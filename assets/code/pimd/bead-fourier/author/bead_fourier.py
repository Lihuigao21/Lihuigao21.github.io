# SPDX-License-Identifier: GPL-3.0-only
# ToyModel BF-PIMD sampler. The helper import is localized for this bundle.
"""One-dimensional Bead–Fourier equilibrium path-integral molecular dynamics.

The action follows Ivanov et al., Phys. Rev. E 67, 066710 (2003).
Gauss–Legendre segment quadrature follows London and Momeni, arXiv:2611.18749.
Like :mod:`toymodel.path_integral`, this module uses exp(-beta/P * H_P).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import irfft, rfft

from ring_helpers import (
    _positive_integer,
    _positive_scalar,
    beta_per_bead,
    real_normal_mode_frequencies,
    ring_polymer_frequency,
)


class BeadFourierPath:
    r"""A cyclic path with `beads` segments and `modes` sine modes per segment.

    `amplitudes[..., j, k-1]` stores the scaled coordinate
    :math:`b_{jk}=k\pi a_{jk}/\sqrt{2}`, where :math:`a_{jk}` multiplies
    :math:`\sin(k\pi s)` along segment `j`. Each scaled amplitude has
    the same spring constant :math:`m[P/(\beta\hbar)]^2` in :math:`H_P`.

    With `modes=0, quadrature_points=None` the potential is evaluated only
    at the beads, so this represents primitive PIMD. A positive quadrature
    order integrates the segment potential, including when `modes=0`.
    For `quadrature_rule="trapezoid"`, the order counts panels. Each bead
    endpoint is evaluated once and contributes to the two adjacent panels.
    """

    def __init__(
        self,
        beads: int,
        beta: float,
        *,
        modes: int = 1,
        quadrature_points: int | None = 2,
        quadrature_rule: str = "gauss-legendre",
        mass: float = 1.0,
        hbar: float = 1.0,
    ):
        self.beads = _positive_integer(beads, "beads")
        self.beta = _positive_scalar(beta, "beta")
        self.mass = _positive_scalar(mass, "mass")
        self.hbar = _positive_scalar(hbar, "hbar")
        if (
            isinstance(modes, (bool, np.bool_))
            or not isinstance(modes, (int, np.integer))
            or modes < 0
        ):
            raise ValueError("modes must be a nonnegative integer.")
        self.modes = int(modes)
        self.beta_p = beta_per_bead(self.beta, self.beads)
        self.omega_p = ring_polymer_frequency(self.beta, self.beads, hbar=self.hbar)
        self.quadrature_points = quadrature_points
        if quadrature_rule not in ("gauss-legendre", "trapezoid"):
            raise ValueError("quadrature_rule must be 'gauss-legendre' or 'trapezoid'.")
        self._wave_numbers = np.pi * np.arange(1, self.modes + 1)
        if quadrature_points is None:
            if self.modes:
                raise ValueError("Sine modes require segment quadrature.")
            self.sites_per_path = self.beads
        else:
            points = _positive_integer(quadrature_points, "quadrature_points")
            if quadrature_rule == "trapezoid":
                # The right endpoint is the next segment's left endpoint.
                # Combine their half weights before evaluating the potential.
                self._nodes = np.arange(points) / points
                self._weights = np.full(points, 1.0 / points)
            else:
                nodes, weights = np.polynomial.legendre.leggauss(points)
                self._nodes = (nodes + 1.0) / 2.0
                self._weights = weights / 2.0
            self._left = 1.0 - self._nodes
            self._basis = self._sine_basis(self._nodes)
            self._left_weights = self._weights * self._left
            self._right_weights = self._weights * self._nodes
            self._mode_weights = self._weights[:, None] * self._basis
            self.sites_per_path = self.beads * points

    def _configuration(self, positions, amplitudes):
        q = np.asarray(positions, dtype=float)
        if q.shape[-1:] != (self.beads,) or not q.size or not np.all(np.isfinite(q)):
            raise ValueError("positions must be nonempty and finite with shape (..., beads).")
        shape = (*q.shape, self.modes)
        b = np.zeros(shape) if amplitudes is None else np.asarray(amplitudes, dtype=float)
        if b.shape != shape or not np.all(np.isfinite(b)):
            raise ValueError("amplitudes must be finite with shape (..., beads, modes).")
        return q, b

    def _sine_basis(self, nodes):
        values = np.sqrt(2.0) * np.sin(nodes[:, None] * self._wave_numbers)
        values[(nodes == 0.0) | (nodes == 1.0)] = 0.0
        return values / self._wave_numbers

    def _segment_positions(self, q, b, left, right, basis):
        positions = q[..., None] * left + np.roll(q, -1, axis=-1)[..., None] * right
        if self.modes:
            positions += b @ basis.T
        return positions

    def reconstruct(self, positions: ArrayLike, amplitudes: ArrayLike, nodes: ArrayLike):
        """Evaluate every segment at the supplied local nodes in [0, 1]."""
        q, b = self._configuration(positions, amplitudes)
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 1 or nodes.size == 0 or not np.all((nodes >= 0) & (nodes <= 1)):
            raise ValueError("nodes must be a nonempty one-dimensional array in [0, 1].")
        return self._segment_positions(q, b, 1.0 - nodes, nodes, self._sine_basis(nodes))

    def _force_positions(self, q, b):
        if self.quadrature_points is None:
            return q
        return self._segment_positions(q, b, self._left, self._nodes, self._basis)

    @staticmethod
    def _evaluate(callback, sites, name):
        values = np.asarray(callback(sites), dtype=float)
        if values.shape != sites.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must return one finite value per position.")
        return values

    def _physical_forces(self, q, b, force):
        sites = self._force_positions(q, b)
        values = self._evaluate(force, sites, "force")
        if self.quadrature_points is None:
            return sites, values, b
        right = values @ self._right_weights
        bead_force = values @ self._left_weights + np.roll(right, 1, axis=-1)
        return sites, bead_force, values @ self._mode_weights

    def physical_forces(self, positions: ArrayLike, amplitudes: ArrayLike, force):
        """Return the bead and bridge forces from the physical potential in H_P."""
        q, b = self._configuration(positions, amplitudes)
        _, bead_force, bridge_force = self._physical_forces(q, b, force)
        return bead_force, bridge_force

    def potential_energy(self, positions: ArrayLike, amplitudes: ArrayLike, potential):
        """Return the configurational part of H_P, including both springs."""
        q, b = self._configuration(positions, amplitudes)
        sites = self._force_positions(q, b)
        values = self._evaluate(potential, sites, "potential")
        physical = values if self.quadrature_points is None else values @ self._weights
        spring = np.sum((q - np.roll(q, -1, axis=-1)) ** 2, axis=-1)
        spring += np.sum(b * b, axis=(-2, -1))
        return np.sum(physical, axis=-1) + 0.5 * self.mass * self.omega_p**2 * spring

    def average_at_force_sites(self, values: ArrayLike):
        """Average a position-diagonal observable evaluated at the action sites."""
        values = np.asarray(values, dtype=float)
        if self.quadrature_points is None:
            if values.shape[-1:] != (self.beads,):
                raise ValueError("values must have shape (..., beads).")
            return values.mean(axis=-1)
        if values.shape[-2:] != (self.beads, self.quadrature_points):
            raise ValueError("values must have shape (..., beads, quadrature_points).")
        return (values @ self._weights).mean(axis=-1)


class BeadFourierPIMD:
    """Normal-mode Langevin sampler for a :class:`BeadFourierPath`.

    `positions` and optional `amplitudes` accept leading batch axes.
    `force` must return the negative physical potential derivative with the
    shape of its input, including the quadrature axis for segment integration.
    Fictitious momenta start from the Maxwell distribution. To use physical
    masses for all auxiliary coordinates, set `mass_frequency=None`; otherwise
    an internal mode of free frequency w gets mass m*(1+(w/mass_frequency)**2).
    With `splitting="obabo"`, two thermostat half steps surround the
    kick, free propagation and kick used in the original PILE scheme.
    The elapsed integration time is a sampling coordinate.
    """

    def __init__(
        self,
        path: BeadFourierPath,
        force,
        positions: ArrayLike,
        *,
        amplitudes: ArrayLike | None = None,
        time_step: float,
        rng: np.random.Generator,
        centroid_friction: float = 1.0,
        mass_frequency: float | None = 1.0,
        splitting: str = "baoab",
    ):
        self.path = path
        self.time_step = _positive_scalar(time_step, "time_step")
        if not callable(force) or not isinstance(rng, np.random.Generator):
            raise TypeError("force must be callable and rng a numpy.random.Generator.")
        self._force = force
        self._rng = rng
        if splitting not in ("baoab", "obabo"):
            raise ValueError("splitting must be 'baoab' or 'obabo'.")
        self._endpoint_thermostat = splitting == "obabo"
        q, b = path._configuration(positions, amplitudes)
        self._fft_scale = np.full(path.beads, np.sqrt(2.0 / path.beads))
        self._fft_scale[0] = 1.0 / np.sqrt(path.beads)
        # FFTPACK stores the negative sine projection and, for even P, a Nyquist mode.
        self._fft_scale[2::2] *= -1.0
        if path.beads % 2 == 0:
            self._fft_scale[-1] = self._fft_scale[0]
        self._inverse_fft_scale = 1.0 / self._fft_scale
        free_frequencies = np.concatenate((
            real_normal_mode_frequencies(path.beta, path.beads, hbar=path.hbar),
            np.full(path.beads * path.modes, path.omega_p),
        ))
        masses = np.full_like(free_frequencies, path.mass)
        if mass_frequency is not None:
            scale = _positive_scalar(mass_frequency, "mass_frequency")
            masses *= 1.0 + (free_frequencies / scale) ** 2
        frequencies = free_frequencies * np.sqrt(path.mass / masses)
        half = 0.5 * self.time_step
        free_time = self.time_step if self._endpoint_thermostat else half
        noise_time = half if self._endpoint_thermostat else self.time_step
        angle = frequencies * free_time
        self._cosine = np.cos(angle)
        self._drift = (free_time / masses) * np.sinc(angle / np.pi)
        self._restore = -masses * frequencies * np.sin(angle)
        friction = 2.0 * frequencies
        friction[0] = _positive_scalar(centroid_friction, "centroid_friction")
        self._decay = np.exp(-friction * noise_time)
        self._noise = np.sqrt(masses / path.beta_p * (-np.expm1(-2.0 * friction * noise_time)))

        self._coordinates = self._join_modes(q, b)
        self._momenta = rng.standard_normal(self._coordinates.shape) * np.sqrt(masses / path.beta_p)
        self._beads, self._amplitudes, self._sites, self._force_modes = self._evaluate(self._coordinates)
        self._batch_size = q.size // path.beads
        self.steps = 0
        self.force_evaluations = self._batch_size * path.sites_per_path

    def _join_modes(self, bead_values, bridge_values):
        modes = rfft(bead_values, axis=-1) * self._fft_scale
        if not self.path.modes:
            return modes
        flat_bridge = bridge_values.reshape(*modes.shape[:-1], -1)
        return np.concatenate((modes, flat_bridge), axis=-1)

    def _evaluate(self, coordinates):
        split = self.path.beads
        q = irfft(coordinates[..., :split] * self._inverse_fft_scale, axis=-1)
        b = coordinates[..., split:].reshape(*q.shape, self.path.modes)
        sites, force_q, force_b = self.path._physical_forces(q, b, self._force)
        return q, b, sites, self._join_modes(force_q, force_b)

    def _free_step(self, q, p):
        return self._cosine * q + self._drift * p, self._cosine * p + self._restore * q

    def _thermostat_step(self, p):
        return self._decay * p + self._noise * self._rng.standard_normal(p.shape)

    def step(self) -> None:
        """Advance one step, reusing the physical force from the old position."""
        half = 0.5 * self.time_step
        p = self._momenta
        if self._endpoint_thermostat:
            p = self._thermostat_step(p)
        coordinates, p = self._free_step(self._coordinates, p + half * self._force_modes)
        if not self._endpoint_thermostat:
            coordinates, p = self._free_step(coordinates, self._thermostat_step(p))
        beads, amplitudes, sites, force_modes = self._evaluate(coordinates)
        p = p + half * force_modes
        if self._endpoint_thermostat:
            p = self._thermostat_step(p)
        self._coordinates = coordinates
        self._momenta = p
        self._beads, self._amplitudes, self._sites, self._force_modes = beads, amplitudes, sites, force_modes
        self.steps += 1
        self.force_evaluations += self._batch_size * self.path.sites_per_path

    @staticmethod
    def _readonly(values):
        view = values.view()
        view.flags.writeable = False
        return view

    @property
    def bead_positions(self):
        """Current bead coordinates."""
        return self._readonly(self._beads)

    @property
    def amplitudes(self):
        """Current scaled sine amplitudes."""
        return self._readonly(self._amplitudes)

    @property
    def force_positions(self):
        """Current physical force sites, cached from the last force evaluation."""
        return self._readonly(self._sites)


__all__ = ["BeadFourierPath", "BeadFourierPIMD"]
