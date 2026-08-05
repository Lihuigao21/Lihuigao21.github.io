"""Minimal, executable CSSH rate algorithm for a two-state 1D model.

The code exposes the parts that determine the physics:

1. forward-flux-conditioned initial conditions,
2. trajectory-fitted fewest-switches electronic propagation,
3. accepted and frustrated hop handling,
4. conversion of conditional transmission into a thermal rate.

It is a compact public companion to the article, not the production driver.
Atomic units are used throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.linalg import expm


KB_AU = 3.166811563e-6
Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass
class InitialEnsemble:
    x: Array
    p: Array
    active: NDArray[np.int64]
    amplitudes: ComplexArray
    total_energy: Array
    barrier_energy: float


@dataclass
class TrajectoryOutcome:
    transmitted: bool
    reflected: bool
    attempted_hops: int
    accepted_hops: int
    frustrated_hops: int


@dataclass
class RateEstimate:
    transmission_probability: float
    binomial_se: float
    cssh_rate: float
    cssh_rate_se: float
    gamma: float | None


class TabulatedCentroidSurface:
    """State-specific centroid free-energy surfaces V_c,n(x)."""

    def __init__(self, grid: Array, values: Array):
        grid = np.asarray(grid, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape != (grid.size, 2):
            raise ValueError("values must have shape (ngrid, 2)")
        self.grid = grid
        self.values = values
        self._splines = [
            CubicSpline(grid, values[:, state], bc_type=((1, 0.0), (1, 0.0)))
            for state in range(2)
        ]

    def energy(self, x: float, state: int) -> float:
        if x <= self.grid[0]:
            return float(self.values[0, state])
        if x >= self.grid[-1]:
            return float(self.values[-1, state])
        return float(self._splines[state](x))

    def force(self, x: float, state: int) -> float:
        if x <= self.grid[0] or x >= self.grid[-1]:
            return 0.0
        return -float(self._splines[state](x, 1))

    def barrier(self, state: int) -> tuple[float, float]:
        i = int(np.argmax(self.values[:, state]))
        return float(self.grid[i]), float(self.values[i, state])


class SingleAvoidedCrossing:
    """Physical two-state SAC Hamiltonian in a smooth real adiabatic gauge."""

    def __init__(
        self,
        a: float = 0.01,
        b: float = 1.6,
        coupling: float = 0.002,
        width: float = 1.0,
    ):
        self.a = a
        self.b = b
        self.coupling = coupling
        self.width = width

    def _diabatic(self, x: float) -> tuple[float, float, float, float]:
        if x >= 0.0:
            v11 = self.a * (1.0 - np.exp(-self.b * x))
            dv11 = self.a * self.b * np.exp(-self.b * x)
        else:
            v11 = -self.a * (1.0 - np.exp(self.b * x))
            dv11 = self.a * self.b * np.exp(self.b * x)
        v12 = self.coupling * np.exp(-self.width * x * x)
        dv12 = -2.0 * self.width * x * v12
        return v11, dv11, v12, dv12

    def energies_and_nac(self, x: float) -> tuple[Array, Array]:
        v11, dv11, v12, dv12 = self._diabatic(x)
        radius = np.sqrt(v11 * v11 + v12 * v12)
        energies = np.array([self.a - radius, self.a + radius])

        # With diabatic gap delta = v11 - (-v11) = 2 v11,
        # d_01 = d(theta)/dx in the selected continuous real gauge.
        delta = 2.0 * v11
        ddelta = 2.0 * dv11
        denominator = delta * delta + 4.0 * v12 * v12
        dtheta = (delta * dv12 - v12 * ddelta) / denominator
        nac = np.array([[0.0, dtheta], [-dtheta, 0.0]])
        return energies, nac


def sample_flux_conditioned_initial_conditions(
    *,
    ntraj: int,
    temperature: float,
    mass: float,
    x_start: float,
    active_state: int,
    centroid: TabulatedCentroidSurface,
    rng: np.random.Generator,
) -> InitialEnsemble:
    """Sample the positive thermal flux conditional on clearing the barrier.

    For positive incoming momenta, the flux-weighted canonical density is

        rho_flux(p) = beta (p/m) exp[-beta p^2/(2m)].

    Hence K = p^2/(2m) is exponential. Conditioning the total energy on
    E >= V_c^dagger gives E = V_c^dagger + Exp(scale=1/beta).
    """

    beta = 1.0 / (KB_AU * temperature)
    _, barrier_energy = centroid.barrier(active_state)
    start_energy = centroid.energy(x_start, active_state)

    excess = rng.exponential(scale=1.0 / beta, size=ntraj)
    total_energy = barrier_energy + excess
    kinetic = total_energy - start_energy
    if np.any(kinetic < 0.0):
        raise ValueError("Starting point lies above the conditioned energies")

    x = np.full(ntraj, x_start)
    p = np.sqrt(2.0 * mass * kinetic)
    active = np.full(ntraj, active_state, dtype=np.int64)
    amplitudes = np.zeros((ntraj, 2), dtype=np.complex128)
    amplitudes[:, active_state] = 1.0
    return InitialEnsemble(
        x=x,
        p=p,
        active=active,
        amplitudes=amplitudes,
        total_energy=total_energy,
        barrier_energy=barrier_energy,
    )


def trajectory_fitted_hamiltonian(
    *,
    x: float,
    p: float,
    mass: float,
    active_state: int,
    physical: SingleAvoidedCrossing,
) -> ComplexArray:
    """Trajectory-fitted electronic Hamiltonian with phase correction.

    The inactive-channel momentum is reconstructed at the same total energy.
    If that channel is closed, its magnitude is clipped to zero. The diagonal
    terms accumulate the momentum-dependent phase correction, while the
    off-diagonal terms contain the physical velocity times physical NAC.
    """

    energies, nac = physical.energies_and_nac(x)
    inactive = 1 - active_state
    channel_argument = (
        p * p + 2.0 * mass * (energies[active_state] - energies[inactive])
    )
    inactive_momentum = np.sign(p) * np.sqrt(max(0.0, channel_argument))
    channel_momenta = np.array([p, p], dtype=float)
    channel_momenta[inactive] = inactive_momentum

    h = np.zeros((2, 2), dtype=np.complex128)
    h[0, 0] = -p * channel_momenta[0] / mass
    h[1, 1] = -p * channel_momenta[1] / mass
    velocity = p / mass
    h[0, 1] = -1j * velocity * nac[0, 1]
    h[1, 0] = -1j * velocity * nac[1, 0]
    return h


def propagate_electronic(
    amplitudes: ComplexArray,
    hamiltonian: ComplexArray,
    dt: float,
) -> ComplexArray:
    """Exact local two-state propagation for a frozen midpoint Hamiltonian."""

    updated = expm(-1j * hamiltonian * dt) @ amplitudes
    return updated / np.linalg.norm(updated)


def fssh_trial_probability(
    *,
    amplitudes: ComplexArray,
    active_state: int,
    x: float,
    p: float,
    mass: float,
    dt: float,
    physical: SingleAvoidedCrossing,
) -> float:
    """Fewest-switches probability for the only possible two-state hop."""

    target = 1 - active_state
    _, nac = physical.energies_and_nac(x)
    coupling = (p / mass) * nac[active_state, target]
    coherence = np.conj(amplitudes[active_state]) * amplitudes[target]
    population = max(abs(amplitudes[active_state]) ** 2, 1.0e-15)
    probability = max(0.0, 2.0 * np.real(coherence * coupling) * dt / population)
    if probability > 1.0 + 1.0e-12:
        raise RuntimeError("Hop probability exceeds one; reduce the time step")
    return min(1.0, probability)


def resolve_hop(
    *,
    x: float,
    p: float,
    mass: float,
    active_state: int,
    centroid: TabulatedCentroidSurface,
) -> tuple[int, float, bool]:
    """Apply centroid-surface energy conservation to an attempted hop.

    Returns (new_state, new_momentum, accepted). In one dimension, a
    frustrated hop reverses momentum along the NAC direction and leaves the
    active state unchanged.
    """

    target = 1 - active_state
    gap = centroid.energy(x, target) - centroid.energy(x, active_state)
    kinetic_after = p * p / (2.0 * mass) - gap
    if kinetic_after >= 0.0:
        direction = 1.0 if p >= 0.0 else -1.0
        new_p = direction * np.sqrt(2.0 * mass * kinetic_after)
        return target, float(new_p), True
    return active_state, float(-p), False


def propagate_one_trajectory(
    *,
    x0: float,
    p0: float,
    active0: int,
    amplitudes0: ComplexArray,
    mass: float,
    dt: float,
    max_steps: int,
    x_reactant: float,
    x_product: float,
    centroid: TabulatedCentroidSurface,
    physical: SingleAvoidedCrossing,
    rng: np.random.Generator,
) -> TrajectoryOutcome:
    """Velocity-Verlet nuclei plus midpoint TFF/FSSH electronic dynamics."""

    x = float(x0)
    p = float(p0)
    active = int(active0)
    amplitudes = np.asarray(amplitudes0, dtype=np.complex128).copy()
    attempted = accepted = frustrated = 0

    for _ in range(max_steps):
        # Nuclear velocity Verlet on the currently active centroid surface.
        p_half = p + 0.5 * dt * centroid.force(x, active)
        x_new = x + dt * p_half / mass
        p_new = p_half + 0.5 * dt * centroid.force(x_new, active)

        # Electronic propagation and hopping use physical PES/NAC at midpoint.
        x_mid = 0.5 * (x + x_new)
        p_mid = 0.5 * (p + p_new)
        h_mid = trajectory_fitted_hamiltonian(
            x=x_mid,
            p=p_mid,
            mass=mass,
            active_state=active,
            physical=physical,
        )
        amplitudes = propagate_electronic(amplitudes, h_mid, dt)
        probability = fssh_trial_probability(
            amplitudes=amplitudes,
            active_state=active,
            x=x_mid,
            p=p_mid,
            mass=mass,
            dt=dt,
            physical=physical,
        )

        if rng.random() < probability:
            attempted += 1
            active, p_new, was_accepted = resolve_hop(
                x=x_new,
                p=p_new,
                mass=mass,
                active_state=active,
                centroid=centroid,
            )
            if was_accepted:
                accepted += 1
            else:
                frustrated += 1

        x, p = x_new, p_new
        if x >= x_product:
            return TrajectoryOutcome(True, False, attempted, accepted, frustrated)
        if x <= x_reactant and p < 0.0:
            return TrajectoryOutcome(False, True, attempted, accepted, frustrated)

    return TrajectoryOutcome(False, False, attempted, accepted, frustrated)


def estimate_cssh_rate(
    outcomes: list[TrajectoryOutcome],
    *,
    cmd_rate: float,
    exact_nonadiabatic_to_adiabatic_ratio: float | None = None,
) -> RateEstimate:
    """Convert a conditional transmission fraction into the CSSH rate."""

    ntraj = len(outcomes)
    if ntraj == 0:
        raise ValueError("At least one trajectory is required")
    transmitted = sum(outcome.transmitted for outcome in outcomes)
    probability = transmitted / ntraj
    probability_se = np.sqrt(probability * (1.0 - probability) / ntraj)
    gamma = None
    if exact_nonadiabatic_to_adiabatic_ratio is not None:
        gamma = probability / exact_nonadiabatic_to_adiabatic_ratio
    return RateEstimate(
        transmission_probability=probability,
        binomial_se=probability_se,
        cssh_rate=cmd_rate * probability,
        cssh_rate_se=cmd_rate * probability_se,
        gamma=gamma,
    )


def _self_check() -> None:
    """Fast checks for the sampling law and hop-resolution branches."""

    grid = np.array([-1.0, 0.0, 1.0])
    surfaces = np.column_stack(
        [
            np.array([0.0, 0.01, 0.0]),
            np.array([0.02, 0.03, 0.02]),
        ]
    )
    centroid = TabulatedCentroidSurface(grid, surfaces)
    rng = np.random.default_rng(20250805)
    ensemble = sample_flux_conditioned_initial_conditions(
        ntraj=200_000,
        temperature=100.0,
        mass=2000.0,
        x_start=-1.0,
        active_state=0,
        centroid=centroid,
        rng=rng,
    )
    beta = 1.0 / (KB_AU * 100.0)
    mean_excess = np.mean(ensemble.total_energy - ensemble.barrier_energy)
    assert np.isclose(mean_excess, 1.0 / beta, rtol=0.01)

    state, momentum, accepted = resolve_hop(
        x=0.0,
        p=np.sqrt(2.0 * 2000.0 * 0.03),
        mass=2000.0,
        active_state=0,
        centroid=centroid,
    )
    assert accepted and state == 1 and momentum > 0.0

    state, momentum, accepted = resolve_hop(
        x=0.0,
        p=np.sqrt(2.0 * 2000.0 * 0.005),
        mass=2000.0,
        active_state=0,
        centroid=centroid,
    )
    assert not accepted and state == 0 and momentum < 0.0
    print("CSSH algorithm self-check passed.")


if __name__ == "__main__":
    _self_check()
