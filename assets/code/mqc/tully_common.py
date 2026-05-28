"""Small utilities for one-dimensional Tully MQC benchmarks.

The model is Tully's simple avoided crossing in atomic units:

    V_11 = sign(x) A (1 - exp(-B |x|))
    V_22 = -V_11
    V_12 = C exp(-D x^2)

The helper functions return adiabatic energies, forces, and derivative
couplings using an analytic two-state parametrization. They are deliberately
dependency-light so the accompanying article scripts can be run directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HBAR = 1.0
MASS = 2000.0


@dataclass(frozen=True)
class SACParameters:
    a: float = 0.01
    b: float = 1.6
    c: float = 0.005
    d: float = 1.0


PARAMS = SACParameters()


def diabatic_terms(x: float, params: SACParameters = PARAMS) -> tuple[float, float, float, float]:
    """Return z, u, dz/dx, du/dx for [[z, u], [u, -z]]."""

    ax = abs(x)
    sign = 1.0 if x >= 0.0 else -1.0
    z = sign * params.a * (1.0 - np.exp(-params.b * ax))
    u = params.c * np.exp(-params.d * x * x)
    dz = params.a * params.b * np.exp(-params.b * ax)
    du = -2.0 * params.c * params.d * x * np.exp(-params.d * x * x)
    return float(z), float(u), float(dz), float(du)


def diabatic_matrix(x: float, params: SACParameters = PARAMS) -> np.ndarray:
    """Return the two-by-two diabatic potential matrix."""

    z, u, _, _ = diabatic_terms(x, params)
    return np.array([[z, u], [u, -z]], dtype=float)


def adiabatic_data(x: float, params: SACParameters = PARAMS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return adiabatic energies, forces, and derivative-coupling matrix."""

    z, u, dz, du = diabatic_terms(x, params)
    radius = float(np.hypot(z, u))
    energies = np.array([-radius, radius], dtype=float)

    if radius <= 1.0e-14:
        d_upper = 0.0
        d01 = 0.0
    else:
        d_upper = (z * dz + u * du) / radius
        d_alpha = (z * du - u * dz) / (radius * radius)
        d01 = 0.5 * d_alpha

    forces = np.array([d_upper, -d_upper], dtype=float)
    nac = np.array([[0.0, d01], [-d01, 0.0]], dtype=float)
    return energies, forces, nac


def electronic_rhs(c: np.ndarray, q: float, p: float, mass: float = MASS) -> np.ndarray:
    """Electronic coefficient equation in the adiabatic basis."""

    energies, _, nac = adiabatic_data(q)
    velocity = p / mass
    return -1j * energies * c - velocity * (nac @ c)


def rk4_electronic(c: np.ndarray, q: float, p: float, dt: float, mass: float = MASS) -> np.ndarray:
    """Fourth-order update for electronic coefficients at fixed q and p."""

    k1 = electronic_rhs(c, q, p, mass)
    k2 = electronic_rhs(c + 0.5 * dt * k1, q, p, mass)
    k3 = electronic_rhs(c + 0.5 * dt * k2, q, p, mass)
    k4 = electronic_rhs(c + dt * k3, q, p, mass)
    out = c + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    norm = np.linalg.norm(out)
    if norm > 0.0:
        out = out / norm
    return out


def velocity_verlet_on_state(
    q: float, p: float, state: int, dt: float, mass: float = MASS
) -> tuple[float, float]:
    """Velocity-Verlet nuclear update on one adiabatic surface."""

    _, forces, _ = adiabatic_data(q)
    p_half = p + 0.5 * dt * forces[state]
    q_new = q + dt * p_half / mass
    _, forces_new, _ = adiabatic_data(q_new)
    p_new = p_half + 0.5 * dt * forces_new[state]
    return float(q_new), float(p_new)


def mean_field_force(q: float, c: np.ndarray) -> float:
    """Ehrenfest force from adiabatic populations and coherences."""

    energies, forces, nac = adiabatic_data(q)
    rho = np.outer(c, c.conjugate())
    diagonal = float(np.real(rho[0, 0]) * forces[0] + np.real(rho[1, 1]) * forces[1])
    coherence = float(2.0 * np.real(nac[0, 1] * rho[0, 1]) * (energies[1] - energies[0]))
    return diagonal + coherence


def classify_scattering(q: float, p: float) -> str:
    """Classify the final nuclear branch by side and momentum direction."""

    if q >= 0.0 and p >= 0.0:
        return "transmitted"
    if q <= 0.0 and p <= 0.0:
        return "reflected"
    return "inside"


def model_grid(xmin: float = -8.0, xmax: float = 8.0, n: int = 900) -> dict[str, np.ndarray]:
    """Return arrays useful for plotting the SAC model."""

    xs = np.linspace(xmin, xmax, n)
    v11 = np.empty_like(xs)
    v22 = np.empty_like(xs)
    coupling = np.empty_like(xs)
    e0 = np.empty_like(xs)
    e1 = np.empty_like(xs)
    d01 = np.empty_like(xs)
    for i, x in enumerate(xs):
        z, u, _, _ = diabatic_terms(float(x))
        energies, _, nac = adiabatic_data(float(x))
        v11[i] = z
        v22[i] = -z
        coupling[i] = u
        e0[i], e1[i] = energies
        d01[i] = nac[0, 1]
    return {
        "x": xs,
        "v11": v11,
        "v22": v22,
        "coupling": coupling,
        "e0": e0,
        "e1": e1,
        "d01": d01,
    }
