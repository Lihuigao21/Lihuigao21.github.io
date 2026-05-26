"""Minimal DVR and Kubo-correlation utilities for the DVR note series.

This file is intentionally compact. It collects the core numerical pieces used
throughout the article series:

- sinc DVR kinetic-energy matrix,
- one-state DVR eigenproblem,
- multi-state DVR Hamiltonian assembly,
- Gaussian wavepacket initialization and propagation,
- operator matrix elements in the energy basis,
- Kubo-transformed correlation functions,
- side and flux operators for flux-side correlation tests.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.linalg


Array = np.ndarray


def sinc_grid(n_grid: int, bound: tuple[float, float]) -> tuple[Array, float]:
    """Return evenly spaced grid points and spacing."""
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    x = np.linspace(bound[0], bound[1], n_grid)
    dx = float(bound[1] - bound[0]) / (n_grid - 1)
    return x, dx


def sinc_kinetic(n_grid: int, bound: tuple[float, float], mass: float = 1.0) -> tuple[Array, Array, float]:
    """Build the sinc DVR kinetic-energy matrix.

    The Hamiltonian convention is H = p^2/(2m) + V(x), with hbar = 1.
    """
    x, dx = sinc_grid(n_grid, bound)
    alternating = np.ones(n_grid - 1)
    alternating[0::2] = -1.0
    first_row = np.concatenate(
        [
            np.array([np.pi**2 / 3.0]),
            2.0 * alternating / (np.arange(1, n_grid) ** 2),
        ]
    )
    kinetic = scipy.linalg.toeplitz(first_row / (2.0 * mass * dx**2))
    return kinetic, x, dx


def finite_difference_kinetic(n_grid: int, bound: tuple[float, float], mass: float = 1.0) -> tuple[Array, Array, float]:
    """Build the local second-order finite-difference kinetic matrix."""
    x, dx = sinc_grid(n_grid, bound)
    laplacian = np.diag(-2.0 * np.ones(n_grid))
    laplacian += np.diag(np.ones(n_grid - 1), 1)
    laplacian += np.diag(np.ones(n_grid - 1), -1)
    kinetic = -laplacian / (2.0 * mass * dx**2)
    return kinetic, x, dx


def solve_sinc_dvr(
    potential: Callable[[Array], Array],
    *,
    n_states: int = 20,
    n_grid: int = 401,
    bound: tuple[float, float] = (-8.0, 8.0),
    mass: float = 1.0,
) -> tuple[Array, Array, Array, float]:
    """Solve a one-dimensional time-independent Schrodinger equation."""
    kinetic, x, dx = sinc_kinetic(n_grid, bound, mass)
    hamiltonian = kinetic + np.diag(potential(x))
    energy, coeff = scipy.linalg.eigh(hamiltonian, subset_by_index=(0, n_states - 1))

    # Normalize columns as wavefunctions: integral |psi(x)|^2 dx = 1.
    norms = np.sqrt(np.trapezoid(coeff**2, x=x, axis=0))
    coeff = coeff / norms
    return energy, coeff, x, dx


def gaussian_wavepacket(x: Array, *, x0: float, k0: float, sigma: float, dx: float) -> Array:
    """Return a normalized Gaussian wavepacket on a DVR grid."""
    packet = np.exp(1j * k0 * (x - x0) - sigma * (x - x0) ** 2)
    packet = packet / np.sqrt(np.sum(np.abs(packet) ** 2) * dx)
    return packet


def propagate_state(hamiltonian: Array, psi0: Array, time: Array, hbar: float = 1.0) -> Array:
    """Propagate a state vector by diagonalizing a time-independent Hamiltonian."""
    energy, coeff = scipy.linalg.eigh(hamiltonian)
    initial_in_energy_basis = coeff.conj().T @ psi0
    phase = np.exp(-1j * energy[:, None] * time[None, :] / hbar)
    return (coeff @ (initial_in_energy_basis[:, None] * phase)).T


def state_populations(psi_t: Array, *, n_state: int, n_grid: int, dx: float) -> Array:
    """Return electronic-state populations for state-major |state, grid> ordering."""
    wavefunction = psi_t.reshape(len(psi_t), n_state, n_grid)
    return np.sum(np.abs(wavefunction) ** 2, axis=2) * dx


def operator_matrix(operator_on_grid: Array, coeff: Array, dx: float) -> Array:
    """Return <i|A|j> from grid values A(x) and eigenvectors coeff[x,i]."""
    weighted = operator_on_grid[:, None] * coeff
    return coeff.T @ weighted * dx


def kubo_correlation(
    time: Array,
    energy: Array,
    a_matrix: Array,
    b_matrix: Array,
    beta: float,
) -> Array:
    """Kubo-transformed correlation function in the energy basis.

    K_AB(t) = 1/(Z beta) int_0^beta d lambda
              Tr[e^{-(beta-lambda)H} A e^{-lambda H} B(t)].
    """
    energy = np.asarray(energy)
    d_e = energy[:, None] - energy[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        prefactor = (np.exp(-beta * energy[None, :]) - np.exp(-beta * energy[:, None])) / d_e / beta
    np.fill_diagonal(prefactor, np.exp(-beta * energy))

    phase = np.exp(-1j * d_e[:, :, None] * time[None, None, :])
    partition = np.exp(-beta * energy).sum()
    return np.einsum("ij,ij,ji,ijt->t", prefactor, a_matrix, b_matrix, phase) / partition


def multistate_hamiltonian(
    potential_matrix_on_grid: Array,
    kinetic: Array,
) -> Array:
    """Assemble a dense multi-state DVR Hamiltonian.

    potential_matrix_on_grid has shape (n_state, n_state, n_grid).
    The basis ordering is |state, grid>.
    """
    n_state, _, n_grid = potential_matrix_on_grid.shape
    dim = n_state * n_grid
    hamiltonian = np.zeros((dim, dim), dtype=float)

    for a in range(n_state):
        row = slice(a * n_grid, (a + 1) * n_grid)
        hamiltonian[row, row] += kinetic

    for a in range(n_state):
        row = slice(a * n_grid, (a + 1) * n_grid)
        for b in range(n_state):
            col = slice(b * n_grid, (b + 1) * n_grid)
            hamiltonian[row, col] += np.diag(potential_matrix_on_grid[a, b])

    return hamiltonian


def tully_simple_avoided_crossing(
    x: Array,
    *,
    a: float = 0.01,
    b: float = 1.6,
    c: float = 0.005,
    d: float = 1.0,
) -> Array:
    """Return V_ab(x) for Tully's simple avoided crossing model.

    Output shape is (2, 2, n_grid), ready for multistate_hamiltonian.
    """
    v11 = np.sign(x) * a * (1.0 - np.exp(-b * np.abs(x)))
    v22 = -v11
    v12 = c * np.exp(-d * x**2)
    potential = np.zeros((2, 2, len(x)), dtype=float)
    potential[0, 0] = v11
    potential[1, 1] = v22
    potential[0, 1] = v12
    potential[1, 0] = v12
    return potential


def side_projector(x: Array, dividing_surface: float = 0.0) -> Array:
    """Grid-diagonal side operator h(x - x0)."""
    return np.diag((x > dividing_surface).astype(float))


def flux_operator(hamiltonian: Array, side: Array, hbar: float = 1.0) -> Array:
    """Flux operator F = i/hbar [H, h]."""
    return 1j / hbar * (hamiltonian @ side - side @ hamiltonian)


def harmonic_demo() -> None:
    """Small x-x Kubo smoke test for a harmonic oscillator."""
    beta = 1.0
    time = np.linspace(0.0, 8.0, 200)
    potential = lambda x: 0.5 * x**2
    energy, coeff, x, dx = solve_sinc_dvr(potential, n_states=30, n_grid=401, bound=(-8, 8), mass=1.0)
    x_matrix = operator_matrix(x, coeff, dx)
    c_xx = kubo_correlation(time, energy, x_matrix, x_matrix, beta)
    print("lowest energies:", np.round(energy[:5], 8))
    print("C_xx(0):", float(np.real(c_xx[0])))


def wavepacket_demo() -> None:
    """Small two-state wavepacket propagation smoke test."""
    n_grid = 161
    kinetic, x, dx = sinc_kinetic(n_grid, (-12.0, 12.0), mass=2000.0)
    potential = tully_simple_avoided_crossing(x)
    hamiltonian = multistate_hamiltonian(potential, kinetic)

    psi0 = np.zeros(2 * n_grid, dtype=complex)
    psi0[:n_grid] = gaussian_wavepacket(x, x0=-8.0, k0=20.0, sigma=0.35, dx=dx)

    time = np.linspace(0.0, 1200.0, 80)
    psi_t = propagate_state(hamiltonian, psi0, time)
    populations = state_populations(psi_t, n_state=2, n_grid=n_grid, dx=dx)
    print("final two-state populations:", np.round(populations[-1], 8))


if __name__ == "__main__":
    harmonic_demo()
    wavepacket_demo()
