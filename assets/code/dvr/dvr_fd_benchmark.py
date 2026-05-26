"""Benchmark sinc DVR against a second-order finite-difference Hamiltonian.

The test problem is the one-dimensional harmonic oscillator with hbar = m =
omega = 1, so the exact bound-state energies are E_n = n + 1/2.  Running this
file prints a small error table and writes the benchmark figure used in the DVR
series Part I article.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "dvr-series" / "dvr0-methods" / "dvr-fd-benchmark.png"


def grid(n_grid: int, bound: tuple[float, float]) -> tuple[np.ndarray, float]:
    x = np.linspace(bound[0], bound[1], n_grid)
    dx = (bound[1] - bound[0]) / (n_grid - 1)
    return x, dx


def sinc_kinetic(n_grid: int, bound: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """Infinite-domain sinc DVR kinetic matrix for hbar = m = 1."""
    x, dx = grid(n_grid, bound)
    alternating = np.ones(n_grid - 1)
    alternating[0::2] = -1.0
    first_row = np.concatenate(
        [
            np.array([np.pi**2 / 3.0]),
            2.0 * alternating / (np.arange(1, n_grid) ** 2),
        ]
    )
    kinetic = scipy.linalg.toeplitz(first_row / (2.0 * dx**2))
    return kinetic, x, dx


def finite_difference_kinetic(n_grid: int, bound: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """Second-order central finite-difference kinetic matrix."""
    x, dx = grid(n_grid, bound)
    laplacian = np.diag(-2.0 * np.ones(n_grid))
    laplacian += np.diag(np.ones(n_grid - 1), 1)
    laplacian += np.diag(np.ones(n_grid - 1), -1)
    kinetic = -laplacian / (2.0 * dx**2)
    return kinetic, x, dx


def oscillator_energies(kind: str, n_grid: int, bound: tuple[float, float], n_state: int) -> np.ndarray:
    if kind == "dvr":
        kinetic, x, _ = sinc_kinetic(n_grid, bound)
    elif kind == "fd":
        kinetic, x, _ = finite_difference_kinetic(n_grid, bound)
    else:
        raise ValueError(f"unknown kinetic-energy type: {kind}")

    potential = 0.5 * x**2
    hamiltonian = kinetic + np.diag(potential)
    energies = scipy.linalg.eigh(hamiltonian, subset_by_index=(0, n_state - 1), eigvals_only=True)
    return energies


def main() -> None:
    bound = (-8.0, 8.0)
    n_grid = 121
    n_state = 14
    exact = np.arange(n_state) + 0.5

    dvr_energy = oscillator_energies("dvr", n_grid, bound, n_state)
    fd_energy = oscillator_energies("fd", n_grid, bound, n_state)
    dvr_error = np.abs(dvr_energy - exact)
    fd_error = np.abs(fd_energy - exact)

    grid_sizes = np.array([41, 61, 81, 101, 121, 161, 201])
    dx_values = []
    dvr_ground_error = []
    fd_ground_error = []
    for n in grid_sizes:
        _, dx = grid(int(n), bound)
        dx_values.append(dx)
        dvr_ground_error.append(abs(oscillator_energies("dvr", int(n), bound, 1)[0] - 0.5))
        fd_ground_error.append(abs(oscillator_energies("fd", int(n), bound, 1)[0] - 0.5))

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    axes[0].semilogy(np.arange(n_state), dvr_error, "o-", label="sinc DVR")
    axes[0].semilogy(np.arange(n_state), fd_error, "s--", label="finite difference")
    axes[0].set_xlabel("state index n")
    axes[0].set_ylabel(r"$|E_n^{num} - (n + 1/2)|$")
    axes[0].set_title("Energy errors on the same grid")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].loglog(dx_values, dvr_ground_error, "o-", label="sinc DVR")
    axes[1].loglog(dx_values, fd_ground_error, "s--", label="finite difference")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"grid spacing $\Delta x$")
    axes[1].set_ylabel(r"ground-state error")
    axes[1].set_title("Ground-state convergence")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Harmonic-oscillator benchmark: spectral DVR vs local stencil", y=1.04)
    fig.savefig(FIGURE_PATH, dpi=180)

    print(f"wrote {FIGURE_PATH}")
    print(" n     DVR error        FD error")
    for n, de, fe in zip(range(n_state), dvr_error, fd_error):
        print(f"{n:2d}  {de:12.4e}  {fe:12.4e}")


if __name__ == "__main__":
    main()
