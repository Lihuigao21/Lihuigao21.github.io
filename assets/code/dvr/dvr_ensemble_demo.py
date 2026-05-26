"""Compact density-matrix ensemble evolution demo for the DVR series.

This script reproduces the main operator sequence used in the Part IV note:
thermal density construction, positive-momentum filtering, exact finite-matrix
unitary propagation between saved times, and population diagnostics.  It uses a
one-dimensional free-particle benchmark so the result is easy to inspect.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "dvr-series" / "dvr3-ensemble" / "density-matrix-free-demo.png"


def fft_matrices(n_grid: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the unitary DFT matrix and the corresponding momentum grid."""
    fourier = np.fft.fft(np.eye(n_grid), axis=0) / np.sqrt(n_grid)
    momentum = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=dx)
    return fourier, momentum


def density_from_thermal_harmonic_state(
    x: np.ndarray,
    *,
    mass: float,
    omega: float,
    beta: float,
    center: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = x[1] - x[0]
    fourier, momentum = fft_matrices(len(x), dx)
    p2 = fourier.conj().T @ np.diag(momentum**2) @ fourier
    potential = 0.5 * mass * omega**2 * np.diag((x - center) ** 2)
    h_ref = p2 / (2.0 * mass) + potential
    energy, coeff = scipy.linalg.eigh(h_ref)
    shifted = energy - energy[0]
    boltzmann = np.exp(-beta * shifted)
    rho = (coeff * boltzmann) @ coeff.conj().T
    rho /= np.trace(rho)
    return rho, fourier, momentum


def positive_momentum_filter(fourier: np.ndarray, momentum: np.ndarray, width: float) -> np.ndarray:
    """Smooth projector that keeps right-moving components."""
    selector = 0.5 * (1.0 + np.tanh(momentum / width))
    return fourier.conj().T @ np.diag(selector) @ fourier


def expectation(rho: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ operator)))


def main() -> None:
    n_grid = 512
    x_min, x_max = -80.0, 80.0
    x = np.linspace(x_min, x_max, n_grid, endpoint=False)
    dx = x[1] - x[0]
    mass = 1.0
    beta = 2.0
    omega = 0.12
    center = -10.0

    rho, fourier, momentum = density_from_thermal_harmonic_state(
        x,
        mass=mass,
        omega=omega,
        beta=beta,
        center=center,
    )
    boost = np.diag(np.exp(1j * 0.6 * x))
    rho = boost @ rho @ boost.conj().T
    filter_plus = positive_momentum_filter(fourier, momentum, width=0.10)
    rho = filter_plus @ rho @ filter_plus.conj().T
    rho /= np.trace(rho)

    p_operator = fourier.conj().T @ np.diag(momentum) @ fourier
    h_free = (fourier.conj().T @ np.diag(momentum**2) @ fourier) / (2.0 * mass)
    energy, coeff = scipy.linalg.eigh(h_free)
    rho_energy = coeff.conj().T @ rho @ coeff

    times = np.linspace(0.0, 50.0, 101)
    x_operator = np.diag(x)
    mean_x = []
    mean_p = []
    density_snapshots: dict[float, np.ndarray] = {}
    for t in times:
        phase = np.exp(-1j * energy * t)
        rho_t = coeff @ ((phase[:, None] * rho_energy) * phase.conj()[None, :]) @ coeff.conj().T
        mean_x.append(expectation(rho_t, x_operator))
        mean_p.append(expectation(rho_t, p_operator))
        if t in {0.0, 20.0, 50.0}:
            density_snapshots[float(t)] = np.real(np.diag(rho_t)) / dx

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)

    axes[0].plot(times, mean_x, color="#1f77b4")
    axes[0].set_title("Mean position")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\langle x\rangle$")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(times, mean_p, color="#d95f02")
    axes[1].set_title("Mean momentum")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\langle p\rangle$")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_ylim(0.82, 0.92)

    for t, density in density_snapshots.items():
        axes[2].plot(x, density, label=f"t={t:g}")
    axes[2].set_title("Position density")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel(r"$\rho(x,x;t)$")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    fig.suptitle("Positive-momentum density-matrix benchmark", y=1.05)
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"wrote {FIGURE_PATH}")
    print(f"Tr(rho0) = {np.trace(rho):.12f}")
    print(f"<x>(0), <x>(final) = {mean_x[0]:.8f}, {mean_x[-1]:.8f}")
    print(f"<p>(0), <p>(final) = {mean_p[0]:.8f}, {mean_p[-1]:.8f}")


if __name__ == "__main__":
    main()
