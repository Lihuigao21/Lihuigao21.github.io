"""LSC-IVR-style phase-space benchmark for the Matsubara note series.

The script makes the result used in Matsubara Series Part I.  It compares a
classical-Liouvillian phase-space estimator with exact Kubo results.  The
harmonic oscillator is a sanity check: the Kubo q-q correlation is exactly the
classical result.  The quartic oscillator shows the practical boundary of the
same approximation.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "matsubara-series" / "lscivr-benchmark.png"

BETA = 2.0
MASS = 1.0
HBAR = 1.0


def sinc_kinetic(n_grid: int, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, n_grid)
    dx = x[1] - x[0]
    idx = np.arange(n_grid)
    diff = idx[:, None] - idx[None, :]
    kinetic = np.zeros((n_grid, n_grid), dtype=float)
    np.fill_diagonal(kinetic, np.pi**2 / 3.0)
    mask = diff != 0
    kinetic[mask] = 2.0 * ((-1.0) ** diff[mask]) / (diff[mask].astype(float) ** 2)
    kinetic *= HBAR**2 / (2.0 * MASS * dx**2)
    return x, kinetic


def quantum_kubo_qcorr(
    times: np.ndarray,
    potential,
    *,
    x_min: float = -8.0,
    x_max: float = 8.0,
    n_grid: int = 321,
    n_states: int = 100,
) -> np.ndarray:
    x, kinetic = sinc_kinetic(n_grid, x_min, x_max)
    hamiltonian = kinetic + np.diag(potential(x))
    energy, coeff = scipy.linalg.eigh(hamiltonian, subset_by_index=(0, n_states - 1))
    q_matrix = coeff.T @ (x[:, None] * coeff)
    q2 = np.abs(q_matrix) ** 2

    energy_i = energy[:, None]
    energy_j = energy[None, :]
    d_energy = energy_j - energy_i
    boltzmann = np.exp(-BETA * energy)
    numerator = boltzmann[:, None] - boltzmann[None, :]
    denominator = BETA * d_energy
    prefactor = np.empty_like(d_energy)
    mask = np.abs(d_energy) > 1e-12
    np.divide(numerator, denominator, out=prefactor, where=mask)
    np.fill_diagonal(prefactor, boltzmann)

    partition = boltzmann.sum()
    return np.array([np.sum(prefactor * q2 * np.cos(d_energy * t)) / partition for t in times])


def classical_grid_corr(
    times: np.ndarray,
    potential,
    force,
    *,
    q_lim: float = 4.5,
    p_lim: float = 5.0,
    n_q: int = 220,
    n_p: int = 180,
    dt: float = 0.004,
) -> np.ndarray:
    q_axis = np.linspace(-q_lim, q_lim, n_q)
    p_axis = np.linspace(-p_lim, p_lim, n_p)
    q0, p0 = np.meshgrid(q_axis, p_axis, indexing="ij")
    weight = np.exp(-BETA * (p0**2 / (2.0 * MASS) + potential(q0)))
    weight /= weight.sum()

    q = q0.copy()
    p = p0.copy()
    corr = np.empty(len(times))
    corr[0] = np.sum(weight * q0 * q)
    current_t = 0.0

    for i, target_t in enumerate(times[1:], start=1):
        n_step = int(round((target_t - current_t) / dt))
        local_dt = (target_t - current_t) / max(n_step, 1)
        for _ in range(max(n_step, 1)):
            p += 0.5 * local_dt * force(q)
            q += local_dt * p / MASS
            p += 0.5 * local_dt * force(q)
        current_t = target_t
        corr[i] = np.sum(weight * q0 * q)
    return corr


def harmonic_potential(q: np.ndarray) -> np.ndarray:
    return 0.5 * q**2


def harmonic_force(q: np.ndarray) -> np.ndarray:
    return -q


def quartic_potential(q: np.ndarray) -> np.ndarray:
    return 0.25 * q**4


def quartic_force(q: np.ndarray) -> np.ndarray:
    return -(q**3)


def main() -> None:
    harmonic_time = np.linspace(0.0, 10.0, 151)
    quartic_time = np.linspace(0.0, 12.0, 121)

    harmonic_quantum = np.cos(harmonic_time) / BETA
    harmonic_classical = classical_grid_corr(
        harmonic_time,
        harmonic_potential,
        harmonic_force,
        q_lim=4.0,
        p_lim=4.0,
        dt=0.005,
    )

    quartic_quantum = quantum_kubo_qcorr(quartic_time, quartic_potential)
    quartic_classical = classical_grid_corr(
        quartic_time,
        quartic_potential,
        quartic_force,
        q_lim=4.0,
        p_lim=4.5,
        dt=0.0025,
    )

    harmonic_error = float(np.max(np.abs(harmonic_quantum - harmonic_classical)))
    quartic_error = float(np.sqrt(np.mean((quartic_quantum - quartic_classical) ** 2)))

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)

    axes[0].plot(harmonic_time, harmonic_quantum, color="black", lw=2.0, label="Kubo exact")
    axes[0].plot(harmonic_time, harmonic_classical, color="#d62728", ls="--", lw=1.7, label="classical Liouvillian")
    axes[0].set_title("Harmonic oscillator")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$C_{qq}(t)$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(quartic_time, quartic_quantum, color="black", lw=2.0, label="DVR Kubo")
    axes[1].plot(quartic_time, quartic_classical, color="#d62728", ls="--", lw=1.7, label="phase-space approximation")
    axes[1].set_title(r"Quartic oscillator, $V(q)=q^4/4$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$C_{qq}(t)$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(FIGURE_PATH, dpi=190)
    print(f"wrote {FIGURE_PATH}")
    print(f"harmonic max abs error = {harmonic_error:.6e}")
    print(f"quartic RMS difference = {quartic_error:.6e}")


if __name__ == "__main__":
    main()
