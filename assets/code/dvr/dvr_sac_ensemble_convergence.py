"""Convergence-aware density-matrix ensemble benchmark for DVR Part IV.

The old SAC figure in the article was a useful smoke test, but it was not a
converged scattering result: only the right transmitted channel was recorded,
so reflected and slow components stayed in the numerical box.  This script uses
a low-rank density-matrix representation and records both left and right
absorber ledgers.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = (
    ROOT
    / "assets"
    / "img"
    / "dvr-series"
    / "dvr3-ensemble"
    / "sac-ensemble-convergence.png"
)


@dataclass(frozen=True)
class Parameters:
    hbar: float = 1.0
    mass: float = 2000.0
    k_b: float = 3.166811563e-6
    temperature: float = 500.0
    x_min: float = -40.0
    x_max: float = 40.0
    n_grid: int = 512
    x_initial: float = -10.0
    omega_initial: float = 0.000911
    initial_adiabatic_state: int = 1
    t_final: float = 300_000.0
    dt: float = 20.0
    n_save: int = 241
    left_absorb_start: float = -34.0
    left_absorb_end: float = -27.0
    right_absorb_start: float = 9.0
    right_absorb_end: float = 16.0
    # A strict p > 0 filter leaves arbitrarily slow components.  The small
    # offset below makes this a finite-time benchmark rather than an algebraic
    # tail-convergence test.
    momentum_filter_center: float = 0.45
    momentum_filter_width: float = 0.08
    density_weight_cutoff: float = 1e-9
    tully_a: float = 0.01
    tully_b: float = 1.6
    tully_c: float = 0.002
    tully_d: float = 1.0


def fft_matrices(n_grid: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    fourier = np.fft.fft(np.eye(n_grid), axis=0) / np.sqrt(n_grid)
    momentum = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=dx)
    return fourier, momentum


def tully_simple_avoided_crossing(x: np.ndarray, p: Parameters) -> np.ndarray:
    v11 = np.where(
        x > 0.0,
        p.tully_a * (1.0 - np.exp(-p.tully_b * x)),
        -p.tully_a * (1.0 - np.exp(p.tully_b * x)),
    )
    v22 = -v11
    v12 = p.tully_c * np.exp(-p.tully_d * x**2)
    potential = np.zeros((len(x), 2, 2), dtype=float)
    potential[:, 0, 0] = v11
    potential[:, 1, 1] = v22
    potential[:, 0, 1] = v12
    potential[:, 1, 0] = v12
    return potential


def thermal_density_matrix(x: np.ndarray, p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = x[1] - x[0]
    fourier, momentum = fft_matrices(len(x), dx)
    p2_operator = fourier.conj().T @ np.diag(momentum**2) @ fourier
    h_ref = p2_operator / (2.0 * p.mass) + np.diag(
        0.5 * p.mass * p.omega_initial**2 * (x - p.x_initial) ** 2
    )
    energy, coeff = scipy.linalg.eigh(0.5 * (h_ref + h_ref.conj().T))
    beta = 1.0 / (p.k_b * p.temperature)
    weights = np.exp(-beta * (energy - energy[0]))
    rho = (coeff * weights) @ coeff.conj().T
    rho /= np.trace(rho)

    selector = 0.5 * (
        1.0
        + np.tanh((momentum - p.momentum_filter_center) / p.momentum_filter_width)
    )
    momentum_filter = fourier.conj().T @ np.diag(selector) @ fourier
    rho = momentum_filter @ rho @ momentum_filter.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho, fourier, momentum


def low_rank_density_components(
    rho: np.ndarray, p: Parameters
) -> tuple[np.ndarray, np.ndarray, float]:
    weights, vectors = scipy.linalg.eigh(rho)
    order = np.argsort(weights)[::-1]
    weights = np.real(weights[order])
    vectors = vectors[:, order]
    keep = weights > p.density_weight_cutoff
    kept_weights = weights[keep]
    kept_vectors = vectors[:, keep]
    retained_weight = float(np.sum(kept_weights))
    kept_weights = kept_weights / retained_weight
    return kept_weights, kept_vectors.T.copy(), retained_weight


def local_adiabatic_basis(potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    energy = np.empty((potential.shape[0], 2), dtype=float)
    basis = np.empty_like(potential)
    for i, matrix in enumerate(potential):
        vals, vecs = np.linalg.eigh(matrix)
        energy[i] = vals
        basis[i] = vecs
    return energy, basis


def lift_to_adiabatic_state(
    nuclear_vectors: np.ndarray, adiabatic_basis: np.ndarray, state: int
) -> np.ndarray:
    electronic = adiabatic_basis[:, :, state]
    return nuclear_vectors[:, :, None] * electronic[None, :, :]


def potential_step_matrix(potential: np.ndarray, dt: float, hbar: float) -> np.ndarray:
    step = np.empty_like(potential, dtype=complex)
    for i, matrix in enumerate(potential):
        vals, vecs = np.linalg.eigh(matrix)
        step[i] = (vecs * np.exp(-1j * vals * dt / hbar)) @ vecs.conj().T
    return step


def split_operator_step(
    psis: np.ndarray,
    kinetic_half_phase: np.ndarray,
    potential_step: np.ndarray,
) -> np.ndarray:
    for electronic_state in range(psis.shape[2]):
        psi_k = np.fft.fft(psis[:, :, electronic_state], axis=1)
        psis[:, :, electronic_state] = np.fft.ifft(
            psi_k * kinetic_half_phase[None, :],
            axis=1,
        )
    psis = np.einsum("iab,nib->nia", potential_step, psis, optimize=True)
    for electronic_state in range(psis.shape[2]):
        psi_k = np.fft.fft(psis[:, :, electronic_state], axis=1)
        psis[:, :, electronic_state] = np.fft.ifft(
            psi_k * kinetic_half_phase[None, :],
            axis=1,
        )
    return psis


def left_keep_mask(x: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = np.ones_like(x)
    mask[x <= start] = 0.0
    ramp = (x > start) & (x < end)
    mask[ramp] = np.sin(0.5 * np.pi * (x[ramp] - start) / (end - start)) ** 2
    return mask


def right_keep_mask(x: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = np.ones_like(x)
    mask[x >= end] = 0.0
    ramp = (x > start) & (x < end)
    mask[ramp] = np.cos(0.5 * np.pi * (x[ramp] - start) / (end - start)) ** 2
    return mask


def adiabatic_populations(
    psis: np.ndarray, weights: np.ndarray, adiabatic_basis: np.ndarray
) -> np.ndarray:
    amplitudes = np.einsum("iak,nia->nik", adiabatic_basis.conj(), psis, optimize=True)
    component_pops = np.sum(np.abs(amplitudes) ** 2, axis=1)
    return np.sum(weights[:, None] * component_pops, axis=0).real


def density_profile(psis: np.ndarray, weights: np.ndarray, dx: float) -> np.ndarray:
    density = np.sum(weights[:, None, None] * np.abs(psis) ** 2, axis=(0, 2)).real
    return density / dx


def weighted_norm(psis: np.ndarray, weights: np.ndarray) -> float:
    component_norms = np.sum(np.abs(psis) ** 2, axis=(1, 2)).real
    return float(np.sum(weights * component_norms))


def absorb_and_count(
    psis: np.ndarray,
    weights: np.ndarray,
    adiabatic_basis: np.ndarray,
    mask_left: np.ndarray,
    mask_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_factor = np.sqrt(np.clip(1.0 - mask_left**2, 0.0, 1.0))
    left_lost = psis * left_factor[None, :, None]
    left_pop = adiabatic_populations(left_lost, weights, adiabatic_basis)
    psis = psis * mask_left[None, :, None]

    right_factor = np.sqrt(np.clip(1.0 - mask_right**2, 0.0, 1.0))
    right_lost = psis * right_factor[None, :, None]
    right_pop = adiabatic_populations(right_lost, weights, adiabatic_basis)
    psis = psis * mask_right[None, :, None]
    return psis, left_pop, right_pop


def run_simulation(p: Parameters) -> dict[str, np.ndarray | float | int]:
    x = np.linspace(p.x_min, p.x_max, p.n_grid, endpoint=False)
    dx = x[1] - x[0]
    rho, _, momentum = thermal_density_matrix(x, p)
    weights, nuclear_vectors, retained_weight = low_rank_density_components(rho, p)

    potential = tully_simple_avoided_crossing(x, p)
    _, adiabatic_basis = local_adiabatic_basis(potential)
    psis = lift_to_adiabatic_state(nuclear_vectors, adiabatic_basis, p.initial_adiabatic_state)

    kinetic_half_phase = np.exp(-0.5j * (momentum**2 / (2.0 * p.mass)) * p.dt / p.hbar)
    potential_step = potential_step_matrix(potential, p.dt, p.hbar)
    mask_left = left_keep_mask(x, p.left_absorb_start, p.left_absorb_end)
    mask_right = right_keep_mask(x, p.right_absorb_start, p.right_absorb_end)

    save_times = np.linspace(0.0, p.t_final, p.n_save)
    save_index = 0
    current_time = 0.0
    cumulative_left = np.zeros(2, dtype=float)
    cumulative_right = np.zeros(2, dtype=float)

    times = []
    remaining_norm = []
    remaining_adiabatic = []
    total_adiabatic = []
    left_history = []
    right_history = []
    trace_check = []
    densities = []
    density_times = [0.0, 0.5 * p.t_final, p.t_final]

    def record() -> None:
        times.append(save_times[save_index])
        remaining = adiabatic_populations(psis, weights, adiabatic_basis)
        remaining_adiabatic.append(remaining)
        total_adiabatic.append(remaining + cumulative_left + cumulative_right)
        left_history.append(cumulative_left.copy())
        right_history.append(cumulative_right.copy())
        remaining_norm.append(weighted_norm(psis, weights))
        trace_check.append(remaining_norm[-1] + np.sum(cumulative_left) + np.sum(cumulative_right))
        if any(np.isclose(save_times[save_index], density_times, atol=0.5 * p.dt)):
            densities.append(density_profile(psis, weights, dx))

    psis, lost_left, lost_right = absorb_and_count(
        psis, weights, adiabatic_basis, mask_left, mask_right
    )
    cumulative_left += lost_left
    cumulative_right += lost_right
    record()

    n_steps = int(np.ceil(p.t_final / p.dt))
    for _ in range(n_steps):
        psis = split_operator_step(psis, kinetic_half_phase, potential_step)
        current_time += p.dt
        psis, lost_left, lost_right = absorb_and_count(
            psis, weights, adiabatic_basis, mask_left, mask_right
        )
        cumulative_left += lost_left
        cumulative_right += lost_right
        while save_index + 1 < len(save_times) and current_time >= save_times[save_index + 1] - 1e-12:
            save_index += 1
            record()

    return {
        "x": x,
        "times": np.array(times),
        "densities": np.array(densities),
        "density_times": np.array(density_times),
        "remaining_norm": np.array(remaining_norm),
        "remaining_adiabatic": np.array(remaining_adiabatic),
        "total_adiabatic": np.array(total_adiabatic),
        "left_history": np.array(left_history),
        "right_history": np.array(right_history),
        "trace_check": np.array(trace_check),
        "retained_weight": retained_weight,
        "n_components": len(weights),
    }


def late_window_drift(values: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    start = int((1.0 - fraction) * len(values))
    window = values[start:]
    return np.max(window, axis=0) - np.min(window, axis=0)


def plot_results(results: dict[str, np.ndarray | float | int]) -> None:
    x = results["x"]
    times = results["times"]
    densities = results["densities"]
    density_times = results["density_times"]
    remaining_norm = results["remaining_norm"]
    remaining_adiabatic = results["remaining_adiabatic"]
    total_adiabatic = results["total_adiabatic"]
    left_history = results["left_history"]
    right_history = results["right_history"]
    trace_check = results["trace_check"]

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6), constrained_layout=True)

    for density, t in zip(densities, density_times):
        axes[0, 0].plot(x, density, label=f"t={t:g}")
    axes[0, 0].axvspan(-34.0, -27.0, color="#777777", alpha=0.12, label="left absorber")
    axes[0, 0].axvspan(9.0, 16.0, color="#2a6fbb", alpha=0.10, label="right absorber")
    axes[0, 0].set_title("Remaining density")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$n(x,t)$")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(times, remaining_norm, color="#333333")
    axes[0, 1].set_title("Residual norm")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("remaining probability")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.25)

    axes[0, 2].plot(times, remaining_adiabatic[:, 0], label="adiabatic 0")
    axes[0, 2].plot(times, remaining_adiabatic[:, 1], label="adiabatic 1")
    axes[0, 2].set_title("Remaining channel populations")
    axes[0, 2].set_xlabel("t")
    axes[0, 2].set_ylabel("population")
    axes[0, 2].legend(frameon=False)

    axes[1, 0].plot(times, left_history[:, 0], color="#1f77b4", label="left, channel 0")
    axes[1, 0].plot(times, left_history[:, 1], color="#ff7f0e", label="left, channel 1")
    axes[1, 0].plot(times, right_history[:, 0], "--", color="#1f77b4", label="right, channel 0")
    axes[1, 0].plot(times, right_history[:, 1], "--", color="#ff7f0e", label="right, channel 1")
    axes[1, 0].set_title("Cumulative absorber ledger")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("population")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(times, total_adiabatic[:, 0], label="channel 0 total")
    axes[1, 1].plot(times, total_adiabatic[:, 1], label="channel 1 total")
    axes[1, 1].set_title("Remaining + absorbed by channel")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("population")
    axes[1, 1].legend(frameon=False)

    axes[1, 2].plot(times, trace_check - 1.0, color="#333333")
    axes[1, 2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set_title("Conservation error")
    axes[1, 2].set_xlabel("t")
    axes[1, 2].set_ylabel(r"$P_{\rm remain}+P_{\rm absorbed}-1$")
    axes[1, 2].grid(True, alpha=0.25)

    fig.savefig(FIGURE_PATH, dpi=180)


def main() -> None:
    params = Parameters()
    results = run_simulation(params)
    plot_results(results)

    left_final = results["left_history"][-1]
    right_final = results["right_history"][-1]
    remaining_final = results["remaining_norm"][-1]
    trace_error_final = results["trace_check"][-1] - 1.0
    drift = late_window_drift(
        np.column_stack(
            [
                results["left_history"],
                results["right_history"],
                results["remaining_norm"],
            ]
        )
    )
    print(f"wrote {FIGURE_PATH}")
    print(f"low-rank components retained = {results['n_components']}")
    print(f"retained density weight = {results['retained_weight']:.12f}")
    print(f"final left outflow channel 0/1 = {left_final[0]:.8f}, {left_final[1]:.8f}")
    print(f"final right outflow channel 0/1 = {right_final[0]:.8f}, {right_final[1]:.8f}")
    print(f"final remaining norm = {remaining_final:.8e}")
    print(f"final conservation error = {trace_error_final:.3e}")
    print(
        "late-window drift "
        "(left0,left1,right0,right1,remaining) = "
        + ", ".join(f"{value:.3e}" for value in drift)
    )


if __name__ == "__main__":
    main()
