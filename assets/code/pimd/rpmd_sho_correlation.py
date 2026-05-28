"""PIMC-initialized RPMD benchmark for a harmonic q-q correlation.

This script supports PIMD Series Part V.  It first samples the finite-P
harmonic-oscillator ring-polymer configuration distribution with local
Metropolis PIMC.  It then attaches canonical RPMD bead momenta and propagates
the ring-polymer Hamiltonian in normal modes to estimate the centroid
position autocorrelation,

    C_qq(t) = < q_c(0) q_c(t) >.

For a harmonic oscillator this RPMD result should match the exact
Kubo-transformed correlation,

    C_qq^K(t) = cos(omega_0 t) / (beta * m * omega_0**2).

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "rpmd-sho-correlation.png"
CORRELATION_PATH = ROOT / "assets" / "img" / "pimd-series" / "rpmd-sho-correlation.csv"
SUMMARY_PATH = ROOT / "assets" / "img" / "pimd-series" / "rpmd-sho-summary.csv"

BETA = 2.0
HBAR = 1.0
MASS = 1.0
OMEGA_0 = 4.0
N_BEADS = 32

N_CHAINS = 6
N_SWEEPS = 50000
BURN_SWEEPS = 10000
THIN = 5
INITIAL_STEP_SIZE = 0.65
TARGET_ACCEPTANCE = 0.45
ADAPT_INTERVAL = 250
ADAPT_RATE = 0.35

N_TIMES = 321
T_MAX = 6.0
SEED = 20260529


@dataclass
class PIMCResult:
    samples: np.ndarray
    accept_rate: float
    final_step_size: float


def potential(x: np.ndarray | float) -> np.ndarray | float:
    """Harmonic potential V(x)."""
    return 0.5 * MASS * OMEGA_0**2 * np.asarray(x) ** 2


def local_action(path: np.ndarray, bead: int, x_value: float, beta_p: float) -> float:
    """Dimensionless action terms affected by one bead value."""
    prev_value = path[(bead - 1) % path.size]
    next_value = path[(bead + 1) % path.size]
    spring = 0.5 * MASS / (beta_p * HBAR**2) * (
        (x_value - prev_value) ** 2 + (next_value - x_value) ** 2
    )
    return float(spring + beta_p * potential(x_value))


def run_pimc_chain(seed: int) -> PIMCResult:
    """Run one adaptive local Metropolis chain for the configurational path."""
    rng = np.random.default_rng(seed)
    beta_p = BETA / N_BEADS
    path = np.zeros(N_BEADS, dtype=float)
    log_step = np.log(INITIAL_STEP_SIZE)
    samples: list[np.ndarray] = []

    accept_total = 0
    trial_total = 0
    accept_window = 0
    trial_window = 0

    for sweep in range(N_SWEEPS):
        for bead in rng.permutation(N_BEADS):
            step_size = float(np.exp(log_step))
            old_value = path[bead]
            new_value = old_value + rng.uniform(-step_size, step_size)

            old_action = local_action(path, int(bead), old_value, beta_p)
            new_action = local_action(path, int(bead), new_value, beta_p)
            delta_action = new_action - old_action

            accepted = delta_action <= 0.0 or rng.random() < np.exp(-delta_action)
            trial_total += 1
            trial_window += 1
            if accepted:
                path[bead] = new_value
                accept_total += 1
                accept_window += 1

        if sweep < BURN_SWEEPS and (sweep + 1) % ADAPT_INTERVAL == 0:
            rate = accept_window / max(1, trial_window)
            log_step += ADAPT_RATE * (rate - TARGET_ACCEPTANCE)
            log_step = float(np.clip(log_step, np.log(1.0e-4), np.log(5.0)))
            accept_window = 0
            trial_window = 0

        if sweep >= BURN_SWEEPS and (sweep - BURN_SWEEPS) % THIN == 0:
            samples.append(path.copy())

    return PIMCResult(
        samples=np.asarray(samples),
        accept_rate=accept_total / max(1, trial_total),
        final_step_size=float(np.exp(log_step)),
    )


def spring_eigensystem(n_beads: int) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and orthonormal eigenvectors of the RP spring matrix."""
    matrix = 2.0 * np.eye(n_beads)
    matrix -= np.roll(np.eye(n_beads), 1, axis=0)
    matrix -= np.roll(np.eye(n_beads), -1, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def sample_rpmd_momenta(n_paths: int, seed: int) -> np.ndarray:
    """Sample bead momenta from exp[-beta_P sum_j p_j^2/(2m)]."""
    rng = np.random.default_rng(seed)
    beta_p = BETA / N_BEADS
    sigma = np.sqrt(MASS / beta_p)
    return rng.normal(0.0, sigma, size=(n_paths, N_BEADS))


def rpmd_centroid_correlation(
    paths: np.ndarray,
    momenta: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Propagate harmonic RPMD normal modes and compute <q_c(0) q_c(t)>."""
    eigenvalues, eigenvectors = spring_eigensystem(N_BEADS)
    omega_p = N_BEADS / (BETA * HBAR)
    mode_frequencies = np.sqrt(OMEGA_0**2 + omega_p**2 * eigenvalues)

    q_modes = paths @ eigenvectors
    p_modes = momenta @ eigenvectors
    centroid_weights = eigenvectors.sum(axis=0) / N_BEADS

    active = np.abs(centroid_weights) > 1.0e-12
    q_modes = q_modes[:, active]
    p_modes = p_modes[:, active]
    mode_frequencies = mode_frequencies[active]
    centroid_weights = centroid_weights[active]

    q_centroid_initial = paths.mean(axis=1)
    correlation = np.empty_like(times, dtype=float)

    for index, time in enumerate(times):
        cos_term = np.cos(mode_frequencies * time)
        sin_term = np.sin(mode_frequencies * time)
        q_modes_t = q_modes * cos_term + p_modes * sin_term / (MASS * mode_frequencies)
        q_centroid_t = q_modes_t @ centroid_weights
        correlation[index] = float(np.mean(q_centroid_initial * q_centroid_t))

    return correlation


def exact_kubo_correlation(times: np.ndarray) -> np.ndarray:
    """Exact Kubo-transformed q-q correlation for the harmonic oscillator."""
    amplitude = 1.0 / (BETA * MASS * OMEGA_0**2)
    return amplitude * np.cos(OMEGA_0 * times)


def batch_standard_error(values: np.ndarray, n_batches: int = 32) -> float:
    """Batch-means standard error for correlated Monte Carlo samples."""
    values = np.asarray(values, dtype=float)
    batch_size = values.size // n_batches
    if batch_size < 2:
        return float(values.std(ddof=1) / np.sqrt(values.size))
    trimmed = values[: batch_size * n_batches]
    means = trimmed.reshape(n_batches, batch_size).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_batches))


def main() -> None:
    results = [run_pimc_chain(SEED + 7919 * chain) for chain in range(N_CHAINS)]
    paths = np.concatenate([result.samples for result in results], axis=0)
    momenta = sample_rpmd_momenta(paths.shape[0], seed=SEED + 404)

    times = np.linspace(0.0, T_MAX, N_TIMES)
    rpmd_corr = rpmd_centroid_correlation(paths, momenta, times)
    exact_corr = exact_kubo_correlation(times)
    error = rpmd_corr - exact_corr

    centroids = paths.mean(axis=1)
    centroid_variance = float(np.mean(centroids**2))
    centroid_se = batch_standard_error(centroids**2)
    exact_c0 = 1.0 / (BETA * MASS * OMEGA_0**2)
    rms_error = float(np.sqrt(np.mean(error**2)))
    max_abs_error = float(np.max(np.abs(error)))
    accept_rate = float(np.mean([result.accept_rate for result in results]))
    final_step = float(np.mean([result.final_step_size for result in results]))

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    x_grid = np.linspace(-0.75, 0.75, 500)
    sigma_qc = np.sqrt(exact_c0)
    gaussian = np.exp(-0.5 * (x_grid / sigma_qc) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma_qc
    )
    axes[0].hist(
        centroids,
        bins=70,
        density=True,
        color="#9ecae1",
        edgecolor="none",
        alpha=0.85,
        label="PIMC centroids",
    )
    axes[0].plot(x_grid, gaussian, color="black", lw=1.8, label="exact centroid marginal")
    axes[0].set_xlabel("centroid coordinate q_c")
    axes[0].set_ylabel("probability density")
    axes[0].set_title("PIMC initial ensemble")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(times, exact_corr, color="black", lw=1.8, label="exact Kubo")
    axes[1].plot(times, rpmd_corr, color="#d62728", lw=1.4, ls="--", label="PIMC + RPMD")
    axes[1].axhline(0.0, color="0.65", lw=0.9)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$C_{qq}(t)$")
    axes[1].set_title("SHO q-q correlation")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(FIGURE_PATH, dpi=190)

    CORRELATION_PATH.write_text(
        "time,rpmd_correlation,exact_kubo,error\n"
        + "\n".join(
            f"{time:.10f},{rpmd:.10f},{exact:.10f},{err:.10f}"
            for time, rpmd, exact, err in zip(times, rpmd_corr, exact_corr, error)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    SUMMARY_PATH.write_text(
        "quantity,value,standard_error\n"
        f"n_paths,{paths.shape[0]},0\n"
        f"n_beads,{N_BEADS},0\n"
        f"pimc_accept_rate,{accept_rate:.10f},0\n"
        f"pimc_final_step_size,{final_step:.10f},0\n"
        f"centroid_variance_pimc,{centroid_variance:.10f},{centroid_se:.10f}\n"
        f"centroid_variance_exact,{exact_c0:.10f},0\n"
        f"correlation_c0_rpmd,{rpmd_corr[0]:.10f},0\n"
        f"correlation_c0_exact,{exact_corr[0]:.10f},0\n"
        f"rms_correlation_error,{rms_error:.10f},0\n"
        f"max_abs_correlation_error,{max_abs_error:.10f},0\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {CORRELATION_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"PIMC centroid variance = {centroid_variance:.10f} +/- {centroid_se:.10f}")
    print(f"exact Kubo C(0) = {exact_c0:.10f}")
    print(f"RPMD C(0) = {rpmd_corr[0]:.10f}")
    print(f"RMS correlation error = {rms_error:.10f}")
    print(f"max abs correlation error = {max_abs_error:.10f}")
    print(f"acceptance rate = {accept_rate:.3f}, final step size = {final_step:.3f}")


if __name__ == "__main__":
    main()
