"""Metropolis PIMC benchmark for a one-dimensional harmonic oscillator.

This script supports PIMD Series Part IV.  It implements an unconstrained
path-integral Monte Carlo sampler for the configurational ring-polymer
distribution and checks one observable,

    V_P(q) = (1/P) sum_j 0.5 * m * omega_0**2 * q_j**2,

against the exact finite-P harmonic result and the infinite-P quantum value.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimc-sho-metropolis.png"
RUNNING_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimc-sho-running.csv"
SUMMARY_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimc-sho-summary.csv"

BETA = 2.0
HBAR = 1.0
MASS = 1.0
OMEGA_0 = 4.0
N_BEADS = 32
N_CHAINS = 8
N_SWEEPS = 60000
BURN_SWEEPS = 10000
THIN = 5
INITIAL_STEP_SIZE = 0.7
TARGET_ACCEPTANCE = 0.45
ADAPT_INTERVAL = 250
ADAPT_RATE = 0.35
SEED = 20260528


@dataclass
class PIMCResult:
    samples: np.ndarray
    accept_rate: float
    final_step_size: float


def normal_mode_frequencies(n_beads: int, beta: float = BETA, hbar: float = HBAR) -> np.ndarray:
    """Return ring-polymer spring frequencies omega_k."""
    k = np.arange(n_beads)
    omega_p = n_beads / (beta * hbar)
    return 2.0 * omega_p * np.sin(np.pi * k / n_beads)


def exact_quantum_potential(beta: float = BETA, omega: float = OMEGA_0, hbar: float = HBAR) -> float:
    """Exact infinite-P quantum <V> for a one-dimensional harmonic oscillator."""
    return 0.25 * hbar * omega / np.tanh(0.5 * beta * hbar * omega)


def finite_p_potential(n_beads: int, beta: float = BETA, omega: float = OMEGA_0) -> float:
    """Analytical finite-P ring-polymer expectation of the potential estimator."""
    omega_k = normal_mode_frequencies(n_beads, beta=beta)
    return 0.5 / beta * np.sum(omega**2 / (omega_k**2 + omega**2))


def potential(x: np.ndarray | float) -> np.ndarray | float:
    """Harmonic potential V(x)."""
    return 0.5 * MASS * OMEGA_0**2 * np.asarray(x) ** 2


def local_action(path: np.ndarray, bead: int, x_value: float, beta_p: float) -> float:
    """Action terms affected by a single bead value."""
    prev_value = path[(bead - 1) % path.size]
    next_value = path[(bead + 1) % path.size]
    spring = 0.5 * MASS / (beta_p * HBAR**2) * (
        (x_value - prev_value) ** 2 + (next_value - x_value) ** 2
    )
    return float(spring + beta_p * potential(x_value))


def potential_estimator(path: np.ndarray) -> float:
    """Bead-average potential estimator for one path."""
    return float(np.mean(potential(path)))


def run_pimc_chain(seed: int) -> PIMCResult:
    """Run one adaptive single-bead Metropolis PIMC chain."""
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


def batch_standard_error(values: np.ndarray, n_batches: int = 40) -> float:
    """Batch-means standard error for correlated Monte Carlo samples."""
    values = np.asarray(values, dtype=float)
    batch_size = values.size // n_batches
    if batch_size < 2:
        return float(values.std(ddof=1) / np.sqrt(values.size))
    trimmed = values[: batch_size * n_batches]
    means = trimmed.reshape(n_batches, batch_size).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_batches))


def main() -> None:
    results = [run_pimc_chain(SEED + 7919 * i) for i in range(N_CHAINS)]
    paths = np.concatenate([result.samples for result in results], axis=0)
    potential_samples = np.mean(potential(paths), axis=1)
    q2_samples = np.mean(paths**2, axis=1)

    finite_target = finite_p_potential(N_BEADS)
    exact_target = exact_quantum_potential()
    pimc_mean = float(potential_samples.mean())
    pimc_error = batch_standard_error(potential_samples)
    q2_mean = float(q2_samples.mean())
    q2_error = batch_standard_error(q2_samples)
    q2_finite = 2.0 * finite_target / (MASS * OMEGA_0**2)
    q2_exact = 2.0 * exact_target / (MASS * OMEGA_0**2)
    accept_rate = float(np.mean([result.accept_rate for result in results]))
    final_step = float(np.mean([result.final_step_size for result in results]))

    running = np.cumsum(potential_samples) / np.arange(1, potential_samples.size + 1)
    running_index = np.unique(np.logspace(0, np.log10(potential_samples.size), 180).astype(int))
    running_values = running[running_index - 1]

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    axes[0].plot(running_index, running_values, color="#d62728", lw=1.7, label="PIMC running mean")
    axes[0].axhline(finite_target, color="0.25", lw=1.6, ls="--", label="finite-P target")
    axes[0].axhline(exact_target, color="black", lw=1.8, label="exact quantum")
    axes[0].fill_between(
        running_index,
        pimc_mean - 2.0 * pimc_error,
        pimc_mean + 2.0 * pimc_error,
        color="#d62728",
        alpha=0.14,
        label="final mean +/- 2 SE",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("retained PIMC paths")
    axes[0].set_ylabel(r"$\langle V\rangle$")
    axes[0].set_title(r"SHO observable mean, $P=32$")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    bead_values = paths.reshape(-1)
    x_grid = np.linspace(-1.0, 1.0, 500)
    finite_sigma = np.sqrt(q2_finite)
    exact_sigma = np.sqrt(q2_exact)
    finite_pdf = np.exp(-0.5 * (x_grid / finite_sigma) ** 2) / (np.sqrt(2.0 * np.pi) * finite_sigma)
    exact_pdf = np.exp(-0.5 * (x_grid / exact_sigma) ** 2) / (np.sqrt(2.0 * np.pi) * exact_sigma)

    axes[1].hist(bead_values, bins=80, density=True, color="#9ecae1", edgecolor="none", alpha=0.85, label="PIMC bead marginal")
    axes[1].plot(x_grid, finite_pdf, color="0.25", lw=1.8, ls="--", label="finite-P Gaussian")
    axes[1].plot(x_grid, exact_pdf, color="black", lw=1.6, label="exact Gaussian")
    axes[1].set_xlabel("bead coordinate q")
    axes[1].set_ylabel("probability density")
    axes[1].set_title(r"bead-position marginal")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(FIGURE_PATH, dpi=190)

    RUNNING_PATH.write_text(
        "retained_paths,running_potential_mean\n"
        + "\n".join(f"{int(n)},{v:.10f}" for n, v in zip(running_index, running_values))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    SUMMARY_PATH.write_text(
        "quantity,value,standard_error\n"
        f"pimc_potential_mean,{pimc_mean:.10f},{pimc_error:.10f}\n"
        f"finite_p_potential,{finite_target:.10f},0\n"
        f"exact_quantum_potential,{exact_target:.10f},0\n"
        f"pimc_q2_mean,{q2_mean:.10f},{q2_error:.10f}\n"
        f"finite_p_q2,{q2_finite:.10f},0\n"
        f"exact_quantum_q2,{q2_exact:.10f},0\n"
        f"accept_rate,{accept_rate:.10f},0\n"
        f"final_step_size,{final_step:.10f},0\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {RUNNING_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"PIMC <V> = {pimc_mean:.10f} +/- {pimc_error:.10f}")
    print(f"finite-P <V> = {finite_target:.10f}")
    print(f"exact quantum <V> = {exact_target:.10f}")
    print(f"PIMC <q^2> = {q2_mean:.10f} +/- {q2_error:.10f}")
    print(f"acceptance rate = {accept_rate:.3f}, final step size = {final_step:.3f}")


if __name__ == "__main__":
    main()
