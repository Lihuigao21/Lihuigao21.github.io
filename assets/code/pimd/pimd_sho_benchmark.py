"""PIMD harmonic-oscillator ensemble benchmark.

This script supports PIMD Series Part I.  It computes the bead-number
convergence of the ring-polymer configurational estimator

    V_P(q) = (1/P) sum_j 0.5 m omega_0^2 q_j^2

for a one-dimensional harmonic oscillator.  The finite-P ring-polymer result
is available analytically in normal modes, and the same estimator is sampled
with independent normal-mode Gaussian draws to mimic the equilibrium output of
a correctly thermostatted PIMD run.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-sho-benchmark.png"
META_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-sho-benchmark.csv"
THRESHOLD_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-sho-convergence-thresholds.csv"

BETA = 2.0
HBAR = 1.0
MASS = 1.0
OMEGA_0 = 4.0
N_SAMPLE = 60000
CONVERGENCE_TOLERANCES = (0.10, 0.01, 0.002, 0.001)


def normal_mode_frequencies(n_beads: int, beta: float = BETA, hbar: float = HBAR) -> np.ndarray:
    """Return ring-polymer spring frequencies omega_k."""
    k = np.arange(n_beads)
    omega_p = n_beads / (beta * hbar)
    return 2.0 * omega_p * np.sin(np.pi * k / n_beads)


def exact_quantum_potential(beta: float = BETA, omega: float = OMEGA_0, hbar: float = HBAR) -> float:
    """Exact quantum <V> for the one-dimensional harmonic oscillator."""
    x = 0.5 * beta * hbar * omega
    return 0.25 * hbar * omega / np.tanh(x)


def finite_p_potential(n_beads: int, beta: float = BETA, omega: float = OMEGA_0) -> float:
    """Analytical finite-P ring-polymer expectation of the potential estimator."""
    omega_k = normal_mode_frequencies(n_beads, beta=beta)
    return 0.5 / beta * np.sum(omega**2 / (omega_k**2 + omega**2))


def sample_potential(
    n_beads: int,
    *,
    beta: float = BETA,
    omega: float = OMEGA_0,
    mass: float = MASS,
    n_sample: int = N_SAMPLE,
    seed: int = 0,
) -> tuple[float, float]:
    """Sample V_P from the exact normal-mode Gaussian distribution."""
    rng = np.random.default_rng(seed)
    beta_p = beta / n_beads
    omega_k = normal_mode_frequencies(n_beads, beta=beta)
    variance = 1.0 / (beta_p * mass * (omega_k**2 + omega**2))
    q_modes = rng.normal(scale=np.sqrt(variance), size=(n_sample, n_beads))
    estimator = 0.5 * mass * omega**2 * np.sum(q_modes**2, axis=1) / n_beads
    return float(estimator.mean()), float(estimator.std(ddof=1) / np.sqrt(n_sample))


def first_converged_p(tolerance: float, *, max_beads: int = 512) -> tuple[int, float]:
    """Return the first integer P whose finite-P error is below tolerance."""
    exact = exact_quantum_potential()
    for n_beads in range(1, max_beads + 1):
        error = abs(finite_p_potential(n_beads) - exact) / exact
        if error <= tolerance:
            return n_beads, float(error)
    raise ValueError(f"no convergence below {tolerance} by P={max_beads}")


def main() -> None:
    n_values = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    dense_p = np.arange(1, 129)
    exact = exact_quantum_potential()
    classical = 0.5 / BETA
    finite = np.array([finite_p_potential(int(n)) for n in n_values])
    finite_dense = np.array([finite_p_potential(int(n)) for n in dense_p])
    sampled = []
    stderr = []

    for i, n_beads in enumerate(n_values):
        mean, err = sample_potential(int(n_beads), seed=100 + i)
        sampled.append(mean)
        stderr.append(err)

    sampled = np.array(sampled)
    stderr = np.array(stderr)
    rel_error = np.abs(finite - exact) / exact
    rel_error_dense = np.abs(finite_dense - exact) / exact
    threshold_rows = [
        (tol, *first_converged_p(tol))
        for tol in CONVERGENCE_TOLERANCES
    ]

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)

    axes[0].axhline(exact, color="black", lw=2.0, label="exact quantum")
    axes[0].axhline(classical, color="0.5", lw=1.3, ls="--", label="classical P = 1 limit")
    axes[0].plot(dense_p, finite_dense, color="#d62728", lw=1.8, label="finite-P formula")
    axes[0].errorbar(
        n_values,
        sampled,
        yerr=2.0 * stderr,
        fmt="s",
        ms=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=2,
        label="normal-mode samples",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("number of beads P")
    axes[0].set_ylabel(r"$\langle V\rangle$")
    axes[0].set_title(r"SHO potential estimator, $\beta=2$, $\omega_0=4$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(dense_p, rel_error_dense, color="#d62728", lw=1.8)
    axes[1].scatter(n_values, rel_error, color="#d62728", s=22, zorder=3)
    for tol, p_required, _ in threshold_rows:
        if tol not in (0.01, 0.002):
            continue
        label = "1%" if tol == 0.01 else "0.2%"
        axes[1].axhline(tol, color="0.45", lw=1.0, ls=":")
        axes[1].axvline(p_required, color="0.45", lw=1.0, ls=":")
        axes[1].text(
            p_required * 1.04,
            tol * 1.15,
            f"{label}: P={p_required}",
            color="0.25",
            fontsize=9,
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of beads P")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("finite-P convergence")
    axes[1].grid(True, alpha=0.25, which="both")

    fig.savefig(FIGURE_PATH, dpi=190)
    META_PATH.write_text(
        "P,finite_potential,sampled_potential,two_sigma_error,relative_error\n"
        + "\n".join(
            f"{int(n)},{f:.10f},{s:.10f},{2*e:.10f},{r:.10e}"
            for n, f, s, e, r in zip(n_values, finite, sampled, stderr, rel_error)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    THRESHOLD_PATH.write_text(
        "relative_error_tolerance,first_converged_P,actual_relative_error\n"
        + "\n".join(
            f"{tol:.6g},{int(p_required)},{actual:.10e}"
            for tol, p_required, actual in threshold_rows
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {META_PATH}")
    print(f"wrote {THRESHOLD_PATH}")
    print(f"exact quantum <V> = {exact:.10f}")
    print(f"classical <V> = {classical:.10f}")
    for tol, p_required, actual in threshold_rows:
        print(f"first P below {tol:.3g} relative error = {p_required} ({actual:.6e})")


if __name__ == "__main__":
    main()
