"""Sampling and finite-bead benchmarks for PIMD thermostats.

This script supports PIMD Series Part II.  It separates the numerical role of
two thermostat ideas:

1. PILE improves sampling efficiency at fixed bead number by assigning a
   near-critical Langevin friction to each ring-polymer normal mode.
2. PIGLET-style GLEs reduce finite-bead error for selected equilibrium
   observables by fitting non-classical normal-mode covariances.

The PIGLET panel is a transparent harmonic-oscillator toy model, not a
production colored-noise parameterization.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-thermostats.png"
SAMPLING_META_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-thermostat-sampling-convergence.csv"
BEAD_META_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-thermostats.csv"

BETA = 2.0
HBAR = 1.0
MASS = 1.0
OMEGA_0 = 4.0
P_SAMPLING = 32
DT = 0.006
N_STEPS = 50000
N_REPLICA = 128
WHITE_GAMMA = 1.0
MAX_TOY_COVARIANCE_BOOST = 0.25
PLOT_ERROR_FLOOR = 1.0e-4


def normal_mode_frequencies(n_beads: int, beta: float = BETA, hbar: float = HBAR) -> np.ndarray:
    """Return ring-polymer spring frequencies omega_k."""
    k = np.arange(n_beads)
    omega_p = n_beads / (beta * hbar)
    return 2.0 * omega_p * np.sin(np.pi * k / n_beads)


def effective_mode_frequencies(n_beads: int) -> np.ndarray:
    """Return harmonic-oscillator ring-polymer frequencies Omega_k."""
    omega_k = normal_mode_frequencies(n_beads)
    return np.sqrt(omega_k**2 + OMEGA_0**2)


def exact_quantum_potential(beta: float = BETA, omega: float = OMEGA_0, hbar: float = HBAR) -> float:
    """Exact quantum <V> for a one-dimensional harmonic oscillator."""
    return 0.25 * hbar * omega / np.tanh(0.5 * beta * hbar * omega)


def finite_p_potential(n_beads: int, beta: float = BETA, omega: float = OMEGA_0) -> float:
    """Analytical finite-P ring-polymer expectation of the potential estimator."""
    omega_k = normal_mode_frequencies(n_beads, beta=beta)
    return 0.5 / beta * np.sum(omega**2 / (omega_k**2 + omega**2))


def first_below(values: np.ndarray, threshold: float, p_values: np.ndarray) -> int:
    """Return the first P whose value is at or below threshold."""
    matches = np.flatnonzero(values <= threshold)
    if len(matches) == 0:
        raise ValueError(f"no value below {threshold}")
    return int(p_values[matches[0]])


def sampling_convergence(
    *,
    scheme: str,
    n_beads: int = P_SAMPLING,
    n_steps: int = N_STEPS,
    n_replica: int = N_REPLICA,
    dt: float = DT,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cold-start convergence of internal-mode variances.

    The observable is the normalized internal-mode variance

        A = mean_{k>0} Q_k^2 / <Q_k^2>,

    whose canonical finite-P expectation is 1.  The returned curve is the RMS
    running-mean error over independent replicas.
    """
    rng = np.random.default_rng(seed)
    beta_p = BETA / n_beads
    omega_eff = effective_mode_frequencies(n_beads)

    if scheme == "white":
        gamma = np.full(n_beads, WHITE_GAMMA)
    elif scheme == "pile":
        gamma = 2.0 * omega_eff
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    q = np.zeros((n_replica, n_beads))
    p = rng.normal(scale=np.sqrt(MASS / beta_p), size=(n_replica, n_beads))
    c = np.exp(-gamma * dt)
    sigma_p = np.sqrt(MASS / beta_p * (1.0 - c**2))
    q_variance = 1.0 / (beta_p * MASS * omega_eff**2)
    checkpoints = np.unique(np.logspace(1, np.log10(n_steps), 140).astype(int))

    running_sum = np.zeros(n_replica)
    rms_error = []
    checkpoint_index = 0

    for step in range(1, n_steps + 1):
        p -= 0.5 * dt * MASS * omega_eff**2 * q
        q += 0.5 * dt * p / MASS
        p = c * p + sigma_p * rng.normal(size=(n_replica, n_beads))
        q += 0.5 * dt * p / MASS
        p -= 0.5 * dt * MASS * omega_eff**2 * q

        normalized_internal_variance = np.mean(q[:, 1:] ** 2 / q_variance[1:], axis=1)
        running_sum += normalized_internal_variance

        if checkpoint_index < len(checkpoints) and step == checkpoints[checkpoint_index]:
            running_mean = running_sum / step
            rms_error.append(np.sqrt(np.mean((running_mean - 1.0) ** 2)))
            checkpoint_index += 1

    return checkpoints, np.asarray(rms_error)


def bead_convergence(n_values: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Return canonical and toy covariance-shaped finite-P errors."""
    exact = exact_quantum_potential()
    canonical = np.array([finite_p_potential(int(n)) for n in n_values])
    canonical_error = np.abs(canonical - exact) / exact

    required_boost = exact / canonical - 1.0
    applied_boost = np.minimum(required_boost, MAX_TOY_COVARIANCE_BOOST)
    shaped = canonical * (1.0 + applied_boost)
    shaped_error = np.abs(shaped - exact) / exact

    dense_p = np.arange(1, 129)
    dense_canonical = np.array([finite_p_potential(int(n)) for n in dense_p])
    dense_canonical_error = np.abs(dense_canonical - exact) / exact
    dense_required_boost = exact / dense_canonical - 1.0
    dense_applied_boost = np.minimum(dense_required_boost, MAX_TOY_COVARIANCE_BOOST)
    dense_shaped_error = np.abs(dense_canonical * (1.0 + dense_applied_boost) - exact) / exact

    return {
        "exact": exact,
        "canonical": canonical,
        "canonical_error": canonical_error,
        "required_boost": required_boost,
        "applied_boost": applied_boost,
        "shaped": shaped,
        "shaped_error": shaped_error,
        "dense_p": dense_p,
        "dense_canonical_error": dense_canonical_error,
        "dense_shaped_error": dense_shaped_error,
        "canonical_p_1pct": first_below(dense_canonical_error, 0.01, dense_p),
        "shaped_p_1pct": first_below(dense_shaped_error, 0.01, dense_p),
    }


def first_checkpoint_below(checkpoints: np.ndarray, errors: np.ndarray, threshold: float) -> int | None:
    matches = np.flatnonzero(errors <= threshold)
    return None if len(matches) == 0 else int(checkpoints[matches[0]])


def main() -> None:
    white_x, white_error = sampling_convergence(scheme="white", seed=3)
    pile_x, pile_error = sampling_convergence(scheme="pile", seed=3)

    n_values = np.array([1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    bead_data = bead_convergence(n_values)
    exact = float(bead_data["exact"])

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    axes[0].plot(
        white_x,
        white_error,
        color="0.25",
        lw=1.8,
        label=rf"single Langevin friction $\gamma={WHITE_GAMMA:g}$",
    )
    axes[0].plot(
        pile_x,
        pile_error,
        color="#d62728",
        lw=1.8,
        label=r"PILE-like $\gamma_k=2\Omega_k$",
    )
    axes[0].axhline(0.05, color="0.55", lw=1.0, ls=":", label="5% RMS error")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("MD samples from cold start")
    axes[0].set_ylabel("RMS error of normalized internal variance")
    axes[0].set_title(r"PILE accelerates fixed-$P$ sampling")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    dense_p = np.asarray(bead_data["dense_p"])
    canonical_curve = np.asarray(bead_data["dense_canonical_error"])
    shaped_curve = np.maximum(np.asarray(bead_data["dense_shaped_error"]), PLOT_ERROR_FLOOR)
    canonical_p_1pct = int(bead_data["canonical_p_1pct"])
    shaped_p_1pct = int(bead_data["shaped_p_1pct"])

    axes[1].plot(
        dense_p,
        canonical_curve,
        color="#d62728",
        lw=1.8,
        label="canonical finite-P PIMD",
    )
    axes[1].plot(
        dense_p,
        shaped_curve,
        color="#1f77b4",
        lw=1.7,
        label="toy PIGLET-style covariance target",
    )
    axes[1].scatter(n_values, np.asarray(bead_data["canonical_error"]), color="#d62728", s=18)
    axes[1].scatter(
        n_values,
        np.maximum(np.asarray(bead_data["shaped_error"]), PLOT_ERROR_FLOOR),
        color="#1f77b4",
        s=18,
    )
    axes[1].axhline(0.01, color="0.45", lw=1.0, ls=":", label="1% error")
    axes[1].axvline(shaped_p_1pct, color="#1f77b4", lw=1.0, ls=":")
    axes[1].axvline(canonical_p_1pct, color="#d62728", lw=1.0, ls=":")
    axes[1].annotate(
        f"P={shaped_p_1pct}",
        xy=(shaped_p_1pct, 0.01),
        xytext=(shaped_p_1pct * 1.12, 0.018),
        color="#1f77b4",
        fontsize=9,
    )
    axes[1].annotate(
        f"P={canonical_p_1pct}",
        xy=(canonical_p_1pct, 0.01),
        xytext=(canonical_p_1pct * 1.06, 0.018),
        color="#d62728",
        fontsize=9,
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of beads P")
    axes[1].set_ylabel(r"relative error in $\langle V\rangle$")
    axes[1].set_title("Covariance target reduces bead bias")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False)

    fig.savefig(FIGURE_PATH, dpi=190)

    SAMPLING_META_PATH.write_text(
        "samples,white_rms_error,pile_rms_error\n"
        + "\n".join(
            f"{int(n)},{w:.10e},{p:.10e}"
            for n, w, p in zip(white_x, white_error, pile_error)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    BEAD_META_PATH.write_text(
        "P,canonical_potential,canonical_relative_error,required_covariance_boost,"
        "applied_covariance_boost,toy_shaped_potential,toy_shaped_relative_error\n"
        + "\n".join(
            f"{int(n)},{c:.10f},{ce:.10e},{rb:.10f},{ab:.10f},{s:.10f},{se:.10e}"
            for n, c, ce, rb, ab, s, se in zip(
                n_values,
                bead_data["canonical"],
                bead_data["canonical_error"],
                bead_data["required_boost"],
                bead_data["applied_boost"],
                bead_data["shaped"],
                bead_data["shaped_error"],
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    white_5pct = first_checkpoint_below(white_x, white_error, 0.05)
    pile_5pct = first_checkpoint_below(pile_x, pile_error, 0.05)
    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {SAMPLING_META_PATH}")
    print(f"wrote {BEAD_META_PATH}")
    print(f"exact quantum <V> = {exact:.10f}")
    print(f"5% internal-variance RMS error: white={white_5pct}, pile={pile_5pct} samples")
    print(f"1% <V> error: canonical P={canonical_p_1pct}, toy covariance target P={shaped_p_1pct}")


if __name__ == "__main__":
    main()
