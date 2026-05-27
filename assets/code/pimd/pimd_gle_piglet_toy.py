"""Small NVT/GLE/PIGLET-style benchmarks for the PIMD series.

The script has two deliberately compact parts:

1. A normal-mode Langevin PIMD sampler for a one-dimensional harmonic
   oscillator.  It compares a single white-noise friction with a PILE-like
   local choice gamma_k = 2 Omega_k for the internal modes.
2. A toy covariance-shaping calculation.  It rescales the finite-P normal-mode
   configurational covariance so that one target observable, <V>, matches the
   exact quantum harmonic oscillator value.  This is not a production PIGLET
   parameter file; it is a minimal reproducible demonstration of the idea that
   GLE/PIGLET thermostats can target quantum fluctuations with fewer beads.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-thermostats.png"
META_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-thermostats.csv"

BETA = 2.0
HBAR = 1.0
MASS = 1.0
OMEGA_0 = 4.0


def normal_mode_frequencies(n_beads: int, beta: float = BETA, hbar: float = HBAR) -> np.ndarray:
    k = np.arange(n_beads)
    omega_p = n_beads / (beta * hbar)
    return 2.0 * omega_p * np.sin(np.pi * k / n_beads)


def exact_quantum_potential(beta: float = BETA, omega: float = OMEGA_0, hbar: float = HBAR) -> float:
    return 0.25 * hbar * omega / np.tanh(0.5 * beta * hbar * omega)


def finite_p_potential(n_beads: int, beta: float = BETA, omega: float = OMEGA_0) -> float:
    omega_k = normal_mode_frequencies(n_beads, beta=beta)
    return 0.5 / beta * np.sum(omega**2 / (omega_k**2 + omega**2))


def langevin_running_error(
    *,
    scheme: str,
    n_beads: int = 32,
    n_steps: int = 70000,
    burn: int = 5000,
    dt: float = 0.012,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return running absolute error of the sampled SHO potential estimator."""
    rng = np.random.default_rng(seed)
    beta_p = BETA / n_beads
    omega_k = normal_mode_frequencies(n_beads)
    omega_total = np.sqrt(omega_k**2 + OMEGA_0**2)

    if scheme == "white":
        gamma = np.full(n_beads, 0.35)
    elif scheme == "pile":
        gamma = 2.0 * omega_total
        gamma[0] = 1.0
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    q = np.zeros(n_beads)
    p = rng.normal(scale=np.sqrt(MASS / beta_p), size=n_beads)
    c = np.exp(-gamma * dt)
    sigma_p = np.sqrt(MASS / beta_p * (1.0 - c**2))
    exact = exact_quantum_potential()

    samples = []
    for step in range(n_steps):
        p -= 0.5 * dt * MASS * omega_total**2 * q
        q += 0.5 * dt * p / MASS
        p = c * p + sigma_p * rng.normal(size=n_beads)
        q += 0.5 * dt * p / MASS
        p -= 0.5 * dt * MASS * omega_total**2 * q

        if step >= burn:
            samples.append(0.5 * MASS * OMEGA_0**2 * np.sum(q**2) / n_beads)

    samples = np.asarray(samples)
    running = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    checkpoints = np.unique(np.logspace(1, np.log10(len(samples)), 140).astype(int))
    return checkpoints, np.abs(running[checkpoints - 1] - exact)


def covariance_shaped_potential(n_beads: int) -> tuple[float, float]:
    """Return the toy GLE/PIGLET-style corrected value and its covariance scale."""
    finite = finite_p_potential(n_beads)
    exact = exact_quantum_potential()
    scale = exact / finite
    return finite * scale, scale


def main() -> None:
    white_x, white_err = langevin_running_error(scheme="white", seed=2)
    pile_x, pile_err = langevin_running_error(scheme="pile", seed=2)

    n_values = np.array([1, 2, 4, 6, 8, 16, 32])
    exact = exact_quantum_potential()
    canonical = np.array([finite_p_potential(int(n)) for n in n_values])
    shaped_pairs = [covariance_shaped_potential(int(n)) for n in n_values]
    shaped = np.array([pair[0] for pair in shaped_pairs])
    scales = np.array([pair[1] for pair in shaped_pairs])

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)

    axes[0].plot(white_x, white_err, color="0.25", lw=1.8, label="single white friction")
    axes[0].plot(pile_x, pile_err, color="#d62728", lw=1.8, label=r"PILE-like $\gamma_k=2\Omega_k$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("post-burn-in samples")
    axes[0].set_ylabel(r"$|\overline{V}-\langle V\rangle_{\rm exact}|$")
    axes[0].set_title("NVT sampling of ring-polymer modes")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].axhline(exact, color="black", lw=2.0, label="exact quantum")
    axes[1].plot(n_values, canonical, color="#d62728", marker="o", lw=1.8, label="canonical finite-P")
    axes[1].plot(n_values, shaped, color="#1f77b4", marker="s", lw=1.6, label="toy GLE/PIGLET target")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("number of beads P")
    axes[1].set_ylabel(r"$\langle V\rangle$")
    axes[1].set_title("Covariance shaping for one SHO observable")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(FIGURE_PATH, dpi=190)
    META_PATH.write_text(
        "P,canonical_potential,toy_shaped_potential,covariance_scale\n"
        + "\n".join(
            f"{int(n)},{c:.10f},{s:.10f},{r:.10f}"
            for n, c, s, r in zip(n_values, canonical, shaped, scales)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {META_PATH}")
    print(f"exact quantum <V> = {exact:.10f}")
    print(f"final white-friction error = {white_err[-1]:.6e}")
    print(f"final PILE-like error = {pile_err[-1]:.6e}")
    print(f"P=6 toy covariance scale = {scales[list(n_values).index(6)]:.6f}")


if __name__ == "__main__":
    main()
