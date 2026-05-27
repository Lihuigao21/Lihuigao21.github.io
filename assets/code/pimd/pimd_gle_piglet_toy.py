"""Mode-friction and covariance-shaping diagnostics for the PIMD series.

This script supports PIMD Series Part II.  It separates two ideas that are
easy to mix together:

1. PILE is a sampling-efficiency idea.  A normal-mode Langevin thermostat uses
   a friction scale matched to each ring-polymer mode, but it leaves the
   canonical finite-P distribution unchanged.
2. PIGLET is a finite-bead-correction idea.  A fitted colored-noise thermostat
   imposes selected non-classical configurational covariances so that chosen
   equilibrium observables can converge with fewer beads.

The second panel is deliberately a toy harmonic-oscillator covariance target,
not a production PIGLET parameterization.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-thermostats.png"
MODE_META_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-mode-diagnostics.csv"
COV_META_PATH = ROOT / "assets" / "img" / "pimd-series" / "pimd-gle-thermostats.csv"

BETA = 2.0
HBAR = 1.0
OMEGA_0 = 4.0


def normal_mode_frequencies(n_beads: int, beta: float = BETA, hbar: float = HBAR) -> np.ndarray:
    """Return ring-polymer spring frequencies omega_k."""
    k = np.arange(n_beads)
    omega_p = n_beads / (beta * hbar)
    return 2.0 * omega_p * np.sin(np.pi * k / n_beads)


def effective_mode_frequencies(n_beads: int) -> np.ndarray:
    """Return SHO normal-mode frequencies Omega_k."""
    omega_k = normal_mode_frequencies(n_beads)
    return np.sqrt(omega_k**2 + OMEGA_0**2)


def exact_quantum_potential(beta: float = BETA, omega: float = OMEGA_0, hbar: float = HBAR) -> float:
    """Exact quantum <V> for a one-dimensional harmonic oscillator."""
    return 0.25 * hbar * omega / np.tanh(0.5 * beta * hbar * omega)


def finite_p_potential(n_beads: int, beta: float = BETA, omega: float = OMEGA_0) -> float:
    """Analytical finite-P ring-polymer expectation of the potential estimator."""
    omega_k = normal_mode_frequencies(n_beads, beta=beta)
    return 0.5 / beta * np.sum(omega**2 / (omega_k**2 + omega**2))


def mode_friction_diagnostics(n_beads: int = 32) -> dict[str, np.ndarray]:
    """Compare a single white friction with PILE-like mode-local friction."""
    omega_k = normal_mode_frequencies(n_beads)
    omega_eff = effective_mode_frequencies(n_beads)
    white_gamma = np.full(n_beads, 2.0 * OMEGA_0)
    pile_gamma = 2.0 * omega_eff
    return {
        "k": np.arange(n_beads),
        "omega_k": omega_k,
        "omega_eff": omega_eff,
        "white_gamma": white_gamma,
        "pile_gamma": pile_gamma,
        "white_ratio": white_gamma / (2.0 * omega_eff),
        "pile_ratio": pile_gamma / (2.0 * omega_eff),
    }


def covariance_diagnostics(n_values: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return canonical finite-P bias and a one-observable covariance target."""
    exact = exact_quantum_potential()
    canonical = np.array([finite_p_potential(int(n)) for n in n_values])
    rel_error = np.abs(canonical - exact) / exact
    scale = exact / canonical
    shaped = canonical * scale
    return {
        "exact": exact,
        "canonical": canonical,
        "rel_error": rel_error,
        "scale": scale,
        "boost": scale - 1.0,
        "shaped": shaped,
    }


def main() -> None:
    mode_data = mode_friction_diagnostics(n_beads=32)
    n_values = np.array([1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    cov_data = covariance_diagnostics(n_values)
    exact = float(cov_data["exact"])

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    axes[0].plot(
        mode_data["k"],
        mode_data["white_ratio"],
        color="0.35",
        marker="o",
        ms=3,
        lw=1.6,
        label=r"single friction $\gamma=2\omega_0$",
    )
    axes[0].plot(
        mode_data["k"],
        mode_data["pile_ratio"],
        color="#d62728",
        marker="s",
        ms=3,
        lw=1.6,
        label=r"PILE-like $\gamma_k=2\Omega_k$",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("normal-mode index k")
    axes[0].set_ylabel(r"thermostat matching ratio $\gamma_k/(2\Omega_k)$")
    axes[0].set_title("PILE matches each mode timescale")
    axes[0].set_ylim(8.0e-3, 1.7)
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    rel_error = np.asarray(cov_data["rel_error"])
    boost = np.asarray(cov_data["boost"])
    axes[1].plot(
        n_values,
        rel_error,
        color="#d62728",
        marker="o",
        lw=1.8,
        label="canonical finite-P bias",
    )
    axes[1].plot(
        n_values,
        boost,
        color="#1f77b4",
        marker="s",
        lw=1.6,
        label=r"toy covariance boost $s_P-1$",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of beads P")
    axes[1].set_ylabel("dimensionless magnitude")
    axes[1].set_title("PIGLET targets finite-P covariance bias")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False)

    p6 = int(np.where(n_values == 6)[0][0])
    axes[1].annotate(
        "P=6: 16.8% canonical bias\nrequires 20.1% covariance boost",
        xy=(6, boost[p6]),
        xytext=(8.7, 0.32),
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0},
        fontsize=9,
        color="0.25",
    )

    fig.savefig(FIGURE_PATH, dpi=190)
    MODE_META_PATH.write_text(
        "k,omega_k,Omega_k,white_gamma,pile_gamma,white_ratio,pile_ratio\n"
        + "\n".join(
            f"{int(k)},{omega:.10f},{omega_eff:.10f},{white:.10f},{pile:.10f},{wr:.10f},{pr:.10f}"
            for k, omega, omega_eff, white, pile, wr, pr in zip(
                mode_data["k"],
                mode_data["omega_k"],
                mode_data["omega_eff"],
                mode_data["white_gamma"],
                mode_data["pile_gamma"],
                mode_data["white_ratio"],
                mode_data["pile_ratio"],
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    COV_META_PATH.write_text(
        "P,canonical_potential,relative_error,toy_covariance_scale,toy_covariance_boost,shaped_potential\n"
        + "\n".join(
            f"{int(n)},{c:.10f},{e:.10e},{s:.10f},{b:.10f},{v:.10f}"
            for n, c, e, s, b, v in zip(
                n_values,
                cov_data["canonical"],
                cov_data["rel_error"],
                cov_data["scale"],
                cov_data["boost"],
                cov_data["shaped"],
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {MODE_META_PATH}")
    print(f"wrote {COV_META_PATH}")
    print(f"exact quantum <V> = {exact:.10f}")
    print(f"P=6 canonical relative error = {rel_error[p6]:.6e}")
    print(f"P=6 toy covariance scale = {np.asarray(cov_data['scale'])[p6]:.6f}")
    print(f"P=32 canonical relative error = {rel_error[np.where(n_values == 32)[0][0]]:.6e}")


if __name__ == "__main__":
    main()
