#!/usr/bin/env python3
"""Minimal Jahn-Teller model for triangular Li3/Na3-like clusters.

The script is intentionally pedagogical.  It is not a fit to experimental
Li3 or Na3 constants.  It checks two points used in the article:

1. Three equivalent s orbitals on an equilateral triangle form one a1'
   orbital and a doubly degenerate e' pair.
2. One electron in that e' pair gains linear electronic energy under an
   e'-type distortion, while the lattice pays a quadratic elastic cost.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = ROOT / "assets" / "img" / "jt"


def triangular_hamiltonian(alpha: float, beta0: float, coupling: float, q: float) -> np.ndarray:
    """Three-site Huckel Hamiltonian with a zero-sum bond distortion.

    The scalar coordinate q follows one component of the e' distortion.  The
    three hoppings remain equivalent at q=0 and split as q, -q/2, -q/2.
    """

    dt12 = coupling * q
    dt23 = -0.5 * coupling * q
    dt31 = -0.5 * coupling * q
    t12 = beta0 + dt12
    t23 = beta0 + dt23
    t31 = beta0 + dt31
    return np.array(
        [
            [alpha, t12, t31],
            [t12, alpha, t23],
            [t31, t23, alpha],
        ],
        dtype=float,
    )


def electronic_levels(q_values: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues for an illustrative equilateral-to-isosceles path."""

    levels = []
    for q in q_values:
        h = triangular_hamiltonian(alpha=0.0, beta0=-1.0, coupling=0.55, q=q)
        levels.append(np.linalg.eigvalsh(h))
    return np.asarray(levels)


def effective_jt_energy(q_values: np.ndarray, g: float, k: float) -> np.ndarray:
    """One-electron E x e Jahn-Teller energy in arbitrary units."""

    return 0.5 * k * q_values**2 - g * np.abs(q_values)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    q = np.linspace(-1.0, 1.0, 401)
    levels = electronic_levels(q)

    # Toy constants: chosen to make the algebraic mechanism visible, not to
    # describe measured Li3 or Na3 energetics.
    systems = {
        "Li3-like": {"g": 0.45, "k": 1.40},
        "Na3-like": {"g": 0.30, "k": 0.90},
    }
    total = {name: effective_jt_energy(q, **params) for name, params in systems.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(q, levels[:, 0], color="#315f7d", lw=2.0, label="lowest MO")
    ax.plot(q, levels[:, 1], color="#b45f3a", lw=2.0, label="split e' branch")
    ax.plot(q, levels[:, 2], color="#5f7f3a", lw=2.0, label="split e' branch")
    ax.axvline(0.0, color="0.72", lw=1.0)
    ax.set_title("Three s orbitals on a triangle")
    ax.set_xlabel("e' distortion coordinate Q")
    ax.set_ylabel("MO energy / arbitrary units")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for name, energy in total.items():
        ax.plot(q, energy, lw=2.0, label=name)
        params = systems[name]
        q0 = params["g"] / params["k"]
        ax.scatter([q0, -q0], [effective_jt_energy(np.array([q0]), **params)[0]] * 2, s=28)
    ax.axhline(0.0, color="0.72", lw=1.0)
    ax.axvline(0.0, color="0.72", lw=1.0)
    ax.set_title("Linear JT gain plus elastic cost")
    ax.set_xlabel("JT distortion amplitude Q")
    ax.set_ylabel("relative total energy / arbitrary units")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(IMG_DIR / "li3-na3-jt-model.png", dpi=180)

    output = IMG_DIR / "li3-na3-jt-model.csv"
    with output.open("w", encoding="utf-8") as handle:
        handle.write("Q,mo_1,mo_2,mo_3,Li3_like_total,Na3_like_total\n")
        for i, qi in enumerate(q):
            handle.write(
                f"{qi:.8f},{levels[i,0]:.8f},{levels[i,1]:.8f},{levels[i,2]:.8f},"
                f"{total['Li3-like'][i]:.8f},{total['Na3-like'][i]:.8f}\n"
            )

    print(f"Wrote {IMG_DIR / 'li3-na3-jt-model.png'}")
    print(f"Wrote {output}")
    for name, params in systems.items():
        q0 = params["g"] / params["k"]
        ejt = params["g"] ** 2 / (2.0 * params["k"])
        print(f"{name}: q0={q0:.6f}, E_JT={ejt:.6f}")


if __name__ == "__main__":
    main()
