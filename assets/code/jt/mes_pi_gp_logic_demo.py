"""Minimal diagnostics for the MES-PI geometric-phase logic.

This script does not reproduce the full MES-PIMD simulations of Zhai,
Shang, and Liu. It isolates two mechanisms that the paper uses:

1. A closed imaginary-time path around a conical intersection acquires a
   Wilson-loop sign from adjacent electronic overlaps.
2. A GP-induced half-integer angular shift changes the low-temperature
   spectral thermodynamics.

Run from the repository root:

    python assets/code/jt/mes_pi_gp_logic_demo.py

Outputs:

    assets/img/jt/mes-pi-gp-logic-demo.png
    assets/img/jt/mes-pi-gp-logic-wilson.csv
    assets/img/jt/mes-pi-gp-logic-thermo.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def single_valued_real_state(theta: float) -> np.ndarray:
    """A real lower-state eigenvector with a branch cut at theta = 0.

    The coordinate theta is folded to [0, 2*pi). The state is single-valued
    as a function of the nuclear coordinate, so a path crossing the branch cut
    carries the sign flip in the overlap with the next bead.
    """

    folded = np.mod(theta, 2.0 * np.pi)
    return np.array([np.cos(0.5 * folded), -np.sin(0.5 * folded)])


def overlap_product(bead_count: int, winding: int) -> float:
    if winding == 0:
        thetas = np.zeros(bead_count + 1)
    else:
        thetas = np.linspace(0.0, 2.0 * np.pi * winding, bead_count + 1)

    product = 1.0
    for theta_a, theta_b in zip(thetas[:-1], thetas[1:]):
        product *= float(np.dot(single_valued_real_state(theta_a), single_valued_real_state(theta_b)))
    return product


def heat_capacity(beta: np.ndarray, energies: np.ndarray) -> np.ndarray:
    shifted = energies - np.min(energies)
    weights = np.exp(-beta[:, None] * shifted[None, :])
    partition = weights.sum(axis=1)
    mean = (weights * shifted[None, :]).sum(axis=1) / partition
    mean2 = (weights * shifted[None, :] ** 2).sum(axis=1) / partition
    return beta**2 * (mean2 - mean**2)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "assets" / "img" / "jt"
    out_dir.mkdir(parents=True, exist_ok=True)

    bead_counts = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    windings = np.array([0, 1, 2])
    products = np.array([[overlap_product(int(n), int(w)) for w in windings] for n in bead_counts])

    wilson_rows = np.column_stack([bead_counts, products])
    with open(out_dir / "mes-pi-gp-logic-wilson.csv", "w", encoding="utf-8", newline="\n") as handle:
        np.savetxt(
            handle,
            wilson_rows,
            delimiter=",",
            header="N_bead,overlap_product_W0,overlap_product_W1,overlap_product_W2",
            comments="",
        )

    m = np.arange(-8, 9, dtype=float)
    energies_no_gp = 0.5 * m**2
    energies_gp = 0.5 * (m + 0.5) ** 2
    beta = np.linspace(0.4, 20.0, 500)
    cv_no_gp = heat_capacity(beta, energies_no_gp)
    cv_gp = heat_capacity(beta, energies_gp)

    thermo_rows = np.column_stack([beta, cv_no_gp, cv_gp])
    with open(out_dir / "mes-pi-gp-logic-thermo.csv", "w", encoding="utf-8", newline="\n") as handle:
        np.savetxt(
            handle,
            thermo_rows,
            delimiter=",",
            header="beta,Cv_no_GP_over_kB,Cv_GP_over_kB",
            comments="",
        )

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    colors = ["#4f6f52", "#c23b32", "#2764ad"]
    for idx, winding in enumerate(windings):
        ax.plot(
            bead_counts,
            products[:, idx],
            marker="o",
            lw=2.0,
            color=colors[idx],
            label=f"W = {winding}",
        )
    ax.axhline(1.0, color="0.65", ls=":", lw=1.2)
    ax.axhline(-1.0, color="0.65", ls=":", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("number of beads")
    ax.set_ylabel("overlap product")
    ax.set_title("Overlap Wilson loop")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    m_small = np.arange(-5, 6)
    ax.scatter(m_small, 0.5 * m_small**2, color="#2764ad", s=36, label="GP excluded")
    ax.scatter(m_small, 0.5 * (m_small + 0.5) ** 2, color="#c23b32", marker="s", s=36, label="GP included")
    ax.set_xlabel("integer index m")
    ax.set_ylabel("angular energy")
    ax.set_title("Low angular spectrum")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(beta, cv_no_gp, color="#2764ad", lw=2.2, label="GP excluded")
    ax.plot(beta, cv_gp, color="#c23b32", lw=2.2, label="GP included")
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"$C_V/k_B$")
    ax.set_title("Toy heat-capacity response")
    ax.set_xlim(beta.min(), beta.max())
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    w = np.arange(0, 5)
    mes_sign = (-1.0) ** w
    cancel_factor = (-1.0) ** w
    excluded_sign = mes_sign * cancel_factor
    width = 0.26
    ax.bar(w - width, mes_sign, width=width, color="#c23b32", label="MES-PI sign")
    ax.bar(w, cancel_factor, width=width, color="#6d5fb3", label="cancelling factor")
    ax.bar(w + width, excluded_sign, width=width, color="#2764ad", label="after cancellation")
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xticks(w)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("winding number W")
    ax.set_ylabel("phase factor")
    ax.set_title("Artificial GP-excluded control")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out_dir / "mes-pi-gp-logic-demo.png", dpi=180)


if __name__ == "__main__":
    main()
