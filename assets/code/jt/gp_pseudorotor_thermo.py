"""Minimal pseudorotor diagnostic for geometric phase thermodynamics.

This script compares a fixed-radius Jahn-Teller pseudorotor with ordinary
periodic boundary conditions and with the geometric-phase-shifted angular
momentum spectrum. It is not a full two-dimensional vibronic calculation; it is
the smallest model that exposes the boundary-condition mechanism behind the
ground-state degeneracy and the heat-capacity change.

Run from the repository root:

    python assets/code/jt/gp_pseudorotor_thermo.py

Outputs:

    assets/img/jt/gp-pseudorotor-thermo.png
    assets/img/jt/gp-pseudorotor-thermo.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def angular_spectrum(alpha: float, mmax: int = 12, inertia: float = 1.0) -> np.ndarray:
    """Return angular energies E_m = (m + alpha)^2 / (2I)."""
    m = np.arange(-mmax, mmax + 1, dtype=float)
    return (m + alpha) ** 2 / (2.0 * inertia)


def heat_capacity(beta: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return <E> and C/kB for a discrete spectrum at inverse temperatures beta."""
    shifted = energies - np.min(energies)
    weights = np.exp(-beta[:, None] * shifted[None, :])
    partition = weights.sum(axis=1)
    mean = (weights * shifted[None, :]).sum(axis=1) / partition
    mean2 = (weights * shifted[None, :] ** 2).sum(axis=1) / partition
    cv = beta**2 * (mean2 - mean**2)
    return mean + np.min(energies), cv


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "assets" / "img" / "jt"
    out_dir.mkdir(parents=True, exist_ok=True)

    beta = np.linspace(0.4, 20.0, 500)
    no_gp = angular_spectrum(alpha=0.0, inertia=1.0)
    gp = angular_spectrum(alpha=0.5, inertia=1.0)

    e_no_gp, cv_no_gp = heat_capacity(beta, no_gp)
    e_gp, cv_gp = heat_capacity(beta, gp)

    rows = np.column_stack([beta, e_no_gp, cv_no_gp, e_gp, cv_gp])
    with open(out_dir / "gp-pseudorotor-thermo.csv", "w", encoding="utf-8", newline="\n") as handle:
        np.savetxt(
            handle,
            rows,
            delimiter=",",
            header="beta,E_no_GP,Cv_no_GP_over_kB,E_GP,Cv_GP_over_kB",
            comments="",
        )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)

    mvals = np.arange(-5, 6)
    axes[0].scatter(mvals, (mvals**2) / 2.0, color="#2764ad", label="GP excluded: m")
    axes[0].scatter(
        mvals,
        ((mvals + 0.5) ** 2) / 2.0,
        color="#c23b32",
        marker="s",
        label="GP included: m + 1/2",
    )
    axes[0].set_xlabel("integer index m")
    axes[0].set_ylabel("angular energy")
    axes[0].set_title("Allowed angular spectra")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(beta, cv_no_gp, color="#2764ad", label="GP excluded")
    axes[1].plot(beta, cv_gp, color="#c23b32", label="GP included")
    axes[1].set_xlabel(r"inverse temperature $\beta$")
    axes[1].set_ylabel(r"$C_V/k_B$")
    axes[1].set_title("Heat capacity from the angular modes")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlim(beta.min(), beta.max())
    axes[1].set_ylim(bottom=0.0)

    fig.savefig(out_dir / "gp-pseudorotor-thermo.png", dpi=180)


if __name__ == "__main__":
    main()
