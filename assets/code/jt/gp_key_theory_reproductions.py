"""Small reproductions of geometric-phase mechanisms near conical intersections.

The calculations here are deliberately minimal. They reproduce the mechanisms
behind several key theory/computation papers rather than attempting full
reactive-scattering or full vibronic eigenvalue calculations:

1. Half-integer angular quantization in a Jahn-Teller pseudorotor.
2. A pi phase shift in two-path interference around a conical intersection.
3. Destructive cancellation of two symmetric tunneling routes, a toy model for
   GP-induced localization.

Run from the repository root:

    python assets/code/jt/gp_key_theory_reproductions.py

Outputs:

    assets/img/jt/gp-key-theory-reproductions.png
    assets/img/jt/gp-key-theory-reproductions.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def heat_capacity(beta: np.ndarray, energies: np.ndarray) -> np.ndarray:
    shifted = energies - np.min(energies)
    weights = np.exp(-beta[:, None] * shifted[None, :])
    z = weights.sum(axis=1)
    mean = (weights * shifted[None, :]).sum(axis=1) / z
    mean2 = (weights * shifted[None, :] ** 2).sum(axis=1) / z
    return beta**2 * (mean2 - mean**2)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "assets" / "img" / "jt"
    out_dir.mkdir(parents=True, exist_ok=True)

    m = np.arange(-6, 7, dtype=float)
    e_no_gp = 0.5 * m**2
    e_gp = 0.5 * (m + 0.5) ** 2

    beta = np.linspace(0.4, 20.0, 500)
    cv_no_gp = heat_capacity(beta, e_no_gp)
    cv_gp = heat_capacity(beta, e_gp)

    phase = np.linspace(0.0, 2.0 * np.pi, 500)
    intensity_no_gp = 0.5 * (1.0 + np.cos(phase))
    intensity_gp = 0.5 * (1.0 - np.cos(phase))

    ratio = np.linspace(0.0, 1.5, 500)
    splitting_no_gp = np.abs(1.0 + ratio)
    splitting_gp = np.abs(1.0 - ratio)

    rows = np.column_stack(
        [
            beta,
            cv_no_gp,
            cv_gp,
            phase,
            intensity_no_gp,
            intensity_gp,
            ratio,
            splitting_no_gp,
            splitting_gp,
        ]
    )
    with open(out_dir / "gp-key-theory-reproductions.csv", "w", encoding="utf-8", newline="\n") as handle:
        np.savetxt(
            handle,
            rows,
            delimiter=",",
            header=(
                "beta,Cv_no_GP_over_kB,Cv_GP_over_kB,phase_delta,"
                "interference_no_GP,interference_GP,path_ratio,"
                "splitting_no_GP,splitting_GP"
            ),
            comments="",
        )

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(m, e_no_gp, s=38, color="#2764ad", label="GP excluded: integer")
    ax.scatter(m, e_gp, s=38, color="#c23b32", marker="s", label="GP included: half-shift")
    ax.set_xlabel("integer index m")
    ax.set_ylabel("angular energy")
    ax.set_title("Ham pseudorotor sequence")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(beta, cv_no_gp, color="#2764ad", lw=2.2, label="GP excluded")
    ax.plot(beta, cv_gp, color="#c23b32", lw=2.2, label="GP included")
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"$C_V/k_B$")
    ax.set_title("Thermodynamic response of shifted spectra")
    ax.set_xlim(beta.min(), beta.max())
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(phase / np.pi, intensity_no_gp, color="#2764ad", lw=2.2, label="without extra GP phase")
    ax.plot(phase / np.pi, intensity_gp, color="#c23b32", lw=2.2, label=r"with $\pi$ GP phase")
    ax.set_xlabel(r"ordinary path phase difference $\delta/\pi$")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Two-path interference around a CI")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.03)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(ratio, splitting_no_gp, color="#2764ad", lw=2.2, label=r"$|t_1+t_2|$")
    ax.plot(ratio, splitting_gp, color="#c23b32", lw=2.2, label=r"$|t_1-t_2|$")
    ax.axvline(1.0, color="0.55", ls=":", lw=1.6)
    ax.text(1.03, 0.08, "symmetric paths", color="0.45", fontsize=9)
    ax.set_xlabel(r"path-amplitude ratio $t_2/t_1$")
    ax.set_ylabel("relative tunneling splitting")
    ax.set_title("GP cancellation and localization toy model")
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 2.6)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out_dir / "gp-key-theory-reproductions.png", dpi=180)


if __name__ == "__main__":
    main()
