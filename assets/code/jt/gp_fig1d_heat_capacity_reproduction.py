"""Redraw the heat-capacity trend from Fig. 1(d) of the GP/JT note.

The input CSV contains a digitized trace of the red and blue curves from the
paper panel used in the accompanying article. This script redraws the visible
curves in a clean, publication-style format for explanatory use on the site. It
is a digitized visual reproduction, not an independent full two-dimensional
vibronic eigenvalue calculation.

Run from the repository root:

    python assets/code/jt/gp_fig1d_heat_capacity_reproduction.py

Outputs:

    assets/img/jt/gp-fig1d-heat-capacity-reproduction.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_trace(data: np.ndarray, beta_col: int, cv_col: int) -> tuple[np.ndarray, np.ndarray]:
    beta = data[:, beta_col]
    cv = data[:, cv_col]
    keep = np.isfinite(beta) & np.isfinite(cv)
    beta = beta[keep]
    cv = cv[keep]
    order = np.argsort(beta)
    return beta[order], cv[order]


def smooth_trace(beta: np.ndarray, cv: np.ndarray, points: int = 700) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(float(beta.min()), float(beta.max()), points)
    values = np.interp(grid, beta, cv)
    window = 21
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return grid, smooth


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "assets" / "img" / "jt" / "gp-fig1d-digitized.csv"
    out_path = repo_root / "assets" / "img" / "jt" / "gp-fig1d-heat-capacity-reproduction.png"

    data = np.genfromtxt(data_path, delimiter=",", names=True)
    matrix = np.column_stack([data[name] for name in data.dtype.names])
    beta_blue, cv_blue = read_trace(matrix, 0, 1)
    beta_red, cv_red = read_trace(matrix, 2, 3)
    beta_blue, cv_blue = smooth_trace(beta_blue, cv_blue)
    beta_red, cv_red = smooth_trace(beta_red, cv_red)

    blue_beta = np.r_[0.0, 1.4, beta_blue]
    blue_cv = np.r_[1.45, 1.30, cv_blue]
    red_beta = np.r_[0.0, 1.4, beta_red]
    red_cv = np.r_[1.50, 1.32, cv_red]

    fig, ax = plt.subplots(figsize=(4.8, 5.2), constrained_layout=True)
    ax.plot(blue_beta, blue_cv, color="blue", lw=2.7)
    ax.plot(red_beta, red_cv, color="red", lw=2.7)
    ax.axvline(8, color="0.55", lw=2.0, ls=":")
    ax.text(8.45, 1.00, r"$\beta=8\ \mathrm{r.u.}$", color="0.55", fontsize=13)
    ax.text(19.4, 1.15, "(d)", fontsize=18, ha="right", va="center")

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks(np.linspace(0, 1.2, 7))
    ax.set_xlabel(r"$\beta\ (\mathrm{r.u.})$", fontsize=16)
    ax.set_ylabel(r"$C_V\ (k_B)$", fontsize=16)
    ax.tick_params(direction="in", top=True, right=True, length=5, width=1.2, labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.savefig(out_path, dpi=220)


if __name__ == "__main__":
    main()
