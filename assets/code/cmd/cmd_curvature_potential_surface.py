"""Plot the 2D champagne-bottle Morse surface used in the CMD curvature note.

The script is self-contained for the potential surface.  If an instanton
directory is provided, it also overlays representative centroid-constrained
ring-polymer paths saved as ``instanton_T*_R*.npz`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def morse_bottle_potential(r, *, req=1.8324, d0=0.18748, alpha=1.1605):
    y = np.exp(-alpha * (np.asarray(r, dtype=float) - req))
    return d0 * (1.0 - y) ** 2


def load_path(instanton_dir: Path | None, name: str):
    if instanton_dir is None:
        return None
    path = instanton_dir / name
    if not path.exists():
        return None
    return np.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cmd-curvature-potential-surface.png")
    parser.add_argument("--instanton-dir", default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    instanton_dir = Path(args.instanton_dir) if args.instanton_dir else None
    req = 1.8324
    x = np.linspace(-3.2, 3.2, 420)
    y = np.linspace(-3.2, 3.2, 420)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X * X + Y * Y)
    V = morse_bottle_potential(R, req=req)
    levels = np.linspace(0.0, 0.22, 18)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax in axes:
        cf = ax.contourf(X, Y, np.minimum(V, 0.22), levels=levels, cmap="viridis", extend="max")
        ax.contour(X, Y, V, levels=[0.002, 0.01, 0.03, 0.08, 0.16], colors="white", linewidths=0.55, alpha=0.72)
        ax.add_patch(plt.Circle((0, 0), req, edgecolor="white", facecolor="none", lw=1.5, ls="--"))
        ax.set_aspect("equal")
        ax.set(xlabel="x / bohr", ylabel="y / bohr", xlim=(-3.0, 3.0), ylim=(-3.0, 3.0))

    axes[0].set_title("2D radial Morse bottle")
    axes[0].text(0.04, 0.94, r"minimum ring: $r_e=1.8324$ bohr", color="white", transform=axes[0].transAxes, fontsize=9, va="top")

    path_specs = [
        ("instanton_T200_R0.40.npz", "200 K, $R_c=0.4$", "#ffcc33", "o"),
        ("instanton_T800_R0.40.npz", "800 K, $R_c=0.4$", "#ff5a5f", "s"),
        ("instanton_T200_R1.80.npz", "200 K, $R_c=1.8$", "#69d2e7", "^"),
    ]
    for filename, label, color, marker in path_specs:
        data = load_path(instanton_dir, filename)
        if data is None:
            continue
        q = data["path"]
        q_closed = np.vstack([q, q[0]])
        axes[1].plot(q_closed[:, 0], q_closed[:, 1], color=color, lw=1.8, label=label)
        axes[1].scatter(q[:, 0], q[:, 1], s=11, color=color, marker=marker, edgecolor="black", linewidth=0.25, zorder=4)
        c = data["centroid"]
        axes[1].scatter([c[0]], [c[1]], s=52, color=color, edgecolor="black", linewidth=0.7, zorder=5)

    axes[1].set_title("Centroid-constrained bead paths")
    if axes[1].lines:
        axes[1].legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", framealpha=0.88)
    cb = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("V(r) / hartree, clipped at 0.22")
    fig.suptitle("Champagne-bottle Morse surface and artificial-instanton geometry", fontsize=14)
    fig.savefig(out, dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
