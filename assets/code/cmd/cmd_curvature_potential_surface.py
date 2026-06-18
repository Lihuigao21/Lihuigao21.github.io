"""Plot the 2D champagne-bottle Morse surface used in the CMD curvature note.

The script is self-contained for the potential surface.  If an instanton
directory is provided, it also overlays one representative low-temperature
centroid-constrained ring-polymer path saved as ``instanton_T200_R0.40.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def draw_surface(ax, X, Y, V, req, levels):
    cf = ax.contourf(X, Y, np.minimum(V, 0.22), levels=levels, cmap="viridis", extend="max")
    ax.contour(X, Y, V, levels=[0.002, 0.01, 0.03, 0.08, 0.16], colors="white", linewidths=0.55, alpha=0.72)
    ax.add_patch(plt.Circle((0, 0), req, edgecolor="white", facecolor="none", lw=1.5, ls="--"))
    ax.set_aspect("equal")
    ax.set(xlabel="x / bohr", ylabel="y / bohr", xlim=(-3.0, 3.0), ylim=(-3.0, 3.0))
    return cf


def draw_path(ax, data, *, color, label):
    q = data["path"]
    q_closed = np.vstack([q, q[0]])
    ax.plot(q_closed[:, 0], q_closed[:, 1], color=color, lw=2.2, label=f"{label} bead path")
    ax.scatter(q[:, 0], q[:, 1], s=18, color=color, marker="o", edgecolor="black", linewidth=0.3, zorder=4, label="beads")
    c = data["centroid"]
    ax.scatter([c[0]], [c[1]], s=150, color="white", marker="*", edgecolor="black", linewidth=1.1, zorder=6, label=r"centroid, $R_c=0.4$")
    radii = np.linalg.norm(q, axis=1)
    return c, radii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cmd-curvature-potential-surface.png")
    parser.add_argument("--comparison-out", default=None)
    parser.add_argument("--instanton-dir", default=None)
    args = parser.parse_args()

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
        cf = draw_surface(ax, X, Y, V, req, levels)

    axes[0].set_title("2D radial Morse bottle")
    axes[0].text(0.04, 0.94, r"minimum ring: $r_e=1.8324$ bohr", color="white", transform=axes[0].transAxes, fontsize=9, va="top")

    data = load_path(instanton_dir, "instanton_T200_R0.40.npz")
    if data is not None:
        c, radii = draw_path(axes[1], data, color="#ffcc33", label="200 K")
        axes[1].annotate(
            "centroid",
            xy=(c[0], c[1]),
            xytext=(0.68, 0.34),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "white"},
            color="white",
            fontsize=9,
        )
        q = data["path"]
        bead = q[np.argmax(q[:, 1])]
        axes[1].annotate(
            "beads remain\nnear the minimum ring",
            xy=(bead[0], bead[1]),
            xytext=(-2.25, 2.35),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "white"},
            color="white",
            fontsize=9,
        )
    else:
        axes[1].text(0.5, 0.08, "provide --instanton-dir to overlay the path", transform=axes[1].transAxes, ha="center", color="white")

    axes[1].set_title("One constrained path: 200 K, $R_c=0.4$")
    if data is not None:
        axes[1].legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", framealpha=0.9)
    cb = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("V(r) / hartree, clipped at 0.22")
    fig.suptitle("Champagne-bottle Morse surface and artificial-instanton geometry", fontsize=14)
    fig.savefig(out, dpi=220, facecolor="white")

    if args.comparison_out is not None:
        comparison_out = Path(args.comparison_out)
        low = load_path(instanton_dir, "instanton_T200_R0.40.npz")
        high = load_path(instanton_dir, "instanton_T800_R0.40.npz")
        if low is None or high is None:
            raise FileNotFoundError("temperature comparison needs instanton_T200_R0.40.npz and instanton_T800_R0.40.npz")

        fig2, axes2 = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
        specs = [
            (axes2[0], low, "200 K", "#ffcc33", "low T: beads stay near the minimum ring"),
            (axes2[1], high, "800 K", "#ff5a5f", "high T: constrained path is more compact"),
        ]
        for ax, dataset, temp_label, color, title in specs:
            cf2 = draw_surface(ax, X, Y, V, req, levels)
            _, radii = draw_path(ax, dataset, color=color, label=temp_label)
            ax.set_title(title)
            ax.text(
                0.04,
                0.94,
                rf"$\langle r_{{bead}}\rangle={radii.mean():.2f}$ bohr,  $R_c=0.4$ bohr",
                color="white",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )
            ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", framealpha=0.9)
        cb2 = fig2.colorbar(cf2, ax=axes2, shrink=0.92, pad=0.02)
        cb2.set_label("V(r) / hartree, clipped at 0.22")
        fig2.suptitle("Temperature dependence of the centroid-constrained path", fontsize=14)
        fig2.savefig(comparison_out, dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
