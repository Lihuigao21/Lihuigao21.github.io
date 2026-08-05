"""Plot normalized Kubo and trajectory flux-side correlations.

Run from the repository root with

    python assets/code/mqc/kubo_trajectory_flux_side_comparison.py

The input CSV files contain the staged 200 and 300 K benchmark data used by
the accompanying technical note.  All curves are normalized by the same
adiabatic quantum rate scale.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "assets" / "img" / "mqc-series"
OUTPUT = DATA_DIR / "kubo-trajectory-flux-side-comparison.png"


def read_series(temperature: int) -> dict[str, list[float]]:
    path = DATA_DIR / f"kubo-trajectory-flux-side-{temperature}K.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data in {path}")
    return {
        key: [float(row[key]) for row in rows]
        for key in rows[0]
    }


def main() -> None:
    colors = {
        "qm_a": "#222222",
        "qm_na": "#111111",
        "cmd": "#2878b5",
        "fssh": "#d64b40",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)

    for ax, temperature in zip(axes, (300, 200)):
        data = read_series(temperature)
        time = data["time_au"]
        ax.plot(
            time,
            data["qm_adiabatic"],
            color=colors["qm_a"],
            linestyle="--",
            linewidth=1.7,
            label="QM adiabatic",
        )
        ax.plot(
            time,
            data["qm_nonadiabatic"],
            color=colors["qm_na"],
            linewidth=1.9,
            label="QM nonadiabatic",
        )
        ax.plot(
            time,
            data["cmd"],
            color=colors["cmd"],
            linestyle="--",
            linewidth=1.8,
            label="CMD",
        )
        ax.fill_between(
            time,
            [y - e for y, e in zip(data["cmd"], data["cmd_se"])],
            [y + e for y, e in zip(data["cmd"], data["cmd_se"])],
            color=colors["cmd"],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            time,
            data["fssh"],
            color=colors["fssh"],
            linewidth=1.8,
            label="FSSH",
        )
        ax.fill_between(
            time,
            [y - e for y, e in zip(data["fssh"], data["fssh_se"])],
            [y + e for y, e in zip(data["fssh"], data["fssh_se"])],
            color=colors["fssh"],
            alpha=0.12,
            linewidth=0,
        )
        ax.set_title(f"{temperature} K")
        ax.set_xlabel("Time (a.u.)")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.text(
            0.97,
            0.95,
            "final sampled values\n"
            f"QM-NA {data['qm_nonadiabatic'][-1]:.3f}\n"
            f"CMD {data['cmd'][-1]:.3f}\n"
            f"FSSH {data['fssh'][-1]:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.9},
        )

    axes[0].set_ylabel(
        r"Normalized flux-side correlation "
        r"$C_{fs}(t)/(2\pi Q_r k_{\mathrm{QM,A}})$"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
