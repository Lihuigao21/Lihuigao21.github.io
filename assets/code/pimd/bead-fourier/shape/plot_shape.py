"""Render the published shape-scan figures from the released numerical data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
BASE, END, FIXED, PATH = "#34587a", "#a64f36", "#6e718f", "#327d75"
TARGET = PROTOCOL["acceptance"]["primary_relative_tolerance"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def force_plot(conditions, output):
    temps = PROTOCOL["reduced_beta"]
    shapes = PROTOCOL["shape"]
    fig, axes = plt.subplots(1, len(temps), figsize=(12, 3.7), layout="constrained", sharey=True)
    audit = []
    for ax, b in zip(axes, temps):
        for readout, choice, label, color, marker in (
            ("bead", "fixed_k1_l2", "Endpoint, K=1 L=2", FIXED, "x"),
            ("bead", "bf", "Endpoint, best BF", END, "o"),
            ("path", "bf", "Action-grid, best BF", PATH, "^"),
        ):
            x, y = [], []
            for s in shapes:
                item = conditions[s, b, TARGET, readout]
                base, selected = item["primitive"], item[choice]
                if base is not None and selected is not None:
                    x.append(s)
                    y.append(base["sites"] / selected["sites"])
            assert x == sorted(set(x)) and np.isfinite(y).all() and np.min(y) > 0
            # Use hollow points for the optimized endpoint series to reveal coincidences.
            ax.plot(x, y, linestyle="none", marker=marker, color=color, ms=7, mew=1.5,
                    markerfacecolor="none", label=label)
            audit.append({"temperature": b, "series": label, "shape": x, "ratio": y,
                          "finite_positive_ordered_unique": True, "discrete": True})
        ax.set(title=f"Reduced inverse temperature = {b:g}", xlabel="Quartic weight s",
               xticks=shapes, yscale="log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhline(1, color=BASE, linewidth=0.9, linestyle="--")
        ax.grid(axis="y", alpha=0.28)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Force sites: primitive / BF")
    axes[-1].legend(fontsize=8, loc="upper right", frameon=False)
    path = output / "force_vs_shape.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return {"file": path.name, "sha256": digest(path), "series": audit}


def tail_plot(mechanism, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), layout="constrained")
    audit = []
    for ax, obs, title in zip(axes, ("x2", "energy"), ("Endpoint ⟨x²⟩", "Endpoint energy")):
        for points, readout, label, color, marker in (
            (2, "plain", "2 nodes, plain BF", END, "o"),
            (8, "plain", "8 nodes, plain BF", "#96758e", "x"),
            (2, "partial_average", "2 nodes, tail averaged", BASE, "s"),
            (8, "partial_average", "8 nodes, tail averaged", PATH, "^"),
        ):
            selected = sorted((r for r in mechanism["rows"] if r["beads"] == 8 and r["points"] == points),
                              key=lambda r: r["shape"])
            x = [r["shape"] for r in selected]
            y = [100 * r["signed_relative_error"][readout][obs] for r in selected]
            assert x == PROTOCOL["shape"] and np.isfinite(y).all()
            ax.plot(x, y, marker=marker, color=color, linestyle="none", label=label, ms=7, mew=1.5,
                    markerfacecolor="none")
            audit.append({"observable": obs, "series": label, "shape": x, "error_percent": y,
                          "finite_ordered_unique": True, "discrete": True})
        ax.axhline(0, linewidth=0.9, color=BASE)
        ax.axhspan(-2, 2, alpha=0.10, color=BASE)
        ax.set(xlabel="Quartic weight s", title=title, xticks=PROTOCOL["shape"])
        ax.grid(axis="y", alpha=0.23)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Signed relative error (%)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    file = output / "tail_control.png"
    fig.savefig(file, dpi=190)
    plt.close(fig)
    return {"file": file.name, "sha256": digest(file), "series": audit}


def timing_plot(summary, output):
    measured = sorted((r for r in summary["efficiency_comparisons"]
                       if r["reduced_beta"] == 8 and "minimum_sites_bf" in r["case"]["roles"]),
                      key=lambda r: r["shape"])
    labels = [f"{r['shape']:g}" for r in measured]
    audit = []
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4), layout="constrained")
    for ax, key, title in zip(axes, ("total_sites_ratio", "total_wall_ratio"),
                              ("Total force sites", "ToyModel wall time")):
        values = [r[key] for r in measured]
        assert np.isfinite(values).all() and np.min(values) > 0 and len(set(labels)) == len(labels)
        bars = ax.bar(np.arange(len(labels)), values, color=[PATH if y < 1 else END for y in values], width=0.65)
        ax.axhline(1, linewidth=1, color=BASE, linestyle="--")
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
        ax.set_ylim(0, max(1.3, 1.25 * max(values)))
        ax.set(title=title, xlabel="Quartic weight s", xticks=np.arange(len(labels)), xticklabels=labels)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
        audit.append({"metric": key, "shape": labels, "ratio": values, "finite_positive_ordered_unique": True,
                      "discrete": True})
    axes[0].set_ylabel("BF / primitive")
    file = output / "timing_by_shape.png"
    fig.savefig(file, dpi=190)
    plt.close(fig)
    return {"file": file.name, "sha256": digest(file), "series": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    choices = {
        (row["shape"], row["reduced_beta"], row["tolerance"], row["readout"]): row
        for row in data["matched_costs"]
    }
    rendered = [force_plot(choices, args.output), tail_plot(data["tail"], args.output),
                timing_plot({"efficiency_comparisons": data["timing"]["comparisons"]}, args.output)]
    for result, published in zip(rendered, data["figures"]):
        if result["file"] != published["file"] or result["series"] != published["series"]:
            raise ValueError("A rendered plot disagrees with the published data series.")
    print(f"Reconstructed {len(rendered)} figure data series collections.")


if __name__ == "__main__":
    main()
