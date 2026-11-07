"""Plot the four measured reports; no model or network calls.

Usage: python assets/code/langgraph-ifscale/make_figures.py
Requires matplotlib. PNG review copies are optional via --png-dir.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def save(fig, name, output, png):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None}, facecolor="white")
    # Explicit LF and no trailing whitespace for portable Git publication.
    text = path.read_bytes().decode("utf-8")
    path.write_bytes(("\n".join(line.rstrip() for line in text.splitlines()) + "\n").encode("utf-8"))
    if png:
        png.mkdir(parents=True, exist_ok=True)
        fig.savefig(png / f"{name}.png", dpi=145, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data.json")
    parser.add_argument("--output-dir", type=Path, default=HERE.parents[2] / "assets/img/langgraph-ifscale")
    parser.add_argument("--png-dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.data.read_bytes())
    arms = data["arms"]
    assert [a["writers"] for a in arms] == [1, 10, 25, 50]
    summaries = [a["summary"] for a in arms]
    x = np.arange(4)
    labels = [str(a["writers"]) for a in arms]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#343b3f", "text.color": "#343b3f",
        "svg.fonttype": "none", "svg.hashsalt": "langgraph-ifscale-20261107"})
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.5), gridspec_kw={"height_ratios": [1.45, 1]}, layout="constrained")
    series = [("assigned_local_hits", "Strict / responsible fragment", "#315f85"),
              ("global_hits", "Strict / whole report", "#a2b9ca"),
              ("loose_local_hits", "Broader / responsible fragment", "#66806e"),
              ("loose_global_hits", "Broader / whole report", "#becbbf")]
    for j, (field, label, color) in enumerate(series):
        bars = axes[0].bar(x + (j - 1.5) * .19, [s[field] for s in summaries], width=.175, color=color, label=label)
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(title="A. Keyword requirements satisfied", ylabel="Matched requirements / 500", ylim=(0, 555), xticks=x, xticklabels=labels)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.21), ncols=2, fontsize=10, frameon=False)
    axes[0].set_xlabel("Number of serial writers")
    axes[0].yaxis.grid(True, alpha=.15)
    axes[0].set_axisbelow(True)
    scores = [s["coherence_score"] for s in summaries]
    axes[1].scatter(x, scores, s=95, color="#315f85", zorder=3)
    axes[1].axhline(data["experiment"]["coherence_threshold"], color="#946b4a", linestyle="--", label="Preset threshold = 6")
    for a, score in zip(x, scores):
        axes[1].annotate(str(score), (a, score), xytext=(0, 10), textcoords="offset points", ha="center")
    axes[1].set(title="B. Blinded coherence judgment", ylabel="Score / 10", xlabel="Number of serial writers", ylim=(.5, 10.5), xlim=(-.55, 3.55), xticks=x, xticklabels=labels, yticks=[1, 3, 5, 7, 9, 10])
    axes[1].legend(loc="upper right", fontsize=10, frameon=False)
    axes[1].yaxis.grid(True, alpha=.15)
    save(fig, "coverage-coherence", args.output_dir, args.png_dir)

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 11.5), layout="constrained")
    total = np.array([s["input_tokens"] for s in summaries])
    cached = np.array([s["cached_input_tokens"] for s in summaries])
    axes[0].bar(x, total - cached, width=.55, label="Uncached input", color="#315f85")
    axes[0].bar(x, cached, width=.55, bottom=total - cached, label="Cached input", color="#aab2b8")
    for a, value in zip(x, total):
        axes[0].text(a, value + 15000, f"{value:,}", ha="center", fontsize=10)
    axes[0].set(title="A. Cumulative successful input", ylabel="Input tokens", ylim=(0, 1040000))
    axes[0].ticklabel_format(axis="y", style="plain")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    out = [s["output_tokens"] for s in summaries]
    bars = axes[1].bar(x, out, width=.55, color="#66806e")
    axes[1].bar_label(bars, labels=[f"{n:,}" for n in out], padding=4, fontsize=10)
    axes[1].set(title="B. Cumulative successful output", ylabel="Output tokens", ylim=(0, 16000))
    successful = [s["successful_writer_seconds"] / 60 for s in summaries]
    elapsed = [s["wall_seconds"] / 60 for s in summaries]
    for off, vals, label, color in [(-.18, successful, "Successful calls summed", "#315f85"), (.18, elapsed, "End-to-end, including recovery", "#b59a80")]:
        bars = axes[2].bar(x + off, vals, width=.32, color=color, label=label)
        axes[2].bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=4, fontsize=10)
    axes[2].set(title="C. Serial-call and end-to-end time", ylabel="Elapsed minutes", ylim=(0, 56))
    axes[2].legend(loc="upper left", frameon=False, fontsize=10)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Number of serial writers")
        ax.yaxis.grid(True, alpha=.15)
        ax.set_axisbelow(True)
    save(fig, "cost-and-time", args.output_dir, args.png_dir)
    print(json.dumps({"figures": 2, "arms": labels, "model_calls": 0}))


if __name__ == "__main__":
    main()
