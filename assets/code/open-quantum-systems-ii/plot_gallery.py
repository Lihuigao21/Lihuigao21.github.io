"""Replot the twelve published TCL2/NZ2 population examples from their CSVs.

No propagation, interpolation, filtering, clipping or normalization is applied.
Run with --output to choose an explicit destination for figures and diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {"exact": "#202124", "TCL2": "#d16b13", "NZ2": "#2478ab"}
STYLES = {"exact": "-", "TCL2": "--", "NZ2": ":"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "path.simplify": False, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
    "svg.hashsalt": "open-quantum-systems-ii",
})


def quality(label, t, y, gates):
    """Use the original benchmark's frozen fourth-difference thresholds."""
    scale = max(float(np.ptp(y)), 1.0)
    residual = np.diff(y, n=4) / (16 * scale)
    maximum = float(np.max(abs(residual)))
    rms = float(np.sqrt(np.mean(abs(residual) ** 2)))
    endpoint = float(max(abs(residual[0]), abs(residual[-1])))
    result = dict(
        series=label, n=len(t),
        finite=bool(np.isfinite(t).all() and np.isfinite(y).all()),
        strictly_increasing=bool(np.all(np.diff(t) > 0)),
        duplicate_times=int(len(t) - len(np.unique(t))),
        high_frequency_residual_rms=rms,
        high_frequency_residual_max=maximum,
        endpoint_artifact=endpoint,
        isolated_spikes=int(np.count_nonzero(abs(residual) > gates["curve_max"])),
        scale=scale,
        algorithm="fourth difference / (16*max(ptp,1)); original data unchanged",
        oscillation_exception=None,
    )
    result["pass"] = bool(
        result["finite"] and result["strictly_increasing"]
        and not result["duplicate_times"]
        and rms <= gates["curve_rms"]
        and maximum <= gates["curve_max"]
        and endpoint <= gates["curve_endpoint"]
        and not result["isolated_spikes"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "gallery.json").read_text(encoding="utf-8"))
    gates = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["gates"]
    records, exports = [], []
    for case in manifest["panels"]:
        source = ROOT / case["csv"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != case["csv_sha256"]:
            raise RuntimeError("CSV identity mismatch: " + source.name)
        data = np.genfromtxt(source, delimiter=",", names=True)
        t = data["time"]
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        values = []
        for method in ("exact", "TCL2", "NZ2"):
            y = data[method]
            item = quality(case["id"] + "_" + method, t, y, gates)
            records.append(item)
            if not item["pass"]:
                raise RuntimeError("Curve-quality gate failed: " + item["series"])
            ax.plot(t, y, STYLES[method], color=COLORS[method],
                    linewidth=2.1 if method == "exact" else 1.9, label=method)
            values.append(y)
        low, high = min(0., float(np.min(values))), max(1., float(np.max(values)))
        margin = 0.045 * (high - low)
        ax.set(xlim=(0, 30), ylim=(low - margin, high + margin),
               xlabel=r"Time $t\omega_{\mathrm{ref}}$",
               ylabel=r"Upper-state population $P_e(t)$")
        ax.set_title(case["letter"] + ". " + case["label"]
                     + f" | g = {case['g']:g}", pad=38)
        ax.axhline(0, color="#b9b9b9", linewidth=.7, zorder=0)
        ax.axhline(1, color="#d1d1d1", linewidth=.7, zorder=0)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncols=3,
                  fontsize=9, frameon=False, borderpad=.4, handlelength=2.2)
        filename = case["id"] + ".png"
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
        exports.append(dict(
            id=case["id"], filename=filename,
            sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
            pixels=[1184, 768], samples=len(t),
        ))
    report = dict(thresholds=gates, series=records, figures=exports,
                  pass_=all(item["pass"] for item in records),
                  sampling_note="Deterministic data; separate time and basis controls.",
                  visual_review="Required separately at exported resolution.")
    (output / "curve-quality.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({"figures": len(exports), "series": len(records),
                      "all_curve_gates_pass": report["pass_"]}))


if __name__ == "__main__":
    main()
