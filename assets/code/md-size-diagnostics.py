"""Analytic finite-cell checks; no MD or material calculation is performed.

Run with Python 3.10+: python assets/code/md-size-diagnostics.py
The temperature convention removes three center-of-mass momenta and assumes
classical Cartesian nuclei with no other constraints. q is expressed in
primitive reciprocal-lattice coordinates for a diagonal repetition cell.
"""
import argparse
from fractions import Fraction
import json
from math import exp, lgamma, log, sqrt
from pathlib import Path

PUBLISHED_SILICON = [
    {"atoms": 512, "unstable_modes": 36, "total_modes": 1536,
     "unstable_displacement_weight": 0.97},
    {"atoms": 1728, "unstable_modes": 120, "total_modes": 5184,
     "unstable_displacement_weight": 0.87},
]


def kinetic_temperature_width(atoms, temperature=300.0):
    dof = 3 * atoms - 3
    if dof <= 0:
        raise ValueError("At least two atoms are needed for this convention.")
    relative_sd = sqrt(2.0 / dof)
    return {"atoms": atoms, "momentum_dof": dof,
            "relative_temperature_sd": relative_sd,
            "temperature_sd_K": temperature * relative_sd}


def compatible(q_fractional, repeats):
    """Test exp(i q.L_j)=1 exactly for a diagonal supercell."""
    if len(q_fractional) != 3 or len(repeats) != 3:
        raise ValueError("Three reciprocal coordinates and repetitions required.")
    if any(not isinstance(n, int) or n <= 0 for n in repeats):
        raise ValueError("Repetitions must be positive integers.")
    return all((Fraction(q) * n).denominator == 1
               for q, n in zip(q_fractional, repeats))


def make_figures(output, preview=None):
    """Plot analytic distributions and reported scalars, not material trajectories.

    Requires matplotlib. Silicon values: Corradini et al., npj Computational
    Materials 11, 151 (2025), doi:10.1038/s41524-025-01614-5, Fig. 12.
    The weights are approximate values near 700 fs, not statistical error bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if preview:
        preview = Path(preview)
        preview.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "md-size-note"})

    def save(fig, name):
        dest = output / (name + ".svg")
        fig.savefig(dest, bbox_inches="tight", metadata={"Date": None})
        # Matplotlib SVG metadata and line endings remain portable public text.
        text = dest.read_text(encoding="utf-8")
        dest.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n",
                        encoding="utf-8", newline="\n")
        if preview:
            fig.savefig(preview / (name + ".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.1, 4.3))
    ratios = [0.002 + 2.398 * i / 1400 for i in range(1401)]
    for atoms, color in zip((6, 64, 162), ("#a96340", "#3d6690", "#327465")):
        dof = 3 * atoms - 3
        shape = dof / 2
        rate = dof / 2
        pdf = [exp(shape * log(rate) - lgamma(shape)
                   + (shape - 1) * log(x) - rate * x) for x in ratios]
        ax.plot(ratios, pdf, color=color, lw=2,
                label=f"N = {atoms}, relative SD = {100 * sqrt(2 / dof):.1f}%")
    ax.axvline(1, color="#777777", ls="--", lw=1, label="Bath temperature")
    ax.set(xlabel=r"Instantaneous kinetic temperature / bath temperature",
           ylabel="Probability density in this ratio", xlim=(0, 2.4), ylim=(0, None))
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    fig.tight_layout()
    save(fig, "canonical-temperature-widths")

    fig, ax = plt.subplots(figsize=(8.1, 4.3))
    x = [0, 1]
    fractions = [100 * r["unstable_modes"] / r["total_modes"] for r in PUBLISHED_SILICON]
    weights = [100 * r["unstable_displacement_weight"] for r in PUBLISHED_SILICON]
    bars_a = ax.bar([v - 0.19 for v in x], fractions, width=0.34,
                    color="#3d6690", label="Unstable modes / all modes")
    bars_b = ax.bar([v + 0.19 for v in x], weights, width=0.34,
                    color="#a96340", label="Squared displacement projection weight")
    for bars, values in ((bars_a, fractions), (bars_b, weights)):
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 1.5,
                    f"{value:.2f}%" if value < 5 else f"~{value:.0f}%",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["512 atoms\n36 / 1536 modes", "1728 atoms\n120 / 5184 modes"])
    ax.set(ylabel="Percent (different denominators)", ylim=(0, 108))
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "silicon-published-mode-weights")
    (output / "silicon-published-values.json").write_text(json.dumps({
        "doi": "10.1038/s41524-025-01614-5", "figure": 12,
        "conditions": {"initial_lattice_temperature_K": 0.1,
                       "photoexcited_electrons_holes_per_atom": 0.2,
                       "approximate_time_fs": 700},
        "definition": "Squared normalized displacement projection onto all unstable modes; not an energy fraction or a single-q contribution.",
        "records": PUBLISHED_SILICON,
        "provenance": "Scalar values reported in the paper; no trajectory digitization or new MD.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="Make the two explanatory figures (requires matplotlib).")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "img/md-supercell-observables")
    parser.add_argument("--preview", type=Path, help="Optional PNG preview directory.")
    args = parser.parse_args()
    zone_corner = ("1/2", "1/2", "1/2")
    result = {
        "scope": "Exact geometry and classical canonical formulas; not a reproduction of either paper.",
        "temperature_K": 300,
        "temperature_widths": [kinetic_temperature_width(n) for n in (6, 64, 162)],
        "illustrative_q_fractional": list(zone_corner),
        "mode_compatibility": [
            {"repeats": [n, n, n], "contains_mode": compatible(zone_corner, (n, n, n))}
            for n in (2, 3, 4, 5)
        ],
    }
    if args.figures:
        make_figures(args.output, args.preview)
        result["figures_generated"] = True
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
