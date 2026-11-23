"""Select equal-accuracy configurations from the independent transfer scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
PROTOCOL_PATH = WORK / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def condition(row):
    return float(row["shape"]), float(row["reduced_beta"])


def config_key(row):
    return (*condition(row), int(row["modes"]), int(row["points"]), int(row["beads"]))


def identifier(row):
    s, b, k, l, p = config_key(row)
    tag = f"s{s:g}-b{b:g}-p{p}-k{k}-l{l}".replace(".", "p")
    return f"{tag}-r{row.get('refinement', 1)}"


def cost(row):
    return int(row["beads"]) * max(1, int(row["points"]))


def convergence(reference):
    """Require a sustained pass on the actual frozen bead grid for each scheme."""
    if not reference["validation"]["accepted"]:
        raise ValueError("Finite-action reference has not passed its numerical checks")
    quantum = {condition(row): row for row in reference["quantum"]}
    rows = {config_key(row): row for row in reference["rows"]}
    if len(rows) != len(reference["rows"]):
        raise ValueError("Duplicate finite-action reference")
    thresholds = PROTOCOL["acceptance"]
    checks = [thresholds["primary_relative_tolerance"], *thresholds["sensitivity_tolerances"]]
    quantum_change = reference["validation"]["max_quantum_relative_change"]
    outputs = []
    for s, b in sorted(quantum):
        expectation = quantum[s, b]
        for scheme in PROTOCOL["schemes"]:
            k, l = scheme["modes"], scheme["points"]
            beads = [p for p in PROTOCOL["bead_grid"] if p <= PROTOCOL["max_beads_by_reduced_beta"][f"{b:g}"]]
            series = [rows[s, b, k, l, p] for p in beads]
            for readout in thresholds["readouts"]:
                obs = thresholds["observables"]
                errors = [max(abs(row[readout][name] / expectation[name] - 1) for name in obs) for row in series]
                uncertainty = [2 * (row["relative_numerical_change"] + quantum_change) for row in series]
                for target in checks:
                    passing = [error <= target for error in errors]
                    first = next((i for i in range(len(series) - 1) if all(passing[i:])), None)
                    neighbors = list(range(max((first if first is not None else len(series) - 1) - 1, 0),
                                           len(series)))
                    borderline = [beads[i] for i in neighbors if abs(errors[i] - target) <= uncertainty[i]]
                    first_under_refinement = [
                        next((beads[i] for i in range(len(series) - 1)
                              if all(e + direction * u <= target for e, u in zip(errors[i:], uncertainty[i:]))), None)
                        for direction in (-1, 1)
                    ]
                    outputs.append({
                        "shape": s, "reduced_beta": b, "modes": k, "points": l,
                        "readout": readout, "tolerance": target, "bead_grid": beads,
                        "max_errors_by_bead": errors,
                        "selected_beads": None if first is None else beads[first],
                        "sites": None if first is None else cost(series[first]),
                        "selected_relative_error": None if first is None else errors[first],
                        "finer_confirmation": [] if first is None else beads[first+1:],
                        "isolated_finest_pass": first is None and passing[-1],
                        "borderline_with_numerical_refinement": borderline,
                        "refinement_stable": first_under_refinement[0] == first_under_refinement[1],
                        "beads_under_refinement_envelope": first_under_refinement,
                    })
    return outputs


def best_by_condition(outcomes):
    pairs = {}
    for row in outcomes:
        key = (*condition(row), row["tolerance"], row["readout"])
        entry = pairs.setdefault(key, {"primitive": None, "bf": None, "fixed_k1_l2": None})
        if row["selected_beads"] is None:
            continue
        if row["modes"] == 0:
            entry["primitive"] = row
            continue
        if (row["modes"], row["points"]) == (1, 2):
            entry["fixed_k1_l2"] = row
        chosen = entry["bf"]
        score = (row["sites"], row["modes"], row["selected_relative_error"])
        if chosen is None or score < (chosen["sites"], chosen["modes"], chosen["selected_relative_error"]):
            entry["bf"] = row
    return pairs


def build_selection(reference, outcomes):
    q = {condition(row): row for row in reference["quantum"]}
    lookup = {config_key(row): row for row in reference["rows"]}
    best = best_by_condition(outcomes)
    md = PROTOCOL["md"]
    target = PROTOCOL["acceptance"]["primary_relative_tolerance"]
    picked = {}

    def add(scheme, beads, role, refinement=1):
        s, b = condition(scheme)
        key = (s, b, scheme["modes"], scheme["points"], beads)
        case = {name: lookup[key][name] for name in ("shape", "reduced_beta", "beta", "beads", "modes", "points")}
        case.update(gap=q[s, b]["gap"], refinement=refinement)
        identity = identifier(case)
        selected = picked.setdefault(identity, case | {"roles": []})
        selected["roles"].append(role)

    for b, shapes in md["shape_by_reduced_beta"].items():
        b = float(b)
        for s in shapes:
            join = best[s, b, target, "bead"]
            base, bf, fixed = join.values()
            if base is None or bf is None:
                raise ValueError(f"Primary endpoint comparison unresolved for s={s}, B={b}")
            add(base, base["selected_beads"], "primitive")
            add(bf, bf["selected_beads"], "minimum_sites_bf")
            if b == 8 and fixed is not None and fixed["selected_beads"] <= 64 and fixed != bf:
                add(fixed, fixed["selected_beads"], "fixed_k1_l2_bf")
            if b == 8 and s in (0, 1):
                add(bf, bf["finer_confirmation"][0], "next_finer_bf")
            if any(condition(c) == (s, b) for c in md["dt_controls"]):
                add(base, base["selected_beads"], "dt_control", refinement=2)
                add(bf, bf["selected_beads"], "dt_control", refinement=2)
    return list(picked.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ref = read(args.reference)
    outcomes = convergence(ref)
    if not all(row["refinement_stable"] for row in outcomes):
        raise ValueError("Refinement envelope can alter a sustained bead threshold")
    selection = build_selection(ref, outcomes)
    args.output.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    matching = best_by_condition(outcomes)
    for s, b, t, label in matching:
        if t == PROTOCOL["acceptance"]["primary_relative_tolerance"] and label == "bead":
            answer = matching[s, b, t, label]
            print(f"s={s:g}, B={b:g} :",
                  [(name, (entry["selected_beads"], entry["modes"], entry["points"], entry["sites"]))
                   for name, entry in answer.items() if entry is not None])
    print(f"selected {len(selection)} ToyModel sampler configurations")


if __name__ == "__main__":
    main()
