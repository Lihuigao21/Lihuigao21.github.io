# SPDX-License-Identifier: GPL-3.0-only
"""Run and time matched shape-scan configurations through ToyModel BAOAB."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time
from uuid import uuid4

import numpy as np
import scipy
from scipy.stats import t

from select_configs import PROTOCOL, PROTOCOL_PATH, config_key, cost, identifier, read
AUTHOR_BUNDLE = Path(__file__).resolve().parents[1] / "author"
sys.path.insert(0, str(AUTHOR_BUNDLE))

import bead_fourier
import ring_helpers
import sampler_statistics
from bead_fourier import BeadFourierPIMD, BeadFourierPath
from ring_helpers import split_rhat
from sampler_statistics import chain_effective_sample_sizes

CONTROL = PROTOCOL["md"]
OBSERVABLES = ("x2", "energy")
READOUTS = ("bead", "path")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    for attempt in range(4):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.05 * (attempt + 1))


def start_record(output, selected, reference):
    paths = [
        Path(__file__), PROTOCOL_PATH, Path(__file__).with_name("select_configs.py"),
        *[Path(x.__file__) for x in (bead_fourier, ring_helpers, sampler_statistics)],
    ]
    data = {
        "inputs": {selected.name: digest(selected), reference.name: digest(reference)},
        "sources": {path.name: digest(path) for path in paths},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "threads": {key: os.environ.get(key)
                    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    file = output / "provenance.json"
    if file.exists():
        if read(file) != data:
            raise ValueError("Sampler inputs changed. A new attempt directory is required.")
    else:
        frozen = output / "frozen_source"
        frozen.mkdir(parents=True, exist_ok=True)
        for p in paths:
            target = frozen / p.name
            shutil.copyfile(p, target)
            if digest(target) != data["sources"][p.name]:
                raise ValueError(f"Source changed while freezing {p}")
        shutil.copyfile(selected, frozen / "selection.json")
        write(file, data)
    return data


class CountedForce:
    def __init__(self, shape):
        self.shape = shape
        self.harmonic = 1.0 - shape
        self.sites = 0

    def __call__(self, coords):
        self.sites += coords.size
        return -(self.harmonic + self.shape * (coords * coords)) * coords


def instantiate(case, seed, chains):
    rng = np.random.default_rng(seed)
    path = BeadFourierPath(case["beads"], case["beta"], modes=case["modes"],
                           quadrature_points=case["points"] or None)
    center = rng.normal(0, 0.55, size=(chains, 1))
    q = center + rng.normal(0, 0.25, size=(chains, case["beads"]))
    b = rng.normal(size=(*q.shape, case["modes"])) * np.sqrt(path.beta_p)
    force = CountedForce(case["shape"])
    sampler = BeadFourierPIMD(
        path, force, q, amplitudes=b, time_step=CONTROL["dt_times_gap"] / case["gap"] / case["refinement"],
        rng=rng, centroid_friction=CONTROL["centroid_friction_over_gap"] * case["gap"],
        mass_frequency=CONTROL["mass_preconditioning_frequency_over_gap"] * case["gap"],
        splitting=CONTROL["splitting"],
    )
    return sampler, force


def readouts(sampler, shape):
    q2 = sampler.bead_positions**2
    x2 = q2.mean(axis=-1)
    e = (1 - shape) * x2 + 0.75 * shape * (q2 * q2).mean(axis=-1)
    if sampler.path.quadrature_points is None:
        return (x2, e), (x2, e)
    s2 = sampler.force_positions**2
    path_x2 = sampler.path.average_at_force_sites(s2)
    path_energy = (1 - shape) * path_x2 + 0.75 * shape * sampler.path.average_at_force_sites(s2 * s2)
    return (x2, e), (path_x2, path_energy)


def chain_statistics(values):
    means = values.mean(axis=1)
    count = means.size
    se = float(means.std(ddof=1) / np.sqrt(count))
    return {
        "mean": float(means.mean()), "sd": float(values.std(ddof=1)),
        "se": se, "ci95_half": se * float(t.ppf(0.975, count - 1)),
        "split_rhat": split_rhat(values),
        "ess": float(chain_effective_sample_sizes(values, max_lag=min(400, values.shape[1] - 1)).sum()),
        "chain_means": means.tolist(),
        "block_means": values.reshape(count, 10, -1).mean(axis=-1).tolist(),
    }


def sample(case, index, smoke=False):
    factor = case["refinement"]
    chains = 8 if smoke else CONTROL["chains"]
    burn = (200 if smoke else CONTROL["burn"]) * factor
    steps = (800 if smoke else CONTROL["steps"]) * factor
    stride = CONTROL["stride"] * factor
    seed = np.random.SeedSequence(CONTROL["seed"], spawn_key=(int(smoke), index))
    started = now()
    begin = time.perf_counter()
    sampler, force = instantiate(case, seed, chains)
    for _ in range(burn):
        sampler.step()
    traces = np.empty((2, 2, chains, steps // stride))
    for step in range(steps):
        sampler.step()
        if (step + 1) % stride == 0:
            traces[..., step // stride] = readouts(sampler, case["shape"])
    elapsed = time.perf_counter() - begin
    expected = (burn + steps + 1) * cost(case) * chains
    if force.sites != expected or sampler.force_evaluations != expected or not np.isfinite(traces).all():
        raise ValueError(f"Invalid sampler trace or force count for {identifier(case)}")
    statistics_by_obs = {
        name: {key: chain_statistics(traces[i, j]) for j, key in enumerate(OBSERVABLES)}
        for i, name in enumerate(READOUTS)
    }
    return {
        "case": case, "time_step": sampler.time_step, "burn": burn, "steps": steps, "stride": stride,
        "chains": chains, "seen_force_sites": force.sites, "seed": CONTROL["seed"], "spawn_key": list(seed.spawn_key),
        "observables": statistics_by_obs, "started": started, "ended": now(), "wall_seconds": elapsed,
    }


def valid_existing(path, case, index):
    if not path.is_file():
        return False
    try:
        present = read(path)
        r = case["refinement"]
        burn, steps = CONTROL["burn"] * r, CONTROL["steps"] * r
        return (
            present["case"] == case and present["seed"] == CONTROL["seed"] and present["spawn_key"] == [0, index]
            and present["steps"] == steps and present["burn"] == burn
            and present["seen_force_sites"] == (burn + steps + 1) * cost(case) * CONTROL["chains"]
            and all(math.isfinite(present["observables"][name][obs][key])
                    for name in READOUTS for obs in OBSERVABLES for key in ("mean", "se", "ess", "split_rhat"))
        )
    except (ValueError, OSError, KeyError, TypeError):
        return False


def validation(cases, rows, reference):
    expected = {config_key(row): row for row in reference["rows"]}
    equilibrium, steps = [], []
    for case in cases:
        name = identifier(case)
        reference_row = expected[config_key(case)]
        stats = rows[name]["observables"]
        for readout in READOUTS:
            for obs in OBSERVABLES:
                raw = stats[readout][obs]
                theoretical = reference_row[readout][obs]
                limit = 3.5 * raw["se"] + 0.002 * theoretical
                equilibrium.append({"case": name, "observable": obs, "readout": readout,
                                    "independent_expectation": theoretical,
                                    "relative_difference": raw["mean"] / theoretical - 1,
                                    "pass": abs(raw["mean"] - theoretical) <= limit
                                    and raw["ess"] >= CONTROL["minimum_ess"]
                                    and raw["split_rhat"] <= CONTROL["maximum_split_rhat"]})
                if case["refinement"] == 2:
                    old = rows[identifier(case | {"refinement": 1})]["observables"][readout][obs]
                    shift = raw["mean"] - old["mean"]
                    bound = 3.5 * math.hypot(raw["se"], old["se"]) + 0.002 * theoretical
                    steps.append({"case": name, "observable": obs, "readout": readout,
                                  "relative_shift": shift / theoretical, "pass": abs(shift) <= bound})
    return {"equilibrium": equilibrium, "timestep": steps,
            "passed": all(r["pass"] for r in equilibrium + steps)}


def equilibrium(output, reference, cases, workers):
    stage = output / "cells"
    stage.mkdir(exist_ok=True)
    scheduled = [(index, case) for index, case in enumerate(cases)
                 if not valid_existing(stage / f"{identifier(case)}.json", case, index)]
    done = len(cases) - len(scheduled)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(sample, case, index): case for index, case in scheduled}
        for future in as_completed(jobs):
            case = jobs[future]
            name = identifier(case)
            try:
                result = future.result()
            except Exception as error:
                write(stage / f"{name}.failure.json", {"case": case, "exception": repr(error), "time": now()})
                raise
            write(stage / f"{name}.json", result)
            done += 1
            print(f"{done}/{len(cases)} {name}, {result['wall_seconds']:.1f}s", flush=True)
    data = {identifier(case): read(stage / f"{identifier(case)}.json") for case in cases}
    gates = validation(cases, data, reference)
    write(output / "validation.json", gates)
    failed = [entry for group in ("equilibrium", "timestep") for entry in gates[group] if not entry["pass"]]
    if failed:
        print(f"Failed checks: {failed}")
        raise RuntimeError("At least one declared scientific check failed")


def cpu_affinity():
    try:
        import psutil
        current = psutil.Process()
        cores = current.cpu_affinity()
        current.cpu_affinity([cores[min(len(cores) - 1, 2)]])
        return current.cpu_affinity()
    except (ImportError, AttributeError, OSError):
        return None


def measure(output, selection):
    if not read(output / "validation.json")["passed"]:
        raise RuntimeError("Timing inputs have failed the sampler check")
    cases = [case for case in selection if case["refinement"] == 1 and
             (case["reduced_beta"] == 8 or case["shape"] in (0, 1))]
    config = PROTOCOL["timings"]
    numbers = {identifier(case): {"case": case, "site_count": cost(case), "batches": {}} for case in cases}
    pinned = cpu_affinity()
    started = now()
    for batch in config["batch_sizes"]:
        for trial in range(config["repeats"]):
            offset = trial % len(cases)
            for case in cases[offset:] + cases[:offset]:
                seed = np.random.SeedSequence(CONTROL["seed"], spawn_key=(2, trial))
                sampler, force = instantiate(case, seed, batch)
                for _ in range(config["warmup"]):
                    sampler.step()
                before = force.sites
                start = time.perf_counter_ns()
                for step in range(config["steps"]):
                    sampler.step()
                    if batch > 1 and (step + 1) % CONTROL["stride"] == 0:
                        readouts(sampler, case["shape"])
                spent = time.perf_counter_ns() - start
                if force.sites - before != cost(case) * batch * config["steps"]:
                    raise ValueError("Unexpected measured force-site count")
                output_row = numbers[identifier(case)]["batches"].setdefault(str(batch), [])
                output_row.append(spent * 1e-9 / config["steps"])
            print(f"{batch} paths, round {trial + 1}/{config['repeats']}", flush=True)
    write(output / "timing.json", {"started": started, "ended": now(), "affinity": pinned, "rows": numbers})


def efficiency(case_timing, cell):
    values = case_timing["batches"]
    cfg = PROTOCOL["timings"]
    med = {key: statistics.median(data) for key, data in values.items()}
    step_time = med[str(cell["chains"])]
    estimates = {}
    for name in OBSERVABLES:
        statistics_ = cell["observables"]["bead"][name]
        effective_per_step = statistics_["ess"] / cell["steps"]
        required_steps = (statistics_["sd"] / (statistics_["mean"] * cfg["target_relative_se"]))**2 / effective_per_step
        estimates[name] = {
            "seconds": required_steps * step_time, "steps": required_steps,
            "force_sites": required_steps * cell["chains"] * case_timing["site_count"],
            "ess_per_step": effective_per_step,
        }
    return {
        "single_path_seconds_per_step": med["1"], "batch_seconds_per_step": step_time,
        "estimated_joint_seconds": max(row["seconds"] for row in estimates.values()),
        "estimated_joint_force_sites": max(row["force_sites"] for row in estimates.values()),
        "by_observable": estimates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cases = [dict(shape=s, reduced_beta=8, beta=8, gap=1.0, beads=6, modes=k, points=l, refinement=1)
                 for s, k, l in [(0, 0, 0), (0, 1, 2), (1, 0, 0), (1, 3, 2)]]
        write(args.out / "smoke.json", [sample(case, i, smoke=True) for i, case in enumerate(cases)])
        print("4 physical sampler and force-accounting smoke cases completed")
        return
    if args.reference is None or args.selection is None:
        parser.error("Production requires --reference and --selection")
    start_record(args.out, args.selection, args.reference)
    cases = read(args.selection)
    if args.timing:
        measure(args.out, cases)
    else:
        equilibrium(args.out, read(args.reference), cases, args.workers)


if __name__ == "__main__":
    main()
