# SPDX-License-Identifier: GPL-3.0-only
"""Reproduce the harmonic BF-PIMD equilibria and timing controls."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import scipy
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.stats import t

import bead_fourier
from bead_fourier import BeadFourierPath, BeadFourierPIMD
from ring_helpers import real_normal_mode_frequencies, real_normal_mode_matrix, split_rhat

WORK = Path(__file__).resolve().parent
PROTOCOL_PATH = WORK / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Case:
    name: str
    beta: float
    beads: int
    modes: int
    rule: str
    sites: int
    dt: float
    tau0: float
    stride: int
    md: bool

    @property
    def site_count(self):
        if self.rule == "analytic":
            return 0
        return self.beads * max(self.sites, 1)


CASES = [Case(**row) for row in PROTOCOL["cases"]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def provenance():
    return {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sampler_sha256": digest(bead_fourier.__file__),
        "ring_helpers_sha256": digest(WORK / "ring_helpers.py"),
        "protocol_sha256": digest(PROTOCOL_PATH), "runner_sha256": digest(__file__),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def local_gram(modes, rule, sites):
    waves = np.pi * np.arange(1, modes + 1)
    if rule == "analytic":
        entries = np.zeros((modes + 2, modes + 2))
        entries[:2, :2] = [[1 / 3, 1 / 6], [1 / 6, 1 / 3]]
        entries[0, 2:] = entries[2:, 0] = np.sqrt(2) / waves**2
        entries[1, 2:] = entries[2:, 1] = (-1)**np.arange(modes) * entries[0, 2:]
        entries[2:, 2:] = np.diag(waves**-2)
        return entries
    if rule == "trapezoid":
        s = np.linspace(0, 1, sites + 1)
        w = np.full(sites + 1, 1 / sites)
        w[[0, -1]] *= 0.5
    else:
        s, w = np.polynomial.legendre.leggauss(sites)
        s, w = (s + 1) / 2, w / 2
    basis = np.column_stack([1 - s, s, np.sqrt(2) * np.sin(s[:, None] * waves) / waves])
    return basis.T @ (w[:, None] * basis)


def potential_matrix(case):
    dimension = case.beads * (1 + case.modes)
    if case.rule == "primitive":
        return np.eye(case.beads)
    matrix = np.zeros((dimension, dimension))
    block = local_gram(case.modes, case.rule, case.sites)
    for j in range(case.beads):
        start = case.beads + j * case.modes
        indices = np.array([j, (j + 1) % case.beads, *range(start, start + case.modes)])
        np.add.at(matrix, (indices[:, None], indices[None, :]), block)
    return matrix


class AnalyticHarmonicPath(BeadFourierPath):
    """Exact harmonic force integration, the special case used by Ivanov."""

    def __init__(self, case):
        super().__init__(case.beads, case.beta, modes=case.modes)
        self.sites_per_path = 0
        self._gram = potential_matrix(case)

    def packed(self, q, b):
        return np.concatenate((q, b.reshape(*q.shape[:-1], self.beads * self.modes)), axis=-1)

    def _physical_forces(self, q, b, force):
        # No external potential evaluations are necessary for this quadratic.
        gradient = self.packed(q, b) @ self._gram
        return q, -gradient[..., :self.beads], -gradient[..., self.beads:].reshape(b.shape)

    def potential_energy(self, positions, amplitudes, potential):
        q, b = self._configuration(positions, amplitudes)
        spring = np.sum((q - np.roll(q, -1, axis=-1))**2, axis=-1) + np.sum(b*b, axis=(-2, -1))
        return 0.5 * (self.beads * self.virial_energy(q, b) + self.mass * self.omega_p**2 * spring)

    def virial_energy(self, q, b):
        x = self.packed(q, b)
        return np.sum((x @ self._gram) * x, axis=-1) / self.beads


class CountedHarmonicForce:
    def __init__(self):
        self.evaluations = 0

    def __call__(self, x):
        self.evaluations += x.size
        return -x


def new_sampler(case, chains, seed):
    rng = np.random.default_rng(seed)
    if case.rule == "analytic":
        path = AnalyticHarmonicPath(case)
    else:
        path = BeadFourierPath(
            case.beads, case.beta, modes=case.modes, quadrature_points=case.sites or None,
            quadrature_rule="trapezoid" if case.rule == "trapezoid" else "gauss-legendre",
        )
    initial_q = rng.normal(size=(chains, 1)) * 0.6
    q = initial_q + 0.3 * rng.normal(size=(chains, path.beads))
    b = rng.normal(size=(*q.shape, path.modes)) * np.sqrt(path.beta_p)
    force = CountedHarmonicForce()
    sampler = BeadFourierPIMD(
        path, force, q, amplitudes=b, time_step=case.dt, rng=rng,
        centroid_friction=1.0 / case.tau0, mass_frequency=PROTOCOL["mass_frequency"],
        splitting=PROTOCOL["splitting"],
    )
    return sampler, force


def observe(sampler, case):
    q = sampler.bead_positions
    bead = (q * q).mean(axis=-1)
    if case.rule == "primitive":
        return bead, bead
    if case.rule == "analytic":
        path = sampler.path.virial_energy(q, sampler.amplitudes)
    else:
        sites = sampler.force_positions
        path = sampler.path.average_at_force_sites(sites * sites)
    return bead, path


def normal_modes(case):
    ring = real_normal_mode_matrix(case.beads)
    count = case.beads * case.modes
    transform = block_diag(ring, np.eye(count))
    omega = np.r_[
        real_normal_mode_frequencies(case.beta, case.beads), np.full(count, case.beads / case.beta)
    ]
    return transform, omega


def linear_reference(case):
    """Exact covariance and asymptotic variance for the linear OBABO chain."""
    transform, frequencies = normal_modes(case)
    physical = potential_matrix(case)
    modes_hessian = transform.T @ physical @ transform
    size = len(frequencies)
    identity, zero = np.eye(size), np.zeros((size, size))
    beta_p = case.beta / case.beads
    stiffness = modes_hessian + np.diag(frequencies**2)
    equilibrium = np.linalg.solve(beta_p * stiffness, identity)
    omega_dt = frequencies * case.dt
    cosine = np.diag(np.cos(omega_dt))
    drift = np.diag(case.dt * np.sinc(omega_dt / np.pi))
    restore = np.diag(-frequencies * np.sin(omega_dt))
    free = np.block([[cosine, drift], [restore, cosine]])
    kick = np.block([[identity, zero], [-0.5 * case.dt * modes_hessian, identity]])
    frictions = 2 * frequencies
    frictions[0] = 1 / case.tau0
    ou = np.diag(np.r_[np.ones(size), np.exp(-frictions * case.dt / 2)])
    half_noise = np.diag(np.r_[np.zeros(size), -np.expm1(-frictions * case.dt) / beta_p])
    left = ou @ kick @ free @ kick
    step = left @ ou
    diffusion = half_noise + left @ half_noise @ left.T
    covariance = solve_discrete_lyapunov(step, diffusion, method="bilinear")
    residual = np.linalg.norm(covariance - step @ covariance @ step.T - diffusion, ord=np.inf)
    radius = float(np.max(np.abs(np.linalg.eigvals(step))))
    if radius >= 1 or residual > 1e-8 or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise RuntimeError(f"{case.name} has an invalid stationary solution")
    bead_quadratic = np.diag(np.arange(size) < case.beads) / case.beads
    path_quadratic = modes_hessian / case.beads
    quantum = 0.5 / np.tanh(case.beta / 2)
    stride_step = np.linalg.matrix_power(step, case.stride)
    estimators = {}
    for label, operator in (("bead", bead_quadratic), ("path", path_quadratic)):
        observable = block_diag(operator, zero)
        stationary_mean = float(np.trace(observable @ covariance))
        finite_mean = float(np.trace(operator @ equilibrium))
        summation = solve_discrete_lyapunov(stride_step.T, observable, method="bilinear")
        cov_projected = covariance @ observable @ covariance
        variance = float(2 * np.sum(observable * cov_projected))
        asymptotic_per_draw = float(4 * np.sum(summation * cov_projected) - variance)
        if not 0 < asymptotic_per_draw < math.inf:
            raise RuntimeError(f"{case.name} invalid autocorrelation sum")
        estimators[label] = {
            "finite_action_energy": finite_mean, "stationary_energy": stationary_mean,
            "finite_action_relative_error": float(finite_mean / quantum - 1),
            "timestep_rel_bias": stationary_mean / finite_mean - 1,
            "variance": variance, "asymptotic_variance_per_draw": asymptotic_per_draw,
            "correlation_samples": asymptotic_per_draw / variance,
        }
    return {"case": asdict(case), "quantum_energy": float(quantum), "stationary_residual": float(residual),
            "spectral_radius": radius, "estimators": estimators}


def finite_action_energy(case):
    transform, frequencies = normal_modes(case)
    hessian = transform.T @ potential_matrix(case) @ transform
    precision = (case.beta / case.beads) * (hessian + np.diag(frequencies**2))
    covariance = np.linalg.solve(precision, np.eye(len(frequencies)))
    return {
        "bead": float(np.trace(covariance[:case.beads, :case.beads]) / case.beads),
        "path": float(np.sum(hessian * covariance) / case.beads),
    }


def check_close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Computed {actual:.12g}, archived {expected:.12g}")


def check_publisher_points():
    london_references = {}
    london_points = set()
    with (WORK / "london_fig1.csv").open(encoding="utf-8", newline="") as table:
        for row in csv.DictReader(table):
            beads, modes, panels = (int(row[k]) for k in ("beads", "segment_modes", "trapezoid_panels"))
            key = beads, modes, panels
            if key not in london_references:
                case = Case("london_check", 8, beads, modes, "trapezoid" if panels else "primitive",
                            panels, 0.01, 50, 250, False)
                london_references[key] = finite_action_energy(case)
                assert case.site_count == int(row["force_sites_per_path_step"])
            point = (*key, row["estimator"])
            if point in london_points:
                raise ValueError(f"Duplicate London marker {point}")
            london_points.add(point)
            value = london_references[key][row["estimator"]]
            check_close(value, float(row["finite_action_energy_au"]))
            low, high, center = (float(row[k]) for k in ("drawn_low_au", "drawn_high_au", "publisher_energy_au"))
            if not low <= value <= high or not low <= center <= high:
                raise ValueError(f"London marker falls outside its recorded bar {point}")
    assert len(london_points) == 70

    ivanov_points = set()
    with (WORK / "ivanov_fig5.csv").open(encoding="utf-8", newline="") as table:
        for row in csv.DictReader(table):
            beads, modes = int(row["beads"]), int(row["segment_modes"])
            point = beads, modes
            if point in ivanov_points:
                raise ValueError(f"Duplicate Ivanov marker {point}")
            ivanov_points.add(point)
            case = Case("ivanov_check", 15.8, beads, modes, "analytic", 0, 0.002, 1, 125, False)
            value = finite_action_energy(case)["bead"]
            check_close(value, float(row["finite_action_endpoint_energy_hbar_omega"]))
            if point == (5, 2):
                assert row["raster_status"] == "overlap" and not row["raster_center_hbar_omega"]
                assert value < float(row["raster_low_hbar_omega"])
    assert len(ivanov_points) == 25
    return len(london_points), len(ivanov_points)


def check_recorded():
    london_points, ivanov_points = check_publisher_points()
    archive = read_json(WORK / "efficiency.json")
    setting = PROTOCOL["timing"]
    chains = PROTOCOL["chains"]
    anchor_cases = ("london_p32", "london_bf4k2", "ivanov_p102", "ivanov_bf5k4")
    for case in CASES:
        saved = archive["reference"][case.name]
        if case.name in anchor_cases:
            fresh = linear_reference(case)
            for label, reference in fresh["estimators"].items():
                for key in ("finite_action_energy", "stationary_energy", "asymptotic_variance_per_draw"):
                    check_close(reference[key], saved["estimators"][label][key])

        timing_row = archive["timing"][case.name]
        assert timing_row["sites_per_path_step"] == case.site_count
        for batch, trials in timing_row["batches"].items():
            assert len(trials) == setting["repeats"] and min(trials) > 0
            check_close(statistics.median(trials), timing_row["medians"][batch])
        for label, estimator in saved["estimators"].items():
            draw_count = estimator["asymptotic_variance_per_draw"] / (
                setting["target_relative_se"] * estimator["stationary_energy"]
            )**2
            batch_steps = draw_count * case.stride / chains
            cost = timing_row["sampling_cost"][label]
            check_close(batch_steps, cost["batch_steps"])
            check_close(batch_steps * timing_row["medians"][str(chains)], cost["wall_seconds"])
            check_close(batch_steps * chains * case.site_count, cost["force_sites"])
    for name, comparison in archive["comparisons"].items():
        left, right = (archive["timing"][comparison[side]] for side in ("left", "right"))
        label = comparison["readout"]
        ratio = right["sampling_cost"][label]["wall_seconds"] / left["sampling_cost"][label]["wall_seconds"]
        check_close(ratio, comparison["stationary_sampling_time_ratio"])
        match = PROTOCOL["comparisons"][name]
        for side in ("left", "right"):
            error = archive["reference"][match[side]]["estimators"][label]["finite_action_relative_error"]
            assert abs(error) <= match["relative_discretization_tolerance"]
    print(f"Verified {london_points} London vector values, {ivanov_points} Ivanov raster entries, "
          f"four stationary references and {len(archive['timing'])} timing rows.")


def run_md(case, seed, *, smoke=False):
    chains = 8 if smoke else PROTOCOL["chains"]
    burn = 100 if smoke else PROTOCOL["thermalize_steps"]
    steps = 2 * case.stride if smoke else PROTOCOL["production_steps"]
    origin = now()
    sampler, force = new_sampler(case, chains, seed)
    for _ in range(burn):
        sampler.step()
    trace = np.empty((2, chains, steps // case.stride))
    for i in range(steps):
        sampler.step()
        if (i + 1) % case.stride == 0:
            trace[..., i // case.stride] = observe(sampler, case)
    sites = (burn + steps + 1) * chains * case.site_count
    if sites != sampler.force_evaluations or sites != force.evaluations or not np.isfinite(trace).all():
        raise RuntimeError(f"{case.name} failed force accounting or produced nonfinite energies")
    payload = {}
    for tag, draws in zip(("bead", "path"), trace):
        means = draws.mean(axis=1)
        stderr = float(means.std(ddof=1) / np.sqrt(chains))
        payload[tag] = {
            "energy": float(means.mean()), "independent_chain_means": means.tolist(), "se": stderr,
            "ci95_half": float(t.ppf(0.975, chains - 1) * stderr),
            "split_rhat": None if smoke else split_rhat(draws),
        }
    return {"schema": 1, "case": asdict(case), "started": origin, "ended": now(),
            "force_sites": sites, "seed": seed, "burn": burn, "steps": steps, "chains": chains,
            "obs": payload}


def validate_md(cell, linear):
    gates = {}
    for readout, sampled in cell["obs"].items():
        stationary = linear["estimators"][readout]["stationary_energy"]
        bound = 3.5 * sampled["se"] + 0.001 * stationary
        gates[readout] = bool(abs(sampled["energy"] - stationary) <= bound
                              and sampled["split_rhat"] < PROTOCOL["max_split_rhat"])
    return gates


def select_cpu():
    try:
        import psutil
        process = psutil.Process()
        available = process.cpu_affinity()
        core = available[min(2, len(available) - 1)]
        process.cpu_affinity([core])
        return process.cpu_affinity()
    except (ImportError, AttributeError, OSError):
        return None


def timing(cases, references):
    setting = PROTOCOL["timing"]
    timing_rows = {case.name: {"case": asdict(case), "sites_per_path_step": case.site_count, "batches": {}}
                   for case in cases}
    affinity = select_cpu()
    started = now()
    for batch in setting["batches"]:
        for trial in range(setting["repeats"]):
            start = trial % len(cases)
            order = cases[start:] + cases[:start]
            for case in order:
                sampler, force = new_sampler(case, batch, PROTOCOL["seed"] + 900 + trial)
                for _ in range(setting["warmup_steps"]):
                    sampler.step()
                before = force.evaluations
                clock = time.perf_counter_ns()
                steps = setting["steps"]
                for i in range(steps):
                    sampler.step()
                    if batch > 1 and (i + 1) % case.stride == 0:
                        observe(sampler, case)
                spent = time.perf_counter_ns() - clock
                if force.evaluations - before != steps * case.site_count * batch:
                    raise RuntimeError("Force counter mismatch in measured loop")
                timing_rows[case.name]["batches"].setdefault(str(batch), []).append(spent / steps * 1e-9)
            print(f"Timing {batch} path(s), round {trial+1}/{setting['repeats']}", flush=True)
    for case in cases:
        row = timing_rows[case.name]
        row["medians"] = {batch: statistics.median(trials) for batch, trials in row["batches"].items()}
        estimators = references[case.name]["estimators"]
        row["sampling_cost"] = {}
        for label, estimator in estimators.items():
            draws = estimator["asymptotic_variance_per_draw"] / (
                setting["target_relative_se"] * estimator["stationary_energy"]
            )**2
            steps = (draws * case.stride) / PROTOCOL["chains"]
            row["sampling_cost"][label] = {
                "batch_steps": steps, "wall_seconds": steps * row["medians"][str(PROTOCOL["chains"])],
                "force_sites": steps * PROTOCOL["chains"] * case.site_count,
            }
    return {"started": started, "ended": now(), "cpu_affinity": affinity, "rows": timing_rows}


def deterministic(cases):
    output = {}
    for case in cases:
        main = linear_reference(case)
        half = linear_reference(replace(case, dt=case.dt / 2, stride=2 * case.stride))
        for tag, values in main["estimators"].items():
            reference = half["estimators"][tag]["stationary_energy"]
            values["dt_refinement_relative_shift"] = values["stationary_energy"] / reference - 1
            if max(abs(values["timestep_rel_bias"]), abs(values["dt_refinement_relative_shift"])) > PROTOCOL["stationary_timestep_relative_tolerance"]:
                raise RuntimeError(f"{case.name}, {tag} failed deterministic timestep gate")
        output[case.name] = main
    for pair in PROTOCOL["comparisons"].values():
        for name in ("left", "right"):
            error = abs(output[pair[name]]["estimators"][pair["readout"]]["finite_action_relative_error"])
            if error > pair["relative_discretization_tolerance"]:
                raise RuntimeError(f"{pair[name]} violates its matched accuracy threshold")
    return output


def run(output, workers, smoke, measure):
    common = provenance()
    if smoke:
        selection = ("london_p32", "london_bf4k1", "ivanov_p102", "ivanov_bf5k2")
        cells = [run_md(case, PROTOCOL["seed"] + i, smoke=True)
                 for i, case in enumerate(CASES) if case.name in selection]
        if output is not None:
            write_json(output / "smoke.json", {"cells": cells, "provenance": common})
        print("four smoke cases passed", flush=True)
        return
    reference_file = output / "reference.json"
    if measure:
        saved_reference = read_json(reference_file)
        assert saved_reference["provenance"] == common
        for i, case in enumerate(CASES):
            if not case.md:
                continue
            cell = read_json(output / f"{case.name}.json")
            assert cell["provenance"] == common and cell["seed"] == PROTOCOL["seed"] + i
            assert all(cell["gates"].values())
        reference = saved_reference["rows"]
        write_json(output / "timing.json", {"provenance": common, "measurement": timing(CASES, reference)})
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_md, case, PROTOCOL["seed"] + index): case
                   for index, case in enumerate(CASES) if case.md}
        reference = deterministic(CASES)
        write_json(reference_file, {"provenance": common, "rows": reference})
        for future in as_completed(futures):
            case = futures[future]
            cell = future.result()
            cell["provenance"] = common
            cell["gates"] = validate_md(cell, reference[case.name])
            write_json(output / f"{case.name}.json", cell)
            print(f"{case.name}: {cell['gates']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--check", action="store_true")
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.check:
        if args.output is not None:
            parser.error("--check uses the bundled data and writes no files")
        check_recorded()
    else:
        if args.output is None and not args.smoke:
            parser.error("provide an output directory for sampling or timing")
        run(args.output, args.workers, args.smoke, args.timing)


if __name__ == "__main__":
    main()
