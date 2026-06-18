"""SAC benchmark for CMD-EH/CMD-FSSH against DVR and ordinary MQC methods."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ensure_repo_imports() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        src = candidate / "src"
        if (src / "toymodel").exists():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return candidate
    raise RuntimeError("Could not locate repository root containing src/toymodel.")


REPO_ROOT = _ensure_repo_imports()

from toymodel.DVRmethods import DVRDensityDynamics  # noqa: E402
from toymodel.methods import (  # noqa: E402
    CMDAdiabaticPMFSurfaces,
    CMD_Ehrenfest,
    CMD_FSSH,
    Ehrenfest,
    FSSH,
)
from toymodel.model.tully_model import TullySimpleAvoidedCrossing  # noqa: E402
from toymodel.utils.constant import kB  # noqa: E402


@dataclass(frozen=True)
class InitialCondition:
    sample_id: int
    q: float
    p: float


@dataclass
class FixedDistribution:
    q: float
    p: float
    mass: float
    T: float
    Ntraj: int = 1

    def sample(self):
        return np.asarray(self.q, dtype=float), np.asarray(self.p, dtype=float)


def make_model(representation: str):
    return TullySimpleAvoidedCrossing(x=None, representation=representation)


def stable_coth(x: float) -> float:
    x = float(x)
    if x > 30.0:
        return 1.0
    if x < 1.0e-8:
        return 1.0 / max(x, 1.0e-300) + x / 3.0
    return float(1.0 / np.tanh(x))


def sho_thermal_widths(*, temperature: float, mass: float, omega: float):
    beta = 1.0 / (kB * float(temperature))
    coth = stable_coth(0.5 * beta * float(omega))
    sigma_q = np.sqrt(coth / (2.0 * float(mass) * float(omega)))
    sigma_p = np.sqrt(0.5 * float(mass) * float(omega) * coth)
    return float(sigma_q), float(sigma_p), float(coth)


def sample_initial_conditions(args, ntraj: int):
    rng = np.random.default_rng(args.seed + 71)
    sigma_q, sigma_p, _ = sho_thermal_widths(
        temperature=args.temperature,
        mass=args.mass,
        omega=args.sho_omega,
    )
    return [
        InitialCondition(
            sample_id=i,
            q=float(args.q0 + rng.normal(0.0, sigma_q)),
            p=float(args.p0 + rng.normal(0.0, sigma_p)),
        )
        for i in range(int(ntraj))
    ]


def initial_population(start_state: int, nstate: int = 2):
    pop = np.zeros(nstate, dtype=float)
    pop[int(start_state)] = 1.0
    return pop


def _append_unique(values, value, *, atol=1.0e-10):
    value = float(value)
    if not values or abs(values[-1] - value) > atol:
        values.append(value)


def _arange_with_endpoint(start: float, stop: float, dx: float):
    start = float(start)
    stop = float(stop)
    dx = float(dx)
    if dx <= 0.0:
        raise ValueError("grid spacing must be positive.")
    if stop < start:
        return np.asarray([], dtype=float)
    n = int(np.floor((stop - start) / dx + 1.0e-12))
    values = [start + i * dx for i in range(n + 1)]
    if not values or abs(values[-1] - stop) > 1.0e-10:
        values.append(stop)
    return np.asarray(values, dtype=float)


def build_cmd_qgrid(args):
    qmin = float(args.cmd_qmin)
    qmax = float(args.cmd_qmax)
    if qmax <= qmin:
        raise ValueError("cmd-qmax must be larger than cmd-qmin.")

    if args.cmd_grid_mode == "uniform":
        if int(args.cmd_qpoints) < 2:
            raise ValueError("cmd-qpoints must be at least 2.")
        return np.linspace(qmin, qmax, int(args.cmd_qpoints)), {
            "mode": "uniform",
            "qmin": qmin,
            "qmax": qmax,
            "qpoints": int(args.cmd_qpoints),
        }

    model = make_model("adiabatic")
    probe = np.linspace(qmin, qmax, int(args.cmd_adaptive_probe_points))
    energies = model.evaluate(probe, need_force=False, need_nac=False).energies
    if energies.shape[1] < 2:
        raise ValueError("adaptive CMD grid requires at least two states.")
    gap = energies[:, 1] - energies[:, 0]
    min_gap = float(np.min(gap))
    center = float(probe[int(np.argmin(gap))])
    threshold = min_gap * float(args.cmd_gap_factor)
    mask = gap <= threshold
    if np.any(mask):
        fine_min = float(np.min(probe[mask]) - float(args.cmd_adaptive_pad))
        fine_max = float(np.max(probe[mask]) + float(args.cmd_adaptive_pad))
    else:
        fine_min = center - float(args.cmd_adaptive_pad)
        fine_max = center + float(args.cmd_adaptive_pad)
    fine_min = max(qmin, fine_min)
    fine_max = min(qmax, fine_max)
    if fine_max <= fine_min:
        fine_min = max(qmin, center - float(args.cmd_dx_fine))
        fine_max = min(qmax, center + float(args.cmd_dx_fine))

    values = []
    for q in _arange_with_endpoint(qmin, fine_min, float(args.cmd_dx_coarse)):
        _append_unique(values, q)
    for q in _arange_with_endpoint(fine_min, fine_max, float(args.cmd_dx_fine)):
        _append_unique(values, q)
    for q in _arange_with_endpoint(fine_max, qmax, float(args.cmd_dx_coarse)):
        _append_unique(values, q)
    qgrid = np.asarray(sorted(set(round(v, 12) for v in values)), dtype=float)
    if qgrid[0] > qmin + 1.0e-10:
        qgrid = np.r_[qmin, qgrid]
    if qgrid[-1] < qmax - 1.0e-10:
        qgrid = np.r_[qgrid, qmax]
    return qgrid, {
        "mode": "adaptive",
        "qmin": qmin,
        "qmax": qmax,
        "qpoints": int(qgrid.size),
        "fine_min": float(fine_min),
        "fine_max": float(fine_max),
        "gap_min": min_gap,
        "gap_center": center,
        "gap_threshold": threshold,
        "dx_fine": float(args.cmd_dx_fine),
        "dx_coarse": float(args.cmd_dx_coarse),
    }


def prepend_initial(pop: np.ndarray, start_state: int):
    arr = np.asarray(pop, dtype=float)
    return np.vstack([initial_population(start_state, arr.shape[-1]), arr])


def run_dvr_density(args):
    solver = DVRDensityDynamics(
        model=make_model("diabatic"),
        total_time=float(args.t_final),
        dt=float(args.dt),
        ndvr=int(args.dvr_ndvr),
        xbound=float(args.dvr_xbound),
        m=float(args.mass),
        temperature=float(args.temperature),
        basis_ordering="grid-major",
        grid_convention=args.dvr_grid_convention,
        kinetic_operator=args.dvr_kinetic_operator,
        initial_representation="adiabatic",
        initial_state=int(args.start_state),
        x0_init=float(args.q0),
        omega_init=float(args.sho_omega),
        use_positive_momentum_only=False,
        preparation_kinetic_operator=args.dvr_preparation_kinetic_operator,
        population_partition="half-grid",
        store_density_matrix=False,
    )

    tic = time.perf_counter()
    if abs(args.p0) <= 1.0e-14:
        result = solver.run()
    else:
        solver.build()
        rho_nuc = solver._build_default_nuclear_density()
        x = solver.xgrid
        p_nyquist = np.pi / solver.dx
        if abs(args.p0) >= 0.85 * p_nyquist:
            raise ValueError(
                "DVR grid is too coarse for the requested boost: "
                f"|p0|={abs(args.p0):.6g}, p_nyquist={p_nyquist:.6g}."
            )
        boost = np.exp(1j * float(args.p0) * (x[:, None] - x[None, :]))
        solver.initial_nuclear_density = boost * rho_nuc
        solver.construct_initial_density_matrix()
        solver._construct_absorber_masks()
        result = solver.compute()
    wall = time.perf_counter() - tic

    pop = np.asarray(result["adiabatic_pop"], dtype=float)
    trace = np.asarray(result["trace"], dtype=float)
    return {
        "times": np.asarray(result["times"], dtype=float),
        "pop": pop,
        "trace": trace,
        "mean_x": np.asarray(result["mean_x"], dtype=float),
        "mean_p": np.asarray(result["mean_p"], dtype=float),
        "wall_time_sec": wall,
    }


def build_cmd_surface(args, *, nsteps: int, seed_offset: int = 0):
    qgrid, _ = build_cmd_qgrid(args)
    return CMDAdiabaticPMFSurfaces.from_model(
        model=make_model("adiabatic"),
        qgrid=qgrid,
        temperature=float(args.temperature),
        nbeads=int(args.cmd_nbeads),
        mass=float(args.mass),
        nsteps=int(nsteps),
        burnin=float(args.cmd_burnin),
        step_size=float(args.cmd_step_size),
        sample_stride=int(args.cmd_sample_stride),
        seed=int(args.seed + seed_offset),
        reference_q=float(args.cmd_reference_q),
    )


def align_cmd_surface_to_reactant_asymptote(surface, args, *, nsteps: int):
    qgrid = np.asarray(surface.qgrid, dtype=float)
    mask = qgrid <= float(args.cmd_reactant_max)
    if not np.any(mask):
        cutoff = np.quantile(qgrid, 0.2)
        mask = qgrid <= cutoff
    qref = qgrid[mask]
    classical = make_model("adiabatic").evaluate(
        qref,
        need_force=False,
        need_nac=False,
    ).energies
    rows = []
    for state in range(surface.nstate):
        shift = float(np.mean(classical[:, state] - surface.pmf[mask, state]))
        surface.pmf[:, state] += shift
        residual = surface.pmf[mask, state] - classical[:, state]
        rows.append(
            {
                "nsteps": int(nsteps),
                "state": state,
                "qmin": float(np.min(qref)),
                "qmax": float(np.max(qref)),
                "npoints": int(qref.size),
                "energy_shift": shift,
                "mean_residual_after_shift": float(np.mean(residual)),
                "max_abs_residual_after_shift": float(np.max(np.abs(residual))),
                "rms_residual_after_shift": float(np.sqrt(np.mean(residual * residual))),
            }
        )
    surface.__post_init__()
    return rows


def run_cmd_pimc_convergence(args, outdir: Path):
    levels = [int(x) for x in args.cmd_pimc_levels]
    qgrid, grid_meta = build_cmd_qgrid(args)
    grid_rows = []
    for i, qval in enumerate(qgrid):
        grid_rows.append(
            {
                "index": i,
                "q": float(qval),
                "dx_prev": "" if i == 0 else float(qval - qgrid[i - 1]),
            }
        )
    write_csv(outdir / "cmd_qgrid.csv", grid_rows)
    (outdir / "cmd_qgrid_metadata.json").write_text(
        json.dumps(grid_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    surfaces = []
    rows = []
    alignment_rows = []
    for ilevel, nsteps in enumerate(levels):
        tic = time.perf_counter()
        surface = build_cmd_surface(args, nsteps=nsteps, seed_offset=10_000 * (ilevel + 1))
        alignment_rows.extend(
            align_cmd_surface_to_reactant_asymptote(
                surface,
                args,
                nsteps=nsteps,
            )
        )
        wall = time.perf_counter() - tic
        surfaces.append(surface)
        rows.append(
            {
                "level_index": ilevel,
                "nsteps": nsteps,
                "wall_time_sec": wall,
                "acceptance_min": float(np.min(surface.acceptance_rate)),
                "acceptance_mean": float(np.mean(surface.acceptance_rate)),
                "acceptance_max": float(np.max(surface.acceptance_rate)),
                "pmf_min": float(np.min(surface.pmf)),
                "pmf_max": float(np.max(surface.pmf)),
                "alignment_qmax": float(args.cmd_reactant_max),
            }
        )

    write_csv(outdir / "cmd_asymptote_alignment.csv", alignment_rows)

    diff_rows = []
    for i in range(1, len(surfaces)):
        diff = surfaces[i].pmf - surfaces[i - 1].pmf
        diff_rows.append(
            {
                "from_nsteps": levels[i - 1],
                "to_nsteps": levels[i],
                "max_abs_pmf_diff": float(np.max(np.abs(diff))),
                "rms_pmf_diff": float(np.sqrt(np.mean(diff * diff))),
            }
        )

    write_csv(outdir / "cmd_pimc_levels.csv", rows)
    write_csv(outdir / "cmd_pimc_diffs.csv", diff_rows)
    np.savez_compressed(
        outdir / "cmd_surface_final.npz",
        qgrid=surfaces[-1].qgrid,
        pmf=surfaces[-1].pmf,
        mean_force=surfaces[-1].mean_force,
        force_stderr=surfaces[-1].force_stderr,
        acceptance_rate=surfaces[-1].acceptance_rate,
        npaths=surfaces[-1].npaths,
        temperature=surfaces[-1].temperature,
        beta=surfaces[-1].beta,
        nbeads=surfaces[-1].nbeads,
        mass=surfaces[-1].mass,
        reference_q=surfaces[-1].reference_q,
    )
    plot_cmd_pimc(outdir / "cmd_pimc_convergence.png", surfaces, levels)
    barrier_rows = write_cmd_barrier_outputs(outdir, surfaces[-1], args)
    return surfaces[-1], rows, diff_rows, barrier_rows


def run_method_sample(args, method_name: str, condition: InitialCondition, cmd_surface=None):
    seed = args.seed + 1_000_003 * int(condition.sample_id) + 7919 * len(method_name)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    dist = FixedDistribution(
        q=condition.q,
        p=condition.p,
        mass=float(args.mass),
        T=float(args.temperature),
    )
    common = {
        "distribution": dist,
        "Nstep": int(round(args.t_final / args.dt)),
        "dt": float(args.dt),
        "start_state": int(args.start_state),
        "is_record": True,
        "legacy_result": False,
        "record_level": "minimal",
        "verbose": False,
    }
    if method_name == "EH":
        method = Ehrenfest(model=make_model("adiabatic"), **common)
    elif method_name == "FSSH":
        method = FSSH(model=make_model("adiabatic"), **common)
    elif method_name == "CMD_EH":
        method = CMD_Ehrenfest(
            model=make_model("adiabatic"),
            cmd_surface=cmd_surface,
            **common,
        )
    elif method_name == "CMD_FSSH":
        method = CMD_FSSH(
            model=make_model("adiabatic"),
            cmd_surface=cmd_surface,
            **common,
        )
    else:
        raise ValueError(f"unknown method {method_name!r}")
    method.run()
    record = method.result[0] if isinstance(method.result, list) else method.result
    pop = prepend_initial(record["pop"], args.start_state)
    if not np.all(np.isfinite(pop)):
        raise FloatingPointError(f"{method_name} produced non-finite populations.")
    return pop


def run_method_ensemble(args, method_name: str, conditions, cmd_surface=None):
    tic = time.perf_counter()
    pops = []
    failures = []
    for condition in conditions:
        try:
            pops.append(run_method_sample(args, method_name, condition, cmd_surface=cmd_surface))
        except Exception as exc:
            failures.append({"sample_id": int(condition.sample_id), "error": repr(exc)})
    if not pops:
        raise RuntimeError(f"all {method_name} samples failed: {failures[:3]}")
    stack = np.asarray(pops, dtype=float)
    mean = np.mean(stack, axis=0)
    stderr = np.std(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(mean)
    return {
        "method": method_name,
        "stack": stack,
        "mean": mean,
        "stderr": stderr,
        "n_completed": int(stack.shape[0]),
        "n_expected": int(len(conditions)),
        "failures": failures,
        "wall_time_sec": time.perf_counter() - tic,
    }


def prefix_mean_stderr(stack: np.ndarray, n: int):
    sub = np.asarray(stack[: int(n)], dtype=float)
    mean = np.mean(sub, axis=0)
    stderr = np.std(sub, axis=0, ddof=1) / np.sqrt(sub.shape[0]) if sub.shape[0] > 1 else np.zeros_like(mean)
    return mean, stderr


def run_fssh_convergence(args, outdir: Path, conditions, cmd_surface):
    rows = []
    stacks = {}
    for method_name in ("FSSH", "CMD_FSSH"):
        result = run_method_ensemble(args, method_name, conditions, cmd_surface=cmd_surface)
        stacks[method_name] = result["stack"]
        for n in args.fssh_ntraj_levels:
            n = int(n)
            if n > result["stack"].shape[0]:
                continue
            mean, stderr = prefix_mean_stderr(result["stack"], n)
            rows.append(
                {
                    "method": method_name,
                    "ntraj": n,
                    "final_pop0": float(mean[-1, 0]),
                    "final_pop1": float(mean[-1, 1]),
                    "final_stderr_pop1": float(stderr[-1, 1]),
                    "pop_sum_min": float(np.min(np.sum(mean, axis=1))),
                    "pop_sum_max": float(np.max(np.sum(mean, axis=1))),
                    "wall_time_sec_total": float(result["wall_time_sec"]),
                    "n_completed_total": int(result["n_completed"]),
                    "n_failures": int(len(result["failures"])),
                }
            )
    diff_rows = []
    for method_name in ("FSSH", "CMD_FSSH"):
        method_rows = [r for r in rows if r["method"] == method_name]
        method_rows = sorted(method_rows, key=lambda row: row["ntraj"])
        for prev, cur in zip(method_rows, method_rows[1:]):
            prev_mean, _ = prefix_mean_stderr(stacks[method_name], int(prev["ntraj"]))
            cur_mean, _ = prefix_mean_stderr(stacks[method_name], int(cur["ntraj"]))
            diff = cur_mean[:, 1] - prev_mean[:, 1]
            diff_rows.append(
                {
                    "method": method_name,
                    "from_ntraj": int(prev["ntraj"]),
                    "to_ntraj": int(cur["ntraj"]),
                    "final_abs_diff_pop1": float(abs(cur_mean[-1, 1] - prev_mean[-1, 1])),
                    "rms_diff_pop1": float(np.sqrt(np.mean(diff * diff))),
                }
            )
    write_csv(outdir / "fssh_convergence.csv", rows)
    write_csv(outdir / "fssh_convergence_diffs.csv", diff_rows)
    plot_fssh_convergence(outdir / "fssh_convergence.png", rows, diff_rows)
    return stacks, rows, diff_rows


def compare_to_dvr(times, dvr_pop, method_results):
    rows = []
    for name, result in method_results.items():
        mean = result["mean"]
        diff = mean[:, 1] - dvr_pop[:, 1]
        rows.append(
            {
                "method": name,
                "n_completed": int(result["n_completed"]),
                "final_pop1": float(mean[-1, 1]),
                "dvr_final_pop1": float(dvr_pop[-1, 1]),
                "final_abs_error_pop1": float(abs(diff[-1])),
                "rms_error_pop1_vs_dvr": float(np.sqrt(np.mean(diff * diff))),
                "max_abs_error_pop1_vs_dvr": float(np.max(np.abs(diff))),
                "pop_sum_min": float(np.min(np.sum(mean, axis=1))),
                "pop_sum_max": float(np.max(np.sum(mean, axis=1))),
                "wall_time_sec": float(result["wall_time_sec"]),
            }
        )
    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def plot_population(path: Path, times, dvr, method_results):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    labels = {
        "EH": "Ehrenfest",
        "FSSH": "FSSH",
        "CMD_EH": "CMD-EH",
        "CMD_FSSH": "CMD-FSSH",
    }
    for state, ax in enumerate(axes):
        ax.plot(times, dvr[:, state], color="black", lw=2.4, label="DVR")
        for name, result in method_results.items():
            mean = result["mean"]
            stderr = result["stderr"]
            ax.plot(times, mean[:, state], lw=1.6, label=labels.get(name, name))
            if name.endswith("FSSH"):
                ax.fill_between(
                    times,
                    mean[:, state] - 2.0 * stderr[:, state],
                    mean[:, state] + 2.0 * stderr[:, state],
                    alpha=0.14,
                    linewidth=0,
                )
        ax.set_title(f"Adiabatic state {state}")
        ax.set_xlabel("time / a.u.")
        ax.set_ylabel("population")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cmd_pimc(path: Path, surfaces, levels):
    nstate = surfaces[-1].nstate
    fig, axes = plt.subplots(1, nstate, figsize=(5.0 * nstate, 3.8), squeeze=False)
    for state, ax in enumerate(axes[0]):
        for surface, level in zip(surfaces, levels):
            ax.plot(surface.qgrid, surface.pmf[:, state], marker="o", ms=3, label=f"{level}")
        ax.set_title(f"CMD PMF state {state}")
        ax.set_xlabel("centroid q")
        ax.set_ylabel("PMF / Hartree")
        ax.legend(title="PIMC steps", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cmd_energies_on_grid(surface, qgrid):
    return np.asarray([surface.energies(np.array([q], dtype=float)) for q in qgrid])


def cmd_barrier_metrics(surface, args):
    qplot = np.linspace(float(surface.qgrid[0]), float(surface.qgrid[-1]), 2001)
    classical = make_model("adiabatic").evaluate(
        qplot,
        need_force=False,
        need_nac=False,
    ).energies
    cmd = _cmd_energies_on_grid(surface, qplot)
    left_mask = qplot <= float(args.cmd_reactant_max)
    if not np.any(left_mask):
        left_mask = qplot <= np.quantile(qplot, 0.2)
    barrier_mask = np.abs(qplot - float(args.cmd_barrier_center)) <= float(args.cmd_barrier_radius)
    if not np.any(barrier_mask):
        barrier_mask = np.abs(qplot - float(args.cmd_barrier_center)) <= np.quantile(
            np.abs(qplot - float(args.cmd_barrier_center)),
            0.1,
        )

    rows = []
    for state in range(surface.nstate):
        e_left = float(np.min(classical[left_mask, state]))
        a_left = float(np.min(cmd[left_mask, state]))
        e_barrier = float(np.max(classical[barrier_mask, state]))
        a_barrier = float(np.max(cmd[barrier_mask, state]))
        rows.append(
            {
                "state": state,
                "left_region_qmax": float(np.max(qplot[left_mask])),
                "barrier_region_qmin": float(np.min(qplot[barrier_mask])),
                "barrier_region_qmax": float(np.max(qplot[barrier_mask])),
                "adiabatic_left_min": e_left,
                "cmd_left_min": a_left,
                "adiabatic_barrier_max": e_barrier,
                "cmd_barrier_max": a_barrier,
                "adiabatic_barrier_height": e_barrier - e_left,
                "cmd_barrier_height": a_barrier - a_left,
                "cmd_barrier_lowering": (e_barrier - e_left) - (a_barrier - a_left),
            }
        )
    return rows


def plot_cmd_vs_adiabatic(path: Path, surface, args, barrier_rows):
    qplot = np.linspace(float(surface.qgrid[0]), float(surface.qgrid[-1]), 1200)
    classical = make_model("adiabatic").evaluate(
        qplot,
        need_force=False,
        need_nac=False,
    ).energies
    cmd = _cmd_energies_on_grid(surface, qplot)
    nstate = surface.nstate
    fig, axes = plt.subplots(1, nstate, figsize=(5.4 * nstate, 4.0), squeeze=False)
    for state, ax in enumerate(axes[0]):
        ax.plot(qplot, classical[:, state], color="black", lw=2.0, ls="--", label="adiabatic E")
        ax.plot(qplot, cmd[:, state], color=f"C{state}", lw=2.1, label="CMD PMF")
        ax.scatter(surface.qgrid, surface.pmf[:, state], color=f"C{state}", s=12, alpha=0.65)
        ax.axvspan(
            float(args.cmd_barrier_center) - float(args.cmd_barrier_radius),
            float(args.cmd_barrier_center) + float(args.cmd_barrier_radius),
            color="0.5",
            alpha=0.08,
            linewidth=0,
        )
        if state == 0 and barrier_rows:
            lowering = barrier_rows[0]["cmd_barrier_lowering"]
            ax.text(
                0.03,
                0.05,
                f"barrier lowering = {lowering:.3e} Ha",
                transform=ax.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
            )
        ax.set_title(f"state {state}")
        ax.set_xlabel("centroid q")
        ax.set_ylabel("energy / Hartree")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_cmd_barrier_outputs(outdir: Path, surface, args):
    barrier_rows = cmd_barrier_metrics(surface, args)
    write_csv(outdir / "cmd_barrier_metrics.csv", barrier_rows)
    plot_cmd_vs_adiabatic(
        outdir / "cmd_pmf_vs_adiabatic.png",
        surface,
        args,
        barrier_rows,
    )
    return barrier_rows


def plot_fssh_convergence(path: Path, rows, diff_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for method in sorted({row["method"] for row in rows}):
        method_rows = sorted([row for row in rows if row["method"] == method], key=lambda row: row["ntraj"])
        axes[0].errorbar(
            [row["ntraj"] for row in method_rows],
            [row["final_pop1"] for row in method_rows],
            yerr=[2.0 * row["final_stderr_pop1"] for row in method_rows],
            marker="o",
            capsize=3,
            label=method,
        )
        method_diffs = sorted(
            [row for row in diff_rows if row["method"] == method],
            key=lambda row: row["to_ntraj"],
        )
        if method_diffs:
            axes[1].plot(
                [row["to_ntraj"] for row in method_diffs],
                [row["rms_diff_pop1"] for row in method_diffs],
                marker="o",
                label=method,
            )
    axes[0].set_xlabel("ntraj")
    axes[0].set_ylabel("final P1")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("to ntraj")
    axes[1].set_ylabel("prefix RMS diff in P1")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    *,
    args,
    summary_rows,
    cmd_diff_rows,
    fssh_diff_rows,
    dvr,
    cmd_surface=None,
    barrier_rows=None,
):
    cmd_last = cmd_diff_rows[-1] if cmd_diff_rows else {}
    fssh_last = {
        row["method"]: row
        for row in fssh_diff_rows
        if row["to_ntraj"] == max([r["to_ntraj"] for r in fssh_diff_rows], default=-1)
    }
    lines = [
        "# SAC CMD Benchmark Report",
        "",
        "## Setup",
        "",
        f"- q0 = {args.q0}, p0 = {args.p0}, T = {args.temperature} K, omega = {args.sho_omega}",
        f"- t_final = {args.t_final}, dt = {args.dt}",
        f"- DVR ndvr = {args.dvr_ndvr}, xbound = {args.dvr_xbound}",
        "- CMD qgrid = "
        f"[{args.cmd_qmin}, {args.cmd_qmax}] with "
        f"{cmd_surface.qgrid.size if cmd_surface is not None else args.cmd_qpoints} points, "
        f"mode = {args.cmd_grid_mode}, nbeads = {args.cmd_nbeads}",
        "",
        "## Convergence",
        "",
        f"- DVR trace range: [{float(np.min(dvr['trace'])):.8f}, {float(np.max(dvr['trace'])):.8f}]",
    ]
    if cmd_last:
        lines.append(
            "- CMD PIMC highest-level PMF change: "
            f"max_abs={cmd_last['max_abs_pmf_diff']:.6e}, rms={cmd_last['rms_pmf_diff']:.6e}"
        )
    for method, row in sorted(fssh_last.items()):
        lines.append(
            f"- {method} trajectory convergence at n={row['to_ntraj']}: "
            f"final_abs_diff={row['final_abs_diff_pop1']:.6e}, "
            f"rms_diff={row['rms_diff_pop1']:.6e}"
        )
    if barrier_rows:
        row = barrier_rows[0]
        lines.append(
            "- State-0 CMD barrier lowering from left reactant basin: "
            f"{row['cmd_barrier_lowering']:.6e} Ha "
            f"(adiabatic={row['adiabatic_barrier_height']:.6e}, "
            f"CMD={row['cmd_barrier_height']:.6e})"
        )
    lines += ["", "## Comparison", ""]
    for row in summary_rows:
        lines.append(
            f"- {row['method']}: final P1={row['final_pop1']:.6f}, "
            f"DVR final P1={row['dvr_final_pop1']:.6f}, "
            f"RMS error={row['rms_error_pop1_vs_dvr']:.6e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nstep = int(round(args.t_final / args.dt))
    args.t_final = nstep * args.dt

    args_json = {}
    for key, value in vars(args).items():
        args_json[key] = str(value) if isinstance(value, Path) else value
    metadata = {
        "args": args_json,
        "repo_root": str(REPO_ROOT),
        "schema": "toymodel.sac_cmd_benchmark.v1",
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[cmd-pimc] start", flush=True)
    cmd_surface, cmd_rows, cmd_diff_rows, barrier_rows = run_cmd_pimc_convergence(args, outdir)
    print("[cmd-pimc] done", flush=True)

    if args.pmf_only:
        print(f"[done] wrote PMF outputs to {outdir}", flush=True)
        return {
            "outdir": str(outdir),
            "cmd_pimc": cmd_diff_rows,
            "cmd_barrier": barrier_rows,
        }

    print("[dvr] start", flush=True)
    dvr = run_dvr_density(args)
    print(
        f"[dvr] done final_P1={dvr['pop'][-1,1]:.6f} "
        f"trace=[{np.min(dvr['trace']):.8f},{np.max(dvr['trace']):.8f}]",
        flush=True,
    )

    max_ntraj = max(int(n) for n in args.fssh_ntraj_levels)
    conditions = sample_initial_conditions(args, max_ntraj)
    np.savez_compressed(
        outdir / "initial_conditions.npz",
        q=np.asarray([c.q for c in conditions], dtype=float),
        p=np.asarray([c.p for c in conditions], dtype=float),
        sample_id=np.asarray([c.sample_id for c in conditions], dtype=int),
    )

    print("[fssh-conv] start", flush=True)
    fssh_stacks, fssh_rows, fssh_diff_rows = run_fssh_convergence(
        args,
        outdir,
        conditions,
        cmd_surface,
    )
    print("[fssh-conv] done", flush=True)

    method_results = {}
    for method_name in ("EH", "CMD_EH"):
        print(f"[{method_name}] start", flush=True)
        method_results[method_name] = run_method_ensemble(
            args,
            method_name,
            conditions,
            cmd_surface=cmd_surface,
        )
        print(f"[{method_name}] done", flush=True)
    for method_name in ("FSSH", "CMD_FSSH"):
        stack = fssh_stacks[method_name]
        mean, stderr = prefix_mean_stderr(stack, max_ntraj)
        method_results[method_name] = {
            "method": method_name,
            "stack": stack,
            "mean": mean,
            "stderr": stderr,
            "n_completed": int(stack.shape[0]),
            "n_expected": int(max_ntraj),
            "failures": [],
            "wall_time_sec": 0.0,
        }

    times = np.asarray(dvr["times"], dtype=float)
    summary_rows = compare_to_dvr(times, dvr["pop"], method_results)
    write_csv(outdir / "summary.csv", summary_rows)
    np.savez_compressed(
        outdir / "populations.npz",
        times=times,
        DVR=dvr["pop"],
        EH=method_results["EH"]["mean"],
        EH_stderr=method_results["EH"]["stderr"],
        FSSH=method_results["FSSH"]["mean"],
        FSSH_stderr=method_results["FSSH"]["stderr"],
        CMD_EH=method_results["CMD_EH"]["mean"],
        CMD_EH_stderr=method_results["CMD_EH"]["stderr"],
        CMD_FSSH=method_results["CMD_FSSH"]["mean"],
        CMD_FSSH_stderr=method_results["CMD_FSSH"]["stderr"],
        dvr_trace=dvr["trace"],
    )
    plot_population(outdir / "population_comparison.png", times, dvr["pop"], method_results)
    write_report(
        outdir / "report.md",
        args=args,
        summary_rows=summary_rows,
        cmd_diff_rows=cmd_diff_rows,
        fssh_diff_rows=fssh_diff_rows,
        dvr=dvr,
        cmd_surface=cmd_surface,
        barrier_rows=barrier_rows,
    )
    print(f"[done] wrote {outdir}", flush=True)
    return {
        "outdir": str(outdir),
        "summary": summary_rows,
        "cmd_pimc": cmd_diff_rows,
        "fssh_convergence": fssh_diff_rows,
        "cmd_barrier": barrier_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("output/sac_cmd_benchmark/smoke"))
    parser.add_argument("--pmf-only", action="store_true")
    parser.add_argument("--q0", type=float, default=-4.0)
    parser.add_argument("--p0", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sho-omega", type=float, default=0.004)
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--t-final", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--dvr-ndvr", type=int, default=256)
    parser.add_argument("--dvr-xbound", type=float, default=12.0)
    parser.add_argument("--dvr-grid-convention", choices=["left-edge", "cell-centered"], default="left-edge")
    parser.add_argument("--dvr-kinetic-operator", choices=["particle_in_box", "sinc", "fft_periodic"], default="particle_in_box")
    parser.add_argument("--dvr-preparation-kinetic-operator", choices=["same-as-main", "fft"], default="same-as-main")
    parser.add_argument("--cmd-qmin", type=float, default=-6.0)
    parser.add_argument("--cmd-qmax", type=float, default=4.0)
    parser.add_argument("--cmd-qpoints", type=int, default=21)
    parser.add_argument("--cmd-grid-mode", choices=["uniform", "adaptive"], default="uniform")
    parser.add_argument("--cmd-dx-fine", type=float, default=0.1)
    parser.add_argument("--cmd-dx-coarse", type=float, default=0.5)
    parser.add_argument("--cmd-gap-factor", type=float, default=1.6)
    parser.add_argument("--cmd-adaptive-pad", type=float, default=0.8)
    parser.add_argument("--cmd-adaptive-probe-points", type=int, default=2001)
    parser.add_argument("--cmd-reference-q", type=float, default=-5.0)
    parser.add_argument("--cmd-reactant-max", type=float, default=-3.0)
    parser.add_argument("--cmd-barrier-center", type=float, default=0.0)
    parser.add_argument("--cmd-barrier-radius", type=float, default=0.5)
    parser.add_argument("--cmd-nbeads", type=int, default=8)
    parser.add_argument("--cmd-pimc-levels", nargs="+", type=int, default=[250, 750])
    parser.add_argument("--cmd-burnin", type=float, default=0.25)
    parser.add_argument("--cmd-step-size", type=float, default=0.08)
    parser.add_argument("--cmd-sample-stride", type=int, default=8)
    parser.add_argument("--fssh-ntraj-levels", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
