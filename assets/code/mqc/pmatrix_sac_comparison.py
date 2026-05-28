"""Run a Tully simple-avoided-crossing comparison.

The script compares the P-Matrix implementation against FSSH, Ehrenfest, and
exact DVR wave-packet propagation on the same SAC toy model.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ensure_repo_imports():
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

from toymodel.DVRmethods import DVRWaveDynamics  # noqa: E402
from toymodel.distribution import UniformDistribution  # noqa: E402
from toymodel.methods import Ehrenfest, FSSH, PMatrix  # noqa: E402
from toymodel.model import TullySimpleAvoidedCrossing  # noqa: E402


def _with_initial_population(pop, start_state, nstate=2):
    initial = np.zeros((1, nstate), dtype=float)
    initial[0, start_state] = 1.0
    return np.vstack([initial, np.asarray(pop, dtype=float)])


def _fixed_distribution(args, ntraj):
    return UniformDistribution(
        Ntraj=ntraj,
        T=args.temperature,
        q0=args.q0,
        pmin=args.p0,
        pmax=args.p0,
        mass=args.mass,
    )


def run_pmatrix(args):
    model = TullySimpleAvoidedCrossing(representation="adiabatic")
    method = PMatrix(
        model=model,
        distribution=_fixed_distribution(args, 1),
        Nstep=args.nstep,
        dt=args.dt,
        start_state=args.start_state,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        decoherence_time=args.tau,
        electronic_substeps=args.electronic_substeps,
        detailed_balance=not args.no_detailed_balance,
        nuclear_force=args.pmatrix_nuclear_force,
        reference_state=args.reference_state,
        verbose=False,
    )
    method.run()
    result = method.result
    return _with_initial_population(result["pop"], args.start_state)


def run_ehrenfest(args):
    model = TullySimpleAvoidedCrossing(representation="adiabatic")
    method = Ehrenfest(
        model=model,
        distribution=_fixed_distribution(args, 1),
        Nstep=args.nstep,
        dt=args.dt,
        start_state=args.start_state,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        verbose=False,
    )
    method.run()
    result = method.result
    return _with_initial_population(result["pop"], args.start_state)


def run_fssh(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = TullySimpleAvoidedCrossing(representation="adiabatic")
    method = FSSH(
        model=model,
        distribution=_fixed_distribution(args, args.ntraj),
        Nstep=args.nstep,
        dt=args.dt,
        start_state=args.start_state,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        verbose=False,
    )
    method.run()
    traj_pops = [np.asarray(traj["pop"], dtype=float) for traj in method.result]
    avg = np.mean(traj_pops, axis=0)
    return _with_initial_population(avg, args.start_state)


def run_dvr(args):
    model = TullySimpleAvoidedCrossing(representation="adiabatic")
    solver = DVRWaveDynamics(
        model=model,
        total_time=args.total_time,
        dt=args.dt,
        ndvr=args.dvr_ndvr,
        xbound=args.dvr_xbound,
        startstate=args.start_state,
        sigma=args.sigma,
        x0=args.q0,
        p0=args.p0,
        m=args.mass,
        is_plot=False,
        basis_ordering="grid-major",
        grid_convention="left-edge",
        kinetic_operator="particle_in_box",
    )
    solver.initialize()
    result = solver.compute()
    return np.asarray(result["pop"], dtype=float).sum(axis=1)


def _write_population_csv(path, times, pops):
    fields = ["time"]
    for name in pops:
        fields.extend([f"{name}_pop0", f"{name}_pop1"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(times):
            row = {"time": float(t)}
            for name, pop in pops.items():
                row[f"{name}_pop0"] = float(pop[i, 0])
                row[f"{name}_pop1"] = float(pop[i, 1])
            writer.writerow(row)


def _write_summary_csv(path, pops):
    dvr = pops["dvr"][:, 1]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "final_pop0",
                "final_pop1",
                "rms_upper_vs_dvr",
                "max_abs_upper_vs_dvr",
            ],
        )
        writer.writeheader()
        for name, pop in pops.items():
            diff = pop[:, 1] - dvr
            writer.writerow(
                {
                    "method": name,
                    "final_pop0": float(pop[-1, 0]),
                    "final_pop1": float(pop[-1, 1]),
                    "rms_upper_vs_dvr": float(np.sqrt(np.mean(diff**2))),
                    "max_abs_upper_vs_dvr": float(np.max(np.abs(diff))),
                }
            )


def _plot(path, times, pops, args):
    colors = {
        "pmatrix": "#1f77b4",
        "fssh": "#d62728",
        "ehrenfest": "#2ca02c",
        "dvr": "#111111",
    }
    labels = {
        "pmatrix": "P-Matrix",
        "fssh": "FSSH",
        "ehrenfest": "Ehrenfest",
        "dvr": "DVR",
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    for name, pop in pops.items():
        axes[0].plot(times, pop[:, 1], label=labels[name], color=colors[name], lw=2)
        axes[1].plot(times, pop[:, 0], label=labels[name], color=colors[name], lw=2)

    axes[0].set_ylabel("Upper adiabatic population")
    axes[1].set_ylabel("Lower adiabatic population")
    axes[1].set_xlabel("Time (a.u.)")
    axes[0].set_title(
        "Tully simple avoided crossing "
        f"(q0={args.q0:g}, p0={args.p0:g}, start={args.start_state})"
    )
    for ax in axes:
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-time", type=float, default=4000.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--q0", type=float, default=-10.0)
    parser.add_argument("--p0", type=float, default=10.0)
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--start-state", type=int, default=1)
    parser.add_argument("--ntraj", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tau", type=float, default=200.0)
    parser.add_argument("--electronic-substeps", type=int, default=10)
    parser.add_argument("--no-detailed-balance", action="store_true")
    parser.add_argument(
        "--pmatrix-nuclear-force",
        choices=["density", "reference", "free"],
        default="density",
    )
    parser.add_argument("--reference-state", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--dvr-ndvr", type=int, default=256)
    parser.add_argument("--dvr-xbound", type=float, default=25.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "sac_pmatrix_comparison",
    )
    args = parser.parse_args()
    args.nstep = int(round(args.total_time / args.dt))
    args.total_time = args.nstep * args.dt
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = np.arange(args.nstep + 1, dtype=float) * args.dt

    pops = {
        "pmatrix": run_pmatrix(args),
        "fssh": run_fssh(args),
        "ehrenfest": run_ehrenfest(args),
        "dvr": run_dvr(args),
    }

    for name, pop in pops.items():
        if pop.shape != (len(times), 2):
            raise RuntimeError(f"{name} returned shape {pop.shape}, expected {(len(times), 2)}.")

    np.savez(
        args.output_dir / "populations.npz",
        time=times,
        **{name: pop for name, pop in pops.items()},
    )
    _write_population_csv(args.output_dir / "populations.csv", times, pops)
    _write_summary_csv(args.output_dir / "summary.csv", pops)
    _plot(args.output_dir / "population_compare.png", times, pops, args)

    print(f"wrote {args.output_dir}")
    for name, pop in pops.items():
        print(f"{name:10s} final pop = [{pop[-1, 0]:.6f}, {pop[-1, 1]:.6f}]")


if __name__ == "__main__":
    main()
