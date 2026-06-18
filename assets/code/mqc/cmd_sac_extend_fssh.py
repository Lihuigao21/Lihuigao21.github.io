"""Extend SAC FSSH convergence from a saved CMD PMF surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _ensure_repo_imports() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        src = candidate / "src"
        if (src / "toymodel").exists():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            repo_str = str(candidate)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            return candidate
    raise RuntimeError("Could not locate repository root containing src/toymodel.")


REPO_ROOT = _ensure_repo_imports()

from scripts import run_sac_cmd_benchmark as bench  # noqa: E402
from toymodel.methods import CMDAdiabaticPMFSurfaces  # noqa: E402


def load_cmd_surface(path: Path):
    data = np.load(path)
    return CMDAdiabaticPMFSurfaces(
        model=bench.make_model("adiabatic"),
        qgrid=np.asarray(data["qgrid"], dtype=float),
        pmf=np.asarray(data["pmf"], dtype=float),
        mean_force=np.asarray(data["mean_force"], dtype=float),
        force_stderr=np.asarray(data["force_stderr"], dtype=float),
        acceptance_rate=np.asarray(data["acceptance_rate"], dtype=float),
        npaths=np.asarray(data["npaths"], dtype=int),
        temperature=float(data["temperature"]),
        beta=float(data["beta"]),
        nbeads=int(data["nbeads"]),
        ndim=1,
        mass=np.asarray(data["mass"], dtype=float),
        reference_q=float(data["reference_q"]),
    )


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd_surface = load_cmd_surface(Path(args.cmd_surface_npz))
    base = Path(args.base_dynamics_dir)
    dvr_npz = np.load(base / "populations.npz")
    times = np.asarray(dvr_npz["times"], dtype=float)
    dvr_pop = np.asarray(dvr_npz["DVR"], dtype=float)

    run_args = SimpleNamespace(
        q0=float(args.q0),
        p0=float(args.p0),
        temperature=float(args.temperature),
        sho_omega=float(args.sho_omega),
        mass=float(args.mass),
        seed=int(args.seed),
        fssh_ntraj_levels=[int(x) for x in args.fssh_ntraj_levels],
        t_final=float(args.t_final),
        dt=float(args.dt),
        start_state=int(args.start_state),
    )
    max_ntraj = max(run_args.fssh_ntraj_levels)
    conditions = bench.sample_initial_conditions(run_args, max_ntraj)
    np.savez_compressed(
        outdir / "initial_conditions.npz",
        q=np.asarray([c.q for c in conditions], dtype=float),
        p=np.asarray([c.p for c in conditions], dtype=float),
        sample_id=np.asarray([c.sample_id for c in conditions], dtype=int),
    )

    metadata = {
        "base_dynamics_dir": str(base),
        "cmd_surface_npz": str(args.cmd_surface_npz),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "repo_root": str(REPO_ROOT),
        "schema": "toymodel.sac_fssh_extension.v1",
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[fssh-extension] start", flush=True)
    stacks, rows, diff_rows = bench.run_fssh_convergence(
        run_args,
        outdir,
        conditions,
        cmd_surface,
    )
    print("[fssh-extension] done", flush=True)

    method_results = {}
    for method_name, stack in stacks.items():
        mean, stderr = bench.prefix_mean_stderr(stack, max_ntraj)
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
    summary_rows = bench.compare_to_dvr(times, dvr_pop, method_results)
    bench.write_csv(outdir / "summary.csv", summary_rows)
    np.savez_compressed(
        outdir / "fssh_populations.npz",
        times=times,
        DVR=dvr_pop,
        FSSH=method_results["FSSH"]["mean"],
        FSSH_stderr=method_results["FSSH"]["stderr"],
        CMD_FSSH=method_results["CMD_FSSH"]["mean"],
        CMD_FSSH_stderr=method_results["CMD_FSSH"]["stderr"],
    )
    np.savez_compressed(
        outdir / "fssh_stacks.npz",
        FSSH=stacks["FSSH"],
        CMD_FSSH=stacks["CMD_FSSH"],
    )
    bench.plot_population(
        outdir / "fssh_population_comparison.png",
        times,
        dvr_pop,
        method_results,
    )
    print(f"[done] wrote {outdir}", flush=True)
    return {
        "outdir": str(outdir),
        "summary": summary_rows,
        "fssh_convergence": diff_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--base-dynamics-dir", type=Path, required=True)
    parser.add_argument("--cmd-surface-npz", type=Path, required=True)
    parser.add_argument("--fssh-ntraj-levels", nargs="+", type=int, required=True)
    parser.add_argument("--q0", type=float, default=-4.0)
    parser.add_argument("--p0", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sho-omega", type=float, default=0.004)
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--t-final", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
