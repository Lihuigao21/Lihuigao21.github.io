"""DVR-only convergence checks for the SAC scattering setup."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
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

_bench_spec = importlib.util.spec_from_file_location(
    "run_sac_cmd_benchmark",
    REPO_ROOT / "scripts" / "run_sac_cmd_benchmark.py",
)
if _bench_spec is None or _bench_spec.loader is None:
    raise RuntimeError("Could not load scripts/run_sac_cmd_benchmark.py.")
_bench = importlib.util.module_from_spec(_bench_spec)
sys.modules[_bench_spec.name] = _bench
_bench_spec.loader.exec_module(_bench)
run_dvr_density = _bench.run_dvr_density


def run_case(base_args, *, name, ndvr, xbound, dt):
    args = argparse.Namespace(**vars(base_args))
    args.dvr_ndvr = int(ndvr)
    args.dvr_xbound = float(xbound)
    args.dt = float(dt)
    args.t_final = float(base_args.t_final)
    nstep = int(round(args.t_final / args.dt))
    args.t_final = nstep * args.dt
    tic = time.perf_counter()
    result = run_dvr_density(args)
    wall = time.perf_counter() - tic
    times = np.asarray(result["times"], dtype=float)
    pop = np.asarray(result["pop"], dtype=float)
    p1 = pop[:, 1]
    dp = np.diff(p1)
    return {
        "name": name,
        "args": args,
        "times": times,
        "pop": pop,
        "trace": np.asarray(result["trace"], dtype=float),
        "mean_x": np.asarray(result["mean_x"], dtype=float),
        "mean_p": np.asarray(result["mean_p"], dtype=float),
        "wall_time_sec": wall,
        "metrics": {
            "name": name,
            "ndvr": int(ndvr),
            "xbound": float(xbound),
            "dt": float(dt),
            "dx": float(2.0 * xbound / ndvr),
            "nt": int(times.size),
            "wall_time_sec": wall,
            "final_pop1": float(p1[-1]),
            "max_pop1": float(np.max(p1)),
            "t_at_max_pop1": float(times[int(np.argmax(p1))]),
            "num_negative_steps": int(np.sum(dp < -1.0e-10)),
            "most_negative_step": float(np.min(dp)) if dp.size else 0.0,
            "last_100au_delta_pop1": float(p1[-1] - p1[np.searchsorted(times, times[-1] - 100.0)]),
            "last_200au_delta_pop1": float(p1[-1] - p1[np.searchsorted(times, times[-1] - 200.0)]),
            "trace_min": float(np.min(result["trace"])),
            "trace_max": float(np.max(result["trace"])),
            "mean_x_final": float(result["mean_x"][-1]),
            "mean_p_final": float(result["mean_p"][-1]),
        },
    }


def write_csv(path: Path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def interpolate_to(times_ref, case):
    values = []
    for state in range(case["pop"].shape[1]):
        values.append(np.interp(times_ref, case["times"], case["pop"][:, state]))
    return np.stack(values, axis=1)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    case_specs = [
        ("coarse320_dt2", 320, 18.0, 2.0),
        ("base384_dt2", 384, 18.0, 2.0),
        ("fine448_dt2", 448, 18.0, 2.0),
        ("wide384_dt2", 384, 22.0, 2.0),
        ("base384_dt1", 384, 18.0, 1.0),
    ]
    (outdir / "metadata.json").write_text(
        json.dumps(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "case_specs": case_specs,
                "repo_root": str(REPO_ROOT),
                "schema": "toymodel.sac_dvr_convergence.v1",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = []
    for spec in case_specs:
        print(f"[dvr] start {spec[0]}", flush=True)
        case = run_case(args, name=spec[0], ndvr=spec[1], xbound=spec[2], dt=spec[3])
        cases.append(case)
        print(
            f"[dvr] done {spec[0]} final_P1={case['metrics']['final_pop1']:.8f} "
            f"trace=[{case['metrics']['trace_min']:.8f},{case['metrics']['trace_max']:.8f}]",
            flush=True,
        )

    write_csv(outdir / "dvr_convergence_metrics.csv", [case["metrics"] for case in cases])

    ref = next(case for case in cases if case["name"] == "base384_dt2")
    comparison_rows = []
    for case in cases:
        pop_i = interpolate_to(ref["times"], case)
        diff = pop_i[:, 1] - ref["pop"][:, 1]
        comparison_rows.append(
            {
                "case": case["name"],
                "reference": ref["name"],
                "final_abs_diff_pop1": float(abs(pop_i[-1, 1] - ref["pop"][-1, 1])),
                "rms_diff_pop1": float(np.sqrt(np.mean(diff * diff))),
                "max_abs_diff_pop1": float(np.max(np.abs(diff))),
            }
        )
    write_csv(outdir / "dvr_convergence_vs_base.csv", comparison_rows)

    np.savez_compressed(
        outdir / "dvr_convergence_data.npz",
        **{f"{case['name']}_times": case["times"] for case in cases},
        **{f"{case['name']}_pop": case["pop"] for case in cases},
        **{f"{case['name']}_trace": case["trace"] for case in cases},
        **{f"{case['name']}_mean_x": case["mean_x"] for case in cases},
        **{f"{case['name']}_mean_p": case["mean_p"] for case in cases},
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
    for case in cases:
        axes[0].plot(case["times"], case["pop"][:, 1], lw=1.5, label=case["name"])
        axes[1].plot(case["times"], case["trace"], lw=1.5, label=case["name"])
    axes[0].set_xlabel("time / a.u.")
    axes[0].set_ylabel("DVR total adiabatic P1")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("time / a.u.")
    axes[1].set_ylabel("trace")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "dvr_convergence_p1_trace.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for case in cases:
        ax.plot(case["times"], case["mean_x"], lw=1.5, label=case["name"])
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("time / a.u.")
    ax.set_ylabel("<x>")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "dvr_mean_x.png", dpi=180)
    plt.close(fig)

    print(f"[done] wrote {outdir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--q0", type=float, default=-4.0)
    parser.add_argument("--p0", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sho-omega", type=float, default=0.004)
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--t-final", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--dvr-ndvr", type=int, default=384)
    parser.add_argument("--dvr-xbound", type=float, default=18.0)
    parser.add_argument("--dvr-grid-convention", choices=["left-edge", "cell-centered"], default="left-edge")
    parser.add_argument("--dvr-kinetic-operator", choices=["particle_in_box", "sinc", "fft_periodic"], default="particle_in_box")
    parser.add_argument("--dvr-preparation-kinetic-operator", choices=["same-as-main", "fft"], default="same-as-main")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
