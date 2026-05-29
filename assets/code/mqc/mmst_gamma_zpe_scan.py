"""MMST gamma scan on a minimal zero-point-energy leakage diagnostic.

The model is a two-state adiabatic Hamiltonian with a constant gap and a
constant derivative coupling.  The nuclei feel no force, so the electronic
problem has a simple exact Rabi solution.  This isolates the MMST mapping
variables: the phase-averaged population can look reasonable while individual
mapping actions leak outside the physical population simplex.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
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

from toymodel.methods import MMST  # noqa: E402
from toymodel.model.modelbase import ModelSnapshot  # noqa: E402


class ConstantGapNACModel:
    """Two-state adiabatic model with fixed gap and fixed derivative coupling."""

    def __init__(self, gap=0.002, nac=0.05):
        self.nstate = 2
        self.gap = float(gap)
        self.nac = float(nac)

    def evaluate(
        self,
        x,
        *,
        wavefunc_for_nac=None,
        energies_for_nac=None,
        need_force=True,
        need_nac=True,
        need_wavefun=True,
    ):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        npoint = len(x_arr)
        energies = np.tile(np.array([0.0, self.gap], dtype=float), (npoint, 1))
        wavefun = None
        if need_wavefun:
            wavefun = np.tile(np.eye(2, dtype=float), (npoint, 1, 1))

        force = None
        force_diag = None
        if need_force:
            force = np.zeros((npoint, 2, 2, 1), dtype=float)
            force_diag = np.zeros((npoint, 2, 1), dtype=float)

        nac = None
        if need_nac:
            nac = np.zeros((npoint, 2, 2, 1), dtype=float)
            nac[:, 0, 1, 0] = self.nac
            nac[:, 1, 0, 0] = -self.nac

        return ModelSnapshot(
            x=x_arr,
            energies=energies,
            wavefun=wavefun,
            force=force,
            force_diag=force_diag,
            nac=nac,
        )


@dataclass
class FixedDistribution:
    q: float
    p: float
    mass: float = 2000.0
    T: float = 300.0
    Ntraj: int = 1

    def sample(self):
        return float(self.q), float(self.p)


def parse_gamma_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def exact_upper_population(times, *, gap, nac, velocity, start_state):
    H = np.array(
        [[0.0, -1j * nac * velocity], [1j * nac * velocity, gap]],
        dtype=np.complex128,
    )
    eigval, eigvec = np.linalg.eigh(H)
    c0 = np.zeros(2, dtype=np.complex128)
    c0[int(start_state)] = 1.0
    coeff = eigvec.conjugate().T @ c0
    out = []
    for t in times:
        phase = np.exp(-1j * eigval * t)
        ct = eigvec @ (phase * coeff)
        out.append(abs(ct[1]) ** 2)
    return np.asarray(out, dtype=float)


def run_mmst_phase_ensemble(args, gamma):
    traj_pop = []
    for iphase in range(args.nphase):
        model = ConstantGapNACModel(gap=args.gap, nac=args.nac)
        method = MMST(
            model=model,
            distribution=FixedDistribution(
                q=args.q0,
                p=args.p0,
                mass=args.mass,
                T=args.temperature,
            ),
            Nstep=args.nstep,
            dt=args.dt,
            start_state=args.start_state,
            is_record=True,
            legacy_result=False,
            record_level="minimal",
            gamma=gamma,
            electronic_substeps=args.substeps,
            population_estimator="action",
            mapping_seed=args.seed + iphase,
            verbose=False,
        )
        method.run()
        initial = np.zeros((1, 2), dtype=float)
        initial[0, args.start_state] = 1.0
        traj_pop.append(np.vstack([initial, np.asarray(method.result["pop"], dtype=float)]))
    return np.asarray(traj_pop, dtype=float)


def leakage_metrics(traj_pop, exact_upper):
    flat = traj_pop.reshape(-1, 2)
    clipped = np.clip(flat, 0.0, None)
    denom = clipped.sum(axis=1, keepdims=True)
    clipped = np.divide(
        clipped,
        denom,
        out=np.full_like(clipped, 0.5),
        where=denom > 1.0e-14,
    )
    mean_pop = traj_pop.mean(axis=0)
    upper_diff = mean_pop[:, 1] - exact_upper
    outside = (flat < -1.0e-12) | (flat > 1.0 + 1.0e-12)
    return {
        "mean_pop": mean_pop,
        "min_any": traj_pop.min(axis=(0, 2)),
        "max_any": traj_pop.max(axis=(0, 2)),
        "upper_min": traj_pop[:, :, 1].min(axis=0),
        "upper_max": traj_pop[:, :, 1].max(axis=0),
        "global_min": float(flat.min()),
        "global_max": float(flat.max()),
        "outside_fraction": float(outside.any(axis=1).mean()),
        "clip_l1": float(np.abs(clipped - flat).sum(axis=1).mean()),
        "rms_upper_vs_exact": float(np.sqrt(np.mean(upper_diff**2))),
        "final_upper_mean": float(mean_pop[-1, 1]),
    }


def _write_summary(path, rows):
    fields = [
        "gamma",
        "final_upper_mean",
        "rms_upper_vs_exact",
        "global_min_pop",
        "theory_min_pop",
        "global_max_pop",
        "theory_max_pop",
        "outside_fraction",
        "clip_l1_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_timeseries(path, times, exact_upper, results):
    fields = ["time", "exact_upper"]
    for gamma in results:
        label = f"gamma_{gamma:g}"
        fields.extend(
            [
                f"{label}_upper_mean",
                f"{label}_upper_min",
                f"{label}_upper_max",
                f"{label}_min_any_pop",
                f"{label}_max_any_pop",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(times):
            row = {"time": float(t), "exact_upper": float(exact_upper[i])}
            for gamma, data in results.items():
                label = f"gamma_{gamma:g}"
                row[f"{label}_upper_mean"] = float(data["mean_pop"][i, 1])
                row[f"{label}_upper_min"] = float(data["upper_min"][i])
                row[f"{label}_upper_max"] = float(data["upper_max"][i])
                row[f"{label}_min_any_pop"] = float(data["min_any"][i])
                row[f"{label}_max_any_pop"] = float(data["max_any"][i])
            writer.writerow(row)


def _plot(path, times, exact_upper, results, rows, args):
    colors = {
        0.0: "#1f77b4",
        1.0 / 3.0: "#2ca02c",
        0.5: "#ff7f0e",
        1.0: "#9467bd",
    }
    fallback = ["#17becf", "#bcbd22", "#8c564b", "#e377c2"]

    def color_for(gamma):
        for key, value in colors.items():
            if abs(gamma - key) < 1.0e-8:
                return value
        return fallback[hash(round(gamma, 8)) % len(fallback)]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.2, 9.4),
        sharex=False,
        gridspec_kw={"height_ratios": [1.05, 1.05, 0.85]},
    )

    ax = axes[0]
    ax.plot(times, exact_upper, color="#111111", lw=2.6, label="Exact electronic")
    for gamma, data in results.items():
        ax.plot(
            times,
            data["mean_pop"][:, 1],
            color=color_for(gamma),
            lw=1.7,
            label=f"MMST gamma={gamma:g}",
        )
    ax.set_ylabel("Mean upper population")
    ax.set_title(
        "MMST gamma scan: constant-gap / constant-NAC electronic precession"
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncols=2)

    ax = axes[1]
    envelope_gammas = [g for g in results if abs(g - 1.0 / 3.0) < 1.0e-8 or abs(g - 1.0) < 1.0e-8]
    if not envelope_gammas:
        envelope_gammas = list(results)[-2:]
    ax.axhspan(-0.6, 0.0, color="#f4b6b6", alpha=0.22, lw=0)
    ax.axhspan(1.0, 1.6, color="#f4b6b6", alpha=0.22, lw=0)
    for gamma in envelope_gammas:
        data = results[gamma]
        color = color_for(gamma)
        ax.fill_between(
            times,
            data["upper_min"],
            data["upper_max"],
            color=color,
            alpha=0.18,
            label=f"phase envelope gamma={gamma:g}",
        )
        ax.plot(times, data["mean_pop"][:, 1], color=color, lw=1.4)
    ax.plot(times, exact_upper, color="#111111", lw=2.1, label="Exact electronic")
    ax.axhline(0.0, color="#666666", lw=0.9)
    ax.axhline(1.0, color="#666666", lw=0.9)
    ax.set_ylim(-0.58, 1.58)
    ax.set_ylabel("Raw upper action")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="upper right")

    ax = axes[2]
    gammas = np.array([row["gamma"] for row in rows], dtype=float)
    global_min = np.array([row["global_min_pop"] for row in rows], dtype=float)
    global_max = np.array([row["global_max_pop"] for row in rows], dtype=float)
    outside = np.array([row["outside_fraction"] for row in rows], dtype=float)
    ax.plot(gammas, global_min, "o-", color="#d62728", lw=1.8, label="observed min pop")
    ax.plot(gammas, -0.5 * gammas, "--", color="#d62728", lw=1.2, label="-gamma/2 bound")
    ax.plot(gammas, global_max, "s-", color="#1f77b4", lw=1.8, label="observed max pop")
    ax.plot(gammas, 1.0 + 0.5 * gammas, "--", color="#1f77b4", lw=1.2, label="1+gamma/2 bound")
    ax2 = ax.twinx()
    ax2.plot(gammas, outside, "^-", color="#4d4d4d", lw=1.6, label="outside fraction")
    ax.set_xlabel("MMST gamma")
    ax.set_ylabel("Population extrema")
    ax2.set_ylabel("Fraction outside [0, 1]")
    ax.grid(alpha=0.22)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncols=2, loc="best")

    fig.text(
        0.02,
        0.01,
        f"gap={args.gap:g}, NAC={args.nac:g}, velocity={args.p0 / args.mass:g}, "
        f"nphase={args.nphase}, start={args.start_state}",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-time", type=float, default=2500.0)
    parser.add_argument("--dt", type=float, default=2.5)
    parser.add_argument("--q0", type=float, default=0.0)
    parser.add_argument("--p0", type=float, default=200.0)
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--gap", type=float, default=0.002)
    parser.add_argument("--nac", type=float, default=0.05)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--gammas", default="0,0.3333333333,0.5,1.0")
    parser.add_argument("--nphase", type=int, default=256)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "mmst_gamma_zpe_scan",
    )
    args = parser.parse_args()
    args.gammas = parse_gamma_list(args.gammas)
    args.nstep = int(round(args.total_time / args.dt))
    args.total_time = args.nstep * args.dt
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = np.arange(args.nstep + 1, dtype=float) * args.dt
    exact_upper = exact_upper_population(
        times,
        gap=args.gap,
        nac=args.nac,
        velocity=args.p0 / args.mass,
        start_state=args.start_state,
    )

    results = {}
    rows = []
    for gamma in args.gammas:
        traj_pop = run_mmst_phase_ensemble(args, gamma)
        data = leakage_metrics(traj_pop, exact_upper)
        results[gamma] = data
        rows.append(
            {
                "gamma": float(gamma),
                "final_upper_mean": data["final_upper_mean"],
                "rms_upper_vs_exact": data["rms_upper_vs_exact"],
                "global_min_pop": data["global_min"],
                "theory_min_pop": -0.5 * gamma,
                "global_max_pop": data["global_max"],
                "theory_max_pop": 1.0 + 0.5 * gamma,
                "outside_fraction": data["outside_fraction"],
                "clip_l1_mean": data["clip_l1"],
            }
        )

    np.savez(
        args.output_dir / "mmst_gamma_zpe_scan.npz",
        time=times,
        exact_upper=exact_upper,
        gamma=np.asarray(args.gammas, dtype=float),
        **{
            f"gamma_{gamma:g}_mean_pop": data["mean_pop"]
            for gamma, data in results.items()
        },
        **{
            f"gamma_{gamma:g}_upper_min": data["upper_min"]
            for gamma, data in results.items()
        },
        **{
            f"gamma_{gamma:g}_upper_max": data["upper_max"]
            for gamma, data in results.items()
        },
    )
    _write_summary(args.output_dir / "summary.csv", rows)
    _write_timeseries(args.output_dir / "timeseries.csv", times, exact_upper, results)
    _plot(args.output_dir / "mmst_gamma_zpe_scan.png", times, exact_upper, results, rows, args)

    print(f"wrote {args.output_dir}")
    for row in rows:
        print(
            f"gamma={row['gamma']:g}: final_upper={row['final_upper_mean']:.6f}, "
            f"rms={row['rms_upper_vs_exact']:.6e}, "
            f"min={row['global_min_pop']:.6f}, max={row['global_max_pop']:.6f}, "
            f"outside={row['outside_fraction']:.3f}, "
            f"clip_l1={row['clip_l1_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
