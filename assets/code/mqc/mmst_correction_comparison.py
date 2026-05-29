"""Compare practical fixes for MMST electronic ZPE leakage.

The script combines two small diagnostics:

1. a constant-gap/constant-NAC electronic-precession model with an exact
   reference, used to expose raw mapping-action leakage;
2. a simple initial-force diagnostic for SQC-like electronic windows, used to
   show what the trajectory-adjusted gamma protocol fixes.
"""

from __future__ import annotations

import argparse
import csv
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
        ct = eigvec @ (np.exp(-1j * eigval * t) * coeff)
        out.append(abs(ct[1]) ** 2)
    return np.asarray(out, dtype=float)


def focused_mapping_z(nphase, *, gamma, start_state, seed):
    if gamma < 0.0:
        raise ValueError("focused_mapping_z requires gamma >= 0.")
    rng = np.random.default_rng(seed)
    actions = np.zeros(2, dtype=float)
    actions[int(start_state)] = 1.0
    amplitudes = np.sqrt(actions + 0.5 * gamma)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(nphase, 2))
    return amplitudes[None, :] * np.exp(1j * phase)


def propagate_constant_hamiltonian(z0, times, *, gap, nac, velocity):
    H = np.array(
        [[0.0, -1j * nac * velocity], [1j * nac * velocity, gap]],
        dtype=np.complex128,
    )
    eigval, eigvec = np.linalg.eigh(H)
    coeff = z0 @ eigvec.conjugate()
    out = np.empty((len(times), z0.shape[0], 2), dtype=np.complex128)
    for it, t in enumerate(times):
        out[it] = (coeff * np.exp(-1j * eigval[None, :] * t)) @ eigvec.T
    return out


def raw_action(z_t, gamma):
    return np.abs(z_t) ** 2 - 0.5 * gamma


def clipped_population(pop):
    clipped = np.clip(pop, 0.0, None)
    denom = clipped.sum(axis=-1, keepdims=True)
    return np.divide(
        clipped,
        denom,
        out=np.full_like(clipped, 0.5),
        where=denom > 1.0e-14,
    )


def majority_window_population(pop):
    state = np.argmax(pop, axis=-1)
    out = np.zeros_like(pop)
    out[..., 0] = state == 0
    out[..., 1] = state == 1
    return out


def square_window_population(pop, width):
    upper = (pop[..., 1] >= 1.0 - width) & (pop[..., 0] <= width)
    lower = (pop[..., 0] >= 1.0 - width) & (pop[..., 1] <= width)
    assigned = upper | lower
    out = np.full_like(pop, np.nan)
    out[assigned, 0] = lower[assigned]
    out[assigned, 1] = upper[assigned]
    return out, assigned


def method_timeseries(args, times, exact_upper):
    configs = [
        ("raw_gamma_1", "Raw gamma=1", 1.0, "raw"),
        ("raw_gamma_1_3", "Reduced gamma=1/3", 1.0 / 3.0, "raw"),
        ("raw_gamma_0", "Gamma=0", 0.0, "raw"),
        ("clipped_gamma_1", "Clip+renorm gamma=1", 1.0, "clipped"),
        ("majority_gamma_1_3", "Window/bin gamma=1/3", 1.0 / 3.0, "majority"),
        ("square_gamma_1_3", "SQC square gamma=1/3", 1.0 / 3.0, "square"),
    ]
    results = {}
    rows = []
    for key, label, gamma, estimator in configs:
        z0 = focused_mapping_z(
            args.nphase,
            gamma=gamma,
            start_state=args.start_state,
            seed=args.seed,
        )
        z_t = propagate_constant_hamiltonian(
            z0,
            times,
            gap=args.gap,
            nac=args.nac,
            velocity=args.p0 / args.mass,
        )
        action = raw_action(z_t, gamma)
        assigned_fraction = 1.0
        if estimator == "raw":
            pop = action
        elif estimator == "clipped":
            pop = clipped_population(action)
        elif estimator == "majority":
            pop = majority_window_population(action)
        elif estimator == "square":
            pop, assigned = square_window_population(action, args.square_width)
            assigned_fraction = float(np.mean(assigned))
            upper_count = np.nansum(pop[..., 1], axis=1)
            lower_count = np.nansum(pop[..., 0], axis=1)
            assigned_count = np.sum(assigned, axis=1)
            mean_pop = np.column_stack(
                [
                    np.divide(
                        lower_count,
                        assigned_count,
                        out=np.zeros_like(lower_count),
                        where=assigned_count > 0,
                    ),
                    np.divide(
                        upper_count,
                        assigned_count,
                        out=np.zeros_like(upper_count),
                        where=assigned_count > 0,
                    ),
                ]
            )
            pop = None
        else:
            raise ValueError(estimator)

        if estimator != "square":
            mean_pop = pop.mean(axis=1)
        upper = mean_pop[:, 1]
        outside = (action < -1.0e-12) | (action > 1.0 + 1.0e-12)
        results[key] = {
            "label": label,
            "gamma": gamma,
            "estimator": estimator,
            "mean_pop": mean_pop,
            "upper_min": action[:, :, 1].min(axis=1),
            "upper_max": action[:, :, 1].max(axis=1),
            "global_min": float(action.min()),
            "global_max": float(action.max()),
            "outside_fraction": float(outside.any(axis=-1).mean()),
            "assigned_fraction": assigned_fraction,
        }
        rows.append(
            {
                "method": key,
                "label": label,
                "gamma": gamma,
                "estimator": estimator,
                "final_upper": float(upper[-1]),
                "rms_upper_vs_exact": float(np.sqrt(np.mean((upper - exact_upper) ** 2))),
                "global_min_pop": results[key]["global_min"],
                "global_max_pop": results[key]["global_max"],
                "outside_fraction": results[key]["outside_fraction"],
                "assigned_fraction": assigned_fraction,
            }
        )
    return results, rows


def negative_gamma_bounds(gammas):
    rows = []
    for gamma in gammas:
        rows.append(
            {
                "gamma": float(gamma),
                "lower_bound": -0.5 * gamma,
                "upper_bound": 1.0 + 0.5 * gamma,
                "focused_pure_state_possible": bool(gamma >= 0.0),
                "constraint_allowed": bool(gamma > -0.5),
            }
        )
    return rows


def trajectory_adjusted_force_sample(args):
    rng = np.random.default_rng(args.seed + 60000)
    gamma = args.force_gamma
    offset = 0.5 * gamma
    force_diag = np.asarray(args.force_diag, dtype=float)
    target = force_diag[args.start_state]
    other = 1 - args.start_state

    e = np.empty((args.nforce, 2), dtype=float)
    e[:, args.start_state] = (
        1.0
        + offset
        + rng.uniform(-args.force_width, args.force_width, size=args.nforce)
    )
    e[:, other] = offset + rng.uniform(-args.force_width, args.force_width, size=args.nforce)
    e = np.clip(e, 1.0e-12, None)

    fixed_actions = e - offset
    fixed_force = fixed_actions @ force_diag

    adjusted_gamma = 2.0 * (e - np.eye(2)[args.start_state])
    adjusted_actions = e - 0.5 * adjusted_gamma
    adjusted_force = adjusted_actions @ force_diag

    return {
        "fixed_force_error": fixed_force - target,
        "adjusted_force_error": adjusted_force - target,
        "fixed_actions": fixed_actions,
        "adjusted_gamma": adjusted_gamma,
        "target_force": target,
    }


def write_summary(path, rows, bounds_rows, force_data):
    fields = [
        "section",
        "method",
        "label",
        "gamma",
        "estimator",
        "final_upper",
        "rms_upper_vs_exact",
        "global_min_pop",
        "global_max_pop",
        "outside_fraction",
        "assigned_fraction",
        "lower_bound",
        "upper_bound",
        "focused_pure_state_possible",
        "constraint_allowed",
        "force_error_mean",
        "force_error_std",
        "force_error_max_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"section": "population", **row})
        for row in bounds_rows:
            writer.writerow({"section": "negative_gamma_bounds", **row})
        for key, values in [
            ("fixed_sqc_gamma", force_data["fixed_force_error"]),
            ("trajectory_adjusted_gamma", force_data["adjusted_force_error"]),
        ]:
            writer.writerow(
                {
                    "section": "initial_force",
                    "method": key,
                    "force_error_mean": float(np.mean(values)),
                    "force_error_std": float(np.std(values)),
                    "force_error_max_abs": float(np.max(np.abs(values))),
                }
            )


def write_timeseries(path, times, exact_upper, results):
    fields = ["time", "exact_upper"]
    for key in results:
        fields.append(f"{key}_upper")
        fields.append(f"{key}_upper_min_raw")
        fields.append(f"{key}_upper_max_raw")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(times):
            row = {"time": float(t), "exact_upper": float(exact_upper[i])}
            for key, data in results.items():
                row[f"{key}_upper"] = float(data["mean_pop"][i, 1])
                row[f"{key}_upper_min_raw"] = float(data["upper_min"][i])
                row[f"{key}_upper_max_raw"] = float(data["upper_max"][i])
            writer.writerow(row)


def plot_comparison(path, times, exact_upper, results, rows, bounds_rows, force_data, args):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    ax = axes[0, 0]
    ax.plot(times, exact_upper, color="#111111", lw=2.5, label="Exact electronic")
    colors = {
        "raw_gamma_1": "#9467bd",
        "raw_gamma_1_3": "#2ca02c",
        "raw_gamma_0": "#1f77b4",
        "clipped_gamma_1": "#d62728",
        "majority_gamma_1_3": "#ff7f0e",
        "square_gamma_1_3": "#8c564b",
    }
    styles = {
        "raw_gamma_1": "-.",
        "raw_gamma_1_3": "-",
        "raw_gamma_0": "--",
        "clipped_gamma_1": ":",
        "majority_gamma_1_3": "-",
        "square_gamma_1_3": "--",
    }
    for key, data in results.items():
        ax.plot(
            times,
            data["mean_pop"][:, 1],
            color=colors[key],
            ls=styles[key],
            lw=1.7,
            label=data["label"],
        )
    ax.set_title("Population estimators on electronic precession")
    ax.set_ylabel("Mean upper population")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncols=2)

    ax = axes[0, 1]
    short_labels = {
        "raw_gamma_1": "raw\n1",
        "raw_gamma_1_3": "raw\n1/3",
        "raw_gamma_0": "raw\n0",
        "clipped_gamma_1": "clip\n1",
        "majority_gamma_1_3": "bin\n1/3",
        "square_gamma_1_3": "SQC\n1/3",
    }
    x = np.arange(len(rows), dtype=float)
    rms = np.asarray([row["rms_upper_vs_exact"] for row in rows], dtype=float)
    outside = np.asarray([row["outside_fraction"] for row in rows], dtype=float)
    unassigned = 1.0 - np.asarray([row["assigned_fraction"] for row in rows], dtype=float)
    ax.bar(x, rms, width=0.58, color="#4c78a8", alpha=0.78, label="RMS upper error")
    ax.set_title("Estimator error and assignment diagnostics")
    ax.set_ylabel("RMS upper error")
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[row["method"]] for row in rows], fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(x, outside, "o-", color="#d62728", lw=1.5, ms=4.5, label="raw outside fraction")
    ax2.plot(x, unassigned, "s--", color="#8c564b", lw=1.3, ms=4.0, label="unassigned fraction")
    ax2.set_ylabel("Fraction")
    ax2.set_ylim(-0.03, 1.03)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    gamma = np.asarray([row["gamma"] for row in bounds_rows], dtype=float)
    lower = np.asarray([row["lower_bound"] for row in bounds_rows], dtype=float)
    upper = np.asarray([row["upper_bound"] for row in bounds_rows], dtype=float)
    ax.plot(gamma, lower, "o-", color="#d62728", label="lower bound")
    ax.plot(gamma, upper, "s-", color="#1f77b4", label="upper bound")
    ax.axhline(0.0, color="#666666", lw=0.9)
    ax.axhline(1.0, color="#666666", lw=0.9)
    ax.axvline(0.0, color="#aaaaaa", lw=0.9)
    ax.set_title("Negative gamma shrinks the raw-action interval")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Allowed raw population interval")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    bins = np.linspace(
        min(
            force_data["fixed_force_error"].min(),
            force_data["adjusted_force_error"].min(),
        ),
        max(
            force_data["fixed_force_error"].max(),
            force_data["adjusted_force_error"].max(),
        ),
        42,
    )
    ax.hist(
        force_data["fixed_force_error"],
        bins=bins,
        density=True,
        alpha=0.58,
        color="#9467bd",
        label="fixed gamma SQC-like window",
    )
    ax.hist(
        force_data["adjusted_force_error"],
        bins=bins,
        density=True,
        alpha=0.72,
        color="#2ca02c",
        label="trajectory-adjusted gamma",
    )
    ax.axvline(0.0, color="#111111", lw=1.4)
    ax.set_title("Initial nuclear-force error")
    ax.set_xlabel("Force error vs target initial state")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "MMST ZPE/gamma correction comparison "
        f"(gap={args.gap:g}, NAC={args.nac:g}, velocity={args.p0 / args.mass:g})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-time", type=float, default=2500.0)
    parser.add_argument("--dt", type=float, default=2.5)
    parser.add_argument("--gap", type=float, default=0.002)
    parser.add_argument("--nac", type=float, default=0.05)
    parser.add_argument("--p0", type=float, default=200.0)
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--nphase", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--square-width", type=float, default=1.0 / 3.0)
    parser.add_argument("--negative-gammas", default="-0.4,-0.2,0,0.3333333333,0.5,1")
    parser.add_argument("--nforce", type=int, default=20000)
    parser.add_argument("--force-gamma", type=float, default=1.0 / 3.0)
    parser.add_argument("--force-width", type=float, default=0.25)
    parser.add_argument("--force-diag", nargs=2, type=float, default=[-1.0, 2.0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "mmst_correction_comparison",
    )
    args = parser.parse_args()
    args.nstep = int(round(args.total_time / args.dt))
    args.total_time = args.nstep * args.dt
    args.negative_gammas = [
        float(item.strip())
        for item in str(args.negative_gammas).split(",")
        if item.strip()
    ]
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
    results, rows = method_timeseries(args, times, exact_upper)
    bounds_rows = negative_gamma_bounds(args.negative_gammas)
    force_data = trajectory_adjusted_force_sample(args)

    np.savez(
        args.output_dir / "mmst_correction_comparison.npz",
        time=times,
        exact_upper=exact_upper,
        **{f"{key}_mean_pop": data["mean_pop"] for key, data in results.items()},
        fixed_force_error=force_data["fixed_force_error"],
        adjusted_force_error=force_data["adjusted_force_error"],
    )
    write_summary(args.output_dir / "summary.csv", rows, bounds_rows, force_data)
    write_timeseries(args.output_dir / "timeseries.csv", times, exact_upper, results)
    plot_comparison(
        args.output_dir / "mmst_correction_comparison.png",
        times,
        exact_upper,
        results,
        rows,
        bounds_rows,
        force_data,
        args,
    )

    print(f"wrote {args.output_dir}")
    for row in rows:
        print(
            f"{row['label']}: rms={row['rms_upper_vs_exact']:.6e}, "
            f"min={row['global_min_pop']:.6f}, max={row['global_max_pop']:.6f}, "
            f"outside={row['outside_fraction']:.3f}, "
            f"assigned={row['assigned_fraction']:.3f}"
        )
    print(
        "force error std: "
        f"fixed={np.std(force_data['fixed_force_error']):.6f}, "
        f"adjusted={np.std(force_data['adjusted_force_error']):.6f}"
    )


if __name__ == "__main__":
    main()
