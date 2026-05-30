"""Compare Spin-MInt with MMST, FSSH, Ehrenfest, and DVR on Tully SAC.

The benchmark uses Tully's simple avoided crossing with an incoming Gaussian
wavepacket reference from DVR.  Spin-MInt is propagated in the diabatic basis,
but its mapped density is projected back to the instantaneous adiabatic basis
for comparison with FSSH, Ehrenfest, MMST, and DVR populations.
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
from toymodel.methods import Ehrenfest, FSSH, MMST, SpinMInt  # noqa: E402
from toymodel.model import TullySimpleAvoidedCrossing  # noqa: E402
from toymodel.utils.su_n import focused_mapping_variables  # noqa: E402


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
    return _with_initial_population(method.result["pop"], args.start_state)


def run_fssh(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = TullySimpleAvoidedCrossing(representation="adiabatic")
    method = FSSH(
        model=model,
        distribution=_fixed_distribution(args, args.fssh_ntraj),
        Nstep=args.nstep,
        dt=args.dt,
        start_state=args.start_state,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        verbose=False,
    )
    method.run()
    pops = np.asarray([traj["pop"] for traj in method.result], dtype=float)
    return _with_initial_population(np.mean(pops, axis=0), args.start_state)


def run_mmst(args):
    pops = []
    for itraj in range(args.mmst_ntraj):
        model = TullySimpleAvoidedCrossing(representation="adiabatic")
        method = MMST(
            model=model,
            distribution=_fixed_distribution(args, 1),
            Nstep=args.nstep,
            dt=args.dt,
            start_state=args.start_state,
            is_record=True,
            legacy_result=False,
            record_level="minimal",
            gamma=args.mmst_gamma,
            electronic_substeps=args.mmst_substeps,
            population_estimator=args.mmst_population_estimator,
            mapping_seed=args.seed + 10000 + itraj,
            verbose=False,
        )
        method.run()
        pops.append(np.asarray(method.result["pop"], dtype=float))
    return _with_initial_population(np.mean(pops, axis=0), args.start_state)


def _adiabatic_coefficients(q):
    ad_model = TullySimpleAvoidedCrossing(representation="adiabatic")
    return np.asarray(ad_model.evaluate(q, need_force=False, need_nac=False).wavefun[0])


def _adiabatic_population_from_diabatic_density(q, rho_diabatic):
    coeff = _adiabatic_coefficients(q)
    rho_ad = coeff.T.conjugate() @ np.asarray(rho_diabatic) @ coeff
    return np.real(np.diag(rho_ad))


def _focused_mapping_in_diabatic_basis(args, rng):
    gamma, _ = SpinMInt.sw_parameters(2, "W")
    x_ad, p_ad = focused_mapping_variables(
        2,
        args.start_state,
        gamma,
        rng,
        phase=args.spin_initial_phase,
    )
    z_ad = (x_ad + 1j * p_ad) / np.sqrt(2.0)
    coeff0 = _adiabatic_coefficients(args.q0)
    z_diabatic = coeff0 @ z_ad
    return np.sqrt(2.0) * np.real(z_diabatic), np.sqrt(2.0) * np.imag(z_diabatic)


def run_spinmint(args):
    rng = np.random.default_rng(args.seed + 20000)
    model = TullySimpleAvoidedCrossing(representation="diabatic")
    pops = []
    for _ in range(args.spin_ntraj):
        mapping_x, mapping_p = _focused_mapping_in_diabatic_basis(args, rng)
        method = SpinMInt(
            model=model,
            distribution=_fixed_distribution(args, 1),
            Nstep=args.nstep,
            dt=args.dt,
            start_state=args.start_state,
            mapping_x=mapping_x,
            mapping_p=mapping_p,
            is_record=False,
            verbose=False,
        )
        method.initialize()
        traj_pop = np.empty((args.nstep + 1, 2), dtype=float)
        traj_pop[0] = _adiabatic_population_from_diabatic_density(method.q, method.rho)
        for istep in range(1, args.nstep + 1):
            method.step()
            traj_pop[istep] = _adiabatic_population_from_diabatic_density(
                method.q,
                method.rho,
            )
        pops.append(traj_pop)
    return np.mean(np.asarray(pops, dtype=float), axis=0)


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
        "dvr": "#111111",
        "spinmint": "#1f77b4",
        "mmst": "#9467bd",
        "fssh": "#d62728",
        "ehrenfest": "#2ca02c",
    }
    labels = {
        "dvr": "Exact DVR",
        "spinmint": f"Spin-MInt W ({args.spin_ntraj} phases)",
        "mmst": f"MMST (gamma={args.mmst_gamma:g}, {args.mmst_ntraj} phases)",
        "fssh": f"FSSH ({args.fssh_ntraj} traj)",
        "ehrenfest": "Ehrenfest",
    }
    linestyles = {
        "dvr": "-",
        "spinmint": "-",
        "mmst": "-.",
        "fssh": "-",
        "ehrenfest": "--",
    }

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True)
    order = ["dvr", "spinmint", "mmst", "fssh", "ehrenfest"]
    for name in order:
        kwargs = {
            "label": labels[name],
            "color": colors[name],
            "lw": 2.5 if name == "dvr" else 1.9,
            "ls": linestyles[name],
        }
        if name == "fssh":
            kwargs["drawstyle"] = "steps-post"
            kwargs["alpha"] = 0.82
        axes[0].plot(times, pops[name][:, 1], **kwargs)
        axes[1].plot(times, pops[name][:, 0], **kwargs)

    axes[0].set_ylabel("Upper adiabatic population")
    axes[1].set_ylabel("Lower adiabatic population")
    axes[1].set_xlabel("Time (a.u.)")
    axes[0].set_title(
        "Tully simple avoided crossing "
        f"(q0={args.q0:g}, p0={args.p0:g}, start={args.start_state})"
    )
    for ax in axes:
        ax.set_ylim(-0.08, 1.08)
        ax.grid(alpha=0.22, lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, ncols=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fssh-ntraj", type=int, default=512)
    parser.add_argument("--mmst-ntraj", type=int, default=64)
    parser.add_argument("--spin-ntraj", type=int, default=128)
    parser.add_argument("--spin-initial-phase", choices=["random", "zero"], default="random")
    parser.add_argument("--mmst-gamma", type=float, default=1.0)
    parser.add_argument("--mmst-substeps", type=int, default=10)
    parser.add_argument(
        "--mmst-population-estimator",
        choices=["action", "raw", "clipped"],
        default="action",
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--dvr-ndvr", type=int, default=256)
    parser.add_argument("--dvr-xbound", type=float, default=25.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "spin_mapping_sac_comparison",
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
        "ehrenfest": run_ehrenfest(args),
        "fssh": run_fssh(args),
        "mmst": run_mmst(args),
        "spinmint": run_spinmint(args),
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
    _write_population_csv(args.output_dir / "spin_mapping_sac_comparison.csv", times, pops)
    _write_summary_csv(args.output_dir / "spin_mapping_sac_comparison_summary.csv", pops)
    _plot(args.output_dir / "spin_mapping_sac_comparison.png", times, pops, args)

    print(f"wrote {args.output_dir}")
    for name, pop in pops.items():
        print(f"{name:10s} final pop = [{pop[-1, 0]:.6f}, {pop[-1, 1]:.6f}]")


if __name__ == "__main__":
    main()
