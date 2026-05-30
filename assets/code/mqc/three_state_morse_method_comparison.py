"""Compare Spin-MInt, FSSH, Ehrenfest, and DVR on three-state Morse Model A.

The initial nuclear state is the Gaussian wavepacket corresponding to the
Wigner distribution used in the Spin-MInt three-state benchmark.  The exact
reference is finite-grid DVR wavepacket propagation.  The MQC methods use
Wigner-sampled classical initial conditions from the same Gaussian.
"""

from __future__ import annotations

import argparse
import csv
import random
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

from toymodel.DVRmethods import DVRWaveDynamics  # noqa: E402
from toymodel.methods import Ehrenfest, FSSH, SpinMInt  # noqa: E402
from toymodel.model import ThreeStateMorse  # noqa: E402
from toymodel.utils.su_n import focused_mapping_variables  # noqa: E402


@dataclass
class FixedDistribution:
    q: float
    p: float
    mass: float
    T: float = 300.0
    Ntraj: int = 1

    def sample(self):
        return float(self.q), float(self.p)


class MorseWignerDistribution:
    def __init__(self, *, ntraj, mass, omega, R0, P0, seed, T=300.0):
        self.Ntraj = int(ntraj)
        self.mass = float(mass)
        self.omega = float(omega)
        self.R0 = float(R0)
        self.P0 = float(P0)
        self.T = float(T)
        self.rng = np.random.default_rng(seed)
        self.q_sigma = np.sqrt(1.0 / (2.0 * self.mass * self.omega))
        self.p_sigma = np.sqrt(self.mass * self.omega / 2.0)

    def sample(self):
        q = self.rng.normal(self.R0, self.q_sigma)
        p = self.rng.normal(self.P0, self.p_sigma)
        return float(q), float(p)


def make_model(representation):
    return ThreeStateMorse(representation=representation)


def _with_initial_population(pop, start_state, nstate=3):
    initial = np.zeros((1, nstate), dtype=float)
    initial[0, start_state] = 1.0
    return np.vstack([initial, np.asarray(pop, dtype=float)])


def _wigner_sample(args, rng):
    q_sigma = np.sqrt(1.0 / (2.0 * args.mass * args.omega))
    p_sigma = np.sqrt(args.mass * args.omega / 2.0)
    q = rng.normal(args.R0, q_sigma)
    p = rng.normal(args.P0, p_sigma)
    return float(q), float(p)


def _adiabatic_coefficients(q):
    model = make_model("adiabatic")
    return np.asarray(model.evaluate(q, need_force=False, need_nac=False).wavefun[0])


def _adiabatic_population_from_diabatic_density(q, rho_diabatic):
    coeff = _adiabatic_coefficients(q)
    rho_ad = coeff.T.conjugate() @ np.asarray(rho_diabatic) @ coeff
    return np.real(np.diag(rho_ad))


def _focused_mapping_in_diabatic_basis(q, args, rng):
    gamma, _ = SpinMInt.sw_parameters(3, "W")
    x_ad, p_ad = focused_mapping_variables(
        3,
        args.start_state,
        gamma,
        rng,
        phase=args.spin_initial_phase,
    )
    z_ad = (x_ad + 1j * p_ad) / np.sqrt(2.0)
    coeff = _adiabatic_coefficients(q)
    z_diabatic = coeff @ z_ad
    return np.sqrt(2.0) * np.real(z_diabatic), np.sqrt(2.0) * np.imag(z_diabatic)


def run_ehrenfest(args):
    rng = np.random.default_rng(args.seed + 1000)
    pop_sum = np.zeros((args.nstep + 1, 3), dtype=float)
    for _ in range(args.ehrenfest_ntraj):
        q, p = _wigner_sample(args, rng)
        method = Ehrenfest(
            model=make_model("adiabatic"),
            distribution=FixedDistribution(q=q, p=p, mass=args.mass),
            Nstep=args.nstep,
            dt=args.dt,
            start_state=args.start_state,
            is_record=True,
            legacy_result=False,
            record_level="minimal",
            verbose=False,
        )
        method.run()
        pop_sum += _with_initial_population(method.result["pop"], args.start_state, 3)
    return pop_sum / args.ehrenfest_ntraj


def run_fssh(args):
    np.random.seed(args.seed + 2000)
    random.seed(args.seed + 2000)
    method = FSSH(
        model=make_model("adiabatic"),
        distribution=MorseWignerDistribution(
            ntraj=args.fssh_ntraj,
            mass=args.mass,
            omega=args.omega,
            R0=args.R0,
            P0=args.P0,
            seed=args.seed + 2000,
        ),
        Nstep=args.nstep,
        dt=args.dt,
        start_state=args.start_state,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        verbose=False,
    )
    method.run()
    pops = np.asarray(
        [_with_initial_population(traj["pop"], args.start_state, 3) for traj in method.result],
        dtype=float,
    )
    return np.mean(pops, axis=0)


def run_spinmint(args):
    rng = np.random.default_rng(args.seed + 3000)
    model = make_model("diabatic")
    pop_sum = np.zeros((args.nstep + 1, 3), dtype=float)
    for _ in range(args.spin_ntraj):
        q, p = _wigner_sample(args, rng)
        mapping_x, mapping_p = _focused_mapping_in_diabatic_basis(q, args, rng)
        method = SpinMInt(
            model=model,
            distribution=FixedDistribution(q=q, p=p, mass=args.mass),
            Nstep=args.nstep,
            dt=args.dt,
            start_state=args.start_state,
            mapping_x=mapping_x,
            mapping_p=mapping_p,
            is_record=False,
            verbose=False,
        )
        method.initialize()
        traj_pop = np.empty((args.nstep + 1, 3), dtype=float)
        traj_pop[0] = _adiabatic_population_from_diabatic_density(method.q, method.rho)
        for istep in range(1, args.nstep + 1):
            method.step()
            traj_pop[istep] = _adiabatic_population_from_diabatic_density(
                method.q,
                method.rho,
            )
        pop_sum += traj_pop
    return pop_sum / args.spin_ntraj


def run_dvr(args):
    solver = DVRWaveDynamics(
        model=make_model("diabatic"),
        total_time=args.total_time,
        dt=args.dt,
        ndvr=args.dvr_ndvr,
        xbound=(args.dvr_xmin, args.dvr_xmax),
        startstate=args.start_state,
        sigma=np.sqrt(1.0 / (2.0 * args.mass * args.omega)),
        x0=args.R0,
        p0=args.P0,
        m=args.mass,
        is_plot=False,
        basis_ordering="grid-major",
        grid_convention="cell-centered",
        kinetic_operator="sinc",
    )
    solver.initialize()
    result = solver.compute()
    return np.asarray(result["pop"], dtype=float).sum(axis=1)


def _write_population_csv(path, times, pops):
    fields = ["time"]
    for name in pops:
        fields.extend([f"{name}_pop{i + 1}" for i in range(3)])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for it, t in enumerate(times):
            row = {"time": float(t)}
            for name, pop in pops.items():
                for istate in range(3):
                    row[f"{name}_pop{istate + 1}"] = float(pop[it, istate])
            writer.writerow(row)


def _write_summary_csv(path, pops):
    dvr = pops["dvr"]
    fields = ["method"]
    fields.extend([f"final_pop{i + 1}" for i in range(3)])
    fields.extend([f"rms_pop{i + 1}_vs_dvr" for i in range(3)])
    fields.append("rms_all_states_vs_dvr")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, pop in pops.items():
            diff = pop - dvr
            row = {"method": name}
            for istate in range(3):
                row[f"final_pop{istate + 1}"] = float(pop[-1, istate])
                row[f"rms_pop{istate + 1}_vs_dvr"] = float(
                    np.sqrt(np.mean(diff[:, istate] ** 2))
                )
            row["rms_all_states_vs_dvr"] = float(np.sqrt(np.mean(diff**2)))
            writer.writerow(row)


def _plot(path, times, pops, args):
    colors = {
        "dvr": "#111111",
        "spinmint": "#1f77b4",
        "fssh": "#d62728",
        "ehrenfest": "#2ca02c",
    }
    labels = {
        "dvr": "Exact DVR",
        "spinmint": f"Spin-MInt W ({args.spin_ntraj} traj)",
        "fssh": f"FSSH ({args.fssh_ntraj} traj)",
        "ehrenfest": f"Ehrenfest ({args.ehrenfest_ntraj} traj)",
    }
    linestyles = {
        "dvr": "-",
        "spinmint": "-",
        "fssh": "-",
        "ehrenfest": "--",
    }
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.4), sharex=True)
    order = ["dvr", "spinmint", "fssh", "ehrenfest"]
    for istate, ax in enumerate(axes):
        for name in order:
            kwargs = {
                "label": labels[name] if istate == 0 else None,
                "color": colors[name],
                "lw": 2.5 if name == "dvr" else 1.9,
                "ls": linestyles[name],
            }
            if name == "fssh":
                kwargs["drawstyle"] = "steps-post"
                kwargs["alpha"] = 0.82
            ax.plot(times, pops[name][:, istate], **kwargs)
        ax.set_ylabel(f"State {istate + 1} pop.")
        ax.set_ylim(-0.08, 1.08)
        ax.grid(alpha=0.22, lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title(
        "Three-state Morse Model A "
        f"(R0={args.R0:g}, P0={args.P0:g}, start={args.start_state + 1})"
    )
    axes[0].legend(frameon=False, ncols=2, loc="best")
    axes[-1].set_xlabel("Time (a.u.)")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-time", type=float, default=3000.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--mass", type=float, default=20000.0)
    parser.add_argument("--omega", type=float, default=0.005)
    parser.add_argument("--R0", type=float, default=2.1)
    parser.add_argument("--P0", type=float, default=0.0)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--spin-ntraj", type=int, default=300)
    parser.add_argument("--fssh-ntraj", type=int, default=300)
    parser.add_argument("--ehrenfest-ntraj", type=int, default=300)
    parser.add_argument("--spin-initial-phase", choices=["random", "zero"], default="random")
    parser.add_argument("--dvr-ndvr", type=int, default=256)
    parser.add_argument("--dvr-xmin", type=float, default=1.6)
    parser.add_argument("--dvr-xmax", type=float, default=13.6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "three_state_morse_method_comparison",
    )
    args = parser.parse_args()
    args.nstep = int(round(args.total_time / args.dt))
    args.total_time = args.nstep * args.dt
    if args.start_state not in (0, 1, 2):
        raise ValueError("start-state must be 0, 1, or 2.")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = np.arange(args.nstep + 1, dtype=float) * args.dt
    pops = {
        "ehrenfest": run_ehrenfest(args),
        "fssh": run_fssh(args),
        "spinmint": run_spinmint(args),
        "dvr": run_dvr(args),
    }
    for name, pop in pops.items():
        if pop.shape != (len(times), 3):
            raise RuntimeError(f"{name} returned shape {pop.shape}, expected {(len(times), 3)}.")

    np.savez(args.output_dir / "populations.npz", time=times, **pops)
    _write_population_csv(args.output_dir / "three_state_morse_method_comparison.csv", times, pops)
    _write_summary_csv(args.output_dir / "three_state_morse_method_comparison_summary.csv", pops)
    _plot(args.output_dir / "three_state_morse_method_comparison.png", times, pops, args)

    print(f"wrote {args.output_dir}")
    for name, pop in pops.items():
        values = " ".join(f"P{i + 1}={value:.6f}" for i, value in enumerate(pop[-1]))
        print(f"{name:10s} final {values}")


if __name__ == "__main__":
    main()
