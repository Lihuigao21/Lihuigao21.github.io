"""Small P-Matrix diagnostics for decoherence and detailed balance.

The examples here are intentionally simpler than a scattering model.  They are
designed to isolate the two corrections in the Kang-Wang P-Matrix formalism:

1. pairwise damping of off-diagonal density-matrix elements;
2. Boltzmann suppression of energy-increasing population transfer.
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

from toymodel.distribution import UniformDistribution  # noqa: E402
from toymodel.methods import Ehrenfest, FSSH, PMatrix  # noqa: E402
from toymodel.model.modelbase import ModelSnapshot  # noqa: E402
from toymodel.utils.constant import kB  # noqa: E402


class ConstantGapNACModel:
    """Two-state adiabatic model with fixed gap and fixed derivative coupling."""

    def __init__(self, gap=0.002, nac=0.02, mass=2000.0):
        self.nstate = 2
        self.gap = float(gap)
        self.nac = float(nac)
        self.mass = np.array(float(mass))

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


def _distribution(args, *, p):
    return UniformDistribution(
        Ntraj=1,
        T=args.temperature,
        q0=0.0,
        pmin=p,
        pmax=p,
        mass=args.mass,
    )


def _case_distribution(args, *, p, temperature, ntraj=1):
    return UniformDistribution(
        Ntraj=ntraj,
        T=temperature,
        q0=0.0,
        pmin=p,
        pmax=p,
        mass=args.mass,
    )


def _set_initial_density(method, density):
    density = np.asarray(density, dtype=np.complex128)
    P = 0.5 * density
    method.P = method._renormalize_pmatrix(P)
    method.rho = method._density_from_pmatrix(method.P)


def _set_ehrenfest_superposition(method):
    method.phi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    method.rho = np.outer(method.phi, method.phi.conjugate())


def _run_manual(method, nstep, *, initial_density=None, ehrenfest_superposition=False):
    method.initialize()
    if initial_density is not None:
        _set_initial_density(method, initial_density)
    if ehrenfest_superposition:
        _set_ehrenfest_superposition(method)

    pop = [np.real(np.diag(method.rho)).copy()]
    coh = [abs(method.rho[0, 1])]
    for _ in range(nstep):
        method.step()
        pop.append(np.real(np.diag(method.rho)).copy())
        coh.append(abs(method.rho[0, 1]))
    return np.asarray(pop), np.asarray(coh)


def decoherence_demo(args):
    nstep = int(round(args.decoherence_time_total / args.decoherence_dt))
    times = np.arange(nstep + 1, dtype=float) * args.decoherence_dt
    density0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

    model = ConstantGapNACModel(gap=args.gap, nac=0.0, mass=args.mass)
    common = dict(
        model=model,
        Nstep=nstep,
        dt=args.decoherence_dt,
        start_state=0,
        is_record=False,
        verbose=False,
    )

    pm_tau = PMatrix(
        distribution=_distribution(args, p=0.0),
        decoherence_time=args.decoherence_tau,
        detailed_balance=False,
        nuclear_force="free",
        **common,
    )
    pop_tau, coh_tau = _run_manual(pm_tau, nstep, initial_density=density0)

    pm_inf = PMatrix(
        distribution=_distribution(args, p=0.0),
        decoherence_time=np.inf,
        detailed_balance=False,
        nuclear_force="free",
        **common,
    )
    pop_inf, coh_inf = _run_manual(pm_inf, nstep, initial_density=density0)

    ehrenfest = Ehrenfest(
        distribution=_distribution(args, p=0.0),
        **common,
    )
    pop_eh, coh_eh = _run_manual(ehrenfest, nstep, ehrenfest_superposition=True)

    return {
        "time": times,
        "pmatrix_tau_pop": pop_tau,
        "pmatrix_tau_coherence": coh_tau,
        "pmatrix_inf_pop": pop_inf,
        "pmatrix_inf_coherence": coh_inf,
        "ehrenfest_pop": pop_eh,
        "ehrenfest_coherence": coh_eh,
    }


def detailed_balance_demo(args):
    nstep = int(round(args.balance_time_total / args.balance_dt))
    times = np.arange(nstep + 1, dtype=float) * args.balance_dt
    model = ConstantGapNACModel(gap=args.gap, nac=args.nac, mass=args.mass)
    common = dict(
        model=model,
        distribution=_case_distribution(
            args, p=args.momentum, temperature=args.temperature
        ),
        Nstep=nstep,
        dt=args.balance_dt,
        start_state=1,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        decoherence_time=args.balance_tau,
        electronic_substeps=args.balance_substeps,
        nuclear_force="free",
        verbose=False,
    )

    with_db = PMatrix(detailed_balance=True, **common)
    with_db_result = with_db.run()
    with_db_pop = np.vstack([[0.0, 1.0], np.asarray(with_db_result["pop"])])

    without_db = PMatrix(detailed_balance=False, **common)
    without_db_result = without_db.run()
    without_db_pop = np.vstack([[0.0, 1.0], np.asarray(without_db_result["pop"])])

    fssh_pop = run_fssh_balance_case(
        args,
        gap=args.gap,
        temperature=args.temperature,
        nstep=nstep,
        dt=args.balance_dt,
    )
    ehrenfest_pop = run_ehrenfest_balance_case(
        args,
        gap=args.gap,
        temperature=args.temperature,
        nstep=nstep,
        dt=args.balance_dt,
    )

    ratio = np.exp(-args.gap / (kB * args.temperature))
    high_target = ratio / (1.0 + ratio)

    return {
        "time": times,
        "with_db_pop": with_db_pop,
        "without_db_pop": without_db_pop,
        "fssh_pop": fssh_pop,
        "ehrenfest_pop": ehrenfest_pop,
        "boltzmann_high": float(high_target),
        "boltzmann_ratio": float(ratio),
    }


def run_pmatrix_balance_case(args, *, gap, temperature, detailed_balance, nstep, dt):
    model = ConstantGapNACModel(gap=gap, nac=args.nac, mass=args.mass)
    method = PMatrix(
        model=model,
        distribution=_case_distribution(
            args, p=args.momentum, temperature=temperature
        ),
        Nstep=nstep,
        dt=dt,
        start_state=1,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        decoherence_time=args.balance_tau,
        electronic_substeps=args.balance_substeps,
        detailed_balance=detailed_balance,
        nuclear_force="free",
        verbose=False,
    )
    result = method.run()
    return np.vstack([[0.0, 1.0], np.asarray(result["pop"])])


def run_fssh_balance_case(args, *, gap, temperature, nstep, dt):
    np.random.seed(args.fssh_seed)
    random.seed(args.fssh_seed)
    model = ConstantGapNACModel(gap=gap, nac=args.nac, mass=args.mass)
    method = FSSH(
        model=model,
        distribution=_case_distribution(
            args,
            p=args.momentum,
            temperature=temperature,
            ntraj=args.fssh_ntraj,
        ),
        Nstep=nstep,
        dt=dt,
        start_state=1,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        verbose=False,
    )
    method.run()
    pops = np.asarray([traj["pop"] for traj in method.result], dtype=float)
    avg = np.mean(pops, axis=0)
    return np.vstack([[0.0, 1.0], avg])


def run_ehrenfest_balance_case(args, *, gap, temperature, nstep, dt):
    model = ConstantGapNACModel(gap=gap, nac=args.nac, mass=args.mass)
    method = Ehrenfest(
        model=model,
        distribution=_case_distribution(
            args, p=args.momentum, temperature=temperature
        ),
        Nstep=nstep,
        dt=dt,
        start_state=1,
        is_record=True,
        legacy_result=False,
        record_level="minimal",
        verbose=False,
    )
    method.run()
    return np.vstack([[0.0, 1.0], np.asarray(method.result["pop"])])


def balance_model_sweep(args):
    nstep = int(round(args.balance_time_total / args.balance_dt))
    cases = [
        ("gap_0.001_T300", 0.001, 300.0),
        ("gap_0.002_T300", 0.002, 300.0),
        ("gap_0.003_T300", 0.003, 300.0),
        ("gap_0.002_T600", 0.002, 600.0),
    ]
    rows = []
    for offset, (name, gap, temperature) in enumerate(cases):
        old_seed = args.fssh_seed
        args.fssh_seed = old_seed + offset * 1000
        with_db = run_pmatrix_balance_case(
            args,
            gap=gap,
            temperature=temperature,
            detailed_balance=True,
            nstep=nstep,
            dt=args.balance_dt,
        )
        without_db = run_pmatrix_balance_case(
            args,
            gap=gap,
            temperature=temperature,
            detailed_balance=False,
            nstep=nstep,
            dt=args.balance_dt,
        )
        fssh = run_fssh_balance_case(
            args,
            gap=gap,
            temperature=temperature,
            nstep=nstep,
            dt=args.balance_dt,
        )
        ehrenfest = run_ehrenfest_balance_case(
            args,
            gap=gap,
            temperature=temperature,
            nstep=nstep,
            dt=args.balance_dt,
        )
        args.fssh_seed = old_seed

        ratio = np.exp(-gap / (kB * temperature))
        target = ratio / (1.0 + ratio)
        rows.append(
            {
                "case": name,
                "gap": gap,
                "temperature": temperature,
                "boltzmann_high": float(target),
                "pmatrix_db_high": float(with_db[-1, 1]),
                "pmatrix_no_db_high": float(without_db[-1, 1]),
                "fssh_high": float(fssh[-1, 1]),
                "ehrenfest_high": float(ehrenfest[-1, 1]),
                "pmatrix_db_abs_error": float(abs(with_db[-1, 1] - target)),
                "pmatrix_no_db_abs_error": float(abs(without_db[-1, 1] - target)),
                "fssh_abs_error": float(abs(fssh[-1, 1] - target)),
                "ehrenfest_abs_error": float(abs(ehrenfest[-1, 1] - target)),
            }
        )
    return rows


def _write_decoherence_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "pmatrix_tau_abs_rho01",
                "pmatrix_inf_abs_rho01",
                "ehrenfest_abs_rho01",
            ],
        )
        writer.writeheader()
        for i, t in enumerate(data["time"]):
            writer.writerow(
                {
                    "time": float(t),
                    "pmatrix_tau_abs_rho01": float(data["pmatrix_tau_coherence"][i]),
                    "pmatrix_inf_abs_rho01": float(data["pmatrix_inf_coherence"][i]),
                    "ehrenfest_abs_rho01": float(data["ehrenfest_coherence"][i]),
                }
            )


def _write_balance_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "pmatrix_db_low",
                "pmatrix_db_high",
                "pmatrix_no_db_low",
                "pmatrix_no_db_high",
                "fssh_low",
                "fssh_high",
                "ehrenfest_low",
                "ehrenfest_high",
                "boltzmann_high",
            ],
        )
        writer.writeheader()
        for i, t in enumerate(data["time"]):
            writer.writerow(
                {
                    "time": float(t),
                    "pmatrix_db_low": float(data["with_db_pop"][i, 0]),
                    "pmatrix_db_high": float(data["with_db_pop"][i, 1]),
                    "pmatrix_no_db_low": float(data["without_db_pop"][i, 0]),
                    "pmatrix_no_db_high": float(data["without_db_pop"][i, 1]),
                    "fssh_low": float(data["fssh_pop"][i, 0]),
                    "fssh_high": float(data["fssh_pop"][i, 1]),
                    "ehrenfest_low": float(data["ehrenfest_pop"][i, 0]),
                    "ehrenfest_high": float(data["ehrenfest_pop"][i, 1]),
                    "boltzmann_high": data["boltzmann_high"],
                }
            )


def _write_balance_sweep_csv(path, rows):
    fieldnames = [
        "case",
        "gap",
        "temperature",
        "boltzmann_high",
        "pmatrix_db_high",
        "pmatrix_no_db_high",
        "fssh_high",
        "ehrenfest_high",
        "pmatrix_db_abs_error",
        "pmatrix_no_db_abs_error",
        "fssh_abs_error",
        "ehrenfest_abs_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_decoherence(path, data, args):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        data["time"],
        data["pmatrix_tau_coherence"],
        lw=2.2,
        label=f"P-Matrix tau={args.decoherence_tau:g}",
        color="#1f77b4",
    )
    ax.plot(
        data["time"],
        data["pmatrix_inf_coherence"],
        lw=2.0,
        label="P-Matrix tau=inf",
        color="#d62728",
    )
    ax.plot(
        data["time"],
        data["ehrenfest_coherence"],
        lw=1.8,
        ls="--",
        label="Ehrenfest",
        color="#2ca02c",
    )
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("|rho_01|")
    ax.set_title("Pairwise decoherence diagnostic")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_balance(path, data, args):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        data["time"],
        data["with_db_pop"][:, 1],
        lw=2.4,
        label="P-Matrix detailed balance",
        color="#1b6ca8",
    )
    ax.plot(
        data["time"],
        data["without_db_pop"][:, 1],
        lw=2.0,
        ls="--",
        label="P-Matrix no detailed balance",
        color="#c43c39",
    )
    ax.plot(
        data["time"],
        data["fssh_pop"][:, 1],
        lw=1.5,
        alpha=0.72,
        label=f"FSSH ({args.fssh_ntraj} traj)",
        color="#3a923a",
        drawstyle="steps-post",
    )
    ax.plot(
        data["time"],
        data["ehrenfest_pop"][:, 1],
        lw=1.8,
        ls="-.",
        label="Ehrenfest",
        color="#7b4ab2",
    )
    ax.axhline(
        data["boltzmann_high"],
        color="#111111",
        lw=1.6,
        ls=":",
        label="Boltzmann target",
    )
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("High-energy-state population")
    ax.set_title(
        "Detailed-balance diagnostic "
        f"(DeltaE={args.gap:g}, T={args.temperature:g} K)"
    )
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_balance_sweep(path, rows):
    labels = [f"DeltaE={row['gap']:.3f}, T={row['temperature']:.0f} K" for row in rows]
    y = np.arange(len(rows))[::-1]
    target = np.array([row["boltzmann_high"] for row in rows])
    series = [
        ("Boltzmann", target, "#111111", "D", 56, 0.0),
        ("P-Matrix DB", np.array([row["pmatrix_db_high"] for row in rows]), "#1b6ca8", "o", 54, -0.18),
        ("P-Matrix no DB", np.array([row["pmatrix_no_db_high"] for row in rows]), "#c43c39", "s", 50, 0.18),
        ("FSSH", np.array([row["fssh_high"] for row in rows]), "#3a923a", "^", 58, 0.36),
        ("Ehrenfest", np.array([row["ehrenfest_high"] for row in rows]), "#7b4ab2", "v", 58, -0.36),
    ]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for idx, row in enumerate(rows):
        vals = [
            row["pmatrix_db_high"],
            row["pmatrix_no_db_high"],
            row["fssh_high"],
            row["ehrenfest_high"],
        ]
        xmin = min(row["boltzmann_high"], *vals)
        xmax = max(row["boltzmann_high"], *vals)
        ax.hlines(y[idx], xmin, xmax, color="#d8d8d8", lw=1.4, zorder=0)

    for label, values, color, marker, size, offset in series:
        ax.scatter(
            values,
            y + offset,
            s=size,
            marker=marker,
            color=color,
            edgecolor="white" if label != "Boltzmann" else color,
            linewidth=0.7,
            label=label,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Final high-energy-state population")
    ax.set_title("Detailed-balance sweep over constant-gap toy models")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_summary(path, decoherence, balance):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow(
            {
                "metric": "decoherence_final_pmatrix_tau_abs_rho01",
                "value": float(decoherence["pmatrix_tau_coherence"][-1]),
            }
        )
        writer.writerow(
            {
                "metric": "decoherence_final_pmatrix_inf_abs_rho01",
                "value": float(decoherence["pmatrix_inf_coherence"][-1]),
            }
        )
        writer.writerow(
            {
                "metric": "decoherence_final_ehrenfest_abs_rho01",
                "value": float(decoherence["ehrenfest_coherence"][-1]),
            }
        )
        writer.writerow(
            {
                "metric": "balance_final_with_db_high_pop",
                "value": float(balance["with_db_pop"][-1, 1]),
            }
        )
        writer.writerow(
            {
                "metric": "balance_final_without_db_high_pop",
                "value": float(balance["without_db_pop"][-1, 1]),
            }
        )
        writer.writerow(
            {
                "metric": "balance_final_fssh_high_pop",
                "value": float(balance["fssh_pop"][-1, 1]),
            }
        )
        writer.writerow(
            {
                "metric": "balance_final_ehrenfest_high_pop",
                "value": float(balance["ehrenfest_pop"][-1, 1]),
            }
        )
        writer.writerow(
            {
                "metric": "balance_boltzmann_high_pop",
                "value": float(balance["boltzmann_high"]),
            }
        )
        writer.writerow(
            {
                "metric": "balance_boltzmann_high_low_ratio",
                "value": float(balance["boltzmann_ratio"]),
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "pmatrix_decoherence_balance",
    )
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--gap", type=float, default=0.002)
    parser.add_argument("--decoherence-time-total", type=float, default=800.0)
    parser.add_argument("--decoherence-dt", type=float, default=2.0)
    parser.add_argument("--decoherence-tau", type=float, default=100.0)
    parser.add_argument("--balance-time-total", type=float, default=10000.0)
    parser.add_argument("--balance-dt", type=float, default=2.0)
    parser.add_argument("--balance-tau", type=float, default=100.0)
    parser.add_argument("--balance-substeps", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=200.0)
    parser.add_argument("--nac", type=float, default=0.02)
    parser.add_argument("--fssh-ntraj", type=int, default=32)
    parser.add_argument("--fssh-seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoherence = decoherence_demo(args)
    balance = detailed_balance_demo(args)
    sweep = balance_model_sweep(args)

    np.savez(
        args.output_dir / "diagnostics.npz",
        **{f"decoherence_{key}": value for key, value in decoherence.items()},
        **{f"balance_{key}": value for key, value in balance.items()},
    )
    _write_decoherence_csv(args.output_dir / "decoherence.csv", decoherence)
    _write_balance_csv(args.output_dir / "detailed_balance.csv", balance)
    _write_balance_sweep_csv(args.output_dir / "detailed_balance_sweep.csv", sweep)
    _write_summary(args.output_dir / "summary.csv", decoherence, balance)
    _plot_decoherence(args.output_dir / "decoherence_rho01.png", decoherence, args)
    _plot_balance(args.output_dir / "detailed_balance_boltzmann.png", balance, args)
    _plot_balance_sweep(args.output_dir / "detailed_balance_sweep.png", sweep)

    print(f"wrote {args.output_dir}")
    print(
        "decoherence final |rho01|: "
        f"PMatrix(tau)={decoherence['pmatrix_tau_coherence'][-1]:.6f}, "
        f"PMatrix(inf)={decoherence['pmatrix_inf_coherence'][-1]:.6f}, "
        f"Ehrenfest={decoherence['ehrenfest_coherence'][-1]:.6f}"
    )
    print(
        "detailed balance final high pop: "
        f"with_db={balance['with_db_pop'][-1, 1]:.6f}, "
        f"without_db={balance['without_db_pop'][-1, 1]:.6f}, "
        f"fssh={balance['fssh_pop'][-1, 1]:.6f}, "
        f"ehrenfest={balance['ehrenfest_pop'][-1, 1]:.6f}, "
        f"boltzmann={balance['boltzmann_high']:.6f}"
    )
    print("detailed-balance sweep:")
    for row in sweep:
        print(
            f"  {row['case']}: P-DB={row['pmatrix_db_high']:.6f}, "
            f"P-noDB={row['pmatrix_no_db_high']:.6f}, "
            f"FSSH={row['fssh_high']:.6f}, "
            f"Ehrenfest={row['ehrenfest_high']:.6f}, "
            f"Boltzmann={row['boltzmann_high']:.6f}"
        )


if __name__ == "__main__":
    main()
