from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

def _find_toymodel_root():
    env_root = os.environ.get("TOYMODEL_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "src" / "toymodel").exists():
            return root
        raise RuntimeError("TOYMODEL_ROOT does not contain src/toymodel.")
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "toymodel").exists():
            return parent
    raise RuntimeError("Set TOYMODEL_ROOT to a Toymodel checkout containing src/toymodel.")


ROOT = _find_toymodel_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cmd_curvature_smoke import (  # noqa: E402
    HARTREE_TO_WAVENUMBER,
    classical_velocity_autocorrelation,
    fermi_window,
    peak_position,
    spectrum_from_correlation,
)
from toymodel.DVRmethods import RadialDVRKubo  # noqa: E402
from toymodel.methods import (  # noqa: E402
    CMDRadialDynamics,
    RingPolymerRadialDynamics,
    estimate_radial_cmd_pmf,
    minimize_centroid_constrained_path,
)
from toymodel.model import ChampagneBottleMorse2D  # noqa: E402
from toymodel.utils.constant import fs_to_au, kB  # noqa: E402


def positive_area(omega_cm, spectrum, lo, hi):
    omega_cm = np.asarray(omega_cm, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    mask = (omega_cm >= float(lo)) & (omega_cm <= float(hi))
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapezoid(np.maximum(spectrum[mask], 0.0), omega_cm[mask]))


def spectrum_metrics(omega_cm, spectrum, *, peak_min_cm, peak_max_cm):
    global_peak, global_intensity = peak_position(
        omega_cm,
        spectrum,
        min_cm=0.0,
        max_cm=float(np.max(omega_cm)),
    )
    stretch_peak, stretch_intensity = peak_position(
        omega_cm,
        spectrum,
        min_cm=peak_min_cm,
        max_cm=peak_max_cm,
    )
    paper_peak, paper_intensity = peak_position(
        omega_cm,
        spectrum,
        min_cm=3000.0,
        max_cm=4000.0,
    )
    soft_area = positive_area(omega_cm, spectrum, 1500.0, 3000.0)
    stretch_area = positive_area(omega_cm, spectrum, 3000.0, 4000.0)
    total_area = positive_area(omega_cm, spectrum, 0.0, 5000.0)
    return {
        "global_peak_cm": global_peak,
        "global_intensity": global_intensity,
        "stretch_peak_cm": stretch_peak,
        "stretch_intensity": stretch_intensity,
        "paper_peak_cm": paper_peak,
        "paper_intensity": paper_intensity,
        "soft_area_1500_3000": soft_area,
        "stretch_area_3000_4000": stretch_area,
        "soft_to_stretch_area": soft_area / stretch_area if stretch_area > 0.0 else np.nan,
        "soft_fraction_total": soft_area / total_area if total_area > 0.0 else np.nan,
    }


def radial_distribution_metrics(pmf, temperature):
    rgrid = np.asarray(pmf.rgrid, dtype=float)
    beta = 1.0 / (kB * float(temperature))
    weights = rgrid ** (pmf.ndim - 1) * np.exp(-beta * (pmf.pmf - np.min(pmf.pmf)))
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)
    return weights, {
        "r_mode": float(rgrid[np.argmax(weights)]),
        "r_mean": float(np.sum(rgrid * weights)),
        "prob_r_lt_0p4": float(np.sum(weights[rgrid < 0.4])),
        "prob_r_lt_0p8": float(np.sum(weights[rgrid < 0.8])),
        "prob_r_lt_1p2": float(np.sum(weights[rgrid < 1.2])),
    }


def summarize_seed_rows(rows):
    grouped = {}
    for row in rows:
        if row["method"] != "cmd_ca":
            continue
        grouped.setdefault(float(row["temperature_K"]), []).append(row)
    out = []
    keys = [
        "global_peak_cm",
        "stretch_peak_cm",
        "paper_peak_cm",
        "soft_to_stretch_area",
        "soft_fraction_total",
        "r_mode",
        "r_mean",
        "prob_r_lt_0p4",
        "prob_r_lt_1p2",
        "energy_drift_rms",
    ]
    for temperature, entries in sorted(grouped.items()):
        summary = {"temperature_K": temperature, "nseed": len(entries)}
        for key in keys:
            values = np.asarray([float(row[key]) for row in entries], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def plot_spectra(path, spectra_by_temp, cmd_seed_spectra_by_temp):
    import matplotlib.pyplot as plt

    ntemp = len(spectra_by_temp)
    fig, axes = plt.subplots(1, ntemp, figsize=(5.2 * ntemp, 3.6), squeeze=False)
    colors = {
        "DVR": "black",
        "Classical": "tab:orange",
        "RPMD": "tab:green",
        "TRPMD": "tab:blue",
        "CMD mean": "tab:red",
    }
    for ax, temperature in zip(axes[0], spectra_by_temp):
        for omega, spectrum in cmd_seed_spectra_by_temp.get(temperature, []):
            ymax = np.max(np.abs(spectrum))
            ax.plot(omega, spectrum / ymax, color="tab:red", alpha=0.25, lw=0.9)
        cmd_stack = [spectrum for _, spectrum in cmd_seed_spectra_by_temp.get(temperature, [])]
        if cmd_stack:
            omega = cmd_seed_spectra_by_temp[temperature][0][0]
            mean = np.mean(np.vstack(cmd_stack), axis=0)
            ymax = np.max(np.abs(mean))
            ax.plot(omega, mean / ymax, color=colors["CMD mean"], lw=2.0, label="CMD seeds mean")
        for label, (omega, spectrum) in spectra_by_temp[temperature].items():
            ymax = np.max(np.abs(spectrum))
            y = spectrum / ymax if ymax > 0.0 else spectrum
            ax.plot(omega, y, lw=1.6, color=colors.get(label), label=label)
        ax.axvspan(1500.0, 3000.0, color="tab:red", alpha=0.07)
        ax.axvspan(3000.0, 4000.0, color="tab:blue", alpha=0.06)
        ax.set_title(f"T={temperature:g} K")
        ax.set_xlim(0.0, 5000.0)
        ax.set_xlabel("frequency / cm$^{-1}$")
        ax.set_ylabel("normalized intensity")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cmd_pmf(path, pmf_records):
    import matplotlib.pyplot as plt

    temps = sorted(pmf_records)
    fig, axes = plt.subplots(2, len(temps), figsize=(5.0 * len(temps), 6.0), squeeze=False)
    for col, temperature in enumerate(temps):
        records = pmf_records[temperature]
        for record in records:
            pmf = record["pmf"]
            weight = record["radial_weight"]
            label = f"seed {record['seed']}"
            axes[0, col].plot(pmf.rgrid, pmf.pmf, lw=1.0, alpha=0.8, label=label)
            axes[1, col].plot(pmf.rgrid, weight, lw=1.0, alpha=0.8, label=label)
        axes[0, col].set_title(f"T={temperature:g} K")
        axes[0, col].set_xlabel("centroid radius / bohr")
        axes[0, col].set_ylabel("CMD PMF / Hartree")
        axes[1, col].set_xlabel("centroid radius / bohr")
        axes[1, col].set_ylabel("normalized radial weight")
        axes[0, col].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_peak_summary(path, rows, cmd_seed_summary):
    import matplotlib.pyplot as plt

    temps = sorted({float(row["temperature_K"]) for row in rows})
    methods = ["exact_dvr", "classical_eh", "rpmd", "trpmd"]
    labels = {
        "exact_dvr": "DVR",
        "classical_eh": "Classical",
        "rpmd": "RPMD",
        "trpmd": "TRPMD",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method in methods:
        y = []
        for temperature in temps:
            row = next(
                row
                for row in rows
                if row["method"] == method and float(row["temperature_K"]) == temperature
            )
            y.append(float(row["paper_peak_cm"]))
        ax.plot(temps, y, marker="o", label=labels[method])
    cmd_by_temp = {float(row["temperature_K"]): row for row in cmd_seed_summary}
    cmd_mean = [float(cmd_by_temp[t]["paper_peak_cm_mean"]) for t in temps]
    cmd_std = [float(cmd_by_temp[t]["paper_peak_cm_std"]) for t in temps]
    ax.errorbar(temps, cmd_mean, yerr=cmd_std, marker="o", capsize=3, label="CMD mean")
    ax.set_xlabel("temperature / K")
    ax.set_ylabel("peak in 3000-4000 cm$^{-1}$ band")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = ChampagneBottleMorse2D()
    omega_cm = np.linspace(0.0, args.omega_max_cm, args.nomega)
    time = np.arange(args.nstep + 1, dtype=float) * args.dt
    window = fermi_window(
        time,
        args.window_half_fs * fs_to_au,
        args.window_tau_fs * fs_to_au,
    )
    rows = []
    cmd_rows = []
    spectra_by_temp = {}
    cmd_seed_spectra_by_temp = {}
    pmf_records = {}

    for itemp, temperature in enumerate(args.temperatures):
        print(f"[temperature] {temperature:g} K", flush=True)
        spectra_by_temp[temperature] = {}
        cmd_seed_spectra_by_temp[temperature] = []
        pmf_records[temperature] = []

        exact = RadialDVRKubo(
            model=model,
            temperature=temperature,
            nr=args.radial_nr,
            rmax=args.radial_rmax,
            angular_max=args.angular_max,
            tgrid=time,
        )
        exact_corr = exact.radial_velocity_autocorrelation()
        _, exact_spectrum = exact.radial_velocity_spectrum_from_lines(
            omega_cm=omega_cm,
            broadening_cm=args.exact_broadening_cm,
        )
        row = {
            "temperature_K": temperature,
            "method": "exact_dvr",
            "seed": "",
            "corr0": float(np.real(exact_corr[0])),
            "energy_drift_rms": "",
            "acceptance_mean": "",
            "acceptance_min": "",
            "nbeads": "",
            **spectrum_metrics(
                omega_cm,
                exact_spectrum,
                peak_min_cm=args.peak_min_cm,
                peak_max_cm=args.peak_max_cm,
            ),
        }
        rows.append(row)
        spectra_by_temp[temperature]["DVR"] = (omega_cm, exact_spectrum)
        np.savez_compressed(
            outdir / f"exact_T{int(temperature)}.npz",
            time=time,
            correlation=exact_corr,
            omega_cm=omega_cm,
            spectrum=exact_spectrum,
            evals=np.concatenate([block.evals for block in exact.blocks]),
            angular_partition=np.asarray(
                [(k, v) for k, v in exact.angular_partition_weights().items()],
                dtype=float,
            ),
        )

        cmd_beads = args.cmd_nbeads_low if temperature <= args.low_temperature_cutoff else args.cmd_nbeads_high
        for iseed, seed in enumerate(args.cmd_seeds):
            print(f"  [cmd] seed={seed}", flush=True)
            pmf = estimate_radial_cmd_pmf(
                model=model,
                rgrid=np.linspace(args.rmin, args.rmax, args.pmf_points),
                temperature=temperature,
                nbeads=cmd_beads,
                nsteps=args.pimc_steps,
                burnin=args.pimc_burnin,
                step_size=args.pimc_step_size,
                sample_stride=args.pimc_stride,
                seed=seed + 1000 * itemp,
                initial_path_mode="curved",
            )
            radial_weight, rmetrics = radial_distribution_metrics(pmf, temperature)
            cmd = CMDRadialDynamics(pmf, seed=seed + 2000 * itemp)
            cmd_data = cmd.radial_velocity_autocorrelation(
                ntraj=args.cmd_ntraj,
                dt=args.dt,
                nstep=args.nstep,
            )
            cmd_spectrum = spectrum_from_correlation(
                cmd_data["time"],
                cmd_data["correlation"],
                omega_cm,
                window=window,
            )
            metrics = spectrum_metrics(
                omega_cm,
                cmd_spectrum,
                peak_min_cm=args.peak_min_cm,
                peak_max_cm=args.peak_max_cm,
            )
            energy_drift_rms = float(np.sqrt(np.mean(cmd_data["energy_drift"] ** 2)))
            cmd_row = {
                "temperature_K": temperature,
                "method": "cmd_ca",
                "seed": seed,
                "corr0": float(cmd_data["correlation"][0]),
                "energy_drift_rms": energy_drift_rms,
                "acceptance_mean": float(np.mean(pmf.acceptance_rate)),
                "acceptance_min": float(np.min(pmf.acceptance_rate)),
                "nbeads": cmd_beads,
                **metrics,
                **rmetrics,
            }
            rows.append(cmd_row)
            cmd_rows.append(cmd_row)
            cmd_seed_spectra_by_temp[temperature].append((omega_cm, cmd_spectrum))
            pmf_records[temperature].append(
                {"seed": seed, "pmf": pmf, "radial_weight": radial_weight}
            )
            np.savez_compressed(
                outdir / f"cmd_T{int(temperature)}_seed{seed}.npz",
                rgrid=pmf.rgrid,
                mean_force=pmf.mean_force,
                force_stderr=pmf.force_stderr,
                pmf=pmf.pmf,
                radial_weight=radial_weight,
                acceptance_rate=pmf.acceptance_rate,
                time=cmd_data["time"],
                correlation=cmd_data["correlation"],
                energy_drift=cmd_data["energy_drift"],
                omega_cm=omega_cm,
                spectrum=cmd_spectrum,
                nbeads=cmd_beads,
            )

        print("  [classical]", flush=True)
        classical = classical_velocity_autocorrelation(
            model,
            temperature=temperature,
            ntraj=args.classical_ntraj,
            dt=args.dt,
            nstep=args.nstep,
            seed=args.seed + 3000 * itemp,
        )
        classical_spectrum = spectrum_from_correlation(
            classical["time"],
            classical["correlation"],
            omega_cm,
            window=window,
        )
        row = {
            "temperature_K": temperature,
            "method": "classical_eh",
            "seed": "",
            "corr0": float(classical["correlation"][0]),
            "energy_drift_rms": float(np.sqrt(np.mean(classical["energy_drift"] ** 2))),
            "acceptance_mean": "",
            "acceptance_min": "",
            "nbeads": "",
            **spectrum_metrics(
                omega_cm,
                classical_spectrum,
                peak_min_cm=args.peak_min_cm,
                peak_max_cm=args.peak_max_cm,
            ),
        }
        rows.append(row)
        spectra_by_temp[temperature]["Classical"] = (omega_cm, classical_spectrum)
        np.savez_compressed(
            outdir / f"classical_T{int(temperature)}.npz",
            **classical,
            omega_cm=omega_cm,
            spectrum=classical_spectrum,
        )

        rp_beads = args.rpmd_nbeads_low if temperature <= args.low_temperature_cutoff else args.rpmd_nbeads_high
        print("  [rpmd]", flush=True)
        rp_dyn = RingPolymerRadialDynamics(
            model=model,
            temperature=temperature,
            nbeads=rp_beads,
            seed=args.seed + 4000 * itemp,
        )
        rpmd = rp_dyn.radial_velocity_autocorrelation(
            ntraj=args.rpmd_ntraj,
            dt=args.dt,
            nstep=args.nstep,
            trpmd_gamma=0.0,
            pimc_steps=args.rpmd_pimc_steps,
            pimc_burnin=args.pimc_burnin,
            pimc_step_size=args.rpmd_pimc_step_size,
            pimc_stride=args.rpmd_pimc_stride,
        )
        rpmd_spectrum = spectrum_from_correlation(
            rpmd["time"],
            rpmd["correlation"],
            omega_cm,
            window=window,
        )
        row = {
            "temperature_K": temperature,
            "method": "rpmd",
            "seed": "",
            "corr0": float(rpmd["correlation"][0]),
            "energy_drift_rms": float(np.sqrt(np.mean(rpmd["energy_drift"] ** 2))),
            "acceptance_mean": float(rpmd["acceptance_rate"]),
            "acceptance_min": "",
            "nbeads": rp_beads,
            **spectrum_metrics(
                omega_cm,
                rpmd_spectrum,
                peak_min_cm=args.peak_min_cm,
                peak_max_cm=args.peak_max_cm,
            ),
        }
        rows.append(row)
        spectra_by_temp[temperature]["RPMD"] = (omega_cm, rpmd_spectrum)
        np.savez_compressed(
            outdir / f"rpmd_T{int(temperature)}.npz",
            **rpmd,
            omega_cm=omega_cm,
            spectrum=rpmd_spectrum,
            nbeads=rp_beads,
        )

        print("  [trpmd]", flush=True)
        tr_dyn = RingPolymerRadialDynamics(
            model=model,
            temperature=temperature,
            nbeads=rp_beads,
            seed=args.seed + 5000 * itemp,
        )
        trpmd = tr_dyn.radial_velocity_autocorrelation(
            ntraj=args.rpmd_ntraj,
            dt=args.dt,
            nstep=args.nstep,
            trpmd_gamma=args.trpmd_gamma,
            pimc_steps=args.rpmd_pimc_steps,
            pimc_burnin=args.pimc_burnin,
            pimc_step_size=args.rpmd_pimc_step_size,
            pimc_stride=args.rpmd_pimc_stride,
        )
        trpmd_spectrum = spectrum_from_correlation(
            trpmd["time"],
            trpmd["correlation"],
            omega_cm,
            window=window,
        )
        row = {
            "temperature_K": temperature,
            "method": "trpmd",
            "seed": "",
            "corr0": float(trpmd["correlation"][0]),
            "energy_drift_rms": float(np.sqrt(np.mean(trpmd["energy_drift"] ** 2))),
            "acceptance_mean": float(trpmd["acceptance_rate"]),
            "acceptance_min": "",
            "nbeads": rp_beads,
            **spectrum_metrics(
                omega_cm,
                trpmd_spectrum,
                peak_min_cm=args.peak_min_cm,
                peak_max_cm=args.peak_max_cm,
            ),
        }
        rows.append(row)
        spectra_by_temp[temperature]["TRPMD"] = (omega_cm, trpmd_spectrum)
        np.savez_compressed(
            outdir / f"trpmd_T{int(temperature)}.npz",
            **trpmd,
            omega_cm=omega_cm,
            spectrum=trpmd_spectrum,
            nbeads=rp_beads,
            gamma=args.trpmd_gamma,
        )

    cmd_seed_summary = summarize_seed_rows(rows)
    write_csv(outdir / "method_summary.csv", rows)
    write_csv(outdir / "cmd_seed_summary.csv", cmd_seed_summary)
    plot_spectra(outdir / "spectra_validation.png", spectra_by_temp, cmd_seed_spectra_by_temp)
    plot_cmd_pmf(outdir / "cmd_pmf_distribution.png", pmf_records)
    plot_peak_summary(outdir / "paper_band_peak_summary.png", rows, cmd_seed_summary)

    for temperature in args.instanton_temperatures:
        for radius in args.instanton_radii:
            nbeads = args.cmd_nbeads_low if temperature <= args.low_temperature_cutoff else args.cmd_nbeads_high
            centroid = np.zeros(model.ndim, dtype=float)
            centroid[0] = radius
            result = minimize_centroid_constrained_path(
                model=model,
                centroid=centroid,
                temperature=temperature,
                nbeads=nbeads,
                seed=args.seed + int(temperature) + int(100 * radius),
                maxiter=args.instanton_maxiter,
            )
            np.savez_compressed(
                outdir / f"instanton_T{int(temperature)}_R{radius:.2f}.npz",
                path=result["path"],
                centroid=result["centroid"],
                action=result["action"],
                success=result["success"],
                niter=result["niter"],
                temperature=temperature,
                radius=radius,
                nbeads=nbeads,
            )

    print(f"[done] wrote {outdir}", flush=True)
    return rows, cmd_seed_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="output/cmd_curvature/validation_medium")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[200.0, 400.0, 800.0])
    parser.add_argument("--cmd-seeds", nargs="+", type=int, default=[20260617, 20260618, 20260619])
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--radial-nr", type=int, default=128)
    parser.add_argument("--radial-rmax", type=float, default=6.0)
    parser.add_argument("--angular-max", type=int, default=20)
    parser.add_argument("--exact-broadening-cm", type=float, default=75.0)
    parser.add_argument("--omega-max-cm", type=float, default=5000.0)
    parser.add_argument("--nomega", type=int, default=900)
    parser.add_argument("--peak-min-cm", type=float, default=2500.0)
    parser.add_argument("--peak-max-cm", type=float, default=4500.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--nstep", type=int, default=2000)
    parser.add_argument("--window-half-fs", type=float, default=20.0)
    parser.add_argument("--window-tau-fs", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=3.8)
    parser.add_argument("--pmf-points", type=int, default=39)
    parser.add_argument("--cmd-nbeads-low", type=int, default=32)
    parser.add_argument("--cmd-nbeads-high", type=int, default=16)
    parser.add_argument("--rpmd-nbeads-low", type=int, default=16)
    parser.add_argument("--rpmd-nbeads-high", type=int, default=8)
    parser.add_argument("--low-temperature-cutoff", type=float, default=300.0)
    parser.add_argument("--pimc-steps", type=int, default=3000)
    parser.add_argument("--pimc-burnin", type=float, default=0.25)
    parser.add_argument("--pimc-stride", type=int, default=10)
    parser.add_argument("--pimc-step-size", type=float, default=0.25)
    parser.add_argument("--cmd-ntraj", type=int, default=200)
    parser.add_argument("--classical-ntraj", type=int, default=250)
    parser.add_argument("--rpmd-pimc-steps", type=int, default=2200)
    parser.add_argument("--rpmd-pimc-stride", type=int, default=8)
    parser.add_argument("--rpmd-pimc-step-size", type=float, default=0.08)
    parser.add_argument("--rpmd-ntraj", type=int, default=120)
    parser.add_argument("--trpmd-gamma", type=float, default=0.01)
    parser.add_argument("--instanton-temperatures", nargs="+", type=float, default=[200.0, 800.0])
    parser.add_argument("--instanton-radii", nargs="+", type=float, default=[0.4, 0.9, 1.8])
    parser.add_argument("--instanton-maxiter", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
