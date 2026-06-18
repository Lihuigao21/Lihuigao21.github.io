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

from toymodel.DVRmethods import CartesianDVRKubo, RadialDVRKubo
from toymodel.methods import (
    CMDRadialDynamics,
    RingPolymerRadialDynamics,
    estimate_radial_cmd_pmf,
)
from toymodel.model import ChampagneBottleMorse2D
from toymodel.utils.constant import fs_to_au, kB

HARTREE_TO_WAVENUMBER = 219474.6313705


def fermi_window(time, t_half, tau):
    return 1.0 / (1.0 + np.exp((np.abs(time) - t_half) / tau))


def spectrum_from_correlation(time, corr, omega_cm, window=None):
    time = np.asarray(time, dtype=float)
    corr = np.asarray(corr)
    omega_cm = np.asarray(omega_cm, dtype=float)
    if window is None:
        window = np.ones_like(time)
    omega_au = omega_cm / HARTREE_TO_WAVENUMBER
    spectrum = np.empty_like(omega_cm, dtype=float)
    for iw, omega in enumerate(omega_au):
        val = np.trapezoid(corr * window * np.exp(-1j * omega * time), time)
        spectrum[iw] = 2.0 * np.real(val)
    return spectrum


def peak_position(
    omega_cm,
    spectrum,
    *,
    min_cm: float | None = None,
    max_cm: float | None = None,
    interpolate: bool = True,
):
    omega_cm = np.asarray(omega_cm, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    mask = np.isfinite(omega_cm) & np.isfinite(spectrum)
    if min_cm is not None:
        mask &= omega_cm >= float(min_cm)
    if max_cm is not None:
        mask &= omega_cm <= float(max_cm)
    if not np.any(mask):
        raise ValueError("no finite spectrum points in the requested peak window.")
    candidates = np.flatnonzero(mask)
    idx = int(candidates[np.nanargmax(spectrum[candidates])])
    peak_omega = float(omega_cm[idx])
    peak_intensity = float(spectrum[idx])
    if interpolate and 0 < idx < omega_cm.size - 1:
        x0, x1, x2 = omega_cm[idx - 1 : idx + 2]
        y0, y1, y2 = spectrum[idx - 1 : idx + 2]
        denom = y0 - 2.0 * y1 + y2
        if np.isfinite(denom) and abs(denom) > 1.0e-30 and denom < 0.0:
            dx = 0.5 * (y0 - y2) / denom
            if abs(dx) <= 1.0:
                step = 0.5 * (x2 - x0)
                peak_omega = float(x1 + dx * step)
                peak_intensity = float(y1 - 0.25 * (y0 - y2) * dx)
    return peak_omega, peak_intensity


def sample_radius_from_potential(model, beta, rgrid, rng):
    potential = model.radial_potential(rgrid)
    shifted = potential - np.min(potential)
    weights = rgrid ** (model.ndim - 1) * np.exp(-beta * shifted)
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return float(np.interp(rng.random(), cdf, rgrid))


def sample_direction(ndim, rng):
    vec = rng.normal(size=ndim)
    norm = np.linalg.norm(vec)
    while norm < 1.0e-14:
        vec = rng.normal(size=ndim)
        norm = np.linalg.norm(vec)
    return vec / norm


def model_force(model, q):
    return -np.asarray(model.dV(q), dtype=float)[:, 0, 0]


def classical_velocity_autocorrelation(
    model,
    *,
    temperature,
    ntraj,
    dt,
    nstep,
    seed,
):
    rng = np.random.default_rng(seed)
    beta = 1.0 / (kB * temperature)
    mass = np.full(model.ndim, model.mass, dtype=float)
    sigma_p = np.sqrt(mass / beta)
    rgrid = np.linspace(0.3, 5.0, 4000)
    corr = np.zeros(nstep + 1, dtype=float)
    energy_drift = []

    for _ in range(ntraj):
        r = sample_radius_from_potential(model, beta, rgrid, rng)
        q = r * sample_direction(model.ndim, rng)
        p = rng.normal(0.0, sigma_p)
        r0 = np.linalg.norm(q)
        vr0 = float(np.dot(q, p / mass) / r0)
        e0 = np.sum(p * p / (2.0 * mass)) + float(model.V(q)[0, 0])

        for istep in range(nstep + 1):
            r = np.linalg.norm(q)
            vr = float(np.dot(q, p / mass) / r) if r > 1.0e-14 else 0.0
            corr[istep] += vr * vr0
            if istep == nstep:
                break
            f = model_force(model, q)
            p_half = p + 0.5 * dt * f
            q = q + dt * p_half / mass
            f_new = model_force(model, q)
            p = p_half + 0.5 * dt * f_new

        e1 = np.sum(p * p / (2.0 * mass)) + float(model.V(q)[0, 0])
        energy_drift.append(e1 - e0)

    corr /= ntraj
    return {
        "time": np.arange(nstep + 1, dtype=float) * dt,
        "correlation": corr,
        "energy_drift": np.asarray(energy_drift, dtype=float),
    }


def write_summary(path, rows):
    fields = [
        "temperature_K",
        "method",
        "peak_cm",
        "peak_intensity",
        "peak_min_cm",
        "peak_max_cm",
        "corr0",
        "acceptance_mean",
        "energy_drift_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def plot_spectra(path, spectra_by_temp):
    import matplotlib.pyplot as plt

    ntemp = len(spectra_by_temp)
    fig, axes = plt.subplots(1, ntemp, figsize=(5.0 * ntemp, 3.5), squeeze=False)
    for ax, (temperature, spectra) in zip(axes[0], spectra_by_temp.items()):
        for label, (omega, spectrum) in spectra.items():
            max_abs = np.max(np.abs(spectrum))
            y = spectrum / max_abs if max_abs > 0.0 else spectrum
            ax.plot(omega, y, label=label, lw=1.5)
        ax.set_title(f"T={temperature:g} K")
        ax.set_xlabel("frequency / cm$^{-1}$")
        ax.set_ylabel("normalized intensity")
        ax.set_xlim(0.0, 5000.0)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cmd_mean_force(path, pmf_by_temp):
    import matplotlib.pyplot as plt

    ntemp = len(pmf_by_temp)
    fig, axes = plt.subplots(2, ntemp, figsize=(5.0 * ntemp, 6.0), squeeze=False)
    for col, (temperature, pmf) in enumerate(pmf_by_temp.items()):
        r = pmf.rgrid
        axes[0, col].plot(r, pmf.mean_force, marker="o", lw=1.4)
        axes[0, col].fill_between(
            r,
            pmf.mean_force - pmf.force_stderr,
            pmf.mean_force + pmf.force_stderr,
            alpha=0.2,
        )
        axes[0, col].set_title(f"T={temperature:g} K")
        axes[0, col].set_xlabel("centroid radius / bohr")
        axes[0, col].set_ylabel("CMD mean force / a.u.")

        radial_weight = r ** (pmf.ndim - 1) * np.exp(-pmf.beta * (pmf.pmf - np.min(pmf.pmf)))
        if np.max(radial_weight) > 0.0:
            radial_weight = radial_weight / np.max(radial_weight)
        axes[1, col].plot(r, pmf.pmf, marker="o", lw=1.4, label="PMF")
        axes2 = axes[1, col].twinx()
        axes2.plot(r, radial_weight, color="tab:red", lw=1.2, label="radial weight")
        axes[1, col].set_xlabel("centroid radius / bohr")
        axes[1, col].set_ylabel("PMF / Hartree")
        axes2.set_ylabel("normalized radial weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = ChampagneBottleMorse2D()
    omega_cm = np.linspace(0.0, 5000.0, args.nomega)
    time = np.arange(args.nstep + 1, dtype=float) * args.dt
    window = fermi_window(
        time,
        args.window_half_fs * fs_to_au,
        args.window_tau_fs * fs_to_au,
    )

    rows = []
    spectra_by_temp = {}
    pmf_by_temp = {}
    for itemp, temperature in enumerate(args.temperatures):
        spectra_by_temp[temperature] = {}

        if args.exact_dvr_type == "radial":
            exact = RadialDVRKubo(
                model=model,
                temperature=temperature,
                nr=args.radial_nr,
                rmax=args.radial_rmax,
                angular_max=args.angular_max,
                tgrid=time,
            )
        else:
            exact = CartesianDVRKubo(
                model=model,
                temperature=temperature,
                ndvr=np.array([args.exact_ndvr, args.exact_ndvr]),
                xbound=np.array([args.xbound, args.xbound]),
                tgrid=time,
                kinetic_operator=args.cartesian_kinetic,
            )
        exact_corr = exact.radial_velocity_autocorrelation()
        if args.exact_spectrum_mode == "lines":
            exact_spectrum = exact.radial_velocity_spectrum_from_lines(
                omega_cm=omega_cm,
                broadening_cm=args.exact_broadening_cm,
            )[1]
        else:
            exact_spectrum = exact.spectrum_from_correlation(
                exact_corr,
                omega_cm=omega_cm,
                window=window,
            )[1]
        exact_peak, exact_intensity = peak_position(
            omega_cm,
            exact_spectrum,
            min_cm=args.peak_min_cm,
            max_cm=args.peak_max_cm,
        )
        if hasattr(exact, "evals") and exact.evals is not None:
            exact_evals = exact.evals
        elif hasattr(exact, "blocks"):
            exact_evals = np.concatenate([block.evals for block in exact.blocks])
        else:
            exact_evals = np.array([], dtype=float)
        np.savez_compressed(
            outdir / f"exact_T{int(temperature)}.npz",
            time=time,
            correlation=exact_corr,
            omega_cm=omega_cm,
            spectrum=exact_spectrum,
            evals=exact_evals,
            partition=getattr(exact, "partition", np.nan),
            energy_shift=getattr(exact, "energy_shift", np.nan),
        )
        rows.append(
            {
                "temperature_K": temperature,
                "method": "exact_dvr",
                "peak_cm": exact_peak,
                "peak_intensity": exact_intensity,
                "peak_min_cm": args.peak_min_cm,
                "peak_max_cm": args.peak_max_cm,
                "corr0": float(np.real(exact_corr[0])),
            }
        )
        spectra_by_temp[temperature]["DVR"] = (omega_cm, exact_spectrum)

        nbeads = args.cmd_nbeads_low if temperature <= 300.0 else args.cmd_nbeads_high
        pmf = estimate_radial_cmd_pmf(
            model=model,
            rgrid=np.linspace(args.rmin, args.rmax, args.pmf_points),
            temperature=temperature,
            nbeads=nbeads,
            nsteps=args.pimc_steps,
            burnin=0.25,
            step_size=args.pimc_step_size,
            sample_stride=args.pimc_stride,
            seed=args.seed + 1000 * itemp,
            initial_path_mode=args.pimc_initial_path,
        )
        pmf_by_temp[temperature] = pmf
        cmd = CMDRadialDynamics(pmf, seed=args.seed + 2000 * itemp)
        cmd_corr_data = cmd.radial_velocity_autocorrelation(
            ntraj=args.cmd_ntraj,
            dt=args.dt,
            nstep=args.nstep,
        )
        cmd_spectrum = spectrum_from_correlation(
            cmd_corr_data["time"],
            cmd_corr_data["correlation"],
            omega_cm,
            window=window,
        )
        cmd_peak, cmd_intensity = peak_position(
            omega_cm,
            cmd_spectrum,
            min_cm=args.peak_min_cm,
            max_cm=args.peak_max_cm,
        )
        np.savez_compressed(
            outdir / f"cmd_T{int(temperature)}.npz",
            rgrid=pmf.rgrid,
            mean_force=pmf.mean_force,
            force_stderr=pmf.force_stderr,
            pmf=pmf.pmf,
            acceptance_rate=pmf.acceptance_rate,
            time=cmd_corr_data["time"],
            correlation=cmd_corr_data["correlation"],
            energy_drift=cmd_corr_data["energy_drift"],
            omega_cm=omega_cm,
            spectrum=cmd_spectrum,
        )
        rows.append(
            {
                "temperature_K": temperature,
                "method": "cmd_ca",
                "peak_cm": cmd_peak,
                "peak_intensity": cmd_intensity,
                "peak_min_cm": args.peak_min_cm,
                "peak_max_cm": args.peak_max_cm,
                "corr0": float(cmd_corr_data["correlation"][0]),
                "acceptance_mean": float(np.mean(pmf.acceptance_rate)),
                "energy_drift_rms": float(
                    np.sqrt(np.mean(cmd_corr_data["energy_drift"] ** 2))
                ),
            }
        )
        spectra_by_temp[temperature]["CMD"] = (omega_cm, cmd_spectrum)

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
        classical_peak, classical_intensity = peak_position(
            omega_cm,
            classical_spectrum,
            min_cm=args.peak_min_cm,
            max_cm=args.peak_max_cm,
        )
        np.savez_compressed(
            outdir / f"classical_T{int(temperature)}.npz",
            time=classical["time"],
            correlation=classical["correlation"],
            energy_drift=classical["energy_drift"],
            omega_cm=omega_cm,
            spectrum=classical_spectrum,
        )
        rows.append(
            {
                "temperature_K": temperature,
                "method": "classical_eh",
                "peak_cm": classical_peak,
                "peak_intensity": classical_intensity,
                "peak_min_cm": args.peak_min_cm,
                "peak_max_cm": args.peak_max_cm,
                "corr0": float(classical["correlation"][0]),
                "energy_drift_rms": float(
                    np.sqrt(np.mean(classical["energy_drift"] ** 2))
                ),
            }
        )
        spectra_by_temp[temperature]["Classical"] = (omega_cm, classical_spectrum)

        rp_beads = args.rpmd_nbeads_low if temperature <= 300.0 else args.rpmd_nbeads_high
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
            pimc_burnin=0.25,
            pimc_step_size=args.rpmd_pimc_step_size,
            pimc_stride=args.rpmd_pimc_stride,
        )
        rpmd_spectrum = spectrum_from_correlation(
            rpmd["time"],
            rpmd["correlation"],
            omega_cm,
            window=window,
        )
        rpmd_peak, rpmd_intensity = peak_position(
            omega_cm,
            rpmd_spectrum,
            min_cm=args.peak_min_cm,
            max_cm=args.peak_max_cm,
        )
        np.savez_compressed(
            outdir / f"rpmd_T{int(temperature)}.npz",
            time=rpmd["time"],
            correlation=rpmd["correlation"],
            energy_drift=rpmd["energy_drift"],
            acceptance_rate=rpmd["acceptance_rate"],
            omega_cm=omega_cm,
            spectrum=rpmd_spectrum,
            nbeads=rp_beads,
        )
        rows.append(
            {
                "temperature_K": temperature,
                "method": "rpmd",
                "peak_cm": rpmd_peak,
                "peak_intensity": rpmd_intensity,
                "peak_min_cm": args.peak_min_cm,
                "peak_max_cm": args.peak_max_cm,
                "corr0": float(rpmd["correlation"][0]),
                "acceptance_mean": float(rpmd["acceptance_rate"]),
                "energy_drift_rms": float(np.sqrt(np.mean(rpmd["energy_drift"] ** 2))),
            }
        )
        spectra_by_temp[temperature]["RPMD"] = (omega_cm, rpmd_spectrum)

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
            pimc_burnin=0.25,
            pimc_step_size=args.rpmd_pimc_step_size,
            pimc_stride=args.rpmd_pimc_stride,
        )
        trpmd_spectrum = spectrum_from_correlation(
            trpmd["time"],
            trpmd["correlation"],
            omega_cm,
            window=window,
        )
        trpmd_peak, trpmd_intensity = peak_position(
            omega_cm,
            trpmd_spectrum,
            min_cm=args.peak_min_cm,
            max_cm=args.peak_max_cm,
        )
        np.savez_compressed(
            outdir / f"trpmd_T{int(temperature)}.npz",
            time=trpmd["time"],
            correlation=trpmd["correlation"],
            energy_drift=trpmd["energy_drift"],
            acceptance_rate=trpmd["acceptance_rate"],
            omega_cm=omega_cm,
            spectrum=trpmd_spectrum,
            nbeads=rp_beads,
            gamma=args.trpmd_gamma,
        )
        rows.append(
            {
                "temperature_K": temperature,
                "method": "trpmd",
                "peak_cm": trpmd_peak,
                "peak_intensity": trpmd_intensity,
                "peak_min_cm": args.peak_min_cm,
                "peak_max_cm": args.peak_max_cm,
                "corr0": float(trpmd["correlation"][0]),
                "acceptance_mean": float(trpmd["acceptance_rate"]),
                "energy_drift_rms": float(np.sqrt(np.mean(trpmd["energy_drift"] ** 2))),
            }
        )
        spectra_by_temp[temperature]["TRPMD"] = (omega_cm, trpmd_spectrum)

    write_summary(outdir / "summary.csv", rows)
    plot_spectra(outdir / "spectra.png", spectra_by_temp)
    plot_cmd_mean_force(outdir / "cmd_mean_force.png", pmf_by_temp)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="output/cmd_curvature/smoke")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[200.0, 800.0])
    parser.add_argument("--exact-ndvr", type=int, default=26)
    parser.add_argument("--xbound", type=float, default=5.0)
    parser.add_argument("--exact-dvr-type", choices=["radial", "cartesian"], default="radial")
    parser.add_argument(
        "--cartesian-kinetic",
        choices=["particle_in_box", "sinc", "fft_periodic"],
        default="particle_in_box",
    )
    parser.add_argument("--radial-nr", type=int, default=192)
    parser.add_argument("--radial-rmax", type=float, default=6.0)
    parser.add_argument("--angular-max", type=int, default=24)
    parser.add_argument("--nstep", type=int, default=500)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--nomega", type=int, default=700)
    parser.add_argument("--exact-spectrum-mode", choices=["lines", "correlation"], default="lines")
    parser.add_argument("--exact-broadening-cm", type=float, default=75.0)
    parser.add_argument("--peak-min-cm", type=float, default=2500.0)
    parser.add_argument("--peak-max-cm", type=float, default=4500.0)
    parser.add_argument("--window-half-fs", type=float, default=20.0)
    parser.add_argument("--window-tau-fs", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=0.9)
    parser.add_argument("--rmax", type=float, default=3.6)
    parser.add_argument("--pmf-points", type=int, default=19)
    parser.add_argument("--cmd-nbeads-low", type=int, default=16)
    parser.add_argument("--cmd-nbeads-high", type=int, default=8)
    parser.add_argument("--pimc-steps", type=int, default=800)
    parser.add_argument("--pimc-stride", type=int, default=8)
    parser.add_argument("--pimc-step-size", type=float, default=0.08)
    parser.add_argument(
        "--pimc-initial-path",
        choices=["collapsed", "curved"],
        default="curved",
    )
    parser.add_argument("--cmd-ntraj", type=int, default=80)
    parser.add_argument("--classical-ntraj", type=int, default=120)
    parser.add_argument("--rpmd-nbeads-low", type=int, default=16)
    parser.add_argument("--rpmd-nbeads-high", type=int, default=8)
    parser.add_argument("--rpmd-pimc-steps", type=int, default=1600)
    parser.add_argument("--rpmd-pimc-stride", type=int, default=8)
    parser.add_argument("--rpmd-pimc-step-size", type=float, default=0.08)
    parser.add_argument("--rpmd-ntraj", type=int, default=60)
    parser.add_argument("--trpmd-gamma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


if __name__ == "__main__":
    for row in run(parse_args()):
        print(row)
