from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


H_EV_FS = 4.135667696


def numeric_rows(path: Path, min_cols: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values: list[float] = []
        for token in stripped.split():
            try:
                values.append(float(token))
            except ValueError:
                pass
        if len(values) >= min_cols:
            rows.append(values[:min_cols])
    if not rows:
        raise RuntimeError(f"no numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def fft_response(signal: np.ndarray, field: np.ndarray | None, dt_fs: float, damping_fs: float, pad: int):
    n = len(signal)
    t = np.arange(n) * dt_fs
    y = signal - np.mean(signal[: min(50, n)])
    if n > 2:
        y -= np.polyval(np.polyfit(t, y, 1), t)
    window = np.exp(-t / damping_fs)
    nfft = 1 << int(math.ceil(math.log2(max(n * pad, 16))))
    energy = np.fft.rfftfreq(nfft, dt_fs) * H_EV_FS
    dipole_fft = np.fft.rfft(y * window, nfft)
    if field is None:
        return energy, dipole_fft, None
    ef = field[:n] - np.mean(field[: min(50, n)])
    field_fft = np.fft.rfft(ef * window, nfft)
    return energy, dipole_fft, field_fft


def select_peaks(energy: np.ndarray, strength: np.ndarray, threshold: float = 0.04):
    mask = (energy >= 4.0) & (energy <= 18.5)
    e = energy[mask]
    y = np.asarray(strength[mask], dtype=float)
    y[~np.isfinite(y)] = 0.0
    if y.max() > 0:
        y = y / y.max()
    loc = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    loc = loc[y[loc] >= threshold]
    loc = loc[np.argsort(y[loc])][::-1]
    peaks: list[dict[str, float]] = []
    for idx in loc:
        ev = float(e[idx])
        if all(abs(ev - old["energy_ev"]) > 0.20 for old in peaks):
            peaks.append({"energy_ev": ev, "relative_height": float(y[idx])})
        if len(peaks) >= 8:
            break
    return sorted(peaks, key=lambda item: item["energy_ev"])


def analyze(root: Path, output: Path, dt_fs: float, damping_fs: float, pad: int):
    cases = [
        ("rt_x_a002", 1, "tab:red"),
        ("rt_y_a002", 2, "tab:green"),
        ("rt_z_a002", 3, "tab:blue"),
        ("rt_y_a001", 2, "limegreen"),
        ("rt_z_a001", 3, "cornflowerblue"),
    ]
    nofield = numeric_rows(root / "rt_nofield" / "OUT.rt_nofield" / "dipole_s1.txt", 4)
    summary: dict[str, object] = {"spectra": {}}

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 10.0), sharex=False)
    iso_energy = None
    iso_strength = None

    for case, component, color in cases:
        dip = numeric_rows(root / case / f"OUT.{case}" / "dipole_s1.txt", 4)
        efield = numeric_rows(root / case / f"OUT.{case}" / "efield_0.txt", 2)
        n = min(len(nofield), len(dip), len(efield))
        time = (dip[:n, 0] - dip[0, 0]) * dt_fs
        induced = dip[:n, component] - nofield[:n, component]
        field = efield[:n, 1]

        energy, dipole_fft, field_fft = fft_response(induced, field, dt_fs, damping_fs, pad)
        direct = np.abs(dipole_fft)
        denom = np.where(np.abs(field_fft) < 1.0e-12, np.nan, field_fft)
        alpha = dipole_fft / denom
        strength = energy * np.imag(alpha)
        band = (energy > 4.0) & (energy < 18.5)
        if np.nanmax(-strength[band]) > np.nanmax(strength[band]):
            strength = -strength
        strength = np.where(np.isfinite(strength), strength, 0.0)
        strength = np.maximum(strength, 0.0)

        mask = (energy >= 4.0) & (energy <= 18.5)
        axes[0].plot(time, induced, lw=0.55, color=color, label=case)
        axes[1].plot(energy[mask], strength[mask] / strength[mask].max(), lw=0.9, color=color, label=case)
        axes[2].plot(energy[mask], direct[mask] / direct[mask].max(), lw=0.75, color=color, label=case)

        if case.endswith("a002"):
            iso_energy = energy.copy()
            iso_strength = strength.copy() if iso_strength is None else iso_strength + strength

        summary["spectra"][case] = {
            "component": component,
            "n_points": int(n),
            "time_window_fs": float((n - 1) * dt_fs),
            "field_peak": float(np.max(np.abs(field))),
            "alpha_peaks": select_peaks(energy, strength),
            "direct_dipole_fft_peaks": select_peaks(energy, direct),
        }

    if iso_energy is not None and iso_strength is not None:
        mask = (iso_energy >= 4.0) & (iso_energy <= 18.5)
        axes[1].plot(iso_energy[mask], iso_strength[mask] / iso_strength[mask].max(), color="k", lw=1.2, label="iso a002")
        summary["isotropic_a002_alpha_peaks"] = select_peaks(iso_energy, iso_strength)

    for ev in (7.447, 9.672):
        axes[1].axvline(ev, color="0.25", ls="--", lw=0.9, alpha=0.65)
        axes[2].axvline(ev, color="0.25", ls="--", lw=0.9, alpha=0.65)
    for ev in (7.553, 9.740):
        axes[1].axvline(ev, color="tab:purple", ls=":", lw=0.9, alpha=0.75)
        axes[2].axvline(ev, color="tab:purple", ls=":", lw=0.9, alpha=0.75)

    axes[0].set_title("H2O LDA RT-TDDFT induced dipoles")
    axes[0].set_xlabel("time (fs)")
    axes[0].set_ylabel("induced dipole (a.u.)")
    axes[1].set_title("Absorption proxy: normalized positive omega Im alpha")
    axes[1].set_xlabel("energy (eV)")
    axes[1].set_ylabel("norm. omega Im alpha")
    axes[2].set_title("Auxiliary direct dipole FFT")
    axes[2].set_xlabel("energy (eV)")
    axes[2].set_ylabel("norm. |FFT(dipole)|")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze ABACUS H2O RT-TDDFT dipole and electric-field outputs.")
    parser.add_argument("root", type=Path, help="Directory containing rt_nofield and rt_* case folders.")
    parser.add_argument("--output", type=Path, default=Path("h2o_rt_tddft_absorption"), help="Output prefix.")
    parser.add_argument("--dt-fs", type=float, default=0.005, help="RT-TDDFT timestep in fs.")
    parser.add_argument("--damping-fs", type=float, default=25.0, help="Exponential damping time for FFT.")
    parser.add_argument("--pad", type=int, default=8, help="FFT zero-padding multiplier.")
    args = parser.parse_args()
    analyze(args.root, args.output, args.dt_fs, args.damping_fs, args.pad)


if __name__ == "__main__":
    main()
