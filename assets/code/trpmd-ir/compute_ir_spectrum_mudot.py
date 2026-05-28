#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CM_PER_CYCLE_PER_PS = 33.35640951981521


def read_dipoles(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row["mux_D"]), float(row["muy_D"]), float(row["muz_D"])])
    if not rows:
        raise ValueError(f"No dipole rows in {path}")
    return np.asarray(rows, dtype=float)


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    nframes = values.shape[0]
    nfft = 1 << (2 * nframes - 1).bit_length()
    corr = np.zeros(nframes, dtype=float)
    for icomp in range(values.shape[1]):
        x = values[:, icomp] - values[:, icomp].mean()
        fx = np.fft.rfft(x, n=nfft)
        corr += np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:nframes]
    corr /= np.arange(nframes, 0, -1)
    return corr


def central_difference(values: np.ndarray, dt_ps: float) -> np.ndarray:
    return (values[2:] - values[:-2]) / (2.0 * dt_ps)


def make_window(kind: str, nframes: int, dt_ps: float, width_ps: float) -> np.ndarray:
    if kind == "none":
        return np.ones(nframes, dtype=float)
    x = np.arange(nframes, dtype=float) / float(nframes - 1)
    time_ps = np.arange(nframes, dtype=float) * dt_ps
    if kind == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * x))
    if kind == "cosine":
        return np.cos(0.5 * np.pi * x)
    if kind == "gaussian":
        return np.exp(-0.5 * (time_ps / width_ps) ** 2)
    if kind == "exponential":
        return np.exp(-time_ps / width_ps)
    raise ValueError(f"Unknown window kind: {kind}")


def spectrum_from_correlation(corr: np.ndarray, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
    freq_cycle_per_ps = np.fft.rfftfreq(len(corr), d=dt_ps)
    wavenumber = freq_cycle_per_ps * CM_PER_CYCLE_PER_PS
    spectrum = np.real(np.fft.rfft(corr)) * dt_ps
    return wavenumber, np.maximum(spectrum, 0.0)


def normalize_intensity(
    wavenumber: np.ndarray, intensity: np.ndarray, max_cm: float
) -> np.ndarray:
    keep = wavenumber <= max_cm
    if np.any(keep) and np.max(intensity[keep]) > 0:
        return intensity / np.max(intensity[keep])
    return intensity


def write_spectrum(path: Path, wavenumber: np.ndarray, intensity: np.ndarray, max_cm: float) -> None:
    keep = wavenumber <= max_cm
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavenumber_cm-1", "intensity"])
        for wn, inten in zip(wavenumber[keep], intensity[keep]):
            writer.writerow([wn, inten])


def write_blocks(
    path: Path,
    wavenumber: np.ndarray,
    block_spectra: np.ndarray,
    max_cm: float,
) -> None:
    keep = wavenumber <= max_cm
    mean = block_spectra.mean(axis=0)
    sem = block_spectra.std(axis=0, ddof=1) / np.sqrt(block_spectra.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavenumber_cm-1", "block_mean", "block_sem"])
        for wn, avg, err in zip(wavenumber[keep], mean[keep], sem[keep]):
            writer.writerow([wn, avg, err])


def plot_spectrum(
    path: Path,
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    max_cm: float,
    block_spectra: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    keep = wavenumber <= max_cm
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    if block_spectra is not None and block_spectra.shape[0] > 1:
        mean = block_spectra.mean(axis=0)
        sem = block_spectra.std(axis=0, ddof=1) / np.sqrt(block_spectra.shape[0])
        ax.fill_between(
            wavenumber[keep],
            np.maximum(mean[keep] - sem[keep], 0.0),
            mean[keep] + sem[keep],
            color="tab:green",
            alpha=0.22,
            linewidth=0,
            label=f"{block_spectra.shape[0]}-block SEM",
        )
    ax.plot(wavenumber[keep], intensity[keep], color="tab:green", linewidth=1.8, label="mu-dot ACF")
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel("Intensity / arb. units")
    ax.set_xlim(0, max_cm)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an RPMD/MD IR spectrum from the dipole-derivative "
            "autocorrelation function."
        )
    )
    parser.add_argument("dipoles", nargs="+", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--dt-fs", type=float, required=True)
    parser.add_argument(
        "--window",
        choices=("none", "hann", "cosine", "gaussian", "exponential"),
        default="gaussian",
    )
    parser.add_argument(
        "--window-width-ps",
        type=float,
        default=2.0,
        help="Width for gaussian/exponential windows. Ignored by hann/cosine.",
    )
    parser.add_argument("--max-cm", type=float, default=4000.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--acf-output", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--block-output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Discard this many initial dipole frames before differentiating.",
    )
    parser.add_argument(
        "--skip-ps",
        type=float,
        default=0.0,
        help="Discard this many initial picoseconds before differentiating.",
    )
    args = parser.parse_args()

    dt_ps = args.dt_fs * 1.0e-3
    skip_frames = max(args.skip_frames, int(np.ceil(args.skip_ps / dt_ps)))
    correlations = []
    for path in args.dipoles:
        values = read_dipoles(path)
        if skip_frames:
            if skip_frames >= len(values) - 2:
                raise ValueError(
                    f"Skip removes too much data from {path}: "
                    f"{skip_frames} of {len(values)} frames"
                )
            values = values[skip_frames:]
        mudot = central_difference(values, dt_ps)
        correlations.append(autocorrelation_fft(mudot))

    nmin = min(len(corr) for corr in correlations)
    corr_array = np.vstack([corr[:nmin] for corr in correlations])
    corr = corr_array.mean(axis=0)
    window = make_window(args.window, nmin, dt_ps, args.window_width_ps)

    if args.acf_output is not None:
        args.acf_output.parent.mkdir(parents=True, exist_ok=True)
        time_ps = np.arange(nmin) * dt_ps
        np.savetxt(
            args.acf_output,
            np.column_stack([time_ps, corr]),
            delimiter=",",
            header="time_ps,corr_D2_per_ps2",
            comments="",
        )

    wavenumber, intensity = spectrum_from_correlation(corr * window, dt_ps)
    if not args.no_normalize:
        intensity = normalize_intensity(wavenumber, intensity, args.max_cm)
    write_spectrum(args.output_csv, wavenumber, intensity, args.max_cm)

    block_spectra = None
    if args.block_size > 0:
        blocks = []
        for start in range(0, len(correlations), args.block_size):
            block = corr_array[start : start + args.block_size]
            if len(block) == args.block_size:
                _, block_intensity = spectrum_from_correlation(block.mean(axis=0) * window, dt_ps)
                if not args.no_normalize:
                    block_intensity = normalize_intensity(
                        wavenumber, block_intensity, args.max_cm
                    )
                blocks.append(block_intensity)
        if len(blocks) > 1:
            block_spectra = np.vstack(blocks)
            if args.block_output is not None:
                write_blocks(args.block_output, wavenumber, block_spectra, args.max_cm)

    if args.plot is not None:
        plot_spectrum(args.plot, wavenumber, intensity, args.max_cm, block_spectra)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
