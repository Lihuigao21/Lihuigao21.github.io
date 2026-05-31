from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    os.environ.get(
        "CONTROLLED_NVT_ROOT",
        "/public/home/gaolihui/codex/controlled_nvt_2000_smassm1_20260530",
    )
)
OUT = ROOT / "analysis"
PLOTS = OUT / "plots"
CSV_DIR = OUT / "csv"
PLOTS.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = {
    "MAPbI3": {"framework": {"Pb", "I"}, "projected": ["Pb", "I"]},
    "CsPbI3": {"framework": {"Pb", "I"}, "projected": ["Pb", "I", "Cs"]},
    "TiO2": {"framework": {"Ti", "O"}, "projected": ["Ti", "O"]},
}

MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Ti": 47.867,
    "Cs": 132.905,
    "I": 126.904,
    "Pb": 207.2,
}


def read_xdatcar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float
    )
    lattice *= scale
    elems = lines[5].split()
    counts = [int(float(x)) for x in lines[6].split()]
    symbols = np.array([elem for elem, count in zip(elems, counts) for _ in range(count)])
    n_atoms = len(symbols)
    frames: list[list[list[float]]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("direct configuration"):
            frames.append(
                [
                    [float(x) for x in lines[i + 1 + j].split()[:3]]
                    for j in range(n_atoms)
                ]
            )
            i += n_atoms + 1
        else:
            i += 1
    return lattice, symbols, np.asarray(frames, dtype=float)


def parse_potim_fs(path: Path) -> float:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().upper() == "POTIM":
            return float(value.split("#")[0].split("!")[0].split()[0])
    return 0.5


def unwrap(frames: np.ndarray) -> np.ndarray:
    steps = np.diff(frames, axis=0)
    steps -= np.round(steps)
    direct = np.empty_like(frames)
    direct[0] = frames[0]
    direct[1:] = frames[0] + np.cumsum(steps, axis=0)
    return direct


def remove_com(cart: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    masses = np.array([MASS.get(str(sym), 1.0) for sym in symbols], dtype=float)
    com = np.einsum("tnc,n->tc", cart, masses) / masses.sum()
    return cart - com[:, None, :]


def gaussian_smooth(y: np.ndarray, sigma_points: float = 1.0) -> np.ndarray:
    if len(y) < 5:
        return y
    radius = max(2, int(round(4 * sigma_points)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_points) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def vdos(
    rel_cart: np.ndarray,
    symbols: np.ndarray,
    idx: np.ndarray,
    dt_fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocities = np.diff(rel_cart, axis=0) / dt_fs
    v = velocities[:, idx, :].reshape(velocities.shape[0], -1)
    v -= v.mean(axis=0, keepdims=True)
    window = np.hanning(v.shape[0])[:, None]
    spec = np.fft.rfft(v * window, axis=0)
    power = np.sum(np.abs(spec) ** 2, axis=1)
    freq = np.fft.rfftfreq(v.shape[0], d=dt_fs * 1e-15) / 1e12
    if len(power):
        power[0] = 0.0
    power = gaussian_smooth(power, sigma_points=1.0)
    total = np.trapz(power, freq)
    if total > 0:
        power = power / total
    return freq, power


def spectral_metrics(freq: np.ndarray, dos: np.ndarray) -> dict[str, float]:
    total = float(np.trapz(dos, freq))
    if total <= 0:
        return {key: math.nan for key in ["low2", "low5", "low10", "centroid", "peak", "q50", "q90"]}
    cdf = np.cumsum((dos[:-1] + dos[1:]) * np.diff(freq) / 2)
    cdf = np.concatenate([[0.0], cdf]) / total
    def frac(limit: float) -> float:
        mask = freq <= limit
        return float(np.trapz(dos[mask], freq[mask]) / total) if np.any(mask) else 0.0

    return {
        "low2": frac(2.0),
        "low5": frac(5.0),
        "low10": frac(10.0),
        "centroid": float(np.trapz(freq * dos, freq) / total),
        "peak": float(freq[int(np.argmax(dos))]),
        "q50": float(np.interp(0.50, cdf, freq)),
        "q90": float(np.interp(0.90, cdf, freq)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, ndigits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{ndigits}f}"


def main() -> None:
    skip_frac = 0.2
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    projected_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    projected_rows: list[dict[str, object]] = []

    for system, spec in SYSTEMS.items():
        run_dir = ROOT / system
        lattice, symbols, frames = read_xdatcar(run_dir / "XDATCAR")
        dt_fs = parse_potim_fs(run_dir / "INCAR")
        skip = int(round(skip_frac * len(frames)))
        rel = remove_com(unwrap(frames) @ lattice, symbols)[skip:]

        framework_idx = np.array(
            [i for i, sym in enumerate(symbols) if str(sym) in spec["framework"]],
            dtype=int,
        )
        freq, dos = vdos(rel, symbols, framework_idx, dt_fs)
        curves[system] = (freq, dos)
        metrics = spectral_metrics(freq, dos)
        rows.append(
            {
                "system": system,
                "group": "framework",
                "frames_used": int(len(rel)),
                "dt_fs": dt_fs,
                "freq_resolution_THz": float(freq[1] - freq[0]) if len(freq) > 1 else math.nan,
                "low_2THz_frac": metrics["low2"],
                "low_5THz_frac": metrics["low5"],
                "low_10THz_frac": metrics["low10"],
                "centroid_THz": metrics["centroid"],
                "peak_THz": metrics["peak"],
                "median_THz": metrics["q50"],
                "q90_THz": metrics["q90"],
            }
        )

        for f, d in zip(freq, dos):
            if f <= 80:
                curve_rows.append({"system": system, "freq_THz": float(f), "dos": float(d)})

        for elem in spec["projected"]:
            idx = np.where(symbols == elem)[0]
            pfreq, pdos = vdos(rel, symbols, idx, dt_fs)
            projected_curves.setdefault(system, {})[elem] = (pfreq, pdos)
            pmetrics = spectral_metrics(pfreq, pdos)
            projected_rows.append(
                {
                    "system": system,
                    "group": elem,
                    "n_atoms": int(len(idx)),
                    "low_5THz_frac": pmetrics["low5"],
                    "centroid_THz": pmetrics["centroid"],
                    "peak_THz": pmetrics["peak"],
                }
            )

    tio2 = next(row for row in rows if row["system"] == "TiO2")
    for row in rows:
        row["low_5THz_vs_TiO2"] = float(row["low_5THz_frac"]) / float(tio2["low_5THz_frac"])
        row["centroid_vs_TiO2"] = float(row["centroid_THz"]) / float(tio2["centroid_THz"])

    write_csv(OUT / "md_phonon_spectrum_metrics.csv", rows)
    write_csv(CSV_DIR / "md_phonon_spectrum_curves.csv", curve_rows)
    write_csv(CSV_DIR / "md_phonon_projected_metrics.csv", projected_rows)

    colors = {"MAPbI3": "#b94141", "CsPbI3": "#3267b1", "TiO2": "#202020"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax, ax2 = axes
    for system, (freq, dos) in curves.items():
        mask = freq <= 50
        ax.plot(freq[mask], dos[mask], label=system, linewidth=2.1, color=colors[system])
        low = freq <= 20
        cumulative = np.cumsum((dos[:-1] + dos[1:]) * np.diff(freq) / 2)
        cumulative = np.concatenate([[0.0], cumulative])
        total = cumulative[-1] if cumulative[-1] > 0 else 1.0
        ax2.plot(freq[low], cumulative[low] / total, label=system, linewidth=2.1, color=colors[system])

    ax.axvspan(0, 5, color="#d8e6f3", alpha=0.45, linewidth=0)
    ax.text(2.5, ax.get_ylim()[1] * 0.88, "<5 THz", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Normalized framework VDOS")
    ax.set_title("MD-derived phonon DOS / VDOS")
    ax.legend(frameon=False)
    ax.set_xlim(0, 50)

    ax2.axvline(5, color="#777777", linewidth=1.0, linestyle="--")
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("Cumulative spectral weight")
    ax2.set_title("Low-frequency accumulation")
    ax2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "md_phonon_spectrum_framework.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharey=True)
    elem_colors = {
        "Pb": "#7a3db8",
        "I": "#2f7d32",
        "Cs": "#b47b00",
        "Ti": "#444444",
        "O": "#b94141",
    }
    for axp, system in zip(axes, SYSTEMS):
        for elem, (freq, dos) in projected_curves[system].items():
            mask = freq <= 40
            axp.plot(
                freq[mask],
                dos[mask],
                label=elem,
                linewidth=1.9,
                color=elem_colors.get(elem, None),
            )
        axp.axvspan(0, 5, color="#d8e6f3", alpha=0.35, linewidth=0)
        axp.set_title(system)
        axp.set_xlabel("Frequency (THz)")
        axp.set_xlim(0, 40)
        axp.legend(frameon=False)
    axes[0].set_ylabel("Normalized projected VDOS")
    fig.tight_layout()
    fig.savefig(PLOTS / "md_phonon_projected_spectra.png", dpi=220)
    plt.close(fig)

    report = [
        "# MD-derived phonon spectrum from controlled NVT",
        "",
        "This is a finite-temperature VDOS estimated from finite-difference velocities in the 330 K NVT trajectories, not a harmonic 0 K phonopy band structure. The last 80 percent of each trajectory is used.",
        "",
        "| system | low <2 THz | low <5 THz | low <10 THz | centroid THz | peak THz | median THz | q90 THz | <5THz/TiO2 | centroid/TiO2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {system} | {low2} | {low5} | {low10} | {centroid} | {peak} | {median} | {q90} | {lowrel} | {centrel} |".format(
                system=row["system"],
                low2=fmt(row["low_2THz_frac"]),
                low5=fmt(row["low_5THz_frac"]),
                low10=fmt(row["low_10THz_frac"]),
                centroid=fmt(row["centroid_THz"], 2),
                peak=fmt(row["peak_THz"], 2),
                median=fmt(row["median_THz"], 2),
                q90=fmt(row["q90_THz"], 2),
                lowrel=fmt(row["low_5THz_vs_TiO2"], 1),
                centrel=fmt(row["centroid_vs_TiO2"], 2),
            )
        )
    report.append("")
    report.append("Projected framework metrics:")
    report.append("")
    report.append("| system | group | low <5 THz | centroid THz | peak THz |")
    report.append("|---|---|---:|---:|---:|")
    for row in projected_rows:
        report.append(
            "| {system} | {group} | {low5} | {centroid} | {peak} |".format(
                system=row["system"],
                group=row["group"],
                low5=fmt(row["low_5THz_frac"]),
                centroid=fmt(row["centroid_THz"], 2),
                peak=fmt(row["peak_THz"], 2),
            )
        )
    text = "\n".join(report) + "\n"
    (OUT / "md_phonon_spectrum_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
