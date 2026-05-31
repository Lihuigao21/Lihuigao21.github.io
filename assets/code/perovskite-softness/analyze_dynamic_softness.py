from __future__ import annotations

import csv
import json
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
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

SYSTEMS = {
    "MAPbI3": {"B": "Pb", "X": "I", "framework": {"Pb", "I"}},
    "CsPbI3": {"B": "Pb", "X": "I", "framework": {"Pb", "I"}},
    "TiO2": {"B": "Ti", "X": "O", "framework": {"Ti", "O"}},
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
            coords = [
                [float(x) for x in lines[i + 1 + j].split()[:3]]
                for j in range(n_atoms)
            ]
            frames.append(coords)
            i += n_atoms + 1
        else:
            i += 1
    return lattice, symbols, np.asarray(frames, dtype=float)


def parse_potim_fs(path: Path) -> float:
    if not path.exists():
        return 0.5
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().upper() == "POTIM":
            return float(value.split("#")[0].split("!")[0].split()[0])
    return 0.5


def unwrap_direct(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return frames
    steps = np.diff(frames, axis=0)
    steps -= np.round(steps)
    out = np.empty_like(frames)
    out[0] = frames[0]
    out[1:] = frames[0] + np.cumsum(steps, axis=0)
    return out


def remove_com(positions: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    masses = np.array([MASS.get(str(sym), 1.0) for sym in symbols], dtype=float)
    com = np.einsum("tnc,n->tc", positions, masses) / masses.sum()
    return positions - com[:, None, :]


def displacement_block(
    lattice: np.ndarray, symbols: np.ndarray, frames: np.ndarray, skip_frac: float
) -> np.ndarray:
    skip = int(round(skip_frac * len(frames)))
    direct = unwrap_direct(frames)
    cart = direct @ lattice
    rel = remove_com(cart, symbols)
    prod = rel[skip:]
    return prod - prod.mean(axis=0, keepdims=True)


def element_msd_rows(system: str, symbols: np.ndarray, disp: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sq = np.sum(disp * disp, axis=2)
    amp = np.sqrt(sq)
    for elem in sorted(set(symbols.tolist()), key=list(symbols).index):
        idx = np.where(symbols == elem)[0]
        values = sq[:, idx].reshape(-1)
        amps = amp[:, idx].reshape(-1)
        rows.append(
            {
                "system": system,
                "group": elem,
                "n_atoms": int(len(idx)),
                "msd_A2": float(np.mean(values)),
                "rms_disp_A": float(np.sqrt(np.mean(values))),
                "disp_p50_A": float(np.percentile(amps, 50)),
                "disp_p95_A": float(np.percentile(amps, 95)),
            }
        )
    return rows


def group_msd(system: str, group: str, idx: np.ndarray, disp: np.ndarray) -> dict[str, object]:
    sq = np.sum(disp[:, idx, :] * disp[:, idx, :], axis=2).reshape(-1)
    amp = np.sqrt(sq)
    return {
        "system": system,
        "group": group,
        "n_atoms": int(len(idx)),
        "msd_A2": float(np.mean(sq)),
        "rms_disp_A": float(np.sqrt(np.mean(sq))),
        "disp_p50_A": float(np.percentile(amp, 50)),
        "disp_p95_A": float(np.percentile(amp, 95)),
    }


def vdos_for_group(
    positions_rel: np.ndarray, symbols: np.ndarray, idx: np.ndarray, dt_fs: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    velocities = np.diff(positions_rel, axis=0) / dt_fs
    v = velocities[:, idx, :].reshape(velocities.shape[0], -1)
    v = v - v.mean(axis=0, keepdims=True)
    if v.shape[0] < 4:
        freq_thz = np.array([], dtype=float)
        power = np.array([], dtype=float)
        metrics = {
            "vdos_low_5THz_frac": math.nan,
            "vdos_low_10THz_frac": math.nan,
            "vdos_centroid_THz": math.nan,
            "vdos_peak_THz": math.nan,
        }
        return freq_thz, power, metrics

    window = np.hanning(v.shape[0])[:, None]
    spectrum = np.fft.rfft(v * window, axis=0)
    power = np.sum(np.abs(spectrum) ** 2, axis=1)
    freq_thz = np.fft.rfftfreq(v.shape[0], d=dt_fs * 1e-15) / 1e12
    if len(power) > 0:
        power[0] = 0.0
    total = float(np.sum(power))
    if total <= 0:
        metrics = {
            "vdos_low_5THz_frac": math.nan,
            "vdos_low_10THz_frac": math.nan,
            "vdos_centroid_THz": math.nan,
            "vdos_peak_THz": math.nan,
        }
    else:
        nz = power.copy()
        metrics = {
            "vdos_low_5THz_frac": float(np.sum(nz[freq_thz <= 5.0]) / total),
            "vdos_low_10THz_frac": float(np.sum(nz[freq_thz <= 10.0]) / total),
            "vdos_centroid_THz": float(np.sum(freq_thz * nz) / total),
            "vdos_peak_THz": float(freq_thz[int(np.argmax(nz))]),
        }
    return freq_thz, power, metrics


def pca_for_group(disp: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    x = disp[:, idx, :].reshape(disp.shape[0], -1)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 3 or x.shape[1] < 1:
        return {
            "pca_pc1_frac": math.nan,
            "pca_pc1_per_atom_rms_A": math.nan,
            "pca_total_rms_A": math.nan,
            "pca_effective_modes": math.nan,
        }
    _, svals, _ = np.linalg.svd(x / math.sqrt(max(x.shape[0] - 1, 1)), full_matrices=False)
    eig = svals * svals
    trace = float(np.sum(eig))
    if trace <= 0:
        return {
            "pca_pc1_frac": math.nan,
            "pca_pc1_per_atom_rms_A": math.nan,
            "pca_total_rms_A": math.nan,
            "pca_effective_modes": math.nan,
        }
    return {
        "pca_pc1_frac": float(eig[0] / trace),
        "pca_pc1_per_atom_rms_A": float(math.sqrt(eig[0] / len(idx))),
        "pca_total_rms_A": float(math.sqrt(trace / len(idx))),
        "pca_effective_modes": float(trace * trace / np.sum(eig * eig)),
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
    if value is None:
        return "nan"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{ndigits}f}"


def main() -> None:
    skip_frac = 0.2
    summary_rows: list[dict[str, object]] = []
    element_msd_all: list[dict[str, object]] = []
    element_vdos_all: list[dict[str, object]] = []
    vdos_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, spec in SYSTEMS.items():
        run_dir = ROOT / name
        lattice, symbols, frames = read_xdatcar(run_dir / "XDATCAR")
        dt_fs = parse_potim_fs(run_dir / "INCAR")
        skip = int(round(skip_frac * len(frames)))
        unwrapped = unwrap_direct(frames) @ lattice
        rel = remove_com(unwrapped, symbols)
        prod_rel = rel[skip:]
        disp = prod_rel - prod_rel.mean(axis=0, keepdims=True)

        framework_idx = np.array(
            [i for i, symbol in enumerate(symbols) if str(symbol) in spec["framework"]],
            dtype=int,
        )
        b_idx = np.where(symbols == spec["B"])[0]
        x_idx = np.where(symbols == spec["X"])[0]

        element_msd_all.extend(element_msd_rows(name, symbols, disp))
        framework_row = group_msd(name, "framework", framework_idx, disp)
        b_row = group_msd(name, str(spec["B"]), b_idx, disp)
        x_row = group_msd(name, str(spec["X"]), x_idx, disp)

        freq, power, framework_vdos = vdos_for_group(prod_rel, symbols, framework_idx, dt_fs)
        vdos_curves[name] = (freq, power / np.max(power) if np.max(power) > 0 else power)
        _, _, b_vdos = vdos_for_group(prod_rel, symbols, b_idx, dt_fs)
        _, _, x_vdos = vdos_for_group(prod_rel, symbols, x_idx, dt_fs)
        pca = pca_for_group(disp, framework_idx)

        for group, idx in [("framework", framework_idx), (str(spec["B"]), b_idx), (str(spec["X"]), x_idx)]:
            _, _, metrics = vdos_for_group(prod_rel, symbols, idx, dt_fs)
            element_vdos_all.append(
                {
                    "system": name,
                    "group": group,
                    "n_atoms": int(len(idx)),
                    **metrics,
                }
            )

        summary_rows.append(
            {
                "system": name,
                "frames_total": int(len(frames)),
                "frames_used": int(len(prod_rel)),
                "dt_fs": dt_fs,
                "framework": "+".join(sorted(spec["framework"])),
                "framework_msd_A2": framework_row["msd_A2"],
                "framework_rms_disp_A": framework_row["rms_disp_A"],
                "framework_disp_p95_A": framework_row["disp_p95_A"],
                "B_rms_disp_A": b_row["rms_disp_A"],
                "X_rms_disp_A": x_row["rms_disp_A"],
                "framework_vdos_low_5THz_frac": framework_vdos["vdos_low_5THz_frac"],
                "framework_vdos_low_10THz_frac": framework_vdos["vdos_low_10THz_frac"],
                "framework_vdos_centroid_THz": framework_vdos["vdos_centroid_THz"],
                "framework_vdos_peak_THz": framework_vdos["vdos_peak_THz"],
                "B_vdos_centroid_THz": b_vdos["vdos_centroid_THz"],
                "X_vdos_centroid_THz": x_vdos["vdos_centroid_THz"],
                **pca,
            }
        )

    tio2 = next(row for row in summary_rows if row["system"] == "TiO2")
    for row in summary_rows:
        for key in [
            "framework_msd_A2",
            "framework_rms_disp_A",
            "framework_vdos_low_5THz_frac",
            "framework_vdos_centroid_THz",
            "pca_pc1_frac",
            "pca_pc1_per_atom_rms_A",
        ]:
            denom = float(tio2[key])
            row[f"{key}_vs_TiO2"] = float(row[key]) / denom if denom else math.nan

    write_csv(OUT / "dynamic_softness_summary.csv", summary_rows)
    write_csv(CSV_DIR / "dynamic_softness_element_msd.csv", element_msd_all)
    write_csv(CSV_DIR / "dynamic_softness_element_vdos.csv", element_vdos_all)
    (OUT / "dynamic_softness_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plt.figure(figsize=(7.2, 4.2))
    for name, (freq, power) in vdos_curves.items():
        mask = freq <= 40.0
        plt.plot(freq[mask], power[mask], label=name, linewidth=1.8)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized VDOS from finite-difference velocities")
    plt.title("Framework VDOS from controlled 330 K NVT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "dynamic_softness_framework_vdos.png", dpi=180)
    plt.close()

    report_lines = [
        "# Dynamic softness from controlled 330 K NVT",
        "",
        "All metrics use the last 80 percent of each 2000-step NVT trajectory. Framework means Pb+I for perovskites and Ti+O for TiO2. VDOS is estimated from finite-difference velocities, so the 1 ps trajectory gives a coarse low-frequency resolution.",
        "",
        "| system | framework RMS disp(A) | B RMS(A) | X RMS(A) | VDOS <5 THz | VDOS centroid(THz) | PC1 frac | PC1 per-atom RMS(A) | effective modes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| {system} | {rms} | {b} | {x} | {low5} | {centroid} | {pc1} | {pc1rms} | {neff} |".format(
                system=row["system"],
                rms=fmt(row["framework_rms_disp_A"], 4),
                b=fmt(row["B_rms_disp_A"], 4),
                x=fmt(row["X_rms_disp_A"], 4),
                low5=fmt(row["framework_vdos_low_5THz_frac"], 3),
                centroid=fmt(row["framework_vdos_centroid_THz"], 2),
                pc1=fmt(row["pca_pc1_frac"], 3),
                pc1rms=fmt(row["pca_pc1_per_atom_rms_A"], 4),
                neff=fmt(row["pca_effective_modes"], 1),
            )
        )

    report_lines.extend(
        [
            "",
            "Relative to TiO2:",
            "",
            "| system | RMS disp/TiO2 | low-5THz/TiO2 | centroid/TiO2 | PC1 frac/TiO2 | PC1 RMS/TiO2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        report_lines.append(
            "| {system} | {rms} | {low5} | {centroid} | {pc1} | {pc1rms} |".format(
                system=row["system"],
                rms=fmt(row["framework_rms_disp_A_vs_TiO2"], 2),
                low5=fmt(row["framework_vdos_low_5THz_frac_vs_TiO2"], 2),
                centroid=fmt(row["framework_vdos_centroid_THz_vs_TiO2"], 2),
                pc1=fmt(row["pca_pc1_frac_vs_TiO2"], 2),
                pc1rms=fmt(row["pca_pc1_per_atom_rms_A_vs_TiO2"], 2),
            )
        )

    report = "\n".join(report_lines) + "\n"
    (OUT / "dynamic_softness_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
