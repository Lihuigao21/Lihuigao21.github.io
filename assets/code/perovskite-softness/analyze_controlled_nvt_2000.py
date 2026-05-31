from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    __import__("os").environ.get(
        "CONTROLLED_NVT_ROOT",
        "/public/home/gaolihui/codex/controlled_nvt_2000_20260530",
    )
)
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
(OUT / "csv").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

SYSTEMS = {
    "MAPbI3": {"B": "Pb", "X": "I", "x_coord": 2},
    "CsPbI3": {"B": "Pb", "X": "I", "x_coord": 2},
    "TiO2": {"B": "Ti", "X": "O", "x_coord": 3},
}


def parse_incar(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "=" not in line or line.lstrip().startswith(("#", "!")):
                continue
            key, value = line.split("=", 1)
            params[key.strip().upper()] = value.split("#")[0].split("!")[0].strip()
    return params


def parse_oszicar(path: Path) -> list[dict[str, float]]:
    rows = []
    pattern = re.compile(
        r"^\s*(\d+)\s+T=\s*([-+0-9.]+).*?E=\s*([-+0-9.Ee]+)\s+F=\s*([-+0-9.Ee]+).*?EK=\s*([-+0-9.Ee]+)"
    )
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                rows.append(
                    {
                        "step": int(match.group(1)),
                        "T": float(match.group(2)),
                        "E": float(match.group(3)),
                        "F": float(match.group(4)),
                        "EK": float(match.group(5)),
                    }
                )
    return rows


def outcar_completed(path: Path) -> bool:
    if not path.exists():
        return False
    tail = path.read_text(encoding="utf-8", errors="ignore")[-50000:]
    return "Voluntary context switches" in tail or "General timing and accounting" in tail


def read_xdatcar(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    elems = lines[5].split()
    counts = [int(float(x)) for x in lines[6].split()]
    symbols = np.array([elem for elem, count in zip(elems, counts) for _ in range(count)])
    n_atoms = len(symbols)
    frames: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("direct configuration"):
            coords = [[float(x) for x in lines[i + 1 + j].split()[:3]] for j in range(n_atoms)]
            frames.append(np.array(coords, dtype=float))
            i += n_atoms + 1
        else:
            i += 1
    return lattice, symbols, frames


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return math.nan
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: math.nan for k in ("n", "mean", "std", "min", "p05", "median", "p95", "max", "p95_p05", "cv_pct")}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "p95_p05": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
        "cv_pct": float(std / abs(mean) * 100) if mean else math.nan,
    }


def prefixed(row: dict[str, object], prefix: str, data: dict[str, float]) -> None:
    for key, value in data.items():
        row[f"{prefix}_{key}"] = value


def analyze_structure(name: str, spec: dict[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    run_dir = ROOT / name
    incar = parse_incar(run_dir / "INCAR")
    osz = parse_oszicar(run_dir / "OSZICAR")
    completed = outcar_completed(run_dir / "OUTCAR")
    lattice, symbols, frames = read_xdatcar(run_dir / "XDATCAR") if (run_dir / "XDATCAR").exists() else (None, None, [])

    skip = int(0.2 * len(frames))
    prod_frames = frames[skip:]
    b_indices = np.where(symbols == spec["B"])[0] if symbols is not None else np.array([], dtype=int)
    x_indices = np.where(symbols == spec["X"])[0] if symbols is not None else np.array([], dtype=int)

    bonds: list[float] = []
    bridge_angles: list[float] = []
    internal_angles: list[float] = []
    oct_devs: list[float] = []
    frame_mean_bond: list[float] = []
    frame_mean_bridge: list[float] = []
    frame_mean_oct_dev: list[float] = []

    for frame in prod_frames:
        b_to_x_frac = frame[x_indices][None, :, :] - frame[b_indices][:, None, :]
        b_to_x_frac -= np.round(b_to_x_frac)
        b_to_x_cart = b_to_x_frac @ lattice
        distances = np.linalg.norm(b_to_x_cart, axis=2)

        nearest_x = np.argpartition(distances, 5, axis=1)[:, :6]
        frame_bonds: list[float] = []
        frame_internal: list[float] = []
        frame_oct: list[float] = []
        for b_local in range(len(b_indices)):
            vectors = b_to_x_cart[b_local, nearest_x[b_local]]
            ds = distances[b_local, nearest_x[b_local]]
            frame_bonds.extend(ds.tolist())
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    angle = angle_deg(vectors[i], vectors[j])
                    frame_internal.append(angle)
                    frame_oct.append(min(abs(angle - 90.0), abs(angle - 180.0)))

        x_to_b_cart = -np.transpose(b_to_x_cart, (1, 0, 2))
        x_to_b_dist = distances.T
        x_coord = int(spec["x_coord"])
        nearest_b = np.argpartition(x_to_b_dist, x_coord - 1, axis=1)[:, :x_coord]
        frame_bridge: list[float] = []
        for x_local in range(len(x_indices)):
            vectors = x_to_b_cart[x_local, nearest_b[x_local]]
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    frame_bridge.append(angle_deg(vectors[i], vectors[j]))

        bonds.extend(frame_bonds)
        internal_angles.extend(frame_internal)
        oct_devs.extend(frame_oct)
        bridge_angles.extend(frame_bridge)
        frame_mean_bond.append(float(np.mean(frame_bonds)))
        frame_mean_bridge.append(float(np.mean(frame_bridge)))
        frame_mean_oct_dev.append(float(np.mean(frame_oct)))

    prod_osz = osz[int(0.2 * len(osz)) :] if osz else []
    temperatures = [row["T"] for row in prod_osz]
    energies = [row["E"] for row in prod_osz]
    potim = float(incar.get("POTIM", "nan").split()[0])
    if len(energies) > 2:
        times_ps = np.array([row["step"] * potim / 1000 for row in prod_osz])
        e_drift = float(np.polyfit(times_ps, np.array(energies), 1)[0])
    else:
        e_drift = math.nan

    row: dict[str, object] = {
        "system": name,
        "completed": completed,
        "frames_total": len(frames),
        "frames_used": len(prod_frames),
        "md_steps": osz[-1]["step"] if osz else 0,
        "B_X": f"{spec['B']}-{spec['X']}",
        "bridge": f"{spec['B']}-{spec['X']}-{spec['B']}",
        "T_mean_K": float(np.mean(temperatures)) if temperatures else math.nan,
        "T_std_K": float(np.std(temperatures)) if temperatures else math.nan,
        "E_drift_eV_per_ps": e_drift,
    }
    raw = {
        "bond_A": np.asarray(bonds),
        "bridge_deg": np.asarray(bridge_angles),
        "internal_deg": np.asarray(internal_angles),
        "oct_dev_deg": np.asarray(oct_devs),
        "frame_mean_bond_A": np.asarray(frame_mean_bond),
        "frame_mean_bridge_deg": np.asarray(frame_mean_bridge),
        "frame_mean_oct_dev_deg": np.asarray(frame_mean_oct_dev),
    }
    for metric, values in raw.items():
        prefixed(row, metric, stats(values))
    return row, raw


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    raw_by_system = {}
    for name, spec in SYSTEMS.items():
        row, raw = analyze_structure(name, spec)
        rows.append(row)
        raw_by_system[name] = raw

    tio2 = next(row for row in rows if row["system"] == "TiO2")
    for row in rows:
        for metric in (
            "bond_A_std",
            "bond_A_p95_p05",
            "bridge_deg_std",
            "bridge_deg_p95_p05",
            "oct_dev_deg_mean",
            "oct_dev_deg_p95_p05",
            "frame_mean_bond_A_std",
            "frame_mean_bridge_deg_std",
            "frame_mean_oct_dev_deg_std",
        ):
            denom = float(tio2.get(metric, math.nan))
            row[f"{metric}_vs_TiO2"] = float(row[metric]) / denom if denom and math.isfinite(denom) else math.nan

    write_csv(OUT / "controlled_nvt_2000_structural_summary.csv", rows)
    (OUT / "controlled_nvt_2000_structural_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    colors = {"MAPbI3": "#9E2A2B", "CsPbI3": "#E09F3E", "TiO2": "#335C67"}
    for metric, xlabel, filename in [
        ("bond_A", "B-X bond length (A)", "controlled_bond_hist.png"),
        ("bridge_deg", "B-X-B angle (deg)", "controlled_bridge_angle_hist.png"),
        ("oct_dev_deg", "X-B-X angle deviation from 90/180 deg", "controlled_oct_dev_hist.png"),
        ("frame_mean_bond_A", "Frame mean B-X bond length (A)", "controlled_frame_mean_bond_hist.png"),
        ("frame_mean_bridge_deg", "Frame mean B-X-B angle (deg)", "controlled_frame_mean_bridge_hist.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        for name, raw in raw_by_system.items():
            data = raw[metric]
            data = data[np.isfinite(data)]
            ax.hist(data, bins=80, density=True, histtype="step", lw=1.8, label=name, color=colors[name])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "plots" / filename)
        plt.close(fig)

    def fmt(value: object, nd: int = 3) -> str:
        try:
            number = float(value)
            if not math.isfinite(number):
                return "NA"
            return f"{number:.{nd}f}"
        except Exception:
            return str(value)

    lines = [
        "# Controlled 2000-step NVT structural comparison",
        "",
        "All three systems use T=330 K, POTIM=0.5 fs, NSW=2000, NBLOCK=1, ENCUT=600 eV, Gamma 2x2x2. The first 20 percent is skipped as equilibration.",
        "",
        "| system | completed | used frames | T mean(K) | T std(K) | bond std(A) | bond 5-95(A) | bond std/TiO2 | B-X-B std(deg) | B-X-B 5-95(deg) | oct.dev mean(deg) | frame-mean bond std(A) | frame-mean B-X-B std(deg) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['completed']} | {row['frames_used']} | {fmt(row['T_mean_K'],1)} | {fmt(row['T_std_K'],1)} | "
            f"{fmt(row['bond_A_std'],3)} | {fmt(row['bond_A_p95_p05'],3)} | {fmt(row['bond_A_std_vs_TiO2'],2)} | "
            f"{fmt(row['bridge_deg_std'],2)} | {fmt(row['bridge_deg_p95_p05'],2)} | {fmt(row['oct_dev_deg_mean'],2)} | "
            f"{fmt(row['frame_mean_bond_A_std'],4)} | {fmt(row['frame_mean_bridge_deg_std'],2)} |"
        )
    report = "\n".join(lines) + "\n"
    (OUT / "controlled_nvt_2000_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
