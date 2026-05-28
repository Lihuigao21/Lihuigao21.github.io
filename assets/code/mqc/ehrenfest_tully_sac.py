"""Ehrenfest mean-field benchmark on Tully's SAC model.

Run from the repository root or directly from this directory:

    python assets/code/mqc/ehrenfest_tully_sac.py

The script writes a PNG figure and CSV tables to assets/img/mqc-series.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tully_common import MASS, adiabatic_data, classify_scattering
from tully_common import electronic_rhs, mean_field_force


def coupled_rhs(q: float, p: float, c: np.ndarray) -> tuple[float, float, np.ndarray]:
    return p / MASS, mean_field_force(q, c), electronic_rhs(c, q, p)


def rk4_step(q: float, p: float, c: np.ndarray, dt: float) -> tuple[float, float, np.ndarray]:
    k1q, k1p, k1c = coupled_rhs(q, p, c)
    k2q, k2p, k2c = coupled_rhs(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p, c + 0.5 * dt * k1c)
    k3q, k3p, k3c = coupled_rhs(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p, c + 0.5 * dt * k2c)
    k4q, k4p, k4c = coupled_rhs(q + dt * k3q, p + dt * k3p, c + dt * k3c)

    q_new = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    p_new = p + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    c_new = c + (dt / 6.0) * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
    norm = np.linalg.norm(c_new)
    if norm > 0.0:
        c_new = c_new / norm
    return float(q_new), float(p_new), c_new


def mean_energy(q: float, p: float, c: np.ndarray) -> float:
    energies, _, _ = adiabatic_data(q)
    pop = np.abs(c) ** 2
    return float(0.5 * p * p / MASS + pop[0] * energies[0] + pop[1] * energies[1])


def run_ehrenfest(
    p0: float,
    dt: float = 0.2,
    q0: float = -10.0,
    q_left: float = -12.0,
    q_right: float = 12.0,
    max_time: float = 7000.0,
    record_stride: int = 10,
) -> tuple[dict[str, float | str], list[dict[str, float]]]:
    q = q0
    p = p0
    c = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    max_steps = int(max_time / dt)
    trace: list[dict[str, float]] = []

    for step in range(max_steps + 1):
        if step % record_stride == 0:
            trace.append(
                {
                    "time": step * dt,
                    "q": q,
                    "p": p,
                    "pop_lower": float(abs(c[0]) ** 2),
                    "pop_upper": float(abs(c[1]) ** 2),
                    "mean_energy": mean_energy(q, p, c),
                }
            )

        if step > 0 and ((q >= q_right and p > 0.0) or (q <= q_left and p < 0.0)):
            break

        q, p, c = rk4_step(q, p, c, dt)

    summary = {
        "p0": p0,
        "dt": dt,
        "steps": step,
        "final_q": q,
        "final_p": p,
        "branch": classify_scattering(q, p),
        "final_pop_lower": float(abs(c[0]) ** 2),
        "final_pop_upper": float(abs(c[1]) ** 2),
        "final_mean_energy": mean_energy(q, p, c),
    }
    return summary, trace


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_exact_reference(out_dir: Path) -> dict[float, dict[str, float]]:
    path = out_dir / "dvr-tully-sac-reference.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {
            float(row["p0"]): {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        }


def make_figure(
    rows: list[dict[str, float | str]],
    trace: list[dict[str, float]],
    output_png: Path,
    exact_reference: dict[float, dict[str, float]] | None = None,
) -> None:
    p_values = np.array([float(row["p0"]) for row in rows])
    upper = np.array([float(row["final_pop_upper"]) for row in rows])
    exact_reference = exact_reference or {}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)

    ax = axes[0]
    time = np.array([row["time"] for row in trace])
    q = np.array([row["q"] for row in trace])
    pop_upper = np.array([row["pop_upper"] for row in trace])
    ax.plot(time, q, color="#1f1f1f", lw=1.7, label="q(t)")
    ax.axhline(0.0, color="#888888", lw=0.9, ls=":")
    ax.set_xlabel("Time")
    ax.set_ylabel("Nuclear coordinate q")
    ax.set_title("Mean-field trajectory, p0 = 20")
    ax.grid(alpha=0.24)
    twin = ax.twinx()
    twin.plot(time, pop_upper, color="#d62728", lw=1.5, ls="--", label=r"$|c_1(t)|^2$")
    twin.set_ylabel("Upper-state population")
    twin.set_ylim(-0.03, 1.03)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    transmitted = [row["branch"] == "transmitted" for row in rows]
    colors = ["#1f77b4" if flag else "#444444" for flag in transmitted]
    ax.plot(p_values, upper, color="#777777", lw=1.0, alpha=0.55)
    ax.scatter(p_values, upper, c=colors, s=44, zorder=3)
    if exact_reference:
        exact_p = [p0 for p0 in p_values if p0 in exact_reference]
        exact_upper = [exact_reference[p0]["adiabatic_upper_exact"] for p0 in exact_p]
        ax.plot(exact_p, exact_upper, "D--", color="#111111", ms=4, lw=1.4, label="DVR exact upper")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("Initial momentum p0")
    ax.set_ylabel("Final upper-state population")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Ehrenfest SAC scan")
    ax.grid(alpha=0.24)
    ax.text(0.02, 0.96, "blue: transmitted final path", transform=ax.transAxes, va="top", fontsize=8)

    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    out_dir = repo / "assets" / "img" / "mqc-series"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_values = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]
    dt = 0.2
    rows: list[dict[str, float | str]] = []
    p20_trace: list[dict[str, float]] = []
    for p0 in p_values:
        summary, trace = run_ehrenfest(p0, dt=dt)
        rows.append(summary)
        if abs(p0 - 20.0) < 1.0e-12:
            p20_trace = trace

    csv_path = out_dir / "ehrenfest-tully-sac.csv"
    trace_path = out_dir / "ehrenfest-tully-sac-trace-p20.csv"
    summary_path = out_dir / "ehrenfest-tully-sac-summary.csv"
    png_path = out_dir / "ehrenfest-tully-sac.png"
    write_csv(csv_path, rows)
    write_csv(trace_path, p20_trace)
    exact_reference = load_exact_reference(out_dir)
    make_figure(rows, p20_trace, png_path, exact_reference=exact_reference)

    p20 = min(rows, key=lambda row: abs(float(row["p0"]) - 20.0))
    exact_p20 = exact_reference.get(float(p20["p0"]), {})
    write_csv(
        summary_path,
        [
            {
                "dt": dt,
                "p20_final_pop_upper": p20["final_pop_upper"],
                "p20_final_q": p20["final_q"],
                "p20_final_p": p20["final_p"],
                "p20_branch": p20["branch"],
                "p20_steps": p20["steps"],
                "p20_dvr_adiabatic_upper_exact": exact_p20.get("adiabatic_upper_exact", np.nan),
                "p20_dvr_R_total_exact": exact_p20.get("R_total_exact", np.nan),
            }
        ],
    )

    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(
        "p0=20: final upper population="
        f"{float(p20['final_pop_upper']):.3f}, branch={p20['branch']}"
    )


if __name__ == "__main__":
    main()
