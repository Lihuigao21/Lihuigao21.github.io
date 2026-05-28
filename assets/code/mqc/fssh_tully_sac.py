"""Fewest-switches surface hopping benchmark on Tully's SAC model.

Run from the repository root or directly from this directory:

    python assets/code/mqc/fssh_tully_sac.py

The script writes a PNG figure and CSV tables to assets/img/mqc-series.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tully_common import MASS, adiabatic_data, classify_scattering, model_grid
from tully_common import rk4_electronic, velocity_verlet_on_state


@dataclass
class FSSHResult:
    branch: str
    state: int
    q: float
    p: float
    hops: int
    frustrated: int
    final_pop_upper: float
    steps: int


def attempt_hop(
    state: int,
    q: float,
    p: float,
    c: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> tuple[int, float, bool, bool]:
    """Apply the two-state FSSH hop test and energy-conserving rescale."""

    energies, _, nac = adiabatic_data(q)
    rho = np.outer(c, c.conjugate())
    target = 1 - state
    rho_aa = max(float(np.real(rho[state, state])), 1.0e-14)
    velocity = p / MASS
    flux = 2.0 * np.real(rho[state, target] * nac[state, target] * velocity)
    probability = min(1.0, max(0.0, float(dt * flux / rho_aa)))

    if rng.random() >= probability:
        return state, p, False, False

    delta_e = float(energies[target] - energies[state])
    kinetic = 0.5 * p * p / MASS
    if kinetic + 1.0e-14 < delta_e:
        return state, p, False, True

    new_p2 = max(0.0, p * p - 2.0 * MASS * delta_e)
    sign = 1.0 if p >= 0.0 else -1.0
    return target, sign * float(np.sqrt(new_p2)), True, False


def run_trajectory(
    p0: float,
    dt: float,
    rng: np.random.Generator,
    q0: float = -10.0,
    q_left: float = -12.0,
    q_right: float = 12.0,
    max_time: float = 7000.0,
) -> FSSHResult:
    """Run one FSSH trajectory initialized on the lower adiabatic state."""

    q = q0
    p = p0
    state = 0
    c = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    hops = 0
    frustrated = 0
    max_steps = int(max_time / dt)

    for step in range(1, max_steps + 1):
        q, p = velocity_verlet_on_state(q, p, state, dt)
        c = rk4_electronic(c, q, p, dt)
        state, p, hopped, failed = attempt_hop(state, q, p, c, dt, rng)
        hops += int(hopped)
        frustrated += int(failed)

        if (q >= q_right and p > 0.0) or (q <= q_left and p < 0.0):
            break

    branch = classify_scattering(q, p)
    final_pop_upper = float(abs(c[1]) ** 2)
    return FSSHResult(branch, state, q, p, hops, frustrated, final_pop_upper, step)


def summarize_momentum(
    p0: float, ntraj: int, dt: float, rng: np.random.Generator
) -> dict[str, float]:
    """Run an ensemble at one initial momentum and return branch probabilities."""

    counts = {
        "T_lower": 0,
        "T_upper": 0,
        "R_lower": 0,
        "R_upper": 0,
        "inside": 0,
    }
    hops = []
    frustrated = []
    final_pop_upper = []
    steps = []
    for _ in range(ntraj):
        result = run_trajectory(p0=p0, dt=dt, rng=rng)
        if result.branch == "transmitted":
            key = "T_upper" if result.state == 1 else "T_lower"
        elif result.branch == "reflected":
            key = "R_upper" if result.state == 1 else "R_lower"
        else:
            key = "inside"
        counts[key] += 1
        hops.append(result.hops)
        frustrated.append(result.frustrated)
        final_pop_upper.append(result.final_pop_upper)
        steps.append(result.steps)

    row = {
        "p0": p0,
        "ntraj": ntraj,
        "T_lower": counts["T_lower"] / ntraj,
        "T_upper": counts["T_upper"] / ntraj,
        "R_lower": counts["R_lower"] / ntraj,
        "R_upper": counts["R_upper"] / ntraj,
        "inside": counts["inside"] / ntraj,
        "mean_hops": float(np.mean(hops)),
        "mean_frustrated_hops": float(np.mean(frustrated)),
        "mean_final_pop_upper": float(np.mean(final_pop_upper)),
        "mean_steps": float(np.mean(steps)),
    }
    row["T_total"] = row["T_lower"] + row["T_upper"]
    row["R_total"] = row["R_lower"] + row["R_upper"]
    return row


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
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
    rows: list[dict[str, float]],
    output_png: Path,
    exact_reference: dict[float, dict[str, float]] | None = None,
) -> None:
    grid = model_grid()
    p = np.array([row["p0"] for row in rows], dtype=float)
    exact_reference = exact_reference or {}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(grid["x"], grid["v11"], "--", color="#5975a4", lw=1.4, label=r"$V_{11}$")
    ax.plot(grid["x"], grid["v22"], "--", color="#b55d60", lw=1.4, label=r"$V_{22}$")
    ax.plot(grid["x"], grid["e0"], color="#1f1f1f", lw=1.8, label=r"$E_0$")
    ax.plot(grid["x"], grid["e1"], color="#666666", lw=1.8, label=r"$E_1$")
    ax.plot(grid["x"], grid["d01"] / 160.0, color="#2f8f6f", lw=1.4, label=r"$d_{01}/160$")
    ax.set_xlabel("Nuclear coordinate x")
    ax.set_ylabel("Energy / scaled coupling")
    ax.set_title("Tully simple avoided crossing")
    ax.set_ylim(-0.017, 0.017)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.24)

    ax = axes[1]
    ax.plot(p, [row["T_lower"] for row in rows], "o-", color="#1f77b4", label="T lower")
    ax.plot(p, [row["T_upper"] for row in rows], "s-", color="#d62728", label="T upper")
    ax.plot(p, [row["R_total"] for row in rows], "^-", color="#444444", label="R total")
    if exact_reference:
        exact_upper = [exact_reference[p0]["T_upper_exact"] for p0 in p if p0 in exact_reference]
        exact_p = [p0 for p0 in p if p0 in exact_reference]
        exact_reflect = [exact_reference[p0]["R_total_exact"] for p0 in exact_p]
        ax.plot(exact_p, exact_upper, "D--", color="#111111", ms=4, lw=1.4, label="DVR exact T upper")
        ax.plot(exact_p, exact_reflect, "v:", color="#777777", ms=4, lw=1.2, label="DVR exact R total")
    ax.plot(p, [row["mean_final_pop_upper"] for row in rows], "x:", color="#2f8f6f", label="mean |c1|^2")
    ax.set_xlabel("Initial momentum p0")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("FSSH final branches")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.24)

    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    out_dir = repo / "assets" / "img" / "mqc-series"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260528)
    p_values = [10.0, 14.0, 18.0, 20.0, 22.0, 26.0, 30.0]
    ntraj = 50
    dt = 0.5
    rows = [summarize_momentum(p0, ntraj=ntraj, dt=dt, rng=rng) for p0 in p_values]

    csv_path = out_dir / "fssh-tully-sac.csv"
    summary_path = out_dir / "fssh-tully-sac-summary.csv"
    png_path = out_dir / "fssh-tully-sac.png"
    write_csv(csv_path, rows)
    exact_reference = load_exact_reference(out_dir)
    make_figure(rows, png_path, exact_reference=exact_reference)

    p20 = min(rows, key=lambda row: abs(row["p0"] - 20.0))
    exact_p20 = exact_reference.get(float(p20["p0"]), {})
    summary_rows = [
        {
            "dt": dt,
            "ntraj_per_momentum": ntraj,
            "near_p20": p20["p0"],
            "near_p20_T_lower": p20["T_lower"],
            "near_p20_T_upper": p20["T_upper"],
            "near_p20_R_total": p20["R_total"],
            "near_p20_mean_hops": p20["mean_hops"],
            "near_p20_mean_final_pop_upper": p20["mean_final_pop_upper"],
            "near_p20_dvr_T_upper_exact": exact_p20.get("T_upper_exact", np.nan),
            "near_p20_dvr_R_total_exact": exact_p20.get("R_total_exact", np.nan),
        }
    ]
    write_csv(summary_path, summary_rows)

    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"p0={p20['p0']:.0f}: T_lower={p20['T_lower']:.3f}, T_upper={p20['T_upper']:.3f}, R_total={p20['R_total']:.3f}")


if __name__ == "__main__":
    main()
