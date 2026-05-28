"""Normal-mode versus bead-coordinate free ring-polymer propagation.

This script supports PIMD Series Part VI.  It compares the free ring-polymer
substep used in PIMD/RPMD integrators:

1. exact normal-mode propagation, implemented with FFTs;
2. bead-coordinate velocity Verlet with one outer timestep;
3. bead-coordinate velocity Verlet with automatic subcycling.

The benchmark is intentionally restricted to the free spring part of the ring
polymer.  That isolates the point made in i-PI's normal-mode code: the spring
matrix is diagonal in normal modes and can be propagated analytically, whereas
Cartesian/bead propagation must resolve the highest spring frequency with a
small enough timestep.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "pimd-series" / "normal-mode-vs-bead.png"
SUBCYCLING_FIGURE_PATH = (
    ROOT / "assets" / "img" / "pimd-series" / "normal-mode-vs-bead-subcycling.png"
)
SCAN_PATH = ROOT / "assets" / "img" / "pimd-series" / "normal-mode-vs-bead.csv"
SUBCYCLING_PATH = (
    ROOT / "assets" / "img" / "pimd-series" / "normal-mode-vs-bead-subcycling.csv"
)
SUMMARY_PATH = ROOT / "assets" / "img" / "pimd-series" / "normal-mode-vs-bead-summary.csv"

BETA = 2.0
HBAR = 1.0
MASS = 1.0
OUTER_DT = 0.04
N_STEPS = 600
BEAD_COUNTS = (8, 16, 32, 64, 128, 256, 512)
SUBCYCLE_TEST_P = 128
SUBCYCLE_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
SUBCYCLE_STABILITY_TARGET = 0.8
SEED = 20260529
BLOWUP_REL_ERROR = 1.0e6


@dataclass
class BenchmarkRow:
    n_beads: int
    method: str
    nmts: int
    omega_max: float
    completed_steps: int
    max_relative_energy_error: float
    final_relative_energy_error: float
    runtime_seconds: float
    stable: bool


def ring_polymer_frequencies(n_beads: int) -> np.ndarray:
    """Return FFT-ordered free ring-polymer spring frequencies."""
    k = np.arange(n_beads)
    omega_p = n_beads / (BETA * HBAR)
    return 2.0 * omega_p * np.abs(np.sin(np.pi * k / n_beads))


def spring_force(q: np.ndarray) -> np.ndarray:
    """Free ring-polymer spring force in bead coordinates."""
    omega_p = q.size / (BETA * HBAR)
    return -MASS * omega_p**2 * (2.0 * q - np.roll(q, 1) - np.roll(q, -1))


def free_ring_energy(q: np.ndarray, p: np.ndarray) -> float:
    """Free ring-polymer Hamiltonian: kinetic plus spring energy."""
    omega_p = q.size / (BETA * HBAR)
    kinetic = 0.5 * np.sum(p**2) / MASS
    spring = 0.5 * MASS * omega_p**2 * np.sum((np.roll(q, -1) - q) ** 2)
    return float(kinetic + spring)


def initial_condition(n_beads: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic, high-mode-rich initial condition."""
    rng = np.random.default_rng(seed + 104729 * n_beads)
    q = rng.normal(0.0, 1.0, size=n_beads)
    p = rng.normal(0.0, 1.0, size=n_beads)
    q -= q.mean()
    p -= p.mean()
    return q, p


def nm_fft_step(q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact free ring-polymer step using complex FFT normal modes."""
    n_beads = q.size
    omega = ring_polymer_frequencies(n_beads)

    q_nm = np.fft.fft(q) / np.sqrt(n_beads)
    p_nm = np.fft.fft(p) / np.sqrt(n_beads)

    q_new = np.empty_like(q_nm)
    p_new = np.empty_like(p_nm)

    q_new[0] = q_nm[0] + dt * p_nm[0] / MASS
    p_new[0] = p_nm[0]

    nonzero = omega > 0.0
    cos_term = np.cos(omega[nonzero] * dt)
    sin_term = np.sin(omega[nonzero] * dt)
    q_new[nonzero] = (
        q_nm[nonzero] * cos_term
        + p_nm[nonzero] * sin_term / (MASS * omega[nonzero])
    )
    p_new[nonzero] = (
        p_nm[nonzero] * cos_term
        - MASS * omega[nonzero] * q_nm[nonzero] * sin_term
    )

    q_out = np.fft.ifft(q_new * np.sqrt(n_beads)).real
    p_out = np.fft.ifft(p_new * np.sqrt(n_beads)).real
    return q_out, p_out


def bead_velocity_verlet_step(
    q: np.ndarray,
    p: np.ndarray,
    dt: float,
    n_substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Free ring-polymer step in bead coordinates with optional subcycling."""
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        p = p + 0.5 * sub_dt * spring_force(q)
        q = q + sub_dt * p / MASS
        p = p + 0.5 * sub_dt * spring_force(q)
    return q, p


def automatic_nmts(n_beads: int, dt: float = OUTER_DT) -> int:
    """Choose a conservative subcycle count for bead-coordinate Verlet."""
    omega_max = float(ring_polymer_frequencies(n_beads).max())
    return max(1, int(np.ceil(dt * omega_max / SUBCYCLE_STABILITY_TARGET)))


def run_trajectory(n_beads: int, method: str, nmts: int = 1) -> BenchmarkRow:
    """Run one free ring-polymer trajectory and return stability diagnostics."""
    q, p = initial_condition(n_beads, SEED)
    e0 = free_ring_energy(q, p)
    max_rel = 0.0
    final_rel = 0.0
    stable = True
    completed = 0

    start = perf_counter()
    for step in range(N_STEPS):
        if method == "normal_mode_fft":
            q, p = nm_fft_step(q, p, OUTER_DT)
        elif method == "bead_verlet":
            q, p = bead_velocity_verlet_step(q, p, OUTER_DT, nmts)
        else:
            raise ValueError(f"Unknown method: {method}")

        energy = free_ring_energy(q, p)
        if not np.isfinite(energy):
            max_rel = BLOWUP_REL_ERROR
            final_rel = BLOWUP_REL_ERROR
            stable = False
            completed = step + 1
            break

        final_rel = abs((energy - e0) / e0)
        max_rel = max(max_rel, final_rel)
        completed = step + 1
        if max_rel > BLOWUP_REL_ERROR:
            max_rel = BLOWUP_REL_ERROR
            final_rel = BLOWUP_REL_ERROR
            stable = False
            break

    runtime = perf_counter() - start
    return BenchmarkRow(
        n_beads=n_beads,
        method=method,
        nmts=nmts,
        omega_max=float(ring_polymer_frequencies(n_beads).max()),
        completed_steps=completed,
        max_relative_energy_error=float(max_rel),
        final_relative_energy_error=float(final_rel),
        runtime_seconds=float(runtime),
        stable=stable and completed == N_STEPS,
    )


def run_bead_scan() -> list[BenchmarkRow]:
    """Scan bead number for normal-mode and bead-coordinate propagation."""
    rows: list[BenchmarkRow] = []
    for n_beads in BEAD_COUNTS:
        rows.append(run_trajectory(n_beads, "normal_mode_fft", nmts=1))
        rows.append(run_trajectory(n_beads, "bead_verlet", nmts=1))
        rows.append(run_trajectory(n_beads, "bead_verlet", nmts=automatic_nmts(n_beads)))
    return rows


def run_subcycling_scan() -> list[BenchmarkRow]:
    """Scan nmts at one high bead count."""
    rows = [run_trajectory(SUBCYCLE_TEST_P, "normal_mode_fft", nmts=1)]
    for nmts in SUBCYCLE_COUNTS:
        rows.append(run_trajectory(SUBCYCLE_TEST_P, "bead_verlet", nmts=nmts))
    return rows


def rows_to_csv(rows: list[BenchmarkRow]) -> str:
    """Serialize benchmark rows."""
    header = (
        "n_beads,method,nmts,omega_max,completed_steps,"
        "max_relative_energy_error,final_relative_energy_error,runtime_seconds,stable"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.n_beads},{row.method},{row.nmts},{row.omega_max:.10f},"
            f"{row.completed_steps},{row.max_relative_energy_error:.10e},"
            f"{row.final_relative_energy_error:.10e},{row.runtime_seconds:.10f},"
            f"{int(row.stable)}"
        )
    return "\n".join(lines) + "\n"


def plot_bead_scan(rows: list[BenchmarkRow]) -> None:
    """Plot robustness and efficiency versus bead number."""
    labels = {
        ("normal_mode_fft", 1): ("normal mode, FFT exact", "black", "o", "-"),
        ("bead_verlet", 1): ("bead Verlet, nmts=1", "#d62728", "s", "--"),
        ("bead_verlet", "auto"): ("bead Verlet, auto nmts", "#1f77b4", "^", "-."),
    }

    grouped: dict[tuple[str, str | int], list[BenchmarkRow]] = {
        ("normal_mode_fft", 1): [],
        ("bead_verlet", 1): [],
        ("bead_verlet", "auto"): [],
    }
    for row in rows:
        if row.method == "normal_mode_fft":
            grouped[("normal_mode_fft", 1)].append(row)
        elif row.method == "bead_verlet" and row.nmts == 1:
            grouped[("bead_verlet", 1)].append(row)
        else:
            grouped[("bead_verlet", "auto")].append(row)

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    for key, group in grouped.items():
        label, color, marker, linestyle = labels[key]
        group = sorted(group, key=lambda item: item.n_beads)
        x = np.array([row.n_beads for row in group])
        err = np.array([max(row.max_relative_energy_error, 1.0e-15) for row in group])
        runtime = np.array([row.runtime_seconds if row.stable else np.nan for row in group])
        axes[0].plot(x, err, color=color, marker=marker, ls=linestyle, lw=1.6, label=label)
        axes[1].plot(x, runtime, color=color, marker=marker, ls=linestyle, lw=1.6, label=label)

    axes[0].axhline(1.0, color="0.65", lw=1.0, ls=":", label="100% energy error")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of beads P")
    axes[0].set_ylabel("max relative energy error")
    axes[0].set_title("free spring robustness")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of beads P")
    axes[1].set_ylabel("runtime for fixed trajectory (s)")
    axes[1].set_title("cost of the free substep")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False)
    fig.savefig(FIGURE_PATH, dpi=190)


def plot_subcycling_scan(rows: list[BenchmarkRow]) -> None:
    """Plot direct bead-coordinate convergence with subcycling."""
    nm_row = next(row for row in rows if row.method == "normal_mode_fft")
    bead_rows = sorted(
        [row for row in rows if row.method == "bead_verlet"],
        key=lambda item: item.nmts,
    )
    nmts = np.array([row.nmts for row in bead_rows])
    errors = np.array([max(row.max_relative_energy_error, 1.0e-15) for row in bead_rows])
    runtimes = np.array([row.runtime_seconds if row.stable else np.nan for row in bead_rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    axes[0].plot(nmts, errors, color="#d62728", marker="s", lw=1.6, label="bead Verlet")
    axes[0].axhline(
        max(nm_row.max_relative_energy_error, 1.0e-15),
        color="black",
        lw=1.4,
        ls="--",
        label="normal mode exact",
    )
    axes[0].axvline(
        int(np.ceil(OUTER_DT * nm_row.omega_max / 2.0)),
        color="0.5",
        lw=1.0,
        ls=":",
        label="linear stability boundary",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bead-coordinate substeps nmts")
    axes[0].set_ylabel("max relative energy error")
    axes[0].set_title(f"subcycling convergence at P={SUBCYCLE_TEST_P}")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].plot(nmts, runtimes, color="#1f77b4", marker="^", lw=1.6, label="bead Verlet")
    axes[1].axhline(nm_row.runtime_seconds, color="black", lw=1.4, ls="--", label="normal mode exact")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("bead-coordinate substeps nmts")
    axes[1].set_ylabel("runtime for fixed trajectory (s)")
    axes[1].set_title("subcycling cost")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False)
    fig.savefig(SUBCYCLING_FIGURE_PATH, dpi=190)


def write_summary(scan_rows: list[BenchmarkRow], sub_rows: list[BenchmarkRow]) -> None:
    """Write a compact summary table."""
    nm_512 = next(row for row in scan_rows if row.n_beads == 512 and row.method == "normal_mode_fft")
    bead_512 = next(row for row in scan_rows if row.n_beads == 512 and row.method == "bead_verlet" and row.nmts == 1)
    auto_512 = next(row for row in scan_rows if row.n_beads == 512 and row.method == "bead_verlet" and row.nmts != 1)
    stable_one_step = [
        row.n_beads
        for row in scan_rows
        if row.method == "bead_verlet" and row.nmts == 1 and row.stable
    ]
    min_unstable = min(
        row.n_beads
        for row in scan_rows
        if row.method == "bead_verlet" and row.nmts == 1 and not row.stable
    )
    best_sub = min(
        (row for row in sub_rows if row.method == "bead_verlet" and row.stable),
        key=lambda item: item.max_relative_energy_error,
    )

    SUMMARY_PATH.write_text(
        "quantity,value\n"
        f"outer_dt,{OUTER_DT}\n"
        f"n_steps,{N_STEPS}\n"
        f"largest_one_step_stable_P,{max(stable_one_step)}\n"
        f"first_one_step_unstable_P,{min_unstable}\n"
        f"nm_fft_P512_max_rel_error,{nm_512.max_relative_energy_error:.10e}\n"
        f"bead_one_step_P512_stable,{int(bead_512.stable)}\n"
        f"bead_auto_P512_nmts,{auto_512.nmts}\n"
        f"bead_auto_P512_max_rel_error,{auto_512.max_relative_energy_error:.10e}\n"
        f"bead_auto_P512_runtime_seconds,{auto_512.runtime_seconds:.10f}\n"
        f"nm_fft_P512_runtime_seconds,{nm_512.runtime_seconds:.10f}\n"
        f"best_subcycling_P128_nmts,{best_sub.nmts}\n"
        f"best_subcycling_P128_max_rel_error,{best_sub.max_relative_energy_error:.10e}\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    scan_rows = run_bead_scan()
    sub_rows = run_subcycling_scan()

    SCAN_PATH.write_text(rows_to_csv(scan_rows), encoding="utf-8", newline="\n")
    SUBCYCLING_PATH.write_text(rows_to_csv(sub_rows), encoding="utf-8", newline="\n")
    write_summary(scan_rows, sub_rows)
    plot_bead_scan(scan_rows)
    plot_subcycling_scan(sub_rows)

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {SUBCYCLING_FIGURE_PATH}")
    print(f"wrote {SCAN_PATH}")
    print(f"wrote {SUBCYCLING_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
