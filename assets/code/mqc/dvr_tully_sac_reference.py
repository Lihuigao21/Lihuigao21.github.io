"""Finite-grid DVR quantum reference for Tully's simple avoided crossing.

The Hamiltonian is built in a two-channel sinc-DVR basis and diagonalized once.
Each initial wavepacket is then propagated by exact phase factors in that
finite Hilbert space. Run from the repository root:

    python assets/code/mqc/dvr_tully_sac_reference.py

The script writes CSV tables to assets/img/mqc-series. The FSSH and Ehrenfest
plotting scripts read those tables when available and overlay the quantum
reference curves.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tully_common import MASS, PARAMS


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def diabatic_arrays(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax = np.abs(x)
    sign = np.where(x >= 0.0, 1.0, -1.0)
    v11 = sign * PARAMS.a * (1.0 - np.exp(-PARAMS.b * ax))
    v22 = -v11
    v12 = PARAMS.c * np.exp(-PARAMS.d * x * x)
    return v11, v22, v12


def sinc_dvr_kinetic(n: int, dx: float, mass: float = MASS) -> np.ndarray:
    """Colbert-Miller infinite-order kinetic-energy matrix."""

    idx = np.arange(n)
    diff = idx[:, None] - idx[None, :]
    kinetic = np.empty((n, n), dtype=float)
    prefactor = 1.0 / (2.0 * mass * dx * dx)
    kinetic[diff == 0] = prefactor * (np.pi * np.pi / 3.0)
    mask = diff != 0
    kinetic[mask] = prefactor * (2.0 * (-1.0) ** diff[mask] / (diff[mask] ** 2))
    return kinetic


def build_hamiltonian(x: np.ndarray) -> np.ndarray:
    n = x.size
    dx = float(x[1] - x[0])
    kinetic = sinc_dvr_kinetic(n, dx)
    v11, v22, v12 = diabatic_arrays(x)

    hamiltonian = np.zeros((2 * n, 2 * n), dtype=float)
    hamiltonian[:n, :n] = kinetic + np.diag(v11)
    hamiltonian[n:, n:] = kinetic + np.diag(v22)
    hamiltonian[:n, n:] = np.diag(v12)
    hamiltonian[n:, :n] = np.diag(v12)
    return hamiltonian


def initial_packet(x: np.ndarray, p0: float, x0: float = -10.0, sigma: float = 1.0) -> np.ndarray:
    """Return a normalized two-channel DVR coefficient vector."""

    dx = float(x[1] - x[0])
    envelope = (1.0 / (2.0 * np.pi * sigma * sigma)) ** 0.25
    wave = envelope * np.exp(-((x - x0) ** 2) / (4.0 * sigma * sigma) + 1j * p0 * (x - x0))
    coeff = np.zeros(2 * x.size, dtype=complex)
    coeff[: x.size] = np.sqrt(dx) * wave
    coeff /= np.linalg.norm(coeff)
    return coeff


def adiabatic_projection_probabilities(x: np.ndarray, coeff: np.ndarray) -> dict[str, float]:
    """Integrate final wavepacket probability by side and adiabatic state."""

    n = x.size
    c0 = coeff[:n]
    c1 = coeff[n:]
    v11, _, v12 = diabatic_arrays(x)

    t_lower = 0.0
    t_upper = 0.0
    r_lower = 0.0
    r_upper = 0.0
    center = 0.0

    for i, xi in enumerate(x):
        matrix = np.array([[v11[i], v12[i]], [v12[i], -v11[i]]], dtype=float)
        _, vecs = np.linalg.eigh(matrix)
        local = np.array([c0[i], c1[i]])
        amp = vecs.T.conjugate() @ local
        lower = float(abs(amp[0]) ** 2)
        upper = float(abs(amp[1]) ** 2)
        if xi > 2.0:
            t_lower += lower
            t_upper += upper
        elif xi < -2.0:
            r_lower += lower
            r_upper += upper
        else:
            center += lower + upper

    return {
        "T_lower_exact": t_lower,
        "T_upper_exact": t_upper,
        "R_lower_exact": r_lower,
        "R_upper_exact": r_upper,
        "center_residual": center,
        "norm": t_lower + t_upper + r_lower + r_upper + center,
    }


def propagate_reference(
    p_values: list[float],
    n_grid: int = 768,
    x_min: float = -32.0,
    x_max: float = 32.0,
    x0: float = -10.0,
    sigma: float = 1.0,
    x_target: float = 20.0,
) -> list[dict[str, float]]:
    """Diagonalize the DVR Hamiltonian and propagate each initial momentum."""

    x = np.linspace(x_min, x_max, n_grid)
    hamiltonian = build_hamiltonian(x)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    rows: list[dict[str, float]] = []
    for p0 in p_values:
        time = max(1800.0, (x_target - x0) * MASS / p0)
        psi0 = initial_packet(x, p0=p0, x0=x0, sigma=sigma)
        amplitudes = eigenvectors.T.conjugate() @ psi0
        phase = np.exp(-1j * eigenvalues * time)
        final = eigenvectors @ (phase * amplitudes)
        probs = adiabatic_projection_probabilities(x, final)
        probs["p0"] = p0
        probs["time"] = time
        probs["T_total_exact"] = probs["T_lower_exact"] + probs["T_upper_exact"]
        probs["R_total_exact"] = probs["R_lower_exact"] + probs["R_upper_exact"]
        probs["adiabatic_upper_exact"] = probs["T_upper_exact"] + probs["R_upper_exact"]
        rows.append(probs)
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    out_dir = repo / "assets" / "img" / "mqc-series"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_values = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    rows = propagate_reference(p_values)
    out_path = out_dir / "dvr-tully-sac-reference.csv"
    summary_path = out_dir / "dvr-tully-sac-reference-summary.csv"
    write_csv(out_path, rows)

    p20 = min(rows, key=lambda row: abs(row["p0"] - 20.0))
    write_csv(
        summary_path,
        [
            {
                "n_grid": 768,
                "x_min": -32.0,
                "x_max": 32.0,
                "x0": -10.0,
                "sigma": 1.0,
                "p20_T_upper_exact": p20["T_upper_exact"],
                "p20_R_total_exact": p20["R_total_exact"],
                "p20_center_residual": p20["center_residual"],
                "p20_norm": p20["norm"],
            }
        ],
    )

    print(f"wrote {out_path}")
    print(
        "p0=20: "
        f"T_upper_exact={p20['T_upper_exact']:.3f}, "
        f"R_total_exact={p20['R_total_exact']:.3e}, "
        f"center_residual={p20['center_residual']:.3e}"
    )


if __name__ == "__main__":
    main()
