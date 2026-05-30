"""Plot PES and adiabatic NACs for the three-state Morse Model A.

This standalone diagnostic mirrors the parameters in
``toymodel.model.TSM.ThreeStateMorse``.  It writes a PNG figure and a CSV table
for the adiabatic potential-energy surfaces, diabatic matrix elements, and
adiabatic derivative couplings.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PARAMS = {
    "d1": 0.02,
    "alpha1": 0.4,
    "r1": 4.0,
    "c1": 0.02,
    "d2": 0.02,
    "alpha2": 0.65,
    "r2": 4.5,
    "c2": 0.0,
    "d3": 0.003,
    "alpha3": 0.65,
    "r3": 6.0,
    "c3": 0.02,
    "a12": 0.005,
    "a13": 0.005,
    "a23": 0.0,
    "alpha12": 32.0,
    "alpha13": 32.0,
    "alpha23": 0.0,
    "r12": 3.40,
    "r13": 4.97,
    "r23": 0.0,
}


def morse(x, d, alpha, r, c):
    return d * (1.0 - np.exp(-alpha * (x - r))) ** 2 + c


def dmorse(x, d, alpha, r):
    ex = np.exp(-alpha * (x - r))
    return 2.0 * d * alpha * (ex - ex**2)


def gaussian_coupling(x, a, alpha, r):
    return a * np.exp(-alpha * (x - r) ** 2)


def d_gaussian_coupling(x, a, alpha, r):
    return -2.0 * a * alpha * (x - r) * np.exp(-alpha * (x - r) ** 2)


def diabatic_matrices(x):
    x = np.asarray(x, dtype=float)
    V = np.zeros((len(x), 3, 3), dtype=float)
    dV = np.zeros_like(V)

    diag_specs = [
        ("d1", "alpha1", "r1", "c1"),
        ("d2", "alpha2", "r2", "c2"),
        ("d3", "alpha3", "r3", "c3"),
    ]
    for i, (d_key, alpha_key, r_key, c_key) in enumerate(diag_specs):
        V[:, i, i] = morse(
            x,
            PARAMS[d_key],
            PARAMS[alpha_key],
            PARAMS[r_key],
            PARAMS[c_key],
        )
        dV[:, i, i] = dmorse(x, PARAMS[d_key], PARAMS[alpha_key], PARAMS[r_key])

    pair_specs = [
        (0, 1, "a12", "alpha12", "r12"),
        (0, 2, "a13", "alpha13", "r13"),
        (1, 2, "a23", "alpha23", "r23"),
    ]
    for i, j, a_key, alpha_key, r_key in pair_specs:
        V[:, i, j] = V[:, j, i] = gaussian_coupling(
            x,
            PARAMS[a_key],
            PARAMS[alpha_key],
            PARAMS[r_key],
        )
        dV[:, i, j] = dV[:, j, i] = d_gaussian_coupling(
            x,
            PARAMS[a_key],
            PARAMS[alpha_key],
            PARAMS[r_key],
        )

    return V, dV


def adiabatic_surfaces_and_nacs(x):
    V, dV = diabatic_matrices(x)
    energies = np.zeros((len(x), 3), dtype=float)
    coeffs = np.zeros((len(x), 3, 3), dtype=float)
    nac = np.zeros((len(x), 3, 3), dtype=float)

    for ipoint in range(len(x)):
        values, vectors = np.linalg.eigh(V[ipoint])
        if ipoint > 0:
            for istate in range(3):
                if np.dot(vectors[:, istate], coeffs[ipoint - 1, :, istate]) < 0.0:
                    vectors[:, istate] *= -1.0
        energies[ipoint] = values
        coeffs[ipoint] = vectors

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            gap = energies[:, j] - energies[:, i]
            safe_gap = gap.copy()
            tiny = np.abs(safe_gap) < 1.0e-10
            safe_gap[tiny] = np.where(safe_gap[tiny] >= 0.0, 1.0e-10, -1.0e-10)
            nac[:, i, j] = np.array(
                [
                    (coeffs[k, :, i] @ dV[k] @ coeffs[k, :, j]) / safe_gap[k]
                    for k in range(len(x))
                ],
                dtype=float,
            )

    return V, energies, nac


def write_csv(path, x, V, energies, nac):
    fields = [
        "R",
        "E1",
        "E2",
        "E3",
        "V11",
        "V22",
        "V33",
        "V12",
        "V13",
        "V23",
        "nac12",
        "nac13",
        "nac23",
        "abs_nac12",
        "abs_nac13",
        "abs_nac23",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k, r_value in enumerate(x):
            row = {
                "R": float(r_value),
                "E1": float(energies[k, 0]),
                "E2": float(energies[k, 1]),
                "E3": float(energies[k, 2]),
                "V11": float(V[k, 0, 0]),
                "V22": float(V[k, 1, 1]),
                "V33": float(V[k, 2, 2]),
                "V12": float(V[k, 0, 1]),
                "V13": float(V[k, 0, 2]),
                "V23": float(V[k, 1, 2]),
                "nac12": float(nac[k, 0, 1]),
                "nac13": float(nac[k, 0, 2]),
                "nac23": float(nac[k, 1, 2]),
                "abs_nac12": float(abs(nac[k, 0, 1])),
                "abs_nac13": float(abs(nac[k, 0, 2])),
                "abs_nac23": float(abs(nac[k, 1, 2])),
            }
            writer.writerow(row)


def plot(path, x, V, energies, nac, args):
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True)

    ax = axes[0]
    for istate, color in enumerate(colors):
        ax.plot(
            x,
            energies[:, istate],
            color=color,
            lw=2.2,
            label=f"Adiabatic E{istate + 1}",
        )
        ax.plot(
            x,
            V[:, istate, istate],
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.58,
            label=f"Diabatic V{istate + 1}{istate + 1}",
        )
    ax.axvline(args.R0, color="#111111", lw=1.1, ls=":", alpha=0.65)
    ax.text(args.R0 + 0.04, 0.061, "R0", fontsize=9, color="#111111")
    ax.set_ylabel("Energy (a.u.)")
    ax.set_ylim(args.energy_min, args.energy_max)
    ax.set_title("Three-state Morse Model A: PES and adiabatic NAC")
    ax.grid(alpha=0.22, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=2, fontsize=8, loc="upper right")

    ax = axes[1]
    pair_styles = [
        ((0, 1), "#9467bd", "|d12|"),
        ((0, 2), "#8c564b", "|d13|"),
        ((1, 2), "#ff7f0e", "|d23|"),
    ]
    for (i, j), color, label in pair_styles:
        ax.plot(x, np.abs(nac[:, i, j]), color=color, lw=2.0, label=label)
    for center, label in ((PARAMS["r12"], "r12"), (PARAMS["r13"], "r13")):
        ax.axvline(center, color="#777777", lw=1.0, ls=":", alpha=0.55)
        ax.text(center + 0.04, args.nac_max * 0.88, label, fontsize=9, color="#555555")
    ax.set_xlabel("Nuclear coordinate R (a.u.)")
    ax.set_ylabel("Adiabatic NAC |dij| (a.u.^-1)")
    ax.set_ylim(0.0, args.nac_max)
    ax.grid(alpha=0.22, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_args():
    here = Path(__file__).resolve()
    default_output = here.parents[2] / "img" / "mqc-series"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmin", type=float, default=1.8)
    parser.add_argument("--xmax", type=float, default=8.2)
    parser.add_argument("--npoints", type=int, default=2000)
    parser.add_argument("--R0", type=float, default=2.1)
    parser.add_argument("--energy-min", type=float, default=-0.002)
    parser.add_argument("--energy-max", type=float, default=0.065)
    parser.add_argument("--nac-max", type=float, default=5.8)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(args.xmin, args.xmax, args.npoints)
    V, energies, nac = adiabatic_surfaces_and_nacs(x)
    png_path = args.output_dir / "three-state-morse-pes-nac.png"
    csv_path = args.output_dir / "three-state-morse-pes-nac.csv"
    write_csv(csv_path, x, V, energies, nac)
    plot(png_path, x, V, energies, nac, args)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
