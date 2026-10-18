"""Four-cell folding and symmetry-breaking diagnostic.

Run from the website root:
    python assets/code/hefeinamd_supercell_folding.py

Requires NumPy and Matplotlib. All examples are dimensionless. This script
checks folding, a monatomic harmonic chain, and Fourier matrix elements;
it does not simulate DFT, electronic dynamics, or surface hopping.
"""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n", encoding="utf-8", newline="\n",
    )


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "assets/img/hefeinamd-series"
    out.mkdir(parents=True, exist_ok=True)
    count = 4
    j = np.arange(count)
    q_over_b = j / count
    waves = np.exp(2j * np.pi * np.outer(j, j) / count) / np.sqrt(count)
    eye = np.eye(count)
    dynamical = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    transformed = waves.conj().T @ dynamical @ waves
    omega = 2 * np.abs(np.sin(np.pi * q_over_b))
    expected_dynamical = np.diag(omega ** 2)
    # Independent analytic Fourier selection rule for a real r=1 cosine.
    potential = np.cos(2 * np.pi * j / count)
    coupling = waves.conj().T @ (potential[:, None] * waves)
    expected_coupling = np.zeros((count, count))
    for initial in range(count):
        expected_coupling[(initial + 1) % count, initial] = 0.5
        expected_coupling[(initial - 1) % count, initial] = 0.5
    uniform = waves.conj().T @ waves
    errors = {
        "dynamical_matrix": float(np.max(np.abs(transformed - expected_dynamical))),
        "dynamical_eigenvalues": float(np.max(np.abs(
            np.linalg.eigvalsh(dynamical) - np.sort(omega ** 2)
        ))),
        "uniform_perturbation": float(np.max(np.abs(uniform - eye))),
        "finite_q_perturbation": float(np.max(np.abs(coupling - expected_coupling))),
    }
    assert max(errors.values()) < 1e-13
    assert np.allclose(np.exp(2j * np.pi * q_over_b * count), 1, atol=1e-14)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    mesh = np.linspace(0, 1, 401)
    ax.plot(mesh, 2 * np.abs(np.sin(np.pi * mesh)), color="#315a73",
            label="Primitive-cell dispersion")
    ax.scatter(q_over_b, omega, color="#a35d36", zorder=3,
               label="Modes at supercell Gamma")
    for r, x, y in zip(j, q_over_b, omega):
        ax.annotate(f"r={r}", (x, y), xytext=(8, 8), textcoords="offset points")
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.12, 2.32),
           xlabel=r"Primitive wavevector $q/b$, with $b=2\pi/a$",
           ylabel=r"Frequency $\omega/\omega_0$")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.1), fontsize=10)
    save_svg(fig, out / "supercell-phonon-folding.svg")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    for ax, matrix, title in zip(
        axes, [uniform, coupling], [r"Uniform $V_j=1$", r"$V_j=\cos(2\pi j/4)$"]
    ):
        im = ax.imshow(np.abs(matrix), origin="lower", cmap="cividis",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set(title=title, xlabel="Initial primitive k index",
               ylabel="Final primitive k index")
        ax.set_xticks(j)
        ax.set_yticks(j)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Absolute matrix element")
    save_svg(fig, out / "supercell-perturbation-matrix.svg")

    with (out / "supercell-folding.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["r", "primitive_q_over_b", "supercell_Q", "omega_over_omega0"])
        for r, q, frequency in zip(j, q_over_b, omega):
            writer.writerow([r, f"{q:.17g}", 0, f"{frequency:.17g}"])
    report = {"primitive_cells": count, "omega0": 1,
              "maximum_absolute_errors": errors,
              "scope": "Folding and perturbation-matrix checks only; no NAMD propagation."}
    (out / "supercell-folding.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
