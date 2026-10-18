"""Reproduce the finite-lattice momentum-selection diagnostic.

Run from the website root:
    python assets/code/hefeinamd_k_selection.py

Requires NumPy and Matplotlib. The model is a dimensionless, 16-site
periodic lattice with a diagonal perturbation in the site basis.
It checks a Fourier identity, not a DFT calculation or an NAMD algorithm.
"""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "assets" / "img" / "hefeinamd-series"
    output.mkdir(parents=True, exist_ok=True)
    sites = 16
    mode = 3
    j = np.arange(sites)
    # Column p is the normalized plane wave exp(i 2*pi*p*j/N).
    fourier = np.exp(2j * np.pi * np.outer(j, j) / sites) / np.sqrt(sites)
    potentials = [np.ones(sites), np.cos(2 * np.pi * mode * j / sites)]
    matrices = [fourier.conj().T @ (v[:, None] * fourier) for v in potentials]
    exact = [np.eye(sites), np.zeros((sites, sites))]
    for p in range(sites):
        exact[1][(p + mode) % sites, p] = 0.5
        exact[1][(p - mode) % sites, p] = 0.5

    errors = [float(np.max(np.abs(m - e))) for m, e in zip(matrices, exact)]
    forbidden = [
        float(np.max(np.abs(m[e == 0]))) for m, e in zip(matrices, exact)
    ]
    hermiticity = [
        float(np.max(np.abs(m - m.conj().T))) for m in matrices
    ]
    assert max(errors) < 1e-13
    assert max(hermiticity) < 1e-13
    assert np.max(np.abs(fourier.conj().T @ fourier - np.eye(sites))) < 1e-13

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), layout="constrained")
    titles = [r"$V_j=1$ ($r=0$)", r"$V_j=\cos(2\pi\,3j/16)$ ($r=\pm 3$)"]
    for ax, matrix, title in zip(axes, matrices, titles):
        plot = ax.imshow(
            np.abs(matrix), origin="lower", cmap="cividis", vmin=0, vmax=1,
            extent=(-0.5, sites - 0.5, -0.5, sites - 0.5),
            interpolation="nearest",
        )
        ax.set(title=title, xlabel=r"Initial momentum index $p$",
               ylabel=r"Final momentum index $p'$")
        ax.set_xticks([0, 4, 8, 12, 15])
        ax.set_yticks([0, 4, 8, 12, 15])
    fig.colorbar(plot, ax=axes, shrink=0.86, label=r"$|\langle p'|V|p\rangle|$")
    svg = output / "momentum-selection.svg"
    fig.savefig(svg, metadata={"Date": None})
    plt.close(fig)
    # Normalize generated text explicitly for Git publication on Windows.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n", encoding="utf-8", newline="\n",
    )
    with (output / "momentum-selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["perturbation", "p_final", "p_initial", "real", "imag"])
        for name, matrix in zip(["uniform", "cosine_q3"], matrices):
            for pf in range(sites):
                for pi in range(sites):
                    writer.writerow([name, pf, pi, f"{matrix[pf, pi].real:.17g}",
                                     f"{matrix[pf, pi].imag:.17g}"])
    result = {
        "sites": sites, "finite_q_index": mode, "amplitude": 1,
        "maximum_absolute_error": dict(zip(["uniform", "cosine_q3"], errors)),
        "maximum_forbidden_element": dict(zip(["uniform", "cosine_q3"], forbidden)),
        "maximum_hermiticity_error": max(hermiticity),
        "scope": "Discrete Fourier selection rule only; no electronic propagation.",
    }
    (output / "momentum-selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
