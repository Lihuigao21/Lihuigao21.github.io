"""Four-site checks for a linear EPC Hamiltonian in a fixed Bloch basis.

Run from the website root:
    python assets/code/hefeinamd_epc_hamiltonian.py

Requires NumPy and Matplotlib. All coordinates, times and energies below are
dimensionless. This is an algebra and expansion-error diagnostic, not a
material calculation or a reproduction of the HefeiNAMD hopping algorithm.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    path.write_text(
        "\n".join(line.rstrip() for line in
                  path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )


def save_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "assets/img/hefeinamd-series"
    output.mkdir(parents=True, exist_ok=True)
    count = 4
    sites = np.arange(count)
    wavevectors = 2 * np.pi * np.arange(count) / count
    fourier = np.exp(1j * np.outer(sites, wavevectors)) / np.sqrt(count)
    identity = np.eye(count)
    reference_site = -np.roll(identity, 1, axis=1) - np.roll(identity, -1, axis=1)
    reference_bloch = np.diag(-2 * np.cos(wavevectors))
    linear_coefficient = 0.4
    quadratic_coefficient = 0.25
    amplitude = 0.4
    times = np.linspace(0, 2 * np.pi, 201)
    rows = []
    matrix_error = 0.0
    hermiticity_error = 0.0
    selected_error = 0.0
    for tau in times:
        displacement = amplitude * np.cos(np.pi * sites / 2 - tau)
        site_hamiltonian = reference_site + np.diag(linear_coefficient * displacement)
        direct_bloch = fourier.conj().T @ site_hamiltonian @ fourier
        # Independent Fourier selection rule: only q=+/-pi/2 is present.
        epc_bloch = reference_bloch.astype(complex).copy()
        for initial in range(count):
            epc_bloch[(initial + 1) % count, initial] += (
                linear_coefficient * amplitude / 2 * np.exp(-1j * tau)
            )
            epc_bloch[(initial - 1) % count, initial] += (
                linear_coefficient * amplitude / 2 * np.exp(1j * tau)
            )
        matrix_error = max(matrix_error, float(np.max(np.abs(direct_bloch - epc_bloch))))
        hermiticity_error = max(
            hermiticity_error, float(np.max(np.abs(epc_bloch - epc_bloch.conj().T)))
        )
        expected = linear_coefficient * amplitude / 2 * np.exp(-1j * tau)
        selected_error = max(selected_error, float(abs(direct_bloch[1, 0] - expected)))
        rows.append([float(tau), float(direct_bloch[1, 0].real),
                     float(direct_bloch[1, 0].imag), float(abs(direct_bloch[1, 0]))])

    amplitudes = np.array([0.025, 0.05, 0.1, 0.2, 0.4, 0.8])
    truncation_errors = []
    for value in amplitudes:
        displacement = value * np.cos(np.pi * sites / 2)
        linear = reference_site + np.diag(linear_coefficient * displacement)
        full = linear + np.diag(quadratic_coefficient * displacement ** 2)
        truncation_errors.append(float(np.linalg.norm(full - linear, ord=2)))
    truncation_errors = np.array(truncation_errors)
    expected_errors = quadratic_coefficient * amplitudes ** 2
    error_law_residual = float(np.max(np.abs(truncation_errors - expected_errors)))
    fitted_slope = float(np.polyfit(np.log(amplitudes), np.log(truncation_errors), 1)[0])
    assert max(matrix_error, hermiticity_error, selected_error, error_law_residual) < 1e-13
    assert abs(fitted_slope - 2) < 1e-10
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "epc-hamiltonian"})
    data = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    ax.plot(data[:, 0], data[:, 1], color="#315a73", label=r"Re $H_{1,0}$")
    ax.plot(data[:, 0], data[:, 2], color="#a35d36", linestyle="--", label=r"Im $H_{1,0}$")
    ax.plot(data[:, 0], data[:, 3], color="#64765b", linestyle=":", label=r"$|H_{1,0}|$")
    ax.set(xlabel=r"Mode phase $\tau=\omega t$",
           ylabel=r"Matrix element / $E_0$", xlim=(0, 2 * np.pi))
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.legend(loc="lower left", ncol=3, fontsize=10)
    save_svg(fig, output / "epc-hamiltonian-time.svg")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    ax.loglog(amplitudes, truncation_errors, "o", color="#315a73",
              label="Direct matrix-norm error", zorder=3)
    ax.loglog(amplitudes, expected_errors, "--", color="#a35d36",
              label=r"Analytic $\eta A^2$")
    ax.set(xlabel=r"Displacement amplitude $A$",
           ylabel=r"$\|H_{\rm full}-H_{\rm linear}\|_2/E_0$")
    ax.legend(loc="upper left", fontsize=10)
    save_svg(fig, output / "epc-linearization-error.svg")
    save_csv(output / "epc-hamiltonian-time.csv",
             ["omega_t", "real_H10_over_E0", "imag_H10_over_E0", "abs_H10_over_E0"], rows)
    save_csv(output / "epc-linearization-error.csv",
             ["amplitude", "direct_error_over_E0", "analytic_error_over_E0"],
             zip(amplitudes, truncation_errors, expected_errors))
    report = {
        "sites": count, "linear_coefficient_over_E0": linear_coefficient,
        "quadratic_coefficient_over_E0": quadratic_coefficient, "time_plot_amplitude": amplitude,
        "maximum_absolute_errors": {
            "independent_Hamiltonian_constructions": matrix_error,
            "Hermiticity": hermiticity_error,
            "selected_matrix_element": selected_error,
            "quadratic_error_law": error_law_residual,
        },
        "fitted_error_power": fitted_slope,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Fixed-basis linear Hamiltonian and Taylor truncation only; no dynamics or material validation.",
    }
    (output / "epc-hamiltonian-check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
