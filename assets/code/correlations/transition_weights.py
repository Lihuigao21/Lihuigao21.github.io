"""Check ordered/Kubo weights and the second-order transition probability.

Units: hbar=1. Bath energies are [0, 1], beta=2, and B=sigma_x.
D(t)=<B(t)B(0)> uses exp(+i omega t) for its Fourier transform.
The reversed C(t)=<B(0)B(t)> with exp(-i omega t) has the same spectrum.
This checks a perturbative identity, not exact coupled-system dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def integrate_interval(order, upper):
    nodes, weights = leggauss(order)
    return (nodes + 1) * upper / 2, weights * upper / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("transition-weight-results"))
    args = parser.parse_args()
    beta = 2.0
    energies = np.array([0.0, 1.0])
    operator = np.array([[0.0, 1.0], [1.0, 0.0]])
    probabilities = np.exp(-beta * energies)
    probabilities /= probabilities.sum()
    rho = np.diag(probabilities)
    # Rows are initial m; columns are final n.
    gaps = energies[None, :] - energies[:, None]
    ordered_weights = probabilities[:, None] * abs(operator.T)**2
    kubo_weights = np.empty_like(gaps)
    for m in range(2):
        for n in range(2):
            kubo_weights[m, n] = (
                probabilities[m] if gaps[m, n] == 0 else
                (probabilities[m] - probabilities[n]) / (beta * gaps[m, n])
            ) * abs(operator[n, m])**2

    def d_spectral(time):
        return np.sum(ordered_weights * np.exp(-1j * gaps * time),
                      axis=(-2, -1))

    lambdas, lambda_weights = integrate_interval(64, beta)
    times = [0.0, 0.13, 0.9, 2.4, 4.1]
    trace_errors, kubo_errors, convention_errors = [], [], []
    for time in times:
        evolution = np.diag(np.exp(1j * energies * time))
        bt = evolution @ operator @ evolution.conj().T
        d_trace = np.trace(rho @ bt @ operator)
        c_trace = np.trace(rho @ operator @ bt)
        trace_errors.append(abs(d_trace - d_spectral(time)))
        convention_errors.append(abs(c_trace - d_trace.conjugate()))
        quadrature = 0j
        partition = np.exp(-beta * energies).sum()
        for lam, weight in zip(lambdas, lambda_weights):
            left = np.diag(np.exp(-(beta - lam) * energies))
            right = np.diag(np.exp(-lam * energies))
            quadrature += weight * np.trace(left @ bt @ right @ operator)
        quadrature /= beta * partition
        spectral_k = np.sum(kubo_weights * np.exp(-1j * gaps * time))
        kubo_errors.append(abs(quadrature - spectral_k))

    conversion_errors = []
    for m, n in [(0, 1), (1, 0)]:
        y = beta * gaps[m, n]
        restored = y / (-np.expm1(-y)) * kubo_weights[m, n]
        conversion_errors.append(abs(restored - ordered_weights[m, n]))

    # Independent double-time quadrature versus squared analytic amplitudes.
    coupling = 0.03
    electron_gaps = [-1.0, 1.0, 0.37]
    horizons = [0.0, 0.13, 0.9, 2.4, 4.1]
    probability_errors = []
    refinement_errors = []
    records = []
    for delta in electron_gaps:
        for time in horizons:
            frequencies = delta + gaps
            integrals = np.empty_like(frequencies, dtype=complex)
            resonant = np.isclose(frequencies, 0, atol=1e-14, rtol=0)
            integrals[resonant] = time
            integrals[~resonant] = np.expm1(
                1j * frequencies[~resonant] * time
            ) / (1j * frequencies[~resonant])
            amplitude_probability = coupling**2 * np.sum(
                ordered_weights * abs(integrals)**2)
            values = []
            for order in [32, 64]:
                nodes, weights = integrate_interval(order, time)
                difference = nodes[:, None] - nodes[None, :]
                # D(tau'-tau), evaluated independently as a matrix trace sum.
                correlations = np.zeros_like(difference, dtype=complex)
                for a in range(2):
                    for b in range(2):
                        bt = operator[a, b] * np.exp(
                            1j * (energies[a] - energies[b]) * (-difference))
                        correlations += probabilities[a] * bt * operator[b, a]
                kernel = np.exp(1j * delta * difference) * correlations
                value = coupling**2 * np.sum(
                    weights[:, None] * weights[None, :] * kernel)
                values.append(value)
                probability_errors.append(abs(value - amplitude_probability))
            refinement_errors.append(abs(values[1] - values[0]))
            records.append({"electronic_gap": delta, "time": time,
                            "second_order_probability": float(amplitude_probability)})

    tolerance = 1e-12
    maxima = {
        "ordered_trace": float(max(trace_errors)),
        "kubo_imaginary_time_quadrature": float(max(kubo_errors)),
        "reversed_order_conjugacy": float(max(convention_errors)),
        "spectral_recovery": float(max(conversion_errors)),
        "finite_time_probability": float(max(probability_errors)),
        "time_quadrature_32_to_64": float(max(refinement_errors)),
    }
    assert all(value < tolerance for value in maxima.values()), maxima
    result = {
        "status": "passed", "units": "hbar=1", "beta": beta,
        "bath_energies": energies.tolist(), "coupling_operator": operator.tolist(),
        "initial_probabilities": probabilities.tolist(),
        "ordered_line_weights_bath_up_down": [float(ordered_weights[0, 1]),
                                              float(ordered_weights[1, 0])],
        "kubo_line_weights_bath_up_down": [float(kubo_weights[0, 1]),
                                           float(kubo_weights[1, 0])],
        "imaginary_time_quadrature_order": 64,
        "time_quadrature_orders": [32, 64], "correlation_times": times,
        "absolute_error_tolerance": tolerance, "maximum_absolute_errors": maxima,
        "coupling_g": coupling, "probability_checks": records,
        "scope": "Algebra and quadrature check of the second-order probability; no irreversible-rate or exact coupled-dynamics claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transition-weights-check.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: value for key, value in result.items()
                      if key != "probability_checks"}, indent=2))


if __name__ == "__main__":
    main()
