"""Compare closed SHO correlations with a thermal energy-basis trace.

Units: hbar=m=Omega=1, so ell^2=1/2. Positive Fourier frequency
uses exp(-i omega t); the ordered correlation is <A A(t)>.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def closed(x, s, t, observable):
    n = 1.0 / np.expm1(x)
    if observable == "q":
        return 0.5 * ((n + 1) * np.exp(-s * x + 1j * t)
                      + n * np.exp(s * x - 1j * t))
    return 0.25 * (4 * n * (n + 1)
                   + 2 * ((n + 1)**2 * np.exp(-2 * s * x + 2j * t)
                          + n**2 * np.exp(2 * s * x - 2j * t)))


def kubo_closed(x, t, observable):
    if observable == "q":
        return np.cos(t) / x
    n = 1.0 / np.expm1(x)
    return 0.25 * (4 * n * (n + 1)
                   + 2 * (2 * n + 1) / x * np.cos(2 * t))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sho-correlation-results"))
    args = parser.parse_args()
    dim = 180
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    q = (a + a.T) / np.sqrt(2)
    energies = np.arange(dim) + 0.5
    energy_difference = energies[None, :] - energies[:, None]
    errors = []
    average_errors = []
    xs = [0.2, 0.7, 2.0, 6.0, 12.0]
    ss = [0.0, 0.17, 0.5, 0.83, 1.0]
    times = [0.0, 0.13, 1.4, np.pi / 2, 4.1]
    quadrature_s, quadrature_w = leggauss(64)
    quadrature_s = (quadrature_s + 1) / 2
    quadrature_w = quadrature_w / 2
    for x in xs:
        log_p = -x * energies
        log_p -= np.log(np.exp(log_p).sum())
        p = np.exp(log_p)
        q2 = q @ q
        q2 -= np.eye(dim) * np.dot(p, np.diag(q2))
        for label, operator in [("q", q), ("delta_q2", q2)]:
            for s in ss:
                weight = np.exp((1 - s) * log_p[:, None] + s * log_p[None, :])
                for t in times:
                    trace = np.sum(weight * operator**2 * np.exp(1j * energy_difference * t))
                    expected = closed(x, s, t, label)
                    errors.append(abs(trace - expected) / max(1, abs(expected)))
            for t in times:
                average = sum(w * closed(x, s, t, label)
                              for s, w in zip(quadrature_s, quadrature_w))
                expected = kubo_closed(x, t, label)
                average_errors.append(abs(average - expected) / max(1, abs(expected)))
    result = {
        "status": "passed",
        "energy_basis_dimension": dim,
        "temperatures_x": xs,
        "split_parameters": ss,
        "times_theta": times,
        "observable_labels": ["q", "delta_q2"],
        "trace_comparisons": len(errors),
        "maximum_relative_or_absolute_trace_error": float(max(errors)),
        "maximum_relative_or_absolute_kubo_integral_error": float(max(average_errors)),
        "scope": "Closed formulas checked against an independent truncated thermal trace and s integration; not a test of RPMD."
    }
    assert max(errors) < 1e-10, result
    assert max(average_errors) < 1e-10, result
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sho-correlations-check.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
