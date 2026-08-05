"""Algorithmic core of the N=1 forward/surface flux-side comparison.

This compact example isolates the sampling identities used in the article.  A
production calculation must supply energy-resolved FSSH transmission
probabilities and physical/proposal hop probabilities from its own trajectory
engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class Estimate:
    value: float
    standard_error: float


def forward_thermal_rate(
    transmission_probability: Callable[[float], float],
    *,
    beta: float,
    mass: float,
    threshold: float,
    quadrature_order: int,
) -> float:
    """Gauss--Laguerre form of the reactant-launched thermal rate."""

    nodes, weights = np.polynomial.laguerre.laggauss(quadrature_order)
    energies = threshold + nodes / beta
    probabilities = np.asarray(
        [transmission_probability(float(energy)) for energy in energies]
    )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("transmission probabilities must lie in [0, 1]")
    free_flux_per_state = 0.5 / np.sqrt(2.0 * np.pi * mass * beta)
    return float(
        free_flux_per_state
        * np.exp(-beta * threshold)
        * np.dot(weights, probabilities)
    )


def positive_centroid_canonical_momenta(
    *, mass: float, beta: float, nbeads: int, rng: np.random.Generator
) -> np.ndarray:
    """Canonical bead momenta conditioned on positive centroid momentum."""

    momenta = rng.normal(0.0, np.sqrt(mass * nbeads / beta), nbeads)
    centroid = float(np.mean(momenta))
    if centroid < 0.0:
        momenta -= 2.0 * centroid
    return momenta


def path_likelihood_ratio(
    target_event_probabilities: Iterable[float],
    proposal_event_probabilities: Iterable[float],
) -> float:
    """Return W=P(history|incoming)/Q(history) from stepwise event factors."""

    target = np.asarray(list(target_event_probabilities), dtype=float)
    proposal = np.asarray(list(proposal_event_probabilities), dtype=float)
    if target.shape != proposal.shape or np.any(target < 0.0):
        raise ValueError("target and proposal arrays must be compatible")
    if np.any(proposal <= 0.0):
        raise ValueError("the proposal must cover the target history support")
    log_weight = np.sum(np.log(target) - np.log(proposal))
    return float(np.exp(log_weight))


def surface_transmission_coefficient(
    centroid_momenta: np.ndarray,
    *,
    mass: float,
    reactive: np.ndarray,
    history_weights: np.ndarray,
    positive_crossings: np.ndarray,
) -> Estimate:
    """Tully Eq. (50): positive flux, history weight, and 1/n+ correction."""

    pbar = np.asarray(centroid_momenta, dtype=float)
    reactive = np.asarray(reactive, dtype=bool)
    weights = np.asarray(history_weights, dtype=float)
    crossings = np.asarray(positive_crossings, dtype=int)
    if not (pbar.shape == reactive.shape == weights.shape == crossings.shape):
        raise ValueError("all trajectory arrays must have the same shape")
    if np.any(pbar <= 0.0) or np.any(weights < 0.0):
        raise ValueError("positive flux and nonnegative history weights required")
    if np.any(reactive & (crossings < 1)):
        raise ValueError("every reactive path needs a positive crossing")

    corrected = np.where(
        reactive, weights / np.maximum(crossings, 1), 0.0
    )
    flux = pbar / mass
    contributions = flux * corrected
    value = float(np.mean(contributions) / np.mean(flux))

    # Independent-block uncertainty; production runs should use multiple seeds.
    nblocks = min(10, pbar.size)
    blocks = np.array_split(np.arange(pbar.size), nblocks)
    block_values = np.asarray(
        [np.mean(contributions[b]) / np.mean(flux[b]) for b in blocks]
    )
    standard_error = float(np.std(block_values, ddof=1) / np.sqrt(nblocks))
    return Estimate(value, standard_error)


def compare(a: Estimate, b: Estimate) -> dict[str, float | bool]:
    """Combined-standard-error and relative-effect convergence diagnostics."""

    combined = float(np.hypot(a.standard_error, b.standard_error))
    difference = b.value - a.value
    scale = max(abs(a.value), abs(b.value))
    relative = abs(difference) / scale if scale else 0.0
    return {
        "difference": difference,
        "combined_standard_error": combined,
        "z": difference / combined if combined else 0.0,
        "relative_difference": relative,
        "within_2sigma": abs(difference) <= 2.0 * combined,
        "within_10_percent": relative <= 0.10,
    }


def _self_check() -> None:
    beta, mass, threshold = 4.0, 2000.0, 0.02
    constant_probability = 0.37
    expected = (
        0.5
        / np.sqrt(2.0 * np.pi * mass * beta)
        * np.exp(-beta * threshold)
        * constant_probability
    )
    calculated = forward_thermal_rate(
        lambda _energy: constant_probability,
        beta=beta,
        mass=mass,
        threshold=threshold,
        quadrature_order=16,
    )
    np.testing.assert_allclose(calculated, expected, rtol=2e-14)

    rng = np.random.default_rng(7)
    centroids = np.asarray(
        [
            np.mean(
                positive_centroid_canonical_momenta(
                    mass=mass, beta=beta, nbeads=1, rng=rng
                )
            )
            for _ in range(20_000)
        ]
    )
    assert np.all(centroids >= 0.0)
    expected_half_normal_mean = np.sqrt(2.0 * mass / (np.pi * beta))
    np.testing.assert_allclose(
        np.mean(centroids), expected_half_normal_mean, rtol=0.02
    )

    np.testing.assert_allclose(
        path_likelihood_ratio([0.2, 0.8], [0.4, 0.5]), 0.8
    )
    estimate = surface_transmission_coefficient(
        np.ones(20),
        mass=1.0,
        reactive=np.ones(20, dtype=bool),
        history_weights=np.ones(20),
        positive_crossings=np.ones(20, dtype=int),
    )
    np.testing.assert_allclose(estimate.value, 1.0)
    print("all N=1 flux-side estimator checks passed")


if __name__ == "__main__":
    _self_check()
