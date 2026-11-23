# SPDX-License-Identifier: GPL-3.0-only
"""Scalar autocorrelation and ESS routines from ToyModel's shape-scan sampler."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _trajectory_1d(samples: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional trajectory.")
    if values.size < 2:
        raise ValueError("samples must contain at least two values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite.")
    return values


def _validated_max_lag(max_lag: int | None, n_samples: int) -> int:
    if max_lag is None:
        return n_samples - 1
    if not isinstance(max_lag, (int, np.integer)):
        raise TypeError("max_lag must be an integer or None.")
    if max_lag < 0 or max_lag >= n_samples:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(samples).")
    return int(max_lag)


def autocorrelation_fft(
    samples: ArrayLike,
    *,
    max_lag: int | None = None,
    unbiased: bool = True,
) -> NDArray[np.float64]:
    """Return the normalized autocorrelation of a scalar trajectory.

    The calculation is ``O(n log n)`` and zero-pads the trajectory before the
    FFT, so the result is the linear rather than circular autocorrelation.
    With ``unbiased=True``, lag ``k`` is divided by ``n-k`` before
    normalization.  A constant trajectory has no defined normalized
    autocorrelation and is rejected.
    """

    values = _trajectory_1d(samples)
    lag_limit = _validated_max_lag(max_lag, values.size)
    centered = values - np.mean(values)
    squared_norm = float(np.dot(centered, centered))
    if squared_norm == 0.0:
        raise ValueError("autocorrelation is undefined for a constant trajectory.")

    fft_length = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    covariance_sum = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_length)
    covariance_sum = covariance_sum[: lag_limit + 1]
    if unbiased:
        autocovariance = covariance_sum / (
            values.size - np.arange(lag_limit + 1, dtype=float)
        )
    else:
        autocovariance = covariance_sum / values.size
    return np.asarray(autocovariance / autocovariance[0], dtype=float)


def _tau_initial_positive(autocorrelation: NDArray[np.float64]) -> tuple[float, int]:
    tau = 0.5
    window = 0
    for lag, correlation in enumerate(autocorrelation[1:], start=1):
        if correlation <= 0.0:
            break
        tau += float(correlation)
        window = lag
    return tau, window


def _tau_self_consistent(
    autocorrelation: NDArray[np.float64], window_factor: float
) -> tuple[float, int]:
    if not np.isfinite(window_factor) or window_factor <= 0.0:
        raise ValueError("window_factor must be positive and finite.")
    if autocorrelation.size == 1:
        return 0.5, 0

    running_tau = 0.5 + np.cumsum(autocorrelation[1:])
    lags = np.arange(1, autocorrelation.size)
    candidates = np.flatnonzero(
        (running_tau > 0.0) & (lags >= window_factor * running_tau)
    )
    if candidates.size:
        index = int(candidates[0])
    else:
        positive = np.flatnonzero(running_tau > 0.0)
        if not positive.size:
            raise ValueError("the estimated integrated autocorrelation time is nonpositive.")
        index = int(positive[-1])
    return float(running_tau[index]), int(lags[index])


def integrated_autocorrelation_time(
    samples: ArrayLike,
    *,
    method: str = "initial_positive",
    max_lag: int | None = None,
    window_factor: float = 5.0,
    return_window: bool = False,
) -> float | tuple[float, int]:
    """Estimate integrated autocorrelation time in units of retained samples.

    ``initial_positive`` truncates immediately before the first non-positive
    empirical autocorrelation.  ``self_consistent`` uses the first window
    satisfying ``window >= window_factor*tau``.  The latter is useful for
    smooth positive correlations such as an AR(1) oracle, while the former
    returns exactly ``tau=1/2`` as soon as the lag-one correlation is
    non-positive.
    """

    autocorrelation = autocorrelation_fft(samples, max_lag=max_lag)
    return integrated_autocorrelation_time_from_acf(
        autocorrelation,
        method=method,
        window_factor=window_factor,
        return_window=return_window,
    )


def integrated_autocorrelation_time_from_acf(
    autocorrelation: ArrayLike,
    *,
    method: str = "initial_positive",
    window_factor: float = 5.0,
    return_window: bool = False,
) -> float | tuple[float, int]:
    """Estimate IAT from a precomputed normalized autocorrelation array."""

    autocorrelation = np.asarray(autocorrelation, dtype=float)
    if autocorrelation.ndim != 1 or autocorrelation.size < 1:
        raise ValueError("autocorrelation must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(autocorrelation)):
        raise ValueError("autocorrelation must be finite.")
    if not np.isclose(autocorrelation[0], 1.0, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("autocorrelation[0] must equal one.")
    if method == "initial_positive":
        tau, window = _tau_initial_positive(autocorrelation)
    elif method == "self_consistent":
        tau, window = _tau_self_consistent(autocorrelation, window_factor)
    else:
        raise ValueError(
            "method must be 'initial_positive' or 'self_consistent'."
        )
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("the estimated integrated autocorrelation time is nonpositive.")
    return (tau, window) if return_window else tau


def effective_sample_size(
    samples: ArrayLike,
    *,
    tau: float | None = None,
    **iat_options,
) -> float:
    """Return ``n / (2*tau)`` for a scalar trajectory."""

    values = _trajectory_1d(samples)
    if tau is None:
        tau = integrated_autocorrelation_time(values, **iat_options)
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive and finite.")
    return float(values.size / (2.0 * tau))


def chain_effective_sample_sizes(chains: ArrayLike, **iat_options) -> np.ndarray:
    """Return the Chapter 1 autocorrelation ESS separately for each chain."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(
            "chains must have shape (at least 1 chain, at least 2 draws)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("chains must be finite.")
    return np.asarray(
        [effective_sample_size(chain, **iat_options) for chain in values],
        dtype=float,
    )
