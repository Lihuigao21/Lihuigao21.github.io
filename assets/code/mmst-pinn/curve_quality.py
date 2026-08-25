"""Reusable quality gates for continuous scientific curves."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def curve_smoothness_metrics(
    x,
    y,
    *,
    window_fraction: float = 0.08,
    polyorder: int = 3,
):
    """Return scale-free high-frequency roughness diagnostics.

    The Savitzky--Golay trend is a diagnostic reference only. This function
    never modifies the reported curve.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 9:
        raise ValueError("x and y must be matching 1D arrays with at least 9 points.")
    finite = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    dx = np.diff(x)
    strictly_increasing = bool(np.all(dx > 0.0))
    duplicate_points = int(np.count_nonzero(dx == 0.0))
    if not finite or not strictly_increasing:
        return {
            "finite": finite,
            "strictly_increasing_x": strictly_increasing,
            "duplicate_x_points": duplicate_points,
            "roughness_rms_fraction": float("inf"),
            "roughness_max_fraction": float("inf"),
            "isolated_spike_fraction": float("inf"),
            "endpoint_roughness_max_fraction": float("inf"),
            "window_points": None,
        }

    window = max(polyorder + 2, int(round(x.size * float(window_fraction))))
    if window % 2 == 0:
        window += 1
    window = min(window, x.size if x.size % 2 else x.size - 1)
    if window <= polyorder:
        window = polyorder + 2 + (polyorder % 2)
    trend = savgol_filter(y, window_length=window, polyorder=polyorder)
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0e-15)
    residual = y - trend
    local_median = np.median(
        np.stack((np.roll(y, 1), y, np.roll(y, -1))), axis=0
    )
    spike = y - local_median
    spike[[0, -1]] = 0.0
    endpoint_points = max(2, min(window // 2, x.size // 10))
    endpoint_residual = np.concatenate(
        (residual[:endpoint_points], residual[-endpoint_points:])
    )
    return {
        "finite": finite,
        "strictly_increasing_x": strictly_increasing,
        "duplicate_x_points": duplicate_points,
        "roughness_rms_fraction": float(np.sqrt(np.mean(residual**2)) / scale),
        "roughness_max_fraction": float(np.max(np.abs(residual)) / scale),
        "isolated_spike_fraction": float(np.max(np.abs(spike)) / scale),
        "endpoint_roughness_max_fraction": float(
            np.max(np.abs(endpoint_residual)) / scale
        ),
        "window_points": int(window),
    }

def curve_smoothness_gate(
    metrics,
    *,
    max_rms_fraction: float,
    max_peak_fraction: float,
    max_spike_fraction: float,
    max_endpoint_roughness_fraction: float = float("inf"),
):
    """Apply explicit acceptance thresholds to smoothness metrics."""

    return bool(
        metrics["finite"]
        and metrics["strictly_increasing_x"]
        and metrics["duplicate_x_points"] == 0
        and metrics["roughness_rms_fraction"] <= float(max_rms_fraction)
        and metrics["roughness_max_fraction"] <= float(max_peak_fraction)
        and metrics["isolated_spike_fraction"] <= float(max_spike_fraction)
        and metrics["endpoint_roughness_max_fraction"]
        <= float(max_endpoint_roughness_fraction)
    )


def cleaning_distortion_metrics(x, raw, cleaned):
    """Quantify how a reference-extraction cleanup changes a curve."""

    x = np.asarray(x, dtype=float)
    raw = np.asarray(raw, dtype=float)
    cleaned = np.asarray(cleaned, dtype=float)
    if x.ndim != 1 or raw.shape != x.shape or cleaned.shape != x.shape:
        raise ValueError("x, raw, and cleaned must be matching 1D arrays.")
    raw_scale = max(float(np.ptp(raw)), float(np.max(np.abs(raw))), 1.0e-15)
    raw_area = float(np.trapezoid(raw, x))
    cleaned_area = float(np.trapezoid(cleaned, x))
    raw_argmax = float(x[int(np.argmax(raw))])
    cleaned_argmax = float(x[int(np.argmax(cleaned))])

    def robust_peak_location(values, top_fraction=0.95):
        threshold = float(top_fraction) * float(np.max(values))
        mask = values >= threshold
        weights = values[mask] - threshold + 1.0e-15
        return float(np.sum(x[mask] * weights) / np.sum(weights))

    raw_peak = robust_peak_location(raw)
    cleaned_peak = robust_peak_location(cleaned)
    return {
        "peak_shift_x": cleaned_peak - raw_peak,
        "peak_location_definition": "weighted centroid above 95% of maximum",
        "argmax_peak_shift_x": cleaned_argmax - raw_argmax,
        "relative_area_change": (
            (cleaned_area - raw_area) / raw_area if abs(raw_area) > 1.0e-15 else 0.0
        ),
        "max_pointwise_change_fraction": float(
            np.max(np.abs(cleaned - raw)) / raw_scale
        ),
    }
