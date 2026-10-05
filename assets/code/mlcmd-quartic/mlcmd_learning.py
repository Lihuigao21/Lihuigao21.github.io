"""One-dimensional conservative force matching for direct and delta ML-CMD.

This is a small NumPy/SciPy representation, not a reproduction of DeepMD.
Labels may be projected instantaneous forces or converged conditional means;
the caller owns their sampling measure, uncertainty and train/test separation.
Optimizer termination is not a scientific accuracy or sampling-convergence gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


def _callback(callback, q, name):
    value = np.asarray(callback(q), dtype=float)
    try:
        value = np.broadcast_to(value, q.shape)
    except ValueError as exc:
        raise ValueError(f"{name} must return values matching the coordinate shape") from exc
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} returned nonfinite values")
    return value


def _force_and_jacobian(parameters, x, prefactor):
    """Force and its parameter derivatives for E=sum(a*tanh(w*x+b))."""
    a, w, b = np.split(parameters, 3)
    t = np.tanh(x[:, None] * w + b)
    s = 1.0 - t * t
    force = -prefactor * np.sum(a * w * s, axis=1)
    jac = np.concatenate((
        -prefactor * w * s,
        -prefactor * a * s * (1.0 - 2.0 * w * x[:, None] * t),
        2.0 * prefactor * a * w * s * t,
    ), axis=1)
    return force, jac


@dataclass
class ForceMatchedPotential1D:
    """A learned energy and its negative derivative within a declared domain.

    The network correction is zero at ``center`` (energy gauge only). Direct
    mode returns that energy; delta mode adds the supplied bare PES. Coordinate
    and force units are inherited unchanged from the input. Base callbacks must
    be vectorized and obey ``base_force = -d(base_energy)/dq``. No extrapolation,
    clipping, periodic wrapping, or fallback to the bare PES is performed.
    """

    parameters: np.ndarray
    center: float
    coordinate_scale: float
    energy_scale: float
    domain: tuple[float, float]
    mode: str = "direct"
    base_id: str | None = None
    base_energy: Callable | None = None
    base_force: Callable | None = None

    def __post_init__(self):
        self.parameters = np.array(self.parameters, dtype=float, copy=True)
        if (self.parameters.ndim != 1 or self.parameters.size < 3
                or self.parameters.size % 3 or not np.all(np.isfinite(self.parameters))):
            raise ValueError("parameters must contain finite a, w, b vectors of equal length")
        if self.mode not in ("direct", "delta"):
            raise ValueError("mode must be direct or delta")
        self.domain = tuple(float(v) for v in self.domain)
        if (len(self.domain) != 2 or not np.all(np.isfinite(self.domain))
                or self.domain[0] >= self.domain[1]):
            raise ValueError("domain must be a finite increasing pair")
        if (not np.isfinite(self.center) or not self.domain[0] <= self.center <= self.domain[1]
                or not np.isfinite(self.coordinate_scale) or self.coordinate_scale <= 0
                or not np.isfinite(self.energy_scale) or self.energy_scale <= 0):
            raise ValueError("invalid coordinate or energy normalization")
        if self.mode == "delta":
            if not self.base_id or not callable(self.base_energy) or not callable(self.base_force):
                raise ValueError("delta mode requires base_id and both base callbacks")
        elif any(v is not None for v in (self.base_id, self.base_energy, self.base_force)):
            raise ValueError("direct mode does not accept a base PES")

    def _coordinates(self, q):
        q = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q)):
            raise ValueError("coordinates must be finite")
        if np.any(q < self.domain[0]) or np.any(q > self.domain[1]):
            raise ValueError(f"coordinate outside trained domain {self.domain}; extrapolation forbidden")
        return q

    def energy(self, q):
        q = self._coordinates(q)
        a, w, b = np.split(self.parameters, 3)
        x = (q.ravel() - self.center) / self.coordinate_scale
        value = self.energy_scale * np.sum(a * (np.tanh(x[:, None] * w + b) - np.tanh(b)), axis=1)
        value = value.reshape(q.shape)
        if self.mode == "delta":
            value = value + _callback(self.base_energy, q, "base_energy")
        return float(value) if q.ndim == 0 else value

    def value(self, q):
        """Alias for :meth:`energy`, matching the CMD potential interface."""
        return self.energy(q)

    def force(self, q):
        q = self._coordinates(q)
        x = (q.ravel() - self.center) / self.coordinate_scale
        value = _force_and_jacobian(self.parameters, x, self.energy_scale / self.coordinate_scale)[0]
        value = value.reshape(q.shape)
        if self.mode == "delta":
            value = value + _callback(self.base_force, q, "base_force")
        return float(value) if q.ndim == 0 else value

    @classmethod
    def fit(cls, q, force, *, mode="direct", weights=None, hidden_units=12,
            seed=0, max_nfev=2000, domain=None, base_energy=None,
            base_force=None, base_id=None, tolerance=1e-10):
        """Return ``(model, diagnostics)`` from weighted force least squares.

        Objective: sum_i normalized_weight_i * (F_model-F_label)^2 / F_scale^2.
        ``weights`` are explicit loss weights, not inferred statistical weights.
        Their meaning must be fixed by the experiment. Zero-weight rows do not
        establish support. The domain is the retained coordinate hull or an
        explicitly narrower interval; coverage within it still requires audit.

        Both modes use the SAME total-label force RMS for normalization and the
        same seeded network initialization (zero output weights). Width, samples,
        weights, seed and evaluation budget should be held fixed in comparisons.
        All hidden slopes, hidden biases and output weights are optimized.
        A zero delta target is an exact zero correction, not learned quantum
        improvement. Energies are determined only up to the stated gauge.
        """
        q = np.asarray(q, dtype=float)
        force = np.asarray(force, dtype=float)
        if q.ndim != 1 or force.shape != q.shape or q.size < 2:
            raise ValueError("q and force must be matching one-dimensional arrays with at least two rows")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(force)):
            raise ValueError("training arrays must be finite")
        weights = np.ones(q.size) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != q.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be matching finite nonnegative values")
        keep = weights > 0
        input_count = int(q.size)
        q, force, weights = q[keep], force[keep], weights[keep]
        if q.size < 2 or np.min(q) == np.max(q):
            raise ValueError("at least two distinct positive-weight coordinates are required")
        if (not isinstance(hidden_units, (int, np.integer)) or hidden_units < 1
                or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1):
            raise ValueError("hidden_units and max_nfev must be positive integers")
        if not np.isfinite(tolerance) or tolerance <= np.finfo(float).eps:
            raise ValueError("tolerance must exceed machine epsilon")
        lo, hi = float(np.min(q)), float(np.max(q))
        selected_domain = (lo, hi) if domain is None else tuple(domain)
        if len(selected_domain) != 2 or selected_domain[0] < lo or selected_domain[1] > hi:
            raise ValueError("domain must lie within positive-weight training support")
        center = 0.5 * (selected_domain[0] + selected_domain[1])
        scale = 0.5 * (hi - lo)
        # Scaling by the largest weight first avoids overflow in normalization.
        weights = weights / np.max(weights)
        weights = weights / np.sum(weights)
        force_scale = float(np.sqrt(np.sum(weights * force * force)))
        if not np.isfinite(force_scale):
            raise ValueError("force magnitudes overflow normalization")
        if force_scale == 0.0:
            force_scale = 1.0
        rng = np.random.default_rng(seed)
        parameters = np.concatenate((np.zeros(hidden_units),
                                     rng.normal(0.0, 0.6, hidden_units),
                                     rng.uniform(-1.0, 1.0, hidden_units)))
        model = cls(parameters, center, scale, force_scale * scale,
                    selected_domain, mode, base_id, base_energy, base_force)
        target = force.copy()
        if mode == "delta":
            # Check both callbacks now, including finite energy at all labels.
            _callback(base_energy, q, "base_energy")
            target -= _callback(base_force, q, "base_force")
        x = (q - center) / scale
        root_weight = np.sqrt(weights) / force_scale

        def residual(p):
            return root_weight * (_force_and_jacobian(p, x, force_scale)[0] - target)

        def jacobian(p):
            return root_weight[:, None] * _force_and_jacobian(p, x, force_scale)[1]

        initial_rmse = float(np.sqrt(np.sum(weights * target * target)))
        result = least_squares(residual, parameters, jac=jacobian,
                               max_nfev=max_nfev, ftol=tolerance,
                               xtol=tolerance, gtol=tolerance)
        model.parameters = result.x.copy()
        error = _force_and_jacobian(result.x, x, force_scale)[0] - target
        diagnostics = {
            "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
            "optimizer_message": str(result.message), "nfev": int(result.nfev),
            "njev": int(result.njev), "max_nfev": int(max_nfev),
            "weighted_force_rmse": float(np.sqrt(np.sum(weights * error * error))),
            "max_abs_force_error": float(np.max(np.abs(error))),
            "initial_weighted_force_rmse": initial_rmse,
            "force_scale": force_scale, "input_count": input_count,
            "positive_weight_count": int(q.size), "hidden_units": int(hidden_units),
            "parameter_count": int(parameters.size), "seed": int(seed), "mode": mode,
            "domain": list(model.domain), "training_support": [lo, hi],
            "normalized_objective": float(np.sum(result.fun * result.fun)),
            "optimality": float(result.optimality), "tolerance": float(tolerance),
            "sampling_convergence_checked": False,
        }
        return model, diagnostics

    def to_dict(self):
        """JSON-safe payload; functions/code are deliberately not serialized."""
        return {"schema": "toymodel.mlcmd_learning.v1", "mode": self.mode,
                "parameters": self.parameters.tolist(), "center": float(self.center),
                "coordinate_scale": float(self.coordinate_scale),
                "energy_scale": float(self.energy_scale), "domain": list(self.domain),
                "base_id": self.base_id}

    @classmethod
    def from_dict(cls, payload, *, base_energy=None, base_force=None, base_id=None):
        """Reload, requiring explicit matching base identity for a delta model."""
        if payload.get("schema") != "toymodel.mlcmd_learning.v1":
            raise ValueError("unsupported ML-CMD model schema")
        if payload["mode"] == "delta" and base_id != payload["base_id"]:
            raise ValueError("delta base_id does not match saved model")
        return cls(payload["parameters"], float(payload["center"]),
                   float(payload["coordinate_scale"]), float(payload["energy_scale"]),
                   tuple(payload["domain"]), payload["mode"], base_id,
                   base_energy, base_force)
