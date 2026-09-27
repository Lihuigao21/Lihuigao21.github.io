"""Scaled bosonic HEOM for stationary Gaussian baths, in units hbar = 1.

The Hamiltonian is H + sum_a Q_a B_a, with Hermitian Q_a and independent,
zero-mean Gaussian bath channels. Initial system and bath are factorized.
Both correlation expansions must be supplied in the *same* exponential basis:
    C_a(t) = sum_k cL_k exp(-nu_k t),
    C_a(t)* = sum_k cR_k exp(-nu_k t).
For complex rates, cR_k is generally NOT cL_k.conjugate().

This is an original sparse implementation of the standard Gaussian hierarchy,
not a method for a general anharmonic/nonlinear nuclear bath. It uses hard tier
truncation, with no Markovian terminator or automatic density normalization.
Convergence of bath expansion and hierarchy depth must be checked separately.

References: Tanimura & Kubo, J. Phys. Soc. Jpn. 58, 101 (1989),
doi:10.1143/JPSJ.58.101; Shi et al., J. Chem. Phys. 130, 084105 (2009),
doi:10.1063/1.3077918; Lambert et al., Nat. Commun. 10, 3721 (2019),
doi:10.1038/s41467-019-11656-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class BathExponent:
    """One term in the paired C(t), C(t)* expansions (energy-squared units)."""

    rate: complex
    left_coefficient: complex
    right_coefficient: complex
    coupling_index: int = 0


@dataclass
class HEOMTrajectory:
    times: np.ndarray
    density_matrices: np.ndarray
    final_ados: np.ndarray
    ados: np.ndarray | None = None

    @property
    def populations(self) -> np.ndarray:
        """Diagonal of the reduced density in the input system basis."""
        return np.diagonal(self.density_matrices, axis1=-2, axis2=-1).real


def _positive(value: float, name: str, *, infinity: bool = False) -> float:
    value = float(value)
    if value <= 0 or np.isnan(value) or (not infinity and not np.isfinite(value)):
        raise ValueError(f"{name} must be positive" + (" and finite" if not infinity else ""))
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def harmonic_mode_exponents(
    omega: float, coupling: float, beta: float, coupling_index: int = 0
) -> tuple[BathExponent, BathExponent]:
    """Thermal mode B = g(b+b†), H_b = omega b†b; beta=inf permits T=0.

    C(t) = g²[(nbar+1) exp(-i omega t) + nbar exp(+i omega t)].
    No counterterm is added; any desired counterterm belongs explicitly in H.
    """
    omega = _positive(omega, "omega")
    beta = _positive(beta, "beta", infinity=True)
    coupling = float(coupling)
    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite")
    coupling_index = _nonnegative_integer(coupling_index, "coupling_index")
    argument = beta * omega
    occupation = 0.0 if argument > 700 else 1.0 / np.expm1(argument)
    emission = coupling**2 * (occupation + 1)
    absorption = coupling**2 * occupation
    return (
        BathExponent(1j * omega, emission, absorption, coupling_index),
        BathExponent(-1j * omega, absorption, emission, coupling_index),
    )


def drude_lorentz_exponents(
    reorganization: float,
    cutoff: float,
    beta: float,
    matsubara_terms: int,
    coupling_index: int = 0,
) -> tuple[BathExponent, ...]:
    """Truncated Matsubara expansion of J(w)=2 lambda gamma w/(w²+gamma²).

    Uses C(t)=(1/pi) integral J(w)[coth(beta*w/2) cos(wt)-i sin(wt)] dw.
    The omitted Matsubara tail is NOT replaced by a terminator. The singular
    case gamma=2*pi*k/beta requires a different expansion and is rejected.
    """
    reorganization = float(reorganization)
    if not np.isfinite(reorganization) or reorganization < 0:
        raise ValueError("reorganization must be finite and nonnegative")
    cutoff = _positive(cutoff, "cutoff")
    beta = _positive(beta, "beta")
    matsubara_terms = _nonnegative_integer(matsubara_terms, "matsubara_terms")
    coupling_index = _nonnegative_integer(coupling_index, "coupling_index")
    phase = beta * cutoff / 2
    if abs(np.sin(phase)) < 1e-10:
        raise ValueError("Drude pole coincides with a Matsubara pole")
    coefficient = reorganization * cutoff * (1 / np.tan(phase) - 1j)
    terms = [BathExponent(cutoff, coefficient, coefficient.conjugate(), coupling_index)]
    for k in range(1, matsubara_terms + 1):
        rate = 2 * np.pi * k / beta
        coefficient = 4 * reorganization * cutoff * rate / (beta * (rate**2 - cutoff**2))
        terms.append(BathExponent(rate, coefficient, coefficient, coupling_index))
    return tuple(terms)


def _hermitian_matrix(value, name: str, dimension: int | None = None) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-13):
        raise ValueError(f"{name} must be Hermitian")
    matrix.setflags(write=False)
    return matrix


def _labels(nexponents: int, max_depth: int) -> tuple[tuple[int, ...], ...]:
    def compositions(total, count):
        if count == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in compositions(total - first, count - 1):
                    yield (first,) + rest

    if nexponents == 0:
        return ((),)
    return tuple(label for tier in range(max_depth + 1) for label in compositions(tier, nexponents))


class BosonicHEOM:
    """Time-independent, total-tier-truncated Gaussian bosonic hierarchy.

    Scaled ADOs are rho_n / sqrt(prod_k n_k! s_k**n_k), where
    s_k=max(abs(cL_k),abs(cR_k)) (or 1 for an identically zero term).
    The equations are
      d rho_n = -i[H,rho_n] - sum n_k nu_k rho_n
        - i sum sqrt((n_k+1)s_k) [Q_k,rho_(n+e_k)]
        - i sum sqrt(n_k/s_k)(cL_k Q_k rho_(n-e_k)
                                             - cR_k rho_(n-e_k) Q_k).
    ADOs above max_depth are set to zero. The root is the physical density.
    """

    def __init__(
        self,
        H: np.ndarray,
        couplings: Sequence[np.ndarray],
        exponents: Sequence[BathExponent],
        max_depth: int,
        *,
        max_ados: int = 100000,
    ):
        self.H = _hermitian_matrix(H, "H")
        self.dimension = self.H.shape[0]
        self.couplings = tuple(_hermitian_matrix(q, "coupling", self.dimension) for q in couplings)
        self.exponents = tuple(exponents)
        self.max_depth = _nonnegative_integer(max_depth, "max_depth")
        max_ados = _nonnegative_integer(max_ados, "max_ados")
        if comb(len(self.exponents) + self.max_depth, self.max_depth) > max_ados:
            raise ValueError("hierarchy exceeds max_ados; explicitly review memory before increasing it")
        for term in self.exponents:
            if not isinstance(term, BathExponent):
                raise TypeError("exponents must contain BathExponent objects")
            index = _nonnegative_integer(term.coupling_index, "coupling_index")
            if index >= len(self.couplings):
                raise ValueError("coupling_index is outside couplings")
            if not np.isfinite([term.rate, term.left_coefficient, term.right_coefficient]).all():
                raise ValueError("bath coefficients and rates must be finite")
            if complex(term.rate).real < 0:
                raise ValueError("bath rates must have nonnegative real parts")
        self._validate_correlations()
        self.labels = _labels(len(self.exponents), self.max_depth)
        self.nados = len(self.labels)
        self.scales = np.array([max(abs(t.left_coefficient), abs(t.right_coefficient)) or 1.0 for t in self.exponents])
        self.generator = self._build_generator()

    def _validate_correlations(self):
        # Include conjugate rates with zero coefficients so a missing partner
        # cannot slip through when one coefficient vanishes (e.g. zero T).
        for term in self.exponents:
            channel = [t for t in self.exponents if t.coupling_index == term.coupling_index]
            for rate in (complex(term.rate), complex(term.rate).conjugate()):
                same = lambda a, b: abs(a - b) <= 1e-13 * max(1.0, abs(a), abs(b))
                right = sum(t.right_coefficient for t in channel if same(t.rate, rate))
                left = sum(complex(t.left_coefficient).conjugate() for t in channel if same(t.rate, rate.conjugate()))
                scale = max(abs(right), abs(left), 1e-300)
                if abs(right - left) > 1e-11 * scale:
                    raise ValueError("right coefficients must expand C(t)* in the same exponential basis")

    def _build_generator(self):
        dim = self.dimension
        eye = sparse.eye(dim, format="csr", dtype=complex)
        unit = sparse.eye(dim * dim, format="csr", dtype=complex)
        left = lambda q: sparse.kron(eye, sparse.csr_matrix(q), format="csr")
        right = lambda q: sparse.kron(sparse.csr_matrix(q.T), eye, format="csr")
        system = -1j * (left(self.H) - right(self.H))
        qleft = [left(q) for q in self.couplings]
        qright = [right(q) for q in self.couplings]
        commutators = [a - b for a, b in zip(qleft, qright)]
        lookup = {n: i for i, n in enumerate(self.labels)}
        rows, cols, values = [], [], []

        def block(i, j, matrix):
            coo = matrix.tocoo()
            rows.extend(coo.row + i * dim * dim)
            cols.extend(coo.col + j * dim * dim)
            values.extend(coo.data)

        for i, n in enumerate(self.labels):
            decay = sum(nk * t.rate for nk, t in zip(n, self.exponents))
            block(i, i, system - decay * unit)
            for k, term in enumerate(self.exponents):
                a = term.coupling_index
                if sum(n) < self.max_depth:
                    upper = list(n)
                    upper[k] += 1
                    block(i, lookup[tuple(upper)], -1j * np.sqrt((n[k] + 1) * self.scales[k]) * commutators[a])
                if n[k]:
                    lower = list(n)
                    lower[k] -= 1
                    block(i, lookup[tuple(lower)], -1j * np.sqrt(n[k] / self.scales[k]) * (
                        term.left_coefficient * qleft[a] - term.right_coefficient * qright[a]
                    ))
        size = self.nados * dim * dim
        result = sparse.coo_matrix((values, (rows, cols)), shape=(size, size), dtype=complex).tocsr()
        result.eliminate_zeros()
        return result

    def propagate(self, rho0: np.ndarray, times: np.ndarray, *, store_ados: bool = False) -> HEOMTrajectory:
        """Evolve from t=0; output times must be finite, nonnegative, increasing.

        The initial state is rho0 tensor the stationary bath state, so all
        non-root ADOs start at zero. Returned ADOs use the scaled convention.
        expm_multiply controls exponential action internally; output spacing is
        not an integration timestep. No positivity/trace repair is performed.
        """
        rho0 = _hermitian_matrix(rho0, "rho0", self.dimension)
        if not np.isclose(np.trace(rho0), 1.0, rtol=0, atol=1e-12):
            raise ValueError("rho0 must have unit trace")
        if np.linalg.eigvalsh(rho0).min() < -1e-12:
            raise ValueError("rho0 must be positive semidefinite")
        times = np.array(times, dtype=float, copy=True)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be a nonempty, finite, nonnegative, strictly increasing vector")
        dim = self.dimension
        state = np.zeros(self.nados * dim * dim, dtype=complex)
        state[:dim * dim] = rho0.ravel(order="F")
        densities = np.empty((len(times), dim, dim), dtype=complex)
        history = np.empty((len(times), self.nados, dim, dim), dtype=complex) if store_ados else None
        previous = 0.0
        trace = self.generator.diagonal().sum()
        for it, time in enumerate(times):
            delta = time - previous
            if delta:
                state = expm_multiply(delta * self.generator, state, traceA=delta * trace)
            ados = state.reshape(self.nados, dim, dim).transpose(0, 2, 1)
            densities[it] = ados[0]
            if history is not None:
                history[it] = ados
            previous = time
        return HEOMTrajectory(times, densities, ados.copy(), history)


__all__ = ["BathExponent", "BosonicHEOM", "HEOMTrajectory", "harmonic_mode_exponents", "drude_lorentz_exponents"]
