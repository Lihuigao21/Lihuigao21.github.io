"""
DVR base class.

This module centralizes the construction of DVR grids / operators and now
supports configurable basis ordering and kinetic-energy operator choices so
wave-packet dynamics and correlation-function style DVR methods can share the
same infrastructure.
"""

from __future__ import annotations

from abc import ABC
from typing import Literal

import numpy as np

from toymodel.utils.constant import kB

BasisOrdering = Literal["state-major", "grid-major"]
GridConvention = Literal["cell-centered", "left-edge"]
KineticOperator = Literal["sinc", "particle_in_box", "fft_periodic"]


class DVRBASE(ABC):
    """
    The base class of DVR methods.

    Parameters
    ----------
    basis_ordering
        "state-major": ``|x0,s0>, |x1,s0>, ..., |x0,s1>, ...``
        "grid-major": ``|x0,s0>, |x0,s1>, ..., |x1,s0>, ...``
    grid_convention
        How the DVR grid points are placed in ``xbound``.
    kinetic_operator
        Choice of kinetic-energy matrix construction.
    """

    def __init__(
        self,
        temperature=300.0,
        model=None,
        Nxgrid=1024,
        xbound=(-10, 10),
        Ntgrid=20,
        tbound=(0, 10),
        m=2000,
        is_CAP=False,
        cap_eta=0.05,
        cap_x0=22.0,
        cap_width=1.0,
        basis_ordering: BasisOrdering = "state-major",
        grid_convention: GridConvention = "cell-centered",
        kinetic_operator: KineticOperator = "sinc",
    ):
        self.temperature = temperature
        self.model = model
        self.Nxgrid = int(Nxgrid)
        self.xbound = self._normalize_xbound(xbound)
        self.dx = (self.xbound[1] - self.xbound[0]) / self.Nxgrid
        self.Ntgrid = int(Ntgrid)
        self.tbound = tbound
        self.m = m
        self.is_CAP = is_CAP
        self.beta = 1 / (self.temperature * kB)
        self.basis_ordering = self._validate_basis_ordering(basis_ordering)
        self.grid_convention = self._validate_grid_convention(grid_convention)
        self.kinetic_operator = self._validate_kinetic_operator(kinetic_operator)

        if self.model is None:
            raise ValueError(
                "model must be provided and must have attribute nstate and method V(x)."
            )
        self.nstate = self.model.nstate

        # CAP params (only used if is_CAP True)
        self.cap_eta = cap_eta
        self.cap_x0 = cap_x0
        self.cap_width = cap_width

        # placeholders
        self.xgrid = None
        self.tgrid = None

        dim = self.Nxgrid * self.nstate
        self.Tn = None
        self.T = None
        self.V = None
        self.Vabs = np.zeros((dim, dim), dtype=complex)
        self.H = np.zeros((dim, dim), dtype=complex)
        self.H0 = np.zeros((dim, dim), dtype=float)
        self.chis = None
        self.local_adiabatic_energies = None

    @staticmethod
    def _normalize_xbound(xbound):
        if np.isscalar(xbound):
            xb = float(xbound)
            return (-xb, xb)
        if len(xbound) != 2:
            raise ValueError("xbound must be a scalar or a 2-tuple/list.")
        return (float(xbound[0]), float(xbound[1]))

    @staticmethod
    def _validate_basis_ordering(value: str) -> BasisOrdering:
        if value not in {"state-major", "grid-major"}:
            raise ValueError(
                "basis_ordering must be 'state-major' or 'grid-major', "
                f"got {value!r}."
            )
        return value  # type: ignore[return-value]

    @staticmethod
    def _validate_grid_convention(value: str) -> GridConvention:
        if value not in {"cell-centered", "left-edge"}:
            raise ValueError(
                "grid_convention must be 'cell-centered' or 'left-edge', "
                f"got {value!r}."
            )
        return value  # type: ignore[return-value]

    @staticmethod
    def _validate_kinetic_operator(value: str) -> KineticOperator:
        if value not in {"sinc", "particle_in_box", "fft_periodic"}:
            raise ValueError(
                "kinetic_operator must be 'sinc', 'particle_in_box', or 'fft_periodic', "
                f"got {value!r}."
            )
        return value  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        return self.Nxgrid * self.nstate

    def build(self):
        """One-stop construction pipeline."""
        self.construct_grid()
        self.construct_T()
        self.construct_V()
        if self.is_CAP:
            self.construct_CAP()
        else:
            self.Vabs = np.zeros((self.dim, self.dim), dtype=complex)
        self.construct_chis()
        self.construct_H()
        self.construct_H0()
        self.construct_timegrid()

    def construct_grid(self):
        """Construct the grid points."""
        x0, x1 = self.xbound
        if self.grid_convention == "cell-centered":
            self.xgrid = (np.arange(self.Nxgrid) + 0.5) * self.dx + x0
        elif self.grid_convention == "left-edge":
            self.xgrid = np.linspace(x0, x1, self.Nxgrid, endpoint=False)
        else:
            raise ValueError(f"Unsupported grid_convention: {self.grid_convention}")

    def construct_timegrid(self):
        """Construct the time grid."""
        self.tgrid = np.linspace(self.tbound[0], self.tbound[1], self.Ntgrid)

    # ------------------------------------------------------------------
    # Basis / indexing helpers
    # ------------------------------------------------------------------
    def linear_index(self, ix: int, istate: int) -> int:
        if self.basis_ordering == "state-major":
            return istate * self.Nxgrid + ix
        return ix * self.nstate + istate

    def reshape_wavefunction(self, coeff: np.ndarray) -> np.ndarray:
        """
        Convert a flattened coefficient vector to shape ``(Nxgrid, nstate)``.

        The returned array is always indexed as ``coeff[ix, istate]`` regardless of
        the underlying flattened ordering.
        """
        coeff = np.asarray(coeff)
        if coeff.shape != (self.dim,):
            raise ValueError(
                f"Expected flattened wavefunction of shape ({self.dim},), got {coeff.shape}."
            )
        if self.basis_ordering == "state-major":
            return coeff.reshape(self.nstate, self.Nxgrid).T.copy()
        return coeff.reshape(self.Nxgrid, self.nstate).copy()

    def flatten_wavefunction(self, coeff_by_grid: np.ndarray) -> np.ndarray:
        coeff_by_grid = np.asarray(coeff_by_grid)
        if coeff_by_grid.shape != (self.Nxgrid, self.nstate):
            raise ValueError(
                "Expected wavefunction in grid/state layout with shape "
                f"({self.Nxgrid}, {self.nstate}), got {coeff_by_grid.shape}."
            )
        if self.basis_ordering == "state-major":
            return coeff_by_grid.T.reshape(-1).copy()
        return coeff_by_grid.reshape(-1).copy()

    def lift_nuclear_operator(self, op: np.ndarray, dtype=None) -> np.ndarray:
        op = np.asarray(op)
        if op.shape != (self.Nxgrid, self.Nxgrid):
            raise ValueError(
                f"Expected nuclear operator of shape ({self.Nxgrid}, {self.Nxgrid}), got {op.shape}."
            )
        if self.basis_ordering == "state-major":
            lifted = np.kron(np.eye(self.nstate, dtype=dtype), op)
        else:
            lifted = np.kron(op, np.eye(self.nstate, dtype=dtype))
        return lifted.astype(dtype) if dtype is not None else lifted

    def local_operator_matrix(self, mats: np.ndarray, dtype=None) -> np.ndarray:
        """
        Build a full matrix from local electronic matrices at each grid point.

        Parameters
        ----------
        mats
            Array with shape ``(Nxgrid, nstate, nstate)``.
        """
        mats = np.asarray(mats)
        if mats.shape != (self.Nxgrid, self.nstate, self.nstate):
            raise ValueError(
                "Expected local matrices with shape "
                f"({self.Nxgrid}, {self.nstate}, {self.nstate}), got {mats.shape}."
            )
        out = np.zeros((self.dim, self.dim), dtype=dtype or mats.dtype)
        if self.basis_ordering == "state-major":
            for s1 in range(self.nstate):
                for s2 in range(self.nstate):
                    out[
                        s1 * self.Nxgrid : (s1 + 1) * self.Nxgrid,
                        s2 * self.Nxgrid : (s2 + 1) * self.Nxgrid,
                    ] = np.diag(mats[:, s1, s2])
        else:
            for ix in range(self.Nxgrid):
                sl = slice(ix * self.nstate, (ix + 1) * self.nstate)
                out[sl, sl] = mats[ix]
        return out

    # ------------------------------------------------------------------
    # Operator construction
    # ------------------------------------------------------------------
    def _construct_T_sinc(self) -> np.ndarray:
        idx = np.arange(self.Nxgrid)
        denom = np.subtract.outer(idx, idx) ** 2 / 2 + np.identity(self.Nxgrid) * 3 / np.pi**2
        numer = np.outer(np.resize([1, -1], self.Nxgrid), np.resize([1, -1], self.Nxgrid))
        Tn = numer / 2 / self.m / self.dx**2
        Tn /= denom
        return Tn.astype(np.float64)

    def _construct_T_particle_in_box(self) -> np.ndarray:
        ndvr = self.Nxgrid
        mass = self.m
        L = self.xbound[1] - self.xbound[0]
        N = ndvr + 1
        T = np.zeros((ndvr, ndvr), dtype=np.float64)
        pref = (np.pi**2) / (4.0 * mass * L**2)
        for i in range(1, N):
            for j in range(i, N):
                tmp = (-1) ** (i - j) * pref
                if i == j:
                    T[i - 1, j - 1] = tmp * (
                        (2 * ndvr**2 + 1) / 3 - 1 / np.sin(i * np.pi / N) ** 2
                    )
                else:
                    T[i - 1, j - 1] = T[j - 1, i - 1] = tmp * (
                        1 / (np.sin(0.5 * np.pi * (i - j) / N)) ** 2
                        - 1 / (np.sin(0.5 * np.pi * (i + j) / N)) ** 2
                    )
        return T

    def _construct_T_fft_periodic(self) -> np.ndarray:
        ks = 2.0 * np.pi * np.fft.fftfreq(self.Nxgrid, d=self.dx)
        idx = np.arange(self.Nxgrid)
        F = np.exp(-2j * np.pi * np.outer(idx, idx) / self.Nxgrid) / np.sqrt(
            self.Nxgrid
        )
        T = F.conj().T @ np.diag(ks**2 / (2.0 * self.m)) @ F
        return np.real(0.5 * (T + T.conj().T)).astype(np.float64)

    def construct_nuclear_T(self) -> np.ndarray:
        """Construct the nuclear kinetic-energy matrix before lifting to full space."""
        if self.kinetic_operator == "sinc":
            return self._construct_T_sinc()
        if self.kinetic_operator == "particle_in_box":
            return self._construct_T_particle_in_box()
        if self.kinetic_operator == "fft_periodic":
            return self._construct_T_fft_periodic()
        raise ValueError(f"Unsupported kinetic_operator: {self.kinetic_operator}")

    def construct_T(self):
        """Construct the kinetic energy matrix and lift it to full space."""
        Tn = self.construct_nuclear_T()
        self.Tn = Tn.copy()
        self.T = self.lift_nuclear_operator(Tn, dtype=np.float64)

    def construct_V(self):
        """Construct the potential-energy matrix in the configured basis ordering."""
        if self.xgrid is None:
            raise RuntimeError("xgrid not constructed. Call construct_grid() first.")
        V_all = np.asarray(self.model.V(self.xgrid), dtype=np.float64)
        if V_all.shape != (self.Nxgrid, self.nstate, self.nstate):
            raise ValueError(
                "model.V(xgrid) must return shape "
                f"({self.Nxgrid}, {self.nstate}, {self.nstate}), got {V_all.shape}."
            )
        self.V = self.local_operator_matrix(V_all, dtype=np.float64)

    def construct_CAP(self):
        """Construct the CAP in full space."""
        if self.xgrid is None:
            raise RuntimeError(
                "xgrid not constructed. Call construct_grid() first (or build())."
            )

        xs = self.xgrid
        Vabs = 2.0 / (1.0 + np.exp((self.cap_x0 - np.abs(xs)) / self.cap_width))
        Vabs = np.diagflat(Vabs)
        self.Vabs = (-1j * self.cap_eta) * self.lift_nuclear_operator(
            Vabs, dtype=complex
        ).astype(complex)

    def construct_H(self):
        """Construct the Hamiltonian matrix."""
        if self.T is None or self.V is None:
            raise RuntimeError(
                "T or V not constructed. Call construct_T/construct_V first (or build())."
            )
        self.H = self.T + self.V + self.Vabs

    def construct_H0(self):
        """Construct the Hamiltonian matrix without CAP."""
        if self.T is None or self.V is None:
            raise RuntimeError(
                "T or V not constructed. Call construct_T/construct_V first (or build())."
            )
        self.H0 = np.real(self.T + self.V)

    def construct_chis(self):
        """
        Construct local adiabatic eigenvectors ``chis[ix]`` from ``V(x_ix)``.

        ``chis[ix][:, a]`` is the local adiabatic eigenvector for adiabatic state ``a``
        expressed in the diabatic basis.
        """
        if self.xgrid is None:
            raise RuntimeError("xgrid not constructed. Call build() or construct_grid().")

        V_all = np.asarray(self.model.V(self.xgrid), dtype=np.float64)
        chis = np.zeros((self.Nxgrid, self.nstate, self.nstate), dtype=complex)
        energies = np.zeros((self.Nxgrid, self.nstate), dtype=np.float64)
        reference = None
        for ix in range(self.Nxgrid):
            en, vecs = np.linalg.eigh(V_all[ix])
            if reference is not None:
                for ist in range(self.nstate):
                    if np.vdot(reference[:, ist], vecs[:, ist]).real < 0:
                        vecs[:, ist] *= -1.0
            energies[ix] = en
            chis[ix] = vecs
            reference = vecs.copy()
        self.local_adiabatic_energies = energies
        self.chis = chis

    # ------------------------------------------------------------------
    # Wavefunction transforms and utility operators
    # ------------------------------------------------------------------
    def local_diabatic_to_adiabatic(self, coeff: np.ndarray) -> np.ndarray:
        if self.chis is None:
            self.construct_chis()
        coeff_grid = self.reshape_wavefunction(coeff)
        coeff_ad = np.zeros_like(coeff_grid, dtype=np.complex128)
        for ix in range(self.Nxgrid):
            coeff_ad[ix] = self.chis[ix].conj().T @ coeff_grid[ix]
        return coeff_ad

    def local_adiabatic_to_diabatic(self, coeff_ad: np.ndarray) -> np.ndarray:
        if self.chis is None:
            self.construct_chis()
        coeff_ad = np.asarray(coeff_ad)
        if coeff_ad.shape != (self.Nxgrid, self.nstate):
            raise ValueError(
                f"Expected local adiabatic coeffs of shape ({self.Nxgrid}, {self.nstate}), got {coeff_ad.shape}."
            )
        coeff_dia = np.zeros_like(coeff_ad, dtype=np.complex128)
        for ix in range(self.Nxgrid):
            coeff_dia[ix] = self.chis[ix] @ coeff_ad[ix]
        return self.flatten_wavefunction(coeff_dia)

    def side_operator(self, x: float) -> np.ndarray:
        if self.xgrid is None:
            raise RuntimeError("xgrid not constructed. Call build() or construct_grid().")
        hs = np.diagflat(self.xgrid >= x)
        return self.lift_nuclear_operator(hs, dtype=complex)

    def projector_electronic(self, istate=0):
        if not (0 <= istate < self.nstate):
            raise ValueError(f"istate must be in [0, {self.nstate - 1}], got {istate}")
        if self.chis is None:
            self.construct_chis()
        mats = np.zeros((self.Nxgrid, self.nstate, self.nstate), dtype=complex)
        for ix, chi in enumerate(self.chis):
            v = chi[:, istate]
            mats[ix] = np.outer(v, v.conjugate())
        return self.local_operator_matrix(mats, dtype=complex)
