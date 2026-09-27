from __future__ import annotations

import copy as cp
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import eigh

"""Base classes and snapshots for electronic model evaluations."""


@dataclass
class ModelSnapshot:
    x: np.ndarray
    energies: np.ndarray | None = None
    wavefun: np.ndarray | None = None
    force: np.ndarray | None = None
    force_diag: np.ndarray | None = None
    nac: np.ndarray | None = None

    def as_dict(self) -> dict:
        out = {
            "energies": self.energies,
            "wavefun": self.wavefun,
        }
        if self.force is not None:
            out["force"] = self.force
        if self.force_diag is not None:
            out["force_diag"] = self.force_diag
        if self.nac is not None:
            out["nac"] = self.nac
        return out


class ElectronicModel(ABC):
    def __init__(
        self,
        x=None,
        representation="adiabatic",
        wavefunc=None,
        nstate=2,
        wavefunc_revised=True,
    ):
        if not isinstance(representation, str):
            raise TypeError("representation must be 'adiabatic' or 'diabatic'.")
        representation = representation.casefold()
        if representation not in {"adiabatic", "diabatic"}:
            raise ValueError(
                "representation must be 'adiabatic' or 'diabatic', "
                f"got {representation!r}."
            )
        self.nstate = nstate
        self.representation = representation
        self.wavefunc = wavefunc
        self.hamiltonian = None
        self.force = None
        self.derivative_coupling = None
        self.position = None
        self.wavefunc_revised = wavefunc_revised
        if x is not None:
            self.position = np.asarray(x)
            self.compute(self.position)

    def update(self, x):
        new_model = cp.copy(self)
        new_model.compute(x)
        new_model.position = np.asarray(x)
        return new_model

    @abstractmethod
    def compute(self, x):
        pass


class AbaticModelModel(ABC):
    """Legacy one-state model base class kept for backward compatibility."""

    def __init__(self, x=None):
        if x is not None:
            self.position = np.asarray(x)
            self.force = self.get_force(self.position)
        self.nstate = 1

    @abstractmethod
    def V(self, x):
        pass

    @abstractmethod
    def dV(self, x):
        pass

    def get_force(self, x):
        return -self.dV(x)

    def get_energies(self, x):
        return self.V(x)


class DiabaticModel(ElectronicModel):
    def __init__(self, x, representation="adiabatic", nstate=2, wavefunc_revised=True):
        super().__init__(
            x, representation, nstate=nstate, wavefunc_revised=wavefunc_revised
        )

    def compute(self, x):
        snap = self.evaluate(x, need_force=True, need_nac=True)
        self.derivative_coupling = snap.nac
        self.force = snap.force
        self.wavefun = snap.wavefun
        self.hamiltonian = snap.energies

    def _compute_basis_states(self, V):
        n_points = V.shape[0]

        if self.representation == "adiabatic":
            energies = np.zeros((n_points, self.nstate))
            coeff = np.zeros((n_points, self.nstate, self.nstate))

            for i in range(n_points):
                e, c = eigh(V[i])
                energies[i] = e
                coeff[i] = c

                if i > 0 and self.wavefunc_revised:
                    for s in range(self.nstate):
                        if np.dot(coeff[i, :, s], coeff[i - 1, :, s]) <= 0:
                            coeff[i, :, s] = -coeff[i, :, s]

            return coeff, energies

        return np.tile(np.eye(self.nstate), (n_points, 1, 1)), V

    def _compute_force(self, dV, coeff):
        return -np.array([coeff[i].T @ dV[i] @ coeff[i] for i in range(len(dV))])

    def _compute_derivative_coupling(self, dV, coeff, energies):
        out = np.zeros((len(dV), self.nstate, self.nstate))
        if self.representation == "diabatic":
            return out
        for i in range(self.nstate):
            for j in range(self.nstate):
                if i != j:
                    dE = energies[:, j] - energies[:, i]
                    tiny = np.abs(dE) < 1e-10
                    safe_dE = dE.copy()
                    safe_dE[tiny] = np.where(safe_dE[tiny] >= 0.0, 1e-10, -1e-10)
                    out[:, i, j] = np.array(
                        [
                            (coeff[k][:, i] @ dV[k] @ coeff[k][:, j]) / safe_dE[k]
                            for k in range(len(dV))
                        ]
                    )
                    out[:, j, i] = -out[:, i, j]
        return out

    def _prepare_V_dV(self, x):
        x = np.asarray(x)
        V = self.V(x)
        dV = self.dV(x)
        if x.ndim == 0:
            x = x.reshape(1)
            V = V[np.newaxis, :, :]
            dV = dV[np.newaxis, :, :]
        return x, V, dV

    def _normalize_wavefun(self, wavefunc, n_points):
        wf = np.asarray(wavefunc)
        if wf.ndim == 2:
            if wf.shape != (self.nstate, self.nstate):
                raise ValueError(
                    f"wavefunc of shape {wf.shape} is not (nstate, nstate) "
                    f"for nstate={self.nstate}"
                )
            wf = wf[np.newaxis, :, :]
        elif wf.ndim == 3:
            if (
                wf.shape[0] != n_points
                or wf.shape[1] != self.nstate
                or wf.shape[2] != self.nstate
            ):
                raise ValueError(
                    f"wavefunc has incompatible shape {wf.shape}; expected "
                    f"(n_points, nstate, nstate)=({n_points}, {self.nstate}, {self.nstate})"
                )
        else:
            raise ValueError(
                f"wavefunc must have ndim 2 or 3, got ndim={wf.ndim} with shape {wf.shape}"
            )
        return wf

    def _normalize_energies(self, energies, n_points):
        en = np.asarray(energies)
        if en.ndim == 1:
            if en.shape[0] != self.nstate:
                raise ValueError(
                    f"energies of shape {en.shape} do not match nstate={self.nstate}"
                )
            en = en[np.newaxis, :]
        elif en.ndim == 2:
            if en.shape != (n_points, self.nstate):
                raise ValueError(
                    f"energies has incompatible shape {en.shape}; expected "
                    f"({n_points}, {self.nstate})"
                )
        else:
            raise ValueError(
                f"energies must have ndim 1 or 2, got ndim={en.ndim} with shape {en.shape}"
            )
        return en

    def evaluate(
        self,
        x,
        *,
        wavefunc_for_nac=None,
        energies_for_nac=None,
        need_force=True,
        need_nac=True,
        need_wavefun=True,
    ) -> ModelSnapshot:
        x_arr, V, dV = self._prepare_V_dV(x)
        coeff, energies = self._compute_basis_states(V)

        force = None
        force_diag = None
        if need_force:
            force = self._compute_force(dV, coeff)
            force_diag = np.diagonal(force, axis1=1, axis2=2)

        nac = None
        if need_nac:
            nac_coeff = coeff
            nac_energies = energies
            if wavefunc_for_nac is not None:
                nac_coeff = self._normalize_wavefun(wavefunc_for_nac, V.shape[0])
            if energies_for_nac is not None:
                nac_energies = self._normalize_energies(energies_for_nac, V.shape[0])
            nac = self._compute_derivative_coupling(dV, nac_coeff, nac_energies)

        return ModelSnapshot(
            x=x_arr,
            energies=energies,
            wavefun=coeff if need_wavefun else None,
            force=force,
            force_diag=force_diag,
            nac=nac,
        )

    def get_nac(self, x, wavefunc=None):
        return self.evaluate(
            x,
            wavefunc_for_nac=wavefunc,
            need_force=False,
            need_nac=True,
        ).nac

    def get_force(self, x):
        return self.evaluate(x, need_force=True, need_nac=False).force

    def get_force_diag(self, x):
        return self.evaluate(x, need_force=True, need_nac=False).force_diag

    def get_energies(self, x):
        return self.evaluate(x, need_force=False, need_nac=False).energies

    def get_wavefun(self, x):
        return self.evaluate(x, need_force=False, need_nac=False).wavefun

    def get_properties(self, x, wavefunc=None, need_force=True, need_nac=True):
        return self.evaluate(
            x,
            wavefunc_for_nac=wavefunc,
            need_force=need_force,
            need_nac=need_nac,
        ).as_dict()

    def get_nac_from_wavefun(self, x, wavefunc, energies=None):
        return self.evaluate(
            x,
            wavefunc_for_nac=wavefunc,
            energies_for_nac=energies,
            need_force=False,
            need_nac=True,
        ).nac

    def get_basis_properties(self, x, need_force=False):
        return self.evaluate(x, need_force=need_force, need_nac=False).as_dict()

    @abstractmethod
    def V(self, x):
        pass

    @abstractmethod
    def dV(self, x):
        pass


class VectorDiabaticModel(ElectronicModel):
    """Adiabatic interface for diabatic models with explicit coordinate axes.

    ``DiabaticModel`` is the historical one-dimensional API where ``dV(x)`` has
    no coordinate axis.  This class keeps that path untouched and adds a
    separate convention for vector coordinates:

    - coordinates: ``(ndim,)`` for one configuration or ``(npoint, ndim)``
    - diabatic potential: ``(npoint, nstate, nstate)``
    - diabatic gradient: ``(npoint, ndim, nstate, nstate)``
    - force/NAC snapshots: coordinate axis is last, e.g.
      ``force_diag.shape == (npoint, nstate, ndim)``.
    """

    def __init__(
        self,
        x=None,
        representation="adiabatic",
        nstate=2,
        ndim=1,
        wavefunc_revised=True,
    ):
        self.ndim = int(ndim)
        if self.ndim < 1:
            raise ValueError("ndim must be at least 1.")
        self.natom = self.ndim
        self.nmode = self.ndim
        super().__init__(
            x,
            representation,
            nstate=nstate,
            wavefunc_revised=wavefunc_revised,
        )

    def _as_batch(self, q):
        arr = np.asarray(q, dtype=float)
        if self.ndim == 1:
            if arr.ndim == 0:
                return arr.reshape(1, 1), True
            if arr.ndim == 1:
                return arr.reshape(-1, 1), arr.size == 1
            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr, arr.shape[0] == 1
        else:
            if arr.ndim == 1 and arr.shape[0] == self.ndim:
                return arr[None, :], True
            if arr.ndim == 2 and arr.shape[1] == self.ndim:
                return arr, arr.shape[0] == 1
        raise ValueError(
            f"coordinates must be scalar/(npoint,) for ndim=1 or shape "
            f"({self.ndim},)/(npoint, {self.ndim}); got {arr.shape}."
        )

    def compute(self, q):
        snap = self.evaluate(q, need_force=True, need_nac=True)
        self.position = np.asarray(q)
        self.hamiltonian = snap.energies
        self.wavefun = snap.wavefun
        self.force = snap.force
        self.derivative_coupling = snap.nac

    def _compute_basis_states(self, V):
        n_points = V.shape[0]
        if self.representation != "adiabatic":
            raise ValueError("VectorDiabaticModel currently exposes adiabatic states only.")

        energies = np.zeros((n_points, self.nstate), dtype=float)
        coeff = np.zeros((n_points, self.nstate, self.nstate), dtype=float)
        for i in range(n_points):
            e, c = eigh(V[i])
            energies[i] = e
            coeff[i] = c
            if i > 0 and self.wavefunc_revised:
                for s in range(self.nstate):
                    if np.dot(coeff[i, :, s], coeff[i - 1, :, s]) <= 0:
                        coeff[i, :, s] *= -1.0
        return coeff, energies

    def _prepare_V_dV(self, q):
        q_batch, _ = self._as_batch(q)
        npoint = q_batch.shape[0]
        V = np.asarray(self.V(q_batch), dtype=float)
        dV = np.asarray(self.dV(q_batch), dtype=float)

        if V.shape == (self.nstate, self.nstate):
            V = V[None, :, :]
        if V.shape != (npoint, self.nstate, self.nstate):
            raise ValueError(
                "V(q) must return shape "
                f"({npoint}, {self.nstate}, {self.nstate}), got {V.shape}."
            )

        if dV.shape == (self.ndim, self.nstate, self.nstate):
            dV = dV[None, :, :, :]
        if self.ndim == 1 and dV.shape == (npoint, self.nstate, self.nstate):
            dV = dV[:, None, :, :]
        if dV.shape != (npoint, self.ndim, self.nstate, self.nstate):
            raise ValueError(
                "dV(q) must return shape "
                f"({npoint}, {self.ndim}, {self.nstate}, {self.nstate}), got {dV.shape}."
            )
        return q_batch, V, dV

    def _compute_force(self, dV, coeff):
        npoint = dV.shape[0]
        force = np.zeros((npoint, self.nstate, self.nstate, self.ndim), dtype=float)
        for ipoint in range(npoint):
            c = coeff[ipoint]
            for dim in range(self.ndim):
                force[ipoint, :, :, dim] = -(c.T @ dV[ipoint, dim] @ c)
        return force

    def _compute_derivative_coupling(self, dV, coeff, energies):
        npoint = dV.shape[0]
        out = np.zeros((npoint, self.nstate, self.nstate, self.ndim), dtype=float)
        for ipoint in range(npoint):
            c = coeff[ipoint]
            projected = np.zeros((self.ndim, self.nstate, self.nstate), dtype=float)
            for dim in range(self.ndim):
                projected[dim] = c.T @ dV[ipoint, dim] @ c
            for ist in range(self.nstate):
                for jst in range(ist + 1, self.nstate):
                    dE = energies[ipoint, jst] - energies[ipoint, ist]
                    if abs(dE) < 1.0e-10:
                        dE = np.copysign(1.0e-10, dE if dE != 0 else 1.0)
                    out[ipoint, ist, jst, :] = projected[:, ist, jst] / dE
                    out[ipoint, jst, ist, :] = -out[ipoint, ist, jst, :]
        return out

    def _normalize_wavefun(self, wavefunc, n_points):
        wf = np.asarray(wavefunc)
        if wf.ndim == 2:
            if wf.shape != (self.nstate, self.nstate):
                raise ValueError(
                    f"wavefunc of shape {wf.shape} is not (nstate, nstate) "
                    f"for nstate={self.nstate}"
                )
            wf = wf[np.newaxis, :, :]
        elif wf.ndim == 3:
            if (
                wf.shape[0] != n_points
                or wf.shape[1] != self.nstate
                or wf.shape[2] != self.nstate
            ):
                raise ValueError(
                    f"wavefunc has incompatible shape {wf.shape}; expected "
                    f"(n_points, nstate, nstate)=({n_points}, {self.nstate}, {self.nstate})"
                )
        else:
            raise ValueError(
                f"wavefunc must have ndim 2 or 3, got ndim={wf.ndim} with shape {wf.shape}"
            )
        return wf

    def _normalize_energies(self, energies, n_points):
        en = np.asarray(energies)
        if en.ndim == 1:
            if en.shape[0] != self.nstate:
                raise ValueError(
                    f"energies of shape {en.shape} do not match nstate={self.nstate}"
                )
            en = en[np.newaxis, :]
        elif en.ndim == 2:
            if en.shape != (n_points, self.nstate):
                raise ValueError(
                    f"energies has incompatible shape {en.shape}; expected "
                    f"({n_points}, {self.nstate})"
                )
        else:
            raise ValueError(
                f"energies must have ndim 1 or 2, got ndim={en.ndim} with shape {en.shape}"
            )
        return en

    def evaluate(
        self,
        q,
        *,
        wavefunc_for_nac=None,
        energies_for_nac=None,
        need_force=True,
        need_nac=True,
        need_wavefun=True,
    ) -> ModelSnapshot:
        q_batch, V, dV = self._prepare_V_dV(q)
        coeff, energies = self._compute_basis_states(V)

        force = None
        force_diag = None
        if need_force:
            force = self._compute_force(dV, coeff)
            diag = np.diagonal(force, axis1=1, axis2=2)
            force_diag = np.moveaxis(diag, -1, 1)

        nac = None
        if need_nac:
            nac_coeff = coeff
            nac_energies = energies
            if wavefunc_for_nac is not None:
                nac_coeff = self._normalize_wavefun(wavefunc_for_nac, V.shape[0])
            if energies_for_nac is not None:
                nac_energies = self._normalize_energies(energies_for_nac, V.shape[0])
            nac = self._compute_derivative_coupling(dV, nac_coeff, nac_energies)

        return ModelSnapshot(
            x=q_batch,
            energies=energies,
            wavefun=coeff if need_wavefun else None,
            force=force,
            force_diag=force_diag,
            nac=nac,
        )

    def get_nac(self, q, wavefunc=None):
        return self.evaluate(
            q,
            wavefunc_for_nac=wavefunc,
            need_force=False,
            need_nac=True,
        ).nac

    def get_force(self, q):
        return self.evaluate(q, need_force=True, need_nac=False).force

    def get_force_diag(self, q):
        return self.evaluate(q, need_force=True, need_nac=False).force_diag

    def get_energies(self, q):
        return self.evaluate(q, need_force=False, need_nac=False).energies

    def get_wavefun(self, q):
        return self.evaluate(q, need_force=False, need_nac=False).wavefun

    def get_properties(self, q, wavefunc=None, need_force=True, need_nac=True):
        return self.evaluate(
            q,
            wavefunc_for_nac=wavefunc,
            need_force=need_force,
            need_nac=need_nac,
        ).as_dict()

    def get_nac_from_wavefun(self, q, wavefunc, energies=None):
        return self.evaluate(
            q,
            wavefunc_for_nac=wavefunc,
            energies_for_nac=energies,
            need_force=False,
            need_nac=True,
        ).nac

    def get_basis_properties(self, q, need_force=False):
        return self.evaluate(q, need_force=need_force, need_nac=False).as_dict()

    @abstractmethod
    def V(self, q):
        pass

    @abstractmethod
    def dV(self, q):
        pass


# Preferred spelling kept alongside the historical name for compatibility.
AdiabaticModel = AbaticModelModel
