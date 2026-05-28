"""P-Matrix density-matrix nonadiabatic dynamics.

The implementation follows the density-matrix ensemble formalism of
Kang and Wang, Phys. Rev. B 99, 224303 (2019), adapted to the local
toy-model method API.  It propagates the directed P-matrix variables
``P_ij`` and reconstructs the physical density matrix as
``D_ij = P_ij + P_ji.conjugate()``.
"""

from __future__ import annotations

import numpy as np

from .methodbase import Method
from toymodel.utils.constant import hbar, kB


class PMatrix(Method):
    """Directed density-matrix dynamics with decoherence and detailed balance.

    Parameters
    ----------
    decoherence_time
        Pairwise decoherence time.  May be ``None`` (estimate from force
        differences using the Wong-Rossky expression used in the paper), a
        scalar, a full ``(nstate, nstate)`` matrix, or a callable returning a
        scalar/matrix.
    decoherence_width
        Gaussian width ``a`` in the force-difference decoherence estimate.
    carrier
        ``"electron"`` penalizes transitions to higher adiabatic energy;
        ``"hole"`` reverses the detailed-balance direction.
    detailed_balance
        Whether to apply the Boltzmann correction in Eq. 13.
    nuclear_force
        ``"density"`` uses a density-matrix mean force for toy-model
        back-reaction.  ``"reference"`` follows one reference adiabatic
        surface, closer to the NBRA setting of the paper.  ``"free"`` keeps
        the momentum constant.
    """

    def __init__(
        self,
        model=None,
        integrator=None,
        distribution=None,
        Nstep=1000,
        dt=1,
        start_state=0,
        **options,
    ):
        self.is_HST = options.pop("is_HST", False)
        self.is_record = options.get("is_record", True)
        self.decoherence_time = options.pop("decoherence_time", None)
        self.decoherence_width = float(options.pop("decoherence_width", 1.0))
        self.min_decoherence_time = float(options.pop("min_decoherence_time", 1.0e-10))
        self.max_decoherence_time = float(options.pop("max_decoherence_time", np.inf))
        self.detailed_balance = bool(options.pop("detailed_balance", True))
        self.carrier = str(options.pop("carrier", "electron")).lower()
        self.electronic_substeps = int(options.pop("electronic_substeps", 1))
        self.include_coherence_force = bool(options.pop("include_coherence_force", True))
        self.nuclear_force = str(options.pop("nuclear_force", "density")).lower()
        self.reference_state = int(options.pop("reference_state", start_state))
        if self.carrier not in {"electron", "hole"}:
            raise ValueError("carrier must be either 'electron' or 'hole'.")
        if self.nuclear_force not in {"density", "reference", "free"}:
            raise ValueError("nuclear_force must be 'density', 'reference', or 'free'.")
        if self.electronic_substeps < 1:
            raise ValueError("electronic_substeps must be >= 1.")

        super().__init__(
            model=model,
            integrator=integrator,
            distribution=distribution,
            Nstep=Nstep,
            dt=dt,
            startstate=start_state,
            **options,
        )

    # ------------------------------------------------------------------
    # Initial state and observables
    # ------------------------------------------------------------------
    def initialize(self):
        self.poplist = []
        self.pmatrix_list = []
        self.q, self.p = self.distribution.sample()
        self._set_lightweight_initial(self.q, self.p)
        self.P = np.zeros((self.nstate, self.nstate), dtype=np.complex128)
        self.P[self.startstate, self.startstate] = 0.5
        self.rho = self._density_from_pmatrix(self.P)
        self.countstep = 0
        self._setup_recorder(extra_fields=["p_matrix"])

    @staticmethod
    def _density_from_pmatrix(P):
        D = P + P.conjugate().T
        return 0.5 * (D + D.conjugate().T)

    def _renormalize_pmatrix(self, P):
        P = np.asarray(P, dtype=np.complex128).copy()
        D = self._density_from_pmatrix(P)
        trace = float(np.real(np.trace(D)))
        if np.isfinite(trace) and abs(trace) > 1.0e-14:
            P /= trace

        pops = np.real(np.diag(self._density_from_pmatrix(P)))
        if np.any(pops < -1.0e-8):
            pops = np.clip(pops, 0.0, None)
            total = np.sum(pops)
            if total <= 1.0e-14:
                pops = np.zeros(self.nstate, dtype=float)
                pops[self.startstate] = 1.0
            else:
                pops /= total
            np.fill_diagonal(P, 0.5 * pops)
        else:
            for istate in range(self.nstate):
                P[istate, istate] = 0.5 * max(pops[istate], 0.0)
        return P

    def _snapshot_at_current_q(self):
        snap = self.model.evaluate(
            self.q, need_force=True, need_nac=(self.countstep == 0)
        )
        energy = snap.energies[0]
        wavefun = snap.wavefun[0]

        if self.countstep != 0:
            wavefun = self.phase_correction(self.funclist[-1], wavefun)
            if self.is_HST:
                nac = self.HST_nac_by_velocity(self.funclist[-1], wavefun)
            else:
                nac = self.model.evaluate(
                    self.q,
                    wavefunc_for_nac=wavefun,
                    energies_for_nac=energy,
                    need_force=False,
                    need_nac=True,
                ).nac[0]
        else:
            nac = snap.nac[0]

        return snap, energy, nac, wavefun

    def _observables_at_current_q(self, previous_wavefun):
        snap = self.model.evaluate(self.q, need_force=True, need_nac=False)
        energy = snap.energies[0]
        wavefun = self.phase_correction(previous_wavefun, snap.wavefun[0])
        if self.is_HST:
            nac = self.HST_nac_by_velocity(previous_wavefun, wavefun)
        else:
            nac = self.model.evaluate(
                self.q,
                wavefunc_for_nac=wavefun,
                energies_for_nac=energy,
                need_force=False,
                need_nac=True,
            ).nac[0]
        return snap, energy, nac, wavefun

    def _hamiltonian_from(self, energy, nac, velocity):
        V = -1j * self._contract_nac_velocity(nac, velocity)
        V = np.asarray(V, dtype=np.complex128)
        np.fill_diagonal(V, energy)
        return V

    def _mean_force_from(self, snapshot, density=None, energies=None, nac=None):
        D = self.rho if density is None else density
        force_diag = snapshot.force_diag[0]
        energies = snapshot.energies[0] if energies is None else energies
        nac = snapshot.nac[0] if nac is None else nac

        if self.nuclear_force == "free":
            return np.zeros_like(force_diag[0], dtype=float)
        if self.nuclear_force == "reference":
            return np.array(force_diag[self.reference_state], copy=True)

        rho_diag = np.real(np.diag(D))
        mean_force = np.tensordot(rho_diag, force_diag, axes=([0], [0]))
        if not self.include_coherence_force:
            return mean_force

        off = np.zeros_like(mean_force, dtype=float)
        for a in range(self.nstate):
            for b in range(a + 1, self.nstate):
                off += (
                    2.0
                    * np.real(nac[a, b] * D[a, b])
                    * (energies[b] - energies[a])
                )
        return mean_force + off

    # ------------------------------------------------------------------
    # Decoherence and detailed balance
    # ------------------------------------------------------------------
    def _temperature(self):
        T = getattr(self.distribution, "T", None)
        return 300.0 if T is None else float(T)

    def _tau_from_force_difference(self, force_diag):
        tau = np.full((self.nstate, self.nstate), np.inf, dtype=float)
        width = max(self.decoherence_width, 1.0e-30)
        for i in range(self.nstate):
            for j in range(self.nstate):
                if i == j:
                    continue
                df = np.asarray(force_diag[i] - force_diag[j], dtype=float).reshape(-1)
                norm = float(np.linalg.norm(df))
                if norm <= 1.0e-14:
                    tau[i, j] = np.inf
                else:
                    tau[i, j] = np.sqrt(2.0 * width) * hbar / norm
        return tau

    def _decoherence_times(self, energy, force_diag):
        source = self.decoherence_time
        if callable(source):
            raw = source(energy=energy, force_diag=force_diag, q=self.q, p=self.p)
        elif source is None:
            raw = self._tau_from_force_difference(force_diag)
        else:
            raw = source

        tau = np.asarray(raw, dtype=float)
        if tau.ndim == 0:
            out = np.full((self.nstate, self.nstate), float(tau), dtype=float)
        else:
            if tau.shape != (self.nstate, self.nstate):
                raise ValueError(
                    "decoherence_time matrix must have shape "
                    f"({self.nstate}, {self.nstate}), got {tau.shape}."
                )
            out = tau.copy()

        out = np.where(np.eye(self.nstate, dtype=bool), np.inf, out)
        out = np.where(out <= 0.0, np.inf, out)
        out = np.clip(out, self.min_decoherence_time, self.max_decoherence_time)
        return out

    def _boltzmann_factor(self, delta):
        if not self.detailed_balance:
            return 1.0
        T = self._temperature()
        if T <= 0.0:
            return 0.0
        arg = -abs(float(delta)) / (kB * T)
        if arg < -745.0:
            return 0.0
        return float(np.exp(arg))

    def _direction_selector(self, delta):
        if self.carrier == "electron":
            return 1.0 if delta > 0.0 else 0.0
        return 0.0 if delta > 0.0 else 1.0

    def _p_rhs(self, P, V, energy, tau):
        comm = V @ P - P @ V
        dP = np.zeros_like(P, dtype=np.complex128)

        for i in range(self.nstate):
            base = -np.real(1j * comm[i, i])
            correction = 0.0
            if self.detailed_balance:
                for j in range(self.nstate):
                    if i == j:
                        continue
                    delta = float(np.real(energy[i] - energy[j]))
                    selector = self._direction_selector(delta)
                    penalty = self._boltzmann_factor(delta) - 1.0
                    correction += (
                        np.real(1j * P[i, j] * V[j, i])
                        * selector
                        * penalty
                    )
                    correction -= (
                        np.real(1j * P[j, i] * V[i, j])
                        * (1.0 - selector)
                        * penalty
                    )
            dP[i, i] = base + correction

        for i in range(self.nstate):
            for j in range(self.nstate):
                if i == j:
                    continue
                decoherence = 0.0 if np.isinf(tau[i, j]) else P[i, j] / tau[i, j]
                dP[i, j] = (
                    -1j * comm[i, j]
                    - 1j * V[i, j] * (P[i, i] + P[j, j].conjugate())
                    - decoherence
                )
        return dP

    def _propagate_pmatrix(self, V, energy, tau):
        h = self.dt / self.electronic_substeps
        P = self.P
        for _ in range(self.electronic_substeps):
            k1 = self._p_rhs(P, V, energy, tau)
            k2 = self._p_rhs(P + 0.5 * h * k1, V, energy, tau)
            k3 = self._p_rhs(P + 0.5 * h * k2, V, energy, tau)
            k4 = self._p_rhs(P + h * k3, V, energy, tau)
            P = P + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            P = self._renormalize_pmatrix(P)
        self.P = P
        self.rho = self._density_from_pmatrix(self.P)

    # ------------------------------------------------------------------
    # Nuclear propagation
    # ------------------------------------------------------------------
    def RK4(self, initial_snapshot=None, initial_energies=None, initial_nac=None):
        q0 = self.q
        p0 = self.p
        m = self.m
        dt = self.dt
        density = self.rho.copy()

        def deriv(q, p, snapshot=None, energies=None, nac=None):
            v = p / m
            snap = snapshot
            if snap is None:
                snap = self.model.evaluate(q, need_force=True, need_nac=True)
            return v, self._mean_force_from(
                snap, density=density, energies=energies, nac=nac
            )

        k1q, k1p = deriv(
            q0,
            p0,
            snapshot=initial_snapshot,
            energies=initial_energies,
            nac=initial_nac,
        )

        q1 = q0 + 0.5 * dt * k1q
        p1 = p0 + 0.5 * dt * k1p
        k2q, k2p = deriv(q1, p1)

        q2 = q0 + 0.5 * dt * k2q
        p2 = p0 + 0.5 * dt * k2p
        k3q, k3p = deriv(q2, p2)

        q3 = q0 + dt * k3q
        p3 = p0 + dt * k3p
        k4q, k4p = deriv(q3, p3)

        dq = (k1q + 2 * k2q + 2 * k3q + k4q) * (dt / 6)
        dp = (k1p + 2 * k2p + 2 * k3p + k4p) * (dt / 6)

        self.q = q0 + dq
        self.p = p0 + dp

    def step(self):
        snap, energy, nac, wavefun = self._snapshot_at_current_q()
        self.funclist.append(wavefun.copy())
        velocity = self.p / self.m
        V = self._hamiltonian_from(energy, nac, velocity)
        tau = self._decoherence_times(energy, snap.force_diag[0])

        self.RK4(
            initial_snapshot=snap,
            initial_energies=energy,
            initial_nac=nac,
        )
        self._propagate_pmatrix(V, energy, tau)

        record_snap, record_energy, record_nac, record_wavefun = (
            self._observables_at_current_q(wavefun)
        )
        self.energy = record_energy
        self.nac = record_nac
        self.wavefun = record_wavefun
        self.force = self._mean_force_from(
            record_snap, energies=record_energy, nac=record_nac
        )

        pop = np.real(np.diag(self.rho))
        self.poplist.append(pop.copy())
        self.pmatrix_list.append(self.P.copy())
        self._record_step(
            q=self.q,
            p=self.p,
            pop=pop,
            state=None,
            energy=record_energy.copy(),
            force=self.force,
            nac=record_nac,
            wavefun=record_wavefun,
            rho=self.rho,
            p_matrix=self.P,
        )
        self.countstep += 1

    def run(self):
        self.initialize()

        while self.countstep < self.Nstep:
            self.step()

        self._save_lightweight_summary(
            pop=np.array(self.poplist, copy=True),
            extra={"p_matrix": np.array(self.pmatrix_list, copy=True)},
        )
        if self.is_record:
            self._finalize_traj_output(
                legacy_order=["q", "energy", "p", "force", "nac", "pop"]
            )
            self.result = self.traj_result if self.legacy_result else self.traj_output
            if self.output is not None:
                self.output = self.traj_output
        return self.result


Pmatrix = PMatrix
P_Matrix = PMatrix
