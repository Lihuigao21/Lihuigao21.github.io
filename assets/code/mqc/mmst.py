"""Meyer-Miller-Stock-Thoss mapping dynamics.

This module implements a compact adiabatic-basis MMST dynamics method for the
local toy-model API.  The electronic state is represented by continuous mapping
variables ``x_i, p_i``; the nuclei move on the mapping-density mean force.
"""

from __future__ import annotations

import numpy as np

from .methodbase import Method


class MMST(Method):
    """Adiabatic-basis MMST mapping-variable MQC dynamics.

    Parameters
    ----------
    gamma:
        Electronic zero-point parameter in the mapping population estimator
        ``rho_ii = 0.5 * (x_i**2 + p_i**2 - gamma)``.
    electronic_substeps:
        Number of RK4 substeps used for the mapping variables during one
        nuclear time step.
    population_estimator:
        ``"action"`` records the raw MMST action estimator. ``"clipped"``
        clips negative populations and renormalizes, useful for plotting.
    initial_phase:
        ``"random"`` samples mapping phases uniformly; ``"zero"`` places all
        mapping momentum initially at zero.
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
        self.gamma = float(
            options.pop("gamma", options.pop("zero_point_parameter", 1.0))
        )
        self.electronic_substeps = int(options.pop("electronic_substeps", 1))
        self.population_estimator = str(
            options.pop("population_estimator", "action")
        ).lower()
        self.initial_phase = str(options.pop("initial_phase", "random")).lower()
        self.mapping_seed = options.pop("mapping_seed", options.pop("seed", None))
        self.include_coherence_force = bool(
            options.pop("include_coherence_force", True)
        )
        self.renormalize_mapping = bool(options.pop("renormalize_mapping", True))
        self.initial_mapping_x = options.pop("mapping_x", None)
        self.initial_mapping_p = options.pop("mapping_p", None)

        if self.electronic_substeps < 1:
            raise ValueError("electronic_substeps must be >= 1.")
        if self.population_estimator not in {"action", "raw", "clipped"}:
            raise ValueError(
                "population_estimator must be 'action', 'raw', or 'clipped'."
            )
        if self.initial_phase not in {"random", "zero"}:
            raise ValueError("initial_phase must be 'random' or 'zero'.")

        self.mapping_rng = np.random.default_rng(self.mapping_seed)

        super().__init__(
            model=model,
            integrator=integrator,
            distribution=distribution,
            Nstep=Nstep,
            dt=dt,
            startstate=start_state,
            **options,
        )
        min_gamma = -1.0 / self.nstate
        if self.gamma <= min_gamma:
            raise ValueError(
                "gamma must be greater than -1/nstate for the MMST "
                f"constraint space; got gamma={self.gamma:g}, "
                f"nstate={self.nstate}."
            )

    # ------------------------------------------------------------------
    # Mapping variables
    # ------------------------------------------------------------------
    def _initialize_mapping_variables(self):
        if (self.initial_mapping_x is None) != (self.initial_mapping_p is None):
            raise ValueError("mapping_x and mapping_p must be provided together.")

        if self.initial_mapping_x is not None:
            x = np.asarray(self.initial_mapping_x, dtype=float)
            p = np.asarray(self.initial_mapping_p, dtype=float)
            if x.shape != (self.nstate,) or p.shape != (self.nstate,):
                raise ValueError(
                    "mapping_x and mapping_p must both have shape "
                    f"({self.nstate},)."
                )
            self.mapping_x = x.copy()
            self.mapping_p = p.copy()
            z = self._mapping_z()
            self._target_mapping_norm = float(np.vdot(z, z).real)
            return
        if self.gamma < 0.0:
            raise ValueError(
                "negative gamma is allowed by the MMST constraint, but the "
                "focused pure-state initializer cannot represent inactive "
                "states with real mapping radii. Provide mapping_x and "
                "mapping_p explicitly or use a constrained/windowed sampler."
            )

        actions = np.zeros(self.nstate, dtype=float)
        actions[self.startstate] = 1.0
        radii = np.sqrt(np.maximum(2.0 * actions + self.gamma, 0.0))

        if self.initial_phase == "random":
            phase = self.mapping_rng.uniform(0.0, 2.0 * np.pi, size=self.nstate)
        else:
            phase = np.zeros(self.nstate, dtype=float)

        self.mapping_x = radii * np.cos(phase)
        self.mapping_p = radii * np.sin(phase)
        self._target_mapping_norm = 1.0 + 0.5 * self.nstate * self.gamma

    def _mapping_z(self):
        return (self.mapping_x + 1j * self.mapping_p) / np.sqrt(2.0)

    def _set_mapping_from_z(self, z):
        self.mapping_x = np.sqrt(2.0) * np.real(z)
        self.mapping_p = np.sqrt(2.0) * np.imag(z)

    def _rho_from_mapping(self):
        z = self._mapping_z()
        rho = np.outer(z, z.conjugate())
        diag = np.diag_indices(self.nstate)
        rho[diag] -= 0.5 * self.gamma
        return 0.5 * (rho + rho.conjugate().T)

    def _population_from_mapping(self):
        action = np.real(np.diag(self.rho))
        if self.population_estimator in {"action", "raw"}:
            return action.copy()

        clipped = np.clip(action, 0.0, None)
        total = float(np.sum(clipped))
        if total <= 1.0e-14:
            out = np.zeros(self.nstate, dtype=float)
            out[self.startstate] = 1.0
            return out
        return clipped / total

    def _mapping_action(self):
        return 0.5 * (self.mapping_x**2 + self.mapping_p**2 - self.gamma)

    # ------------------------------------------------------------------
    # Model snapshots and forces
    # ------------------------------------------------------------------
    def initialize(self):
        self.poplist = []
        self.mapping_action_list = []
        self.q, self.p = self.distribution.sample()
        self.countstep = 0
        self.funclist = []
        self._initialize_mapping_variables()
        self.rho = self._rho_from_mapping()
        self._set_lightweight_initial(self.q, self.p)
        self._setup_recorder(
            extra_fields=["mapping_x", "mapping_p", "mapping_action", "rho_map"]
        )

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
        H = -1j * self._contract_nac_velocity(nac, velocity)
        H = np.asarray(H, dtype=np.complex128)
        np.fill_diagonal(H, energy)
        return H

    def _mean_force_from(self, snapshot, density=None, energies=None, nac=None):
        D = self.rho if density is None else density
        force_diag = snapshot.force_diag[0]
        energies = snapshot.energies[0] if energies is None else energies
        nac = snapshot.nac[0] if nac is None else nac

        rho_diag = np.real(np.diag(D))
        mean_force = np.tensordot(rho_diag, force_diag, axes=([0], [0]))
        if not self.include_coherence_force:
            return np.asarray(mean_force, dtype=float)

        off = np.zeros_like(mean_force, dtype=float)
        for a in range(self.nstate):
            for b in range(a + 1, self.nstate):
                off += (
                    2.0
                    * np.real(nac[a, b] * D[a, b])
                    * (energies[b] - energies[a])
                )
        return np.asarray(mean_force + off, dtype=float)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def _propagate_mapping(self, H):
        h = self.dt / self.electronic_substeps
        z = self._mapping_z()

        def rhs(z_vec):
            return -1j * (H @ z_vec)

        for _ in range(self.electronic_substeps):
            k1 = rhs(z)
            k2 = rhs(z + 0.5 * h * k1)
            k3 = rhs(z + 0.5 * h * k2)
            k4 = rhs(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if self.renormalize_mapping:
                norm = float(np.vdot(z, z).real)
                if norm > 1.0e-30:
                    z *= np.sqrt(self._target_mapping_norm / norm)

        self._set_mapping_from_z(z)
        self.rho = self._rho_from_mapping()

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
        self.energy = energy
        self.nac = nac
        self.wavefun = wavefun
        self.funclist.append(wavefun.copy())

        H = self._hamiltonian_from(energy, nac, self.p / self.m)
        self.RK4(
            initial_snapshot=snap,
            initial_energies=energy,
            initial_nac=nac,
        )
        self._propagate_mapping(H)

        record_snap, record_energy, record_nac, record_wavefun = (
            self._observables_at_current_q(wavefun)
        )
        self.energy = record_energy
        self.nac = record_nac
        self.wavefun = record_wavefun
        self.force = self._mean_force_from(
            record_snap, energies=record_energy, nac=record_nac
        )

        pop = self._population_from_mapping()
        action = self._mapping_action()
        self.poplist.append(pop.copy())
        self.mapping_action_list.append(action.copy())
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
            mapping_x=self.mapping_x,
            mapping_p=self.mapping_p,
            mapping_action=action,
            rho_map=self.rho,
        )
        self.countstep += 1

    def run(self):
        self.initialize()

        while self.countstep < self.Nstep:
            self.step()

        self._save_lightweight_summary(
            pop=np.array(self.poplist, copy=True),
            extra={"mapping_action": np.array(self.mapping_action_list, copy=True)},
        )
        if self.is_record:
            self._finalize_traj_output(
                legacy_order=["q", "energy", "p", "force", "nac", "pop"]
            )
            self.result = self.traj_result if self.legacy_result else self.traj_output
            if self.output is not None:
                self.output = self.traj_output
        return self.result


Mmst = MMST
