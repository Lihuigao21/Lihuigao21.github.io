"""Exact operator-block hierarchy for an explicitly retained finite bath.

This is **not** Gaussian HEOM or a non-Gaussian cumulant HEOM closure.  For
``H = Hs (x) I + I (x) Hb + sum_a Sa (x) Ba`` we retain every electronic
operator ``R[m,n] = <m|rho|n>``.  Its equation is

    i hbar dR_mn/dt = [Hs,R_mn] + sum_k (Hb_mk R_kn - R_mk Hb_kn)
                       + sum_a,k (Ba_mk Sa R_kn - Ba_kn R_mk Sa).

The reduced system density is ``sum_m R[m,m]``.  Bath energy eigenstates
give natural indices, but no basis transformation is performed implicitly.
The caller must transform bath operators and initial states together.
Retaining N bath states costs O(N**2) blocks; removing bath states is a
physical basis truncation, not an auxiliary-density hierarchy-depth cutoff.

Factor propagation represents exactly rho = F F^dagger, without changing
its initial rank or normalizing it.  For pure states this reduces storage,
but does not provide general bath compression or remove bath-basis errors.
Joint vectors use system-major ordering: index = system * N + bath.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply


@dataclass
class FiniteBathResult:
    """No normalization, symmetrization, or clipping is applied to outputs."""

    times: np.ndarray
    density_matrices: np.ndarray
    joint_factors: Optional[np.ndarray] = None
    joint_density_matrices: Optional[np.ndarray] = None


def _hermitian_matrix(value, name):
    array = np.array(value, dtype=complex, copy=True)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(array, array.conj().T, rtol=1e-12, atol=1e-13):
        raise ValueError(f"{name} must be Hermitian")
    return array


class FiniteBathHierarchy:
    """Closed finite-bath hierarchy with a dense joint-Hamiltonian guard.

    ``max_joint_dimension`` bounds the dense Hamiltonian (16 D**2 bytes,
    excluding construction/Krylov workspace). ``max_density_dimension``
    separately limits the more expensive full-density control. Stored joint
    trajectories additionally cost 16 Nt D rank or 16 Nt D**2 bytes. Both
    solvers temporarily retain those trajectories even if store_joint=False;
    that flag controls returned data, not peak working memory. Initial
    density/factors must have unit trace unless ``check_normalized=False``;
    that option permits subnormalized positive projected states (trace <= 1).
    """

    def __init__(
        self,
        system_hamiltonian,
        bath_hamiltonian,
        couplings: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
        *,
        hbar=1.0,
        max_joint_dimension=2048,
        max_density_dimension=256,
    ):
        if not np.isfinite(hbar) or hbar <= 0:
            raise ValueError("hbar must be finite and positive")
        for name, limit in (("max_joint_dimension", max_joint_dimension),
                            ("max_density_dimension", max_density_dimension)):
            if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.system_hamiltonian = _hermitian_matrix(system_hamiltonian, "system_hamiltonian")
        self.bath_hamiltonian = _hermitian_matrix(bath_hamiltonian, "bath_hamiltonian")
        self.nsystem = self.system_hamiltonian.shape[0]
        self.nbath = self.bath_hamiltonian.shape[0]
        self.dimension = self.nsystem * self.nbath
        if self.dimension > max_joint_dimension:
            raise ValueError("joint dimension exceeds max_joint_dimension memory guard")
        self.hbar = float(hbar)
        self.max_density_dimension = int(max_density_dimension)
        terms = []
        for index, pair in enumerate(couplings):
            if len(pair) != 2:
                raise ValueError("couplings must contain (system_operator, bath_operator) pairs")
            system = _hermitian_matrix(pair[0], f"couplings[{index}].system")
            bath = _hermitian_matrix(pair[1], f"couplings[{index}].bath")
            if system.shape != self.system_hamiltonian.shape or bath.shape != self.bath_hamiltonian.shape:
                raise ValueError("coupling dimensions must match system and bath Hamiltonians")
            terms.append((system, bath))
        self.couplings = tuple(terms)
        self.joint_hamiltonian = (
            np.kron(self.system_hamiltonian, np.eye(self.nbath))
            + np.kron(np.eye(self.nsystem), self.bath_hamiltonian)
        )
        for system, bath in self.couplings:
            self.joint_hamiltonian += np.kron(system, bath)
        # Keep both propagation routes tied to the same frozen operator inputs.
        for array in (self.system_hamiltonian, self.bath_hamiltonian, self.joint_hamiltonian):
            array.setflags(write=False)
        for system, bath in self.couplings:
            system.setflags(write=False)
            bath.setflags(write=False)

    def density_to_blocks(self, density):
        """Convert a system-major joint matrix to R[m,n,i,j], without mutation."""
        density = np.asarray(density, dtype=complex)
        if density.shape != (self.dimension, self.dimension):
            raise ValueError("joint density has the wrong shape")
        return density.reshape(self.nsystem, self.nbath, self.nsystem, self.nbath).transpose(1, 3, 0, 2)

    def blocks_to_density(self, blocks):
        """Inverse of density_to_blocks (also works on derivative blocks)."""
        blocks = np.asarray(blocks, dtype=complex)
        if blocks.shape != (self.nbath, self.nbath, self.nsystem, self.nsystem):
            raise ValueError("operator blocks have the wrong shape")
        return blocks.transpose(2, 0, 3, 1).reshape(self.dimension, self.dimension)

    def rhs_blocks(self, blocks):
        """Evaluate the explicit block hierarchy, independently of joint H."""
        blocks = np.asarray(blocks, dtype=complex)
        if blocks.shape != (self.nbath, self.nbath, self.nsystem, self.nsystem):
            raise ValueError("operator blocks have the wrong shape")
        hs, hb = self.system_hamiltonian, self.bath_hamiltonian
        rhs = np.einsum("ab,mnbc->mnac", hs, blocks) - np.einsum("mnab,bc->mnac", blocks, hs)
        rhs += np.einsum("mk,knab->mnab", hb, blocks) - np.einsum("mkab,kn->mnab", blocks, hb)
        for system, bath in self.couplings:
            left = np.einsum("mk,knab->mnab", bath, blocks)
            right = np.einsum("mkab,kn->mnab", blocks, bath)
            rhs += np.einsum("ab,mnbc->mnac", system, left) - np.einsum("mnab,bc->mnac", right, system)
        return (-1j / self.hbar) * rhs

    def _times(self, times):
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        times = np.array(times, dtype=float, copy=True)
        if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
            raise ValueError("times must be a nonempty finite one-dimensional array")
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be nonnegative and strictly increasing")
        return times

    @staticmethod
    def _check_trace(trace, check_normalized):
        if abs(np.imag(trace)) > 1e-11 or not np.isfinite(trace):
            raise ValueError("initial trace must be finite and real")
        trace = float(np.real(trace))
        if trace < -1e-12 or trace > 1 + 1e-10:
            raise ValueError("initial trace must lie between zero and one")
        if check_normalized and not np.isclose(trace, 1.0, rtol=0, atol=1e-10):
            raise ValueError("initial state must have unit trace (or set check_normalized=False)")

    @staticmethod
    def _propagate(generator, initial, times, trace_generator):
        # Uniform output times permit scipy's shared interval Krylov algorithm.
        if len(times) > 1 and np.array_equal(times, np.linspace(times[0], times[-1], len(times))):
            return expm_multiply(generator, initial, start=times[0], stop=times[-1],
                                 num=len(times), endpoint=True, traceA=trace_generator)
        outputs = []
        current = initial.copy()
        previous = 0.0
        for time in times:
            dt = time - previous
            if dt != 0:
                current = expm_multiply(generator * dt, current, traceA=trace_generator * dt)
            outputs.append(current.copy())
            previous = time
        return np.asarray(outputs)

    def evolve_factors(self, factors, times, *, store_joint=False, check_normalized=True):
        """Propagate every column of F with exp(-iHt/hbar); never truncate rank.

        Columns need not be orthogonal, and can encode arbitrary mixed,
        system-bath-correlated states. ``F.shape`` must be (D, rank).
        """
        factors = np.array(factors, dtype=complex, copy=True)
        if factors.ndim != 2 or factors.shape[0] != self.dimension or factors.shape[1] < 1:
            raise ValueError("factors must have shape (joint dimension, positive rank)")
        if not np.all(np.isfinite(factors)):
            raise ValueError("factors must be finite")
        self._check_trace(np.vdot(factors, factors), check_normalized)
        times = self._times(times)
        generator = (-1j / self.hbar) * self.joint_hamiltonian
        trajectory = self._propagate(generator, factors, times, np.trace(generator))
        shaped = trajectory.reshape(len(times), self.nsystem, self.nbath, factors.shape[1])
        density = np.einsum("tibr,tjbr->tij", shaped, shaped.conj())
        return FiniteBathResult(times, density, trajectory if store_joint else None)

    def evolve_density(self, density, times, *, store_joint=False, check_normalized=True):
        """Control solver: exponentiate the explicit block-hierarchy generator.

        Accepts arbitrary positive mixed/correlated joint density matrices.
        Uses matrix-free Liouville propagation, not the factor solver or a
        preassembled joint commutator. The initial PSD check is cubic in D.
        """
        if self.dimension > self.max_density_dimension:
            raise ValueError("full-density control exceeds max_density_dimension memory guard")
        density = _hermitian_matrix(density, "density")
        if density.shape != (self.dimension, self.dimension):
            raise ValueError("joint density has the wrong shape")
        self._check_trace(np.trace(density), check_normalized)
        if np.linalg.eigvalsh(density).min() < -1e-11:
            raise ValueError("initial density must be positive semidefinite")
        times = self._times(times)
        size = self.dimension ** 2

        def matvec(vector):
            joint = np.asarray(vector).reshape(self.dimension, self.dimension)
            return self.blocks_to_density(self.rhs_blocks(self.density_to_blocks(joint))).reshape(size)

        # A Hermitian Hamiltonian induces an anti-Hermitian Liouville generator
        # in the Hilbert-Schmidt metric. This is used only for scipy norm probes.
        generator = LinearOperator((size, size), matvec=matvec,
                                   rmatvec=lambda vector: -matvec(vector), dtype=complex)
        trajectory = self._propagate(generator, density.reshape(size), times, 0.0)
        joint = trajectory.reshape(len(times), self.dimension, self.dimension)
        shaped = joint.reshape(len(times), self.nsystem, self.nbath, self.nsystem, self.nbath)
        reduced = np.einsum("tibjb->tij", shaped)
        return FiniteBathResult(times, reduced, joint_density_matrices=joint if store_joint else None)


__all__ = ["FiniteBathHierarchy", "FiniteBathResult"]
