"""Standalone CPS/eCMM basis-control adapter from ToyModel.

Atomic units, one nuclear coordinate, smooth complete electronic frame.
Dependencies: NumPy. See cps_basis_benchmark.py for reproducible experiments.
The implementation is a coordinate transformation of a classical mapping
Hamiltonian, not a new quantization of a covariant nuclear kinetic operator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def validate_density(density, nstate, *, tolerance=1.e-10):
    """Validate a physical initial density without normalizing or clipping it."""
    rho = np.asarray(density, dtype=complex)
    if rho.shape != (nstate, nstate) or not np.all(np.isfinite(rho)):
        raise ValueError(f"initial_density must be finite with shape ({nstate}, {nstate}).")
    if not np.allclose(rho, rho.conj().T, atol=tolerance, rtol=0):
        raise ValueError("initial_density must be Hermitian.")
    if abs(np.trace(rho) - 1) > tolerance:
        raise ValueError("initial_density must have trace one.")
    if np.linalg.eigvalsh(rho).min() < -tolerance:
        raise ValueError("initial_density must be positive semidefinite.")
    return rho.copy()


def cps_gamma(nstate, kernel="W"):
    """Q/P/W parameters in the CPS convention (half the old spin gamma)."""
    if not isinstance(nstate, (int, np.integer)) or isinstance(nstate, (bool, np.bool_)) or nstate < 1:
        raise ValueError("nstate must be a positive integer.")
    name = str(kernel).upper()
    if name == "Q":
        return 0.0
    if name == "P":
        return 1.0
    if name == "W":
        return (np.sqrt(nstate + 1.) - 1.) / nstate
    raise ValueError("kernel must be Q, P or W.")


@dataclass(frozen=True)
class CPSKernel:
    """A scalar CPS representation together with its required dual kernel."""

    nstate: int
    gamma: float

    def __post_init__(self):
        f = self.nstate
        if not isinstance(f, (int, np.integer)) or isinstance(f, (bool, np.bool_)) or f < 1:
            raise ValueError("nstate must be a positive integer.")
        if isinstance(self.gamma, (bool, np.bool_)) or not np.isscalar(self.gamma):
            raise ValueError("gamma must be a finite real scalar.")
        gamma = float(self.gamma)
        if not np.isfinite(gamma) or gamma <= -1. / f:
            raise ValueError("gamma must be finite and greater than -1/nstate.")
        object.__setattr__(self, "nstate", int(f))
        object.__setattr__(self, "gamma", gamma)

    @classmethod
    def from_sw(cls, nstate, kernel="W"):
        return cls(nstate, cps_gamma(nstate, kernel))

    @property
    def radius_squared(self):
        """Squared radius of z, i.e. total unshifted oscillator action."""
        return 1. + self.nstate * self.gamma

    @property
    def dual_gamma(self):
        """Dual parameter on the same unit vector, not on the same z radius."""
        return (1. - self.gamma) / self.radius_squared

    @property
    def is_self_dual(self):
        return self.nstate == 1 or bool(np.isclose(self.radius_squared**2, self.nstate + 1., atol=1.e-12, rtol=1.e-12))

    def validate_z(self, z):
        z = np.asarray(z, dtype=complex)
        if z.ndim < 1 or z.shape[-1] != self.nstate or not np.all(np.isfinite(z)):
            raise ValueError("z must have a finite final electronic-state axis.")
        if not np.allclose(np.sum(abs(z)**2, axis=-1), self.radius_squared, atol=1.e-10, rtol=1.e-10):
            raise ValueError("mapping_z does not satisfy this kernel's CPS sphere constraint.")
        return z

    def sample(self, size=None, *, rng=None):
        """Independent samples of z from the normalized complex sphere."""
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        if size is None:
            shape = ()
        else:
            shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
            if any(not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1 for k in shape):
                raise ValueError("sample dimensions must be positive integers.")
        z = rng.normal(size=shape + (self.nstate,)) + 1j*rng.normal(size=shape + (self.nstate,))
        norm = np.linalg.norm(z, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise RuntimeError("sphere RNG returned a zero vector.")
        return z * np.sqrt(self.radius_squared) / norm

    def mapping(self, z):
        z = self.validate_z(z)
        return z[..., :, None] * z[..., None, :].conj() - self.gamma * np.eye(self.nstate)

    def dual(self, z):
        z = self.validate_z(z)
        c = self.radius_squared
        return (self.nstate + 1.)/c**2 * z[..., :, None]*z[..., None, :].conj() - (1.-self.gamma)/c*np.eye(self.nstate)

    def dual_from_mapping(self, matrix):
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.shape[-2:] != (self.nstate, self.nstate):
            raise ValueError("mapping matrix has the wrong electronic shape.")
        identity = np.eye(self.nstate)/self.nstate
        return identity + (self.nstate+1.)/self.radius_squared**2 * (matrix-identity)

    def symbol(self, operator, z, *, dual=False):
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (self.nstate, self.nstate) or not np.all(np.isfinite(operator)):
            raise ValueError("operator must be a finite electronic matrix.")
        matrix = self.dual(z) if dual else self.mapping(z)
        return np.einsum("ij,...ji->...", operator, matrix)

    def density_estimator(self, z, z_initial, initial_density):
        """F*Tr[rho0*K(z0)]*D(z), to average with equal sphere weights."""
        rho = validate_density(initial_density, self.nstate)
        weight = self.symbol(rho, z_initial).real
        return self.nstate * weight[..., None, None] * self.dual(z)

    def observable_estimator(self, operator, z, z_initial, initial_density):
        rho = self.density_estimator(z, z_initial, initial_density)
        return np.einsum("ij,...ji->...", np.asarray(operator), rho)


def real_two_state_frame(potential, gradient):
    """Smooth frame for a real two-state avoided crossing with positive V01.

    Columns of ``vectors`` are adiabatic eigenvectors. A fixed angle gauge is
    used instead of phase matching between independent trajectory samples.
    This patch supports V01 >= 0; it is not a global conical-intersection gauge.
    """
    v = np.asarray(potential, dtype=float)
    dv = np.asarray(gradient, dtype=float)
    if v.shape != dv.shape or v.shape[-2:] != (2, 2):
        raise ValueError("potential/gradient must have matching (..., 2, 2) shapes.")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(dv)):
        raise ValueError("potential/gradient must be finite.")
    if not np.allclose(v, v.swapaxes(-1, -2), atol=1.e-13, rtol=0):
        raise ValueError("potential must be real symmetric.")
    if not np.allclose(dv, dv.swapaxes(-1, -2), atol=1.e-13, rtol=0):
        raise ValueError("gradient must be real symmetric.")
    a, b = .5*(v[..., 0, 0]-v[..., 1, 1]), v[..., 0, 1]
    da, db = .5*(dv[..., 0, 0]-dv[..., 1, 1]), dv[..., 0, 1]
    if np.any(b < 0):
        raise ValueError("this smooth frame requires nonnegative offdiagonal potential.")
    radius = np.hypot(a, b)
    if np.any(radius < 1.e-12):
        raise ValueError("degenerate two-state frame is undefined.")
    theta = np.arctan2(b, a)
    sine, cosine = np.sin(theta/2), np.cos(theta/2)
    vectors = np.empty_like(v)
    vectors[..., 0, 0], vectors[..., 1, 0] = -sine, cosine
    vectors[..., 0, 1], vectors[..., 1, 1] = cosine, sine
    center = .5*(v[..., 0, 0]+v[..., 1, 1])
    dc = .5*(dv[..., 0, 0]+dv[..., 1, 1])
    dr = (a*da+b*db)/radius
    energy = np.stack((center-radius, center+radius), axis=-1)
    grad = np.stack((dc-dr, dc+dr), axis=-1)
    nac = np.zeros_like(v)
    nac[..., 0, 1] = .5*(a*db-b*da)/radius**2
    nac[..., 1, 0] = -nac[..., 0, 1]
    return {"energy": energy, "vectors": vectors, "gradient": grad, "nac": nac}


class CPSBasisRK4:
    """Independent batched diabatic or adiabatic integration of scalar eCMM.

    q/p have shape (N,), z has shape (N,F). The supplied initial z and density
    always refer to the fixed diabatic basis, even for adiabatic propagation.
    ``adiabatic_provider(q)`` supplies energy, vectors, gradient and nac in a
    smooth frame. It is mandatory for adiabatic propagation and basis readout.
    Single-trajectory signed estimators are neither clipped nor normalized.
    """

    def __init__(self, model, q, p, z, *, mass, gamma, representation="diabatic",
                 initial_density=None, adiabatic_provider=None, frozen_nuclei=False):
        if representation not in {"diabatic", "adiabatic"}:
            raise ValueError("representation must be diabatic or adiabatic.")
        self.model = model
        self.representation = representation
        self.adiabatic_provider = adiabatic_provider
        if representation == "adiabatic" and adiabatic_provider is None:
            raise ValueError("adiabatic propagation requires a smooth adiabatic_provider.")
        self.kernel = CPSKernel(model.nstate, gamma)
        self.gamma, self.nstate = self.kernel.gamma, self.kernel.nstate
        self.q, self.p = np.asarray(q, dtype=float).copy(), np.asarray(p, dtype=float).copy()
        if self.q.ndim != 1 or self.q.shape != self.p.shape or self.q.size == 0:
            raise ValueError("q/p must have matching nonempty one-dimensional batch axes.")
        if not np.all(np.isfinite(self.q)) or not np.all(np.isfinite(self.p)):
            raise ValueError("q/p must be finite.")
        self.mass = float(mass)
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive.")
        self.z = self.kernel.validate_z(z).copy()
        if self.z.shape != (self.q.size, self.nstate):
            raise ValueError("z must have shape (number of trajectories, nstate).")
        if initial_density is None:
            initial_density = np.diag([1., *([0.]*(self.nstate-1))])
        self.initial_density = validate_density(initial_density, self.nstate)
        self.initial_weight = self.nstate*self.kernel.symbol(self.initial_density, self.z).real
        self.frozen_nuclei = bool(frozen_nuclei)
        if representation == "adiabatic":
            u = self.frame_data["vectors"]
            self.z = np.einsum("nji,nj->ni", u.conj(), self.z)
        self.initial_energy = self.mapping_energy()

    @property
    def frame_data(self):
        if self.adiabatic_provider is None:
            raise ValueError("basis readout requires adiabatic_provider.")
        return self.adiabatic_provider(self.q)

    def force_density(self, z=None):
        z = self.z if z is None else z
        return z[..., :, None]*z[..., None, :].conj()-self.gamma*np.eye(self.nstate)

    def _rhs(self, q, p, z):
        k = self.force_density(z)
        if self.representation == "diabatic":
            v, dv = np.asarray(self.model.V(q)), np.asarray(self.model.dV(q))
            zdot = -1j*np.einsum("nij,nj->ni", v, z)
            force = -np.einsum("nij,nji->n", k, dv).real
        else:
            frame = self.adiabatic_provider(q)
            e, d, de = frame["energy"], frame["nac"], frame["gradient"]
            zdot = -1j*e*z
            if not self.frozen_nuclei:
                zdot -= (p/self.mass)[:, None]*np.einsum("nij,nj->ni", d, z)
            # <phi_i|dV|phi_j>=(Ej-Ei)dij off the diagonal.
            grad = (e[:, None, :]-e[:, :, None])*d
            idx = np.arange(self.nstate)
            grad[:, idx, idx] = de
            force = -np.einsum("nij,nji->n", k, grad).real
        if self.frozen_nuclei:
            return np.zeros_like(q), np.zeros_like(p), zdot
        return p/self.mass, force, zdot

    def step(self, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        y = (self.q, self.p, self.z)
        k1 = self._rhs(*y)
        k2 = self._rhs(*(v+.5*dt*k for v, k in zip(y, k1)))
        k3 = self._rhs(*(v+.5*dt*k for v, k in zip(y, k2)))
        k4 = self._rhs(*(v+dt*k for v, k in zip(y, k3)))
        self.q, self.p, self.z = tuple(v+dt*(a+2*b+2*c+d)/6
            for v, a, b, c, d in zip(y, k1, k2, k3, k4))
        # Preserve integration errors for diagnostics; no sphere projection.

    def density_estimator(self, basis="diabatic"):
        c = self.kernel.radius_squared
        dual = (self.nstate+1)/c**2*self.z[..., :, None]*self.z[..., None, :].conj()
        dual -= (1-self.gamma)/c*np.eye(self.nstate)
        if basis not in {"diabatic", "adiabatic"}:
            raise ValueError("basis must be diabatic or adiabatic.")
        if basis != self.representation:
            u = self.frame_data["vectors"]
            if basis == "diabatic":
                dual = u @ dual @ u.conj().swapaxes(-1, -2)
            else:
                dual = u.conj().swapaxes(-1, -2) @ dual @ u
        return self.initial_weight[:, None, None]*dual

    def mapping_energy(self):
        k = self.force_density()
        if self.representation == "diabatic":
            potential = np.einsum("nij,nji->n", k, self.model.V(self.q)).real
        else:
            potential = np.einsum("nii,ni->n", k, self.frame_data["energy"]).real
        return .5*self.p**2/self.mass+potential

    def sphere_error(self):
        return np.sum(abs(self.z)**2, axis=-1)-self.kernel.radius_squared


class TullySAC:
    """ToyModel's real two-state simple avoided crossing (atomic units)."""
    nstate = 2
    def __init__(self):
        self.A, self.B, self.C, self.D = .01, 1.6, .005, 1.
        self.mass, self.bias = 2000., False

    def V(self, X) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        v11 = np.sign(x) * self.A * (1 - np.exp(-self.B * np.abs(x)))
        v22 = -v11
        if self.bias:
            v11 += self.A
            v22 += self.A
        v12 = self.C * np.exp(-self.D * x**2)
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = v11
        out[:, 1, 1] = v22
        out[:, 0, 1] = out[:, 1, 0] = v12
        return out[0] if scalar_input else out

    def dV(self, X) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        v11 = self.A * self.B * np.exp(-self.B * np.abs(x))
        v22 = -v11
        v12 = -2 * self.C * self.D * x * np.exp(-self.D * x**2)
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = v11
        out[:, 1, 1] = v22
        out[:, 0, 1] = out[:, 1, 0] = v12
        return out[0] if scalar_input else out

def tully_frame(model, q):
    return real_two_state_frame(model.V(q), model.dV(q))
