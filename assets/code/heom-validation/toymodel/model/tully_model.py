import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Union
from .modelbase import DiabaticModel

"""
Tully model implementations for nonadiabatic molecular dynamics:
1. Simple Avoided Crossing (Tully model 1)
2. Dual Avoided Crossing (Tully model 2) 
3. Extended Coupling with Reflection (Tully model 3)

All models implement:
- V(x): returns potential matrix of shape (n_points, 2, 2)
- dV(x): returns derivative matrix of shape (n_points, 2, 2)
- Standard parameters from Tully's 1991 paper
"""


class TullyModel(DiabaticModel):
    def __init__(
        self, x, representation: str, mass: float = 2000.0, wavefunc_revised=True
    ):
        super().__init__(
            x,
            representation=representation,
            nstate=2,
            wavefunc_revised=wavefunc_revised,
        )
        self.mass = np.array(mass, dtype=np.float64)


class TullySimpleAvoidedCrossing(TullyModel):
    def __init__(
        self,
        x=None,
        representation: str = "adiabatic",
        a=0.01,
        b=1.6,
        c=0.005,
        d=1.0,
        mass=2000.0,
        wavefunc_revised=True,
        bias=False,
    ):
        self.A, self.B, self.C, self.D = a, b, c, d
        self.bias = bias
        super().__init__(x, representation, mass, wavefunc_revised=wavefunc_revised)

    def V(self, X: Union[float, ArrayLike]) -> np.ndarray:
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

    def dV(self, X: Union[float, ArrayLike]) -> np.ndarray:
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

    def plot_model(
        self,
        x_range=(-10, 10),
        num_points=1000,
        nac_only=False,
        adiabatic_only=False,
        diabatic_only=False,
        nac_scale=50,
    ):
        x = np.linspace(*x_range, num_points)
        self.compute(x)
        V, nac, E = self.V(x), self.derivative_coupling, self.hamiltonian
        if diabatic_only:
            plt.plot(x, V[:, 0, 0], "--", color="b", label=r"$\varepsilon_1^d$")
            plt.plot(x, V[:, 1, 1], "--", color="r", label=r"$\varepsilon_2^d$")
        elif adiabatic_only:
            plt.plot(x, E[:, 0], "-", color="b", label=r"$\varepsilon_1^a$")
            plt.plot(x, E[:, 1], "-", color="r", label=r"$\varepsilon_2^a$")
        elif nac_only:
            plt.plot(x, -nac[:, 0, 1], "-", color="g", label="NAC")
        else:
            plt.plot(x, V[:, 0, 0], "--", color="b", label=r"$\varepsilon_1^d$")
            plt.plot(x, V[:, 1, 1], "--", color="r", label=r"$\varepsilon_2^d$")
            plt.plot(x, E[:, 0], "-", color="b", label=r"$\varepsilon_1^a$")
            plt.plot(x, E[:, 1], "-", color="r", label=r"$\varepsilon_2^a$")
            plt.plot(x, -nac[:, 0, 1] / nac_scale, "-", color="g", label="NAC/50")

        plt.ylim(-0.04, 0.04)
        plt.xlabel("Coordinate")
        plt.ylabel("Energy / Coupling")
        plt.legend()
        plt.grid()
        plt.show()


class TullyDualAvoidedCrossing(TullyModel):
    def __init__(
        self,
        x=None,
        representation: str = "adiabatic",
        a=0.1,
        b=0.28,
        c=0.015,
        d=0.06,
        e=0.05,
        mass=2000.0,
        wavefunc_revised=True,
    ):
        self.A, self.B, self.C, self.D, self.E0 = a, b, c, d, e
        super().__init__(x, representation, mass, wavefunc_revised=wavefunc_revised)

    def V(self, X: Union[float, ArrayLike]) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        v11 = np.zeros_like(x)
        v22 = -self.A * np.exp(-self.B * x**2) + self.E0
        v12 = self.C * np.exp(-self.D * x**2)
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = v11
        out[:, 1, 1] = v22
        out[:, 0, 1] = out[:, 1, 0] = v12
        return out[0] if scalar_input else out

    def dV(self, X: Union[float, ArrayLike]) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        v11 = np.zeros_like(x)
        v22 = 2 * self.A * self.B * x * np.exp(-self.B * x**2)
        v12 = -2 * self.C * self.D * x * np.exp(-self.D * x**2)
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = v11
        out[:, 1, 1] = v22
        out[:, 0, 1] = out[:, 1, 0] = v12
        return out[0] if scalar_input else out

    def plot_model(self, x_range=(-10, 10), num_points=1000):
        x = np.linspace(*x_range, num_points)
        self.compute(x)
        V, nac, E = self.V(x), self.derivative_coupling, self.hamiltonian

        plt.plot(x, V[:, 0, 0], "--", color="b", label=r"$\varepsilon_1^d$")
        plt.plot(x, V[:, 1, 1], "--", color="r", label=r"$\varepsilon_2^d$")
        plt.plot(x, E[:, 0], "-", color="b", label=r"$\varepsilon_1^a$")
        plt.plot(x, E[:, 1], "-", color="r", label=r"$\varepsilon_2^a$")
        plt.plot(x, -nac[:, 0, 1] / 12, "-", color="g", label="NAC/12")

        plt.ylim(-0.08, 0.08)
        plt.xlabel("Coordinate")
        plt.ylabel("Energy / Coupling")
        plt.legend()
        plt.grid()
        plt.show()


class TullyExtendedCouplingReflection(TullyModel):
    def __init__(
        self,
        x=None,
        representation: str = "adiabatic",
        a=0.0006,
        b=0.10,
        c=0.90,
        mass=2000.0,
        wavefunc_revised=True,
    ):
        self.A, self.B, self.C = a, b, c
        super().__init__(x, representation, mass, wavefunc_revised=wavefunc_revised)

    def V(self, X: Union[float, ArrayLike]) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        v11 = np.full_like(x, self.A)
        v22 = np.full_like(x, -self.A)
        exp_term = np.exp(-self.C * np.abs(x))
        v12 = np.where(x < 0, self.B * exp_term, self.B * (2 - exp_term))
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = v11
        out[:, 1, 1] = v22
        out[:, 0, 1] = out[:, 1, 0] = v12
        return out[0] if scalar_input else out

    def dV(self, X: Union[float, ArrayLike]) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        v11 = np.zeros_like(x)
        v22 = np.zeros_like(x)
        v12 = self.B * self.C * np.exp(-self.C * np.abs(x))
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = v11
        out[:, 1, 1] = v22
        out[:, 0, 1] = out[:, 1, 0] = v12
        return out[0] if scalar_input else out

    def plot_model(self, x_range=(-10, 10), num_points=1000):
        x = np.linspace(*x_range, num_points)
        self.compute(x)
        V, nac, E = self.V(x), self.derivative_coupling, self.hamiltonian

        plt.plot(x, V[:, 0, 0], "--", color="b", label=r"$\varepsilon_1^d$")
        plt.plot(x, V[:, 1, 1], "--", color="r", label=r"$\varepsilon_2^d$")
        plt.plot(x, E[:, 0], "-", color="b", label=r"$\varepsilon_1^a$")
        plt.plot(x, E[:, 1], "-", color="r", label=r"$\varepsilon_2^a$")
        plt.plot(x, -nac[:, 0, 1], "-", color="g", label="NAC")

        plt.ylim(-0.25, 0.25)
        plt.xlabel("Coordinate")
        plt.ylabel("Energy / Coupling")
        plt.legend()
        plt.grid()
        plt.show()


class TullyDumbbellExtendedCoupling(TullyModel):
    """Dumbbell extended-coupling model used in A-FSSH tests.

    This is the one-dimensional two-state "dumbbell" model from
    Subotnik and Shenvi's augmented FSSH work.  The asymptotic coupling is
    large on both sides and nearly zero in the middle, producing two
    separated nonadiabatic regions that test whether surface hopping carries
    ghost electronic amplitudes after wave-packet branching.
    """

    def __init__(
        self,
        x=None,
        representation: str = "adiabatic",
        a=0.0006,
        b=0.10,
        c=0.90,
        z=10.0,
        mass=2000.0,
        wavefunc_revised=True,
    ):
        self.A, self.B, self.C, self.Z = a, b, c, z
        super().__init__(x, representation, mass, wavefunc_revised=wavefunc_revised)

    def _offdiag(self, x):
        left = x < -self.Z
        center = (x >= -self.Z) & (x <= self.Z)
        right = x > self.Z
        v12 = np.empty_like(x, dtype=np.float64)
        v12[left] = self.B * np.exp(self.C * (x[left] - self.Z)) + self.B * (
            2.0 - np.exp(self.C * (x[left] + self.Z))
        )
        v12[center] = self.B * np.exp(self.C * (x[center] - self.Z)) + self.B * np.exp(
            -self.C * (x[center] + self.Z)
        )
        v12[right] = self.B * (
            2.0 - np.exp(-self.C * (x[right] - self.Z))
        ) + self.B * np.exp(-self.C * (x[right] + self.Z))
        return v12

    def _d_offdiag(self, x):
        left = x < -self.Z
        center = (x >= -self.Z) & (x <= self.Z)
        right = x > self.Z
        dv12 = np.empty_like(x, dtype=np.float64)
        dv12[left] = self.B * self.C * np.exp(self.C * (x[left] - self.Z)) - (
            self.B * self.C * np.exp(self.C * (x[left] + self.Z))
        )
        dv12[center] = self.B * self.C * np.exp(self.C * (x[center] - self.Z)) - (
            self.B * self.C * np.exp(-self.C * (x[center] + self.Z))
        )
        dv12[right] = self.B * self.C * np.exp(-self.C * (x[right] - self.Z)) - (
            self.B * self.C * np.exp(-self.C * (x[right] + self.Z))
        )
        return dv12

    def V(self, X: Union[float, ArrayLike]) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.empty((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 0] = self.A
        out[:, 1, 1] = -self.A
        out[:, 0, 1] = out[:, 1, 0] = self._offdiag(x)
        return out[0] if scalar_input else out

    def dV(self, X: Union[float, ArrayLike]) -> np.ndarray:
        x = np.asarray(X)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.zeros((len(x), 2, 2), dtype=np.float64)
        out[:, 0, 1] = out[:, 1, 0] = self._d_offdiag(x)
        return out[0] if scalar_input else out

    def plot_model(
        self,
        x_range=(-25, 25),
        num_points=1200,
        nac_scale=2.0,
    ):
        x = np.linspace(*x_range, num_points)
        self.compute(x)
        V, nac, E = self.V(x), self.derivative_coupling, self.hamiltonian

        plt.plot(x, V[:, 0, 0], "--", color="b", label=r"$V_{00}^d$")
        plt.plot(x, V[:, 1, 1], "--", color="r", label=r"$V_{11}^d$")
        plt.plot(x, V[:, 0, 1], "--", color="k", label=r"$V_{01}^d$")
        plt.plot(x, E[:, 0], "-", color="b", label=r"$E_0^a$")
        plt.plot(x, E[:, 1], "-", color="r", label=r"$E_1^a$")
        plt.plot(x, -nac[:, 0, 1] / nac_scale, "-", color="g", label="NAC/scaled")
        plt.xlabel("Coordinate")
        plt.ylabel("Energy / coupling")
        plt.legend()
        plt.grid()
        plt.show()
