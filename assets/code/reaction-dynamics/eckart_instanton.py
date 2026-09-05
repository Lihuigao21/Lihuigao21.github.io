"""Part V: thermal WKB and a 1D instanton against exact Eckart scattering.

Run: python assets/code/reaction-dynamics/eckart_instanton.py
Dependencies: numpy, scipy, matplotlib. Atomic units unless stated otherwise.
We compare the thermal flux numerator k*Q_R, not absolute molecular rates.
The instanton rate is the leading Gaussian energy-saddle approximation.
The analytic periodic path is sampled to test its ring-polymer discretization;
this script is NOT a general multidimensional instanton optimizer or RPMD.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M, V0, A = 1836.152673, .01, 1.5
KB = 3.166811563e-6
OMEGA = A*np.sqrt(2*V0/M)
TC = OMEGA/(2*np.pi*KB)
B = 2*np.pi*np.sqrt(2*M)/A
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "assets/data/reaction-dynamics"
FIG = ROOT / "assets/img/reaction-dynamics"


def potential(x):
    return V0/np.cosh(A*x)**2


def gradient(x):
    return -2*A*potential(x)*np.tanh(A*x)


def action(e):
    """Abbreviated, round-trip forbidden-region action W(E), hbar=1."""
    return B*(np.sqrt(V0)-np.sqrt(e))


def transmission(e):
    """Exact symmetric Eckart transmission, evaluated stably."""
    z = np.pi*np.sqrt(2*M*np.maximum(e, 0))/A
    c = .5*np.pi*np.sqrt(8*M*V0/A**2-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logsinh = z+np.log(-np.expm1(-2*z))-np.log(2)
    logcosh = c+np.log1p(np.exp(-2*c))-np.log(2)
    return expit(2*(logsinh-logcosh))


def saddle(temperature):
    beta = 1/(KB*temperature)
    if temperature >= TC:
        return np.nan, np.nan, np.nan
    e = (B/(2*beta))**2
    phi = beta*e+action(e)
    curvature = B/(4*e**1.5)
    return e, phi, curvature


def rates(temperature, tolerance=1e-10):
    beta = 1/(KB*temperature)
    e, phi, curvature = saddle(temperature)
    shift = phi if np.isfinite(phi) else beta*V0
    exact_low = quad(lambda en: np.exp(shift-beta*en)*transmission(en),
                     0, V0, epsabs=1e-12, epsrel=tolerance)[0]
    exact_high = quad(lambda en: np.exp(shift-beta*en)*transmission(en),
                      V0, V0+50/beta, epsabs=1e-12, epsrel=tolerance)[0]
    exact = np.exp(-shift)*(exact_low+exact_high)/(2*np.pi)
    tst = np.exp(-beta*V0)/(2*np.pi*beta)
    wkb_low = quad(lambda en: np.exp(shift-beta*en-action(en)),
                   0, V0, epsabs=1e-12, epsrel=tolerance)[0]
    wkb = np.exp(-shift)*wkb_low/(2*np.pi)+tst
    inst = np.sqrt(2*np.pi/curvature)*np.exp(-phi)/(2*np.pi) if temperature<TC else np.nan
    width = 1/np.sqrt(curvature) if temperature<TC else np.nan
    return dict(T_K=temperature, T_over_Tc=temperature/TC, E_star=e,
                instanton_action=phi, gaussian_energy_width=width,
                exact_flux=exact, classical_TST_flux=tst, thermal_WKB_flux=wkb,
                instanton_flux=inst, exact_over_TST=exact/tst, WKB_over_TST=wkb/tst,
                instanton_over_TST=inst/tst, instanton_over_exact=inst/exact,
                WKB_over_exact=wkb/exact)


def periodic_path(temperature, n):
    if temperature>=TC:
        raise ValueError("Nontrivial periodic instanton requires T<Tc in this model.")
    e, _, _ = saddle(temperature)
    phase = np.arange(n)/n
    return phase, np.arcsinh(np.sqrt(V0/e-1)*np.cos(2*np.pi*phase))/A


def path_check(temperature, n):
    beta = 1/(KB*temperature)
    _, x = periodic_path(temperature, n)
    dtau = beta/n
    sn = np.sum(M*(x-np.roll(x,1))**2/(2*dtau)+dtau*potential(x))
    residual = M*(np.roll(x,-1)-2*x+np.roll(x,1))/dtau**2-gradient(x)
    return dict(T_K=temperature, beads=n, discrete_action=float(sn),
                continuum_action=saddle(temperature)[1],
                relative_force_residual=float(np.max(abs(residual))/np.max(abs(gradient(x)))))


def action_quadrature(e):
    turn = np.arccosh(np.sqrt(V0/e))/A
    return 4*quad(lambda x: np.sqrt(max(0,2*M*(potential(x)-e))),
                  0, turn, epsabs=1e-11, epsrel=1e-11)[0]


def ode_transmission(e, bound=8., tolerance=1e-10):
    """Independent stationary Schrodinger integration, outgoing right boundary."""
    k = np.sqrt(2*M*e)
    def rhs(x, y):
        return np.array([y[1],2*M*(potential(x)-e)*y[0]])
    result = solve_ivp(rhs, [bound,-bound], [1.+0j,1j*k], method="DOP853",
                       rtol=tolerance, atol=tolerance*.01)
    if not result.success:
        raise RuntimeError(result.message)
    wave, derivative = result.y[:,-1]
    incident = .5*(wave+derivative/(1j*k))
    reflected = .5*(wave-derivative/(1j*k))
    t = 1/abs(incident)**2
    r = abs(reflected/incident)**2
    return float(t), float(t+r-1)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, mobile=False):
    fig, axes = plt.subplots(4 if mobile else 2, 1 if mobile else 2,
        figsize=(6.4,15) if mobile else (12,8), layout="constrained")
    ax = np.ravel(axes)
    colors = ["#386c91", "#b95d48", "#6e8659"]
    temps = [100.,180.,230.]
    x = np.linspace(-2.4,2.4,500)
    ax[0].plot(x, potential(x)/V0, color="#253955", label=r"$V(x)/V_0$")
    en = np.linspace(.00001,V0,500)
    for t, color in zip(temps, colors):
        e, phi, _ = saddle(t)
        turn = np.arccosh(np.sqrt(V0/e))/A
        ax[0].plot([-turn,turn], [e/V0,e/V0], "o-", ms=4, color=color, label=f"{t:g} K turning points")
        phase, q = periodic_path(t, 1024)
        ax[1].plot(np.r_[phase,1.], np.r_[q,q[0]], color=color, label=f"{t:g} K")
        beta = 1/(KB*t)
        ax[2].plot(en/V0, beta*en+action(en)-phi, color=color, label=f"{t:g} K")
        ax[2].plot(e/V0, 0., "o", color=color)
    ax[0].set(xlabel="Position x (bohr)", ylabel="Energy / barrier height", title="A. Thermally selected tunneling energies")
    ax[1].set(xlabel=r"Imaginary-time fraction $\tau/(\beta\hbar)$", ylabel="Path position x (bohr)", title="B. Periodic instantons, not real-time paths")
    ax[2].set(xlabel=r"Energy $E/V_0$", ylabel=r"$\Phi(E)-\Phi(E_*)$ (dimensionless)", ylim=(-.3,8), title="C. Balance thermal cost and barrier penetration")
    temperatures = np.array([r["T_K"] for r in rows])
    for key, label, color, style in [("exact_over_TST","Exact quantum", "#253955","-"),
           ("WKB_over_TST","Thermally integrated WKB", "#71859e","--"),
           ("instanton_over_TST","Leading 1D instanton", "#b95d48","-")]:
        ax[3].semilogy(temperatures,[r[key] for r in rows],style,color=color,label=label)
    ax[3].axhline(1.,ls=":",color="gray",label="Classical TST")
    ax[3].axvline(TC,ls=":",color="#6e8659",label=f"Tc = {TC:.1f} K")
    ax[3].set(xlabel="Temperature (K)", ylabel=r"Thermal flux / classical TST flux", title="D. A rate benchmark with an exact reference")
    for a in ax:
        a.legend(frameon=False,fontsize=8)
        a.grid(alpha=.18)
    fig.savefig(FIG/("eckart-instanton-mobile.png" if mobile else "eckart-instanton.png"),dpi=170)
    plt.close(fig)


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    temperatures = np.unique(np.r_[np.linspace(80,350,136),100,150,180,200,230,240,248,250,300])
    rows = [rates(float(t)) for t in temperatures]
    write_csv(DATA/"eckart-instanton-rates.csv",rows)
    quadrature = []
    for t in [100.,150.,180.,200.,230.,240.,250.,300.]:
        normal, fine = rates(t), rates(t, tolerance=1e-12)
        quadrature.append(dict(T_K=t,
            exact_relative_change=abs(fine["exact_flux"]/normal["exact_flux"]-1),
            WKB_relative_change=abs(fine["thermal_WKB_flux"]/normal["thermal_WKB_flux"]-1)))
    write_csv(DATA/"eckart-instanton-quadrature.csv", quadrature)
    write_csv(DATA/"eckart-instanton-beads.csv",
        [path_check(t,n) for t in [100.,180.,230.] for n in [32,64,128,256,512,1024]])
    checks=[]
    for e in [.001,.003,.006,.01,.014]:
        ode, err = ode_transmission(e)
        fine, _ = ode_transmission(e,bound=10.,tolerance=1e-12)
        checks.append(dict(E=e,analytic_P=float(transmission(e)),ODE_P=ode,
            fine_ODE_P=fine,flux_balance_error=err,
            action_analytic=float(action(e)) if e<V0 else np.nan,
            action_quadrature=action_quadrature(e) if e<V0 else np.nan))
    write_csv(DATA/"eckart-instanton-checks.csv",checks)
    assert max(abs(c["fine_ODE_P"]/c["analytic_P"]-1) for c in checks) < 1e-8
    assert max(abs(c["flux_balance_error"]) for c in checks) < 1e-8
    assert max(q["exact_relative_change"] for q in quadrature) < 1e-8
    assert max(q["WKB_relative_change"] for q in quadrature) < 1e-8
    pathrows=[]
    for t in [100.,180.,230.]:
        phase,q=periodic_path(t,256)
        pathrows.extend(dict(T_K=t,imaginary_time_fraction=float(s),x=float(v)) for s,v in zip(phase,q))
    write_csv(DATA/"eckart-instanton-paths.csv",pathrows)
    plot(rows)
    plot(rows,mobile=True)
    print(f"Crossover temperature: {TC:.6f} K")
    for t in [100,150,180,200,230,240,250,300]:
        r=rates(t)
        print(t, {k:r[k] for k in ["exact_over_TST","WKB_over_exact","instanton_over_exact"]})


if __name__=="__main__":
    main()
