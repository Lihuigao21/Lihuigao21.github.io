"""Two-state scattering: single-surface dynamics, FSSH, and quantum packets.

Run: python assets/code/reaction-dynamics/nonadiabatic_scattering.py
Dependencies: numpy, scipy, matplotlib. All quantities are in atomic units.
The positive Gaussian Wigner distribution is shared by the trajectory
ensemble and quantum packet. No decoherence correction is applied to FSSH.
Quantum reference: periodic Fourier-grid (Fourier DVR), split-operator
propagation; convergence is checked in space, time, and observation time.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 2000.0
X0, SIGMA = -12.0, 1.0
MOMENTA = [6.0, 12.0, 20.0, 30.0]
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "assets/data/reaction-dynamics"
FIG = ROOT / "assets/img/reaction-dynamics"


def potential(x):
    z = np.sign(x) * .01 * (1 - np.exp(-1.6 * np.abs(x)))
    u = .005 * np.exp(-x*x)
    dz = .016 * np.exp(-1.6 * np.abs(x))
    du = -2*x*u
    r = np.hypot(z, u)
    dr = (z*dz + u*du)/r
    d = (z*du - u*dz)/(2*r*r)
    return z, u, r, dr, d


def quantum(p0, n=2048, dt=.5, halfbox=64., time_factor=40.):
    x = np.linspace(-halfbox, halfbox, n, endpoint=False)
    dx = x[1]-x[0]
    p = 2*np.pi*fftfreq(n, dx)
    z, u, r, _, _ = potential(x)
    # V has eigenvalues +/-r, so its exponential is analytic.
    co, si = np.cos(r*dt/2), -1j*np.sin(r*dt/2)/r
    kinetic = np.exp(-1j*p*p*dt/(2*M))
    psi = np.zeros((2, n), complex)
    psi[0] = np.exp(-(x-X0)**2/(4*SIGMA**2)+1j*p0*(x-X0))
    psi /= np.sqrt(np.sum(abs(psi)**2)*dx)
    def vstep(w):
        return np.array([co*w[0]+si*(z*w[0]+u*w[1]),
                         co*w[1]+si*(u*w[0]-z*w[1])])
    steps = round(time_factor*M/p0/dt)
    for _ in range(steps):
        psi = vstep(ifft(fft(vstep(psi), axis=1)*kinetic, axis=1))
    # Spectral projectors P_+ = (I+V/r)/2, P_- = I-P_+.
    total = np.sum(abs(psi)**2, axis=0)
    vexpect = z*(abs(psi[0])**2-abs(psi[1])**2)
    vexpect += 2*u*np.real(psi[0].conj()*psi[1])
    upper = (total+vexpect/r)/2
    lower = total-upper
    return dict(T_lower=float(np.sum(lower[x>4])*dx),
                T_upper=float(np.sum(upper[x>4])*dx),
                R_lower=float(np.sum(lower[x<-4])*dx),
                R_upper=float(np.sum(upper[x<-4])*dx),
                center=float(np.sum(total[abs(x)<=4])*dx),
                edge=float(np.sum(total[abs(x)>halfbox-8])*dx),
                norm=float(np.sum(total)*dx))


def trajectories(p0, ntraj=4000, dt=.5, hopping=True, seed=20260905):
    rng = np.random.default_rng(seed)
    q = rng.normal(X0, SIGMA, ntraj)
    p = rng.normal(p0, 1/(2*SIGMA), ntraj)
    state = np.zeros(ntraj, dtype=int)
    c = np.zeros((2, ntraj), complex)
    c[0] = 1.
    _, _, r, _, _ = potential(q)
    e_initial = p*p/(2*M)-r
    active = np.ones(ntraj, bool)
    max_de, max_g, norm_error, frustrated = 0., 0., 0., 0
    for _ in range(round(160*M/p0/dt)):
        ix = np.flatnonzero(active)
        if len(ix) == 0:
            break
        qi, pi, a = q[ix], p[ix], state[ix]
        _, _, _, dr, _ = potential(qi)
        ph = pi + .5*dt*(1-2*a)*dr
        qn = qi + dt*ph/M
        _, _, rn, drn, _ = potential(qn)
        pn = ph + .5*dt*(1-2*a)*drn
        if hopping:
            _, _, rm, _, d = potential(.5*(qi+qn))
            vd = ph/M*d
            w = np.hypot(rm, vd)
            # Integrate electronic amplitudes unitarily at the midpoint.
            old = c[:, ix]
            hc = np.array([-rm*old[0]-1j*vd*old[1],
                           1j*vd*old[0]+rm*old[1]])
            new = np.cos(w*dt)*old-1j*np.sin(w*dt)/w*hc
            mid = .5*(old+new)
            popa = abs(mid[a, np.arange(len(ix))])**2
            outflow = (1-2*a)*2*vd*np.real(mid[0].conj()*mid[1])
            g = np.maximum(0., dt*outflow/np.maximum(popa, 1e-14))
            max_g = max(max_g, float(g.max()))
            proposed = rng.random(len(ix)) < np.minimum(1., g)
            delta = 2*rn*(1-2*a)
            allowed = pn*pn/(2*M) >= delta
            frustrated += int(np.sum(proposed & ~allowed))
            hop = proposed & allowed
            pn[hop] = np.sign(pn[hop])*np.sqrt(pn[hop]**2-2*M*delta[hop])
            a[hop] = 1-a[hop]
            c[:, ix] = new
            norm_error = max(norm_error, float(np.max(abs(np.sum(abs(new)**2, axis=0)-1))))
        q[ix], p[ix], state[ix] = qn, pn, a
        max_de = max(max_de, float(np.max(abs(pn*pn/(2*M)+(2*a-1)*rn-e_initial[ix]))))
        active[ix] = ~(((qn>10)&(pn>0)) | ((qn<-18)&(pn<0)))
    out = {}
    for name, mask in [("T", (q>10)&(p>0)), ("R", (q<-18)&(p<0))]:
        for j, label in enumerate(["lower", "upper"]):
            val = np.mean(mask & (state==j))
            out[f"{name}_{label}"] = float(val)
            out[f"{name}_{label}_se"] = float(np.sqrt(val*(1-val)/ntraj))
    out.update(center=float(np.mean(active)), max_energy_error=max_de,
               max_raw_hop_probability=max_g, electronic_norm_error=norm_error,
               frustrated_attempts=frustrated, ntraj=ntraj)
    return out


def write_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, mobile=False):
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3,
                             figsize=(6, 11) if mobile else (14, 4.3), layout="constrained")
    x = np.linspace(-5, 5, 1000)
    z, u, r, _, _ = potential(x)
    ax = axes[0]
    ax.plot(x, z, "--", c="#74869f", label=r"$V_{11},V_{22}$")
    ax.plot(x, -z, "--", c="#74869f")
    ax.plot(x, -r, c="#253955", label=r"$E_-$ (lower)")
    ax.plot(x, r, c="#bf624b", label=r"$E_+$ (upper)")
    ax.set(xlabel="Nuclear coordinate x (bohr)", ylabel="Energy (hartree)", title="One Hamiltonian, two representations")
    ax.legend(frameon=False, fontsize=9)
    for ax, field, title in [(axes[1], "T_upper", "Transmission on upper state"),
                             (axes[2], "R_lower", "Reflection on lower state")]:
        for method, color, marker in [("quantum", "#253955", "o"), ("FSSH", "#bf624b", "s"),
                                      ("single", "#74869f", "^")]:
            selected = [r for r in rows if r["method"]==method]
            ys = [r[field] for r in selected]
            err = np.zeros((2, len(selected)))
            for i, row in enumerate(selected):
                if method == "quantum":
                    continue
                # Wilson intervals remain meaningful when no events occur.
                prob, count, z95 = row[field], row["ntraj"], 1.96
                denom = 1 + z95*z95/count
                center = (prob + z95*z95/(2*count))/denom
                radius = z95*np.sqrt(prob*(1-prob)/count+z95*z95/(4*count*count))/denom
                err[:, i] = [max(0., prob-(center-radius)), max(0., center+radius-prob)]
            ax.errorbar(MOMENTA, ys, yerr=err, color=color, marker=marker,
                        ls="--" if method=="single" else "-", capsize=3,
                        label={"quantum":"Quantum grid", "FSSH":"FSSH (95% sampling bars)",
                               "single":"Lower-surface only"}[method])
        ax.set(xlabel="Mean incident momentum (a.u.)", ylabel="Probability", title=title,
               ylim=(-.001, .027) if field=="R_lower" else (-.03, 1.03))
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(FIG / ("nonadiabatic-scattering-mobile.png" if mobile else "nonadiabatic-scattering.png"), dpi=170)
    plt.close(fig)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for p0 in MOMENTA:
        coarse = quantum(p0)
        fine = quantum(p0, n=4096, dt=.25)
        later = quantum(p0, n=4096, dt=.25, halfbox=80., time_factor=44.)
        channels = ["T_lower", "T_upper", "R_lower", "R_upper"]
        checks.append(dict(p0=p0,
            grid_time_max_difference=max(abs(coarse[k]-fine[k]) for k in channels),
            box_observation_max_difference=max(abs(later[k]-fine[k]) for k in channels),
            center=fine["center"], edge=fine["edge"], norm=fine["norm"]))
        rows.append(dict(p0=p0, method="quantum", **fine))
        for hopping in [False, True]:
            row = dict(p0=p0, method="FSSH" if hopping else "single",
                       **trajectories(p0, hopping=hopping))
            rows.append(row)
        # Same seed and initial Wigner samples; hopping draws diverge after hops.
        halfstep = trajectories(p0, dt=.25)
        checks[-1]["FSSH_halfstep_T_upper"] = halfstep["T_upper"]
        checks[-1]["FSSH_halfstep_R_lower"] = halfstep["R_lower"]
        print(f"p0={p0}: quantum={fine}; FSSH={rows[-1]}", flush=True)
    write_table(DATA / "nonadiabatic-scattering.csv", rows)
    write_table(DATA / "nonadiabatic-convergence.csv", checks)
    plot(rows)
    plot(rows, mobile=True)
    print("Wrote benchmark tables and desktop/mobile figures.", flush=True)


if __name__ == "__main__":
    main()
