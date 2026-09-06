"""Part V: one barrier, four thermal-rate approximations.

Run from any directory: python eckart_four_methods.py
Dependencies: numpy, scipy, matplotlib. All calculations use atomic units.
When kept in the website tree, outputs go to assets/data and assets/img;
otherwise they go into an eckart-results folder beside this script.

DVR is an independent Hamiltonian calculation, NOT analytic transmission
relabeled as DVR. It uses the symmetrically thermalized flux-side function,
not the ordinary or Kubo-transformed function. The Eckart analytic solution
is only a validation reference. Values are flux numerators F=k*Q_R; no
absolute molecular rate is claimed for this unbound scattering coordinate.

Instanton here means the leading 1D Gaussian energy-saddle formula. It is
not a bead optimizer, a uniform crossover approximation, or RPMD.
"""
from pathlib import Path
import csv

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, toeplitz
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M, V0, A = 1836.152673, 0.01, 1.5
KB, AU_TO_FS = 3.166811563e-6, 0.024188843
B = 2 * np.pi * np.sqrt(2 * M) / A
TC = A * np.sqrt(2 * V0 / M) / (2 * np.pi * KB)
TEMPERATURES = [100., 150., 180., 200., 230., 240., 260., 300.]


def potential(x):
    return V0 / np.cosh(A * x)**2


def action(energy):
    """Round-trip forbidden-region action; hbar=1."""
    return B * (np.sqrt(V0) - np.sqrt(energy))


def analytic_transmission(energy):
    z = np.pi * np.sqrt(2 * M * np.maximum(energy, 0)) / A
    c = 0.5 * np.pi * np.sqrt(8 * M * V0 / A**2 - 1)
    with np.errstate(divide="ignore"):
        logsinh = z + np.log(-np.expm1(-2*z)) - np.log(2)
    logcosh = c + np.log1p(np.exp(-2*c)) - np.log(2)
    return expit(2 * (logsinh - logcosh))


def energy_rates(temperature, tolerance=1e-10):
    beta = 1 / (KB * temperature)
    energy = (B / (2 * beta))**2
    phi = beta * energy + action(energy)
    shift = phi if temperature < TC else beta * V0
    scale = np.exp(-shift) / (2*np.pi)
    def integrate(function, low, high):
        return quad(function, low, high, epsabs=1e-12,
                    epsrel=tolerance)[0]
    tst = np.exp(-beta * V0) / (2*np.pi*beta)
    wkb = scale * integrate(
        lambda e: np.exp(shift-beta*e-action(e)), 0, V0) + tst
    analytic = scale * sum(integrate(
        lambda e: np.exp(shift-beta*e)*analytic_transmission(e), low, high)
        for low, high in [(0, V0), (V0, V0+50/beta)])
    curvature = B / (4*energy**1.5)
    instanton = (np.sqrt(2*np.pi/curvature)*np.exp(-phi)/(2*np.pi)
                 if temperature < TC else np.nan)
    return dict(T_K=temperature, TST_flux_au=tst, WKB_flux_au=wkb,
                instanton_flux_au=instanton, analytic_flux_au=analytic,
                E_star_au=energy if temperature < TC else np.nan,
                sigma_E_au=1/np.sqrt(curvature) if temperature < TC else np.nan)


def dvr_spectrum(bound=50., spacing=0.1, cutoff=0.06):
    n = round(2*bound/spacing)+1
    x = np.linspace(-bound, bound, n)
    dx = x[1]-x[0]
    offsets = np.arange(1, n)
    row = np.r_[np.pi**2/3, 2*(-1.)**offsets/offsets**2]
    hamiltonian = toeplitz(row/(2*M*dx**2)) + np.diag(potential(x))
    energies, vectors = eigh(hamiltonian, subset_by_value=(-1., cutoff))
    positive = vectors[x > 0, :]
    side = positive.T @ positive
    return energies, side, n


def flux_side(energies, side, temperature, times):
    """Tr[e^(-beta H/2) F e^(-beta H/2) h(t)], F=i[H,h].

    The real spectral sum is sum_nm e^[-beta(En+Em)/2]
    (En-Em)*|h_nm|^2*sin[(En-Em)t]. No analytic P(E) is used.
    """
    beta = 1/(KB*temperature)
    delta = energies[:, None]-energies[None, :]
    amplitude = (np.exp(-0.5*beta*(energies[:, None]+energies[None, :]))
                 * delta * np.abs(side)**2)
    return np.array([np.sum(amplitude*np.sin(delta*t)) for t in times])


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figures(rows, path, mobile=False):
    temperatures = np.linspace(100, 300, 241)
    smooth = [energy_rates(t) for t in temperatures]
    fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                             figsize=(6, 8.8) if mobile else (11.5, 4.3))
    colors = {"TST": "#78828f", "WKB": "#398077", "instanton": "#c05d43"}
    for method in colors:
        axes[0].semilogy(temperatures, [r[method+"_flux_au"] for r in smooth],
                         color=colors[method], label=method, lw=1.8)
    axes[0].scatter([r["T_K"] for r in rows], [r["DVR_flux_au"] for r in rows],
                    s=30, color="#22324b", label="DVR", zorder=4)
    axes[0].set_ylabel(r"thermal flux $\mathcal{F}=kQ_R$ (atomic units)")
    axes[0].set_title("Same potential, four calculations")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].axhline(0, color="#22324b", lw=1, label="DVR reference")
    for method in colors:
        axes[1].plot([r["T_K"] for r in rows],
                     [100*(r[method+"_flux_au"]/r["DVR_flux_au"]-1) for r in rows],
                     "o-", color=colors[method], label=method, lw=1.5, ms=4)
    axes[1].set_ylabel(r"relative error against DVR (%)")
    axes[1].set_ylim(-105, 90)
    axes[1].set_title("A plausible trend is not exact agreement")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.axvline(TC, color="#a4937b", ls=":", lw=1.2)
        ax.text(TC+3, 0.94, r"$T_c$", transform=ax.get_xaxis_transform(),
                color="#88765f")
        ax.set_xlabel("temperature (K)")
        ax.set_xlim(95, 305)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    source = Path(__file__).resolve()
    if source.parent.name == "reaction-dynamics" and source.parents[1].name == "code":
        root = source.parents[3]
        data, images = root/"assets/data/reaction-dynamics", root/"assets/img/reaction-dynamics"
    else:
        data = images = source.parent/"eckart-results"
    data.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    times = np.arange(6000., 10001., 100.)
    energy, side, n_grid = dvr_spectrum()
    rows, plateaus, checks = [], [], []
    for temperature in TEMPERATURES:
        row = energy_rates(temperature)
        curve = flux_side(energy, side, temperature, times)
        row.update(DVR_flux_au=float(curve.mean()), DVR_window_std_au=float(curve.std()))
        for method in ["TST", "WKB", "instanton", "DVR"]:
            row[method+"_over_DVR"] = row[method+"_flux_au"]/row["DVR_flux_au"]
        row["DVR_relative_error_analytic"] = row["DVR_flux_au"]/row["analytic_flux_au"]-1
        assert abs(row["DVR_relative_error_analytic"]) < 0.001
        assert np.std(curve)/np.mean(curve) < 0.001
        rows.append(row)
        for t, value in zip(times, curve):
            plateaus.append(dict(T_K=temperature, time_au=t, time_fs=t*AU_TO_FS,
                                 DVR_flux_au=value, analytic_flux_au=row["analytic_flux_au"]))
        tighter = energy_rates(temperature, tolerance=1e-12)
        for method in ["WKB", "analytic"]:
            assert abs(tighter[method+"_flux_au"]/row[method+"_flux_au"]-1) < 1e-8
    variants = [("base", 50., .1, .06, 6000., 10000.),
                ("smaller_box", 35., .1, .06, 6000., 10000.),
                ("finer_grid", 50., .08, .06, 6000., 10000.),
                ("higher_cutoff", 50., .1, .09, 6000., 10000.),
                ("later_window", 50., .1, .06, 8000., 12000.)]
    for label, bound, dx, cutoff, start, stop in variants:
        if label in ["base", "later_window"]:
            e, h, n = energy, side, n_grid
        else:
            e, h, n = dvr_spectrum(bound, dx, cutoff)
        sample_times = np.arange(start, stop+1, 100.)
        for row in rows:
            curve = flux_side(e, h, row["T_K"], sample_times)
            value = float(curve.mean())
            relative = value/row["DVR_flux_au"]-1
            assert abs(relative) < 0.001
            checks.append(dict(variant=label, T_K=row["T_K"], bound_bohr=bound,
                               spacing_bohr=dx, cutoff_Ha=cutoff, n_grid=n,
                               n_states=len(e), window_start_au=start, window_stop_au=stop,
                               DVR_flux_au=value, relative_change_from_base=relative,
                               relative_error_analytic=value/row["analytic_flux_au"]-1,
                               window_relative_std=float(curve.std()/curve.mean())))
    # Independent quadrature of W verifies the analytic action used by both approximations.
    for e in V0*np.array([.05, .2, .5, .8, .95]):
        turning_point = np.arccosh(np.sqrt(V0/e))/A
        numerical = 4*quad(lambda x: np.sqrt(max(0., 2*M*(potential(x)-e))),
                           0, turning_point, epsabs=1e-12, epsrel=1e-11)[0]
        assert abs(numerical/action(e)-1) < 1e-9
    write_csv(data/"eckart-four-methods.csv", rows)
    write_csv(data/"eckart-four-methods-convergence.csv", checks)
    write_csv(data/"eckart-four-methods-plateaus.csv", plateaus)
    figures(rows, images/"eckart-four-methods.png")
    figures(rows, images/"eckart-four-methods-mobile.png", mobile=True)
    print(f"Tc = {TC:.6f} K; DVR grid = {n_grid}; states = {len(energy)}")
    for row in rows:
        print(f"{row['T_K']:5.0f} K: TST/DVR={row['TST_over_DVR']:.6g}, "
              f"WKB/DVR={row['WKB_over_DVR']:.6f}, inst/DVR={row['instanton_over_DVR']:.6f}")
    print("max DVR/analytic relative error:", max(abs(r["DVR_relative_error_analytic"]) for r in rows))
    print("max convergence relative change:", max(abs(r["relative_change_from_base"]) for r in checks))
    print("All numerical assertions passed.")


if __name__ == "__main__":
    main()
