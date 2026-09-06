"""Part VI: Richardson SC3 and Lawrence Eq. (51), symmetric 1D Eckart.

Run beside eckart_four_methods.py; numpy/scipy/matplotlib required.
All fluxes are k*Q_R in atomic units (hbar=1). This is a specialization
of the uniform instanton formula, NOT exact scattering renamed instanton.
The optional --dvr flag recomputes independent DVR values and convergence.
Without it, previously published Part V DVR values are explicitly reused.
"""
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eckart_four_methods as base

TAUC = 1 / (base.KB * base.TC)
ALPHA = TAUC * base.V0


def uniform_flux(temperature, terms=512):
    """Algebraically pole-free evaluation of Lawrence Eq. (51)/App. A21.

    r=tau/tauc, alpha=tauc*V0, Sn=alpha*(2*n-n*n/r).
    In App. A14 the two endpoint terms sum exactly to -F_TST for
    this action. Add Jn=F_TST*r/(n+r), with alternating signs, to
    obtain this convergent series; no clipping/interpolation at crossover.
    This rearranges pairs from App. A18, not the two separate divergent
    sums. erfcx evaluates exp(z*z)*erfc(z) without overflow for z>0.
    """
    r = base.TC / temperature
    n = np.arange(1., terms+1)
    z = (n-r) * np.sqrt(ALPHA/r)
    half_orbit_ratio = n * np.sqrt(np.pi*ALPHA/r) * erfcx(z)
    combined = half_orbit_ratio - 1 - r/(n+r)
    signs = np.where(n.astype(int) % 2, 1., -1.)
    ratio = 1 + math.fsum((signs*combined).tolist())
    tst = np.exp(-ALPHA*r) / (2*np.pi*TAUC*r)
    return tst * ratio


def direct_uniform_flux(temperature, terms=512):
    """Independent transcription of Eq. (51); only away from its poles."""
    r = base.TC / temperature
    n = np.arange(1., terms+1)
    z = (n-r) * np.sqrt(ALPHA/r)
    signs = np.where(n.astype(int) % 2, 1., -1.)
    pb = np.exp(-ALPHA*r)/(2*TAUC*np.sin(np.pi*r))
    endpoint = n/(2*np.pi*TAUC*r*(r-n))
    orbit = n/TAUC*np.sqrt(ALPHA/(np.pi*r**3))*erfcx(z)/2
    return pb + np.exp(-ALPHA*r)*math.fsum((signs*(endpoint+orbit)).tolist())


def sc3_flux(temperature, tol=1e-10):
    """Richardson 2016 Eq. (17),(19): actual W below, parabola above."""
    beta = 1/(base.KB*temperature)
    energy = (base.B/(2*beta))**2
    shift = beta*energy + base.action(energy) if temperature < base.TC else beta*base.V0
    def integrand(e):
        w = base.action(e) if e < base.V0 else TAUC*(base.V0-e)
        return np.exp(shift-beta*e) * expit(-w)
    value = sum(quad(integrand, lo, hi, epsabs=tol, epsrel=tol)[0]
                for lo, hi in [(0, base.V0), (base.V0, base.V0+50/beta)])
    return np.exp(-shift)*value/(2*np.pi)


def extrapolated_flux(temperature):
    """Deliberately invalid continuation of the Part V energy-saddle formula."""
    beta = 1/(base.KB*temperature)
    e = (base.B/(2*beta))**2
    return np.sqrt(2*np.pi/(base.B/(4*e**1.5))) * np.exp(-beta*e-base.action(e))/(2*np.pi)


def make_figure(path, dvr_rows, mobile=False):
    ts = np.unique(np.r_[np.linspace(100, 400, 361), base.TC, base.TC/2])
    exact = np.array([base.energy_rates(t)['analytic_flux_au'] for t in ts])
    raw = np.array([extrapolated_flux(t) for t in ts])
    law = np.array([uniform_flux(t) for t in ts])
    sc3 = np.array([sc3_flux(t) for t in ts])
    fig, axs = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                            figsize=(6.2, 8.6) if mobile else (12.2, 4.6))
    ax = axs[0]
    ax.semilogy(ts, exact, color='#22324b', lw=1.4, label='Exact scattering')
    ax.semilogy(ts, law, color='#446a93', lw=2, label='Lawrence uniform')
    ax.semilogy(ts, sc3, color='#398077', lw=1.7, ls='--', label='Richardson SC3')
    low = ts < base.TC
    ax.semilogy(ts[low], raw[low], color='#c05d43', lw=1.7, label='Ordinary instanton')
    ax.semilogy(ts[~low], raw[~low], color='#c05d43', lw=1.5, ls=':', label='Invalid extrapolation')
    if dvr_rows:
        ax.scatter([r['T_K'] for r in dvr_rows], [r['DVR_flux_au'] for r in dvr_rows],
                   s=24, color='#22324b', zorder=4, label='DVR')
    ax.set_ylabel(r'Thermal flux $\mathcal{F}=kQ_R$ (atomic units)')
    ax.set_title('A stopped line is not a divergence')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax = axs[1]
    ax.axhline(0, color='#22324b', lw=1)
    ax.plot(ts, 100*(law/exact-1), color='#446a93', lw=2, label='Lawrence uniform')
    ax.plot(ts, 100*(sc3/exact-1), color='#398077', lw=1.7, ls='--', label='Richardson SC3')
    ax.plot(ts[low], 100*(raw[low]/exact[low]-1), color='#c05d43', lw=1.7, label='Ordinary instanton')
    ax.set_ylabel('Relative error against exact scattering (%)')
    ax.set_ylim(-25, 115)
    ax.set_title('Uniform does not mean exact')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    for ax in axs:
        ax.axvline(base.TC, color='#a4937b', ls=':', lw=1)
        ax.text(base.TC+4, .94, r'$T_c$', transform=ax.get_xaxis_transform(), color='#88765f')
        ax.set_xlabel('Temperature (K)')
        ax.set_xlim(95, 405)
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dvr', action='store_true', help='Recompute DVR and its convergence at new benchmark temperatures')
    args = parser.parse_args()
    source = Path(__file__).resolve()
    in_site = source.parent.name == 'reaction-dynamics' and source.parents[1].name == 'code'
    root = source.parents[3] if in_site else source.parent
    data = root/'assets/data/reaction-dynamics' if in_site else root/'uniform-results'
    images = root/'assets/img/reaction-dynamics' if in_site else data
    data.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    dvr_rows = []
    original = data/'eckart-four-methods.csv'
    if original.exists():
        with original.open(encoding='utf-8') as f:
            dvr_rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    temperatures = [100., 150., 200., 240., base.TC, 260., 300., 400.]
    rows = []
    series_error = quadrature_error = 0.
    for t in temperatures:
        r = base.energy_rates(t)
        r['SC3_flux_au'] = sc3_flux(t)
        r['Lawrence_flux_au'] = uniform_flux(t)
        r['invalid_extrapolation_flux_au'] = extrapolated_flux(t) if t >= base.TC else float('nan')
        for key in ['SC3', 'Lawrence', 'instanton']:
            r[key+'_relative_error_exact'] = r[key+'_flux_au']/r['analytic_flux_au']-1
        rows.append(r)
        series_error = max(series_error, abs(uniform_flux(t, 1024)/uniform_flux(t, 512)-1))
        quadrature_error = max(quadrature_error, abs(sc3_flux(t, 1e-12)/sc3_flux(t)-1))
    direct_errors = [abs(direct_uniform_flux(t, 4096)/uniform_flux(t, 4096)-1)
                     for t in [110., 150., 200., 240., 260., 300., 400.]]
    # Cross both the primary and repeated-orbit crossover. No averaging or
    # hand-set values are used at either point.
    continuity = []
    for n in [1, 2]:
        center = base.TC/n
        fc = uniform_flux(center)
        continuity.append(max(abs(uniform_flux(center*(1+s*1e-7))/fc-1) for s in [-1, 1]))
    assert series_error < 1e-5
    assert quadrature_error < 1e-8
    assert max(direct_errors) < 1e-7
    assert max(continuity) < 1e-5
    checks = dict(Tc_K=base.TC, alpha=ALPHA, series_512_to_1024_max_relative_change=series_error,
                  sc3_quadrature_max_relative_change=quadrature_error,
                  direct_eq51_max_relative_difference=max(direct_errors),
                  crossover_relative_changes=continuity,
                  interpretation='Model semiclassical benchmark, not a molecular rate or full-dimensional implementation')
    if args.dvr:
        dvr_checks = []
        variants = [('base', 50., .1, .06, 6000., 10000.),
                    ('finer_grid', 50., .08, .06, 6000., 10000.),
                    ('smaller_box', 35., .1, .06, 6000., 10000.),
                    ('higher_cutoff', 50., .1, .09, 6000., 10000.),
                    ('later_window', 50., .1, .06, 8000., 12000.)]
        for label, bound, dx, cutoff, start, stop in variants:
            if label != 'later_window':
                e, side, _ = base.dvr_spectrum(bound, dx, cutoff)
            else:
                e, side, _ = base.dvr_spectrum()
            for r in rows:
                curve = base.flux_side(e, side, r['T_K'], np.arange(start, stop+1, 100.))
                value = float(curve.mean())
                if label == 'base':
                    r['DVR_flux_au'] = value
                relative = value/r['DVR_flux_au']-1
                err = value/r['analytic_flux_au']-1
                spread = float(curve.std()/curve.mean())
                assert abs(relative) < .001 and abs(err) < .001 and spread < .001
                dvr_checks.append(dict(variant=label, T_K=r['T_K'], DVR_flux_au=value,
                    relative_change_from_base=relative, relative_error_exact=err, plateau_relative_std=spread))
        base.write_csv(data/'eckart-uniform-dvr-checks.csv', dvr_checks)
        checks['DVR_max_relative_error_exact'] = max(abs(r['relative_error_exact']) for r in dvr_checks)
        checks['DVR_max_relative_change'] = max(abs(r['relative_change_from_base']) for r in dvr_checks)
        dvr_rows = rows
    base.write_csv(data/'eckart-uniform.csv', rows)
    with (data/'eckart-uniform-checks.json').open('w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(checks, indent=2)+'\n')
    make_figure(images/'eckart-uniform.png', dvr_rows)
    make_figure(images/'eckart-uniform-mobile.png', dvr_rows, mobile=True)
    print(json.dumps(checks, indent=2))
    for r in rows:
        print(f"{r['T_K']:.6f} K: exact={r['analytic_flux_au']:.7e}, "
              f"SC3/exact={r['SC3_flux_au']/r['analytic_flux_au']:.7f}, "
              f"Lawrence/exact={r['Lawrence_flux_au']/r['analytic_flux_au']:.7f}")


if __name__ == '__main__':
    main()
