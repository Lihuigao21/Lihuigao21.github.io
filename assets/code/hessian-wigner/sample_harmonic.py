"""Reproduce the harmonic-Wigner teaching checks and figures.

Requires numpy and matplotlib. Example, from this directory:
  python sample_harmonic.py --output wigner-example --samples 50000 --seed 20261105

The Gaussian draws target the exact *harmonic* thermal Wigner distribution.
They do not replay SHARC 4.1 finite-temperature sampling or run a new quantum-
chemistry calculation. The input contains the actual tutorial motor's xTB modes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

# Declared before evaluating any generated curve; no curve is smoothed.
CURVE_CONTRACT = {
    'grid_points': 1601,
    'maximum_normalized_high_frequency_rms': 5e-4,
    'maximum_normalized_high_frequency_residual': 3e-3,
    'maximum_normalized_grid_refinement_error': 3e-4,
    'maximum_endpoint_excess_ratio': 3.0,
    'maximum_isolated_spikes': 0,
    'scope': 'Analytic teaching curves on stated finite plotting intervals; not NAMD convergence.'
}


def write_json(path, data):
    path.write_bytes((json.dumps(data, indent=2, allow_nan=False) + '\n').encode())


def thermal_factor(freq_cm, temperature, constants):
    omega = 2 * np.pi * constants['c_cm_s'] * np.asarray(freq_cm)
    if temperature == 0:
        return np.ones_like(omega)
    u = constants['hbar_Js'] * omega / (2 * constants['kB_J_K'] * temperature)
    return 1 / np.tanh(u)


def normal_pdf(x, variance):
    return np.exp(-x*x / (2*variance)) / np.sqrt(2*np.pi*variance)


def curve_quality(x, y, exact):
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    increasing = bool(np.all(np.diff(x) > 0))
    peak = max(float(np.max(np.abs(y))), 1e-300)
    lowpass = np.convolve(y, np.ones(5)/5, mode='valid')
    residual = (y[2:-2] - lowpass)/peak
    fine = np.linspace(x[0], x[-1], 2*len(x)-1)
    error = float(np.max(np.abs(exact(fine) - np.interp(fine, x, y)))/peak)
    dy = np.diff(y)
    endpoint_ratio = max(abs(dy[0])/max(abs(dy[1]), peak*1e-12),
                         abs(dy[-1])/max(abs(dy[-2]), peak*1e-12))
    spikes = int(np.sum((np.abs(y[1:-1]-(y[:-2]+y[2:])/2) > .1*peak)
                        & (np.abs(y[:-2]-y[2:]) < .02*peak)))
    result = {'finite': finite, 'strictly_increasing_abscissa': increasing,
              'duplicate_points': int(np.sum(np.diff(x)==0)),
              'normalized_high_frequency_residual_rms': float(np.sqrt(np.mean(residual**2))),
              'normalized_high_frequency_residual_max': float(np.max(np.abs(residual))),
              'normalized_grid_refinement_error': error,
              'endpoint_excess_ratio': float(endpoint_ratio), 'isolated_spikes': spikes,
              'values_smoothed': False}
    result['passed'] = (finite and increasing and result['duplicate_points']==0
        and result['normalized_high_frequency_residual_rms'] <= CURVE_CONTRACT['maximum_normalized_high_frequency_rms']
        and result['normalized_high_frequency_residual_max'] <= CURVE_CONTRACT['maximum_normalized_high_frequency_residual']
        and error <= CURVE_CONTRACT['maximum_normalized_grid_refinement_error']
        and endpoint_ratio <= CURVE_CONTRACT['maximum_endpoint_excess_ratio']
        and spikes <= CURVE_CONTRACT['maximum_isolated_spikes'])
    return result


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    data = json.loads(data_path.read_bytes())
    constants = data['constants']
    frequency = np.asarray(data['frequencies_cm1'])
    modes = np.asarray(data['modes_mass_weighted'])
    mass = np.asarray(data['masses_amu']) * constants['amu_kg']
    omega = 2*np.pi*constants['c_cm_s']*frequency
    factor = thermal_factor(frequency, args.temperature, constants)
    rng = np.random.default_rng(args.seed)
    zq = rng.standard_normal((args.samples, len(frequency)))
    zp = rng.standard_normal((args.samples, len(frequency)))
    means = np.concatenate([zq.mean(axis=0), zp.mean(axis=0)])
    variances = np.concatenate([zq.var(axis=0, ddof=1), zp.var(axis=0, ddof=1)])
    corr = np.corrcoef(np.concatenate([zq, zp], axis=1), rowvar=False)
    np.fill_diagonal(corr, 0)
    gates = {'max_abs_standardized_mean': 6/np.sqrt(args.samples),
             'max_abs_relative_variance_error': 6*np.sqrt(2/(args.samples-1)),
             'max_abs_offdiagonal_correlation': 6/np.sqrt(args.samples),
             'reconstruction_relative_error': 1e-10}
    # SI mass-weighted Q and P, transformed back through the actual modes.
    sigma_q = np.sqrt(constants['hbar_Js']*factor/(2*omega))
    sigma_p = np.sqrt(constants['hbar_Js']*omega*factor/2)
    q = zq[:128]*sigma_q
    p = zp[:128]*sigma_p
    flat = modes.reshape(len(frequency), -1)
    displacement = (q @ flat)/np.repeat(np.sqrt(mass), 3)
    velocity = (p @ flat)/np.repeat(np.sqrt(mass), 3)
    recovered_q = (displacement*np.repeat(np.sqrt(mass), 3)) @ flat.T
    recovered_p = (velocity*np.repeat(np.sqrt(mass), 3)) @ flat.T
    reconstruction_error = max(float(np.max(np.abs((recovered_q-q)/sigma_q))),
                               float(np.max(np.abs((recovered_p-p)/sigma_p))))
    summary = {
        'distribution': 'Exact harmonic Gaussian; not actual SHARC candidate-pool moments.',
        'samples': args.samples, 'seed': args.seed, 'temperature_K': args.temperature,
        'mode_count': len(frequency), 'atom_count': len(mass),
        'frequency_min_cm1': float(frequency.min()), 'frequency_max_cm1': float(frequency.max()),
        'maximum_standardized_mean_abs': float(np.max(np.abs(means))),
        'maximum_relative_variance_error_abs': float(np.max(np.abs(variances-1))),
        'maximum_offdiagonal_correlation_abs': float(np.max(np.abs(corr))),
        'reconstruction_max_error_in_mode_standard_deviations': reconstruction_error,
        'zero_temperature_zpe_eV': float(np.sum(.5*constants['hbar_Js']*omega)/constants['eV_J']),
        'finite_temperature_mean_harmonic_energy_eV': float(np.sum(.5*constants['hbar_Js']*omega*factor)/constants['eV_J']),
        'acceptance_limits': gates,
        'input_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scientific_scope': 'Sampler and coordinate-transformation checks only; no electronic-surface or dynamics accuracy claim.'
    }
    summary['passed'] = (summary['maximum_standardized_mean_abs'] <= gates['max_abs_standardized_mean']
        and summary['maximum_relative_variance_error_abs'] <= gates['max_abs_relative_variance_error']
        and summary['maximum_offdiagonal_correlation_abs'] <= gates['max_abs_offdiagonal_correlation']
        and reconstruction_error <= gates['reconstruction_relative_error'])
    np.savez_compressed(out/'teaching-draws.npz', zq=zq, zp=zp)
    write_json(out/'numerical-check.json', summary)
    curves = figures(data, out, Path(args.figures) if args.figures else out)
    write_json(out/'curve-quality.json', {'contract': CURVE_CONTRACT, 'series': curves,
                                        'passed': all(x['passed'] for x in curves.values())})
    assert summary['passed'], summary
    assert all(x['passed'] for x in curves.values()), curves
    print(json.dumps(summary, indent=2))


def figures(data, out, figure_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'hessian-wigner-v1'})
    blue, brown, gray = '#176078', '#976c43', '#899397'
    quality = {}

    def save(fig, name):
        fig.savefig(figure_dir/(name+'.svg'), bbox_inches='tight', metadata={'Date': None})
        fig.savefig(figure_dir/(name+'.png'), bbox_inches='tight', dpi=160)
        plt.close(fig)
        p = figure_dir/(name+'.svg')
        p.write_bytes(('\n'.join(x.rstrip() for x in p.read_text(encoding='utf-8').splitlines())+'\n').encode())

    # Three collective displacement patterns from the actual molecular Hessian.
    coords = np.asarray(data['coordinates_angstrom'])
    coords -= np.average(coords, axis=0, weights=data['masses_amu'])
    yaw, pitch = np.radians(18), np.radians(22)
    rotate = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    tilt = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    projection = rotate @ tilt
    points = coords @ projection
    colors = {'C': '#626b70', 'H': '#ecece6', 'N': blue, 'O': brown}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    for ax, k in zip(axes, [0, 35, 83]):
        c = data['constants']
        om = 2*np.pi*c['c_cm_s']*data['frequencies_cm1'][k]
        dx = np.asarray(data['modes_mass_weighted'][k])*np.sqrt(c['hbar_Js']/(2*om*np.asarray(data['masses_amu'])*c['amu_kg']))[:, None]/1e-10
        arrows = (3*dx) @ projection
        for i, j in data['bonds']:
            ax.plot(points[[i,j],0], points[[i,j],1], color='#b8b8b2', lw=1.2, zorder=1)
        ax.scatter(points[:,0], points[:,1], s=[55 if s!='H' else 23 for s in data['symbols']],
                   c=[colors[s] for s in data['symbols']], edgecolor='#4f5659', linewidth=.6, zorder=3)
        ax.quiver(points[:,0], points[:,1], arrows[:,0], arrows[:,1], angles='xy', scale_units='xy', scale=1,
                  color=blue, width=.005, zorder=4)
        visible = np.concatenate([points[:, :2], points[:, :2] + arrows[:, :2]])
        ax.set_aspect('equal')
        ax.set_xlim(float(visible[:, 0].min())-.7, float(visible[:, 0].max())+.7)
        ax.set_ylim(float(visible[:, 1].min())-.7, float(visible[:, 1].max())+.7)
        ax.set_title('Mode %d: %.1f cm$^{-1}$'%(k+1, data['frequencies_cm1'][k]))
        ax.set_xlabel('Projected position (\u00c5)'); ax.set_ylabel('Projected position (\u00c5)')
    save(fig, 'motor-mode-patterns')

    # At 50 K a soft mode broadens thermally while a bond-stretching mode retains ZPE.
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.8), layout='constrained')
    x = np.linspace(-4, 4, CURVE_CONTRACT['grid_points'])
    raw = {'dimensionless_coordinate': x}
    for ax, freq in zip(axes, [data['frequencies_cm1'][0], 1000.]):
        quantum_var = float(thermal_factor(freq, 50., data['constants']))/2
        classical_var = data['constants']['kB_J_K']*50/(data['constants']['hbar_Js']*2*np.pi*data['constants']['c_cm_s']*freq)
        for name, var, color, style in [('Quantum harmonic', quantum_var, blue, '-'), ('Classical', classical_var, brown, '--')]:
            exact = lambda v, variance=var: normal_pdf(v, variance)
            y = exact(x)
            key = ('%.5f_%s'%(freq, name)).replace(' ', '_')
            quality[key] = curve_quality(x, y, exact)
            raw[key] = y
            ax.plot(x, y, color=color, ls=style, lw=2, label=name)
        ax.set_title('50 K; %.1f cm$^{-1}$'%freq)
        ax.set_xlabel(r'$q=\sqrt{\omega/\hbar}\,Q$'); ax.set_ylabel('Probability density')
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, fontsize=9)
        ax.set_xlim(-4, 4)
    save(fig, 'wigner-marginals')
    np.savez_compressed(out/'analytic-marginal-curves.npz', **raw)

    # A Wigner point's classical energy is not a quantum projective measurement.
    e = np.linspace(0, 3, CURVE_CONTRACT['grid_points'])
    exact = lambda x: 2*np.exp(-2*x)
    y = exact(e)
    quality['ground_state_point_energy_density'] = curve_quality(e, y, exact)
    fig, ax = plt.subplots(figsize=(7.8, 3.9), layout='constrained')
    ax.plot(e, y, color=blue, lw=2, label=r'Wigner-point classical energy: $2e^{-2\epsilon}$')
    ax.axvline(.5, color=brown, ls='--', lw=1.8, label=r'Ensemble mean: $\epsilon=1/2$')
    ax.set_xlabel(r'$\epsilon=E_{\rm cl}/(\hbar\omega)$'); ax.set_ylabel('Probability density')
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.15)
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    ax.text(1.15, .6, 'Quantum energy measurement in the ground state:\nonly epsilon = 1/2. This curve is not that distribution.', fontsize=9)
    save(fig, 'wigner-point-energy')
    np.savez_compressed(out/'analytic-energy-curve.npz', epsilon=e, density=y)
    return quality


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(Path(__file__).with_name('motor-normal-modes.json')))
    parser.add_argument('--output', default='wigner-example')
    parser.add_argument('--figures')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--temperature', type=float, default=50.)
    args = parser.parse_args()
    if args.samples < 1000 or args.temperature < 0:
        parser.error('Use at least 1000 independent samples and a nonnegative temperature.')
    run(args)
