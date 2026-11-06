"""Four seeded random-feature surrogates: a reproducible DP-GEN teaching example.

Run from any directory: python committee_demo.py --output demo-output
Requires NumPy and Matplotlib. Dimensionless quantities; no MD or chemistry run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

CONFIG = {
    'seeds': [2026110511, 2026110512, 2026110513, 2026110514],
    'training_range': [-0.9, 0.9], 'training_count': 13,
    'display_range': [-2.2, 2.2], 'display_count': 601,
    'features': 24, 'center_range': [-2.2, 2.2], 'width_range': [0.35, 1.05],
    'ridge': 1e-6, 'force_row_weight': 0.5,
    'common_bias_slope': 0.4, 'default_sigma_thresholds': [0.03, 0.15],
}
GATE = {
    'normalized_hf_rms_max': 2e-4, 'normalized_hf_abs_max': 2e-3,
    'normalized_endpoint_error_max': 1e-10,
    'normalized_analytic_vertex_error_max': 1e-10,
    'normalized_integral_refinement_max': 2e-4,
    'unexpected_isolated_spikes_max': 0,
}


def reference(q):
    return (0.12*q**4-0.18*q**2+0.08*np.sin(3*q),
            -0.48*q**3+0.36*q-0.24*np.cos(3*q))


def basis(q, centers, widths):
    delta = (q[:, None]-centers[None, :])
    phi = np.exp(-0.5*(delta/widths)**2)
    energy = np.column_stack([phi, np.ones(len(q))])
    force = np.column_stack([phi*delta/widths**2, np.zeros(len(q))])
    return energy, force


def fit_models():
    q = np.linspace(*CONFIG['training_range'], CONFIG['training_count'])
    energy, force = reference(q)
    models = []
    for seed in CONFIG['seeds']:
        rng = np.random.default_rng(seed)
        centers = rng.uniform(*CONFIG['center_range'], CONFIG['features'])
        widths = rng.uniform(*CONFIG['width_range'], CONFIG['features'])
        a, d = basis(q, centers, widths)
        x = np.vstack([a, CONFIG['force_row_weight']*d])
        y = np.r_[energy, CONFIG['force_row_weight']*force]
        # Stable ridge least squares; the intercept receives the same penalty.
        x = np.vstack([x, np.sqrt(CONFIG['ridge'])*np.eye(x.shape[1])])
        y = np.r_[y, np.zeros(x.shape[1])]
        weights = np.linalg.lstsq(x, y, rcond=None)[0]
        models.append({'seed': seed, 'centers': centers.tolist(),
                       'widths': widths.tolist(), 'weights': weights.tolist()})
    return q, energy, force, models


def evaluate(q, models, bias=False):
    energies, forces = [], []
    slope = CONFIG['common_bias_slope'] if bias else 0.0
    for model in models:
        a, d = basis(q, np.array(model['centers']), np.array(model['widths']))
        energies.append(a@model['weights']+slope*q)
        forces.append(d@model['weights']-slope)
    energies, forces = np.array(energies), np.array(forces)
    truth_e, truth_f = reference(q)
    mean_f = forces.mean(axis=0)
    return {'energies': energies, 'forces': forces, 'mean_energy': energies.mean(axis=0),
            'mean_force': mean_f, 'sigma_force': forces.std(axis=0),
            'signed_force_error': mean_f-truth_f, 'absolute_force_error': abs(mean_f-truth_f),
            'reference_energy': truth_e, 'reference_force': truth_f}


def curve_audit(q, y, fine_q, fine_y, analytic_y=None):
    scale = max(float(np.ptp(y)), float(np.max(abs(y))), 1e-12)
    hf = (y[:-4]-4*y[1:-3]+6*y[2:-2]-4*y[3:-1]+y[4:])/16
    integral = np.trapezoid(y, q)
    fine_integral = np.trapezoid(fine_y, fine_q)
    refinement = abs(float(integral-fine_integral))/(scale*(q[-1]-q[0]))
    # Compare every coarse vertex with a direct evaluation on the doubled grid.
    parent = fine_y[::2] if analytic_y is None else analytic_y
    residual = y-parent
    spikes = sum(abs(residual[1:-1]) > 1e-10*scale)
    result = {'all_finite': bool(np.isfinite(y).all()),
              'strictly_increasing_x': bool(np.all(np.diff(q)>0)),
              'duplicate_x': len(q)-len(np.unique(q)),
              'normalized_hf_rms': float(np.sqrt(np.mean(hf**2))/scale),
              'normalized_hf_abs_max': float(np.max(abs(hf))/scale),
              'normalized_endpoint_error': float(max(abs(residual[[0, -1]]))/scale),
              'normalized_analytic_vertex_error': float(np.max(abs(residual))/scale),
              'normalized_integral_refinement': refinement,
              'unexpected_isolated_spikes': int(spikes)}
    result['passed'] = (result['all_finite'] and result['strictly_increasing_x'] and
        result['duplicate_x']==0 and
        all(result[name] <= GATE[threshold] for name, threshold in [
            ('normalized_hf_rms', 'normalized_hf_rms_max'),
            ('normalized_hf_abs_max', 'normalized_hf_abs_max'),
            ('normalized_endpoint_error', 'normalized_endpoint_error_max'),
            ('normalized_analytic_vertex_error', 'normalized_analytic_vertex_error_max'),
            ('normalized_integral_refinement', 'normalized_integral_refinement_max'),
            ('unexpected_isolated_spikes', 'unexpected_isolated_spikes_max')]))
    return result


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    train_q, train_e, train_f, models = fit_models()
    q = np.linspace(*CONFIG['display_range'], CONFIG['display_count'])
    fine_q = np.linspace(*CONFIG['display_range'], 2*CONFIG['display_count']-1)
    base, biased = evaluate(q, models), evaluate(q, models, True)
    fine_base, fine_biased = evaluate(fine_q, models), evaluate(fine_q, models, True)
    series = {}
    for label, result, refined in [('base', base, fine_base), ('common_bias', biased, fine_biased)]:
        for kind in ['energies', 'forces']:
            for i in range(4):
                series[f'{label}_{kind}_{i}'] = curve_audit(q, result[kind][i], fine_q, refined[kind][i])
        for kind in ['mean_energy', 'mean_force', 'reference_energy', 'reference_force',
                     'sigma_force', 'signed_force_error']:
            series[f'{label}_{kind}'] = curve_audit(q, result[kind], fine_q, refined[kind])
        absolute = curve_audit(q, result['absolute_force_error'], fine_q, refined['absolute_force_error'])
        # The absolute-value cusp is an exact mathematical feature. Do not smooth it.
        parent = series[f'{label}_signed_force_error']
        absolute['method_specific_exception'] = 'Derivative cusps at signed-error zeros: audit HF and spikes on signed parent; displayed abs values remain unchanged.'
        absolute['passed'] = bool(absolute['all_finite'] and absolute['strictly_increasing_x'] and
                              absolute['duplicate_x']==0 and parent['passed'] and
                              absolute['normalized_analytic_vertex_error']<1e-10 and
                              absolute['normalized_integral_refinement']<2e-4)
        series[f'{label}_absolute_force_error'] = absolute
    lo, hi = CONFIG['default_sigma_thresholds']
    sigma = base['sigma_force']
    counts = {'below': int(np.sum(sigma<lo)), 'candidate': int(np.sum((sigma>=lo)&(sigma<hi))),
              'above': int(np.sum(sigma>=hi))}
    invariant = float(np.max(abs(base['sigma_force']-biased['sigma_force'])))
    train = evaluate(train_q, models)
    inside = abs(q) <= 0.9+1e-12
    summary = {'config': CONFIG, 'scope': 'Synthetic random-feature teaching example; no MD, HamGNN, CASSCF, or active-learning convergence result.',
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'default_class_counts': counts, 'sigma_bias_invariance_max_abs': invariant,
               'training_max_member_force_error': float(np.max(abs(train['forces']-train_f))),
               'inside_mean_force_rmse': float(np.sqrt(np.mean(base['signed_force_error'][inside]**2))),
               'outside_mean_force_rmse': float(np.sqrt(np.mean(base['signed_force_error'][~inside]**2))),
               'global_mean_force_rmse': float(np.sqrt(np.mean(base['signed_force_error']**2))),
               'biased_global_mean_force_rmse': float(np.sqrt(np.mean(biased['signed_force_error']**2))),
               'sigma_force_range': [float(sigma.min()), float(sigma.max())],
               'point_checks': []}
    for point in [0.0, 0.6, 1.2, 1.8]:
        a, b = evaluate(np.array([point]), models), evaluate(np.array([point]), models, True)
        summary['point_checks'].append({'q': point, 'reference_force': a['reference_force'][0],
            'mean_force': a['mean_force'][0], 'sigma_force': a['sigma_force'][0],
            'absolute_force_error': a['absolute_force_error'][0],
            'biased_mean_force': b['mean_force'][0], 'biased_absolute_force_error': b['absolute_force_error'][0]})
    payload = {'schema': 'dpgen-committee-teaching-v1', 'config': CONFIG, 'models': models,
               'training': {'q': train_q.tolist(), 'energy': train_e.tolist(), 'force': train_f.tolist()},
               'reference': {'energy': '0.12*q^4-0.18*q^2+0.08*sin(3*q)', 'force': '-dE/dq'},
               'scope': summary['scope']}
    dump(output/'committee-models.json', payload)
    summary['models_sha256'] = hashlib.sha256((output/'committee-models.json').read_bytes()).hexdigest()
    summary['passed'] = invariant<1e-12 and all(v['passed'] for v in series.values())
    dump(output/'numerical-check.json', summary)
    dump(output/'curve-quality.json', {'passed': all(v['passed'] for v in series.values()),
         'thresholds': GATE, 'no_smoothing': True, 'series': series})
    make_figure(output, q, base, biased, train_q, train_f)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['point_checks','config']}))
    if not summary['passed']: raise RuntimeError('Teaching-example verification failed; retain outputs for diagnosis')


def make_figure(output, q, base, biased, train_q, train_f):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.hashsalt': 'dpgen-committee-teaching-20261105'})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax in axes:
        ax.axvspan(-0.9, 0.9, color='#e5edf1', alpha=.8, zorder=0)
        ax.set_xlabel('Dimensionless coordinate q')
        ax.set_xlim(q[0], q[-1]); ax.spines[['top','right']].set_visible(False)
    for row in base['forces']: axes[0].plot(q, row, color='#8d9fac', lw=.8, alpha=.7)
    axes[0].plot(q, base['reference_force'], color='#292d30', lw=1.5, label='Reference')
    axes[0].plot(q, base['mean_force'], color='#315f85', lw=1.8, label='Committee mean')
    axes[0].scatter(train_q, train_f, s=12, color='#292d30', zorder=4, label='Shared training data')
    axes[0].set_ylabel('Dimensionless force'); axes[0].set_title('Four seeds, one training set')
    axes[1].plot(q, base['sigma_force'], color='#315f85', lw=1.8, label='Committee spread')
    axes[1].plot(q, base['absolute_force_error'], color='#925e35', lw=1.5, label='Actual mean-force error')
    axes[1].plot(q, biased['absolute_force_error'], color='#925e35', lw=1.2, ls='--', label='Error after common force bias -0.4')
    axes[1].set_ylabel('Dimensionless force'); axes[1].set_title('Shared bias leaves spread unchanged')
    axes[1].set_ylim(bottom=0)
    for ax in axes: ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.22))
    fig.subplots_adjust(left=.08, right=.985, bottom=.30, top=.88, wspace=.29)
    fig.savefig(output/'committee-example.png', dpi=170)
    fig.savefig(output/'committee-example.svg', metadata={'Date': None})
    plt.close(fig)
    svg = output/'committee-example.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    run(parser.parse_args().output)
