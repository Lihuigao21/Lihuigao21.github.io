"""Verify and redraw the stored one-dimensional ML-CMD benchmark.

No ToyModel installation is required. The default path loads frozen networks
and stored data; it performs no PIMC, fitting, trajectory propagation, or DVR
calculation. Optional --refit writes separate new networks only, and does not
replace the stored-model plots or pretend to regenerate their correlations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import CubicSpline

from mlcmd_learning import ForceMatchedPotential1D


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_checksums(directory):
    manifest = read_json(directory / 'checksums.json')
    for filename, expected in manifest['sha256'].items():
        if Path(filename).name != filename:
            raise ValueError('checksum manifest requires plain filenames')
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'checksum mismatch: {filename}')
    return len(manifest['sha256'])


def load_table(directory, name, kind):
    table = np.genfromtxt(directory / f'{name}_{kind}.csv', delimiter=',', names=True)
    if table.ndim != 1 or not np.all(np.isfinite(table.view(float))):
        raise ValueError(f'{name}_{kind}: malformed or nonfinite data')
    coordinate = table['time'] if 'time' in table.dtype.names else table['q']
    if np.any(np.diff(coordinate) <= 0):
        raise ValueError(f'{name}_{kind}: coordinates are not strictly increasing')
    return table


def bare_callbacks(coefficients):
    derivative = np.polynomial.polynomial.polyder(coefficients)
    value = lambda q: np.polynomial.polynomial.polyval(q, coefficients)
    force = lambda q: -np.polynomial.polynomial.polyval(q, derivative)
    return value, force


def load_models(directory, name, spec):
    value, force = bare_callbacks(spec['coefficients'])
    models = {}
    for mode in ('direct', 'delta'):
        kwargs = {} if mode == 'direct' else {'base_id': name, 'base_energy': value, 'base_force': force}
        models[mode] = ForceMatchedPotential1D.from_dict(read_json(directory / f'{name}_{mode}.json'), **kwargs)
    return models


def inspect_case(directory, name, spec):
    tables = {kind: load_table(directory, name, kind) for kind in
              ('training', 'reference', 'validation', 'curves', 'cmd_chains')}
    train, reference, validation, curves, chains = (tables[k] for k in
        ('training', 'reference', 'validation', 'curves', 'cmd_chains'))
    if not np.array_equal(train['q'], reference['q']):
        raise ValueError('reference and training table coordinates differ')
    if not np.array_equal(chains['time'], curves['time']):
        raise ValueError('chain curves and main curves have different times')
    dt, tmax = spec['dynamics']['dt'], spec['dynamics']['tmax']
    expected_times = np.arange(round(tmax / dt) + 1) * dt
    if len(curves) != len(expected_times) or not np.allclose(curves['time'], expected_times, atol=1e-14, rtol=0):
        raise ValueError('stored correlations must retain every integration time')
    unseen = ~np.any(np.isclose(validation['q'][:, None], train['q'][None, :], atol=1e-12, rtol=0), axis=1)
    if not np.array_equal(unseen, validation['unseen_training_coordinate'].astype(bool)):
        raise ValueError('validation coordinate mask is inconsistent')
    if not np.any(unseen):
        raise ValueError('missing independent interpolation coordinates')
    if any(np.any(tables[k]['standard_error'] < 0) for k in ('training','reference','validation')):
        raise ValueError('standard errors cannot be negative')
    models = load_models(directory, name, spec)
    metrics = {'rows': {kind: len(table) for kind, table in tables.items()},
               'unseen_validation_coordinates': int(unseen.sum()),
               'max_abs_CMD_exact': float(np.max(np.abs(curves['cmd'] - curves['exact']))),
               'models': {}}
    expected = {check['name']: check['value'] for check in spec['expected_checks']}
    np.testing.assert_allclose(metrics['max_abs_CMD_exact'], spec['expected_curve_differences']['cmd_exact'], atol=1e-12, rtol=1e-10)
    q = np.linspace(-3., 3., 301)
    h = 1e-5
    for mode, model in models.items():
        errors = {label: model.force(table['q']) - table['mean_force']
                  for label, table in [('reference',reference),('validation',validation)]}
        finite_difference = -(model.energy(q+h) - model.energy(q-h))/(2*h)
        conservative_error = float(np.max(np.abs(model.force(q)-finite_difference)))
        if conservative_error > 1e-6:
            raise ValueError(f'{mode}: conservative-force finite-difference check failed')
        m = {'max_abs_force_reference': float(np.max(np.abs(errors['reference']))),
             'max_abs_force_validation': float(np.max(np.abs(errors['validation']))),
             'max_abs_force_unseen': float(np.max(np.abs(errors['validation'][unseen]))),
             'max_abs_correlation_CMD': float(np.max(np.abs(curves[mode]-curves['cmd']))),
             'max_abs_correlation_exact': float(np.max(np.abs(curves[mode]-curves['exact']))),
             'force_energy_finite_difference_max_error': conservative_error}
        for metric, key in [('max_abs_force_reference',f'{mode}_independent_force'),
                            ('max_abs_force_validation',f'{mode}_refined_independent_force'),
                            ('max_abs_force_unseen',f'{mode}_unseen_centroid_force'),
                            ('max_abs_correlation_CMD',f'{mode}_CMD_correlation')]:
            np.testing.assert_allclose(m[metric], expected[key], atol=1e-12, rtol=1e-10,
                                       err_msg=f'{name}/{key}: public data disagree with saved benchmark')
        np.testing.assert_allclose(m['max_abs_correlation_exact'], spec['expected_curve_differences'][mode+'_exact'], atol=1e-12, rtol=1e-10)
        metrics['models'][mode] = m
    return tables, models, metrics


def draw_case(output, name, spec, tables, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reference, validation, curves = (tables[k] for k in ('reference','validation','curves'))
    value, force = bare_callbacks(spec['coefficients'])
    force_spline = CubicSpline(reference['q'], reference['mean_force'], extrapolate=False)
    primitive = force_spline.antiderivative()
    reference_energy = lambda q: -primitive(q)+primitive(0.)
    q = np.linspace(-3.,3.,601)
    colors = {'direct':'#b65327','delta':'#3975a8'}
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'legend.fontsize':8.5,
                         'xtick.labelsize':9,'ytick.labelsize':9})
    fig, ax = plt.subplots(figsize=(7,4.5), constrained_layout=True)
    ax.plot(q,value(q)-value(0.),color='.65',label='Bare PES')
    ax.plot(q,reference_energy(q),color='.15',label='Reference centroid PMF')
    for mode, model in models.items():
        energy = model.energy(q)-model.energy(0.)
        style = {'color':colors[mode],'label':mode.capitalize(),'linestyle':'--' if mode=='direct' else ':'}
        ax.plot(q,energy,**style)
    ax.set(xlabel='Centroid q (model units)',ylabel='Potential (model units)',
           title=f'{name.capitalize()}: all potentials set to zero at q = 0')
    ax.legend()
    fig.savefig(output/f'{name}_potentials.png',dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(7,6.5),constrained_layout=True)
    axes[0].plot(q,force_spline(q)-force(q),color='.15',label='Independent reference')
    unseen = validation['unseen_training_coordinate'].astype(bool)
    for mode, model in models.items():
        style={'color':colors[mode],'label':mode.capitalize(),'linestyle':'--' if mode=='direct' else ':'}
        axes[0].plot(q,model.force(q)-force(q),**style)
        residual = model.force(validation['q'][unseen])-validation['mean_force'][unseen]
        axes[1].errorbar(validation['q'][unseen],residual,yerr=validation['standard_error'][unseen],
                         fmt='o' if mode=='direct' else 's',ms=3,capsize=2,
                         color=colors[mode],label=mode.capitalize()+'; reference +/- 1 SE')
    axes[0].set(xlabel='Centroid q (model units)',ylabel='Centroid force minus bare force',
                title=f'{name.capitalize()}: learned force correction')
    axes[1].axhline(0,color='.5',lw=.8)
    axes[1].set(xlabel='Centroid q (model units)',ylabel='Predicted minus reference force',
                title=f'{int(unseen.sum())} independent coordinates absent from training')
    for ax in axes:
        ax.legend()
    fig.savefig(output/f'{name}_force_validation.png',dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(3,1,figsize=(7,8),constrained_layout=True)
    t = curves['time']
    chain_values = np.column_stack([tables['cmd_chains'][key] for key in tables['cmd_chains'].dtype.names if key != 'time'])
    axes[0].fill_between(t,chain_values.min(axis=1),chain_values.max(axis=1),color='.7',alpha=.25,
                         label='Independent-chain CMD range (not CI)')
    axes[0].plot(t,curves['exact'],color='black',label='Quantum Kubo reference')
    axes[0].plot(t,curves['cmd'],color='#5d8065',label='Reference CMD')
    for mode in ('direct','delta'):
        style={'color':colors[mode],'label':mode.capitalize()+' ML-CMD','linestyle':'--' if mode=='direct' else ':'}
        axes[0].plot(t,curves[mode],**style)
        axes[1].plot(t,curves[mode]-curves['cmd'],**style)
    axes[0].set(xlabel='Time (model units)',ylabel='Position correlation')
    axes[1].set(xlabel='Time (model units)',ylabel='ML-CMD minus reference CMD')
    axes[2].plot(t,curves['cmd']-curves['exact'],color='#5d8065',label='Reference CMD minus quantum')
    axes[2].set(xlabel='Time (model units)',ylabel='CMD minus quantum Kubo')
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f'{name.capitalize()}: all stored integration times')
    fig.savefig(output/f'{name}_correlations.png',dpi=200)
    plt.close(fig)


def optional_refit(output, name, spec, training):
    value, force = bare_callbacks(spec['coefficients'])
    options = {key:spec['ml'][key] for key in ('hidden_units','seed','max_nfev')}
    diagnostics = {}
    for mode in ('direct','delta'):
        base = {} if mode=='direct' else {'base_energy':value,'base_force':force,'base_id':name}
        model, diagnostic = ForceMatchedPotential1D.fit(training['q'],training['mean_force'],mode=mode,
            weights=1/(1+training['q']**2)**2,**options,**base)
        (output/f'{name}_{mode}_refitted.json').write_text(json.dumps(model.to_dict(),indent=2)+'\n',encoding='utf-8')
        diagnostics[mode]=diagnostic
    return diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--output',type=Path,required=True,help='directory for regenerated plots and verification JSON')
    parser.add_argument('--model',choices=('both','harmonic','quartic'),default='both')
    parser.add_argument('--refit',action='store_true',help='optionally fit separate new models; does not alter stored results')
    args=parser.parse_args()
    directory=Path(__file__).resolve().parent
    verified=verify_checksums(directory)
    metadata=read_json(directory/'metadata.json')
    args.output.mkdir(parents=True,exist_ok=True)
    report={'scope':'stored-data and stored-model verification; no fresh sampling or dynamics',
            'checksum_verified_files':verified,'numpy':np.__version__,'scipy':scipy.__version__,'models':{}}
    selected=metadata['models'] if args.model=='both' else [args.model]
    for name in selected:
        spec=metadata['models'][name]
        tables,models,metrics=inspect_case(directory,name,spec)
        draw_case(args.output,name,spec,tables,models)
        if args.refit:
            metrics['optional_refit']=optional_refit(args.output,name,spec,tables['training'])
            metrics['optional_refit_scope']='new fitted models only; stored plots/correlations still use the supplied networks'
        report['models'][name]=metrics
    (args.output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'verified':True,'models':list(selected),'refit_executed':args.refit,'files_checked':verified}))


if __name__=='__main__':
    main()
