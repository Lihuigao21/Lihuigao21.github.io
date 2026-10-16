"""Finite-time TCL2/NZ2 benchmark; all units hbar=omega_ref=1.

Independent finite-Hilbert-space exact dynamics and analytic dephasing controls.
No density clipping, renormalization, Markov/secular or thermal corrections.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm
import scipy

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1., -1.]).astype(complex)
I2 = np.eye(2, dtype=complex)
INITIAL = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]]], complex)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, np.generic):
        return serial(x.item())
    return x


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(serial(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for trial in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if trial == 7:
                raise
            time.sleep(.05 * (trial + 1))


def distance(a, b):
    diff = a - b
    # No positivity assumption: trace norm is computed on the Hermitian difference.
    return .5 * np.abs(np.linalg.eigvalsh((diff + diff.conj().swapaxes(-1, -2)) / 2)).sum(-1)


def density_metrics(r):
    errtrace = float(np.max(abs(np.trace(r, axis1=-2, axis2=-1) - 1)))
    errherm = float(np.max(abs(r - r.conj().swapaxes(-1, -2))))
    ev = np.linalg.eigvalsh((r + r.conj().swapaxes(-1, -2)) / 2)
    return dict(finite=bool(np.isfinite(r).all()), trace_error=errtrace,
                hermiticity_error=errherm, min_eigenvalue=float(ev.min()),
                max_eigenvalue=float(ev.max()))


def bath(model, n):
    beta = CFG['beta']
    if model in ('gaussian_dephasing', 'spin_dephasing'):
        # These exact ordered correlations coincide for every beta at this normalization.
        c = np.array([(1 + np.tanh(beta / 2)) / 2, (1 - np.tanh(beta / 2)) / 2])
        return dict(H=.5 * SZ, A=SZ, c=c, d=c[::-1].copy(), w=np.array([1., -1.]),
                    delta=1., variance=1., omitted_weight=0., n=0,
                    correlation_definition='C(t)=cos(t)-i*tanh(beta/2)*sin(t)')
    kappa = dict(quartic_weak=.01, quartic_strong=.5).get(model, 0.)
    aa = np.diag(np.sqrt(np.arange(1, n)), 1)
    q = (aa + aa.T) / np.sqrt(2)
    hb0 = np.diag(np.arange(n) + .5) + kappa * (q @ q @ q @ q)
    eb, vb = eigh(hb0, driver='evr')
    p = np.exp(-beta * (eb - eb[0]))
    p /= p.sum()
    b = vb.T @ q @ vb
    mean = np.sum(p * np.diag(b))
    b -= mean * np.eye(n)
    var = np.sum(p[:, None] * abs(b)**2)
    b /= np.sqrt(var)
    weights = p[:, None] * abs(b)**2
    pairs = [(float(weights[i, j] + (weights[j, i] if i != j else 0)), i, j)
             for i in range(n) for j in range(i, n)]
    pairs.sort()
    omitted = 0.
    kept = []
    for val, i, j in pairs:
        if omitted + val <= CFG['pole_omitted_weight']:
            omitted += val
        else:
            kept.append((i, j))
            if i != j:
                kept.append((j, i))
    ii, jj = np.array(kept, int).T
    c, d, w = weights[ii, jj], weights[jj, ii], eb[jj] - eb[ii]
    delta = 1.6 if model == 'harmonic_detuned' else (eb[1] - eb[0] if kappa else 1.)
    return dict(H=.5 * delta * SZ, A=SX, c=c, d=d, w=w, delta=float(delta),
                variance=float(var), omitted_weight=float(omitted), n=n,
                eb=eb, b=b, p=p, q=q, vb=vb, hb0=hb0, kappa=kappa,
                tail_thermal_weight=float(p[-4:].sum()),
                bath_eigen_residual=float(np.max(abs(hb0 @ vb - vb * eb))),
                bath_orthogonality=float(np.max(abs(vb.T @ vb - np.eye(n)))))


def phi(z, t):
    out = np.empty_like(z, dtype=complex)
    small = abs(z * t) < 1e-5
    u = -z[small] * t
    out[small] = t * (1 + u/2 + u*u/6 + u**3/24 + u**4/120 + u**5/720)
    np.divide(-np.expm1(-z*t), z, out=out, where=~small)
    return out


def approximate(b, g, times, settings, method):
    H, A = b['H'], b['A']
    diff = np.diag(H).real[:, None] - np.diag(H).real[None, :]
    l0 = -1j * diff
    c, d, w = b['c'], b['d'], b['w']
    started = time.perf_counter()
    if method == 'TCL2':
        z = 1j * (w[:, None, None] + diff[None])
        def rhs(t, y):
            rho = y.reshape(3, 2, 2)
            lam = A * np.sum(c[:, None, None] * phi(z, t), axis=0)
            u, v = lam @ rho, rho @ lam.conj().T
            return (l0 * rho - g*g * (A@u - u@A + v@A - A@v)).ravel()
        initial = INITIAL.ravel()
    elif method == 'NZ2':
        rate = l0[None, None] - 1j * w[:, None, None, None]
        cc, dd = c[:, None, None, None], d[:, None, None, None]
        def rhs(t, y):
            arr = y.reshape(1 + len(c), 3, 2, 2)
            rho, aux = arr[0], arr[1:]
            summed = aux.sum(axis=0)
            result = np.empty_like(arr)
            result[0] = l0 * rho - g*g * (A@summed - summed@A)
            result[1:] = rate * aux + cc * (A@rho)[None] - dd * (rho@A)[None]
            return result.ravel()
        initial = np.zeros((1 + len(c), 3, 2, 2), complex)
        initial[0] = INITIAL
        initial = initial.ravel()
    else:
        raise ValueError(method)
    sol = solve_ivp(rhs, (0, float(times[-1])), initial, t_eval=times, method='DOP853',
                    rtol=settings['rtol'], atol=settings['atol'], max_step=settings['max_step'])
    if not sol.success or len(sol.t) != len(times) or not np.isfinite(sol.y).all():
        raise RuntimeError(f'{method}: {sol.message}')
    # Root is always the first 12 complex entries; auxiliaries are not saved.
    rho = sol.y[:12].T.reshape(-1, 3, 2, 2).copy()
    info = dict(nfev=int(sol.nfev), seconds=time.perf_counter()-started,
                npoles=len(c), status=sol.message, integrator='DOP853',
                output='all requested t_eval points; adaptive internal integration', settings=settings)
    return rho, info


def exact(model, b, g, times):
    started = time.perf_counter()
    if model in ('gaussian_dephasing', 'spin_dephasing'):
        if model == 'gaussian_dephasing':
            f = np.exp(-4 * g*g * (1-np.cos(times)))
        else:
            r2 = 1 + 4*g*g
            f = (1 + 4*g*g*np.cos(np.sqrt(r2)*times)) / r2
        rho = np.broadcast_to(INITIAL, (len(times), 3, 2, 2)).copy()
        rho[:, :, 0, 1] *= (f * np.exp(-1j*times))[:, None]
        rho[:, :, 1, 0] = rho[:, :, 0, 1].conj()
        return rho, dict(seconds=time.perf_counter()-started, reference='analytic full reduced map')
    n = b['n']
    h = np.kron(b['H'], np.eye(n)) + np.kron(I2, np.diag(b['eb'])) + g*np.kron(b['A'], b['b'])
    e, v = eigh(h, driver='evr')
    v3 = v.reshape(2, n, 2*n)
    partial = np.einsum('akm,bkn->abmn', v3, v3.conj(), optimize=True)
    amps = []
    for rs in INITIAL:
        r0 = np.kron(rs, np.diag(b['p']))
        re = v.conj().T @ r0 @ v
        amps.append((partial * re[None, None]).reshape(4, -1))
    amps = np.array(amps).reshape(12, -1)
    freqs = (e[:, None] - e[None, :]).ravel()
    rho = np.empty((len(times), 3, 2, 2), complex)
    for first in range(0, len(times), 128):
        tt = times[first:first+128]
        phase = np.exp(-1j * freqs[:, None] * tt[None])
        rho[first:first+len(tt)] = (amps @ phase).T.reshape(-1, 3, 2, 2)
    info = dict(seconds=time.perf_counter()-started, reference='full Hamiltonian spectral propagation',
                dimension=2*n, eigen_residual=float(np.max(abs(h@v-v*e))),
                orthogonality=float(np.max(abs(v.conj().T@v-np.eye(2*n)))),
                initial_error=float(np.max(abs(rho[0]-INITIAL))))
    return rho, info


def cell(model, g, role, identity):
    settings = CFG[role]
    n = settings['basis']
    times = np.linspace(0, CFG['tf'], round(CFG['tf']/settings['output_dt'])+1)
    out = ROOT / 'runs' / model / f'g-{g:g}' / role / 'attempt-001'
    out.mkdir(parents=True, exist_ok=True)
    meta_path, data_path = out/'run.json', out/'densities.npz'
    expected = dict(identity=identity, model=model, g=g, role=role, settings=settings)
    if meta_path.exists() and data_path.exists():
        m = json.loads(meta_path.read_text(encoding='utf-8'))
        if all(m.get(k) == v for k, v in expected.items()) and m.get('status') == 'completed':
            with np.load(data_path) as data:
                valid = np.array_equal(data['time'], times) and all(data[k].shape == (len(times),3,2,2)
                         and np.isfinite(data[k]).all() for k in ('exact', 'TCL2', 'NZ2'))
            if valid and sha(data_path) == m['data_sha256']:
                return m
        raise RuntimeError('Existing cell has incompatible or invalid identity: ' + str(out))
    b = bath(model, n)
    metadata = {k: b[k] for k in ('delta', 'variance', 'omitted_weight', 'n')}
    metadata.update({k: b[k] for k in ('tail_thermal_weight','bath_eigen_residual','bath_orthogonality') if k in b})
    arrays = dict(time=times)
    infos = {}
    arrays['exact'], infos['exact'] = exact(model,b,g,times)
    for method in ('TCL2','NZ2'):
        arrays[method], infos[method] = approximate(b,g,times,settings,method)
    np.savez_compressed(data_path, **arrays)
    m = dict(**expected, status='completed', bath=metadata, methods=infos,
             metrics={method:density_metrics(arrays[method]) for method in ('exact','TCL2','NZ2')},
             data_sha256=sha(data_path), numerical_completed=True)
    save_json(meta_path,m)
    return m


def audit():
    results = {}
    # Independent full-bath double commutator, not the spectral formula twice.
    hs = .7*SZ
    hb = .35*SZ
    a = .4*SX + .6*SZ
    b = SX + .13*SZ
    e,pv = eigh(hb)
    p = np.exp(-2*(e-e.min())); p /= p.sum()
    rb = pv@np.diag(p)@pv.conj().T
    b -= np.trace(rb@b)*I2
    bb = pv.conj().T@b@pv
    v = np.kron(a,b)
    rho = np.array([[.63,.17+.09j],[.17-.09j,.37]])
    errors = []
    for tau in (0., .17, .83, 2.1):
        u = expm(-1j*hs*tau); ub = expm(-1j*hb*tau)
        rp = np.kron(rho,rb)
        inner = v@rp-rp@v
        uu = np.kron(u,ub)
        inner = uu@inner@uu.conj().T
        full = -(v@inner-inner@v).reshape(2,2,2,2).trace(axis1=1,axis2=3)
        c = np.sum(p[:,None]*abs(bb)**2*np.exp(-1j*(e[None,:]-e[:,None])*tau))
        q = c*u@a@rho@u.conj().T - c.conjugate()*u@rho@a@u.conj().T
        spec = -(a@q-q@a)
        errors.append(np.max(abs(full-spec)))
    results['full_bath_NZ2_kernel_error'] = float(max(errors))
    # Independent finite 4x4 spin-bath exact control, at a nontrivial coupling.
    times=np.linspace(0,30,1201); g=.4
    bdict=bath('spin_dephasing',2)
    analytic,_=exact('spin_dephasing',bdict,g,times)
    h=np.kron(.5*SZ,I2)+np.kron(I2,.5*SZ)+g*np.kron(SZ,SX)
    vals,vec=eigh(h); rb=np.diag(np.exp(-np.diag(.5*SZ).real*2)); rb/=np.trace(rb)
    vals_r=[]
    for t in times[::40]:
        ut=(vec*np.exp(-1j*vals*t))@vec.conj().T
        rt=ut@np.kron(INITIAL[2],rb)@ut.conj().T
        vals_r.append(rt.reshape(2,2,2,2).trace(axis1=1,axis2=3))
    results['spin_analytic_vs_full_H_error']=float(np.max(abs(np.array(vals_r)-analytic[::40,2])))
    strict=dict(CFG['ode_validation'])
    gauss=bath('gaussian_dephasing',40)
    gt,_=approximate(gauss,g,times,strict,'TCL2')
    ge,_=exact('gaussian_dephasing',gauss,g,times)
    sn,_=approximate(bdict,g,times,strict,'NZ2')
    results['gaussian_TCL2_analytic_error']=float(np.max(distance(gt,ge)))
    results['spin_NZ2_analytic_error']=float(np.max(distance(sn,analytic)))
    # Zero coupling and fixed-time fourth-order difference.
    tr=np.linspace(0,3,121); br=bath('harmonic_resonant',40)
    zero,_=approximate(br,0,tr,strict,'NZ2'); ze,_=exact('harmonic_resonant',br,0,tr)
    results['zero_coupling_error']=float(np.max(distance(zero,ze)))
    differences=[]
    for smallg in (.01,.02,.04,.08):
        rt,_=approximate(br,smallg,tr,strict,'TCL2')
        rn,_=approximate(br,smallg,tr,strict,'NZ2')
        differences.append(float(np.max(distance(rt,rn))))
    slope=float(np.polyfit(np.log([.01,.02,.04,.08]),np.log(differences),1)[0])
    results['fixed_t_weak_coupling_difference']=dict(g=[.01,.02,.04,.08],max_trace_distance=differences,slope=slope)
    results['pass']=bool(max(errors)<1e-12 and results['spin_analytic_vs_full_H_error']<2e-12
                         and results['gaussian_TCL2_analytic_error']<2e-8
                         and results['spin_NZ2_analytic_error']<2e-8
                         and results['zero_coupling_error']<2e-9 and 3.9<slope<4.1)
    save_json(ROOT/'records'/'implementation_audit.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
    if not results['pass']:
        raise RuntimeError('Implementation audit failed')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command', choices=['audit','smoke','run']); args=ap.parse_args()
    if args.command=='audit':
        audit(); return
    config_hash=sha(ROOT/'config.json'); source_hash=sha(__file__)
    identity=hashlib.sha256((source_hash+config_hash).encode()).hexdigest()
    save_json(ROOT/'records'/f'launch-{args.command}-{identity[:12]}.json',dict(
        command=sys.argv,identity=identity,source_sha256=source_hash,config_sha256=config_hash,
        python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        seed='not applicable: deterministic dynamics',pid=os.getpid(),started=time.time()))
    work = [(m,g,r) for m in CFG['models'] for g in CFG['couplings'] for r in
            (['production','ode_validation'] if 'dephasing' in m else ['production','ode_validation','basis_validation'])]
    if args.command=='smoke':
        work=[('quartic_strong',.6,'production'),('quartic_strong',.6,'ode_validation'),
              ('quartic_strong',.6,'basis_validation')]
    started=time.perf_counter()
    for index,(model,g,role) in enumerate(work,1):
        m=cell(model,g,role,identity)
        progress=dict(status='running',pid=os.getpid(),index=index,total=len(work),model=model,g=g,role=role,
                      identity=identity,seconds=time.perf_counter()-started,last_metrics=m['metrics'],last_update=time.time())
        save_json(ROOT/'records'/f'progress-{args.command}.json',progress)
        print(f'{index}/{len(work)} {model} g={g:g} {role} seconds={progress["seconds"]:.1f} '
              f'TCLmin={m["metrics"]["TCL2"]["min_eigenvalue"]:.4g} NZmin={m["metrics"]["NZ2"]["min_eigenvalue"]:.4g}',flush=True)
    progress.update(status='completed',finished=time.time())
    save_json(ROOT/'records'/f'progress-{args.command}.json',progress)
    if sha(__file__)!=source_hash or sha(ROOT/'config.json')!=config_hash:
        raise RuntimeError('Frozen input bytes changed during computation')


if __name__=='__main__':
    main()
