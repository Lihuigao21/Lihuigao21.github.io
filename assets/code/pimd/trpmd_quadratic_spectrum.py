"""Controlled harmonic nonlinear spectrum. No imported ToyModel dynamics claimed.

Independent layers: energy-basis Kubo sum, modal analytic Laplace transform,
numerical Gaussian-moment ODE, and actual sampled stationary OU trajectories.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json, math, sys, time, hashlib, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import expm
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('trpmd-q2-output'))
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
CFG = {'model': 'H=(p^2+q^2)/2; A=q^2; connected Kubo correlation',
 'units': 'hbar=m=omega0=1',
 'beta': 8.0,
 'beads_production': 64,
 'beads_validation': 128,
 'beads_diagnostics': [16, 32],
 'lambda_production': 1.0,
 'lambda_scan': [0.0, 0.25, 0.5, 1.0, 2.0],
 'spectral_window_eta': 0.05,
 'omega_range': [0.0, 8.0],
 'omega_step': 0.002,
 'numeric_tmax': 400.0,
 'numeric_dt_production': 0.01,
 'numeric_dt_validation': 0.005,
 'ode_rtol': 1e-11,
 'ode_atol': 1e-13,
 'numerical_spectrum_max_relative_error': 0.0002,
 'bead_spectrum_relative_L1_tolerance': 0.01,
 'mc_seed': 2026111701,
 'mc_batches': 16,
 'mc_paths_per_batch': 32768,
 'mc_times': [0.0, 8.0, 0.05],
 'mc_modes': [1, 2],
 'mc_lambdas': [0.0, 1.0],
 'mc_simultaneous_z_tolerance': 5.0,
 'curve_gate': {'finite': True,
                'strict_x': True,
                'hf_residual_rms_over_peak_max': 0.0005,
                'hf_residual_max_over_peak_max': 0.005,
                'isolated_spike_over_peak_max': 0.005,
                'endpoint_local_extrapolation_over_peak_max': 0.005,
                'filter': 'Savitzky-Golay 7 points polynomial4 for diagnostic only; never alter '
                          'plotted arrays',
                'mc_display': 'errorbar markers, no continuous final scientific curve; all time '
                              'points retained'},
 'output_policy': 'small arrays, figures, diagnostics; no full raw trajectories',
 'runtime_class': 'bounded local analytic/moment calculation and stochastic smoke, expected '
                  'minutes; no long campaign'}
OUT = ROOT
REC = ROOT / 'diagnostics'
REC.mkdir(exist_ok=True)
B = CFG['beta']; ETA = CFG['spectral_window_eta']
W = np.arange(0,8.0001,.002)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':190,
                     'path.simplify':False})
COL = {'exact':'#222222','RPMD':'#cc4b37','TRPMD':'#176d91'}

def publish(stage, **kw):
    data = dict(stage=stage,observed_at=time.time(),pid=os.getpid(),
                **kw)
    (REC/'progress.json').write_text(json.dumps(data,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(data),flush=True)

def save_csv(name, values, header):
    with (OUT/name).open('w',encoding='utf-8',newline='\n') as handle:
        np.savetxt(handle,values,delimiter=',',header=header,comments='')

def modes(P):
    k=np.arange(P); spring=2*P/B*np.sin(np.pi*k/P)
    return spring,np.sqrt(1+spring**2)

def lorentz(w,center):
    return ETA/(np.pi*((w-center)**2+ETA**2))

def exact_constants():
    return 1/(4*np.sinh(B/2)**2), 1/np.tanh(B/2)/(2*B)

def exact_spectrum(w=W):
    D,A=exact_constants()
    return 2*D*lorentz(w,0)+A*(lorentz(w,2)+lorentz(w,-2))

def laplace_g(z,gamma,Omega):
    return (z*z+3*gamma*z+2*gamma**2+2*Omega**2)/((z+gamma)*(z*z+2*gamma*z+4*Omega**2))

def spectrum(P,lam,w=W,resolved=False):
    spring,O=modes(P); gamma=2*lam*spring
    z=ETA-1j*w[None,:]
    s=4/(np.pi*B**2*O[:,None]**4)*np.real(laplace_g(z,gamma[:,None],O[:,None]))
    return s if resolved else s.sum(axis=0)

def exact_f(t,spring,O,lam):
    # matrix exponential, used only to cross-check analytic underdamped form
    gamma=2*lam*spring
    nu=np.sqrt((O**2-gamma**2/4).astype(complex))
    f=np.exp(-gamma[:,None]*t/2)*(np.cos(nu[:,None]*t)+
              gamma[:,None]/2*t*np.sinc(nu[:,None]*t/np.pi))
    return f.real

def numeric_moments(P,lam,dt,tmax):
    spring,O=modes(P); gamma=2*lam*spring
    # f is coordinate cross-covariance / stationary coordinate variance.
    def rhs(t,y):
        return np.concatenate((y[P:],-O**2*y[:P]-gamma*y[P:]))
    ts=np.arange(round(tmax/dt)+1)*dt
    sol=solve_ivp(rhs,(0,tmax),np.r_[np.ones(P),np.zeros(P)],t_eval=ts,
                  rtol=CFG['ode_rtol'],atol=CFG['ode_atol'],method='DOP853')
    if not sol.success: raise RuntimeError(sol.message)
    f=sol.y[:P]
    C=(2/(B**2*O[:,None]**4)*f*f).sum(axis=0)
    analytic=exact_f(ts,spring,O,lam)
    error=float(np.max(np.abs(f-analytic)))
    return ts,C,error,sol.nfev

def transform(t,C):
    dt=t[1]-t[0]
    # Same exponential observation window for all methods. Padding evaluates
    # the finite-time Fourier polynomial more densely, adds no information.
    y=C*np.exp(-ETA*t); y=y.copy(); y[0]*=.5; y[-1]*=.5
    nfft=2**math.ceil(math.log2(2*np.pi/(dt*.0003)))
    freq=2*np.pi*rfftfreq(nfft,dt)
    return np.interp(W,freq,2*dt/np.pi*rfft(y,n=nfft).real)

GATES={}
def audit_curve(name,x,y):
    span=max(float(np.max(np.abs(y))),1e-15)
    residual=y-savgol_filter(y,7,4)
    end=np.array([y[0]-(3*y[1]-3*y[2]+y[3]), y[-1]-(3*y[-2]-3*y[-3]+y[-4])])
    # Isolated one-point spikes leave a large residual against the local quartic.
    metrics=dict(finite=bool(np.isfinite(y).all()),strict_x=bool(np.all(np.diff(x)>0)),
        duplicate_points=int(np.count_nonzero(np.diff(x)==0)),
        normalized_hf_rms=float(np.sqrt(np.mean(residual**2))/span),
        normalized_hf_max=float(np.max(np.abs(residual))/span),
        isolated_spike=float(np.max(np.abs(residual))/span),
        endpoint_local_extrapolation=float(np.max(np.abs(end))/span))
    g=CFG['curve_gate']
    metrics['pass']=bool(metrics['finite'] and metrics['strict_x'] and
        metrics['normalized_hf_rms']<=g['hf_residual_rms_over_peak_max'] and
        metrics['normalized_hf_max']<=g['hf_residual_max_over_peak_max'] and
        metrics['isolated_spike']<=g['isolated_spike_over_peak_max'] and
        metrics['endpoint_local_extrapolation']<=g['endpoint_local_extrapolation_over_peak_max'])
    GATES[name]=metrics

def savefig(name,fig):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def sampled_ou(mode,lam):
    P=CFG['beads_production']; spring,O=modes(P); s=spring[mode]; o=O[mode]
    dt=CFG['mc_times'][2]; ts=np.arange(round(CFG['mc_times'][1]/dt)+1)*dt
    gamma=2*lam*s
    M=expm(np.array([[0,1],[-o*o,-gamma]])*dt)
    V=np.diag([1/(B*o*o),1/B]); Q=V-M@V@M.T
    ev,U=np.linalg.eigh((Q+Q.T)/2)
    if np.min(ev)<-1e-12: raise RuntimeError('nonpositive OU covariance')
    L=U*np.sqrt(np.maximum(ev,0))
    if lam==0: L*=0
    rows=[]; initial=[]
    for batch in range(CFG['mc_batches']):
        rng=np.random.default_rng(np.random.SeedSequence([CFG['mc_seed'],mode,int(lam*100),batch]))
        X=rng.standard_normal((CFG['mc_paths_per_batch'],2))*np.sqrt(np.diag(V))
        initial.append([float(np.mean(X[:,0]**2)),float(np.mean(X[:,1]**2))])
        a0=X[:,0]**2-V[0,0]; vals=[np.mean(a0*a0)]
        for j in range(1,len(ts)):
            X=X@M.T
            if lam: X+=rng.standard_normal(X.shape)@L.T
            vals.append(np.mean(a0*(X[:,0]**2-V[0,0])))
        rows.append(vals)
    rows=np.array(rows); avg=rows.mean(axis=0); se=rows.std(axis=0,ddof=1)/np.sqrt(len(rows))
    f=exact_f(ts,np.array([s]),np.array([o]),lam)[0]
    target=2*V[0,0]**2*f*f
    maxz=float(np.max(np.abs(avg-target)/se))
    identity=f'k{mode}_lambda{lam:g}'
    np.savez_compressed(OUT/(identity+'_mc.npz'),t=ts,mean=avg,se=se,
                         batch_means=rows,target=target,initial_moments=initial,M=M,Q=Q)
    publish('stochastic_validation',mode=mode,lam=lam,max_z=maxz)
    return ts,avg,se,target,dict(max_z=maxz,pass_=maxz<=CFG['mc_simultaneous_z_tolerance'])

def main():
    started=time.time(); P=CFG['beads_production']
    publish('smoke')
    # Independent projected q², not the square of a truncated q matrix.
    fock=[]
    D,A=exact_constants()
    for N in [16,32,64]:
        n=np.arange(N); prob=np.exp(-B*n); prob/=prob.sum()
        mu=np.dot(prob,n+.5)
        d=np.dot(prob,(n+.5-mu)**2)
        a=sum((prob[i]-prob[i+2])*(i+1)*(i+2)/(4*B) for i in range(N-2))
        fock.append(dict(N=N,D=float(d),A=float(a),error=float(max(abs(d-D),abs(a-A)))))
    smoke_t,smoke_c,smoke_err,_=numeric_moments(16,1,.02,2)
    assert smoke_err<1e-9 and np.isfinite(smoke_c).all()
    publish('production',smoke_max_f_error=smoke_err)
    spec={'exact':exact_spectrum(),'RPMD':spectrum(P,0),'TRPMD':spectrum(P,1)}
    numeric={}; numeric_checks={}
    for name,lam in [('RPMD',0),('TRPMD',1)]:
        t,c,err,nfev=numeric_moments(P,lam,CFG['numeric_dt_production'],CFG['numeric_tmax'])
        ns=transform(t,c); numeric[name]=ns
        maxerr=float(np.max(abs(ns-spec[name]))/np.max(spec[name]))
        numeric_checks[name]=dict(max_relative_spectrum_error=maxerr,max_f_error=err,nfev=nfev,
            pass_=maxerr<CFG['numerical_spectrum_max_relative_error'])
        audit_curve(name+'_correlation',t,c)
        np.savez_compressed(OUT/(name+'_numeric.npz'),t=t,C=c,omega=W,spectrum=ns)
        publish('numeric_spectrum',method=name,**numeric_checks[name])
    t2,c2,err2,_=numeric_moments(P,1,CFG['numeric_dt_validation'],CFG['numeric_tmax'])
    strict=transform(t2,c2)
    numeric_checks['strict_dt']=dict(max_relative_difference=float(np.max(abs(strict-numeric['TRPMD']))/max(spec['TRPMD'])),
        pass_=bool(np.max(abs(strict-numeric['TRPMD']))/max(spec['TRPMD'])<CFG['numerical_spectrum_max_relative_error']))
    audit_curve('TRPMD_strict_correlation',t2,c2)
    # Spectral analysis and bead convergence.
    spring,O=modes(P); components={lam:spectrum(P,lam,resolved=True) for lam in [0.,.25,.5,1.,2.]}
    bead=[]
    for p in [16,32,64,128]:
        row=dict(P=p,C0=float((2/(B*B*modes(p)[1]**4)).sum()))
        for name,lam in [('RPMD',0),('TRPMD',1)]:
            sp=spectrum(p,lam)
            row[name+'_L1_to_P128']=float(trapezoid(abs(sp-spectrum(128,lam)),W)/trapezoid(spectrum(128,lam),W))
            audit_curve(f'{name}_P{p}_spectrum',W,sp)
        bead.append(row)
    line_table=[]
    for k in [0,1,2,3,4]:
        deg=1 if k==0 else 2
        freq=2*O[k]; sr=spectrum(P,0,w=np.array([freq]))[0]; st=spectrum(P,1,w=np.array([freq]))[0]
        ownr=spectrum(P,0,w=np.array([freq]),resolved=True)[k,0]
        ownt=spectrum(P,1,w=np.array([freq]),resolved=True)[k,0]
        line_table.append(dict(k=k,degeneracy=deg,spring=float(spring[k]),
            RPMD_frequency=float(freq),paired_cosine_weight=float(deg/(B*B*O[k]**4)),
            gamma_TRPMD=float(2*spring[k]),TRPMD_carrier=2.,TRPMD_envelope_rate=float(2*spring[k]),
            old_peak_full_ratio=float(st/sr),old_peak_mode_ratio=float(ownt/ownr)))
    # Main figure: match analytic and numerically propagated ensemble dynamics.
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for a in [ax[0,0],ax[0,1]]:
        for name in spec:
            a.plot(W,spec[name],label=name if a is ax[0,0] else None,c=COL[name],lw=1.6)
        for name in numeric:
            ix=np.arange(0,len(W),65)
            a.plot(W[ix],numeric[name][ix],ls='none',marker='o',ms=3,mfc='none',mec=COL[name],
                   label=name+' moment simulation' if a is ax[0,0] else None)
        a.set(xlim=(0,8),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$')
    ax[0,0].set_title('(a) Nonlinear observable $q^2$: full spectrum')
    ax[0,0].legend(fontsize=8)
    ax[0,1].set(xlim=(1.7,5.4),ylim=(0,.12),title='(b) Fake lines and residual broad background')
    for row in line_table[1:4]:
        ax[0,1].axvline(row['RPMD_frequency'],c='#999999',ls=':',lw=.8)
    for lam in [.25,.5,1.,2.]:
        sp=components[lam].sum(axis=0)
        ax[1,0].plot(W,sp,label=rf'$\lambda={lam:g}$',lw=1.4)
        audit_curve(f'lambda{lam}_spectrum',W,sp)
    ax[1,0].plot(W,spec['exact'],c='k',ls='--',lw=1,label='exact')
    ax[1,0].set(xlim=(1.3,5.3),ylim=(0,.16),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title='(c) Friction scan: broadening is not exact recovery')
    ax[1,0].legend(fontsize=8,ncol=3)
    tt=np.linspace(0,12,2401)
    for k in [1,2]:
        a=1/(B*B*O[k]**4)
        y0=2*a*exact_f(tt,spring[[k]],O[[k]],0)[0]**2
        y1=2*a*exact_f(tt,spring[[k]],O[[k]],1)[0]**2
        ax[1,1].plot(tt,y0/(2*a),ls='--',alpha=.7,label=f'RPMD k={k}')
        ax[1,1].plot(tt,y1/(2*a),label=f'TRPMD k={k}')
        audit_curve(f'k{k}_RPMD_normalized_C',tt,y0/(2*a));audit_curve(f'k{k}_TRPMD_normalized_C',tt,y1/(2*a))
    ax[1,1].set(xlabel=r'$t\omega_0$',ylabel=r'$C_k(t)/C_k(0)$',title='(d) Internal modes lose long-lived coherence')
    ax[1,1].legend(fontsize=8,ncol=2)
    fig.suptitle(r'Harmonic potential; $A=q^2$, $\beta\hbar\omega_0=8$, $P=64$, $\eta=0.05\omega_0$',fontsize=14)
    fig.tight_layout();savefig('spectral_comparison',fig)
    # Mode decomposition figure, including the zero-frequency component.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for k in [0,1,2,3]:
        deg=1 if k==0 else 2
        for i,lam in enumerate([0.,1.]):
            y=deg*components[lam][k]
            axs[i].plot(W,y,label=f'k={k}'+(' pair' if k else ' centroid'))
            audit_curve(f'mode{k}_lam{lam}_spectrum',W,y)
    for i,a in enumerate(axs):
        a.set(xlim=(0,6),ylim=(0,.22),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_{+,k}(\omega)$',title=['RPMD: resolved fake peaks','TRPMD: damped internal modes'][i]);a.legend(fontsize=8)
    fig.tight_layout();savefig('mode_resolved',fig)
    # Actual stochastic trajectory validation; markers, not smoothed MC curves.
    fig,axs=plt.subplots(2,2,figsize=(11,7));mc={}
    for i,k in enumerate(CFG['mc_modes']):
        for j,lam in enumerate(CFG['mc_lambdas']):
            ts,avg,se,target,check=sampled_ou(k,lam)
            mc[f'k{k}_lambda{lam:g}']=check
            a=axs[i,j]; norm=target[0]
            a.plot(ts,target/norm,c='k',lw=1.3,label='analytic Gaussian result')
            ix=np.arange(0,len(ts),5)
            a.errorbar(ts[ix],avg[ix]/norm,yerr=1.96*se[ix]/norm,fmt='o',ms=3,c=COL['TRPMD' if lam else 'RPMD'],capsize=2,label='OU paths; 95% pointwise bars')
            audit_curve(f'MC_reference_k{k}_lambda{lam}',ts,target/norm)
            a.set(title=f'k={k}, '+('RPMD' if lam==0 else 'TRPMD lambda=1'),xlabel=r'$t\omega_0$',ylabel=r'$C_k(t)/C_k(0)$')
            a.legend(fontsize=7)
    fig.tight_layout();savefig('trajectory_validation',fig)
    # Convergence shown as actual spectra plus measured errors; no joined sparse points.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for p in [16,32,64,128]:
        axs[0].plot(W,spectrum(p,0),label=f'P={p}')
    axs[0].set(xlim=(2.35,2.75),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title='First fake peak persists as P increases');axs[0].legend(fontsize=8)
    for name in ['RPMD','TRPMD']:
        vals=[r[name+'_L1_to_P128'] for r in bead[:-1]]
        axs[1].plot([16,32,64],vals,'o',label=name)
    axs[1].axhline(.01,c='k',ls=':',label='1% criterion')
    axs[1].set(xscale='log',yscale='log',xlabel='P (compared with P=128)',ylabel='relative spectral L1 difference',title='Finite-P numerical convergence, not quantum accuracy');axs[1].legend(fontsize=8)
    fig.tight_layout();savefig('convergence',fig)
    for name,s in spec.items():audit_curve(name+'_main_spectrum',W,s)
    for name,s in numeric.items():audit_curve(name+'_numeric_spectrum',W,s)
    # Linear centroid control is exactly the quantum Kubo q spectrum.
    linear=(lorentz(W,1)+lorentz(W,-1))/B
    audit_curve('linear_q_control',W,linear)
    np.savez_compressed(OUT/'spectra.npz',omega=W,exact=spec['exact'],RPMD=spec['RPMD'],TRPMD=spec['TRPMD'],linear_q=linear)
    save_csv('spectra.csv',np.c_[W,spec['exact'],spec['RPMD'],spec['TRPMD']],'omega,exact,RPMD,TRPMD')
    static=dict(exact_D=D,exact_A=A,exact_C0=D+A,
                RPMD_P64_D=float(sum(1/(B*B*O**4))),
                centroid_cosine_weight=1/B**2,
                centroid_weight_over_exact=float(1/B**2/A))
    checks=dict(fock=fock,numerical=numeric_checks,bead=bead,mc=mc,
                curve_failures=[k for k,v in GATES.items() if not v['pass']])
    report=dict(parameters=CFG,static=static,lines=line_table,checks=checks,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest(),
        numpy=np.__version__,scipy=scipy.__version__,elapsed_seconds=time.time()-started)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    (OUT/'curve_quality.json').write_text(json.dumps(GATES,indent=2),encoding='utf-8',newline='\n')
    passed=(all(x['pass_'] for x in numeric_checks.values()) and all(x['pass_'] for x in mc.values())
            and all(bead[2][n+'_L1_to_P128']<.01 for n in ['RPMD','TRPMD']) and not checks['curve_failures'])
    publish('completed',validation_pass=passed,elapsed_seconds=time.time()-started,curve_failures=checks['curve_failures'])
    if not passed:sys.exit(2)


def render_final_figures():
    P=64
    d=np.load(OUT/'spectra.npz')
    spectra={k:d[k] for k in ['exact','RPMD','TRPMD']}
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for a in [ax[0,0],ax[0,1]]:
        for name,sp in spectra.items():a.plot(W,sp,c=COL[name],lw=1.6,label=name)
        for name in ['RPMD','TRPMD']:
            ns=np.load(OUT/(name+'_numeric.npz'))['spectrum'];ix=np.arange(0,len(W),65)
            a.plot(W[ix],ns[ix],ls='none',marker='o',ms=3,mfc='none',mec=COL[name],label=name+' moment simulation')
        a.set(xlim=(0,8),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$')
    ax[0,0].set_title('(a) Full nonlinear $q^2$ spectrum');ax[0,0].legend(fontsize=8)
    ax[0,1].set(xlim=(1.7,5.4),ylim=(.00015,.5),yscale='log',title='(b) Log scale: physical and fake lines')
    for k in [1,2,3]:ax[0,1].axvline(2*modes(P)[1][k],c='#999999',ls=':',lw=.8)
    for lam in [.25,.5,1.,2.]:ax[1,0].plot(W,spectrum(P,lam),label=rf'$\lambda={lam:g}$')
    ax[1,0].plot(W,spectra['exact'],c='k',ls='--',lw=1,label='exact')
    ax[1,0].set(xlim=(1.3,5.3),ylim=(.00015,.5),yscale='log',xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title='(c) Friction scan; all physical peak heights visible')
    ax[1,0].legend(fontsize=8,ncol=3)
    tt=np.linspace(0,12,2401);s,O=modes(P)
    for k,c in [(1,'#2878b5'),(2,'#e48724')]:
        for lam in [0.,1.]:
            y=exact_f(tt,s[[k]],O[[k]],lam)[0]**2
            ax[1,1].plot(tt,y,c=c,ls='--' if lam==0 else '-',alpha=.75 if lam==0 else 1,label=('RPMD' if lam==0 else 'TRPMD')+f' k={k}')
    ax[1,1].set(xlabel=r'$t\omega_0$',ylabel=r'$C_k(t)/C_k(0)$',title='(d) Suppression of persistent internal-mode motion')
    ax[1,1].legend(fontsize=8,ncol=2)
    fig.suptitle(r'$A=q^2$; harmonic potential; $\beta\hbar\omega_0=8$, $P=64$, common $\eta=0.05\omega_0$',fontsize=14)
    fig.tight_layout();savefig('spectral_comparison_final',fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for i,k in enumerate([1,2]):
        for j,lam in enumerate([0,1]):
            data=np.load(OUT/f'k{k}_lambda{lam}_mc.npz');ts=data['t'];norm=data['target'][0];a=axs[i,j]
            a.plot(ts,data['target']/norm,c='k',lw=1.2,label='analytic')
            a.errorbar(ts,data['mean']/norm,yerr=1.96*data['se']/norm,fmt='o',ms=1.8,
                elinewidth=.5,c=COL['RPMD' if lam==0 else 'TRPMD'],capsize=1,label='OU paths (all saved times); 95% bars')
            a.set(title=f'k={k}, '+('RPMD' if lam==0 else 'TRPMD lambda=1'),xlabel=r'$t\omega_0$',ylabel=r'$C_k(t)/C_k(0)$')
            a.legend(fontsize=7)
    fig.tight_layout();savefig('trajectory_validation_final',fig)
    print('rendered two final figures using completed arrays')

def weight_budget():
    D,A=exact_constants();spring,O=modes(CFG['beads_production'])
    ak=1/(B*B*O**4);dc=float(ak.sum());a0=float(ak[0])
    rows=np.array([[D,A,0,0],[dc,a0,dc-a0,0],[a0,a0,0,2*(dc-a0)]])
    labels=['Zero-frequency delta','Physical 2 omega0 delta','Internal-mode false deltas','Internal-mode continuous spectrum']
    colors=['#b7a899','#2878b5','#cc4b37','#43a08a']
    fig,ax=plt.subplots(figsize=(10,3.6));left=np.zeros(3)
    for j in range(4):
        ax.barh(np.arange(3),rows[:,j],left=left,color=colors[j],label=labels[j],height=.55)
        left+=rows[:,j]
    ax.set(yticks=range(3),yticklabels=['Exact','RPMD','TRPMD'],xlabel='Full positive-frequency spectral area',xlim=(0,.067))
    ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=8)
    fig.tight_layout();savefig('spectral_weight_budget',fig)
    save_csv('weight_budget.csv',rows,','.join(labels))
    P=CFG['beads_production'];k=np.arange(P//2+1)
    degeneracy=np.where((k==0)|(k==P//2),1,2)
    lines=np.c_[k,degeneracy,spring[k],2*O[k],degeneracy*ak[k],2*spring[k]]
    save_csv('mode_lines.csv',lines,
        'k,degeneracy,spring_frequency,line_frequency,cosine_coefficient,gamma_lambda1')

if __name__=='__main__':
    main()
    render_final_figures()
    weight_budget()
