"""Centroid-surface RPMD on the Part V Eckart barrier (atomic units).

Run beside eckart_four_methods.py: python eckart_rpmd.py
Independent free-ring proposals give the absolute constrained density.
Importance resampling supplies canonical surface configurations; paired
positive/negative flux launches supply the signed recrossing estimator.
No thermostat, fitted transmission, or instanton path enters RPMD.
Dependencies: numpy, scipy, matplotlib. Output: --output (default rpmd-results).
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eckart_four_methods import M, V0, A, KB, AU_TO_FS, TC, energy_rates


def vf(q):
    z = np.tanh(A*q)
    v = V0*(1-z*z)
    return v, 2*A*v*z


def run(t, n, dt, seed, proposals=131072, launches=1024):
    rng = np.random.default_rng(seed)
    beta = 1/(KB*t)
    omega = 2*n/beta*np.sin(np.pi*np.arange(n)/n)
    # FFT is orthonormal: each nonzero real normal coordinate has this variance.
    sd = np.zeros(n)
    sd[1:] = np.sqrt(n/(beta*M))/omega[1:]
    q = np.fft.ifft(np.fft.fft(rng.normal(size=(proposals,n)), norm='ortho')
                    *sd, norm='ortho').real
    weights = np.exp(-beta*vf(q)[0].mean(axis=1))
    static = weights.mean()/(2*np.pi*beta)
    ess = weights.sum()**2/np.sum(weights**2)
    q = q[rng.choice(proposals, launches, p=weights/weights.sum())]
    # Internal Gaussian momenta; the centroid is sampled from positive FLUX,
    # not a half-Maxwell distribution. Pair (q,p) with (q,-p).
    p = rng.normal(scale=np.sqrt(M*n/beta), size=q.shape)
    p -= p.mean(axis=1, keepdims=True)
    pc = np.sqrt(2*M*rng.exponential(1/beta,size=launches))
    p += pc[:,None]
    q = np.concatenate((q,q))
    p = np.concatenate((p,-p))
    c, s = np.cos(omega*dt), np.sin(omega*dt)
    drift = np.full(n, dt/M)
    drift[1:] = s[1:]/(M*omega[1:])
    def energy(q,p):
        return (p*p/(2*M)+.5*M*(n/beta)**2*(q-np.roll(q,1,axis=1))**2
                +vf(q)[0]).sum(axis=1)
    e0 = energy(q,p)
    max_de = 0.
    times, curve = [0.], [1.]
    steps = round(120/AU_TO_FS/dt)
    stride = max(1,round(2/AU_TO_FS/dt))
    for step in range(1,steps+1):
        p += .5*dt*vf(q)[1]
        qk, pk = np.fft.fft(q,norm='ortho'), np.fft.fft(p,norm='ortho')
        q, p = (np.fft.ifft(c*qk+drift*pk,norm='ortho').real,
                np.fft.ifft(c*pk-M*omega*s*qk,norm='ortho').real)
        p += .5*dt*vf(q)[1]
        if step % stride == 0:
            h = (q.mean(axis=1)>0).astype(float)
            times.append(step*dt*AU_TO_FS)
            curve.append(np.mean(h[:launches]-h[launches:]))
            max_de=max(max_de,float(np.max(np.abs(energy(q,p)-e0)))/n)
    times,curve=np.array(times),np.array(curve)
    plateau=curve[(times>=80)&(times<=120)].mean()
    early=curve[(times>=40)&(times<=80)].mean()
    return dict(T_K=t,beads=n,dt_au=dt,seed=seed,proposals=proposals,
                launches=launches,ESS=ess,static_flux=static,kappa=plateau,
                kappa_40_80=early,RPMD_flux=static*plateau,
                max_energy_drift_per_bead_Ha=max_de),times,curve


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('rpmd-results'))
    ap.add_argument('--plot-only',action='store_true',help='Read saved CSVs without rerunning trajectories')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        def read(name):
            with (args.output/name).open() as f:
                return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        figures(read('rpmd-summary.csv'),read('rpmd-plateaus.csv'),args.output)
        return
    blocks=[]; curves=[]
    variants=[(t,n,4.) for t in [200.,260.,300.] for n in [32,64]]
    variants += [(200.,64,2.)]
    for t,n,dt in variants:
        for seed in range(4):
            row,times,curve=run(t,n,dt,7100+seed)
            blocks.append(row)
            curves.extend(dict(T_K=t,beads=n,dt_au=dt,seed=seed,time_fs=x,kappa=y)
                          for x,y in zip(times,curve))
            print(json.dumps(row),flush=True)
    def save(name,rows):
        with (args.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('rpmd-blocks.csv',blocks); save('rpmd-plateaus.csv',curves)
    summary=[]
    for t,n,dt in variants:
        b=[r for r in blocks if (r['T_K'],r['beads'],r['dt_au'])==(t,n,dt)]
        row=dict(T_K=t,beads=n,dt_au=dt)
        for key in ['static_flux','kappa','kappa_40_80','RPMD_flux']:
            v=np.array([r[key] for r in b]);row[key]=v.mean();row[key+'_SE']=v.std(ddof=1)/np.sqrt(len(v))
        row['min_ESS']=min(r['ESS'] for r in b)
        row['max_energy_drift_per_bead_Ha']=max(r['max_energy_drift_per_bead_Ha'] for r in b)
        summary.append(row)
    save('rpmd-summary.csv',summary)
    figures(summary,curves,args.output)


def figures(summary,curves,output):
    # Independent DVR values are read from the separately executed Part V code.
    ref_path=Path(__file__).resolve().parents[2]/'data/reaction-dynamics/eckart-four-methods.csv'
    with ref_path.open() as f: refs={float(r['T_K']):r for r in csv.DictReader(f)}
    selected=[r for r in summary if r['beads']==64 and r['dt_au']==4]
    for mobile in [False,True]:
        fig,ax=plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(6,8) if mobile else (11,4))
        for t in [200,260,300]:
            cc=[r for r in curves if r['T_K']==t and r['beads']==64 and r['dt_au']==4]
            xx=np.unique([r['time_fs'] for r in cc]); yy=np.array([[r['kappa'] for r in cc if r['seed']==seed] for seed in range(4)])
            mean=yy.mean(axis=0);se=yy.std(axis=0,ddof=1)/2
            ax[0].plot(xx,mean,label=f'{t} K');ax[0].fill_between(xx,mean-se,mean+se,alpha=.15)
        ax[0].set(xlabel='propagation time (fs)',ylabel=r'$\kappa_{\rm RP}(t)$',ylim=(.975,1.002))
        ax[0].axvspan(80,120,color='gray',alpha=.1);ax[0].legend(frameon=False)
        tt=np.linspace(180,310,140);er=[energy_rates(t) for t in tt]
        for name,color in [('TST','#78828f'),('instanton','#c05d43')]:
            ax[1].semilogy(tt,[r[name+'_flux_au'] for r in er],label=name,color=color)
        ax[1].scatter(list(refs),[float(r['DVR_flux_au']) for r in refs.values()],label='DVR',color='#22324b',s=22)
        ax[1].errorbar([r['T_K'] for r in selected],[r['RPMD_flux'] for r in selected],yerr=[r['RPMD_flux_SE'] for r in selected],fmt='s',color='#398077',capsize=4,label='RPMD (64 beads)')
        ax[1].axvline(TC,ls=':',color='gray');ax[1].set(xlim=(180,310),ylim=(5e-11,3e-8),xlabel='temperature (K)',ylabel=r'$\mathcal{F}=kQ_R$ (atomic units)');ax[1].legend(frameon=False)
        for a in ax:a.grid(alpha=.15);a.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(output/('rpmd-comparison-mobile.png' if mobile else 'rpmd-comparison.png'),dpi=180);plt.close(fig)


if __name__=='__main__':main()
