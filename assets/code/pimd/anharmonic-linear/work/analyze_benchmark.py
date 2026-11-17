"""Read validated batch aggregates; compare absolute linear Kubo spectra."""
from benchmark_support import *
import argparse
from scipy.stats import t as student_t
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,'savefig.dpi':190,'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
COL={'exact':'#202020',0.:'#c64c38',.5:'#6f5ca8',1.:'#176f9b',2.:'#29946d'}
FIGURE_GATES={}
def path_for(a,l,p=64,dt=.025,mu=1.):
    phase='main' if p==64 and dt==.025 and mu==1 else 'refine'
    name=f'alpha{a:.3f}_P{p}_lambda{l:.2f}_dt{dt:.4f}_mu{mu:.1f}'
    return ROOT/f'runs/{phase}/{name}/attempt-001'
def get(a,l,p=64,dt=.025,mu=1.):
    folder=path_for(a,l,p,dt,mu)
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_batches']==CFG['batches'] and summary['paths']==CFG['batches']*CFG['paths_per_batch']
    zfile=np.load(folder/'aggregate.npz',allow_pickle=False)
    z={key:zfile[key].copy() for key in zfile.files};zfile.close()
    assert all(np.isfinite(z[k]).all() for k in ['time','omega','correlation','spectrum','spectrum_batches'])
    ext=ROOT/'runs/extension'/folder.parent.name/'attempt-001'
    if (ext/'aggregate.npz').exists():
        extra=np.load(ext/'aggregate.npz',allow_pickle=False)
        original_id=json.loads(str(z['identity']));extra_id=json.loads(str(extra['identity']))
        assert original_id['config']==extra_id['config'] and original_id['seed_family']!=extra_id['seed_family']
        for key in ['correlation_batches','spectrum_batches','aux_correlation_batches','initial_moments','diagnostics']:
            z[key]=np.concatenate((z[key],extra[key]),axis=0)
        count=len(z['correlation_batches'])
        for name,key in [('correlation','correlation_batches'),('spectrum','spectrum_batches')]:
            z[name]=z[key].mean(0);z[name+'_se']=z[key].std(0,ddof=1)/np.sqrt(count)
        z['aux_correlation_mean']=z['aux_correlation_batches'].mean(0)
        summary=dict(summary);summary.update(paths=count*CFG['paths_per_batch'],completed_batches=count,C0=float(z['correlation'][0]),C0_SE=float(z['correlation_se'][0]),
            max_correlation_SE_over_C0=float(np.max(z['correlation_se'])/z['correlation'][0]),sources=[str(folder.relative_to(ROOT)),str(ext.relative_to(ROOT))])
    return z,summary,folder
def exact(a):return load_exact(a)[0]
def band(a):return (.8,1.3) if a<.1 else (1.25,2.15)
def peak_values(w,s,a):
    lo,hi=band(a);indices=np.flatnonzero((w>=lo)&(w<=hi));i=indices[np.argmax(s[indices])]
    h=float(s[i]);half=h/2
    left=np.flatnonzero(s[:i]<half);right=np.flatnonzero(s[i+1:]<half)
    if len(left) and len(right):
        j=left[-1];k=i+1+right[0]
        xl=float(np.interp(half,[s[j],s[j+1]],[w[j],w[j+1]]))
        xr=float(np.interp(half,[s[k],s[k-1]],[w[k],w[k-1]]))
        width=xr-xl
    else:width=None
    return dict(peak_frequency=float(w[i]),peak_height=h,fwhm=width,band_area=float(trapezoid(s[indices],w[indices])))
def metrics(a,l):
    z,summary,folder=get(a,l);e=exact(a);m=peak_values(W,z['spectrum'],a);em=peak_values(W,e['spectrum'],a)
    use=(W>=band(a)[0])&(W<=band(a)[1]);areas=trapezoid(z['spectrum_batches'][:,use],W[use],axis=1)
    m.update(alpha=a,lambda_=l,band=list(band(a)),band_area_SE=float(areas.std(ddof=1)/np.sqrt(len(areas))),
        band_area_exact=em['band_area'],peak_frequency_exact=em['peak_frequency'],peak_height_exact=em['peak_height'],fwhm_exact=em['fwhm'],
        relative_spectral_L1=float(trapezoid(abs(z['spectrum']-e['spectrum']),W)/trapezoid(e['spectrum'],W)),
        C0=summary['C0'],C0_SE=summary['C0_SE'],C0_exact=float(e['C0']),
        max_correlation_SE_over_C0=summary['max_correlation_SE_over_C0'],source=str(folder.relative_to(ROOT)),
        all_frequency_area_estimator=summary['C0'],finite_frequency_area=float(trapezoid(z['spectrum'],W)))
    return m
def save(name,fig):
    for i,axis in enumerate(fig.axes):
        for j,item in enumerate(axis.lines):
            x=np.asarray(item.get_xdata(),dtype=float);y=np.asarray(item.get_ydata(),dtype=float)
            if len(x)>10:FIGURE_GATES[f'{name}/axes{i}/line{j}/{item.get_label()}']=curve_quality(x,y)
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def line(ax,z,label,color,ci=True):
    ax.plot(W,z['spectrum'],color=color,label=label,lw=1.5)
    if ci:
        c=student_t.ppf(.975,len(z['spectrum_batches'])-1)*z['spectrum_se']
        ax.fill_between(W,z['spectrum']-c,z['spectrum']+c,color=color,alpha=.16,lw=0)

def draw_main():
    fig,axs=plt.subplots(3,2,figsize=(11,10.2))
    for row,a in enumerate([0.,.01,.5]):
        e=exact(a);r,_,_=get(a,0);tr,_,_=get(a,1)
        ax=axs[row,0];ax.plot(W,e['spectrum'],color=COL['exact'],label='Exact')
        line(ax,r,'RPMD',COL[0.]);line(ax,tr,'TRPMD lambda=1',COL[1.])
        ax.set(xlim=((.7,1.35) if a<.1 else (1.15,2.3)),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title=f'alpha={a:g}: absolute physical band');ax.legend(fontsize=8)
        ax=axs[row,1];t=r['time'];et=e['time'];ax.plot(et,e['correlation'],color=COL['exact'],label='Exact')
        for z,l,label in [(r,0.,'RPMD'),(tr,1.,'TRPMD lambda=1')]:
            ax.plot(t,z['correlation'],color=COL[l],label=label,lw=1.1)
            c=student_t.ppf(.975,len(z['correlation_batches'])-1)*z['correlation_se'];ax.fill_between(t,z['correlation']-c,z['correlation']+c,color=COL[l],alpha=.12,lw=0)
        ax.set(xlim=(0,30),xlabel=r'$t\omega_0$',ylabel=r'$C_K(t)$',title='Linear observable q; absolute correlation')
    fig.suptitle('Same linear observable q; harmonic, weak and strong quartic potentials\nbeta=8, P=64, common eta=0.05; no spectrum normalization',y=1.015)
    save('linear_q_comparison',fig)

def draw_friction():
    fig,axs=plt.subplots(2,2,figsize=(11,7.5))
    for row,a in enumerate([.01,.5]):
        e=exact(a)
        for col in range(2):
            ax=axs[row,col];ax.plot(W,e['spectrum'],color=COL['exact'],label='Exact',lw=1.8)
            for l in [0.,.5,1.,2.]:
                z,_,_=get(a,l);line(ax,z,'RPMD' if l==0 else f'TRPMD lambda={l:g}',COL[l],ci=False)
            ax.set(xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title=f'alpha={a:g}: '+('physical band' if col==0 else 'full spectrum, logarithmic scale'))
            if col==0:ax.set_xlim((.8,1.3) if a<.1 else (1.2,2.2))
            else:ax.set(xlim=(0,8),yscale='log',ylim=(1e-7,2.))
            ax.legend(fontsize=7.5)
    save('friction_scan',fig)

def draw_sources():
    fig,axs=plt.subplots(2,2,figsize=(11,7.4))
    for row,a in enumerate([.01,.5]):
        for l in [0.,1.]:
            z,_,_=get(a,l);t=z['time'];ac=z['aux_correlation_mean']
            for col in [0,1]:
                axs[row,col].plot(W,spectrum(t,ac[col]),color=COL[l],label='RPMD' if l==0 else 'TRPMD lambda=1')
        for col in [0,1]:
            ax=axs[row,col];ax.set(xlim=(0,7),xlabel=r'$\omega/\omega_0$',ylabel='Absolute diagnostic spectrum',title=f'alpha={a:g}: '+('internal width B' if col==0 else 'full internal coupling force'))
            all_y=np.concatenate([item.get_ydata() for item in ax.lines])
            if np.min(all_y)>0:ax.set_yscale('log')
            else:ax.set_yscale('symlog',linthresh=1e-7)
            ax.legend(fontsize=8)
        if a==.01:
            spring=2*64/BETA*np.sin(np.pi/64);om=np.sqrt(1+spring*spring)
            axs[row,0].axvline(2*om,c='.5',ls=':',lw=1)
            for f in [2*om-1,2*om+1]:axs[row,1].axvline(f,c='.5',ls=':',lw=1)
    save('internal_coupling_diagnostics',fig)

def draw_width_memory():
    fig,axs=plt.subplots(1,2,figsize=(11,3.9));results=[]
    for l in [0.,.5,1.,2.]:
        z,_,_=get(.5,l);t=z['time'];batch=z['aux_correlation_batches'][:,0,:];c=batch.mean(0)
        tau=trapezoid(np.exp(-CFG['eta']*t)*c,t)/c[0]
        taus=trapezoid(np.exp(-CFG['eta']*t)[None,:]*batch,t,axis=1)/batch[:,0]
        label='RPMD' if l==0 else f'TRPMD lambda={l:g}'
        axs[0].plot(t,c/c[0],color=COL[l],label=label)
        ratio=c/c[0];ratio_se=(batch-ratio[None,:]*batch[:,[0]]).std(0,ddof=1)/(c[0]*np.sqrt(len(batch)))
        interval=student_t.ppf(.975,len(batch)-1)*ratio_se
        axs[0].fill_between(t,ratio-interval,ratio+interval,color=COL[l],alpha=.13,lw=0)
        FIGURE_GATES[f'width_memory/lambda{l:g}/CI_lower']=curve_quality(t,ratio-interval)
        FIGURE_GATES[f'width_memory/lambda{l:g}/CI_upper']=curve_quality(t,ratio+interval)
        results.append(dict(lambda_=l,width_variance=float(c[0]),windowed_memory_time=float(tau),
            batch_memory_time_SE=float(taus.std(ddof=1)/np.sqrt(len(taus))),time_window_eta=CFG['eta'],
            width_C20_over_C0=float(c[np.argmin(abs(t-20))]/c[0])))
    axs[0].set(xlim=(0,30),xlabel=r'$t\omega_0$',ylabel=r'$C_B(t)/C_B(0)$',title='Strong quartic: auxiliary width memory');axs[0].legend(fontsize=8)
    axs[1].bar(np.arange(4),[x['windowed_memory_time'] for x in results],color=[COL[l] for l in [0.,.5,1.,2.]])
    axs[1].set(xticks=np.arange(4),xticklabels=['RPMD','TR .5','TR 1','TR 2'],ylabel=r'$\tau_B^{(\eta)}\omega_0$',title='Integral memory with the same eta=.05 window')
    save('width_memory',fig);return results

def difference_stats(a,l,p=64,dt=.025):
    base,_,_=get(a,l);fine,_,_=get(a,l,p=p,dt=dt)
    difference=fine['spectrum']-base['spectrum']
    if p==64:
        d=fine['spectrum_batches']-base['spectrum_batches'];se=d.std(0,ddof=1)/np.sqrt(len(d));kind='paired batches'
    else:
        se=np.sqrt(fine['spectrum_se']**2+base['spectrum_se']**2);kind='independent canonical ensembles'
    denom=trapezoid(abs(fine['spectrum']),W)
    return dict(relative_L1=float(trapezoid(abs(difference),W)/denom),integrated_pointwise_SE_scale=float(trapezoid(se,W)/denom),comparison=kind,
        pass_central_L1=bool(trapezoid(abs(difference),W)/denom<CFG['gates']['step_and_bead_L1_relative_tolerance']))

def draw_convergence():
    fig,axs=plt.subplots(2,2,figsize=(11,7.6));checks={}
    for row,l in enumerate([0.,1.]):
        base,_,_=get(.5,l);finep,_,_=get(.5,l,p=128);finedt,_,_=get(.5,l,dt=.0125)
        for z,label,color,ls in [(base,'P64, dt=.025',COL[l],'-'),(finep,'P128, dt=.025','#d89528','--'),(finedt,'P64, dt=.0125','#786292',':')]:
            axs[row,0].plot(W,z['spectrum'],label=label,color=color,ls=ls,lw=1.5)
        axs[row,0].plot(W,exact(.5)['spectrum'],color='k',label='Exact',lw=1)
        axs[row,0].set(xlim=(1.15,2.3),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title='RPMD' if l==0 else 'TRPMD lambda=1');axs[row,0].legend(fontsize=8)
        for z,label,color in [(finep,'P128 - P64','#d89528'),(finedt,'dt/2 - dt','#786292')]:
            delta=z['spectrum']-base['spectrum'];axs[row,1].plot(W,delta,label=label,color=color)
        axs[row,1].axhline(0,color='.7',lw=.8);axs[row,1].set(xlim=(1.15,2.3),xlabel=r'$\omega/\omega_0$',ylabel='Absolute spectral difference');axs[row,1].legend(fontsize=8)
        checks[f'lambda{l:g}_beads']=difference_stats(.5,l,p=128)
        checks[f'lambda{l:g}_step']=difference_stats(.5,l,dt=.0125)
    save('discretization_checks',fig);return checks

def draw_mass_diagnostic():
    p=path_for(.01,0.,mu=.5)/'aggregate.npz'
    if not p.exists():return None
    z,_,_=get(.01,0.,mu=.5);base,_,_=get(.01,0.)
    fig,axs=plt.subplots(1,2,figsize=(11,3.8))
    for d,label,color in [(base,'Physical internal masses=1',COL[0.]),(z,'Diagnostic internal masses=.5','#986b9e')]:
        axs[0].plot(W,d['spectrum'],label=label,color=color)
        axs[1].plot(W,spectrum(d['time'],d['aux_correlation_mean'][0]),label=label,color=color)
    axs[0].plot(W,exact(.01)['spectrum'],c='k',label='Exact')
    axs[0].set(xlim=(.8,1.3),xlabel=r'$\omega/\omega_0$',ylabel=r'$S_+(\omega)$',title='Linear physical observable q')
    axs[1].set(xlim=(1.5,5),xlabel=r'$\omega/\omega_0$',ylabel='Internal width spectrum',title='Auxiliary width: frequency sensitivity')
    if min(np.min(item.get_ydata()) for item in axs[1].lines)>0:axs[1].set_yscale('log')
    else:axs[1].set_yscale('symlog',linthresh=1e-7)
    spring=2*64/BETA*np.sin(np.pi/64);om=np.sqrt(1+spring*spring)
    axs[1].axvline(2*om,c=COL[0.],ls=':');axs[1].axvline(2*om/np.sqrt(.5),c='#986b9e',ls=':')
    for ax in axs:ax.legend(fontsize=7.5)
    save('internal_mass_diagnostic',fig)
    return dict(physical_Hamiltonian_unchanged=True,configuration_distribution_unchanged=True,standard_RPMD=False,internal_mass=.5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    if args.preview:
        fig,axs=plt.subplots(1,2,figsize=(10,3.6))
        for a,ax in zip([.01,.5],axs):
            e=exact(a);ax.plot(W,e['spectrum'],c='k',label='Exact')
            for l in [0.,.5,1.,2.]:
                if (path_for(a,l)/'aggregate.npz').exists():z,_,_=get(a,l);line(ax,z,f'lambda={l:g}',COL[l])
            ax.set(xlim=(.8,1.3) if a<.1 else (1.15,2.3),title=f'DRAFT: alpha={a:g}, completed cases only');ax.legend(fontsize=8)
        save('preview_running',fig);return
    rows=[metrics(a,l) for a in [0.,.01,.5] for l in ([0.,1.] if a==0 else [0.,.5,1.,2.])]
    draw_main();draw_friction();draw_sources();memory=draw_width_memory();convergence=draw_convergence();mass=draw_mass_diagnostic()
    harmonic_r,_,_=get(0.,0.);harmonic_t,_,_=get(0.,1.)
    harmonic_diff=float(np.max(abs(harmonic_r['correlation']-harmonic_t['correlation'])))
    result=dict(parameters=CFG,metrics=rows,convergence=convergence,mass_diagnostic=mass,width_memory=memory,harmonic_RPMD_TRPMD_max_correlation_difference=harmonic_diff,
        analysis_source_sha256=sha(Path(__file__)))
    json_write(OUT/'figure_curve_quality.json',FIGURE_GATES)
    failures=[key for key,value in FIGURE_GATES.items() if not value['pass']]
    assert not failures,failures
    json_write(OUT/'comparison.json',result)
    with (OUT/'comparison.csv').open('w',encoding='utf-8',newline='\n') as h:
        import csv
        keys=['alpha','lambda_','peak_frequency','peak_frequency_exact','peak_height','peak_height_exact','fwhm','fwhm_exact','band_area','band_area_SE','band_area_exact','relative_spectral_L1','C0','C0_exact']
        writer=csv.DictWriter(h,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'harmonic_difference':harmonic_diff,'convergence':convergence}))
if __name__=='__main__':main()
