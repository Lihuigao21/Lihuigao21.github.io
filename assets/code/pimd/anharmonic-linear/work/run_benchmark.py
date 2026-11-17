"""Finite, restartable one-dimensional quartic RPMD/TRPMD benchmark."""
from benchmark_support import *
import argparse,ctypes,sys,traceback,platform
from datetime import datetime,timezone

DP=ctypes.POINTER(ctypes.c_double);UP=ctypes.POINTER(ctypes.c_uint64)
from native_loader import load_native, LIBRARY_NAME
DLL=load_native(ROOT)
RUN=DLL.native_run_1d
RUN.argtypes=[ctypes.c_int]*4+[ctypes.c_double]*4+[DP]*3+[ctypes.c_uint64,UP,DP,DP,ctypes.c_void_p,ctypes.c_int]
RUN.restype=ctypes.c_int

def ptr(a,typ=DP):return a.ctypes.data_as(typ)
def case_name(c):
    return f"alpha{c['alpha']:.3f}_P{c['p']}_lambda{c['lam']:.2f}_dt{c['dt']:.4f}_mu{c['mu']:.1f}"
def cases_for(phase):
    if phase=='smoke':return [dict(alpha=.5,p=64,lam=x,dt=.025,mu=1.,duration=20.,n=128,batches=1) for x in [0.,1.]]
    specs=[]
    if phase=='main':
        for a in [0.,.01,.5]:
            for l in ([0.,1.] if a==0 else [0.,.5,1.,2.]):specs.append((a,64,l,.025,1.))
    elif phase=='refine':
        for p,dt in [(128,.025),(64,.0125)]:
            for l in [0.,1.]:specs.append((.5,p,l,dt,1.))
        specs.append((.01,64,0.,.025,.5))
    return [dict(alpha=a,p=p,lam=l,dt=dt,mu=mu,duration=CFG['duration'],n=CFG['paths_per_batch'],batches=CFG['batches']) for a,p,l,dt,mu in specs]

def progress(stage,**kw):
    value=dict(stage=stage,observed_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw)
    json_write(ROOT/'records/progress.json',value)
    print(json.dumps(value),flush=True)

def aggregate(folder,c,identity):
    files=[folder/f'batch{i:03d}.npz' for i in range(c['batches'])]
    correlations_list=[];spectra_list=[];aux_list=[];diagnostics=[];initial=[]
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            assert json.loads(str(z['identity']))==identity and int(z['n'])==c['n']
            assert z['correlation'].shape==z['time'].shape
            assert all(np.isfinite(z[k]).all() for k in ['time','correlation','spectrum','aux_correlation','diagnostics','initial_moments'])
            t=z['time'].copy();correlations_list.append(z['correlation'].copy());spectra_list.append(z['spectrum'].copy())
            aux_list.append(z['aux_correlation'].copy());diagnostics.append(z['diagnostics'].copy());initial.append(z['initial_moments'].copy())
    cb=np.array(correlations_list);sb=np.array(spectra_list);ab=np.array(aux_list)
    mean=cb.mean(0);se=cb.std(0,ddof=1)/np.sqrt(len(cb)) if len(cb)>1 else np.zeros_like(mean)
    sm=sb.mean(0);ss=sb.std(0,ddof=1)/np.sqrt(len(sb)) if len(sb)>1 else np.zeros_like(sm)
    np.savez_compressed(folder/'aggregate.npz',time=t,omega=W,correlation=mean,correlation_se=se,spectrum=sm,spectrum_se=ss,
        correlation_batches=cb,spectrum_batches=sb,aux_correlation_batches=ab,aux_correlation_mean=ab.mean(0),
        initial_moments=np.concatenate(initial),diagnostics=np.concatenate(diagnostics),identity=json.dumps(identity,sort_keys=True))
    metrics=dict(correlation=curve_quality(t,mean),spectrum=curve_quality(W,sm))
    for j,name in enumerate(['width','internal_force']):
        metrics[name+'_correlation']=curve_quality(t,ab.mean(0)[j])
        metrics[name+'_spectrum']=curve_quality(W,spectrum(t,ab.mean(0)[j])) if np.max(np.abs(ab))>0 else dict(pass_=True,zero_signal=True)
    d=np.concatenate(diagnostics);im=np.concatenate(initial)
    static_mean=im.mean(0);static_se=im.std(0,ddof=1)/np.sqrt(len(im))
    summary=dict(case=c,identity=identity,completed_batches=len(files),paths=len(im),elapsed_seconds=sum(float(np.load(f)['elapsed_seconds']) for f in files),
        C0=float(mean[0]),C0_SE=float(se[0]),max_correlation_SE_over_C0=float(np.max(se)/max(abs(mean[0]),1e-15)),
        initial_moments=static_mean.tolist(),initial_moments_SE=static_se.tolist(),max_nve_relative_drift=float(d[:,2].max()) if c['lam']==0 else None,
        initial_final_q2=[float(d[:,4].mean()),float(d[:,5].mean())],initial_final_q4=[float(d[:,6].mean()),float(d[:,7].mean())],
        max_abs_q=float(d[:,3].max()),curve_quality=metrics)
    json_write(folder/'summary.json',summary)
    return summary

def run_case(c,phase):
    attempt='attempt-002' if phase=='smoke' else 'attempt-001'
    folder=ROOT/f'runs/{phase}/{case_name(c)}/{attempt}';folder.mkdir(parents=True,exist_ok=True)
    code={p:sha(ROOT/'work'/p) for p in ['run_benchmark.py','benchmark_support.py','native_engine.cpp',LIBRARY_NAME]}
    identity=dict(config=c,protocol_sha256=sha(ROOT/'inputs/protocol.json'),code_sha256=code,seed_family=CFG['seed_family'])
    if (folder/'input.json').exists():
        assert json.loads((folder/'input.json').read_text(encoding='utf-8'))==identity,'Resume identity mismatch'
    else:json_write(folder/'input.json',identity)
    n=c['n'];p=c['p'];dt=c['dt'];nsteps=round(c['duration']/dt);stride=round(CFG['output_dt']/dt)
    assert abs(nsteps*dt-c['duration'])<1e-10 and abs(stride*dt-CFG['output_dt'])<1e-10
    t=np.arange(nsteps//stride+1)*CFG['output_dt'];coeff=ao_coefficients(p,c['lam'],dt,c['mu'])
    for batch in range(c['batches']):
        dest=folder/f'batch{batch:03d}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                assert json.loads(str(z['identity']))==identity and int(z['n'])==n and len(z['time'])==len(t)
                assert all(np.isfinite(z[k]).all() for k in ['correlation','spectrum','diagnostics'])
            continue
        seed=[CFG['seed_family'],round(c['alpha']*1000),p,batch,1]
        q,v,sampling=sample_rings(n,p,c['alpha'],seed,c['mu'])
        initial=np.c_[q.mean(1)**2,np.mean(q*q,axis=1),np.mean(q**4,axis=1)]
        output=np.empty((n,len(t)));diag=np.empty((n,10));obs=np.empty((n,len(t),2))
        noise_seed=int(np.random.SeedSequence([CFG['seed_family'],round(c['alpha']*1000),p,batch,2]).generate_state(1,dtype=np.uint64)[0])
        states=np.random.default_rng(noise_seed).integers(0,np.iinfo(np.uint64).max,size=n,dtype=np.uint64)
        assert len(np.unique(states))==n,'Repeated trajectory noise states'
        start=time.perf_counter()
        rc=RUN(n,p,nsteps,stride,dt,c['alpha'],BETA,c['mu'],ptr(coeff),ptr(q),ptr(v),noise_seed,ptr(states,UP),ptr(output),ptr(diag),obs.ctypes.data_as(ctypes.c_void_p),CFG['threads'])
        elapsed=time.perf_counter()-start
        if rc or not np.isfinite(output).all() or not np.isfinite(diag).all():
            json_write(folder/f'failure-batch{batch:03d}.json',dict(return_code=rc,case=c,sampling=sampling,elapsed=elapsed));raise RuntimeError('Native nonfinite or refused batch')
        corr=correlations(output)
        aux=np.array([correlations(obs[:,:,0],float(obs[:,:,0].mean())),correlations(obs[:,:,1])])
        spec=spectrum(t,corr)
        with dest.open('wb') as handle:
            np.savez_compressed(handle,time=t,omega=W,correlation=corr,spectrum=spec,aux_correlation=aux,diagnostics=diag,initial_moments=initial,
                n=n,identity=json.dumps(identity,sort_keys=True),sampling=json.dumps(sampling),noise_seed=noise_seed,elapsed_seconds=elapsed)
        del output,obs,q,v
        progress('running',phase=phase,case=case_name(c),batch=batch+1,batches=c['batches'],seconds=round(elapsed,3),
            energy_drift=float(diag[:,2].max()) if c['lam']==0 else None,initial_q2=float(initial[:,1].mean()),final_q2=float(diag[:,5].mean()))
        if c['lam']==0 and float(diag[:,2].max())>CFG['gates']['nve_max_relative_energy_drift']:
            raise RuntimeError('NVE energy drift gate failed; batch retained')
    result=aggregate(folder,c,identity)
    progress('case_completed',phase=phase,case=case_name(c),paths=result['paths'],C0=result['C0'],max_SE_over_C0=result['max_correlation_SE_over_C0'])
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','main','refine'],required=True);parser.add_argument('--case-index',type=int)
    args=parser.parse_args();cases=cases_for(args.phase)
    if args.case_index is not None:cases=[cases[args.case_index]]
    progress('starting',phase=args.phase,cases=len(cases),python=platform.python_version(),numpy=np.__version__)
    for c in cases:run_case(c,args.phase)
    progress('phase_completed',phase=args.phase,cases=len(cases))
if __name__=='__main__':
    try:main()
    except BaseException as error:
        progress('execution_failed',error_type=type(error).__name__,error=str(error));traceback.print_exc();sys.exit(1)
