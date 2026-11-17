from benchmark_support import *
from analyze_benchmark import get,exact,peak_values,band,difference_stats
from run_benchmark import RUN,DP,UP,ptr
import ctypes
from scipy.stats import t as student_t

validation={};quality={};paths=[]
for phase in ['main','refine','extension']:
    paths.extend(sorted((ROOT/'runs'/phase).glob('*/attempt-001/summary.json')))
assert len(paths)==18,('Expected 18 completed ensembles',len(paths))
for path in paths:
    report=json.loads(path.read_text(encoding='utf-8'))
    assert report['paths']==8192 and report['completed_batches']==16
    assert report['max_correlation_SE_over_C0']<CFG['gates']['production_pointwise_correlation_SE_over_C0']
    if report['max_nve_relative_drift'] is not None:assert report['max_nve_relative_drift']<CFG['gates']['nve_max_relative_energy_drift']
    with np.load(path.parent/'aggregate.npz') as z:
        for name,x,y in [('Cq',z['time'],z['correlation']),('Sq',z['omega'],z['spectrum'])]:quality[str(path.parent.relative_to(ROOT))+'/'+name]=curve_quality(x,y)
        for i,name in enumerate(['width','internal_force']):
            c=z['aux_correlation_mean'][i];quality[str(path.parent.relative_to(ROOT))+'/'+name]=curve_quality(z['time'],c)
            quality[str(path.parent.relative_to(ROOT))+'/'+name+'_spectrum']=curve_quality(W,spectrum(z['time'],c))
    e=exact(report['case']['alpha']);reference=np.array([float(e['C0']),float(e['q2']),float(e['q4'])])
    difference=np.abs(np.array(report['initial_moments'])-reference);se=np.array(report['initial_moments_SE'])
    assert np.all(difference < np.maximum(.02*reference,5*se)),('Static moment gate',str(path))
validation['completed_ensemble_cases']=len(paths)
validation['frozen_producer_gates_pass']=True
validation['curve_failures']=[k for k,v in quality.items() if not v['pass']]
assert not validation['curve_failures'],validation['curve_failures']

# Bootstrap paired original 16 batches to quantify the thermostat comparison.
rng=np.random.default_rng(2026111706);bootstrap={}
for alpha in [.01,.5]:
    rp=get(alpha,0.)[0];tr=get(alpha,1.)[0];e=exact(alpha);denom=trapezoid(e['spectrum'],W)
    # The first 16 batches share initial rings. Additional RPMD samples are
    # excluded here, preserving the exact matching used by this comparison.
    rp_b=rp['spectrum_batches'][:16];tr_b=tr['spectrum_batches'][:16]
    records=[]
    for j in range(600):
        ix=rng.integers(0,16,size=16);rs=rp_b[ix].mean(0);ts=tr_b[ix].mean(0)
        rm=peak_values(W,rs,alpha);tm=peak_values(W,ts,alpha)
        records.append([tm['peak_height']-rm['peak_height'],rm['fwhm']-tm['fwhm'],
            (trapezoid(abs(rs-e['spectrum']),W)-trapezoid(abs(ts-e['spectrum']),W))/denom])
    records=np.array(records)
    bootstrap[str(alpha)]=dict(seed=2026111706,replicates=600,paired_original_batches=16,
        quantities=['TRminusRP_peak_height','RPminusTR_FWHM','RPminusTR_relative_L1'],
        percentile_95_interval=np.percentile(records,[2.5,97.5],axis=0).T.tolist())
validation['paired_bootstrap']=bootstrap

# Same trajectory, two output cadences: no new nuclear method or initial state.
alias=[]
for lam in [0.,1.]:
    n=1024;p=128;dt=.025;steps=8000;frames=steps+1
    q,v,_=sample_rings(n,p,.5,[2026111707,round(lam),1]);coeff=ao_coefficients(p,lam,dt)
    states=np.random.default_rng([2026111707,round(lam),2]).integers(0,np.iinfo(np.uint64).max,size=n,dtype=np.uint64)
    output=np.empty((n,frames));diagnostics=np.empty((n,10))
    rc=RUN(n,p,steps,1,dt,.5,BETA,1.,ptr(coeff),ptr(q),ptr(v),2026111707,ptr(states,UP),ptr(output),ptr(diagnostics),None,CFG['threads'])
    assert rc==0
    tf=np.arange(frames)*dt;tc=tf[::2];cf=correlations(output);cc=correlations(output[:,::2])
    sf=spectrum(tf,cf);sc=spectrum(tc,cc)
    relative=float(trapezoid(abs(sf-sc),W)/trapezoid(abs(sf),W))
    item=dict(lambda_=lam,p=p,n=n,duration=200.,native_dt=dt,output_dt=[.025,.05],relative_L1_shared_0to8=relative,pass_=relative<.001)
    alias.append(item);assert item['pass_'],item
    np.savez_compressed(ROOT/f'results/output_cadence_lambda{lam:g}.npz',omega=W,fine_spectrum=sf,coarse_spectrum=sc,time_fine=tf,correlation_fine=cf,time_coarse=tc,correlation_coarse=cc)
    print(json.dumps({'stage':'output_cadence_check',**item}),flush=True)
validation['output_cadence']=alias
validation['curve_count']=len(quality);validation['status']='pass'
validation['source_sha256']=sha(Path(__file__))
json_write(ROOT/'results/curve_quality.json',quality);json_write(ROOT/'results/validation.json',validation)
print(json.dumps({'status':'pass','cases':len(paths),'curve_count':len(quality),'bootstrap':bootstrap,'output_cadence':alias}))
