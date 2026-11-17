"""Converged quantum Kubo spectra for V=q**2/2+alpha*q**4, A=q.

The physical projected powers q**2 and q**4 are formed before truncation.
The model has m=hbar=omega0=1. All outputs remain within the selected output directory.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

TASK=Path(__file__).resolve().parents[1]
CONFIG=dict(model='H=p^2/2+q^2/2+alpha*q^4; m=hbar=omega0=1',
            observable='connected linear q; Kubo autocorrelation',beta=8.,eta=.05,
            omega_range=[0.,8.],omega_step=.002,time_range=[0.,240.],time_step=.05,
            thermal_probability_cut=1e-17,saved_line_coefficient_cut=1e-20,
            basis_main_line_absolute_tolerance=1e-4,basis_relative_L1_tolerance=.002,
            basis_static_relative_tolerance=.002,
            finite_field_values=[1e-4,5e-5],finite_field_Kubo_relative_tolerance=1e-6,
            curve_gate=dict(normalized_hf_rms_max=.0005,normalized_hf_max_max=.005,
                            normalized_endpoint_extrapolation_max=.005,normalized_isolated_spike_max=.005,
                            filter='Savitzky-Golay 7 points polynomial4; diagnostic only; never smooth arrays'))
SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
W=np.arange(0,8.0001,.002);T=np.arange(0,240.0001,.05)

def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')

def projected_operators(n):
    off=np.sqrt(np.arange(1,n+4)/2)
    q=np.diag(off,1)+np.diag(off,-1)
    q2=q@q;q4=q2@q2
    return q[:n,:n],q2[:n,:n],q4[:n,:n]

def audit_curve(x,y):
    scale=max(float(np.max(np.abs(y))),1e-30)
    residual=y-savgol_filter(y,7,4)
    endpoints=np.array([y[0]-3*y[1]+3*y[2]-y[3],y[-1]-3*y[-2]+3*y[-3]-y[-4]])
    record=dict(finite=bool(np.isfinite(y).all()),strict_x=bool((np.diff(x)>0).all()),
                duplicate_points=int(np.count_nonzero(np.diff(x)==0)),
                normalized_hf_rms=float(np.sqrt(np.mean(residual**2))/scale),
                normalized_hf_max=float(np.max(np.abs(residual))/scale),
                normalized_endpoint_extrapolation=float(np.max(np.abs(endpoints))/scale),
                normalized_isolated_spike=float(np.max(np.abs(residual))/scale))
    record['pass']=bool(record['finite'] and record['strict_x'] and
                        all(record[key[:-4]]<=value for key,value in CONFIG['curve_gate'].items() if key.endswith('_max')))
    return record

def arrays_from_lines(frequency,coefficient,dc):
    eta=CONFIG['eta'];spectrum=2*dc*eta/(np.pi*(W**2+eta**2));correlation=np.full_like(T,dc)
    for start in range(0,len(frequency),128):
        freq=frequency[start:start+128,None];weight=coefficient[start:start+128,None]
        spectrum+=(weight*eta/np.pi*(1/((W-freq)**2+eta**2)+1/((W+freq)**2+eta**2))).sum(axis=0)
        correlation+=(weight*np.cos(freq*T)).sum(axis=0)
    return spectrum,correlation

def finite_field_check(H,Q,beta,reference):
    def mean(field):
        E,U=eigh(H-field*Q,check_finite=False,driver='evd')
        p=np.exp(-beta*(E-E[0]));p/=p.sum()
        return float(p@np.sum(U*(Q@U),axis=0))
    rows=[]
    for eps in CONFIG['finite_field_values']:
        plus=mean(eps);minus=mean(-eps);C0=(plus-minus)/(2*eps*beta)
        rows.append(dict(field=eps,plus_mean=plus,minus_mean=minus,response_Kubo_C0=C0,
                         relative_error=abs(C0/reference-1)))
    return dict(method='thermodynamic derivative (d<q>/df)/beta for H-f*q',rows=rows,
                tolerance=CONFIG['finite_field_Kubo_relative_tolerance'],
                passed=all(x['relative_error']<CONFIG['finite_field_Kubo_relative_tolerance'] for x in rows))

def calculate(alpha,n,output):
    start=time.time();beta=CONFIG['beta'];folder=output/f'alpha{alpha:.3f}'/f'basis{n}'
    folder.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(stage='exact_1d_start',alpha=alpha,basis=n)),flush=True)
    Q,Q2,Q4=projected_operators(n);H=np.diag(np.arange(n)+.5)+alpha*Q4
    E,U=eigh(H,check_finite=False,driver='evd')
    p=np.exp(-beta*(E-E[0]));p/=p.sum();keep=np.flatnonzero(p>CONFIG['thermal_probability_cut'])
    Uk=U[:,keep];pk=p[keep];M=U.T@Q@Uk;diag=M[keep,np.arange(len(keep))]
    mean=float(pk@diag);q2=float(pk@np.sum(Uk*(Q2@Uk),axis=0));q4=float(pk@np.sum(Uk*(Q4@Uk),axis=0))
    dc=float(pk@((diag-mean)**2));freq=[];weight=[];initial=[];final=[];matrix2=[]
    for j,k in enumerate(keep):
        dest=np.arange(k+1,n);gap=E[dest]-E[k]
        coeff=2*p[k]*(-np.expm1(-beta*gap))/(beta*gap)*M[dest,j]**2
        select=coeff>0
        freq.extend(gap[select]);weight.extend(coeff[select]);initial.extend(np.full(np.count_nonzero(select),k));final.extend(dest[select]);matrix2.extend(M[dest[select],j]**2)
    freq=np.asarray(freq);weight=np.asarray(weight);initial=np.asarray(initial);final=np.asarray(final);matrix2=np.asarray(matrix2)
    spectrum,correlation=arrays_from_lines(freq,weight,dc);C0=float(dc+weight.sum())
    selected=weight>CONFIG['saved_line_coefficient_cut'];sf=freq[selected];sc=weight[selected]
    tail=float(p[p<=CONFIG['thermal_probability_cut']].sum());norm=float(np.linalg.norm(Q,ord=2))
    strongest=np.argsort(sc)[::-1][:20]
    lines=[dict(initial=int(initial[selected][i]),final=int(final[selected][i]),frequency=float(sf[i]),coefficient=float(sc[i])) for i in strongest]
    residual=H@Uk-Uk*E[keep]
    static=dict(mean_q=mean,q2=q2,q4=q4,dc=dc,C0=C0,ordinary_q_variance=q2-mean*mean,
                main_frequency=lines[0]['frequency'],main_single_transition_coefficient=lines[0]['coefficient'],
                line_area=float(weight.sum()),saved_line_area=float(sc.sum()),
                omitted_line_area=float(weight[~selected].sum()),omitted_initial_probability=tail,
                omitted_initial_Kubo_area_bound=2*tail*norm*norm,
                thermal_last4_basis_population=float(pk@np.sum(Uk[-4:]**2,axis=0)),
                low_state_eigen_residual_max=float(np.max(np.abs(residual))),
                low_state_orthogonality_max=float(np.max(np.abs(Uk.T@Uk-np.eye(len(keep))))))
    quality=dict(spectrum=audit_curve(W,spectrum),correlation=audit_curve(T,correlation))
    field=finite_field_check(H,Q,beta,C0) if n==180 else None
    report=dict(config=CONFIG,source_sha256=SOURCE_HASH,alpha=alpha,basis=n,static=static,strongest_lines=lines,
                retained_thermal_initial_states=len(keep),curve_quality=quality,finite_field_check=field,
                numpy=np.__version__,scipy=scipy.__version__,elapsed_seconds=time.time()-start)
    np.savez_compressed(folder/'reference.npz',omega=W,spectrum=spectrum,time=T,correlation=correlation,
                        line_frequency=sf,line_coefficient=sc,line_initial=initial[selected],line_final=final[selected],
                        dc=dc,C0=C0,mean_q=mean,q2=q2,q4=q4,alpha=alpha,beta=beta,eta=CONFIG['eta'],energy=E,thermal_probability=p)
    with (folder/'lines.csv').open('w',encoding='utf-8',newline='\n') as handle:
        np.savetxt(handle,np.c_[initial[selected],final[selected],sf,sc,p[initial[selected]],p[final[selected]],matrix2[selected]],delimiter=',',
                   header='initial,final,frequency,cosine_coefficient,p_initial,p_final,matrixelement_squared',comments='')
    with (folder/'spectrum.csv').open('w',encoding='utf-8',newline='\n') as handle:
        np.savetxt(handle,np.c_[W,spectrum],delimiter=',',header='omega,S_plus',comments='')
    write_json(folder/'summary.json',report)
    print(json.dumps(dict(stage='exact_1d_done',alpha=alpha,basis=n,C0=C0,main=static['main_frequency'],
                         curve_pass=all(x['pass'] for x in quality.values()),elapsed=report['elapsed_seconds'])),flush=True)
    return report,spectrum

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--alphas',type=float,nargs='+',default=[0,.01,.5]);parser.add_argument('--bases',type=int,nargs='+',default=[80,120,180]);parser.add_argument('--output',type=Path,default=TASK/'runs/exact-1d/attempt-001')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'config.json',dict(config=CONFIG,source_sha256=SOURCE_HASH,alphas=args.alphas,bases=args.bases))
    rows=[];curve_pass=True;field_pass=True
    for alpha in args.alphas:
        previous=None
        for n in args.bases:
            report,spectrum=calculate(alpha,n,args.output)
            curve_pass &= all(x['pass'] for x in report['curve_quality'].values())
            if report['finite_field_check'] is not None:field_pass &= report['finite_field_check']['passed']
            if previous:
                old,old_s=previous
                delta=dict(alpha=alpha,coarse_basis=old['basis'],fine_basis=n,
                           main_frequency_absolute=abs(report['static']['main_frequency']-old['static']['main_frequency']),
                           spectrum_relative_L1=float(trapezoid(abs(spectrum-old_s),W)/trapezoid(abs(spectrum),W)))
                for key in ['C0','q2','q4']:
                    delta[key+'_relative']=abs(report['static'][key]-old['static'][key])/abs(report['static'][key])
                delta['pass']=bool(delta['main_frequency_absolute']<CONFIG['basis_main_line_absolute_tolerance'] and delta['spectrum_relative_L1']<CONFIG['basis_relative_L1_tolerance'] and all(delta[k+'_relative']<CONFIG['basis_static_relative_tolerance'] for k in ['C0','q2','q4']))
                rows.append(delta)
            previous=report,spectrum
    checks=dict(source_sha256=SOURCE_HASH,comparisons=rows,all_basis_pass=all(x['pass'] for x in rows),all_curve_pass=bool(curve_pass),all_finite_field_pass=bool(field_pass))
    write_json(args.output/'convergence.json',checks)
    (args.output/'source_exact_1d.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(stage='exact_1d_all_done',**{k:v for k,v in checks.items() if k!='comparisons'})),flush=True)
    assert checks['all_basis_pass'] and checks['all_curve_pass'] and checks['all_finite_field_pass']

if __name__=='__main__':main()
