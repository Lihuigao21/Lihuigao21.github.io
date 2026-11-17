"""Independent canonical rings and absolute all-origin spectral estimators."""
import os
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy.linalg import expm
from scipy.fft import rfft,irfft,next_fast_len,rfftfreq
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'inputs/protocol.json').read_text(encoding='utf-8'))
BETA=CFG['beta'];W=np.arange(0,CFG['omega_max']+1e-9,CFG['omega_step'])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    for attempt in range(6):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==5:raise
            time.sleep(.05*(attempt+1))

def sample_rings(n,p,alpha,seed,internal_mass=1.):
    rng=np.random.default_rng(seed)
    spring=2*p/BETA*np.sin(np.pi*np.arange(p//2+1)/p)
    omega=np.sqrt(1+spring*spring)
    chunks=[];accepted=0;proposed=0;actions=[]
    while accepted<n:
        m=min(8192,max(256,2*(n-accepted)))
        z=rng.normal(size=(m,p))*np.sqrt(p/BETA)
        q=irfft(rfft(z,axis=1)/omega[None,:],n=p,axis=1)
        action=BETA*alpha*np.mean(q**4,axis=1)
        keep=np.log(rng.random(m)) < -action
        selected=q[keep][:n-accepted]
        chunks.append(selected);accepted+=len(selected);proposed+=m
        if len(selected):actions.append(action[keep][:len(selected)])
    q=np.ascontiguousarray(np.concatenate(chunks))
    # Uniform internal-mode mass: v is the actual derivative dq/dt.
    white=rng.normal(size=q.shape)*np.sqrt(p/BETA)
    v=white/np.sqrt(internal_mass)+(1-1/np.sqrt(internal_mass))*white.mean(axis=1,keepdims=True)
    return q,np.ascontiguousarray(v),dict(proposed=proposed,retained=n,acceptance_lower_bound=n/proposed,
        mean_retained_quartic_action=float(np.mean(np.concatenate(actions))),seed=seed,internal_mass=internal_mass)

def ao_coefficients(p,lam,dt,internal_mass=1.):
    spring=2*p/BETA*np.sin(np.pi*np.arange(p//2+1)/p)
    mu=np.full_like(spring,internal_mass);mu[0]=1.
    omega=np.sqrt((1+spring**2)/mu)
    gamma=2*lam*spring/np.sqrt(mu)
    coeff=np.zeros((len(spring),7))
    for k,(w,fr,mass) in enumerate(zip(omega,gamma,mu)):
        F=expm(np.array([[0.,1.],[-w*w,-fr]])*dt)
        coeff[k,:4]=F.ravel()
        if fr:
            covariance=np.diag([p/(BETA*(1+spring[k]**2)),p/(BETA*mass)])
            Q=covariance-F@covariance@F.T;Q=(Q+Q.T)/2
            assert np.linalg.eigvalsh(Q).min()>-1e-10*np.trace(covariance)
            l0=np.sqrt(max(Q[0,0],0.));l1=Q[1,0]/l0
            coeff[k,4:]=[l0,l1,np.sqrt(max(Q[1,1]-l1*l1,0.))]
    return np.ascontiguousarray(coeff)

def correlations(series,mean=0.):
    n=series.shape[1];nfft=next_fast_len(2*n)
    F=rfft(series-mean,n=nfft,axis=1,workers=2)
    c=irfft(F*F.conj(),n=nfft,axis=1,workers=2)[:,:n]
    return np.mean(c,axis=0)/np.arange(n,0,-1)

def spectrum(t,c):
    dt=t[1]-t[0];y=np.asarray(c)*np.exp(-CFG['eta']*t)
    y=y.copy();y[0]*=.5;y[-1]*=.5
    nfft=2**int(np.ceil(np.log2(2*np.pi/(dt*.0003))))
    f=2*np.pi*rfftfreq(nfft,dt)
    return np.interp(W,f,2*dt/np.pi*rfft(y,n=nfft).real)

def curve_quality(x,y):
    scale=max(float(np.max(np.abs(y))),1e-15)
    residual=y-savgol_filter(y,7,4)
    endpoint=np.array([y[0]-(3*y[1]-3*y[2]+y[3]),y[-1]-(3*y[-2]-3*y[-3]+y[-4])])
    q=dict(finite=bool(np.isfinite(y).all()),strictly_increasing_x=bool(np.all(np.diff(x)>0)),duplicate_points=int(np.sum(np.diff(x)==0)),
        normalized_hf_residual_rms=float(np.sqrt(np.mean(residual**2))/scale),normalized_hf_residual_max=float(np.max(np.abs(residual))/scale),
        endpoint_artifact=float(np.max(np.abs(endpoint))/scale),isolated_spike=float(np.max(np.abs(residual))/scale))
    g=CFG['gates'];q['pass']=bool(q['finite'] and q['strictly_increasing_x'] and q['duplicate_points']==0 and
        q['normalized_hf_residual_rms']<g['curve_hf_residual_RMS_over_peak'] and q['normalized_hf_residual_max']<g['curve_hf_residual_max_over_peak'] and
        q['endpoint_artifact']<g['curve_endpoint_artifact_over_peak'] and q['isolated_spike']<g['curve_isolated_spike_over_peak'])
    return q

def load_exact(alpha):
    p=ROOT/f'runs/exact-1d/attempt-001/alpha{alpha:.3f}/basis180/reference.npz'
    return np.load(p),p
