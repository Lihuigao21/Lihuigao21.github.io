"""Independent NumPy/rFFT and exact-OU controls for native 1D quartic runs."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='2'
import ctypes as ct
import hashlib,json,time
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1];RECORD=ROOT/'records';RECORD.mkdir(exist_ok=True)
from native_loader import load_native, LIBRARY_NAME
LIB=load_native(ROOT)
DP=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
UP=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
LIB.native_run_1d.argtypes=[ct.c_int]*4+[ct.c_double]*4+[DP]*3+[ct.c_uint64,UP,DP,DP,ct.c_void_p,ct.c_int]
LIB.native_run_1d.restype=ct.c_int
LIB.native_step_explicit_1d.argtypes=[ct.c_int]*2+[ct.c_double]*3+[DP]*4+[ct.c_int]
LIB.native_step_explicit_1d.restype=ct.c_int
LIB.native_version.restype=ct.c_char_p
BETA=8.
LIMITS=dict(explicit_max_abs=2e-11,force_max_abs=1e-8,harmonic_energy_drift=3e-11,
    harmonic_centroid_error=3e-11,noise_z=5.,quartic_energy_drift=.01,quartic_order_ratio_min=3.5,
    observables_max_abs=1e-12)

def coefficients(p,dt,lam,mu=1.,beta=BETA):
    result=np.zeros((p//2+1,7))
    spring=2*p/beta*np.sin(np.pi*np.arange(p//2+1)/p)
    for k,s in enumerate(spring):
        mass=1. if k==0 else mu;K=1+s*s;O=np.sqrt(K/mass);gamma=2*lam*s
        if gamma==0:
            F=np.array([[np.cos(O*dt),np.sin(O*dt)/O],[-O*np.sin(O*dt),np.cos(O*dt)]])
            L=np.zeros((2,2))
        else:
            F=expm(np.array([[0.,1.],[-O*O,-gamma]])*dt)
            Sigma=np.diag([p/(beta*K),p/(beta*mass)])
            C=Sigma-F@Sigma@F.T;C=(C+C.T)/2;L=np.linalg.cholesky(C)
        result[k]=[*F.ravel(),L[0,0],L[1,0],L[1,1]]
    return result

def kick(q,v,h,alpha,mu):
    force=-4*alpha*q**3
    mass=np.full(q.shape[-1],mu);mass[0]=1.
    acceleration=np.fft.ifft(np.fft.fft(force,norm='ortho')/mass,norm='ortho').real
    v+=h*acceleration

def reference_step(q,v,c,noise,dt,alpha,mu):
    q=q.copy();v=v.copy();p=q.shape[-1];m=p//2+1
    kick(q,v,dt/2,alpha,mu)
    Q=np.fft.rfft(q,norm='ortho');V=np.fft.rfft(v,norm='ortho')
    z0=noise[:,:,0]+1j*noise[:,:,2];z1=noise[:,:,1]+1j*noise[:,:,3]
    z0[:,[0,-1]]=z0[:,[0,-1]].real;z1[:,[0,-1]]=z1[:,[0,-1]].real
    scale=np.full(m,2**-.5);scale[[0,-1]]=1.
    Qn=c[:,0]*Q+c[:,1]*V+scale*c[:,4]*z0
    Vn=c[:,2]*Q+c[:,3]*V+scale*(c[:,5]*z0+c[:,6]*z1)
    q=np.fft.irfft(Qn,n=p,norm='ortho');v=np.fft.irfft(Vn,n=p,norm='ortho')
    kick(q,v,dt/2,alpha,mu)
    return q,v

def run(q,v,c,steps,stride,dt,alpha,states,threads=6,mu=1.,want_obs=False):
    n,p=q.shape;frames=1+steps//stride;out=np.empty((n,frames));diag=np.empty((n,10))
    obs=np.empty((n,frames,2)) if want_obs else None
    obs_ptr=obs.ctypes.data_as(ct.c_void_p) if obs is not None else None
    rc=LIB.native_run_1d(n,p,steps,stride,dt,alpha,BETA,mu,c,q,v,2026111704,states,out,diag,obs_ptr,threads)
    assert rc==0,rc
    return out,diag,obs

def initial(n,p,rng,mu=1.):
    spring=2*p/BETA*np.sin(np.pi*np.arange(p)/p);K=1+spring*spring
    q=np.fft.ifft(np.fft.fft(rng.normal(size=(n,p)),norm='ortho')/np.sqrt(K),norm='ortho').real*np.sqrt(p/BETA)
    mass=np.full(p,mu);mass[0]=1.
    v=np.fft.ifft(np.fft.fft(rng.normal(size=(n,p)),norm='ortho')/np.sqrt(mass),norm='ortho').real*np.sqrt(p/BETA)
    return np.ascontiguousarray(q),np.ascontiguousarray(v)

def main():
    rng=np.random.default_rng(2026111704);report=dict(limits=LIMITS,seed=2026111704,version=LIB.native_version().decode(),
        source_sha256=hashlib.sha256((ROOT/'work/native_engine.cpp').read_bytes()).hexdigest(),
        dll_sha256=hashlib.sha256((ROOT/'work'/LIBRARY_NAME).read_bytes()).hexdigest(),explicit=[])
    for p in [2,32,64,128]:
        for lam in [0.,.5,1.,2.]:
            for alpha in [0.,.01,.5]:
                for mu in [1.,2.]:
                    n=7;dt=.025;c=coefficients(p,dt,lam,mu)
                    q,v=initial(n,p,rng,mu);noise=rng.normal(size=(n,p//2+1,4))
                    a,b=reference_step(q,v,c,noise,dt,alpha,mu)
                    rc=LIB.native_step_explicit_1d(n,p,dt,alpha,mu,c,noise,q,v,2);assert rc==0
                    err=max(float(np.max(abs(a-q))),float(np.max(abs(b-v))))
                    report['explicit'].append(dict(p=p,lam=lam,alpha=alpha,mu=mu,error=err))
    assert max(x['error'] for x in report['explicit'])<LIMITS['explicit_max_abs']
    x=rng.normal(size=101);eps=1e-5;alpha=.5
    fd=-((.5*(x+eps)**2+alpha*(x+eps)**4)-(.5*(x-eps)**2+alpha*(x-eps)**4))/(2*eps)
    ferr=float(np.max(abs(fd+x+4*alpha*x**3)))
    report['force_error']=ferr;assert ferr<LIMITS['force_max_abs']
    harmonic=[];benchmark=[]
    # Requested T240 pilot, with exact harmonic centroid at every saved time.
    p=64;n=128;dt=.025;steps=9600;stride=2
    q0,v0=initial(n,p,rng);states=rng.integers(0,2**64-1,size=n,dtype=np.uint64)
    t=np.arange(1+steps//stride)*stride*dt
    exact=q0.mean(-1)[:,None]*np.cos(t)+v0.mean(-1)[:,None]*np.sin(t)
    for lam in [0.,.5,1.,2.]:
        q,v=q0.copy(),v0.copy();start=time.perf_counter()
        out,diag,obs=run(q,v,coefficients(p,dt,lam),steps,stride,dt,0.,states.copy(),want_obs=True)
        elapsed=time.perf_counter()-start;err=float(np.max(abs(out-exact)));drift=float(diag[:,2].max())
        harmonic.append(dict(lam=lam,centroid_error=err,max_sampled_energy_drift=drift))
        benchmark.append(dict(lam=lam,alpha=0.,n=n,p=p,steps=steps,dt=dt,T=steps*dt,obs=True,
            threads=6,seconds=elapsed,trajectory_steps_per_second=n*steps/elapsed))
        assert err<LIMITS['harmonic_centroid_error']
        if lam==0:assert drift<LIMITS['harmonic_energy_drift']
    report['harmonic']=harmonic
    # Check strong-quartic NVE conservation/order and mass-2 implementation.
    report['quartic_energy']=[]
    for mu in [1.,2.]:
        q0,v0=initial(n,p,rng,mu);states=rng.integers(0,2**64-1,size=n,dtype=np.uint64);drifts=[]
        for dt in [.025,.0125]:
            q,v=q0.copy(),v0.copy();steps=round(240/dt);stride=round(.05/dt);start=time.perf_counter()
            out,diag,obs=run(q,v,coefficients(p,dt,0.,mu),steps,stride,dt,.5,states.copy(),mu=mu,want_obs=True)
            elapsed=time.perf_counter()-start;drifts.append(float(diag[:,2].max()))
            # Initial/final coupling observables independently computed in bead space.
            for array,idx in [(q0,0),(q,-1)]:
                centroid=array.mean(-1);wanted=np.c_[np.mean(array*array,axis=-1)-centroid**2,
                    -2*(np.mean(array**3,axis=-1)-centroid**3)]
                assert np.max(abs(obs[:,idx]-wanted))<LIMITS['observables_max_abs']
            benchmark.append(dict(lam=0.,alpha=.5,mu=mu,n=n,p=p,steps=steps,dt=dt,T=steps*dt,obs=True,
                threads=6,seconds=elapsed,trajectory_steps_per_second=n*steps/elapsed))
        ratio=drifts[0]/drifts[1]
        report['quartic_energy'].append(dict(mu=mu,dt_coarse=.025,dt_fine=.0125,
            coarse_max_sampled_relative_drift=drifts[0],fine_max_sampled_relative_drift=drifts[1],ratio=ratio))
        assert max(drifts)<LIMITS['quartic_energy_drift'] and ratio>LIMITS['quartic_order_ratio_min']
    # Strong quartic native random-noise cost at the production timestep.
    q,v=initial(n,p,rng);states=rng.integers(0,2**64-1,size=n,dtype=np.uint64);start=time.perf_counter()
    run(q,v,coefficients(p,.025,1.),9600,2,.025,.5,states,want_obs=True)
    elapsed=time.perf_counter()-start
    benchmark.append(dict(lam=1.,alpha=.5,mu=1.,n=n,p=p,steps=9600,dt=.025,T=240.,obs=True,
        threads=6,seconds=elapsed,trajectory_steps_per_second=n*9600/elapsed))
    report['benchmark']=benchmark
    # Complete persistent states imply exact agreement across chunking/threads.
    p=32;n=17;dt=.025;c=coefficients(p,dt,1.,2.)
    q0,v0=initial(n,p,rng,2.);s0=rng.integers(0,2**64-1,size=n,dtype=np.uint64)
    q,v=q0.copy(),v0.copy();s=s0.copy();full,_,fo=run(q,v,c,400,10,dt,.5,s,threads=1,mu=2.,want_obs=True)
    q2,v2=q0.copy(),v0.copy();s2=s0.copy();a,_,ao=run(q2,v2,c,200,10,dt,.5,s2,threads=6,mu=2.,want_obs=True);b,_,bo=run(q2,v2,c,200,10,dt,.5,s2,threads=6,mu=2.,want_obs=True)
    assert np.array_equal(q,q2) and np.array_equal(v,v2) and np.array_equal(s,s2)
    assert np.array_equal(full,np.concatenate([a,b[:,1:]],axis=1))
    assert np.array_equal(fo,np.concatenate([ao,bo[:,1:]],axis=1))
    report['chunk_thread_and_observables_bitwise_equal']=True
    report['noise']=[]
    for mu in [1.,2.]:
        p=32;n=16384;dt=.025;c=coefficients(p,dt,1.,mu)
        q,v=initial(n,p,rng,mu);Q=np.fft.fft(q,norm='ortho');V=np.fft.fft(v,norm='ortho')
        states=rng.integers(0,2**64-1,size=n,dtype=np.uint64)
        run(q,v,c,1,1,dt,0.,states,mu=mu);Q1=np.fft.fft(q,norm='ortho');V1=np.fft.fft(v,norm='ortho')
        cc=c[1];L=np.array([[cc[4],0.],[cc[5],cc[6]]])
        residual=np.c_[Q1[:,1]-cc[0]*Q[:,1]-cc[1]*V[:,1],V1[:,1]-cc[2]*Q[:,1]-cc[3]*V[:,1]]
        sample=np.r_[residual.real,residual.imag]*np.sqrt(2)
        whitened=np.linalg.solve(L,sample.T).T;mean=whitened.mean(0);cov=np.cov(whitened.T);nn=len(sample)
        zm=float(np.max(abs(mean))*np.sqrt(nn));zv=float(np.max(abs(np.diag(cov)-1))*np.sqrt(nn/2));zc=float(abs(cov[0,1])*np.sqrt(nn))
        report['noise'].append(dict(mu=mu,mean_z=zm,variance_z=zv,covariance_z=zc));assert max(zm,zv,zc)<LIMITS['noise_z']
    report['pass']=True
    (RECORD/'native_validation_1d.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pass':True,'explicit_max_abs':max(x['error'] for x in report['explicit']),
        'force_error':report['force_error'],'harmonic':harmonic,'quartic_energy':report['quartic_energy'],
        'noise':report['noise'],'benchmark':benchmark}),flush=True)

if __name__=='__main__':main()
