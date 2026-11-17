"""Check public runner shapes, independent RNG initialization and exact output."""
from benchmark_support import *
from run_benchmark import RUN, ptr, UP
from native_loader import LIBRARY_NAME
import ctypes

assert ROOT.name != 'anharmonic-linear', 'Run only inside the explicitly staged output.'
n=11;p=64;alpha=.5;dt=.025;steps=20;stride=2
q0,v0,sampling=sample_rings(n,p,alpha,[2026111703,500,p,0,1])
q_repeat,v_repeat,_=sample_rings(n,p,alpha,[2026111703,500,p,0,1])
q_next,_,_=sample_rings(n,p,alpha,[2026111703,500,p,1,1])
assert np.array_equal(q0,q_repeat) and np.array_equal(v0,v_repeat)
assert not np.array_equal(q0,q_next)
noise_seed=int(np.random.SeedSequence([2026111703,500,p,0,2]).generate_state(1,dtype=np.uint64)[0])
states0=np.random.default_rng(noise_seed).integers(0,np.iinfo(np.uint64).max,size=n,dtype=np.uint64)
assert len(np.unique(states0))==n
q,v=q0.copy(),v0.copy();states=states0.copy();coeff=ao_coefficients(p,1.,dt)
out=np.empty((n,1+steps//stride));diagnostics=np.empty((n,10));obs=np.empty((n,out.shape[1],2))
rc=RUN(n,p,steps,stride,dt,alpha,BETA,1.,ptr(coeff),ptr(q),ptr(v),noise_seed,
       ptr(states,UP),ptr(out),ptr(diagnostics),obs.ctypes.data_as(ctypes.c_void_p),2)
assert rc==0 and np.isfinite(out).all() and np.isfinite(diagnostics).all() and np.isfinite(obs).all()
assert np.allclose(out[:,0],q0.mean(1),rtol=0,atol=1e-14)
assert np.allclose(out[:,-1],q.mean(1),rtol=0,atol=1e-14)
assert np.allclose(diagnostics[:,8],out[:,0],rtol=0,atol=1e-14)
assert np.allclose(diagnostics[:,9],out[:,-1],rtol=0,atol=1e-14)
assert np.allclose(obs[:,0,0],np.mean(q0*q0,1)-q0.mean(1)**2,rtol=0,atol=1e-14)
assert np.allclose(obs[:,-1,1],-4*alpha*(np.mean(q**3,1)-q.mean(1)**3),rtol=0,atol=1e-14)
assert not np.array_equal(states,states0)
runner_shapes=[]
for folder in sorted((ROOT/'runs/smoke').glob('*/attempt-002')):
    with np.load(folder/'batch000.npz',allow_pickle=False) as z:
        assert z['correlation'].shape==z['time'].shape==(401,)
        assert z['aux_correlation'].shape==(2,401) and z['diagnostics'].shape==(128,10)
        assert z['initial_moments'].shape==(128,3) and int(z['n'])==128
        runner_shapes.append(dict(case=folder.parent.name,time=list(z['time'].shape),
            spectrum=list(z['spectrum'].shape),diagnostics=list(z['diagnostics'].shape)))
assert len(runner_shapes)==2
reference=ROOT/'runs/exact-1d/attempt-001/alpha0.500/basis180/reference.npz'
with np.load(reference) as z:
    assert z['spectrum'].shape==z['omega'].shape==(4001,)
    assert z['correlation'].shape==z['time'].shape==(4801,)
    assert np.isfinite(z['spectrum']).all()
    exact_frequency=float(z['line_frequency'][np.argmax(z['line_coefficient'])])
record=dict(status='pass',seed_family=2026111703,unique_initial_noise_states=n,
    distinct_batch_rings=True,repeated_seed_exactly_reproducible=True,
    native_input_shape=list(q0.shape),native_output_shape=list(out.shape),
    native_auxiliary_shape=list(obs.shape),diagnostic_shape=list(diagnostics.shape),
    initial_and_final_centroid_verified=True,auxiliary_initial_final_verified=True,
    exact_main_frequency=exact_frequency,runner_shapes=runner_shapes,
    public_library_sha256=sha(ROOT/'work'/LIBRARY_NAME),scope='implementation smoke; no production sampling claim')
json_write(ROOT/'records/public_smoke_checks.json',record)
print(json.dumps(record))
