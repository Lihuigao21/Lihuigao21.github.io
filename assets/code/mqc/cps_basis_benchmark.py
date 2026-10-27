"""Reproduce MQC VIII: paired diabatic/adiabatic eCMM on Tully SAC.

Dependencies: numpy, scipy, matplotlib; keep cps_basis_dynamics.py beside this file.
Run from any directory, with an explicit output directory:
  python cps_basis_benchmark.py --output results --validate
  python cps_basis_benchmark.py --output smoke --smoke 50
  python cps_basis_benchmark.py --output figures --plot-data path/to/mqc-viii

The full run uses independent batches and the published stopping rule.
Smoke output is an implementation check, not a converged scientific result.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.signal import savgol_filter
from cps_basis_dynamics import CPSBasisRK4, TullySAC, real_two_state_frame

MODEL = TullySAC()
MASS, END = 2000., 800.
GAMMAS = {"W": (np.sqrt(3.)-1)/2, "half": .5}
FEATURES = ["dia0", "dia1", "adi0", "adi1", "coherence_real", "coherence_imag", "q", "p"]
ROOT = None

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False)+"\n").encode("utf-8"))

def frame(q):
    return real_two_state_frame(MODEL.V(q), MODEL.dV(q))

def initial_pool():
    parts = []
    for n, ns, es in ((10000, 20261027, 20261028), (40000, 20261029, 20261030)):
        rng = np.random.default_rng(ns)
        q, p = rng.normal(-4., .5, n), rng.normal(20., 1., n)
        rng = np.random.default_rng(es)
        z = rng.normal(size=(n, 2))+1j*rng.normal(size=(n, 2))
        z /= np.linalg.norm(z, axis=1)[:, None]
        parts.append(dict(q=q, p=p, unit_z=z))
    return {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}

def quantum(box=20., ngrid=2048, dt=.5, label="base"):
    out = ROOT/"results"/f"quantum_{label}.npz"
    if out.exists():
        data = dict(np.load(out))
        if data["obs"].shape != (int(END/dt)+1, 8) or not np.isfinite(data["obs"]).all():
            raise RuntimeError("invalid existing quantum record")
        return data
    begin = time.perf_counter()
    q = np.linspace(-box, box, ngrid, endpoint=False)
    dx = q[1]-q[0]
    p = 2*np.pi*np.fft.fftfreq(ngrid, dx)
    v = MODEL.V(q)
    e, u = np.linalg.eigh(v)
    uv = (u*np.exp(-1j*dt*e)[:, None, :]) @ u.swapaxes(-1, -2)
    kinetic = np.exp(-1j*dt*p*p/(4*MASS))
    psi = np.zeros((ngrid, 2), dtype=complex)
    psi[:, 0] = np.exp(-(q+4.)**2+20j*q)
    psi /= np.sqrt(np.sum(abs(psi)**2)*dx)
    obs = np.empty((int(round(END/dt))+1, 8))
    norm_error = edge_max = energy_drift = 0.
    initial_energy = None
    for k in range(len(obs)):
        density = abs(psi)**2
        pa = np.einsum("nji,nj->ni", u, psi)
        coh = np.sum(psi[:, 0]*psi[:, 1].conj())*dx
        derivative = np.fft.ifft(p[:, None]*np.fft.fft(psi, axis=0), axis=0)
        mom = np.sum((psi.conj()*derivative).real)*dx
        obs[k] = [*(density.sum(axis=0)*dx), *(np.sum(abs(pa)**2, axis=0)*dx),
                  coh.real, coh.imag, np.sum(q[:, None]*density)*dx, mom]
        norm_error = max(norm_error, abs(density.sum()*dx-1.))
        edge_max = max(edge_max, density[abs(q)>box-2].sum()*dx)
        energy = .5/MASS*np.sum(abs(derivative)**2)*dx + np.einsum("ni,nij,nj->", psi.conj(), v, psi).real*dx
        if initial_energy is None:
            initial_energy = energy
        energy_drift = max(energy_drift, abs(energy-initial_energy))
        if k+1 < len(obs):
            psi = np.fft.ifft(kinetic[:, None]*np.fft.fft(psi, axis=0), axis=0)
            psi = np.einsum("nij,nj->ni", uv, psi)
            psi = np.fft.ifft(kinetic[:, None]*np.fft.fft(psi, axis=0), axis=0)
    result = dict(time=np.arange(len(obs))*dt, obs=obs, norm_error=np.array(norm_error),
                  edge_max=np.array(edge_max), energy_drift=np.array(energy_drift))
    np.savez_compressed(out, **result)
    write_json(ROOT/"records"/f"quantum_{label}.json", {"box": box, "ngrid": ngrid, "dt": dt,
        "runtime_seconds": time.perf_counter()-begin, "norm_error": norm_error,
        "edge_max": edge_max, "energy_drift": energy_drift})
    print("quantum", label, "seconds", round(time.perf_counter()-begin, 2), flush=True)
    return result


def quantum_controls():
    base = quantum()
    controls = {"dt": quantum(dt=.25, label="dt"),
                "grid": quantum(ngrid=4096, label="grid"),
                "box": quantum(box=40, ngrid=4096, label="box")}
    report = {}
    for name, data in controls.items():
        stride = int(round(len(data["time"][:-1])/len(base["time"][:-1])))
        diff = abs(base["obs"]-data["obs"][::stride])
        record = {"population_max_difference": float(diff[:, :4].max()),
                  "q_max_difference": float(diff[:, 6].max()), "p_max_difference": float(diff[:, 7].max()),
                  "norm_error": float(data["norm_error"]), "edge_max": float(data["edge_max"])}
        record["passed"] = bool(record["population_max_difference"] < 2e-4 and
            record["q_max_difference"] < 2e-3 and record["p_max_difference"] < 2e-3 and
            record["norm_error"] < 1e-9 and record["edge_max"] < 1e-9)
        report[name] = record
    report["passed"] = all(v["passed"] for v in report.values()) and float(base["norm_error"])<1e-9 and float(base["edge_max"])<1e-9
    write_json(ROOT/"results/quantum_controls.json", report)
    if not report["passed"]:
        raise RuntimeError("quantum reference convergence failed")
    return base


def observable(method):
    dia = method.density_estimator("diabatic")
    adi = method.density_estimator("adiabatic")
    return np.column_stack((dia[:, 0, 0].real, dia[:, 1, 1].real,
        adi[:, 0, 0].real, adi[:, 1, 1].real, dia[:, 0, 1].real, dia[:, 0, 1].imag,
        method.initial_weight*method.q, method.initial_weight*method.p))


def propagate(pool, start, stop, gamma, dt):
    begin = time.perf_counter()
    z = pool["unit_z"][start:stop]*np.sqrt(1+2*gamma)
    methods = [CPSBasisRK4(MODEL, pool["q"][start:stop], pool["p"][start:stop], z,
        mass=MASS, gamma=gamma, representation=rep, adiabatic_provider=frame)
        for rep in ("diabatic", "adiabatic")]
    nt = int(round(END/dt))+1
    sums, squares = np.empty((2, nt, 8)), np.empty((2, nt, 8))
    energy_max = np.zeros(2)
    sphere_max = np.zeros(2)
    coordinate_pair_max = momentum_pair_max = electronic_pair_max = 0.
    for k in range(nt):
        for j, method in enumerate(methods):
            o = observable(method)
            sums[j, k], squares[j, k] = o.sum(axis=0), (o*o).sum(axis=0)
            energy_max[j] = max(energy_max[j], np.max(abs(method.mapping_energy()-method.initial_energy)))
            sphere_max[j] = max(sphere_max[j], np.max(abs(method.sphere_error())))
        coordinate_pair_max = max(coordinate_pair_max, np.max(abs(methods[0].q-methods[1].q)))
        momentum_pair_max = max(momentum_pair_max, np.max(abs(methods[0].p-methods[1].p)))
        electronic_pair_max = max(electronic_pair_max, np.max(abs(
            methods[0].density_estimator("diabatic")-methods[1].density_estimator("diabatic"))))
        if k+1 < nt:
            for method in methods:
                method.step(dt)
    return dict(time=np.arange(nt)*dt, sums=sums, squares=squares, n=np.array(stop-start),
        energy_max=energy_max, sphere_max=sphere_max, coordinate_pair_max=np.array(coordinate_pair_max),
        momentum_pair_max=np.array(momentum_pair_max), electronic_pair_max=np.array(electronic_pair_max),
        runtime_seconds=np.array(time.perf_counter()-begin))


def curve_quality(t, y):
    amplitude = max(float(np.ptp(y)), 1e-8)
    residual = y-savgol_filter(y, 7, 3, mode="interp")
    r = {"finite": bool(np.isfinite(y).all()), "increasing_time": bool(np.all(np.diff(t)>0)),
         "duplicates": int(len(t)-len(np.unique(t))),
         "normalized_residual_rms": float(np.sqrt(np.mean(residual**2))/amplitude),
         "normalized_residual_max": float(abs(residual).max()/amplitude),
         "normalized_second_difference_max": float(abs(np.diff(y, n=2)).max()/amplitude),
         "normalized_endpoint_residual": float(max(abs(residual[:3]).max(), abs(residual[-3:]).max())/amplitude)}
    r["passed"] = bool(r["finite"] and r["increasing_time"] and r["duplicates"]==0 and
        r["normalized_residual_rms"]<.002 and r["normalized_residual_max"]<.02 and
        r["normalized_second_difference_max"]<.02 and r["normalized_endpoint_residual"]<.02)
    return r

def run_ensemble(pool, label, gamma, maximum, smoke):
    total = squares = previous = None
    stable = 0
    history = []
    maximum = smoke if smoke else maximum
    batch_size = smoke if smoke else 500
    energy = sphere = 0.
    for start in range(0, maximum, batch_size):
        stop = min(start+batch_size, maximum)
        batch = propagate(pool, start, stop, gamma, 1.)
        if total is None:
            total, squares = batch["sums"].copy(), batch["squares"].copy()
        else:
            total += batch["sums"]
            squares += batch["squares"]
        mean = total/stop
        se = np.sqrt(np.maximum((squares-total*total/stop)/(stop-1), 0)/stop)
        change = float(abs(mean[:, :, :4]-previous).max()) if previous is not None else 1.
        stable = stable+1 if change<.003 else 0
        halfwidth = float(1.96*se[:, :, :4].max())
        passed = bool(not smoke and stop>=500 and halfwidth<.03 and stable>=3)
        energy = max(energy, float(batch["energy_max"].max()))
        sphere = max(sphere, float(batch["sphere_max"].max()))
        row = dict(n=stop, halfwidth_max=halfwidth, mean_change_max=change,
            stable_batches=stable, passed=passed, energy_max=energy, sphere_max=sphere,
            basis_max=float(abs(mean[0,:,:4]-mean[1,:,:4]).max()))
        history.append(row)
        print(label, row, flush=True)
        previous = mean[:,:,:4].copy()
        if energy>=2e-6 or sphere>=2e-6:
            raise RuntimeError("invariant gate failed")
        if passed:
            break
    np.savez_compressed(ROOT/"results"/f"ensemble_{label}.npz",time=batch["time"],mean=mean,se=se,n=np.array(stop))
    for r, rep in enumerate(("diabatic", "adiabatic")):
        table = np.column_stack((batch["time"], mean[r], se[r]))
        text = ",".join(["time"]+FEATURES+[f"se_{f}" for f in FEATURES])
        with (ROOT/f"{label}_{rep}.csv").open("w", encoding="utf-8", newline="\n") as handle:
            np.savetxt(handle, table, delimiter=",", header=text, comments="")
    write_json(ROOT/f"sampling_{label}.json", history)
    return row


def checks(pool):
    report = {}
    for label, gamma in GAMMAS.items():
        controls = {}
        for dt in (2., 1., .5):
            data = propagate(pool, 0, 50, gamma, dt)
            np.savez_compressed(ROOT/"results"/f"dt_{label}_{dt:g}.npz", **data)
            controls[dt] = data
        diff = abs(controls[1.]["sums"]/50-controls[.5]["sums"][:,::2]/50)
        pair = abs(controls[.5]["sums"][0]-controls[.5]["sums"][1])/50
        report[label] = dict(population_dt_difference=float(diff[:,:,:4].max()),
            population_basis_difference_dt_half=float(pair[:,:4].max()))
        assert report[label]["population_dt_difference"]<2e-4
        assert report[label]["population_basis_difference_dt_half"]<2e-4
    write_json(ROOT/"timestep_controls.json", report)


def plot_data(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11,"path.simplify":False,"axes.spines.top":False,"axes.spines.right":False})
    ref = np.loadtxt(directory/"quantum.csv", delimiter=",", skiprows=1)
    data = {(g,r):np.loadtxt(directory/f"{g}_{r}.csv", delimiter=",", skiprows=1)
            for g in GAMMAS for r in ("diabatic","adiabatic")}
    colors = {"W":"#0077AA", "half":"#C65D24"}
    labels = {"W":"W kernel", "half":"gamma = 1/2"}
    fig, axes = plt.subplots(2,2,figsize=(11.5,7.4),constrained_layout=True)
    quality = {}
    for ax,j,title in zip(axes.flat,range(4),["Fixed diabatic state 0","Fixed diabatic state 1","Local adiabatic lower state","Local adiabatic upper state"]):
        ax.plot(ref[:,0], ref[:,j+1], color="black",lw=1.8,label="Quantum wavepacket")
        quality[f"quantum_{j}"] = curve_quality(ref[:,0],ref[:,j+1])
        for g in GAMMAS:
            dia,adi=data[g,"diabatic"],data[g,"adiabatic"]
            lo,hi=dia[:,j+1]-1.96*dia[:,j+9],dia[:,j+1]+1.96*dia[:,j+9]
            ax.fill_between(dia[:,0],lo,hi,color=colors[g],alpha=.14)
            ax.plot(dia[:,0],dia[:,j+1],color=colors[g],lw=1.6,label=labels[g]+", diabatic EOM")
            ax.plot(adi[:,0],adi[:,j+1],color=colors[g],lw=1.2,ls="--",label=labels[g]+", adiabatic EOM")
            for name,y in (("dia",dia[:,j+1]),("adi",adi[:,j+1]),("lo",lo),("hi",hi)):
                quality[f"{g}_{j}_{name}"]=curve_quality(dia[:,0],y)
        ax.set(xlabel="Time (a.u.)",ylabel="Population",title=title,xlim=(0,800))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7.4)
    fig.suptitle("eCMM: the same preparation, two propagation bases")
    for suffix in ("png","svg"):
        fig.savefig(ROOT/f"cps-basis-populations.{suffix}",dpi=180)
    plt.close(fig)
    q=np.linspace(-5,5,4001)
    f,v=frame(q),MODEL.V(q)
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),constrained_layout=True)
    for name,arr,style in (("E lower",f["energy"][:,0],"-"),("E upper",f["energy"][:,1],"-"),
                           ("V00",v[:,0,0],"--"),("V11",v[:,1,1],"--")):
        axes[0].plot(q,arr,style,label=name)
        quality[name]=curve_quality(q,arr)
    axes[0].set(xlabel="R (Bohr)",ylabel="Energy (Hartree)",title="Tully simple avoided crossing")
    axes[0].legend(fontsize=8)
    axes[1].plot(q,f["nac"][:,0,1])
    axes[1].set(xlabel="R (Bohr)",ylabel="d01 (1/Bohr)",title="Analytic moving-basis coupling")
    quality["NAC"]=curve_quality(q,f["nac"][:,0,1])
    errors=[float(abs(data[g,"diabatic"][:,1:5]-data[g,"adiabatic"][:,1:5]).max()) for g in GAMMAS]
    axes[2].bar(["W kernel","gamma = 1/2"],errors,color=list(colors.values()),width=.5)
    axes[2].axhline(2e-4,color="black",ls=":",label="Declared tolerance")
    axes[2].set(ylabel="Max population difference",title="Same physical readout",yscale="log",ylim=(1e-8,1e-3))
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    for suffix in ("png","svg"):
        fig.savefig(ROOT/f"cps-basis-model-check.{suffix}",dpi=180)
    plt.close(fig)
    write_json(ROOT/"curve_quality.json",quality)
    # Explicit LF and no trailing whitespace in generated vector files.
    for path in ROOT.glob("*.svg"):
        text="\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n"
        path.write_bytes(text.encode("utf-8"))
    assert all(x["passed"] for x in quality.values()), "rendered curve gate failed"


def main():
    global ROOT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-samples",type=int,default=50000)
    ap.add_argument("--smoke",type=int,default=0)
    ap.add_argument("--validate",action="store_true")
    ap.add_argument("--plot-data",type=Path)
    args=ap.parse_args()
    ROOT=args.output.resolve()
    (ROOT/"results").mkdir(parents=True,exist_ok=True)
    if args.plot_data:
        plot_data(args.plot_data.resolve())
        return
    if not 2<=max(args.smoke,args.max_samples)<=50000:
        ap.error("sample count must lie between 2 and 50000")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path(__file__).with_name("cps_basis_dynamics.py"))}
    write_json(ROOT/"config.json",dict(max_samples=args.max_samples,smoke=args.smoke,
        validate=args.validate,seeds=[20261027,20261028,20261029,20261030],source_sha256=hashes))
    pool=initial_pool()
    ref=quantum_controls() if args.validate else quantum()
    with (ROOT/"quantum.csv").open("w",encoding="utf-8",newline="\n") as h:
        np.savetxt(h,np.column_stack((ref["time"],ref["obs"])),delimiter=",",
            header=",".join(["time"]+FEATURES),comments="")
    if args.validate: checks(pool)
    status={g:run_ensemble(pool,g,gamma,args.max_samples,args.smoke) for g,gamma in GAMMAS.items()}
    write_json(ROOT/"sampling_controls.json",status)
    if not args.smoke and not all(r["passed"] for r in status.values()):
        raise RuntimeError("resource cap reached before sampling stop criterion")
    plot_data(ROOT)


if __name__=="__main__":
    main()
