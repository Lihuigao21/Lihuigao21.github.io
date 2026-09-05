"""Part IV: a 2D equilibrium sampling benchmark with an analytic marginal.

Run from the repository root:
    python assets/code/reaction-dynamics/umbrella_free_energy.py
Requires NumPy, SciPy, and Matplotlib. Reduced units: beta = 1.
Sampling uses conditional Gaussian y updates and local Metropolis x updates.
Monte Carlo sweeps are NOT physical time. The analytic F is never supplied
to the sampler or to WHAM; it is used only for reference and diagnostics.
Only compact histograms, tables and plots are saved, not raw trajectories.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import cumulative_trapezoid, trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "assets/data/reaction-dynamics"
FIG = ROOT / "assets/img/reaction-dynamics"
CENTERS = np.linspace(-1.5, 1.5, 21)
SPRING = 60.
EDGES = np.linspace(-1.8, 1.8, 601)
REPLICAS, BURN, PRODUCTION, THIN = 12, 5000, 40000, 5


def stiffness(x):
    return np.exp(4*(1-x*x))


def energy(x, y):
    return 8*(x*x-1)**2 + .5*stiffness(x)*y*y


def reference(x):
    """Analytic F(x)-F(1) at beta=1, obtained by integrating out y."""
    return 8*(x*x-1)**2 + 2*(1-x*x)


def sample():
    # Axis 0: plain or umbrella; axis 1: independent full replicate;
    # axis 2: independent chain (plain) or window (umbrella).
    rng = np.random.default_rng(20260906)
    shape = (2, REPLICAS, len(CENTERS))
    x = np.full(shape, -1.)
    x[1] = CENTERS
    k = np.array([0., SPRING])[:, None, None]
    center = np.broadcast_to(CENTERS, shape)
    hist = np.zeros((*shape, len(EDGES)-1), dtype=np.int64)
    offsets = np.arange(np.prod(shape)).reshape(shape)*(len(EDGES)-1)
    traces, acceptance = [], np.zeros(shape)
    for step in range(BURN+PRODUCTION):
        y = rng.normal(size=shape)/np.sqrt(stiffness(x))
        proposed = x + rng.normal(scale=.14, size=shape)
        du = energy(proposed, y)-energy(x, y)
        du += .5*k*((proposed-center)**2-(x-center)**2)
        accept = np.log(rng.random(shape)) < -du
        x = np.where(accept, proposed, x)
        if step >= BURN:
            acceptance += accept
            if (step-BURN) % THIN == 0:
                bins = np.searchsorted(EDGES, x, side="right")-1
                if np.any((bins<0)|(bins>=len(EDGES)-1)):
                    raise RuntimeError("Histogram domain too small; do not silently discard samples.")
                np.add.at(hist.ravel(), (offsets+bins).ravel(), 1)
            if (step-BURN) % 100 == 0:
                traces.append(x[0, 0, :4].copy())
    return hist, np.array(traces), acceptance/PRODUCTION


def wham(hist, edges, tolerance=1e-10):
    """Binned WHAM, probabilities per bin, common beta=1.

    Equal-length windows are used; correlated sampling is handled for
    uncertainty by independent full replicates, not iid-bin error bars.
    """
    x = .5*(edges[:-1]+edges[1:])
    bias = .5*SPRING*(x[None, :]-CENTERS[:, None])**2
    counts = hist.sum(axis=0)
    n = hist.sum(axis=1)
    support = counts > 0
    logh = np.full(len(x), -np.inf)
    logh[support] = np.log(counts[support])
    f = np.zeros(len(n))
    for iteration in range(20000):
        logp = logh - logsumexp(np.log(n)[:, None]+f[:, None]-bias, axis=0)
        logp -= logsumexp(logp)
        updated = -logsumexp(logp[None, :]-bias, axis=1)
        delta = float(np.max(abs(updated-f)))
        f = updated
        if delta < tolerance:
            break
    else:
        raise RuntimeError("WHAM did not converge")
    return np.exp(logp), iteration+1, delta


def align_free_energy(p, x):
    f = np.full_like(p, np.nan)
    positive = p>0
    f[positive] = -np.log(p[positive]/(x[1]-x[0]))
    f -= np.interp(1., x[positive], f[positive])
    return f


def mfpt(diffusion=1., n=40001):
    """Separate 1D Smoluchowski model: reflecting -1.6, absorbing +1.

    This is not claimed to be the exact projected dynamics of the 2D sampler.
    Initial coordinate -1; tau = integral exp(F) integral exp(-F) / D.
    """
    x = np.linspace(-1.6, 1., n)
    f = reference(x)
    inner = cumulative_trapezoid(np.exp(-f), x, initial=0)
    integrand = np.exp(f)*inner
    # Integrate with the exact start point included, not a snapped grid index.
    xx = np.r_[-1., x[x>-1.]]
    yy = np.r_[np.interp(-1., x, integrand), integrand[x>-1.]]
    return float(trapezoid(yy, xx)/diffusion)


def noiseless_bin_check(edges):
    """Feed integrated analytic window probabilities to WHAM, diagnostics only."""
    subgrid = edges[:-1, None]+np.linspace(0.,1.,33)[None,:]*np.diff(edges)[:,None]
    counts = []
    for center in CENTERS:
        weights = np.exp(-reference(subgrid)-.5*SPRING*(subgrid-center)**2)
        masses = trapezoid(weights, subgrid, axis=1)
        counts.append(8000*masses/masses.sum())
    p, _, _ = wham(np.array(counts), edges)
    x = .5*(edges[:-1]+edges[1:])
    return float(np.interp(0., x, align_free_energy(p, x)))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def plot(x, umbrella, plain, hist, traces, mobile=False):
    fig, axes = plt.subplots(4 if mobile else 2, 1 if mobile else 2,
        figsize=(6.4, 15) if mobile else (12, 8), layout="constrained")
    ax = np.ravel(axes)
    ax[0].plot(x, 8*(x*x-1)**2, "--", c="#71859e", label="Minimum potential U(x,0)")
    ax[0].plot(x, reference(x), c="#253955", label="Analytic marginal free energy")
    ax[0].set(xlim=(-1.4,1.4), ylim=(-.5,14), xlabel="Coordinate x (reduced units)",
              ylabel="Energy / kBT, referenced at x=1", title="A. Entropy changes the landscape")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(np.arange(len(traces))*100, traces, lw=.7, alpha=.8)
    ax[1].set(xlabel="Production Monte Carlo sweeps (not time)", ylabel="Coordinate x",
              ylim=(-1.5,1.5), title="B. Four unbiased chains, initially left")
    dx = x[1]-x[0]
    for i in range(len(CENTERS)):
        density = hist[i]/hist[i].sum()/dx
        ax[2].plot(x, density, lw=.9, alpha=.8)
    ax[2].set(xlim=(-1.5,1.5), xlabel="Coordinate x", ylabel="Biased probability density",
              title="C. Overlapping umbrella windows")
    mean, spread = np.mean(umbrella, axis=0), np.std(umbrella, axis=0, ddof=1)
    ax[3].plot(x, reference(x), c="#253955", label="Analytic reference")
    ax[3].plot(x, mean, c="#bd614c", label="Umbrella + WHAM")
    ax[3].fill_between(x, mean-spread, mean+spread, color="#bd614c", alpha=.2,
                       label="One SD across 12 independent runs")
    ax[3].plot(x, plain, ":", c="#71859e", lw=1.3, label="Unbiased, pooled same-budget samples")
    ax[3].set(xlim=(-1.3,1.3), ylim=(-5,13), xlabel="Coordinate x",
              ylabel="[F(x)-F(1)] / kBT", title="D. Recover the unbiased landscape")
    ax[3].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(alpha=.18)
    fig.savefig(FIG / ("umbrella-free-energy-mobile.png" if mobile else "umbrella-free-energy.png"), dpi=170)
    plt.close(fig)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    finehist, traces, acceptance = sample()
    hist = finehist.reshape(2, REPLICAS, len(CENTERS), 300, 2).sum(axis=-1)
    edges = EDGES[::2]
    x = .5*(edges[:-1]+edges[1:])
    xf = .5*(EDGES[:-1]+EDGES[1:])
    rows, fs = [], []
    for j in range(REPLICAS):
        p, iterations, residual = wham(hist[1,j], edges)
        pf, _, _ = wham(finehist[1,j], EDGES)
        f, ff = align_free_energy(p, x), align_free_energy(pf, xf)
        fs.append(f)
        direct = hist[0,j].sum(axis=0)
        pd = direct/direct.sum()
        normalized_windows = hist[1,j]/hist[1,j].sum(axis=1)[:,None]
        overlap = np.sum(np.sqrt(normalized_windows[:-1]*normalized_windows[1:]), axis=1)
        rows.append(dict(replica=j, umbrella_right_probability=float(p[x>0].sum()),
            plain_right_probability=float(pd[x>0].sum()),
            umbrella_F0_minus_F1=float(np.interp(0., x, f)),
            finebin_F0_minus_F1=float(np.interp(0., xf, ff)),
            min_adjacent_overlap=float(overlap.min()), iterations=iterations,
            wham_residual=residual, samples_per_method=int(direct.sum()),
            plain_acceptance=float(acceptance[0,j].mean()),
            umbrella_acceptance=float(acceptance[1,j].mean())))
    fs = np.array(fs)
    pooled = hist[0].sum(axis=(0,1))
    directf = align_free_energy(pooled/pooled.sum(), x)
    write_csv(DATA / "umbrella-replicates.csv", rows)
    write_csv(DATA / "umbrella-profile.csv", [dict(x=float(xx), analytic_F=float(reference(xx)),
        wham_mean=float(np.mean(fs[:,i])), wham_run_sd=float(np.std(fs[:,i],ddof=1)),
        plain_pooled_F=float(directf[i])) for i,xx in enumerate(x)])
    write_csv(DATA / "umbrella-windows.csv", [dict(window=j, center=float(CENTERS[j]), x=float(xx),
        pooled_count=int(hist[1,:,j,i].sum())) for j in range(len(CENTERS)) for i,xx in enumerate(x)])
    write_csv(DATA / "free-energy-mfpt.csv", [dict(D=d, mfpt=mfpt(d), inverse_mfpt=1/mfpt(d),
        quadrature_change=abs(mfpt(d)-mfpt(d,n=80001))) for d in [1., .1]])
    write_csv(DATA / "umbrella-bin-check.csv", [dict(bin_width=float(e[1]-e[0]),
        noiseless_F0_minus_F1=noiseless_bin_check(e), analytic_F0_minus_F1=10.) for e in [edges, EDGES]])
    write_csv(DATA / "umbrella-plain-traces.csv", [dict(sweep=i*100,
        chain_0=float(t[0]), chain_1=float(t[1]), chain_2=float(t[2]), chain_3=float(t[3]))
        for i,t in enumerate(traces)])
    plot(x, fs, directf, hist[1].sum(axis=0), traces)
    plot(x, fs, directf, hist[1].sum(axis=0), traces, mobile=True)
    for key in ["umbrella_right_probability", "plain_right_probability", "umbrella_F0_minus_F1"]:
        values = np.array([r[key] for r in rows])
        print(f"{key}: mean={values.mean():.6f}, run SD={values.std(ddof=1):.6f}")
    print("Minimum adjacent histogram overlap:", min(r["min_adjacent_overlap"] for r in rows))
    print("MFPT at D=1:", mfpt())


if __name__ == "__main__":
    main()
