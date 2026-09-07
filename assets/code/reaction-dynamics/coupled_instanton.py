"""Coupled 2D instanton: explicit damped Newton, convergence, stability prefactor.

Units hbar=mx=my=kB=1. V=2 sech(x)^2+(y-c exp(-x*x))^2/2.
The coupled orbit is solved numerically, not from a 1D action formula.
No coupled exact quantum calculation or DVR is claimed here.
Run with NumPy, SciPy and Matplotlib; --output selects the results directory.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
TAU = 2*np.pi
C = .6


def potential(q, c=C):
    q = np.asarray(q)
    x, y = q[..., 0], q[..., 1]
    g = c*np.exp(-x*x)
    gp = -2*x*g
    gpp = (4*x*x-2)*g
    s = 1/np.cosh(x)**2
    d = y-g
    v = 2*s+d*d/2
    grad = np.stack((-4*np.tanh(x)*s-d*gp, d), axis=-1)
    hess = np.empty(np.shape(x)+(2, 2))
    hess[..., 0, 0] = 8*s-12*s*s+gp*gp-d*gpp
    hess[..., 0, 1] = hess[..., 1, 0] = -gp
    hess[..., 1, 1] = 1
    return v, grad, hess


def resample(q, n):
    """Interpolate the old periodic orbit, without imposing a spatial curve."""
    return CubicSpline(np.linspace(0, 1, len(q)+1), np.vstack((q, q[0])),
                       bc_type='periodic')(np.arange(n)/n)


def orbit(n, tau=TAU, c=C, initial=None, amplitude=1.15, yshift=.1):
    dt = tau/n
    lap = 2*np.eye(n)-np.roll(np.eye(n), 1, 0)-np.roll(np.eye(n), -1, 0)
    idx = np.minimum(np.arange(n), n-np.arange(n))
    p = n//2+1
    P = np.eye(p)[idx]
    L = P.T@lap@P/dt
    if initial is None:
        phase = 2*np.pi*np.arange(n)/n
        q = np.column_stack((amplitude*np.cos(phase), np.full(n, yshift)))
    else:
        q = resample(initial, n)
    z = np.r_[q[:p, 0], q[:p, 1]]

    def evaluate(z, need_hessian=True):
        q = np.column_stack((P@z[:p], P@z[p:]))
        v, vg, vh = potential(q, c)
        G = (2*q-np.roll(q, 1, 0)-np.roll(q, -1, 0))/dt+dt*vg
        gr = np.r_[P.T@G[:, 0], P.T@G[:, 1]]
        dq = np.roll(q, -1, 0)-q
        action = np.sum(dq*dq)/(2*dt)+dt*np.sum(v)
        H = None
        if need_hessian:
            xx = L+dt*(P.T@(vh[:, 0, 0, None]*P))
            xy = dt*(P.T@(vh[:, 0, 1, None]*P))
            yy = L+dt*(P.T@(vh[:, 1, 1, None]*P))
            H = np.block([[xx, xy], [xy, yy]])
        return q, action, G, gr, H

    history, snapshots = [], []
    for iteration in range(60):
        q, action, G, gr, H = evaluate(z)
        residual = float(np.max(np.abs(G)))
        entry = dict(iteration=iteration, action=float(action), residual=residual,
                     endpoint=q[0].tolist(), middle=q[n//4].tolist())
        snapshots.append(q.tolist())
        if residual < 2e-11:
            entry.update(alpha=0., max_step=0.)
            history.append(entry)
            break
        # This is a saddle-point root solve. Never replace negative eigenvalues
        # with their absolute values, and never require the action to decrease.
        delta = np.linalg.solve(H, -gr)
        alpha = min(1., .35/max(float(np.max(np.abs(delta))), 1e-300))
        merit = .5*np.dot(gr, gr)
        for backtrack in range(30):
            trial = evaluate(z+alpha*delta, False)
            if .5*np.dot(trial[3], trial[3]) <= (1-1e-4*alpha)*merit:
                break
            alpha *= .5
        else:
            raise RuntimeError(('line search failed', n, iteration, residual))
        entry.update(alpha=float(alpha), max_step=float(np.max(np.abs(alpha*delta))))
        history.append(entry)
        z += alpha*delta
    else:
        raise RuntimeError(('Newton did not converge', n, history[-1]))
    if np.ptp(q[:, 0]) < 1:
        raise RuntimeError('Converged to collapsed/nonreactive solution')
    return dict(n=n, tau=tau, c=c, q=q, action=float(action), residual=residual,
                history=history, snapshots=snapshots)


def stability(o, rtol=2e-10):
    """Linearize BOTH coordinates around the orbit; not just local V_yy."""
    q, tau = o['q'], o['tau']
    spline = CubicSpline(np.linspace(0, tau, len(q)+1),
                         np.vstack((q, q[0])), bc_type='periodic')

    def rhs(t, flat):
        h = potential(spline(t), o['c'])[2]
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [h, np.zeros((2, 2))]])
        return (A@flat.reshape(4, 4)).ravel()
    sol = solve_ivp(rhs, (0, tau), np.eye(4).ravel(), method='DOP853',
                    rtol=rtol, atol=rtol*.01, max_step=tau/100)
    assert sol.success
    M = sol.y[:, -1].reshape(4, 4)
    eig = np.linalg.eigvals(M)
    lead = eig[np.argmax(np.abs(eig))]
    assert abs(lead.imag) < 1e-7 and lead.real > 1
    u = float(np.log(lead.real))
    J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    t = np.linspace(0, tau, 2001)
    energy = potential(spline(t), o['c'])[0]-.5*np.sum(spline(t, 1)**2, axis=1)
    return dict(u=u, z=1/(2*np.sinh(u/2)), matrix=M.tolist(),
                eigenvalues=[[float(e.real), float(e.imag)] for e in eig],
                determinant=float(np.linalg.det(M)),
                symplectic_error=float(np.max(np.abs(M.T@J@M-J))),
                energy_mean=float(np.mean(energy)), energy_range=float(np.ptp(energy)))


def rate(o, fraction=.002):
    h = fraction*o['tau']
    sm2, sm1, sp1, sp2 = [orbit(o['n'], o['tau']+i*h, o['c'], o['q'])['action']
                         for i in [-2, -1, 1, 2]]
    d1 = (sm2-8*sm1+8*sp1-sp2)/(12*h)
    d2 = (-sp2+16*sp1-30*o['action']+16*sm1-sm2)/(12*h*h)
    assert d2 < 0
    stab = stability(o)
    flux = np.sqrt(-d2/(2*np.pi))*stab['z']*np.exp(-o['action'])
    qr = 1/(2*np.sinh(o['tau']/2))/np.sqrt(2*np.pi*o['tau'])
    return dict(n=o['n'], action=o['action'], energy=d1, minus_S_second=-d2,
                longitudinal_factor=float(np.sqrt(-d2/(2*np.pi))),
                exponential=float(np.exp(-o['action'])), stability=stab,
                flux=float(flux), qr_per_length=float(qr),
                rate_per_density=float(flux/qr), residual=o['residual'],
                endpoint=o['q'][0].tolist(), middle=o['q'][o['n']//4].tolist())


def derivative_check():
    q = np.array([.7, .2]); h = 1e-5
    v, g, H = potential(q)
    gn = np.array([(potential(q+h*e)[0]-potential(q-h*e)[0])/(2*h)
                   for e in np.eye(2)])
    Hn = np.column_stack([(potential(q+h*e)[1]-potential(q-h*e)[1])/(2*h)
                          for e in np.eye(2)])
    return dict(gradient_error=float(np.max(np.abs(g-gn))),
                hessian_error=float(np.max(np.abs(H-Hn))))


def shooting_check(o):
    """Independent continuum boundary-value check, NOT a quantum benchmark.

    Start at a turning point with v=0; solve two initial coordinates so that
    both velocities vanish again at tau/2. The bead path seeds this small solve.
    This check also distinguishes an exact ODE orbit from a spline through beads.
    """
    def continuous(tau):
        def rhs(t, a):
            q, v = a[:2], a[2:4]
            pot, grad, hess = potential(q, o['c'])
            return np.r_[v, grad, .5*np.dot(v, v)+pot]

        def integrate(q0, end, dense=False):
            sol = solve_ivp(rhs, (0, end), np.r_[q0, 0., 0., 0.],
                            method='DOP853', rtol=2e-12, atol=2e-14,
                            max_step=tau/150, dense_output=dense)
            assert sol.success
            return sol
        sol = root(lambda q0: integrate(q0, tau/2).y[2:4, -1],
                   o['q'][0], tol=1e-10)
        half = integrate(sol.x, tau/2)
        assert np.max(np.abs(half.y[2:4, -1])) < 1e-9
        full = integrate(sol.x, tau, True)
        err = float(np.max(np.abs(full.y[:4, -1]-full.y[:4, 0])))
        assert err < 1e-8
        return full, float(2*half.y[4, -1]), float(potential(sol.x, o['c'])[0]), err

    full, action, energy, closure = continuous(o['tau'])
    h = .001*o['tau']
    en = [continuous(o['tau']+j*h)[2] for j in [-2, -1, 1, 2]]
    d2 = (en[0]-8*en[1]+8*en[2]-en[3])/(12*h)
    def linear_rhs(t, flat):
        H = potential(full.sol(t)[:2], o['c'])[2]
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [H, np.zeros((2, 2))]])
        return (A@flat.reshape(4, 4)).ravel()
    ms = solve_ivp(linear_rhs, (0, o['tau']), np.eye(4).ravel(), method='DOP853',
                   rtol=2e-12, atol=2e-14, max_step=o['tau']/150)
    eig = np.linalg.eigvals(ms.y[:, -1].reshape(4, 4))
    u = float(np.log(max(eig.real)))
    z = float(1/(2*np.sinh(u/2)))
    t = np.linspace(0, o['tau'], 1001)
    a = full.sol(t)
    e = potential(a[:2].T, o['c'])[0]-.5*np.sum(a[2:4]**2, axis=0)
    return dict(action=action, energy=energy, minus_S_second=float(-d2), u=u, z=z,
                flux=float(np.sqrt(-d2/(2*np.pi))*z*np.exp(-action)),
                periodic_closure_error=closure, energy_range=float(np.ptp(e)),
                endpoint=a[:2, 0].tolist(), middle=full.sol(o['tau']/4)[:2].tolist(),
                monodromy_eigenvalues=[[float(v.real), float(v.imag)] for v in eig])


def draw(o, first):
    plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    x = np.linspace(-2.3, 2.3, 180); y = np.linspace(-.45, 1.25, 130)
    X, Y = np.meshgrid(x, y)
    V = potential(np.stack((X, Y), -1))[0]
    q = o['q'][:o['n']//2+1]
    fig = plt.figure(figsize=(13, 5.1), layout='constrained')
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, V, cmap='viridis', alpha=.65, rcount=60, ccount=80,
                    linewidth=0, antialiased=True)
    ax.plot(q[:, 0], q[:, 1], potential(q)[0]+.035, color='#d63e2a', lw=4)
    ax.plot(x, C*np.exp(-x*x), 2/np.cosh(x)**2+.015, '--', color='#13263d', lw=2)
    ax.scatter([0], [C], [2.04], c='black', s=30)
    ax.set(xlabel='x', ylabel='y', zlabel='V(x,y)', title='Coupled potential and instanton')
    ax.view_init(elev=32, azim=-63)
    ax = fig.add_subplot(122)
    im = ax.contourf(X, Y, V, levels=25, cmap='viridis', alpha=.65)
    ax.contour(X, Y, V, levels=[.5, 1, 1.5, 2], colors='white', linewidths=.6)
    ax.plot(x, C*np.exp(-x*x), '--', color='#13263d', label='Minimum of V at fixed x')
    init = np.array(first['snapshots'][0]); ax.plot(init[:, 0], init[:, 1], ':', color='#7f3c8d',
                                                 lw=2, label='Initial guess')
    ax.plot(q[:, 0], q[:, 1], color='#d63e2a', lw=3, label='Solved instanton (out and back)')
    ax.scatter(q[::16, 0], q[::16, 1], c='#d63e2a', s=16)
    ax.scatter([0], [C], c='black', marker='x', s=70, label='Potential saddle')
    ax.scatter(q[[0, -1], 0], q[[0, -1], 1], c='#d63e2a', marker='s', s=40)
    ax.annotate('The path leaves the curved valley', xy=q[len(q)//2], xytext=(-1.9, .91),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    ax.set(xlabel='x', ylabel='y', title='Top view: solve the path, do not prescribe it',
           xlim=(-2.1, 2.1), ylim=(-.2, 1.1))
    ax.legend(loc='lower center', fontsize=8)
    fig.savefig(OUT/'potential-and-path.png', dpi=180)
    fig.set_layout_engine(None); fig.set_size_inches(7, 10)
    fig.axes[0].set_position([.12, .55, .82, .4])
    fig.axes[1].set_position([.12, .08, .82, .38])
    fig.axes[1].set_title('Top view: the solved path')
    fig.savefig(OUT/'potential-and-path-mobile.png', dpi=180); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    axs[0].contour(X, Y, V, levels=12, colors='#aaaaaa', linewidths=.6)
    for i in sorted(set([0, 1, 2, len(first['snapshots'])-1])):
        qi = np.array(first['snapshots'][i])[:first['n']//2+1]
        axs[0].plot(qi[:, 0], qi[:, 1], '.-', label=f'Newton iteration {i}')
    axs[0].plot(x, C*np.exp(-x*x), 'k--', alpha=.5, label='Fixed-x valley')
    axs[0].set(xlabel='x', ylabel='y', xlim=(-1.6, 1.6), ylim=(-.03, .8),
               title='Actual node updates (N=32)')
    axs[0].legend(fontsize=9)
    hist = first['history']
    axs[1].semilogy([r['iteration'] for r in hist], [r['residual'] for r in hist], 'o-')
    axs[1].set(xlabel='Newton iteration', ylabel='max |action gradient|',
               title='Convergence of the stationarity equations')
    axs[1].grid(alpha=.2)
    fig.savefig(OUT/'newton-iterations.png', dpi=180)
    fig.set_layout_engine(None); fig.set_size_inches(7, 9)
    axs[0].set_position([.15, .57, .8, .36])
    axs[1].set_position([.15, .09, .8, .36])
    fig.savefig(OUT/'newton-iterations-mobile.png', dpi=180); plt.close(fig)


def main():
    tests = derivative_check()
    assert max(tests.values()) < 1e-7
    orbits, rates = [], []
    for n in [32, 64, 128, 256, 512]:
        o = orbit(n, initial=orbits[-1]['q'] if orbits else None)
        r = rate(o); orbits.append(o); rates.append(r)
        print('CONVERGENCE', {k:v for k,v in r.items() if k!='stability'}, flush=True)
    final = orbits[-1]
    half = rate(final, .001)
    tests['derivative_step_flux_relative_change'] = abs(half['flux']/rates[-1]['flux']-1)
    tighter = stability(final, 1e-11)
    tests['ode_tolerance_z_relative_change'] = abs(tighter['z']/rates[-1]['stability']['z']-1)
    seeds = []
    for a, y0 in [(1., 0.), (1.4, .5), (1.2, -.1)]:
        oo = orbit(64, amplitude=a, yshift=y0)
        seeds.append(dict(amplitude=a, yshift=y0, action=oo['action'],
                          iterations=len(oo['history'])-1, residual=oo['residual']))
    n = final['n']; dt = TAU/n
    lap = 2*np.eye(n)-np.roll(np.eye(n), 1, 0)-np.roll(np.eye(n), -1, 0)
    h = potential(final['q'])[2]
    H = np.block([[lap/dt+dt*np.diag(h[:, 0, 0]), dt*np.diag(h[:, 0, 1])],
                  [dt*np.diag(h[:, 1, 0]), lap/dt+dt*np.diag(h[:, 1, 1])]])
    ev = np.linalg.eigvalsh(H)
    modes = dict(negative=int(np.sum(ev < -1e-7)), near_zero=int(np.sum(np.abs(ev)<1e-7)),
                 smallest=ev[:8].tolist())
    assert modes['negative']==1 and modes['near_zero']==1, modes
    # Independent analytic LIMIT test, not an analytic solution of c=0.6.
    sep = rate(orbit(512, c=0.))
    expected = np.exp(-3*np.pi)/(2*np.pi)/(2*np.sinh(np.pi))
    tests['separable_limit_flux_relative_error'] = abs(sep['flux']/expected-1)
    tests['separable_limit_u_relative_error'] = abs(sep['stability']['u']/TAU-1)
    tests['512_vs_256_flux_relative_change'] = abs(rates[-1]['flux']/rates[-2]['flux']-1)
    continuum = shooting_check(final)
    tests['512_vs_continuum_flux_relative_error'] = abs(rates[-1]['flux']/continuum['flux']-1)
    assert tests['512_vs_256_flux_relative_change'] < .001
    assert tests['512_vs_continuum_flux_relative_error'] < .0001
    assert all(abs(seed['action']-rates[1]['action']) < 1e-8 for seed in seeds)
    assert rates[-1]['stability']['symplectic_error'] < 1e-7
    assert tests['separable_limit_flux_relative_error'] < .0002
    result = dict(model='2 sech(x)^2 + (y - 0.6 exp(-x^2))^2/2',
                  units='hbar=mx=my=kB=1', tau=TAU, T=1/TAU, Tc=1/np.pi,
                  rates=rates, tests=tests, continuum_check=continuum,
                  hessian_modes=modes, initial_guesses=seeds,
                  newton_history_32=orbits[0]['history'],
                  newton_snapshots_32=orbits[0]['snapshots'],
                  final_path=final['q'].tolist(),
                  tst_flux=float(np.exp(-2*TAU)/(2*np.pi*TAU)/(2*np.sinh(TAU/2))),
                  boundary='No coupled quantum/DVR reference; convergence is not theory accuracy.')
    (OUT/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    draw(final, orbits[0])
    print('TESTS', tests, flush=True)
    print('MODES', modes, flush=True)
    print('HISTORY', orbits[0]['history'], flush=True)
    print('STABILITY', rates[-1]['stability'], flush=True)
    print('CONTINUUM', continuum, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('coupled-instanton-output'))
    OUT = parser.parse_args().output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    main()
