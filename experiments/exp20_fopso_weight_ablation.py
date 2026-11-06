"""
FO-PSO velocity-memory weight ablation (R1, reviewer R3-2.1).
Variants (src/algorithms/fo_pso.py, memory_weights=...):
  uniform : published FO-PSO, alpha*v_t + (1-alpha)/(L-1) sum_{j>=1} v_{t-j}
  gl_def6 : Definition 6 applied to the velocity history, alpha*v_t + (1-alpha)/L sum_{j>=0} omega_j v_{t-j}
  gl_abs  : |omega_j| weights normalised to total mass (1-alpha) over the past velocities
alpha in {0.5, 0.7, 0.9}; PSO (alpha = 1) as reference. Suites canonical and shiftrot (exp18 instances), n in {10, 30},
30 runs, seed = 42 + run, 500 iterations, Table 4 parameters.
Theory column: spectral radius of the expected extended-state map (e_t, v_t, ..., v_{t-8}) for each variant, full window,
built exactly as in Theorem 2(iv) (r1, r2 replaced by 1/2, attractor fixed).
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp20_fopso_weight_ablation.py
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from exp18_r1_benchmark import Objective, FUNCS
from src.algorithms.fo_pso import FO_PSO
from src.algorithms.pso import PSO

VARIANTS = ['uniform', 'gl_def6', 'gl_abs']; ALPHAS = [0.5, 0.7, 0.9]; SUITES = ['canonical', 'shiftrot']; DIMS = [10, 30]
RUNS, ITERS, B, FLOOR = 30, 500, (-100, 100), 1e-10
W, C = 0.7, 1.5; M = 10


def omega(alpha, n=M):
    w = [1.0]
    for k in range(1, n):
        w.append(w[-1] * (k - 1 - alpha) / k)
    return np.array(w)


def mean_map_radius(variant, alpha):
    """Expected extended map, state (e, v_{t-1}, ..., v_{t-M}) before the step; v_t = w v_{t-1} - phi e."""
    phi = C  # (c1 + c2)/2 with E[r] = 1/2
    d = 1 + M
    vnew = np.zeros(d); vnew[0] = -phi; vnew[1] = W
    past = np.zeros((M, d)); past[0] = vnew                      # v_{t-j}: j = 0 is the new velocity
    for j in range(1, M):
        past[j, 1 + j - 1] = 1.0                                 # v_{t-j} = stored state index j
    if variant == 'uniform':
        coef = np.r_[alpha, np.full(M - 1, (1 - alpha) / (M - 1))]
    elif variant == 'gl_def6':
        coef = (1 - alpha) * omega(alpha) / M; coef[0] += alpha
    else:
        a = np.abs(omega(alpha))[1:]; coef = np.r_[alpha, (1 - alpha) * a / a.sum()]
    xnew = np.zeros(d); xnew[0] = 1.0; xnew += coef @ past
    T = np.zeros((d, d)); T[0] = xnew; T[1] = vnew
    for j in range(2, d):
        T[j, j - 1] = 1.0
    return float(max(abs(np.linalg.eigvals(T))))


def job(a):
    s, d, f, var, al, r = a
    obj = Objective(s, f, d)
    if var == 'PSO':
        alg = PSO(n_particles=30, dim=d, w=W, c1=C, c2=C, bounds=B, seed=42 + r)
    else:
        alg = FO_PSO(n_particles=30, dim=d, w=W, c1=C, c2=C, alpha=al, bounds=B, seed=42 + r, memory_weights=var)
    _, best, _ = alg.optimize(obj, max_iter=ITERS)
    return a, float(best)


def main():
    jobs = [(s, d, f, v, al, r) for s in SUITES for d in DIMS for f in FUNCS for v in VARIANTS for al in ALPHAS for r in range(RUNS)]
    jobs += [(s, d, f, 'PSO', 1.0, r) for s in SUITES for d in DIMS for f in FUNCS for r in range(RUNS)]
    res = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        for (s, d, f, v, al, r), b in ex.map(job, jobs, chunksize=8):
            res.setdefault(f'{s}|{d}|{f}|{v}|{al}', np.full(RUNS, np.nan))[r] = b
    print('Theory: spectral radius of the expected extended map (full window)')
    for v in VARIANTS:
        print(f'  {v:8s} ' + '  '.join(f'alpha={al}: {mean_map_radius(v, al):.4f}' for al in ALPHAS))
    print(f'  PSO (alpha=1): {mean_map_radius("uniform", 1.0):.4f}')
    print('\nmedian final f (Wilcoxon vs uniform at the same alpha, two-sided, floor 1e-10; Holm within suite x n over 30 tests)')
    out = {}
    for s in SUITES:
        for d in DIMS:
            tests = []
            for f in FUNCS:
                for al in ALPHAS:
                    u = np.maximum(res[f'{s}|{d}|{f}|uniform|{al}'], FLOOR)
                    for v in ['gl_def6', 'gl_abs']:
                        x = np.maximum(res[f'{s}|{d}|{f}|{v}|{al}'], FLOOR)
                        p = 1.0 if np.all(x == u) else stats.wilcoxon(x, u, zero_method='zsplit').pvalue
                        tests.append((f, al, v, p, np.median(x), np.median(u)))
            ps = np.array([t[3] for t in tests]); order = np.argsort(ps); adj = np.empty_like(ps); run = 0
            for i, k in enumerate(order):
                run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(1, run)
            print(f'\n## suite={s} n={d}')
            for f in FUNCS:
                pso = np.median(res[f'{s}|{d}|{f}|PSO|1.0'])
                line = f'  {f:10s} PSO {pso:.3g} | ' + ' | '.join(
                    f'a={al}: ' + ', '.join(f'{v} {np.median(res[f"{s}|{d}|{f}|{v}|{al}"]):.3g}' for v in VARIANTS) for al in ALPHAS)
                print(line)
            sig = [(t, a_) for t, a_ in zip(tests, adj) if a_ < 0.05]
            print(f'  Holm-significant differences vs uniform: {len(sig)}/{len(tests)}')
            for (f, al, v, p, mx, mu), a_ in sig:
                print(f'    {f:10s} alpha={al} {v:8s} median {mx:.3g} vs uniform {mu:.3g}  p_holm={a_:.2g}  ({"better" if mx < mu else "worse"})')
    json.dump({k: v.tolist() for k, v in res.items()}, open(os.path.join(ROOT, 'results', 'data', 'exp20_fopso_weight_ablation.json'), 'w'))


if __name__ == '__main__':
    main()
