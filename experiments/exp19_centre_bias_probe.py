"""
Centre-bias probe (R1, Proposition 1). For each FO algorithm and alpha in {0.3, 0.5, 0.7, 0.9, 1.0}, 30 runs
(seed = 42 + run), n = 10, 500 iterations, box [-100,100]^n, on the SHIFTED Sphere and Rastrigin of exp18
(same shift o, drawn from the exp18 problem generator). We record the final best position p and report the
projection ratio  lambda = <p, o> / <o, o>  (lambda = 1: optimum found; lambda = 0: stuck at the origin),
together with tau(alpha, M) and kappa(alpha, M) of Proposition 1 for M = 10.
NOTE: Proposition 1 gives the per-step pull of the memory term on a frozen population/archive. Selection and
archive updates act against it, so lambda is NOT predicted to equal tau or kappa; the probe tests the qualitative
predictions: lambda < 1 for alpha < 1 (JAYA/WFO/ACOR), lambda increasing with alpha, lambda = 1 at alpha = 1,
and no bias for FO-PSO (memory on velocities, translation-equivariant).
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp19_centre_bias_probe.py
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from exp18_r1_benchmark import Objective
from src.algorithms.fo_pso import FO_PSO
from src.algorithms.fo_jaya import FO_JAYA
from src.algorithms.fo_wfo import FO_WFO
from src.algorithms.fo_acor import FO_ACOR

ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]; FUNCS = ['sphere', 'rastrigin']; ALGS = ['FO-PSO', 'FO-JAYA', 'FO-WFO', 'FO-ACOR']
DIM, RUNS, ITERS, B = 10, 30, 500, (-100, 100)


def make(name, alpha, seed):
    return {'FO-PSO': lambda: FO_PSO(n_particles=30, dim=DIM, w=0.7, c1=1.5, c2=1.5, alpha=alpha, bounds=B, seed=seed),
            'FO-JAYA': lambda: FO_JAYA(n_population=30, dim=DIM, alpha=alpha, memory_size=10, bounds=B, seed=seed),
            'FO-WFO': lambda: FO_WFO(n_population=30, dim=DIM, alpha=alpha, memory_size=10, pl=0.25, pe=0.2, bounds=B, seed=seed),
            'FO-ACOR': lambda: FO_ACOR(n_ants=30, dim=DIM, alpha=alpha, archive_size=50, q=0.01, xi=0.85, memory_size=10, bounds=B, seed=seed)}[name]()


def job(a):
    f, g, al, r = a
    obj = Objective('shift', f, DIM)
    p, best, _ = make(g, al, 42 + r).optimize(obj, max_iter=ITERS)
    o = obj.o
    return f, g, al, r, float(np.dot(p, o) / np.dot(o, o)), float(np.linalg.norm(p - o)), float(best)


def tau_kappa(alpha, M=10):
    S = np.prod([1 - alpha / i for i in range(1, M)])
    tau = alpha + (1 - alpha) * S / M
    kappa = alpha / (1 - (1 - alpha) * S / M)
    return tau, kappa


def main():
    jobs = [(f, g, al, r) for f in FUNCS for g in ALGS for al in ALPHAS for r in range(RUNS)]
    res = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        for f, g, al, r, lam, dist, best in ex.map(job, jobs, chunksize=4):
            res.setdefault((f, g, al), []).append((r, lam, dist, best))
    out = os.path.join(ROOT, 'results', 'data', 'exp19_centre_bias_probe.json')
    rows = []
    print(f'{"function":10s} {"algorithm":8s} {"alpha":>5s} {"tau":>6s} {"kappa":>6s} {"lambda median [IQR]":>26s} {"|p-o| median":>12s}')
    for f in FUNCS:
        for g in ALGS:
            for al in ALPHAS:
                v = sorted(res[(f, g, al)]); lam = np.array([x[1] for x in v]); dist = np.array([x[2] for x in v])
                t, k = tau_kappa(al); q1, m, q3 = np.percentile(lam, [25, 50, 75])
                rows.append(dict(function=f, algorithm=g, alpha=al, tau=t, kappa=k, lam=lam.tolist(), dist=dist.tolist(),
                                 best=[x[3] for x in v]))
                print(f'{f:10s} {g:8s} {al:5.1f} {t:6.4f} {k:6.4f} {m:10.4f} [{q1:.4f}, {q3:.4f}] {np.median(dist):12.3g}')
    json.dump(dict(protocol=dict(dim=DIM, runs=RUNS, iters=ITERS, bounds=B, suite='shift (exp18 instance)'), rows=rows),
              open(out, 'w'))
    print('saved', out)


if __name__ == '__main__':
    main()
