"""
R1 alpha sweep (Section 5.3, Table 14; replaces the submitted sweep, which used mixed bounds and code v1).
FO-PSO, FO-JAYA, FO-WFO, FO-ACOR at alpha in {0.5, 0.7, 0.9, 1.0} (alpha = 1.0 is the baseline algorithm, bit-identical),
suites canonical / shift / shiftrot of exp18, n = 10, 30 runs (seed = 42 + run), 500 iterations, box [-100,100]^n,
Table 4 parameters. Output: median final f and success rate (f < 1e-6) per cell, and the best alpha per
(suite, function, algorithm) by median with ties (equal medians) listed.
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp22_alpha_sweep.py
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from exp18_r1_benchmark import Objective, FUNCS
from src.algorithms.fo_pso import FO_PSO
from src.algorithms.fo_jaya import FO_JAYA
from src.algorithms.fo_wfo import FO_WFO
from src.algorithms.fo_acor import FO_ACOR

ALPHAS = [0.5, 0.7, 0.9, 1.0]; ALGS = ['FO-PSO', 'FO-JAYA', 'FO-WFO', 'FO-ACOR']; SUITES = ['canonical', 'shift', 'shiftrot']
DIM, RUNS, ITERS, B = 10, 30, 500, (-100, 100)


def make(name, alpha, seed):
    return {'FO-PSO': lambda: FO_PSO(n_particles=30, dim=DIM, w=0.7, c1=1.5, c2=1.5, alpha=alpha, bounds=B, seed=seed),
            'FO-JAYA': lambda: FO_JAYA(n_population=30, dim=DIM, alpha=alpha, memory_size=10, bounds=B, seed=seed),
            'FO-WFO': lambda: FO_WFO(n_population=30, dim=DIM, alpha=alpha, memory_size=10, pl=0.25, pe=0.2, bounds=B, seed=seed),
            'FO-ACOR': lambda: FO_ACOR(n_ants=30, dim=DIM, alpha=alpha, archive_size=50, q=0.01, xi=0.85, memory_size=10, bounds=B, seed=seed)}[name]()


def job(a):
    s, f, g, al, r = a
    _, best, _ = make(g, al, 42 + r).optimize(Objective(s, f, DIM), max_iter=ITERS)
    return a, float(best)


def main():
    jobs = [(s, f, g, al, r) for s in SUITES for f in FUNCS for g in ALGS for al in ALPHAS for r in range(RUNS)]
    res = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        for (s, f, g, al, r), b in ex.map(job, jobs, chunksize=8):
            res.setdefault(f'{s}|{f}|{g}|{al}', np.full(RUNS, np.nan))[r] = b
    for s in SUITES:
        print(f'\n## suite={s}  n={DIM}: median (success %) for alpha = ' + ' / '.join(map(str, ALPHAS)) + '  -> best alpha by median')
        for f in FUNCS:
            for g in ALGS:
                meds = [np.median(res[f'{s}|{f}|{g}|{al}']) for al in ALPHAS]
                sr = [100 * np.mean(res[f'{s}|{f}|{g}|{al}'] < 1e-6) for al in ALPHAS]
                mfloor = [max(m, 1e-10) for m in meds]
                best = [al for al, m in zip(ALPHAS, mfloor) if m == min(mfloor)]
                print(f'  {f:10s} {g:8s} ' + ' / '.join(f'{m:.3g} ({p:.0f}%)' for m, p in zip(meds, sr)) + f'  -> {best}')
    json.dump({k: v.tolist() for k, v in res.items()}, open(os.path.join(ROOT, 'results', 'data', 'exp22_alpha_sweep.json'), 'w'))


if __name__ == '__main__':
    main()
