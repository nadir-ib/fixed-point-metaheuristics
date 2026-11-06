"""
R1 additional checks quoted in the manuscript (Sections 4.1.2, 5.2.7, 5.2.10).
A. FO-PSO weight ablation summary from results/data/exp20_fopso_weight_ablation.json (Holm over the 30 tests of each
   suite x n, floor 1e-10, two-sided Wilcoxon, zsplit), per weighting scheme.
B. Friedman test with the five functions as blocks (median over 30 runs per function and algorithm, floored at 1e-10),
   for every suite and dimension of exp18.
C. WFO on Rastrigin (n = 10, 10 runs, seeds 42..51): shift with equal components (-20,...,-20) versus a random
   direction of the same norm (np.random.default_rng(7)); success = f < 1e-6.
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp24_r1_extra_checks.py > results/exp24_r1_extra_checks.txt
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import stats
sys.path.insert(0, '.')
FUNCS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); a = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); a[k] = min(1, run)
    return a


print('A. FO-PSO velocity-memory weight ablation (vs uniform weights at the same alpha)')
R = json.load(open('results/data/exp20_fopso_weight_ablation.json'))
tot = {'gl_def6': [0, 0, []], 'gl_abs': [0, 0, []]}; unstable = [0, 0, []]
for s in ['canonical', 'shiftrot']:
    for d in [10, 30]:
        tests = []
        for f in FUNCS:
            for al in [0.5, 0.7, 0.9]:
                u = np.maximum(R[f'{s}|{d}|{f}|uniform|{al}'], 1e-10)
                for v in ['gl_def6', 'gl_abs']:
                    x = np.maximum(R[f'{s}|{d}|{f}|{v}|{al}'], 1e-10)
                    p = 1.0 if np.all(x == u) else stats.wilcoxon(x, u, zero_method='zsplit').pvalue
                    tests.append((f, al, v, p, np.median(x) < np.median(u)))
        a = holm([t[3] for t in tests])
        for (f, al, v, p, better), q in zip(tests, a):
            if q < 0.05:
                tot[v][1 if better else 0] += 1
                if better: tot[v][2].append((s, d, f, al))
            if v == 'gl_abs' and al in (0.5, 0.7):
                unstable[0] += 1
                if q < 0.05 and not better: unstable[1] += 1
                else: unstable[2].append((s, d, f, al, 'better' if q < 0.05 else 'ns'))
        print(f'  {s:9s} n={d}: significant gl_def6 {sum(1 for t, q in zip(tests, a) if t[2] == "gl_def6" and q < 0.05)}/15, '
              f'gl_abs {sum(1 for t, q in zip(tests, a) if t[2] == "gl_abs" and q < 0.05)}/15')
for v in tot:
    print(f'  {v}: significantly worse {tot[v][0]}, significantly better {tot[v][1]} {tot[v][2]}')
print(f'  gl_abs at alpha 0.5/0.7 (expected map unstable): significantly worse in {unstable[1]} of {unstable[0]}; exceptions {unstable[2]}')

print('\nB. Friedman test with the functions as blocks (5 blocks, medians floored at 1e-10)')
Z = np.load('results/data/exp18_r1_20261106_090957.npz')
A = ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)', 'ACOR', 'FO-ACOR(0.5)']
for s in ['canonical', 'shift', 'shiftrot']:
    for d in [10, 30, 50]:
        Mx = np.array([[max(np.median(Z[f'best::{s}|{d}|{f}|{a}']), 1e-10) for a in A] for f in FUNCS])
        chi, p = stats.friedmanchisquare(*Mx.T); rk = np.mean(np.apply_along_axis(stats.rankdata, 1, Mx), axis=0)
        order = sorted(zip(A, rk), key=lambda t: t[1])
        print(f'  {s:9s} n={d}: chi2={chi:.2f} p={p:.2g}; best {[(a, round(float(r), 1)) for a, r in order[:3]]}; worst {[(a, round(float(r), 1)) for a, r in order[-3:]]}')

print('\nC. WFO, Rastrigin n=10, equal-component vs random-direction shift of the same norm (10 runs)')
from src.algorithms.wfo import WFO
from src.benchmarks.functions import rastrigin
diag = np.full(10, -20.0); rnd = np.random.default_rng(7).uniform(-1, 1, 10); rnd = rnd / np.linalg.norm(rnd) * np.linalg.norm(diag)
for name, sh in [('origin', np.zeros(10)), ('equal components', diag), ('random direction', rnd)]:
    v = np.array([WFO(n_population=30, dim=10, pl=0.25, pe=0.2, bounds=(-100, 100), seed=42 + r).optimize(lambda x: rastrigin(np.asarray(x) - sh), 500)[1] for r in range(10)])
    print(f'  {name:17s}: success {int(np.sum(v < 1e-6))}/10, median {np.median(v):.3g}')
