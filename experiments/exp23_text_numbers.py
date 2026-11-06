"""
Every number quoted in the R1 text of Sections 5-7 that is not already a table cell, computed from the R1 data.
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp23_text_numbers.py > results/exp23_text_numbers.txt
Inputs: results/data/exp18_r1_20261106_090957.npz (main), results/data/exp18_r1_convdomain_20261106_095722.npz,
results/data/exp19_centre_bias_probe.json, results/data/exp22_alpha_sweep.json, results/data/exp21_timing.json.
Part F runs FO-PSO at alpha = 0.35 and 0.3 (canonical Sphere, n = 10, 30 runs) for the instability ablation of 5.2.8,
whose submitted numbers had no script in the repository.
"""
import os, sys, json, math
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import stats
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
FUNCS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']
ALGS = ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)', 'ACOR', 'FO-ACOR(0.5)']
PAIRS = [('PSO', 'FO-PSO(0.9)'), ('JAYA', 'FO-JAYA(0.5)'), ('WFO', 'FO-WFO(0.5)'), ('ACOR', 'FO-ACOR(0.5)')]
FLOOR = 1e-10
Z = np.load('results/data/exp18_r1_20261106_090957.npz')
ZC = np.load('results/data/exp18_r1_convdomain_20261106_095722.npz')
g = lambda s, d, f, a, z=Z: z[f'best::{s}|{d}|{f}|{a}']


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj


def rb(x, y):
    dlt = x - y; dlt = dlt[dlt != 0]
    if len(dlt) == 0: return 0.0
    r = stats.rankdata(np.abs(dlt)); return (r[dlt > 0].sum() - r[dlt < 0].sum()) / r.sum()


def suite_summary(s, d, z=Z):
    ps, dirs, rbs = [], [], []
    for f in FUNCS:
        for b, fo in PAIRS:
            x = np.maximum(g(s, d, f, b, z), FLOOR); y = np.maximum(g(s, d, f, fo, z), FLOOR)
            ps.append(1.0 if np.all(x == y) else stats.wilcoxon(x, y, zero_method='zsplit').pvalue)
            dirs.append('+' if np.median(y) < np.median(x) else ('-' if np.median(y) > np.median(x) else '='))
            rbs.append(rb(x, y))
    adj = holm(ps); sig = adj < 0.05
    blocks = np.vstack([np.column_stack([np.maximum(g(s, d, f, a, z), FLOOR) for a in ALGS]) for f in FUNCS])
    chi, pf = stats.friedmanchisquare(*blocks.T); rk = np.mean(np.apply_along_axis(stats.rankdata, 1, blocks), axis=0)
    order = sorted(zip(ALGS, rk), key=lambda t: t[1])
    return dict(sig=int(sig.sum()), fo_better=int(sum(1 for q, dd in zip(sig, dirs) if q and dd == '+')),
                base_better=int(sum(1 for q, dd in zip(sig, dirs) if q and dd == '-')), chi2=float(chi), p=float(pf),
                ranks=[(a, round(float(r), 2)) for a, r in order], rb=rbs, dirs=dirs)


print('A. Holm-significant FO-vs-baseline comparisons (of 20), Friedman (150 blocks), floor 1e-10')
for s in ['canonical', 'shift', 'shiftrot']:
    for d in (10, 30, 50):
        S = suite_summary(s, d)
        print(f"  {s:9s} n={d}: sig {S['sig']} (FO better {S['fo_better']}, baseline better {S['base_better']}); "
              f"Friedman chi2={S['chi2']:.2f} p={S['p']:.2g}; best ranks {S['ranks'][:3]}; worst {S['ranks'][-3:]}")
for d in (10, 30, 50):
    S = suite_summary('convdomain', d, ZC)
    print(f"  convdomain n={d}: sig {S['sig']} (FO better {S['fo_better']}, baseline better {S['base_better']}); "
          f"Friedman chi2={S['chi2']:.2f} p={S['p']:.2g}; best ranks {S['ranks'][:3]}")

print('\nB. Rank-biserial effect sizes, canonical n=10 (floor 1e-10); r > 0: FO variant better')
S = suite_summary('canonical', 10); k = 0; large_fo = large_b = 0
for f in FUNCS:
    row = []
    for b, fo in PAIRS:
        r = S['rb'][k]; k += 1; row.append(f'{fo} {r:+.2f}')
        if abs(r) > 0.5: large_fo += r > 0; large_b += r < 0
    print(f'  {f:10s} ' + ' | '.join(row))
print(f'  |r| > 0.5: {large_fo + large_b} of 20 ({large_fo} FO better, {large_b} baseline better)')

print('\nC. Selected ratios (canonical n=10, means)')
m = lambda f, a: float(np.mean(g('canonical', 10, f, a)))
print(f"  Sphere ACOR/FO-ACOR mean ratio: 10^{math.log10(m('sphere', 'ACOR') / m('sphere', 'FO-ACOR(0.5)')):.1f}")
print(f"  Sphere PSO/FO-PSO mean ratio: {m('sphere', 'PSO') / m('sphere', 'FO-PSO(0.9)'):.3g} "
      f"(PSO {m('sphere', 'PSO'):.3g}, FO-PSO {m('sphere', 'FO-PSO(0.9)'):.3g})")
print(f"  Rosenbrock means: WFO {m('rosenbrock', 'WFO'):.3g}, FO-WFO {m('rosenbrock', 'FO-WFO(0.5)'):.3g}, JAYA mean Rastrigin {m('rastrigin', 'JAYA'):.4g}")
print(f"  PSO Sphere mean n=50: {np.mean(g('canonical', 50, 'sphere', 'PSO')):.4g}, median {np.median(g('canonical', 50, 'sphere', 'PSO')):.4g}; "
      f"runs >= 1e4: {int(np.sum(g('canonical', 50, 'sphere', 'PSO') >= 1e4))}/30")

print('\nD. FO-PSO rate check (Sphere, canonical n=10, iterations 10..149, floor 1e-30, per-run log-linear fit)')
def rate(c):
    y = np.log(np.maximum(c[10:150], 1e-30)); return math.exp(np.polyfit(np.arange(y.size), y, 1)[0])
for a in ['FO-PSO(0.9)', 'PSO']:
    rr = np.array([rate(c) for c in Z[f'curve::canonical|10|sphere|{a}']])
    print(f'  {a:12s} decay ratio mean {rr.mean():.3f} +- {rr.std():.3f} (sd over runs, ddof=0)')

print('\nE. Centre-bias probe exp19 (median lambda; kappa)')
P = json.load(open('results/data/exp19_centre_bias_probe.json'))['rows']
for row in P:
    if row['function'] == 'sphere':
        print(f"  {row['algorithm']:8s} alpha={row['alpha']}: lambda {np.median(row['lam']):.3f}  (kappa {row['kappa']:.3f}, tau {row['tau']:.3f})")

print('\nF. FO-PSO instability ablation (canonical Sphere, n=10, 30 runs): median final f')
from exp18_r1_benchmark import Objective
from src.algorithms.fo_pso import FO_PSO
A22 = json.load(open('results/data/exp22_alpha_sweep.json'))
for al in (0.9, 0.5):
    print(f"  alpha={al}: {np.median(A22[f'canonical|sphere|FO-PSO|{al}']):.3g} (exp22)")
for al in (0.35, 0.3):
    v = [FO_PSO(n_particles=30, dim=10, w=0.7, c1=1.5, c2=1.5, alpha=al, bounds=(-100, 100), seed=42 + r).optimize(Objective('canonical', 'sphere', 10), 500)[1] for r in range(30)]
    print(f'  alpha={al}: {np.median(v):.3g} (computed here)')
