"""
R1 checks requested by the second verification round (07 Nov 2026); numbers quoted in Sections 4.1.1, 4.3, 5.1.3,
5.2.8 and 5.2.10.
A. Friedman test with the five functions as blocks (medians over 30 runs, floored at 1e-10), chi-square and the
   Iman-Davenport F statistic F = (N-1) chi2 / (N (k-1) - chi2), df (k-1, (k-1)(N-1)), N = 5 blocks, k = 8 methods
   [Demsar 2006], for every suite and dimension of exp18, including the conventional-domain suite.
B. FO-PSO linearised stochastic system (as review-1/revision/math/pso_mean_square_check.py, p_i = g fixed): almost-sure
   rate exp(top Lyapunov exponent) from 20 independent products of 200,000 steps each, mean and 95% t-interval.
C. Memory weight on the past states only, (1-alpha)(A_M - 1)/M = (1-alpha)(1 - S_M)/M, M = 10: maximum over alpha.
D. L_W of Theorem 4 at n = 10, 30, 50 (p_l = 0.25, p_e = 0.2, N = 30).
E. Initial points: for seed 42 + r the 30 initial positions of PSO, JAYA, WFO (and their FO variants) coincide, and
   the 50-point initial ACOR archive starts with the same 30 points.
F. Rate check of Section 5.2.8: on Sphere the fitness is the squared distance, so fitness ratios are compared with
   squared rates (mean map, mean square, almost sure); observed ratios from results/exp23_text_numbers.txt.
G. Direction rule: for every Holm-significant comparison, the median rule (Section 5.1.3) against the sign of r.
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp26_r1_council2_checks.py > results/exp26_r1_council2_checks.txt
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import stats
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
FUNCS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']
A = ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)', 'ACOR', 'FO-ACOR(0.5)']

print('A. Friedman test with the five functions as blocks; Iman-Davenport F, df (7, 28)')
for path, suites in [('results/data/exp18_r1_20261106_090957.npz', ['canonical', 'shift', 'shiftrot']),
                     ('results/data/exp18_r1_convdomain_20261106_095722.npz', ['convdomain'])]:
    Z = np.load(path)
    for s in suites:
        for d in [10, 30, 50]:
            Mx = np.array([[max(np.median(Z[f'best::{s}|{d}|{f}|{a}']), 1e-10) for a in A] for f in FUNCS])
            chi, p = stats.friedmanchisquare(*Mx.T)
            N, k = Mx.shape
            F = (N - 1) * chi / (N * (k - 1) - chi); pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
            rk = np.mean(np.apply_along_axis(stats.rankdata, 1, Mx), axis=0)
            order = sorted(zip(A, rk), key=lambda t: t[1])
            print(f'  {s:10s} n={d}: chi2={chi:.2f} p={p:.3g} | F_ID={F:.2f} p={pF:.3g} | best {[(a, round(float(r), 1)) for a, r in order[:3]]}'
                  f' worst {[(a, round(float(r), 1)) for a, r in order[-3:]]}')

print('\nB. FO-PSO linearised stochastic system: almost-sure rate, 20 products x 200,000 steps')
sys.path.insert(0, 'math'); sys.path.insert(0, 'review-1/revision/math')
from pso_mean_square_check import mats, c1, c2
AS = {}
for lab, a, mem in [('alpha=0.9 with velocity memory', 0.9, True), ('alpha=1.0 memoryless', 1.0, False)]:
    B0, B1 = mats(a, mem); vals = []
    for rep in range(20):
        rng = np.random.default_rng(1000 + rep); T = 200000
        phis = c1 * rng.random(T) + c2 * rng.random(T)
        Q = np.eye(B0.shape[0]); s = 0.0
        for t in range(T):
            Q, R = np.linalg.qr((B0 + phis[t] * B1) @ Q); s += np.log(abs(R[0, 0]))
        vals.append(s / T)
    vals = np.array(vals); m = vals.mean(); h = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
    AS[a] = float(np.exp(m))
    print(f'  {lab:32s} rate {np.exp(m):.4f}, 95% CI [{np.exp(m - h):.4f}, {np.exp(m + h):.4f}]')

print('\nC. Memory weight on the past states only, (1-alpha)(A_M - 1)/M, M = 10')
def past_weight(al, M=10):
    w = [1.0]
    for j in range(1, M):
        w.append(w[-1] * (j - 1 - al) / j)
    return (1 - al) * (np.sum(np.abs(w)) - 1) / M
grid = np.linspace(0.001, 1, 100000); pw = np.array([past_weight(a) for a in grid])
print(f'  maximum {pw.max():.4f} at alpha = {grid[pw.argmax()]:.3f}; at alpha 0.3/0.5/0.7/0.9: '
      + ', '.join(f'{past_weight(a):.4f}' for a in (0.3, 0.5, 0.7, 0.9)))

print('\nD. L_W of Theorem 4 (p_l = 0.25, p_e = 0.2, N = 30)')
pl, pe, Np = 0.25, 0.2, 30
for n in (10, 30, 50):
    LW = pl + (1 - pl) * ((1 - 1 / n) + (1 / n) * (2 * pe + (1 - pe) / Np))
    print(f'  n={n}: L_W = {LW:.4f}')

print('\nE. Initial points shared across algorithms (seed 42 + r, r = 0..29, n = 10)')
from exp18_r1_benchmark import make_alg
ok_pos, ok_arch = True, True
for r in range(30):
    ref = None
    for a in ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)']:
        alg = make_alg(a, 10, 42 + r)
        if a in ('PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)'):
            alg._initialize()                      # WFO/FO-WFO draw their positions in __init__
        P = np.array(alg.positions, dtype=float)
        if ref is None:
            ref = P.copy()
        ok_pos &= bool(np.array_equal(P, ref))
    for a in ['ACOR', 'FO-ACOR(0.5)']:
        alg = make_alg(a, 10, 42 + r); alg._initialize_archive(lambda x: float(np.sum(np.asarray(x) ** 2)))
        arch = np.array(alg.archive, dtype=float)     # sorted by fitness after initialisation
        rows = {tuple(x) for x in arch}
        ok_arch &= (arch.shape[0] == 50) and all(tuple(x) in rows for x in ref)
print(f'  initial positions of PSO, JAYA, WFO and their FO variants identical in all 30 seeds: {ok_pos}')
print(f'  50-point ACOR / FO-ACOR archive contains these 30 points in all 30 seeds: {ok_arch}')

print('\nF. FO-PSO rate check on Sphere (Section 5.2.8): fitness ratio = squared distance rate')
import re
from pso_mean_square_check import rates
txt = open('results/exp23_text_numbers.txt').read()
obs = {k: float(re.search(rf'{re.escape(k)}\s+decay ratio mean ([0-9.]+)', txt).group(1)) for k in ('FO-PSO(0.9)', 'PSO')}
mm9, ms9, _ = rates(0.9, True, T=1); mm1, ms1, _ = rates(1.0, False, T=1)
print(f'  observed ratios (exp23): FO-PSO {obs["FO-PSO(0.9)"]:.3f}, PSO {obs["PSO"]:.3f}; quotient {obs["FO-PSO(0.9)"] / obs["PSO"]:.3f}')
print(f'  mean-map rates {mm9:.4f} / {mm1:.4f}: squared {mm9**2:.3f} / {mm1**2:.3f}, quotient {(mm9 / mm1)**2:.3f}')
print(f'  mean-square rates {ms9:.4f} / {ms1:.4f}: squared {ms9**2:.3f} / {ms1**2:.3f}, quotient {(ms9 / ms1)**2:.3f}')
print('  almost-sure rates: see part B (quotient of squares printed below)')
print(f'  almost-sure quotient of squares from part B: {(AS[0.9] / AS[1.0])**2:.3f}; squared rates {AS[0.9]**2:.3f} / {AS[1.0]**2:.3f}')

print('\nG. Direction of significant comparisons: median rule vs sign of the rank-biserial r (all suites and dimensions)')
sys.path.insert(0, 'experiments')
from exp18_r1_analysis import holm, rank_biserial, PAIRS
tot = agree = 0; bad = []
for path, suites in [('results/data/exp18_r1_20261106_090957.npz', ['canonical', 'shift', 'shiftrot']),
                     ('results/data/exp18_r1_convdomain_20261106_095722.npz', ['convdomain'])]:
    Z = np.load(path)
    for s in suites:
        for d in [10, 30, 50]:
            ps, info = [], []
            for f in FUNCS:
                for b, fo in PAIRS:
                    x = np.maximum(Z[f'best::{s}|{d}|{f}|{b}'], 1e-10); y = np.maximum(Z[f'best::{s}|{d}|{f}|{fo}'], 1e-10)
                    ps.append(1.0 if np.all(x == y) else stats.wilcoxon(x, y, zero_method='zsplit').pvalue)
                    info.append((f, fo, np.sign(np.median(x) - np.median(y)), np.sign(rank_biserial(x, y))))
            for (f, fo, sm, sr), q in zip(info, holm(ps)):
                if q < 0.05:
                    tot += 1
                    if sm == sr and sm != 0: agree += 1
                    else: bad.append((s, d, f, fo, int(sm), int(sr)))
print(f'  significant comparisons: {tot}; median direction equals sign of r: {agree}; exceptions: {bad}')
