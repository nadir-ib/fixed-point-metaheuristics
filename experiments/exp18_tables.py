"""
Cell contents of manuscript Tables 5-13 (R1) from the R1 data. No number is typed by hand.
  Tables 5-9  : canonical suite, n = 10: Mean ± Std (ddof = 1, computed on rescaled values so that tiny values do not
                underflow), Best (minimum), success rate (f < 1e-6, raw values), 30 runs.
  Table 10    : serial timing, n = 10 (results/data/exp21_timing.json).
  Table 11    : Wilcoxon signed-rank FO vs baseline, canonical n = 10, values floored at 1e-10, two-sided, zsplit;
                Holm-adjusted over the 20 comparisons; direction + (FO lower median) / - (baseline lower median).
  Table 12    : Friedman over 8 algorithms, 150 blocks (5 functions x 30 runs), floored values; mean ranks.
                Per-function Friedman tests are printed separately (primary analysis, R1-M3b / R4-13).
  Table 13    : canonical suite n = 30 and 50 (same protocol as n = 10): median [Q1, Q3] (success rate),
                thresholds Sphere < 1e-10, Rastrigin < 10 (as in the submitted Table 13).
Usage: python3 experiments/exp18_tables.py results/data/exp18_r1_20261106_090957.npz results/data/exp21_timing.json
Writes results/tables/R1_tables.json and prints a readable version.
"""
import sys, json, os
import numpy as np
from scipy import stats
FUNCS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']
ALGS = ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)', 'ACOR', 'FO-ACOR(0.5)']
PAIRS = [('PSO', 'FO-PSO(0.9)'), ('JAYA', 'FO-JAYA(0.5)'), ('WFO', 'FO-WFO(0.5)'), ('ACOR', 'FO-ACOR(0.5)')]
FLOOR = 1e-10


def num(v, latex=True):
    """Same convention as the submitted tables: 2 decimals in [0.01, 1e6), otherwise a x 10^b with 2 decimals."""
    if v == 0:
        return '0'
    if 1e-2 <= abs(v) < 1e6:
        return f'{v:.2f}'
    e = int(np.floor(np.log10(abs(v)))); m = v / 10 ** e
    if round(abs(m), 2) >= 10:
        m /= 10; e += 1
    return f'{m:.2f}\\times10^{{{e}}}' if latex else f'{m:.2f}e{e}'


def mean_std(v):
    v = np.asarray(v, float); s = np.max(np.abs(v))
    if s == 0:
        return 0.0, 0.0
    w = v / s
    return float(s * w.mean()), float(s * w.std(ddof=1))


def holm(p):
    p = np.asarray(p); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def main(npz, timing):
    Z = np.load(npz); T = json.load(open(timing)); out = {}
    g = lambda s, d, f, a: Z[f'best::{s}|{d}|{f}|{a}']
    for ti, f in zip(range(5, 10), FUNCS):
        rows = []
        for a in ALGS:
            v = g('canonical', 10, f, a); mu, sd = mean_std(v)
            rows.append(dict(alg=a, mean_std=f'{num(mu)}\\pm{num(sd)}', best=num(float(v.min())), sr=f'{100 * np.mean(v < 1e-6):.1f}',
                             mean=mu, std=sd, median=float(np.median(v))))
        out[f'T{ti}'] = rows
    base = {'FO-PSO(0.9)': 'PSO', 'FO-JAYA(0.5)': 'JAYA', 'FO-WFO(0.5)': 'WFO', 'FO-ACOR(0.5)': 'ACOR'}
    row = T['A|10']
    out['T10'] = [dict(alg=a, time=f'{row[a]:.3f}', overhead='-' if a not in base else f'{100 * (row[a] / row[base[a]] - 1):+.1f}%') for a in ALGS]
    ps, cells = [], []
    for b, fo in PAIRS:
        for f in FUNCS:
            x = np.maximum(g('canonical', 10, f, b), FLOOR); y = np.maximum(g('canonical', 10, f, fo), FLOOR)
            p = 1.0 if np.all(x == y) else stats.wilcoxon(x, y, zero_method='zsplit').pvalue
            sign = '+' if np.median(y) < np.median(x) else ('-' if np.median(y) > np.median(x) else '=')
            ps.append(p); cells.append((b, fo, f, sign))
    adj = holm(ps); t11 = {}
    for (b, fo, f, sign), p, pa in zip(cells, ps, adj):
        txt = ('< 0.001' if pa < 0.001 else f'{pa:.3f}') + (f' ({sign})' if pa < 0.05 else ' ns')
        t11.setdefault(f'{b} vs {fo}', {})[f] = dict(text=txt, p_raw=float(p), p_holm=float(pa), dir=sign)
    out['T11'] = t11
    blocks = np.vstack([np.column_stack([np.maximum(g('canonical', 10, f, a), FLOOR) for a in ALGS]) for f in FUNCS])
    chi, pf = stats.friedmanchisquare(*blocks.T); rk = np.mean(np.apply_along_axis(stats.rankdata, 1, blocks), axis=0)
    out['T12'] = dict(chi2=float(chi), p=float(pf), ranks=sorted([(a, float(r)) for a, r in zip(ALGS, rk)], key=lambda t: t[1]))
    per = {}
    for f in FUNCS:
        bl = np.column_stack([np.maximum(g('canonical', 10, f, a), FLOOR) for a in ALGS])
        c, p = stats.friedmanchisquare(*bl.T); r = np.mean(np.apply_along_axis(stats.rankdata, 1, bl), axis=0)
        per[f] = dict(chi2=float(c), p=float(p), ranks={a: float(x) for a, x in zip(ALGS, r)})
    out['friedman_per_function'] = per
    thr = {'sphere': 1e-10, 'rastrigin': 10.0}
    t13 = {}
    for a in ['PSO', 'JAYA', 'WFO', 'ACOR', 'FO-JAYA(0.5)', 'FO-WFO(0.5)', 'FO-ACOR(0.5)', 'FO-PSO(0.9)']:
        for f in ['sphere', 'rastrigin']:
            for d in (30, 50):
                v = g('canonical', d, f, a); q1, m, q3 = np.percentile(v, [25, 50, 75])
                t13.setdefault(a, {})[f'{f}|{d}'] = dict(median=num(m), q1=num(q1), q3=num(q3), sr=f'{100 * np.mean(v < thr[f]):.0f}%')
    out['T13'] = t13
    os.makedirs('results/tables', exist_ok=True)
    json.dump(out, open('results/tables/R1_tables.json', 'w'), indent=1)
    for ti in range(5, 10):
        print(f'Table {ti}'); [print(f"  {r['alg']:13s} {r['mean_std']:42s} best {r['best']:22s} SR {r['sr']}") for r in out[f'T{ti}']]
    print('Table 10'); [print(f"  {r['alg']:13s} {r['time']} {r['overhead']}") for r in out['T10']]
    print('Table 11 (Holm-adjusted)'); [print(f'  {k:22s} ' + ' | '.join(f"{f}: {v[f]['text']}" for f in FUNCS)) for k, v in t11.items()]
    print(f"Table 12 Friedman chi2={chi:.2f} p={pf:.3g}"); [print(f'  {a:13s} {r:.2f}') for a, r in out['T12']['ranks']]
    print('Per-function Friedman'); [print(f"  {f:10s} chi2={v['chi2']:.1f} p={v['p']:.2g}") for f, v in per.items()]
    print('Table 13'); [print(f'  {a:13s} ' + ' | '.join(f"{k}: {c['median']} [{c['q1']}, {c['q3']}] ({c['sr']})" for k, c in v.items())) for a, v in t13.items()]


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
