"""
Statistics for exp18 (R1). Usage: python3 experiments/exp18_r1_analysis.py results/data/exp18_r1_<stamp>.npz
One floor policy for every statistic: values below FLOOR = 1e-10 are set to FLOOR (ties at the floor are ties).
Reports per suite x dim:
  - median [IQR] and success rate (f < 1e-6, on raw values) per function x algorithm
  - paired Wilcoxon signed-rank FO vs own baseline (two-sided, zero_method='zsplit' so all-tied pairs give p = 1),
    Holm correction over the 20 pairs of that suite x dim, direction (+ = FO better median, - = baseline better)
  - matched-pairs rank-biserial effect size r
  - Friedman test over the 8 algorithms with runs x functions as blocks, and per function
"""
import sys, json
import numpy as np
from scipy import stats

FLOOR = 1e-10
FUNCS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']
ALGS = ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)', 'ACOR', 'FO-ACOR(0.5)']
PAIRS = [('PSO', 'FO-PSO(0.9)'), ('JAYA', 'FO-JAYA(0.5)'), ('WFO', 'FO-WFO(0.5)'), ('ACOR', 'FO-ACOR(0.5)')]


def holm(p):
    p = np.asarray(p); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def rank_biserial(base, fo):
    d = base - fo                       # positive = FO smaller (better)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def main(path):
    Z = np.load(path); meta = json.loads(str(Z['meta']))
    suites = meta['protocol']['suites']; dims = meta['protocol']['dims']
    get = lambda s, d, f, a: Z[f'best::{s}|{d}|{f}|{a}']
    print(f'# exp18 analysis of {path}\n# floor {FLOOR:g}; success = raw f < 1e-6; code sha256 recorded in meta\n')
    for s in suites:
        for d in dims:
            print(f'\n## suite={s}  n={d}')
            print('median [Q1, Q3] (success %)')
            for f in FUNCS:
                row = []
                for a in ALGS:
                    v = get(s, d, f, a); q1, med, q3 = np.percentile(v, [25, 50, 75])
                    row.append(f'{a}: {med:.3g} [{q1:.2g},{q3:.2g}] ({100*np.mean(v < 1e-6):.0f}%)')
                print(f'  {f:10s} ' + ' | '.join(row))
            ps, lines = [], []
            for f in FUNCS:
                for b, fo in PAIRS:
                    x = np.maximum(get(s, d, f, b), FLOOR); y = np.maximum(get(s, d, f, fo), FLOOR)
                    p = 1.0 if np.all(x == y) else stats.wilcoxon(x, y, zero_method='zsplit').pvalue
                    ps.append(p)
                    sign = '+' if np.median(y) < np.median(x) else ('-' if np.median(y) > np.median(x) else '=')
                    lines.append((f, fo, p, sign, rank_biserial(x, y)))
            adj = holm(ps)
            nsig = sum(a_ < 0.05 for a_ in adj)
            nplus = sum(1 for (l, a_) in zip(lines, adj) if a_ < 0.05 and l[3] == '+')
            nminus = sum(1 for (l, a_) in zip(lines, adj) if a_ < 0.05 and l[3] == '-')
            print(f'Wilcoxon FO vs baseline (Holm over 20): significant {nsig}/20  (FO better {nplus}, baseline better {nminus})')
            for (f, fo, p, sign, r), a_ in zip(lines, adj):
                print(f'  {f:10s} {fo:13s} p={p:.3g} p_holm={a_:.3g} dir={sign} r={r:+.2f}')
            blocks = np.vstack([np.column_stack([np.maximum(get(s, d, f, a), FLOOR) for a in ALGS]) for f in FUNCS])
            chi, pf = stats.friedmanchisquare(*blocks.T)
            ranks = np.mean(np.apply_along_axis(stats.rankdata, 1, blocks), axis=0)
            print(f'Friedman (150 blocks): chi2={chi:.2f} p={pf:.3g}; mean ranks: ' +
                  ', '.join(f'{a} {r:.2f}' for a, r in sorted(zip(ALGS, ranks), key=lambda t: t[1])))
            for f in FUNCS:
                bl = np.column_stack([np.maximum(get(s, d, f, a), FLOOR) for a in ALGS])
                chi, pf = stats.friedmanchisquare(*bl.T)
                rk = np.mean(np.apply_along_axis(stats.rankdata, 1, bl), axis=0)
                print(f'  Friedman {f:10s} chi2={chi:.1f} p={pf:.2g}; best: ' +
                      ', '.join(f'{a} {r:.2f}' for a, r in sorted(zip(ALGS, rk), key=lambda t: t[1])[:3]))


if __name__ == '__main__':
    main(sys.argv[1])
