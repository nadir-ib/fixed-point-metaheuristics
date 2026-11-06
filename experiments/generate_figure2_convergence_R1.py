"""
Figure 2 (R1): median best-so-far convergence (30 runs, IQR band) from the R1 benchmark (exp18, code v2), n = 10.
Top row: canonical suite (optimum at the origin, the centre of [-100,100]^n). Bottom row: shifted and rotated suite
(same functions, optimum moved to o ~ U[-80,80]^n, random rotation). Same three baseline/FO pairs as the submitted
Figure 2. Annotations are computed from the data (median final values), not typed in.
Run from repo root: python3 generate_figure2_convergence_R1.py results/data/exp18_r1_20261106_090957.npz
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'results/data/exp18_r1_20261106_090957.npz'
OUT = os.path.join('results', 'figures', 'figure2_convergence_R1')
FLOOR = 1e-35
z = np.load(NPZ)
PAIRS = [('sphere', 'PSO', 'FO-PSO(0.9)', 'Sphere'), ('rastrigin', 'JAYA', 'FO-JAYA(0.5)', 'Rastrigin'),
         ('rosenbrock', 'WFO', 'FO-WFO(0.5)', 'Rosenbrock')]
ROWS = [('canonical', 'optimum at the origin'), ('shiftrot', 'shifted and rotated')]
BASE_C, FO_C = '#0072B2', '#D55E00'
fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
report = []
for r, (suite, rlab) in enumerate(ROWS):
    for c, (fn, base, fo, flab) in enumerate(PAIRS):
        ax = axes[r, c]; it = np.arange(1, 501); finals = {}
        for alg, color, ls in [(base, BASE_C, '-'), (fo, FO_C, '--')]:
            arr = np.maximum(z[f'curve::{suite}|10|{fn}|{alg}'], FLOOR)
            med = np.median(arr, axis=0)
            ax.plot(it, med, ls, color=color, lw=1.8, label=alg)
            ax.fill_between(it, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), color=color, alpha=0.18, lw=0)
            fin = np.median(z[f'best::{suite}|10|{fn}|{alg}'])
            assert np.isclose(np.median(z[f'curve::{suite}|10|{fn}|{alg}'][:, -1]), fin, rtol=0, atol=0) or fin == 0
            finals[alg] = fin
        ax.set_yscale('log'); ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_title(f'({"abc"[c] if r == 0 else "def"[c]}) {flab}, {rlab}', fontsize=11.5, fontweight='bold')
        ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
        ax.text(0.03, 0.04, f'median final: {base} {finals[base]:.3g}\n{fo} {finals[fo]:.3g}', transform=ax.transAxes,
                fontsize=8.5, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.45))
        if r == 1: ax.set_xlabel('Iteration', fontsize=11)
        report.append((suite, fn, base, finals[base], fo, finals[fo]))
    axes[r, 0].set_ylabel('Best-so-far objective value (log scale)', fontsize=10.5)
fig.suptitle('Median best-so-far convergence over 30 runs, n = 10 (shaded: interquartile range); '
             'zeros shown at the 1e-35 display floor', fontsize=10.5, y=1.0)
plt.tight_layout()
plt.savefig(OUT + '.png', dpi=300, bbox_inches='tight'); plt.savefig(OUT + '.pdf', bbox_inches='tight')
print('saved', OUT + '.png/.pdf')
for row in report:
    print('  %-9s %-10s %-5s %.4g | %-13s %.4g' % row)
