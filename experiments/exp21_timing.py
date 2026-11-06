"""
R1 timing (reviewer R3-2.3; Table 10). MUST run serially on an otherwise idle machine, one process.
Part A: wall-clock per run (time.perf_counter around optimize) for the 8 algorithms, canonical suite, box [-100,100]^n,
        n = 10 (30 runs, seeds 42+run, as Table 10), n = 30 and 50 (5 runs). Code version v2 (CHANGES.md).
Part B: the GL memory term for a whole population, N = 30, M = 10, n in {10, 30, 50}:
        (i)  per-particle Python loop over the history, as in fo_jaya.py / fo_wfo.py;
        (ii) one vectorised contraction np.tensordot(omega, H, axes=(0, 0)) over a ring buffer H of shape (M, N, n).
        Both are checked to agree to rtol 1e-12 on the same data; median time per call over 2000 calls.
Run from repo root: OMP_NUM_THREADS=1 python3 experiments/exp21_timing.py
"""
import os, sys, time, json, platform
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from exp18_r1_benchmark import make_alg, ALGS, FUNCS, BASE

OUT = {}


def part_a():
    print('Part A: mean wall-clock seconds per run (serial)')
    for dim, runs in [(10, 30), (30, 5), (50, 5)]:
        row = {}
        for a in ALGS:
            ts = []
            for f in FUNCS:
                for r in range(runs):
                    alg = make_alg(a, dim, 42 + r)
                    t = time.perf_counter(); alg.optimize(BASE[f], max_iter=500); ts.append(time.perf_counter() - t)
            row[a] = float(np.mean(ts))
        OUT[f'A|{dim}'] = row
        base = {'FO-PSO(0.9)': 'PSO', 'FO-JAYA(0.5)': 'JAYA', 'FO-WFO(0.5)': 'WFO', 'FO-ACOR(0.5)': 'ACOR'}
        print(f'  n={dim}: ' + ', '.join(f'{a} {row[a]:.3f}' for a in ALGS))
        print(f'         overhead: ' + ', '.join(f'{fo} {100*(row[fo]/row[b]-1):+.1f}%' for fo, b in base.items()))


def part_b():
    print('\nPart B: GL memory term for a population (N=30, M=10), median microseconds per call')
    rng = np.random.default_rng(0); N, M, alpha = 30, 10, 0.5
    om = [1.0]
    for k in range(1, M):
        om.append(om[-1] * (k - 1 - alpha) / k)
    om = np.array(om)
    for n in (10, 30, 50):
        hist = [[rng.normal(size=n) for _ in range(M)] for _ in range(N)]      # per-particle lists, oldest first
        H = np.stack([np.stack([hist[i][M - 1 - j] for i in range(N)]) for j in range(M)])   # H[j] = x_{t-j}, (M, N, n)

        def loop():
            out = np.zeros((N, n))
            for i in range(N):
                L = len(hist[i]); m = np.zeros(n)
                for k in range(L):
                    m += om[k] * hist[i][-(k + 1)]
                out[i] = (1 - alpha) * m / L
            return out

        def vec():
            return (1 - alpha) * np.tensordot(om, H, axes=(0, 0)) / M

        assert np.allclose(loop(), vec(), rtol=1e-12, atol=1e-12)
        tl, tv = [], []
        for _ in range(2000):
            t = time.perf_counter(); loop(); tl.append(time.perf_counter() - t)
            t = time.perf_counter(); vec(); tv.append(time.perf_counter() - t)
        a_, b_ = 1e6 * np.median(tl), 1e6 * np.median(tv)
        OUT[f'B|{n}'] = dict(loop_us=a_, vec_us=b_)
        print(f'  n={n}: loop {a_:.1f} us, vectorised {b_:.1f} us, speed-up x{a_/b_:.0f} (results agree to 1e-12)')


if __name__ == '__main__':
    part_a(); part_b()
    OUT['platform'] = platform.platform(); OUT['python'] = platform.python_version(); OUT['numpy'] = np.__version__
    OUT['cpu'] = platform.processor()
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'data', 'exp21_timing.json'), 'w'), indent=1)
