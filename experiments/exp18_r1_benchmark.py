"""
R1 benchmark (revision round 1): 8 algorithms x 5 functions x {canonical, shift, shift+rotation} x n in {10, 30, 50}.

Protocol (Table 4 of the manuscript, unchanged): 30 runs, seed = 42 + run, 500 iterations, population 30,
search box [-100, 100]^n for every function, ACOR archive k = 50, q = 0.01, xi = 0.85, memory M = 10.
Code version: src/ after CHANGES.md (C001) (ACOR batch archive update) and C002 (JAYA evaluation budget).

Problem suites (reviewer R4-11, internal verification):
  canonical : f(x)                         optimum at the origin (Rosenbrock: at 1)
  convdomain: f(x) on the conventional domain of each function (CONV_BOUNDS, Yao et al. 1999), R4-11
  shift     : f(x - o)                     o ~ U[-80, 80]^n, a NON-diagonal shift (different value per coordinate)
  shiftrot  : f(Q (x - o))                 Q random orthogonal (QR of a Gaussian matrix with sign fix)
  Rosenbrock in shift/shiftrot uses f(Q(x - o) + 1), so its optimum is exactly o.
Problem instances come from a separate generator np.random.default_rng(PROBLEM_SEED + 100*dim + fidx) and are
identical for every algorithm and run; they are stored in the output file. Algorithms seed the global numpy RNG in
their constructors, so each job runs in its own worker process and constructs the algorithm immediately before
optimize(), exactly as in exp6.

Run from the repo root:  OMP_NUM_THREADS=1 python3 experiments/exp18_r1_benchmark.py [--workers 30] [--quick]
"""
import os, sys, time, json, hashlib, platform, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from src.algorithms.pso import PSO
from src.algorithms.fo_pso import FO_PSO
from src.algorithms.jaya import JAYA
from src.algorithms.fo_jaya import FO_JAYA
from src.algorithms.wfo import WFO
from src.algorithms.fo_wfo import FO_WFO
from src.algorithms.acor import ACOR
from src.algorithms.fo_acor import FO_ACOR
from src.benchmarks.functions import sphere, rastrigin, ackley, rosenbrock, griewank

BOUNDS = (-100, 100)
FUNCS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']
BASE = dict(sphere=sphere, rastrigin=rastrigin, ackley=ackley, rosenbrock=rosenbrock, griewank=griewank)
ALGS = ['PSO', 'FO-PSO(0.9)', 'JAYA', 'FO-JAYA(0.5)', 'WFO', 'FO-WFO(0.5)', 'ACOR', 'FO-ACOR(0.5)']
SUITES = ['canonical', 'shift', 'shiftrot']          # default; 'convdomain' is run separately (R4-11)
# Conventional domains, Yao, Liu & Lin (1999) IEEE TEC 3(2):82-102, Table I (f1, f9, f10, f5, f11); verified 06 Nov 2026.
CONV_BOUNDS = dict(sphere=(-100, 100), rastrigin=(-5.12, 5.12), ackley=(-32, 32), rosenbrock=(-30, 30), griewank=(-600, 600))
DIMS = [10, 30, 50]
PROBLEM_SEED = 20261106


def make_alg(name, dim, seed, b=BOUNDS):
    return {
        'PSO': lambda: PSO(n_particles=30, dim=dim, w=0.7, c1=1.5, c2=1.5, bounds=b, seed=seed),
        'FO-PSO(0.9)': lambda: FO_PSO(n_particles=30, dim=dim, w=0.7, c1=1.5, c2=1.5, alpha=0.9, bounds=b, seed=seed),
        'JAYA': lambda: JAYA(n_population=30, dim=dim, bounds=b, seed=seed),
        'FO-JAYA(0.5)': lambda: FO_JAYA(n_population=30, dim=dim, alpha=0.5, memory_size=10, bounds=b, seed=seed),
        'WFO': lambda: WFO(n_population=30, dim=dim, pl=0.25, pe=0.2, bounds=b, seed=seed),
        'FO-WFO(0.5)': lambda: FO_WFO(n_population=30, dim=dim, alpha=0.5, memory_size=10, pl=0.25, pe=0.2, bounds=b, seed=seed),
        'ACOR': lambda: ACOR(n_ants=30, dim=dim, archive_size=50, q=0.01, xi=0.85, bounds=b, seed=seed),
        'FO-ACOR(0.5)': lambda: FO_ACOR(n_ants=30, dim=dim, alpha=0.5, archive_size=50, q=0.01, xi=0.85, memory_size=10, bounds=b, seed=seed),
    }[name]()


def random_orthogonal(rng, n):
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))          # Haar-distributed (Mezzadri 2007 sign fix)


def problem_instance(suite, fname, dim):
    """Returns (o, Q); o = shift (None for canonical), Q = rotation (None unless shiftrot)."""
    if suite in ('canonical', 'convdomain'):
        return None, None
    rng = np.random.default_rng(PROBLEM_SEED + 100 * dim + FUNCS.index(fname))
    o = rng.uniform(-80.0, 80.0, dim)
    Q = random_orthogonal(rng, dim) if suite == 'shiftrot' else None   # same o for shift and shiftrot
    return o, Q


class Objective:
    def __init__(self, suite, fname, dim):
        self.f = BASE[fname]; self.o, self.Q = problem_instance(suite, fname, dim)
        self.plus1 = (fname == 'rosenbrock' and suite != 'canonical'); self.nfe = 0

    def __call__(self, x):
        self.nfe += 1
        x = np.asarray(x, dtype=float)
        if self.o is None:
            return float(self.f(x))
        z = x - self.o
        if self.Q is not None:
            z = self.Q @ z
        if self.plus1:
            z = z + 1.0
        return float(self.f(z))


def job(args):
    suite, dim, fname, alg, run, max_iter = args
    obj = Objective(suite, fname, dim)
    a = make_alg(alg, dim, 42 + run, CONV_BOUNDS[fname] if suite == 'convdomain' else BOUNDS)
    t = time.perf_counter()
    _, best, curve = a.optimize(obj, max_iter=max_iter)
    return (suite, dim, fname, alg, run, float(best), np.asarray(curve, dtype=float), obj.nfe, time.perf_counter() - t)


def file_sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--suites', default=','.join(SUITES))
    ap.add_argument('--dims', default=','.join(map(str, DIMS)))
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 runs, 20 iterations')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.quick:
        a.runs, a.iters = 2, 20
    suites = a.suites.split(','); dims = [int(d) for d in a.dims.split(',')]
    jobs = [(s, d, f, g, r, a.iters) for s in suites for d in dims for f in FUNCS for g in ALGS for r in range(a.runs)]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = a.out or os.path.join(ROOT, 'results', 'data', f'exp18_r1_{"quick_" if a.quick else ""}{stamp}.npz')
    print(f'{len(jobs)} jobs, {a.workers} workers -> {out}', flush=True)
    best = {}; curves = {}; nfe = {}; secs = {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(job, jobs, chunksize=4):
            s, d, f, g, r, b, c, n, tt = res
            key = f'{s}|{d}|{f}|{g}'
            best.setdefault(key, np.full(a.runs, np.nan))[r] = b
            curves.setdefault(key, np.full((a.runs, a.iters), np.nan))[r] = c
            nfe.setdefault(key, np.zeros(a.runs, dtype=np.int64))[r] = n
            secs.setdefault(key, np.zeros(a.runs))[r] = tt
            done += 1
            if done % 500 == 0:
                print(f'  {done}/{len(jobs)}  {time.time()-t0:.0f}s', flush=True)
    meta = dict(protocol=dict(runs=a.runs, iters=a.iters, seed='42+run', bounds=BOUNDS, suites=suites, dims=dims,
                              funcs=FUNCS, algs=ALGS, problem_seed=PROBLEM_SEED, shift='U[-80,80]^n',
                              rotation='Haar orthogonal (QR + sign fix)', rosenbrock_shift='f(Q(x-o)+1)', conv_bounds=CONV_BOUNDS if 'convdomain' in suites else None),
                numpy=np.__version__, python=platform.python_version(), platform=platform.platform(),
                code_sha256={p: file_sha(os.path.join(ROOT, 'src', p)) for p in
                             ['algorithms/pso.py', 'algorithms/fo_pso.py', 'algorithms/jaya.py', 'algorithms/fo_jaya.py',
                              'algorithms/wfo.py', 'algorithms/fo_wfo.py', 'algorithms/acor.py', 'algorithms/fo_acor.py',
                              'benchmarks/functions.py']},
                wall_seconds=time.time() - t0, note='timings measured under parallel load; NOT for Table 10')
    inst = {}
    for s in suites:
        for d in dims:
            for f in FUNCS:
                o, Q = problem_instance(s, f, d)
                if o is not None: inst[f'shift|{s}|{d}|{f}'] = o
                if Q is not None: inst[f'rot|{s}|{d}|{f}'] = Q
    np.savez_compressed(out, meta=json.dumps(meta),
                        **{f'best::{k}': v for k, v in best.items()}, **{f'curve::{k}': v for k, v in curves.items()},
                        **{f'nfe::{k}': v for k, v in nfe.items()}, **{f'sec::{k}': v for k, v in secs.items()},
                        **{f'inst::{k}': v for k, v in inst.items()})
    print(f'saved {out}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
