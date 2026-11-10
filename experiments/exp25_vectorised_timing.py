"""
R1, reviewer R3-2.3: measured end-to-end runtime of FO-JAYA and FO-WFO with the GL memory term computed for the
whole population at once (Sections 6.2 and 7). The library code in src/ is NOT changed; the vectorised variants are
subclasses defined here and used only for timing.

Why this is exact. In FO-JAYA and FO-WFO the memory term of particle i depends only on the stored history of
particle i, and that history does not change between the start of an iteration and the update of particle i
(FO-JAYA pushes x_i(t) immediately before its own update; FO-WFO pushes x_i after its own selection, i.e. into the
next iteration). Hence the memory terms of all N particles can be computed once at the start of the iteration from a
ring buffer H of shape (M, N, n). The sum over k is accumulated in the same order (k = 0, 1, ..., L-1, starting from
zeros) as in src/, and the blend alpha*u + (1-alpha)*m/L is the same element-wise expression, so every result is
bit-identical; the random-number stream is untouched. This is checked below on every timed run (best value, best
position and convergence curve) and on the shifted and rotated suite.

Protocol as experiments/exp21_timing.py part A: serial, one process, canonical suite, box [-100,100]^n,
n = 10 (30 runs x 5 functions), n = 30 and 50 (5 runs x 5 functions), seeds 42 + run, 500 iterations.
The baselines and the loop implementations are timed again in the same session, interleaved run by run.
Run from repo root on an idle machine:
  OMP_NUM_THREADS=1 python3 experiments/exp25_vectorised_timing.py > results/exp25_vectorised_timing.txt
"""
import os, sys, time, json, platform
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import scipy
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from exp18_r1_benchmark import make_alg, FUNCS, BASE, Objective
from src.algorithms.fo_jaya import FO_JAYA
from src.algorithms.fo_wfo import FO_WFO


def _memory_all(H, L, w):
    """Memory sums of all particles; H[k] = x_{t-k} for k < L, shape (M, N, n). Same summation order as src/."""
    m = np.zeros(H.shape[1:])
    for k in range(L):
        m += w[k] * H[k]
    return m


class FO_JAYA_vec(FO_JAYA):
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        self._initialize()
        N, n, M = self.n_population, self.dim, self.memory_size
        H = np.zeros((M, N, n)); L = 0
        convergence_curve = np.zeros(max_iter)
        for iteration in range(max_iter):
            if iteration == 0:
                for i in range(N):
                    self.fitness[i] = fitness_func(self.positions[i])
            best_idx = np.argmin(self.fitness); worst_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]; self.best_position = self.positions[best_idx].copy()
            self.worst_fitness = self.fitness[worst_idx]; self.worst_position = self.positions[worst_idx].copy()
            # push x(t) of all particles (src/: per particle, just before its own update; equal, see docstring)
            H = np.roll(H, 1, axis=0); H[0] = self.positions; L = min(L + 1, M)
            use_mem = not (self.alpha == 1.0 or L < 2)
            if use_mem:
                mem = _memory_all(H, L, self.weights)
            for i in range(N):
                r1 = np.random.rand(n); r2 = np.random.rand(n)
                jaya_update = (self.positions[i] + r1 * (self.best_position - np.abs(self.positions[i]))
                               - r2 * (self.worst_position - np.abs(self.positions[i])))
                if use_mem:
                    new_position = self.alpha * jaya_update
                    new_position += (1.0 - self.alpha) * mem[i] / L
                else:
                    new_position = jaya_update
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                new_fitness = fitness_func(new_position)
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_position; self.fitness[i] = new_fitness
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]; self.best_position = self.positions[best_idx].copy()
            convergence_curve[iteration] = self.best_fitness
        return self.best_position, self.best_fitness, convergence_curve


class FO_WFO_vec(FO_WFO):
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        N, n, M = self.n_population, self.dim, self.memory_size
        H = np.zeros((M, N, n)); L = 0
        convergence_curve = []
        for i in range(N):
            self.fitness[i] = fitness_func(self.positions[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]; self.best_position = self.positions[i].copy()
        H[0] = self.positions; L = 1
        for iteration in range(max_iter):
            use_mem = not (self.alpha == 1.0 or L < 2)
            if use_mem:
                mem = _memory_all(H, L, self.weights)
            for i in range(N):
                if np.random.rand() < self.pl:
                    wfo_update = self._laminar_flow(i)
                else:
                    wfo_update = self._turbulent_flow(i)
                if use_mem:
                    new_position = self.alpha * wfo_update
                    new_position += (1.0 - self.alpha) * mem[i] / L
                    new_position = self._clip_position(new_position)
                else:
                    new_position = wfo_update
                new_fitness = fitness_func(new_position)
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_position; self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness; self.best_position = new_position.copy()
            # push x(t+1) of all particles (src/: per particle after its own selection; equal, see docstring)
            H = np.roll(H, 1, axis=0); H[0] = self.positions; L = min(L + 1, M)
            convergence_curve.append(self.best_fitness)
        return self.best_position, self.best_fitness, convergence_curve


def make(name, dim, seed):
    if name == 'FO-JAYA(0.5) vec':
        return FO_JAYA_vec(n_population=30, dim=dim, alpha=0.5, memory_size=10, bounds=(-100, 100), seed=seed)
    if name == 'FO-WFO(0.5) vec':
        return FO_WFO_vec(n_population=30, dim=dim, alpha=0.5, memory_size=10, pl=0.25, pe=0.2, bounds=(-100, 100), seed=seed)
    return make_alg(name, dim, seed)


def same(a, b):
    return (np.array_equal(a[0], b[0]) and a[1] == b[1] and np.array_equal(np.asarray(a[2]), np.asarray(b[2])))


GROUPS = [('JAYA', 'FO-JAYA(0.5)', 'FO-JAYA(0.5) vec'), ('WFO', 'FO-WFO(0.5)', 'FO-WFO(0.5) vec')]
OUT = {}

if __name__ == '__main__':
    print('Check 1: bit-identity on the shifted and rotated suite (n = 10 and 30, 3 runs per function)')
    nchk = 0
    for dim in (10, 30):
        for f in FUNCS:
            obj = Objective('shiftrot', f, dim)
            for r in range(3):
                for _, loop, vec in GROUPS:
                    a = make(loop, dim, 42 + r).optimize(obj, max_iter=500)
                    b = make(vec, dim, 42 + r).optimize(obj, max_iter=500)
                    assert same(a, b), (dim, f, r, loop)
                    nchk += 1
    print(f'  {nchk} paired runs: identical best value, best position and convergence curve')

    print('\nTiming: mean wall-clock seconds per run (serial, interleaved), canonical suite')
    for dim, runs in [(10, 30), (30, 5), (50, 5)]:
        ts = {a: [] for g in GROUPS for a in g}; ident = 0
        for f in FUNCS:
            for r in range(runs):
                for g in GROUPS:
                    res = {}
                    for a in g:
                        alg = make(a, dim, 42 + r)
                        t = time.perf_counter(); res[a] = alg.optimize(BASE[f], max_iter=500); ts[a].append(time.perf_counter() - t)
                    assert same(res[g[1]], res[g[2]]), (dim, f, r, g)
                    ident += 1
        row = {a: float(np.mean(v)) for a, v in ts.items()}; OUT[f'n={dim}'] = row
        print(f'  n={dim} ({ident} paired runs, all bit-identical):')
        for base, loop, vec in GROUPS:
            ov_l = row[loop] - row[base]; ov_v = row[vec] - row[base]
            print(f'    {base:5s} {row[base]:.3f} s | {loop} loop {row[loop]:.3f} s (overhead {ov_l:.3f} s, {100*ov_l/row[base]:+.1f}%)'
                  f' | vectorised {row[vec]:.3f} s (overhead {ov_v:.3f} s, {100*ov_v/row[base]:+.1f}%)'
                  f' | overhead removed {100*(1-ov_v/ov_l):.0f}%')
    OUT['platform'] = platform.platform(); OUT['cpu'] = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.machine())
    OUT['python'] = platform.python_version(); OUT['numpy'] = np.__version__; OUT['scipy'] = scipy.__version__
    print(f"\ncpu {OUT['cpu']}; platform {OUT['platform']}; python {OUT['python']}; numpy {OUT['numpy']}; scipy {OUT['scipy']}")
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'data', 'exp25_vectorised_timing.json'), 'w'), indent=1)
