"""
Verification suite for the revised mathematics (Revision R1).
Every number that will appear in the revised manuscript is computed here.
Run from the repository root:  python3 math/verify_math.py
Exit code != 0 if any check fails.
"""
import sys, math, numpy as np
from math import gamma
sys.path.insert(0, '.')
FAIL = []
def check(name, cond, detail=''):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ''))
    if not cond: FAIL.append(name)

ALPHAS = (0.3, 0.5, 0.7, 0.9)
def gl(a, M):                       # recurrence used in the code
    w = [1.0]
    for j in range(1, M): w.append(w[-1] * (j - 1 - a) / j)
    return np.array(w)
def S(a, M): return gl(a, M).sum()
def A(a, M): return np.abs(gl(a, M)).sum()
def S_closed(a, M): return gamma(M - a) / (gamma(1 - a) * gamma(M))

print("T1  GL finite-memory coefficient facts")
ok_sign = ok_cf = ok_A = ok_mono = True
for a in np.linspace(0.05, 0.95, 19):
    for M in range(1, 41):
        w = gl(a, M)
        if M > 1 and not (np.all(w[1:] < 0) and np.all(np.diff(np.abs(w)) < 0)): ok_sign = False
        if abs(S(a, M) - S_closed(a, M)) > 1e-12: ok_cf = False
        if abs(A(a, M) - (2 - S(a, M))) > 1e-12 or not (1 <= A(a, M) < 2): ok_A = False
        if M > 1 and not (0 < S(a, M) < S(a, M - 1)): ok_mono = False
check("w_0=1, w_j<0 and |w_j| strictly decreasing for j>=1", ok_sign)
check("S_M = sum_{j<M} w_j = Gamma(M-a)/(Gamma(1-a)Gamma(M))  (closed form)", ok_cf)
check("A_M = sum|w_j| = 2 - S_M, 1 <= A_M < 2", ok_A)
check("0 < S_M strictly decreasing in M", ok_mono)
print("      M=10 table:")
for a in ALPHAS: print(f"      alpha={a}: S_10={S(a,10):.4f}  A_10={A(a,10):.4f}")

print("T2  translation factor tau and fixed-point gain kappa (M=10)")
for a in ALPHAS:
    tau = a + (1 - a) * S(a, 10) / 10
    kap = a / (1 - (1 - a) * S(a, 10) / 10)
    print(f"      alpha={a}: tau={tau:.4f}  kappa={kap:.4f}")
check("tau(0.5,10) = 0.5093 (R1 predicted ~0.51)", abs(0.5 + 0.5 * S(0.5, 10) / 10 - 0.5093) < 5e-5)
check("tau = 1 when M=1 (no bias without memory window)", abs(0.5 + 0.5 * S(0.5, 1) / 1 - 1) < 1e-12)

print("T3  Lemma 0: homogeneous memory recursion x_{t+1}=sum c_j x_{t-j}, c_j=(1-a)w_j/M")
def companion(c):
    M = len(c); C = np.zeros((M, M)); C[0, :] = c
    for i in range(1, M): C[i, i - 1] = 1
    return C
ok = True
for a in np.linspace(0.01, 0.99, 50):
    for M in range(1, 31):
        c = (1 - a) * gl(a, M) / M
        if not (np.abs(c).sum() < 1 and max(abs(np.linalg.eigvals(companion(c)))) < 1): ok = False
check("sum|c_j| < 1 and rho(C) < 1 for all a in (0,1), M=1..30", ok)
for a in ALPHAS:
    c = (1 - a) * gl(a, 10) / 10
    print(f"      alpha={a}: sum|c|={np.abs(c).sum():.4f}  rho(C)={max(abs(np.linalg.eigvals(companion(c)))):.4f}")

print("T4  nonnegative comparison recursion: q <= r <= q^(1/M)")
def perron(a0, rest):
    coeff = np.r_[a0, rest]; C = companion(coeff)
    ev, lv = np.linalg.eig(C.T); i = np.argmax(ev.real); r = ev[i].real
    u = np.abs(lv[:, i].real); return r, u
def q_and_r(a, L, M=10):
    w = gl(a, M); cabs = (1 - a) * np.abs(w) / M
    a0 = a * L + cabs[0]; q = a0 + cabs[1:].sum()
    r, u = perron(a0, cabs[1:]); return q, r, u
ok = True
for a in np.linspace(0.05, 0.95, 19):
    for L in (0.0, 0.5, 0.9, 1.0):
        q, r, u = q_and_r(a, L)
        if q < 1 and not (q - 1e-12 <= r <= q ** (1 / 10) + 1e-12): ok = False
        if q < 1 and not np.all(u > 0): ok = False
check("q <= r <= q^(1/M), left Perron vector strictly positive", ok)

print("T5  WFO vortex factor  U = theta*cos(theta), theta~U(-pi,pi)")
th = np.linspace(-np.pi, np.pi, 2_000_001); Uv = th * np.cos(th)
EU = np.trapezoid(np.abs(Uv), th) / (2 * np.pi)
check("E|theta cos theta| = 1 (analytic)", abs(EU - 1) < 1e-6, f"numeric {EU:.7f}")
cs = [np.trapezoid(np.abs(1 + s * Uv), th) / (2 * np.pi) for s in np.linspace(-1, 1, 201)]
c_star = max(cs)
check("c* = sup_{|s|<=1} E|1+sU| <= 2", c_star <= 2, f"c* = {c_star:.4f}")
pl, pe, N, n = 0.25, 0.2, 30, 10
LW_bound = pl + (1 - pl) * ((1 - 1 / n) + (1 / n) * (pe * c_star + (1 - pe) / N))
LW_crude = pl + (1 - pl) * ((1 - 1 / n) + (1 / n) * (pe * 2 + (1 - pe) / N))
print(f"      L_W (with c*) = {LW_bound:.4f};  L_W (with c<=2) = {LW_crude:.4f}")

print("T6  contraction constants q and rates r (M=10)")
rows = []
for a in ALPHAS:
    qJ, rJ, _ = q_and_r(a, 1.0); qW, rW, _ = q_and_r(a, LW_crude); qA, rA, _ = q_and_r(a, 0.0)
    rows.append((a, qJ, rJ, qW, rW, qA, rA))
    print(f"      alpha={a}: JAYA q={qJ:.4f} r={rJ:.4f} | WFO q={qW:.4f} r={rW:.4f} | ACOR q={qA:.4f} r={rA:.4f}")
check("all q < 1 for alpha in {0.3,...,0.9}", all(max(r[1], r[3], r[5]) < 1 for r in rows))
qJ1 = 1.0 + 0 * 0; check("JAYA at alpha=1 has q = 1 (no contraction without memory)", abs(q_and_r(1.0, 1.0)[0] - 1) < 1e-12)
# transient window t<M: worst row sum is at window 2 (memory active from L=2)
for a in ALPHAS:
    worst = max(a * 1.0 + (1 - a) * A(a, L) / L for L in range(2, 11))
    check(f"JAYA transient worst q over windows L=2..10 < 1 (alpha={a})", worst < 1, f"{worst:.4f} at L=2 -> {(a+(1-a)*(1+a)/2):.4f}")

print("T7  Monte Carlo: coupled FO-JAYA candidate map (code-faithful, frozen best/worst, no selection)")
def fit_rate(D):
    D = np.asarray(D); D0 = D[0]
    ok = np.where((D > 1e-9 * D0) & (D < 1e-1 * D0))[0]      # far above machine eps, past the first transient
    if len(ok) < 15: return float('nan'), 0
    k = ok[:]; return math.exp(np.polyfit(k, np.log(D[k]), 1)[0]), len(k)
def run_jaya_coupled(a, drift_scale, T=200, R=20000, n=10, seed=0):
    rng = np.random.default_rng(seed); w = gl(a, 10)
    b = rng.uniform(-drift_scale, drift_scale, n); wo = rng.uniform(-drift_scale, drift_scale, n)
    X = rng.uniform(-20, 20, (R, n)); Y = X + rng.normal(0, 1.0, (R, n))
    HX = [X.copy()]; HY = [Y.copy()]; D = []; clipped = 0
    for t in range(T):
        r1 = rng.random((R, n)); r2 = rng.random((R, n))
        ux = X + r1 * (b - np.abs(X)) - r2 * (wo - np.abs(X)); uy = Y + r1 * (b - np.abs(Y)) - r2 * (wo - np.abs(Y))
        L = len(HX)
        if L >= 2:
            nx = a * ux + (1 - a) * sum(w[k] * HX[-(k + 1)] for k in range(L)) / L
            ny = a * uy + (1 - a) * sum(w[k] * HY[-(k + 1)] for k in range(L)) / L
        else: nx, ny = ux, uy
        clipped += int((np.abs(nx) > 100).sum() + (np.abs(ny) > 100).sum())
        X = np.clip(nx, -100, 100); Y = np.clip(ny, -100, 100)
        HX.append(X.copy()); HY.append(Y.copy()); HX = HX[-10:]; HY = HY[-10:]
        D.append(np.abs(X - Y).sum(axis=1).mean())
    return D, clipped
for a, ds in ((0.5, 20.0), (0.7, 5.0), (0.9, 1.0)):
    D, clipped = run_jaya_coupled(a, ds); rate, npts = fit_rate(D); q, r, _ = q_and_r(a, 1.0)
    check(f"FO-JAYA empirical coupled rate <= r (alpha={a}; clip events={clipped})", rate <= r + 1e-3,   # projection is part of the theorem (1-Lipschitz), so clipping does not void the bound

          f"empirical {rate:.4f} vs r={r:.4f} (q={q:.4f}), fit on {npts} pts")

print("T7b Monte Carlo: coupled FO-WFO candidate map (code-faithful operators, frozen population, no selection)")
def run_wfo_coupled(a, T=200, R=20000, n=10, N=30, pl=0.25, pe=0.2, seed=5):
    rng = np.random.default_rng(seed); w = gl(a, 10)
    P = rng.uniform(-20, 20, (N, n)); best = P[0].copy(); i_self = 3     # frozen population; particle i_self is the coupled one
    X = np.tile(P[i_self], (R, 1)) + rng.normal(0, 1, (R, n)); Y = X + rng.normal(0, 1, (R, n))
    HX = [X.copy()]; HY = [Y.copy()]; D = []; clipped = 0; rows = np.arange(R)
    for t in range(T):
        lam = rng.random(R) < pl; s_ = rng.random(R); k = rng.integers(0, N, R)
        j1 = rng.integers(0, n, R); vort = rng.random(R) < pe; th = rng.uniform(-np.pi, np.pi, R)
        j2 = (j1 + rng.integers(1, n, R)) % n                                  # uniform over j != j1
        def base(Z):
            Pk = P[k].copy(); self_k = (k == i_self); Pk[self_k] = Z[self_k]   # if k == i, the 'other' particle is itself
            d = best - Pk; d[(Pk == best).all(1)] = 0
            U = Z.copy()
            U[lam] = Z[lam] + s_[lam, None] * d[lam]
            tb = ~lam
            zv = Z[rows, j1]; pk = Pk[rows, j1]
            vort_val = zv + np.abs(zv - pk) * th * np.cos(th)
            cross_val = Pk[rows, j2]
            newj = np.where(vort, vort_val, cross_val)
            U[rows[tb], j1[tb]] = newj[tb]
            return np.clip(U, -100, 100)
        ux, uy = base(X), base(Y); L = len(HX)
        if L >= 2:
            nx = a * ux + (1 - a) * sum(w[kk] * HX[-(kk + 1)] for kk in range(L)) / L
            ny = a * uy + (1 - a) * sum(w[kk] * HY[-(kk + 1)] for kk in range(L)) / L
        else: nx, ny = ux, uy
        clipped += int((np.abs(nx) > 100).sum() + (np.abs(ny) > 100).sum())
        X = np.clip(nx, -100, 100); Y = np.clip(ny, -100, 100)
        HX.append(X.copy()); HY.append(Y.copy()); HX = HX[-10:]; HY = HY[-10:]
        D.append(np.abs(X - Y).sum(axis=1).mean())
    return D, clipped
for a in (0.5, 0.9, 1.0):
    D, clipped = run_wfo_coupled(a); rate, npts = fit_rate(D); q, r, _ = q_and_r(a, LW_crude)
    check(f"FO-WFO empirical coupled rate <= r (alpha={a}; clip events={clipped})", rate <= r + 1e-3,
          f"empirical {rate:.4f} vs r={r:.4f} (q={q:.4f}, L_W<= {LW_crude:.4f})")

print("T8  Monte Carlo: ACTUAL FO_ACOR class, frozen archive -> mean of generated solutions = kappa * mixture mean")
from src.algorithms.fo_acor import FO_ACOR
for a in (0.5, 0.9):
    alg = FO_ACOR(n_ants=30, dim=10, alpha=a, memory_size=10, bounds=(-100, 100), seed=7)
    rng = np.random.default_rng(1)
    centre = np.full(10, 40.0)
    alg.archive = [centre + rng.normal(0, 2.0, 10) for _ in range(alg.archive_size)]
    alg.archive_fitness = list(np.arange(alg.archive_size, dtype=float))
    wp = alg._compute_weights_prob(); sd = alg._compute_standard_deviations()
    mbar = sum(wp[l] * alg.archive[l] for l in range(alg.archive_size))
    gen = []
    for it in range(60000):
        s = alg._generate_solution(wp, sd); x = alg._apply_fractional_memory(s)
        alg.solution_history.append(x.copy())
        if len(alg.solution_history) > alg.memory_size: alg.solution_history.pop(0)
        if it > 200: gen.append(x)
    gen = np.array(gen); kap = a / (1 - (1 - a) * S(a, 10) / 10)
    emp = gen.mean(0).mean() / mbar.mean()
    check(f"generated mean / archive-mixture mean = kappa (alpha={a})", abs(emp - kap) < 0.01,
          f"empirical {emp:.4f}, kappa {kap:.4f}, mixture mean {mbar.mean():.2f}, generated mean {gen.mean():.2f}")

print("T9  Lemma M (FO-ACOR, frozen archive): stationary variance factor and non-normal transient")
for a in ALPHAS:
    c = (1 - a) * gl(a, 10) / 10; C = companion(c)
    Ck = np.eye(10); s11 = 0.0; supn = 0.0
    for k in range(2000):
        s11 += Ck[0, 0] ** 2; supn = max(supn, np.linalg.norm(Ck, 2)); Ck = C @ Ck
    print(f"      alpha={a}: sum_k (C^k)_11^2 = {s11:.4f}  -> stationary sd = alpha*sigma*{math.sqrt(s11):.4f};  "
          f"sup_k ||C^k||_2 = {supn:.4f};  ||C||_2 = {np.linalg.norm(C,2):.4f}")

print("T10 FO-PSO: Theorem 2 numbers and the mean-square claim of P172")
w_, c1, c2, Mv = 0.7, 1.5, 1.5, 10; phi = (c1 + c2) / 2
def pso_ext(a):
    n_ = 1 + (Mv - 1); Mx = np.zeros((n_, n_)); vnew = np.zeros(n_); vnew[0] = -phi; vnew[1] = w_
    Mx[0, :] = a * vnew; Mx[0, 0] += 1; Mx[0, 1:] += (1 - a) / (Mv - 1); Mx[1, :] = vnew
    for k in range(2, n_): Mx[k, k - 1] = 1
    return max(abs(np.linalg.eigvals(Mx)))
al = np.linspace(0.2, 1, 8001); rs = np.array([pso_ext(x) for x in al]); i = rs.argmin()
cross = al[np.where((rs[:-1] >= 1) & (rs[1:] < 1))[0][0] + 1]
check("min rho ~0.78 near alpha~0.86", abs(rs[i] - 0.783) < 0.002 and abs(al[i] - 0.857) < 0.005, f"{rs[i]:.4f} at {al[i]:.4f}")
check("rho(0.9)=0.81, rho(0.7)=0.90, rho(0.5)=0.97", abs(pso_ext(0.9)-0.81)<0.005 and abs(pso_ext(0.7)-0.90)<0.005 and abs(pso_ext(0.5)-0.97)<0.005,
      f"{pso_ext(0.9):.4f}, {pso_ext(0.7):.4f}, {pso_ext(0.5):.4f}")
check("instability threshold ~0.33", abs(cross - 0.329) < 0.003, f"{cross:.4f}")
A2 = np.array([[1 - 0.9 * phi, 0.9 * w_], [-phi, w_]])
check("||A||_2 at alpha=0.9 = 1.76", abs(np.linalg.norm(A2, 2) - 1.76) < 0.005, f"{np.linalg.norm(A2,2):.4f}")
# mean-square: the submitted text quotes 0.96 in the alpha=0.9 context; that value belongs to alpha=1, memoryless.
Ephi = 1.5; Ephi2 = (c1**2 + c2**2) / 12 + Ephi**2
def extB(a, memory):
    n_ = 10 if memory else 2; B0 = np.zeros((n_, n_)); B1 = np.zeros((n_, n_))
    B0[1, 1] = w_; B1[1, 0] = -1; B0[0, 0] = 1; B0[0, 1] = a * w_; B1[0, 0] = -a
    if memory:
        B0[0, 1:] += (1 - a) / (Mv - 1)
        for k in range(2, n_): B0[k, k - 1] = 1
    return B0, B1
def msrate(a, memory):
    B0, B1 = extB(a, memory); K = np.kron
    E = K(B0, B0) + Ephi * (K(B0, B1) + K(B1, B0)) + Ephi2 * K(B1, B1)
    return math.sqrt(max(abs(np.linalg.eigvals(E))))
check("submitted '0.96' equals the alpha=1 memoryless mean-square rate", abs(msrate(1.0, False) - 0.9568) < 1e-3, f"{msrate(1.0,False):.4f}")
check("correct FO-PSO (alpha=0.9, with memory) mean-square rate = 0.937", abs(msrate(0.9, True) - 0.9374) < 1e-3, f"{msrate(0.9,True):.4f}")

print("T11 Lemma M radius: exact factor vs submitted 1/sqrt(1-rho(C)^2)")
for a in ALPHAS:
    c = (1 - a) * gl(a, 10) / 10; C = companion(c); rho = max(abs(np.linalg.eigvals(C)))
    Ck = np.eye(10); s11 = 0.0
    for k in range(2000): s11 += Ck[0, 0] ** 2; Ck = C @ Ck
    exact, sub = math.sqrt(s11), 1 / math.sqrt(1 - rho ** 2)
    print(f"      alpha={a}: exact {exact:.4f}   submitted-style {sub:.4f}")
    check(f"submitted radius is a valid upper bound (alpha={a})", sub >= exact)

print()
print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL: {FAIL}")
sys.exit(1 if FAIL else 0)
