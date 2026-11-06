"""
Exact lower bounds for the coupled contraction rates of Theorems 3 and 4 (R1, internal verification).
Under synchronous coupling and without clipping, the MEAN coupled difference E[Delta_t] obeys a linear recursion:
  FO-JAYA : E[Delta_{t+1}] = (alpha + c_0) E[Delta_t] + sum_{j>=1} c_j E[Delta_{t-j}]     (E[K | past] = 1)
  FO-WFO  : E[Delta_{t+1}] = (alpha Lbar + c_0 I) E[Delta_t] + sum_{j>=1} c_j E[Delta_{t-j}],
            Lbar = p_l (1 - 1/(2N)) I + (1-p_l)[(1-1/n) I + (1/n)(p_e I + (1-p_e)(1/N)(J - I)/(n-1))]
            (laminar: factor 1-s, E s = 1/2, only if k = i; vortex: E[theta cos theta] = 0; cross-layer: 0 if k != i,
             Delta_{j2} with j2 uniform over j != j1 if k = i), c_j = (1-alpha) omega_j / M.
Since E||Delta_t||_1 >= ||E Delta_t||_1, the decay rate of E||Delta_t||_1 is at least the spectral radius rho_mean of
this recursion (for initial differences with a component on the dominant mode). Together with Lemma C:
   rho_mean <= rate of E d(Z_t, Z_t') <= r  (r = Perron root of Theorems 3/4).
"""
import numpy as np
M, N, n, pl, pe = 10, 30, 10, 0.25, 0.2
def omega(a):
    w = [1.0]
    for k in range(1, M): w.append(w[-1] * (k - 1 - a) / k)
    return np.array(w)
def rho_companion(coefs):
    m = len(coefs); C = np.zeros((m, m)); C[0] = coefs; C[1:, :-1] = np.eye(m - 1)
    return max(abs(np.linalg.eigvals(C)))
def perron(a0, a, LW=1.0):
    w = omega(a); coefs = (1 - a) * np.abs(w) / M; coefs[0] = a * LW + (1 - a) / M
    return rho_companion(coefs)
J = np.ones((n, n)); I = np.eye(n)
Lbar = pl * (1 - 1 / (2 * N)) * I + (1 - pl) * ((1 - 1 / n) * I + (1 / n) * (pe * I + (1 - pe) * (1 / N) * (J - I) / (n - 1)))
lam = np.linalg.eigvalsh(Lbar)
LW = pl + (1 - pl) * ((1 - 1 / n) + (1 / n) * (2 * pe + (1 - pe) / N))
print(f'Lbar eigenvalues: {np.round(sorted(set(np.round(lam, 10))), 6)}   L_W = {LW:.4f}')
for a in (0.5, 0.7, 0.9, 1.0):
    w = omega(a); c = (1 - a) * w / M
    if a < 1:
        cj = c.copy(); cj[0] += a; rj = rho_companion(cj); RJ = perron(None, a)
    rw = max(rho_companion(np.r_[a * l + c[0], c[1:]]) for l in lam)
    RW = perron(None, a, LW) if a < 1 else LW
    s = f'alpha={a}: ' + (f'FO-JAYA bracket [{rj:.4f}, {RJ:.4f}]   ' if a < 1 else 'FO-JAYA q_J = 1 (no contraction)   ')
    print(s + f'FO-WFO bracket [{rw:.4f}, {RW:.4f}]')
