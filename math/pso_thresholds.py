"""FO-PSO thresholds for Theorem 2(iv) (R1, internal verification). Uses the matrices of pso_mean_square_check.py.
(a) interval of alpha where the extended mean-map rate is below sqrt(w); (b) minimum of the mean-map rate;
(c) alpha below which the mean-map rate exceeds one; (d) the same for the mean-square rate sqrt(rho(E[B(x)B]))."""
import numpy as np
from scipy.optimize import brentq
import pso_mean_square_check as P
def mm(a): B0, B1 = P.mats(a, True); return max(abs(np.linalg.eigvals(B0 + P.Ephi * B1)))
def ms(a):
    B0, B1 = P.mats(a, True)
    EBB = np.kron(B0, B0) + P.Ephi * (np.kron(B0, B1) + np.kron(B1, B0)) + P.Ephi2 * np.kron(B1, B1)
    return np.sqrt(max(abs(np.linalg.eigvals(EBB))))
sw = np.sqrt(P.w)
g = np.linspace(0.2, 0.999, 8000); v = np.array([mm(a) for a in g])
lo = brentq(lambda a: mm(a) - sw, 0.7, 0.85)
print(f'(a) mean-map rate < sqrt(w) = {sw:.4f} for alpha in ({lo:.4f}, 1)')
k = v.argmin(); print(f'(b) minimum mean-map rate {v[k]:.4f} at alpha = {g[k]:.4f}')
print(f'(c) mean-map rate = 1 at alpha = {brentq(lambda a: mm(a) - 1, 0.25, 0.4):.4f}')
print(f'(d) mean-square rate = 1 at alpha = {brentq(lambda a: ms(a) - 1, 0.25, 0.5):.4f};  ms(0.9) = {ms(0.9):.4f}, ms(0.7) = {ms(0.7):.4f}, ms(0.5) = {ms(0.5):.4f}')
gm = np.linspace(0.35, 0.999, 3000); vm = np.array([ms(a) for a in gm]); km = vm.argmin()
print(f'    minimum mean-square rate {vm[km]:.4f} at alpha = {gm[km]:.4f}')
