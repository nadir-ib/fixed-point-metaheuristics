"""R4-5: quantitative memory-strength measures for the finite-memory filter
x_{t+1} = alpha*u + (1-alpha)/m * sum_{j<m} omega_j x_{t-j}  (code: m = current history length <= M = 10)."""
import numpy as np
from math import gamma
def om(a, m):
    w = [1.0]
    for j in range(1, m): w.append(w[-1] * (1 - (a + 1) / j))
    return np.array(w)
grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
print('alpha  |w1|    |w5|    |w9|   total_abs_on_history=(1-a)A_m/m  weight_on_x_t=(1-a)/m  weight_on_older=(1-a)(A_m-1)/m')
for a in [0.1,0.3,0.5,0.7,0.9]:
    w = om(a, 10); A = np.abs(w).sum()
    print(f'{a:.1f}  {abs(w[1]):.4f} {abs(w[5]):.4f} {abs(w[9]):.4f}   {(1-a)*A/10:.4f}   {(1-a)/10:.4f}   {(1-a)*(A-1)/10:.4f}')
# monotonicity of total absolute history weight in alpha, for every m = 2..30
ok = True
for m in range(2, 31):
    a = np.linspace(0.001, 0.999, 2000)
    h = np.array([(1-x)*np.abs(om(x, m)).sum()/m for x in a])
    if not np.all(np.diff(h) < 0): ok = False; print('NOT monotone, m =', m)
print('total |weight| on stored states strictly decreasing in alpha for all m=2..30:', ok)
# older-iterate weight non-monotone?
a = np.linspace(0.001, 0.999, 2000)
h = np.array([(1-x)*(np.abs(om(x, 10)).sum()-1)/10 for x in a])
print('weight on older iterates (j>=1), m=10: argmax alpha = %.3f, max = %.4f' % (a[h.argmax()], h.max()))
# asymptotic decay |omega_j| ~ j^{-(1+alpha)}/|Gamma(-alpha)|
for a in [0.3, 0.5, 0.9]:
    w = om(a, 2001); j = 2000
    print(f'alpha {a}: |w_2000| * 2000^(1+a) * |Gamma(-a)| = {abs(w[j])*j**(1+a)*abs(gamma(-a)):.4f} (->1)')
print('--- part 2')
for m in (2, 5, 10, 20):
    a = np.linspace(0.0005, 0.9995, 20000)
    h = np.array([(1-x)*np.abs(om(x, m)).sum()/m for x in a])
    k = h.argmax(); dec = np.all(np.diff(h[k:]) < 0)
    print(f'm={m}: total history weight max at alpha={a[k]:.4f} (value {h[k]:.4f}); strictly decreasing for alpha above it: {dec}')
