"""Shape of h(alpha, M) = (1-alpha) A_M / M for M = 10 (R1, internal verification)."""
import numpy as np
M = 10
def h(a):
    S = np.prod([1 - a / i for i in range(1, M)]); return (1 - a) * (2 - S) / M
g = np.linspace(1e-6, 1 - 1e-6, 200001); H = np.array([h(a) for a in g]); k = H.argmax()
print(f'h(0+) = {h(1e-12):.4f} (= 1/M), max h = {H[k]:.4f} at alpha = {g[k]:.4f}')
print('strictly increasing on (0, alpha*):', bool(np.all(np.diff(H[:k + 1]) > 0)))
print('strictly decreasing on (alpha*, 1):', bool(np.all(np.diff(H[k:]) < 0)))
