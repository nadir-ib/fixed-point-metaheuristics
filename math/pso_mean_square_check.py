"""FO-PSO linearised stochastic dynamics (per coordinate, p_i = g frozen, i.e. common attractor x*):
  v_{t+1} = w v_t - phi_t e_t,  phi_t = c1 r1 + c2 r2,  r1, r2 ~ U(0,1) i.i.d. per step and coordinate
  e_{t+1} = e_t + alpha v_{t+1} + (1-alpha)/(M-1) * sum_{j=1}^{M-1} v_{t-j+1}     (Eq. (1); code fo_pso.py 147-178)
State s_t = (e_t, v_t, v_{t-1}, ..., v_{t-M+2}). B_t = B0 + phi_t B1 (affine in phi_t).
Outputs: mean-map rate rho(E B), mean-square rate sqrt(rho(E[B (x) B])) (exact, from E phi and E phi^2),
and the almost-sure rate exp(top Lyapunov exponent) by QR iteration. Settings w=0.7, c1=c2=1.5, M=10 (Table 4)."""
import numpy as np
w, c1, c2, M = 0.7, 1.5, 1.5, 10
Ephi = (c1 + c2) / 2; Ephi2 = c1**2 / 12 + c2**2 / 12 + Ephi**2
def mats(a, memory=True):
    n = M if memory else 2
    B0 = np.zeros((n, n)); B1 = np.zeros((n, n))
    # v_{t+1} row (index 1): w v_t - phi e_t
    B0[1, 1] = w; B1[1, 0] = -1
    # e_{t+1} = e_t + a v_{t+1} + mem
    B0[0, 0] = 1; B0[0, 1] += a * w; B1[0, 0] += -a
    if memory:
        for k in range(1, n): B0[0, k] += (1 - a) / (M - 1)   # v_t, ..., v_{t-M+2}: M-1 past velocities
        for k in range(2, n): B0[k, k - 1] = 1                 # shift
    return B0, B1
def rates(a, memory=True, T=400000, seed=0):
    B0, B1 = mats(a, memory)
    EB = B0 + Ephi * B1
    EBB = np.kron(B0, B0) + Ephi * (np.kron(B0, B1) + np.kron(B1, B0)) + Ephi2 * np.kron(B1, B1)
    rng = np.random.default_rng(seed); Q = np.eye(B0.shape[0]); s = 0.0
    phis = c1 * rng.random(T) + c2 * rng.random(T)
    for t in range(T):
        Q, R = np.linalg.qr((B0 + phis[t] * B1) @ Q); s += np.log(abs(R[0, 0]))
    # R[0,0] after QR on the first column tracks the top exponent
    return max(abs(np.linalg.eigvals(EB))), np.sqrt(max(abs(np.linalg.eigvals(EBB)))), np.exp(s / T)
if __name__ == '__main__':
    for lab, a, mem in [('alpha=1.0 memoryless', 1.0, False), ('alpha=0.9 with velocity memory (FO-PSO)', 0.9, True),
                        ('alpha=0.7 with velocity memory', 0.7, True), ('alpha=0.5 with velocity memory', 0.5, True)]:
        r1, r2, r3 = rates(a, mem)
        print(f'{lab:42s} mean-map rho(E B) = {r1:.4f} | mean-square sqrt(rho(E[B(x)B])) = {r2:.4f} | a.s. exp(lyap) = {r3:.4f}')
