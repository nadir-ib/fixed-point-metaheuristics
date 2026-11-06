"""
exp16 — Adversarial validation of FO-PSO (Theorem 2).

FO-PSO (src/algorithms/fo_pso.py), single particle/coordinate, attractor frozen
at 0 (p=g=0):
    v' = w*v - phi_r * x,   phi_r = c1*r1 + c2*r2,  r1,r2 ~ U(0,1)
    x' = x + [ alpha*v' + (1-alpha)/(H-1) * sum_past v ]   (uniform velocity memory)

Manuscript Theorem 2 (expected-value analysis, r_i -> 1/2):
  - joint (x,v) system, Jury conditions;
  - oscillatory-regime spectral radius rho(E[A]) = sqrt(w), INDEPENDENT of alpha;
  - at alpha=1 reduces to the classical Trelea/Clerc-Kennedy/Van den Bergh triangle
    0<w<1, 0<(c1+c2)/2<2(1+w).

Checks (no fabrication):
  C1. alpha=1 expected system: reproduce the classical stability triangle via Jury
      and confirm rho = sqrt(w) in the oscillatory regime.
  C2. THE HONESTY CHECK — multiplicative noise: the TRUE mean-square rate is
      sqrt(rho(E[A⊗A])), which can EXCEED the expected-value rate sqrt(w). Compute
      it, and compute it exactly from the second-moment recursion (R1); report the a.s. rate separately.
  C3. alpha-dependence of the extended expected-value rate (Thm 2 iv): build the augmented
      expected system with uniform velocity memory and report its spectral radius vs alpha.

Refs (verified): Trelea, Information Processing Letters 85(6):317-325 (2003);
Bonyadi & Michalewicz, Evolutionary Computation 25(1):1-54 (2017).

Run from repo root:  python experiments/exp16_pso_validation.py
"""
from __future__ import annotations

import numpy as np

W = 0.7
C1 = C2 = 1.5
H = 10  # velocity-history window


def expected_A(w=W, c1=C1, c2=C2):
    phi = (c1 + c2) / 2.0            # E[phi_r]
    # x' = (1-phi)x + w v ; v' = -phi x + w v
    return np.array([[1 - phi, w], [-phi, w]])


def C1_trelea_triangle():
    print("C1. alpha=1 expected system: Trelea triangle + sqrt(w) oscillatory rate")
    A = expected_A()
    ev = np.linalg.eigvals(A)
    rho = max(abs(ev))
    print(f"   w={W}, c1=c2={C1}: eig={np.round(ev,4)}  |lambda|={rho:.4f}  "
          f"sqrt(w)={np.sqrt(W):.4f}  (complex => oscillatory)")
    # scan the triangle in the manuscript's stated region 0<w<1 (eigenvalue
    # stability actually extends to -1<w<1, so 0<w<1 is a correct practical subset).
    ok = True
    mism = 0
    rng = np.random.RandomState(0)
    for _ in range(40000):
        w = rng.uniform(1e-3, 1 - 1e-3)
        phi = rng.uniform(1e-3, 5.0)
        A = np.array([[1 - phi, w], [-phi, w]])
        stable_eig = max(abs(np.linalg.eigvals(A))) < 1 - 1e-9
        stable_tri = (0 < phi < 2 * (1 + w))          # w in (0,1) already
        if stable_eig != stable_tri:
            ok = False
            mism += 1
    print(f"   in 0<w<1: eigenvalue-stability == (0<phi<2(1+w)) : {ok} "
          f"(mismatches={mism}/40000)")
    print(f"   => {'PASS' if (abs(rho-np.sqrt(W))<1e-9 and ok) else 'FAIL'}\n")


def C2_multiplicative_noise():
    print("C2. Multiplicative-noise honesty check: mean-square rate vs expected-value")
    w, c1, c2 = W, C1, C2
    # E[A] rate
    rho_mean = max(abs(np.linalg.eigvals(expected_A())))
    # E[A⊗A]: A(phi) = [[1-phi, w],[-phi, w]], phi=c1 r1 + c2 r2
    # E over r1,r2 ~ U(0,1): need E[phi]=1.5, E[phi^2]=Var+mean^2
    Ephi = (c1 + c2) / 2.0
    Vphi = (c1**2 + c2**2) / 12.0
    Ephi2 = Vphi + Ephi**2
    # A = A0 + phi*A1, A0=[[1,w],[0,w]], A1=[[-1,0],[-1,0]]
    A0 = np.array([[1.0, w], [0.0, w]])
    A1 = np.array([[-1.0, 0.0], [-1.0, 0.0]])
    # E[A⊗A] = A0⊗A0 + E[phi](A0⊗A1 + A1⊗A0) + E[phi^2](A1⊗A1)
    kron = np.kron
    EAA = (kron(A0, A0) + Ephi * (kron(A0, A1) + kron(A1, A0)) + Ephi2 * kron(A1, A1))
    rho_ms = max(abs(np.linalg.eigvals(EAA)))          # 2nd-moment per-step factor
    rms_rate = np.sqrt(rho_ms)                          # RMS decay per step
    print(f"   expected-value rate rho(E[A]) = {rho_mean:.4f} (= sqrt(w) = {np.sqrt(w):.4f})")
    print(f"   TRUE mean-square: rho(E[A⊗A]) = {rho_ms:.4f}  => RMS rate = {rms_rate:.4f}")
    print(f"   mean-square stable? {rho_ms < 1}   (RMS rate {'<1' if rms_rate<1 else '>=1'})")

    # (R1 fix, CHANGES.md (C003)) The former Monte-Carlo RMS slope was biased low: ||z||^2 is heavy-tailed,
    # so a finite-sample mean drifts toward the almost-sure (Lyapunov) rate (0.933 at N=2e3, 0.942 at 2e4,
    # 0.948 at 2e5). We now (a) iterate the EXACT second-moment recursion
    #     S' = A0 S A0^T + E[phi](A0 S A1^T + A1 S A0^T) + E[phi^2] A1 S A1^T,
    # whose per-step growth gives the mean-square rate, and (b) report the almost-sure rate separately as the
    # exponential of the mean log-growth of one long product (top Lyapunov exponent).
    S = np.eye(2)
    tr = [np.trace(S)]
    for t in range(400):
        S = A0 @ S @ A0.T + Ephi * (A0 @ S @ A1.T + A1 @ S @ A0.T) + Ephi2 * (A1 @ S @ A1.T)
        tr.append(np.trace(S))
    tr = np.array(tr)
    ms_exact = np.sqrt(tr[-1] / tr[-2])
    print(f"   exact second-moment recursion: RMS rate = {ms_exact:.4f}  (matches sqrt(rho(E[A(x)A])) = {rms_rate:.4f})")
    rng = np.random.RandomState(1)
    z = np.array([1.0, 0.0]); logsum = 0.0; T = 400000
    for t in range(T):
        phi = c1 * rng.rand() + c2 * rng.rand()
        z = (A0 + phi * A1) @ z
        nz = np.linalg.norm(z); logsum += np.log(nz); z /= nz
    lyap = np.exp(logsum / T)
    print(f"   almost-sure (Lyapunov) rate, {T} steps = {lyap:.4f}  (a different mode; not a mean-square rate)")
    print(f"   => expected-value sqrt(w)={np.sqrt(w):.4f} UNDERSTATES the mean-square rate "
          f"{rms_rate:.4f} by {100*(rms_rate/np.sqrt(w)-1):.1f}%\n")


def C3_alpha_independence():
    print("C3. alpha-dependence of the extended expected-value rate (uniform velocity memory)")
    w, c1, c2 = W, C1, C2
    phi = (c1 + c2) / 2.0
    # Augmented expected state: z = (x, v_t, v_{t-1}, ..., v_{t-(H-1)})  (H past vels)
    # v' = w v - phi x                                    (expected)
    # x' = x + alpha v' + (1-alpha)/(H-1) * sum_{j=1..H-1} v_{t-j}
    for alpha in [1.0, 0.9, 0.7, 0.5, 0.3]:
        dim = 1 + H
        M = np.zeros((dim, dim))
        # indices: 0=x, 1=v_t(new), 2.. = v_{t-1..}
        # v' (goes to index1): w*v_t - phi*x ; current v_t is at index 1 (previous new)
        # We treat state BEFORE update: x=idx0, v_t=idx1, v_{t-1}=idx2, ...
        vnew = np.zeros(dim); vnew[0] = -phi; vnew[1] = w      # v' = w v_t - phi x
        # x' = x + alpha*v' + (1-alpha)/(H-1)*sum of the H-1 PAST velocities.
        # Code sums velocity_history[:-1] = {v_t, v_{t-1}, ..., v_{t-(H-2)}}
        # = state indices 1..H-1 (NOT the brand-new v'). Fixed off-by-one.
        xnew = np.zeros(dim); xnew[0] = 1.0
        xnew += alpha * vnew
        if alpha < 1.0:
            for j in range(1, H):              # v_t..v_{t-(H-2)} at idx 1..H-1
                xnew[j] += (1 - alpha) / (H - 1)
        M[0, :] = xnew
        M[1, :] = vnew
        for j in range(2, dim):                # shift velocity history
            M[j, j - 1] = 1.0
        rho = max(abs(np.linalg.eigvals(M)))
        print(f"   alpha={alpha}: rho(augmented E-system)={rho:.4f}  sqrt(w)={np.sqrt(w):.4f}")
    print("   (Thm 2(iv), R1: the extended mean-map rate DEPENDS on alpha: 0.809/0.903/0.967 at 0.9/0.7/0.5; unstable below ~0.33)\n")


if __name__ == "__main__":
    C1_trelea_triangle()
    C2_multiplicative_noise()
    C3_alpha_independence()
