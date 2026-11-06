"""
Caputo Fractional Calculus Implementation

This module implements the Caputo fractional derivative and integral operators
used in fractional-order swarm intelligence algorithms.

Mathematical Definitions:
- Caputo fractional integral: D^(-α) f(t) = (1/Γ(α)) ∫[0,t] (t-τ)^(α-1) f(τ) dτ
- Caputo fractional derivative: D^α f(t) = (1/Γ(n-α)) ∫[0,t] (t-τ)^(n-α-1) f^(n)(τ) dτ
- Mittag-Leffler function: E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from functools import lru_cache


def caputo_integral(f_values, alpha, dt=1.0):
    """
    Compute Caputo fractional integral of order alpha.

    Uses simplified Riemann-Liouville approximation for discrete-time implementation:
    D^(-α) f(t_n) ≈ (dt^α / Γ(α+1)) Σ_{j=0}^n (n-j+1)^α - (n-j)^α) * f(t_j)

    This is a more stable approximation that avoids numerical overflow.

    Parameters:
    -----------
    f_values : array-like
        Function values at discrete time points [f(t_0), f(t_1), ..., f(t_n)]
    alpha : float
        Fractional order, 0 < alpha <= 1
    dt : float, optional
        Time step (default: 1.0)

    Returns:
    --------
    result : float or ndarray
        Caputo fractional integral at time t_n

    References:
    -----------
    [1] Podlubny, I. (1999). Fractional Differential Equations. Academic Press.
    [2] Li, C., & Zeng, F. (2015). Numerical Methods for Fractional Calculus. CRC Press.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in (0, 1]")

    if alpha == 1.0:
        # Standard integral (cumulative sum)
        return np.cumsum(f_values) * dt

    # Convert to numpy array
    f_values = np.atleast_1d(f_values)
    n = len(f_values)

    # Compute fractional integral using Riemann-Liouville approximation
    result = np.zeros_like(f_values, dtype=float)

    # Normalization factor
    norm_factor = (dt**alpha) / gamma(alpha + 1)

    for i in range(n):
        # Compute weights for this time step
        for j in range(i + 1):
            weight = (i - j + 1)**alpha - (i - j)**alpha
            result[i] += weight * f_values[j]

        result[i] *= norm_factor

    return result


@lru_cache(maxsize=128)
def _compute_gl_weights(n, alpha):
    """
    Compute Grünwald-Letnikov weights for fractional integral.
    
    Parameters:
    -----------
    n : int
        Number of weights to compute
    alpha : float
        Fractional order
    
    Returns:
    --------
    weights : ndarray
        Array of GL weights [w_0, w_1, ..., w_{n-1}]
    """
    weights = np.zeros(n)
    weights[0] = 1.0
    
    for j in range(1, n):
        weights[j] = (1.0 - (alpha + 1.0) / j) * weights[j-1]
    
    return weights


def caputo_derivative(f_values, alpha, dt=1.0):
    """
    Compute Caputo fractional derivative of order alpha.
    
    For 0 < alpha < 1:
    D^α f(t) = D^(-(1-α)) [df/dt]
    
    Parameters:
    -----------
    f_values : array-like
        Function values at discrete time points
    alpha : float
        Fractional order, 0 < alpha < 1
    dt : float, optional
        Time step (default: 1.0)
    
    Returns:
    --------
    result : ndarray
        Caputo fractional derivative
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be in (0, 1)")
    
    # Compute first derivative using finite differences
    f_values = np.atleast_1d(f_values)
    df_dt = np.gradient(f_values, dt)
    
    # Apply fractional integral of order (1-alpha) to the derivative
    result = caputo_integral(df_dt, 1.0 - alpha, dt)
    
    return result


def mittag_leffler(z, alpha, beta=1.0, terms=100):
    """
    Compute Mittag-Leffler function E_{α,β}(z).
    
    E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)
    
    Parameters:
    -----------
    z : float or complex
        Argument
    alpha : float
        Parameter α > 0
    beta : float, optional
        Parameter β (default: 1.0)
    terms : int, optional
        Number of terms in series (default: 100)
    
    Returns:
    --------
    result : float or complex
        Mittag-Leffler function value
    
    References:
    -----------
    [1] Gorenflo, R., Kilbas, A. A., Mainardi, F., & Rogosin, S. V. (2014).
        Mittag-Leffler Functions, Related Topics and Applications. Springer.
    """
    result = 0.0
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    
    return result


# Vectorized versions for efficiency
caputo_integral_vec = np.vectorize(caputo_integral, excluded=['alpha', 'dt'])
caputo_derivative_vec = np.vectorize(caputo_derivative, excluded=['alpha', 'dt'])
mittag_leffler_vec = np.vectorize(mittag_leffler, excluded=['alpha', 'beta', 'terms'])

