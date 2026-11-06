"""
Standard Benchmark Functions for Optimization

This module implements commonly used benchmark functions for testing
optimization algorithms. All functions are minimization problems.

References:
-----------
[1] Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions
    for global optimisation problems. International Journal of Mathematical
    Modelling and Numerical Optimisation, 4(2), 150-194.
[2] Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation
    Experiments: Test Functions and Datasets.
"""

import numpy as np


def sphere(x):
    """
    Sphere function (unimodal, convex, separable)
    
    f(x) = Σ x_i²
    
    Global minimum: f(0, ..., 0) = 0
    Domain: [-100, 100]^n
    
    Properties:
    - Unimodal: Single global minimum
    - Convex: Easy to optimize
    - Separable: Can be optimized dimension by dimension
    """
    return np.sum(x**2)


def rastrigin(x):
    """
    Rastrigin function (highly multimodal, non-convex, separable)
    
    f(x) = 10n + Σ [x_i² - 10cos(2πx_i)]
    
    Global minimum: f(0, ..., 0) = 0
    Domain: [-5.12, 5.12]^n
    
    Properties:
    - Highly multimodal: Many local minima
    - Non-convex: Difficult to optimize
    - Separable: Can be optimized dimension by dimension
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    """
    Ackley function (multimodal, non-convex, non-separable)
    
    f(x) = -20exp(-0.2√(Σx_i²/n)) - exp(Σcos(2πx_i)/n) + 20 + e
    
    Global minimum: f(0, ..., 0) = 0
    Domain: [-32, 32]^n
    
    Properties:
    - Multimodal: Many local minima
    - Non-convex: Difficult to optimize
    - Non-separable: Dimensions are coupled
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    
    return term1 + term2 + 20 + np.e


def rosenbrock(x):
    """
    Rosenbrock function (unimodal, non-convex, non-separable)
    
    f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]
    
    Global minimum: f(1, ..., 1) = 0
    Domain: [-5, 10]^n
    
    Properties:
    - Unimodal: Single global minimum
    - Non-convex: Narrow curved valley
    - Non-separable: Dimensions are coupled
    - Difficult: Valley is easy to find but minimum is hard to reach
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def griewank(x):
    """
    Griewank function (multimodal, non-convex, non-separable)
    
    f(x) = Σ x_i²/4000 - Π cos(x_i/√i) + 1
    
    Global minimum: f(0, ..., 0) = 0
    Domain: [-600, 600]^n
    
    Properties:
    - Multimodal: Many local minima
    - Non-convex: Difficult to optimize
    - Non-separable: Dimensions are coupled
    """
    n = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    
    return sum_term - prod_term + 1


# Function metadata
FUNCTION_INFO = {
    'sphere': {
        'name': 'Sphere',
        'bounds': [-100, 100],
        'global_optimum': 0.0,
        'optimal_x': lambda dim: np.zeros(dim),
        'properties': ['unimodal', 'convex', 'separable']
    },
    'rastrigin': {
        'name': 'Rastrigin',
        'bounds': [-5.12, 5.12],
        'global_optimum': 0.0,
        'optimal_x': lambda dim: np.zeros(dim),
        'properties': ['multimodal', 'non-convex', 'separable']
    },
    'ackley': {
        'name': 'Ackley',
        'bounds': [-32, 32],
        'global_optimum': 0.0,
        'optimal_x': lambda dim: np.zeros(dim),
        'properties': ['multimodal', 'non-convex', 'non-separable']
    },
    'rosenbrock': {
        'name': 'Rosenbrock',
        'bounds': [-5, 10],
        'global_optimum': 0.0,
        'optimal_x': lambda dim: np.ones(dim),
        'properties': ['unimodal', 'non-convex', 'non-separable']
    },
    'griewank': {
        'name': 'Griewank',
        'bounds': [-600, 600],
        'global_optimum': 0.0,
        'optimal_x': lambda dim: np.zeros(dim),
        'properties': ['multimodal', 'non-convex', 'non-separable']
    }
}


def get_bounds(func_name):
    """Get search space bounds for a function."""
    return FUNCTION_INFO[func_name]['bounds']


def get_global_optimum(func_name):
    """Get global optimum value for a function."""
    return FUNCTION_INFO[func_name]['global_optimum']


def get_function_info(func_name):
    """Get complete information about a function."""
    return FUNCTION_INFO[func_name]

