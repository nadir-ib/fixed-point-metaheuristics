"""
Fractional-Order JAYA Algorithm (FO-JAYA)

Fractional-order extension of JAYA using memory effect.

Key Innovation:
- Incorporates fractional-order memory into JAYA's update mechanism
- Uses Grünwald-Letnikov fractional-order difference
- Leverages historical information from previous iterations

References:
-----------
[1] Peng, Y., et al. (2025). A fractional-order JAYA algorithm with memory effect 
    for solving global optimization problem. 
    Expert Systems with Applications, 237, 121406.
[2] Rao, R. (2016). Jaya: A simple and new optimization algorithm for solving 
    constrained and unconstrained optimization problems. 
    International Journal of Industrial Engineering Computations, 7(1), 19-34.
"""

import numpy as np
from collections import deque


class FO_JAYA:
    """
    Fractional-Order JAYA Optimization Algorithm
    
    Parameters:
    -----------
    n_population : int
        Number of individuals in the population
    dim : int
        Dimension of the search space
    alpha : float
        Fractional order, 0 < alpha <= 1
        - alpha = 1.0: Standard JAYA
        - alpha < 1.0: Fractional-order JAYA with memory effect
    memory_size : int
        Number of past iterations to remember (default: 10)
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_population=30, dim=10, alpha=0.7, memory_size=10,
                 bounds=(-100, 100), seed=None):
        self.n_population = n_population
        self.dim = dim
        self.alpha = alpha
        self.memory_size = memory_size
        self.bounds = bounds
        self.seed = seed
        
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be in (0, 1]")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population
        self.positions = None
        self.fitness = None
        self.position_history = None  # For fractional-order memory
        self.best_position = None
        self.best_fitness = None
        self.worst_position = None
        self.worst_fitness = None
        
        # Fractional-order weights (Grünwald-Letnikov coefficients)
        self.weights = self._compute_gl_weights()
        
    def _compute_gl_weights(self):
        """
        Compute Grünwald-Letnikov fractional-order weights.
        
        w_k = (-1)^k * C(alpha, k)
        where C(alpha, k) is the binomial coefficient
        """
        weights = np.zeros(self.memory_size)
        weights[0] = 1.0
        for k in range(1, self.memory_size):
            weights[k] = weights[k-1] * (k - 1 - self.alpha) / k
        return weights
        
    def _initialize(self):
        """Initialize population positions."""
        self.positions = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            (self.n_population, self.dim)
        )
        self.fitness = np.full(self.n_population, np.inf)
        self.position_history = [deque(maxlen=self.memory_size) 
                                for _ in range(self.n_population)]
        self.best_fitness = np.inf
        self.worst_fitness = -np.inf
        
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """
        Run FO-JAYA optimization.
        
        Parameters:
        -----------
        fitness_func : callable
            Fitness function to minimize
        max_iter : int
            Maximum number of iterations
        verbose : bool
            Print progress information
        
        Returns:
        --------
        best_position : ndarray
            Best position found
        best_fitness : float
            Best fitness value
        convergence_curve : ndarray
            Best fitness at each iteration
        """
        self._initialize()
        convergence_curve = np.zeros(max_iter)
        
        for iteration in range(max_iter):
            # Evaluate the initial population once; afterwards self.fitness is kept current by
            # the greedy selection below, so re-evaluation would only repeat known values (R1:
            # equal evaluation budget, N per iteration).
            if iteration == 0:
                for i in range(self.n_population):
                    self.fitness[i] = fitness_func(self.positions[i])
            
            # Find best and worst solutions
            best_idx = np.argmin(self.fitness)
            worst_idx = np.argmax(self.fitness)
            
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_position = self.positions[best_idx].copy()
            
            self.worst_fitness = self.fitness[worst_idx]
            self.worst_position = self.positions[worst_idx].copy()
            
            # Update positions using FO-JAYA equation
            for i in range(self.n_population):
                # Store current position in history
                self.position_history[i].append(self.positions[i].copy())
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Standard JAYA update
                jaya_update = (self.positions[i] + 
                              r1 * (self.best_position - np.abs(self.positions[i])) -
                              r2 * (self.worst_position - np.abs(self.positions[i])))
                
                # Apply fractional-order memory effect
                if self.alpha == 1.0 or len(self.position_history[i]) < 2:
                    # Standard JAYA (no memory effect)
                    new_position = jaya_update
                else:
                    # Fractional-order JAYA with memory
                    # Current update contribution (scaled by alpha)
                    new_position = self.alpha * jaya_update
                    
                    # Add memory effect from past positions
                    memory_term = np.zeros(self.dim)
                    history_len = len(self.position_history[i])
                    for k in range(history_len):
                        if k < len(self.weights):
                            memory_term += self.weights[k] * self.position_history[i][-(k+1)]
                    
                    new_position += (1.0 - self.alpha) * memory_term / history_len
                
                # Apply bounds
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                
                # Greedy selection
                new_fitness = fitness_func(new_position)
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_position
                    self.fitness[i] = new_fitness
            
            # Credit improvements accepted in this iteration (R1: no one-iteration lag)
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_position = self.positions[best_idx].copy()
            
            # Record convergence
            convergence_curve[iteration] = self.best_fitness
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.best_fitness:.6e}, Alpha: {self.alpha}")
        
        return self.best_position, self.best_fitness, convergence_curve

