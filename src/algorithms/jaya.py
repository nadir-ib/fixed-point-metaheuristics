"""
JAYA Algorithm

A simple and powerful optimization algorithm proposed by Rao (2016).
The name "JAYA" means "victory" in Sanskrit.

Key Principle:
- Move towards the best solution
- Move away from the worst solution
- No algorithm-specific parameters (only population size)

References:
-----------
[1] Rao, R. (2016). Jaya: A simple and new optimization algorithm for solving 
    constrained and unconstrained optimization problems. 
    International Journal of Industrial Engineering Computations, 7(1), 19-34.
"""

import numpy as np


class JAYA:
    """
    JAYA Optimization Algorithm
    
    Parameters:
    -----------
    n_population : int
        Number of individuals in the population
    dim : int
        Dimension of the search space
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_population=30, dim=10, bounds=(-100, 100), seed=None):
        self.n_population = n_population
        self.dim = dim
        self.bounds = bounds
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population
        self.positions = None
        self.fitness = None
        self.best_position = None
        self.best_fitness = None
        self.worst_position = None
        self.worst_fitness = None
        
    def _initialize(self):
        """Initialize population positions."""
        self.positions = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            (self.n_population, self.dim)
        )
        self.fitness = np.full(self.n_population, np.inf)
        self.best_fitness = np.inf
        self.worst_fitness = -np.inf
        
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """
        Run JAYA optimization.
        
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
            
            # Update positions using JAYA equation
            for i in range(self.n_population):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # JAYA update equation
                new_position = (self.positions[i] + 
                               r1 * (self.best_position - np.abs(self.positions[i])) -
                               r2 * (self.worst_position - np.abs(self.positions[i])))
                
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
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.best_fitness:.6e}")
        
        return self.best_position, self.best_fitness, convergence_curve

