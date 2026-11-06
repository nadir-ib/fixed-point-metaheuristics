"""
Fractional-Order Water Flow Optimizer (FO-WFO)

Reference:
Tang, Z., Wang, K., Zang, Y., Zhu, Q., Todo, Y., & Gao, S. (2024). 
Fractional-Order Water Flow Optimizer. International Journal of Computational 
Intelligence Systems, 17(1), 84.
DOI: 10.1007/s44196-024-00445-4

The FO-WFO incorporates fractional calculus to add memory effects to the 
standard WFO algorithm, using Grünwald-Letnikov fractional-order derivative.
"""

import numpy as np


class FO_WFO:
    """
    Fractional-Order Water Flow Optimizer (FO-WFO) algorithm.
    
    Parameters:
    -----------
    n_population : int
        Number of water particles in the population
    dim : int
        Dimension of the problem
    alpha : float
        Fractional order parameter (0 < alpha <= 1)
        alpha = 1: standard WFO
        alpha < 1: fractional-order with memory effect
    pl : float
        Probability of laminar flow (default: 0.25)
    pe : float
        Vortex probability in turbulent flow (default: 0.2)
    memory_size : int
        Number of past positions to remember (default: 10)
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_population=30, dim=10, alpha=0.7, pl=0.25, pe=0.2, 
                 memory_size=10, bounds=(-100, 100), seed=None):
        self.n_population = n_population
        self.dim = dim
        self.alpha = alpha
        self.pl = pl
        self.pe = pe
        self.memory_size = memory_size
        self.bounds = bounds
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_population, dim))
        self.fitness = np.full(n_population, np.inf)
        self.best_position = None
        self.best_fitness = np.inf
        
        # Memory for fractional-order
        self.position_history = [[] for _ in range(n_population)]
        
        # Compute Grünwald-Letnikov weights
        self.weights = self._compute_gl_weights()
        
    def _compute_gl_weights(self):
        """Compute Grünwald-Letnikov fractional-order weights."""
        weights = np.zeros(self.memory_size)
        weights[0] = 1.0
        for k in range(1, self.memory_size):
            weights[k] = weights[k-1] * (k - 1 - self.alpha) / k
        return weights
    
    def _clip_position(self, position):
        """Clip position to bounds."""
        return np.clip(position, self.bounds[0], self.bounds[1])
    
    def _laminar_flow(self, i):
        """Laminar flow operator."""
        s = np.random.rand()
        k = np.random.randint(0, self.n_population)
        
        if not np.array_equal(self.best_position, self.positions[k]):
            d = self.best_position - self.positions[k]
        else:
            d = np.zeros(self.dim)
        
        new_position = self.positions[i] + s * d
        return self._clip_position(new_position)
    
    def _turbulent_flow(self, i):
        """Turbulent flow operator."""
        new_position = self.positions[i].copy()
        j1 = np.random.randint(0, self.dim)
        k = np.random.randint(0, self.n_population)
        r = np.random.rand()
        
        if r < self.pe:
            theta = np.random.uniform(-np.pi, np.pi)
            new_position[j1] = (self.positions[i, j1] + 
                               np.abs(self.positions[i, j1] - self.positions[k, j1]) * 
                               theta * np.cos(theta))
        else:
            j2 = np.random.randint(0, self.dim)
            while j2 == j1:
                j2 = np.random.randint(0, self.dim)
            
            ub_j1, lb_j1 = self.bounds[1], self.bounds[0]
            ub_j2, lb_j2 = self.bounds[1], self.bounds[0]
            
            new_position[j1] = ((ub_j1 - lb_j1) * (self.positions[k, j2] - lb_j2) / 
                               (ub_j2 - lb_j2) + lb_j1)
        
        return self._clip_position(new_position)
    
    def _apply_fractional_memory(self, i, wfo_update):
        """
        Apply fractional-order memory effect to WFO update.
        
        X(t+1) = alpha * X_wfo(t) + (1-alpha) * sum(w_k * X(t-k))
        """
        if self.alpha == 1.0 or len(self.position_history[i]) < 2:
            return wfo_update
        
        # Current update contribution (scaled by alpha)
        new_position = self.alpha * wfo_update
        
        # Add memory effect from past positions
        memory_term = np.zeros(self.dim)
        history_len = len(self.position_history[i])
        
        for k in range(history_len):
            if k < len(self.weights):
                memory_term += self.weights[k] * self.position_history[i][-(k+1)]
        
        new_position += (1.0 - self.alpha) * memory_term / history_len
        
        return self._clip_position(new_position)
    
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """
        Run the FO-WFO optimization algorithm.
        
        Parameters:
        -----------
        fitness_func : callable
            The fitness function to minimize
        max_iter : int
            Maximum number of iterations
        verbose : bool
            Print progress information
            
        Returns:
        --------
        best_position : ndarray
            Best solution found
        best_fitness : float
            Best fitness value
        convergence_curve : list
            Best fitness at each iteration
        """
        convergence_curve = []
        
        # Evaluate initial population
        for i in range(self.n_population):
            self.fitness[i] = fitness_func(self.positions[i])
            self.position_history[i].append(self.positions[i].copy())
            
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.positions[i].copy()
        
        # Main loop
        for iteration in range(max_iter):
            for i in range(self.n_population):
                # Standard WFO update
                if np.random.rand() < self.pl:
                    wfo_update = self._laminar_flow(i)
                else:
                    wfo_update = self._turbulent_flow(i)
                
                # Apply fractional-order memory effect
                new_position = self._apply_fractional_memory(i, wfo_update)
                
                # Evaluate new position
                new_fitness = fitness_func(new_position)
                
                # Greedy selection
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_position
                    self.fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_position.copy()
                
                # Update position history
                self.position_history[i].append(self.positions[i].copy())
                if len(self.position_history[i]) > self.memory_size:
                    self.position_history[i].pop(0)
            
            convergence_curve.append(self.best_fitness)
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.best_fitness:.6e}")
        
        return self.best_position, self.best_fitness, convergence_curve

