"""
Fractional-Order Ant Colony Optimization for Continuous Domains (FO-ACOR)

Reference:
Pu, Y. F., et al. (2021). Fractional-order ant colony algorithm: A fractional 
long term memory based cooperative learning approach. Swarm and Evolutionary 
Computation, 69, 101024.
DOI: 10.1016/j.swevo.2021.101024

FO-ACOR incorporates fractional calculus to add memory effects to the standard 
ACOR algorithm, using Grünwald-Letnikov fractional-order derivative.
"""

import numpy as np


class FO_ACOR:
    """
    Fractional-Order Ant Colony Optimization for Continuous Domains (FO-ACOR).
    
    Parameters:
    -----------
    n_ants : int
        Number of ants (solutions generated per iteration)
    dim : int
        Dimension of the problem
    alpha : float
        Fractional order parameter (0 < alpha <= 1)
        alpha = 1: standard ACOR
        alpha < 1: fractional-order with memory effect
    archive_size : int
        Size of solution archive (default: 50)
    q : float
        Locality of search parameter (default: 0.01)
    xi : float
        Speed of convergence parameter (default: 0.85)
    memory_size : int
        Number of past solutions to remember (default: 10)
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_ants=30, dim=10, alpha=0.7, archive_size=50, q=0.01, 
                 xi=0.85, memory_size=10, bounds=(-100, 100), seed=None):
        self.n_ants = n_ants
        self.dim = dim
        self.alpha = alpha
        self.archive_size = archive_size
        self.q = q
        self.xi = xi
        self.memory_size = memory_size
        self.bounds = bounds
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Solution archive
        self.archive = []
        self.archive_fitness = []
        
        # Memory for fractional-order
        self.solution_history = []
        
        # Best solution tracking
        self.best_position = None
        self.best_fitness = np.inf
        
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
    
    def _initialize_archive(self, fitness_func):
        """Initialize solution archive with random solutions."""
        for _ in range(self.archive_size):
            solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            fitness = fitness_func(solution)
            self.archive.append(solution)
            self.archive_fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = solution.copy()
        
        # Sort archive by fitness
        sorted_indices = np.argsort(self.archive_fitness)
        self.archive = [self.archive[i] for i in sorted_indices]
        self.archive_fitness = [self.archive_fitness[i] for i in sorted_indices]
    
    def _compute_weights_prob(self):
        """Compute selection weights for archive solutions."""
        weights = np.zeros(self.archive_size)
        for l in range(self.archive_size):
            weights[l] = (1.0 / (self.q * self.archive_size * np.sqrt(2 * np.pi))) * \
                        np.exp(-(l ** 2) / (2 * self.q ** 2 * self.archive_size ** 2))
        return weights / np.sum(weights)
    
    def _compute_standard_deviations(self):
        """Compute standard deviations for Gaussian sampling."""
        # sigma_{l,j} = xi/(k-1) * sum_r |s_{l,j} - s_{r,j}|  (vectorised; R1)
        A = np.asarray(self.archive)
        return self.xi / (self.archive_size - 1) * np.abs(A[:, None, :] - A[None, :, :]).sum(axis=1)
    
    def _generate_solution(self, weights_prob, stds):
        """Generate a new solution using Gaussian sampling."""
        # One kernel index per coordinate (per-coordinate mixture, Sec. 3.3.4); mean and sigma
        # are taken from the SAME archive member of the archive frozen for this iteration (R1).
        kernel_idx = np.random.choice(self.archive_size, size=self.dim, p=weights_prob)
        cols = np.arange(self.dim)
        A = np.asarray(self.archive)
        new_solution = np.random.normal(A[kernel_idx, cols], stds[kernel_idx, cols])
        
        return self._clip_position(new_solution)
    
    def _apply_fractional_memory(self, acor_solution):
        """Apply fractional-order memory effect to ACOR solution."""
        if self.alpha == 1.0 or len(self.solution_history) < 2:
            return acor_solution
        
        # Current solution contribution (scaled by alpha)
        new_solution = self.alpha * acor_solution
        
        # Add memory effect from past solutions
        memory_term = np.zeros(self.dim)
        history_len = len(self.solution_history)
        
        for k in range(history_len):
            if k < len(self.weights):
                memory_term += self.weights[k] * self.solution_history[-(k+1)]
        
        new_solution += (1.0 - self.alpha) * memory_term / history_len
        
        return self._clip_position(new_solution)
    
    def _update_archive(self, new_solutions, new_fitnesses):
        """Batch archive update after all ants of an iteration (Socha & Dorigo 2008):
        merge the k archive members with the m new solutions, sort, keep the best k (R1)."""
        pool = list(self.archive) + [s.copy() for s in new_solutions]
        pool_f = list(self.archive_fitness) + list(new_fitnesses)
        order = np.argsort(pool_f, kind='stable')[:self.archive_size]
        self.archive = [pool[i] for i in order]
        self.archive_fitness = [pool_f[i] for i in order]
        if self.archive_fitness[0] < self.best_fitness:
            self.best_fitness = self.archive_fitness[0]
            self.best_position = self.archive[0].copy()

    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """Run the FO-ACOR optimization algorithm."""
        convergence_curve = []
        
        # Initialize solution archive
        self._initialize_archive(fitness_func)
        
        # Main loop
        for iteration in range(max_iter):
            weights_prob = self._compute_weights_prob()
            stds = self._compute_standard_deviations()
            
            new_solutions, new_fitnesses = [], []
            
            for _ in range(self.n_ants):
                # Standard ACOR solution generation
                acor_solution = self._generate_solution(weights_prob, stds)
                
                # Apply fractional-order memory effect
                new_solution = self._apply_fractional_memory(acor_solution)
                new_fitness = fitness_func(new_solution)
                
                # Update solution history
                self.solution_history.append(new_solution.copy())
                if len(self.solution_history) > self.memory_size:
                    self.solution_history.pop(0)
                
                new_solutions.append(new_solution)
                new_fitnesses.append(new_fitness)
            
            self._update_archive(new_solutions, new_fitnesses)

            
            convergence_curve.append(self.best_fitness)
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.best_fitness:.6e}")
        
        return self.best_position, self.best_fitness, convergence_curve

