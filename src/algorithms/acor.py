"""
Ant Colony Optimization for Continuous Domains (ACOR)

Reference:
Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
European Journal of Operational Research, 185(3), 1155-1173.
DOI: 10.1016/j.ejor.2006.06.046

ACOR uses a solution archive and Gaussian kernel probability density functions
to generate new solutions in continuous search spaces.
"""

import numpy as np


class ACOR:
    """
    Ant Colony Optimization for Continuous Domains (ACOR) algorithm.
    
    Parameters:
    -----------
    n_ants : int
        Number of ants (solutions generated per iteration)
    dim : int
        Dimension of the problem
    archive_size : int
        Size of solution archive (default: 50)
    q : float
        Locality of search parameter (default: 0.01)
    xi : float
        Speed of convergence parameter (default: 0.85)
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_ants=30, dim=10, archive_size=50, q=0.01, xi=0.85, 
                 bounds=(-100, 100), seed=None):
        self.n_ants = n_ants
        self.dim = dim
        self.archive_size = archive_size
        self.q = q
        self.xi = xi
        self.bounds = bounds
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Solution archive: stores best solutions
        self.archive = []
        self.archive_fitness = []
        
        # Best solution tracking
        self.best_position = None
        self.best_fitness = np.inf
        
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
    
    def _compute_weights(self):
        """Compute selection weights for archive solutions."""
        weights = np.zeros(self.archive_size)
        for l in range(self.archive_size):
            weights[l] = (1.0 / (self.q * self.archive_size * np.sqrt(2 * np.pi))) * \
                        np.exp(-(l ** 2) / (2 * self.q ** 2 * self.archive_size ** 2))
        return weights / np.sum(weights)  # Normalize
    
    def _compute_standard_deviations(self):
        """Compute standard deviations for Gaussian sampling."""
        # sigma_{l,j} = xi/(k-1) * sum_r |s_{l,j} - s_{r,j}|  (vectorised; R1)
        A = np.asarray(self.archive)
        return self.xi / (self.archive_size - 1) * np.abs(A[:, None, :] - A[None, :, :]).sum(axis=1)
    
    def _generate_solution(self, weights, stds):
        """Generate a new solution using Gaussian sampling."""
        # Select solution from archive based on weights
        # One kernel index per coordinate (per-coordinate mixture, Sec. 3.3.4); mean and sigma
        # are taken from the SAME archive member of the archive frozen for this iteration (R1).
        kernel_idx = np.random.choice(self.archive_size, size=self.dim, p=weights)
        cols = np.arange(self.dim)
        A = np.asarray(self.archive)
        new_solution = np.random.normal(A[kernel_idx, cols], stds[kernel_idx, cols])
        
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
        """
        Run the ACOR optimization algorithm.
        
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
        
        # Initialize solution archive
        self._initialize_archive(fitness_func)
        
        # Main loop
        for iteration in range(max_iter):
            # Compute weights and standard deviations
            weights = self._compute_weights()
            stds = self._compute_standard_deviations()
            
            # Generate new solutions
            new_solutions, new_fitnesses = [], []
            for _ in range(self.n_ants):
                new_solution = self._generate_solution(weights, stds)
                new_fitness = fitness_func(new_solution)
                
                new_solutions.append(new_solution)
                new_fitnesses.append(new_fitness)
            
            self._update_archive(new_solutions, new_fitnesses)

            
            convergence_curve.append(self.best_fitness)
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.best_fitness:.6e}")
        
        return self.best_position, self.best_fitness, convergence_curve

