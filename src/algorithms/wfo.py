"""
Water Flow Optimizer (WFO)

Reference:
Luo, K. (2022). Water Flow Optimizer: A Nature-Inspired Evolutionary Algorithm 
for Global Optimization. IEEE Transactions on Cybernetics, 52(7), 7753-7764.
DOI: 10.1109/TCYB.2021.3049607

The WFO algorithm is inspired by the two distinctive types of water flows:
1. Laminar flow: All particles move parallel in the same direction
2. Turbulent flow: Particles show irregular rotational movements
"""

import numpy as np


class WFO:
    """
    Water Flow Optimizer (WFO) algorithm.
    
    Parameters:
    -----------
    n_population : int
        Number of water particles in the population
    dim : int
        Dimension of the problem
    pl : float
        Probability of laminar flow (default: 0.25)
    pe : float
        Vortex probability in turbulent flow (default: 0.2)
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_population=30, dim=10, pl=0.25, pe=0.2, bounds=(-100, 100), seed=None):
        self.n_population = n_population
        self.dim = dim
        self.pl = pl  # Laminar flow probability
        self.pe = pe  # Vortex probability
        self.bounds = bounds
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_population, dim))
        self.fitness = np.full(n_population, np.inf)
        self.best_position = None
        self.best_fitness = np.inf
        
    def _clip_position(self, position):
        """Clip position to bounds."""
        return np.clip(position, self.bounds[0], self.bounds[1])
    
    def _laminar_flow(self, i):
        """
        Laminar flow operator: particles move parallel in the same direction.
        y_i = x_i + s * d
        where d = x_best - x_k (common direction)
        """
        # Random step size
        s = np.random.rand()
        
        # Select a random particle
        k = np.random.randint(0, self.n_population)
        
        # Common direction vector
        if not np.array_equal(self.best_position, self.positions[k]):
            d = self.best_position - self.positions[k]
        else:
            d = np.zeros(self.dim)
        
        # New position
        new_position = self.positions[i] + s * d
        
        return self._clip_position(new_position)
    
    def _turbulent_flow(self, i):
        """
        Turbulent flow operator: particles show irregular rotational movements.
        Two types:
        1. Vortex transformation (same layer)
        2. Cross-layer movement
        """
        new_position = self.positions[i].copy()
        
        # Select a random dimension
        j1 = np.random.randint(0, self.dim)
        
        # Select a random particle
        k = np.random.randint(0, self.n_population)
        
        r = np.random.rand()
        
        if r < self.pe:
            # Vortex transformation (same layer)
            theta = np.random.uniform(-np.pi, np.pi)
            new_position[j1] = (self.positions[i, j1] + 
                               np.abs(self.positions[i, j1] - self.positions[k, j1]) * 
                               theta * np.cos(theta))
        else:
            # Cross-layer movement
            j2 = np.random.randint(0, self.dim)
            while j2 == j1:
                j2 = np.random.randint(0, self.dim)
            
            ub_j1, lb_j1 = self.bounds[1], self.bounds[0]
            ub_j2, lb_j2 = self.bounds[1], self.bounds[0]
            
            new_position[j1] = ((ub_j1 - lb_j1) * (self.positions[k, j2] - lb_j2) / 
                               (ub_j2 - lb_j2) + lb_j1)
        
        return self._clip_position(new_position)
    
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """
        Run the WFO optimization algorithm.
        
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
            
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.positions[i].copy()
        
        # Main loop
        for iteration in range(max_iter):
            for i in range(self.n_population):
                # Decide flow type
                if np.random.rand() < self.pl:
                    # Laminar flow
                    new_position = self._laminar_flow(i)
                else:
                    # Turbulent flow
                    new_position = self._turbulent_flow(i)
                
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
            
            convergence_curve.append(self.best_fitness)
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.best_fitness:.6e}")
        
        return self.best_position, self.best_fitness, convergence_curve

