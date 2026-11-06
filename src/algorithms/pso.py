"""
Particle Swarm Optimization (PSO)

Standard PSO algorithm as proposed by Kennedy and Eberhart (1995).

References:
-----------
[1] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95-International Conference on Neural Networks, 4, 1942-1948.
[2] Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
    IEEE International Conference on Evolutionary Computation, 69-73.
"""

import numpy as np


class PSO:
    """
    Particle Swarm Optimization Algorithm
    
    Parameters:
    -----------
    n_particles : int
        Number of particles in the swarm
    dim : int
        Dimension of the search space
    w : float
        Inertia weight (typically 0.4-0.9)
    c1 : float
        Cognitive coefficient (typically 1.5-2.0)
    c2 : float
        Social coefficient (typically 1.5-2.0)
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_particles=30, dim=10, w=0.7, c1=1.5, c2=1.5, 
                 bounds=(-100, 100), seed=None):
        self.n_particles = n_particles
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize particles and velocities
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_fitness = None
        self.gbest_position = None
        self.gbest_fitness = None
        
    def _initialize(self):
        """Initialize particle positions and velocities."""
        # Random positions within bounds
        self.positions = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            (self.n_particles, self.dim)
        )
        
        # Random velocities (typically 10% of search space)
        v_max = 0.1 * (self.bounds[1] - self.bounds[0])
        self.velocities = np.random.uniform(
            -v_max, v_max, 
            (self.n_particles, self.dim)
        )
        
        # Initialize personal best
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.n_particles, np.inf)
        
        # Initialize global best
        self.gbest_position = None
        self.gbest_fitness = np.inf
        
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """
        Run PSO optimization.
        
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
            # Evaluate fitness for all particles
            for i in range(self.n_particles):
                fitness = fitness_func(self.positions[i])
                
                # Update personal best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # Update global best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
            
            # Update velocities and positions
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            
            # Velocity update
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social
            
            # Position update
            self.positions = self.positions + self.velocities
            
            # Apply bounds
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
            
            # Record convergence
            convergence_curve[iteration] = self.gbest_fitness
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.gbest_fitness:.6e}")
        
        return self.gbest_position, self.gbest_fitness, convergence_curve

