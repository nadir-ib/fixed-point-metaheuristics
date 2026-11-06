"""
Fractional-Order Particle Swarm Optimization (FO-PSO)

Fractional-order extension of PSO with a memory-augmented position update.

Position update:
    x_i(t+1) = x_i(t) + alpha * v_i(t+1)
                       + ((1 - alpha) / (M - 1)) * sum_{j=1}^{M-1} v_i(t-j+1)
where M is the size of the retained velocity-history window (M = 10 here).
This is a finite-memory blend that captures the long-memory effect of a
fractional integral while remaining inexpensive to evaluate; it is not a
strict Grunwald-Letnikov or Caputo discretization. The manuscript's
Theorem 2 is derived for exactly this update.

References
----------
[1] Couceiro, M. S., & Ghamisi, P. (2016). Fractional order Darwinian particle
    swarm optimization. Springer.
[2] Pires, E. S., et al. (2010). Particle swarm optimization with fractional-order
    velocity. Nonlinear Dynamics, 61(1-2), 295-301.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FO_PSO:
    """
    Fractional-Order Particle Swarm Optimization Algorithm
    
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
    alpha : float
        Fractional order, 0 < alpha <= 1
        - alpha = 1.0: Standard PSO
        - alpha < 1.0: Fractional-order PSO with memory effect
    bounds : tuple
        Search space bounds (min, max)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_particles=30, dim=10, w=0.7, c1=1.5, c2=1.5, 
                 alpha=0.8, bounds=(-100, 100), seed=None, memory_weights='uniform'):
        # memory_weights (R1 ablation, reviewer R3-2.1; default 'uniform' = the published FO-PSO):
        #   'uniform' : alpha*v_t + (1-alpha)/(L-1) * sum_{j=1}^{L-1} v_{t-j}
        #   'gl_def6' : alpha*v_t + (1-alpha)/L * sum_{j=0}^{L-1} omega_j v_{t-j}  (Definition 6 applied to velocities)
        #   'gl_abs'  : alpha*v_t + (1-alpha) * sum_{j=1}^{L-1} |omega_j| v_{t-j} / sum_{i=1}^{L-1} |omega_i|
        if memory_weights not in ('uniform', 'gl_def6', 'gl_abs'):
            raise ValueError(memory_weights)
        self.memory_weights = memory_weights
        self._omega = [1.0]
        for k in range(1, 10):
            self._omega.append(self._omega[-1] * (k - 1 - alpha) / k)
        self.n_particles = n_particles
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.bounds = bounds
        self.seed = seed
        
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be in (0, 1]")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize particles and velocities
        self.positions = None
        self.velocities = None
        self.velocity_history = None  # For fractional integral
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
        
        # Initialize velocity history for fractional integral
        self.velocity_history = []
        
        # Initialize personal best
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.n_particles, np.inf)
        
        # Initialize global best
        self.gbest_position = None
        self.gbest_fitness = np.inf
        
    def optimize(self, fitness_func, max_iter=500, verbose=False):
        """
        Run FO-PSO optimization.
        
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
            
            # Update velocities
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social
            
            # Store velocity history (keep only last 10 iterations for efficiency)
            self.velocity_history.append(self.velocities.copy())
            if len(self.velocity_history) > 10:
                self.velocity_history.pop(0)

            # Update positions using fractional-order approach
            if self.alpha == 1.0:
                # Standard PSO (no fractional effect)
                position_update = self.velocities
            else:
                # Fractional-order PSO
                # Use memory effect: current velocity + weighted past velocities
                # This approximates the fractional integral's memory property

                # Current velocity contribution (scaled by alpha)
                position_update = self.alpha * self.velocities

                # Add memory effect from past velocities
                L = len(self.velocity_history)
                if L > 1 and self.memory_weights == 'uniform':
                    # Uniform weights over the L-1 stored past velocities (published FO-PSO)
                    memory_weight = (1.0 - self.alpha) / (len(self.velocity_history) - 1)
                    for v_past in self.velocity_history[:-1]:
                        position_update += memory_weight * v_past
                elif L > 1 and self.memory_weights == 'gl_def6':
                    mem = np.zeros_like(self.velocities)
                    for j in range(L):                      # v_{t-j} = velocity_history[-(j+1)]
                        mem += self._omega[j] * self.velocity_history[-(j + 1)]
                    position_update += (1.0 - self.alpha) * mem / L
                elif L > 1 and self.memory_weights == 'gl_abs':
                    tot = sum(abs(self._omega[j]) for j in range(1, L))
                    for j in range(1, L):
                        position_update += (1.0 - self.alpha) * abs(self._omega[j]) / tot * self.velocity_history[-(j + 1)]

            self.positions = self.positions + position_update
            
            # Apply bounds
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
            
            # Record convergence
            convergence_curve[iteration] = self.gbest_fitness
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {self.gbest_fitness:.6e}, Alpha: {self.alpha}")
        
        return self.gbest_position, self.gbest_fitness, convergence_curve

