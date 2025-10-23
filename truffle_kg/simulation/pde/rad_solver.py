"""
Reaction-Advection-Diffusion (RAD) PDE Solver for Truffle Cultivation
Simulates nutrient and signal transport in hydroponic systems
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Species:
    """Represents a chemical species in the system"""
    name: str
    diffusion_coeff: float  # m²/s
    molecular_weight: float  # g/mol
    charge: int = 0
    initial_concentration: float = 0.0

@dataclass
class BoundaryCondition:
    """Boundary condition for PDE"""
    type: str  # 'dirichlet', 'neumann', 'robin'
    value: float
    location: str  # 'left', 'right', 'top', 'bottom'
    
@dataclass
class Reaction:
    """Chemical reaction in the system"""
    reactants: List[str]
    products: List[str]
    rate_constant: float
    stoichiometry: Dict[str, float]

class RADSolver:
    """Solver for Reaction-Advection-Diffusion PDEs"""
    
    def __init__(self, domain: Tuple[float, float, float, float], 
                 nx: int = 50, ny: int = 50, dt: float = 0.01):
        """
        Initialize the RAD solver
        
        Args:
            domain: (x_min, x_max, y_min, y_max) domain boundaries
            nx, ny: Number of grid points in x and y directions
            dt: Time step size
        """
        self.domain = domain
        self.nx = nx
        self.ny = ny
        self.dt = dt
        
        # Create spatial grid
        x_min, x_max, y_min, y_max = domain
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Grid spacing
        self.dx = (x_max - x_min) / (nx - 1)
        self.dy = (y_max - y_min) / (ny - 1)
        
        # Species and reactions
        self.species = {}
        self.reactions = []
        self.boundary_conditions = {}
        
        # Velocity field
        self.velocity_field = None
        
        # Hyphal density field (from ABM)
        self.hyphal_density = np.zeros((ny, nx))
        
        logger.info(f"Initialized RAD solver with {nx}x{ny} grid")
    
    def add_species(self, species: Species):
        """Add a chemical species to the system"""
        self.species[species.name] = species
        logger.info(f"Added species: {species.name}")
    
    def add_reaction(self, reaction: Reaction):
        """Add a chemical reaction to the system"""
        self.reactions.append(reaction)
        logger.info(f"Added reaction: {reaction.reactants} -> {reaction.products}")
    
    def set_boundary_condition(self, species_name: str, bc: BoundaryCondition):
        """Set boundary condition for a species"""
        if species_name not in self.boundary_conditions:
            self.boundary_conditions[species_name] = {}
        self.boundary_conditions[species_name][bc.location] = bc
        logger.info(f"Set {bc.type} BC for {species_name} at {bc.location}")
    
    def set_velocity_field(self, velocity_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]):
        """Set the velocity field for advection"""
        self.velocity_field = velocity_func
        logger.info("Set velocity field")
    
    def set_hyphal_density(self, density: np.ndarray):
        """Set hyphal density field from ABM"""
        if density.shape != (self.ny, self.nx):
            raise ValueError(f"Hyphal density shape {density.shape} doesn't match grid {self.ny}x{self.nx}")
        self.hyphal_density = density
        logger.info("Updated hyphal density field")
    
    def _create_laplacian_matrix(self, species_name: str) -> sp.csr_matrix:
        """Create finite difference Laplacian matrix"""
        n = self.nx * self.ny
        D = self.species[species_name].diffusion_coeff
        
        # Create 5-point stencil matrix
        data = []
        row_indices = []
        col_indices = []
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx = i * self.nx + j
                
                # Center point
                data.append(-4 * D / (self.dx**2) - 4 * D / (self.dy**2))
                row_indices.append(idx)
                col_indices.append(idx)
                
                # Neighbors
                if i > 0:  # Bottom
                    data.append(D / (self.dy**2))
                    row_indices.append(idx)
                    col_indices.append(idx - self.nx)
                
                if i < self.ny - 1:  # Top
                    data.append(D / (self.dy**2))
                    row_indices.append(idx)
                    col_indices.append(idx + self.nx)
                
                if j > 0:  # Left
                    data.append(D / (self.dx**2))
                    row_indices.append(idx)
                    col_indices.append(idx - 1)
                
                if j < self.nx - 1:  # Right
                    data.append(D / (self.dx**2))
                    row_indices.append(idx)
                    col_indices.append(idx + 1)
        
        return sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    def _create_advection_matrix(self, species_name: str) -> sp.csr_matrix:
        """Create finite difference advection matrix"""
        n = self.nx * self.ny
        
        if self.velocity_field is None:
            return sp.csr_matrix((n, n))
        
        # Get velocity field
        u, v = self.velocity_field(self.X, self.Y)
        
        data = []
        row_indices = []
        col_indices = []
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx = i * self.nx + j
                
                # Upwind scheme for advection
                if u[i, j] > 0:  # Flow to the right
                    if j > 0:
                        data.append(u[i, j] / self.dx)
                        row_indices.append(idx)
                        col_indices.append(idx - 1)
                    data.append(-u[i, j] / self.dx)
                    row_indices.append(idx)
                    col_indices.append(idx)
                else:  # Flow to the left
                    if j < self.nx - 1:
                        data.append(-u[i, j] / self.dx)
                        row_indices.append(idx)
                        col_indices.append(idx + 1)
                    data.append(u[i, j] / self.dx)
                    row_indices.append(idx)
                    col_indices.append(idx)
                
                if v[i, j] > 0:  # Flow upward
                    if i > 0:
                        data.append(v[i, j] / self.dy)
                        row_indices.append(idx)
                        col_indices.append(idx - self.nx)
                    data.append(-v[i, j] / self.dy)
                    row_indices.append(idx)
                    col_indices.append(idx)
                else:  # Flow downward
                    if i < self.ny - 1:
                        data.append(-v[i, j] / self.dy)
                        row_indices.append(idx)
                        col_indices.append(idx + self.nx)
                    data.append(v[i, j] / self.dy)
                    row_indices.append(idx)
                    col_indices.append(idx)
        
        return sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    def _apply_boundary_conditions(self, species_name: str, matrix: sp.csr_matrix, 
                                 rhs: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Apply boundary conditions to the system"""
        n = self.nx * self.ny
        
        if species_name not in self.boundary_conditions:
            return matrix, rhs
        
        bcs = self.boundary_conditions[species_name]
        
        # Apply Dirichlet boundary conditions
        for location, bc in bcs.items():
            if bc.type == 'dirichlet':
                if location == 'left':
                    for i in range(self.ny):
                        idx = i * self.nx
                        matrix[idx, :] = 0
                        matrix[idx, idx] = 1
                        rhs[idx] = bc.value
                elif location == 'right':
                    for i in range(self.ny):
                        idx = i * self.nx + (self.nx - 1)
                        matrix[idx, :] = 0
                        matrix[idx, idx] = 1
                        rhs[idx] = bc.value
                elif location == 'bottom':
                    for j in range(self.nx):
                        idx = j
                        matrix[idx, :] = 0
                        matrix[idx, idx] = 1
                        rhs[idx] = bc.value
                elif location == 'top':
                    for j in range(self.nx):
                        idx = (self.ny - 1) * self.nx + j
                        matrix[idx, :] = 0
                        matrix[idx, idx] = 1
                        rhs[idx] = bc.value
        
        return matrix, rhs
    
    def _compute_reaction_rates(self, concentrations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute reaction rates for all species"""
        reaction_rates = {name: np.zeros((self.ny, self.nx)) for name in self.species.keys()}
        
        for reaction in self.reactions:
            # Compute reaction rate (simplified first-order kinetics)
            rate = np.ones((self.ny, self.nx)) * reaction.rate_constant
            
            for reactant in reaction.reactants:
                if reactant in concentrations:
                    rate *= concentrations[reactant]
            
            # Update production/consumption rates
            for reactant in reaction.reactants:
                if reactant in reaction_rates:
                    reaction_rates[reactant] -= rate * reaction.stoichiometry.get(reactant, 1.0)
            
            for product in reaction.products:
                if product in reaction_rates:
                    reaction_rates[product] += rate * reaction.stoichiometry.get(product, 1.0)
        
        return reaction_rates
    
    def _compute_uptake_rates(self, concentrations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute nutrient uptake rates based on hyphal density"""
        uptake_rates = {}
        
        for species_name, species in self.species.items():
            if species_name in concentrations:
                # Michaelis-Menten uptake kinetics
                Vmax = 1.0  # Maximum uptake rate (mol/m²/s)
                Km = 0.1    # Half-saturation constant (mol/m³)
                
                S = concentrations[species_name]
                uptake = self.hyphal_density * Vmax * S / (Km + S)
                uptake_rates[species_name] = uptake
        
        return uptake_rates
    
    def solve_timestep(self, concentrations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Solve one time step of the RAD system"""
        new_concentrations = {}
        
        # Compute reaction rates
        reaction_rates = self._compute_reaction_rates(concentrations)
        
        # Compute uptake rates
        uptake_rates = self._compute_uptake_rates(concentrations)
        
        for species_name, species in self.species.items():
            if species_name not in concentrations:
                continue
            
            # Flatten concentration for matrix operations
            C = concentrations[species_name].flatten()
            
            # Create system matrices
            L = self._create_laplacian_matrix(species_name)
            A = self._create_advection_matrix(species_name)
            
            # Combine diffusion and advection
            system_matrix = L + A
            
            # Create right-hand side
            rhs = C.copy()
            
            # Add reaction terms
            if species_name in reaction_rates:
                rhs += self.dt * reaction_rates[species_name].flatten()
            
            # Add uptake terms
            if species_name in uptake_rates:
                rhs -= self.dt * uptake_rates[species_name].flatten()
            
            # Apply boundary conditions
            system_matrix, rhs = self._apply_boundary_conditions(species_name, system_matrix, rhs)
            
            # Solve linear system
            try:
                C_new = spsolve(system_matrix, rhs)
                new_concentrations[species_name] = C_new.reshape(self.ny, self.nx)
            except Exception as e:
                logger.error(f"Error solving for {species_name}: {e}")
                new_concentrations[species_name] = concentrations[species_name]
        
        return new_concentrations
    
    def solve(self, initial_concentrations: Dict[str, np.ndarray], 
              t_end: float) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
        """Solve the RAD system over time"""
        logger.info(f"Solving RAD system from t=0 to t={t_end}")
        
        # Time points
        t_points = np.arange(0, t_end + self.dt, self.dt)
        n_steps = len(t_points)
        
        # Initialize solution storage
        solution = {name: [] for name in self.species.keys()}
        
        # Set initial conditions
        concentrations = initial_concentrations.copy()
        for name in self.species.keys():
            if name not in concentrations:
                concentrations[name] = np.full((self.ny, self.nx), 
                                             self.species[name].initial_concentration)
        
        # Store initial conditions
        for name, conc in concentrations.items():
            solution[name].append(conc.copy())
        
        # Time stepping
        for i in range(1, n_steps):
            logger.info(f"Time step {i}/{n_steps-1}, t={t_points[i]:.3f}")
            
            # Solve one time step
            concentrations = self.solve_timestep(concentrations)
            
            # Store solution
            for name, conc in concentrations.items():
                solution[name].append(conc.copy())
        
        return t_points, solution
    
    def plot_solution(self, solution: Dict[str, List[np.ndarray]], 
                     t_points: np.ndarray, species_name: str, 
                     time_indices: List[int] = None):
        """Plot the solution for a species"""
        if species_name not in solution:
            logger.error(f"Species {species_name} not found in solution")
            return
        
        if time_indices is None:
            time_indices = [0, len(t_points)//4, len(t_points)//2, 3*len(t_points)//4, -1]
        
        fig, axes = plt.subplots(1, len(time_indices), figsize=(15, 3))
        if len(time_indices) == 1:
            axes = [axes]
        
        for i, t_idx in enumerate(time_indices):
            if t_idx < 0:
                t_idx = len(t_points) + t_idx
            
            conc = solution[species_name][t_idx]
            im = axes[i].contourf(self.X, self.Y, conc, levels=20)
            axes[i].set_title(f't = {t_points[t_idx]:.2f}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()

def create_truffle_system() -> RADSolver:
    """Create a typical truffle cultivation system"""
    # Domain: 1m x 1m hydroponic chamber
    domain = (0.0, 1.0, 0.0, 1.0)
    solver = RADSolver(domain, nx=50, ny=50, dt=0.01)
    
    # Add species
    solver.add_species(Species("NO3", 1.5e-9, 62.0, -1, 0.1))  # Nitrate
    solver.add_species(Species("H2PO4", 8.0e-10, 97.0, -2, 0.05))  # Phosphate
    solver.add_species(Species("K", 1.9e-9, 39.1, 1, 0.2))  # Potassium
    solver.add_species(Species("Ca", 7.9e-10, 40.1, 2, 0.15))  # Calcium
    solver.add_species(Species("O2", 2.0e-9, 32.0, 0, 0.008))  # Oxygen
    
    # Add reactions (simplified)
    solver.add_reaction(Reaction(
        reactants=["NO3"],
        products=["NO2"],
        rate_constant=0.01,
        stoichiometry={"NO3": 1.0, "NO2": 1.0}
    ))
    
    # Set boundary conditions
    solver.set_boundary_condition("NO3", BoundaryCondition("dirichlet", 0.1, "left"))
    solver.set_boundary_condition("H2PO4", BoundaryCondition("dirichlet", 0.05, "left"))
    solver.set_boundary_condition("K", BoundaryCondition("dirichlet", 0.2, "left"))
    solver.set_boundary_condition("Ca", BoundaryCondition("dirichlet", 0.15, "left"))
    solver.set_boundary_condition("O2", BoundaryCondition("dirichlet", 0.008, "top"))
    
    # Set velocity field (circular flow)
    def velocity_field(x, y):
        u = -0.1 * (y - 0.5)
        v = 0.1 * (x - 0.5)
        return u, v
    
    solver.set_velocity_field(velocity_field)
    
    return solver

def main():
    """Example usage"""
    # Create system
    solver = create_truffle_system()
    
    # Set initial conditions
    initial_concentrations = {
        "NO3": np.full((solver.ny, solver.nx), 0.05),
        "H2PO4": np.full((solver.ny, solver.nx), 0.025),
        "K": np.full((solver.ny, solver.nx), 0.1),
        "Ca": np.full((solver.ny, solver.nx), 0.075),
        "O2": np.full((solver.ny, solver.nx), 0.004)
    }
    
    # Solve
    t_points, solution = solver.solve(initial_concentrations, t_end=10.0)
    
    # Plot results
    solver.plot_solution(solution, t_points, "NO3")

if __name__ == "__main__":
    main()