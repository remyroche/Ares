"""
Agent-Based Model (ABM) for Hyphal Growth in Truffle Cultivation
Simulates individual hyphal tips as agents with growth, branching, and tropism behaviors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import random
import logging
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

@dataclass
class HyphalTip:
    """Represents a single hyphal tip agent"""
    x: float
    y: float
    z: float
    direction: float  # Angle in radians
    age: float = 0.0
    length: float = 0.0
    parent_id: Optional[int] = None
    generation: int = 0
    active: bool = True
    growth_rate: float = 1.0
    branching_probability: float = 0.1
    anastomosis_probability: float = 0.05

@dataclass
class Environment:
    """Environmental conditions affecting hyphal growth"""
    nutrient_concentration: np.ndarray  # 3D array
    ph_gradient: np.ndarray
    temperature: np.ndarray
    oxygen_concentration: np.ndarray
    root_exudates: np.ndarray
    obstacles: np.ndarray  # Binary array for obstacles
    chemotaxis_strength: float = 1.0
    thigmotaxis_strength: float = 0.5
    ph_tropism_strength: float = 0.3

@dataclass
class GrowthParameters:
    """Parameters controlling hyphal growth behavior"""
    base_growth_rate: float = 1.0
    max_growth_rate: float = 3.0
    min_growth_rate: float = 0.1
    branching_rate: float = 0.1
    anastomosis_distance: float = 5.0
    max_tip_age: float = 100.0
    direction_persistence: float = 0.8
    noise_strength: float = 0.1
    chemotaxis_sensitivity: float = 1.0
    thigmotaxis_sensitivity: float = 0.5

class HyphalGrowthABM:
    """Agent-Based Model for hyphal growth simulation"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 50),
                 growth_params: GrowthParameters = None):
        """
        Initialize the hyphal growth ABM
        
        Args:
            domain_size: (width, height, depth) of the simulation domain
            growth_params: Parameters controlling growth behavior
        """
        self.domain_size = domain_size
        self.growth_params = growth_params or GrowthParameters()
        
        # Initialize hyphal tips
        self.hyphal_tips: List[HyphalTip] = []
        self.hyphal_segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        
        # Initialize environment
        self.environment = self._create_default_environment()
        
        # Simulation state
        self.time = 0.0
        self.dt = 0.1
        self.step_count = 0
        
        # Statistics
        self.stats = {
            'total_tips': [],
            'active_tips': [],
            'total_length': [],
            'branching_events': 0,
            'anastomosis_events': 0
        }
        
        logger.info(f"Initialized HyphalGrowthABM with domain {domain_size}")
    
    def _create_default_environment(self) -> Environment:
        """Create a default environment for testing"""
        w, h, d = self.domain_size
        
        # Create nutrient gradient (higher near center)
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        z = np.linspace(0, 1, d)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Nutrient concentration decreases from center
        center_dist = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
        nutrient_conc = np.exp(-center_dist * 2)
        
        # pH gradient (slightly acidic in center)
        ph_gradient = 6.0 + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        
        # Temperature gradient (warmer in center)
        temperature = 22.0 + 2.0 * (1 - center_dist)
        
        # Oxygen concentration (higher at surface)
        oxygen_conc = 0.008 * (1 - Z)
        
        # Root exudates (higher near bottom)
        root_exudates = 0.1 * np.exp(-Z * 3)
        
        # No obstacles initially
        obstacles = np.zeros((w, h, d), dtype=bool)
        
        return Environment(
            nutrient_concentration=nutrient_conc,
            ph_gradient=ph_gradient,
            temperature=temperature,
            oxygen_concentration=oxygen_conc,
            root_exudates=root_exudates,
            obstacles=obstacles
        )
    
    def add_hyphal_tip(self, x: float, y: float, z: float, 
                      direction: float = None, parent_id: int = None):
        """Add a new hyphal tip to the simulation"""
        if direction is None:
            direction = random.uniform(0, 2 * np.pi)
        
        tip = HyphalTip(
            x=x, y=y, z=z,
            direction=direction,
            parent_id=parent_id,
            generation=0 if parent_id is None else self.hyphal_tips[parent_id].generation + 1
        )
        
        self.hyphal_tips.append(tip)
        logger.debug(f"Added hyphal tip at ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def _get_environment_at_position(self, x: float, y: float, z: float) -> Dict[str, float]:
        """Get environmental conditions at a specific position"""
        w, h, d = self.domain_size
        
        # Clamp coordinates to domain bounds
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        z = max(0, min(d-1, z))
        
        # Interpolate (simple nearest neighbor for now)
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        
        return {
            'nutrient': self.environment.nutrient_concentration[xi, yi, zi],
            'ph': self.environment.ph_gradient[xi, yi, zi],
            'temperature': self.environment.temperature[xi, yi, zi],
            'oxygen': self.environment.oxygen_concentration[xi, yi, zi],
            'root_exudates': self.environment.root_exudates[xi, yi, zi]
        }
    
    def _compute_chemotaxis_direction(self, tip: HyphalTip) -> float:
        """Compute direction change due to chemotaxis"""
        env = self._get_environment_at_position(tip.x, tip.y, tip.z)
        
        # Compute gradient of nutrient concentration
        dx = 1.0
        dy = 1.0
        dz = 1.0
        
        # Finite difference approximation
        env_xp = self._get_environment_at_position(tip.x + dx, tip.y, tip.z)
        env_xm = self._get_environment_at_position(tip.x - dx, tip.y, tip.z)
        env_yp = self._get_environment_at_position(tip.x, tip.y + dy, tip.z)
        env_ym = self._get_environment_at_position(tip.x, tip.y - dy, tip.z)
        env_zp = self._get_environment_at_position(tip.x, tip.y, tip.z + dz)
        env_zm = self._get_environment_at_position(tip.x, tip.y, tip.z - dz)
        
        grad_nutrient_x = (env_xp['nutrient'] - env_xm['nutrient']) / (2 * dx)
        grad_nutrient_y = (env_yp['nutrient'] - env_ym['nutrient']) / (2 * dy)
        grad_nutrient_z = (env_zp['nutrient'] - env_zm['nutrient']) / (2 * dz)
        
        # Compute gradient direction
        if np.linalg.norm([grad_nutrient_x, grad_nutrient_y, grad_nutrient_z]) > 1e-6:
            gradient_direction = np.arctan2(grad_nutrient_y, grad_nutrient_x)
            return gradient_direction
        else:
            return tip.direction
    
    def _compute_thigmotaxis_direction(self, tip: HyphalTip) -> float:
        """Compute direction change due to thigmotaxis (contact guidance)"""
        # Check for nearby obstacles or surfaces
        search_radius = 2.0
        w, h, d = self.domain_size
        
        # Simple obstacle avoidance
        if (tip.x < search_radius or tip.x > w - search_radius or
            tip.y < search_radius or tip.y > h - search_radius or
            tip.z < search_radius or tip.z > d - search_radius):
            # Turn away from boundary
            center_x, center_y, center_z = w/2, h/2, d/2
            away_direction = np.arctan2(tip.y - center_y, tip.x - center_x)
            return away_direction
        
        return tip.direction
    
    def _compute_ph_tropism_direction(self, tip: HyphalTip) -> float:
        """Compute direction change due to pH tropism"""
        env = self._get_environment_at_position(tip.x, tip.y, tip.z)
        optimal_ph = 6.0
        
        # Compute pH gradient
        dx = 1.0
        dy = 1.0
        
        env_xp = self._get_environment_at_position(tip.x + dx, tip.y, tip.z)
        env_xm = self._get_environment_at_position(tip.x - dx, tip.y, tip.z)
        env_yp = self._get_environment_at_position(tip.x, tip.y + dy, tip.z)
        env_ym = self._get_environment_at_position(tip.x, tip.y - dy, tip.z)
        
        grad_ph_x = (env_xp['ph'] - env_xm['ph']) / (2 * dx)
        grad_ph_y = (env_yp['ph'] - env_ym['ph']) / (2 * dy)
        
        # Move towards optimal pH
        ph_diff = env['ph'] - optimal_ph
        if abs(ph_diff) > 0.1:  # Only respond to significant pH differences
            gradient_direction = np.arctan2(-grad_ph_y, -grad_ph_x)  # Negative gradient
            return gradient_direction
        
        return tip.direction
    
    def _update_hyphal_tip(self, tip: HyphalTip) -> List[HyphalTip]:
        """Update a single hyphal tip and return new tips from branching"""
        if not tip.active:
            return []
        
        new_tips = []
        
        # Update age
        tip.age += self.dt
        
        # Check if tip should die
        if tip.age > self.growth_params.max_tip_age:
            tip.active = False
            return []
        
        # Get environmental conditions
        env = self._get_environment_at_position(tip.x, tip.y, tip.z)
        
        # Compute growth rate based on environment
        nutrient_factor = min(1.0, env['nutrient'] / 0.5)  # Normalize to max nutrient
        oxygen_factor = min(1.0, env['oxygen'] / 0.008)  # Normalize to max oxygen
        ph_factor = 1.0 - abs(env['ph'] - 6.0) / 2.0  # Optimal at pH 6.0
        ph_factor = max(0.1, ph_factor)  # Minimum growth rate
        
        growth_rate = (self.growth_params.base_growth_rate * 
                      nutrient_factor * oxygen_factor * ph_factor)
        growth_rate = np.clip(growth_rate, self.growth_params.min_growth_rate, 
                             self.growth_params.max_growth_rate)
        
        # Update direction based on tropisms
        chemotaxis_dir = self._compute_chemotaxis_direction(tip)
        thigmotaxis_dir = self._compute_thigmotaxis_direction(tip)
        ph_tropism_dir = self._compute_ph_tropism_direction(tip)
        
        # Combine directions with weights
        weights = [
            self.environment.chemotaxis_strength,
            self.environment.thigmotaxis_strength,
            self.environment.ph_tropism_strength
        ]
        directions = [chemotaxis_dir, thigmotaxis_dir, ph_tropism_dir]
        
        # Weighted average of directions
        weighted_direction = np.average(directions, weights=weights)
        
        # Add noise
        noise = np.random.normal(0, self.growth_params.noise_strength)
        tip.direction = weighted_direction + noise
        
        # Normalize direction
        tip.direction = tip.direction % (2 * np.pi)
        
        # Move tip
        step_size = growth_rate * self.dt
        tip.x += step_size * np.cos(tip.direction)
        tip.y += step_size * np.sin(tip.direction)
        tip.z += step_size * np.sin(tip.direction) * 0.1  # Slight vertical component
        
        # Update length
        tip.length += step_size
        
        # Check for branching
        if (random.random() < self.growth_params.branching_rate * self.dt and
            tip.age > 10.0):  # Minimum age before branching
            
            # Create new branch
            branch_direction = tip.direction + random.uniform(-np.pi/3, np.pi/3)
            new_tip = HyphalTip(
                x=tip.x, y=tip.y, z=tip.z,
                direction=branch_direction,
                parent_id=len(self.hyphal_tips),
                generation=tip.generation + 1,
                growth_rate=tip.growth_rate * random.uniform(0.8, 1.2)
            )
            new_tips.append(new_tip)
            
            self.stats['branching_events'] += 1
        
        # Check for anastomosis with nearby tips
        for other_tip in self.hyphal_tips:
            if (other_tip != tip and other_tip.active and
                np.sqrt((tip.x - other_tip.x)**2 + (tip.y - other_tip.y)**2 + 
                       (tip.z - other_tip.z)**2) < self.growth_params.anastomosis_distance):
                
                if random.random() < self.growth_params.anastomosis_probability:
                    # Merge tips (simplified - just deactivate one)
                    other_tip.active = False
                    self.stats['anastomosis_events'] += 1
                    break
        
        return new_tips
    
    def step(self):
        """Perform one simulation step"""
        self.step_count += 1
        self.time += self.dt
        
        # Update all active tips
        new_tips = []
        for tip in self.hyphal_tips:
            if tip.active:
                new_tips.extend(self._update_hyphal_tip(tip))
        
        # Add new tips from branching
        self.hyphal_tips.extend(new_tips)
        
        # Update statistics
        active_tips = sum(1 for tip in self.hyphal_tips if tip.active)
        total_length = sum(tip.length for tip in self.hyphal_tips)
        
        self.stats['total_tips'].append(len(self.hyphal_tips))
        self.stats['active_tips'].append(active_tips)
        self.stats['total_length'].append(total_length)
        
        if self.step_count % 100 == 0:
            logger.info(f"Step {self.step_count}: {len(self.hyphal_tips)} tips, "
                       f"{active_tips} active, total length {total_length:.2f}")
    
    def run(self, n_steps: int):
        """Run the simulation for n_steps"""
        logger.info(f"Running simulation for {n_steps} steps")
        
        for i in range(n_steps):
            self.step()
        
        logger.info(f"Simulation completed. Final stats: {len(self.hyphal_tips)} tips, "
                   f"{self.stats['branching_events']} branching events, "
                   f"{self.stats['anastomosis_events']} anastomosis events")
    
    def get_hyphal_density(self) -> np.ndarray:
        """Compute hyphal density field from current tips"""
        w, h, d = self.domain_size
        density = np.zeros((w, h, d))
        
        for tip in self.hyphal_tips:
            if tip.active:
                # Add Gaussian kernel centered at tip position
                x, y, z = int(round(tip.x)), int(round(tip.y)), int(round(tip.z))
                if 0 <= x < w and 0 <= y < h and 0 <= z < d:
                    # Simple density addition (could be improved with proper kernel)
                    density[x, y, z] += 1.0
        
        # Smooth the density field
        density = gaussian_filter(density, sigma=1.0)
        
        return density
    
    def plot_hyphae(self, ax=None, show_3d=False):
        """Plot the current hyphal network"""
        if ax is None:
            if show_3d:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot hyphal segments (simplified - just points for now)
        if show_3d:
            for tip in self.hyphal_tips:
                if tip.active:
                    ax.scatter(tip.x, tip.y, tip.z, c='b', s=1)
                else:
                    ax.scatter(tip.x, tip.y, tip.z, c='r', s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            for tip in self.hyphal_tips:
                if tip.active:
                    ax.scatter(tip.x, tip.y, c='b', s=1)
                else:
                    ax.scatter(tip.x, tip.y, c='r', s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        ax.set_title(f'Hyphal Network (t={self.time:.2f})')
        plt.tight_layout()
        return ax
    
    def plot_statistics(self):
        """Plot simulation statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total tips over time
        axes[0, 0].plot(self.stats['total_tips'])
        axes[0, 0].set_title('Total Tips')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Count')
        
        # Active tips over time
        axes[0, 1].plot(self.stats['active_tips'])
        axes[0, 1].set_title('Active Tips')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Count')
        
        # Total length over time
        axes[1, 0].plot(self.stats['total_length'])
        axes[1, 0].set_title('Total Hyphal Length')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Length')
        
        # Growth rate distribution
        growth_rates = [tip.growth_rate for tip in self.hyphal_tips if tip.active]
        if growth_rates:
            axes[1, 1].hist(growth_rates, bins=20, alpha=0.7)
            axes[1, 1].set_title('Growth Rate Distribution')
            axes[1, 1].set_xlabel('Growth Rate')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of the hyphal growth ABM"""
    # Create ABM
    growth_params = GrowthParameters(
        base_growth_rate=1.0,
        branching_rate=0.05,
        anastomosis_distance=3.0
    )
    
    abm = HyphalGrowthABM(domain_size=(50, 50, 25), growth_params=growth_params)
    
    # Add initial hyphal tips
    for i in range(5):
        x = random.uniform(20, 30)
        y = random.uniform(20, 30)
        z = random.uniform(10, 15)
        abm.add_hyphal_tip(x, y, z)
    
    # Run simulation
    abm.run(500)
    
    # Plot results
    abm.plot_hyphae()
    abm.plot_statistics()
    
    # Get hyphal density for coupling with PDE solver
    density = abm.get_hyphal_density()
    print(f"Hyphal density shape: {density.shape}")
    print(f"Max density: {density.max():.3f}")

if __name__ == "__main__":
    main()