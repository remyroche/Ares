"""
Neural ODEs

Implementation for Neural Ordinary Differential Equations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.integrate import odeint


class ODESolver(Enum):
    """Types of ODE solvers."""
    EULER = "euler"
    RK4 = "rk4"
    DORMAND_PRINCE = "dormand_prince"
    ADAPTIVE = "adaptive"


@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODE."""
    hidden_dim: int
    num_layers: int
    activation: str = "tanh"
    solver: ODESolver = ODESolver.RK4
    rtol: float = 1e-3
    atol: float = 1e-4


class NeuralODE:
    """Neural Ordinary Differential Equation implementation."""
    
    def __init__(self, config: NeuralODEConfig):
        """Initialize Neural ODE.
        
        Args:
            config: Configuration for the Neural ODE
        """
        self.config = config
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
        self.activation_fn = self._get_activation_function()
        
    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize network weights."""
        weights = []
        dims = [self.config.hidden_dim] * self.config.num_layers
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * 0.1
            weights.append(w)
        
        return weights
    
    def _initialize_biases(self) -> List[np.ndarray]:
        """Initialize network biases."""
        biases = []
        dims = [self.config.hidden_dim] * self.config.num_layers
        
        for i in range(len(dims) - 1):
            b = np.zeros(dims[i + 1])
            biases.append(b)
        
        return biases
    
    def _get_activation_function(self) -> Callable:
        """Get activation function."""
        if self.config.activation == "tanh":
            return np.tanh
        elif self.config.activation == "relu":
            return lambda x: np.maximum(0, x)
        elif self.config.activation == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif self.config.activation == "swish":
            return lambda x: x * (1 / (1 + np.exp(-x)))
        else:
            return np.tanh
    
    def _neural_network(self, t: float, y: np.ndarray) -> np.ndarray:
        """Neural network function for ODE."""
        x = y.copy()
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            if i < len(self.weights) - 1:  # Don't apply activation to output
                x = self.activation_fn(x)
        
        return x
    
    def forward(self, t_span: Tuple[float, float], y0: np.ndarray, 
                t_eval: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through Neural ODE.
        
        Args:
            t_span: Time span (t_start, t_end)
            y0: Initial conditions
            t_eval: Optional time points for evaluation
            
        Returns:
            Solution array
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        
        if self.config.solver == ODESolver.EULER:
            return self._euler_solve(t_span, y0, t_eval)
        elif self.config.solver == ODESolver.RK4:
            return self._rk4_solve(t_span, y0, t_eval)
        else:
            return self._adaptive_solve(t_span, y0, t_eval)
    
    def _euler_solve(self, t_span: Tuple[float, float], y0: np.ndarray, 
                    t_eval: np.ndarray) -> np.ndarray:
        """Euler method solver."""
        dt = t_eval[1] - t_eval[0]
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t_eval)):
            dy = self._neural_network(t_eval[i-1], y[i-1])
            y[i] = y[i-1] + dt * dy
        
        return y
    
    def _rk4_solve(self, t_span: Tuple[float, float], y0: np.ndarray, 
                   t_eval: np.ndarray) -> np.ndarray:
        """Runge-Kutta 4th order solver."""
        dt = t_eval[1] - t_eval[0]
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t_eval)):
            k1 = self._neural_network(t_eval[i-1], y[i-1])
            k2 = self._neural_network(t_eval[i-1] + dt/2, y[i-1] + dt*k1/2)
            k3 = self._neural_network(t_eval[i-1] + dt/2, y[i-1] + dt*k2/2)
            k4 = self._neural_network(t_eval[i-1] + dt, y[i-1] + dt*k3)
            
            y[i] = y[i-1] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return y
    
    def _adaptive_solve(self, t_span: Tuple[float, float], y0: np.ndarray, 
                       t_eval: np.ndarray) -> np.ndarray:
        """Adaptive solver using scipy."""
        try:
            sol = odeint(self._neural_network, y0, t_eval, 
                        rtol=self.config.rtol, atol=self.config.atol)
            return sol
        except Exception:
            # Fallback to RK4 if adaptive solver fails
            return self._rk4_solve(t_span, y0, t_eval)
    
    def predict(self, t_span: Tuple[float, float], y0: np.ndarray, 
                t_eval: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions using the Neural ODE."""
        return self.forward(t_span, y0, t_eval)
    
    def get_derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """Get derivative at given time and state."""
        return self._neural_network(t, y)
    
    def update_weights(self, new_weights: List[np.ndarray]):
        """Update network weights."""
        if len(new_weights) == len(self.weights):
            self.weights = new_weights
        else:
            raise ValueError("Number of weight matrices must match current architecture")
    
    def update_biases(self, new_biases: List[np.ndarray]):
        """Update network biases."""
        if len(new_biases) == len(self.biases):
            self.biases = new_biases
        else:
            raise ValueError("Number of bias vectors must match current architecture")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'biases': self.biases,
            'config': self.config
        }
    
    def set_parameters(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        """Set model parameters."""
        self.weights = weights
        self.biases = biases
