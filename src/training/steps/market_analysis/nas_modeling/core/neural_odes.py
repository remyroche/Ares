"""
Neural Ordinary Differential Equations (Neural ODEs) for NAS

This module implements Neural ODEs for continuous-time modeling:
- Neural ODE layers for time series
- Continuous-time regime detection
- ODE-based architecture search
- Adaptive time-stepping
- Event detection and handling
- Continuous normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import copy
from pathlib import Path
try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None
    odeint_adjoint = None

logger = logging.getLogger(__name__)

@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODEs."""
    state_size: int = 64
    hidden_size: int = 128
    time_points: int = 10
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-6
    adjoint: bool = True
    use_adjoint: bool = True
    use_checkpointing: bool = True
    max_num_steps: int = 1000
    step_size: float = 0.1
    use_adaptive_stepping: bool = True
    event_detection: bool = True

class ODEFunction(nn.Module):
    """
    Neural network that defines the ODE dynamics.

    f(t, x) where x is the state at time t.
    """

    def __init__(self, state_size: int, hidden_size: int = 128):
        """Initialize ODE function.

        Args:
            state_size: Size of the state vector
            hidden_size: Size of hidden layers
        """
        super(ODEFunction, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_size)
        )

        # Initialize weights for better numerical stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass defining ODE dynamics.

        Args:
            t: Time point
            x: State at time t

        Returns:
            State derivative dx/dt
        """
        return self.net(x)

class NeuralODE(nn.Module):
    """
    Neural ODE layer for continuous-time modeling.

    Solves ODEs defined by ODEFunction to model continuous dynamics.
    """

    def __init__(self, config: NeuralODEConfig):
        """Initialize Neural ODE.

        Args:
            config: Neural ODE configuration
        """
        super(NeuralODE, self).__init__()
        self.config = config

        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required for Neural ODEs. Install with: pip install torchdiffeq")

        # ODE function
        self.ode_func = ODEFunction(config.state_size, config.hidden_size)

        # Time span for integration
        self.time_span = torch.linspace(0, 1, config.time_points)

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Neural ODE.

        Args:
            x: Input tensor

        Returns:
            Solution of ODE at final time point
        """
        # Initial state
        x0 = x

        # Solve ODE
        if self.config.use_adjoint and self.config.adjoint:
            solution = odeint_adjoint(
                self.ode_func,
                x0,
                self.time_span,
                rtol=self.config.rtol,
                atol=self.config.atol,
                method=self.config.method
            )
        else:
            solution = odeint(
                self.ode_func,
                x0,
                self.time_span,
                rtol=self.config.rtol,
                atol=self.config.atol,
                method=self.config.method
            )

        # Return final state
        return solution[-1]

class ContinuousTimeRegimeDetector(nn.Module):
    """
    Continuous-time regime detector using Neural ODEs.

    Models market regime evolution as continuous-time dynamics.
    """

    def __init__(self, input_size: int = 4, state_size: int = 64, num_regimes: int = 5):
        """Initialize continuous-time regime detector.

        Args:
            input_size: Input feature dimension
            state_size: Size of continuous state
            num_regimes: Number of market regimes
        """
        super(ContinuousTimeRegimeDetector, self).__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.num_regimes = num_regimes

        # Initial state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_size, state_size),
            nn.Tanh(),
            nn.Linear(state_size, state_size)
        )

        # Neural ODE for continuous dynamics
        ode_config = NeuralODEConfig(state_size=state_size, time_points=20)
        self.ode_layer = NeuralODE(ode_config)

        # Regime classifier from continuous states
        self.regime_classifier = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.ReLU(),
            nn.Linear(state_size // 2, num_regimes),
            nn.LogSoftmax(dim=1)
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through continuous-time regime detector.

        Args:
            x: Input time series data

        Returns:
            Regime probabilities
        """
        batch_size, seq_len, input_size = x.shape

        # Encode initial state from first time step
        initial_state = self.state_encoder(x[:, 0, :])  # (batch_size, state_size)

        # Apply Neural ODE for continuous evolution
        final_state = self.ode_layer(initial_state)  # (batch_size, state_size)

        # Classify regime from final state
        regime_logits = self.regime_classifier(final_state)

        return regime_logits

    def get_continuous_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get continuous state evolution over time.

        Args:
            x: Input time series

        Returns:
            State evolution over time
        """
        batch_size = x.size(0)

        # Encode initial state
        x0 = self.state_encoder(x[:, 0, :])

        # Solve ODE over time span
        time_span = torch.linspace(0, 1, x.size(1))
        states_over_time = odeint(self.ode_func, x0, time_span)

        return states_over_time

class AdaptiveNeuralODE(nn.Module):
    """
    Neural ODE with adaptive time-stepping and event detection.

    Adapts integration steps based on solution dynamics.
    """

    def __init__(self, config: NeuralODEConfig):
        """Initialize adaptive Neural ODE.

        Args:
            config: Neural ODE configuration
        """
        super(AdaptiveNeuralODE, self).__init__()
        self.config = config

        self.ode_func = ODEFunction(config.state_size, config.hidden_size)
        self.event_detector = EventDetector() if config.event_detection else None

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor, return_states: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with adaptive stepping.

        Args:
            x: Input tensor
            return_states: Whether to return all states

        Returns:
            Final state or tuple of (final_state, all_states)
        """
        x0 = x

        def adaptive_ode_func(t, x):
            # Add adaptive control based on state
            dx_dt = self.ode_func(t, x)

            # Detect events
            if self.event_detector is not None:
                events = self.event_detector.detect_events(x)
                if events.any():
                    # Adjust dynamics based on events
                    dx_dt = self._handle_events(dx_dt, events)

            return dx_dt

        # Solve with adaptive time-stepping
        if self.config.use_adaptive_stepping:
            solution = odeint(
                adaptive_ode_func,
                x0,
                self.config.time_span,
                rtol=self.config.rtol,
                atol=self.config.atol,
                method=self.config.method,
                options={'max_num_steps': self.config.max_num_steps}
            )
        else:
            solution = odeint(adaptive_ode_func, x0, self.config.time_span)

        if return_states:
            return solution[-1], solution
        else:
            return solution[-1]

    def _handle_events(self, dx_dt: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Handle detected events by adjusting dynamics."""
        # Modify dynamics based on detected events
        # This could include changing regime, applying interventions, etc.
        return dx_dt * (1 + events.float() * 0.1)  # Small perturbation for events

class EventDetector(nn.Module):
    """
    Event detection for Neural ODEs.

    Detects important events in the continuous state evolution.
    """

    def __init__(self, state_size: int = 64, event_threshold: float = 0.1):
        """Initialize event detector.

        Args:
            state_size: Size of state vector
            event_threshold: Threshold for event detection
        """
        super(EventDetector, self).__init__()
        self.state_size = state_size
        self.event_threshold = event_threshold

        # Event detection network
        self.event_network = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.ReLU(),
            nn.Linear(state_size // 2, state_size // 4),
            nn.ReLU(),
            nn.Linear(state_size // 4, 1),
            nn.Sigmoid()
        )

    def detect_events(self, state: torch.Tensor) -> torch.Tensor:
        """Detect events in current state.

        Args:
            state: Current state

        Returns:
            Event indicators
        """
        event_probs = self.event_network(state)
        events = (event_probs > self.event_threshold).float()

        return events

class ContinuousNormalization(nn.Module):
    """
    Continuous normalization for Neural ODEs.

    Provides continuous-time normalization that evolves with the ODE solution.
    """

    def __init__(self, state_size: int, affine: bool = True):
        """Initialize continuous normalization.

        Args:
            state_size: Size of state vector
            affine: Whether to use affine parameters
        """
        super(ContinuousNormalization, self).__init__()
        self.state_size = state_size
        self.affine = affine

        # Normalization parameters
        self.register_buffer('running_mean', torch.zeros(state_size))
        self.register_buffer('running_var', torch.ones(state_size))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        if affine:
            self.weight = nn.Parameter(torch.ones(state_size))
            self.bias = nn.Parameter(torch.zeros(state_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through continuous normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        if self.training:
            # Update running statistics during training
            self._update_running_stats(x)

        # Normalize
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)

        if self.affine:
            normalized = normalized * self.weight + self.bias

        return normalized

    def _update_running_stats(self, x: torch.Tensor):
        """Update running statistics."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)

        # Update running statistics using momentum
        momentum = 0.1
        self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
        self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
        self.num_batches_tracked += 1

class ODESearchSpace:
    """
    Search space for Neural ODE architectures.

    Defines the space of possible Neural ODE configurations.
    """

    def __init__(self):
        """Initialize ODE search space."""
        self.state_sizes = [32, 64, 128, 256]
        self.hidden_sizes = [64, 128, 256, 512]
        self.time_points_options = [5, 10, 20, 50]
        self.integration_methods = ["dopri5", "rk4", "euler", "adams"]
        self.use_adjoint_options = [True, False]
        self.use_events_options = [True, False]

    def sample_architecture(self) -> Dict[str, Any]:
        """Sample random Neural ODE architecture."""
        architecture = {
            'state_size': np.random.choice(self.state_sizes),
            'hidden_size': np.random.choice(self.hidden_sizes),
            'time_points': np.random.choice(self.time_points_options),
            'method': np.random.choice(self.integration_methods),
            'use_adjoint': np.random.choice(self.use_adjoint_options),
            'use_events': np.random.choice(self.use_events_options),
            'rtol': 10 ** np.random.uniform(-5, -3),  # 1e-5 to 1e-3
            'atol': 10 ** np.random.uniform(-8, -5),  # 1e-8 to 1e-5
        }

        return architecture

    def create_ode_config(self, architecture: Dict[str, Any]) -> NeuralODEConfig:
        """Create NeuralODEConfig from architecture."""
        return NeuralODEConfig(
            state_size=architecture['state_size'],
            hidden_size=architecture['hidden_size'],
            time_points=architecture['time_points'],
            method=architecture['method'],
            rtol=architecture['rtol'],
            atol=architecture['atol'],
            adjoint=architecture['use_adjoint'],
            event_detection=architecture['use_events']
        )

class NeuralODE_NAS:
    """
    Neural Architecture Search for Neural ODEs.

    Searches for optimal Neural ODE architectures using various search strategies.
    """

    def __init__(self, search_strategy: str = "random"):
        """Initialize Neural ODE NAS.

        Args:
            search_strategy: Search strategy ("random", "evolution", "bayesian")
        """
        self.search_strategy = search_strategy
        self.search_space = ODESearchSpace()
        self.logger = logging.getLogger(self.__class__.__name__)

    def search(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray],
              num_trials: int = 50) -> Dict[str, Any]:
        """
        Search for optimal Neural ODE architecture.

        Args:
            train_data: Training data
            val_data: Validation data
            num_trials: Number of search trials

        Returns:
            Search results
        """
        logger.info(f"🚀 Starting Neural ODE NAS with {num_trials} trials")

        best_architecture = None
        best_performance = float('-inf')

        for trial in range(num_trials):
            # Sample architecture
            architecture = self.search_space.sample_architecture()

            # Create Neural ODE
            ode_config = self.search_space.create_ode_config(architecture)
            model = ContinuousTimeRegimeDetector(
                input_size=train_data[0].shape[-1],
                state_size=ode_config.state_size,
                num_regimes=5  # Assuming 5 regimes
            )

            # Evaluate
            performance = self._evaluate_ode_model(model, train_data, val_data)

            if performance > best_performance:
                best_performance = performance
                best_architecture = architecture

            if trial % 10 == 0:
                self.logger.info(f"📈 Trial {trial}: Best performance = {best_performance:.4f}")

        results = {
            'best_architecture': best_architecture,
            'best_performance': best_performance,
            'search_strategy': self.search_strategy,
            'num_trials': num_trials,
            'ode_config': self.search_space.create_ode_config(best_architecture)
        }

        self.logger.info(f"✅ Neural ODE NAS completed with best performance: {best_performance:.4f}")
        return results

    def _evaluate_ode_model(self, model: nn.Module,
                           train_data: Tuple[np.ndarray, np.ndarray],
                           val_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate Neural ODE model."""
        try:
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(train_data, val_data)

            # Train for a few epochs
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.NLLLoss()

            model.train()
            for epoch in range(3):  # Quick evaluation
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)

            accuracy = correct / total
            return accuracy

        except Exception as e:
            self.logger.warning(f"⚠️ ODE model evaluation failed: {e}")
            return 0.0

    def _create_data_loaders(self, train_data: Tuple[np.ndarray, np.ndarray],
                           val_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create data loaders."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

class HybridNeuralODE(nn.Module):
    """
    Hybrid Neural ODE combining discrete and continuous modeling.

    Uses Neural ODEs for continuous evolution between discrete events.
    """

    def __init__(self, discrete_model: nn.Module, continuous_config: NeuralODEConfig):
        """Initialize hybrid Neural ODE.

        Args:
            discrete_model: Discrete neural network model
            continuous_config: Configuration for continuous dynamics
        """
        super(HybridNeuralODE, self).__init__()
        self.discrete_model = discrete_model

        # Continuous dynamics between discrete steps
        self.continuous_dynamics = NeuralODE(continuous_config)

        # Event detection for switching between discrete and continuous
        self.event_detector = EventDetector(continuous_config.state_size)

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        # Initial processing with discrete model
        discrete_output = self.discrete_model(x)

        # Continuous evolution
        continuous_output = self.continuous_dynamics(discrete_output)

        # Detect events
        events = self.event_detector.detect_events(continuous_output)

        # Combine discrete and continuous representations
        combined_output = discrete_output + continuous_output

        return combined_output

# Utility functions
def create_neural_ode(config: NeuralODEConfig) -> NeuralODE:
    """Create Neural ODE model."""
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq is required for Neural ODEs")
    return NeuralODE(config)

def create_continuous_regime_detector(input_size: int = 4, state_size: int = 64,
                                    num_regimes: int = 5) -> ContinuousTimeRegimeDetector:
    """Create continuous-time regime detector."""
    return ContinuousTimeRegimeDetector(input_size, state_size, num_regimes)

def create_adaptive_neural_ode(config: NeuralODEConfig) -> AdaptiveNeuralODE:
    """Create adaptive Neural ODE."""
    return AdaptiveNeuralODE(config)

def create_hybrid_ode(discrete_model: nn.Module,
                     continuous_config: NeuralODEConfig) -> HybridNeuralODE:
    """Create hybrid Neural ODE."""
    return HybridNeuralODE(discrete_model, continuous_config)

def solve_ode_with_events(ode_func: ODEFunction, x0: torch.Tensor,
                         time_span: torch.Tensor, event_fn: Optional[Callable] = None) -> torch.Tensor:
    """Solve ODE with event detection."""
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq is required")

    def event_wrapper(t, x):
        dx_dt = ode_func(t, x)

        if event_fn is not None:
            events = event_fn(x)
            # Modify dynamics based on events
            dx_dt = dx_dt * (1 + events.float() * 0.1)

        return dx_dt

    solution = odeint(event_wrapper, x0, time_span)
    return solution[-1]  # Return final state