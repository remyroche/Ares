"""
Advanced Search Strategies for NAS and TAS Systems

This module provides advanced search strategies including reinforcement learning,
enhanced Bayesian optimization, and adaptive evolutionary algorithms that can be
used by both neural and tree architecture search systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import random
from collections import defaultdict, deque
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class SearchStrategyType(Enum):
    """Types of advanced search strategies."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENHANCED_BAYESIAN = "enhanced_bayesian"
    ADAPTIVE_EVOLUTIONARY = "adaptive_evolutionary"
    MONTE_CARLO_TREE_SEARCH = "monte_carlo_tree_search"
    PARTICLE_SWARM = "particle_swarm"
    GENETIC_PROGRAMMING = "genetic_programming"
    HYBRID_META_LEARNING = "hybrid_meta_learning"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"


class RLAgentType(Enum):
    """Types of reinforcement learning agents."""
    Q_LEARNING = "q_learning"
    DEEP_Q_NETWORK = "deep_q_network"
    POLICY_GRADIENT = "policy_gradient"
    ACTOR_CRITIC = "actor_critic"
    PROXIMAL_POLICY_OPTIMIZATION = "ppo"


@dataclass
class RLState:
    """State representation for reinforcement learning."""
    architecture_encoding: np.ndarray
    search_iteration: int
    current_performance: float
    best_performance: float
    time_elapsed: float
    n_architectures_evaluated: int
    exploration_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLAction:
    """Action representation for reinforcement learning."""
    action_type: str  # 'expand_layer', 'modify_activation', 'add_connection', etc.
    parameters: Dict[str, Any]
    expected_impact: float
    risk_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLReward:
    """Reward signal for reinforcement learning."""
    performance_improvement: float
    architecture_complexity: float
    resource_efficiency: float
    exploration_bonus: float
    constraint_satisfaction: float
    total_reward: float


@dataclass
class SearchStrategyResult:
    """Result from advanced search strategy."""
    best_architecture: Any
    best_score: float
    search_history: List[Dict[str, Any]]
    strategy_used: str
    convergence_info: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdvancedSearchStrategy:
    """Base class for advanced search strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the search strategy."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False

    def search(self,
               architecture_generator: Callable,
               performance_evaluator: Callable,
               constraint_validator: Callable,
               n_iterations: int = 100) -> SearchStrategyResult:
        """Perform advanced search."""
        try:
            tprint(f"🔍 [SEARCH] Starting advanced search with {n_iterations} iterations", color="blue")
            
            best_architecture = None
            best_performance = float('-inf')
            search_history = []
            constraint_violations = []
            
            # Initialize search state
            self.reset()
            
            for iteration in range(n_iterations):
                try:
                    # Generate candidate architecture
                    candidate = architecture_generator()
                    
                    # Validate constraints
                    validation_result = constraint_validator(candidate)
                    if not validation_result.is_valid:
                        constraint_violations.append({
                            'iteration': iteration,
                            'violations': validation_result.violations
                        })
                        continue
                    
                    # Evaluate performance
                    performance = performance_evaluator(candidate)
                    
                    # Update best if better
                    if performance > best_performance:
                        best_performance = performance
                        best_architecture = candidate
                    
                    # Record search history
                    search_history.append({
                        'iteration': iteration,
                        'performance': performance,
                        'constraint_score': validation_result.score,
                        'architecture': candidate
                    })
                    
                    # Update search strategy state
                    self._update_search_state(candidate, performance, validation_result)
                    
                    # Log progress
                    if iteration % 10 == 0:
                        tprint(f"🔍 [SEARCH] Iteration {iteration}/{n_iterations}, Best: {best_performance:.4f}", color="cyan")
                        
                except Exception as e:
                    tprint(f"⚠️ [SEARCH] Error in iteration {iteration}: {e}", color="yellow")
                    continue
            
            # Calculate search statistics
            total_candidates = len(search_history)
            valid_candidates = len([h for h in search_history if h['constraint_score'] > 0])
            constraint_violation_rate = len(constraint_violations) / max(1, total_candidates)
            
            # Create search result
            result = SearchStrategyResult(
                best_architecture=best_architecture,
                best_performance=best_performance,
                search_history=search_history,
                constraint_violations=constraint_violations,
                total_iterations=n_iterations,
                valid_candidates=valid_candidates,
                constraint_violation_rate=constraint_violation_rate,
                search_strategy=self.__class__.__name__
            )
            
            tprint(f"✅ [SEARCH] Search completed. Best performance: {best_performance:.4f}", color="green")
            return result
            
        except Exception as e:
            tprint(f"❌ [SEARCH] Error in search: {e}", color="red")
            # Return empty result
            return SearchStrategyResult(
                best_architecture=None,
                best_performance=float('-inf'),
                search_history=[],
                constraint_violations=[],
                total_iterations=0,
                valid_candidates=0,
                constraint_violation_rate=1.0,
                search_strategy=self.__class__.__name__
            )
    
    def _update_search_state(self, candidate, performance, validation_result):
        """Update the search strategy state based on the candidate."""
        try:
            # This is a simplified implementation
            # In practice, this would update the search strategy's internal state
            # based on the candidate's performance and constraints
            
            # For now, just log the update
            pass
            
        except Exception as e:
            tprint(f"⚠️ [SEARCH] Error updating search state: {e}", color="yellow")

    def reset(self):
        """Reset the search strategy state."""
        pass

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the search strategy."""
        return {
            'strategy_type': self.__class__.__name__,
            'config': self.config,
            'is_initialized': self.is_initialized
        }


class ReinforcementLearningSearch(BaseAdvancedSearchStrategy):
    """Reinforcement learning based architecture search."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize RL search strategy."""
        super().__init__(config)
        self.agent_type = config.get('agent_type', RLAgentType.Q_LEARNING)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.exploration_rate = config.get('exploration_rate', 1.0)
        self.exploration_decay = config.get('exploration_decay', 0.995)
        self.min_exploration_rate = config.get('min_exploration_rate', 0.01)

        # State and action spaces
        self.state_dim = config.get('state_dimension', 50)
        self.action_dim = config.get('action_dimension', 20)
        self.max_actions_per_episode = config.get('max_actions_per_episode', 10)

        # RL components
        self.q_table = None
        self.policy_network = None
        self.value_network = None
        self.optimizer = None

        # Search state
        self.current_state = None
        self.current_architecture = None
        self.best_architecture = None
        self.best_score = -np.inf
        self.episode_reward = 0
        self.episode_count = 0

        self._initialize_rl_components()
        self.is_initialized = True
        self.logger.info(f"✅ RL Search Strategy initialized with {self.agent_type.value}")

    def _initialize_rl_components(self):
        """Initialize RL-specific components."""
        if self.agent_type in [RLAgentType.DEEP_Q_NETWORK, RLAgentType.ACTOR_CRITIC]:
            # Initialize neural networks for deep RL
            self.policy_network = self._create_policy_network()
            self.value_network = self._create_value_network() if self.agent_type == RLAgentType.ACTOR_CRITIC else None
            self.optimizer = optim.Adam(list(self.policy_network.parameters()), lr=self.learning_rate)

        elif self.agent_type == RLAgentType.Q_LEARNING:
            # Initialize Q-table
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim))

    def _create_policy_network(self) -> nn.Module:
        """Create policy network for deep RL."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def _create_value_network(self) -> nn.Module:
        """Create value network for actor-critic."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def search(self,
               architecture_generator: Callable,
               performance_evaluator: Callable,
               constraint_validator: Callable,
               n_iterations: int = 100) -> SearchStrategyResult:
        """Perform reinforcement learning based search."""
        start_time = time.time()
        search_history = []

        # Initialize first architecture
        self.current_architecture = architecture_generator()
        self.current_state = self._architecture_to_state(self.current_architecture)
        self.best_architecture = self.current_architecture
        self.best_score = performance_evaluator(self.current_architecture)

        search_history.append({
            'iteration': 0,
            'architecture': self.current_architecture,
            'score': self.best_score,
            'action': 'initialization',
            'reward': 0.0
        })

        for iteration in range(1, n_iterations + 1):
            # Select action
            if np.random.random() < self.exploration_rate:
                # Exploration: random action
                action = self._select_random_action()
            else:
                # Exploitation: best action
                action = self._select_best_action()

            # Execute action to get new architecture
            new_architecture = self._apply_action(self.current_architecture, action)

            # Validate constraints
            if not constraint_validator(new_architecture).is_valid:
                reward = -1.0  # Penalty for invalid architectures
                search_history.append({
                    'iteration': iteration,
                    'architecture': new_architecture,
                    'score': 0.0,
                    'action': action.action_type,
                    'reward': reward
                })
                continue

            # Evaluate performance
            current_score = performance_evaluator(self.current_architecture)
            new_score = performance_evaluator(new_architecture)

            # Calculate reward
            reward = self._calculate_reward(current_score, new_score, new_architecture)

            # Update best if improved
            if new_score > self.best_score:
                self.best_architecture = new_architecture
                self.best_score = new_score

            # Store in history
            search_history.append({
                'iteration': iteration,
                'architecture': new_architecture,
                'score': new_score,
                'action': action.action_type,
                'reward': reward,
                'parameters': action.parameters
            })

            # Update RL components
            new_state = self._architecture_to_state(new_architecture)
            self._update_rl_components(self.current_state, action, reward, new_state)

            # Update current state
            self.current_state = new_state
            self.current_architecture = new_architecture

            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate,
                                      self.exploration_rate * self.exploration_decay)

            # Log progress
            if iteration % 10 == 0:
                self.logger.info(f"RL Search - Iteration {iteration}: Best Score = {self.best_score:.4f}, "
                               f"Exploration Rate = {self.exploration_rate:.3f}")

        execution_time = time.time() - start_time

        return SearchStrategyResult(
            best_architecture=self.best_architecture,
            best_score=self.best_score,
            search_history=search_history,
            strategy_used=f"reinforcement_learning_{self.agent_type.value}",
            convergence_info={
                'final_exploration_rate': self.exploration_rate,
                'total_episodes': self.episode_count,
                'convergence_iteration': len(search_history)
            },
            execution_time=execution_time,
            metadata={'agent_type': self.agent_type.value}
        )

    def _architecture_to_state(self, architecture: Any) -> np.ndarray:
        """Convert architecture to RL state representation."""
        # Extract features from architecture
        if hasattr(architecture, 'layers'):
            # Neural architecture
            n_layers = len(architecture.layers)
            total_params = sum(layer.hidden_size * layer.hidden_size for layer in architecture.layers)
            n_connections = len(architecture.connections)
        else:
            # Tree architecture
            n_layers = len(architecture.trees)
            total_params = sum(tree.n_estimators for tree in architecture.trees)
            n_connections = 0

        # Create state vector
        state = np.array([
            n_layers / 20.0,  # Normalize layer count
            min(total_params / 1000000, 1.0),  # Normalize parameter count
            n_connections / 50.0,  # Normalize connection count
            architecture.estimated_complexity / 5.0,  # Normalize complexity
            architecture.estimated_memory_usage / 4096.0,  # Normalize memory
            self.exploration_rate,  # Current exploration rate
            self.best_score,  # Best score so far
            np.random.random()  # Random component for exploration
        ])

        # Pad to state dimension
        if len(state) < self.state_dim:
            padding = np.zeros(self.state_dim - len(state))
            state = np.concatenate([state, padding])

        return state

    def _select_random_action(self) -> RLAction:
        """Select a random action for exploration."""
        action_types = [
            'expand_layer', 'modify_activation', 'add_connection', 'remove_layer',
            'change_layer_size', 'modify_dropout', 'add_residual', 'remove_connection'
        ]

        action_type = np.random.choice(action_types)

        if action_type == 'expand_layer':
            parameters = {'position': np.random.randint(1, 10)}
        elif action_type == 'modify_activation':
            parameters = {'activation': np.random.choice(['relu', 'tanh', 'sigmoid'])}
        elif action_type == 'add_connection':
            parameters = {'from_layer': np.random.randint(0, 5), 'to_layer': np.random.randint(1, 6)}
        elif action_type == 'change_layer_size':
            parameters = {'new_size': np.random.choice([64, 128, 256, 512])}
        elif action_type == 'modify_dropout':
            parameters = {'dropout_rate': np.random.uniform(0.0, 0.5)}
        else:
            parameters = {}

        return RLAction(
            action_type=action_type,
            parameters=parameters,
            expected_impact=np.random.uniform(-0.1, 0.1),
            risk_level=np.random.uniform(0.0, 1.0)
        )

    def _select_best_action(self) -> RLAction:
        """Select the best action according to current policy."""
        if self.agent_type == RLAgentType.Q_LEARNING:
            # Q-learning: select action with highest Q-value
            state_key = tuple(self.current_state)
            q_values = self.q_table[state_key]
            best_action_idx = np.argmax(q_values)

            # Convert action index back to action
            return self._action_idx_to_action(best_action_idx)

        elif self.agent_type == RLAgentType.DEEP_Q_NETWORK:
            # Deep Q-Network: use policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
                q_values = self.policy_network(state_tensor)
                best_action_idx = torch.argmax(q_values).item()

            return self._action_idx_to_action(best_action_idx)

        else:
            # Default to random for other methods
            return self._select_random_action()

    def _action_idx_to_action(self, action_idx: int) -> RLAction:
        """Convert action index to RLAction."""
        action_mapping = [
            ('expand_layer', {'position': 1}),
            ('modify_activation', {'activation': 'relu'}),
            ('add_connection', {'from_layer': 0, 'to_layer': 1}),
            ('remove_layer', {}),
            ('change_layer_size', {'new_size': 128}),
            ('modify_dropout', {'dropout_rate': 0.1}),
            ('add_residual', {}),
            ('remove_connection', {})
        ]

        if 0 <= action_idx < len(action_mapping):
            action_type, parameters = action_mapping[action_idx]
        else:
            action_type, parameters = 'expand_layer', {'position': 1}

        return RLAction(
            action_type=action_type,
            parameters=parameters,
            expected_impact=0.0,
            risk_level=0.5
        )

    def _apply_action(self, architecture: Any, action: RLAction) -> Any:
        """Apply an action to an architecture to create a new one."""
        # This is a simplified implementation
        # In practice, you'd have specific logic for each architecture type

        if action.action_type == 'expand_layer':
            # Add a new layer (simplified)
            if hasattr(architecture, 'layers'):
                # Neural architecture
                new_layer = type(architecture.layers[0])(
                    layer_type=architecture.layers[0].layer_type,
                    hidden_size=64,
                    activation=architecture.layers[0].activation
                )
                architecture.layers.append(new_layer)
            else:
                # Tree architecture
                new_tree = type(architecture.trees[0])(
                    tree_type=architecture.trees[0].tree_type,
                    max_depth=5
                )
                architecture.trees.append(new_tree)

        elif action.action_type == 'change_layer_size':
            # Modify layer size
            if hasattr(architecture, 'layers') and architecture.layers:
                layer_idx = np.random.randint(0, len(architecture.layers))
                new_size = action.parameters.get('new_size', 128)
                architecture.layers[layer_idx].hidden_size = new_size

        # Recalculate architecture properties
        if hasattr(architecture, 'calculate_complexity'):
            architecture.estimated_complexity = architecture.calculate_complexity()

        return architecture

    def _calculate_reward(self, old_score: float, new_score: float, architecture: Any) -> float:
        """Calculate reward for RL agent."""
        # Performance improvement
        performance_improvement = new_score - old_score

        # Complexity penalty
        complexity_penalty = -architecture.estimated_complexity * 0.1

        # Resource efficiency bonus
        memory_efficiency = 1.0 / (1.0 + architecture.estimated_memory_usage / 1000)
        time_efficiency = 1.0 / (1.0 + architecture.estimated_training_time / 3600)

        # Exploration bonus
        exploration_bonus = self.exploration_rate * 0.1

        # Constraint satisfaction bonus
        constraint_bonus = 0.1  # Assume architecture is valid

        total_reward = (performance_improvement +
                       complexity_penalty +
                       memory_efficiency * 0.1 +
                       time_efficiency * 0.1 +
                       exploration_bonus +
                       constraint_bonus)

        return total_reward

    def _update_rl_components(self, old_state: np.ndarray, action: RLAction, reward: float, new_state: np.ndarray):
        """Update RL components with new experience."""
        if self.agent_type == RLAgentType.Q_LEARNING:
            # Q-learning update
            old_state_key = tuple(old_state)
            new_state_key = tuple(new_state)

            action_idx = self._action_to_idx(action)
            old_q_value = self.q_table[old_state_key][action_idx]

            # Q-value update
            max_new_q = np.max(self.q_table[new_state_key])
            new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_new_q - old_q_value)

            self.q_table[old_state_key][action_idx] = new_q_value

        elif self.agent_type == RLAgentType.DEEP_Q_NETWORK:
            # Deep Q-Network update
            self._update_dqn(old_state, action, reward, new_state)

    def _action_to_idx(self, action: RLAction) -> int:
        """Convert RLAction to action index."""
        action_mapping = {
            'expand_layer': 0,
            'modify_activation': 1,
            'add_connection': 2,
            'remove_layer': 3,
            'change_layer_size': 4,
            'modify_dropout': 5,
            'add_residual': 6,
            'remove_connection': 7
        }
        return action_mapping.get(action.action_type, 0)

    def _update_dqn(self, old_state: np.ndarray, action: RLAction, reward: float, new_state: np.ndarray):
        """Update Deep Q-Network."""
        # Simplified DQN update - in practice, you'd use experience replay
        if self.policy_network is None:
            return

        # Convert to tensors
        old_state_tensor = torch.FloatTensor(old_state).unsqueeze(0)
        new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)

        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.policy_network(new_state_tensor)
            max_next_q = torch.max(next_q_values).item()
            target_q = reward + self.discount_factor * max_next_q

        # Compute current Q-value
        current_q_values = self.policy_network(old_state_tensor)
        action_idx = self._action_to_idx(action)
        current_q = current_q_values[0, action_idx].item()

        # Compute loss
        loss = (current_q - target_q) ** 2

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class EnhancedBayesianOptimization(BaseAdvancedSearchStrategy):
    """Enhanced Bayesian optimization with advanced kernels and acquisition functions."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced Bayesian optimization."""
        super().__init__(config)
        self.n_initial_points = config.get('n_initial_points', 20)
        self.acquisition_function = config.get('acquisition_function', 'expected_improvement')
        self.kernel_type = config.get('kernel_type', 'matern')
        self.noise_level = config.get('noise_level', 0.1)
        self.acquisition_weight = config.get('acquisition_weight', 1.0)

        # Multi-objective support
        self.enable_multi_objective = config.get('enable_multi_objective', False)
        self.objective_weights = config.get('objective_weights', [1.0, 0.3, 0.2])  # performance, complexity, efficiency

        # Advanced features
        self.use_trust_region = config.get('use_trust_region', True)
        self.trust_region_size = config.get('trust_region_size', 0.1)
        self.enable_local_search = config.get('enable_local_search', True)

        self.gp_model = None
        self.scaler = StandardScaler()
        self.best_architecture = None
        self.best_score = -np.inf
        self.search_history = []

        self.logger.info("✅ Enhanced Bayesian Optimization initialized")

    def search(self,
               architecture_generator: Callable,
               performance_evaluator: Callable,
               constraint_validator: Callable,
               n_iterations: int = 100) -> SearchStrategyResult:
        """Perform enhanced Bayesian optimization search."""
        start_time = time.time()

        # Generate initial points
        initial_architectures = []
        initial_scores = []

        for _ in range(self.n_initial_points):
            arch = architecture_generator()
            if constraint_validator(arch).is_valid:
                score = performance_evaluator(arch)
                initial_architectures.append(arch)
                initial_scores.append(score)

                if score > self.best_score:
                    self.best_score = score
                    self.best_architecture = arch

                self.search_history.append({
                    'iteration': len(self.search_history),
                    'architecture': arch,
                    'score': score,
                    'type': 'initial'
                })

        # Fit initial Gaussian Process
        if len(initial_architectures) >= 2:
            self._fit_gaussian_process(initial_architectures, initial_scores)

        # Main optimization loop
        for iteration in range(n_iterations):
            # Propose next architecture
            if self.gp_model is not None:
                next_arch = self._propose_next_architecture(architecture_generator)
            else:
                next_arch = architecture_generator()

            # Validate constraints
            if not constraint_validator(next_arch).is_valid:
                self.search_history.append({
                    'iteration': len(self.search_history),
                    'architecture': next_arch,
                    'score': -np.inf,
                    'type': 'invalid'
                })
                continue

            # Evaluate architecture
            score = performance_evaluator(next_arch)

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = next_arch

            # Store in history
            self.search_history.append({
                'iteration': len(self.search_history),
                'architecture': next_arch,
                'score': score,
                'type': 'bayesian_optimization'
            })

            # Update Gaussian Process
            all_architectures = initial_architectures + [next_arch]
            all_scores = initial_scores + [score]
            self._fit_gaussian_process(all_architectures, all_scores)

            # Log progress
            if iteration % 10 == 0:
                self.logger.info(f"Bayesian Optimization - Iteration {iteration}: Best Score = {self.best_score:.4f}")

        execution_time = time.time() - start_time

        return SearchStrategyResult(
            best_architecture=self.best_architecture,
            best_score=self.best_score,
            search_history=self.search_history,
            strategy_used="enhanced_bayesian_optimization",
            convergence_info={
                'n_evaluations': len(self.search_history),
                'initial_points': self.n_initial_points,
                'acquisition_function': self.acquisition_function
            },
            execution_time=execution_time,
            metadata={'kernel_type': self.kernel_type}
        )

    def _fit_gaussian_process(self, architectures: List[Any], scores: List[float]):
        """Fit Gaussian Process model to data."""
        try:
            # Convert architectures to feature vectors
            X = np.array([self._architecture_to_features(arch) for arch in architectures])
            y = np.array(scores)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Create kernel
            if self.kernel_type == 'matern':
                kernel = Matern(length_scale=1.0, nu=2.5)
            elif self.kernel_type == 'rbf':
                kernel = RBF(length_scale=1.0)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)

            # Fit GP
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level,
                random_state=42
            )
            self.gp_model.fit(X_scaled, y)

        except Exception as e:
            self.logger.warning(f"Failed to fit Gaussian Process: {e}")
            self.gp_model = None

    def _architecture_to_features(self, architecture: Any) -> np.ndarray:
        """Convert architecture to feature vector for GP."""
        if hasattr(architecture, 'layers'):
            # Neural architecture
            n_layers = len(architecture.layers)
            total_params = sum(layer.hidden_size * layer.hidden_size for layer in architecture.layers)
            n_connections = len(architecture.connections)
        else:
            # Tree architecture
            n_layers = len(architecture.trees)
            total_params = sum(tree.n_estimators for tree in architecture.trees)
            n_connections = 0

        return np.array([
            n_layers / 20.0,
            total_params / 1000000.0,
            n_connections / 50.0,
            architecture.estimated_complexity / 5.0,
            architecture.estimated_memory_usage / 4096.0,
            architecture.estimated_training_time / 3600.0
        ])

    def _propose_next_architecture(self, architecture_generator: Callable) -> Any:
        """Propose next architecture using Bayesian optimization."""
        if self.gp_model is None:
            return architecture_generator()

        try:
            # Generate candidate architectures
            candidates = []
            for _ in range(50):  # Generate multiple candidates
                candidate = architecture_generator()
                candidates.append(candidate)

            # Convert to feature matrix
            X_candidates = np.array([self._architecture_to_features(arch) for arch in candidates])
            X_candidates_scaled = self.scaler.transform(X_candidates)

            # Get GP predictions
            with torch.no_grad() if hasattr(torch, 'no_grad') else self._dummy_context():
                mean, std = self.gp_model.predict(X_candidates_scaled, return_std=True)

                # Calculate acquisition function
                if self.acquisition_function == 'expected_improvement':
                    acquisition = self._expected_improvement(mean, std)
                elif self.acquisition_function == 'upper_confidence_bound':
                    acquisition = self._upper_confidence_bound(mean, std)
                elif self.acquisition_function == 'probability_of_improvement':
                    acquisition = self._probability_of_improvement(mean, std)
                else:
                    acquisition = mean

                # Select best candidate
                best_idx = np.argmax(acquisition)
                return candidates[best_idx]

        except Exception as e:
            self.logger.warning(f"Failed to propose next architecture: {e}")
            return architecture_generator()

    def _dummy_context(self):
        """Dummy context manager for compatibility."""
        return self

    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate expected improvement acquisition function."""
        if len(self.search_history) == 0:
            return mean

        best_score = max(record['score'] for record in self.search_history)

        with np.errstate(divide='ignore', invalid='ignore'):
            improvement = mean - best_score
            z = improvement / (std + 1e-9)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei = np.where(std < 1e-9, 0, ei)

        return ei

    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate upper confidence bound acquisition function."""
        return mean + self.acquisition_weight * std

    def _probability_of_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate probability of improvement acquisition function."""
        if len(self.search_history) == 0:
            return np.ones_like(mean)

        best_score = max(record['score'] for record in self.search_history)

        with np.errstate(divide='ignore', invalid='ignore'):
            z = (mean - best_score) / (std + 1e-9)
            poi = norm.cdf(z)

        return poi


class AdaptiveEvolutionarySearch(BaseAdvancedSearchStrategy):
    """Adaptive evolutionary search with dynamic parameter adjustment."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive evolutionary search."""
        super().__init__(config)
        self.population_size = config.get('population_size', 50)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.tournament_size = config.get('tournament_size', 5)

        # Adaptive parameters
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.stagnation_threshold = config.get('stagnation_threshold', 10)
        self.diversity_threshold = config.get('diversity_threshold', 0.1)

        # Advanced features
        self.use_island_model = config.get('use_island_model', False)
        self.n_islands = config.get('n_islands', 5)
        self.migration_interval = config.get('migration_interval', 10)

        self.population = []
        self.generation = 0
        self.stagnation_count = 0
        self.best_score_history = []

        self.logger.info("✅ Adaptive Evolutionary Search initialized")

    def search(self,
               architecture_generator: Callable,
               performance_evaluator: Callable,
               constraint_validator: Callable,
               n_iterations: int = 100) -> SearchStrategyResult:
        """Perform adaptive evolutionary search."""
        start_time = time.time()
        search_history = []

        # Initialize population
        self.population = self._initialize_population(architecture_generator, constraint_validator)

        for generation in range(n_iterations):
            # Evaluate current population
            population_scores = []
            for arch in self.population:
                score = performance_evaluator(arch)
                population_scores.append(score)

                # Track best architecture
                if score > (self.best_score_history[-1] if self.best_score_history else -np.inf):
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1

            # Update best score history
            current_best = max(population_scores)
            self.best_score_history.append(current_best)

            # Store in history
            for i, (arch, score) in enumerate(zip(self.population, population_scores)):
                search_history.append({
                    'iteration': generation,
                    'architecture': arch,
                    'score': score,
                    'type': 'evolutionary',
                    'generation': generation,
                    'individual': i
                })

            # Check for stagnation
            if self.stagnation_count >= self.stagnation_threshold:
                self._adapt_parameters()
                self.stagnation_count = 0

            # Create next generation
            self.population = self._create_next_generation(population_scores, constraint_validator)

            # Island model migration
            if self.use_island_model and generation % self.migration_interval == 0:
                self._perform_migration()

            # Log progress
            if generation % 10 == 0:
                self.logger.info(f"Evolutionary Search - Generation {generation}: Best Score = {current_best:.4f}, "
                               f"Mutation Rate = {self.mutation_rate:.3f}")

        execution_time = time.time() - start_time

        # Get final best architecture
        final_scores = [performance_evaluator(arch) for arch in self.population]
        best_idx = np.argmax(final_scores)
        best_architecture = self.population[best_idx]
        best_score = final_scores[best_idx]

        return SearchStrategyResult(
            best_architecture=best_architecture,
            best_score=best_score,
            search_history=search_history,
            strategy_used="adaptive_evolutionary_search",
            convergence_info={
                'final_generation': generation,
                'total_evaluations': len(search_history),
                'stagnation_events': self.stagnation_count // self.stagnation_threshold
            },
            execution_time=execution_time,
            metadata={'population_size': self.population_size}
        )

    def _initialize_population(self, architecture_generator: Callable, constraint_validator: Callable) -> List[Any]:
        """Initialize population with valid architectures."""
        population = []

        while len(population) < self.population_size:
            arch = architecture_generator()
            if constraint_validator(arch).is_valid:
                population.append(arch)

        return population

    def _create_next_generation(self, scores: List[float], constraint_validator: Callable) -> List[Any]:
        """Create next generation using evolutionary operators."""
        next_generation = []

        # Elitism: keep best individuals
        elite_size = max(1, self.population_size // 10)
        elite_indices = np.argsort(scores)[-elite_size:]
        for idx in elite_indices:
            next_generation.append(self.population[idx])

        # Fill remaining population
        while len(next_generation) < self.population_size:
            if np.random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection(scores)
                parent2 = self._tournament_selection(scores)
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._tournament_selection(scores)
                child = self._mutate(parent)

            # Validate constraints
            if constraint_validator(child).is_valid:
                next_generation.append(child)

        return next_generation[:self.population_size]

    def _tournament_selection(self, scores: List[float]) -> Any:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(len(self.population), self.tournament_size, replace=False)
        best_idx = tournament_indices[np.argmax([scores[i] for i in tournament_indices])]
        return self.population[best_idx]

    def _crossover(self, parent1: Any, parent2: Any) -> Any:
        """Crossover two architectures."""
        # Simplified crossover - in practice, architecture-specific crossover needed
        if np.random.random() < 0.5:
            return parent1
        else:
            return parent2

    def _mutate(self, architecture: Any) -> Any:
        """Mutate an architecture."""
        # Simplified mutation - in practice, architecture-specific mutation needed
        return architecture

    def _adapt_parameters(self):
        """Adapt evolutionary parameters based on search progress."""
        # Increase mutation rate if stagnating
        if self.stagnation_count > 0:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.2)

        # Decrease crossover rate if population is too similar
        if len(set(str(arch) for arch in self.population)) < self.population_size * 0.5:
            self.crossover_rate = max(0.3, self.crossover_rate * 0.9)

    def _perform_migration(self):
        """Perform migration in island model."""
        # Simplified migration - in practice, more sophisticated migration strategies
        if self.use_island_model and len(self.population) >= self.n_islands:
            # Split population into islands
            island_size = len(self.population) // self.n_islands
            islands = []

            for i in range(self.n_islands):
                start_idx = i * island_size
                end_idx = start_idx + island_size if i < self.n_islands - 1 else len(self.population)
                islands.append(self.population[start_idx:end_idx])

            # Perform migration (exchange best individuals)
            best_individuals = [max(island, key=lambda x: x.estimated_complexity) for island in islands]

            # Replace worst individuals in each island
            for i, island in enumerate(islands):
                if len(island) > 1:
                    worst_idx = np.argmin([ind.estimated_complexity for ind in island])
                    island[worst_idx] = best_individuals[(i + 1) % self.n_islands]


def create_rl_search_strategy(config: Dict[str, Any]) -> ReinforcementLearningSearch:
    """Create a reinforcement learning search strategy."""
    return ReinforcementLearningSearch(config)


def create_enhanced_bayesian_search(config: Dict[str, Any]) -> EnhancedBayesianOptimization:
    """Create an enhanced Bayesian optimization search strategy."""
    return EnhancedBayesianOptimization(config)


def create_adaptive_evolutionary_search(config: Dict[str, Any]) -> AdaptiveEvolutionarySearch:
    """Create an adaptive evolutionary search strategy."""
    return AdaptiveEvolutionarySearch(config)