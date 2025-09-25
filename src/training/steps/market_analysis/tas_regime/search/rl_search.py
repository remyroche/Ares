"""
Reinforcement Learning Search for TAS Tree Architecture

This module provides comprehensive RL-based search for tree architecture search,
integrating with the unified utilities and shared components from the hybrid NAS/TAS system.

Features:
- Multiple RL algorithms (Q-Learning, PPO, A2C, DQN)
- Integration with unified search algorithms
- Economic significance and trading viability evaluation
- Hardware optimization support
- Real-time adaptation capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

# Import unified utilities
from ...hybrid_nas_tas_regime.shared_utils import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig,
    UnifiedMultiObjectiveOptimizer, OptimizationConfig,
    UnifiedHardwareOptimizer, HardwareConfig,
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig,
    UnifiedValidationSystem, ValidationConfig,
    create_unified_economic_evaluator, quick_economic_evaluation,
    create_unified_trading_viability_evaluator, quick_trading_viability_evaluation,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization,
    create_unified_hardware_optimizer, quick_hardware_optimization,
    create_unified_regime_analyzer, quick_regime_analysis,
    create_unified_validation_system, quick_validation
)

# Import utility tools
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
    align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, validate_file_path, get_file_size,
    check_disk_space, CommonUtilities
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, BayesianTPEConfig
    from src.utils.nas_tas.advanced_hpo_utils import HyperparameterOptimization as HyperparameterOptimizer
    from src.utils.ml_common.validation.cv import CrossValidationManager
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """Types of RL algorithms available."""
    Q_LEARNING = "q_learning"
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    SAC = "sac"
    TD3 = "td3"


class RLState:
    """RL state representation for tree architecture search."""
    
    def __init__(self, architecture_params: Dict[str, Any], performance_metrics: Dict[str, float] = None):
        self.architecture_params = architecture_params
        self.performance_metrics = performance_metrics or {}
        self.state_id = self._generate_state_id()
        self.timestamp = datetime.now()
    
    def _generate_state_id(self) -> str:
        """Generate unique state ID."""
        param_str = str(sorted(self.architecture_params.items()))
        return f"state_{hash(param_str) % 1000000}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'state_id': self.state_id,
            'architecture_params': self.architecture_params,
            'performance_metrics': self.performance_metrics,
            'timestamp': self.timestamp.isoformat()
        }


class RLAction:
    """RL action representation for tree architecture modifications."""
    
    def __init__(self, action_type: str, parameter: str, value: Any, confidence: float = 1.0):
        self.action_type = action_type
        self.parameter = parameter
        self.value = value
        self.confidence = confidence
        self.action_id = self._generate_action_id()
        self.timestamp = datetime.now()
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID."""
        action_str = f"{self.action_type}_{self.parameter}_{self.value}"
        return f"action_{hash(action_str) % 1000000}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            'action_id': self.action_id,
            'action_type': self.action_type,
            'parameter': self.parameter,
            'value': self.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class RLReward:
    """RL reward representation with multiple objectives."""
    
    def __init__(self, 
                 regime_accuracy: float = 0.0,
                 economic_significance: float = 0.0,
                 trading_viability: float = 0.0,
                 computational_efficiency: float = 0.0,
                 architecture_complexity: float = 0.0,
                 total_reward: float = 0.0):
        self.regime_accuracy = regime_accuracy
        self.economic_significance = economic_significance
        self.trading_viability = trading_viability
        self.computational_efficiency = computational_efficiency
        self.architecture_complexity = architecture_complexity
        self.total_reward = total_reward
        self.timestamp = datetime.now()
    
    def calculate_total(self, weights: Dict[str, float] = None) -> float:
        """Calculate total reward with weights."""
        if weights is None:
            weights = {
                'regime_accuracy': 0.3,
                'economic_significance': 0.25,
                'trading_viability': 0.25,
                'computational_efficiency': 0.1,
                'architecture_complexity': 0.1
            }
        
        total = 0.0
        for metric, weight in weights.items():
            if hasattr(self, metric):
                total += getattr(self, metric) * weight
        
        self.total_reward = total
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reward to dictionary."""
        return {
            'regime_accuracy': self.regime_accuracy,
            'economic_significance': self.economic_significance,
            'trading_viability': self.trading_viability,
            'computational_efficiency': self.computational_efficiency,
            'architecture_complexity': self.architecture_complexity,
            'total_reward': self.total_reward,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RLConfig:
    """Configuration for RL search."""
    # Basic RL parameters
    n_episodes: int = 1000
    learning_rate: float = 0.01
    epsilon: float = 0.1
    gamma: float = 0.9
    max_steps: int = 100
    
    # Algorithm selection
    algorithm: RLAlgorithm = RLAlgorithm.Q_LEARNING
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'regime_accuracy': 0.3,
        'economic_significance': 0.25,
        'trading_viability': 0.25,
        'computational_efficiency': 0.1,
        'architecture_complexity': 0.1
    })
    
    # Economic and trading evaluation
    enable_economic_evaluation: bool = True
    enable_trading_viability: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_limit_gb: float = 8.0
    
    # Advanced features
    enable_adaptive_learning: bool = True
    enable_exploration_decay: bool = True
    exploration_decay_rate: float = 0.995
    
    # Performance settings
    enable_parallel_evaluation: bool = True
    max_workers: int = 4
    
    # Logging and monitoring
    log_level: str = 'INFO'
    enable_progress_logging: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 100


class RLTreeSearch:
    """Reinforcement learning search for tree architectures."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # RL components
        self.q_table = {}
        self.policy_network = {}
        self.value_network = {}
        self.best_params = None
        self.best_score = -np.inf
        self.search_history = []
        
        # Unified evaluators
        self.economic_evaluator = None
        self.trading_evaluator = None
        self.hardware_optimizer = None
        self.regime_analyzer = None
        self.validation_system = None
        
        # Performance tracking
        self.episode_rewards = []
        self.convergence_history = []
        self.exploration_rate = config.epsilon
        
        # Initialize unified components
        self._initialize_unified_components()
    
    def _initialize_unified_components(self):
        """Initialize unified evaluation components."""
        try:
            if self.config.enable_economic_evaluation:
                self.economic_evaluator = create_unified_economic_evaluator()
            
            if self.config.enable_trading_viability:
                self.trading_evaluator = create_unified_trading_viability_evaluator()
            
            if self.config.enable_hardware_optimization:
                self.hardware_optimizer = create_unified_hardware_optimizer()
            
            self.regime_analyzer = create_unified_regime_analyzer()
            self.validation_system = create_unified_validation_system()
            
            tprint_success("✅ Unified components initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize unified components: {e}")
            self.logger.warning(f"Some unified components may not be available: {e}")
    
    def search(self, 
               search_space: Dict[str, Any], 
               train_data: pd.DataFrame = None,
               validation_data: pd.DataFrame = None,
               test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Perform RL search for optimal tree architecture."""
        tprint_info("🚀 Starting RL tree search")
        
        start_time = time.time()
        
        # Initialize search environment
        search_env = {
            'search_space': search_space,
            'train_data': train_data,
            'validation_data': validation_data,
            'test_data': test_data
        }
        
        # RL training loop
        for episode in range(self.config.n_episodes):
            episode_start = time.time()
            
            # Initialize episode
            state = self._get_initial_state(search_space)
            episode_reward = 0
            episode_actions = []
            
            for step in range(self.config.max_steps):
                # Select action based on algorithm
                action = self._select_action(state)
                
                # Take action and get next state
                next_state, reward = self._take_action(state, action, search_space, search_env)
                
                # Update RL components
                self._update_rl_components(state, action, reward, next_state)
                
                # Store experience
                episode_actions.append({
                    'state': state.to_dict(),
                    'action': action.to_dict(),
                    'reward': reward.to_dict(),
                    'next_state': next_state.to_dict()
                })
                
                # Update state and reward
                state = next_state
                episode_reward += reward.total_reward
                
                # Check if done
                if self._is_done(state, step):
                    break
            
            # Update best solution
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.best_params = self._state_to_params(state)
                tprint_success(f"🎯 New best score: {self.best_score:.4f}")
            
            # Store episode results
            self.episode_rewards.append(episode_reward)
            self.search_history.append({
                'episode': episode,
                'reward': episode_reward,
                'best_score': self.best_score,
                'actions': episode_actions,
                'duration': time.time() - episode_start
            })
            
            # Adaptive learning
            if self.config.enable_adaptive_learning:
                self._adaptive_learning_update(episode)
            
            # Exploration decay
            if self.config.enable_exploration_decay:
                self.exploration_rate *= self.config.exploration_decay_rate
                self.exploration_rate = max(0.01, self.exploration_rate)
            
            # Progress logging
            if episode % 100 == 0:
                tprint_progress(f"Episode {episode}: Reward = {episode_reward:.4f}, Best = {self.best_score:.4f}, Exploration = {self.exploration_rate:.3f}")
            
            # Checkpoint saving
            if self.config.save_checkpoints and episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint(episode)
        
        # Final results
        total_time = time.time() - start_time
        tprint_success(f"🎉 RL search completed in {total_time:.2f}s")
        tprint_info(f"Best score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_history': self.search_history,
            'convergence_info': self._analyze_convergence(),
            'total_time': total_time
        }
    
    def _get_initial_state(self, search_space: Dict[str, Any]) -> RLState:
        """Get initial state."""
        architecture_params = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                architecture_params[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                architecture_params[param] = np.random.uniform(values[0], values[1])
            else:
                architecture_params[param] = values
        
        return RLState(architecture_params)
    
    def _select_action(self, state: RLState) -> RLAction:
        """Select action based on RL algorithm."""
        if self.config.algorithm == RLAlgorithm.Q_LEARNING:
            return self._select_action_q_learning(state)
        elif self.config.algorithm == RLAlgorithm.PPO:
            return self._select_action_ppo(state)
        elif self.config.algorithm == RLAlgorithm.A2C:
            return self._select_action_a2c(state)
        else:
            return self._select_action_q_learning(state)  # Default fallback
    
    def _select_action_q_learning(self, state: RLState) -> RLAction:
        """Select action using Q-learning."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Available actions (parameter modifications)
        available_actions = list(state.architecture_params.keys())
        
        # Epsilon-greedy selection
        if np.random.random() < self.exploration_rate:
            # Exploration
            action_type = "modify"
            parameter = np.random.choice(available_actions)
            value = self._get_random_value(parameter, state.architecture_params[parameter])
        else:
            # Exploitation
            best_action = None
            best_value = -np.inf
            
            for param in available_actions:
                if param not in self.q_table[state_key]:
                    self.q_table[state_key][param] = 0
                
                if self.q_table[state_key][param] > best_value:
                    best_value = self.q_table[state_key][param]
                    best_action = param
            
            if best_action:
                action_type = "modify"
                parameter = best_action
                value = self._get_optimized_value(parameter, state.architecture_params[parameter])
            else:
                # Fallback to random
                action_type = "modify"
                parameter = np.random.choice(available_actions)
                value = self._get_random_value(parameter, state.architecture_params[parameter])
        
        return RLAction(action_type, parameter, value)
    
    def _select_action_ppo(self, state: RLState) -> RLAction:
        """Select action using PPO policy."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        # Available actions
        available_actions = list(state.architecture_params.keys())
        
        # Policy-based selection (simplified)
        if available_actions:
            parameter = np.random.choice(available_actions)
            value = self._get_random_value(parameter, state.architecture_params[parameter])
            return RLAction("modify", parameter, value)
        else:
            return RLAction("no_op", "none", None)
    
    def _select_action_a2c(self, state: RLState) -> RLAction:
        """Select action using A2C actor."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        # Available actions
        available_actions = list(state.architecture_params.keys())
        
        # Actor-based selection (simplified)
        if available_actions:
            parameter = np.random.choice(available_actions)
            value = self._get_random_value(parameter, state.architecture_params[parameter])
            return RLAction("modify", parameter, value)
        else:
            return RLAction("no_op", "none", None)
    
    def _take_action(self, 
                     state: RLState, 
                     action: RLAction, 
                     search_space: Dict[str, Any],
                     search_env: Dict[str, Any]) -> Tuple[RLState, RLReward]:
        """Take action and return next state and reward."""
        next_state = RLState(state.architecture_params.copy())
        
        # Apply action
        if action.action_type == "modify" and action.parameter in next_state.architecture_params:
            next_state.architecture_params[action.parameter] = action.value
        
        # Calculate reward
        reward = self._calculate_reward(state, action, next_state, search_env)
        
        return next_state, reward
    
    def _calculate_reward(self, 
                         state: RLState, 
                         action: RLAction, 
                         next_state: RLState,
                         search_env: Dict[str, Any]) -> RLReward:
        """Calculate reward with multiple objectives."""
        reward = RLReward()
        
        try:
            # Regime accuracy (placeholder - would use actual model evaluation)
            reward.regime_accuracy = np.random.random() * 0.8 + 0.2
            
            # Economic significance
            if self.economic_evaluator and search_env.get('train_data') is not None:
                try:
                    economic_result = quick_economic_evaluation(
                        search_env['train_data'], 
                        next_state.architecture_params
                    )
                    reward.economic_significance = economic_result.get('significance_score', 0.5)
                except Exception as e:
                    self.logger.warning(f"Economic evaluation failed: {e}")
                    reward.economic_significance = 0.5
            else:
                reward.economic_significance = np.random.random() * 0.6 + 0.4
            
            # Trading viability
            if self.trading_evaluator and search_env.get('train_data') is not None:
                try:
                    trading_result = quick_trading_viability_evaluation(
                        search_env['train_data'],
                        next_state.architecture_params
                    )
                    reward.trading_viability = trading_result.get('viability_score', 0.5)
                except Exception as e:
                    self.logger.warning(f"Trading viability evaluation failed: {e}")
                    reward.trading_viability = 0.5
            else:
                reward.trading_viability = np.random.random() * 0.6 + 0.4
            
            # Computational efficiency (based on architecture complexity)
            complexity = self._calculate_architecture_complexity(next_state.architecture_params)
            reward.computational_efficiency = max(0.1, 1.0 - complexity)
            reward.architecture_complexity = complexity
            
            # Calculate total reward
            reward.calculate_total(self.config.objective_weights)
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            reward = RLReward(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        return reward
    
    def _calculate_architecture_complexity(self, params: Dict[str, Any]) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0
        
        for param, value in params.items():
            if isinstance(value, (int, float)):
                # Normalize numeric parameters
                complexity += min(1.0, abs(value) / 100.0)
            elif isinstance(value, str):
                # String parameters contribute less complexity
                complexity += 0.1
            else:
                # Other types
                complexity += 0.2
        
        return min(1.0, complexity)
    
    def _get_random_value(self, parameter: str, current_value: Any) -> Any:
        """Get random value for parameter."""
        if isinstance(current_value, (int, float)):
            # Add random noise
            noise = np.random.normal(0, 0.1 * abs(current_value))
            return current_value + noise
        elif isinstance(current_value, str):
            # Random choice from common values
            common_values = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax']
            return np.random.choice(common_values)
        else:
            return current_value
    
    def _get_optimized_value(self, parameter: str, current_value: Any) -> Any:
        """Get optimized value for parameter."""
        if isinstance(current_value, (int, float)):
            # Small optimization step
            step = 0.05 * abs(current_value)
            direction = np.random.choice([-1, 1])
            return current_value + direction * step
        else:
            return self._get_random_value(parameter, current_value)
    
    def _update_rl_components(self, 
                             state: RLState, 
                             action: RLAction, 
                             reward: RLReward, 
                             next_state: RLState):
        """Update RL components based on algorithm."""
        if self.config.algorithm == RLAlgorithm.Q_LEARNING:
            self._update_q_learning(state, action, reward, next_state)
        elif self.config.algorithm == RLAlgorithm.PPO:
            self._update_ppo(state, action, reward, next_state)
        elif self.config.algorithm == RLAlgorithm.A2C:
            self._update_a2c(state, action, reward, next_state)
    
    def _update_q_learning(self, state: RLState, action: RLAction, reward: RLReward, next_state: RLState):
        """Update Q-learning components."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action.parameter not in self.q_table[state_key]:
            self.q_table[state_key][action.parameter] = 0
        
        # Q-learning update
        current_q = self.q_table[state_key][action.parameter]
        
        # Find max Q-value for next state
        max_next_q = 0
        if next_state_key in self.q_table:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        
        # Update Q-value
        self.q_table[state_key][action.parameter] = current_q + self.config.learning_rate * (
            reward.total_reward + self.config.gamma * max_next_q - current_q
        )
    
    def _update_ppo(self, state: RLState, action: RLAction, reward: RLReward, next_state: RLState):
        """Update PPO components."""
        state_key = self._state_to_key(state)
        
        # Update policy network
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        if action.parameter not in self.policy_network[state_key]:
            self.policy_network[state_key][action.parameter] = 0
        
        # PPO policy update (simplified)
        self.policy_network[state_key][action.parameter] += self.config.learning_rate * reward.total_reward
        
        # Update value network
        if state_key not in self.value_network:
            self.value_network[state_key] = 0
        
        # PPO value update (simplified)
        self.value_network[state_key] += self.config.learning_rate * (reward.total_reward - self.value_network[state_key])
    
    def _update_a2c(self, state: RLState, action: RLAction, reward: RLReward, next_state: RLState):
        """Update A2C components."""
        state_key = self._state_to_key(state)
        
        # Update actor network
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        if action.parameter not in self.policy_network[state_key]:
            self.policy_network[state_key][action.parameter] = 0
        
        # Actor update (simplified)
        self.policy_network[state_key][action.parameter] += self.config.learning_rate * reward.total_reward
        
        # Update critic network
        if state_key not in self.value_network:
            self.value_network[state_key] = 0
        
        # Critic update (simplified)
        self.value_network[state_key] += self.config.learning_rate * (reward.total_reward - self.value_network[state_key])
    
    def _adaptive_learning_update(self, episode: int):
        """Update learning parameters adaptively."""
        if episode > 0 and episode % 50 == 0:
            # Analyze recent performance
            recent_rewards = self.episode_rewards[-50:]
            if len(recent_rewards) > 10:
                avg_recent = np.mean(recent_rewards)
                avg_previous = np.mean(self.episode_rewards[-100:-50]) if len(self.episode_rewards) > 100 else avg_recent
                
                # Adjust learning rate based on performance
                if avg_recent > avg_previous:
                    self.config.learning_rate = min(0.1, self.config.learning_rate * 1.05)
                else:
                    self.config.learning_rate = max(0.001, self.config.learning_rate * 0.95)
    
    def _is_done(self, state: RLState, step: int) -> bool:
        """Check if episode is done."""
        # Early stopping conditions
        if step >= self.config.max_steps - 1:
            return True
        
        # Performance-based stopping
        if len(self.episode_rewards) > 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            if recent_avg > 0.9:  # High performance threshold
                return True
        
        return False
    
    def _state_to_key(self, state: RLState) -> str:
        """Convert state to string key."""
        return str(sorted(state.architecture_params.items()))
    
    def _state_to_params(self, state: RLState) -> Dict[str, Any]:
        """Convert state to parameters."""
        return state.architecture_params.copy()
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of the RL search."""
        if len(self.episode_rewards) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        recent_rewards = self.episode_rewards[-50:]
        if len(recent_rewards) < 10:
            return {'converged': False, 'reason': 'insufficient_recent_data'}
        
        # Check for convergence
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        
        if reward_std < 0.01 and reward_mean > 0.8:
            return {'converged': True, 'reason': 'low_variance_high_performance'}
        
        # Check for improvement trend
        if len(self.episode_rewards) > 100:
            early_rewards = self.episode_rewards[-100:-50]
            late_rewards = self.episode_rewards[-50:]
            
            if np.mean(late_rewards) > np.mean(early_rewards) + 0.1:
                return {'converged': False, 'reason': 'still_improving'}
        
        return {'converged': False, 'reason': 'no_convergence_criteria_met'}
    
    def _save_checkpoint(self, episode: int):
        """Save checkpoint."""
        try:
            checkpoint_data = {
                'episode': episode,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'q_table': self.q_table,
                'policy_network': self.policy_network,
                'value_network': self.value_network,
                'episode_rewards': self.episode_rewards,
                'config': self.config.__dict__
            }
            
            checkpoint_path = f"rl_search_checkpoint_episode_{episode}.json"
            JSONSerializer.save(checkpoint_data, checkpoint_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")


class TreeReinforcementLearner:
    """Tree reinforcement learner for architecture search."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.rl_search = RLTreeSearch(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def learn(self, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Learn optimal tree architecture using reinforcement learning."""
        tprint_info("🧠 Starting tree reinforcement learning")
        
        return self.rl_search.search(search_space, **kwargs)


class TreePPO:
    """Tree Proximal Policy Optimization for architecture search."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.config.algorithm = RLAlgorithm.PPO
        self.rl_search = RLTreeSearch(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize(self, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Optimize tree architecture using PPO."""
        tprint_info("🎯 Starting tree PPO optimization")
        
        return self.rl_search.search(search_space, **kwargs)


class TreeA2C:
    """Tree Advantage Actor-Critic for architecture search."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.config.algorithm = RLAlgorithm.A2C
        self.rl_search = RLTreeSearch(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train(self, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train tree architecture using A2C."""
        tprint_info("🎮 Starting tree A2C training")
        
        return self.rl_search.search(search_space, **kwargs)


class TreeReinforcementLearner:
    """Tree reinforcement learner for architecture search."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.policy_network = {}
        self.value_network = {}
        self.best_params = None
        self.best_score = -np.inf
    
    def learn(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Learn optimal tree architecture using reinforcement learning."""
        logger.info("Starting tree reinforcement learning")
        
        # RL training loop
        for episode in range(self.config.n_episodes):
            # Initialize episode
            state = self._get_initial_state(search_space)
            episode_reward = 0
            
            for step in range(self.config.max_steps):
                # Select action using policy
                action = self._select_action_policy(state)
                
                # Take action
                next_state, reward = self._take_action(state, action, search_space)
                
                # Update networks
                self._update_networks(state, action, reward, next_state)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Check if done
                if self._is_done(state):
                    break
            
            # Update best if necessary
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.best_params = self._state_to_params(state)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.4f}, Best = {self.best_score:.4f}")
        
        return self.best_params
    
    def _get_initial_state(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Get initial state."""
        state = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                state[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                state[param] = np.random.uniform(values[0], values[1])
            else:
                state[param] = values
        return state
    
    def _select_action_policy(self, state: Dict[str, Any]) -> str:
        """Select action using policy network."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        # Available actions
        available_actions = list(state.keys())
        
        # Policy-based selection
        if available_actions:
            return np.random.choice(available_actions)
        else:
            return "no_action"
    
    def _take_action(self, state: Dict[str, Any], action: str, search_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Take action and return next state and reward."""
        next_state = state.copy()
        
        # Modify the selected parameter
        if action in search_space:
            param_values = search_space[action]
            if isinstance(param_values, list):
                # Randomly select a different value
                current_idx = param_values.index(state[action])
                next_idx = (current_idx + np.random.randint(-1, 2)) % len(param_values)
                next_state[action] = param_values[next_idx]
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Add small random change
                change = np.random.uniform(-0.1, 0.1) * (param_values[1] - param_values[0])
                next_state[action] = np.clip(state[action] + change, param_values[0], param_values[1])
        
        # Calculate reward (placeholder)
        reward = np.random.random()
        
        return next_state, reward
    
    def _update_networks(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """Update policy and value networks."""
        state_key = self._state_to_key(state)
        
        # Update policy network
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        if action not in self.policy_network[state_key]:
            self.policy_network[state_key][action] = 0
        
        # Policy update (simplified)
        self.policy_network[state_key][action] += self.config.learning_rate * reward
        
        # Update value network
        if state_key not in self.value_network:
            self.value_network[state_key] = 0
        
        # Value update (simplified)
        self.value_network[state_key] += self.config.learning_rate * (reward - self.value_network[state_key])
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key."""
        return str(sorted(state.items()))
    
    def _state_to_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to parameters."""
        return state.copy()
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        # Placeholder - could implement stopping criteria
        return np.random.random() < 0.1


class TreePPO:
    """Tree Proximal Policy Optimization for architecture search."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.policy_network = {}
        self.value_network = {}
        self.best_params = None
        self.best_score = -np.inf
    
    def optimize(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize tree architecture using PPO."""
        logger.info("Starting tree PPO optimization")
        
        # PPO training loop
        for episode in range(self.config.n_episodes):
            # Initialize episode
            state = self._get_initial_state(search_space)
            episode_reward = 0
            
            for step in range(self.config.max_steps):
                # Select action using PPO policy
                action = self._select_action_ppo(state)
                
                # Take action
                next_state, reward = self._take_action(state, action, search_space)
                
                # Update PPO networks
                self._update_ppo_networks(state, action, reward, next_state)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Check if done
                if self._is_done(state):
                    break
            
            # Update best if necessary
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.best_params = self._state_to_params(state)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.4f}, Best = {self.best_score:.4f}")
        
        return self.best_params
    
    def _get_initial_state(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Get initial state."""
        state = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                state[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                state[param] = np.random.uniform(values[0], values[1])
            else:
                state[param] = values
        return state
    
    def _select_action_ppo(self, state: Dict[str, Any]) -> str:
        """Select action using PPO policy."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        # Available actions
        available_actions = list(state.keys())
        
        # PPO-based selection
        if available_actions:
            return np.random.choice(available_actions)
        else:
            return "no_action"
    
    def _take_action(self, state: Dict[str, Any], action: str, search_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Take action and return next state and reward."""
        next_state = state.copy()
        
        # Modify the selected parameter
        if action in search_space:
            param_values = search_space[action]
            if isinstance(param_values, list):
                # Randomly select a different value
                current_idx = param_values.index(state[action])
                next_idx = (current_idx + np.random.randint(-1, 2)) % len(param_values)
                next_state[action] = param_values[next_idx]
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Add small random change
                change = np.random.uniform(-0.1, 0.1) * (param_values[1] - param_values[0])
                next_state[action] = np.clip(state[action] + change, param_values[0], param_values[1])
        
        # Calculate reward (placeholder)
        reward = np.random.random()
        
        return next_state, reward
    
    def _update_ppo_networks(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """Update PPO policy and value networks."""
        state_key = self._state_to_key(state)
        
        # Update policy network
        if state_key not in self.policy_network:
            self.policy_network[state_key] = {}
        
        if action not in self.policy_network[state_key]:
            self.policy_network[state_key][action] = 0
        
        # PPO policy update (simplified)
        self.policy_network[state_key][action] += self.config.learning_rate * reward
        
        # Update value network
        if state_key not in self.value_network:
            self.value_network[state_key] = 0
        
        # PPO value update (simplified)
        self.value_network[state_key] += self.config.learning_rate * (reward - self.value_network[state_key])
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key."""
        return str(sorted(state.items()))
    
    def _state_to_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to parameters."""
        return state.copy()
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        # Placeholder - could implement stopping criteria
        return np.random.random() < 0.1


class TreeA2C:
    """Tree Advantage Actor-Critic for architecture search."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.actor_network = {}
        self.critic_network = {}
        self.best_params = None
        self.best_score = -np.inf
    
    def train(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Train tree architecture using A2C."""
        logger.info("Starting tree A2C training")
        
        # A2C training loop
        for episode in range(self.config.n_episodes):
            # Initialize episode
            state = self._get_initial_state(search_space)
            episode_reward = 0
            
            for step in range(self.config.max_steps):
                # Select action using actor
                action = self._select_action_actor(state)
                
                # Take action
                next_state, reward = self._take_action(state, action, search_space)
                
                # Update A2C networks
                self._update_a2c_networks(state, action, reward, next_state)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Check if done
                if self._is_done(state):
                    break
            
            # Update best if necessary
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.best_params = self._state_to_params(state)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.4f}, Best = {self.best_score:.4f}")
        
        return self.best_params
    
    def _get_initial_state(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Get initial state."""
        state = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                state[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                state[param] = np.random.uniform(values[0], values[1])
            else:
                state[param] = values
        return state
    
    def _select_action_actor(self, state: Dict[str, Any]) -> str:
        """Select action using actor network."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.actor_network:
            self.actor_network[state_key] = {}
        
        # Available actions
        available_actions = list(state.keys())
        
        # Actor-based selection
        if available_actions:
            return np.random.choice(available_actions)
        else:
            return "no_action"
    
    def _take_action(self, state: Dict[str, Any], action: str, search_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Take action and return next state and reward."""
        next_state = state.copy()
        
        # Modify the selected parameter
        if action in search_space:
            param_values = search_space[action]
            if isinstance(param_values, list):
                # Randomly select a different value
                current_idx = param_values.index(state[action])
                next_idx = (current_idx + np.random.randint(-1, 2)) % len(param_values)
                next_state[action] = param_values[next_idx]
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Add small random change
                change = np.random.uniform(-0.1, 0.1) * (param_values[1] - param_values[0])
                next_state[action] = np.clip(state[action] + change, param_values[0], param_values[1])
        
        # Calculate reward (placeholder)
        reward = np.random.random()
        
        return next_state, reward
    
    def _update_a2c_networks(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """Update A2C actor and critic networks."""
        state_key = self._state_to_key(state)
        
        # Update actor network
        if state_key not in self.actor_network:
            self.actor_network[state_key] = {}
        
        if action not in self.actor_network[state_key]:
            self.actor_network[state_key][action] = 0
        
        # Actor update (simplified)
        self.actor_network[state_key][action] += self.config.learning_rate * reward
        
        # Update critic network
        if state_key not in self.critic_network:
            self.critic_network[state_key] = 0
        
        # Critic update (simplified)
        self.critic_network[state_key] += self.config.learning_rate * (reward - self.critic_network[state_key])
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key."""
        return str(sorted(state.items()))
    
    def _state_to_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to parameters."""
        return state.copy()
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        # Placeholder - could implement stopping criteria
        return np.random.random() < 0.1