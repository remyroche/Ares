"""
Enhanced TPE Early Stopping

This module provides enhanced early stopping capabilities for Bayesian TPE optimization,
including aggressive early stopping, adaptive patience, and confidence-based stopping.

Enhancement: TPE Early Stopping for All Current Usage
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import time
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEarlyStoppingConfig:
    """Configuration for enhanced early stopping."""
    
    # Basic early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Adaptive patience
    adaptive_patience: bool = True
    min_patience: int = 2
    max_patience: int = 20
    patience_adjustment_factor: float = 1.5
    
    # Confidence-based stopping
    confidence_based_stopping: bool = True
    confidence_level: float = 0.95
    min_history_for_confidence: int = 20
    
    # Multi-objective stopping
    multi_objective_stopping: bool = False
    pareto_improvement_threshold: float = 0.01
    
    # Learning rate schedules
    threshold_schedule_enabled: bool = True
    threshold_schedule_type: str = "exponential"  # "exponential", "linear", "step", "adaptive"
    initial_threshold: Optional[float] = None
    final_threshold: Optional[float] = None
    schedule_params: Dict[str, Any] = None
    
    # Performance tracking
    track_convergence: bool = True
    convergence_window: int = 10
    min_improvement_rate: float = 0.001


class EarlyStoppingStrategy(ABC):
    """Abstract base class for early stopping strategies."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig = None):
        """Initialize early stopping strategy."""
        self.config = config or EnhancedEarlyStoppingConfig()
        self.stopping_reason = None
        self.trials_without_improvement = 0
        self.best_value = float('-inf') if self.config.direction == 'maximize' else float('inf')
        self.best_trial = 0
    
    @abstractmethod
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Determine if optimization should stop early."""
        pass
    
    @abstractmethod
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        pass
    
    def reset(self):
        """Reset the early stopping strategy state."""
        self.stopping_reason = None
        self.trials_without_improvement = 0
        self.best_value = float('-inf') if self.config.direction == 'maximize' else float('inf')
        self.best_trial = 0
    
    def update_best_value(self, value: float, trial: int):
        """Update the best value and trial."""
        if self.config.direction == 'maximize':
            if value > self.best_value:
                self.best_value = value
                self.best_trial = trial
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1
        else:
            if value < self.best_value:
                self.best_value = value
                self.best_trial = trial
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1


class AdaptivePatienceStrategy(EarlyStoppingStrategy):
    """Adaptive patience early stopping strategy."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig):
        self.config = config
        self.current_patience = config.early_stopping_patience
        self.trials_without_improvement = 0
        self.best_value_history = []
        self.convergence_rate_history = []
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if adaptive patience criteria is met."""
        if len(history) < self.config.min_patience + 1:
            return False
        
        current_best = max(history) if self.config.direction == 'maximize' else min(history)
        self.best_value_history.append(current_best)
        
        # Calculate convergence rate
        if len(self.best_value_history) >= 10:
            convergence_rate = self._calculate_convergence_rate()
            self.convergence_rate_history.append(convergence_rate)
            
            # Adjust patience based on convergence rate
            if self.config.adaptive_patience:
                self.current_patience = self._calculate_adaptive_patience(convergence_rate)
        
        # Check for improvement
        if len(self.best_value_history) >= self.current_patience + 1:
            recent_history = self.best_value_history[-self.current_patience:]
            if len(recent_history) >= 2:
                if self.config.direction == 'maximize':
                    improvement = recent_history[-1] - recent_history[0]
                else:
                    improvement = recent_history[0] - recent_history[-1]
                
                if improvement < self.config.early_stopping_threshold:
                    self.trials_without_improvement += 1
                else:
                    self.trials_without_improvement = 0
        
        return self.trials_without_improvement >= self.current_patience
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        return f"Adaptive patience exceeded: {self.trials_without_improvement}/{self.current_patience} trials without improvement"
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate from recent history."""
        if len(self.best_value_history) < 10:
            return 1.0
        
        recent_values = self.best_value_history[-20:]
        if len(recent_values) < 5:
            return 1.0
        
        # Calculate improvements
        improvements = []
        for i in range(1, len(recent_values)):
            if self.config.direction == 'maximize':
                improvement = recent_values[i] - recent_values[i-1]
            else:
                improvement = recent_values[i-1] - recent_values[i]
            improvements.append(improvement)
        
        if not improvements:
            return 0.1
        
        # Calculate average improvement rate
        avg_improvement = np.mean(improvements)
        value_range = max(recent_values) - min(recent_values)
        
        if value_range == 0:
            return 1.0
        
        convergence_rate = max(0.1, min(2.0, avg_improvement / value_range * 10))
        return convergence_rate


class ConvergenceBasedStrategy(EarlyStoppingStrategy):
    """Convergence-based early stopping strategy."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig = None):
        super().__init__(config)
        self.convergence_window = config.convergence_window if config else 10
        self.min_improvement_rate = config.min_improvement_rate if config else 0.001
        self.convergence_history = []
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if convergence criteria is met."""
        if len(history) < self.convergence_window:
            return False
        
        # Calculate convergence rate
        recent_values = history[-self.convergence_window:]
        if len(recent_values) < 5:
            return False
        
        # Calculate improvement rate
        improvements = []
        for i in range(1, len(recent_values)):
            if self.config.direction == 'maximize':
                improvement = recent_values[i] - recent_values[i-1]
            else:
                improvement = recent_values[i-1] - recent_values[i]
            improvements.append(improvement)
        
        if not improvements:
            return False
        
        avg_improvement = np.mean(improvements)
        value_range = max(recent_values) - min(recent_values)
        
        if value_range == 0:
            return True  # No variation, consider converged
        
        improvement_rate = avg_improvement / value_range
        self.convergence_history.append(improvement_rate)
        
        # Check if improvement rate is below threshold
        if improvement_rate < self.min_improvement_rate:
            self.stopping_reason = f"Convergence achieved: improvement rate {improvement_rate:.6f} < {self.min_improvement_rate}"
            return True
        
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        return self.stopping_reason or "No stopping reason available"


class PerformanceBasedStrategy(EarlyStoppingStrategy):
    """Performance-based early stopping strategy."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig = None):
        super().__init__(config)
        self.performance_threshold = config.performance_threshold if config else 0.95
        self.performance_window = config.performance_window if config else 5
        self.performance_history = []
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if performance criteria is met."""
        if len(history) < self.performance_window:
            return False
        
        # Calculate current performance relative to best possible
        current_best = max(history) if self.config.direction == 'maximize' else min(history)
        theoretical_best = max(history) if self.config.direction == 'maximize' else min(history)
        
        if theoretical_best == 0:
            performance_ratio = 1.0
        else:
            performance_ratio = current_best / theoretical_best if self.config.direction == 'maximize' else theoretical_best / current_best
        
        self.performance_history.append(performance_ratio)
        
        # Check if performance is above threshold
        if performance_ratio >= self.performance_threshold:
            self.stopping_reason = f"Performance threshold reached: {performance_ratio:.4f} >= {self.performance_threshold}"
            return True
        
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        return self.stopping_reason or "No stopping reason available"


class TimeBasedStrategy(EarlyStoppingStrategy):
    """Time-based early stopping strategy."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig = None):
        super().__init__(config)
        self.max_time_seconds = config.max_time_seconds if config else 3600  # 1 hour default
        self.start_time = time.time()
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if time limit is exceeded."""
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= self.max_time_seconds:
            self.stopping_reason = f"Time limit exceeded: {elapsed_time:.2f}s >= {self.max_time_seconds}s"
            return True
        
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        return self.stopping_reason or "No stopping reason available"


class TrialBasedStrategy(EarlyStoppingStrategy):
    """Trial-based early stopping strategy."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig = None):
        super().__init__(config)
        self.max_trials = config.max_trials if config else 1000
        self.min_trials = config.min_trials if config else 10
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if trial limit is exceeded."""
        if current_trial >= self.max_trials:
            self.stopping_reason = f"Trial limit exceeded: {current_trial} >= {self.max_trials}"
            return True
        
        if current_trial < self.min_trials:
            return False
        
        # Also check for improvement
        if len(history) >= 2:
            recent_improvement = abs(history[-1] - history[-2])
            if recent_improvement < self.config.early_stopping_threshold:
                self.trials_without_improvement += 1
            else:
                self.trials_without_improvement = 0
            
            if self.trials_without_improvement >= self.config.early_stopping_patience:
                self.stopping_reason = f"No improvement for {self.trials_without_improvement} trials"
                return True
        
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        return self.stopping_reason or "No stopping reason available"


class CompositeStrategy(EarlyStoppingStrategy):
    """Composite early stopping strategy that combines multiple strategies."""
    
    def __init__(self, strategies: List[EarlyStoppingStrategy], config: EnhancedEarlyStoppingConfig = None):
        super().__init__(config)
        self.strategies = strategies
        self.stopping_strategy = None
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if any strategy indicates stopping."""
        for strategy in self.strategies:
            if strategy.should_stop(history, current_trial):
                self.stopping_strategy = strategy
                return True
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        if self.stopping_strategy:
            return f"Composite strategy: {self.stopping_strategy.get_stopping_reason()}"
        return "No stopping reason available"
    
    def _calculate_adaptive_patience(self, convergence_rate: float) -> int:
        """Calculate adaptive patience based on convergence rate."""
        base_patience = self.config.early_stopping_patience
        
        if convergence_rate > 1.0:
            # Fast convergence - reduce patience
            patience_factor = 0.7
        elif convergence_rate > 0.5:
            # Moderate convergence - normal patience
            patience_factor = 1.0
        elif convergence_rate > 0.2:
            # Slow convergence - increase patience
            patience_factor = 1.5
        else:
            # Very slow convergence - increase patience significantly
            patience_factor = 2.0
        
        adaptive_patience = int(base_patience * patience_factor)
        return max(self.config.min_patience, min(self.config.max_patience, adaptive_patience))


class ConfidenceBasedStrategy(EarlyStoppingStrategy):
    """Confidence-based early stopping strategy."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig):
        self.config = config
        self.confidence_history = []
        self.stopping_confidence = None
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if confidence-based stopping criteria is met."""
        if len(history) < self.config.min_history_for_confidence:
            return False
        
        self.confidence_history = history[-self.config.min_history_for_confidence:]
        
        # Perform statistical test for convergence
        if len(self.confidence_history) >= 10:
            # Test for no significant improvement
            recent_half = self.confidence_history[-len(self.confidence_history)//2:]
            older_half = self.confidence_history[:len(self.confidence_history)//2]
            
            if len(recent_half) >= 5 and len(older_half) >= 5:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(recent_half, older_half)
                
                # If p-value is high, there's no significant difference
                if p_value > (1 - self.config.confidence_level):
                    self.stopping_confidence = p_value
                    return True
        
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        if self.stopping_confidence:
            return f"Confidence-based stopping: p-value={self.stopping_confidence:.4f} (no significant improvement)"
        return "Confidence-based stopping: no significant improvement detected"


class MultiObjectiveStrategy(EarlyStoppingStrategy):
    """Multi-objective early stopping strategy using Pareto front analysis."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig):
        self.config = config
        self.pareto_solutions = []
        self.objective_history = {}
        self.trials_without_pareto_improvement = 0
    
    def should_stop(self, history: List[float], current_trial: int) -> bool:
        """Check if multi-objective stopping criteria is met."""
        if not self.config.multi_objective_stopping:
            return False
        
        # This would need to be implemented with actual multi-objective data
        # For now, return False as this is a placeholder
        return False
    
    def get_stopping_reason(self) -> str:
        """Get reason for early stopping."""
        return "Multi-objective stopping: no Pareto improvement detected"


class ThresholdScheduleStrategy:
    """Threshold scheduling strategy for adaptive thresholds."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig):
        self.config = config
        self.schedule_params = config.schedule_params or {}
    
    def calculate_threshold(self, current_trial: int, total_trials: int, 
                          base_threshold: float) -> float:
        """Calculate current threshold based on schedule."""
        if not self.config.threshold_schedule_enabled:
            return base_threshold
        
        progress = current_trial / total_trials if total_trials > 0 else 0
        
        if self.config.threshold_schedule_type == "exponential":
            decay_rate = self.schedule_params.get('decay_rate', 0.9)
            return base_threshold * (decay_rate ** progress)
        
        elif self.config.threshold_schedule_type == "linear":
            if self.config.final_threshold is not None:
                return base_threshold + (self.config.final_threshold - base_threshold) * progress
            else:
                return base_threshold * (1.0 - progress * 0.5)
        
        elif self.config.threshold_schedule_type == "step":
            step_points = self.schedule_params.get('step_points', [0.25, 0.5, 0.75])
            step_factors = self.schedule_params.get('step_factors', [0.8, 0.6, 0.4])
            
            current_threshold = base_threshold
            for i, point in enumerate(step_points):
                if progress >= point:
                    factor = step_factors[i] if i < len(step_factors) else step_factors[-1]
                    current_threshold = base_threshold * factor
            
            return current_threshold
        
        elif self.config.threshold_schedule_type == "adaptive":
            # Adaptive based on convergence rate (would need convergence rate input)
            return base_threshold
        
        return base_threshold


class EnhancedEarlyStoppingManager:
    """Manager for enhanced early stopping strategies."""
    
    def __init__(self, config: EnhancedEarlyStoppingConfig):
        self.config = config
        self.strategies = []
        self.threshold_scheduler = ThresholdScheduleStrategy(config)
        self.direction = 'maximize'  # Will be set by optimizer
        
        # Initialize strategies
        if config.adaptive_patience:
            self.strategies.append(AdaptivePatienceStrategy(config))
        
        if config.confidence_based_stopping:
            self.strategies.append(ConfidenceBasedStrategy(config))
        
        if config.multi_objective_stopping:
            self.strategies.append(MultiObjectiveStrategy(config))
    
    def set_direction(self, direction: str):
        """Set optimization direction."""
        self.direction = direction
        for strategy in self.strategies:
            if hasattr(strategy, 'direction'):
                strategy.direction = direction
    
    def should_stop(self, history: List[float], current_trial: int, 
                   total_trials: int) -> bool:
        """Check if any strategy suggests stopping."""
        if not self.strategies:
            return False
        
        # Update threshold based on schedule
        if self.config.threshold_schedule_enabled:
            base_threshold = self.config.early_stopping_threshold
            current_threshold = self.threshold_scheduler.calculate_threshold(
                current_trial, total_trials, base_threshold
            )
            # Update strategies with new threshold
            for strategy in self.strategies:
                if hasattr(strategy, 'config'):
                    strategy.config.early_stopping_threshold = current_threshold
        
        # Check each strategy
        for strategy in self.strategies:
            if strategy.should_stop(history, current_trial):
                return True
        
        return False
    
    def get_stopping_reasons(self) -> List[str]:
        """Get reasons from all strategies that suggest stopping."""
        reasons = []
        for strategy in self.strategies:
            if hasattr(strategy, 'should_stop') and strategy.should_stop([], 0):
                reasons.append(strategy.get_stopping_reason())
        return reasons
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from strategies."""
        metrics = {}
        
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'convergence_rate_history'):
                metrics[f'strategy_{i}_convergence_rate'] = strategy.convergence_rate_history
            if hasattr(strategy, 'best_value_history'):
                metrics[f'strategy_{i}_best_value_history'] = strategy.best_value_history
        
        return metrics


def create_enhanced_early_stopping_config(
    use_case: str = "general",
    n_trials: int = 100,
    **kwargs
) -> EnhancedEarlyStoppingConfig:
    """Create enhanced early stopping configuration for specific use case."""
    
    base_config = {
        'early_stopping_patience': 5,
        'early_stopping_threshold': 0.001,
        'adaptive_patience': True,
        'confidence_based_stopping': True,
        'threshold_schedule_enabled': True,
        'track_convergence': True,
    }
    
    # Use case specific configurations
    use_case_configs = {
        'model_training': {
            'early_stopping_patience': 5,
            'early_stopping_threshold': 0.001,
            'min_patience': 3,
            'max_patience': 15,
        },
        'clustering': {
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.001,
            'min_patience': 2,
            'max_patience': 10,
        },
        'backtesting': {
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.001,
            'min_patience': 2,
            'max_patience': 10,
        },
        'simple_parameters': {
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.001,
            'min_patience': 2,
            'max_patience': 8,
        }
    }
    
    # Merge configurations
    config = base_config.copy()
    if use_case in use_case_configs:
        config.update(use_case_configs[use_case])
    config.update(kwargs)
    
    return EnhancedEarlyStoppingConfig(**config)


# Convenience functions
def get_enhanced_tpe_config(use_case: str = "general", **kwargs) -> Dict[str, Any]:
    """Get enhanced TPE configuration with early stopping."""
    early_stopping_config = create_enhanced_early_stopping_config(use_case, **kwargs)
    
    return {
        'early_stopping_patience': early_stopping_config.early_stopping_patience,
        'early_stopping_threshold': early_stopping_config.early_stopping_threshold,
        'enable_pruner': True,
        'pruner_type': 'hyperband',
        'adaptive_patience': early_stopping_config.adaptive_patience,
        'confidence_based_stopping': early_stopping_config.confidence_based_stopping,
        'threshold_schedule_enabled': early_stopping_config.threshold_schedule_enabled,
        'track_convergence': early_stopping_config.track_convergence,
    }


def create_early_stopping_manager(use_case: str = "general", **kwargs) -> EnhancedEarlyStoppingManager:
    """Create enhanced early stopping manager for specific use case."""
    config = create_enhanced_early_stopping_config(use_case, **kwargs)
    return EnhancedEarlyStoppingManager(config)