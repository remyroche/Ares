"""
Standalone Early Stopping Strategies

This module provides early stopping strategies that can be used independently
without heavy dependencies like numpy, scipy, or the full ml_common module.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping strategies."""
    
    # Basic early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Adaptive patience
    adaptive_patience: bool = True
    min_patience: int = 3
    max_patience: int = 20
    
    # Performance tracking
    track_convergence: bool = True
    convergence_window: int = 10
    min_improvement_rate: float = 0.001
    
    # Performance thresholds
    performance_threshold: float = 0.95
    performance_window: int = 5
    
    # Time and trial limits
    max_time_seconds: int = 3600  # 1 hour
    max_trials: int = 1000
    min_trials: int = 10
    
    # Direction
    direction: str = 'maximize'  # 'maximize' or 'minimize'


class EarlyStoppingStrategy(ABC):
    """Abstract base class for early stopping strategies."""
    
    def __init__(self, config: EarlyStoppingConfig = None):
        """Initialize early stopping strategy."""
        self.config = config or EarlyStoppingConfig()
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
    
    def __init__(self, config: EarlyStoppingConfig = None):
        super().__init__(config)
        self.current_patience = self.config.early_stopping_patience
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
        avg_improvement = sum(improvements) / len(improvements)
        value_range = max(recent_values) - min(recent_values)
        
        if value_range == 0:
            return 1.0
        
        convergence_rate = max(0.1, min(2.0, avg_improvement / value_range * 10))
        return convergence_rate
    
    def _calculate_adaptive_patience(self, convergence_rate: float) -> int:
        """Calculate adaptive patience based on convergence rate."""
        base_patience = self.config.early_stopping_patience
        
        if convergence_rate > 1.5:
            # Fast convergence - reduce patience
            return max(self.config.min_patience, int(base_patience * 0.7))
        elif convergence_rate < 0.5:
            # Slow convergence - increase patience
            return min(self.config.max_patience, int(base_patience * 1.5))
        else:
            # Normal convergence - keep base patience
            return base_patience


class ConvergenceBasedStrategy(EarlyStoppingStrategy):
    """Convergence-based early stopping strategy."""
    
    def __init__(self, config: EarlyStoppingConfig = None):
        super().__init__(config)
        self.convergence_window = self.config.convergence_window
        self.min_improvement_rate = self.config.min_improvement_rate
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
        
        avg_improvement = sum(improvements) / len(improvements)
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
    
    def __init__(self, config: EarlyStoppingConfig = None):
        super().__init__(config)
        self.performance_threshold = self.config.performance_threshold
        self.performance_window = self.config.performance_window
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
    
    def __init__(self, config: EarlyStoppingConfig = None):
        super().__init__(config)
        self.max_time_seconds = self.config.max_time_seconds
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
    
    def __init__(self, config: EarlyStoppingConfig = None):
        super().__init__(config)
        self.max_trials = self.config.max_trials
        self.min_trials = self.config.min_trials
    
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
    
    def __init__(self, strategies: List[EarlyStoppingStrategy], config: EarlyStoppingConfig = None):
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


# Factory functions for easy creation
def create_adaptive_patience_strategy(config: EarlyStoppingConfig = None) -> AdaptivePatienceStrategy:
    """Create an adaptive patience early stopping strategy."""
    return AdaptivePatienceStrategy(config)


def create_convergence_strategy(config: EarlyStoppingConfig = None) -> ConvergenceBasedStrategy:
    """Create a convergence-based early stopping strategy."""
    return ConvergenceBasedStrategy(config)


def create_performance_strategy(config: EarlyStoppingConfig = None) -> PerformanceBasedStrategy:
    """Create a performance-based early stopping strategy."""
    return PerformanceBasedStrategy(config)


def create_time_strategy(config: EarlyStoppingConfig = None) -> TimeBasedStrategy:
    """Create a time-based early stopping strategy."""
    return TimeBasedStrategy(config)


def create_trial_strategy(config: EarlyStoppingConfig = None) -> TrialBasedStrategy:
    """Create a trial-based early stopping strategy."""
    return TrialBasedStrategy(config)


def create_composite_strategy(strategies: List[EarlyStoppingStrategy], config: EarlyStoppingConfig = None) -> CompositeStrategy:
    """Create a composite early stopping strategy."""
    return CompositeStrategy(strategies, config)


def create_default_strategy(config: EarlyStoppingConfig = None) -> CompositeStrategy:
    """Create a default composite strategy with common strategies."""
    if config is None:
        config = EarlyStoppingConfig()
    
    strategies = [
        AdaptivePatienceStrategy(config),
        ConvergenceBasedStrategy(config),
        TimeBasedStrategy(config),
        TrialBasedStrategy(config)
    ]
    
    return CompositeStrategy(strategies, config)