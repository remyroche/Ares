"""
Enhanced Early Stopping Integration

This module integrates the enhanced early stopping capabilities with the existing
HPO engine, providing seamless early stopping across all optimization strategies.

Enhancement: Early stopping for HPO trials
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import existing early stopping components
from .enhanced_tpe_early_stopping import (
    EnhancedEarlyStoppingConfig,
    EnhancedEarlyStoppingManager,
    create_enhanced_early_stopping_config,
    create_early_stopping_manager
)

# Import HPO components
from .core.optimization_strategy import OptimizationStrategy, OptimizationContext
from .validation import HPOConfig
from .results import HPOResult

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingIntegrationConfig:
    """Configuration for early stopping integration."""
    
    # Enable early stopping
    enable_early_stopping: bool = True
    
    # Strategy-specific early stopping
    enable_bayesian_early_stopping: bool = True
    enable_grid_early_stopping: bool = True
    enable_random_early_stopping: bool = True
    enable_bohb_early_stopping: bool = True
    
    # Early stopping configuration
    early_stopping_config: Optional[EnhancedEarlyStoppingConfig] = None
    
    # Performance tracking
    track_early_stopping_metrics: bool = True
    save_early_stopping_log: bool = True
    early_stopping_log_file: str = "early_stopping_log.json"
    
    # Adaptive settings
    adaptive_early_stopping: bool = True
    min_trials_before_stopping: int = 10
    max_early_stopping_patience: int = 50


class EarlyStoppingIntegration:
    """Integrates enhanced early stopping with HPO strategies."""
    
    def __init__(self, config: EarlyStoppingIntegrationConfig):
        self.config = config
        self.early_stopping_managers: Dict[str, EnhancedEarlyStoppingManager] = {}
        self.early_stopping_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.early_stopping_log: List[Dict[str, Any]] = []
        
        # Initialize early stopping managers for each strategy
        self._initialize_early_stopping_managers()
        
        logger.info("Early stopping integration initialized")
    
    def _initialize_early_stopping_managers(self):
        """Initialize early stopping managers for each strategy."""
        strategies = ['bayesian', 'grid', 'random', 'bohb']
        
        for strategy in strategies:
            if self._should_enable_early_stopping(strategy):
                # Create strategy-specific early stopping config
                early_stopping_config = self._create_strategy_early_stopping_config(strategy)
                
                # Create early stopping manager
                manager = create_early_stopping_manager(
                    use_case=self._get_strategy_use_case(strategy),
                    **early_stopping_config.__dict__
                )
                
                self.early_stopping_managers[strategy] = manager
                self.early_stopping_metrics[strategy] = []
                
                logger.info(f"Early stopping enabled for {strategy} strategy")
    
    def _should_enable_early_stopping(self, strategy: str) -> bool:
        """Check if early stopping should be enabled for a strategy."""
        if not self.config.enable_early_stopping:
            return False
        
        strategy_enabled = {
            'bayesian': self.config.enable_bayesian_early_stopping,
            'grid': self.config.enable_grid_early_stopping,
            'random': self.config.enable_random_early_stopping,
            'bohb': self.config.enable_bohb_early_stopping
        }
        
        return strategy_enabled.get(strategy, True)
    
    def _create_strategy_early_stopping_config(self, strategy: str) -> EnhancedEarlyStoppingConfig:
        """Create early stopping configuration for a specific strategy."""
        if self.config.early_stopping_config:
            return self.config.early_stopping_config
        
        # Strategy-specific configurations
        strategy_configs = {
            'bayesian': {
                'early_stopping_patience': 8,
                'early_stopping_threshold': 0.001,
                'adaptive_patience': True,
                'confidence_based_stopping': True,
                'min_patience': 3,
                'max_patience': 20
            },
            'grid': {
                'early_stopping_patience': 5,
                'early_stopping_threshold': 0.001,
                'adaptive_patience': False,  # Grid search is deterministic
                'confidence_based_stopping': False,
                'min_patience': 2,
                'max_patience': 10
            },
            'random': {
                'early_stopping_patience': 10,
                'early_stopping_threshold': 0.001,
                'adaptive_patience': True,
                'confidence_based_stopping': True,
                'min_patience': 5,
                'max_patience': 25
            },
            'bohb': {
                'early_stopping_patience': 6,
                'early_stopping_threshold': 0.001,
                'adaptive_patience': True,
                'confidence_based_stopping': True,
                'min_patience': 3,
                'max_patience': 15
            }
        }
        
        config_params = strategy_configs.get(strategy, {})
        return create_enhanced_early_stopping_config(**config_params)
    
    def _get_strategy_use_case(self, strategy: str) -> str:
        """Get use case for strategy-specific early stopping."""
        use_case_map = {
            'bayesian': 'model_training',
            'grid': 'simple_parameters',
            'random': 'model_training',
            'bohb': 'model_training'
        }
        return use_case_map.get(strategy, 'general')
    
    def should_stop_early(self, strategy: str, trial_history: List[float], 
                         current_trial: int, total_trials: int) -> bool:
        """Check if optimization should stop early."""
        if strategy not in self.early_stopping_managers:
            return False
        
        manager = self.early_stopping_managers[strategy]
        
        # Check minimum trials requirement
        if current_trial < self.config.min_trials_before_stopping:
            return False
        
        # Check early stopping
        should_stop = manager.should_stop(trial_history, current_trial, total_trials)
        
        # Track metrics if enabled
        if self.config.track_early_stopping_metrics:
            self._track_early_stopping_metrics(
                strategy, trial_history, current_trial, should_stop
            )
        
        # Log early stopping decision
        if should_stop:
            self._log_early_stopping_decision(strategy, trial_history, current_trial)
        
        return should_stop
    
    def _track_early_stopping_metrics(self, strategy: str, trial_history: List[float],
                                    current_trial: int, should_stop: bool):
        """Track early stopping metrics."""
        if strategy not in self.early_stopping_metrics:
            self.early_stopping_metrics[strategy] = []
        
        metrics = {
            'trial': current_trial,
            'timestamp': time.time(),
            'history_length': len(trial_history),
            'best_value': max(trial_history) if trial_history else 0.0,
            'worst_value': min(trial_history) if trial_history else 0.0,
            'mean_value': np.mean(trial_history) if trial_history else 0.0,
            'std_value': np.std(trial_history) if trial_history else 0.0,
            'should_stop': should_stop,
            'improvement_rate': self._calculate_improvement_rate(trial_history)
        }
        
        self.early_stopping_metrics[strategy].append(metrics)
    
    def _calculate_improvement_rate(self, trial_history: List[float]) -> float:
        """Calculate improvement rate from trial history."""
        if len(trial_history) < 2:
            return 0.0
        
        recent_window = min(10, len(trial_history))
        recent_values = trial_history[-recent_window:]
        
        if len(recent_values) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(recent_values)):
            improvement = recent_values[i] - recent_values[i-1]
            improvements.append(improvement)
        
        return float(np.mean(improvements)) if improvements else 0.0
    
    def _log_early_stopping_decision(self, strategy: str, trial_history: List[float],
                                   current_trial: int):
        """Log early stopping decision."""
        if not self.config.save_early_stopping_log:
            return
        
        log_entry = {
            'timestamp': time.time(),
            'strategy': strategy,
            'trial': current_trial,
            'history_length': len(trial_history),
            'best_value': max(trial_history) if trial_history else 0.0,
            'stopping_reason': self._get_stopping_reason(strategy),
            'trial_history': trial_history[-20:]  # Keep last 20 trials
        }
        
        self.early_stopping_log.append(log_entry)
        
        # Save to file periodically
        if len(self.early_stopping_log) % 10 == 0:
            self._save_early_stopping_log()
    
    def _get_stopping_reason(self, strategy: str) -> str:
        """Get stopping reason from early stopping manager."""
        if strategy in self.early_stopping_managers:
            manager = self.early_stopping_managers[strategy]
            return manager.get_stopping_reasons()
        return "Unknown"
    
    def _save_early_stopping_log(self):
        """Save early stopping log to file."""
        try:
            import json
            with open(self.config.early_stopping_log_file, 'w') as f:
                json.dump(self.early_stopping_log, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save early stopping log: {e}")
    
    def get_early_stopping_summary(self) -> Dict[str, Any]:
        """Get summary of early stopping performance."""
        summary = {}
        
        for strategy, metrics in self.early_stopping_metrics.items():
            if not metrics:
                continue
            
            strategy_summary = {
                'total_trials': len(metrics),
                'early_stops': sum(1 for m in metrics if m['should_stop']),
                'avg_improvement_rate': np.mean([m['improvement_rate'] for m in metrics]),
                'avg_best_value': np.mean([m['best_value'] for m in metrics]),
                'final_best_value': metrics[-1]['best_value'] if metrics else 0.0
            }
            
            summary[strategy] = strategy_summary
        
        return summary
    
    def reset_early_stopping(self, strategy: Optional[str] = None):
        """Reset early stopping state."""
        if strategy:
            if strategy in self.early_stopping_managers:
                # Reset specific strategy
                self.early_stopping_managers[strategy].strategies = []
                self.early_stopping_metrics[strategy] = []
        else:
            # Reset all strategies
            for strategy_name in self.early_stopping_managers:
                self.reset_early_stopping(strategy_name)
            self.early_stopping_log = []


class EarlyStoppingOptimizationStrategy(OptimizationStrategy):
    """Base class for optimization strategies with early stopping support."""
    
    def __init__(self, config: HPOConfig, early_stopping_integration: EarlyStoppingIntegration):
        super().__init__(config)
        self.early_stopping_integration = early_stopping_integration
        self.trial_history: List[float] = []
        self.early_stopped = False
        self.early_stopping_reason = None
    
    def should_stop_early(self, current_trial: int) -> bool:
        """Check if optimization should stop early."""
        strategy_name = self.config.strategy.value
        
        should_stop = self.early_stopping_integration.should_stop_early(
            strategy_name, self.trial_history, current_trial, self.config.n_trials
        )
        
        if should_stop:
            self.early_stopped = True
            self.early_stopping_reason = self.early_stopping_integration._get_stopping_reason(strategy_name)
            logger.info(f"Early stopping triggered for {strategy_name} at trial {current_trial}")
        
        return should_stop
    
    def update_trial_history(self, score: float):
        """Update trial history with new score."""
        self.trial_history.append(score)
    
    def get_early_stopping_info(self) -> Dict[str, Any]:
        """Get early stopping information."""
        return {
            'early_stopped': self.early_stopped,
            'early_stopping_reason': self.early_stopping_reason,
            'trial_history_length': len(self.trial_history),
            'best_score': max(self.trial_history) if self.trial_history else 0.0
        }


def create_early_stopping_integration(
    enable_early_stopping: bool = True,
    **kwargs
) -> EarlyStoppingIntegration:
    """Create early stopping integration with default settings."""
    config = EarlyStoppingIntegrationConfig(
        enable_early_stopping=enable_early_stopping,
        **kwargs
    )
    return EarlyStoppingIntegration(config)


def integrate_early_stopping_with_strategy(
    strategy: OptimizationStrategy,
    early_stopping_integration: EarlyStoppingIntegration
) -> EarlyStoppingOptimizationStrategy:
    """Integrate early stopping with an existing optimization strategy."""
    return EarlyStoppingOptimizationStrategy(
        strategy.config,
        early_stopping_integration
    )