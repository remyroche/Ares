"""
Enhanced Pruner System for HPO

This module provides an improved pruner system that replaces the basic MedianPruner
with more sophisticated early stopping strategies and integrates with Ares launcher
execution modes for adaptive optimization intensity.

Key Features:
- Multiple pruner strategies (Adaptive, Confidence-based, Multi-fidelity)
- Ares launcher mode integration (light/blank/full)
- Dynamic intensity scaling based on execution mode
- Better convergence detection
- Resource-aware pruning
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
import logging
import time
from abc import ABC, abstractmethod
from scipy import stats
from enum import Enum

# Optuna imports
try:
    import optuna
    from optuna.pruners import BasePruner, MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    BasePruner = object

logger = logging.getLogger(__name__)


class AresExecutionMode(Enum):
    """Ares launcher execution modes."""
    LIGHT = "light"      # 10% intensity - quick testing
    BLANK = "blank"      # 25% intensity - moderate testing  
    FULL = "full"        # 100% intensity - full optimization


class PrunerStrategy(Enum):
    """Available pruner strategies."""
    ADAPTIVE = "adaptive"
    CONFIDENCE_BASED = "confidence_based"
    MULTI_FIDELITY = "multi_fidelity"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    MEDIAN = "median"  # Fallback to original


@dataclass
class AresModeConfig:
    """Configuration for Ares launcher execution modes."""
    
    # Intensity scaling factors
    light_intensity: float = 0.05    # 5% of full intensity
    blank_intensity: float = 0.25    # 25% of full intensity  
    full_intensity: float = 1.00     # 100% intensity
    
    # Mode-specific parameter adjustments
    mode_adjustments: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "light": {
            "n_trials_multiplier": 0.05,
            "patience_multiplier": 0.5,
            "threshold_multiplier": 2.0,
            "enable_aggressive_pruning": True,
            "max_trials": 10,
            "timeout_multiplier": 0.2
        },
        "blank": {
            "n_trials_multiplier": 0.25,
            "patience_multiplier": 0.7,
            "threshold_multiplier": 1.5,
            "enable_aggressive_pruning": True,
            "max_trials": 50,
            "timeout_multiplier": 0.6
        },
        "full": {
            "n_trials_multiplier": 1.0,
            "patience_multiplier": 1.0,
            "threshold_multiplier": 1.0,
            "enable_aggressive_pruning": False,
            "max_trials": 200,
            "timeout_multiplier": 1.0
        }
    })


@dataclass
class EnhancedPrunerConfig:
    """Configuration for enhanced pruner system."""
    
    # Basic settings
    strategy: PrunerStrategy = PrunerStrategy.ADAPTIVE
    ares_mode: AresExecutionMode = AresExecutionMode.FULL
    
    # Adaptive pruning settings
    base_patience: int = 10
    min_patience: int = 3
    max_patience: int = 50
    improvement_threshold: float = 0.001
    
    # Confidence-based settings
    confidence_level: float = 0.95
    min_trials_for_confidence: int = 15
    convergence_threshold: float = 0.01
    
    # Multi-fidelity settings
    min_resource: int = 1
    max_resource: int = 100
    reduction_factor: float = 3.0
    
    # Performance tracking
    track_convergence: bool = True
    convergence_window: int = 10
    min_improvement_rate: float = 0.0001
    
    # Ares mode integration
    enable_mode_scaling: bool = True
    mode_config: AresModeConfig = field(default_factory=AresModeConfig)


class EnhancedPruner(BasePruner):
    """
    Enhanced pruner that adapts to Ares launcher execution modes.
    
    This pruner provides intelligent early stopping based on:
    - Ares execution mode (light/blank/full)
    - Convergence detection
    - Confidence intervals
    - Resource efficiency
    """
    
    def __init__(self, config: EnhancedPrunerConfig = None):
        """Initialize enhanced pruner."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for EnhancedPruner")
            
        super().__init__()
        self.config = config or EnhancedPrunerConfig()
        self.logger = logger.getChild('EnhancedPruner')
        
        # Initialize mode-specific settings
        self._apply_mode_scaling()
        
        # Performance tracking
        self.trial_scores = []
        self.trial_resources = []
        self.convergence_history = []
        self.pruning_decisions = []
        
        # Initialize strategy-specific components
        self._initialize_strategy()
        
        self.logger.info(f"Enhanced pruner initialized with {self.config.strategy.value} strategy for {self.config.ares_mode.value} mode")
    
    def _apply_mode_scaling(self):
        """Apply Ares mode scaling to configuration."""
        if not self.config.enable_mode_scaling:
            return
            
        mode = self.config.ares_mode.value
        adjustments = self.config.mode_config.mode_adjustments.get(mode, {})
        
        # Apply intensity scaling
        self.config.base_patience = int(self.config.base_patience * adjustments.get('patience_multiplier', 1.0))
        self.config.improvement_threshold *= adjustments.get('threshold_multiplier', 1.0)
        
        # Ensure minimum values
        self.config.base_patience = max(self.config.min_patience, self.config.base_patience)
        self.config.improvement_threshold = max(0.0001, self.config.improvement_threshold)
        
        self.logger.info(f"Applied {mode} mode scaling: patience={self.config.base_patience}, threshold={self.config.improvement_threshold:.6f}")
    
    def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        if self.config.strategy == PrunerStrategy.ADAPTIVE:
            self._init_adaptive_strategy()
        elif self.config.strategy == PrunerStrategy.CONFIDENCE_BASED:
            self._init_confidence_strategy()
        elif self.config.strategy == PrunerStrategy.MULTI_FIDELITY:
            self._init_multi_fidelity_strategy()
        else:
            self._init_fallback_strategy()
    
    def _init_adaptive_strategy(self):
        """Initialize adaptive pruning strategy."""
        self.convergence_window = self.config.convergence_window
        self.min_improvement_rate = self.config.min_improvement_rate
        self.adaptive_patience = self.config.base_patience
        self.last_improvement_trial = 0
        
    def _init_confidence_strategy(self):
        """Initialize confidence-based strategy."""
        self.confidence_level = self.config.confidence_level
        self.min_trials = self.config.min_trials_for_confidence
        self.convergence_threshold = self.config.convergence_threshold
        
    def _init_multi_fidelity_strategy(self):
        """Initialize multi-fidelity strategy."""
        self.min_resource = self.config.min_resource
        self.max_resource = self.config.max_resource
        self.reduction_factor = self.config.reduction_factor
        
    def _init_fallback_strategy(self):
        """Initialize fallback strategy."""
        self.fallback_pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    
    def prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """
        Determine if a trial should be pruned.
        
        Args:
            study: Optuna study object
            trial: Current trial to evaluate
            
        Returns:
            True if trial should be pruned, False otherwise
        """
        try:
            # Get trial value and step
            trial_value = trial.value
            trial_step = trial.number
            
            if trial_value is None:
                return False  # Don't prune incomplete trials
            
            # Record trial information
            self.trial_scores.append(trial_value)
            self.trial_resources.append(trial_step)
            
            # Apply strategy-specific pruning logic
            should_prune = self._apply_pruning_strategy(study, trial)
            
            # Record pruning decision
            self.pruning_decisions.append({
                'trial': trial_step,
                'value': trial_value,
                'pruned': should_prune,
                'timestamp': time.time()
            })
            
            if should_prune:
                self.logger.debug(f"Pruned trial {trial_step} with value {trial_value:.6f}")
            
            return should_prune
            
        except Exception as e:
            self.logger.error(f"Error in pruning logic: {e}")
            return False  # Don't prune on error
    
    def _apply_pruning_strategy(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Apply the configured pruning strategy."""
        if self.config.strategy == PrunerStrategy.ADAPTIVE:
            return self._adaptive_prune(study, trial)
        elif self.config.strategy == PrunerStrategy.CONFIDENCE_BASED:
            return self._confidence_based_prune(study, trial)
        elif self.config.strategy == PrunerStrategy.MULTI_FIDELITY:
            return self._multi_fidelity_prune(study, trial)
        elif self.config.strategy == PrunerStrategy.HYPERBAND:
            return self._hyperband_prune(study, trial)
        elif self.config.strategy == PrunerStrategy.SUCCESSIVE_HALVING:
            return self._successive_halving_prune(study, trial)
        else:
            return self._fallback_prune(study, trial)
    
    def _adaptive_prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Adaptive pruning based on convergence patterns."""
        if len(self.trial_scores) < self.config.min_patience:
            return False
        
        # Calculate convergence metrics
        recent_scores = self.trial_scores[-self.convergence_window:]
        if len(recent_scores) < 3:
            return False
        
        # Check for improvement rate
        improvement_rate = self._calculate_improvement_rate(recent_scores)
        
        # Adaptive patience based on improvement rate
        if improvement_rate < self.min_improvement_rate:
            self.adaptive_patience = max(self.config.min_patience, self.adaptive_patience - 1)
        else:
            self.adaptive_patience = min(self.config.max_patience, self.adaptive_patience + 1)
        
        # Check if we've exceeded adaptive patience
        trials_since_improvement = trial.number - self.last_improvement_trial
        if trials_since_improvement >= self.adaptive_patience:
            return True
        
        # Check for convergence
        if self._is_converged(recent_scores):
            return True
        
        return False
    
    def _confidence_based_prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Confidence-based pruning using statistical tests."""
        if len(self.trial_scores) < self.min_trials:
            return False
        
        # Calculate confidence interval for recent performance
        recent_scores = self.trial_scores[-self.min_trials:]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if std_score == 0:
            return False
        
        # Calculate confidence interval
        confidence_interval = stats.t.interval(
            self.confidence_level,
            len(recent_scores) - 1,
            loc=mean_score,
            scale=std_score / np.sqrt(len(recent_scores))
        )
        
        # Check if current trial is significantly below confidence interval
        current_score = trial.value
        if current_score < confidence_interval[0]:
            return True
        
        # Check for convergence (narrow confidence interval)
        interval_width = confidence_interval[1] - confidence_interval[0]
        if interval_width < self.convergence_threshold:
            return True
        
        return False
    
    def _multi_fidelity_prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Multi-fidelity pruning for resource-aware optimization."""
        # This would integrate with resource allocation
        # For now, use a simplified version
        if len(self.trial_scores) < 5:
            return False
        
        # Simple multi-fidelity logic
        recent_scores = self.trial_scores[-5:]
        current_score = trial.value
        
        # Prune if current score is significantly worse than recent average
        recent_avg = np.mean(recent_scores)
        if current_score < recent_avg - 2 * np.std(recent_scores):
            return True
        
        return False
    
    def _hyperband_prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Hyperband-style pruning."""
        if not hasattr(self, 'hyperband_pruner'):
            self.hyperband_pruner = HyperbandPruner(
                min_resource=self.min_resource,
                max_resource=self.max_resource,
                reduction_factor=self.reduction_factor
            )
        
        return self.hyperband_pruner.prune(study, trial)
    
    def _successive_halving_prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Successive halving pruning."""
        if not hasattr(self, 'sh_pruner'):
            self.sh_pruner = SuccessiveHalvingPruner(
                min_resource=self.min_resource,
                reduction_factor=self.reduction_factor
            )
        
        return self.sh_pruner.prune(study, trial)
    
    def _fallback_prune(self, study: 'optuna.Study', trial: 'optuna.Trial') -> bool:
        """Fallback to median pruner."""
        return self.fallback_pruner.prune(study, trial)
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate improvement rate over recent scores."""
        if len(scores) < 2:
            return 0.0
        
        # Calculate slope of recent performance
        x = np.arange(len(scores))
        slope, _, _, _, _ = stats.linregress(x, scores)
        return slope
    
    def _is_converged(self, scores: List[float]) -> bool:
        """Check if optimization has converged."""
        if len(scores) < 3:
            return False
        
        # Check if variance is very low (converged)
        variance = np.var(scores)
        if variance < self.min_improvement_rate:
            return True
        
        # Check if improvement rate is very low
        improvement_rate = self._calculate_improvement_rate(scores)
        if abs(improvement_rate) < self.min_improvement_rate:
            return True
        
        return False
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        if not self.pruning_decisions:
            return {}
        
        total_trials = len(self.pruning_decisions)
        pruned_trials = sum(1 for d in self.pruning_decisions if d['pruned'])
        pruning_rate = pruned_trials / total_trials if total_trials > 0 else 0
        
        return {
            'total_trials': total_trials,
            'pruned_trials': pruned_trials,
            'pruning_rate': pruning_rate,
            'strategy': self.config.strategy.value,
            'ares_mode': self.config.ares_mode.value,
            'adaptive_patience': getattr(self, 'adaptive_patience', self.config.base_patience)
        }


def create_enhanced_pruner(
    ares_mode: str = "full",
    strategy: str = "adaptive",
    **kwargs
) -> EnhancedPruner:
    """
    Create an enhanced pruner with Ares mode integration.
    
    Args:
        ares_mode: Ares execution mode ('light', 'blank', 'full')
        strategy: Pruner strategy ('adaptive', 'confidence_based', 'multi_fidelity', etc.)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnhancedPruner instance
    """
    # Parse execution mode
    try:
        execution_mode = AresExecutionMode(ares_mode.lower())
    except ValueError:
        logger.warning(f"Invalid ares_mode '{ares_mode}', defaulting to 'full'")
        execution_mode = AresExecutionMode.FULL
    
    # Parse strategy
    try:
        pruner_strategy = PrunerStrategy(strategy.lower())
    except ValueError:
        logger.warning(f"Invalid strategy '{strategy}', defaulting to 'adaptive'")
        pruner_strategy = PrunerStrategy.ADAPTIVE
    
    # Create configuration
    config = EnhancedPrunerConfig(
        strategy=pruner_strategy,
        ares_mode=execution_mode,
        **kwargs
    )
    
    return EnhancedPruner(config)


def get_ares_mode_from_context() -> str:
    """
    Extract Ares execution mode from context (environment variables, config, etc.).
    
    Returns:
        Ares execution mode string ('light', 'blank', 'full')
    """
    import os
    
    # Check environment variable first
    mode = os.getenv('ARES_EXECUTION_MODE', '').lower()
    if mode in ['light', 'blank', 'full']:
        return mode
    
    # Check for ares_launcher context
    # This would be set by the launcher when running steps
    launcher_mode = os.getenv('ARES_LAUNCHER_MODE', '').lower()
    if launcher_mode in ['light', 'blank', 'full']:
        return launcher_mode
    
    # Default to full mode
    return 'full'


# Convenience functions for easy integration
def create_light_mode_pruner(**kwargs) -> EnhancedPruner:
    """Create pruner optimized for light mode (10% intensity)."""
    return create_enhanced_pruner(ares_mode="light", **kwargs)


def create_blank_mode_pruner(**kwargs) -> EnhancedPruner:
    """Create pruner optimized for blank mode (25% intensity)."""
    return create_enhanced_pruner(ares_mode="blank", **kwargs)


def create_full_mode_pruner(**kwargs) -> EnhancedPruner:
    """Create pruner optimized for full mode (100% intensity)."""
    return create_enhanced_pruner(ares_mode="full", **kwargs)


def create_auto_mode_pruner(**kwargs) -> EnhancedPruner:
    """Create pruner with automatic mode detection."""
    mode = get_ares_mode_from_context()
    return create_enhanced_pruner(ares_mode=mode, **kwargs)


# Export main classes and functions
__all__ = [
    'EnhancedPruner',
    'EnhancedPrunerConfig', 
    'AresModeConfig',
    'AresExecutionMode',
    'PrunerStrategy',
    'create_enhanced_pruner',
    'create_light_mode_pruner',
    'create_blank_mode_pruner', 
    'create_full_mode_pruner',
    'create_auto_mode_pruner',
    'get_ares_mode_from_context'
]