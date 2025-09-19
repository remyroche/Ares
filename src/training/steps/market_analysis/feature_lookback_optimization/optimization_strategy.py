"""
Optimization Strategy Abstraction Layer.

This module provides abstract base classes and interfaces for different
optimization strategies, enabling pluggable optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from .constants import OPTIMIZATION_CONSTANTS

# Import math validation utilities for safe operations
try:
    from src.utils.math_validation import (
        safe_correlation, validate_finite, safe_mean, safe_std
    )
    from src.utils.core.common import safe_list_get
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    
    # Fallback implementations
    def safe_list_get(lst, index, default=None):
        try:
            return lst[index] if lst and 0 <= index < len(lst) else default
        except (IndexError, TypeError):
            return default
    
    def safe_correlation(x, y, default=0.0):
        try:
            if len(x) != len(y) or len(x) < 2:
                return default
            corr = np.corrcoef(x, y)[0, 1]
            return corr if np.isfinite(corr) else default
        except:
            return default


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    TWO_STEP_GRID_TPE = "two_step_grid_tpe"
    MRMR = "mrmr"
    RANDOM_SEARCH = "random_search"


@dataclass
class OptimizationResult:
    """Standardized optimization result."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_trials: int
    optimization_time: float
    convergence_achieved: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_lookback_period': self.best_lookback_period,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'total_trials': self.total_trials,
            'optimization_time': self.optimization_time,
            'convergence_achieved': self.convergence_achieved,
            'metadata': self.metadata
        }


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the optimization strategy."""
        self.config = config or {}
        self.method_name = "base_strategy"
    
    @abstractmethod
    def optimize(self, 
                 data: pd.DataFrame, 
                 feature_name: str, 
                 target_column: str,
                 **kwargs) -> OptimizationResult:
        """
        Perform optimization.
        
        Args:
            data: Input data for optimization
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult with optimization results
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, 
                       data: pd.DataFrame, 
                       feature_name: str, 
                       target_column: str) -> Tuple[bool, str]:
        """
        Validate inputs for optimization.
        
        Args:
            data: Input data
            feature_name: Feature name
            target_column: Target column name
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this strategy."""
        return {
            'min_lookback': OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK,
            'max_lookback': OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK,
            'random_state': 42
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(new_config)


class GridSearchStrategy(OptimizationStrategy):
    """Grid search optimization strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.method_name = "grid_search"
    
    def optimize(self, 
                 data: pd.DataFrame, 
                 feature_name: str, 
                 target_column: str,
                 **kwargs) -> OptimizationResult:
        """Perform grid search optimization."""
        import time
        import numpy as np
        from sklearn.feature_selection import mutual_info_regression
        
        start_time = time.time()
        
        # Get parameters
        min_lookback = self.config.get('min_lookback', OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK)
        max_lookback = self.config.get('max_lookback', OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK)
        grid_size = self.config.get('grid_size', 20)
        
        # Create grid
        lookback_values = np.linspace(min_lookback, max_lookback, grid_size, dtype=int)
        
        best_score = -np.inf
        best_lookback = min_lookback
        total_trials = 0
        
        # Grid search
        for lookback in lookback_values:
            try:
                # Generate feature with current lookback
                feature_values = self._generate_feature_with_lookback(data, feature_name, lookback)
                target_values = data[target_column].values
                
                # Safely align and clean data
                if len(feature_values) == 0 or len(target_values) == 0:
                    continue
                    
                min_length = min(len(feature_values), len(target_values))
                feature_values = feature_values[:min_length]
                target_values = target_values[:min_length]
                
                # Remove NaN values with safe operations
                if MATH_VALIDATION_AVAILABLE:
                    valid_mask = np.isfinite(feature_values) & np.isfinite(target_values)
                    if not np.any(valid_mask):
                        continue
                    feature_clean = feature_values[valid_mask]
                    target_clean = target_values[valid_mask]
                else:
                    mask = ~(np.isnan(feature_values) | np.isnan(target_values))
                    if mask.sum() < 10:
                        continue
                    feature_clean = feature_values[mask]
                    target_clean = target_values[mask]
                
                # Check minimum data requirement (increased for reliability)
                min_samples = max(30, lookback * 2)
                if len(feature_clean) < min_samples:
                    continue
                
                # Calculate score with safe operations
                score = 0.0
                try:
                    mi_scores = mutual_info_regression(
                        feature_clean.reshape(-1, 1), 
                        target_clean,
                        random_state=self.config.get('random_state', 42)
                    )
                    # Safe array access
                    score = safe_list_get(mi_scores, 0, 0.0)
                except Exception:
                    score = 0.0
                
                # Fallback to safe correlation if MI failed
                if score == 0.0:
                    if MATH_VALIDATION_AVAILABLE:
                        correlation = safe_correlation(feature_clean, target_clean, default=0.0)
                    else:
                        try:
                            corr_matrix = np.corrcoef(feature_clean, target_clean)
                            if corr_matrix.shape == (2, 2):
                                correlation = corr_matrix[0, 1]
                                correlation = correlation if np.isfinite(correlation) else 0.0
                            else:
                                correlation = 0.0
                        except Exception:
                            correlation = 0.0
                    score = abs(correlation)
                
                # Validate final score
                if MATH_VALIDATION_AVAILABLE:
                    try:
                        score = validate_finite(score, "optimization_score")
                        score = max(0.0, score)  # Ensure non-negative
                    except Exception:
                        score = 0.0
                
                total_trials += 1
                
                if score > best_score:
                    best_score = score
                    best_lookback = lookback
                    
            except Exception:
                continue
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_lookback_period=best_lookback,
            best_score=best_score,
            optimization_method=self.method_name,
            total_trials=total_trials,
            optimization_time=optimization_time,
            convergence_achieved=total_trials > 0,
            metadata={
                'grid_size': grid_size,
                'lookback_range': (min_lookback, max_lookback)
            }
        )
    
    def validate_inputs(self, 
                       data: pd.DataFrame, 
                       feature_name: str, 
                       target_column: str) -> Tuple[bool, str]:
        """Validate inputs for grid search."""
        if data.empty:
            return False, "Data is empty"
        
        if target_column not in data.columns:
            return False, f"Target column '{target_column}' not found"
        
        min_required_data = self.config.get('max_lookback', OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK) * 3  # Increased for reliability
        if len(data) < min_required_data:
            return False, f"Insufficient data for optimization: {len(data)} < {min_required_data}"
        
        return True, ""
    
    def _generate_feature_with_lookback(self, 
                                      data: pd.DataFrame, 
                                      feature_name: str, 
                                      lookback_period: int) -> np.ndarray:
        """Generate feature values with specific lookback period."""
        if feature_name in data.columns:
            return data[feature_name].rolling(window=lookback_period).mean().values
        else:
            # Default to simple moving average of close price
            return data['close'].rolling(window=lookback_period).mean().values


class OptimizationStrategyFactory:
    """Factory for creating optimization strategies."""
    
    _strategies = {
        OptimizationMethod.GRID_SEARCH: GridSearchStrategy,
        # Add more strategies as they are implemented
    }
    
    @classmethod
    def create_strategy(cls, 
                       method: OptimizationMethod, 
                       config: Optional[Dict[str, Any]] = None) -> OptimizationStrategy:
        """
        Create an optimization strategy.
        
        Args:
            method: Optimization method to use
            config: Configuration for the strategy
            
        Returns:
            OptimizationStrategy instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in cls._strategies:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        strategy_class = cls._strategies[method]
        return strategy_class(config)
    
    @classmethod
    def get_available_methods(cls) -> List[OptimizationMethod]:
        """Get list of available optimization methods."""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, 
                         method: OptimizationMethod, 
                         strategy_class: type) -> None:
        """
        Register a new optimization strategy.
        
        Args:
            method: Optimization method enum
            strategy_class: Strategy class to register
        """
        cls._strategies[method] = strategy_class