#!/usr/bin/env python3
"""
Shared Components for NAS & TAS Systems

This module provides shared components that can be used by both Neural Architecture Search (NAS)
and Tree Architecture Search (TAS) systems, eliminating code duplication and ensuring consistency.

Key Components:
- Shared configuration management
- Common evaluation metrics
- Unified hardware optimization
- Shared search algorithms
- Common data processing utilities
- Shared utility functions
"""

import os
import sys
import time
import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import concurrent.futures
import threading
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# Import utility modules
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, 
        safe_convert_dtypes, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics,
        optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory
    )
    from src.utils.common_utilities import (
        CommonUtilities, safe_dataframe_operation as safe_df_op,
        validate_dataframe_columns as validate_df_cols,
        get_data_summary, safe_convert_dtypes as safe_convert_dt
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_covariance, safe_mean, safe_std,
        MathValidation
    )
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_progress, tprint_performance, tprint_structured,
        tprint_timer, configure_tprint, TPrintConfig, LogLevel
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")
    UTILITY_MODULES_AVAILABLE = False
    # Define fallback functions
    def safe_dataframe_operation(df, operation, *args, **kwargs):
        return operation(df, *args, **kwargs)
    def validate_dataframe_columns(df, required_columns):
        return True
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)
    def tprint_progress(step, total, message="", **kwargs):
        print(f"Progress [{step}/{total}]: {message}")

logger = logging.getLogger(__name__)


# ============================================================================
# SHARED CONFIGURATION MANAGEMENT
# ============================================================================

class SharedConfigManager:
    """Shared configuration manager for both NAS and TAS systems."""
    
    @staticmethod
    def create_base_config(architecture_type: str = "neural") -> Dict[str, Any]:
        """Create base configuration shared by both systems."""
        base_config = {
            # Common search parameters
            'search_strategy': 'random',
            'max_trials': 100,
            'max_epochs': 50,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'cross_validation_folds': 5,
            
            # Common optimization parameters
            'learning_rate_range': (1e-5, 1e-2),
            'batch_size_range': (16, 256),
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            
            # Common hardware optimization
            'enable_hardware_optimization': True,
            'memory_limit_gb': 8.0,
            'parallel_evaluations': 4,
            'enable_m1_optimization': True,
            
            # Common performance settings
            'save_results': True,
            'save_models': True,
            'output_dir': f"{architecture_type}_results",
            'verbose': True,
            
            # Common advanced features
            'enable_regime_awareness': True,
            'enable_uncertainty_quantification': True,
            'enable_meta_learning': True,
            'enable_real_time_adaptation': False,
            
            # Common data processing
            'enable_feature_selection': True,
            'max_features': 100,
            'normalize_data': True,
            'standardize_data': True,
            'handle_missing_values': True,
            'outlier_detection': True
        }
        
        return base_config
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom configuration with base configuration."""
        merged_config = base_config.copy()
        merged_config.update(custom_config)
        return merged_config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration parameters."""
        errors = []
        
        # Validate numeric ranges
        if config.get('max_trials', 0) <= 0:
            errors.append("max_trials must be positive")
        
        if config.get('max_epochs', 0) <= 0:
            errors.append("max_epochs must be positive")
        
        if not (0 < config.get('validation_split', 0) < 1):
            errors.append("validation_split must be between 0 and 1")
        
        if config.get('cross_validation_folds', 0) < 2:
            errors.append("cross_validation_folds must be at least 2")
        
        # Validate learning rate range
        lr_range = config.get('learning_rate_range', (1e-5, 1e-2))
        if not (0 < lr_range[0] < lr_range[1] < 1):
            errors.append("learning_rate_range must be ascending and between 0 and 1")
        
        # Validate batch size range
        batch_range = config.get('batch_size_range', (16, 256))
        if not (0 < batch_range[0] <= batch_range[1]):
            errors.append("batch_size_range must be ascending and positive")
        
        return len(errors) == 0, errors


# ============================================================================
# SHARED EVALUATION METRICS
# ============================================================================

class SharedEvaluationMetrics:
    """Shared evaluation metrics for both NAS and TAS systems."""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic evaluation metrics."""
        metrics = {}
        
        # Classification metrics
        if len(np.unique(y_true)) <= 10:  # Classification
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
                except ValueError:
                    metrics['roc_auc'] = 0.0
        
        # Regression metrics
        else:
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            metrics['r2_score'] = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        return metrics
    
    @staticmethod
    def calculate_trading_metrics(returns: np.ndarray,
                                predictions: np.ndarray,
                                actual_returns: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        metrics = {}
        
        if len(returns) == 0:
            return metrics
        
        # Basic return metrics
        total_return = np.sum(returns)
        metrics['total_return'] = total_return
        metrics['annualized_return'] = total_return * 252 / len(returns) if len(returns) > 0 else 0
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        metrics['volatility'] = volatility
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = (metrics['annualized_return'] - risk_free_rate) / volatility if volatility > 0 else 0
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        metrics['max_drawdown'] = abs(max_drawdown)
        
        # Win rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns[returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        metrics['win_rate'] = win_rate
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        metrics['profit_factor'] = min(profit_factor, 10.0)  # Cap at 10
        
        # Calmar ratio
        calmar_ratio = metrics['annualized_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        metrics['calmar_ratio'] = calmar_ratio
        
        return metrics
    
    @staticmethod
    def calculate_economic_metrics(predictions: np.ndarray,
                                 actual_values: np.ndarray,
                                 market_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Calculate economic significance metrics."""
        metrics = {}
        
        # Information coefficient (correlation between predictions and actual returns)
        if len(predictions) > 1 and len(actual_values) > 1:
            try:
                ic = np.corrcoef(predictions, actual_values)[0, 1]
                metrics['information_coefficient'] = ic if not np.isnan(ic) else 0.0
            except:
                metrics['information_coefficient'] = 0.0
        
        # Hit rate (percentage of correct directional predictions)
        if len(predictions) > 0 and len(actual_values) > 0:
            directional_correct = np.sum(np.sign(predictions) == np.sign(actual_values))
            hit_rate = directional_correct / len(predictions)
            metrics['hit_rate'] = hit_rate
        
        # Economic significance score (combination of IC and hit rate)
        ic_score = abs(metrics.get('information_coefficient', 0))
        hit_rate_score = metrics.get('hit_rate', 0)
        economic_significance = (ic_score * 0.6 + hit_rate_score * 0.4)
        metrics['economic_significance_score'] = economic_significance
        
        # Trading viability score (based on consistency and magnitude)
        if market_data and 'volume' in market_data:
            volume = market_data['volume']
            avg_volume = np.mean(volume)
            volume_consistency = 1 - (np.std(volume) / avg_volume) if avg_volume > 0 else 0
            trading_viability = economic_significance * volume_consistency
            metrics['trading_viability_score'] = trading_viability
        
        return metrics
    
    @staticmethod
    def calculate_model_complexity(model: Any) -> Dict[str, float]:
        """Calculate model complexity metrics."""
        metrics = {}
        
        try:
            # Try to get model parameters
            if hasattr(model, 'n_features_'):
                metrics['n_features'] = model.n_features_
            
            if hasattr(model, 'n_estimators'):
                metrics['n_estimators'] = model.n_estimators
            
            if hasattr(model, 'max_depth'):
                metrics['max_depth'] = model.max_depth
            
            if hasattr(model, 'layers'):
                metrics['n_layers'] = len(model.layers)
            
            # Estimate parameter count
            param_count = 0
            if hasattr(model, 'get_params'):
                params = model.get_params()
                param_count = sum(1 for v in params.values() if v is not None)
            
            metrics['parameter_count'] = param_count
            metrics['complexity_score'] = min(param_count / 1000, 10.0)  # Normalize to 0-10
            
        except Exception as e:
            tprint_warning(f"Could not calculate model complexity: {e}")
            metrics['complexity_score'] = 1.0
        
        return metrics


# ============================================================================
# SHARED HARDWARE OPTIMIZATION
# ============================================================================

class SharedHardwareOptimizer:
    """Shared hardware optimization for both NAS and TAS systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hardware optimizer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if self.config.get('enable_hardware_optimization', True):
            self._initialize_hardware_optimization()
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            if UTILITY_MODULES_AVAILABLE:
                if self.config.get('enable_m1_optimization', True):
                    self.gpu_manager = get_m1_gpu_manager()
                    self.memory_optimizer = get_m1_memory_optimizer()
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    
                    tprint_info("M1 hardware optimization initialized")
                else:
                    tprint_info("Hardware optimization disabled")
            else:
                tprint_warning("Utility modules not available, using fallback optimization")
        except Exception as e:
            tprint_warning(f"Could not initialize hardware optimization: {e}")
    
    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations."""
        if self.gpu_manager:
            try:
                with gpu_context():
                    yield
            except Exception as e:
                tprint_warning(f"GPU context failed: {e}")
                yield
        else:
            yield
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory optimization."""
        if self.memory_optimizer:
            try:
                with memory_checkpoint():
                    yield
            except Exception as e:
                tprint_warning(f"Memory context failed: {e}")
                yield
        else:
            yield
    
    def optimize_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize data for hardware."""
        if self.memory_optimizer and isinstance(data, pd.DataFrame):
            try:
                return optimize_dataframe_dtypes(data)
            except Exception as e:
                tprint_warning(f"Data optimization failed: {e}")
        
        return data
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """Cleanup hardware optimization resources."""
        try:
            if UTILITY_MODULES_AVAILABLE:
                cleanup_m1_optimizers()
                tprint_info("Hardware optimization resources cleaned up")
        except Exception as e:
            tprint_warning(f"Hardware cleanup failed: {e}")


# ============================================================================
# SHARED SEARCH ALGORITHMS
# ============================================================================

class SharedSearchAlgorithms:
    """Shared search algorithms for both NAS and TAS systems."""
    
    @staticmethod
    def random_search(parameter_space: Dict[str, Any], 
                     n_trials: int = 100) -> List[Dict[str, Any]]:
        """Random search algorithm."""
        candidates = []
        
        for trial in range(n_trials):
            candidate = {}
            
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    param_type = param_config.get('type', 'uniform')
                    param_min = param_config.get('min', 0)
                    param_max = param_config.get('max', 1)
                    
                    if param_type == 'uniform':
                        if isinstance(param_min, int) and isinstance(param_max, int):
                            candidate[param_name] = np.random.randint(param_min, param_max + 1)
                        else:
                            candidate[param_name] = np.random.uniform(param_min, param_max)
                    elif param_type == 'choice':
                        choices = param_config.get('choices', [])
                        candidate[param_name] = np.random.choice(choices)
                    elif param_type == 'loguniform':
                        candidate[param_name] = np.exp(np.random.uniform(np.log(param_min), np.log(param_max)))
                else:
                    # Assume it's a list of choices
                    candidate[param_name] = np.random.choice(param_config)
            
            candidates.append(candidate)
        
        return candidates
    
    @staticmethod
    def grid_search(parameter_space: Dict[str, Any], 
                   max_combinations: int = 1000) -> List[Dict[str, Any]]:
        """Grid search algorithm."""
        from sklearn.model_selection import ParameterGrid
        
        # Create parameter grid
        param_grid = {}
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'uniform')
                
                if param_type == 'uniform':
                    param_min = param_config.get('min', 0)
                    param_max = param_config.get('max', 1)
                    n_values = param_config.get('n_values', 5)
                    
                    if isinstance(param_min, int) and isinstance(param_max, int):
                        param_grid[param_name] = np.linspace(param_min, param_max, n_values, dtype=int).tolist()
                    else:
                        param_grid[param_name] = np.linspace(param_min, param_max, n_values).tolist()
                elif param_type == 'choice':
                    param_grid[param_name] = param_config.get('choices', [])
            else:
                param_grid[param_name] = param_config
        
        # Generate grid
        grid = ParameterGrid(param_grid)
        
        # Limit combinations if too many
        if len(grid) > max_combinations:
            grid = list(grid)[:max_combinations]
        
        return list(grid)
    
    @staticmethod
    def evolutionary_search(parameter_space: Dict[str, Any],
                          population_size: int = 50,
                          n_generations: int = 100,
                          mutation_rate: float = 0.1,
                          crossover_rate: float = 0.8) -> List[Dict[str, Any]]:
        """Evolutionary search algorithm."""
        # Initialize population
        population = SharedSearchAlgorithms.random_search(parameter_space, population_size)
        
        # Add fitness scores (placeholder)
        for candidate in population:
            candidate['_fitness'] = np.random.random()
        
        # Evolution loop
        for generation in range(n_generations):
            # Sort by fitness
            population.sort(key=lambda x: x['_fitness'], reverse=True)
            
            # Selection (keep top 50%)
            elite_size = population_size // 2
            elite = population[:elite_size]
            
            # Generate new population
            new_population = elite.copy()
            
            # Crossover and mutation
            while len(new_population) < population_size:
                parent1 = np.random.choice(elite)
                parent2 = np.random.choice(elite)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child = SharedSearchAlgorithms._crossover(parent1, parent2, parameter_space)
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = SharedSearchAlgorithms._mutate(child, parameter_space)
                
                child['_fitness'] = np.random.random()  # Placeholder fitness
                new_population.append(child)
            
            population = new_population
        
        # Remove fitness scores and return
        for candidate in population:
            candidate.pop('_fitness', None)
        
        return population
    
    @staticmethod
    def _crossover(parent1: Dict[str, Any], 
                  parent2: Dict[str, Any],
                  parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}
        
        for param_name in parameter_space.keys():
            if np.random.random() < 0.5:
                child[param_name] = parent1.get(param_name)
            else:
                child[param_name] = parent2.get(param_name)
        
        return child
    
    @staticmethod
    def _mutate(candidate: Dict[str, Any], 
               parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mutation on a candidate."""
        mutated = candidate.copy()
        
        # Mutate a random parameter
        param_name = np.random.choice(list(parameter_space.keys()))
        param_config = parameter_space[param_name]
        
        if isinstance(param_config, dict):
            param_type = param_config.get('type', 'uniform')
            param_min = param_config.get('min', 0)
            param_max = param_config.get('max', 1)
            
            if param_type == 'uniform':
                if isinstance(param_min, int) and isinstance(param_max, int):
                    mutated[param_name] = np.random.randint(param_min, param_max + 1)
                else:
                    mutated[param_name] = np.random.uniform(param_min, param_max)
            elif param_type == 'choice':
                choices = param_config.get('choices', [])
                mutated[param_name] = np.random.choice(choices)
        else:
            mutated[param_name] = np.random.choice(param_config)
        
        return mutated


# ============================================================================
# SHARED DATA PROCESSING
# ============================================================================

class SharedDataProcessor:
    """Shared data processing utilities for both NAS and TAS systems."""
    
    @staticmethod
    def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate input data."""
        errors = []
        
        # Check data types
        if not isinstance(X, np.ndarray):
            errors.append("X must be a numpy array")
        
        if not isinstance(y, np.ndarray):
            errors.append("y must be a numpy array")
        
        if len(errors) > 0:
            return False, errors
        
        # Check shapes
        if X.shape[0] != y.shape[0]:
            errors.append("X and y must have the same number of samples")
        
        # Check for missing values
        if np.any(np.isnan(X)):
            errors.append("X contains NaN values")
        
        if np.any(np.isnan(y)):
            errors.append("y contains NaN values")
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            errors.append("X contains infinite values")
        
        if np.any(np.isinf(y)):
            errors.append("y contains infinite values")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def preprocess_data(X: np.ndarray, 
                       y: np.ndarray,
                       config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data based on configuration."""
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values
        if config.get('handle_missing_values', True):
            X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
            y_processed = np.nan_to_num(y_processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize data
        if config.get('normalize_data', True):
            X_processed = (X_processed - np.min(X_processed)) / (np.max(X_processed) - np.min(X_processed) + 1e-8)
        
        # Standardize data
        if config.get('standardize_data', True):
            X_processed = (X_processed - np.mean(X_processed, axis=0)) / (np.std(X_processed, axis=0) + 1e-8)
        
        # Outlier detection and removal
        if config.get('outlier_detection', True):
            X_processed, y_processed = SharedDataProcessor._remove_outliers(X_processed, y_processed)
        
        return X_processed, y_processed
    
    @staticmethod
    def _remove_outliers(X: np.ndarray, y: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using z-score method."""
        try:
            z_scores = np.abs(stats.zscore(X))
            outlier_mask = np.all(z_scores < threshold, axis=1)
            
            X_filtered = X[outlier_mask]
            y_filtered = y[outlier_mask]
            
            return X_filtered, y_filtered
        except Exception as e:
            tprint_warning(f"Outlier removal failed: {e}")
            return X, y
    
    @staticmethod
    def split_data(X: np.ndarray, 
                  y: np.ndarray,
                  config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        validation_split = config.get('validation_split', 0.2)
        
        if validation_split <= 0 or validation_split >= 1:
            return X, np.array([]), y, np.array([])
        
        # For time series data, use the last portion as validation
        if config.get('time_series_split', False):
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            # Random split
            indices = np.random.permutation(len(X))
            split_idx = int(len(X) * (1 - validation_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
        
        return X_train, X_val, y_train, y_val
    
    @staticmethod
    def feature_selection(X: np.ndarray, 
                         y: np.ndarray,
                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature selection."""
        if not config.get('enable_feature_selection', True):
            return X, np.arange(X.shape[1])
        
        max_features = config.get('max_features', 100)
        
        if X.shape[1] <= max_features:
            return X, np.arange(X.shape[1])
        
        try:
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            
            # Determine if classification or regression
            n_classes = len(np.unique(y))
            
            if n_classes <= 10:  # Classification
                selector = SelectKBest(score_func=f_classif, k=max_features)
            else:  # Regression
                selector = SelectKBest(score_func=f_regression, k=max_features)
            
            X_selected = selector.fit_transform(X, y)
            selected_features = selector.get_support(indices=True)
            
            return X_selected, selected_features
            
        except ImportError:
            tprint_warning("sklearn not available for feature selection, using random selection")
            # Random feature selection as fallback
            n_features = min(max_features, X.shape[1])
            selected_features = np.random.choice(X.shape[1], n_features, replace=False)
            return X[:, selected_features], selected_features


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

class SharedUtilities:
    """Shared utility functions for both NAS and TAS systems."""
    
    @staticmethod
    def create_sample_data(n_samples: int = 1000, 
                          n_features: int = 20,
                          n_classes: int = 2,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(random_state)
        
        if n_classes <= 10:  # Classification
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, n_classes, n_samples)
        else:  # Regression
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)
        
        return X, y
    
    @staticmethod
    def save_results(results: Dict[str, Any], 
                    filepath: Union[str, Path],
                    format: str = 'json') -> bool:
        """Save results to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)
            else:
                tprint_error(f"Unsupported format: {format}")
                return False
            
            tprint_success(f"Results saved to {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to save results: {e}")
            return False
    
    @staticmethod
    def load_results(filepath: Union[str, Path],
                    format: str = 'json') -> Optional[Dict[str, Any]]:
        """Load results from file."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                tprint_error(f"File not found: {filepath}")
                return None
            
            if format.lower() == 'json':
                with open(filepath, 'r') as f:
                    results = json.load(f)
            elif format.lower() == 'pickle':
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)
            else:
                tprint_error(f"Unsupported format: {format}")
                return None
            
            tprint_success(f"Results loaded from {filepath}")
            return results
            
        except Exception as e:
            tprint_error(f"Failed to load results: {e}")
            return None
    
    @staticmethod
    def calculate_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Calculate execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    
    @staticmethod
    def create_progress_bar(current: int, total: int, width: int = 50) -> str:
        """Create a text progress bar."""
        progress = current / total
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        percentage = progress * 100
        return f"[{bar}] {percentage:.1f}% ({current}/{total})"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of shared components
    
    # Create sample data
    X, y = SharedUtilities.create_sample_data(n_samples=1000, n_features=20, n_classes=2)
    
    # Create configuration
    base_config = SharedConfigManager.create_base_config("neural")
    custom_config = {
        'max_trials': 50,
        'validation_split': 0.2,
        'enable_feature_selection': True,
        'max_features': 10
    }
    config = SharedConfigManager.merge_configs(base_config, custom_config)
    
    # Validate configuration
    is_valid, errors = SharedConfigManager.validate_config(config)
    if not is_valid:
        print("Configuration errors:", errors)
    else:
        print("Configuration is valid")
    
    # Validate data
    is_valid, errors = SharedDataProcessor.validate_data(X, y)
    if not is_valid:
        print("Data validation errors:", errors)
    else:
        print("Data is valid")
    
    # Preprocess data
    X_processed, y_processed = SharedDataProcessor.preprocess_data(X, y, config)
    print(f"Original shape: {X.shape}, Processed shape: {X_processed.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = SharedDataProcessor.split_data(X_processed, y_processed, config)
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Feature selection
    X_selected, selected_features = SharedDataProcessor.feature_selection(X_train, y_train, config)
    print(f"Selected features: {len(selected_features)}")
    
    # Calculate metrics
    dummy_predictions = np.random.randint(0, 2, len(y_val))
    basic_metrics = SharedEvaluationMetrics.calculate_basic_metrics(y_val, dummy_predictions)
    print("Basic metrics:", basic_metrics)
    
    # Hardware optimization
    hardware_optimizer = SharedHardwareOptimizer(config)
    X_optimized = hardware_optimizer.optimize_data(X_selected)
    print(f"Memory usage: {hardware_optimizer.get_memory_usage():.2f} MB")
    
    # Search algorithms
    parameter_space = {
        'learning_rate': {'type': 'loguniform', 'min': 1e-5, 'max': 1e-2},
        'batch_size': {'type': 'uniform', 'min': 16, 'max': 256, 'n_values': 5},
        'activation': {'type': 'choice', 'choices': ['relu', 'tanh', 'swish']}
    }
    
    random_candidates = SharedSearchAlgorithms.random_search(parameter_space, n_trials=10)
    print(f"Generated {len(random_candidates)} random candidates")
    
    grid_candidates = SharedSearchAlgorithms.grid_search(parameter_space, max_combinations=50)
    print(f"Generated {len(grid_candidates)} grid candidates")
    
    # Save results
    results = {
        'config': config,
        'data_shape': X.shape,
        'processed_shape': X_processed.shape,
        'selected_features': len(selected_features),
        'basic_metrics': basic_metrics,
        'candidates_generated': len(random_candidates) + len(grid_candidates)
    }
    
    SharedUtilities.save_results(results, "shared_components_test_results.json")
    
    print("\n" + "="*50)
    print("SHARED COMPONENTS TEST COMPLETED SUCCESSFULLY")
    print("="*50)