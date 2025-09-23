"""
Regime-Specific Hyperparameter Optimization Framework

This module implements a comprehensive regime-specific hyperparameter optimization system
that adapts model hyperparameters based on current market conditions.

Key Features:
1. Regime-specific hyperparameter spaces
2. Dynamic optimization based on regime characteristics
3. Multi-objective optimization for different metrics
4. Bayesian optimization with regime awareness
5. Adaptive search space based on regime transitions
6. Uncertainty quantification for optimization results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
import logging
import time
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("⚠️ scikit-optimize not available, using fallback optimization")


@dataclass
class RegimeHyperparameterConfig:
    """Configuration for regime-specific hyperparameters."""

    regime_id: str
    regime_type: str  # 'high_volatility', 'trending', 'mean_reverting', 'low_volatility'
    base_hyperparams: Dict[str, Any]
    regime_specific_ranges: Dict[str, Any]
    optimization_objectives: List[str] = field(default_factory=lambda: ['mse', 'mae'])
    optimization_weights: Dict[str, float] = field(default_factory=lambda: {'mse': 0.7, 'mae': 0.3})
    search_iterations: int = 50
    cross_validation_folds: int = 5
    random_state: int = 42
    adaptive_ranges: bool = True
    uncertainty_threshold: float = 0.1


class RegimeCharacteristicsAnalyzer:
    """Analyzes regime characteristics to inform hyperparameter optimization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime characteristics analyzer."""
        self.config = config or {}
        self.regime_profiles = self._load_regime_profiles()

    def _load_regime_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined regime profiles with optimal hyperparameter ranges."""
        return {
            'high_volatility': {
                'description': 'High price variance, unpredictable movements',
                'characteristics': {
                    'volatility_level': 'high',
                    'trend_strength': 'weak',
                    'noise_level': 'high',
                    'regime_persistence': 'low',
                    'market_efficiency': 'low'
                },
                'optimal_hyperparams': {
                    'learning_rate': {'low': 1e-5, 'high': 1e-3},
                    'batch_size': {'low': 16, 'high': 64},
                    'dropout': {'low': 0.3, 'high': 0.5},
                    'l2_regularization': {'low': 1e-4, 'high': 1e-2},
                    'n_layers': {'low': 2, 'high': 4},
                    'n_units': {'low': 32, 'high': 128},
                    'early_stopping_patience': {'low': 5, 'high': 15}
                },
                'optimization_priorities': ['robustness', 'stability', 'uncertainty'],
                'risk_tolerance': 'conservative'
            },
            'trending': {
                'description': 'Strong directional movement with momentum',
                'characteristics': {
                    'volatility_level': 'medium',
                    'trend_strength': 'strong',
                    'noise_level': 'medium',
                    'regime_persistence': 'high',
                    'market_efficiency': 'medium'
                },
                'optimal_hyperparams': {
                    'learning_rate': {'low': 1e-4, 'high': 1e-2},
                    'batch_size': {'low': 64, 'high': 256},
                    'dropout': {'low': 0.1, 'high': 0.3},
                    'l2_regularization': {'low': 1e-5, 'high': 1e-3},
                    'n_layers': {'low': 4, 'high': 8},
                    'n_units': {'low': 128, 'high': 512},
                    'early_stopping_patience': {'low': 15, 'high': 30}
                },
                'optimization_priorities': ['accuracy', 'trend_tracking', 'momentum'],
                'risk_tolerance': 'moderate'
            },
            'mean_reverting': {
                'description': 'Price oscillates around mean with reversion patterns',
                'characteristics': {
                    'volatility_level': 'medium',
                    'trend_strength': 'weak',
                    'noise_level': 'medium',
                    'regime_persistence': 'medium',
                    'market_efficiency': 'high'
                },
                'optimal_hyperparams': {
                    'learning_rate': {'low': 1e-4, 'high': 5e-3},
                    'batch_size': {'low': 32, 'high': 128},
                    'dropout': {'low': 0.2, 'high': 0.4},
                    'l2_regularization': {'low': 1e-4, 'high': 5e-3},
                    'n_layers': {'low': 3, 'high': 6},
                    'n_units': {'low': 64, 'high': 256},
                    'early_stopping_patience': {'low': 10, 'high': 25}
                },
                'optimization_priorities': ['precision', 'reversion_timing', 'mean_accuracy'],
                'risk_tolerance': 'balanced'
            },
            'low_volatility': {
                'description': 'Stable, low variance environment with predictable patterns',
                'characteristics': {
                    'volatility_level': 'low',
                    'trend_strength': 'medium',
                    'noise_level': 'low',
                    'regime_persistence': 'high',
                    'market_efficiency': 'high'
                },
                'optimal_hyperparams': {
                    'learning_rate': {'low': 1e-4, 'high': 1e-3},
                    'batch_size': {'low': 128, 'high': 512},
                    'dropout': {'low': 0.05, 'high': 0.2},
                    'l2_regularization': {'low': 1e-6, 'high': 1e-4},
                    'n_layers': {'low': 5, 'high': 10},
                    'n_units': {'low': 256, 'high': 1024},
                    'early_stopping_patience': {'low': 20, 'high': 50}
                },
                'optimization_priorities': ['accuracy', 'efficiency', 'complexity'],
                'risk_tolerance': 'aggressive'
            }
        }

    def analyze_regime_characteristics(self, regime_data: pd.DataFrame,
                                     regime_labels: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Analyze characteristics of each regime from data."""
        regime_stats = {}

        for regime_id in np.unique(regime_labels):
            regime_mask = regime_labels == regime_id
            regime_subset = regime_data[regime_mask]

            if len(regime_subset) < 10:  # Skip if insufficient data
                continue

            # Calculate regime statistics
            volatility = regime_subset.select_dtypes(include=[np.number]).std().mean()
            trend_strength = self._calculate_trend_strength(regime_subset)
            noise_level = self._calculate_noise_level(regime_subset)
            persistence = self._calculate_regime_persistence(regime_labels, regime_id)

            regime_stats[regime_id] = {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'noise_level': noise_level,
                'persistence': persistence,
                'sample_size': len(regime_subset),
                'data_quality': 'good' if len(regime_subset) > 100 else 'fair' if len(regime_subset) > 50 else 'poor'
            }

        return regime_stats

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using autocorrelation."""
        try:
            # Use price-like columns for trend calculation
            price_cols = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
            if not price_cols:
                # Fallback to first numeric column
                price_cols = data.select_dtypes(include=[np.number]).columns[:1]

            if price_cols:
                price_data = data[price_cols[0]].values
                # Calculate autocorrelation at lag 1
                autocorr = np.corrcoef(price_data[:-1], price_data[1:])[0, 1]
                return abs(autocorr)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_noise_level(self, data: pd.DataFrame) -> float:
        """Calculate noise level using signal-to-noise ratio."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 1.0

            # Calculate variance of differences (noise)
            noise = numeric_data.diff().var().mean()

            # Calculate variance of signal
            signal = numeric_data.var().mean()

            if signal > 0:
                return noise / signal
            return 1.0
        except Exception:
            return 1.0

    def _calculate_regime_persistence(self, regime_labels: np.ndarray, regime_id: str) -> float:
        """Calculate how persistent a regime is."""
        try:
            regime_mask = regime_labels == regime_id
            # Calculate average run length
            run_lengths = []
            current_run = 0

            for label in regime_mask:
                if label:
                    current_run += 1
                else:
                    if current_run > 0:
                        run_lengths.append(current_run)
                        current_run = 0

            if current_run > 0:
                run_lengths.append(current_run)

            if run_lengths:
                return np.mean(run_lengths)
            return 1.0
        except Exception:
            return 1.0

    def get_regime_specific_ranges(self, regime_id: str, regime_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get regime-specific hyperparameter ranges based on regime characteristics."""
        if regime_id not in self.regime_profiles:
            return {}

        profile = self.regime_profiles[regime_id]
        base_ranges = profile['optimal_hyperparams']

        # Adapt ranges based on actual regime statistics
        adapted_ranges = {}

        for param, range_dict in base_ranges.items():
            base_low = range_dict.get('low', 0)
            base_high = range_dict.get('high', 1)

            # Adapt based on regime characteristics
            if regime_id in regime_stats:
                stats = regime_stats[regime_id]

                # Adjust for volatility
                if stats['volatility'] > 0.1:  # High volatility
                    if param == 'learning_rate':
                        base_high *= 0.1  # Lower learning rate for high volatility
                    elif param == 'batch_size':
                        base_high *= 0.5  # Smaller batches for high volatility
                    elif param == 'dropout':
                        base_low = max(base_low, 0.3)  # Higher dropout

                # Adjust for trend strength
                if stats['trend_strength'] > 0.7:  # Strong trend
                    if param == 'n_layers':
                        base_high = min(base_high, 8)  # Limit layers for strong trends
                    elif param == 'early_stopping_patience':
                        base_high *= 1.5  # More patience for stable trends

                # Adjust for noise level
                if stats['noise_level'] > 0.5:  # High noise
                    if param == 'l2_regularization':
                        base_low = max(base_low, 1e-3)  # Higher regularization
                    elif param == 'dropout':
                        base_low = max(base_low, 0.4)  # Higher dropout

            adapted_ranges[param] = {
                'low': base_low,
                'high': base_high,
                'type': 'real' if isinstance(base_low, float) else 'integer'
            }

        return adapted_ranges


class RegimeSpecificHyperparameterOptimizer:
    """Main optimizer for regime-specific hyperparameters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime-specific hyperparameter optimizer."""
        self.config = config or {}
        self.regime_analyzer = RegimeCharacteristicsAnalyzer(config)
        self.optimization_history = {}
        self.best_params_per_regime = {}

        # Check for optimization libraries
        if not SKOPT_AVAILABLE:
            logger.warning("⚠️ scikit-optimize not available, using random search")
            self.use_bayesian_optimization = False
        else:
            self.use_bayesian_optimization = True

    def create_optimization_config(self, regime_id: str, model_type: str,
                                 regime_stats: Optional[Dict[str, Any]] = None) -> RegimeHyperparameterConfig:
        """Create optimization configuration for a specific regime."""

        # Base hyperparameter ranges by model type
        base_hyperparams = self._get_base_hyperparams(model_type)

        # Get regime-specific ranges
        regime_ranges = self.regime_analyzer.get_regime_specific_ranges(regime_id, regime_stats or {})

        # Create configuration
        config = RegimeHyperparameterConfig(
            regime_id=regime_id,
            regime_type=self._get_regime_type(regime_id),
            base_hyperparams=base_hyperparams,
            regime_specific_ranges=regime_ranges,
            optimization_objectives=self._get_optimization_objectives(regime_id),
            optimization_weights=self._get_optimization_weights(regime_id),
            search_iterations=self.config.get('search_iterations', 50),
            cross_validation_folds=self.config.get('cv_folds', 5),
            random_state=self.config.get('random_state', 42),
            adaptive_ranges=self.config.get('adaptive_ranges', True),
            uncertainty_threshold=self.config.get('uncertainty_threshold', 0.1)
        )

        return config

    def _get_base_hyperparams(self, model_type: str) -> Dict[str, Any]:
        """Get base hyperparameters for different model types."""
        base_params = {
            'neural_network': {
                'learning_rate': 1e-3,
                'batch_size': 64,
                'epochs': 100,
                'dropout': 0.2,
                'l2_regularization': 1e-4,
                'n_layers': 4,
                'n_units': 128,
                'activation': 'relu',
                'optimizer': 'adam'
            },
            'xgboost': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.0
            },
            'lightgbm': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.0
            },
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'random_strength': 1.0
            }
        }

        return base_params.get(model_type, base_params['neural_network'])

    def _get_regime_type(self, regime_id: str) -> str:
        """Map regime ID to regime type."""
        regime_mapping = {
            'high_volatility': ['volatile', 'high_vol', 'crisis', 'extreme'],
            'trending': ['trending', 'bullish', 'bearish', 'momentum'],
            'mean_reverting': ['reverting', 'oscillating', 'range_bound'],
            'low_volatility': ['calm', 'stable', 'low_vol', 'sideways']
        }

        for regime_type, keywords in regime_mapping.items():
            if any(keyword in regime_id.lower() for keyword in keywords):
                return regime_type

        return 'unknown'

    def _get_optimization_objectives(self, regime_id: str) -> List[str]:
        """Get optimization objectives for a regime."""
        objectives = ['mse', 'mae']  # Default objectives

        regime_type = self._get_regime_type(regime_id)
        profile = self.regime_analyzer.regime_profiles.get(regime_type, {})

        if 'optimization_priorities' in profile:
            priorities = profile['optimization_priorities']

            if 'robustness' in priorities:
                objectives.append('robustness')
            if 'uncertainty' in priorities:
                objectives.append('uncertainty')
            if 'trend_tracking' in priorities:
                objectives.extend(['trend_accuracy', 'momentum'])

        return list(set(objectives))  # Remove duplicates

    def _get_optimization_weights(self, regime_id: str) -> Dict[str, float]:
        """Get optimization weights for a regime."""
        base_weights = {'mse': 0.7, 'mae': 0.3}

        regime_type = self._get_regime_type(regime_id)
        profile = self.regime_analyzer.regime_profiles.get(regime_type, {})

        if 'optimization_priorities' in profile:
            priorities = profile['optimization_priorities']

            if 'robustness' in priorities:
                base_weights['mse'] = 0.8  # Higher weight on accuracy for robustness
                base_weights['robustness'] = 0.2
            if 'uncertainty' in priorities:
                base_weights['uncertainty'] = 0.2
                base_weights['mse'] = 0.6  # Reduce accuracy weight for uncertainty focus

        return base_weights

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                               model_factory: Callable,
                               regime_config: RegimeHyperparameterConfig,
                               X_val: Optional[np.ndarray] = None,
                               y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific regime."""

        logger.info(f"🔬 Optimizing hyperparameters for regime: {regime_config.regime_id}")

        # Create search space
        search_space = self._create_search_space(regime_config)

        if not search_space:
            logger.warning(f"⚠️ No valid search space for regime {regime_config.regime_id}")
            return regime_config.base_hyperparams.copy()

        # Define objective function
        @use_named_args(search_space) if self.use_bayesian_optimization else None
        def objective_function(**params):
            return self._evaluate_hyperparameters(
                params, X, y, model_factory, regime_config, X_val, y_val
            )

        # Perform optimization
        if self.use_bayesian_optimization:
            result = gp_minimize(
                objective_function,
                search_space,
                n_calls=regime_config.search_iterations,
                random_state=regime_config.random_state,
                n_jobs=1,  # Sequential for reproducibility
                verbose=True
            )

            # Get best parameters
            best_params = dict(zip([dim.name for dim in search_space], result.x))
            best_score = result.fun

            logger.info(f"✅ Bayesian optimization completed for regime {regime_config.regime_id}")
            logger.info(f"   Best score: {best_score:.6f}")
            logger.info(f"   Best params: {best_params}")

        else:
            # Fallback to random search
            best_params, best_score = self._random_search_optimization(
                objective_function, search_space, regime_config
            )

        # Store optimization results
        optimization_result = {
            'regime_id': regime_config.regime_id,
            'best_params': best_params,
            'best_score': best_score,
            'optimization_method': 'bayesian' if self.use_bayesian_optimization else 'random_search',
            'search_iterations': regime_config.search_iterations,
            'timestamp': time.time()
        }

        self.optimization_history[regime_config.regime_id] = optimization_result
        self.best_params_per_regime[regime_config.regime_id] = best_params

        return best_params

    def _create_search_space(self, regime_config: RegimeHyperparameterConfig) -> List:
        """Create search space for optimization."""
        search_space = []

        try:
            for param, ranges in regime_config.regime_specific_ranges.items():
                if ranges['type'] == 'real':
                    search_space.append(
                        Real(low=ranges['low'], high=ranges['high'], name=param)
                    )
                elif ranges['type'] == 'integer':
                    search_space.append(
                        Integer(low=int(ranges['low']), high=int(ranges['high']), name=param)
                    )
                elif ranges['type'] == 'categorical':
                    search_space.append(
                        Categorical(ranges['choices'], name=param)
                    )

            return search_space
        except Exception as e:
            logger.error(f"❌ Failed to create search space: {e}")
            return []

    def _evaluate_hyperparameters(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                                model_factory: Callable, regime_config: RegimeHyperparameterConfig,
                                X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> float:
        """Evaluate a set of hyperparameters."""
        try:
            # Create model with current parameters
            model_params = regime_config.base_hyperparams.copy()
            model_params.update(params)

            model = model_factory(**model_params)

            # Evaluate using cross-validation
            if X_val is not None and y_val is not None:
                # Use validation set
                model.fit(X, y)
                predictions = model.predict(X_val)
                mse = mean_squared_error(y_val, predictions)
            else:
                # Use cross-validation
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=KFold(n_splits=regime_config.cross_validation_folds, shuffle=True,
                           random_state=regime_config.random_state),
                    scoring='neg_mean_squared_error'
                )
                mse = -cv_scores.mean()

            # Calculate weighted objective
            objective_score = mse * regime_config.optimization_weights.get('mse', 1.0)

            # Add other objectives if available
            if 'mae' in regime_config.optimization_weights:
                if X_val is not None and y_val is not None:
                    mae = np.mean(np.abs(y_val - predictions))
                else:
                    mae = np.mean(np.abs(y - model.predict(X)))
                objective_score += mae * regime_config.optimization_weights['mae']

            return objective_score

        except Exception as e:
            logger.warning(f"⚠️ Error evaluating hyperparameters: {e}")
            return float('inf')  # Bad score for failed evaluation

    def _random_search_optimization(self, objective_function: Callable,
                                  search_space: List, regime_config: RegimeHyperparameterConfig) -> Tuple[Dict[str, Any], float]:
        """Perform random search optimization as fallback."""
        logger.info(f"🔄 Performing random search optimization for regime {regime_config.regime_id}")

        best_score = float('inf')
        best_params = None

        for i in range(regime_config.search_iterations):
            # Sample random parameters
            params = {}
            for dim in search_space:
                if hasattr(dim, 'rvs'):
                    # scikit-optimize dimension
                    params[dim.name] = dim.rvs(random_state=regime_config.random_state + i)
                else:
                    # Fallback random sampling
                    if dim.name in regime_config.regime_specific_ranges:
                        ranges = regime_config.regime_specific_ranges[dim.name]
                        if ranges['type'] == 'real':
                            params[dim.name] = np.random.uniform(ranges['low'], ranges['high'])
                        elif ranges['type'] == 'integer':
                            params[dim.name] = np.random.randint(int(ranges['low']), int(ranges['high']) + 1)

            # Evaluate
            score = objective_function(**params)

            if score < best_score:
                best_score = score
                best_params = params

            if (i + 1) % 10 == 0:
                logger.info(f"Random search iteration {i + 1}/{regime_config.search_iterations}, "
                          f"best score: {best_score:.6f}")

        return best_params, best_score

    def get_regime_optimized_params(self, regime_id: str) -> Dict[str, Any]:
        """Get the best parameters for a specific regime."""
        return self.best_params_per_regime.get(regime_id, {})

    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file."""
        try:
            results = {
                'optimization_history': self.optimization_history,
                'best_params_per_regime': self.best_params_per_regime,
                'timestamp': time.time()
            }

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"💾 Optimization results saved to {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save optimization results: {e}")

    def load_optimization_results(self, filepath: str) -> None:
        """Load optimization results from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results = json.load(f)

                self.optimization_history = results.get('optimization_history', {})
                self.best_params_per_regime = results.get('best_params_per_regime', {})

                logger.info(f"📂 Optimization results loaded from {filepath}")
            else:
                logger.warning(f"⚠️ Optimization results file not found: {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to load optimization results: {e}")


# Factory functions for creating optimizers
def create_regime_hyperparam_optimizer(config: Dict[str, Any]) -> RegimeSpecificHyperparameterOptimizer:
    """Create regime-specific hyperparameter optimizer."""
    return RegimeSpecificHyperparameterOptimizer(config)


def optimize_model_hyperparameters(X: np.ndarray, y: np.ndarray,
                                 model_factory: Callable,
                                 regime_id: str,
                                 model_type: str = 'neural_network',
                                 regime_stats: Optional[Dict[str, Any]] = None,
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimize hyperparameters for a specific regime and model type."""
    optimizer = create_regime_hyperparam_optimizer(config or {})

    regime_config = optimizer.create_optimization_config(regime_id, model_type, regime_stats)

    best_params = optimizer.optimize_hyperparameters(
        X, y, model_factory, regime_config
    )

    return best_params


# Example usage and configuration
def get_example_optimization_config() -> Dict[str, Any]:
    """Get example configuration for regime-specific hyperparameter optimization."""
    return {
        'search_iterations': 30,
        'cv_folds': 3,
        'random_state': 42,
        'adaptive_ranges': True,
        'uncertainty_threshold': 0.15,
        'optimization_method': 'bayesian',  # or 'random_search'
        'save_results': True,
        'results_filepath': 'regime_optimization_results.json'
    }