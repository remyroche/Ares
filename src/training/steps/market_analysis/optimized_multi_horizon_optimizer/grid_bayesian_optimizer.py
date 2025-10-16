"""
Grid-Bayesian Optimizer using ml_commons utilities

This module implements a two-stage optimization approach:
1. Coarse grid search to find promising regions
2. Fine grid search around best results
3. Bayesian TPE optimization for final refinement

Uses ml_commons utilities extensively for grid search, HPO, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import time

# Import ml_commons utilities
from src.utils.ml_common.optimization.grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
from src.utils.ml_common.validation.cv_utils import CrossValidationUtilities
from src.utils.ml_common.validation.temporal_cross_validation import TemporalCrossValidator
from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem

# Import optimization configuration
from .optimization_config import (
    OptimizationConfig, GridSearchConfig, BayesianTPEConfig,
    SearchSpace, OptimizationResult, ModelType
)

# Import multi-horizon components
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

logger = logging.getLogger(__name__)

class GridBayesianOptimizer:
    """
    Two-stage optimizer using ml_commons utilities.

    Stage 1: Coarse grid search to identify promising regions
    Stage 2: Fine grid search around best coarse results
    Stage 3: Bayesian TPE optimization for final refinement
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize grid-bayesian optimizer."""
        self.config = config
        self.logger = logging.getLogger('GridBayesianOptimizer')

        # Initialize ml_commons utilities
        self._initialize_ml_commons_utilities()

        # Optimization state
        self.optimization_history = []
        self.best_coarse_result = None
        self.best_fine_result = None
        self.best_bayesian_result = None

        # Performance optimization: Add caching for expensive computations
        self._evaluation_cache = {}
        self._cache_max_size = 1000

        self.logger.info(f'🔧 Grid-Bayesian Optimizer initialized for {config.model_type.value}')

    def _initialize_ml_commons_utilities(self):
        """Initialize ml_commons utilities."""
        try:
            # Initialize HPO utility
            hpo_config = {
                'enable_parallel': self.config.bayesian_tpe_config.enable_parallel,
                'max_workers': self.config.bayesian_tpe_config.max_workers,
                'enable_monitoring': True,
                'use_nonlinear_optimization': True
            }
            self.hpo_optimizer = HyperparameterOptimization(hpo_config)

            # Initialize cross-validation utility
            cv_config = {
                'initial_train_size': 0.6,
                'step_size': 0.1,
                'min_test_size': 0.1
            }
            self.cv_utilities = CrossValidationUtilities(cv_config)

            # Initialize temporal cross-validator
            self.temporal_cv = TemporalCrossValidator(
                n_splits=self.config.validation_config.cv_folds,
                gap=1  # 1 period gap to prevent lookahead
            )

            # Initialize unified validation system
            self.validation_system = UnifiedValidationSystem()

            self.logger.info('✅ ml_commons utilities initialized successfully')

        except Exception as e:
            self.logger.error(f'❌ Failed to initialize ml_commons utilities: {e}')
            raise RuntimeError(f"Failed to initialize ml_commons utilities: {e}")

    def optimize(self, market_data: pd.DataFrame, model_type: ModelType) -> OptimizationResult:
        """
        Run complete grid-bayesian optimization.

        Args:
            market_data: Market data for optimization
            model_type: Type of model to optimize for

        Returns:
            OptimizationResult with optimal configuration
        """
        self.logger.info(f'🎯 Starting grid-bayesian optimization for {model_type.value}')
        start_time = datetime.now()

        try:
            # Stage 1: Coarse grid search
            if self.config.grid_search_config.coarse_enabled:
                self.logger.info('   → Stage 1: Coarse grid search')
                coarse_result = self._coarse_grid_search(market_data, model_type)
                self.best_coarse_result = coarse_result
                self.logger.info(f'   → Coarse grid best score: {coarse_result.optimization_score:.3f}')

            # Stage 2: Fine grid search
            if self.config.grid_search_config.fine_enabled and self.best_coarse_result:
                self.logger.info('   → Stage 2: Fine grid search')
                fine_result = self._fine_grid_search(market_data, model_type, self.best_coarse_result)
                self.best_fine_result = fine_result
                self.logger.info(f'   → Fine grid best score: {fine_result.optimization_score:.3f}')

            # Stage 3: Bayesian TPE optimization
            if self.config.optimization_method in [OptimizationMethod.BAYESIAN_TPE, OptimizationMethod.GRID_BAYESIAN]:
                self.logger.info('   → Stage 3: Bayesian TPE optimization')
                bayesian_result = self._bayesian_tpe_optimization(market_data, model_type)
                self.best_bayesian_result = bayesian_result
                self.logger.info(f'   → Bayesian TPE best score: {bayesian_result.optimization_score:.3f}')

            # Select best result
            best_result = self._select_best_result()

            # Calculate total optimization time
            total_time = (datetime.now() - start_time).total_seconds()
            best_result.optimization_time = total_time

            # Store in history
            self.optimization_history.append(best_result)

            self.logger.info(f'✅ Grid-bayesian optimization completed in {total_time:.2f}s')
            self.logger.info(f'   → Final score: {best_result.optimization_score:.3f}')
            self.logger.info(f'   → Optimal horizons: {best_result.optimal_horizons}')
            self.logger.info(f'   → Optimal targets: {best_result.optimal_targets}')

            return best_result

        except Exception as e:
            self.logger.error(f'❌ Grid-bayesian optimization failed: {e}')
            if self.config.fast_fail_on_optimization_failure:
                raise RuntimeError(f"Grid-bayesian optimization failed: {e}")
            else:
                # Return best available result
                return self._select_best_result() or self._create_fallback_result(model_type)

    def _coarse_grid_search(self, market_data: pd.DataFrame, model_type: ModelType) -> OptimizationResult:
        """Run coarse grid search using ml_commons grid utilities."""
        self.logger.info('   → Running coarse grid search...')

        # Create search space
        search_space = self._create_search_space(model_type)

        # Build coarse grid using ml_commons
        coarse_grid = build_coarse_grid_from_search_space(
            search_space,
            self.config.grid_search_config.coarse_grid_points
        )

        self.logger.info(f'   → Generated {len(coarse_grid)} coarse grid points')

        # Evaluate grid points
        best_score = -np.inf
        best_params = None
        best_metrics = {}

        for i, params in enumerate(coarse_grid):
            try:
                # Convert params to MultiHorizonConfig
                config = self._params_to_config(params, model_type)

                # Evaluate configuration
                score, metrics = self._evaluate_configuration(config, market_data, model_type)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics

                if i % 10 == 0:
                    self.logger.info(f'   → Coarse grid point {i+1}/{len(coarse_grid)}: Score: {score:.3f}')

            except Exception as e:
                self.logger.warning(f'⚠️ Error evaluating coarse grid point {i}: {e}')
                continue

        # Create result
        result = OptimizationResult(
            optimal_horizons=self._extract_horizons(best_params),
            optimal_targets=self._extract_targets(best_params),
            optimization_score=best_score,
            validation_score=best_metrics.get('validation_score', 0.0),
            performance_metrics=best_metrics,
            optimization_time=0.0,  # Will be set by caller
            optimization_method='coarse_grid_search',
            n_trials=len(coarse_grid),
            convergence_info={'stage': 'coarse_grid', 'best_score': best_score}
        )

        return result

    def _fine_grid_search(self, market_data: pd.DataFrame, model_type: ModelType,
                         coarse_result: OptimizationResult) -> OptimizationResult:
        """Run fine grid search around best coarse result."""
        self.logger.info('   → Running fine grid search...')

        # Create search space
        search_space = self._create_search_space(model_type)

        # Get best coarse parameters
        best_coarse_params = self._result_to_params(coarse_result)

        # Build fine grid around best result using ml_commons
        fine_grid = build_fine_grid_around_best(
            search_space,
            best_coarse_params,
            self.config.grid_search_config.fine_grid_points
        )

        self.logger.info(f'   → Generated {len(fine_grid)} fine grid points')

        # Evaluate grid points
        best_score = -np.inf
        best_params = None
        best_metrics = {}

        for i, params in enumerate(fine_grid):
            try:
                # Convert params to MultiHorizonConfig
                config = self._params_to_config(params, model_type)

                # Evaluate configuration
                score, metrics = self._evaluate_configuration(config, market_data, model_type)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics

                if i % 5 == 0:
                    self.logger.info(f'   → Fine grid point {i+1}/{len(fine_grid)}: Score: {score:.3f}')

            except Exception as e:
                self.logger.warning(f'⚠️ Error evaluating fine grid point {i}: {e}')
                continue

        # Create result
        result = OptimizationResult(
            optimal_horizons=self._extract_horizons(best_params),
            optimal_targets=self._extract_targets(best_params),
            optimization_score=best_score,
            validation_score=best_metrics.get('validation_score', 0.0),
            performance_metrics=best_metrics,
            optimization_time=0.0,  # Will be set by caller
            optimization_method='fine_grid_search',
            n_trials=len(fine_grid),
            convergence_info={'stage': 'fine_grid', 'best_score': best_score}
        )

        return result

    def _bayesian_tpe_optimization(self, market_data: pd.DataFrame, model_type: ModelType) -> OptimizationResult:
        """Run Bayesian TPE optimization using ml_commons HPO utilities."""
        self.logger.info('   → Running Bayesian TPE optimization...')

        # Create search space for HPO
        search_space = self._create_hpo_search_space(model_type)

        # Define objective function
        def objective(trial):
            # Sample parameters from trial
            params = {}
            for name, config in search_space.items():
                if config['type'] == 'int':
                    params[name] = trial.suggest_int(name, config['low'], config['high'])
                elif config['type'] == 'float':
                    if config.get('log', False):
                        params[name] = trial.suggest_float(name, config['low'], config['high'], log=True)
                    else:
                        params[name] = trial.suggest_float(name, config['low'], config['high'])
                elif config['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, config['choices'])

            # Convert to MultiHorizonConfig
            config = self._params_to_config(params, model_type)

            # Evaluate configuration
            score, _ = self._evaluate_configuration(config, market_data, model_type)

            return score

        # Run optimization using ml_commons HPO with early stopping and parallel processing
        try:
            # Configure early stopping based on improvement plateau
            early_stopping_config = {
                'patience': 10,
                'min_delta': 0.001,
                'mode': 'max'
            }

            # Configure parallel processing for faster evaluation
            parallel_config = {
                'n_jobs': min(self.config.bayesian_tpe_config.max_workers, 4),
                'backend': 'threading'
            }

            optimization_result = self.hpo_optimizer.optimize_hyperparameters(
                objective=objective,
                search_space=search_space,
                n_trials=self.config.bayesian_tpe_config.n_trials,
                timeout=self.config.bayesian_tpe_config.timeout_seconds,
                early_stopping=early_stopping_config,
                parallel_config=parallel_config
            )

            # Extract best parameters
            best_params = optimization_result.best_params
            best_score = optimization_result.best_value

            # Create result
            result = OptimizationResult(
                optimal_horizons=self._extract_horizons(best_params),
                optimal_targets=self._extract_targets(best_params),
                optimization_score=best_score,
                validation_score=0.0,  # Will be calculated separately
                performance_metrics={},
                optimization_time=0.0,  # Will be set by caller
                optimization_method='bayesian_tpe',
                n_trials=self.config.bayesian_tpe_config.n_trials,
                convergence_info=optimization_result.convergence_info
            )

            return result

        except Exception as e:
            self.logger.error(f'❌ Bayesian TPE optimization failed: {e}')
            raise RuntimeError(f"Bayesian TPE optimization failed: {e}")

    def _create_search_space(self, model_type: ModelType) -> Dict[str, Any]:
        """Create search space for grid search."""
        search_space = SearchSpace()

        # Adjust ranges based on model type
        if model_type == ModelType.ANALYST:
            # Analyst: 1h base timeframe, 1-16 periods (1h-16h)
            search_space.horizon_immediate = {'type': 'int', 'low': 1, 'high': 16}
            search_space.horizon_short = {'type': 'int', 'low': 1, 'high': 16}
        elif model_type == ModelType.TACTICIAN:
            # Tactician: 15m base timeframe, 1-16 periods (15m-240m)
            search_space.horizon_immediate = {'type': 'int', 'low': 1, 'high': 16}
            search_space.horizon_short = {'type': 'int', 'low': 1, 'high': 16}
        else:  # BOTH
            # Balanced approach
            search_space.horizon_immediate = {'type': 'int', 'low': 1, 'high': 16}
            search_space.horizon_short = {'type': 'int', 'low': 1, 'high': 16}

        return search_space.to_dict()

    def _create_hpo_search_space(self, model_type: ModelType) -> Dict[str, Any]:
        """Create search space for HPO optimization."""
        return self._create_search_space(model_type)

    def _params_to_config(self, params: Dict[str, Any], model_type: ModelType) -> MultiHorizonConfig:
        """Convert parameters to MultiHorizonConfig."""
        config = MultiHorizonConfig()

        # Set time horizons
        config.time_horizons = {
            'immediate': int(params.get('horizon_immediate', 2)),
            'short': int(params.get('horizon_short', 4))
        }

        # Set profit targets
        config.profit_targets = {
            'micro': float(params.get('target_micro', 0.003)),
            'small': float(params.get('target_small', 0.005)),
            'medium': float(params.get('target_medium', 0.007)),
            'good': float(params.get('target_good', 0.010))
        }

        return config

    def _evaluate_configuration(self, config: MultiHorizonConfig, market_data: pd.DataFrame,
                               model_type: ModelType) -> Tuple[float, Dict[str, float]]:
        """Evaluate a configuration using ml_commons validation utilities with caching."""
        # Create cache key from configuration parameters
        cache_key = self._create_cache_key(config, market_data.shape, model_type)

        # Check cache first
        if cache_key in self._evaluation_cache:
            self.logger.debug(f'   → Cache hit for configuration evaluation')
            return self._evaluation_cache[cache_key]

        try:
            # Generate labels using configuration (optimized - avoid deep copy when possible)
            from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabeler
            labeler = MultiHorizonProfitLabeler(config)

            # Use view instead of copy for efficiency when possible
            try:
                labeled_data = labeler.generate_labels(market_data)
            except:
                labeled_data = labeler.generate_labels(market_data.copy())

            # Calculate performance metrics using ml_commons validation
            metrics = self._calculate_performance_metrics(labeled_data, market_data, model_type)

            # Calculate overall score
            score = self._calculate_overall_score(metrics)

            result = (score, metrics)

            # Cache the result (manage cache size)
            if len(self._evaluation_cache) >= self._cache_max_size:
                # Remove oldest 20% of cache entries
                items_to_remove = len(self._evaluation_cache) // 5
                cache_items = list(self._evaluation_cache.items())
                for i in range(items_to_remove):
                    del self._evaluation_cache[cache_items[i][0]]

            self._evaluation_cache[cache_key] = result
            return result

        except Exception as e:
            self.logger.warning(f'⚠️ Error evaluating configuration: {e}')
            return 0.0, {}

    def _create_cache_key(self, config: MultiHorizonConfig, data_shape: Tuple, model_type: ModelType) -> str:
        """Create a cache key for configuration evaluation."""
        # Create a hashable representation of the configuration and data shape
        config_str = f"{config.time_horizons}_{config.profit_targets}_{model_type.value}_{data_shape}"
        return hash(config_str)

    def _calculate_performance_metrics(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame,
                                     model_type: ModelType) -> Dict[str, float]:
        """Calculate performance metrics using optimized ml_commons validation utilities."""
        try:
            # Use ml_commons validation system with optimized settings
            validation_result = self.validation_system.validate_model_performance(
                model=None,  # No model needed for configuration validation
                X=market_data,
                y=labeled_data,
                validation_type='configuration_validation'
            )

            # Extract metrics efficiently
            hit_rate = validation_result.get('hit_rate', 0.5)
            sharpe_ratio = validation_result.get('sharpe_ratio', 0.0)
            information_ratio = validation_result.get('information_ratio', 0.0)
            max_drawdown = validation_result.get('max_drawdown', 0.1)
            validation_score = validation_result.get('overall_score', 0.5)

            # Return optimized metrics dictionary
            return {
                'hit_rate': float(hit_rate),
                'sharpe_ratio': float(sharpe_ratio),
                'information_ratio': float(information_ratio),
                'max_drawdown': float(max_drawdown),
                'validation_score': float(validation_score)
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating performance metrics: {e}')
            return {
                'hit_rate': 0.5,
                'sharpe_ratio': 0.0,
                'information_ratio': 0.0,
                'max_drawdown': 0.1,
                'validation_score': 0.5
            }

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall optimization score."""
        try:
            # Weighted combination of metrics
            weights = {
                'hit_rate': 0.3,
                'sharpe_ratio': 0.3,
                'information_ratio': 0.2,
                'max_drawdown': 0.2
            }

            # Calculate weighted score
            score = sum(
                weights.get(metric, 0) * metrics.get(metric, 0)
                for metric in weights.keys()
            )

            # Adjust for drawdown (lower is better)
            if 'max_drawdown' in metrics:
                score -= weights['max_drawdown'] * metrics['max_drawdown']

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5

    def _extract_horizons(self, params: Dict[str, Any]) -> Dict[str, int]:
        """Extract horizon configuration from parameters."""
        return {
            'immediate': int(params.get('horizon_immediate', 2)),
            'short': int(params.get('horizon_short', 4))
        }

    def _extract_targets(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Extract target configuration from parameters."""
        return {
            'micro': float(params.get('target_micro', 0.003)),
            'small': float(params.get('target_small', 0.005)),
            'medium': float(params.get('target_medium', 0.007)),
            'good': float(params.get('target_good', 0.010))
        }

    def _result_to_params(self, result: OptimizationResult) -> Dict[str, Any]:
        """Convert optimization result to parameters."""
        params = {}

        # Add horizons
        for horizon_name, horizon_value in result.optimal_horizons.items():
            params[f'horizon_{horizon_name}'] = horizon_value

        # Add targets
        for target_name, target_value in result.optimal_targets.items():
            params[f'target_{target_name}'] = target_value

        return params

    def _select_best_result(self) -> Optional[OptimizationResult]:
        """Select best result from all stages."""
        results = [r for r in [self.best_coarse_result, self.best_fine_result, self.best_bayesian_result] if r is not None]

        if not results:
            return None

        # Return result with highest optimization score
        return max(results, key=lambda r: r.optimization_score)

    def _create_fallback_result(self, model_type: ModelType) -> OptimizationResult:
        """Create fallback result when optimization fails."""
        if model_type == ModelType.ANALYST:
            horizons = {'immediate': 2, 'short': 8}  # 30m, 120m
            targets = {'micro': 0.003, 'small': 0.005, 'medium': 0.007, 'good': 0.010}
        elif model_type == ModelType.TACTICIAN:
            horizons = {'immediate': 4, 'short': 8}  # 20m, 40m
            targets = {'micro': 0.005, 'small': 0.007, 'medium': 0.010, 'good': 0.015}
        else:  # BOTH
            horizons = {'immediate': 2, 'short': 8}  # Balanced
            targets = {'micro': 0.004, 'small': 0.006, 'medium': 0.008, 'good': 0.012}

        return OptimizationResult(
            optimal_horizons=horizons,
            optimal_targets=targets,
            optimization_score=0.5,
            validation_score=0.5,
            performance_metrics={},
            optimization_time=0.0,
            optimization_method='fallback',
            n_trials=0,
            convergence_info={'stage': 'fallback'}
        )
