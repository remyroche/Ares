"""
Optimized Timeframe Optimizer using ml_commons utilities extensively

This module provides the main optimized timeframe optimizer that leverages
ml_commons utilities for grid search, Bayesian TPE optimization, cross-validation,
and comprehensive validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import time
import json
from pathlib import Path

# Import ml_commons utilities extensively
from src.utils.ml_common.utils.hpo_utils import HyperparameterOptimization
from src.utils.ml_common.optimization.grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)
from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem
from src.utils.ml_common.validation.temporal_cross_validation import TemporalCrossValidator
from src.utils.ml_common.validation.cv_utils import CrossValidationUtilities
from src.utils.ml_common.validation.enhanced_overfitting_detection import EnhancedOverfittingDetector
from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention
from src.utils.ml_common.validation.stability import StabilityValidator
from src.utils.ml_common.utils.memory_optimization import MemoryOptimizer
from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection

# Import optimization components
from .grid_bayesian_optimizer import GridBayesianOptimizer
from .enhanced_validation import EnhancedValidator
from .optimization_config import (
    OptimizationConfig, ModelType, OptimizationMethod,
    OptimizationResult, ValidationConfig, ValidationLevel
)

# Import multi-horizon components
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

logger = logging.getLogger(__name__)

class OptimizedTimeframeOptimizer:
    """
    Main optimized timeframe optimizer using ml_commons utilities extensively.

    Features:
    - Grid search (coarse + fine) using ml_commons grid_utils
    - Bayesian TPE optimization using ml_commons hpo_utils
    - Cross-validation using ml_commons validation utilities
    - Comprehensive validation using ml_commons validation framework
    - Performance monitoring and caching
    - Memory optimization using ml_commons memory utilities
    - Lookahead protection using ml_commons protection utilities
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize optimized timeframe optimizer."""
        self.config = config
        self.logger = logging.getLogger('OptimizedTimeframeOptimizer')

        # Initialize ml_commons utilities extensively
        self._initialize_ml_commons_utilities()

        # Initialize optimization components
        self._initialize_optimization_components()

        # Optimization state
        self.optimization_results = {}
        self.optimization_history = []
        self.performance_cache = {}

        self.logger.info(f'🚀 Optimized Timeframe Optimizer initialized for {config.model_type.value}')
        self.logger.info(f'   → Optimization method: {config.optimization_method.value}')
        self.logger.info(f'   → Validation level: {config.validation_config.validation_level.value}')

    def _initialize_ml_commons_utilities(self):
        """Initialize ml_commons utilities extensively."""
        try:
            # Initialize HPO utility
            hpo_config = {
                'enable_parallel': self.config.bayesian_tpe_config.enable_parallel,
                'max_workers': self.config.bayesian_tpe_config.max_workers,
                'enable_monitoring': True,
                'use_nonlinear_optimization': True,
                'enable_pruning': self.config.bayesian_tpe_config.enable_pruning,
                'pruning_patience': self.config.bayesian_tpe_config.pruning_patience
            }
            self.hpo_optimizer = HyperparameterOptimization(hpo_config)

            # Initialize unified validation system
            self.unified_validator = UnifiedValidationSystem()

            # Initialize temporal cross-validator
            self.temporal_cv = TemporalCrossValidator(
                n_splits=self.config.validation_config.cv_folds,
                gap=1  # 1 period gap to prevent lookahead
            )

            # Initialize cross-validation utilities
            cv_config = {
                'initial_train_size': 0.6,
                'step_size': 0.1,
                'min_test_size': 0.1
            }
            self.cv_utilities = CrossValidationUtilities(cv_config)

            # Initialize overfitting detector
            self.overfitting_detector = EnhancedOverfittingDetector()

            # Initialize data leakage prevention
            self.leakage_prevention = DataLeakagePrevention()

            # Initialize stability validator
            self.stability_validator = StabilityValidator()

            # Initialize memory optimizer
            self.memory_optimizer = MemoryOptimizer()

            # Initialize lookahead protection
            self.lookahead_protection = LookaheadProtection()

            self.logger.info('✅ ml_commons utilities initialized extensively')

        except Exception as e:
            self.logger.error(f'❌ Failed to initialize ml_commons utilities: {e}')
            raise RuntimeError(f"Failed to initialize ml_commons utilities: {e}")

    def _initialize_optimization_components(self):
        """Initialize optimization components."""
        try:
            # Initialize grid-bayesian optimizer
            self.grid_bayesian_optimizer = GridBayesianOptimizer(self.config)

            # Initialize enhanced validation framework
            self.validation_framework = EnhancedValidator(self.config.validation_config)

            self.logger.info('✅ Optimization components initialized')

        except Exception as e:
            self.logger.error(f'❌ Failed to initialize optimization components: {e}')
            raise RuntimeError(f"Failed to initialize optimization components: {e}")

    def optimize_for_model(self, model_type: ModelType, market_data: pd.DataFrame,
                          force_optimization: bool = False) -> OptimizationResult:
        """
        Optimize timeframes and profit targets for a specific model type.

        Args:
            model_type: Type of model to optimize for
            market_data: Market data for optimization
            force_optimization: Force re-optimization even if cached results exist

        Returns:
            OptimizationResult with optimal configuration
        """
        self.logger.info(f'🎯 Starting optimization for {model_type.value} model...')
        start_time = datetime.now()

        # Check for cached results
        if not force_optimization and model_type in self.optimization_results:
            self.logger.info(f'📋 Using cached optimization results for {model_type.value}')
            return self.optimization_results[model_type]

        try:
            # Apply lookahead protection
            with self.lookahead_protection.protect_data(market_data):
                # Apply memory optimization
                with self.memory_optimizer.optimize_memory_usage():
                    # Run optimization using grid-bayesian optimizer
                    optimization_result = self.grid_bayesian_optimizer.optimize(
                        market_data=market_data,
                        model_type=model_type
                    )

                    # Validate optimization result using enhanced validation framework
                    validation_result = self.validation_framework.validate_optimized_configuration(
                        config=self._result_to_config(optimization_result),
                        market_data=market_data,
                        model_type=model_type.value
                    )

                    # Update optimization result with validation score
                    optimization_result.validation_score = validation_result.validation_score
                    optimization_result.performance_metrics.update(validation_result.to_dict())

                    # Fast fail if validation score is too low
                    if optimization_result.validation_score < self.config.min_validation_score:
                        self.logger.error(f'❌ FAST FAIL: Low validation score ({optimization_result.validation_score:.3f}) for {model_type.value}')
                        raise RuntimeError(
                            f"❌ FAST FAIL: Validation score ({optimization_result.validation_score:.3f}) below minimum threshold ({self.config.min_validation_score}). "
                            f"Cannot proceed with invalid timeframe configuration. Training pipeline will terminate."
                        )

                    # Store results
                    self.optimization_results[model_type] = optimization_result
                    self.optimization_history.append(optimization_result)

                    # Cache performance metrics
                    self._cache_performance_metrics(model_type, optimization_result)

                    # Calculate total optimization time
                    total_time = (datetime.now() - start_time).total_seconds()
                    optimization_result.optimization_time = total_time

                    self.logger.info(f'✅ Optimization completed for {model_type.value} in {total_time:.2f}s')
                    self.logger.info(f'   → Optimization score: {optimization_result.optimization_score:.3f}')
                    self.logger.info(f'   → Validation score: {optimization_result.validation_score:.3f}')
                    self.logger.info(f'   → Optimal horizons: {optimization_result.optimal_horizons}')
                    self.logger.info(f'   → Optimal targets: {optimization_result.optimal_targets}')

                    return optimization_result

        except Exception as e:
            self.logger.error(f'❌ FAST FAIL: Optimization failed for {model_type.value}: {e}')
            if self.config.fast_fail_on_optimization_failure:
                raise RuntimeError(
                    f"❌ FAST FAIL: Optimization failed for {model_type.value} model. "
                    f"Error: {e}. Cannot proceed without optimal timeframe discovery. "
                    f"Training pipeline will terminate."
                )
            else:
                # Return fallback result
                return self._create_fallback_result(model_type)

    def optimize_for_both_models(self, market_data: pd.DataFrame) -> Dict[str, OptimizationResult]:
        """Optimize for both Analyst and Tactician models."""
        self.logger.info('🎯 Optimizing for both Analyst and Tactician models')

        results = {}

        # Optimize for Analyst model
        if self.config.model_type in [ModelType.ANALYST, ModelType.BOTH]:
            try:
                analyst_result = self.optimize_for_model(ModelType.ANALYST, market_data)
                results['analyst'] = analyst_result
                self.logger.info(f'   ✅ Analyst optimization completed (score: {analyst_result.optimization_score:.3f})')
            except Exception as e:
                self.logger.error(f'❌ Analyst optimization failed: {e}')
                if self.config.fast_fail_on_optimization_failure:
                    raise

        # Optimize for Tactician model
        if self.config.model_type in [ModelType.TACTICIAN, ModelType.BOTH]:
            try:
                tactician_result = self.optimize_for_model(ModelType.TACTICIAN, market_data)
                results['tactician'] = tactician_result
                self.logger.info(f'   ✅ Tactician optimization completed (score: {tactician_result.optimization_score:.3f})')
            except Exception as e:
                self.logger.error(f'❌ Tactician optimization failed: {e}')
                if self.config.fast_fail_on_optimization_failure:
                    raise

        return results

    def get_optimal_timeframes_for_models(self, market_data: pd.DataFrame,
                                        model_type: ModelType = ModelType.BOTH,
                                        force_optimization: bool = False) -> Dict[str, OptimizationResult]:
        """
        Main entry point to get optimal timeframes for specified model types.

        Args:
            market_data: Market data for optimization
            model_type: Type of model to optimize for
            force_optimization: Force re-optimization even if cached results exist

        Returns:
            Dictionary with optimization results for each model type
        """
        if model_type == ModelType.BOTH:
            return self.optimize_for_both_models(market_data)
        else:
            result = self.optimize_for_model(model_type, market_data, force_optimization)
            return {model_type.value: result}

    def _result_to_config(self, result: OptimizationResult) -> MultiHorizonConfig:
        """Convert optimization result to MultiHorizonConfig."""
        config = MultiHorizonConfig()
        config.time_horizons = result.optimal_horizons
        config.profit_targets = result.optimal_targets
        return config

    def _cache_performance_metrics(self, model_type: ModelType, result: OptimizationResult):
        """Cache performance metrics for future use."""
        try:
            cache_key = f"{model_type.value}_{datetime.now().strftime('%Y%m%d')}"
            self.performance_cache[cache_key] = {
                'optimization_score': result.optimization_score,
                'validation_score': result.validation_score,
                'optimal_horizons': result.optimal_horizons,
                'optimal_targets': result.optimal_targets,
                'timestamp': datetime.now().isoformat()
            }

            # Save to file if caching is enabled
            if self.config.enable_caching:
                self._save_performance_cache()

        except Exception as e:
            self.logger.warning(f'⚠️ Error caching performance metrics: {e}')

    def _save_performance_cache(self):
        """Save performance cache to file."""
        try:
            cache_dir = Path("optimization_cache")
            cache_dir.mkdir(exist_ok=True)

            cache_file = cache_dir / "performance_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.performance_cache, f, indent=2)

        except Exception as e:
            self.logger.warning(f'⚠️ Error saving performance cache: {e}')

    def _create_fallback_result(self, model_type: ModelType) -> OptimizationResult:
        """Create fallback result when optimization fails."""
        if model_type == ModelType.ANALYST:
            horizons = {'immediate': 2, 'short': 8}  # 2h, 8h (1h base)
            targets = {'micro': 0.003, 'small': 0.005, 'medium': 0.007, 'good': 0.010}
        elif model_type == ModelType.TACTICIAN:
            horizons = {'immediate': 4, 'short': 8}  # 1h, 2h (15m base)
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

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        summary = {
            'total_optimizations': len(self.optimization_results),
            'optimization_history_count': len(self.optimization_history),
            'performance_cache_size': len(self.performance_cache),
            'model_results': {}
        }

        for model_type, result in self.optimization_results.items():
            summary['model_results'][model_type.value] = {
                'optimization_score': result.optimization_score,
                'validation_score': result.validation_score,
                'optimal_horizons': result.optimal_horizons,
                'optimal_targets': result.optimal_targets,
                'optimization_time': result.optimization_time,
                'optimization_method': result.optimization_method
            }

        return summary

    def export_optimization_results(self, output_file: str) -> None:
        """Export optimization results to file."""
        try:
            export_data = {
                'optimization_results': {
                    model_type.value: result.to_dict()
                    for model_type, result in self.optimization_results.items()
                },
                'optimization_history': [result.to_dict() for result in self.optimization_history],
                'performance_cache': self.performance_cache,
                'summary': self.get_optimization_summary()
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f'📊 Exported optimization results to {output_file}')

        except Exception as e:
            self.logger.error(f'❌ Error exporting optimization results: {e}')

    def clear_cache(self, model_type: Optional[ModelType] = None) -> None:
        """Clear optimization cache."""
        if model_type is None:
            self.optimization_results.clear()
            self.performance_cache.clear()
            self.logger.info('🗑️ Cleared all optimization cache')
        else:
            if model_type in self.optimization_results:
                del self.optimization_results[model_type]
            # Clear performance cache for model type
            keys_to_remove = [k for k in self.performance_cache.keys() if k.startswith(model_type.value)]
            for key in keys_to_remove:
                del self.performance_cache[key]
            self.logger.info(f'🗑️ Cleared optimization cache for {model_type.value}')

        # Save updated cache
        if self.config.enable_caching:
            self._save_performance_cache()
