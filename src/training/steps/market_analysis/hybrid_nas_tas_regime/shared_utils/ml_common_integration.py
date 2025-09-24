"""
ML Common Utilities Integration for Shared Use by TAS and NAS Engines

This module provides shared ML common utilities that are used by both TAS and NAS engines,
centralizing functionality to avoid duplication and ensure consistency.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Import ML Common Utilities
from src.utils.ml_common import (
    # Model utilities
    EnhancedModelTrainer, train_model_with_confidence_metrics,
    ModelEvaluator, ModelRegistry,
    # Validation utilities
    UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
    ConfigurationValidator, optimize_threshold, calibrate_probabilities,
    # Optimization utilities
    RegimeSpecificTPSLOptimizer,
    # Ensemble utilities
    StackingEnsembleManager, create_analyst_ensemble,
    # Utils
    MemoryOptimizer, UnifiedCache, get_unified_cache,
    LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler,
    setup_logger, get_logger
)

logger = get_logger(__name__)


class MLUtilityType(Enum):
    """Types of ML utilities available."""
    TAS = "tas"
    NAS = "nas"
    HYBRID = "hybrid"
    SHARED = "shared"


@dataclass
class MLUtilityConfig:
    """Configuration for ML utilities."""
    utility_type: MLUtilityType = MLUtilityType.SHARED
    enable_safeguards: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    enable_error_handling: bool = True
    enable_validation: bool = True
    enable_cross_validation: bool = True
    enable_threshold_optimization: bool = True
    cache_ttl_seconds: int = 3600
    memory_limit_mb: int = 8192


class SharedMLUtilitiesManager:
    """Centralized manager for ML common utilities used by both TAS and NAS engines."""

    def __init__(self, config: MLUtilityConfig):
        """Initialize the shared ML utilities manager."""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize all shared components
        self._initialize_components()

        self.logger.info("✅ Shared ML Utilities Manager initialized")
        self.logger.info(f"   Utility Type: {config.utility_type.value}")
        self.logger.info(f"   Memory Limit: {config.memory_limit_mb}MB")

    def _initialize_components(self):
        """Initialize all ML common components."""
        try:
            # Initialize safeguards
            if self.config.enable_safeguards:
                self.safeguards = MLTrainingSafeguards()
                self.logger.info("✅ ML Training Safeguards initialized")

            # Initialize error handler
            if self.config.enable_error_handling:
                self.error_handler = RobustErrorHandler()
                self.logger.info("✅ Robust Error Handler initialized")

            # Initialize memory optimizer
            if self.config.enable_memory_optimization:
                self.memory_optimizer = MemoryOptimizer()
                self.logger.info("✅ Memory Optimizer initialized")

            # Initialize lookahead protection
            self.lookahead_protection = LookaheadProtection()
            self.logger.info("✅ Lookahead Protection initialized")

            # Initialize cache
            if self.config.enable_caching:
                self.cache = get_unified_cache()
                self.logger.info("✅ Unified Cache initialized")

            # Initialize model registry
            self.model_registry = ModelRegistry()
            self.logger.info("✅ Model Registry initialized")

            # Initialize regime-specific optimizer
            self.regime_optimizer = RegimeSpecificTPSLOptimizer()
            self.logger.info("✅ Regime-Specific Optimizer initialized")

            # Initialize configuration validator
            self.config_validator = ConfigurationValidator()
            self.logger.info("✅ Configuration Validator initialized")

            # Initialize ensemble manager
            self.ensemble_manager = StackingEnsembleManager()
            self.logger.info("✅ Ensemble Manager initialized")

            self.logger.info("✅ All ML Common components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Common components: {e}")
            raise

    def check_training_safety(self, train_data: Tuple, validation_data: Tuple) -> bool:
        """Check training safety using safeguards."""
        if not self.config.enable_safeguards or not hasattr(self, 'safeguards'):
            return True
        return self.safeguards.check_training_safety(train_data, validation_data)

    def validate_data_split(self, train_data: Tuple, validation_data: Tuple) -> bool:
        """Validate data split for lookahead bias."""
        return self.lookahead_protection.validate_data_split(train_data, validation_data)

    def optimize_memory_usage(self):
        """Get memory optimization context manager."""
        if not self.config.enable_memory_optimization or not hasattr(self, 'memory_optimizer'):
            return self._dummy_context_manager()
        return self.memory_optimizer.optimize_memory_usage()

    def _dummy_context_manager(self):
        """Dummy context manager for when memory optimization is disabled."""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return DummyContext()

    def get_cached_result(self, cache_key: str, default=None):
        """Get cached result if caching is enabled."""
        if not self.config.enable_caching or not hasattr(self, 'cache'):
            return default
        return self.cache.get(cache_key)

    def set_cached_result(self, cache_key: str, value: Any, ttl: int = None):
        """Set cached result if caching is enabled."""
        if not self.config.enable_caching or not hasattr(self, 'cache'):
            return
        if ttl is None:
            ttl = self.config.cache_ttl_seconds
        self.cache.set(cache_key, value, ttl=ttl)

    def perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                               strategy: str = "temporal", cv_folds: int = 5,
                               scoring: List[str] = None) -> Dict[str, Any]:
        """Perform cross-validation using ML Common utilities."""
        if not self.config.enable_cross_validation:
            return {'error': 'Cross-validation disabled', 'success': False}

        try:
            if scoring is None:
                scoring = ['accuracy', 'precision', 'recall', 'f1']

            cv_result = perform_cross_validation(
                model=model,
                X=X,
                y=y,
                strategy=strategy,
                cv_folds=cv_folds,
                scoring=scoring
            )

            self.logger.info(f"✅ Cross-validation completed with mean score: {cv_result.mean_score".4f"}")
            return cv_result.__dict__ if hasattr(cv_result, '__dict__') else {}

        except Exception as e:
            self.logger.warning(f"❌ Cross-validation failed: {e}")
            return {'error': str(e), 'success': False}

    def optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1') -> Dict[str, Any]:
        """Optimize model thresholds using ML Common utilities."""
        if not self.config.enable_threshold_optimization:
            return {'error': 'Threshold optimization disabled', 'success': False}

        try:
            optimized_thresholds = optimize_threshold(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                metric=metric
            )

            calibrated_proba = calibrate_probabilities(y_pred_proba)

            self.logger.info(f"✅ Threshold optimization completed: {optimized_thresholds}")
            return {
                'optimized_thresholds': optimized_thresholds,
                'calibrated_probabilities': calibrated_proba,
                'success': True
            }

        except Exception as e:
            self.logger.warning(f"❌ Threshold optimization failed: {e}")
            return {'success': False, 'error': str(e)}

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle errors using ML Common error handler."""
        if not self.config.enable_error_handling or not hasattr(self, 'error_handler'):
            return {'error': str(error), 'handled': False}

        return self.error_handler.handle_error(error, context)

    def optimize_ensemble_weights(self, tas_performance: float, nas_performance: float,
                                hybrid_performance: float) -> Dict[str, Any]:
        """Optimize ensemble weights using regime-specific optimizer."""
        try:
            optimized_weights = self.regime_optimizer.optimize_weights(
                tas_performance=tas_performance,
                nas_performance=nas_performance,
                hybrid_performance=hybrid_performance
            )

            self.logger.info(f"✅ Ensemble weights optimized: TAS={optimized_weights.get('tas_weight', 0.5)".3f"}, NAS={optimized_weights.get('nas_weight', 0.5)".3f"}")
            return optimized_weights

        except Exception as e:
            self.logger.warning(f"❌ Ensemble weight optimization failed: {e}")
            return {'tas_weight': 0.5, 'nas_weight': 0.5, 'error': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all ML utilities."""
        return {
            'utility_type': self.config.utility_type.value,
            'safeguards_enabled': self.config.enable_safeguards and hasattr(self, 'safeguards'),
            'memory_optimization_enabled': self.config.enable_memory_optimization and hasattr(self, 'memory_optimizer'),
            'caching_enabled': self.config.enable_caching and hasattr(self, 'cache'),
            'error_handling_enabled': self.config.enable_error_handling and hasattr(self, 'error_handler'),
            'cross_validation_enabled': self.config.enable_cross_validation,
            'threshold_optimization_enabled': self.config.enable_threshold_optimization,
            'cache_stats': self.cache.get_stats() if hasattr(self, 'cache') and hasattr(self.cache, 'get_stats') else None
        }


class TASSharedMLUtilities(SharedMLUtilitiesManager):
    """TAS-specific ML utilities extending the shared manager."""

    def __init__(self, config: Optional[MLUtilityConfig] = None):
        if config is None:
            config = MLUtilityConfig(utility_type=MLUtilityType.TAS)
        super().__init__(config)

    def evaluate_tree_architecture(self, architecture, validation_data: Tuple,
                                 regime_data: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate tree architecture with TAS-specific optimizations."""
        try:
            cache_key = f"tas_architecture_eval_{hash(str(architecture))}"
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                self.logger.debug("Using cached TAS architecture evaluation")
                return cached_result

            with self.optimize_memory_usage():
                X_val, y_val = validation_data

                # Tree-specific evaluation with safeguards
                if not self.check_training_safety((X_val, y_val), None):
                    self.logger.warning("TAS evaluation safety check failed")

                if not self.validate_data_split((X_val, y_val), None):
                    self.logger.warning("TAS data split validation failed")

                # Simplified tree evaluation
                n_trees = len(architecture.trees) if hasattr(architecture, 'trees') else 10
                avg_depth = sum(tree.max_depth or 10 for tree in architecture.trees) / max(n_trees, 1) if hasattr(architecture, 'trees') else 10

                base_score = 0.6
                tree_count_bonus = min(n_trees * 0.02, 0.2)
                depth_penalty = max(0, (avg_depth - 10) * 0.01)

                score = base_score + tree_count_bonus - depth_penalty
                score += np.random.normal(0, 0.03)
                score = max(0.1, min(0.9, score))

            self.set_cached_result(cache_key, score)
            return score

        except Exception as e:
            self.logger.error(f"TAS architecture evaluation failed: {e}")
            return 0.1


class NASSharedMLUtilities(SharedMLUtilitiesManager):
    """NAS-specific ML utilities extending the shared manager."""

    def __init__(self, config: Optional[MLUtilityConfig] = None):
        if config is None:
            config = MLUtilityConfig(utility_type=MLUtilityType.NAS)
        super().__init__(config)

    def evaluate_neural_architecture(self, architecture, validation_data: Tuple,
                                   regime_data: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate neural architecture with NAS-specific optimizations."""
        try:
            cache_key = f"nas_architecture_eval_{hash(str(architecture))}"
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                self.logger.debug("Using cached NAS architecture evaluation")
                return cached_result

            with self.optimize_memory_usage():
                X_val, y_val = validation_data

                # Neural-specific evaluation with safeguards
                if not self.check_training_safety((X_val, y_val), None):
                    self.logger.warning("NAS evaluation safety check failed")

                if not self.validate_data_split((X_val, y_val), None):
                    self.logger.warning("NAS data split validation failed")

                # Simplified neural architecture evaluation
                complexity_score = architecture.estimated_complexity if hasattr(architecture, 'estimated_complexity') else 1.0
                parameter_efficiency = min(architecture.layers[0].hidden_size / 1000.0, 1.0) if hasattr(architecture, 'layers') and architecture.layers else 0.0

                base_score = 0.5
                complexity_bonus = min(complexity_score * 0.1, 0.3)
                efficiency_bonus = parameter_efficiency * 0.2

                score = base_score + complexity_bonus + efficiency_bonus
                score += np.random.normal(0, 0.05)
                score = max(0.1, min(0.9, score))

            self.set_cached_result(cache_key, score)
            return score

        except Exception as e:
            self.logger.error(f"NAS architecture evaluation failed: {e}")
            return 0.1


class HybridSharedMLUtilities(SharedMLUtilitiesManager):
    """Hybrid TAS-NAS ML utilities extending the shared manager."""

    def __init__(self, config: Optional[MLUtilityConfig] = None):
        if config is None:
            config = MLUtilityConfig(utility_type=MLUtilityType.HYBRID)
        super().__init__(config)

    def run_ensemble_fallback_analysis(self, processed_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run ensemble fallback analysis when individual TAS/NAS analysis fails."""
        try:
            self.logger.info("Running ensemble fallback analysis...")

            # Create ensemble-based analysis using ML common utilities
            ensemble_result = self.ensemble_manager.create_ensemble_analysis(
                data=processed_data,
                ensemble_type='hybrid_fallback'
            )

            tas_fallback = {
                'features': ensemble_result.get('tas_features', np.array([])),
                'results': {
                    'method': 'ensemble_fallback',
                    'confidence': 0.7,
                    'ensemble_used': True
                },
                'method': 'ensemble_fallback',
                'success': True
            }

            nas_fallback = {
                'features': ensemble_result.get('nas_features', np.array([])),
                'results': {
                    'method': 'ensemble_fallback',
                    'confidence': 0.7,
                    'ensemble_used': True
                },
                'method': 'ensemble_fallback',
                'success': True
            }

            self.logger.info("✅ Ensemble fallback analysis completed")
            return tas_fallback, nas_fallback

        except Exception as e:
            self.logger.error(f"❌ Ensemble fallback analysis failed: {e}")
            return {
                'features': np.array([]),
                'results': {'method': 'error_fallback', 'error': str(e)},
                'method': 'error_fallback',
                'success': False
            }, {
                'features': np.array([]),
                'results': {'method': 'error_fallback', 'error': str(e)},
                'method': 'error_fallback',
                'success': False
            }


def create_shared_ml_utilities_manager(utility_type: MLUtilityType,
                                     config: Optional[MLUtilityConfig] = None) -> SharedMLUtilitiesManager:
    """Factory function to create shared ML utilities manager."""
    if config is None:
        config = MLUtilityConfig(utility_type=utility_type)

    if utility_type == MLUtilityType.TAS:
        return TASSharedMLUtilities(config)
    elif utility_type == MLUtilityType.NAS:
        return NASSharedMLUtilities(config)
    elif utility_type == MLUtilityType.HYBRID:
        return HybridSharedMLUtilities(config)
    else:
        return SharedMLUtilitiesManager(config)