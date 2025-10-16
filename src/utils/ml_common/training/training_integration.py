"""
Training Integration Module

This module provides integration examples and decorators to easily integrate
the enhanced training utilities into existing training steps.

Usage:
    from src.utils.ml_common.training.training_integration import enhanced_training

    @enhanced_training
    def train_model(X, y, model):
        # Your existing training code
        model.fit(X, y)
        return model
"""

import functools
import inspect
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import enhanced training utilities
from .enhanced_training_utils import (
    EnhancedTrainingUtils,
    EarlyStoppingConfig,
    PurgedCVConfig,
    OverfittingMonitorConfig,
    RegularizationConfig,
    create_enhanced_training_utils
)

# Import existing utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")

@dataclass
class TrainingIntegrationConfig:
    """Configuration for training integration."""
    enable_early_stopping: bool = True
    enable_purged_cv: bool = True
    enable_lookahead_detection: bool = True
    enable_temporal_splits: bool = True
    enable_regularization: bool = True
    enable_overfitting_monitoring: bool = True
    enable_validation_curves: bool = False
    enable_walk_forward: bool = False
    enable_ensemble_diversity: bool = False

    # Model-specific settings
    model_type: str = 'auto'  # 'auto', 'xgboost', 'lightgbm', 'catboost', 'randomforest', 'elasticnet'

    # Early stopping settings
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Purged CV settings
    cv_n_splits: int = 5
    cv_purge_pct: float = 0.01

    # Overfitting monitoring
    overfitting_threshold: float = 0.1

    # Regularization
    l1_alpha: float = 0.01
    l2_alpha: float = 0.01

def enhanced_training(config: Optional[TrainingIntegrationConfig] = None):
    """
    Decorator to enhance any training function with comprehensive overfitting prevention
    and lookahead bias detection.

    Args:
        config: Training integration configuration

    Usage:
        @enhanced_training()
        def train_model(X, y, model):
            model.fit(X, y)
            return model

        @enhanced_training(TrainingIntegrationConfig(enable_early_stopping=True))
        def train_ensemble(X, y, models):
            for model in models:
                model.fit(X, y)
            return models
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get configuration
            integration_config = config or TrainingIntegrationConfig()

            # Initialize enhanced training utilities
            early_stopping_config = EarlyStoppingConfig(
                enabled=integration_config.enable_early_stopping,
                patience=integration_config.early_stopping_patience,
                min_delta=integration_config.early_stopping_min_delta
            )

            purged_cv_config = PurgedCVConfig(
                enabled=integration_config.enable_purged_cv,
                n_splits=integration_config.cv_n_splits,
                purge_pct=integration_config.cv_purge_pct
            )

            overfitting_config = OverfittingMonitorConfig(
                enabled=integration_config.enable_overfitting_monitoring,
                threshold=integration_config.overfitting_threshold
            )

            regularization_config = RegularizationConfig(
                enabled=integration_config.enable_regularization,
                l1_alpha=integration_config.l1_alpha,
                l2_alpha=integration_config.l2_alpha
            )

            enhanced_utils = EnhancedTrainingUtils(
                early_stopping_config=early_stopping_config,
                purged_cv_config=purged_cv_config,
                overfitting_config=overfitting_config,
                regularization_config=regularization_config
            )

            # Extract common arguments
            X = kwargs.get('X') or (args[0] if len(args) > 0 else None)
            y = kwargs.get('y') or (args[1] if len(args) > 1 else None)
            model = kwargs.get('model') or (args[2] if len(args) > 2 else None)
            timestamps = kwargs.get('timestamps')

            if X is None or y is None or model is None:
                tprint_warning("⚠️ Enhanced training: Missing required arguments (X, y, model)")
                return func(*args, **kwargs)

            try:
                # Step 1: Validate temporal data
                if integration_config.enable_lookahead_detection:
                    tprint_info("🔍 Validating temporal data for lookahead bias...")
                    is_valid, warnings = enhanced_utils.validate_temporal_data(
                        X, y, timestamps, strict_mode=True
                    )
                    if warnings:
                        for warning in warnings:
                            tprint_warning(f"⚠️ {warning}")

                # Step 2: Apply enhanced regularization
                if integration_config.enable_regularization:
                    tprint_info("🔧 Applying enhanced regularization...")
                    model = enhanced_utils.apply_enhanced_regularization(
                        model, integration_config.model_type
                    )

                # Step 3: Create temporal splits if needed
                if integration_config.enable_temporal_splits and len(X) > 1000:
                    tprint_info("📊 Creating temporal splits...")
                    # This would be used if the function needs CV
                    temporal_splits = enhanced_utils.create_temporal_splits(X, y, timestamps)
                    kwargs['temporal_splits'] = temporal_splits

                # Step 4: Execute original training function
                tprint_info("🚀 Executing enhanced training...")
                start_time = time.time()

                result = func(*args, **kwargs)

                training_time = time.time() - start_time
                tprint_success(f"✅ Training completed in {training_time:.2f}s")

                # Step 5: Post-training monitoring
                if integration_config.enable_overfitting_monitoring and hasattr(result, 'predict'):
                    tprint_info("📊 Monitoring for overfitting...")

                    # Create validation split for monitoring
                    if len(X) > 200:
                        split_point = int(len(X) * 0.8)
                        X_train, X_val = X[:split_point], X[split_point:]
                        y_train, y_val = y[:split_point], y[split_point:]

                        overfitting_results = enhanced_utils.monitor_overfitting(
                            result, X_train, y_train, X_val, y_val,
                            model_name=type(result).__name__
                        )

                        if overfitting_results.get('is_overfitting', False):
                            tprint_warning("⚠️ Overfitting detected in trained model")
                        else:
                            tprint_success("✅ No overfitting detected")

                # Step 6: Add training metadata
                if hasattr(result, '__dict__'):
                    result._enhanced_training_metadata = {
                        'training_time': training_time,
                        'integration_config': integration_config.__dict__,
                        'overfitting_monitoring': overfitting_results if 'overfitting_results' in locals() else None
                    }

                return result

            except Exception as e:
                tprint_error(f"❌ Enhanced training failed: {e}")
                # Fallback to original function
                tprint_warning("⚠️ Falling back to original training function")
                return func(*args, **kwargs)

        return wrapper
    return decorator

def enhanced_ensemble_training(config: Optional[TrainingIntegrationConfig] = None):
    """
    Specialized decorator for ensemble training with diversity monitoring.

    Args:
        config: Training integration configuration

    Usage:
        @enhanced_ensemble_training()
        def train_ensemble(X, y, models):
            for model in models:
                model.fit(X, y)
            return models
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Enable ensemble-specific features
            ensemble_config = config or TrainingIntegrationConfig()
            ensemble_config.enable_ensemble_diversity = True
            ensemble_config.enable_walk_forward = True

            # Use the enhanced training decorator
            enhanced_decorator = enhanced_training(ensemble_config)
            enhanced_func = enhanced_decorator(func)

            return enhanced_func(*args, **kwargs)

        return wrapper
    return decorator

def enhanced_cross_validation(config: Optional[TrainingIntegrationConfig] = None):
    """
    Decorator for cross-validation with temporal integrity.

    Args:
        config: Training integration configuration

    Usage:
        @enhanced_cross_validation()
        def cross_validate_model(X, y, model):
            # Your CV implementation
            return cv_scores
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Enable CV-specific features
            cv_config = config or TrainingIntegrationConfig()
            cv_config.enable_purged_cv = True
            cv_config.enable_temporal_splits = True
            cv_config.enable_lookahead_detection = True

            # Use the enhanced training decorator (downstream CV calls should delegate to unified CV)
            enhanced_decorator = enhanced_training(cv_config)
            enhanced_func = enhanced_decorator(func)

            return enhanced_func(*args, **kwargs)

        return wrapper
    return decorator

class TrainingStepEnhancer:
    """
    Class-based approach to enhance training steps with comprehensive utilities.
    """

    def __init__(self, config: Optional[TrainingIntegrationConfig] = None):
        """Initialize training step enhancer."""
        self.config = config or TrainingIntegrationConfig()
        self.enhanced_utils = create_enhanced_training_utils(
            early_stopping_config=EarlyStoppingConfig(
                enabled=self.config.enable_early_stopping,
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta
            ),
            purged_cv_config=PurgedCVConfig(
                enabled=self.config.enable_purged_cv,
                n_splits=self.config.cv_n_splits,
                purge_pct=self.config.cv_purge_pct
            ),
            overfitting_config=OverfittingMonitorConfig(
                enabled=self.config.enable_overfitting_monitoring,
                threshold=self.config.overfitting_threshold
            ),
            regularization_config=RegularizationConfig(
                enabled=self.config.enable_regularization,
                l1_alpha=self.config.l1_alpha,
                l2_alpha=self.config.l2_alpha
            )
        )

        tprint_success("✅ Training Step Enhancer initialized")

    def enhance_training_step(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            model: Any,
                            timestamps: Optional[np.ndarray] = None,
                            model_name: str = 'model',
                            regime_labels: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Enhance a single training step with all available utilities.

        Args:
            X: Feature matrix
            y: Target array
            model: Model to train
            timestamps: Timestamp array (optional)
            model_name: Name of the model

        Returns:
            Tuple of (trained_model, training_metadata)
        """
        training_metadata = {
            'model_name': model_name,
            'training_time': 0,
            'enhancements_applied': [],
            'warnings': [],
            'overfitting_detected': False
        }

        start_time = time.time()

        try:
            fit_signature = inspect.signature(model.fit)
            accepts_timestamps = 'timestamps' in fit_signature.parameters

            # Step 1: Validate temporal data
            if self.config.enable_lookahead_detection:
                tprint_info(f"🔍 Validating temporal data for {model_name}...")
                is_valid, warnings = self.enhanced_utils.validate_temporal_data(
                    X, y, timestamps, strict_mode=True
                )
                training_metadata['warnings'].extend(warnings)
                if warnings:
                    tprint_warning(f"⚠️ {len(warnings)} warnings found for {model_name}")

            # Step 2: Apply regularization
            if self.config.enable_regularization:
                tprint_info(f"🔧 Applying regularization to {model_name}...")
                model = self.enhanced_utils.apply_enhanced_regularization(
                    model, self.config.model_type
                )
                training_metadata['enhancements_applied'].append('regularization')

            # Step 3: Train with enhanced cross-validation and early stopping
            if self.config.enable_early_stopping and len(X) > 200 and not accepts_timestamps:
                tprint_info(f"⏹️ Training {model_name} with enhanced cross-validation and early stopping...")

                # Enhanced cross-validation strategy based on regime information
                if regime_labels is not None and len(np.unique(regime_labels)) > 1:
                    # Use regime-aware CV strategies
                    try:
                        from src.training.steps.model_training.enhanced_regime_aware_hpo import EnhancedCVStrategies
                        cv_strategies = EnhancedCVStrategies()

                        # Use regime-aware time series split
                        splits = cv_strategies.regime_aware_time_series_split(
                            X, regime_labels, n_splits=min(5, len(X) // 50)
                        )

                        if splits:
                            # Use the first split for training/validation
                            train_idx, val_idx = splits[0]
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            training_metadata['enhancements_applied'].append('regime_aware_cv')
                            tprint_info(f"✅ Using regime-aware cross-validation for {model_name}")
                        else:
                            # Fallback to standard split
                            split_point = int(len(X) * 0.8)
                            X_train, X_val = X[:split_point], X[split_point:]
                            y_train, y_val = y[:split_point], y[split_point:]
                    except Exception as e:
                        tprint_error(f"❌ Regime-aware CV failed: {e}")
                        raise RuntimeError(f"❌ Enhanced training requires regime-aware CV to succeed. Fix the error: {e}") from e
                else:
                    # Use standard time series split
                    split_point = int(len(X) * 0.8)
                    X_train, X_val = X[:split_point], X[split_point:]
                    y_train, y_val = y[:split_point], y[split_point:]

                model, early_stopping_info = self.enhanced_utils.apply_early_stopping(
                    model, X_train, y_train, X_val, y_val, self.config.model_type
                )

                training_metadata['early_stopping'] = early_stopping_info
                training_metadata['enhancements_applied'].append('early_stopping')
            else:
                # Standard training
                tprint_info(f"🚀 Training {model_name}...")
                fit_kwargs: Dict[str, Any] = {}
                if timestamps is not None and accepts_timestamps:
                    fit_kwargs['timestamps'] = timestamps
                model.fit(X, y, **fit_kwargs)

            # Step 4: Monitor for overfitting with enhanced CV
            if self.config.enable_overfitting_monitoring and len(X) > 200:
                tprint_info(f"📊 Monitoring {model_name} for overfitting with enhanced CV...")

                # Enhanced validation split for overfitting monitoring
                if regime_labels is not None and len(np.unique(regime_labels)) > 1:
                    # Use regime-aware CV for overfitting monitoring
                    try:
                        cv_strategies = EnhancedCVStrategies()

                        splits = cv_strategies.regime_aware_time_series_split(
                            X, regime_labels, n_splits=min(3, len(X) // 100)
                        )

                        if splits:
                            # Use the last split for overfitting monitoring (more recent data)
                            train_idx, val_idx = splits[-1]
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            training_metadata['enhancements_applied'].append('regime_aware_overfitting_monitoring')
                            tprint_info(f"✅ Using regime-aware CV for overfitting monitoring")
                        else:
                            # Fallback to standard split
                            split_point = int(len(X) * 0.8)
                            X_train, X_val = X[:split_point], X[split_point:]
                            y_train, y_val = y[:split_point], y[split_point:]
                    except Exception as e:
                        tprint_error(f"❌ Regime-aware overfitting monitoring failed: {e}")
                        raise RuntimeError(f"❌ Enhanced overfitting monitoring requires regime-aware CV to succeed. Fix the error: {e}") from e
                else:
                    # Standard split
                    split_point = int(len(X) * 0.8)
                    X_train, X_val = X[:split_point], X[split_point:]
                    y_train, y_val = y[:split_point], y[split_point:]

                overfitting_results = self.enhanced_utils.monitor_overfitting(
                    model, X_train, y_train, X_val, y_val, model_name
                )

                training_metadata['overfitting_monitoring'] = overfitting_results
                training_metadata['overfitting_detected'] = overfitting_results.get('is_overfitting', False)

                if overfitting_results.get('is_overfitting', False):
                    tprint_warning(f"⚠️ Overfitting detected in {model_name}")
                else:
                    tprint_success(f"✅ No overfitting detected in {model_name}")

            training_time = time.time() - start_time
            training_metadata['training_time'] = training_time

            tprint_success(f"✅ {model_name} training completed in {training_time:.2f}s")
            return model, training_metadata

        except Exception as e:
            tprint_error(f"❌ Enhanced training failed for {model_name}: {e}")
            training_metadata['error'] = str(e)
            training_metadata['training_time'] = time.time() - start_time
            return model, training_metadata

    def enhance_ensemble_training(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                models: List[Any],
                                timestamps: Optional[np.ndarray] = None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Enhance ensemble training with diversity monitoring.

        Args:
            X: Feature matrix
            y: Target array
            models: List of models to train
            timestamps: Timestamp array (optional)

        Returns:
            Tuple of (trained_models, training_metadata)
        """
        ensemble_metadata = {
            'models_trained': len(models),
            'training_times': [],
            'enhancements_applied': [],
            'ensemble_diversity': None,
            'overfitting_detected': False
        }

        trained_models = []

        try:
            # Train each model with enhancements
            for i, model in enumerate(models):
                model_name = f"model_{i}_{type(model).__name__}"
                tprint_info(f"🚀 Training {model_name}...")

                trained_model, model_metadata = self.enhance_training_step(
                    X, y, model, timestamps, model_name
                )

                trained_models.append(trained_model)
                ensemble_metadata['training_times'].append(model_metadata['training_time'])

                if model_metadata.get('overfitting_detected', False):
                    ensemble_metadata['overfitting_detected'] = True

            # Calculate ensemble diversity
            if self.config.enable_ensemble_diversity and len(trained_models) > 1:
                tprint_info("📊 Calculating ensemble diversity...")
                diversity_metrics = self.enhanced_utils.calculate_ensemble_diversity(
                    trained_models, X, y
                )
                ensemble_metadata['ensemble_diversity'] = diversity_metrics

                if diversity_metrics.get('diversity_score', 0) < 0.1:
                    tprint_warning("⚠️ Low ensemble diversity detected")
                else:
                    tprint_success("✅ Good ensemble diversity")

            ensemble_metadata['enhancements_applied'] = [
                'individual_model_enhancement',
                'ensemble_diversity_monitoring'
            ]

            tprint_success(f"✅ Ensemble training completed: {len(trained_models)} models")
            return trained_models, ensemble_metadata

        except Exception as e:
            tprint_error(f"❌ Enhanced ensemble training failed: {e}")
            ensemble_metadata['error'] = str(e)
            return trained_models, ensemble_metadata

# Convenience functions for easy integration
def create_training_enhancer(config: Optional[TrainingIntegrationConfig] = None) -> TrainingStepEnhancer:
    """Create a training step enhancer with custom configuration."""
    return TrainingStepEnhancer(config)

def quick_enhance_training(X: np.ndarray,
                          y: np.ndarray,
                          model: Any,
                          timestamps: Optional[np.ndarray] = None,
                          model_name: str = 'model') -> Tuple[Any, Dict[str, Any]]:
    """Quick function to enhance a single training step."""
    enhancer = TrainingStepEnhancer()
    return enhancer.enhance_training_step(X, y, model, timestamps, model_name)

def quick_enhance_ensemble(X: np.ndarray,
                          y: np.ndarray,
                          models: List[Any],
                          timestamps: Optional[np.ndarray] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """Quick function to enhance ensemble training."""
    enhancer = TrainingStepEnhancer()
    return enhancer.enhance_ensemble_training(X, y, models, timestamps)
