"""
Enhanced Tactician Training Component

This component implements enhanced Tactician training with comprehensive feature integration:
- Trains on all base features + all Analyst outputs + all HMM outputs
- Uses enhanced data filtering (confidence > 0.5 + 45 min after drop)
- Comprehensive feature integration for optimal performance
- Enhanced validation and error handling
- Memory-efficient processing with hardware optimization

Enhanced Features:
- Integration with Analyst confidence scores and directional predictions
- Integration with HMM regime data and predictions
- Time-based data filtering for realistic trading conditions
- Memory-efficient processing with optional chunking
- Comprehensive validation and error handling
- Detailed training statistics and performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import traceback
import time

# Enhanced logging imports
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_structured,
        LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

# Common utilities imports
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics, get_memory_usage,
        optimize_dataframe_dtypes, safe_divide, validate_finite, safe_mean
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False

# Math validation imports
try:
    from src.utils.math_validation import (
        safe_divide as math_safe_divide, validate_finite as math_validate_finite,
        validate_positive, validate_range, safe_mean as math_safe_mean
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

# Hardware optimization imports
try:
    from src.utils.hardware import (
        get_advanced_memory_optimizer, get_unified_hardware_manager,
        optimize_dataframe_advanced, ADVANCED_MEMORY_AVAILABLE, UNIFIED_MANAGER_AVAILABLE
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    ADVANCED_MEMORY_AVAILABLE = False
    UNIFIED_MANAGER_AVAILABLE = False

# Import our custom components
from .comprehensive_feature_integration import create_comprehensive_feature_integrator, integrate_all_features
from .enhanced_tactician_data_filtering import create_enhanced_data_filter, filter_tactician_training_data

# Initialize logger
logger = logging.getLogger(__name__)


class EnhancedTacticianTraining:
    """
    Enhanced Tactician training component with comprehensive feature integration.

    Key Features:
    - Trains on all base features + Analyst outputs + HMM outputs
    - Uses enhanced data filtering (confidence > 0.5 + 45 min after drop)
    - Comprehensive feature integration for optimal performance
    - Memory-efficient processing with hardware optimization
    - Enhanced validation and error handling
    - Detailed training statistics and performance tracking
    """

    def __init__(
        self,
        model_types: List[str] = None,
        confidence_threshold: float = 0.5,
        ride_duration_minutes: int = 45,
        enable_memory_optimization: bool = True,
        enable_hardware_acceleration: bool = True,
        validate_inputs: bool = True,
        max_missing_ratio: float = 0.1,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize the enhanced Tactician training component.

        Args:
            model_types: List of Tactician model types to train
            confidence_threshold: Minimum Analyst confidence for training
            ride_duration_minutes: Duration to include after confidence drops
            enable_memory_optimization: Whether to use memory-efficient processing
            enable_hardware_acceleration: Whether to use hardware optimization
            validate_inputs: Whether to validate input data thoroughly
            max_missing_ratio: Maximum allowed missing data ratio per feature
            chunk_size: Size of data chunks for memory optimization (None = auto)
        """
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Initializing Enhanced Tactician Training")
            tprint_debug(f"   Model types: {model_types or 'default'}")
            tprint_debug(f"   Confidence threshold: {confidence_threshold}")
            tprint_debug(f"   Ride duration: {ride_duration_minutes} minutes")
            tprint_debug(f"   Memory optimization: {enable_memory_optimization}")
            tprint_debug(f"   Hardware acceleration: {enable_hardware_acceleration}")

        # Set default model types if not provided
        if model_types is None:
            model_types = ['xgboost', 'catboost', 'lightgbm', 'elastic_net']

        self.model_types = model_types
        self.confidence_threshold = confidence_threshold
        self.ride_duration_minutes = ride_duration_minutes
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_hardware_acceleration = enable_hardware_acceleration
        self.validate_inputs = validate_inputs
        self.max_missing_ratio = max_missing_ratio
        self.chunk_size = chunk_size

        # Initialize state
        self._validate_configuration()
        self._initialize_components()

        if TPRINT_AVAILABLE:
            tprint_success(f"✅ Enhanced Tactician Training initialized with {len(self.model_types)} model types")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        try:
            if not self.model_types:
                raise ValueError("Model types cannot be empty")

            if not (0.0 < self.confidence_threshold <= 1.0):
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")

            if not (0 < self.ride_duration_minutes <= 1440):  # Max 24 hours
                raise ValueError(f"Ride duration must be between 1 and 1440 minutes, got {self.ride_duration_minutes}")

            if not (0.0 <= self.max_missing_ratio <= 1.0):
                raise ValueError(f"Max missing ratio must be between 0.0 and 1.0, got {self.max_missing_ratio}")

            if self.chunk_size is not None and self.chunk_size <= 0:
                raise ValueError(f"Chunk size must be positive or None, got {self.chunk_size}")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Configuration validation failed: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize internal components and state."""
        self.training_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'green_light_samples': 0,
            'ride_samples': 0,
            'base_features_count': 0,
            'analyst_features_count': 0,
            'hmm_features_count': 0,
            'total_features_used': 0,
            'models_trained': 0,
            'successful_models': 0,
            'failed_models': 0,
            'training_time': 0.0,
            'memory_usage_mb': 0.0,
            'hardware_accelerated': False,
            'filtering_ratio': 0.0,
            'green_light_ratio': 0.0
        }

        self.trained_models = {}
        self.model_performance = {}
        self.last_filtering_result = None

        # Initialize hardware optimizers
        self._initialize_hardware_optimizers()

        # Initialize feature integrator
        self.feature_integrator = create_comprehensive_feature_integrator(
            enable_memory_optimization=self.enable_memory_optimization,
            enable_hardware_acceleration=self.enable_hardware_acceleration,
            validate_inputs=self.validate_inputs,
            max_missing_ratio=self.max_missing_ratio,
            chunk_size=self.chunk_size
        )

        # Initialize data filter
        self.data_filter = create_enhanced_data_filter(
            confidence_threshold=self.confidence_threshold,
            ride_duration_minutes=self.ride_duration_minutes,
            enable_memory_optimization=self.enable_memory_optimization,
            chunk_size=self.chunk_size,
            validate_inputs=self.validate_inputs
        )

        if TPRINT_AVAILABLE:
            tprint_debug("✅ Internal components initialized")

    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimization tools."""
        if not self.enable_hardware_acceleration:
            return

        try:
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                self.memory_optimizer = get_advanced_memory_optimizer() if ADVANCED_MEMORY_AVAILABLE else None
                self.hardware_manager = get_unified_hardware_manager() if UNIFIED_MANAGER_AVAILABLE else None

                if self.memory_optimizer or self.hardware_manager:
                    self.training_stats['hardware_accelerated'] = True
                    if TPRINT_AVAILABLE:
                        tprint_success("🚀 Hardware optimization enabled for Tactician training")
                else:
                    if TPRINT_AVAILABLE:
                        tprint_warning("⚠️ Hardware optimization requested but not available")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Hardware optimization module not available")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Hardware optimizer initialization failed: {e}")

    def train_enhanced_tactician(
        self,
        base_features: Union[pd.DataFrame, np.ndarray],
        targets: np.ndarray,
        analyst_outputs: Optional[Dict[str, Any]] = None,
        hmm_outputs: Optional[Dict[str, Any]] = None,
        ensemble_outputs: Optional[Dict[str, Any]] = None,
        timestamps: Optional[np.ndarray] = None,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Train enhanced Tactician models with comprehensive feature integration and filtering.

        Args:
            base_features: Core features (technical indicators, price data, etc.)
            targets: Target values for Tactician training
            analyst_outputs: Outputs from Analyst models (confidence scores, predictions)
            hmm_outputs: Outputs from HMM models (regime predictions, features)
            ensemble_outputs: Outputs from ensemble models (optional)
            timestamps: Timestamps for each sample (required for filtering)
            return_stats: Whether to return detailed training statistics

        Returns:
            Dictionary containing:
            - 'trained_models': Dictionary of trained Tactician models
            - 'model_performance': Performance metrics for each model
            - 'training_stats': Detailed training statistics
            - 'feature_integration': Feature integration results
            - 'filtering_stats': Data filtering statistics
        """
        start_time = time.time()
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Starting enhanced Tactician training with comprehensive features and filtering")

        try:
            # Input validation
            if self.validate_inputs:
                self._validate_training_inputs(
                    base_features, targets, analyst_outputs, hmm_outputs, ensemble_outputs, timestamps
                )

            # Convert targets to numpy array
            targets_array = np.asarray(targets)

            # Step 1: Apply enhanced data filtering
            if TPRINT_AVAILABLE:
                tprint_info("🔍 Applying enhanced data filtering...")

            # For filtering, we need confidence scores from analyst_outputs
            confidence_scores = None
            if analyst_outputs and 'directional_confidence' in analyst_outputs:
                confidence_scores = analyst_outputs['directional_confidence']
            elif analyst_outputs:
                # Try to find confidence scores with different names
                confidence_keys = [k for k in analyst_outputs.keys() if 'confidence' in k.lower()]
                if confidence_keys:
                    confidence_scores = analyst_outputs[confidence_keys[0]]

            if confidence_scores is not None:
                filtering_result = self.data_filter.filter_training_data(
                    data=base_features,
                    confidence_scores=confidence_scores,
                    timestamps=timestamps,
                    return_stats=True
                )

                filtered_features = filtering_result['filtered_data']
                filtered_targets = targets_array[filtering_result['filtering_mask']]
                self.last_filtering_result = filtering_result

                if TPRINT_AVAILABLE:
                    filter_stats = filtering_result['filtering_stats']
                    tprint_success(f"✅ Data filtering completed: {len(filtered_features)}/{len(base_features)} samples selected")
                    tprint_info(f"   Green light ratio: {filter_stats['green_light_ratio']:.2%}")
                    tprint_info(f"   Ride ratio: {filter_stats['ride_ratio']:.2%}")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ No confidence scores found, using all data")
                filtered_features = base_features
                filtered_targets = targets_array
                self.last_filtering_result = None

            # Step 2: Integrate features
            if TPRINT_AVAILABLE:
                tprint_info("🔄 Integrating features for Tactician training...")

            feature_integration = self.feature_integrator.integrate_features(
                base_features=filtered_features,
                analyst_outputs=analyst_outputs,
                hmm_outputs=hmm_outputs,
                ensemble_outputs=ensemble_outputs,
                return_stats=True
            )

            integrated_features = feature_integration['integrated_features']
            feature_names = feature_integration['feature_names']

            # Step 3: Train individual models
            if TPRINT_AVAILABLE:
                tprint_info(f"🏋️ Training {len(self.model_types)} Tactician model types...")

            trained_models = {}
            model_performance = {}

            for model_type in self.model_types:
                try:
                    if TPRINT_AVAILABLE:
                        tprint_debug(f"   Training {model_type} model...")

                    model, performance = self._train_single_model(
                        model_type, integrated_features, filtered_targets
                    )

                    if model is not None:
                        trained_models[model_type] = model
                        model_performance[model_type] = performance
                        self.training_stats['successful_models'] += 1

                        if TPRINT_AVAILABLE:
                            tprint_success(f"   ✅ {model_type} trained successfully")
                    else:
                        self.training_stats['failed_models'] += 1
                        if TPRINT_AVAILABLE:
                            tprint_warning(f"   ❌ {model_type} training failed")

                except Exception as e:
                    self.training_stats['failed_models'] += 1
                    if TPRINT_AVAILABLE:
                        tprint_error(f"   ❌ {model_type} training failed: {e}")

            # Step 4: Update statistics
            self._update_training_stats(
                base_features=base_features.shape[1] if hasattr(base_features, 'shape') else len(base_features[0]),
                analyst_features=len(analyst_outputs) if analyst_outputs else 0,
                hmm_features=len(hmm_outputs) if hmm_outputs else 0,
                total_features=integrated_features.shape[1],
                total_samples=len(base_features),
                filtered_samples=len(filtered_features),
                models_trained=len(self.model_types),
                training_time=time.time() - start_time
            )

            # Step 5: Log results
            if TPRINT_AVAILABLE:
                self._log_training_results()

            # Prepare result
            result = {
                'trained_models': trained_models,
                'model_performance': model_performance,
                'feature_integration': feature_integration,
                'training_stats': self.training_stats.copy()
            }

            if self.last_filtering_result:
                result['filtering_stats'] = self.last_filtering_result['filtering_stats']

            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Enhanced Tactician training completed: {len(trained_models)}/{len(self.model_types)} models successful")

            return result

        except Exception as e:
            error_msg = f"Enhanced Tactician training failed: {str(e)}"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
                tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_training_inputs(
        self,
        base_features: Union[pd.DataFrame, np.ndarray],
        targets: np.ndarray,
        analyst_outputs: Optional[Dict[str, Any]],
        hmm_outputs: Optional[Dict[str, Any]],
        ensemble_outputs: Optional[Dict[str, Any]],
        timestamps: Optional[np.ndarray]
    ) -> None:
        """Validate inputs for Tactician training."""
        try:
            if base_features is None:
                raise ValueError("Base features cannot be None")

            if targets is None:
                raise ValueError("Targets cannot be None")

            base_array = self.feature_integrator._convert_to_array(base_features)
            targets_array = np.asarray(targets)

            if base_array.shape[0] == 0:
                raise ValueError("Base features cannot be empty")
            if base_array.shape[0] != len(targets_array):
                raise ValueError(f"Features ({base_array.shape[0]}) and targets ({len(targets_array)}) length mismatch")
            if len(targets_array) == 0:
                raise ValueError("Targets cannot be empty")

            # Check for confidence scores in analyst outputs
            if analyst_outputs:
                confidence_keys = [k for k in analyst_outputs.keys() if 'confidence' in k.lower()]
                if confidence_keys and timestamps is None:
                    if TPRINT_AVAILABLE:
                        tprint_warning("⚠️ Confidence scores provided but no timestamps - filtering will be limited")

            # Validate analyst outputs
            if analyst_outputs:
                for output_name, outputs in analyst_outputs.items():
                    if outputs is not None:
                        output_array = np.asarray(outputs)
                        if output_array.shape[0] != base_array.shape[0]:
                            raise ValueError(f"Analyst output '{output_name}' shape mismatch: {output_array.shape[0]} vs {base_array.shape[0]}")

            # Validate HMM outputs
            if hmm_outputs:
                for output_name, outputs in hmm_outputs.items():
                    if outputs is not None:
                        output_array = np.asarray(outputs)
                        if output_array.shape[0] != base_array.shape[0]:
                            raise ValueError(f"HMM output '{output_name}' shape mismatch: {output_array.shape[0]} vs {base_array.shape[0]}")

            # Validate ensemble outputs
            if ensemble_outputs:
                for ensemble_name, outputs in ensemble_outputs.items():
                    if outputs is not None:
                        output_array = np.asarray(outputs)
                        if output_array.shape[0] != base_array.shape[0]:
                            raise ValueError(f"Ensemble '{ensemble_name}' shape mismatch: {output_array.shape[0]} vs {base_array.shape[0]}")

            # Validate timestamps if provided
            if timestamps is not None:
                timestamp_array = np.asarray(timestamps)
                if len(timestamp_array) != base_array.shape[0]:
                    raise ValueError(f"Timestamps ({len(timestamp_array)}) and features ({base_array.shape[0]}) length mismatch")

                try:
                    pd.to_datetime(timestamp_array)
                except Exception:
                    raise ValueError("Timestamps must be convertible to datetime")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Training input validation failed: {e}")
            raise

    def _train_single_model(
        self,
        model_type: str,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Train a single Tactician model."""
        try:
            # Import model libraries dynamically
            if model_type.lower() == 'xgboost':
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=300,
                    max_depth=12,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    objective='reg:squarederror'
                )
            elif model_type.lower() == 'catboost':
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=500,
                    depth=8,
                    learning_rate=0.05,
                    random_seed=42,
                    verbose=False
                )
            elif model_type.lower() == 'lightgbm':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    n_estimators=300,
                    max_depth=12,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            elif model_type.lower() == 'elastic_net':
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    random_state=42,
                    max_iter=10000
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Train model with hardware optimization
            if self.enable_hardware_acceleration and self.hardware_manager:
                with self.hardware_manager.optimized_context(
                    operation_type="ml_training",
                    expected_duration_minutes=15
                ):
                    model.fit(features, targets)
            else:
                model.fit(features, targets)

            # Calculate basic performance metrics
            train_predictions = model.predict(features)
            mse = np.mean((train_predictions - targets) ** 2)
            mae = np.mean(np.abs(train_predictions - targets))
            r2 = 1 - (mse / np.var(targets)) if np.var(targets) > 0 else 0

            performance = {
                'train_mse': mse,
                'train_mae': mae,
                'train_r2': r2,
                'feature_count': features.shape[1],
                'sample_count': features.shape[0],
                'training_successful': True,
                'model_type': model_type
            }

            return model, performance

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Failed to train {model_type} model: {e}")

            return None, {
                'training_successful': False,
                'error': str(e),
                'model_type': model_type
            }

    def _update_training_stats(
        self,
        base_features: int,
        analyst_features: int,
        hmm_features: int,
        total_features: int,
        total_samples: int,
        filtered_samples: int,
        models_trained: int,
        training_time: float
    ) -> None:
        """Update training statistics."""
        self.training_stats.update({
            'base_features_count': base_features,
            'analyst_features_count': analyst_features,
            'hmm_features_count': hmm_features,
            'total_features_used': total_features,
            'total_samples': total_samples,
            'filtered_samples': filtered_samples,
            'models_trained': models_trained,
            'training_time': training_time,
            'filtering_ratio': filtered_samples / max(total_samples, 1),
            'green_light_ratio': self.last_filtering_result['filtering_stats']['green_light_ratio'] if self.last_filtering_result else 0.0
        })

        # Update memory usage if available
        if COMMON_UTILITIES_AVAILABLE:
            try:
                self.training_stats['memory_usage_mb'] = get_memory_usage() / (1024 * 1024)
            except Exception:
                pass

    def _log_training_results(self) -> None:
        """Log detailed training results."""
        if not TPRINT_AVAILABLE:
            return

        stats = self.training_stats
        tprint_info("📊 Enhanced Tactician Training Results:")
        tprint_info(f"   Total samples: {stats['total_samples']","}")
        tprint_info(f"   Filtered samples: {stats['filtered_samples']} ({stats['filtering_ratio']:.2%})")
        tprint_info(f"   Green light ratio: {stats['green_light_ratio']:.2%}")
        tprint_info(f"   Base features: {stats['base_features_count']}")
        tprint_info(f"   Analyst features: {stats['analyst_features_count']}")
        tprint_info(f"   HMM features: {stats['hmm_features_count']}")
        tprint_info(f"   Total features: {stats['total_features_used']}")
        tprint_info(f"   Models trained: {stats['models_trained']}")
        tprint_info(f"   Successful models: {stats['successful_models']}")
        tprint_info(f"   Failed models: {stats['failed_models']}")
        tprint_info(f"   Training time: {stats['training_time']:.2f}s")
        if stats['hardware_accelerated']:
            tprint_info(f"   Hardware accelerated: ✅")
        if stats['memory_usage_mb'] > 0:
            tprint_info(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return self.training_stats.copy()

    def reset_statistics(self) -> None:
        """Reset training statistics."""
        self.training_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'green_light_samples': 0,
            'ride_samples': 0,
            'base_features_count': 0,
            'analyst_features_count': 0,
            'hmm_features_count': 0,
            'total_features_used': 0,
            'models_trained': 0,
            'successful_models': 0,
            'failed_models': 0,
            'training_time': 0.0,
            'memory_usage_mb': 0.0,
            'hardware_accelerated': False,
            'filtering_ratio': 0.0,
            'green_light_ratio': 0.0
        }

        if TPRINT_AVAILABLE:
            tprint_info("📊 Training statistics reset")

    def cleanup_resources(self) -> None:
        """Clean up resources and reset state."""
        self.reset_statistics()

        # Clean up feature integrator
        if hasattr(self, 'feature_integrator'):
            self.feature_integrator.cleanup_resources()

        # Clean up data filter
        if hasattr(self, 'data_filter'):
            self.data_filter.cleanup_resources()

        # Clean up hardware optimizers
        if self.enable_hardware_acceleration:
            try:
                if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                    self.memory_optimizer.cleanup_temporary_arrays()
                if hasattr(self, 'hardware_manager') and self.hardware_manager:
                    self.hardware_manager.cleanup_resources()
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Hardware cleanup warning: {e}")

        if TPRINT_AVAILABLE:
            tprint_info("🧹 Enhanced Tactician Training resources cleaned up")


# Convenience functions for easy integration
def create_enhanced_tactician_training(**kwargs) -> EnhancedTacticianTraining:
    """Create an enhanced Tactician training instance."""
    return EnhancedTacticianTraining(**kwargs)


def train_enhanced_tactician(
    base_features: Union[pd.DataFrame, np.ndarray],
    targets: np.ndarray,
    analyst_outputs: Optional[Dict[str, Any]] = None,
    hmm_outputs: Optional[Dict[str, Any]] = None,
    ensemble_outputs: Optional[Dict[str, Any]] = None,
    timestamps: Optional[np.ndarray] = None,
    model_types: List[str] = None,
    confidence_threshold: float = 0.5,
    ride_duration_minutes: int = 45,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train enhanced Tactician models.

    Args:
        base_features: Core features (technical indicators, price data, etc.)
        targets: Target values for Tactician training
        analyst_outputs: Outputs from Analyst models (confidence scores, predictions)
        hmm_outputs: Outputs from HMM models (regime predictions, features)
        ensemble_outputs: Outputs from ensemble models (optional)
        timestamps: Timestamps for each sample (required for filtering)
        model_types: List of model types to train
        confidence_threshold: Minimum Analyst confidence for training
        ride_duration_minutes: Duration to include after confidence drops
        **kwargs: Additional arguments for training configuration

    Returns:
        Dictionary with trained models and statistics
    """
    trainer = create_enhanced_tactician_training(
        model_types=model_types,
        confidence_threshold=confidence_threshold,
        ride_duration_minutes=ride_duration_minutes,
        **kwargs
    )

    return trainer.train_enhanced_tactician(
        base_features=base_features,
        targets=targets,
        analyst_outputs=analyst_outputs,
        hmm_outputs=hmm_outputs,
        ensemble_outputs=ensemble_outputs,
        timestamps=timestamps
    )


if __name__ == "__main__":
    # Example usage
    print("Enhanced Tactician Training Component")
    print("=" * 42)

    # Create sample data
    np.random.seed(42)
    n_samples = 3000
    n_base_features = 30

    base_features = np.random.randn(n_samples, n_base_features)
    targets = np.random.uniform(-0.01, 0.01, n_samples)  # Small price movements
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')

    # Create Analyst outputs with confidence scores
    analyst_outputs = {
        'directional_confidence': np.random.uniform(0.3, 0.9, n_samples),
        'directional_bias': np.random.uniform(-1, 1, n_samples),
        'overall_opportunity': np.random.uniform(0.4, 0.8, n_samples),
        'long_overall_opportunity': np.random.uniform(0.3, 0.9, n_samples),
        'short_overall_opportunity': np.random.uniform(0.3, 0.9, n_samples),
        'best_direction': np.random.choice([-1, 0, 1], n_samples),
        'opportunity_asymmetry': np.random.uniform(-0.5, 0.5, n_samples)
    }

    hmm_outputs = {
        'regime_predictions': np.random.randint(0, 5, n_samples),
        'regime_features': np.random.randn(n_samples, 12),
        'transition_probabilities': np.random.uniform(0, 1, (n_samples, 4)),
        'regime_confidence': np.random.uniform(0.6, 1.0, n_samples),
        'regime_stability': np.random.uniform(0.5, 1.0, n_samples)
    }

    ensemble_outputs = {
        'analyst_ensemble': np.random.uniform(0.4, 0.8, n_samples),
        'hmm_ensemble': np.random.uniform(0, 1, (n_samples, 2)),
        'meta_learner': np.random.uniform(-0.005, 0.005, n_samples)
    }

    print(f"Base features shape: {base_features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Timestamps: {len(timestamps)} entries")
    print(f"Analyst outputs: {len(analyst_outputs)} types")
    print(f"HMM outputs: {len(hmm_outputs)} types")
    print(f"Ensemble outputs: {len(ensemble_outputs)} types")

    # Create enhanced Tactician trainer
    trainer = create_enhanced_tactician_training(
        model_types=['xgboost', 'catboost', 'lightgbm', 'elastic_net'],
        confidence_threshold=0.5,
        ride_duration_minutes=45,
        enable_memory_optimization=True,
        enable_hardware_acceleration=True,
        validate_inputs=True
    )

    # Train enhanced Tactician models
    result = trainer.train_enhanced_tactician(
        base_features=base_features,
        targets=targets,
        analyst_outputs=analyst_outputs,
        hmm_outputs=hmm_outputs,
        ensemble_outputs=ensemble_outputs,
        timestamps=timestamps.values
    )

    print("
Training Results:")
    print(f"Models trained: {len(result['trained_models'])}")
    print(f"Successful models: {trainer.training_stats['successful_models']}")
    print(f"Failed models: {trainer.training_stats['failed_models']}")

    print("
Data Filtering:")
    if 'filtering_stats' in result:
        filter_stats = result['filtering_stats']
        print(f"Total samples: {filter_stats['original_samples']}")
        print(f"Filtered samples: {filter_stats['filtered_samples']}")
        print(f"Green light samples: {filter_stats['green_light_samples']}")
        print(f"Ride samples: {filter_stats['ride_samples']}")
        print(f"Green light ratio: {filter_stats['green_light_ratio']:.2%}")
        print(f"Ride ratio: {filter_stats['ride_ratio']:.2%}")

    print("
Feature Integration:")
    print(f"Total features: {result['feature_integration']['integration_stats']['total_features']}")
    print(f"Base features: {result['feature_integration']['integration_stats']['base_features_count']}")
    print(f"Analyst features: {result['feature_integration']['integration_stats']['analyst_features_count']}")
    print(f"HMM features: {result['feature_integration']['integration_stats']['hmm_features_count']}")
    print(f"Ensemble features: {result['feature_integration']['integration_stats']['ensemble_features_count']}")

    print("
Training Statistics:")
    stats = trainer.get_training_statistics()
    print(f"Training time: {stats['training_time']:.2f}s")
    print(f"Hardware accelerated: {stats['hardware_accelerated']}")
    if stats['memory_usage_mb'] > 0:
        print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")

    print("\n✅ Enhanced Tactician Training ready for realistic trading conditions!")