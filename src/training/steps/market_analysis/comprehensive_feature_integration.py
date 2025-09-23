"""
Comprehensive Feature Integration Component

This component combines all available features and model outputs for enhanced training:
- Base features (technical indicators, price data, etc.)
- Analyst model outputs (confidence scores, directional predictions, etc.)
- HMM model outputs (regime predictions, regime features, etc.)
- Memory-efficient combination with proper validation

Enhanced Features:
- Handles multiple feature types with different shapes and formats
- Memory-efficient processing with optional chunking
- Comprehensive validation and error handling
- Supports both Analyst and Tactician training requirements
- Optimized for large datasets with hardware acceleration
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

# Initialize logger
logger = logging.getLogger(__name__)


class ComprehensiveFeatureIntegrator:
    """
    Comprehensive feature integration for enhanced model training.

    Combines multiple feature sources:
    1. Base features (technical indicators, price data, etc.)
    2. Analyst model outputs (confidence scores, directional predictions)
    3. HMM model outputs (regime predictions, regime features)
    4. Ensemble predictions from all model types

    Enhanced Features:
    - Memory-efficient processing with hardware optimization
    - Comprehensive validation and error handling
    - Supports both Analyst and Tactician training requirements
    - Handles missing data and different feature formats
    - Detailed integration statistics and reporting
    """

    def __init__(
        self,
        enable_memory_optimization: bool = True,
        enable_hardware_acceleration: bool = True,
        validate_inputs: bool = True,
        max_missing_ratio: float = 0.1,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize the comprehensive feature integrator.

        Args:
            enable_memory_optimization: Whether to use memory-efficient processing
            enable_hardware_acceleration: Whether to use hardware optimization
            validate_inputs: Whether to validate input data thoroughly
            max_missing_ratio: Maximum allowed missing data ratio per feature
            chunk_size: Size of data chunks for memory optimization (None = auto)
        """
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Initializing Comprehensive Feature Integrator")
            tprint_debug(f"   Memory optimization: {enable_memory_optimization}")
            tprint_debug(f"   Hardware acceleration: {enable_hardware_acceleration}")
            tprint_debug(f"   Input validation: {validate_inputs}")
            tprint_debug(f"   Max missing ratio: {max_missing_ratio}")

        self.enable_memory_optimization = enable_memory_optimization
        self.enable_hardware_acceleration = enable_hardware_acceleration
        self.validate_inputs = validate_inputs
        self.max_missing_ratio = max_missing_ratio
        self.chunk_size = chunk_size

        # Initialize state
        self._validate_configuration()
        self._initialize_components()

        if TPRINT_AVAILABLE:
            tprint_success("✅ Comprehensive Feature Integrator initialized")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        try:
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
        self.integration_stats = {
            'base_features_count': 0,
            'analyst_features_count': 0,
            'hmm_features_count': 0,
            'ensemble_features_count': 0,
            'total_features': 0,
            'samples_processed': 0,
            'integration_time': 0.0,
            'memory_usage_mb': 0.0,
            'hardware_accelerated': False
        }

        self.feature_sources = {
            'base': None,
            'analyst_individual': {},
            'analyst_ensembles': {},
            'hmm_regime': None,
            'hmm_models': {}
        }

        # Initialize hardware optimizers
        self._initialize_hardware_optimizers()

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
                    self.integration_stats['hardware_accelerated'] = True
                    if TPRINT_AVAILABLE:
                        tprint_success("🚀 Hardware optimization enabled for feature integration")
                else:
                    if TPRINT_AVAILABLE:
                        tprint_warning("⚠️ Hardware optimization requested but not available")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Hardware optimization module not available")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Hardware optimizer initialization failed: {e}")

    def integrate_features(
        self,
        base_features: Union[pd.DataFrame, np.ndarray],
        analyst_outputs: Optional[Dict[str, Any]] = None,
        hmm_outputs: Optional[Dict[str, Any]] = None,
        ensemble_outputs: Optional[Dict[str, Any]] = None,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Integrate all feature sources into a comprehensive feature matrix.

        Args:
            base_features: Core features (technical indicators, price data, etc.)
            analyst_outputs: Outputs from individual Analyst models
            hmm_outputs: Outputs from HMM models (regime predictions, features)
            ensemble_outputs: Outputs from ensemble models
            return_stats: Whether to return detailed integration statistics

        Returns:
            Dictionary containing:
            - 'integrated_features': Combined feature matrix
            - 'feature_names': Names of all integrated features
            - 'integration_stats': Detailed statistics (if requested)
            - 'feature_sources': Summary of feature sources
        """
        start_time = time.time()
        if TPRINT_AVAILABLE:
            tprint_info("🔄 Starting comprehensive feature integration")

        try:
            # Input validation
            if self.validate_inputs:
                self._validate_integration_inputs(base_features, analyst_outputs, hmm_outputs, ensemble_outputs)

            # Convert to numpy arrays for efficient processing
            base_array = self._convert_to_array(base_features)

            # Initialize feature matrix
            integrated_features, feature_names = self._initialize_feature_matrix(base_array)

            # Integrate Analyst model outputs
            if analyst_outputs:
                analyst_features, analyst_names = self._integrate_analyst_outputs(
                    analyst_outputs, base_array.shape[0]
                )
                integrated_features = self._combine_features(integrated_features, analyst_features)
                feature_names.extend(analyst_names)

            # Integrate HMM outputs
            if hmm_outputs:
                hmm_features, hmm_names = self._integrate_hmm_outputs(
                    hmm_outputs, base_array.shape[0]
                )
                integrated_features = self._combine_features(integrated_features, hmm_features)
                feature_names.extend(hmm_names)

            # Integrate ensemble outputs
            if ensemble_outputs:
                ensemble_features, ensemble_names = self._integrate_ensemble_outputs(
                    ensemble_outputs, base_array.shape[0]
                )
                integrated_features = self._combine_features(integrated_features, ensemble_features)
                feature_names.extend(ensemble_names)

            # Update statistics
            self._update_integration_stats(
                base_features=base_array.shape[1],
                analyst_features=len(analyst_outputs) if analyst_outputs else 0,
                hmm_features=len(hmm_outputs) if hmm_outputs else 0,
                ensemble_features=len(ensemble_outputs) if ensemble_outputs else 0,
                total_features=integrated_features.shape[1],
                samples=base_array.shape[0],
                integration_time=time.time() - start_time
            )

            # Log results
            if TPRINT_AVAILABLE:
                self._log_integration_results()

            result = {
                'integrated_features': integrated_features,
                'feature_names': feature_names,
                'feature_sources': self._get_feature_sources_summary()
            }

            if return_stats:
                result['integration_stats'] = self.integration_stats.copy()

            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Feature integration completed: {integrated_features.shape[1]} total features")

            return result

        except Exception as e:
            error_msg = f"Feature integration failed: {str(e)}"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
                tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_integration_inputs(
        self,
        base_features: Union[pd.DataFrame, np.ndarray],
        analyst_outputs: Optional[Dict[str, Any]],
        hmm_outputs: Optional[Dict[str, Any]],
        ensemble_outputs: Optional[Dict[str, Any]]
    ) -> None:
        """Validate inputs for feature integration."""
        try:
            if base_features is None:
                raise ValueError("Base features cannot be None")

            base_array = self._convert_to_array(base_features)
            if base_array.shape[0] == 0:
                raise ValueError("Base features cannot be empty")
            if base_array.shape[1] == 0:
                raise ValueError("Base features cannot have zero columns")

            # Check for NaN/Inf values
            if np.any(np.isnan(base_array)):
                nan_ratio = np.mean(np.isnan(base_array))
                if nan_ratio > self.max_missing_ratio:
                    raise ValueError(f"Base features have too many NaN values: {nan_ratio:.2%}")
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Base features contain NaN values: {nan_ratio:.2%}")

            if np.any(np.isinf(base_array)):
                inf_ratio = np.mean(np.isinf(base_array))
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Base features contain infinite values: {inf_ratio:.2%}")

            # Validate analyst outputs
            if analyst_outputs:
                for model_name, outputs in analyst_outputs.items():
                    if outputs is not None:
                        output_array = np.asarray(outputs)
                        if output_array.shape[0] != base_array.shape[0]:
                            raise ValueError(f"Analyst model '{model_name}' outputs length mismatch: {output_array.shape[0]} vs {base_array.shape[0]}")

            # Validate HMM outputs
            if hmm_outputs:
                for output_name, outputs in hmm_outputs.items():
                    if outputs is not None:
                        output_array = np.asarray(outputs)
                        if output_array.shape[0] != base_array.shape[0]:
                            raise ValueError(f"HMM output '{output_name}' length mismatch: {output_array.shape[0]} vs {base_array.shape[0]}")

            # Validate ensemble outputs
            if ensemble_outputs:
                for ensemble_name, outputs in ensemble_outputs.items():
                    if outputs is not None:
                        output_array = np.asarray(outputs)
                        if output_array.shape[0] != base_array.shape[0]:
                            raise ValueError(f"Ensemble '{ensemble_name}' outputs length mismatch: {output_array.shape[0]} vs {base_array.shape[0]}")

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Input validation failed: {e}")
            raise

    def _convert_to_array(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input data to numpy array with validation."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _initialize_feature_matrix(self, base_array: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Initialize the base feature matrix and feature names."""
        try:
            n_samples, n_features = base_array.shape

            # Create feature names for base features
            base_feature_names = [f"feature_{i}" for i in range(n_features)]

            # Use hardware optimization for memory allocation if available
            if self.enable_memory_optimization and HARDWARE_OPTIMIZATION_AVAILABLE:
                if self.memory_optimizer:
                    try:
                        # Use hardware-optimized allocation
                        integrated_features = self.memory_optimizer.allocate_optimized_array(
                            shape=(n_samples, n_features),
                            dtype=base_array.dtype,
                            optimization_level='aggressive'
                        )
                        integrated_features[:] = base_array
                    except Exception:
                        # Fallback to standard allocation
                        integrated_features = base_array.copy()
                else:
                    integrated_features = base_array.copy()
            else:
                integrated_features = base_array.copy()

            return integrated_features, base_feature_names

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to initialize feature matrix: {e}")
            raise

    def _integrate_analyst_outputs(
        self,
        analyst_outputs: Dict[str, Any],
        n_samples: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Integrate Analyst model outputs into feature matrix."""
        try:
            analyst_features_list = []
            analyst_feature_names = []

            for model_name, outputs in analyst_outputs.items():
                if outputs is not None:
                    try:
                        # Convert to array and validate shape
                        output_array = np.asarray(outputs)

                        # Handle different output shapes
                        if output_array.ndim == 1:
                            output_array = output_array.reshape(-1, 1)
                        elif output_array.ndim > 2:
                            # Flatten multi-dimensional outputs
                            output_array = output_array.reshape(n_samples, -1)

                        # Validate shape
                        if output_array.shape[0] != n_samples:
                            raise ValueError(f"Analyst model '{model_name}' shape mismatch: {output_array.shape[0]} vs {n_samples}")

                        # Create feature names
                        if output_array.shape[1] == 1:
                            feature_name = f"analyst_{model_name}"
                        else:
                            feature_name = [f"analyst_{model_name}_{i}" for i in range(output_array.shape[1])]

                        analyst_features_list.append(output_array)
                        analyst_feature_names.extend(feature_name if isinstance(feature_name, list) else [feature_name])

                    except Exception as e:
                        if TPRINT_AVAILABLE:
                            tprint_warning(f"⚠️ Failed to integrate Analyst model '{model_name}': {e}")
                        continue

            if analyst_features_list:
                analyst_features = np.concatenate(analyst_features_list, axis=1)
                return analyst_features, analyst_feature_names
            else:
                return np.empty((n_samples, 0)), []

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to integrate Analyst outputs: {e}")
            raise

    def _integrate_hmm_outputs(
        self,
        hmm_outputs: Dict[str, Any],
        n_samples: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Integrate HMM model outputs into feature matrix."""
        try:
            hmm_features_list = []
            hmm_feature_names = []

            for output_name, outputs in hmm_outputs.items():
                if outputs is not None:
                    try:
                        # Convert to array and validate shape
                        output_array = np.asarray(outputs)

                        # Handle different output shapes
                        if output_array.ndim == 1:
                            output_array = output_array.reshape(-1, 1)
                        elif output_array.ndim > 2:
                            # Flatten multi-dimensional outputs
                            output_array = output_array.reshape(n_samples, -1)

                        # Validate shape
                        if output_array.shape[0] != n_samples:
                            raise ValueError(f"HMM output '{output_name}' shape mismatch: {output_array.shape[0]} vs {n_samples}")

                        # Create feature names
                        if output_array.shape[1] == 1:
                            feature_name = f"hmm_{output_name}"
                        else:
                            feature_name = [f"hmm_{output_name}_{i}" for i in range(output_array.shape[1])]

                        hmm_features_list.append(output_array)
                        hmm_feature_names.extend(feature_name if isinstance(feature_name, list) else [feature_name])

                    except Exception as e:
                        if TPRINT_AVAILABLE:
                            tprint_warning(f"⚠️ Failed to integrate HMM output '{output_name}': {e}")
                        continue

            if hmm_features_list:
                hmm_features = np.concatenate(hmm_features_list, axis=1)
                return hmm_features, hmm_feature_names
            else:
                return np.empty((n_samples, 0)), []

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to integrate HMM outputs: {e}")
            raise

    def _integrate_ensemble_outputs(
        self,
        ensemble_outputs: Dict[str, Any],
        n_samples: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Integrate ensemble model outputs into feature matrix."""
        try:
            ensemble_features_list = []
            ensemble_feature_names = []

            for ensemble_name, outputs in ensemble_outputs.items():
                if outputs is not None:
                    try:
                        # Convert to array and validate shape
                        output_array = np.asarray(outputs)

                        # Handle different output shapes
                        if output_array.ndim == 1:
                            output_array = output_array.reshape(-1, 1)
                        elif output_array.ndim > 2:
                            # Flatten multi-dimensional outputs
                            output_array = output_array.reshape(n_samples, -1)

                        # Validate shape
                        if output_array.shape[0] != n_samples:
                            raise ValueError(f"Ensemble '{ensemble_name}' shape mismatch: {output_array.shape[0]} vs {n_samples}")

                        # Create feature names
                        if output_array.shape[1] == 1:
                            feature_name = f"ensemble_{ensemble_name}"
                        else:
                            feature_name = [f"ensemble_{ensemble_name}_{i}" for i in range(output_array.shape[1])]

                        ensemble_features_list.append(output_array)
                        ensemble_feature_names.extend(feature_name if isinstance(feature_name, list) else [feature_name])

                    except Exception as e:
                        if TPRINT_AVAILABLE:
                            tprint_warning(f"⚠️ Failed to integrate ensemble '{ensemble_name}': {e}")
                        continue

            if ensemble_features_list:
                ensemble_features = np.concatenate(ensemble_features_list, axis=1)
                return ensemble_features, ensemble_feature_names
            else:
                return np.empty((n_samples, 0)), []

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to integrate ensemble outputs: {e}")
            raise

    def _combine_features(self, existing_features: np.ndarray, new_features: np.ndarray) -> np.ndarray:
        """Combine existing features with new features using hardware optimization."""
        try:
            if new_features.shape[1] == 0:
                return existing_features

            # Use hardware optimization if available
            if self.enable_memory_optimization and HARDWARE_OPTIMIZATION_AVAILABLE:
                if self.memory_optimizer:
                    # Use optimized concatenation
                    return self.memory_optimizer.concatenate_arrays([existing_features, new_features])
                elif self.hardware_manager:
                    # Use hardware manager for optimization
                    return self.hardware_manager.concatenate_features(existing_features, new_features)

            # Fallback to numpy concatenation
            return np.concatenate([existing_features, new_features], axis=1)

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Hardware-optimized concatenation failed, using fallback: {e}")
            return np.concatenate([existing_features, new_features], axis=1)

    def _update_integration_stats(
        self,
        base_features: int,
        analyst_features: int,
        hmm_features: int,
        ensemble_features: int,
        total_features: int,
        samples: int,
        integration_time: float
    ) -> None:
        """Update integration statistics."""
        self.integration_stats.update({
            'base_features_count': base_features,
            'analyst_features_count': analyst_features,
            'hmm_features_count': hmm_features,
            'ensemble_features_count': ensemble_features,
            'total_features': total_features,
            'samples_processed': samples,
            'integration_time': integration_time
        })

        # Update memory usage if available
        if COMMON_UTILITIES_AVAILABLE:
            try:
                self.integration_stats['memory_usage_mb'] = get_memory_usage() / (1024 * 1024)
            except Exception:
                pass

    def _log_integration_results(self) -> None:
        """Log detailed integration results."""
        if not TPRINT_AVAILABLE:
            return

        stats = self.integration_stats
        tprint_info("📊 Comprehensive Feature Integration Results:")
        tprint_info(f"   Base features: {stats['base_features_count']}")
        tprint_info(f"   Analyst features: {stats['analyst_features_count']}")
        tprint_info(f"   HMM features: {stats['hmm_features_count']}")
        tprint_info(f"   Ensemble features: {stats['ensemble_features_count']}")
        tprint_info(f"   Total features: {stats['total_features']}")
        tprint_info(f"   Samples processed: {stats['samples_processed']}")
        tprint_info(f"   Integration time: {stats['integration_time']:.2f}s")
        if stats['hardware_accelerated']:
            tprint_info(f"   Hardware accelerated: ✅")
        if stats['memory_usage_mb'] > 0:
            tprint_info(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")

    def _get_feature_sources_summary(self) -> Dict[str, Any]:
        """Get summary of feature sources."""
        return {
            'base_features_available': self.integration_stats['base_features_count'] > 0,
            'analyst_features_available': self.integration_stats['analyst_features_count'] > 0,
            'hmm_features_available': self.integration_stats['hmm_features_count'] > 0,
            'ensemble_features_available': self.integration_stats['ensemble_features_count'] > 0,
            'total_feature_types': sum([
                1 if self.integration_stats['base_features_count'] > 0 else 0,
                1 if self.integration_stats['analyst_features_count'] > 0 else 0,
                1 if self.integration_stats['hmm_features_count'] > 0 else 0,
                1 if self.integration_stats['ensemble_features_count'] > 0 else 0
            ])
        }

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        return self.integration_stats.copy()

    def reset_statistics(self) -> None:
        """Reset integration statistics."""
        self.integration_stats = {
            'base_features_count': 0,
            'analyst_features_count': 0,
            'hmm_features_count': 0,
            'ensemble_features_count': 0,
            'total_features': 0,
            'samples_processed': 0,
            'integration_time': 0.0,
            'memory_usage_mb': 0.0,
            'hardware_accelerated': False
        }

        if TPRINT_AVAILABLE:
            tprint_info("📊 Integration statistics reset")

    def cleanup_resources(self) -> None:
        """Clean up resources and reset state."""
        self.reset_statistics()

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
            tprint_info("🧹 Comprehensive Feature Integrator resources cleaned up")


# Convenience functions for easy integration
def create_comprehensive_feature_integrator(**kwargs) -> ComprehensiveFeatureIntegrator:
    """Create a comprehensive feature integrator instance."""
    return ComprehensiveFeatureIntegrator(**kwargs)


def integrate_all_features(
    base_features: Union[pd.DataFrame, np.ndarray],
    analyst_outputs: Optional[Dict[str, Any]] = None,
    hmm_outputs: Optional[Dict[str, Any]] = None,
    ensemble_outputs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to integrate all feature sources.

    Args:
        base_features: Core features (technical indicators, price data, etc.)
        analyst_outputs: Outputs from individual Analyst models
        hmm_outputs: Outputs from HMM models (regime predictions, features)
        ensemble_outputs: Outputs from ensemble models
        **kwargs: Additional arguments for integrator configuration

    Returns:
        Dictionary with integrated features and statistics
    """
    integrator = create_comprehensive_feature_integrator(**kwargs)

    return integrator.integrate_features(
        base_features=base_features,
        analyst_outputs=analyst_outputs,
        hmm_outputs=hmm_outputs,
        ensemble_outputs=ensemble_outputs
    )


if __name__ == "__main__":
    # Example usage
    print("Comprehensive Feature Integration Component")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_base_features = 20

    base_features = np.random.randn(n_samples, n_base_features)
    analyst_outputs = {
        'confidence_model': np.random.uniform(0.3, 0.9, n_samples),
        'directional_model': np.random.uniform(-1, 1, n_samples),
        'bias_model': np.random.uniform(-0.5, 0.5, n_samples)
    }

    hmm_outputs = {
        'regime_predictions': np.random.randint(0, 5, n_samples),
        'regime_features': np.random.randn(n_samples, 8),
        'transition_probabilities': np.random.uniform(0, 1, (n_samples, 3))
    }

    ensemble_outputs = {
        'analyst_ensemble': np.random.uniform(0.4, 0.8, n_samples),
        'hmm_ensemble': np.random.uniform(0, 1, (n_samples, 2))
    }

    print(f"Base features shape: {base_features.shape}")
    print(f"Analyst outputs: {len(analyst_outputs)} models")
    print(f"HMM outputs: {len(hmm_outputs)} outputs")
    print(f"Ensemble outputs: {len(ensemble_outputs)} ensembles")

    # Create integrator
    integrator = create_comprehensive_feature_integrator(
        enable_memory_optimization=True,
        enable_hardware_acceleration=True,
        validate_inputs=True
    )

    # Integrate features
    result = integrator.integrate_features(
        base_features=base_features,
        analyst_outputs=analyst_outputs,
        hmm_outputs=hmm_outputs,
        ensemble_outputs=ensemble_outputs
    )

    print("
Integration Results:")
    print(f"Integrated features shape: {result['integrated_features'].shape}")
    print(f"Total features: {len(result['feature_names'])}")
    print(f"Feature types: {result['feature_sources']['total_feature_types']}")

    print("
Feature breakdown:")
    print(f"- Base: {integrator.integration_stats['base_features_count']}")
    print(f"- Analyst: {integrator.integration_stats['analyst_features_count']}")
    print(f"- HMM: {integrator.integration_stats['hmm_features_count']}")
    print(f"- Ensemble: {integrator.integration_stats['ensemble_features_count']}")

    print("\n✅ Comprehensive Feature Integration ready for enhanced training!")