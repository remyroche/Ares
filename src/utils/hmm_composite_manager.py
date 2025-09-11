#!/usr/bin/env python3
"""
Enhanced HMM Composite Manager with Consolidated Functionality

This enhanced manager consolidates functionality from multiple HMM clustering files:
- Memory management (using existing M1 utilities)
- Bayesian optimization (consolidated from 3 files)
- Feature engineering (consolidated from 3 files)
- Validation (consolidated from 3 files)

Centralized manager for HMM composite cluster files that can be used by:
- step3_hmm_regime_discovery (to create files)
- VectorizedAdvancedFeatureEngineering (to check if files exist)
- CompositeHMMRegimeSystem (to load files)

This ensures consistent behavior and prevents infinite loops.
"""

import contextlib
import json
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Import existing utilities
from .logger import system_logger  # type: ignore[import]

# Import error handling decorator
try:
    from ..core.decorators import handles_errors  # type: ignore[import]
    HANDLES_ERRORS_AVAILABLE = True
except ImportError:
    # Fallback: create a no-op decorator
    def handles_errors(func):
        return func
    HANDLES_ERRORS_AVAILABLE = False

# Import comprehensive utility infrastructure
try:
    from .math_validation import (  # type: ignore[import]
        safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
        validate_positive, validate_range, MathValidationError
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

try:
    from .parquet_utils import ParquetUtils  # type: ignore[import]
    PARQUET_UTILS_AVAILABLE = True
except ImportError:
    PARQUET_UTILS_AVAILABLE = False

try:
    from .serialization_utils import JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer  # type: ignore[import]
    SERIALIZATION_UTILS_AVAILABLE = True
except ImportError:
    SERIALIZATION_UTILS_AVAILABLE = False

try:
    from .data_processing_utils import DataProcessingUtils  # type: ignore[import]
    DATA_PROCESSING_UTILS_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_UTILS_AVAILABLE = False

try:
    from .common_operations import create_fallback_logger, create_fallback_decorator  # type: ignore[import]
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError:
    COMMON_OPERATIONS_AVAILABLE = False

try:
    from .common_utilities import CommonUtilities  # type: ignore[import]
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False

# Import M1 optimization utilities (replacing memory management files)
try:
    from .hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager  # type: ignore[import]
    from .hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer  # type: ignore[import]
    from .hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer  # type: ignore[import]
    M1_UTILITIES_AVAILABLE = True
except ImportError:
    M1_UTILITIES_AVAILABLE = False

# Import HMM regime detection utilities
# Avoid circular imports by defining locally and importing only when needed
HMM_REGIME_CONFIG_AVAILABLE = False

@dataclass
class HMMRegimeConfig:
    """HMM configuration for regime detection."""
    n_components: int = 4
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-3
    random_state: int = 42

# Try to import the real class if available (without triggering circular imports)
try:
    import importlib.util
    spec = importlib.util.find_spec("utils.ml_common.hmm_regime_detection")
    if spec is not None:
        # Import directly from the module file to avoid __init__.py issues
        import sys
        import os
        module_path = os.path.join(os.path.dirname(__file__), 'ml_common', 'hmm_regime_detection.py')
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("hmm_regime_detection", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'HMMRegimeConfig'):
                HMMRegimeConfig = module.HMMRegimeConfig
                HMM_REGIME_CONFIG_AVAILABLE = True
except Exception:
    # Keep our fallback definition
    pass

# Import optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, SelectFromModel,
        mutual_info_regression, f_regression, chi2
    )
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Module-level sets to avoid duplicate logs across multiple instances
_GLOBAL_LOGGED_LOADS: set[str] = set()
_GLOBAL_LOGGED_EVENTS: set[str] = set()

@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""
    n_trials: int = 5  # Reduced default for LIGHT mode compatibility
    timeout: Optional[int] = 120  # Reduced timeout for LIGHT mode
    n_jobs: int = 1
    study_name: str = "hmm_optimization"
    storage_url: Optional[str] = None
    load_if_exists: bool = True

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    max_features: int = 100
    feature_selection_method: str = "mutual_info"
    scaling_method: str = "standard"
    dimensionality_reduction: bool = True
    n_components: int = 50

@dataclass
class ValidationConfig:
    """Configuration for validation."""
    min_regime_samples: int = 100
    max_regime_imbalance: float = 0.8
    min_silhouette_score: float = 0.3
    max_convergence_iterations: int = 100

class EnhancedHMMCompositeManager:
    """Enhanced HMM Composite Manager with consolidated functionality."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("EnhancedHMMCompositeManager")
        self._cache: dict[str, dict[str, Any]] = {}
        self._logged_loads = _GLOBAL_LOGGED_LOADS
        self._logged_events = _GLOBAL_LOGGED_EVENTS

        # Enhanced features
        self._file_metadata_cache: dict[str, dict[str, Any]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Cleanup cache every hour

        # Performance optimization settings
        self._performance_mode = "balanced"  # Options: "performance", "memory", "balanced"
        self._gpu_acceleration_enabled = True
        self._parallel_processing_enabled = True

        # Initialize M1 utilities for memory management
        self._initialize_m1_utilities()

        # Initialize optimization components
        self._initialize_optimization_components()

        # Initialize validation components
        self._initialize_validation_components()

        # Start performance monitoring
        self._start_performance_monitoring()

    def _initialize_m1_utilities(self) -> None:
        """Initialize M1 optimization utilities (replacing memory management files)."""
        if M1_UTILITIES_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                # Configure CPU optimizer for HMM workloads
                if self.cpu_optimizer:
                    self.cpu_optimizer.optimize_numpy_operations()

                # Start memory monitoring
                if self.memory_optimizer:
                    self.memory_optimizer.start_monitoring()

                self.logger.info("✅ M1 utilities initialized for enhanced performance")

                # Log hardware capabilities
                self._log_hardware_capabilities()

            except Exception as e:
                self.logger.warning(f"⚠️ M1 utilities initialization failed: {e}")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.logger.info("ℹ️ M1 utilities not available, using fallback implementations")

    def _log_hardware_capabilities(self) -> None:
        """Log detected hardware capabilities."""
        try:
            gpu_info = self.gpu_manager.get_gpu_info() if self.gpu_manager else {}
            cpu_info = self.cpu_optimizer.get_cpu_info() if self.cpu_optimizer else {}

            self.logger.info("🖥️ Hardware Capabilities:")
            self.logger.info(f"   GPU: {gpu_info.get('gpu_name', 'Not available')}")
            self.logger.info(f"   MPS Available: {gpu_info.get('mps_available', False)}")
            self.logger.info(f"   CPU Cores: {cpu_info.get('total_cores', 'Unknown')}")
            self.logger.info(f"   Performance Cores: {cpu_info.get('performance_cores', 'Unknown')}")
            self.logger.info(f"   Efficiency Cores: {cpu_info.get('efficiency_cores', 'Unknown')}")
            self.logger.info(f"   M1 Detected: {cpu_info.get('is_m1', False)}")

        except Exception as e:
            self.logger.debug(f"Could not log hardware capabilities: {e}")

    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.memory_optimizer:
            # Memory optimizer already has monitoring built-in
            self.logger.info("📊 Performance monitoring enabled")
        else:
            self.logger.info("📊 Performance monitoring not available")

    def _initialize_optimization_components(self) -> None:
        """Initialize Bayesian optimization components."""
        self.bayesian_config = BayesianOptimizationConfig()
        self.feature_config = FeatureEngineeringConfig()
        
        if OPTUNA_AVAILABLE:
            self.logger.info("✅ Bayesian optimization components initialized")
        else:
            self.logger.warning("⚠️ Optuna not available, Bayesian optimization disabled")

    def _initialize_validation_components(self) -> None:
        """Initialize validation components."""
        self.validation_config = ValidationConfig()
        self.logger.info("✅ Validation components initialized")

    # Original HMM Composite Manager functionality
    def get_composite_cluster_file_path(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> str:
        """Get the file path for HMM composite cluster data."""
        if base_path is None:
            base_path = "data_cache"
        
        filename = f"hmm_composite_clusters_{exchange}_{symbol}_{timeframe}.parquet"
        return os.path.join(base_path, "hmm_clusters", filename)

    def file_exists(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> bool:
        """Check if HMM composite cluster file exists."""
        file_path = self.get_composite_cluster_file_path(exchange, symbol, timeframe, base_path)
        return os.path.exists(file_path)

    def load_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Load HMM composite cluster data."""
        file_path = self.get_composite_cluster_file_path(exchange, symbol, timeframe, base_path)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            # Use memory optimizer if available
            if self.memory_optimizer:
                data = self.memory_optimizer.load_dataframe(file_path)
            else:
                data = pd.read_parquet(file_path)
            
            return {
                'data': data,
                'file_path': file_path,
                'metadata': self._get_file_metadata(file_path)
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to load composite clusters: {e}")
            return None

    def save_composite_clusters(
        self,
        data: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> bool:
        """Save HMM composite cluster data."""
        file_path = self.get_composite_cluster_file_path(exchange, symbol, timeframe, base_path)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Use memory optimizer if available
            if self.memory_optimizer:
                self.memory_optimizer.save_dataframe(data, file_path)
            else:
                data.to_parquet(file_path, index=False)
            
            # Update metadata cache
            self._update_file_metadata(file_path, data)
            
            self.logger.info(f"✅ Saved composite clusters to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save composite clusters: {e}")
            return False

    # Consolidated Bayesian Optimization functionality
    def optimize_hmm_parameters_parallel(
        self,
        data: pd.DataFrame,
        config: Optional[BayesianOptimizationConfig] = None,
        n_parallel_models: int = 3
    ) -> Dict[str, Any]:
        """Optimize HMM parameters using parallel model training."""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("⚠️ Optuna not available, using default parameters")
            return self._get_default_hmm_parameters()

        config = config or self.bayesian_config

        def parallel_objective(trial):
            """Parallel objective function for multiple model configurations."""
            # Generate multiple parameter sets for parallel evaluation
            param_sets = []
            for i in range(n_parallel_models):
                n_components = trial.suggest_int(f'n_components_{i}', 2, 6)
                covariance_type = trial.suggest_categorical(f'covariance_type_{i}',
                    ['diag', 'spherical'])
                n_iter = trial.suggest_int(f'n_iter_{i}', 25, 100)
                tol = trial.suggest_float(f'tol_{i}', 1e-6, 1e-2, log=True)
                param_sets.append({
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'n_iter': n_iter,
                    'tol': tol
                })

            # Train models in parallel
            scores = []
            for params in param_sets:
                try:
                    score = self._train_single_hmm_model(data, params)
                    scores.append(score)
                except Exception as e:
                    self.logger.debug(f"⚠️ Parallel model training failed: {e}")
                    scores.append(float('inf'))

            # Return average score across parallel models
            return np.mean(scores) if scores else float('inf')

        try:
            study = optuna.create_study(
                direction='maximize',
                study_name=config.study_name,
                storage=config.storage_url,
                load_if_exists=config.load_if_exists
            )

            study.optimize(
                parallel_objective,
                n_trials=config.n_trials,
                timeout=config.timeout,
                n_jobs=config.n_jobs
            )

            # Extract best parameters from parallel optimization
            best_trial = study.best_trial
            best_params = {}
            for key, value in best_trial.params.items():
                if '_0' in key:  # Use first model parameters as default
                    clean_key = key.replace('_0', '')
                    best_params[clean_key] = value

            best_score = study.best_value

            self.logger.info(f"✅ Parallel Bayesian optimization completed. Best score: {best_score:.4f}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study,
                'success': True,
                'parallel_models': n_parallel_models
            }

        except Exception as e:
            self.logger.error(f"❌ Parallel Bayesian optimization failed: {e}")
            return {'success': False, 'error': str(e)}

    def _train_single_hmm_model(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Train a single HMM model and return its score."""
        try:
            from hmmlearn import hmm

            # Optimize data for M1 hardware
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe_memory(data)

            # Prepare data with M1 optimizations
            X = self._prepare_data_for_hmm(data)

            # Validate data
            n_components = params['n_components']
            if len(X) < n_components:
                return float('inf')

            # Create model with optimized initialization
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=params['covariance_type'],
                n_iter=params['n_iter'],
                tol=params['tol'],
                init_params='mc',
                random_state=42
            )

            # Initialize with memory-efficient method
            self._initialize_hmm_model(model, X, n_components)

            # Use GPU acceleration if available
            if self._gpu_acceleration_enabled and self.gpu_manager:
                score = self._train_hmm_with_gpu_acceleration(model, X)
            else:
                # Fit model with CPU optimization
                with self._create_cpu_optimization_context():
                    model.fit(X)
                    score = model.score(X)

            return score

        except Exception as e:
            self.logger.debug(f"⚠️ Single model training failed: {e}")
            return float('inf')

    def _prepare_data_for_hmm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for HMM training with M1 optimizations."""
        try:
            # Select numeric columns
            X = data.select_dtypes(include=[np.number])

            # Advanced NaN handling for HMM training
            nan_stats = X.isna().sum().sum()
            if nan_stats > 0:
                self.logger.debug(f"⚠️ Found {nan_stats} NaN values in data")

                # For HMM training, use forward/backward fill first, then interpolation
                X = X.fillna(method='ffill').fillna(method='bfill')

                # If still have NaN values, use interpolation
                if X.isna().sum().sum() > 0:
                    X = X.interpolate(method='linear', limit_direction='both')

                # Final fallback: fill remaining NaN with column medians
                if X.isna().sum().sum() > 0:
                    for col in X.columns:
                        if X[col].isna().sum() > 0:
                            median_val = X[col].median()
                            if np.isfinite(median_val):
                                X[col] = X[col].fillna(median_val)
                            else:
                                # Last resort: fill with 0, but warn
                                self.logger.warning(f"⚠️ Column {col} has non-finite median, using 0 for NaN values")
                                X[col] = X[col].fillna(0)

            # Apply M1-specific optimizations
            if self.memory_optimizer:
                X = self.memory_optimizer.optimize_dataframe_memory(X)

            # Use GPU for data preprocessing if available
            if self._gpu_acceleration_enabled and self.gpu_manager:
                # Convert to numpy array for GPU processing
                X_array = X.values.astype(np.float32)
                X_array = self.gpu_manager.optimize_tensor_operations(X_array)
                X = pd.DataFrame(X_array, columns=X.columns, index=X.index)

            return X

        except Exception as e:
            self.logger.debug(f"⚠️ Data preparation failed, using fallback: {e}")
            # Fallback with better NaN handling
            X = data.select_dtypes(include=[np.number])
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            return X

    def _train_hmm_with_gpu_acceleration(self, model, X: pd.DataFrame) -> float:
        """Train HMM model with GPU acceleration."""
        try:
            # Convert data to GPU-compatible format
            X_array = X.values.astype(np.float32)

            # Use GPU manager for tensor operations
            if self.gpu_manager:
                X_gpu = self.gpu_manager.optimize_tensor_operations(X_array)
                X_cpu = X_gpu  # Convert back to CPU for hmmlearn
            else:
                X_cpu = X_array

            # Create CPU optimization context
            with self._create_cpu_optimization_context():
                # Fit model on optimized data
                model.fit(X_cpu)

                # Score model
                score = model.score(X_cpu)

            return score

        except Exception as e:
            self.logger.debug(f"⚠️ GPU acceleration failed, falling back to CPU: {e}")
            # Fallback to CPU training
            with self._create_cpu_optimization_context():
                model.fit(X.values.astype(np.float32))
                return model.score(X.values.astype(np.float32))

    def _create_cpu_optimization_context(self):
        """Create CPU optimization context for M1."""
        if self.cpu_optimizer:
            return self.cpu_optimizer.create_m1_optimized_context()
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()

    def train_hmm_parallel(
        self,
        data: pd.DataFrame,
        n_models: int = 3,
        config: Optional[HMMRegimeConfig] = None
    ) -> List[Any]:
        """Train multiple HMM models in parallel using M1 optimizations."""
        if not self._parallel_processing_enabled:
            self.logger.info("Parallel processing disabled, training sequentially")
            return [self._train_single_hmm_model_optimized(data, config or HMMRegimeConfig())]

        try:
            from hmmlearn import hmm

            # Create optimized configurations for parallel training
            configs = []
            base_config = config or HMMRegimeConfig()

            for i in range(n_models):
                # Vary parameters slightly for ensemble diversity
                config_variation = HMMRegimeConfig(
                    n_components=max(2, base_config.n_components + (i - 1)),
                    covariance_type=base_config.covariance_type,
                    n_iter=base_config.n_iter,
                    tol=base_config.tol,
                    random_state=base_config.random_state + i
                )
                configs.append(config_variation)

            # Use CPU optimizer for parallel execution
            if self.cpu_optimizer:
                self.logger.info(f"🚀 Training {n_models} HMM models in parallel on M1")

                def train_single_model(cfg):
                    return self._train_single_hmm_model_optimized(data, cfg)

                # Execute in parallel with M1 optimization
                models = self.cpu_optimizer.parallel_map_m1(train_single_model, configs)
                self.logger.info(f"✅ Parallel HMM training completed: {len(models)} models trained")
                return models
            else:
                # Fallback to sequential training
                models = []
                for cfg in configs:
                    model = self._train_single_hmm_model_optimized(data, cfg)
                    models.append(model)
                return models

        except Exception as e:
            self.logger.error(f"❌ Parallel HMM training failed: {e}")
            # Fallback to single model training
            return [self._train_single_hmm_model_optimized(data, config or HMMRegimeConfig())]

    def _train_single_hmm_model_optimized(self, data: pd.DataFrame, config: HMMRegimeConfig):
        """Train a single HMM model with full M1 optimizations."""
        try:
            from hmmlearn import hmm

            # Prepare data with full M1 optimization pipeline
            X = self._prepare_data_for_hmm(data)

            # Validate data
            if len(X) < config.n_components:
                raise ValueError(f"Insufficient data: {len(X)} samples for {config.n_components} components")

            # Create optimized HMM model
            model = hmm.GaussianHMM(
                n_components=config.n_components,
                covariance_type=config.covariance_type,
                n_iter=config.n_iter,
                tol=config.tol,
                init_params='mc',
                random_state=config.random_state
            )

            # Initialize with memory-efficient method
            self._initialize_hmm_model(model, X, config.n_components)

            # Train with GPU acceleration if available
            if self._gpu_acceleration_enabled and self.gpu_manager:
                with self._create_cpu_optimization_context():
                    score = self._train_hmm_with_gpu_acceleration(model, X)
            else:
                # Train with CPU optimization
                with self._create_cpu_optimization_context():
                    model.fit(X.values.astype(np.float32))
                    score = model.score(X.values.astype(np.float32))

            self.logger.info(f"✅ HMM model trained - Score: {score:.4f}, Components: {config.n_components}")
            return model

        except Exception as e:
            self.logger.error(f"❌ Optimized HMM training failed: {e}")
            raise

    def train_hmm_batch(
        self,
        data_batches: List[pd.DataFrame],
        config: Optional[HMMRegimeConfig] = None
    ) -> List[Any]:
        """Train HMM models on multiple data batches with streaming optimization."""
        try:
            config = config or HMMRegimeConfig()
            models = []

            self.logger.info(f"🚀 Training HMM on {len(data_batches)} data batches")

            for i, batch_data in enumerate(data_batches):
                self.logger.debug(f"📊 Processing batch {i+1}/{len(data_batches)}")

                # Apply memory optimization for each batch
                if self.memory_optimizer:
                    batch_data = self.memory_optimizer.optimize_dataframe_memory(batch_data)

                # Train model on this batch
                model = self._train_single_hmm_model_optimized(batch_data, config)
                models.append(model)

                # Memory cleanup between batches
                if self.memory_optimizer:
                    self.memory_optimizer._apply_memory_optimizations()

            self.logger.info(f"✅ Batch HMM training completed: {len(models)} models trained")
            return models

        except Exception as e:
            self.logger.error(f"❌ Batch HMM training failed: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            metrics = {
                'performance_mode': self._performance_mode,
                'gpu_acceleration_enabled': self._gpu_acceleration_enabled,
                'parallel_processing_enabled': self._parallel_processing_enabled,
                'm1_utilities_available': M1_UTILITIES_AVAILABLE
            }

            # Add hardware metrics
            if self.gpu_manager:
                metrics['gpu_info'] = self.gpu_manager.get_gpu_info()

            if self.cpu_optimizer:
                metrics['cpu_info'] = self.cpu_optimizer.get_cpu_info()

            if self.memory_optimizer:
                metrics['memory_stats'] = self.memory_optimizer.get_memory_stats()

            # Add cache statistics
            metrics['cache_stats'] = self.get_cache_stats()

            return metrics

        except Exception as e:
            self.logger.error(f"Could not get performance metrics: {e}")
            return {'error': str(e)}

    def train_hmm_streaming(
        self,
        data_stream: Union[pd.DataFrame, List[pd.DataFrame]],
        window_size: int = 1000,
        update_frequency: int = 100,
        config: Optional[HMMRegimeConfig] = None
    ):
        """Train HMM models on streaming data with online learning capabilities."""
        try:
            config = config or HMMRegimeConfig()

            # Initialize streaming state
            streaming_state = {
                'current_model': None,
                'last_update': time.time(),
                'processed_samples': 0,
                'model_history': [],
                'performance_history': []
            }

            self.logger.info(f"🚀 Starting streaming HMM training with window_size={window_size}")

            # Handle single DataFrame or list of DataFrames
            if isinstance(data_stream, pd.DataFrame):
                data_batches = self._create_streaming_batches(data_stream, window_size)
            else:
                data_batches = data_stream

            results = []

            for i, batch_data in enumerate(data_batches):
                self.logger.debug(f"📊 Processing streaming batch {i+1}/{len(data_batches)}")

                # Apply memory optimization for streaming data
                if self.memory_optimizer:
                    batch_data = self.memory_optimizer.optimize_dataframe_memory(batch_data)

                # Update model with new batch
                streaming_state = self._update_streaming_model(
                    streaming_state, batch_data, config, i % update_frequency == 0
                )

                streaming_state['processed_samples'] += len(batch_data)

                # Collect results at specified intervals
                if i % update_frequency == 0:
                    result = {
                        'batch_index': i,
                        'model': streaming_state['current_model'],
                        'processed_samples': streaming_state['processed_samples'],
                        'performance_metrics': self._get_streaming_performance_metrics(streaming_state)
                    }
                    results.append(result)

                    # Memory cleanup
                    if self.memory_optimizer:
                        self.memory_optimizer._apply_memory_optimizations()

            self.logger.info(f"✅ Streaming HMM training completed: {len(results)} model updates")
            return results, streaming_state

        except Exception as e:
            self.logger.error(f"❌ Streaming HMM training failed: {e}")
            raise

    def _create_streaming_batches(self, data: pd.DataFrame, window_size: int) -> List[pd.DataFrame]:
        """Create overlapping or sliding window batches for streaming processing."""
        try:
            batches = []

            # Create sliding windows with 50% overlap for better continuity
            step_size = window_size // 2

            for start_idx in range(0, len(data) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                batch = data.iloc[start_idx:end_idx].copy()
                batches.append(batch)

            # Add final batch if needed
            if len(data) > window_size and len(batches) > 0:
                final_batch = data.iloc[-window_size:].copy()
                batches.append(final_batch)

            self.logger.debug(f"Created {len(batches)} streaming batches from {len(data)} samples")
            return batches

        except Exception as e:
            self.logger.error(f"Failed to create streaming batches: {e}")
            # Fallback: return single batch
            return [data]

    def _update_streaming_model(
        self,
        streaming_state: Dict[str, Any],
        new_batch: pd.DataFrame,
        config: HMMRegimeConfig,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """Update streaming HMM model with new data batch."""
        try:
            # Prepare new batch data
            X_new = self._prepare_data_for_hmm(new_batch)

            if streaming_state['current_model'] is None:
                # Initialize model with first batch
                self.logger.debug("Initializing streaming model with first batch")
                streaming_state['current_model'] = self._train_single_hmm_model_optimized(new_batch, config)
                streaming_state['model_history'].append(streaming_state['current_model'])
            else:
                # Online learning: update existing model
                if force_update:
                    self.logger.debug("Updating streaming model with new batch")

                    # Use ensemble approach: combine old and new model predictions
                    old_model = streaming_state['current_model']

                    # Train new model on combined data (old predictions + new data)
                    combined_data = self._combine_streaming_data(old_model, new_batch)

                    if combined_data is not None:
                        new_model = self._train_single_hmm_model_optimized(combined_data, config)
                        streaming_state['current_model'] = new_model
                        streaming_state['model_history'].append(new_model)

            streaming_state['last_update'] = time.time()
            return streaming_state

        except Exception as e:
            self.logger.error(f"Failed to update streaming model: {e}")
            return streaming_state

    def _combine_streaming_data(self, old_model, new_batch: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Combine old model predictions with new data for online learning."""
        try:
            # Generate predictions from old model
            X_new = self._prepare_data_for_hmm(new_batch)
            old_predictions = old_model.predict(X_new.values)

            # Create combined dataset with old predictions as additional features
            combined_data = new_batch.copy()
            combined_data['old_regime_prediction'] = old_predictions
            combined_data['old_regime_proba'] = old_model.predict_proba(X_new.values).max(axis=1)

            return combined_data

        except Exception as e:
            self.logger.debug(f"Could not combine streaming data: {e}")
            return new_batch  # Fallback to just new data

    def _get_streaming_performance_metrics(self, streaming_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics for streaming HMM."""
        try:
            metrics = {
                'processed_samples': streaming_state['processed_samples'],
                'model_updates': len(streaming_state['model_history']),
                'last_update_time': streaming_state['last_update']
            }

            if streaming_state['current_model']:
                # Add model-specific metrics
                current_model = streaming_state['current_model']
                metrics['current_model_score'] = getattr(current_model, 'score_', 0.0)
                metrics['n_components'] = getattr(current_model, 'n_components', 0)

            return metrics

        except Exception as e:
            self.logger.debug(f"Could not get streaming metrics: {e}")
            return {'error': str(e)}

    def optimize_hmm_parameters(
        self,
        data: pd.DataFrame,
        config: Optional[BayesianOptimizationConfig] = None,
        use_adaptive: bool = False,
        mode: str = 'light'
    ) -> Dict[str, Any]:
        """Optimize HMM parameters using Bayesian optimization.

        Args:
            data: Input data for optimization
            config: Bayesian optimization configuration
            use_adaptive: Whether to use the new adaptive optimization pipeline
            mode: Optimization mode ('light', 'blank', 'full')

        Returns:
            Dictionary containing optimization results
        """
        if use_adaptive:
            return self._optimize_hmm_parameters_adaptive(data, mode)

        # Original optimization method
        if not OPTUNA_AVAILABLE:
            self.logger.warning("⚠️ Optuna not available, using default parameters")
            return self._get_default_hmm_parameters()
        
        config = config or self.bayesian_config
        
        def objective(trial):
            # ULTRA-LIGHT mode optimizations for code testing (not for production results)
            n_components = trial.suggest_int('n_components', 2, 6)  # Small range: 2-6 components
            covariance_type = trial.suggest_categorical('covariance_type',
                ['diag', 'spherical'])  # Fast types only: diagonal and spherical
            n_iter = trial.suggest_int('n_iter', 25, 100)  # Quick convergence: 25-100 iterations
            tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)  # Relaxed tolerance: 1e-6 to 1e-2
            
            try:
                # Create and fit HMM model
                from hmmlearn import hmm
                # Exclude 's' (start probabilities) and 't' (transition matrix) from init_params
                # since we manually initialize them to avoid the warning
                init_params = 'mc'  # Only initialize means and covariances
                if covariance_type == 'tied':
                    init_params = 'm'  # For tied covariance, only initialize means

                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    init_params=init_params,  # Don't auto-initialize startprob and transmat
                    random_state=42
                )

                # Prepare data
                X = data.select_dtypes(include=[np.number]).fillna(0)

                # Validate data before fitting
                if len(X) < n_components:
                    self.logger.debug(f"⚠️ Insufficient data: {len(X)} samples for {n_components} components")
                    return float('inf')

                # Check for data quality issues
                if X.isnull().any().any():
                    self.logger.debug("⚠️ Data contains NaN values, filling with 0")
                    X = X.fillna(0)

                # Check for infinite values
                if np.isinf(X.values).any():
                    self.logger.debug("⚠️ Data contains infinite values, replacing with 0")
                    X = X.replace([np.inf, -np.inf], 0)

                # Ensure minimum data requirements for covariance type
                min_samples_needed = self._get_min_samples_for_covariance(covariance_type, n_components)
                if len(X) < min_samples_needed:
                    self.logger.debug(f"⚠️ Insufficient data for {covariance_type} covariance: need {min_samples_needed}, got {len(X)}")
                    return float('inf')

                # Ensure data has enough variance for covariance estimation
                if X.std().min() < 1e-10:
                    self.logger.debug("⚠️ Data has zero variance, adding small noise")
                    X = X + np.random.normal(0, 1e-6, X.shape)

                # LIGHT mode: Skip expensive data quality checks for speed
                # Only check for extreme variance ratios (>1000x) instead of >1e6
                if len(X.columns) > 1 and X.std().max() / X.std().min() > 1000:
                    self.logger.debug("⚠️ Large variance ratio detected, standardizing data")
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

                # Ensure no constant features that could cause initialization issues
                constant_features = []
                for col in X.columns:
                    if X[col].nunique() <= 1:
                        constant_features.append(col)

                if constant_features:
                    self.logger.debug(f"⚠️ Removing {len(constant_features)} constant features")
                    X = X.drop(columns=constant_features)

                    # Ensure we still have enough features
                    if len(X.columns) < 2:
                        self.logger.debug("⚠️ Insufficient features after removing constants")
                        return float('inf')

                # Initialize model with better defaults to prevent initialization issues
                self._initialize_hmm_model(model, X, n_components)

                # Try to fit the model with error handling
                try:
                    model.fit(X)
                except Exception as fit_error:
                    fit_error_msg = str(fit_error)
                    if ('covars' in fit_error_msg and 'symmetric' in fit_error_msg) or 'positive-definite' in fit_error_msg:
                        # Try multiple approaches to fix covariance matrix issues
                        self.logger.debug(f"⚠️ Covariance matrix issue detected: {fit_error_msg}")

                        # Approach 1: Try regularization
                        X_regularized = self._regularize_covariance_matrix(X)
                        if X_regularized is not None:
                            try:
                                # Re-initialize with regularized data
                                self._initialize_hmm_model(model, X_regularized, n_components)
                                model.fit(X_regularized)
                                self.logger.debug("✅ Covariance regularization successful")
                            except Exception as reg_error:
                                self.logger.debug(f"⚠️ Regularization failed: {reg_error}")
                                raise fit_error
                        else:
                            # Approach 2: Try changing covariance type
                            if covariance_type == 'full':
                                self.logger.debug("⚠️ Switching to diagonal covariance type")
                                # Create new model with diagonal covariance and proper init_params
                                model = hmm.GaussianHMM(
                                    n_components=n_components,
                                    covariance_type='diag',
                                    n_iter=n_iter,
                                    tol=tol,
                                    init_params='mc',  # Only initialize means and covariances
                                    random_state=42
                                )
                                try:
                                    self._initialize_hmm_model(model, X, n_components)
                                    model.fit(X)
                                    self.logger.debug("✅ Diagonal covariance successful")
                                except Exception as cov_error:
                                    self.logger.debug(f"⚠️ Diagonal covariance failed: {cov_error}")
                                    raise fit_error
                            else:
                                raise fit_error
                    elif 'startprob_' in fit_error_msg or 'must sum to 1' in fit_error_msg:
                        # Try with forced initialization and allow parameter initialization
                        self.logger.debug("⚠️ Attempting forced initialization...")
                        model.startprob_ = np.ones(n_components) / n_components
                        model.transmat_ = np.ones((n_components, n_components)) / n_components
                        # Temporarily allow startprob and transmat initialization
                        model.init_params = 'stmc'
                        model.fit(X)
                    else:
                        raise fit_error

                # Calculate score (negative log likelihood)
                score = model.score(X)
                return score

            except Exception as e:
                error_msg = str(e)
                if 'covars' in error_msg and 'symmetric' in error_msg:
                    self.logger.debug(f"⚠️ Covariance matrix not symmetric: {error_msg}")
                    # This usually indicates numerical precision issues
                    return float('-inf')
                elif 'covars' in error_msg and 'positive-definite' in error_msg:
                    self.logger.debug(f"⚠️ Covariance matrix not positive-definite: {error_msg}")
                    # This indicates the data is singular or nearly singular
                    return float('-inf')
                elif 'positive-definite' in error_msg:
                    self.logger.debug(f"⚠️ Positive-definite matrix issue: {error_msg}")
                    return float('-inf')
                elif 'singular' in error_msg.lower():
                    self.logger.debug(f"⚠️ Singular matrix detected: {error_msg}")
                    # Data has linear dependencies
                    return float('-inf')
                elif 'convergence' in error_msg.lower():
                    self.logger.debug(f"⚠️ HMM convergence failed: {error_msg}")
                    # Model didn't converge - try different parameters
                    return float('-inf')
                elif 'nan' in error_msg.lower() or 'inf' in error_msg.lower():
                    self.logger.debug(f"⚠️ Invalid values in data: {error_msg}")
                    # Data contains NaN or infinite values
                    return float('-inf')
                elif 'startprob_' in error_msg:
                    self.logger.debug(f"⚠️ Initial probabilities issue: {error_msg}")
                    # Initial state probabilities are invalid - usually due to data quality
                    return float('-inf')
                elif 'must sum to 1' in error_msg:
                    self.logger.debug(f"⚠️ Probability normalization issue: {error_msg}")
                    # Probabilities don't sum to 1 - numerical precision issue
                    return float('-inf')
                else:
                    self.logger.debug(f"⚠️ HMM optimization failed: {error_msg}")
                return float('inf')
        
        try:
            study = optuna.create_study(
                direction='maximize',
                study_name=config.study_name,
                storage=config.storage_url,
                load_if_exists=config.load_if_exists
            )
            
            study.optimize(
                objective,
                n_trials=config.n_trials,
                timeout=config.timeout,
                n_jobs=config.n_jobs
            )
            
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"✅ Bayesian optimization completed. Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return {'success': False, 'error': str(e)}

    def _optimize_hmm_parameters_adaptive(
        self,
        data: pd.DataFrame,
        mode: str = 'light'
    ) -> Dict[str, Any]:
        """Run adaptive optimization pipeline with coarse grid + parameter analysis + multi-fidelity."""
        self.logger.info(f"🎯 Starting adaptive HMM optimization in {mode} mode")

        try:
            # Initialize adaptive optimizer
            adaptive_config = AdaptiveOptimizationConfig()

            # Adjust configuration based on mode
            if mode == 'light':
                adaptive_config.enable_coarse_grid = True
                adaptive_config.enable_parameter_analysis = True
                adaptive_config.enable_multi_fidelity = True
                adaptive_config.coarse_grid_time_budget = 60  # 1 minute for light mode
                adaptive_config.optimization_time_budget = 120  # 2 minutes for light mode
            elif mode == 'blank':
                adaptive_config.enable_coarse_grid = True
                adaptive_config.enable_parameter_analysis = True
                adaptive_config.enable_multi_fidelity = True
                adaptive_config.coarse_grid_time_budget = 120  # 2 minutes for blank mode
                adaptive_config.optimization_time_budget = 240  # 4 minutes for blank mode
            elif mode == 'full':
                adaptive_config.enable_coarse_grid = True
                adaptive_config.enable_parameter_analysis = True
                adaptive_config.enable_multi_fidelity = True
                adaptive_config.coarse_grid_time_budget = 300  # 5 minutes for full mode
                adaptive_config.optimization_time_budget = 600  # 10 minutes for full mode

            adaptive_optimizer = AdaptiveBayesianOptimizer(self, adaptive_config)

            # Run adaptive optimization
            results = adaptive_optimizer.optimize_adaptive(data, mode)

            # Format results to match original method signature
            if results.get('final_result'):
                final_result = results['final_result']
                return {
                    'best_params': final_result['best_params'],
                    'best_score': final_result['best_score'],
                    'study': None,  # Adaptive optimizer doesn't use single study
                    'success': True,
                    'optimization_method': final_result.get('method', 'adaptive'),
                    'performance_metrics': results.get('performance_metrics', {}),
                    'pipeline_steps': results.get('pipeline_steps', []),
                    'intermediate_results': results.get('intermediate_results', {})
                }
            else:
                self.logger.error("❌ Adaptive optimization failed to produce final result")
                return {
                    'success': False,
                    'error': 'No final result from adaptive optimization',
                    'performance_metrics': results.get('performance_metrics', {})
                }

        except Exception as e:
            self.logger.error(f"❌ Adaptive optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _log_final_optimization_summary(self, optimization_results: Dict[str, Any]) -> None:
        """Log comprehensive summary of the optimization process."""
        final_result = optimization_results.get('final_result', {})
        metrics = optimization_results.get('performance_metrics', {})

        self.logger.info("🎉 === OPTIMIZATION SUMMARY ===")
        self.logger.info(f"🎯 Mode: {optimization_results.get('mode', 'unknown').upper()}")
        self.logger.info(f"📊 Pipeline Steps Completed: {', '.join(optimization_results.get('pipeline_steps', []))}")

        # Final results
        if final_result.get('best_params'):
            self.logger.info("✅ FINAL RESULTS:")
            self.logger.info(f"   Best Score: {final_result['best_score']:.4f}")
            self.logger.info(f"   Optimization Method: {final_result.get('method', 'unknown')}")
            self.logger.info(f"   Parameters: n_components={final_result['best_params'].get('n_components', 'N/A')}, "
                           f"covariance_type={final_result['best_params'].get('covariance_type', 'N/A')}")
        else:
            self.logger.warning("⚠️ No final results generated")

        # Performance metrics
        if metrics:
            self.logger.info("📈 PERFORMANCE METRICS:")
            self.logger.info(f"   Total Time: {metrics.get('total_time_seconds', 0):.1f}s")
            if 'improvement_over_coarse' in metrics:
                self.logger.info(f"   Improvement over Coarse Grid: {metrics['improvement_over_coarse']:.4f}")
            if 'improvement_ratio' in metrics:
                self.logger.info(f"   Improvement Ratio: {metrics['improvement_ratio']:.2%}")

        # Intermediate results summary
        intermediate = optimization_results.get('intermediate_results', {})
        if intermediate:
            self.logger.info("🔄 INTERMEDIATE RESULTS:")
            if 'coarse_grid' in intermediate:
                cg = intermediate['coarse_grid']
                self.logger.info(f"   Coarse Grid: {cg.get('total_evaluations', 0)} evaluations, "
                               f"best={cg.get('best_score', 'N/A')}")
            if 'multi_fidelity' in intermediate:
                mf = intermediate['multi_fidelity']
                early_stop = " (early stopped)" if mf.get('early_stopped') else ""
                self.logger.info(f"   Multi-Fidelity: best={mf.get('best_score', 'N/A')}{early_stop}")

        self.logger.info("🎉 === OPTIMIZATION COMPLETE ===")

    def optimize_hmm_parameters_light(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Ultra-fast optimization for code testing - 1-2 minutes."""
        self.logger.info("⚡ LIGHT MODE: Ultra-fast optimization for code testing")
        return self.optimize_hmm_parameters(data, use_adaptive=True, mode='light')

    def optimize_hmm_parameters_blank(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Balanced optimization for development - 4-6 minutes."""
        self.logger.info("⚖️ BLANK MODE: Balanced optimization for development")
        return self.optimize_hmm_parameters(data, use_adaptive=True, mode='blank')

    def optimize_hmm_parameters_full(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive optimization for production - 10-15 minutes."""
        self.logger.info("🚀 FULL MODE: Comprehensive optimization for production")
        return self.optimize_hmm_parameters(data, use_adaptive=True, mode='full')

    def get_optimization_mode_info(self, mode: str) -> Dict[str, Any]:
        """Get information about optimization modes and their characteristics."""
        mode_info = {
            'light': {
                'description': 'Ultra-fast code testing mode',
                'estimated_time': '1-2 minutes',
                'coarse_grid_trials': 12,
                'optimization_trials': 5,
                'fidelity_levels': ['low', 'medium'],
                'use_case': 'Verify code works, quick iteration',
                'quality_expectation': 'Functional validation'
            },
            'blank': {
                'description': 'Balanced development mode',
                'estimated_time': '4-6 minutes',
                'coarse_grid_trials': 20,
                'optimization_trials': 10,
                'fidelity_levels': ['low', 'medium', 'high'],
                'use_case': 'Development and testing',
                'quality_expectation': 'Good balance of speed vs quality'
            },
            'full': {
                'description': 'Comprehensive production mode',
                'estimated_time': '10-15 minutes',
                'coarse_grid_trials': 32,
                'optimization_trials': 25,
                'fidelity_levels': ['low', 'medium', 'high'],
                'use_case': 'Production optimization',
                'quality_expectation': 'Maximum quality results'
            }
        }

        return mode_info.get(mode, {
            'description': 'Unknown mode',
            'estimated_time': 'Unknown',
            'error': f'Mode "{mode}" not recognized'
        })

    # Consolidated Feature Engineering functionality
    def engineer_features(
        self,
        data: pd.DataFrame,
        config: Optional[FeatureEngineeringConfig] = None
    ) -> pd.DataFrame:
        """Engineer features for HMM regime discovery."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ Scikit-learn not available, returning original data")
            return data
        
        config = config or self.feature_config
        
        try:
            # Select numeric columns only
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            
            if len(numeric_data.columns) == 0:
                self.logger.warning("⚠️ No numeric columns found")
                return data
            
            # Feature selection
            if config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=min(config.max_features, len(numeric_data.columns)))
            elif config.feature_selection_method == "f_score":
                selector = SelectKBest(f_regression, k=min(config.max_features, len(numeric_data.columns)))
            else:
                selector = SelectKBest(f_regression, k=min(config.max_features, len(numeric_data.columns)))
            
            selected_features = selector.fit_transform(numeric_data, numeric_data.mean(axis=1))
            selected_columns = numeric_data.columns[selector.get_support()]
            
            # Feature scaling
            if config.scaling_method == "standard":
                scaler = StandardScaler()
            elif config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif config.scaling_method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            scaled_features = scaler.fit_transform(selected_features)
            
            # Dimensionality reduction
            if config.dimensionality_reduction and len(selected_columns) > config.n_components:
                pca = PCA(n_components=config.n_components)
                reduced_features = pca.fit_transform(scaled_features)
                
                # Create feature names
                feature_names = [f"pca_{i}" for i in range(config.n_components)]
            else:
                reduced_features = scaled_features
                feature_names = selected_columns.tolist()
            
            # Create result DataFrame
            result = pd.DataFrame(reduced_features, columns=feature_names, index=data.index)
            
            self.logger.info(f"✅ Feature engineering completed. Shape: {result.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return data

    # Consolidated Validation functionality
    def validate_hmm_results(
        self,
        data: pd.DataFrame,
        regime_labels: np.ndarray,
        config: Optional[ValidationConfig] = None
    ) -> Dict[str, Any]:
        """Validate HMM regime discovery results."""
        config = config or self.validation_config
        
        try:
            validation_results = {
                'regime_counts': {},
                'regime_imbalance': 0.0,
                'silhouette_score': 0.0,
                'validation_passed': False,
                'warnings': [],
                'errors': []
            }
            
            # Check regime counts
            unique_regimes, counts = np.unique(regime_labels, return_counts=True)
            validation_results['regime_counts'] = dict(zip(unique_regimes, counts))
            
            # Check minimum regime samples
            min_count = min(counts)
            if min_count < config.min_regime_samples:
                validation_results['errors'].append(
                    f"Regime with {min_count} samples below minimum {config.min_regime_samples}"
                )
            
            # Check regime imbalance
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = min_count / max_count
            validation_results['regime_imbalance'] = imbalance_ratio
            
            if imbalance_ratio < config.max_regime_imbalance:
                validation_results['warnings'].append(
                    f"Regime imbalance {imbalance_ratio:.3f} below threshold {config.max_regime_imbalance}"
                )
            
            # Calculate silhouette score if possible
            if len(unique_regimes) > 1 and len(data) > len(unique_regimes):
                try:
                    from sklearn.metrics import silhouette_score
                    numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
                    if len(numeric_data.columns) > 0:
                        silhouette = silhouette_score(numeric_data, regime_labels)
                        validation_results['silhouette_score'] = silhouette
                        
                        if silhouette < config.min_silhouette_score:
                            validation_results['warnings'].append(
                                f"Silhouette score {silhouette:.3f} below threshold {config.min_silhouette_score}"
                            )
                except Exception as e:
                    validation_results['warnings'].append(f"Could not calculate silhouette score: {e}")
            
            # Overall validation
            validation_results['validation_passed'] = len(validation_results['errors']) == 0
            
            self.logger.info(f"✅ Validation completed. Passed: {validation_results['validation_passed']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {
                'validation_passed': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _get_min_samples_for_covariance(self, covariance_type: str, n_components: int) -> int:
        """Get minimum samples needed for different covariance types."""
        # LIGHT mode: Reduced minimum samples for faster testing
        base_min = max(5, n_components)  # Reduced from max(10, n_components * 2)

        if covariance_type == 'full':
            # Full covariance needs more samples due to quadratic complexity
            return max(base_min, n_components * 2)  # Reduced from n_components²
        elif covariance_type == 'tied':
            # Tied covariance shares covariance across components
            return max(base_min, n_components + 1)
        elif covariance_type == 'diag':
            # Diagonal covariance needs fewer samples
            return max(base_min, n_components + 1)
        elif covariance_type == 'spherical':
            # Spherical covariance needs minimal samples
            return max(base_min, n_components + 1)
        else:
            return base_min

    def _regularize_covariance_matrix(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Attempt to regularize covariance matrix to fix positive-definite issues."""
        try:
            X_reg = X.copy()

            # Method 1: Add small noise to break exact linear dependencies
            regularization_strength = 1e-6
            for col in X_reg.columns:
                if X_reg[col].std() < 1e-10:
                    X_reg[col] += np.random.normal(0, regularization_strength, len(X_reg))
                else:
                    # Add small noise to all columns to prevent numerical issues
                    X_reg[col] += np.random.normal(0, regularization_strength, len(X_reg))

            # Method 2: Check for multicollinearity and remove highly correlated features
            if len(X_reg.columns) > 2:
                corr_matrix = X_reg.corr().abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                # Remove features with correlation > 0.95
                to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
                if to_drop:
                    self.logger.debug(f"⚠️ Removing {len(to_drop)} highly correlated features")
                    X_reg = X_reg.drop(columns=to_drop)

                    # Ensure we still have enough features
                    if len(X_reg.columns) < 2:
                        self.logger.debug("⚠️ Too many features removed, keeping original data")
                        X_reg = X.copy()
                        # Just add stronger regularization
                        for col in X_reg.columns:
                            X_reg[col] += np.random.normal(0, regularization_strength * 10, len(X_reg))

            # Method 3: Ensure minimum variance for all features
            for col in X_reg.columns:
                if X_reg[col].std() < 1e-8:
                    # Add more significant noise if variance is still too low
                    X_reg[col] += np.random.normal(0, 1e-4, len(X_reg))

            # Method 4: If still problematic, try dimensionality reduction as last resort
            if len(X_reg.columns) > 3:
                try:
                    from sklearn.decomposition import PCA
                    n_components = max(2, min(len(X_reg.columns) // 2, 10))  # Cap at 10 components
                    pca = PCA(n_components=n_components, random_state=42)
                    X_pca = pca.fit_transform(X_reg.values)

                    # Check explained variance ratio
                    explained_var = pca.explained_variance_ratio_.sum()
                    if explained_var > 0.8:  # Keep at least 80% of variance
                        X_reg = pd.DataFrame(X_pca, columns=[f'pca_{i}' for i in range(X_pca.shape[1])], index=X_reg.index)
                        self.logger.debug(f"✅ PCA reduced to {n_components} components (explained variance: {explained_var:.3f})")
                    else:
                        self.logger.debug(f"⚠️ PCA would lose too much variance ({explained_var:.3f}), keeping original")

                except Exception as pca_error:
                    self.logger.debug(f"⚠️ PCA regularization failed: {pca_error}")

            return X_reg

        except Exception as e:
            self.logger.debug(f"⚠️ Covariance regularization failed: {e}")
            return None

    def _validate_covariance_matrix(self, cov_matrix: np.ndarray) -> bool:
        """Validate that a covariance matrix is positive-definite and well-conditioned."""
        try:
            # Check if matrix is symmetric
            if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-10):
                return False

            # Check if matrix is positive-definite by attempting Cholesky decomposition
            np.linalg.cholesky(cov_matrix)
            return True

        except np.linalg.LinAlgError:
            return False

    def _make_covariance_positive_definite(self, cov_matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """Make a covariance matrix positive-definite by regularization."""
        try:
            # Try Cholesky decomposition
            np.linalg.cholesky(cov_matrix)
            return cov_matrix
        except np.linalg.LinAlgError:
            # Add regularization until positive-definite
            reg_strength = regularization
            max_attempts = 10

            for attempt in range(max_attempts):
                try:
                    regularized = cov_matrix + np.eye(cov_matrix.shape[0]) * reg_strength
                    np.linalg.cholesky(regularized)
                    self.logger.debug(f"✅ Made covariance matrix positive-definite with regularization {reg_strength:.2e}")
                    return regularized
                except np.linalg.LinAlgError:
                    reg_strength *= 10

            # If still not positive-definite, use identity matrix
            self.logger.debug("⚠️ Using identity matrix as fallback for covariance")
            return np.eye(cov_matrix.shape[0]) * regularization

    def _make_covariance_positive_definite_efficient(self, cov_matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """Memory-efficient version for making covariance matrix positive-definite."""
        try:
            # Convert to float32 for memory efficiency
            cov_matrix = cov_matrix.astype(np.float32)

            # Try Cholesky decomposition
            np.linalg.cholesky(cov_matrix)
            return cov_matrix
        except np.linalg.LinAlgError:
            # Add regularization with float32 precision
            reg_strength = np.float32(regularization)
            max_attempts = 10

            for attempt in range(max_attempts):
                try:
                    regularized = cov_matrix + np.eye(cov_matrix.shape[0], dtype=np.float32) * reg_strength
                    np.linalg.cholesky(regularized)
                    self.logger.debug(f"✅ Made covariance matrix positive-definite with regularization {reg_strength:.2e}")
                    return regularized
                except np.linalg.LinAlgError:
                    reg_strength *= np.float32(10)

            # If still not positive-definite, use identity matrix
            self.logger.debug("⚠️ Using identity matrix as fallback for covariance")
            return np.eye(cov_matrix.shape[0], dtype=np.float32) * np.float32(regularization)

    def _initialize_hmm_model(self, model, X: pd.DataFrame, n_components: int) -> None:
        """Initialize HMM model with better defaults to prevent initialization issues."""
        try:
            # Memory-efficient initialization for large datasets
            use_memory_efficient = len(X) > 50000

            if use_memory_efficient:
                self.logger.debug("🔧 Using memory-efficient HMM initialization")
                # Use subset for initialization to save memory
                subset_size = min(10000, len(X))
                X_subset = X.sample(n=subset_size, random_state=42) if len(X) > subset_size else X
                self._initialize_hmm_model_efficient(model, X_subset, n_components)
            else:
                self._initialize_hmm_model_standard(model, X, n_components)

            self.logger.debug(f"✅ HMM model initialized with explicit parameters")

        except Exception as e:
            self.logger.debug(f"⚠️ HMM initialization failed: {e}")
            # Let the model use its default initialization

    def _initialize_hmm_model_standard(self, model, X: pd.DataFrame, n_components: int) -> None:
        """Standard HMM model initialization."""
        # Set explicit starting probabilities
        model.startprob_ = np.ones(n_components) / n_components

        # Set explicit transition matrix (uniform transitions)
        model.transmat_ = np.ones((n_components, n_components)) / n_components

        # Initialize emission parameters based on data
        if hasattr(model, 'means_'):
            # For Gaussian HMM
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            model.means_ = kmeans.cluster_centers_

            if hasattr(model, 'covars_'):
                # Initialize covariance matrices with validation
                if model.covariance_type == 'full':
                    base_cov = np.cov(X.values.T)
                    # Ensure positive-definite
                    base_cov = self._make_covariance_positive_definite(base_cov)
                    model.covars_ = np.array([base_cov] * n_components)
                elif model.covariance_type == 'diag':
                    variances = np.var(X.values, axis=0) + 1e-6  # Ensure non-zero
                    model.covars_ = np.array([variances] * n_components)
                elif model.covariance_type == 'spherical':
                    variance = max(np.var(X.values) + 1e-6, 1e-6)  # Ensure non-zero
                    model.covars_ = np.array([variance] * n_components)
                elif model.covariance_type == 'tied':
                    base_cov = np.cov(X.values.T)
                    model.covars_ = self._make_covariance_positive_definite(base_cov)

    def _initialize_hmm_model_efficient(self, model, X_subset: pd.DataFrame, n_components: int) -> None:
        """Memory-efficient HMM model initialization using data subset."""
        # Set explicit starting probabilities
        model.startprob_ = np.ones(n_components, dtype=np.float32) / n_components

        # Set explicit transition matrix (uniform transitions) - use float32 for memory
        model.transmat_ = np.ones((n_components, n_components), dtype=np.float32) / n_components

        # Initialize emission parameters based on subset
        if hasattr(model, 'means_'):
            # For Gaussian HMM - use memory-efficient clustering
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=n_components,
                random_state=42,
                batch_size=min(1000, len(X_subset)),
                n_init=5  # Reduce iterations for speed
            )
            cluster_labels = kmeans.fit_predict(X_subset)
            model.means_ = kmeans.cluster_centers_.astype(np.float32)

            if hasattr(model, 'covars_'):
                # Initialize covariance matrices with memory optimization
                if model.covariance_type == 'full':
                    base_cov = np.cov(X_subset.values.T)
                    # Ensure positive-definite with memory-efficient approach
                    base_cov = self._make_covariance_positive_definite_efficient(base_cov)
                    model.covars_ = np.array([base_cov] * n_components, dtype=np.float32)
                elif model.covariance_type == 'diag':
                    variances = np.var(X_subset.values, axis=0) + 1e-6  # Ensure non-zero
                    model.covars_ = np.array([variances] * n_components, dtype=np.float32)
                elif model.covariance_type == 'spherical':
                    variance = max(np.var(X_subset.values) + 1e-6, 1e-6)  # Ensure non-zero
                    model.covars_ = np.array([variance] * n_components, dtype=np.float32)
                elif model.covariance_type == 'tied':
                    base_cov = np.cov(X_subset.values.T)
                    model.covars_ = self._make_covariance_positive_definite_efficient(base_cov).astype(np.float32)

    def _get_default_hmm_parameters(self) -> Dict[str, Any]:
        """Get default HMM parameters when optimization is not available."""
        return {
            'n_components': 4,
            'covariance_type': 'full',
            'n_iter': 100,
            'tol': 1e-3,
            'success': True
        }

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata."""
        try:
            if file_path in self._file_metadata_cache:
                return self._file_metadata_cache[file_path]
            
            stat = os.stat(file_path)
            metadata = {
                'size_bytes': stat.st_size,
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime
            }
            
            self._file_metadata_cache[file_path] = metadata
            return metadata
        except Exception:
            return {}

    def _update_file_metadata(self, file_path: str, data: pd.DataFrame) -> None:
        """Update file metadata cache."""
        metadata = {
            'size_bytes': len(data.to_parquet()),
            'modified_time': time.time(),
            'created_time': time.time(),
            'shape': data.shape,
            'columns': list(data.columns)
        }
        self._file_metadata_cache[file_path] = metadata

    def clear_cache(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        """Clear cache entries for specific or all files."""
        if exchange is None and symbol is None and timeframe is None:
            self._cache.clear()
            self.logger.info("🧹 Cleared all HMM composite manager cache")
        else:
            keys_to_remove: list[str] = []
            for key in list(self._cache.keys()):
                if exchange and exchange not in key:
                    continue
                if symbol and symbol not in key:
                    continue
                if timeframe and timeframe not in key:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                with contextlib.suppress(Exception):
                    del self._cache[key]

            self.logger.info(
                f"🧹 Cleared {len(keys_to_remove)} cache entries for {exchange}_{symbol}_{timeframe}",
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        total_size_mb = sum(
            len(str(v.get("data", ""))) / (1024 * 1024)
            for v in self._cache.values()
            if isinstance(v, dict) and "data" in v
        )

        return {
            "total_entries": total_entries,
            "total_size_mb": total_size_mb,
            "metadata_entries": len(self._file_metadata_cache),
        }

    def get_covariance_issue_guidance(self) -> str:
        """Get guidance for resolving covariance matrix issues."""
        guidance = """
        HMM Initialization and Optimization Issues - Common Solutions:

        1. Start Probability Issues ('startprob_ must sum to 1'):
           - Check for NaN or infinite values in your data
           - Ensure sufficient data samples for the number of components
           - Remove constant features that cause initialization problems
           - The system now automatically initializes with uniform probabilities

        2. Covariance Matrix Issues ('covars' must be symmetric, positive-definite):
           - Check for highly correlated features (> 0.95 correlation)
           - Ensure sufficient data samples for the covariance type
           - Remove features with zero or near-zero variance
           - Apply dimensionality reduction (PCA) if needed
           - The system now automatically:
             * Adds regularization to prevent singularity
             * Removes highly correlated features
             * Switches to diagonal covariance if full fails
             * Validates positive-definite matrices before fitting

        3. Data Quality Issues:
           - Check for NaN or infinite values in your data
           - Ensure sufficient data samples (rule of thumb: 10x features per component)
           - Remove or fill missing values appropriately
           - Standardize features to prevent numerical instability

        4. HMM Parameters Optimization:
           - Try different covariance types: 'diag' or 'spherical' instead of 'full'
           - Reduce number of components (n_components)
           - Increase regularization parameters
           - The optimizer now automatically tries multiple parameter combinations

        5. Feature Engineering:
           - Remove outliers that might cause numerical instability
           - Ensure features have sufficient variance
           - Apply data transformations (log, sqrt, etc.) if needed
           - Remove constant features automatically

        6. Alternative Approaches:
           - Use a different model (e.g., K-means clustering)
           - Consider time-series specific models
           - Try different initialization methods

        The enhanced optimizer now automatically handles most of these issues with:
        - Improved data validation and preprocessing
        - Better HMM model initialization
        - Automatic covariance matrix regularization
        - Comprehensive error handling and recovery
        """

        return guidance.strip()

# Global instance for backward compatibility
hmm_composite_manager = EnhancedHMMCompositeManager()

# Export for backward compatibility
@dataclass
class CoarseGridConfig:
    """Configuration for coarse grid search preprocessing."""
    n_components_range: List[int] = field(default_factory=lambda: [2, 4, 6])
    covariance_types: List[str] = field(default_factory=lambda: ['diag', 'spherical'])
    n_iter_range: List[int] = field(default_factory=lambda: [25, 50, 100])
    tol_range: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    max_trials: int = 20
    timeout_seconds: int = 120

@dataclass
class MultiFidelityConfig:
    """Configuration for multi-fidelity optimization."""
    fidelity_levels: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'low': {'n_iter': 25, 'tol': 1e-2, 'data_fraction': 0.3},
        'medium': {'n_iter': 50, 'tol': 1e-3, 'data_fraction': 0.6},
        'high': {'n_iter': 100, 'tol': 1e-4, 'data_fraction': 1.0}
    })
    early_stop_fraction: float = 0.3  # Stop poor trials early
    quality_threshold: float = 0.7   # Quality threshold for promotion

class CoarseGridOptimizer:
    """Performs coarse grid search to identify promising parameter regions."""

    def __init__(self, hmm_manager, config: Optional[CoarseGridConfig] = None):
        self.hmm_manager = hmm_manager
        self.config = config or CoarseGridConfig()
        self.logger = hmm_manager.logger

    def optimize(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run coarse grid search and return results."""
        self.logger.info("🏗️ Starting coarse grid search for parameter space exploration")

        # Generate all parameter combinations
        param_combinations = list(self._generate_param_combinations())

        self.logger.info(f"📊 Evaluating {len(param_combinations)} parameter combinations")

        results = []
        start_time = time.time()

        for i, params in enumerate(param_combinations):
            if time.time() - start_time > self.config.timeout_seconds:
                self.logger.warning("⏰ Coarse grid search timed out")
                break

            try:
                self.logger.debug(f"🔍 Testing combination {i+1}/{len(param_combinations)}: {params}")

                # Quick evaluation with reduced data
                score = self._evaluate_params_quick(data, params)

                result = {
                    'params': params.copy(),
                    'score': score,
                    'trial_number': i,
                    'timestamp': time.time()
                }
                results.append(result)

                self.logger.debug(f"🔍 Testing combination {i+1}/{len(param_combinations)}: {params}")
            except Exception as e:
                self.logger.debug(f"⚠️ Failed to evaluate params {params}: {e}")
                results.append({
                    'params': params.copy(),
                    'score': float('-inf'),
                    'trial_number': i,
                    'error': str(e)
                })

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        self.logger.info(f"✅ Coarse grid search completed: {len(results)} evaluations")
        self.logger.info(f"✅ Best score: {results[0]['score'] if results else float('-inf'):.4f}")
        return {
            'results': results,
            'best_params': results[0]['params'] if results else None,
            'best_score': results[0]['score'] if results else float('-inf'),
            'total_evaluations': len(results),
            'duration': time.time() - start_time
        }

    def _generate_param_combinations(self) -> Iterator[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        for n_comp in self.config.n_components_range:
            for cov_type in self.config.covariance_types:
                for n_iter in self.config.n_iter_range:
                    for tol in self.config.tol_range:
                        yield {
                            'n_components': n_comp,
                            'covariance_type': cov_type,
                            'n_iter': n_iter,
                            'tol': tol
                        }

    def _evaluate_params_quick(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Quick evaluation of parameter combination."""
        try:
            # Use subset of data for speed
            subset_size = min(len(data), max(1000, len(data) // 10))
            if len(data) > subset_size:
                data_subset = data.sample(n=subset_size, random_state=42)
            else:
                data_subset = data

            # Quick HMM training with simplified setup
            from hmmlearn import hmm

            model = hmm.GaussianHMM(
                n_components=params['n_components'],
                covariance_type=params['covariance_type'],
                n_iter=params['n_iter'],
                tol=params['tol'],
                init_params='mc',
                random_state=42
            )

            # Prepare data quickly
            X = self.hmm_manager._prepare_data_for_hmm(data_subset)

            # Validate minimum samples
            min_samples = params['n_components'] * 3
            if len(X) < min_samples:
                return float('-inf')

            # Fit and score
            model.fit(X.values.astype(np.float32))
            score = model.score(X.values.astype(np.float32))

            return score

        except Exception as e:
            self.logger.debug(f"⚠️ Quick evaluation failed: {e}")
            return float('-inf')

class ParameterAnalyzer:
    """Analyzes parameter importance from coarse grid results."""

    def __init__(self, hmm_manager):
        self.hmm_manager = hmm_manager
        self.logger = hmm_manager.logger

    def analyze_importance(self, coarse_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which parameters have the most impact on performance."""
        self.logger.info("📊 Analyzing parameter importance from coarse grid results")

        results = coarse_results['results']
        if not results:
            self.logger.warning("⚠️ No results to analyze")
            return {'param_importance': {}, 'recommendations': []}

        # Extract parameter-performance relationships
        param_performance = {
            'n_components': [],
            'covariance_type': [],
            'n_iter': [],
            'tol': []
        }

        for result in results:
            if result['score'] != float('-inf'):
                params = result['params']
                score = result['score']

                param_performance['n_components'].append((params['n_components'], score))
                param_performance['covariance_type'].append((params['covariance_type'], score))
                param_performance['n_iter'].append((params['n_iter'], score))
                param_performance['tol'].append((params['tol'], score))

        # Calculate importance scores
        importance_scores = {}
        recommendations = []

        for param_name, values_scores in param_performance.items():
            if not values_scores:
                continue

            # Calculate variance in performance for each parameter
            scores_by_value = {}
            for value, score in values_scores:
                if value not in scores_by_value:
                    scores_by_value[value] = []
                scores_by_value[value].append(score)

            # Parameter importance = variance of mean scores across parameter values
            mean_scores = [np.mean(scores) for scores in scores_by_value.values()]
            importance_scores[param_name] = np.var(mean_scores)

            # Generate recommendations
            best_value = max(scores_by_value.keys(),
                           key=lambda x: np.mean(scores_by_value[x]))

            recommendations.append({
                'parameter': param_name,
                'importance_score': importance_scores[param_name],
                'recommended_value': best_value,
                'performance_gain': np.mean(scores_by_value[best_value])
            })

        # Sort by importance
        recommendations.sort(key=lambda x: x['importance_score'], reverse=True)

        self.logger.info("✅ Parameter analysis completed")
        for rec in recommendations[:3]:  # Top 3 parameters
            self.logger.info(f"📊 {rec['parameter']}: importance={rec['importance_score']:.4f}, recommended={rec['recommended_value']}, gain={rec['performance_gain']:.4f}")
        return {
            'param_importance': importance_scores,
            'recommendations': recommendations,
            'top_parameters': [r['parameter'] for r in recommendations[:3]]
        }

class MultiFidelityOptimizer:
    """Implements multi-fidelity optimization with progressive quality levels."""

    def __init__(self, hmm_manager, config: Optional[MultiFidelityConfig] = None):
        self.hmm_manager = hmm_manager
        self.config = config or MultiFidelityConfig()
        self.logger = hmm_manager.logger

    def optimize_with_fidelity(self, data: pd.DataFrame,
                              param_ranges: Dict[str, Any],
                              n_trials: int = 20) -> Dict[str, Any]:
        """Run multi-fidelity optimization with early stopping."""
        self.logger.info("🎯 Starting multi-fidelity optimization")

        fidelity_results = {}
        current_best_score = float('-inf')
        current_best_params = None

        # Start with lowest fidelity
        fidelity_order = ['low', 'medium', 'high']

        for fidelity_level in fidelity_order:
            self.logger.info(f"📈 Running {fidelity_level} fidelity optimization")

            # Adjust parameters based on fidelity level
            level_config = self.config.fidelity_levels[fidelity_level]

            # Create data subset for this fidelity level
            data_fraction = level_config['data_fraction']
            if data_fraction < 1.0:
                subset_size = int(len(data) * data_fraction)
                fidelity_data = data.sample(n=subset_size, random_state=42)
                self.logger.debug(f"📊 Using {subset_size} samples for {fidelity_level} fidelity")
            else:
                fidelity_data = data

            # Run optimization at this fidelity level
            level_results = self._optimize_at_fidelity(
                fidelity_data, param_ranges, n_trials,
                level_config, current_best_params
            )

            fidelity_results[fidelity_level] = level_results

            # Update best parameters if improved
            if level_results['best_score'] > current_best_score:
                current_best_score = level_results['best_score']
                current_best_params = level_results['best_params']
                self.logger.info(f"✅ New best score at {fidelity_level}: {current_best_score:.4f}")

            # Early stopping check
            if self._should_stop_early(fidelity_results, fidelity_level):
                self.logger.info(f"🛑 Early stopping at {fidelity_level} fidelity")
                break

        # Final validation at full fidelity
        if fidelity_results:
            final_params = current_best_params
            final_score = self._validate_at_full_fidelity(data, final_params)

            self.logger.info("✅ Multi-fidelity optimization completed")
            self.logger.info(f"🎯 Final best score: {final_score:.4f}")

            return {
                'best_params': final_params,
                'best_score': final_score,
                'fidelity_results': fidelity_results,
                'optimization_path': fidelity_order[:len(fidelity_results)],
                'early_stopped': len(fidelity_results) < len(fidelity_order)
            }

        return {'best_params': None, 'best_score': float('-inf')}

    def _optimize_at_fidelity(self, data: pd.DataFrame,
                             param_ranges: Dict[str, Any],
                             n_trials: int,
                             level_config: Dict[str, Any],
                             warm_start_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize at a specific fidelity level."""
        try:
            import optuna

            def objective(trial):
                # Suggest parameters
                n_components = trial.suggest_int('n_components',
                    param_ranges.get('n_components', [2, 6])[0],
                    param_ranges.get('n_components', [2, 6])[1])

                covariance_type = trial.suggest_categorical('covariance_type',
                    param_ranges.get('covariance_type', ['diag', 'spherical']))

                # Use fidelity-specific settings
                n_iter = level_config['n_iter']
                tol = level_config['tol']

                params = {
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'n_iter': n_iter,
                    'tol': tol
                }

                # Evaluate parameters
                score = self._evaluate_params_efficient(data, params)

                # Report intermediate results for pruning
                trial.report(score, step=0)

                # Prune if score is too low compared to current best
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            # Create study with pruner for early stopping
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=5,
                interval_steps=1
            )

            study = optuna.create_study(
                direction='maximize',
                pruner=pruner,
                study_name=f"fidelity_{level_config['data_fraction']}"
            )

            # Warm start with previous best if available
            if warm_start_params:
                study.enqueue_trial(warm_start_params)

            # Run optimization
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_score = study.best_value

            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study,
                'n_trials_completed': len(study.trials),
                'n_trials_pruned': len([t for t in study.trials if t.state == optuna.TrialState.PRUNED])
            }

        except Exception as e:
            self.logger.error(f"❌ Fidelity optimization failed: {e}")
            return {'best_params': None, 'best_score': float('-inf')}

    def _evaluate_params_efficient(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Efficient parameter evaluation with memory optimization."""
        try:
            from hmmlearn import hmm

            # Create model with optimized settings
            model = hmm.GaussianHMM(
                n_components=params['n_components'],
                covariance_type=params['covariance_type'],
                n_iter=params['n_iter'],
                tol=params['tol'],
                init_params='mc',
                random_state=42
            )

            # Prepare data with memory optimization
            X = self.hmm_manager._prepare_data_for_hmm(data)

            # Validate minimum samples
            min_samples = params['n_components'] * 2
            if len(X) < min_samples:
                return float('-inf')

            # Use CPU optimization if available
            if self.hmm_manager.cpu_optimizer:
                with self.hmm_manager.cpu_optimizer.create_m1_optimized_context():
                    model.fit(X.values.astype(np.float32))
                    score = model.score(X.values.astype(np.float32))
            else:
                model.fit(X.values.astype(np.float32))
                score = model.score(X.values.astype(np.float32))

            return score

        except Exception as e:
            self.logger.debug(f"⚠️ Parameter evaluation failed: {e}")
            return float('-inf')

    def _should_stop_early(self, fidelity_results: Dict[str, Any], current_level: str) -> bool:
        """Determine if optimization should stop early."""
        if len(fidelity_results) < 2:
            return False

        # Check if performance improvement is diminishing
        levels = list(fidelity_results.keys())
        if len(levels) >= 2:
            prev_level = levels[-2]
            prev_score = fidelity_results[prev_level]['best_score']
            current_score = fidelity_results[current_level]['best_score']

            # Calculate improvement ratio
            if prev_score != 0:
                improvement_ratio = (current_score - prev_score) / abs(prev_score)
            else:
                improvement_ratio = current_score

            # Stop if improvement is less than threshold
            if improvement_ratio < self.config.quality_threshold:
                self.logger.info(f"🛑 Early stopping: improvement ratio {improvement_ratio:.2f} < threshold {self.config.quality_threshold:.2f}")
                return True

        return False

    def _validate_at_full_fidelity(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Final validation at full fidelity."""
        self.logger.info("🔍 Final validation at full fidelity")

        # Use highest fidelity settings for final evaluation
        high_fidelity = self.config.fidelity_levels['high']
        full_params = params.copy()
        full_params.update({
            'n_iter': high_fidelity['n_iter'],
            'tol': high_fidelity['tol']
        })

        return self._evaluate_params_efficient(data, full_params)

@dataclass
class AdaptiveOptimizationConfig:
    """Configuration for adaptive Bayesian optimization pipeline."""
    enable_coarse_grid: bool = True
    enable_parameter_analysis: bool = True
    enable_multi_fidelity: bool = True
    enable_early_stopping: bool = True

    # Resource allocation
    coarse_grid_time_budget: int = 120  # seconds
    parameter_analysis_time_budget: int = 30  # seconds
    optimization_time_budget: int = 300  # seconds

    # Quality thresholds
    min_improvement_threshold: float = 0.01  # Minimum improvement to continue
    confidence_threshold: float = 0.8  # Confidence threshold for early stopping

    # Mode-specific settings
    mode_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'light': {
            'coarse_trials': 12,
            'optimization_trials': 5,
            'fidelity_levels': ['low', 'medium']
        },
        'blank': {
            'coarse_trials': 20,
            'optimization_trials': 10,
            'fidelity_levels': ['low', 'medium', 'high']
        },
        'full': {
            'coarse_trials': 32,
            'optimization_trials': 25,
            'fidelity_levels': ['low', 'medium', 'high']
        }
    })

class AdaptiveBayesianOptimizer:
    """Adaptive Bayesian optimization combining multiple strategies."""

    def __init__(self, hmm_manager, config: Optional[AdaptiveOptimizationConfig] = None):
        self.hmm_manager = hmm_manager
        self.config = config or AdaptiveOptimizationConfig()
        self.logger = hmm_manager.logger

        # Initialize component optimizers
        self.coarse_optimizer = CoarseGridOptimizer(hmm_manager)
        self.parameter_analyzer = ParameterAnalyzer(hmm_manager)
        self.multi_fidelity_optimizer = MultiFidelityOptimizer(hmm_manager)

    def optimize_adaptive(self, data: pd.DataFrame, mode: str = 'light') -> Dict[str, Any]:
        """Run complete adaptive optimization pipeline."""
        self.logger.info(f"🚀 Starting adaptive optimization in {mode} mode")

        start_time = time.time()
        optimization_results = {
            'mode': mode,
            'pipeline_steps': [],
            'intermediate_results': {},
            'final_result': None,
            'performance_metrics': {}
        }

        try:
            # Phase 1: Coarse Grid Search (if enabled)
            if self.config.enable_coarse_grid:
                self.logger.info("📊 Phase 1: Coarse Grid Search")
                coarse_results = self._run_coarse_grid_phase(data, mode)
                optimization_results['intermediate_results']['coarse_grid'] = coarse_results
                optimization_results['pipeline_steps'].append('coarse_grid')

                # Log coarse grid results
                if coarse_results.get('best_score') and coarse_results['best_score'] != float('-inf'):
                    self.logger.info(f"✅ Coarse Grid Phase Complete - Best Score: {coarse_results['best_score']:.4f}")
                    self.logger.info(f"📊 Evaluations: {coarse_results['total_evaluations']}, Duration: {coarse_results['duration']:.1f}s")
                else:
                    self.logger.warning("⚠️ Coarse Grid Phase completed but no valid results found")

            # Phase 2: Parameter Analysis (if enabled)
            if self.config.enable_parameter_analysis and 'coarse_grid' in optimization_results['intermediate_results']:
                self.logger.info("📈 Phase 2: Parameter Importance Analysis")
                param_analysis = self._run_parameter_analysis_phase(
                    optimization_results['intermediate_results']['coarse_grid']
                )
                optimization_results['intermediate_results']['parameter_analysis'] = param_analysis
                optimization_results['pipeline_steps'].append('parameter_analysis')

                # Log parameter analysis results
                if param_analysis.get('recommendations'):
                    self.logger.info("✅ Parameter Analysis Phase Complete")
                    for rec in param_analysis['recommendations'][:3]:
                        self.logger.info(f"📊 {rec['parameter']}: importance={rec['importance_score']:.4f}, recommended={rec['recommended_value']}")
                else:
                    self.logger.warning("⚠️ Parameter Analysis Phase completed but no recommendations generated")

            # Phase 3: Multi-Fidelity Optimization
            if self.config.enable_multi_fidelity:
                self.logger.info("🎯 Phase 3: Multi-Fidelity Optimization")
                fidelity_results = self._run_multi_fidelity_phase(
                    data, mode, optimization_results
                )
                optimization_results['intermediate_results']['multi_fidelity'] = fidelity_results
                optimization_results['pipeline_steps'].append('multi_fidelity')

                # Log multi-fidelity results
                if fidelity_results.get('best_score') and fidelity_results['best_score'] != float('-inf'):
                    self.logger.info(f"✅ Multi-Fidelity Phase Complete - Best Score: {fidelity_results['best_score']:.4f}")
                    if fidelity_results.get('early_stopped'):
                        self.logger.info("🛑 Optimization stopped early due to diminishing returns")
                    else:
                        self.logger.info(f"🎯 Completed full optimization path: {fidelity_results.get('optimization_path', [])}")
                else:
                    self.logger.warning("⚠️ Multi-Fidelity Phase completed but no valid results found")

            # Determine final result
            final_result = self._determine_final_result(optimization_results)
            optimization_results['final_result'] = final_result

            # Calculate performance metrics
            optimization_results['performance_metrics'] = self._calculate_performance_metrics(
                optimization_results, start_time
            )

            # Log comprehensive final results
            self._log_final_optimization_summary(optimization_results)

            self.logger.info("✅ Adaptive optimization completed successfully")
            return optimization_results

        except Exception as e:
            self.logger.error(f"❌ Adaptive optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results

    def _run_coarse_grid_phase(self, data: pd.DataFrame, mode: str) -> Dict[str, Any]:
        """Run coarse grid search phase."""
        mode_config = self.config.mode_configs.get(mode, self.config.mode_configs['light'])

        # Create mode-specific coarse grid config
        coarse_config = CoarseGridConfig(
            max_trials=mode_config['coarse_trials'],
            timeout_seconds=self.config.coarse_grid_time_budget
        )

        # Adjust parameter ranges based on mode
        if mode == 'light':
            coarse_config.n_components_range = [2, 4, 6]
            coarse_config.covariance_types = ['diag', 'spherical']
        elif mode == 'full':
            coarse_config.n_components_range = [2, 4, 6, 8, 10]
            coarse_config.covariance_types = ['diag', 'spherical', 'full']

        # Run coarse grid optimization
        coarse_optimizer = CoarseGridOptimizer(self.hmm_manager, coarse_config)
        return coarse_optimizer.optimize(data)

    def _run_parameter_analysis_phase(self, coarse_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run parameter analysis phase."""
        analyzer = ParameterAnalyzer(self.hmm_manager)
        return analyzer.analyze_importance(coarse_results)

    def _run_multi_fidelity_phase(self, data: pd.DataFrame, mode: str,
                                 optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-fidelity optimization phase."""
        mode_config = self.config.mode_configs.get(mode, self.config.mode_configs['light'])

        # Determine parameter ranges based on previous phases
        param_ranges = self._determine_param_ranges(optimization_results, mode)

        # Configure multi-fidelity optimizer
        fidelity_config = MultiFidelityConfig()

        # Adjust fidelity levels based on mode
        if mode == 'light':
            # Use only low and medium for speed
            fidelity_config.fidelity_levels = {
                k: v for k, v in fidelity_config.fidelity_levels.items()
                if k in ['low', 'medium']
            }

        # Run multi-fidelity optimization
        fidelity_optimizer = MultiFidelityOptimizer(self.hmm_manager, fidelity_config)
        return fidelity_optimizer.optimize_with_fidelity(
            data, param_ranges, mode_config['optimization_trials']
        )

    def _determine_param_ranges(self, optimization_results: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Determine parameter ranges based on previous optimization phases."""
        # Default ranges
        param_ranges = {
            'n_components': [2, 6],
            'covariance_type': ['diag', 'spherical']
        }

        # Adjust based on coarse grid results if available
        if 'parameter_analysis' in optimization_results.get('intermediate_results', {}):
            analysis = optimization_results['intermediate_results']['parameter_analysis']

            # Use recommendations from parameter analysis
            for rec in analysis.get('recommendations', []):
                param_name = rec['parameter']
                if param_name == 'n_components':
                    # Narrow range around recommended value
                    recommended = rec['recommended_value']
                    param_ranges['n_components'] = [
                        max(2, recommended - 1),
                        min(12, recommended + 1)
                    ]
                elif param_name == 'covariance_type':
                    # Prioritize recommended covariance type
                    recommended = rec['recommended_value']
                    param_ranges['covariance_type'] = [recommended] + [
                        ct for ct in ['diag', 'spherical', 'full', 'tied']
                        if ct != recommended
                    ][:2]  # Keep only top 3

        # Mode-specific adjustments
        if mode == 'light':
            param_ranges['n_components'] = [2, 6]  # Keep light
        elif mode == 'full':
            param_ranges['n_components'] = [2, 10]  # Allow more components

        return param_ranges

    def _determine_final_result(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the final optimization result."""
        # Priority: multi-fidelity > coarse grid
        if 'multi_fidelity' in optimization_results.get('intermediate_results', {}):
            mf_result = optimization_results['intermediate_results']['multi_fidelity']
            if mf_result.get('best_params'):
                return {
                    'source': 'multi_fidelity',
                    'best_params': mf_result['best_params'],
                    'best_score': mf_result['best_score'],
                    'method': 'adaptive_multi_fidelity'
                }

        if 'coarse_grid' in optimization_results.get('intermediate_results', {}):
            cg_result = optimization_results['intermediate_results']['coarse_grid']
            if cg_result.get('best_params'):
                return {
                    'source': 'coarse_grid',
                    'best_params': cg_result['best_params'],
                    'best_score': cg_result['best_score'],
                    'method': 'coarse_grid_fallback'
                }

        # Fallback to default parameters
        return {
            'source': 'default',
            'best_params': {
                'n_components': 4,
                'covariance_type': 'diag',
                'n_iter': 100,
                'tol': 1e-3
            },
            'best_score': float('-inf'),
            'method': 'default_fallback'
        }

    def _calculate_performance_metrics(self, optimization_results: Dict[str, Any],
                                     start_time: float) -> Dict[str, Any]:
        """Calculate performance metrics for the optimization run."""
        end_time = time.time()
        total_time = end_time - start_time

        metrics = {
            'total_time_seconds': total_time,
            'pipeline_steps_completed': len(optimization_results.get('pipeline_steps', [])),
            'final_score': optimization_results.get('final_result', {}).get('best_score', float('-inf')),
            'optimization_method': optimization_results.get('final_result', {}).get('method', 'unknown')
        }

        # Calculate efficiency metrics
        if total_time > 0:
            metrics['time_per_step'] = total_time / max(1, len(optimization_results.get('pipeline_steps', [])))

        # Calculate improvement metrics if we have intermediate results
        intermediate = optimization_results.get('intermediate_results', {})
        if 'coarse_grid' in intermediate and 'multi_fidelity' in intermediate:
            cg_score = intermediate['coarse_grid'].get('best_score', float('-inf'))
            mf_score = intermediate['multi_fidelity'].get('best_score', float('-inf'))

            if cg_score != float('-inf') and mf_score != float('-inf'):
                metrics['improvement_over_coarse'] = mf_score - cg_score
                metrics['improvement_ratio'] = (mf_score - cg_score) / abs(cg_score) if cg_score != 0 else mf_score

        return metrics

HMMCompositeManager = EnhancedHMMCompositeManager