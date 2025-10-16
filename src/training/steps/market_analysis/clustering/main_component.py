"""
Main NAS-TAS Clustering Component.

This module provides the main entry point for the clustering system, orchestrating all services
and exposing a clean API for the clustering workflow.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import traceback
import logging

from .config.clustering_config import NASTASClusteringConfig, ClusteringContext
from .services import (
    ClusteringService, FeatureService, OptimizationService, HardwareService
)
from ..shared_utils import (
    get_logger, log_info, log_error, log_success, log_warning, log_debug,
    validate_regime_count, normalize_weights, validate_algorithm_type,
    BaseConfig, ConfigValidator
)

# Import utility tools
from src.utils.tprint import tprint
from src.utils.enhanced_artifact_manager import get_artifact_manager

# Utility function for directory creation
def ensure_directory_exists(path: Path) -> None:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
from src.utils.common_operations import (
    ensure_directory, safe_dataframe_operation,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
)
from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    calculate_data_quality_metrics, safe_convert_dtypes
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_mean, safe_std
)
from src.utils.matrix_operations import (
    UnifiedMatrixOperations, get_unified_matrix_operations
)
from src.utils.hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer

# Try to import ML utilities (some may not be available)
try:
    from src.utils.data.quality.data_quality import DataQualityChecker
except ImportError:
    DataQualityChecker = None

try:
    from src.utils.data.validation.validators import DataValidator
except ImportError:
    DataValidator = None

try:
    from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention
except ImportError:
    DataLeakagePrevention = None

try:
    from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
    LOOKAHEAD_PROTECTION_AVAILABLE = True
except ImportError:
    class LookaheadProtection:
        pass
    LOOKAHEAD_PROTECTION_AVAILABLE = False

try:
    from src.utils.ml_common.cvlsa.cvlsa_architecture import CVLSAArchitecture
except ImportError:
    CVLSAArchitecture = None

try:
    from src.utils.ml_common.optimization.hpo_utils import HPOUtils
except ImportError:
    HPOUtils = None

try:
    from src.utils.ml_common.optimization.grid_utils import GridSearchOptimizer
except ImportError:
    GridSearchOptimizer = None

try:
    from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import BayesianEntryTimingOptimizer
except ImportError:
    BayesianEntryTimingOptimizer = None

# Utility functions are now imported from utils modules

class NASTASClusteringComponent:
    """
    Main NAS-TAS Clustering Component.

    Entry point of the clustering system that orchestrates all services and exposes
    a clean API for the clustering workflow.
    """

    def __init__(self, config: Optional[NASTASClusteringConfig] = None,
                 context: Optional[ClusteringContext] = None):
        """
        Initialize the NAS-TAS Clustering Component.

        Args:
            config: Configuration object for clustering parameters
            context: Clustering context containing weights, thresholds, etc.
        """
        self.logger = get_logger(__name__)
        self.artifact_manager = get_artifact_manager()

        # Initialize configuration and context
        self.config = config or NASTASClusteringConfig()
        if context is None:
            # Create empty context for now - will be populated during fit()
            self.context = ClusteringContext(original_features=None, market_data=None)
        else:
            self.context = context

        # Initialize hardware optimization
        self._initialize_hardware_optimization()

        # Initialize ML utilities
        self._initialize_ml_utilities()

        # Initialize services
        self._initialize_services()

        # State management
        self.is_fitted = False
        self.current_results = None
        self.clustering_history = []

        tprint("🚀 NASTASClusteringComponent initialized successfully", "SUCCESS")
        log_info("NASTASClusteringComponent initialized successfully")

    @property
    def services(self):
        """Return a dictionary of all services for easy access."""
        return {
            'feature': self.feature_service,
            'clustering': self.clustering_service,
            'optimization': self.optimization_service,
            'hardware': self.hardware_service
        }

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization utilities."""
        try:
            tprint("🔧 Initializing hardware optimization...", "INFO")

            # Initialize M1 hardware utilities
            self.m1_gpu_manager = M1GPUManager() if is_m1_available() else None
            self.m1_memory_optimizer = M1MemoryOptimizer() if is_m1_available() else None
            self.m1_cpu_optimizer = M1CPUOptimizer() if is_m1_available() else None

            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations()

            # Log hardware status
            if self.m1_gpu_manager and self.m1_gpu_manager.mps_available:
                tprint("✅ M1 GPU acceleration available", "SUCCESS")
            else:
                tprint("⚠️ M1 GPU acceleration not available, using CPU", "WARNING")

            tprint("✅ Hardware optimization initialized", "SUCCESS")
            log_info("Hardware optimization initialized successfully")

        except Exception as e:
            tprint(f"❌ Hardware optimization initialization failed: {e}", "ERROR")
            log_error(f"Hardware optimization initialization failed: {e}")
            # Continue without hardware optimization
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.matrix_ops = None

    def _initialize_ml_utilities(self):
        """Initialize ML utilities for validation and optimization."""
        try:
            tprint("🧠 Initializing ML utilities...", "INFO")

            # Initialize data quality and validation (if available)
            self.data_quality_checker = DataQualityChecker() if DataQualityChecker else None
            self.data_validator = DataValidator() if DataValidator else None

            # Initialize data leakage prevention (if available)
            self.data_leakage_prevention = DataLeakagePrevention() if DataLeakagePrevention else None
            self.lookahead_protection = LookaheadProtection() if LookaheadProtection else None

            # Initialize CVLSA architecture (if available)
            self.cvlsa_architecture = CVLSAArchitecture() if CVLSAArchitecture else None

            # Initialize optimization utilities (if available)
            self.hpo_utils = HPOUtils if HPOUtils else None
            self.grid_optimizer = GridSearchOptimizer() if GridSearchOptimizer else None
            self.bayesian_optimizer = BayesianEntryTimingOptimizer() if BayesianEntryTimingOptimizer else None

            # Count initialized utilities
            initialized_count = sum([
                self.data_quality_checker is not None,
                self.data_validator is not None,
                self.data_leakage_prevention is not None,
                self.lookahead_protection is not None,
                self.cvlsa_architecture is not None,
                self.hpo_utils is not None,
                self.grid_optimizer is not None,
                self.bayesian_optimizer is not None
            ])

            tprint(f"✅ ML utilities initialized ({initialized_count}/8 available)", "SUCCESS")
            log_info(f"ML utilities initialized successfully ({initialized_count}/8 available)")

        except Exception as e:
            tprint(f"❌ ML utilities initialization failed: {e}", "ERROR")
            log_error(f"ML utilities initialization failed: {e}")
            # Continue without ML utilities
            self.data_quality_checker = None
            self.data_validator = None
            self.data_leakage_prevention = None
            self.lookahead_protection = None
            self.cvlsa_architecture = None
            self.hpo_utils = None
            self.grid_optimizer = None
            self.bayesian_optimizer = None

    def _initialize_services(self):
        """Initialize all required services."""
        try:
            tprint("🔧 Initializing clustering services...", "INFO")

            # Initialize hardware service with M1 optimizations
            try:
                self.hardware_service = HardwareService(
                    m1_gpu_manager=self.m1_gpu_manager,
                    m1_memory_optimizer=self.m1_memory_optimizer,
                    m1_cpu_optimizer=self.m1_cpu_optimizer,
                    matrix_ops=self.matrix_ops
                )
            except TypeError:
                # Fallback if HardwareService doesn't accept these parameters
                self.hardware_service = HardwareService()

            # Initialize feature service
            self.feature_service = FeatureService(verbose=True)

            # Initialize clustering service
            self.clustering_service = ClusteringService(verbose=True)

            # Initialize optimization service
            self.optimization_service = OptimizationService(verbose=True)

            tprint("✅ All services initialized successfully", "SUCCESS")
            log_success("All services initialized successfully")

        except Exception as e:
            tprint(f"❌ Failed to initialize services: {e}", "ERROR")
            log_error(f"Failed to initialize services: {e}")
            raise

    async def fit(self, data: Union[pd.DataFrame, np.ndarray],
            labels: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None) -> 'NASTASClusteringComponent':
        """
        Fit the clustering model to the provided data.

        Args:
            data: Input data for clustering (DataFrame or numpy array)
            labels: Optional ground truth labels for evaluation
            metadata: Optional metadata about the data

        Returns:
            Self for method chaining
        """
        try:
            log_info("Starting clustering fit process...")
            tprint("🚀 Starting NAS-TAS Clustering Fit Process", "INFO")
            tprint("🔍 DEBUG: About to validate input data", "DEBUG")

            # Validate input data using utility functions
            self._validate_input_data(data, labels)
            tprint("✅ DEBUG: Input data validation completed", "DEBUG")

            # Convert data to DataFrame if needed and validate
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

            # Perform data quality validation
            if self.data_quality_checker:
                quality_metrics = self.data_quality_checker.check_data_quality(data)
                tprint(f"📊 Data quality metrics: {quality_metrics.get('overall_score', 'N/A')}", "INFO")
            else:
                quality_metrics = calculate_data_quality_metrics(data)
                tprint(f"📊 Data quality metrics: {quality_metrics.get('shape', 'N/A')}", "INFO")

            # Initialize clustering context
            self.context.data_shape = data.shape
            self.context.feature_names = list(data.columns)
            self.context.metadata = metadata or {}
            self.context.start_time = datetime.now()

            # Step 1: Feature preparation and preprocessing
            log_info("Step 1: Preparing and preprocessing features...")
            tprint("📊 Preparing and preprocessing features...", "INFO")
            tprint("🔍 DEBUG: About to call feature_service.prepare_features", "DEBUG")

            feature_result = await self.feature_service.prepare_features(
                data, self.context
            )

            # Extract processed data and feature info from result
            if hasattr(feature_result, 'features_array'):
                processed_data = feature_result.features_array
                feature_info = feature_result.metadata if hasattr(feature_result, 'metadata') else {}
                feature_names = getattr(feature_result, 'feature_names', [])
            elif hasattr(feature_result, 'features'):
                processed_data = feature_result.features
                feature_info = feature_result.metadata if hasattr(feature_result, 'metadata') else {}
                feature_names = getattr(feature_result, 'feature_names', [])
            else:
                # Fallback: assume it's a direct array
                processed_data = feature_result
                feature_info = {}
                feature_names = []

            tprint(f"✅ DEBUG: Feature preparation completed - processed_data shape: {processed_data.shape}", "DEBUG")

            self.context.feature_info = feature_info
            self.context.optimized_features = processed_data
            # Store feature names for later use in consolidation
            if feature_names:
                self.context.feature_names = feature_names
            log_success(f"Features prepared: {processed_data.shape[1]} features")

            # Step 2: Initial clustering
            log_info("Step 2: Performing initial clustering...")
            tprint("🔍 Performing initial clustering...", "INFO")
            tprint("🔍 DEBUG: About to call clustering_service.run_initial_clustering_only", "DEBUG")

            initial_assignments, optimal_k = await self.clustering_service.run_initial_clustering_only(
                processed_data, data, self.config
            )

            tprint(f"✅ DEBUG: Initial clustering completed - assignments shape: {initial_assignments.shape}, optimal_k: {optimal_k}", "DEBUG")

            initial_results = {
                'cluster_assignments': initial_assignments,
                'n_clusters': optimal_k
            }
            self.context.initial_clustering_results = initial_results
            self.context.initial_assignments = initial_assignments
            log_success(f"Initial clustering completed: {optimal_k} clusters")

            # Step 3: Optimization loop
            log_info("Step 3: Running optimization loop...")
            tprint("⚡ Running optimization loop...", "INFO")
            tprint("🔍 DEBUG: About to start optimization loop", "DEBUG")

            # Update context with initial results
            self.context.optimized_assignments = initial_results['cluster_assignments']
            self.context.optimal_k = initial_results['n_clusters']

            # Run optimization
            tprint("🔍 DEBUG: About to call optimization_service.run_optimization", "DEBUG")
            optimization_result = await self.optimization_service.run_optimization(
                self.context, self.config, max_iterations=100
            )

            tprint("✅ DEBUG: Optimization loop completed", "DEBUG")

            self.context = optimization_result.final_context
            log_success("Optimization loop completed")

            # Step 4: Prepare clustering results for validation
            log_info("Step 4: Preparing clustering results...")
            tprint("📋 Preparing clustering results...", "INFO")

            clustering_results = {
                'assignments': self.context.optimized_assignments,
                'n_clusters': self.context.optimal_k,
                'optimization_history': optimization_result.optimization_history
            }

            # Step 5: Validation and evaluation
            log_info("Step 5: Validating and evaluating results...")
            tprint("✅ Validating and evaluating results...", "INFO")

            validation_results = self._validate_and_evaluate(
                processed_data, clustering_results, labels
            )

            self.context.validation_results = validation_results
            log_success("Validation completed")

            # Calculate duration before consolidation
            self.context.end_time = datetime.now()
            self.context.duration = (self.context.end_time - self.context.start_time).total_seconds()

            # Step 6: Results consolidation
            log_info("Step 6: Consolidating final results...")
            tprint("📋 Consolidating final results...", "INFO")
            self.current_results = self._consolidate_results(
                processed_data, clustering_results, validation_results
            )

            # Update state
            self.is_fitted = True

            # Store in history
            self.clustering_history.append({
                'timestamp': self.context.start_time,
                'config': self.config.to_dict(),
                'results': self.current_results,
                'duration': self.context.duration
            })

            log_success("Clustering fit process completed successfully")
            tprint("🎉 Clustering fit process completed successfully!", "SUCCESS")

            return self

        except Exception as e:
            log_error(f"Clustering fit failed: {e}")
            tprint(f"❌ Clustering fit failed: {e}", "ERROR")
            raise

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster assignments for new data.

        Args:
            data: Input data for prediction

        Returns:
            Cluster assignments as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            log_info("Making predictions on new data...")

            # Validate input
            self._validate_input_data(data)

            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=self.context.feature_names)

            # Use feature service to preprocess
            processed_data, _ = self.feature_service.prepare_features(data, self.context)

            # Use clustering service to predict
            predictions = self.clustering_service.predict_clusters(
                processed_data, self.context.optimized_results
            )

            log_success(f"Predictions completed for {len(predictions)} samples")
            return predictions

        except Exception as e:
            log_error(f"Prediction failed: {e}")
            raise

    def evaluate(self, data: Union[pd.DataFrame, np.ndarray],
                 labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the clustering performance.

        Args:
            data: Input data
            labels: Ground truth labels

        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        try:
            log_info("Evaluating clustering performance...")

            # Get predictions
            predictions = self.predict(data)

            # Calculate evaluation metrics
            evaluation_results = self.clustering_service.evaluate_clustering(
                predictions, labels, self.context.optimized_results
            )

            log_success("Evaluation completed")
            return evaluation_results

        except Exception as e:
            log_error(f"Evaluation failed: {e}")
            raise

    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save clustering results to disk.

        Args:
            path: Path to save the results
        """
        if not self.is_fitted:
            raise ValueError("No results to save - model must be fitted first")

        try:
            log_info(f"Saving results to {path}...")

            path = Path(path)
            ensure_directory_exists(path.parent)

            # Prepare results data
            results_data = {
                'config': self.config.to_dict(),
                'context': self.context.to_dict(),
                'results': self.current_results,
                'clustering_history': self.clustering_history,
                'timestamp': datetime.now().isoformat(),
                'component_version': '1.0.0'
            }

            # Save to pickle file
            with open(path, 'wb') as f:
                pickle.dump(results_data, f)

            # Also save as artifact
            self.artifact_manager.save_artifact(
                'nas_tas_clustering_results',
                results_data,
                metadata={
                    'component': 'NASTASClusteringComponent',
                    'timestamp': datetime.now().isoformat(),
                    'config_summary': self.config.get_summary()
                }
            )

            log_success(f"Results saved successfully to {path}")

        except Exception as e:
            log_error(f"Failed to save results: {e}")
            raise

    def _validate_input_data(self, data: Union[pd.DataFrame, np.ndarray],
                           labels: Optional[np.ndarray] = None) -> None:
        """Validate input data quality and format using utility functions."""
        if data is None:
            raise ValueError("Data cannot be None")

        if hasattr(data, 'shape'):
            if len(data.shape) != 2:
                raise ValueError("Data must be 2-dimensional")

            if data.shape[0] == 0:
                raise ValueError("Data cannot be empty")

            # Convert to DataFrame if needed for column handling
            if isinstance(data, np.ndarray):
                data_df = pd.DataFrame(data)
            else:
                data_df = data

            # Only validate numeric columns for finiteness
            numeric_columns = data_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                numeric_data = data_df[numeric_columns]
                # Check for non-finite values in numeric data
                finite_mask = np.isfinite(numeric_data.values)
                if not finite_mask.all():
                    raise ValueError("Numeric data contains non-finite values")

            tprint(f"📊 Data validation: {data_df.shape[0]} rows, {data_df.shape[1]} columns ({len(numeric_columns)} numeric)", "INFO")

        if labels is not None:
            if len(labels) != len(data):
                raise ValueError("Labels length must match data length")

            # Use math validation utilities for labels
            finite_mask = np.isfinite(labels)
            if not finite_mask.all():
                raise ValueError("Labels contain non-finite values")

        # Additional validation using data validator if available
        if self.data_validator and hasattr(self.data_validator, 'validate_input_data'):
            try:
                validation_result = self.data_validator.validate_input_data(data, labels)
                if not validation_result['is_valid']:
                    tprint(f"⚠️ Data validation warning: {validation_result.get('warnings', [])}", "WARNING")
            except Exception as e:
                tprint(f"⚠️ Data validator error: {e}", "WARNING")

    def _validate_and_evaluate(self, data: pd.DataFrame,
                              clustering_results: Dict[str, Any],
                              labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate and evaluate clustering results using utility functions."""
        try:
            tprint("🔍 Validating and evaluating clustering results...", "INFO")

            # Use data quality checker if available
            if self.data_quality_checker:
                data_quality = self.data_quality_checker.check_data_quality(data)
            else:
                data_quality = calculate_data_quality_metrics(data)

            # Use data leakage prevention if available
            leakage_check = None
            if self.data_leakage_prevention:
                leakage_check = self.data_leakage_prevention.check_data_leakage(data, labels)
                # Debug: Check the type of leakage_check
                tprint(f"🔍 Leakage check type: {type(leakage_check)}", "DEBUG")
                if isinstance(leakage_check, dict):
                    tprint(f"🔒 Data leakage check: {leakage_check.get('status', 'unknown')}", "INFO")
                else:
                    tprint(f"⚠️ Leakage check returned unexpected type: {type(leakage_check)}", "WARNING")

            # Use lookahead protection if available
            lookahead_check = None
            if self.lookahead_protection:
                # Convert numpy array to DataFrame if needed
                if isinstance(data, np.ndarray):
                    # Create DataFrame from numpy array with column names
                    feature_names = getattr(self.context, 'feature_names', [f'feature_{i}' for i in range(data.shape[1])])
                    data_df = pd.DataFrame(data, columns=feature_names)
                else:
                    data_df = data

                lookahead_check = self.lookahead_protection.check_lookahead_bias(data_df, labels)
                # Debug: Check the type of lookahead_check
                tprint(f"🔍 Lookahead check type: {type(lookahead_check)}", "DEBUG")
                if isinstance(lookahead_check, dict):
                    tprint(f"👁️ Lookahead bias check: {lookahead_check.get('status', 'unknown')}", "INFO")
                else:
                    tprint(f"⚠️ Lookahead check returned unexpected type: {type(lookahead_check)}", "WARNING")

            validation_results = {
                'data_quality': data_quality,
                'clustering_quality': self.clustering_service.validate_clustering_quality(
                    clustering_results.get('assignments', np.array([])),
                    clustering_results.get('features', np.array([]))
                ),
                'data_leakage_check': leakage_check,
                'lookahead_bias_check': lookahead_check,
                'timestamp': datetime.now().isoformat()
            }

            # Debug: Check validation results types
            tprint(f"🔍 Validation results data_quality type: {type(validation_results['data_quality'])}", "DEBUG")
            tprint(f"🔍 Validation results clustering_quality type: {type(validation_results['clustering_quality'])}", "DEBUG")

            if labels is not None:
                validation_results['ground_truth_evaluation'] = self.evaluate(data, labels)

            tprint("✅ Validation and evaluation completed", "SUCCESS")
            return validation_results

        except Exception as e:
            tprint(f"❌ Validation failed: {e}", "ERROR")
            log_error(f"Validation failed: {e}")
            return {'error': str(e)}

    def _consolidate_results(self, data,
                           clustering_results: Dict[str, Any],
                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate all results into a comprehensive output using utility functions."""
        try:
            tprint("📋 Consolidating results...", "INFO")

            # Convert numpy array to DataFrame if needed
            if isinstance(data, np.ndarray):
                # Get feature names from context or create generic ones
                feature_names = getattr(self.context, 'feature_names', None)
                if feature_names is None or len(feature_names) != data.shape[1]:
                    feature_names = [f'feature_{i}' for i in range(data.shape[1])]
                    tprint(f"🔧 Created generic feature names for numpy array: {len(feature_names)} features", "INFO")

                # Convert to DataFrame
                data = pd.DataFrame(data, columns=feature_names)
                tprint(f"✅ Converted numpy array to DataFrame with {data.shape[1]} features", "SUCCESS")

            # Now data should be a DataFrame
            if hasattr(data, 'columns'):
                data_shape = data.shape
                feature_names = list(data.columns)
                tprint(f"📊 Processing DataFrame with shape {data_shape} and {len(feature_names)} features", "INFO")
            else:
                raise ValueError(f"Expected DataFrame or numpy array, got {type(data)}")

            # Use math validation for metrics
            clustering_metrics = clustering_results.get('metrics', {})
            safe_metrics = {}
            for key, value in clustering_metrics.items():
                if validate_finite(value, f"metric_{key}"):
                    safe_metrics[key] = value
                else:
                    tprint(f"⚠️ Invalid metric {key}: {value}", "WARNING")

            # Use safe math operations for performance metrics
            duration = getattr(self.context, 'duration', None)
            if duration is None:
                # Calculate duration if not set
                if hasattr(self.context, 'start_time') and hasattr(self.context, 'end_time'):
                    if self.context.start_time and self.context.end_time:
                        duration = (self.context.end_time - self.context.start_time).total_seconds()
                    else:
                        duration = 0.0
                        tprint("⚠️ Missing start_time or end_time, setting duration to 0", "WARNING")
                else:
                    duration = 0.0
                    tprint("⚠️ No timing information available, setting duration to 0", "WARNING")
            elif not validate_finite(duration, "duration"):
                duration = 0.0
                tprint("⚠️ Invalid duration detected, setting to 0", "WARNING")

            consolidated_results = {
                # Core clustering results
                'clusters': clustering_results.get('clusters', []),
                'cluster_assignments': clustering_results.get('assignments', []),
                'cluster_centers': clustering_results.get('centers', []),
                'n_clusters': validate_positive(clustering_results.get('n_clusters', 0), "n_clusters"),

                # Metrics and scores (with validation)
                'clustering_metrics': safe_metrics,
                'optimization_metrics': clustering_results.get('optimization_metrics', {}),

                # Feature information
                'feature_info': self.context.feature_info,
                'selected_features': clustering_results.get('selected_features', []),
                'feature_importance': clustering_results.get('feature_importance', {}),

                # Validation results
                'validation': validation_results,

                # Context information
                'data_shape': data_shape,
                'feature_names': feature_names,
                'metadata': self.context.metadata,

                # Performance information (with validation)
                'duration': duration,
                'timestamp': datetime.now().isoformat(),

                # Hardware optimization info
                'hardware_info': {
                    'm1_gpu_available': self.m1_gpu_manager is not None,
                    'mps_available': self.m1_gpu_manager.mps_available if self.m1_gpu_manager else False,
                    'matrix_ops_available': self.matrix_ops is not None
                }
            }

            tprint("✅ Results consolidation completed", "SUCCESS")
            return consolidated_results

        except Exception as e:
            tprint(f"❌ Results consolidation failed: {e}", "ERROR")
            log_error(f"Results consolidation failed: {e}")
            return {'error': str(e)}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current clustering results."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}

        return {
            'status': 'fitted',
            'n_clusters': self.current_results.get('n_clusters', 0),
            'data_shape': self.current_results.get('data_shape', (0, 0)),
            'duration': self.context.duration if self.context.duration is not None else 0.0,
            'metrics': self.current_results.get('clustering_metrics', {}),
            'feature_count': len(self.current_results.get('selected_features', [])),
            'timestamp': self.current_results.get('timestamp', '')
        }

    def reset(self) -> None:
        """Reset the component to initial state."""
        self.is_fitted = False
        self.current_results = None
        self.clustering_history = []
        self.context = ClusteringContext()
        log_info("Component reset to initial state")
