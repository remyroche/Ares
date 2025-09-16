#!/usr/bin/env python3
"""
Comprehensive Validation and Error Handling for HMM Clustering

This module provides comprehensive validation and error handling using
all available common utilities for robust HMM clustering operations.
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings
from contextlib import contextmanager

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# HMM dependencies
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None

# Import common utilities
from src.utils.common_operations import (
    safe_dataframe_operation,
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    safe_correlation, validate_positive, validate_range
)
from src.utils.error_handler import ErrorHandler, ErrorRecovery
from src.utils.validation import DataValidator, ModelValidator
from src.utils.logger import system_logger

# Setup logging
logger = system_logger.getChild('ComprehensiveValidation')

@dataclass
class ValidationConfig:
    """Configuration for comprehensive validation."""
    # Data validation
    enable_data_validation: bool = True
    min_samples: int = 10
    max_samples: int = 1000000
    min_features: int = 1
    max_features: int = 1000
    max_missing_ratio: float = 0.5
    max_outlier_ratio: float = 0.1
    
    # Model validation
    enable_model_validation: bool = True
    min_components: int = 2
    max_components: int = 20
    min_iterations: int = 10
    max_iterations: int = 10000
    convergence_threshold: float = 1e-6
    
    # Error handling
    enable_error_recovery: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fallback: bool = True
    
    # Performance validation
    enable_performance_validation: bool = True
    min_silhouette_score: float = -1.0
    max_training_time: float = 3600.0  # 1 hour
    min_cluster_size: int = 5
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_validation_steps: bool = True
    enable_profiling: bool = False

class ComprehensiveValidator:
    """
    Comprehensive validator for HMM clustering operations.
    
    This class provides extensive validation and error handling
    using all available common utilities.
    """
    
    def __init__(self, config: ValidationConfig):
        """Initialize comprehensive validator."""
        self.config = config
        self.logger = logger.getChild('ComprehensiveValidator')
        
        # Initialize error handling
        self.error_handler = ErrorHandler()
        self.error_recovery = ErrorRecovery()
        
        # Initialize validators
        self.data_validator = DataValidator()
        self.model_validator = ModelValidator()
        
        # Validation state
        self.validation_history = []
        self.error_history = []
        self.recovery_attempts = 0
        
        self.logger.info("🔍 Comprehensive Validator initialized")
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log validation capabilities."""
        self.logger.info("🔍 Validation Capabilities:")
        self.logger.info(f"   Data Validation: {'✅ Enabled' if self.config.enable_data_validation else '❌ Disabled'}")
        self.logger.info(f"   Model Validation: {'✅ Enabled' if self.config.enable_model_validation else '❌ Disabled'}")
        self.logger.info(f"   Error Recovery: {'✅ Enabled' if self.config.enable_error_recovery else '❌ Disabled'}")
        self.logger.info(f"   Performance Validation: {'✅ Enabled' if self.config.enable_performance_validation else '❌ Disabled'}")
        self.logger.info(f"   Detailed Logging: {'✅ Enabled' if self.config.enable_detailed_logging else '❌ Disabled'}")
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     context: str = "data") -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive data validation."""
        if not self.config.enable_data_validation:
            return True, {}
        
        self.logger.info(f"🔍 Validating {context}...")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                data_array = data.values
                feature_names = data.columns.tolist()
            else:
                data_array = np.array(data)
                feature_names = None
            
            # Basic shape validation
            if data_array.ndim != 2:
                validation_results['errors'].append(f"Data must be 2D, got {data_array.ndim}D")
                validation_results['is_valid'] = False
            
            n_samples, n_features = data_array.shape
            
            # Sample count validation
            if n_samples < self.config.min_samples:
                validation_results['errors'].append(f"Too few samples: {n_samples} < {self.config.min_samples}")
                validation_results['is_valid'] = False
            elif n_samples > self.config.max_samples:
                validation_results['warnings'].append(f"Large dataset: {n_samples} > {self.config.max_samples}")
            
            # Feature count validation
            if n_features < self.config.min_features:
                validation_results['errors'].append(f"Too few features: {n_features} < {self.config.min_features}")
                validation_results['is_valid'] = False
            elif n_features > self.config.max_features:
                validation_results['warnings'].append(f"Many features: {n_features} > {self.config.max_features}")
            
            # Data quality validation
            if validation_results['is_valid']:
                quality_metrics = self._validate_data_quality(data_array)
                validation_results['metrics'].update(quality_metrics)
                
                # Check for missing values
                missing_ratio = np.isnan(data_array).sum() / data_array.size
                if missing_ratio > self.config.max_missing_ratio:
                    validation_results['errors'].append(f"Too many missing values: {missing_ratio:.2%} > {self.config.max_missing_ratio:.2%}")
                    validation_results['is_valid'] = False
                elif missing_ratio > 0:
                    validation_results['warnings'].append(f"Missing values present: {missing_ratio:.2%}")
                
                # Check for infinite values
                inf_count = np.isinf(data_array).sum()
                if inf_count > 0:
                    validation_results['errors'].append(f"Infinite values present: {inf_count}")
                    validation_results['is_valid'] = False
                
                # Check for outliers
                outlier_ratio = self._calculate_outlier_ratio(data_array)
                if outlier_ratio > self.config.max_outlier_ratio:
                    validation_results['warnings'].append(f"High outlier ratio: {outlier_ratio:.2%} > {self.config.max_outlier_ratio:.2%}")
                
                # Validate data types
                if not np.issubdtype(data_array.dtype, np.floating):
                    validation_results['warnings'].append(f"Data type is not floating point: {data_array.dtype}")
            
            # Log validation results
            if self.config.log_validation_steps:
                if validation_results['is_valid']:
                    self.logger.info(f"✅ {context} validation passed")
                else:
                    self.logger.error(f"❌ {context} validation failed: {validation_results['errors']}")
                
                if validation_results['warnings']:
                    self.logger.warning(f"⚠️ {context} validation warnings: {validation_results['warnings']}")
            
            # Store validation history
            self.validation_history.append({
                'timestamp': time.time(),
                'context': context,
                'results': validation_results.copy()
            })
            
            return validation_results['is_valid'], validation_results
            
        except Exception as e:
            error_msg = f"Data validation failed: {e}"
            self.logger.error(error_msg)
            validation_results['errors'].append(error_msg)
            validation_results['is_valid'] = False
            
            # Store error
            self.error_history.append({
                'timestamp': time.time(),
                'context': context,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            return False, validation_results
    
    def _validate_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Validate data quality metrics."""
        try:
            # Calculate basic statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data.T)
            max_correlation = np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
            
            # Calculate condition number
            try:
                condition_number = np.linalg.cond(data)
            except:
                condition_number = float('inf')
            
            return {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'max_correlation': max_correlation,
                'condition_number': condition_number,
                'shape': data.shape
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data quality validation failed: {e}")
            return {}
    
    def _calculate_outlier_ratio(self, data: np.ndarray) -> float:
        """Calculate ratio of outliers using IQR method."""
        try:
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)
            return np.mean(outliers)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Outlier calculation failed: {e}")
            return 0.0
    
    def validate_model_parameters(self, params: Dict[str, Any], 
                                context: str = "model parameters") -> Tuple[bool, Dict[str, Any]]:
        """Validate model parameters."""
        if not self.config.enable_model_validation:
            return True, {}
        
        self.logger.info(f"🔍 Validating {context}...")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'corrected_params': params.copy()
        }
        
        try:
            # Validate n_components
            n_components = params.get('n_components', 3)
            if not isinstance(n_components, int) or n_components < self.config.min_components:
                validation_results['errors'].append(f"Invalid n_components: {n_components}")
                validation_results['is_valid'] = False
            elif n_components > self.config.max_components:
                validation_results['warnings'].append(f"Large n_components: {n_components}")
                validation_results['corrected_params']['n_components'] = min(n_components, self.config.max_components)
            
            # Validate covariance_type
            covariance_type = params.get('covariance_type', 'full')
            valid_cov_types = ['full', 'tied', 'diag', 'spherical']
            if covariance_type not in valid_cov_types:
                validation_results['errors'].append(f"Invalid covariance_type: {covariance_type}")
                validation_results['is_valid'] = False
            
            # Validate n_iter
            n_iter = params.get('n_iter', 100)
            if not isinstance(n_iter, int) or n_iter < self.config.min_iterations:
                validation_results['errors'].append(f"Invalid n_iter: {n_iter}")
                validation_results['is_valid'] = False
            elif n_iter > self.config.max_iterations:
                validation_results['warnings'].append(f"Large n_iter: {n_iter}")
                validation_results['corrected_params']['n_iter'] = min(n_iter, self.config.max_iterations)
            
            # Validate random_state
            random_state = params.get('random_state', 42)
            if not isinstance(random_state, int) or random_state < 0:
                validation_results['warnings'].append(f"Invalid random_state: {random_state}")
                validation_results['corrected_params']['random_state'] = 42
            
            # Log validation results
            if self.config.log_validation_steps:
                if validation_results['is_valid']:
                    self.logger.info(f"✅ {context} validation passed")
                else:
                    self.logger.error(f"❌ {context} validation failed: {validation_results['errors']}")
                
                if validation_results['warnings']:
                    self.logger.warning(f"⚠️ {context} validation warnings: {validation_results['warnings']}")
            
            return validation_results['is_valid'], validation_results
            
        except Exception as e:
            error_msg = f"Model parameter validation failed: {e}"
            self.logger.error(error_msg)
            validation_results['errors'].append(error_msg)
            validation_results['is_valid'] = False
            
            return False, validation_results
    
    def validate_model_performance(self, model: Any, data: np.ndarray, 
                                 labels: np.ndarray, training_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Validate model performance."""
        if not self.config.enable_performance_validation:
            return True, {}
        
        self.logger.info("🔍 Validating model performance...")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Validate training time
            if training_time > self.config.max_training_time:
                validation_results['errors'].append(f"Training time too long: {training_time:.2f}s > {self.config.max_training_time:.2f}s")
                validation_results['is_valid'] = False
            
            # Validate cluster count
            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                validation_results['errors'].append(f"Too few clusters: {n_clusters}")
                validation_results['is_valid'] = False
            
            # Validate cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            min_cluster_size = np.min(counts)
            if min_cluster_size < self.config.min_cluster_size:
                validation_results['warnings'].append(f"Small cluster size: {min_cluster_size} < {self.config.min_cluster_size}")
            
            # Calculate clustering metrics
            if len(unique_labels) > 1:
                try:
                    silhouette = silhouette_score(data, labels)
                    calinski_harabasz = calinski_harabasz_score(data, labels)
                    davies_bouldin = davies_bouldin_score(data, labels)
                    
                    validation_results['metrics'] = {
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin,
                        'n_clusters': n_clusters,
                        'min_cluster_size': min_cluster_size,
                        'max_cluster_size': np.max(counts),
                        'training_time': training_time
                    }
                    
                    # Validate silhouette score
                    if silhouette < self.config.min_silhouette_score:
                        validation_results['warnings'].append(f"Low silhouette score: {silhouette:.3f} < {self.config.min_silhouette_score}")
                    
                except Exception as e:
                    validation_results['warnings'].append(f"Could not calculate clustering metrics: {e}")
            
            # Log validation results
            if self.config.log_validation_steps:
                if validation_results['is_valid']:
                    self.logger.info("✅ Model performance validation passed")
                else:
                    self.logger.error(f"❌ Model performance validation failed: {validation_results['errors']}")
                
                if validation_results['warnings']:
                    self.logger.warning(f"⚠️ Model performance validation warnings: {validation_results['warnings']}")
            
            return validation_results['is_valid'], validation_results
            
        except Exception as e:
            error_msg = f"Model performance validation failed: {e}"
            self.logger.error(error_msg)
            validation_results['errors'].append(error_msg)
            validation_results['is_valid'] = False
            
            return False, validation_results
    
    @contextmanager
    def error_handling_context(self, operation_name: str):
        """Context manager for error handling."""
        try:
            self.logger.info(f"🔄 Starting {operation_name}...")
            yield
            self.logger.info(f"✅ {operation_name} completed successfully")
        except Exception as e:
            self.logger.error(f"❌ {operation_name} failed: {e}")
            
            if self.config.enable_error_recovery:
                self._attempt_error_recovery(operation_name, e)
            else:
                raise
    
    def _attempt_error_recovery(self, operation_name: str, error: Exception):
        """Attempt error recovery."""
        if self.recovery_attempts >= self.config.max_retries:
            self.logger.error(f"❌ Maximum recovery attempts reached for {operation_name}")
            raise error
        
        self.recovery_attempts += 1
        self.logger.warning(f"🔄 Attempting error recovery for {operation_name} (attempt {self.recovery_attempts})")
        
        # Store error for analysis
        self.error_history.append({
            'timestamp': time.time(),
            'operation': operation_name,
            'error': str(error),
            'recovery_attempt': self.recovery_attempts
        })
        
        # Wait before retry
        time.sleep(self.config.retry_delay)
    
    def validate_hmm_training(self, data: Union[pd.DataFrame, np.ndarray], 
                            params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive HMM training validation."""
        self.logger.info("🔍 Starting comprehensive HMM training validation...")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'data_validation': {},
            'parameter_validation': {},
            'performance_validation': {}
        }
        
        try:
            # Validate data
            data_valid, data_results = self.validate_data(data, "training data")
            validation_results['data_validation'] = data_results
            if not data_valid:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(data_results['errors'])
            validation_results['warnings'].extend(data_results['warnings'])
            
            # Validate parameters
            param_valid, param_results = self.validate_model_parameters(params, "HMM parameters")
            validation_results['parameter_validation'] = param_results
            if not param_valid:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(param_results['errors'])
            validation_results['warnings'].extend(param_results['warnings'])
            
            # If validation passed, attempt training
            if validation_results['is_valid'] and HMM_AVAILABLE:
                try:
                    # Use corrected parameters if available
                    corrected_params = param_results.get('corrected_params', params)
                    
                    # Create and train model
                    model = hmm.GaussianHMM(**corrected_params)
                    
                    start_time = time.time()
                    model.fit(data)
                    training_time = time.time() - start_time
                    
                    # Get predictions
                    labels = model.predict(data)
                    
                    # Validate performance
                    perf_valid, perf_results = self.validate_model_performance(
                        model, data, labels, training_time
                    )
                    validation_results['performance_validation'] = perf_results
                    
                    if not perf_valid:
                        validation_results['is_valid'] = False
                        validation_results['errors'].extend(perf_results['errors'])
                    validation_results['warnings'].extend(perf_results['warnings'])
                    
                    # Add model to results
                    validation_results['model'] = model
                    validation_results['labels'] = labels
                    validation_results['training_time'] = training_time
                    
                except Exception as e:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"HMM training failed: {e}")
            
            # Log final results
            if validation_results['is_valid']:
                self.logger.info("✅ Comprehensive HMM training validation passed")
            else:
                self.logger.error(f"❌ Comprehensive HMM training validation failed: {validation_results['errors']}")
            
            return validation_results['is_valid'], validation_results
            
        except Exception as e:
            error_msg = f"Comprehensive validation failed: {e}"
            self.logger.error(error_msg)
            validation_results['is_valid'] = False
            validation_results['errors'].append(error_msg)
            
            return False, validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        return {
            'validation_history': self.validation_history,
            'error_history': self.error_history,
            'recovery_attempts': self.recovery_attempts,
            'config': self.config.__dict__
        }


def create_comprehensive_validator(config: Optional[ValidationConfig] = None) -> ComprehensiveValidator:
    """Factory function to create comprehensive validator instance."""
    if config is None:
        config = ValidationConfig()
    
    return ComprehensiveValidator(config)


# Example usage
if __name__ == "__main__":
    # Example usage
    logger.info("🔍 Comprehensive Validation Example")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 4
    
    # Generate sample data with 3 clusters
    cluster1 = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4), n_samples // 3)
    cluster2 = np.random.multivariate_normal([3, 3, 3, 3], np.eye(4), n_samples // 3)
    cluster3 = np.random.multivariate_normal([-3, -3, -3, -3], np.eye(4), n_samples - 2 * (n_samples // 3))
    
    sample_data = np.vstack([cluster1, cluster2, cluster3])
    
    # Create configuration
    config = ValidationConfig(
        enable_data_validation=True,
        enable_model_validation=True,
        enable_performance_validation=True,
        enable_error_recovery=True,
        enable_detailed_logging=True
    )
    
    # Create and use comprehensive validator
    validator = create_comprehensive_validator(config)
    
    # Test parameters
    test_params = {
        'n_components': 3,
        'covariance_type': 'full',
        'n_iter': 100,
        'random_state': 42
    }
    
    # Run comprehensive validation
    is_valid, results = validator.validate_hmm_training(sample_data, test_params)
    
    # Print results
    print(f"Validation passed: {is_valid}")
    print(f"Errors: {results['errors']}")
    print(f"Warnings: {results['warnings']}")
    
    if 'model' in results:
        print(f"Model trained successfully in {results['training_time']:.3f}s")
    
    # Get validation summary
    summary = validator.get_validation_summary()
    print(f"Validation summary: {len(summary['validation_history'])} validations performed")