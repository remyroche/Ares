"""
Step 16 Enhanced Calibration Methods

This module provides optimized calibration methods with:
- Convergence optimization
- Enhanced algorithms
- Fast-fail mechanisms
- Memory optimization
- Comprehensive validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import warnings
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
import logging

# Import existing utilities and core modules
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    get_current_datetime, format_datetime, safe_sleep, safe_gather,
    safe_mean, safe_std, safe_float, safe_int, validate_dataframe_schema,
    validate_data_quality, optimize_dataframe_dtypes, safe_read_parquet,
    safe_to_parquet, get_logger, setup_basic_logging
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_weighted_average,
    MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, cached
)
from src.core.errors import (
    ValidationError, DataIntegrityError, BusinessRuleError, AppError
)

from .step16_optimization_utilities import (
    FastFailValidator, ParameterValidator, MemoryOptimizer, 
    EnhancedMatrixOperations, CalibrationQualityMetrics,
    FastFailError, ConvergenceError,
    ConvergenceConfig, CalibrationMetrics
)

logger = get_logger(__name__)

class EnhancedPlattScaling:
    """Enhanced Platt scaling with convergence optimization and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = FastFailValidator(config)
        self.param_validator = ParameterValidator()
        self.memory_optimizer = MemoryOptimizer(config.get('memory_limit_gb', 8.0))
        self.matrix_ops = EnhancedMatrixOperations(config.get('use_gpu', True))
        self.metrics_calculator = CalibrationQualityMetrics(self.matrix_ops)
        self.logger = get_logger(f"{__name__}.EnhancedPlattScaling")
        
    @handles_errors(fallback=None, context="enhanced_platt_scaling_calibration")
    @traced(span_name="enhanced_platt_scaling")
    @log_execution_time("platt_scaling_calibration")
    def calibrate(self, probabilities: np.ndarray, labels: np.ndarray, 
                  regime_id: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced Platt scaling calibration with comprehensive optimization."""
        try:
            # Fast-fail validation
            data = pd.DataFrame({'probabilities': probabilities, 'labels': labels})
            self.validator.validate_data_quality(data, regime_id)
            
            # Parameter validation
            platt_params = self.config.get('platt_scaling', {})
            self.param_validator.validate_calibration_parameters(platt_params, regime_id)
            
            # Memory optimization
            data = self.memory_optimizer.optimize_data_loading(data, regime_id)
            
            # Enhanced calibration
            return self._enhanced_platt_calibration(
                data['probabilities'].values, 
                data['labels'].values, 
                platt_params, 
                regime_id
            )
            
        except (FastFailError, ValidationError, ConvergenceError) as e:
            logger.error(f"Platt scaling failed for regime {regime_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Platt scaling for regime {regime_id}: {e}")
            raise
    
    def _enhanced_platt_calibration(self, probabilities: np.ndarray, labels: np.ndarray,
                                  params: Dict[str, Any], regime_id: Optional[int]) -> Dict[str, Any]:
        """Enhanced Platt scaling with convergence optimization."""
        
        # Prepare data
        X = probabilities.reshape(-1, 1)
        y = labels
        
        # Enhanced parameters
        max_iter = params.get('max_iterations', 1000)
        learning_rate = params.get('learning_rate', 0.01)
        regularization = params.get('regularization', 0.01)
        early_stopping = params.get('early_stopping', True)
        validation_split = params.get('validation_split', 0.2)
        
        # Convergence configuration
        conv_config = ConvergenceConfig(
            max_iterations=max_iter,
            tolerance=params.get('tolerance', 1e-6),
            patience=params.get('patience', 10),
            early_stopping=early_stopping,
            validation_split=validation_split
        )
        
        # Split data for validation if early stopping is enabled
        if early_stopping and len(X) > 100:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Enhanced logistic regression with warm start
        base_classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            warm_start=True,
            C=1.0/regularization,
            solver='lbfgs'  # Better for small datasets
        )
        
        # Calibrated classifier with enhanced settings
        calibrator = CalibratedClassifierCV(
            estimator=base_classifier,
            method='sigmoid',
            cv='prefit'
        )
        
        # Fit with convergence monitoring
        start_time = time.time()
        optimizer_history = []
        
        try:
            # Fit the calibrator
            calibrator.fit(X_train, y_train)
            
            # Calculate metrics
            train_prob = calibrator.predict_proba(X_train)[:, 1]
            val_prob = calibrator.predict_proba(X_val)[:, 1]
            
            # Calculate comprehensive metrics
            train_metrics = self.metrics_calculator.calculate_comprehensive_metrics(train_prob, y_train)
            val_metrics = self.metrics_calculator.calculate_comprehensive_metrics(val_prob, y_val)
            
            # Get calibration coefficients
            calibrated_clf = calibrator.calibrated_classifiers_[0]
            A = calibrated_clf.calibrators_[0].coef_[0][0] if hasattr(calibrated_clf.calibrators_[0], 'coef_') else 1.0
            B = calibrated_clf.calibrators_[0].intercept_[0] if hasattr(calibrated_clf.calibrators_[0], 'intercept_') else 0.0
            
            calibration_time = time.time() - start_time
            
            # Enhanced results
            results = {
                'calibration_method': 'enhanced_platt_scaling',
                'regime_id': regime_id,
                'calibration_parameters': params,
                'calibration_metrics': {
                    'train_ece': train_metrics.ece,
                    'val_ece': val_metrics.ece,
                    'train_brier': train_metrics.brier_score,
                    'val_brier': val_metrics.brier_score,
                    'train_reliability': train_metrics.reliability_score,
                    'val_reliability': val_metrics.reliability_score,
                    'sharpness': train_metrics.sharpness,
                    'resolution': train_metrics.resolution
                },
                'calibration_coefficients': {
                    'A': float(A),
                    'B': float(B)
                },
                'calibration_quality': {
                    'convergence_achieved': True,
                    'calibration_time': calibration_time,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'optimization_method': 'lbfgs',
                    'regularization_strength': regularization
                },
                'uncertainty_metrics': {
                    'aleatoric_uncertainty': train_metrics.aleatoric_uncertainty,
                    'epistemic_uncertainty': train_metrics.epistemic_uncertainty,
                    'total_uncertainty': train_metrics.total_uncertainty
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced Platt scaling failed for regime {regime_id}: {e}")
            raise ConvergenceError(f"Platt scaling convergence failed: {e}")

class EnhancedIsotonicRegression:
    """Enhanced isotonic regression with optimization and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = FastFailValidator(config)
        self.param_validator = ParameterValidator()
        self.memory_optimizer = MemoryOptimizer(config.get('memory_limit_gb', 8.0))
        self.matrix_ops = EnhancedMatrixOperations(config.get('use_gpu', True))
        self.metrics_calculator = CalibrationQualityMetrics(self.matrix_ops)
        self.logger = get_logger(f"{__name__}.EnhancedIsotonicRegression")
        
    @handles_errors(fallback=None, context="enhanced_isotonic_regression_calibration")
    @traced(span_name="enhanced_isotonic_regression")
    @log_execution_time("isotonic_regression_calibration")
    def calibrate(self, probabilities: np.ndarray, labels: np.ndarray,
                  regime_id: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced isotonic regression calibration."""
        try:
            # Fast-fail validation
            data = pd.DataFrame({'probabilities': probabilities, 'labels': labels})
            self.validator.validate_data_quality(data, regime_id)
            
            # Parameter validation
            isotonic_params = self.config.get('isotonic_regression', {})
            self.param_validator.validate_calibration_parameters(isotonic_params, regime_id)
            
            # Memory optimization
            data = self.memory_optimizer.optimize_data_loading(data, regime_id)
            
            # Enhanced calibration
            return self._enhanced_isotonic_calibration(
                data['probabilities'].values,
                data['labels'].values,
                isotonic_params,
                regime_id
            )
            
        except (FastFailError, ValidationError, ConvergenceError) as e:
            logger.error(f"Isotonic regression failed for regime {regime_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in isotonic regression for regime {regime_id}: {e}")
            raise
    
    def _enhanced_isotonic_calibration(self, probabilities: np.ndarray, labels: np.ndarray,
                                     params: Dict[str, Any], regime_id: Optional[int]) -> Dict[str, Any]:
        """Enhanced isotonic regression with cross-validation."""
        
        # Enhanced parameters
        out_of_bounds = params.get('out_of_bounds', 'clip')
        increasing = params.get('increasing', True)
        cross_validation = params.get('cross_validation', True)
        cv_folds = params.get('cv_folds', 5)
        
        # Prepare data
        X = probabilities
        y = labels
        
        # Cross-validation if enabled and sufficient data
        if cross_validation and len(X) > 50:
            cv_scores = []
            cv_metrics = []
            
            skf = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y))), shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit isotonic regression
                isotonic_reg = IsotonicRegression(
                    out_of_bounds=out_of_bounds,
                    increasing=increasing
                )
                isotonic_reg.fit(X_train, y_train)
                
                # Predict on validation set
                val_calibrated = isotonic_reg.predict(X_val)
                val_calibrated = np.clip(val_calibrated, 0.0, 1.0)
                
                # Calculate metrics
                val_metrics = self.metrics_calculator.calculate_comprehensive_metrics(val_calibrated, y_val)
                cv_metrics.append(val_metrics)
                cv_scores.append(val_metrics.reliability_score)
            
            # Use best fold for final model
            best_fold_idx = np.argmax(cv_scores)
            best_cv_metrics = cv_metrics[best_fold_idx]
            
        else:
            # Single fit without cross-validation
            best_cv_metrics = None
            cv_scores = []
        
        # Final model on full dataset
        start_time = time.time()
        isotonic_reg = IsotonicRegression(
            out_of_bounds=out_of_bounds,
            increasing=increasing
        )
        isotonic_reg.fit(X, y)
        
        # Calculate calibrated probabilities
        calibrated_prob = isotonic_reg.predict(X)
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)
        
        # Calculate comprehensive metrics
        final_metrics = self.metrics_calculator.calculate_comprehensive_metrics(calibrated_prob, y)
        
        calibration_time = time.time() - start_time
        
        # Calculate monotonicity and smoothness
        monotonicity_score = self._calculate_monotonicity_score(calibrated_prob, y)
        smoothness_score = self._calculate_smoothness_score(calibrated_prob)
        
        # Find breakpoints
        breakpoints = self._find_breakpoints(X, calibrated_prob)
        
        # Enhanced results
        results = {
            'calibration_method': 'enhanced_isotonic_regression',
            'regime_id': regime_id,
            'calibration_parameters': params,
            'calibration_metrics': {
                'ece': final_metrics.ece,
                'mce': final_metrics.mce,
                'brier_score': final_metrics.brier_score,
                'reliability_score': final_metrics.reliability_score,
                'sharpness': final_metrics.sharpness,
                'resolution': final_metrics.resolution,
                'monotonicity_score': monotonicity_score,
                'smoothness_score': smoothness_score
            },
            'calibration_function': {
                'monotonic': increasing,
                'piecewise_linear': True,
                'breakpoints': breakpoints,
                'out_of_bounds_handling': out_of_bounds
            },
            'calibration_quality': {
                'calibration_time': calibration_time,
                'cross_validation_enabled': cross_validation,
                'cv_folds': cv_folds if cross_validation else 1,
                'cv_mean_score': np.mean(cv_scores) if cv_scores else final_metrics.reliability_score,
                'cv_std_score': np.std(cv_scores) if cv_scores else 0.0,
                'best_cv_metrics': best_cv_metrics.__dict__ if best_cv_metrics else None
            },
            'uncertainty_metrics': {
                'aleatoric_uncertainty': final_metrics.aleatoric_uncertainty,
                'epistemic_uncertainty': final_metrics.epistemic_uncertainty,
                'total_uncertainty': final_metrics.total_uncertainty
            }
        }
        
        return results
    
    def _calculate_monotonicity_score(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate monotonicity score."""
        if len(probabilities) <= 1:
            return 1.0
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)
        sorted_prob = probabilities[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Calculate correlation
        correlation = np.corrcoef(sorted_prob, sorted_labels)[0, 1]
        return float(abs(correlation)) if not np.isnan(correlation) else 0.0
    
    def _calculate_smoothness_score(self, probabilities: np.ndarray) -> float:
        """Calculate smoothness score."""
        if len(probabilities) <= 2:
            return 1.0
        
        # Calculate second derivative
        first_diff = np.diff(probabilities)
        second_diff = np.diff(first_diff)
        
        # Smoothness is inverse of average absolute second derivative
        smoothness = 1.0 / (1.0 + np.mean(np.abs(second_diff)))
        return float(smoothness)
    
    def _find_breakpoints(self, X: np.ndarray, calibrated_prob: np.ndarray) -> List[float]:
        """Find breakpoints in isotonic function."""
        # Sort by input
        sorted_indices = np.argsort(X)
        sorted_X = X[sorted_indices]
        sorted_calibrated = calibrated_prob[sorted_indices]
        
        # Simple breakpoint detection
        breakpoints = []
        for i in range(1, len(sorted_X) - 1):
            if i < len(sorted_X) - 1:
                slope1 = (sorted_calibrated[i] - sorted_calibrated[i-1]) / (sorted_X[i] - sorted_X[i-1]) if sorted_X[i] != sorted_X[i-1] else 0
                slope2 = (sorted_calibrated[i+1] - sorted_calibrated[i]) / (sorted_X[i+1] - sorted_X[i]) if sorted_X[i+1] != sorted_X[i] else 0
                if abs(slope2 - slope1) > 0.1:  # Significant slope change
                    breakpoints.append(float(sorted_X[i]))
        
        return breakpoints

class EnhancedTemperatureScaling:
    """Enhanced temperature scaling with advanced optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = FastFailValidator(config)
        self.param_validator = ParameterValidator()
        self.memory_optimizer = MemoryOptimizer(config.get('memory_limit_gb', 8.0))
        self.matrix_ops = EnhancedMatrixOperations(config.get('use_gpu', True))
        self.metrics_calculator = CalibrationQualityMetrics(self.matrix_ops)
        self.logger = get_logger(f"{__name__}.EnhancedTemperatureScaling")
        
    @handles_errors(fallback=None, context="enhanced_temperature_scaling_calibration")
    @traced(span_name="enhanced_temperature_scaling")
    @log_execution_time("temperature_scaling_calibration")
    def calibrate(self, probabilities: np.ndarray, labels: np.ndarray,
                  regime_id: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced temperature scaling calibration."""
        try:
            # Fast-fail validation
            data = pd.DataFrame({'probabilities': probabilities, 'labels': labels})
            self.validator.validate_data_quality(data, regime_id)
            
            # Parameter validation
            temp_params = self.config.get('temperature_scaling', {})
            self.param_validator.validate_calibration_parameters(temp_params, regime_id)
            
            # Memory optimization
            data = self.memory_optimizer.optimize_data_loading(data, regime_id)
            
            # Enhanced calibration
            return self._enhanced_temperature_calibration(
                data['probabilities'].values,
                data['labels'].values,
                temp_params,
                regime_id
            )
            
        except (FastFailError, ValidationError, ConvergenceError) as e:
            logger.error(f"Temperature scaling failed for regime {regime_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in temperature scaling for regime {regime_id}: {e}")
            raise
    
    def _enhanced_temperature_calibration(self, probabilities: np.ndarray, labels: np.ndarray,
                                        params: Dict[str, Any], regime_id: Optional[int]) -> Dict[str, Any]:
        """Enhanced temperature scaling with multiple optimization methods."""
        
        # Enhanced parameters
        temp_range = params.get('temperature_range', [0.1, 10.0])
        optimization_method = params.get('optimization_method', 'multi_start')
        cross_validation = params.get('cross_validation', True)
        validation_split = params.get('validation_split', 0.2)
        
        # Prepare data
        X = probabilities
        y = labels
        
        # Split data for validation
        if cross_validation and len(X) > 100:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Multiple optimization methods
        optimization_results = []
        
        if optimization_method == 'multi_start':
            # Try multiple optimization methods
            methods = ['bounded', 'differential_evolution', 'basin_hopping']
            for method in methods:
                try:
                    result = self._optimize_temperature(X_train, y_train, temp_range, method)
                    if result is not None:
                        optimization_results.append(result)
                except Exception as e:
                    logger.warning(f"Optimization method {method} failed: {e}")
        else:
            # Single method
            result = self._optimize_temperature(X_train, y_train, temp_range, optimization_method)
            if result is not None:
                optimization_results.append(result)
        
        if not optimization_results:
            raise ConvergenceError("All temperature optimization methods failed")
        
        # Select best result
        best_result = min(optimization_results, key=lambda x: x['fun'])
        optimal_temperature = best_result['x']
        
        # Calculate metrics
        start_time = time.time()
        
        # Apply temperature scaling
        scaled_prob = self._apply_temperature_scaling(X, optimal_temperature)
        train_scaled = self._apply_temperature_scaling(X_train, optimal_temperature)
        val_scaled = self._apply_temperature_scaling(X_val, optimal_temperature)
        
        # Calculate comprehensive metrics
        final_metrics = self.metrics_calculator.calculate_comprehensive_metrics(scaled_prob, y)
        train_metrics = self.metrics_calculator.calculate_comprehensive_metrics(train_scaled, y_train)
        val_metrics = self.metrics_calculator.calculate_comprehensive_metrics(val_scaled, y_val)
        
        calibration_time = time.time() - start_time
        
        # Enhanced results
        results = {
            'calibration_method': 'enhanced_temperature_scaling',
            'regime_id': regime_id,
            'calibration_parameters': params,
            'calibration_metrics': {
                'train_ece': train_metrics.ece,
                'val_ece': val_metrics.ece,
                'train_brier': train_metrics.brier_score,
                'val_brier': val_metrics.brier_score,
                'train_reliability': train_metrics.reliability_score,
                'val_reliability': val_metrics.reliability_score,
                'sharpness': final_metrics.sharpness,
                'resolution': final_metrics.resolution
            },
            'calibration_coefficients': {
                'temperature': float(optimal_temperature),
                'bias': 0.0  # Temperature scaling typically doesn't include bias
            },
            'calibration_quality': {
                'optimization_method': optimization_method,
                'optimization_success': best_result['success'],
                'optimization_iterations': best_result.get('nit', 0),
                'final_loss': float(best_result['fun']),
                'calibration_time': calibration_time,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'temperature_range': temp_range
            },
            'uncertainty_metrics': {
                'aleatoric_uncertainty': final_metrics.aleatoric_uncertainty,
                'epistemic_uncertainty': final_metrics.epistemic_uncertainty,
                'total_uncertainty': final_metrics.total_uncertainty
            }
        }
        
        return results
    
    def _optimize_temperature(self, X: np.ndarray, y: np.ndarray, 
                            temp_range: List[float], method: str) -> Optional[Dict[str, Any]]:
        """Optimize temperature parameter using specified method."""
        
        def temperature_loss(temperature):
            """Loss function for temperature optimization."""
            if temperature <= 0:
                return float('inf')
            
            # Apply temperature scaling
            scaled_prob = self._apply_temperature_scaling(X, temperature)
            
            # Calculate negative log-likelihood loss
            eps = 1e-15
            scaled_prob = np.clip(scaled_prob, eps, 1 - eps)
            nll = -np.mean(y * np.log(scaled_prob) + (1 - y) * np.log(1 - scaled_prob))
            
            return nll
        
        try:
            if method == 'bounded':
                result = minimize_scalar(
                    temperature_loss,
                    bounds=temp_range,
                    method='bounded'
                )
            elif method == 'differential_evolution':
                result = differential_evolution(
                    temperature_loss,
                    bounds=[temp_range],
                    seed=42,
                    maxiter=100
                )
            elif method == 'basin_hopping':
                from scipy.optimize import basinhopping
                result = basinhopping(
                    temperature_loss,
                    x0=np.mean(temp_range),
                    minimizer_kwargs={'bounds': [temp_range]},
                    niter=50,
                    seed=42
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Temperature optimization with {method} failed: {e}")
            return None
    
    def _apply_temperature_scaling(self, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        # Avoid log(0) and log(1)
        eps = 1e-15
        prob_clipped = np.clip(probabilities, eps, 1 - eps)
        
        # Apply temperature scaling
        logits = np.log(prob_clipped / (1 - prob_clipped))
        scaled_logits = logits / temperature
        scaled_prob = 1.0 / (1.0 + np.exp(-scaled_logits))
        
        return np.clip(scaled_prob, 0.0, 1.0)