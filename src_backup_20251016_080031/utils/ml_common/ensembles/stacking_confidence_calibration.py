"""
Stacking Confidence Calibration for Multi-Output Models

This module provides comprehensive confidence calibration for multi-output stacking
ensemble models, including Platt scaling and isotonic calibration methods.

Key Features:
- StackingConfidenceCalibrator for multi-output confidence
- Platt scaling and isotonic calibration methods
- Per-output confidence calibration
- Calibration metrics calculation
- M1 hardware optimization integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

# M1 Optimization imports
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.memory_optimization import get_memory_manager, MemoryMonitor

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError
)

# Import confidence metrics
from ..confidence_metrics import (
    calculate_confidence_metrics, calculate_calibration_metrics,
    calculate_expected_calibration_error, ModelConfidenceCalibration
)

logger = logging.getLogger(__name__)


@dataclass
class StackingCalibrationConfig:
    """Configuration for stacking confidence calibration."""
    # Basic configuration
    calibrator_name: str
    n_outputs: int = 4
    output_names: List[str] = field(default_factory=lambda: ["output_1", "output_2", "output_3", "output_4"])
    
    # Calibration methods
    calibration_methods: List[str] = field(default_factory=lambda: ["platt_scaling", "isotonic_regression"])
    enable_temperature_scaling: bool = True
    enable_histogram_binning: bool = True
    enable_bayesian_calibration: bool = True
    
    # Calibration parameters
    platt_scaling_params: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.01,
        'max_iterations': 1000,
        'convergence_threshold': 1e-6
    })
    isotonic_regression_params: Dict[str, Any] = field(default_factory=lambda: {
        'out_of_bounds': 'clip',
        'increasing': True
    })
    temperature_scaling_params: Dict[str, Any] = field(default_factory=lambda: {
        'temperature_range': [0.1, 10.0],
        'optimization_method': 'lbfgs'
    })
    histogram_binning_params: Dict[str, Any] = field(default_factory=lambda: {
        'bin_count': 10,
        'bin_strategy': 'uniform'
    })
    bayesian_calibration_params: Dict[str, Any] = field(default_factory=lambda: {
        'prior_strength': 1.0,
        'mcmc_samples': 1000,
        'burn_in_samples': 100
    })
    
    # Multi-output specific settings
    enable_output_correlation: bool = True
    correlation_threshold: float = 0.7
    enable_joint_calibration: bool = False
    joint_calibration_weight: float = 0.5
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_online_learning: bool = False
    
    # Output settings
    save_calibrators: bool = True
    save_predictions: bool = True
    generate_reports: bool = True


@dataclass
class StackingCalibrationResult:
    """Result from stacking confidence calibration."""
    # Basic info
    calibrator_name: str
    n_outputs: int
    output_names: List[str]
    created_at: datetime
    total_duration: float
    
    # Calibration results
    calibrated_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    calibration_metrics: Dict[str, Any] = field(default_factory=dict)
    per_output_calibration: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Model performance
    calibration_quality: Dict[str, str] = field(default_factory=dict)
    calibration_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    config: StackingCalibrationConfig = field(default_factory=StackingCalibrationConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class StackingConfidenceCalibrator:
    """Comprehensive confidence calibration for multi-output stacking models."""
    
    def __init__(self, config: StackingCalibrationConfig):
        """Initialize the stacking confidence calibrator."""
        self.logger = logger.getChild('StackingConfidenceCalibrator')
        self.logger.info(f"🚀 Initializing StackingConfidenceCalibrator: {config.calibrator_name}")
        start_time = time.time()
        
        self.config = config
        
        # Initialize M1 optimizers
        self.logger.debug("🔧 Initializing M1 optimizers...")
        self.m1_gpu = get_m1_memory_optimizer() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_memory_manager() if config.enable_parallel_processing else None
        
        self.logger.debug("✅ M1 optimizers initialized")
        
        # Initialize calibration models
        self.calibrators: Dict[str, Dict[str, Any]] = {}
        self.is_fitted = False
        
        # Performance tracking
        self.calibration_history: List[Dict[str, Any]] = []
        self.prediction_history: List[Dict[str, Any]] = []
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ StackingConfidenceCalibrator initialized in {init_time:.3f}s")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"📊 Outputs: {config.n_outputs} ({config.output_names})")
        self.logger.info(f"🎯 Calibration methods: {config.calibration_methods}")
    
    @traced(span_name='calibrate_confidence')
    def calibrate_confidence(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None) -> StackingCalibrationResult:
        """Calibrate confidence for multi-output predictions."""
        
        self.logger.info(f"🔄 Calibrating confidence for {y_true.shape[0]} samples")
        start_time = time.time()
        
        try:
            # Validate inputs
            self.logger.debug("🔍 Validating inputs...")
            self._validate_calibration_inputs(y_true, y_pred, y_pred_proba)
            self.logger.debug("✅ Input validation passed")
            
            # Memory optimization context
            if self.m1_memory:
                self.logger.debug("🧠 Using memory optimization context...")
                with self.m1_memory.optimization_context():
                    result = self._calibrate_confidence_internal(y_true, y_pred, y_pred_proba)
            else:
                self.logger.debug("🧠 No memory optimization available, proceeding normally...")
                result = self._calibrate_confidence_internal(y_true, y_pred, y_pred_proba)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Log memory usage
            if self.m1_memory:
                result.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
                self.logger.info(f"🧠 Memory usage: {result.memory_usage_mb:.1f} MB")
            
            self.logger.info(f"✅ Confidence calibration completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Calibration quality: {result.calibration_quality}")
            self.logger.info(f"📈 Calibration improvement: {result.calibration_improvement}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Failed to calibrate confidence after {execution_time:.3f}s: {e}")
            raise
    
    def _validate_calibration_inputs(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: Optional[np.ndarray]) -> None:
        """Validate calibration inputs."""
        
        # Check shapes
        if len(y_true.shape) != 2 or y_true.shape[1] != self.config.n_outputs:
            raise ValidationError(f"Invalid y_true shape: {y_true.shape}, expected (n_samples, {self.config.n_outputs})")
        
        if len(y_pred.shape) != 2 or y_pred.shape[1] != self.config.n_outputs:
            raise ValidationError(f"Invalid y_pred shape: {y_pred.shape}, expected (n_samples, {self.config.n_outputs})")
        
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) != 2 or y_pred_proba.shape[1] != self.config.n_outputs:
                raise ValidationError(f"Invalid y_pred_proba shape: {y_pred_proba.shape}, expected (n_samples, {self.config.n_outputs})")
        
        # Check sample count
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValidationError("Sample count mismatch between y_true and y_pred")
        
        if y_pred_proba is not None and y_true.shape[0] != y_pred_proba.shape[0]:
            raise ValidationError("Sample count mismatch between y_true and y_pred_proba")
        
        self.logger.debug("✅ Input validation passed")
    
    def _calibrate_confidence_internal(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_pred_proba: Optional[np.ndarray]) -> StackingCalibrationResult:
        """Internal confidence calibration logic."""
        
        self.logger.debug("🔄 Starting internal confidence calibration...")
        internal_start_time = time.time()
        
        # Initialize calibration results
        calibrated_predictions = y_pred.copy()
        calibration_metrics = {}
        per_output_calibration = {}
        calibration_quality = {}
        calibration_improvement = {}
        
        # Calibrate each output separately
        self.logger.debug("🔄 Calibrating each output separately...")
        for output_idx, output_name in enumerate(self.config.output_names):
            self.logger.debug(f"🔄 Calibrating output {output_name}...")
            
            # Get data for this output
            y_true_output = y_true[:, output_idx]
            y_pred_output = y_pred[:, output_idx]
            y_pred_proba_output = y_pred_proba[:, output_idx] if y_pred_proba is not None else None
            
            # Calibrate this output
            output_result = self._calibrate_single_output(
                output_name, y_true_output, y_pred_output, y_pred_proba_output
            )
            
            # Store results
            per_output_calibration[output_name] = output_result
            calibrated_predictions[:, output_idx] = output_result['calibrated_predictions']
            calibration_quality[output_name] = output_result['calibration_quality']
            calibration_improvement[output_name] = output_result['calibration_improvement']
            
            self.logger.debug(f"✅ Output {output_name} calibrated: {output_result['calibration_quality']}")
        
        # Joint calibration if enabled
        if self.config.enable_joint_calibration:
            self.logger.debug("🔄 Performing joint calibration...")
            joint_result = self._calibrate_joint_outputs(y_true, y_pred, y_pred_proba)
            calibration_metrics['joint_calibration'] = joint_result
        
        # Calculate overall calibration metrics
        self.logger.debug("📊 Calculating overall calibration metrics...")
        overall_metrics = self._calculate_overall_calibration_metrics(
            y_true, y_pred, calibrated_predictions
        )
        calibration_metrics['overall'] = overall_metrics
        
        # Create result
        result = StackingCalibrationResult(
            calibrator_name=self.config.calibrator_name,
            n_outputs=self.config.n_outputs,
            output_names=self.config.output_names,
            created_at=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            calibrated_predictions=calibrated_predictions,
            calibration_metrics=calibration_metrics,
            per_output_calibration=per_output_calibration,
            calibration_quality=calibration_quality,
            calibration_improvement=calibration_improvement,
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        # Update state
        self.is_fitted = True
        
        # Record calibration history
        internal_time = time.time() - internal_start_time
        self.calibration_history.append({
            'timestamp': datetime.now(),
            'duration': internal_time,
            'n_samples': y_true.shape[0],
            'n_outputs': y_true.shape[1],
            'calibration_quality': calibration_quality,
            'calibration_improvement': calibration_improvement
        })
        
        self.logger.info(f"✅ Internal confidence calibration completed in {internal_time:.3f}s")
        
        return result
    
    def _calibrate_single_output(self, output_name: str, y_true: np.ndarray,
                                y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calibrate a single output."""
        
        self.logger.debug(f"🔄 Calibrating single output: {output_name}")
        
        # Initialize calibration result
        result = {
            'output_name': output_name,
            'calibrated_predictions': y_pred.copy(),
            'calibration_quality': 'unknown',
            'calibration_improvement': 0.0,
            'calibration_methods': {}
        }
        
        # Try different calibration methods
        best_method = None
        best_improvement = -float('inf')
        
        for method in self.config.calibration_methods:
            try:
                self.logger.debug(f"🔄 Trying calibration method: {method}")
                
                if method == "platt_scaling":
                    method_result = self._apply_platt_scaling(y_true, y_pred, y_pred_proba)
                elif method == "isotonic_regression":
                    method_result = self._apply_isotonic_regression(y_true, y_pred, y_pred_proba)
                elif method == "temperature_scaling" and self.config.enable_temperature_scaling:
                    method_result = self._apply_temperature_scaling(y_true, y_pred, y_pred_proba)
                elif method == "histogram_binning" and self.config.enable_histogram_binning:
                    method_result = self._apply_histogram_binning(y_true, y_pred, y_pred_proba)
                elif method == "bayesian_calibration" and self.config.enable_bayesian_calibration:
                    method_result = self._apply_bayesian_calibration(y_true, y_pred, y_pred_proba)
                else:
                    continue
                
                if method_result and 'error' not in method_result:
                    result['calibration_methods'][method] = method_result
                    
                    # Check if this is the best method
                    improvement = method_result.get('calibration_improvement', 0.0)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_method = method
                        result['calibrated_predictions'] = method_result['calibrated_predictions']
                        result['calibration_quality'] = method_result.get('calibration_quality', 'unknown')
                        result['calibration_improvement'] = improvement
                
                self.logger.debug(f"✅ Method {method} completed: improvement={improvement:.4f}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Method {method} failed for {output_name}: {e}")
                continue
        
        if best_method:
            result['best_method'] = best_method
            self.logger.debug(f"✅ Best method for {output_name}: {best_method} (improvement: {best_improvement:.4f})")
        else:
            self.logger.warning(f"⚠️ No successful calibration method for {output_name}")
            result['best_method'] = 'none'
        
        return result
    
    def _apply_platt_scaling(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply Platt scaling calibration."""
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            
            # Use probabilities if available, otherwise use predictions
            if y_pred_proba is not None:
                prob_data = y_pred_proba.reshape(-1, 1)
            else:
                prob_data = y_pred.reshape(-1, 1)
            
            # Create and fit Platt scaling calibrator
            base_classifier = LogisticRegression(
                max_iter=self.config.platt_scaling_params['max_iterations'],
                random_state=42
            )
            
            calibrator = CalibratedClassifierCV(
                estimator=base_classifier,
                method='sigmoid',
                cv='prefit'
            )
            
            # Fit calibrator
            calibrator.fit(prob_data, y_true)
            
            # Calculate calibrated probabilities
            calibrated_prob = calibrator.predict_proba(prob_data)[:, 1]
            
            # Calculate metrics
            brier_before = self._calculate_brier_score(y_true, prob_data[:, 0])
            brier_after = self._calculate_brier_score(y_true, calibrated_prob)
            improvement = brier_before - brier_after
            
            return {
                'method': 'platt_scaling',
                'calibrated_predictions': calibrated_prob,
                'calibration_improvement': float(improvement),
                'calibration_quality': 'good' if improvement > 0.01 else 'fair' if improvement > 0 else 'poor',
                'brier_before': float(brier_before),
                'brier_after': float(brier_after)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Platt scaling failed: {e}")
            return {'error': str(e)}
    
    def _apply_isotonic_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply isotonic regression calibration."""
        
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Use probabilities if available, otherwise use predictions
            if y_pred_proba is not None:
                prob_data = y_pred_proba
            else:
                prob_data = y_pred
            
            # Create and fit isotonic regression
            isotonic_reg = IsotonicRegression(
                out_of_bounds=self.config.isotonic_regression_params['out_of_bounds'],
                increasing=self.config.isotonic_regression_params['increasing']
            )
            
            isotonic_reg.fit(prob_data, y_true)
            
            # Calculate calibrated probabilities
            calibrated_prob = isotonic_reg.predict(prob_data)
            calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)
            
            # Calculate metrics
            brier_before = self._calculate_brier_score(y_true, prob_data)
            brier_after = self._calculate_brier_score(y_true, calibrated_prob)
            improvement = brier_before - brier_after
            
            return {
                'method': 'isotonic_regression',
                'calibrated_predictions': calibrated_prob,
                'calibration_improvement': float(improvement),
                'calibration_quality': 'good' if improvement > 0.01 else 'fair' if improvement > 0 else 'poor',
                'brier_before': float(brier_before),
                'brier_after': float(brier_after)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Isotonic regression failed: {e}")
            return {'error': str(e)}
    
    def _apply_temperature_scaling(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply temperature scaling calibration."""
        
        try:
            from scipy.optimize import minimize_scalar
            
            # Use probabilities if available, otherwise use predictions
            if y_pred_proba is not None:
                prob_data = y_pred_proba
            else:
                prob_data = y_pred
            
            # Get temperature range
            temp_range = self.config.temperature_scaling_params['temperature_range']
            
            # Optimize temperature parameter
            def temperature_loss(temperature):
                if temperature <= 0:
                    return float('inf')
                
                # Apply temperature scaling
                scaled_prob = 1.0 / (1.0 + np.exp(-(np.log(prob_data / (1 - prob_data + 1e-8)) / temperature)))
                
                # Calculate negative log-likelihood loss
                eps = 1e-15
                scaled_prob = np.clip(scaled_prob, eps, 1 - eps)
                nll = -np.mean(y_true * np.log(scaled_prob) + (1 - y_true) * np.log(1 - scaled_prob))
                
                return nll
            
            # Optimize temperature
            result = minimize_scalar(
                temperature_loss,
                bounds=temp_range,
                method='bounded'
            )
            
            optimal_temperature = result.x
            optimization_converged = result.success
            
            # Apply optimal temperature scaling
            scaled_prob = 1.0 / (1.0 + np.exp(-(np.log(prob_data / (1 - prob_data + 1e-8)) / optimal_temperature)))
            scaled_prob = np.clip(scaled_prob, 0.0, 1.0)
            
            # Calculate metrics
            brier_before = self._calculate_brier_score(y_true, prob_data)
            brier_after = self._calculate_brier_score(y_true, scaled_prob)
            improvement = brier_before - brier_after
            
            return {
                'method': 'temperature_scaling',
                'calibrated_predictions': scaled_prob,
                'calibration_improvement': float(improvement),
                'calibration_quality': 'good' if improvement > 0.01 else 'fair' if improvement > 0 else 'poor',
                'brier_before': float(brier_before),
                'brier_after': float(brier_after),
                'optimal_temperature': float(optimal_temperature),
                'optimization_converged': bool(optimization_converged)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Temperature scaling failed: {e}")
            return {'error': str(e)}
    
    def _apply_histogram_binning(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply histogram binning calibration."""
        
        try:
            # Use probabilities if available, otherwise use predictions
            if y_pred_proba is not None:
                prob_data = y_pred_proba
            else:
                prob_data = y_pred
            
            # Get bin parameters
            bin_count = self.config.histogram_binning_params['bin_count']
            bin_strategy = self.config.histogram_binning_params['bin_strategy']
            
            # Create bins
            if bin_strategy == 'uniform':
                bins = np.linspace(0, 1, bin_count + 1)
            else:
                # Quantile-based bins
                bins = np.quantile(prob_data, np.linspace(0, 1, bin_count + 1))
                bins[0] = 0.0
                bins[-1] = 1.0
            
            # Digitize probabilities into bins
            bin_indices = np.digitize(prob_data, bins) - 1
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)
            
            # Calculate bin statistics
            bin_counts = []
            bin_accuracies = []
            
            for bin_idx in range(bin_count):
                bin_mask = bin_indices == bin_idx
                bin_count_val = np.sum(bin_mask)
                bin_counts.append(int(bin_count_val))
                
                if bin_count_val > 0:
                    bin_accuracy = np.mean(y_true[bin_mask])
                else:
                    bin_accuracy = 0.5
                
                bin_accuracies.append(float(bin_accuracy))
            
            # Create calibrated probabilities using bin averages
            calibrated_prob = np.zeros_like(prob_data)
            for bin_idx in range(bin_count):
                bin_mask = bin_indices == bin_idx
                calibrated_prob[bin_mask] = bin_accuracies[bin_idx]
            
            # Calculate metrics
            brier_before = self._calculate_brier_score(y_true, prob_data)
            brier_after = self._calculate_brier_score(y_true, calibrated_prob)
            improvement = brier_before - brier_after
            
            return {
                'method': 'histogram_binning',
                'calibrated_predictions': calibrated_prob,
                'calibration_improvement': float(improvement),
                'calibration_quality': 'good' if improvement > 0.01 else 'fair' if improvement > 0 else 'poor',
                'brier_before': float(brier_before),
                'brier_after': float(brier_after),
                'bin_counts': bin_counts,
                'bin_accuracies': bin_accuracies
            }
            
        except Exception as e:
            self.logger.error(f"❌ Histogram binning failed: {e}")
            return {'error': str(e)}
    
    def _apply_bayesian_calibration(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply Bayesian calibration."""
        
        try:
            from scipy.stats import beta
            
            # Use probabilities if available, otherwise use predictions
            if y_pred_proba is not None:
                prob_data = y_pred_proba
            else:
                prob_data = y_pred
            
            # Bayesian calibration using beta distribution
            prior_strength = self.config.bayesian_calibration_params['prior_strength']
            
            # Group predictions into bins for Bayesian estimation
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(prob_data, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Estimate beta parameters for each bin
            alpha_estimates = []
            beta_estimates = []
            
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                bin_labels = y_true[bin_mask]
                
                if len(bin_labels) == 0:
                    # Use prior for empty bins
                    alpha_estimates.append(prior_strength)
                    beta_estimates.append(prior_strength)
                else:
                    # Calculate posterior parameters
                    successes = np.sum(bin_labels)
                    failures = len(bin_labels) - successes
                    
                    alpha_post = prior_strength + successes
                    beta_post = prior_strength + failures
                    
                    alpha_estimates.append(alpha_post)
                    beta_estimates.append(beta_post)
            
            # Create calibrated probabilities using beta means
            calibrated_prob = np.zeros_like(prob_data)
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if np.any(bin_mask):
                    # Use beta mean for calibration
                    calibrated_prob[bin_mask] = alpha_estimates[bin_idx] / (alpha_estimates[bin_idx] + beta_estimates[bin_idx])
            
            # Calculate metrics
            brier_before = self._calculate_brier_score(y_true, prob_data)
            brier_after = self._calculate_brier_score(y_true, calibrated_prob)
            improvement = brier_before - brier_after
            
            return {
                'method': 'bayesian_calibration',
                'calibrated_predictions': calibrated_prob,
                'calibration_improvement': float(improvement),
                'calibration_quality': 'good' if improvement > 0.01 else 'fair' if improvement > 0 else 'poor',
                'brier_before': float(brier_before),
                'brier_after': float(brier_after),
                'alpha_parameters': alpha_estimates,
                'beta_parameters': beta_estimates
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian calibration failed: {e}")
            return {'error': str(e)}
    
    def _calibrate_joint_outputs(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calibrate outputs jointly."""
        
        self.logger.debug("🔄 Performing joint calibration...")
        
        try:
            # This is a placeholder for joint calibration
            # In practice, you would implement joint calibration logic
            # that considers correlations between outputs
            
            return {
                'method': 'joint_calibration',
                'calibration_improvement': 0.0,
                'calibration_quality': 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Joint calibration failed: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                             calibrated_predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate overall calibration metrics."""
        
        try:
            # Calculate overall Brier score
            brier_before = np.mean([self._calculate_brier_score(y_true[:, i], y_pred[:, i]) 
                                  for i in range(y_true.shape[1])])
            brier_after = np.mean([self._calculate_brier_score(y_true[:, i], calibrated_predictions[:, i]) 
                                 for i in range(y_true.shape[1])])
            
            # Calculate overall improvement
            improvement = brier_before - brier_after
            
            # Calculate overall calibration quality
            if improvement > 0.05:
                quality = 'excellent'
            elif improvement > 0.01:
                quality = 'good'
            elif improvement > 0:
                quality = 'fair'
            else:
                quality = 'poor'
            
            return {
                'brier_before': float(brier_before),
                'brier_after': float(brier_after),
                'improvement': float(improvement),
                'calibration_quality': quality
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate overall calibration metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Brier score."""
        try:
            return float(np.mean((y_true - y_pred) ** 2))
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate Brier score: {e}")
            return 0.0
    
    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        self.logger.debug("🔍 Getting list of optimizations used...")
        
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
            self.logger.debug("✅ M1 GPU acceleration enabled")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
            self.logger.debug("✅ M1 memory optimization enabled")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
            self.logger.debug("✅ M1 parallel processing enabled")
        
        self.logger.debug(f"📊 Optimizations used: {optimizations}")
        return optimizations
    
    def save_calibrator(self, file_path: str) -> None:
        """Save the calibrator to disk."""
        
        try:
            import pickle
            
            calibrator_data = {
                'config': self.config,
                'is_fitted': self.is_fitted,
                'calibrators': self.calibrators,
                'calibration_history': self.calibration_history,
                'prediction_history': self.prediction_history
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(calibrator_data, f)
            
            self.logger.info(f"💾 Calibrator saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save calibrator: {e}")
            raise
    
    def load_calibrator(self, file_path: str) -> None:
        """Load the calibrator from disk."""
        
        try:
            
            with open(file_path, 'rb') as f:
                calibrator_data = pickle.load(f)
            
            self.config = calibrator_data['config']
            self.is_fitted = calibrator_data['is_fitted']
            self.calibrators = calibrator_data['calibrators']
            self.calibration_history = calibrator_data['calibration_history']
            self.prediction_history = calibrator_data['prediction_history']
            
            self.logger.info(f"📂 Calibrator loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load calibrator: {e}")
            raise


# Convenience functions for creating specific calibrators
def create_analyst_calibrator(output_dir: str = "./analyst_calibrator") -> StackingConfidenceCalibrator:
    """Create an Analyst (5m) confidence calibrator."""
    
    config = StackingCalibrationConfig(
        calibrator_name="analyst_calibrator",
        n_outputs=4,
        output_names=["signal_strength", "confidence", "risk_score", "regime_label"],
        calibration_methods=["platt_scaling", "isotonic_regression"],
        enable_temperature_scaling=True,
        enable_histogram_binning=True,
        enable_bayesian_calibration=True
    )
    
    return StackingConfidenceCalibrator(config)


def create_tactician_calibrator(output_dir: str = "./tactician_calibrator") -> StackingConfidenceCalibrator:
    """Create a Tactician (1m) confidence calibrator."""
    
    config = StackingCalibrationConfig(
        calibrator_name="tactician_calibrator",
        n_outputs=4,
        output_names=["entry_timing", "position_size", "stop_loss", "take_profit"],
        calibration_methods=["platt_scaling", "isotonic_regression"],
        enable_temperature_scaling=True,
        enable_histogram_binning=True,
        enable_bayesian_calibration=True
    )
    
    return StackingConfidenceCalibrator(config)