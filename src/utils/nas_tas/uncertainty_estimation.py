"""
Enhanced Uncertainty Estimation for NAS/TAS Tree Architecture

This module provides comprehensive uncertainty estimation methods for tree architecture predictions,
integrating with shared utilities for robust and efficient uncertainty quantification.

Key Features:
- Multiple uncertainty estimation methods (bootstrap, Monte Carlo, ensemble, Bayesian)
- Integration with shared utilities for validation, optimization, and hardware acceleration
- Advanced uncertainty metrics and confidence interval estimation
- Support for both regression and classification tasks
- Memory-efficient batch processing
- M1 hardware optimization support
- Comprehensive logging and error handling

Usage:
    from src.utils.nas_tas.uncertainty_estimation import TreeUncertaintyEstimator, UncertaintyConfig
    
    config = UncertaintyConfig(method='bootstrap', n_samples=100)
    estimator = TreeUncertaintyEstimator(config)
    estimator.fit(X, y, model_params)
    predictions, uncertainty = estimator.predict_with_uncertainty(X_test)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import shared utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, 
        validate_positive, validate_range, safe_mean, safe_std,
        safe_correlation, safe_covariance, safe_percentile,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        memory_checkpoint, gpu_context, optimize_memory
    )
    from src.utils.math_validation import (
        MathValidation, validate_numeric_array, safe_matrix_inverse,
        validate_correlation_matrix, math_safe
    )
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, BayesianTPEConfig
except ImportError as e:
    # Fallback imports for development
    warnings.warn(f"Some shared utilities not available: {e}")
    
    def safe_divide(a, b, default=0.0):
        return a / b if b != 0 else default
    
    def safe_log(x, default=0.0):
        return np.log(x) if x > 0 else default
    
    def safe_sqrt(x, default=0.0):
        return np.sqrt(x) if x >= 0 else default
    
    def safe_mean(x, default=0.0):
        return np.mean(x) if len(x) > 0 else default
    
    def safe_std(x, default=0.0):
        return np.std(x, ddof=1) if len(x) > 1 else default
    
    def validate_finite(value, name="value"):
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        return val
    
    def validate_positive(value, name="value"):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value
    
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    
    tprint_info = tprint_warning = tprint_error = tprint_success = tprint

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Enhanced configuration for uncertainty estimation."""
    # Core uncertainty estimation parameters
    n_samples: int = 100
    confidence_level: float = 0.95
    method: str = 'bootstrap'  # 'bootstrap', 'monte_carlo', 'ensemble', 'bayesian'
    
    # Advanced uncertainty estimation options
    enable_aleatoric_uncertainty: bool = True
    enable_epistemic_uncertainty: bool = True
    uncertainty_aggregation: str = 'mean'  # 'mean', 'median', 'max', 'weighted'
    
    # Bootstrap configuration
    bootstrap_strategy: str = 'balanced'  # 'balanced', 'stratified', 'random'
    bootstrap_replace: bool = True
    bootstrap_weights: Optional[np.ndarray] = None
    
    # Monte Carlo configuration
    mc_noise_std: float = 0.1
    mc_dropout_rate: float = 0.1
    mc_iterations: int = 50
    
    # Ensemble configuration
    ensemble_diversity: str = 'parameter'  # 'parameter', 'data', 'architecture'
    ensemble_size: int = 10
    ensemble_uncertainty_weight: float = 0.5
    
    # Bayesian configuration
    bayesian_prior_strength: float = 1.0
    bayesian_posterior_samples: int = 1000
    bayesian_mcmc_steps: int = 1000
    
    # Performance and optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    enable_m1_optimization: bool = True
    memory_efficient: bool = True
    
    # Validation and quality control
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    min_samples_for_uncertainty: int = 10
    uncertainty_threshold: float = 0.01
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_uncertainty_metrics: bool = True
    save_uncertainty_history: bool = False
    
    # Advanced features
    enable_uncertainty_calibration: bool = False
    enable_uncertainty_optimization: bool = False
    uncertainty_optimization_config: Optional[Dict[str, Any]] = None


class TreeUncertaintyEstimator:
    """Enhanced uncertainty estimator for tree architectures with comprehensive functionality."""
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize enhanced uncertainty estimator."""
        self.config = config
        self.models = []
        self.predictions = []
        self.uncertainty_history = []
        self.performance_metrics = {}
        
        # Initialize hardware optimizations
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        
        if config.enable_m1_optimization:
            try:
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            except Exception as e:
                tprint_warning(f"M1 optimization setup failed: {e}")
        
        # Initialize math validation
        self.math_validator = MathValidation()
        
        # Initialize serialization for saving uncertainty history
        if config.save_uncertainty_history:
            self.serializer = UniversalSerializer()
        
        tprint_info(f"🌳 TreeUncertaintyEstimator initialized with method: {config.method}")
        tprint_info(f"   → Samples: {config.n_samples}, Confidence: {config.confidence_level}")
        tprint_info(f"   → M1 optimization: {'enabled' if config.enable_m1_optimization else 'disabled'}")
        tprint_info(f"   → Parallel processing: {'enabled' if config.enable_parallel_processing else 'disabled'}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Fit uncertainty estimation models with enhanced validation and optimization."""
        start_time = time.time()
        tprint_info("🔧 Fitting uncertainty estimation models")
        
        # Input validation
        if self.config.enable_input_validation:
            self._validate_inputs(X, y, model_params)
        
        # Memory optimization
        if self.config.enable_m1_optimization and self.m1_memory_optimizer:
            with memory_checkpoint("uncertainty_fitting"):
                self._fit_with_memory_optimization(X, y, model_params)
        else:
            self._fit_standard(X, y, model_params)
        
        # Log performance metrics
        fit_time = time.time() - start_time
        self.performance_metrics['fit_time'] = fit_time
        self.performance_metrics['n_models'] = len(self.models)
        
        tprint_success(f"✅ Uncertainty models fitted in {fit_time:.2f}s")
        tprint_info(f"   → Models created: {len(self.models)}")
        
        if self.config.log_uncertainty_metrics:
            self._log_uncertainty_metrics()
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Validate input data and parameters."""
        try:
            # Validate X
            if self.config.enable_input_validation:
                validate_numeric_array(X, "X")
                if X.shape[0] < self.config.min_samples_for_uncertainty:
                    raise ValueError(f"Insufficient samples: {X.shape[0]} < {self.config.min_samples_for_uncertainty}")
            
            # Validate y
            validate_numeric_array(y, "y")
            if len(X) != len(y):
                raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
            
            # Validate model parameters
            if not isinstance(model_params, dict):
                raise ValueError("model_params must be a dictionary")
            
            tprint_info("✅ Input validation passed")
            
        except Exception as e:
            tprint_error(f"❌ Input validation failed: {e}")
            raise
    
    def _fit_with_memory_optimization(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Fit models with M1 memory optimization."""
        try:
            # Optimize memory before fitting
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer.optimize_memory()
            
            # Fit models
            self._fit_standard(X, y, model_params)
            
            # Cleanup after fitting
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer.cleanup_memory()
                
        except Exception as e:
            tprint_error(f"❌ Memory-optimized fitting failed: {e}")
            raise
    
    def _fit_standard(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Standard fitting without memory optimization."""
        if self.config.method == 'bootstrap':
            self._fit_bootstrap_models(X, y, model_params)
        elif self.config.method == 'monte_carlo':
            self._fit_monte_carlo_models(X, y, model_params)
        elif self.config.method == 'ensemble':
            self._fit_ensemble_models(X, y, model_params)
        elif self.config.method == 'bayesian':
            self._fit_bayesian_models(X, y, model_params)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.config.method}")
    
    def _log_uncertainty_metrics(self):
        """Log uncertainty estimation metrics."""
        if not self.uncertainty_history:
            return
        
        metrics = {
            'mean_uncertainty': safe_mean([h['uncertainty'] for h in self.uncertainty_history]),
            'std_uncertainty': safe_std([h['uncertainty'] for h in self.uncertainty_history]),
            'max_uncertainty': max([h['uncertainty'] for h in self.uncertainty_history]),
            'min_uncertainty': min([h['uncertainty'] for h in self.uncertainty_history])
        }
        
        tprint_info("📊 Uncertainty metrics:")
        for metric, value in metrics.items():
            tprint_info(f"   → {metric}: {value:.4f}")
        
        self.performance_metrics.update(metrics)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced prediction with comprehensive uncertainty estimates."""
        start_time = time.time()
        tprint_info("🔮 Making predictions with uncertainty estimates")
        
        # Input validation
        if self.config.enable_input_validation:
            validate_numeric_array(X, "X")
        
        if not self.models:
            tprint_warning("⚠️ No models available, returning default predictions")
            return np.zeros(X.shape[0]), np.ones(X.shape[0])
        
        # Batch processing for large datasets
        if X.shape[0] > self.config.batch_size:
            return self._predict_in_batches(X)
        
        # Get predictions from all models
        all_predictions = self._get_all_predictions(X)
        
        if len(all_predictions) == 0:
            tprint_warning("⚠️ No valid predictions obtained")
            return np.zeros(X.shape[0]), np.ones(X.shape[0])
        
        # Calculate uncertainty metrics
        mean_predictions, uncertainty = self._calculate_uncertainty_metrics(all_predictions)
        
        # Output validation
        if self.config.enable_output_validation:
            self._validate_predictions(mean_predictions, uncertainty)
        
        # Log performance
        pred_time = time.time() - start_time
        tprint_success(f"✅ Predictions completed in {pred_time:.2f}s")
        
        # Save uncertainty history if enabled
        if self.config.save_uncertainty_history:
            self._save_uncertainty_history(mean_predictions, uncertainty)
        
        return mean_predictions, uncertainty
    
    def _predict_in_batches(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict in batches for large datasets."""
        tprint_info(f"📦 Processing {X.shape[0]} samples in batches of {self.config.batch_size}")
        
        batch_size = self.config.batch_size
        n_batches = (X.shape[0] + batch_size - 1) // batch_size
        
        all_mean_predictions = []
        all_uncertainties = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X.shape[0])
            X_batch = X[start_idx:end_idx]
            
            # Get predictions for this batch
            batch_predictions = self._get_all_predictions(X_batch)
            if len(batch_predictions) > 0:
                mean_pred, uncertainty = self._calculate_uncertainty_metrics(batch_predictions)
                all_mean_predictions.append(mean_pred)
                all_uncertainties.append(uncertainty)
            else:
                all_mean_predictions.append(np.zeros(X_batch.shape[0]))
                all_uncertainties.append(np.ones(X_batch.shape[0]))
        
        return np.concatenate(all_mean_predictions), np.concatenate(all_uncertainties)
    
    def _get_all_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """Get predictions from all models with error handling."""
        all_predictions = []
        
        if self.config.enable_parallel_processing and len(self.models) > 1:
            all_predictions = self._get_predictions_parallel(X)
        else:
            all_predictions = self._get_predictions_sequential(X)
        
        return all_predictions
    
    def _get_predictions_parallel(self, X: np.ndarray) -> List[np.ndarray]:
        """Get predictions using parallel processing."""
        all_predictions = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_model = {
                    executor.submit(self._predict_single_model, model, X): model 
                    for model in self.models
                }
                
                for future in as_completed(future_to_model):
                    try:
                        predictions = future.result()
                        if predictions is not None:
                            all_predictions.append(predictions)
                    except Exception as e:
                        tprint_warning(f"⚠️ Model prediction failed: {e}")
                        
        except Exception as e:
            tprint_error(f"❌ Parallel prediction failed: {e}")
            # Fallback to sequential
            return self._get_predictions_sequential(X)
        
        return all_predictions
    
    def _get_predictions_sequential(self, X: np.ndarray) -> List[np.ndarray]:
        """Get predictions sequentially."""
        all_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                predictions = self._predict_single_model(model, X)
                if predictions is not None:
                    all_predictions.append(predictions)
            except Exception as e:
                tprint_warning(f"⚠️ Model {i} prediction failed: {e}")
        
        return all_predictions
    
    def _predict_single_model(self, model, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict using a single model with error handling."""
        try:
            # This is a placeholder - replace with actual model prediction
            # In real implementation, this would call model.predict(X)
            predictions = np.random.random((X.shape[0],))
            
            # Validate predictions
            if not np.all(np.isfinite(predictions)):
                tprint_warning("⚠️ Model produced non-finite predictions")
                return None
            
            return predictions
            
        except Exception as e:
            tprint_warning(f"⚠️ Single model prediction failed: {e}")
            return None
    
    def _calculate_uncertainty_metrics(self, all_predictions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate comprehensive uncertainty metrics."""
        if not all_predictions:
            return np.array([]), np.array([])
        
        predictions_array = np.array(all_predictions)
        
        # Calculate mean predictions
        if self.config.uncertainty_aggregation == 'mean':
            mean_predictions = np.mean(predictions_array, axis=0)
        elif self.config.uncertainty_aggregation == 'median':
            mean_predictions = np.median(predictions_array, axis=0)
        elif self.config.uncertainty_aggregation == 'max':
            mean_predictions = np.max(predictions_array, axis=0)
        else:  # weighted
            weights = np.ones(len(all_predictions))
            mean_predictions = np.average(predictions_array, axis=0, weights=weights)
        
        # Calculate uncertainty (standard deviation)
        uncertainty = np.std(predictions_array, axis=0)
        
        # Apply uncertainty threshold
        uncertainty = np.maximum(uncertainty, self.config.uncertainty_threshold)
        
        return mean_predictions, uncertainty
    
    def _validate_predictions(self, predictions: np.ndarray, uncertainty: np.ndarray):
        """Validate prediction outputs."""
        try:
            # Check for finite values
            if not np.all(np.isfinite(predictions)):
                raise ValueError("Predictions contain non-finite values")
            
            if not np.all(np.isfinite(uncertainty)):
                raise ValueError("Uncertainty contains non-finite values")
            
            # Check for reasonable uncertainty values
            if np.any(uncertainty < 0):
                raise ValueError("Uncertainty contains negative values")
            
            tprint_info("✅ Output validation passed")
            
        except Exception as e:
            tprint_error(f"❌ Output validation failed: {e}")
            raise
    
    def _save_uncertainty_history(self, predictions: np.ndarray, uncertainty: np.ndarray):
        """Save uncertainty history for analysis."""
        if not self.config.save_uncertainty_history:
            return
        
        history_entry = {
            'timestamp': time.time(),
            'predictions': predictions.tolist(),
            'uncertainty': uncertainty.tolist(),
            'mean_uncertainty': float(np.mean(uncertainty)),
            'max_uncertainty': float(np.max(uncertainty)),
            'min_uncertainty': float(np.min(uncertainty))
        }
        
        self.uncertainty_history.append(history_entry)
        
        # Save to file if serializer is available
        if hasattr(self, 'serializer'):
            try:
                self.serializer.save(
                    self.uncertainty_history, 
                    f"uncertainty_history_{int(time.time())}.json"
                )
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save uncertainty history: {e}")
    
    def get_confidence_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals for predictions."""
        mean_pred, uncertainty = self.predict_with_uncertainty(X)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        z_score = 1.96  # For 95% confidence
        
        lower_bound = mean_pred - z_score * uncertainty
        upper_bound = mean_pred + z_score * uncertainty
        
        return lower_bound, upper_bound
    
    def _fit_bootstrap_models(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Enhanced bootstrap models for uncertainty estimation."""
        tprint_info(f"🔄 Fitting {self.config.n_samples} bootstrap models")
        
        n_samples = X.shape[0]
        successful_models = 0
        
        for i in range(self.config.n_samples):
            try:
                # Enhanced bootstrap sampling
                if self.config.bootstrap_strategy == 'balanced':
                    indices = self._balanced_bootstrap_sample(n_samples)
                elif self.config.bootstrap_strategy == 'stratified':
                    indices = self._stratified_bootstrap_sample(X, y)
                else:  # random
                    indices = np.random.choice(n_samples, size=n_samples, replace=self.config.bootstrap_replace)
                
                X_boot = X[indices]
                y_boot = y[indices]
                
                # Apply bootstrap weights if provided
                if self.config.bootstrap_weights is not None:
                    weights = self.config.bootstrap_weights[indices]
                else:
                    weights = None
                
                # Create and fit model
                model = self._create_model(model_params)
                model.fit(X_boot, y_boot, sample_weight=weights)
                self.models.append(model)
                successful_models += 1
                
                if self.config.enable_detailed_logging and (i + 1) % 10 == 0:
                    tprint_info(f"   → Bootstrap model {i + 1}/{self.config.n_samples} completed")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Bootstrap model {i + 1} failed: {e}")
                continue
        
        tprint_success(f"✅ Bootstrap fitting completed: {successful_models}/{self.config.n_samples} models successful")
    
    def _balanced_bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """Create balanced bootstrap sample."""
        # Ensure each sample appears roughly the same number of times
        base_indices = np.arange(n_samples)
        repeated_indices = np.repeat(base_indices, self.config.n_samples // n_samples + 1)
        np.random.shuffle(repeated_indices)
        return repeated_indices[:n_samples]
    
    def _stratified_bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create stratified bootstrap sample."""
        # Simple stratification based on y values
        unique_y = np.unique(y)
        indices = []
        
        for y_val in unique_y:
            y_mask = (y == y_val)
            y_indices = np.where(y_mask)[0]
            if len(y_indices) > 0:
                # Sample from this stratum
                stratum_size = max(1, len(y_indices) // len(unique_y))
                stratum_sample = np.random.choice(y_indices, size=stratum_size, replace=True)
                indices.extend(stratum_sample)
        
        # Fill remaining samples randomly
        while len(indices) < n_samples:
            indices.append(np.random.choice(n_samples))
        
        return np.array(indices[:n_samples])
    
    def _fit_monte_carlo_models(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Enhanced Monte Carlo models for uncertainty estimation."""
        tprint_info(f"🎲 Fitting {self.config.n_samples} Monte Carlo models")
        
        successful_models = 0
        
        for i in range(self.config.n_samples):
            try:
                # Enhanced Monte Carlo sampling
                noisy_params = self._add_monte_carlo_noise(model_params)
                
                # Add data noise if configured
                if self.config.mc_noise_std > 0:
                    X_noisy = self._add_data_noise(X)
                else:
                    X_noisy = X
                
                # Create and fit model
                model = self._create_model(noisy_params)
                model.fit(X_noisy, y)
                self.models.append(model)
                successful_models += 1
                
                if self.config.enable_detailed_logging and (i + 1) % 10 == 0:
                    tprint_info(f"   → Monte Carlo model {i + 1}/{self.config.n_samples} completed")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Monte Carlo model {i + 1} failed: {e}")
                continue
        
        tprint_success(f"✅ Monte Carlo fitting completed: {successful_models}/{self.config.n_samples} models successful")
    
    def _add_monte_carlo_noise(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Add Monte Carlo noise to model parameters."""
        noisy_params = model_params.copy()
        
        for param, value in noisy_params.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise proportional to parameter magnitude
                noise_std = self.config.mc_noise_std * abs(value)
                noise = np.random.normal(0, noise_std)
                noisy_params[param] = value + noise
                
                # Ensure parameter stays within reasonable bounds
                if isinstance(value, int):
                    noisy_params[param] = max(1, int(noisy_params[param]))
                else:
                    noisy_params[param] = max(0.001, noisy_params[param])
        
        return noisy_params
    
    def _add_data_noise(self, X: np.ndarray) -> np.ndarray:
        """Add noise to input data for Monte Carlo sampling."""
        noise = np.random.normal(0, self.config.mc_noise_std, X.shape)
        return X + noise
    
    def _fit_ensemble_models(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Enhanced ensemble models for uncertainty estimation."""
        tprint_info(f"🎭 Fitting {self.config.ensemble_size} ensemble models")
        
        successful_models = 0
        
        for i in range(self.config.ensemble_size):
            try:
                # Create diverse ensemble members
                if self.config.ensemble_diversity == 'parameter':
                    varied_params = self._vary_parameters(model_params)
                elif self.config.ensemble_diversity == 'data':
                    varied_params = model_params
                    X_varied, y_varied = self._vary_data(X, y)
                else:  # architecture
                    varied_params = self._vary_architecture(model_params)
                    X_varied, y_varied = X, y
                
                # Create and fit model
                model = self._create_model(varied_params)
                if self.config.ensemble_diversity == 'data':
                    model.fit(X_varied, y_varied)
                else:
                    model.fit(X, y)
                
                self.models.append(model)
                successful_models += 1
                
                if self.config.enable_detailed_logging and (i + 1) % 5 == 0:
                    tprint_info(f"   → Ensemble model {i + 1}/{self.config.ensemble_size} completed")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Ensemble model {i + 1} failed: {e}")
                continue
        
        tprint_success(f"✅ Ensemble fitting completed: {successful_models}/{self.config.ensemble_size} models successful")
    
    def _vary_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Vary model parameters for ensemble diversity."""
        varied_params = model_params.copy()
        
        for param, value in varied_params.items():
            if isinstance(value, (int, float)):
                # Add controlled variation
                variation = np.random.uniform(0.8, 1.2)
                varied_params[param] = value * variation
                
                # Ensure reasonable bounds
                if isinstance(value, int):
                    varied_params[param] = max(1, int(varied_params[param]))
                else:
                    varied_params[param] = max(0.001, varied_params[param])
        
        return varied_params
    
    def _vary_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vary data for ensemble diversity."""
        # Add noise or subsample
        if np.random.random() < 0.5:
            # Add noise
            noise = np.random.normal(0, 0.01, X.shape)
            X_varied = X + noise
            y_varied = y
        else:
            # Subsample
            n_samples = X.shape[0]
            sample_size = int(0.8 * n_samples)
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_varied = X[indices]
            y_varied = y[indices]
        
        return X_varied, y_varied
    
    def _vary_architecture(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Vary model architecture for ensemble diversity."""
        varied_params = model_params.copy()
        
        # Add architecture-specific variations
        if 'n_estimators' in varied_params:
            varied_params['n_estimators'] = int(varied_params['n_estimators'] * np.random.uniform(0.7, 1.3))
        
        if 'max_depth' in varied_params:
            varied_params['max_depth'] = int(varied_params['max_depth'] * np.random.uniform(0.8, 1.2))
        
        return varied_params
    
    def _fit_bayesian_models(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Enhanced Bayesian models for uncertainty estimation."""
        tprint_info(f"🧠 Fitting {self.config.bayesian_posterior_samples} Bayesian models")
        
        successful_models = 0
        
        for i in range(self.config.bayesian_posterior_samples):
            try:
                # Sample from posterior distribution
                posterior_params = self._sample_posterior_distribution(model_params)
                
                # Create and fit model
                model = self._create_model(posterior_params)
                model.fit(X, y)
                self.models.append(model)
                successful_models += 1
                
                if self.config.enable_detailed_logging and (i + 1) % 20 == 0:
                    tprint_info(f"   → Bayesian model {i + 1}/{self.config.bayesian_posterior_samples} completed")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Bayesian model {i + 1} failed: {e}")
                continue
        
        tprint_success(f"✅ Bayesian fitting completed: {successful_models}/{self.config.bayesian_posterior_samples} models successful")
    
    def _sample_posterior_distribution(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample from Bayesian posterior distribution."""
        posterior_params = model_params.copy()
        
        for param, value in posterior_params.items():
            if isinstance(value, (int, float)):
                # Use prior strength to control sampling variance
                prior_std = self.config.bayesian_prior_strength * abs(value)
                posterior_value = np.random.normal(value, prior_std)
                
                # Ensure reasonable bounds
                if isinstance(value, int):
                    posterior_params[param] = max(1, int(posterior_value))
                else:
                    posterior_params[param] = max(0.001, posterior_value)
        
        return posterior_params
    
    def _create_model(self, params: Dict[str, Any]):
        """Create a model with given parameters."""
        # Enhanced placeholder model class with better functionality
        class EnhancedPlaceholderModel:
            def __init__(self, params):
                self.params = params
                self.fitted = False
                self.feature_importance_ = None
                self.training_score_ = None
            
            def fit(self, X, y, sample_weight=None):
                self.fitted = True
                # Simulate training score
                self.training_score_ = np.random.random()
                # Simulate feature importance
                self.feature_importance_ = np.random.random(X.shape[1])
                self.feature_importance_ = self.feature_importance_ / np.sum(self.feature_importance_)
            
            def predict(self, X):
                if not self.fitted:
                    raise ValueError("Model must be fitted before prediction")
                # More realistic prediction simulation
                base_prediction = np.random.random((X.shape[0],))
                # Add some correlation with input features
                if X.shape[1] > 0:
                    feature_weights = np.random.random(X.shape[1])
                    feature_weights = feature_weights / np.sum(feature_weights)
                    feature_contribution = np.dot(X, feature_weights)
                    base_prediction = 0.7 * base_prediction + 0.3 * feature_contribution
                return base_prediction
            
            def predict_proba(self, X):
                """Predict class probabilities (for classification)."""
                if not self.fitted:
                    raise ValueError("Model must be fitted before prediction")
                # Simulate probability predictions
                n_classes = 2  # Binary classification
                proba = np.random.random((X.shape[0], n_classes))
                proba = proba / np.sum(proba, axis=1, keepdims=True)
                return proba
        
        return EnhancedPlaceholderModel(params)
    
    def get_uncertainty_metrics(self) -> Dict[str, Any]:
        """Get comprehensive uncertainty metrics."""
        if not self.uncertainty_history:
            return {}
        
        uncertainties = [h['uncertainty'] for h in self.uncertainty_history]
        
        metrics = {
            'mean_uncertainty': safe_mean(uncertainties),
            'std_uncertainty': safe_std(uncertainties),
            'max_uncertainty': max(uncertainties),
            'min_uncertainty': min(uncertainties),
            'uncertainty_range': max(uncertainties) - min(uncertainties),
            'uncertainty_cv': safe_divide(safe_std(uncertainties), safe_mean(uncertainties)),
            'n_predictions': len(uncertainties),
            'performance_metrics': self.performance_metrics
        }
        
        return metrics
    
    def save_uncertainty_estimator(self, filepath: str) -> bool:
        """Save uncertainty estimator to file."""
        try:
            if hasattr(self, 'serializer'):
                estimator_data = {
                    'config': self.config.__dict__,
                    'models': self.models,
                    'uncertainty_history': self.uncertainty_history,
                    'performance_metrics': self.performance_metrics
                }
                return self.serializer.save(estimator_data, filepath)
            else:
                tprint_warning("⚠️ Serializer not available for saving")
                return False
        except Exception as e:
            tprint_error(f"❌ Failed to save uncertainty estimator: {e}")
            return False
    
    def load_uncertainty_estimator(self, filepath: str) -> bool:
        """Load uncertainty estimator from file."""
        try:
            if hasattr(self, 'serializer'):
                estimator_data = self.serializer.load(filepath)
                if estimator_data:
                    self.uncertainty_history = estimator_data.get('uncertainty_history', [])
                    self.performance_metrics = estimator_data.get('performance_metrics', {})
                    tprint_success(f"✅ Uncertainty estimator loaded from {filepath}")
                    return True
            else:
                tprint_warning("⚠️ Serializer not available for loading")
                return False
        except Exception as e:
            tprint_error(f"❌ Failed to load uncertainty estimator: {e}")
            return False
    
    def calculate_prediction_entropy(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction entropy as uncertainty measure."""
        mean_pred, uncertainty = self.predict_with_uncertainty(X)
        
        # Use uncertainty as entropy proxy
        entropy = -uncertainty * np.log(uncertainty + 1e-8)
        return entropy
    
    def get_uncertainty_ranking(self, X: np.ndarray) -> np.ndarray:
        """Get ranking of samples by uncertainty."""
        uncertainty = self.calculate_prediction_entropy(X)
        ranking = np.argsort(uncertainty)[::-1]  # Highest uncertainty first
        return ranking


class TreeEnsembleUncertainty:
    """Ensemble uncertainty estimator for tree architectures."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.ensemble_models = []
        self.uncertainty_estimator = TreeUncertaintyEstimator(config)
    
    def fit(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Fit ensemble uncertainty models."""
        logger.info("Fitting ensemble uncertainty models")
        
        # Create ensemble of models
        n_models = self.config.n_samples
        for i in range(n_models):
            # Create bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train model on bootstrap sample
            model = self._create_model(model_params)
            model.fit(X_bootstrap, y_bootstrap)
            self.ensemble_models.append(model)
        
        # Fit uncertainty estimator
        self.uncertainty_estimator.fit(X, y, model_params)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble uncertainty estimates."""
        logger.info("Making ensemble predictions with uncertainty estimates")
        
        # Get predictions from all ensemble models
        ensemble_predictions = []
        for model in self.ensemble_models:
            pred = model.predict(X)
            ensemble_predictions.append(pred)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate mean and uncertainty
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        uncertainty = np.std(ensemble_predictions, axis=0)
        
        return mean_predictions, uncertainty
    
    def _create_model(self, model_params: Dict[str, Any]):
        """Create a model instance."""
        # This would create the appropriate model based on parameters
        # For now, return a simple placeholder
        class SimpleModel:
            def __init__(self, params):
                self.params = params
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.random.rand(len(X))
        
        return SimpleModel(model_params)


class TreeBayesianUncertainty:
    """Bayesian uncertainty estimator for tree architectures."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.bayesian_models = []
        self.posterior_samples = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]):
        """Fit Bayesian uncertainty models."""
        logger.info("Fitting Bayesian uncertainty models")
        
        # Create Bayesian ensemble
        n_models = self.config.n_samples
        for i in range(n_models):
            # Sample from posterior
            posterior_params = self._sample_posterior(model_params)
            
            # Create model with posterior parameters
            model = self._create_model(posterior_params)
            model.fit(X, y)
            self.bayesian_models.append(model)
            
            # Store posterior samples
            self.posterior_samples.append(posterior_params)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with Bayesian uncertainty estimates."""
        logger.info("Making Bayesian predictions with uncertainty estimates")
        
        # Get predictions from all Bayesian models
        bayesian_predictions = []
        for model in self.bayesian_models:
            pred = model.predict(X)
            bayesian_predictions.append(pred)
        
        bayesian_predictions = np.array(bayesian_predictions)
        
        # Calculate mean and uncertainty
        mean_predictions = np.mean(bayesian_predictions, axis=0)
        uncertainty = np.std(bayesian_predictions, axis=0)
        
        return mean_predictions, uncertainty
    
    def _sample_posterior(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample from posterior distribution."""
        posterior_params = model_params.copy()
        
        # Add noise to parameters (simplified Bayesian sampling)
        for param, value in posterior_params.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise
                noise = np.random.normal(0, 0.1 * abs(value))
                posterior_params[param] = value + noise
        
        return posterior_params
    
    def _create_model(self, model_params: Dict[str, Any]):
        """Create a model instance."""
        # This would create the appropriate model based on parameters
        # For now, return a simple placeholder
        class SimpleModel:
            def __init__(self, params):
                self.params = params
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.random.rand(len(X))
        
        return SimpleModel(model_params)