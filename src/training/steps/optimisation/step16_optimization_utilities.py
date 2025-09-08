"""
Step 16 Optimization Utilities

This module provides optimized utilities for confidence calibration including:
- Fast-fail validation mechanisms
- Memory optimization utilities
- Enhanced matrix operations
- Convergence optimization
- Calibration quality metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import gc
import time
import warnings
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

class FastFailError(Exception):
    """Exception raised when fast-fail conditions are met."""
    pass

class ValidationError(Exception):
    """Exception raised when validation checks fail."""
    pass

class ConvergenceError(Exception):
    """Exception raised when convergence fails."""
    pass

class OptimizationLevel(Enum):
    """Optimization level enumeration."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class CalibrationMetrics:
    """Enhanced calibration metrics container."""
    ece: float
    mce: float
    brier_score: float
    reliability_score: float
    sharpness: float
    resolution: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    total_uncertainty: float
    calibration_auc: float
    entropy_score: float

@dataclass
class ConvergenceConfig:
    """Convergence configuration for optimization."""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    patience: int = 10
    min_improvement: float = 1e-8
    early_stopping: bool = True
    validation_split: float = 0.2

class FastFailValidator:
    """Fast-fail validation utilities for calibration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_samples = config.get('min_samples', 100)
        self.max_missing_ratio = config.get('max_missing_ratio', 0.1)
        self.min_class_balance = config.get('min_class_balance', 0.05)
        
    def validate_data_quality(self, data: pd.DataFrame, regime_id: Optional[int] = None) -> bool:
        """Fast-fail data quality validation."""
        try:
            # Check data size
            if len(data) < self.min_samples:
                raise FastFailError(
                    f"Insufficient data for regime {regime_id}: {len(data)} samples "
                    f"(minimum required: {self.min_samples})"
                )
            
            # Check missing values
            missing_ratio = data.isnull().sum().sum() / data.size
            if missing_ratio > self.max_missing_ratio:
                raise FastFailError(
                    f"Too many missing values in regime {regime_id}: {missing_ratio:.2%} "
                    f"(maximum allowed: {self.max_missing_ratio:.2%})"
                )
            
            # Check probability range validity
            if 'probabilities' in data.columns:
                prob_data = data['probabilities'].dropna()
                if len(prob_data) > 0:
                    if not ((prob_data >= 0) & (prob_data <= 1)).all():
                        raise FastFailError(
                            f"Invalid probability range in regime {regime_id}: "
                            f"min={prob_data.min():.3f}, max={prob_data.max():.3f}"
                        )
            
            # Check label distribution
            if 'labels' in data.columns:
                label_counts = data['labels'].value_counts()
                if len(label_counts) < 2:
                    raise FastFailError(
                        f"Single class labels in regime {regime_id}: {label_counts.index[0]}"
                    )
                
                # Check class balance
                min_class_ratio = label_counts.min() / label_counts.sum()
                if min_class_ratio < self.min_class_balance:
                    raise FastFailError(
                        f"Severe class imbalance in regime {regime_id}: "
                        f"minority class ratio {min_class_ratio:.3f} "
                        f"(minimum required: {self.min_class_balance:.3f})"
                    )
            
            return True
            
        except FastFailError:
            raise
        except Exception as e:
            raise FastFailError(f"Data quality validation failed for regime {regime_id}: {e}")
    
    def validate_convergence(self, optimizer_history: List[float], config: ConvergenceConfig) -> bool:
        """Fast-fail convergence validation."""
        try:
            if len(optimizer_history) < 2:
                return False
            
            # Check if converged
            if len(optimizer_history) >= config.max_iterations:
                raise ConvergenceError(
                    f"Optimization failed to converge after {config.max_iterations} iterations"
                )
            
            # Check for convergence
            recent_losses = optimizer_history[-config.patience:]
            if len(recent_losses) >= config.patience:
                improvement = max(recent_losses) - min(recent_losses)
                if improvement < config.min_improvement:
                    return True  # Converged
            
            # Check for oscillation
            if len(optimizer_history) > 20:
                recent_std = np.std(optimizer_history[-20:])
                if recent_std < config.tolerance * 0.1:
                    raise ConvergenceError("Optimization oscillating, likely non-convergent")
            
            return False
            
        except (ConvergenceError, FastFailError):
            raise
        except Exception as e:
            raise FastFailError(f"Convergence validation failed: {e}")

class ParameterValidator:
    """Enhanced parameter validation utilities."""
    
    @staticmethod
    def validate_calibration_parameters(params: Dict[str, Any], regime_id: Optional[int] = None) -> bool:
        """Validate calibration parameters with comprehensive checks."""
        try:
            # Temperature scaling bounds
            if 'temperature_range' in params:
                temp_range = params['temperature_range']
                if not isinstance(temp_range, (list, tuple)) or len(temp_range) != 2:
                    raise ValidationError(f"Invalid temperature_range format for regime {regime_id}")
                if temp_range[0] <= 0 or temp_range[1] > 100 or temp_range[0] >= temp_range[1]:
                    raise ValidationError(
                        f"Invalid temperature range for regime {regime_id}: {temp_range} "
                        f"(must be 0 < min < max <= 100)"
                    )
            
            # Platt scaling parameters
            if 'max_iterations' in params:
                max_iter = params['max_iterations']
                if not isinstance(max_iter, int) or max_iter < 100 or max_iter > 10000:
                    raise ValidationError(
                        f"Invalid max_iterations for regime {regime_id}: {max_iter} "
                        f"(must be integer between 100 and 10000)"
                    )
            
            # Bin count validation
            if 'calibration_bins' in params:
                bins = params['calibration_bins']
                if not isinstance(bins, int) or bins < 5 or bins > 50:
                    raise ValidationError(
                        f"Invalid bin count for regime {regime_id}: {bins} "
                        f"(must be integer between 5 and 50)"
                    )
            
            # Learning rate validation
            if 'learning_rate' in params:
                lr = params['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    raise ValidationError(
                        f"Invalid learning_rate for regime {regime_id}: {lr} "
                        f"(must be positive number <= 1)"
                    )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Parameter validation failed for regime {regime_id}: {e}")
    
    @staticmethod
    def validate_data_integrity(data: pd.DataFrame, regime_id: Optional[int] = None) -> bool:
        """Enhanced data integrity validation."""
        try:
            # Statistical significance test
            if len(data) < 30:
                raise ValidationError(
                    f"Insufficient data for statistical significance in regime {regime_id}: "
                    f"{len(data)} samples (minimum required: 30)"
                )
            
            # Check for data leakage (simplified check)
            if 'timestamp' in data.columns and 'probabilities' in data.columns:
                # Check if probabilities are perfectly correlated with future timestamps
                correlation = data['timestamp'].corr(data['probabilities'])
                if abs(correlation) > 0.95:
                    raise ValidationError(
                        f"Potential data leakage detected in regime {regime_id}: "
                        f"timestamp-probability correlation {correlation:.3f}"
                    )
            
            # Validate regime consistency
            if regime_id is not None and 'regime_id' in data.columns:
                regime_consistency = (data['regime_id'] == regime_id).all()
                if not regime_consistency:
                    raise ValidationError(
                        f"Regime data inconsistent for regime {regime_id}: "
                        f"contains data from other regimes"
                    )
            
            # Check for extreme outliers
            if 'probabilities' in data.columns:
                prob_data = data['probabilities'].dropna()
                if len(prob_data) > 0:
                    q1, q3 = prob_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = prob_data[(prob_data < q1 - 3*iqr) | (prob_data > q3 + 3*iqr)]
                    if len(outliers) > len(prob_data) * 0.05:  # More than 5% outliers
                        raise ValidationError(
                            f"Too many extreme outliers in regime {regime_id}: "
                            f"{len(outliers)} outliers ({len(outliers)/len(prob_data):.1%})"
                        )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Data integrity validation failed for regime {regime_id}: {e}")

class MemoryOptimizer:
    """Memory optimization utilities for large-scale calibration."""
    
    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        
    def estimate_memory_usage(self, data: pd.DataFrame) -> float:
        """Estimate memory usage in GB."""
        return data.memory_usage(deep=True).sum() / 1024**3
    
    def optimize_data_loading(self, data: pd.DataFrame, regime_id: Optional[int] = None) -> pd.DataFrame:
        """Optimize data loading and memory usage."""
        try:
            # Check memory usage
            memory_usage = self.estimate_memory_usage(data)
            
            if memory_usage > self.memory_limit_gb * 0.8:  # Use 80% of limit
                logger.warning(f"Large dataset detected for regime {regime_id}: {memory_usage:.2f}GB")
                
                # Optimize data types
                data = self._optimize_dtypes(data)
                
                # Use chunked processing if still too large
                if self.estimate_memory_usage(data) > self.memory_limit_gb * 0.6:
                    return self._prepare_chunked_processing(data, regime_id)
            
            return data
            
        except Exception as e:
            logger.warning(f"Memory optimization failed for regime {regime_id}: {e}")
            return data
    
    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        optimized_data = data.copy()
        
        for col in optimized_data.columns:
            if optimized_data[col].dtype == 'float64':
                # Try to downcast to float32
                if optimized_data[col].min() >= np.finfo(np.float32).min and \
                   optimized_data[col].max() <= np.finfo(np.float32).max:
                    optimized_data[col] = optimized_data[col].astype(np.float32)
            
            elif optimized_data[col].dtype == 'int64':
                # Try to downcast to smaller int types
                if optimized_data[col].min() >= np.iinfo(np.int32).min and \
                   optimized_data[col].max() <= np.iinfo(np.int32).max:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
                elif optimized_data[col].min() >= np.iinfo(np.int16).min and \
                     optimized_data[col].max() <= np.iinfo(np.int16).max:
                    optimized_data[col] = optimized_data[col].astype(np.int16)
        
        return optimized_data
    
    def _prepare_chunked_processing(self, data: pd.DataFrame, regime_id: Optional[int] = None) -> pd.DataFrame:
        """Prepare data for chunked processing."""
        logger.info(f"Preparing chunked processing for regime {regime_id}")
        
        # Sort by index for consistent chunking
        data = data.sort_index()
        
        # Add chunk identifier
        chunk_size = max(1000, len(data) // 10)  # At least 10 chunks
        data['chunk_id'] = data.index // chunk_size
        
        return data
    
    def cleanup_memory(self):
        """Clean up memory and force garbage collection."""
        gc.collect()
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

class EnhancedMatrixOperations:
    """Enhanced matrix operations with GPU acceleration support."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.gpu_available = False
        
        # Try to initialize GPU support
        if use_gpu:
            try:
                import torch
                self.torch = torch
                self.gpu_available = torch.cuda.is_available() or hasattr(torch.backends, 'mps')
                if self.gpu_available:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
                else:
                    self.device = torch.device('cpu')
            except ImportError:
                self.gpu_available = False
                self.device = None
    
    def to_tensor(self, data: np.ndarray) -> Any:
        """Convert numpy array to tensor."""
        if self.gpu_available and self.torch:
            return self.torch.from_numpy(data).to(self.device)
        return data
    
    def calculate_ece_vectorized(self, probabilities: np.ndarray, labels: np.ndarray, 
                                n_bins: int = 10) -> float:
        """Calculate ECE using vectorized operations."""
        try:
            if self.gpu_available and self.torch:
                return self._calculate_ece_gpu(probabilities, labels, n_bins)
            else:
                return self._calculate_ece_cpu_optimized(probabilities, labels, n_bins)
        except Exception as e:
            logger.warning(f"Vectorized ECE calculation failed: {e}")
            return self._calculate_ece_fallback(probabilities, labels, n_bins)
    
    def _calculate_ece_gpu(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
        """GPU-accelerated ECE calculation."""
        prob_tensor = self.to_tensor(probabilities)
        labels_tensor = self.to_tensor(labels)
        
        # Create bins
        bins = self.torch.linspace(0, 1, n_bins + 1, device=self.device)
        bin_indices = self.torch.bucketize(prob_tensor, bins) - 1
        bin_indices = self.torch.clamp(bin_indices, 0, n_bins - 1)
        
        # Calculate ECE
        ece = 0.0
        total_samples = len(probabilities)
        
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if not bin_mask.any():
                continue
            
            bin_probabilities = prob_tensor[bin_mask]
            bin_labels = labels_tensor[bin_mask]
            bin_size = bin_mask.sum().item()
            
            avg_pred_prob = bin_probabilities.mean().item()
            avg_accuracy = bin_labels.mean().item()
            
            ece += (bin_size / total_samples) * abs(avg_pred_prob - avg_accuracy)
        
        return float(ece)
    
    def _calculate_ece_cpu_optimized(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
        """CPU-optimized ECE calculation."""
        # Use numpy's digitize for efficient binning
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Vectorized calculation
        ece = 0.0
        total_samples = len(probabilities)
        
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if not np.any(bin_mask):
                continue
            
            bin_probabilities = probabilities[bin_mask]
            bin_labels = labels[bin_mask]
            bin_size = np.sum(bin_mask)
            
            avg_pred_prob = np.mean(bin_probabilities)
            avg_accuracy = np.mean(bin_labels)
            
            ece += (bin_size / total_samples) * abs(avg_pred_prob - avg_accuracy)
        
        return float(ece)
    
    def _calculate_ece_fallback(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
        """Fallback ECE calculation."""
        if len(probabilities) == 0 or len(labels) == 0:
            return 0.0
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        total_samples = len(probabilities)
        
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if not np.any(bin_mask):
                continue
            
            bin_probabilities = probabilities[bin_mask]
            bin_labels = labels[bin_mask]
            bin_size = len(bin_probabilities)
            
            avg_pred_prob = np.mean(bin_probabilities)
            avg_accuracy = np.mean(bin_labels)
            
            ece += (bin_size / total_samples) * abs(avg_pred_prob - avg_accuracy)
        
        return float(ece)

class CalibrationQualityMetrics:
    """Enhanced calibration quality metrics calculator."""
    
    def __init__(self, matrix_ops: Optional[EnhancedMatrixOperations] = None):
        self.matrix_ops = matrix_ops or EnhancedMatrixOperations(use_gpu=False)
    
    def calculate_comprehensive_metrics(self, probabilities: np.ndarray, labels: np.ndarray) -> CalibrationMetrics:
        """Calculate comprehensive calibration quality metrics."""
        try:
            # Basic metrics
            ece = self._calculate_ece(probabilities, labels)
            mce = self._calculate_mce(probabilities, labels)
            brier_score = brier_score_loss(labels, probabilities)
            
            # Enhanced metrics
            reliability_score = self._calculate_reliability_score(probabilities, labels)
            sharpness = self._calculate_sharpness(probabilities)
            resolution = self._calculate_resolution(probabilities, labels)
            
            # Uncertainty metrics
            aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(probabilities)
            epistemic_uncertainty = self._calculate_epistemic_uncertainty(probabilities)
            total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
            
            # Additional metrics
            calibration_auc = self._calculate_calibration_auc(probabilities, labels)
            entropy_score = self._calculate_entropy_score(probabilities)
            
            return CalibrationMetrics(
                ece=ece,
                mce=mce,
                brier_score=brier_score,
                reliability_score=reliability_score,
                sharpness=sharpness,
                resolution=resolution,
                aleatoric_uncertainty=aleatoric_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                total_uncertainty=total_uncertainty,
                calibration_auc=calibration_auc,
                entropy_score=entropy_score
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive metrics: {e}")
            # Return default metrics
            return CalibrationMetrics(
                ece=0.0, mce=0.0, brier_score=0.0, reliability_score=0.0,
                sharpness=0.0, resolution=0.0, aleatoric_uncertainty=0.0,
                epistemic_uncertainty=0.0, total_uncertainty=0.0,
                calibration_auc=0.0, entropy_score=0.0
            )
    
    def _calculate_ece(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        return self.matrix_ops.calculate_ece_vectorized(probabilities, labels, n_bins)
    
    def _calculate_mce(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        if len(probabilities) == 0 or len(labels) == 0:
            return 0.0
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        max_error = 0.0
        
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if not np.any(bin_mask):
                continue
            
            bin_probabilities = probabilities[bin_mask]
            bin_labels = labels[bin_mask]
            
            avg_pred_prob = np.mean(bin_probabilities)
            avg_accuracy = np.mean(bin_labels)
            
            max_error = max(max_error, abs(avg_pred_prob - avg_accuracy))
        
        return float(max_error)
    
    def _calculate_reliability_score(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate reliability score (1 - ECE)."""
        ece = self._calculate_ece(probabilities, labels)
        return max(0.0, 1.0 - ece)
    
    def _calculate_sharpness(self, probabilities: np.ndarray) -> float:
        """Calculate sharpness (variance of predictions)."""
        if len(probabilities) == 0:
            return 0.0
        return float(np.var(probabilities))
    
    def _calculate_resolution(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate resolution (variance of accuracies)."""
        if len(probabilities) == 0 or len(labels) == 0:
            return 0.0
        
        # Bin probabilities and calculate accuracies
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        accuracies = []
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if np.any(bin_mask):
                bin_accuracy = np.mean(labels[bin_mask])
                accuracies.append(bin_accuracy)
        
        return float(np.var(accuracies)) if accuracies else 0.0
    
    def _calculate_aleatoric_uncertainty(self, probabilities: np.ndarray) -> float:
        """Calculate aleatoric uncertainty (data uncertainty)."""
        if len(probabilities) == 0:
            return 0.0
        
        # Use entropy as proxy for aleatoric uncertainty
        eps = 1e-15
        prob_clipped = np.clip(probabilities, eps, 1 - eps)
        entropy = -np.mean(prob_clipped * np.log(prob_clipped) + (1 - prob_clipped) * np.log(1 - prob_clipped))
        return float(entropy)
    
    def _calculate_epistemic_uncertainty(self, probabilities: np.ndarray) -> float:
        """Calculate epistemic uncertainty (model uncertainty)."""
        if len(probabilities) == 0:
            return 0.0
        
        # Use variance as proxy for epistemic uncertainty
        return float(np.var(probabilities))
    
    def _calculate_calibration_auc(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate calibration AUC."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(labels, probabilities))
        except Exception:
            return 0.0
    
    def _calculate_entropy_score(self, probabilities: np.ndarray) -> float:
        """Calculate entropy score."""
        if len(probabilities) == 0:
            return 0.0
        
        eps = 1e-15
        prob_clipped = np.clip(probabilities, eps, 1 - eps)
        entropy = -np.mean(prob_clipped * np.log(prob_clipped) + (1 - prob_clipped) * np.log(1 - prob_clipped))
        return float(entropy)