"""
Data Validation Component

This module provides comprehensive data validation utilities for feature selection,
including quality checks, anomaly detection, and data preprocessing validation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

# Import utilities
try:
    from ...utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile
    )
    from ...utils.common_operations import create_fallback_logger, safe_dataframe_operation
except ImportError as e:
    print(f"⚠️ Some utilities not available: {e}")
    # Create fallback implementations
    def safe_divide(a, b): return a / b if b != 0 else 0
    def safe_log(x): return np.log(np.maximum(x, 1e-10))
    def safe_sqrt(x): return np.sqrt(np.maximum(x, 0))
    def safe_power(x, p): return np.power(np.maximum(x, 0), p)
    def validate_finite(x): return np.isfinite(x).all()
    def safe_correlation(x, y): return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    def safe_covariance(x, y): return np.cov(x, y)[0, 1] if len(x) > 1 else 0
    def safe_mean(x): return np.mean(x) if len(x) > 0 else 0
    def safe_std(x): return np.std(x) if len(x) > 1 else 0
    def safe_percentile(x, p): return np.percentile(x, p) if len(x) > 0 else 0

# Enhanced dependency management
try:
    from ...utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.DataValidation")
    print("✅ Custom logger available for FeatureSelection.DataValidation")
except Exception as e:
    print(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.DataValidation")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr, spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited data validation functionality")


class DataValidator:
    """Comprehensive data validation for feature selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data validator with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('DataValidator')
        
        # Validation thresholds
        self.correlation_threshold = self.config.get('correlation_threshold', 0.99)
        self.mutual_info_threshold = self.config.get('mutual_info_threshold', 0.99)
        self.variance_threshold = self.config.get('variance_threshold', 1e-10)
        self.nan_threshold = self.config.get('nan_threshold', 0.1)  # Max 10% NaN values
        
        _LOGGER.info("🔍 DataValidator initialized with comprehensive validation capabilities")
        _LOGGER.info(f"⚙️ Correlation threshold: {self.correlation_threshold}")
        _LOGGER.info(f"⚙️ Mutual info threshold: {self.mutual_info_threshold}")
        _LOGGER.info(f"⚙️ Variance threshold: {self.variance_threshold}")
        _LOGGER.info(f"⚙️ NaN threshold: {self.nan_threshold}")

    def validate_data_quality(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Validate data quality with comprehensive checks and detailed context."""
        _LOGGER.info("🔍 Starting comprehensive data quality validation...")
        
        try:
            issues = []
            warnings = []
            suspicious_features = []
            
            # Basic data shape validation
            if X is None or X.size == 0:
                issues.append("Input data X is None or empty")
                return self._create_validation_result(False, issues, warnings, suspicious_features)
            
            if X.ndim != 2:
                issues.append(f"Input data X must be 2D, got {X.ndim}D")
                return self._create_validation_result(False, issues, warnings, suspicious_features)
            
            n_samples, n_features = X.shape
            _LOGGER.info(f"📊 Data shape: {n_samples} samples, {n_features} features")
            
            # Check for constant features
            constant_features = self.detect_constant_features(X)
            if constant_features:
                issues.append(f"Constant features detected: {constant_features}")
                suspicious_features.extend(constant_features)
                _LOGGER.warning(f"⚠️ Found {len(constant_features)} constant features")
            
            # Check for high correlation features
            high_corr_features = self.detect_high_correlation_features(X)
            if high_corr_features:
                warnings.append(f"High correlation features detected: {len(high_corr_features)} pairs")
                suspicious_features.extend([pair[0] for pair in high_corr_features])
                suspicious_features.extend([pair[1] for pair in high_corr_features])
                _LOGGER.warning(f"⚠️ Found {len(high_corr_features)} highly correlated feature pairs")
            
            # Check for suspicious correlations with target
            if y is not None:
                suspicious_target_corr = self.detect_suspicious_target_correlations(X, y)
                if suspicious_target_corr:
                    warnings.append(f"Suspicious target correlations: {len(suspicious_target_corr)} features")
                    suspicious_features.extend([feat[0] for feat in suspicious_target_corr])
                    _LOGGER.warning(f"⚠️ Found {len(suspicious_target_corr)} features with suspicious target correlations")
            
            # Check for NaN/Inf values
            nan_features = self.detect_nan_inf_features(X)
            if nan_features:
                issues.append(f"NaN/Inf values in features: {nan_features}")
                suspicious_features.extend(nan_features)
                _LOGGER.warning(f"⚠️ Found {len(nan_features)} features with NaN/Inf values")
            
            # Check for zero variance features
            zero_var_features = self.detect_zero_variance_features(X)
            if zero_var_features:
                issues.append(f"Zero variance features: {zero_var_features}")
                suspicious_features.extend(zero_var_features)
                _LOGGER.warning(f"⚠️ Found {len(zero_var_features)} zero variance features")
            
            # Check for perfect correlations (suspicious)
            perfect_corr = self.detect_perfect_correlations(X)
            if perfect_corr:
                warnings.append(f"Perfect correlations detected: {len(perfect_corr)} pairs")
                suspicious_features.extend([pair[0] for pair in perfect_corr])
                suspicious_features.extend([pair[1] for pair in perfect_corr])
                _LOGGER.warning(f"⚠️ Found {len(perfect_corr)} perfectly correlated feature pairs")
            
            # Check for suspicious mutual information
            if y is not None and SKLEARN_AVAILABLE:
                suspicious_mi = self.detect_suspicious_mutual_information(X, y)
                if suspicious_mi:
                    warnings.append(f"Suspicious mutual information: {len(suspicious_mi)} features")
                    suspicious_features.extend([feat[0] for feat in suspicious_mi])
                    _LOGGER.warning(f"⚠️ Found {len(suspicious_mi)} features with suspicious mutual information")
            
            # Check data distribution
            distribution_issues = self.check_data_distribution(X)
            if distribution_issues:
                warnings.extend(distribution_issues)
                _LOGGER.warning(f"⚠️ Found {len(distribution_issues)} data distribution issues")
            
            # Check for outliers
            outlier_features = self.detect_outlier_features(X)
            if outlier_features:
                warnings.append(f"Features with excessive outliers: {len(outlier_features)}")
                _LOGGER.warning(f"⚠️ Found {len(outlier_features)} features with excessive outliers")
            
            is_valid = len(issues) == 0
            
            # Remove duplicates from suspicious features
            suspicious_features = list(set(suspicious_features))
            
            _LOGGER.info(f"✅ Data validation completed - Valid: {is_valid}, Issues: {len(issues)}, Warnings: {len(warnings)}")
            
            return self._create_validation_result(
                is_valid, issues, warnings, suspicious_features,
                constant_features, high_corr_features, suspicious_target_corr if y is not None else [],
                nan_features, zero_var_features, perfect_corr,
                suspicious_mi if y is not None and SKLEARN_AVAILABLE else [],
                distribution_issues, outlier_features
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Data quality validation failed: {e}")
            return {
                'is_valid': False, 
                'issues': [f"Validation error: {e}"], 
                'warnings': [],
                'suspicious_features': [],
                'error': str(e)
            }

    def _create_validation_result(self, is_valid: bool, issues: List[str], warnings: List[str], 
                                 suspicious_features: List[int], **kwargs) -> Dict[str, Any]:
        """Create standardized validation result."""
        result = {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'suspicious_features': suspicious_features,
            'validation_details': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        return result

    def detect_constant_features(self, X: np.ndarray) -> List[int]:
        """Detect constant features (zero variance)."""
        _LOGGER.debug("🔍 Detecting constant features...")
        try:
            constant_indices = []
            for i in range(X.shape[1]):
                if safe_std(X[:, i]) < self.variance_threshold:
                    constant_indices.append(i)
            _LOGGER.debug(f"📊 Found {len(constant_indices)} constant features")
            return constant_indices
        except Exception as e:
            _LOGGER.warning(f"⚠️ Constant feature detection failed: {e}")
            return []

    def detect_high_correlation_features(self, X: np.ndarray, threshold: float = None) -> List[Tuple[int, int, float]]:
        """Detect features with suspiciously high correlations."""
        if threshold is None:
            threshold = self.correlation_threshold
            
        _LOGGER.debug(f"🔍 Detecting high correlation features (threshold: {threshold})...")
        try:
            high_corr_pairs = []
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    corr = abs(safe_correlation(X[:, i], X[:, j]))
                    if corr > threshold:
                        high_corr_pairs.append((i, j, corr))
            _LOGGER.debug(f"📊 Found {len(high_corr_pairs)} highly correlated feature pairs")
            return high_corr_pairs
        except Exception as e:
            _LOGGER.warning(f"⚠️ High correlation detection failed: {e}")
            return []

    def detect_suspicious_target_correlations(self, X: np.ndarray, y: np.ndarray, 
                                             threshold: float = None) -> List[Tuple[int, float]]:
        """Detect suspiciously high correlations with target."""
        if threshold is None:
            threshold = self.correlation_threshold
            
        _LOGGER.debug(f"🔍 Detecting suspicious target correlations (threshold: {threshold})...")
        try:
            suspicious_features = []
            for i in range(X.shape[1]):
                corr = abs(safe_correlation(X[:, i], y))
                if corr > threshold:
                    suspicious_features.append((i, corr))
            _LOGGER.debug(f"📊 Found {len(suspicious_features)} features with suspicious target correlations")
            return suspicious_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Suspicious correlation detection failed: {e}")
            return []

    def detect_nan_inf_features(self, X: np.ndarray) -> List[int]:
        """Detect features with NaN or Inf values."""
        _LOGGER.debug("🔍 Detecting NaN/Inf features...")
        try:
            problematic_features = []
            for i in range(X.shape[1]):
                feature_data = X[:, i]
                nan_count = np.isnan(feature_data).sum()
                inf_count = np.isinf(feature_data).sum()
                total_count = len(feature_data)
                
                if nan_count > 0 or inf_count > 0:
                    nan_ratio = nan_count / total_count
                    inf_ratio = inf_count / total_count
                    
                    if nan_ratio > self.nan_threshold or inf_ratio > self.nan_threshold:
                        problematic_features.append(i)
            _LOGGER.debug(f"📊 Found {len(problematic_features)} features with excessive NaN/Inf values")
            return problematic_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ NaN/Inf detection failed: {e}")
            return []

    def detect_zero_variance_features(self, X: np.ndarray) -> List[int]:
        """Detect features with zero variance."""
        _LOGGER.debug("🔍 Detecting zero variance features...")
        try:
            zero_var_features = []
            for i in range(X.shape[1]):
                if safe_std(X[:, i]) < self.variance_threshold:
                    zero_var_features.append(i)
            _LOGGER.debug(f"📊 Found {len(zero_var_features)} zero variance features")
            return zero_var_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Zero variance detection failed: {e}")
            return []

    def detect_perfect_correlations(self, X: np.ndarray, threshold: float = 0.999) -> List[Tuple[int, int, float]]:
        """Detect perfectly correlated feature pairs."""
        return self.detect_high_correlation_features(X, threshold)

    def detect_suspicious_mutual_information(self, X: np.ndarray, y: np.ndarray, 
                                           threshold: float = None) -> List[Tuple[int, float]]:
        """Detect suspiciously high mutual information with target."""
        if not SKLEARN_AVAILABLE:
            _LOGGER.warning("⚠️ Scikit-learn not available for mutual information calculation")
            return []
            
        if threshold is None:
            threshold = self.mutual_info_threshold
            
        _LOGGER.debug(f"🔍 Detecting suspicious mutual information (threshold: {threshold})...")
        try:
            suspicious_features = []
            for i in range(X.shape[1]):
                try:
                    mi = mutual_info_regression(X[:, i].reshape(-1, 1), y)[0]
                    if mi > threshold:
                        suspicious_features.append((i, mi))
                except Exception as e:
                    _LOGGER.debug(f"⚠️ MI calculation failed for feature {i}: {e}")
                    continue
            _LOGGER.debug(f"📊 Found {len(suspicious_features)} features with suspicious mutual information")
            return suspicious_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Suspicious MI detection failed: {e}")
            return []

    def check_data_distribution(self, X: np.ndarray) -> List[str]:
        """Check for data distribution issues."""
        _LOGGER.debug("🔍 Checking data distribution...")
        try:
            issues = []
            
            # Check for extreme skewness
            for i in range(X.shape[1]):
                feature_data = X[:, i]
                if len(feature_data) > 10:  # Need sufficient data
                    try:
                        # Simple skewness check
                        mean_val = safe_mean(feature_data)
                        std_val = safe_std(feature_data)
                        if std_val > 0:
                            skewness = safe_mean(((feature_data - mean_val) / std_val) ** 3)
                            if abs(skewness) > 3:  # Highly skewed
                                issues.append(f"Feature {i} is highly skewed (skewness: {skewness:.3f})")
                    except Exception:
                        continue
            
            # Check for extreme kurtosis
            for i in range(X.shape[1]):
                feature_data = X[:, i]
                if len(feature_data) > 10:  # Need sufficient data
                    try:
                        # Simple kurtosis check
                        mean_val = safe_mean(feature_data)
                        std_val = safe_std(feature_data)
                        if std_val > 0:
                            kurtosis = safe_mean(((feature_data - mean_val) / std_val) ** 4) - 3
                            if abs(kurtosis) > 5:  # Extreme kurtosis
                                issues.append(f"Feature {i} has extreme kurtosis (kurtosis: {kurtosis:.3f})")
                    except Exception:
                        continue
            
            _LOGGER.debug(f"📊 Found {len(issues)} data distribution issues")
            return issues
        except Exception as e:
            _LOGGER.warning(f"⚠️ Data distribution check failed: {e}")
            return []

    def detect_outlier_features(self, X: np.ndarray, outlier_threshold: float = 3.0) -> List[int]:
        """Detect features with excessive outliers."""
        _LOGGER.debug(f"🔍 Detecting outlier features (threshold: {outlier_threshold})...")
        try:
            outlier_features = []
            
            for i in range(X.shape[1]):
                feature_data = X[:, i]
                if len(feature_data) > 10:  # Need sufficient data
                    try:
                        mean_val = safe_mean(feature_data)
                        std_val = safe_std(feature_data)
                        
                        if std_val > 0:
                            # Count outliers using z-score
                            z_scores = np.abs((feature_data - mean_val) / std_val)
                            outlier_count = np.sum(z_scores > outlier_threshold)
                            outlier_ratio = outlier_count / len(feature_data)
                            
                            if outlier_ratio > 0.1:  # More than 10% outliers
                                outlier_features.append(i)
                    except Exception:
                        continue
            
            _LOGGER.debug(f"📊 Found {len(outlier_features)} features with excessive outliers")
            return outlier_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Outlier detection failed: {e}")
            return []

    def clean_data(self, X: np.ndarray, y: np.ndarray = None, 
                   remove_constant: bool = True, remove_high_corr: bool = True,
                   remove_nan_inf: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Clean data by removing problematic features."""
        _LOGGER.info("🧹 Starting data cleaning...")
        
        try:
            original_shape = X.shape
            features_to_remove = set()
            cleaning_log = {
                'original_shape': original_shape,
                'removed_features': [],
                'cleaning_steps': []
            }
            
            # Remove constant features
            if remove_constant:
                constant_features = self.detect_constant_features(X)
                if constant_features:
                    features_to_remove.update(constant_features)
                    cleaning_log['cleaning_steps'].append(f"Removed {len(constant_features)} constant features")
                    _LOGGER.info(f"🧹 Removed {len(constant_features)} constant features")
            
            # Remove features with excessive NaN/Inf
            if remove_nan_inf:
                nan_features = self.detect_nan_inf_features(X)
                if nan_features:
                    features_to_remove.update(nan_features)
                    cleaning_log['cleaning_steps'].append(f"Removed {len(nan_features)} features with excessive NaN/Inf")
                    _LOGGER.info(f"🧹 Removed {len(nan_features)} features with excessive NaN/Inf")
            
            # Remove highly correlated features (keep one from each pair)
            if remove_high_corr:
                high_corr_pairs = self.detect_high_correlation_features(X)
                if high_corr_pairs:
                    # Keep the first feature from each pair, remove the second
                    features_to_remove.update([pair[1] for pair in high_corr_pairs])
                    cleaning_log['cleaning_steps'].append(f"Removed {len(high_corr_pairs)} highly correlated features")
                    _LOGGER.info(f"🧹 Removed {len(high_corr_pairs)} highly correlated features")
            
            # Convert to sorted list
            features_to_remove = sorted(list(features_to_remove))
            cleaning_log['removed_features'] = features_to_remove
            
            # Remove features
            if features_to_remove:
                # Create mask for features to keep
                keep_mask = np.ones(X.shape[1], dtype=bool)
                keep_mask[features_to_remove] = False
                
                # Apply mask
                X_cleaned = X[:, keep_mask]
                y_cleaned = y if y is None else y
                
                cleaning_log['final_shape'] = X_cleaned.shape
                cleaning_log['features_removed_count'] = len(features_to_remove)
                
                _LOGGER.info(f"✅ Data cleaning completed - Removed {len(features_to_remove)} features")
                _LOGGER.info(f"📊 Shape: {original_shape} -> {X_cleaned.shape}")
                
                return X_cleaned, y_cleaned, cleaning_log
            else:
                _LOGGER.info("✅ No features needed to be removed")
                cleaning_log['final_shape'] = original_shape
                cleaning_log['features_removed_count'] = 0
                return X, y, cleaning_log
                
        except Exception as e:
            _LOGGER.error(f"❌ Data cleaning failed: {e}")
            return X, y, {'error': str(e)}

    def get_data_summary(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        _LOGGER.info("📊 Generating data summary...")
        
        try:
            summary = {
                'shape': X.shape,
                'dtype': str(X.dtype),
                'memory_usage_mb': X.nbytes / (1024 * 1024),
                'feature_stats': {},
                'target_stats': {}
            }
            
            # Feature statistics
            for i in range(X.shape[1]):
                feature_data = X[:, i]
                summary['feature_stats'][f'feature_{i}'] = {
                    'mean': safe_mean(feature_data),
                    'std': safe_std(feature_data),
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'nan_count': np.isnan(feature_data).sum(),
                    'inf_count': np.isinf(feature_data).sum()
                }
            
            # Target statistics
            if y is not None:
                summary['target_stats'] = {
                    'mean': safe_mean(y),
                    'std': safe_std(y),
                    'min': np.min(y),
                    'max': np.max(y),
                    'nan_count': np.isnan(y).sum(),
                    'inf_count': np.isinf(y).sum(),
                    'unique_values': len(np.unique(y))
                }
            
            _LOGGER.info("✅ Data summary generated successfully")
            return summary
            
        except Exception as e:
            _LOGGER.error(f"❌ Data summary generation failed: {e}")
            return {'error': str(e)}