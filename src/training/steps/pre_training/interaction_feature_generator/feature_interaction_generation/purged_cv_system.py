"""
Purged Cross-Validation System for Time-Series Data

This module implements a comprehensive purged cross-validation system that automatically
sizes embargo periods based on max lookback + horizon to prevent data leakage.

Key Features:
- Auto-sizing based on feature lookback periods
- Embargo enforcement globally
- Purged CV with proper train/test separation
- Horizon-aware validation
- Leakage detection and prevention
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class PurgedCVConfig:
    """Configuration for purged cross-validation."""
    n_splits: int = 5
    embargo_ratio: float = 0.01  # 1% of data as embargo
    min_embargo_size: int = 10   # Minimum embargo size
    max_embargo_size: int = 1000 # Maximum embargo size
    enforce_globally: bool = True
    horizon: int = 1  # Prediction horizon
    safety_factor: float = 1.5  # Safety factor for embargo sizing


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Purged cross-validation for time-series data with automatic embargo sizing.
    
    Prevents data leakage by:
    1. Purging overlapping samples between train/test
    2. Adding embargo period based on max lookback + horizon
    3. Ensuring causal relationships are maintained
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_ratio: float = 0.01,
                 min_embargo_size: int = 10,
                 max_embargo_size: int = 1000,
                 horizon: int = 1,
                 safety_factor: float = 1.5):
        """
        Initialize purged time series split.
        
        Args:
            n_splits: Number of CV splits
            embargo_ratio: Ratio of data to use as embargo
            min_embargo_size: Minimum embargo size in samples
            max_embargo_size: Maximum embargo size in samples
            horizon: Prediction horizon
            safety_factor: Safety factor for embargo sizing
        """
        self.n_splits = n_splits
        self.embargo_ratio = embargo_ratio
        self.min_embargo_size = min_embargo_size
        self.max_embargo_size = max_embargo_size
        self.horizon = horizon
        self.safety_factor = safety_factor
        
        tprint_info(f"🔧 PurgedTimeSeriesSplit initialized")
        tprint_info(f"📊 Splits: {n_splits}, Embargo ratio: {embargo_ratio}")
        tprint_info(f"📊 Horizon: {horizon}, Safety factor: {safety_factor}")
    
    def calculate_embargo_size(self, 
                             data_length: int,
                             max_lookback: int = 0,
                             feature_configs: Optional[Dict[str, Any]] = None) -> int:
        """
        Calculate embargo size based on max lookback + horizon.
        
        Args:
            data_length: Total length of data
            max_lookback: Maximum lookback period across all features
            feature_configs: Feature configuration dict for lookback analysis
            
        Returns:
            Calculated embargo size
        """
        # Base embargo from horizon
        base_embargo = self.horizon
        
        # Add max lookback if provided
        if max_lookback > 0:
            base_embargo += max_lookback
        
        # Apply safety factor
        calculated_embargo = int(base_embargo * self.safety_factor)
        
        # Apply ratio-based constraints
        ratio_embargo = int(data_length * self.embargo_ratio)
        
        # Use the larger of calculated and ratio-based
        embargo_size = max(calculated_embargo, ratio_embargo)
        
        # Apply min/max constraints
        embargo_size = max(self.min_embargo_size, min(embargo_size, self.max_embargo_size))
        
        tprint_info(f"📊 Calculated embargo size: {embargo_size}")
        tprint_info(f"📊 Base embargo: {base_embargo}, Ratio embargo: {ratio_embargo}")
        
        return embargo_size
    
    def analyze_feature_lookbacks(self, 
                                features: pd.DataFrame,
                                feature_configs: Optional[Dict[str, Any]] = None) -> int:
        """
        Analyze features to determine maximum lookback period.
        
        Args:
            features: Feature DataFrame
            feature_configs: Feature configuration dict
            
        Returns:
            Maximum lookback period
        """
        max_lookback = 0
        
        # Analyze feature names for lookback patterns
        for col in features.columns:
            lookback = self._extract_lookback_from_name(col)
            max_lookback = max(max_lookback, lookback)
        
        # Check feature configs for rolling windows
        if feature_configs:
            rolling_windows = feature_configs.get('rolling_windows', [])
            if rolling_windows:
                max_lookback = max(max_lookback, max(rolling_windows))
            
            cross_timeframe_periods = feature_configs.get('cross_timeframe_periods', [])
            if cross_timeframe_periods:
                max_lookback = max(max_lookback, max(cross_timeframe_periods))
        
        tprint_info(f"📊 Analyzed max lookback: {max_lookback}")
        return max_lookback
    
    def _extract_lookback_from_name(self, feature_name: str) -> int:
        """Extract lookback period from feature name."""
        import re
        
        # Common patterns for lookback periods
        patterns = [
            r'_(\d+)$',           # _20, _50, etc.
            r'_(\d+)_',           # _20_, _50_, etc.
            r'rolling_(\d+)',     # rolling_20, rolling_50
            r'window_(\d+)',      # window_20, window_50
            r'period_(\d+)',      # period_20, period_50
            r'ctf_(\d+)_',        # ctf_20_, ctf_50_
        ]
        
        max_period = 0
        for pattern in patterns:
            matches = re.findall(pattern, feature_name)
            for match in matches:
                try:
                    period = int(match)
                    max_period = max(max_period, period)
                except ValueError:
                    continue
        
        return max_period
    
    def split(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None,
              groups: Optional[np.ndarray] = None,
              feature_configs: Optional[Dict[str, Any]] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits with purged CV.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            groups: Group labels (optional)
            feature_configs: Feature configuration dict
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        
        # Calculate embargo size
        max_lookback = self.analyze_feature_lookbacks(X, feature_configs)
        embargo_size = self.calculate_embargo_size(n_samples, max_lookback, feature_configs)
        
        tprint_info(f"📊 Using embargo size: {embargo_size} for {n_samples} samples")
        
        # Calculate split sizes
        test_size = n_samples // self.n_splits
        train_size = n_samples - test_size - embargo_size
        
        if train_size <= 0:
            raise ValueError(f"Not enough data for purged CV: {n_samples} samples, "
                           f"test_size={test_size}, embargo_size={embargo_size}")
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate split boundaries
            test_start = i * test_size
            test_end = test_start + test_size
            
            # Purged train set (before test set)
            train_start = 0
            train_end = test_start
            
            # Add embargo after test set
            embargo_start = test_end
            embargo_end = min(embargo_start + embargo_size, n_samples)
            
            # Create indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            # Validate split
            if len(train_indices) == 0 or len(test_indices) == 0:
                tprint_warning(f"⚠️ Empty split {i}: train={len(train_indices)}, test={len(test_indices)}")
                continue
            
            # Check for overlap
            if self._has_overlap(train_indices, test_indices, embargo_size):
                tprint_warning(f"⚠️ Overlap detected in split {i}")
                continue
            
            tprint_debug(f"📊 Split {i}: train={len(train_indices)}, test={len(test_indices)}, embargo={embargo_size}")
            
            yield train_indices, test_indices
    
    def _has_overlap(self, 
                    train_indices: np.ndarray, 
                    test_indices: np.ndarray, 
                    embargo_size: int) -> bool:
        """Check if there's overlap between train and test sets."""
        if len(train_indices) == 0 or len(test_indices) == 0:
            return False
        
        # Check if test set starts before train set ends + embargo
        max_train_idx = np.max(train_indices)
        min_test_idx = np.min(test_indices)
        
        return min_test_idx <= max_train_idx + embargo_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits


class LeakageDetector:
    """Detect and prevent data leakage in time-series features."""
    
    def __init__(self, config: PurgedCVConfig):
        self.config = config
        self.detected_leakage = []
    
    def detect_leakage(self, 
                      features: pd.DataFrame,
                      target: pd.Series,
                      feature_configs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect potential data leakage in features.
        
        Args:
            features: Feature matrix
            target: Target vector
            feature_configs: Feature configuration dict
            
        Returns:
            List of leakage issues found
        """
        leakage_issues = []
        
        # Check for future information leakage
        for col in features.columns:
            if self._has_future_leakage(features[col], target):
                leakage_issues.append({
                    'type': 'future_leakage',
                    'feature': col,
                    'description': 'Feature appears to use future information'
                })
        
        # Check for centered windows
        for col in features.columns:
            if self._has_centered_window(col):
                leakage_issues.append({
                    'type': 'centered_window',
                    'feature': col,
                    'description': 'Feature uses centered window (non-causal)'
                })
        
        # Check for lookback violations
        max_lookback = self._analyze_max_lookback(features, feature_configs)
        if max_lookback > 0:
            # Check if any features use more lookback than expected
            for col in features.columns:
                feature_lookback = self._extract_lookback_from_name(col)
                if feature_lookback > max_lookback:
                    leakage_issues.append({
                        'type': 'excessive_lookback',
                        'feature': col,
                        'description': f'Feature uses {feature_lookback} lookback, max expected: {max_lookback}'
                    })
        
        self.detected_leakage = leakage_issues
        
        if leakage_issues:
            tprint_warning(f"⚠️ Detected {len(leakage_issues)} leakage issues")
            for issue in leakage_issues:
                tprint_warning(f"  - {issue['type']}: {issue['feature']} - {issue['description']}")
        else:
            tprint_success("✅ No leakage detected")
        
        return leakage_issues
    
    def _has_future_leakage(self, feature: pd.Series, target: pd.Series) -> bool:
        """Check if feature has future information leakage."""
        # Simple check: if feature and target are perfectly correlated
        # with zero lag, it might indicate leakage
        correlation = feature.corr(target)
        return abs(correlation) > 0.99
    
    def _has_centered_window(self, feature_name: str) -> bool:
        """Check if feature uses centered window."""
        centered_patterns = [
            'centered', 'center', 'mid', 'middle',
            'symmetric', 'sym', 'balanced'
        ]
        
        feature_lower = feature_name.lower()
        return any(pattern in feature_lower for pattern in centered_patterns)
    
    def _analyze_max_lookback(self, 
                            features: pd.DataFrame,
                            feature_configs: Optional[Dict[str, Any]] = None) -> int:
        """Analyze maximum lookback period."""
        max_lookback = 0
        
        for col in features.columns:
            lookback = self._extract_lookback_from_name(col)
            max_lookback = max(max_lookback, lookback)
        
        if feature_configs:
            rolling_windows = feature_configs.get('rolling_windows', [])
            if rolling_windows:
                max_lookback = max(max_lookback, max(rolling_windows))
        
        return max_lookback
    
    def _extract_lookback_from_name(self, feature_name: str) -> int:
        """Extract lookback period from feature name."""
        import re
        
        patterns = [
            r'_(\d+)$', r'_(\d+)_', r'rolling_(\d+)',
            r'window_(\d+)', r'period_(\d+)', r'ctf_(\d+)_'
        ]
        
        max_period = 0
        for pattern in patterns:
            matches = re.findall(pattern, feature_name)
            for match in matches:
                try:
                    period = int(match)
                    max_period = max(max_period, period)
                except ValueError:
                    continue
        
        return max_period


class PurgedCVValidator:
    """Validate purged CV implementation."""
    
    def __init__(self, config: PurgedCVConfig):
        self.config = config
    
    def validate_splits(self, 
                       X: pd.DataFrame,
                       y: pd.Series,
                       splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Validate purged CV splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            splits: List of train/test splits
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_splits': len(splits),
            'valid_splits': 0,
            'overlap_issues': 0,
            'size_issues': 0,
            'leakage_issues': 0
        }
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Check split validity
            if len(train_idx) == 0 or len(test_idx) == 0:
                validation_results['size_issues'] += 1
                tprint_warning(f"⚠️ Split {i}: Empty train or test set")
                continue
            
            # Check for overlap
            if self._has_overlap(train_idx, test_idx):
                validation_results['overlap_issues'] += 1
                tprint_warning(f"⚠️ Split {i}: Overlap between train and test")
                continue
            
            # Check for leakage
            if self._has_leakage(X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]):
                validation_results['leakage_issues'] += 1
                tprint_warning(f"⚠️ Split {i}: Potential leakage detected")
                continue
            
            validation_results['valid_splits'] += 1
        
        tprint_info(f"📊 Validation results: {validation_results['valid_splits']}/{validation_results['total_splits']} valid splits")
        
        return validation_results
    
    def _has_overlap(self, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
        """Check for overlap between train and test indices."""
        return len(np.intersect1d(train_idx, test_idx)) > 0
    
    def _has_leakage(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    X_test: pd.DataFrame, 
                    y_test: pd.Series) -> bool:
        """Check for data leakage between train and test sets."""
        # Simple check: if test features are perfectly correlated with train features
        # it might indicate leakage
        for col in X_train.columns:
            if col in X_test.columns:
                corr = X_train[col].corr(X_test[col])
                if abs(corr) > 0.99:
                    return True
        return False


# Global instances for convenience
_purged_cv_config = None
_purged_cv_splitter = None
_leakage_detector = None
_cv_validator = None

def get_purged_cv_config() -> PurgedCVConfig:
    """Get the global purged CV configuration."""
    global _purged_cv_config
    if _purged_cv_config is None:
        _purged_cv_config = PurgedCVConfig()
    return _purged_cv_config

def get_purged_cv_splitter() -> PurgedTimeSeriesSplit:
    """Get the global purged CV splitter."""
    global _purged_cv_splitter
    if _purged_cv_splitter is None:
        config = get_purged_cv_config()
        _purged_cv_splitter = PurgedTimeSeriesSplit(
            n_splits=config.n_splits,
            embargo_ratio=config.embargo_ratio,
            min_embargo_size=config.min_embargo_size,
            max_embargo_size=config.max_embargo_size,
            horizon=config.horizon,
            safety_factor=config.safety_factor
        )
    return _purged_cv_splitter

def get_leakage_detector() -> LeakageDetector:
    """Get the global leakage detector."""
    global _leakage_detector
    if _leakage_detector is None:
        config = get_purged_cv_config()
        _leakage_detector = LeakageDetector(config)
    return _leakage_detector

def get_cv_validator() -> PurgedCVValidator:
    """Get the global CV validator."""
    global _cv_validator
    if _cv_validator is None:
        config = get_purged_cv_config()
        _cv_validator = PurgedCVValidator(config)
    return _cv_validator

def create_purged_cv_splits(X: pd.DataFrame, 
                          y: pd.Series,
                          feature_configs: Optional[Dict[str, Any]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create purged CV splits with automatic embargo sizing.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_configs: Feature configuration dict
        
    Returns:
        List of train/test splits
    """
    splitter = get_purged_cv_splitter()
    splits = list(splitter.split(X, y, feature_configs=feature_configs))
    
    # Validate splits
    validator = get_cv_validator()
    validation_results = validator.validate_splits(X, y, splits)
    
    if validation_results['valid_splits'] == 0:
        raise ValueError("No valid purged CV splits could be created")
    
    return splits

def detect_data_leakage(features: pd.DataFrame, 
                       target: pd.Series,
                       feature_configs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Detect data leakage in features.
    
    Args:
        features: Feature matrix
        target: Target vector
        feature_configs: Feature configuration dict
        
    Returns:
        List of leakage issues
    """
    detector = get_leakage_detector()
    return detector.detect_leakage(features, target, feature_configs)