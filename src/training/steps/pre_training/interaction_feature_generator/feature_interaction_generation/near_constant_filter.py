"""
Near-Constant Filter Using IQR and Entropy

This module implements a sophisticated near-constant filter that uses IQR and
entropy measures instead of just variance to identify and remove features with
very low information content.

Key Features:
- IQR-based filtering for continuous features
- Entropy-based filtering for discrete features
- Adaptive thresholds per feature family
- Fold-aware filtering
- Information content preservation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class NearConstantFilterConfig:
    """Configuration for near-constant filter."""
    iqr_threshold: float = 0.01  # IQR threshold for continuous features
    entropy_threshold: float = 0.1  # Entropy threshold for discrete features
    variance_threshold: float = 1e-8  # Fallback variance threshold
    min_unique_values: int = 2  # Minimum unique values required
    max_constant_ratio: float = 0.95  # Maximum ratio of constant values
    adaptive_thresholds: bool = True  # Use adaptive thresholds per family
    fold_aware: bool = True  # Apply filtering per fold
    preserve_info: bool = True  # Preserve high-information features


class NearConstantFilter:
    """Filter near-constant features using IQR and entropy measures."""
    
    def __init__(self, config: NearConstantFilterConfig):
        self.config = config
        self.filter_stats = {}
        self.filtered_features = []
        
        tprint_info("🔍 Near-constant filter initialized")
        tprint_info(f"📊 IQR threshold: {config.iqr_threshold}")
        tprint_info(f"📊 Entropy threshold: {config.entropy_threshold}")
        tprint_info(f"📊 Adaptive thresholds: {config.adaptive_thresholds}")
    
    def filter_features(self, 
                       features: pd.DataFrame,
                       target: Optional[pd.Series] = None,
                       feature_families: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Filter near-constant features from the dataset.
        
        Args:
            features: Feature matrix
            target: Target vector (optional, for entropy calculation)
            feature_families: Feature families for adaptive thresholds
            
        Returns:
            Filtered feature matrix
        """
        tprint_info(f"🔍 Filtering {len(features.columns)} features for near-constant values")
        
        if features.empty:
            return features
        
        # Initialize filter statistics
        self.filter_stats = {
            'total_features': len(features.columns),
            'filtered_features': 0,
            'filter_reasons': {},
            'family_stats': {}
        }
        
        # Get feature families if not provided
        if feature_families is None:
            feature_families = self._group_features_by_family(features.columns)
        
        # Filter features
        filtered_features = []
        filtered_out = []
        
        for family, family_features in feature_families.items():
            family_data = features[family_features]
            
            # Calculate adaptive thresholds for this family
            if self.config.adaptive_thresholds:
                iqr_thresh, entropy_thresh = self._calculate_adaptive_thresholds(
                    family_data, target, family
                )
            else:
                iqr_thresh = self.config.iqr_threshold
                entropy_thresh = self.config.entropy_threshold
            
            # Filter features in this family
            family_filtered, family_reasons = self._filter_family_features(
                family_data, target, iqr_thresh, entropy_thresh, family
            )
            
            # Update statistics
            self.filter_stats['family_stats'][family] = {
                'total_features': len(family_features),
                'filtered_features': len(family_features) - len(family_filtered.columns),
                'iqr_threshold': iqr_thresh,
                'entropy_threshold': entropy_thresh
            }
            
            # Collect results
            filtered_features.append(family_filtered)
            filtered_out.extend(family_reasons)
        
        # Combine filtered features
        if filtered_features:
            result = pd.concat(filtered_features, axis=1)
        else:
            result = pd.DataFrame(index=features.index)
        
        # Update statistics
        self.filter_stats['filtered_features'] = len(features.columns) - len(result.columns)
        self.filter_stats['filter_reasons'] = filtered_out
        self.filtered_features = [reason['feature'] for reason in filtered_out]
        
        tprint_info(f"✅ Filtered {self.filter_stats['filtered_features']} near-constant features")
        tprint_info(f"📊 Remaining features: {len(result.columns)}")
        
        return result
    
    def _group_features_by_family(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features by family based on naming patterns."""
        families = {}
        
        for feature in feature_names:
            family = self._extract_feature_family(feature)
            if family not in families:
                families[family] = []
            families[family].append(feature)
        
        return families
    
    def _extract_feature_family(self, feature_name: str) -> str:
        """Extract feature family from feature name."""
        # Common feature family patterns
        family_patterns = [
            (r'^rsi', 'rsi'),
            (r'^macd', 'macd'),
            (r'^bb_', 'bollinger'),
            (r'^atr', 'atr'),
            (r'^sma_', 'sma'),
            (r'^ema_', 'ema'),
            (r'^rolling_', 'rolling'),
            (r'^ctf_', 'cross_timeframe'),
            (r'^volume', 'volume'),
            (r'^volatility', 'volatility'),
            (r'^momentum', 'momentum'),
            (r'^trend', 'trend'),
            (r'^mean_reversion', 'mean_reversion'),
        ]
        
        for pattern, family in family_patterns:
            if re.match(pattern, feature_name):
                return family
        
        return 'other'
    
    def _calculate_adaptive_thresholds(self, 
                                     family_data: pd.DataFrame,
                                     target: Optional[pd.Series],
                                     family: str) -> Tuple[float, float]:
        """Calculate adaptive thresholds for a feature family."""
        if family_data.empty:
            return self.config.iqr_threshold, self.config.entropy_threshold
        
        # Calculate IQR for continuous features
        iqr_values = []
        for col in family_data.columns:
            if family_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                iqr = family_data[col].quantile(0.75) - family_data[col].quantile(0.25)
                if not np.isnan(iqr) and iqr > 0:
                    iqr_values.append(iqr)
        
        # Calculate entropy for discrete features
        entropy_values = []
        for col in family_data.columns:
            if family_data[col].dtype in ['object', 'category']:
                entropy = self._calculate_entropy(family_data[col])
                if not np.isnan(entropy):
                    entropy_values.append(entropy)
        
        # Set adaptive thresholds
        if iqr_values:
            # Use 10th percentile of IQR values as threshold
            iqr_threshold = np.percentile(iqr_values, 10)
            iqr_threshold = max(iqr_threshold, self.config.iqr_threshold * 0.1)
        else:
            iqr_threshold = self.config.iqr_threshold
        
        if entropy_values:
            # Use 10th percentile of entropy values as threshold
            entropy_threshold = np.percentile(entropy_values, 10)
            entropy_threshold = max(entropy_threshold, self.config.entropy_threshold * 0.1)
        else:
            entropy_threshold = self.config.entropy_threshold
        
        return iqr_threshold, entropy_threshold
    
    def _filter_family_features(self, 
                               family_data: pd.DataFrame,
                               target: Optional[pd.Series],
                               iqr_threshold: float,
                               entropy_threshold: float,
                               family: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Filter features within a family."""
        filtered_features = []
        filter_reasons = []
        
        for col in family_data.columns:
            feature_data = family_data[col].dropna()
            
            if len(feature_data) == 0:
                filter_reasons.append({
                    'feature': col,
                    'reason': 'all_nan',
                    'family': family
                })
                continue
            
            # Check minimum unique values
            if feature_data.nunique() < self.config.min_unique_values:
                filter_reasons.append({
                    'feature': col,
                    'reason': 'insufficient_unique_values',
                    'family': family,
                    'unique_count': feature_data.nunique()
                })
                continue
            
            # Check constant ratio
            constant_ratio = (feature_data == feature_data.mode().iloc[0]).mean()
            if constant_ratio > self.config.max_constant_ratio:
                filter_reasons.append({
                    'feature': col,
                    'reason': 'high_constant_ratio',
                    'family': family,
                    'constant_ratio': constant_ratio
                })
                continue
            
            # Check IQR for continuous features
            if feature_data.dtype in ['float64', 'float32', 'int64', 'int32']:
                iqr = feature_data.quantile(0.75) - feature_data.quantile(0.25)
                if iqr < iqr_threshold:
                    filter_reasons.append({
                        'feature': col,
                        'reason': 'low_iqr',
                        'family': family,
                        'iqr': iqr,
                        'threshold': iqr_threshold
                    })
                    continue
            
            # Check entropy for discrete features
            elif feature_data.dtype in ['object', 'category']:
                entropy = self._calculate_entropy(feature_data)
                if entropy < entropy_threshold:
                    filter_reasons.append({
                        'feature': col,
                        'reason': 'low_entropy',
                        'family': family,
                        'entropy': entropy,
                        'threshold': entropy_threshold
                    })
                    continue
            
            # Check variance as fallback
            if hasattr(feature_data, 'var'):
                variance = feature_data.var()
                if variance < self.config.variance_threshold:
                    filter_reasons.append({
                        'feature': col,
                        'reason': 'low_variance',
                        'family': family,
                        'variance': variance,
                        'threshold': self.config.variance_threshold
                    })
                    continue
            
            # Feature passed all checks
            filtered_features.append(col)
        
        # Return filtered data
        if filtered_features:
            result = family_data[filtered_features]
        else:
            result = pd.DataFrame(index=family_data.index)
        
        return result, filter_reasons
    
    def _calculate_entropy(self, data: pd.Series) -> float:
        """Calculate entropy of a discrete feature."""
        try:
            # Count value frequencies
            value_counts = data.value_counts()
            
            # Calculate probabilities
            probabilities = value_counts / len(data)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return entropy
            
        except Exception as e:
            tprint_debug(f"⚠️ Entropy calculation failed: {e}")
            return 0.0
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get detailed filter statistics."""
        return self.filter_stats
    
    def get_filtered_features(self) -> List[str]:
        """Get list of filtered features."""
        return self.filtered_features


class FoldAwareNearConstantFilter:
    """Near-constant filter that works with cross-validation folds."""
    
    def __init__(self, config: NearConstantFilterConfig):
        self.config = config
        self.fold_filters = {}
        self.global_filter = NearConstantFilter(config)
    
    def fit_fold(self, 
                 features: pd.DataFrame,
                 target: pd.Series,
                 fold_id: int) -> NearConstantFilter:
        """Fit filter on a specific fold."""
        filter_instance = NearConstantFilter(self.config)
        filter_instance.filter_features(features, target)
        self.fold_filters[fold_id] = filter_instance
        return filter_instance
    
    def transform_fold(self, 
                      features: pd.DataFrame,
                      fold_id: int) -> pd.DataFrame:
        """Transform features using fold-specific filter."""
        if fold_id not in self.fold_filters:
            tprint_warning(f"⚠️ No filter found for fold {fold_id}, using global filter")
            return self.global_filter.filter_features(features)
        
        filter_instance = self.fold_filters[fold_id]
        return filter_instance.filter_features(features)
    
    def fit_global(self, 
                  features: pd.DataFrame,
                  target: pd.Series) -> NearConstantFilter:
        """Fit global filter on all data."""
        self.global_filter.filter_features(features, target)
        return self.global_filter
    
    def get_fold_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all folds."""
        stats = {}
        for fold_id, filter_instance in self.fold_filters.items():
            stats[fold_id] = filter_instance.get_filter_statistics()
        return stats


# Global instances
_near_constant_filter = None
_fold_aware_filter = None

def get_near_constant_filter() -> NearConstantFilter:
    """Get the global near-constant filter."""
    global _near_constant_filter
    if _near_constant_filter is None:
        config = NearConstantFilterConfig()
        _near_constant_filter = NearConstantFilter(config)
    return _near_constant_filter

def get_fold_aware_filter() -> FoldAwareNearConstantFilter:
    """Get the global fold-aware filter."""
    global _fold_aware_filter
    if _fold_aware_filter is None:
        config = NearConstantFilterConfig()
        _fold_aware_filter = FoldAwareNearConstantFilter(config)
    return _fold_aware_filter

def filter_near_constant_features(features: pd.DataFrame,
                                 target: Optional[pd.Series] = None,
                                 feature_families: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Filter near-constant features using IQR and entropy.
    
    Args:
        features: Feature matrix
        target: Target vector (optional)
        feature_families: Feature families for adaptive thresholds
        
    Returns:
        Filtered feature matrix
    """
    filter_instance = get_near_constant_filter()
    return filter_instance.filter_features(features, target, feature_families)

def get_filter_statistics() -> Dict[str, Any]:
    """Get near-constant filter statistics."""
    filter_instance = get_near_constant_filter()
    return filter_instance.get_filter_statistics()

def get_filtered_features() -> List[str]:
    """Get list of filtered features."""
    filter_instance = get_near_constant_filter()
    return filter_instance.get_filtered_features()