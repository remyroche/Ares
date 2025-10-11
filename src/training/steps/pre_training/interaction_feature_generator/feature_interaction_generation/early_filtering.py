"""
Early Filtering for Interactive Feature Generation

This module implements time-series safe early filtering that:
- Runs on 10% downsample with purged folds
- Applies variance threshold (per-fold) to drop near-constant features
- Uses autocorr length (ACL) to deprioritize features with ACL >> horizon
- Performs quick IC/MI vs label to keep top-k per transform family
- Filters redundancy within family (corr > 0.97 duplicates)

Key Features:
- Time-series safe filtering (respects purged folds)
- Fast filtering on downsampled data
- Prevents family collapse by keeping top-k per family
- Removes redundant features early
- Maintains statistical validity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
import time
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class EarlyFilteringConfig:
    """Configuration for early filtering."""
    # Downsampling
    downsample_ratio: float = 0.1  # Use 10% of data for filtering
    min_samples_per_fold: int = 100
    
    # Variance filtering
    variance_threshold: float = 1e-6  # Drop features with variance below this
    per_fold_variance_check: bool = True
    
    # Autocorrelation filtering
    max_autocorr_length_ratio: float = 0.5  # Max ACL / horizon ratio
    autocorr_lag_max: int = 50
    
    # IC/MI filtering
    min_ic_threshold: float = 0.01  # Minimum IC to keep
    min_mi_threshold: float = 0.01  # Minimum MI to keep
    top_k_per_family: int = 5  # Keep top-k features per transform family
    
    # Redundancy filtering
    max_correlation_threshold: float = 0.97  # Drop highly correlated features
    correlation_window: int = 1000  # Window for correlation calculation
    
    # Purged folds
    purging_period: int = 5  # Bars to purge around each split
    embargo_period: int = 2  # Bars to embargo after each split


@dataclass
class FilteringResult:
    """Result of early filtering."""
    selected_features: List[str]
    rejected_features: Dict[str, str]  # feature -> reason
    filtering_stats: Dict[str, Any]
    family_breakdown: Dict[str, List[str]]
    performance_metrics: Dict[str, float]


class EarlyFilteringSystem:
    """
    Early filtering system for interactive feature generation.
    
    Performs time-series safe filtering to remove low-quality features
    before expensive computation steps.
    """
    
    def __init__(self, config: Optional[EarlyFilteringConfig] = None):
        """Initialize the early filtering system."""
        self.config = config or EarlyFilteringConfig()
        self.scaler = StandardScaler()
        
        tprint_info(f"🚀 Early filtering system initialized")
        tprint_info(f"📊 Downsample ratio: {self.config.downsample_ratio}")
        tprint_info(f"📊 Variance threshold: {self.config.variance_threshold}")
        tprint_info(f"📊 Top-k per family: {self.config.top_k_per_family}")
    
    def create_purged_folds(self, data: pd.DataFrame, n_folds: int = 5) -> List[Tuple[int, int]]:
        """Create purged time-series folds for cross-validation."""
        n_samples = len(data)
        fold_size = n_samples // n_folds
        
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = min((i + 1) * fold_size, n_samples)
            
            # Apply purging and embargo
            purged_start = start + self.config.purging_period
            purged_end = end - self.config.embargo_period
            
            if purged_end > purged_start:
                folds.append((purged_start, purged_end))
        
        tprint_debug(f"📊 Created {len(folds)} purged folds")
        return folds
    
    def downsample_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Downsample data for fast filtering while maintaining time-series structure."""
        n_samples = len(data)
        sample_size = max(int(n_samples * self.config.downsample_ratio), self.config.min_samples_per_fold)
        
        # Use systematic sampling to maintain time-series structure
        step = n_samples // sample_size
        indices = np.arange(0, n_samples, step)[:sample_size]
        
        sampled_data = data.iloc[indices].copy()
        sampled_target = data[target_column].iloc[indices].copy()
        
        tprint_debug(f"📊 Downsampled from {n_samples} to {len(sampled_data)} samples")
        return sampled_data, sampled_target
    
    def filter_by_variance(self, data: pd.DataFrame, target: pd.Series, 
                          purged_folds: List[Tuple[int, int]]) -> Dict[str, str]:
        """Filter features by variance threshold across purged folds."""
        rejected_features = {}
        
        tprint_debug("🔍 Filtering by variance...")
        
        for col in data.columns:
            if col == target.name:
                continue
            
            feature_data = data[col].dropna()
            if len(feature_data) < 10:
                rejected_features[col] = "insufficient_data"
                continue
            
            # Check variance across folds
            fold_variances = []
            for start, end in purged_folds:
                if end <= len(feature_data):
                    fold_data = feature_data.iloc[start:end]
                    if len(fold_data) > 0:
                        fold_var = fold_data.var()
                        if not np.isnan(fold_var):
                            fold_variances.append(fold_var)
            
            if not fold_variances:
                rejected_features[col] = "no_valid_folds"
                continue
            
            # Check if any fold has sufficient variance
            max_variance = max(fold_variances)
            if max_variance < self.config.variance_threshold:
                rejected_features[col] = f"low_variance_{max_variance:.2e}"
        
        tprint_info(f"📊 Variance filtering: {len(rejected_features)} features rejected")
        return rejected_features
    
    def filter_by_autocorrelation(self, data: pd.DataFrame, target: pd.Series,
                                 horizon: int) -> Dict[str, str]:
        """Filter features by autocorrelation length relative to horizon."""
        rejected_features = {}
        
        tprint_debug("🔍 Filtering by autocorrelation...")
        
        max_acl = int(horizon * self.config.max_autocorr_length_ratio)
        
        for col in data.columns:
            if col == target.name:
                continue
            
            feature_data = data[col].dropna()
            if len(feature_data) < max_acl + 10:
                continue
            
            # Calculate autocorrelation function
            try:
                autocorr = np.correlate(feature_data, feature_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Find autocorrelation length (where ACF drops below 0.1)
                acl = np.where(autocorr < 0.1)[0]
                if len(acl) > 0:
                    acl = acl[0]
                else:
                    acl = len(autocorr)
                
                if acl > max_acl:
                    rejected_features[col] = f"high_autocorr_{acl}"
            
            except Exception as e:
                tprint_debug(f"⚠️ Autocorr calculation failed for {col}: {e}")
                continue
        
        tprint_info(f"📊 Autocorr filtering: {len(rejected_features)} features rejected")
        return rejected_features
    
    def calculate_ic_scores(self, data: pd.DataFrame, target: pd.Series,
                           purged_folds: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate Information Coefficient scores across purged folds."""
        ic_scores = {}
        
        tprint_debug("🔍 Calculating IC scores...")
        
        for col in data.columns:
            if col == target.name:
                continue
            
            feature_data = data[col].dropna()
            if len(feature_data) < 10:
                continue
            
            fold_ics = []
            for start, end in purged_folds:
                if end <= len(feature_data):
                    fold_data = feature_data.iloc[start:end]
                    fold_target = target.iloc[start:end]
                    
                    # Align data
                    valid_idx = ~(fold_data.isna() | fold_target.isna())
                    if valid_idx.sum() > 5:
                        fold_data_clean = fold_data[valid_idx]
                        fold_target_clean = fold_target[valid_idx]
                        
                        # Calculate IC (correlation)
                        try:
                            ic = np.corrcoef(fold_data_clean, fold_target_clean)[0, 1]
                            if not np.isnan(ic):
                                fold_ics.append(abs(ic))
                        except:
                            continue
            
            if fold_ics:
                ic_scores[col] = np.mean(fold_ics)
        
        tprint_debug(f"📊 Calculated IC scores for {len(ic_scores)} features")
        return ic_scores
    
    def calculate_mi_scores(self, data: pd.DataFrame, target: pd.Series,
                           purged_folds: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate Mutual Information scores across purged folds."""
        mi_scores = {}
        
        tprint_debug("🔍 Calculating MI scores...")
        
        for col in data.columns:
            if col == target.name:
                continue
            
            feature_data = data[col].dropna()
            if len(feature_data) < 10:
                continue
            
            fold_mis = []
            for start, end in purged_folds:
                if end <= len(feature_data):
                    fold_data = feature_data.iloc[start:end]
                    fold_target = target.iloc[start:end]
                    
                    # Align data
                    valid_idx = ~(fold_data.isna() | fold_target.isna())
                    if valid_idx.sum() > 5:
                        fold_data_clean = fold_data[valid_idx].values.reshape(-1, 1)
                        fold_target_clean = fold_target[valid_idx].values
                        
                        # Calculate MI
                        try:
                            mi = mutual_info_regression(fold_data_clean, fold_target_clean, random_state=42)[0]
                            if not np.isnan(mi):
                                fold_mis.append(mi)
                        except:
                            continue
            
            if fold_mis:
                mi_scores[col] = np.mean(fold_mis)
        
        tprint_debug(f"📊 Calculated MI scores for {len(mi_scores)} features")
        return mi_scores
    
    def group_features_by_family(self, features: List[str]) -> Dict[str, List[str]]:
        """Group features by transform family based on naming patterns."""
        families = defaultdict(list)
        
        for feature in features:
            # Extract family from feature name
            if '/' in feature:
                family = feature.split('/')[1]  # e.g., "momentum/sma_20" -> "sma"
            elif '_' in feature:
                parts = feature.split('_')
                if len(parts) > 1:
                    family = parts[0]  # e.g., "rsi_14" -> "rsi"
                else:
                    family = "other"
            else:
                family = "other"
            
            families[family].append(feature)
        
        tprint_debug(f"📊 Grouped features into {len(families)} families")
        return dict(families)
    
    def select_top_k_per_family(self, ic_scores: Dict[str, float], mi_scores: Dict[str, float],
                               families: Dict[str, List[str]]) -> List[str]:
        """Select top-k features per family based on IC and MI scores."""
        selected_features = []
        
        tprint_debug("🔍 Selecting top-k per family...")
        
        for family, family_features in families.items():
            # Filter features that exist in scores
            valid_features = [f for f in family_features if f in ic_scores or f in mi_scores]
            
            if not valid_features:
                continue
            
            # Calculate combined scores (weighted average of IC and MI)
            combined_scores = {}
            for feature in valid_features:
                ic_score = ic_scores.get(feature, 0.0)
                mi_score = mi_scores.get(feature, 0.0)
                
                # Normalize scores to [0, 1] range
                ic_norm = min(ic_score / 0.1, 1.0) if ic_score > 0 else 0.0
                mi_norm = min(mi_score / 0.1, 1.0) if mi_score > 0 else 0.0
                
                # Combined score (equal weight)
                combined_scores[feature] = 0.5 * ic_norm + 0.5 * mi_norm
            
            # Select top-k features
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_k = sorted_features[:self.config.top_k_per_family]
            
            selected_features.extend([f for f, _ in top_k])
            tprint_debug(f"📊 Family {family}: selected {len(top_k)} features")
        
        tprint_info(f"📊 Selected {len(selected_features)} features across families")
        return selected_features
    
    def filter_redundant_features(self, data: pd.DataFrame, features: List[str],
                                 target: pd.Series) -> List[str]:
        """Filter redundant features based on correlation threshold."""
        if len(features) <= 1:
            return features
        
        tprint_debug("🔍 Filtering redundant features...")
        
        # Fast-fail: Skip correlation check for large feature sets to avoid memory issues
        if len(features) > 500:
            tprint_warning("⚠️ Skipping redundancy filtering for large feature set to avoid memory issues")
            return features
        
        # Calculate correlation matrix
        feature_data = data[features].dropna()
        if len(feature_data) < 10:
            return features
        
        try:
            corr_matrix = feature_data.corr().abs()
            
            # Find highly correlated pairs
            redundant_pairs = []
            for i, feature1 in enumerate(features):
                for j, feature2 in enumerate(features[i+1:], i+1):
                    if corr_matrix.iloc[i, j] > self.config.max_correlation_threshold:
                        redundant_pairs.append((feature1, feature2))
            
            # Remove redundant features (keep the one with higher IC/MI)
            features_to_remove = set()
            for feature1, feature2 in redundant_pairs:
                # Simple heuristic: keep the feature with more data
                if len(data[feature1].dropna()) >= len(data[feature2].dropna()):
                    features_to_remove.add(feature2)
                else:
                    features_to_remove.add(feature1)
            
            filtered_features = [f for f in features if f not in features_to_remove]
            
            tprint_info(f"📊 Redundancy filtering: removed {len(features_to_remove)} redundant features")
            return filtered_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Redundancy filtering failed: {e}")
            return features
    
    def filter_features(self, data: pd.DataFrame, target_column: str, 
                       horizon: int = 20) -> FilteringResult:
        """
        Perform comprehensive early filtering of features.
        
        Args:
            data: Input DataFrame with features
            target_column: Name of target column
            horizon: Prediction horizon for autocorrelation filtering
            
        Returns:
            FilteringResult with selected features and statistics
        """
        tprint_success("🚀 Starting early feature filtering")
        start_time = time.time()
        
        # Create purged folds
        purged_folds = self.create_purged_folds(data)
        
        # Downsample data for fast filtering
        sampled_data, sampled_target = self.downsample_data(data, target_column)
        
        # Initialize rejection tracking
        rejected_features = {}
        filtering_stats = {
            'total_features': len(data.columns),
            'sampled_features': len(sampled_data.columns),
            'variance_rejected': 0,
            'autocorr_rejected': 0,
            'ic_rejected': 0,
            'mi_rejected': 0,
            'redundancy_rejected': 0
        }
        
        # Step 1: Variance filtering
        variance_rejected = self.filter_by_variance(sampled_data, sampled_target, purged_folds)
        rejected_features.update(variance_rejected)
        filtering_stats['variance_rejected'] = len(variance_rejected)
        
        # Step 2: Autocorrelation filtering
        autocorr_rejected = self.filter_by_autocorrelation(sampled_data, sampled_target, horizon)
        rejected_features.update(autocorr_rejected)
        filtering_stats['autocorr_rejected'] = len(autocorr_rejected)
        
        # Get remaining features
        remaining_features = [col for col in sampled_data.columns 
                            if col not in rejected_features and col != target_column]
        
        if not remaining_features:
            tprint_warning("⚠️ No features remaining after initial filtering")
            return FilteringResult(
                selected_features=[],
                rejected_features=rejected_features,
                filtering_stats=filtering_stats,
                family_breakdown={},
                performance_metrics={}
            )
        
        # Step 3: IC/MI scoring
        ic_scores = self.calculate_ic_scores(sampled_data[remaining_features], sampled_target, purged_folds)
        mi_scores = self.calculate_mi_scores(sampled_data[remaining_features], sampled_target, purged_folds)
        
        # Filter by IC/MI thresholds
        ic_rejected = {f: f"low_ic_{ic_scores.get(f, 0):.3f}" 
                      for f in remaining_features 
                      if ic_scores.get(f, 0) < self.config.min_ic_threshold}
        mi_rejected = {f: f"low_mi_{mi_scores.get(f, 0):.3f}" 
                      for f in remaining_features 
                      if mi_scores.get(f, 0) < self.config.min_mi_threshold}
        
        rejected_features.update(ic_rejected)
        rejected_features.update(mi_rejected)
        filtering_stats['ic_rejected'] = len(ic_rejected)
        filtering_stats['mi_rejected'] = len(mi_rejected)
        
        # Get features that passed IC/MI filtering
        passed_features = [f for f in remaining_features 
                          if f not in ic_rejected and f not in mi_rejected]
        
        if not passed_features:
            tprint_warning("⚠️ No features remaining after IC/MI filtering")
            return FilteringResult(
                selected_features=[],
                rejected_features=rejected_features,
                filtering_stats=filtering_stats,
                family_breakdown={},
                performance_metrics={}
            )
        
        # Step 4: Group by family and select top-k per family
        families = self.group_features_by_family(passed_features)
        selected_features = self.select_top_k_per_family(ic_scores, mi_scores, families)
        
        # Step 5: Filter redundant features
        final_features = self.filter_redundant_features(sampled_data, selected_features, sampled_target)
        redundancy_rejected = len(selected_features) - len(final_features)
        filtering_stats['redundancy_rejected'] = redundancy_rejected
        
        # Calculate performance metrics
        execution_time = time.time() - start_time
        performance_metrics = {
            'execution_time': execution_time,
            'filtering_efficiency': len(final_features) / len(data.columns),
            'rejection_rate': len(rejected_features) / len(data.columns),
            'family_diversity': len(families)
        }
        
        # Create result
        result = FilteringResult(
            selected_features=final_features,
            rejected_features=rejected_features,
            filtering_stats=filtering_stats,
            family_breakdown=families,
            performance_metrics=performance_metrics
        )
        
        tprint_success(f"✅ Early filtering completed in {execution_time:.3f}s")
        tprint_info(f"📊 Selected {len(final_features)} features from {len(data.columns)}")
        tprint_info(f"📊 Rejection rate: {performance_metrics['rejection_rate']:.1%}")
        tprint_info(f"📊 Family diversity: {performance_metrics['family_diversity']} families")
        
        return result


# Convenience functions

def create_early_filtering_system(config: Optional[EarlyFilteringConfig] = None) -> EarlyFilteringSystem:
    """Create an early filtering system with the given configuration."""
    return EarlyFilteringSystem(config)


def filter_features_early(data: pd.DataFrame, target_column: str, 
                         horizon: int = 20, config: Optional[EarlyFilteringConfig] = None) -> FilteringResult:
    """Convenience function for early feature filtering."""
    system = create_early_filtering_system(config)
    return system.filter_features(data, target_column, horizon)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    
    data = pd.DataFrame({
        'target': np.random.randn(n_samples).cumsum(),
        'feature1': np.random.randn(n_samples),  # Good feature
        'feature2': np.random.randn(n_samples) * 0.1,  # Low variance
        'feature3': np.random.randn(n_samples),  # Good feature
        'feature4': np.ones(n_samples),  # Constant feature
        'feature5': np.random.randn(n_samples),  # Good feature
        'feature6': np.random.randn(n_samples) * 0.01,  # Very low variance
    })
    
    # Add some autocorrelated features
    data['feature7'] = data['feature1'].rolling(window=10).mean()  # High autocorr
    data['feature8'] = data['feature1'].shift(1)  # High autocorr
    
    # Test early filtering
    config = EarlyFilteringConfig(
        downsample_ratio=0.2,
        variance_threshold=1e-4,
        top_k_per_family=3
    )
    
    result = filter_features_early(data, 'target', horizon=20, config=config)
    
    print(f"Selected features: {result.selected_features}")
    print(f"Rejected features: {len(result.rejected_features)}")
    print(f"Filtering stats: {result.filtering_stats}")
    print(f"Performance metrics: {result.performance_metrics}")