"""
Feature Drift Monitoring and Nested Validation System

This module implements:
1. Feature drift monitoring (KL divergence, distribution shifts)
2. Nested cross-validation for feature selection
3. Feature correlation clustering
4. VIF (Variance Inflation Factor) analysis for multicollinearity
5. Feature stability tracking over time

Addresses Section 3: Feature Engineering & Selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import json

from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.purged_kfold import PurgedKFoldTime
from src.utils.ml_common.validation.universal_temporal_validation import UniversalTimeSeriesSplit
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


logger = logging.getLogger(__name__)


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    
    # KL divergence
    max_kl_divergence: float = 0.5  # Maximum KL divergence
    
    # Distribution shift (in standard deviations)
    max_mean_shift: float = 2.0  # Maximum mean shift
    max_std_ratio: float = 2.0  # Maximum std ratio change
    
    # Correlation thresholds
    max_correlation: float = 0.9  # Maximum correlation for redundancy
    
    # VIF threshold
    max_vif: float = 10.0  # Maximum VIF for multicollinearity
    
    # Information coefficient
    min_ic: float = 0.01  # Minimum IC to retain feature
    ic_stability_threshold: float = 0.5  # Minimum IC stability (Sharpe ratio)


@dataclass
class FeatureDriftReport:
    """Report of feature drift analysis."""
    
    feature_name: str
    drift_detected: bool
    drift_score: float
    
    # Distribution statistics
    train_mean: float
    val_mean: float
    train_std: float
    val_std: float
    
    # Drift metrics
    kl_divergence: Optional[float] = None
    js_distance: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    
    # Metadata
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'drift_detected': self.drift_detected,
            'drift_score': self.drift_score,
            'train_mean': self.train_mean,
            'val_mean': self.val_mean,
            'train_std': self.train_std,
            'val_std': self.val_std,
            'kl_divergence': self.kl_divergence,
            'js_distance': self.js_distance,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'timestamp': self.timestamp
        }


@dataclass
class NestedCVResult:
    """Results from nested cross-validation."""
    
    feature_name: str
    inner_cv_scores: List[float]
    outer_cv_scores: List[float]
    
    mean_inner_score: float
    std_inner_score: float
    mean_outer_score: float
    std_outer_score: float
    
    selected_in_folds: int  # Number of folds where feature was selected
    total_folds: int
    selection_frequency: float
    
    stable: bool  # Whether feature appears in >60% of folds


class FeatureDriftMonitor:
    """
    Feature drift monitor with nested validation support.
    
    Key Features:
    1. Distribution drift detection (KL divergence, KS test)
    2. Nested cross-validation for feature selection
    3. Feature correlation clustering
    4. VIF analysis for multicollinearity detection
    5. Temporal stability tracking
    """
    
    def __init__(self, thresholds: Optional[DriftThresholds] = None):
        """
        Initialize feature drift monitor.
        
        Args:
            thresholds: Drift detection thresholds
        """
        self.thresholds = thresholds or DriftThresholds()
        self.drift_history: List[FeatureDriftReport] = []
        
        tprint_success("✅ FeatureDriftMonitor initialized")
        tprint_info(f"   → KL divergence threshold: {self.thresholds.max_kl_divergence}")
        tprint_info(f"   → Mean shift threshold: {self.thresholds.max_mean_shift}σ")
        tprint_info(f"   → Max VIF: {self.thresholds.max_vif}")
    
    def detect_feature_drift(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, FeatureDriftReport]:
        """
        Detect feature drift between training and validation sets.
        
        Args:
            train_features: Training features
            val_features: Validation features
            feature_names: Optional subset of features to check
        
        Returns:
            Dictionary of drift reports per feature
        """
        tprint_info("🔍 Detecting feature drift...")
        
        if feature_names is None:
            feature_names = train_features.columns.tolist()
        
        drift_reports = {}
        n_drifted = 0
        
        for feature in feature_names:
            if feature not in train_features.columns or feature not in val_features.columns:
                continue
            
            train_data = train_features[feature].dropna()
            val_data = val_features[feature].dropna()
            
            if len(train_data) < 10 or len(val_data) < 10:
                continue
            
            # Calculate distribution statistics
            train_mean = train_data.mean()
            train_std = train_data.std()
            val_mean = val_data.mean()
            val_std = val_data.std()
            
            # KS test
            ks_stat, ks_pval = stats.ks_2samp(train_data, val_data)
            
            # KL divergence (using histograms)
            kl_div = self._calculate_kl_divergence(train_data, val_data)
            
            # Jensen-Shannon distance
            js_dist = self._calculate_js_distance(train_data, val_data)
            
            # Mean shift (in standard deviations)
            mean_shift = abs(val_mean - train_mean) / (train_std + 1e-8)
            
            # Std ratio
            std_ratio = val_std / (train_std + 1e-8)
            
            # Determine if drift detected
            drift_detected = (
                kl_div > self.thresholds.max_kl_divergence or
                mean_shift > self.thresholds.max_mean_shift or
                std_ratio > self.thresholds.max_std_ratio or
                std_ratio < (1.0 / self.thresholds.max_std_ratio)
            )
            
            if drift_detected:
                n_drifted += 1
            
            # Create drift report
            report = FeatureDriftReport(
                feature_name=feature,
                drift_detected=drift_detected,
                drift_score=float(kl_div),
                train_mean=float(train_mean),
                val_mean=float(val_mean),
                train_std=float(train_std),
                val_std=float(val_std),
                kl_divergence=float(kl_div),
                js_distance=float(js_dist),
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pval)
            )
            
            drift_reports[feature] = report
            self.drift_history.append(report)
        
        drift_ratio = n_drifted / len(drift_reports) if drift_reports else 0.0
        
        if drift_ratio > 0.2:
            tprint_warning(f"⚠️ Significant drift detected: {n_drifted}/{len(drift_reports)} features ({drift_ratio:.1%})")
        else:
            tprint_success(f"✅ Drift check passed: {n_drifted}/{len(drift_reports)} features drifted")
        
        return drift_reports
    
    def _calculate_kl_divergence(
        self,
        data1: pd.Series,
        data2: pd.Series,
        n_bins: int = 50
    ) -> float:
        """Calculate KL divergence between two distributions."""
        try:
            # Create histograms
            min_val = min(data1.min(), data2.min())
            max_val = max(data1.max(), data2.max())
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            hist1, _ = np.histogram(data1, bins=bins, density=True)
            hist2, _ = np.histogram(data2, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            
            # Normalize
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Calculate KL divergence
            kl_div = np.sum(hist1 * np.log(hist1 / hist2))
            
            return kl_div
            
        except Exception as e:
            logger.warning(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def _calculate_js_distance(
        self,
        data1: pd.Series,
        data2: pd.Series,
        n_bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon distance between two distributions."""
        try:
            # Create histograms
            min_val = min(data1.min(), data2.min())
            max_val = max(data1.max(), data2.max())
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            hist1, _ = np.histogram(data1, bins=bins, density=True)
            hist2, _ = np.histogram(data2, bins=bins, density=True)
            
            # Normalize
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Calculate JS distance
            js_dist = jensenshannon(hist1, hist2)
            
            return js_dist
            
        except Exception as e:
            logger.warning(f"Error calculating JS distance: {e}")
            return 0.0
    
    def perform_nested_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator,
        inner_cv: int = 3,
        outer_cv: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, NestedCVResult]:
        """
        Perform nested cross-validation for feature selection.
        
        Outer loop: Evaluation
        Inner loop: Feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            estimator: Estimator for scoring
            inner_cv: Number of inner CV folds
            outer_cv: Number of outer CV folds
            feature_names: Optional subset of features to evaluate
        
        Returns:
            Dictionary of nested CV results per feature
        """
        tprint_info(f"🔄 Performing nested CV ({outer_cv} outer, {inner_cv} inner folds)...")
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Create time series splitters
        outer_splitter = UniversalTimeSeriesSplit(
            n_splits=outer_cv,
            test_size=0.2,
            gap_size=1
        )
        
        inner_splitter = UniversalTimeSeriesSplit(
            n_splits=inner_cv,
            test_size=0.2,
            gap_size=1
        )
        
        nested_results = {}
        
        for feature in feature_names:
            if feature not in X.columns:
                continue
            
            try:
                # Outer CV loop
                outer_scores = []
                feature_selected_count = 0
                inner_scores_all = []
                
                for outer_train_idx, outer_test_idx in outer_splitter.split(X):
                    X_outer_train = X.iloc[outer_train_idx]
                    y_outer_train = y.iloc[outer_train_idx]
                    X_outer_test = X.iloc[outer_test_idx]
                    y_outer_test = y.iloc[outer_test_idx]
                    
                    # Inner CV loop for feature selection
                    inner_scores = cross_val_score(
                        estimator,
                        X_outer_train[[feature]],
                        y_outer_train,
                        cv=inner_splitter,
                        scoring='r2'
                    )
                    
                    inner_scores_all.extend(inner_scores.tolist())
                    
                    # Decide if feature should be selected based on inner CV
                    if inner_scores.mean() > 0:  # Positive R²
                        feature_selected_count += 1
                        
                        # Evaluate on outer test set
                        estimator.fit(X_outer_train[[feature]], y_outer_train)
                        outer_score = estimator.score(X_outer_test[[feature]], y_outer_test)
                        outer_scores.append(outer_score)
                
                # Calculate selection frequency
                selection_frequency = feature_selected_count / outer_cv
                stable = selection_frequency >= 0.6  # 60% threshold
                
                result = NestedCVResult(
                    feature_name=feature,
                    inner_cv_scores=inner_scores_all,
                    outer_cv_scores=outer_scores if outer_scores else [0.0],
                    mean_inner_score=float(np.mean(inner_scores_all)) if inner_scores_all else 0.0,
                    std_inner_score=float(np.std(inner_scores_all)) if inner_scores_all else 0.0,
                    mean_outer_score=float(np.mean(outer_scores)) if outer_scores else 0.0,
                    std_outer_score=float(np.std(outer_scores)) if outer_scores else 0.0,
                    selected_in_folds=feature_selected_count,
                    total_folds=outer_cv,
                    selection_frequency=selection_frequency,
                    stable=stable
                )
                
                nested_results[feature] = result
                
            except Exception as e:
                logger.warning(f"Error in nested CV for feature {feature}: {e}")
                continue
        
        # Log summary
        stable_features = sum(1 for r in nested_results.values() if r.stable)
        tprint_success(f"✅ Nested CV completed: {stable_features}/{len(nested_results)} stable features")
        
        return nested_results
    
    def cluster_correlated_features(
        self,
        features: pd.DataFrame,
        correlation_threshold: float = None
    ) -> Dict[int, List[str]]:
        """
        Cluster highly correlated features using hierarchical clustering.
        
        Args:
            features: Feature DataFrame
            correlation_threshold: Correlation threshold for clustering
        
        Returns:
            Dictionary mapping cluster ID to feature names
        """
        if correlation_threshold is None:
            correlation_threshold = self.thresholds.max_correlation
        
        tprint_info(f"🔗 Clustering correlated features (threshold={correlation_threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - correlation_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        # Fit clustering
        labels = clustering.fit_predict(distance_matrix)
        
        # Group features by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(features.columns[i])
        
        # Log results
        tprint_success(f"✅ Found {len(clusters)} feature clusters")
        for cluster_id, feature_list in clusters.items():
            if len(feature_list) > 1:
                tprint_info(f"   → Cluster {cluster_id}: {len(feature_list)} features")
        
        return clusters
    
    def calculate_vif(
        self,
        features: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
        
        Args:
            features: Feature DataFrame
            feature_names: Optional subset of features to check
        
        Returns:
            Dictionary mapping feature names to VIF values
        """
        tprint_info("📊 Calculating VIF for multicollinearity detection...")
        
        if feature_names is None:
            feature_names = features.columns.tolist()
        
        # Select only numeric features
        numeric_features = features[feature_names].select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            tprint_warning("⚠️ No numeric features found for VIF calculation")
            return {}
        
        # Drop columns with NaN or infinite values
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        
        if numeric_features.shape[1] < 2:
            tprint_warning("⚠️ Not enough features for VIF calculation")
            return {}
        
        vif_data = {}
        high_vif_count = 0
        
        for i, feature in enumerate(numeric_features.columns):
            try:
                vif = variance_inflation_factor(numeric_features.values, i)
                vif_data[feature] = float(vif)
                
                if vif > self.thresholds.max_vif:
                    high_vif_count += 1
                    
            except Exception as e:
                logger.warning(f"Error calculating VIF for {feature}: {e}")
                vif_data[feature] = np.nan
        
        if high_vif_count > 0:
            tprint_warning(f"⚠️ {high_vif_count} features with VIF > {self.thresholds.max_vif}")
        else:
            tprint_success(f"✅ No multicollinearity detected (all VIF < {self.thresholds.max_vif})")
        
        return vif_data
    
    def track_feature_stability_over_time(
        self,
        features: pd.DataFrame,
        window_size: int = 100,
        step_size: int = 20
    ) -> Dict[str, List[float]]:
        """
        Track feature stability over time using rolling windows.
        
        Args:
            features: Feature DataFrame with datetime index
            window_size: Size of rolling window
            step_size: Step size between windows
        
        Returns:
            Dictionary mapping feature names to stability scores over time
        """
        tprint_info(f"📈 Tracking feature stability over time (window={window_size}, step={step_size})...")
        
        stability_data = {}
        
        for feature in features.columns:
            stability_scores = []
            
            for start_idx in range(0, len(features) - window_size, step_size):
                window_data = features[feature].iloc[start_idx:start_idx+window_size]
                
                # Calculate coefficient of variation as stability metric
                mean = window_data.mean()
                std = window_data.std()
                
                if mean != 0:
                    cv = std / abs(mean)
                else:
                    cv = 0.0
                
                stability_scores.append(cv)
            
            stability_data[feature] = stability_scores
        
        tprint_success(f"✅ Stability tracking completed for {len(stability_data)} features")
        
        return stability_data
    
    def export_drift_report(
        self,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Export drift report to JSON file.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'thresholds': {
                'max_kl_divergence': self.thresholds.max_kl_divergence,
                'max_mean_shift': self.thresholds.max_mean_shift,
                'max_std_ratio': self.thresholds.max_std_ratio,
                'max_correlation': self.thresholds.max_correlation,
                'max_vif': self.thresholds.max_vif
            },
            'drift_history': [report.to_dict() for report in self.drift_history],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        tprint_success(f"✅ Drift report exported to {output_path}")
        return output_path


def create_feature_drift_monitor(
    thresholds: Optional[DriftThresholds] = None
) -> FeatureDriftMonitor:
    """
    Factory function to create FeatureDriftMonitor.
    
    Args:
        thresholds: Optional drift thresholds
    
    Returns:
        FeatureDriftMonitor instance
    """
    return FeatureDriftMonitor(thresholds)