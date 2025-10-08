"""
Feature Redundancy Control and Drift Monitoring.

This module implements:
1. Correlation-based feature clustering and redundancy removal
2. VIF (Variance Inflation Factor) analysis for multicollinearity
3. Hierarchical feature clustering for representative selection
4. Feature drift monitoring using KL divergence
5. Feature orthogonalization and de-biasing

References:
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Feature Engineering for Machine Learning (Zheng & Casari, 2018)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_debug,
)
from src.utils.matrix_operations import (
    safe_matrix_multiply,
    safe_correlation_matrix,
    correlation_matrix_gpu,
)


@dataclass
class RedundancyConfig:
    """Configuration for redundancy control."""
    
    # Correlation clustering
    correlation_threshold: float = 0.85  # Features with |corr| > threshold are redundant
    clustering_method: str = "average"  # 'single', 'complete', 'average', 'ward'
    
    # VIF analysis
    enable_vif: bool = True
    vif_threshold: float = 10.0  # Features with VIF > threshold are multicollinear
    max_vif_iterations: int = 5  # Max iterations for iterative VIF removal
    
    # Feature selection strategy
    selection_strategy: str = "best_per_cluster"  # 'best_per_cluster', 'representative', 'all_below_threshold'
    selection_metric: str = "importance"  # 'importance', 'ic', 'mi'
    
    # Orthogonalization
    enable_orthogonalization: bool = False
    orthog_method: str = "gram_schmidt"  # 'gram_schmidt', 'pca', 'ica'


@dataclass
class DriftConfig:
    """Configuration for drift monitoring."""
    
    # KL divergence settings
    kl_threshold: float = 0.15  # Maximum acceptable KL divergence
    kl_bins: int = 20  # Number of bins for histogram comparison
    
    # Drift detection
    detect_distribution_shift: bool = True
    detect_correlation_shift: bool = True
    detect_importance_shift: bool = False
    
    # Reference period
    reference_period: str = "train"  # 'train', 'first_half', 'custom'
    min_samples: int = 100  # Minimum samples for drift detection


@dataclass
class FeatureCluster:
    """Represents a cluster of correlated features."""
    
    cluster_id: int
    feature_names: List[str]
    representative_feature: str
    mean_correlation: float
    max_correlation: float
    cluster_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedundancyReport:
    """Report from redundancy analysis."""
    
    original_features: List[str]
    retained_features: List[str]
    removed_features: List[str]
    feature_clusters: List[FeatureCluster]
    vif_scores: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def reduction_rate(self) -> float:
        """Feature reduction rate."""
        if len(self.original_features) == 0:
            return 0.0
        return len(self.removed_features) / len(self.original_features)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            'original_count': len(self.original_features),
            'retained_count': len(self.retained_features),
            'removed_count': len(self.removed_features),
            'reduction_rate': self.reduction_rate,
            'n_clusters': len(self.feature_clusters),
            'mean_cluster_size': np.mean([c.cluster_size for c in self.feature_clusters]) if self.feature_clusters else 0
        }


@dataclass
class DriftReport:
    """Report from drift analysis."""
    
    feature_drifts: Dict[str, float]  # Feature -> KL divergence
    drifted_features: List[str]  # Features exceeding threshold
    correlation_drift: Optional[float] = None
    importance_drift: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            'n_features': len(self.feature_drifts),
            'n_drifted': len(self.drifted_features),
            'drift_rate': len(self.drifted_features) / len(self.feature_drifts) if self.feature_drifts else 0,
            'mean_drift': np.mean(list(self.feature_drifts.values())) if self.feature_drifts else 0,
            'max_drift': max(self.feature_drifts.values()) if self.feature_drifts else 0,
            'correlation_drift': self.correlation_drift,
            'importance_drift': self.importance_drift
        }


class RedundancyController:
    """
    Controls feature redundancy through clustering and VIF analysis.
    """
    
    def __init__(
        self,
        config: Optional[RedundancyConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the redundancy controller.
        
        Args:
            config: Redundancy configuration
            logger: Optional logger instance
        """
        self.config = config or RedundancyConfig()
        self.logger = logger or system_logger.getChild('RedundancyController')

        tprint_info("🔧 Initializing RedundancyController...")
        tprint_debug(f"🎯 Correlation threshold: {self.config.correlation_threshold}")
        tprint_debug(f"📊 VIF enabled: {'Yes' if self.config.enable_vif else 'No'}")
        tprint_debug(f"🔗 Clustering method: {self.config.clustering_method}")
        tprint_debug(f"📈 Selection strategy: {self.config.selection_strategy}")
        tprint_success("✅ RedundancyController initialized")
    
    def analyze_and_reduce(
        self,
        features: pd.DataFrame,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> RedundancyReport:
        """
        Analyze feature redundancy and select non-redundant subset.
        
        Args:
            features: DataFrame with feature columns
            feature_importance: Optional importance scores for selection
        
        Returns:
            RedundancyReport with analysis results
        """
        tprint_info("🔍 Starting redundancy analysis and feature reduction")
        tprint_debug(f"📊 Input features shape: {features.shape}")
        tprint_debug(f"🎯 Feature importance provided: {'Yes' if feature_importance else 'No'}")

        numeric_features = features.select_dtypes(include=[np.number])

        if numeric_features.empty:
            tprint_warning("⚠️ No numeric features found for redundancy analysis")
            self.logger.warning("No numeric features found")
            return RedundancyReport(
                original_features=[],
                retained_features=[],
                removed_features=[],
                feature_clusters=[],
                vif_scores={}
            )
        
        # Step 1: Correlation-based clustering
        tprint_info(f"📊 Step 1: Correlation-based clustering ({self.config.clustering_method} method)...")
        clusters = self._cluster_correlated_features(numeric_features)
        tprint_success(f"✅ Identified {len(clusters)} feature clusters")
        
        # Step 2: Select representative from each cluster
        tprint_info(f"🎯 Step 2: Selecting representatives using '{self.config.selection_strategy}' strategy...")
        retained_features = self._select_representatives(
            features=numeric_features,
            clusters=clusters,
            importance=feature_importance
        )
        tprint_success(f"✅ Selected {len(retained_features)} representative features from {len(clusters)} clusters")
        
        # Step 3: VIF analysis on retained features
        if self.config.enable_vif:
            tprint_info(f"🔢 Step 3: VIF analysis (threshold={self.config.vif_threshold})...")
            pre_vif_count = len(retained_features)
            retained_features = self._remove_high_vif(
                features=numeric_features[retained_features]
            )
            tprint_success(f"✅ VIF analysis complete: {pre_vif_count} → {len(retained_features)} features")
        
        # Compute VIF scores for final set
        vif_scores = {}
        if self.config.enable_vif and len(retained_features) > 1:
            tprint_debug(f"🔢 Computing final VIF scores for {len(retained_features)} features")
            vif_scores = self._compute_vif(numeric_features[retained_features])
            max_vif = max(vif_scores.values()) if vif_scores else 0
            tprint_debug(f"📊 VIF range: [{min(vif_scores.values()):.2f}, {max_vif:.2f}]")
        
        # Build report
        all_features = numeric_features.columns.tolist()
        removed_features = [f for f in all_features if f not in retained_features]
        
        report = RedundancyReport(
            original_features=all_features,
            retained_features=retained_features,
            removed_features=removed_features,
            feature_clusters=clusters,
            vif_scores=vif_scores,
            correlation_matrix=numeric_features[retained_features].corr() if len(retained_features) > 1 else None
        )
        
        self.logger.info(
            f"Redundancy analysis complete: {len(all_features)} -> {len(retained_features)} features "
            f"({report.reduction_rate:.1%} reduction)"
        )
        tprint_success(f"✅ Redundancy analysis complete: {len(all_features)} → {len(retained_features)} features ({report.reduction_rate:.1%} reduction)")
        tprint_info(f"📊 {len(removed_features)} redundant features removed, {len(report.feature_clusters)} clusters identified")
        
        return report
    
    def _cluster_correlated_features(
        self,
        features: pd.DataFrame
    ) -> List[FeatureCluster]:
        """
        Cluster features based on correlation using hierarchical clustering.
        
        Args:
            features: DataFrame with numeric features
        
        Returns:
            List of FeatureCluster objects
        """
        if features.shape[1] < 2:
            # Single feature, no clustering needed
            return [FeatureCluster(
                cluster_id=0,
                feature_names=features.columns.tolist(),
                representative_feature=features.columns[0],
                mean_correlation=1.0,
                max_correlation=1.0,
                cluster_size=1
            )]
        
        # Compute correlation matrix using matrix operations
        tprint_debug(f"🔢 Computing correlation matrix for {features.shape[1]} features")
        try:
            corr_matrix = correlation_matrix_gpu(features.values)
            corr_matrix = pd.DataFrame(corr_matrix, index=features.columns, columns=features.columns)
        except Exception as e:
            tprint_debug(f"⚠️ Matrix correlation failed: {e}, using pandas fallback")
            corr_matrix = features.corr().fillna(0)

        # Convert to distance matrix (distance = 1 - |correlation|)
        dist_matrix = 1 - np.abs(corr_matrix.values)
        
        # Ensure distance matrix is valid
        dist_matrix = np.clip(dist_matrix, 0, 2)
        np.fill_diagonal(dist_matrix, 0)
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(dist_matrix, checks=False)
        
        # Perform hierarchical clustering
        try:
            linkage_matrix = hierarchy.linkage(
                condensed_dist,
                method=self.config.clustering_method
            )
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}, using single feature per cluster")
            tprint_error(f"❌ Clustering failed: {e}")
            tprint_warning(f"⚠️ Falling back to single-feature clusters")
            # Fallback: each feature is its own cluster
            fallback_clusters = [
                FeatureCluster(
                    cluster_id=i,
                    feature_names=[col],
                    representative_feature=col,
                    mean_correlation=1.0,
                    max_correlation=1.0,
                    cluster_size=1
                )
                for i, col in enumerate(features.columns)
            ]
            tprint_info(f"📊 Created {len(fallback_clusters)} single-feature clusters as fallback")
            return fallback_clusters
        
        # Cut dendrogram at correlation threshold
        distance_threshold = 1 - self.config.correlation_threshold
        cluster_labels = hierarchy.fcluster(
            linkage_matrix,
            t=distance_threshold,
            criterion='distance'
        )
        
        # Build cluster objects
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            cluster_features = features.columns[cluster_labels == label].tolist()
            
            if len(cluster_features) == 1:
                mean_corr = 1.0
                max_corr = 1.0
                representative = cluster_features[0]
            else:
                # Compute mean and max correlation within cluster
                cluster_corr = corr_matrix.loc[cluster_features, cluster_features]
                # Exclude diagonal
                mask = ~np.eye(len(cluster_features), dtype=bool)
                corr_values = np.abs(cluster_corr.values[mask])
                mean_corr = np.mean(corr_values)
                max_corr = np.max(corr_values)
                
                # Representative is feature with highest average correlation to others
                avg_corr = cluster_corr.abs().mean()
                representative = avg_corr.idxmax()
            
            clusters.append(FeatureCluster(
                cluster_id=int(label),
                feature_names=cluster_features,
                representative_feature=representative,
                mean_correlation=float(mean_corr),
                max_correlation=float(max_corr),
                cluster_size=len(cluster_features)
            ))
        
        self.logger.info(
            f"Identified {len(clusters)} feature clusters "
            f"(threshold: {self.config.correlation_threshold})"
        )
        
        return clusters
    
    def _select_representatives(
        self,
        features: pd.DataFrame,
        clusters: List[FeatureCluster],
        importance: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Select representative features from each cluster.
        
        Args:
            features: DataFrame with features
            clusters: List of feature clusters
            importance: Optional importance scores
        
        Returns:
            List of selected feature names
        """
        selected = []
        
        for cluster in clusters:
            if self.config.selection_strategy == "best_per_cluster":
                # Select feature with highest importance in cluster
                if importance:
                    cluster_importance = {
                        f: importance.get(f, 0.0)
                        for f in cluster.feature_names
                    }
                    best_feature = max(cluster_importance, key=cluster_importance.get)
                else:
                    # No importance, use representative
                    best_feature = cluster.representative_feature
                
                selected.append(best_feature)
            
            elif self.config.selection_strategy == "representative":
                # Use pre-computed representative
                selected.append(cluster.representative_feature)
            
            elif self.config.selection_strategy == "all_below_threshold":
                # Keep all features in cluster if cluster is small
                if cluster.cluster_size <= 3:
                    selected.extend(cluster.feature_names)
                else:
                    selected.append(cluster.representative_feature)
        
        return selected
    
    def _compute_vif(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Variance Inflation Factor for each feature using vectorized operations.
        
        VIF measures multicollinearity:
        - VIF = 1: No correlation with other features
        - VIF < 5: Low multicollinearity
        - VIF 5-10: Moderate multicollinearity
        - VIF > 10: High multicollinearity
        
        Args:
            features: DataFrame with features
        
        Returns:
            Dictionary mapping feature names to VIF scores
        """
        if features.shape[1] < 2:
            return {features.columns[0]: 1.0} if features.shape[1] == 1 else {}
        
        tprint_debug(f"🔢 Computing VIF for {features.shape[1]} features using vectorized operations")
        
        vif_scores = {}
        
        # Standardize features once
        features_std = (features - features.mean()) / (features.std() + 1e-8)
        features_std = features_std.fillna(0)
        features_array = features_std.values
        
        # Try using batch matrix operations if available
        try:
            from src.utils.matrix_operations import get_unified_matrix_operations
            matrix_ops = get_unified_matrix_operations()
            
            # Compute correlation matrix once
            corr_matrix = matrix_ops.safe_correlation_matrix(features_array)
            
            # Compute VIF from correlation matrix
            for i, col in enumerate(features.columns):
                try:
                    # Get correlation with other features
                    other_indices = [j for j in range(len(features.columns)) if j != i]
                    if not other_indices:
                        vif_scores[col] = 1.0
                        continue
                    
                    # R² = sum of squared correlations (approximation)
                    # More accurate: use matrix inversion
                    try:
                        # Extract sub-correlation matrix excluding current feature
                        sub_corr = corr_matrix[np.ix_(other_indices, other_indices)]
                        cross_corr = corr_matrix[i, other_indices]
                        
                        # Solve: beta = (X'X)^{-1} X'y
                        beta = np.linalg.solve(sub_corr, cross_corr)
                        r_squared = np.dot(cross_corr, beta)
                        
                        # Ensure R² is in valid range
                        r_squared = np.clip(r_squared, 0, 0.9999)
                        
                        # VIF = 1 / (1 - R²)
                        vif = 1.0 / (1.0 - r_squared)
                        vif_scores[col] = float(vif)
                        
                    except np.linalg.LinAlgError:
                        # Singular matrix - perfect multicollinearity
                        vif_scores[col] = 1000.0
                        tprint_debug(f"⚠️ Singular matrix for {col}, VIF=1000")
                    
                except Exception as e:
                    self.logger.warning(f"VIF computation failed for {col}: {e}")
                    tprint_debug(f"⚠️ VIF computation failed for {col}: {e}, using default VIF=1.0")
                    vif_scores[col] = 1.0
            
            tprint_debug(f"✅ VIF computed using matrix operations")
            
        except ImportError:
            # Fallback to column-wise computation
            tprint_debug("⚠️ Matrix operations not available, using column-wise VIF computation")
            
            for i, col in enumerate(features.columns):
                try:
                    # Regress feature on all other features
                    X = features_std.drop(columns=[col]).values
                    y = features_std[col].values
                    
                    # Compute R²
                    if X.shape[1] == 0:
                        vif = 1.0
                    else:
                        # Simple linear regression
                        X_with_intercept = np.column_stack([np.ones(len(X)), X])
                        
                        try:
                            # Solve normal equations
                            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                            y_pred = X_with_intercept @ beta
                            
                            # R²
                            ss_res = np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - y.mean()) ** 2)
                            
                            if ss_tot < 1e-10:
                                r_squared = 0.0
                            else:
                                r_squared = 1 - ss_res / ss_tot
                            
                            # VIF = 1 / (1 - R²)
                            if r_squared >= 0.9999:
                                vif = 1000.0  # Cap at 1000
                            else:
                                vif = 1.0 / (1.0 - r_squared)
                        except np.linalg.LinAlgError:
                            vif = 1000.0  # Singular matrix indicates perfect multicollinearity
                    
                    vif_scores[col] = float(vif)
                    
                except Exception as e:
                    self.logger.warning(f"Could not compute VIF for {col}: {e}")
                    vif_scores[col] = 1.0
        
        return vif_scores
    
    def _remove_high_vif(self, features: pd.DataFrame) -> List[str]:
        """
        Iteratively remove features with high VIF.
        
        Args:
            features: DataFrame with features
        
        Returns:
            List of retained feature names
        """
        retained_features = features.columns.tolist()
        
        for iteration in range(self.config.max_vif_iterations):
            if len(retained_features) < 2:
                break
            
            vif_scores = self._compute_vif(features[retained_features])
            
            # Find feature with highest VIF
            max_vif_feature = max(vif_scores, key=vif_scores.get)
            max_vif = vif_scores[max_vif_feature]
            
            if max_vif > self.config.vif_threshold:
                self.logger.info(
                    f"Removing {max_vif_feature} (VIF={max_vif:.2f}) "
                    f"[iteration {iteration + 1}]"
                )
                retained_features.remove(max_vif_feature)
            else:
                # All features below threshold
                break
        
        return retained_features


class DriftMonitor:
    """
    Monitors feature drift between train and validation/test sets.
    """
    
    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the drift monitor.
        
        Args:
            config: Drift configuration
            logger: Optional logger instance
        """
        self.config = config or DriftConfig()
        self.logger = logger or system_logger.getChild('DriftMonitor')
    
    def detect_drift(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame
    ) -> DriftReport:
        """
        Detect distribution drift between reference and current features.
        
        Args:
            reference_features: Reference feature distribution (e.g., training data)
            current_features: Current feature distribution (e.g., validation/test data)
        
        Returns:
            DriftReport with drift analysis
        """
        tprint_info("🔍 Starting feature drift detection")
        tprint_debug(f"📊 Reference: {reference_features.shape}, Current: {current_features.shape}")
        
        # Align columns
        common_cols = reference_features.columns.intersection(current_features.columns)
        common_cols = common_cols[reference_features[common_cols].dtypes.apply(lambda x: np.issubdtype(x, np.number))]
        
        if len(common_cols) == 0:
            self.logger.warning("No common numeric columns for drift detection")
            tprint_error("❌ No common numeric columns found for drift detection")
            return DriftReport(feature_drifts={}, drifted_features=[])
        
        reference = reference_features[common_cols]
        current = current_features[common_cols]
        
        # Compute KL divergence for each feature
        feature_drifts = {}
        
        for col in common_cols:
            ref_series = reference[col].dropna()
            cur_series = current[col].dropna()
            
            if len(ref_series) < self.config.min_samples or len(cur_series) < self.config.min_samples:
                continue
            
            # Compute KL divergence
            kl_div = self._compute_kl_divergence(ref_series, cur_series)
            feature_drifts[col] = kl_div
        
        # Identify drifted features
        drifted_features = [
            col for col, kl in feature_drifts.items()
            if kl > self.config.kl_threshold
        ]
        
        # Compute correlation drift if enabled
        correlation_drift = None
        if self.config.detect_correlation_shift and len(common_cols) > 1:
            tprint_debug("🔢 Computing correlation drift using matrix operations")
            try:
                ref_corr = correlation_matrix_gpu(reference.values)
                cur_corr = correlation_matrix_gpu(current.values)

                # Frobenius norm of difference
                correlation_drift = float(np.linalg.norm(ref_corr - cur_corr, 'fro'))
            except Exception as e:
                tprint_debug(f"⚠️ Matrix correlation drift failed: {e}, using pandas fallback")
                ref_corr = reference.corr().fillna(0).values
                cur_corr = current.corr().fillna(0).values

                # Frobenius norm of difference
                correlation_drift = float(np.linalg.norm(ref_corr - cur_corr, 'fro'))
        
        report = DriftReport(
            feature_drifts=feature_drifts,
            drifted_features=drifted_features,
            correlation_drift=correlation_drift
        )
        
        self.logger.info(
            f"Drift detection complete: {len(drifted_features)}/{len(feature_drifts)} features drifted "
            f"(threshold: {self.config.kl_threshold})"
        )
        
        if len(drifted_features) > 0:
            tprint_warning(f"⚠️ Drift detected in {len(drifted_features)}/{len(feature_drifts)} features (threshold={self.config.kl_threshold})")
            tprint_debug(f"📊 Drifted features: {drifted_features[:10]}" + ("..." if len(drifted_features) > 10 else ""))
        else:
            tprint_success(f"✅ No feature drift detected ({len(feature_drifts)} features checked)")
        
        if correlation_drift is not None:
            tprint_debug(f"📊 Correlation drift (Frobenius norm): {correlation_drift:.4f}")
        
        return report
    
    def _compute_kl_divergence(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Compute KL divergence between two distributions using adaptive binning.
        
        Args:
            reference: Reference distribution
            current: Current distribution
        
        Returns:
            KL divergence (0 = identical distributions)
        """
        try:
            # Remove NaN values
            reference = reference.dropna()
            current = current.dropna()
            
            if len(reference) < 10 or len(current) < 10:
                tprint_debug(f"⚠️ Insufficient data for KL divergence: ref={len(reference)}, cur={len(current)}")
                return 0.0
            
            # Create histograms with common bins
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if np.isclose(min_val, max_val):
                return 0.0  # Constant feature, no drift
            
            # Use adaptive binning based on data distribution
            # Use Freedman-Diaconis rule or Scott's rule
            try:
                from sklearn.preprocessing import KBinsDiscretizer
                
                # Determine optimal number of bins using Freedman-Diaconis
                iqr = np.percentile(reference, 75) - np.percentile(reference, 25)
                if iqr > 0:
                    h = 2 * iqr / (len(reference) ** (1/3))  # Freedman-Diaconis bin width
                    n_bins = max(5, min(int((max_val - min_val) / h), self.config.kl_bins))
                else:
                    n_bins = self.config.kl_bins
                
                tprint_debug(f"🔢 Using {n_bins} adaptive bins for KL divergence")
                
            except ImportError:
                n_bins = self.config.kl_bins
            
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            ref_hist, _ = np.histogram(reference, bins=bins, density=True)
            cur_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            cur_hist = cur_hist / cur_hist.sum()
            
            # Compute KL divergence: D_KL(P || Q) = sum(P * log(P / Q))
            kl_div = np.sum(ref_hist * np.log(ref_hist / cur_hist))
            
            return float(max(0.0, kl_div))  # KL divergence is non-negative
        
        except Exception as e:
            self.logger.warning(f"Could not compute KL divergence: {e}")
            tprint_debug(f"⚠️ KL divergence computation failed: {e}")
            return 0.0


__all__ = [
    'RedundancyController',
    'DriftMonitor',
    'RedundancyConfig',
    'DriftConfig',
    'RedundancyReport',
    'DriftReport',
    'FeatureCluster',
]