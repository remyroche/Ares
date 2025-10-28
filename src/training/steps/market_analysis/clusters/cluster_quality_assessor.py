"""
Unified Cluster Quality Assessor

This module provides a unified, standardized way to assess cluster quality
across different clustering approaches (HDBSCAN, regime clustering, etc.).

It integrates with BaseStep's artifact manager and provides comprehensive
quality metrics including:
- Silhouette scores (global and per-cluster)
- Davies-Bouldin Index (DBI)
- Calinski-Harabasz Index (CH)
- Within/Between regime coefficient of variation
- Temporal smoothness
- Regime persistence
- Economic validation
- Predictive power
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Import sklearn metrics
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Import tprint utilities
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_debug,
    tprint_data_preview,
    tprint_data_format,
    tprint_timer,
    tprint_logged
)

# Import hardware utilities
try:
    from src.utils.hardware.unified_hardware_manager import (
        get_unified_hardware_manager,
        WorkloadType,
        OptimizationLevel
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    tprint_warning("Hardware optimization utilities not available")

# Import vectorization utilities
try:
    from src.features_common.utils import (
        VectorBTRollingOptimizer,
        UnifiedVectorizationManager,
        get_vectorbt_rolling_optimizer,
        get_unified_vectorization_manager
    )
    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False
    tprint_warning("Vectorization utilities not available")

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Enumeration of regime types for cluster classification."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class ClusterQualityMetrics:
    """
    Comprehensive cluster quality metrics.
    
    Attributes:
        silhouette_score: Global silhouette score (-1 to 1, higher is better)
        silhouette_per_cluster: Per-cluster silhouette scores
        davies_bouldin_score: Davies-Bouldin Index (lower is better)
        calinski_harabasz_score: Calinski-Harabasz Index (higher is better)
        within_regime_cv: Within-regime coefficient of variation
        between_regime_cv: Between-regime coefficient of variation
        temporal_smoothness: Temporal smoothness score (0 to 1, higher is better)
        regime_persistence: Average regime duration
        n_regimes: Number of regimes (excluding noise)
        noise_ratio: Ratio of noise points
        per_regime_metrics: Per-regime detailed metrics
        economic_validation: Economic validation results
        predictive_power: Predictive power score (cross-validation)
        quality_score: Overall composite quality score (0 to 1)
    """
    # Core clustering metrics
    silhouette_score: Optional[float] = None
    silhouette_per_cluster: Optional[Dict[int, Dict[str, float]]] = None
    davies_bouldin_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    
    # Coefficient of variation metrics (with std dev)
    within_regime_cv: Optional[float] = None
    within_regime_cv_std: Optional[float] = None
    between_regime_cv: Optional[float] = None
    between_regime_cv_std: Optional[float] = None
    per_regime_cv: Optional[Dict[int, float]] = None  # Per-regime CV values
    
    # Temporal metrics
    temporal_smoothness: Optional[float] = None
    regime_persistence: Optional[float] = None
    
    # Cluster composition
    n_regimes: int = 0
    noise_ratio: float = 0.0
    
    # Balance metrics
    balance_score: Optional[float] = None  # Global balance score (0-1, higher is better)
    min_cluster_size_pct: Optional[float] = None  # Smallest cluster as % of total
    max_cluster_size_pct: Optional[float] = None  # Largest cluster as % of total
    cluster_size_std: Optional[float] = None  # Std dev of cluster sizes
    cluster_size_distribution: Optional[List[float]] = None  # Size of each cluster as %
    
    # Model-specific metrics
    log_likelihood: Optional[float] = None  # For Markov-Switching, HMM models
    
    # Per-regime metrics
    per_regime_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Regime type classification
    regime_type_per_cluster: Optional[Dict[int, str]] = None
    
    # Economic validation
    economic_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Economic interpretation (data-driven insights)
    economic_interpretation: Dict[str, Any] = field(default_factory=dict)
    
    # Predictive power
    predictive_power: Optional[float] = None
    
    # Overall quality
    quality_score: Optional[float] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            # Core metrics
            'silhouette_score': self.silhouette_score,
            'silhouette_per_cluster': self.silhouette_per_cluster,
            'davies_bouldin_score': self.davies_bouldin_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            
            # CV metrics with std dev
            'within_regime_cv': self.within_regime_cv,
            'within_regime_cv_std': self.within_regime_cv_std,
            'between_regime_cv': self.between_regime_cv,
            'between_regime_cv_std': self.between_regime_cv_std,
            'per_regime_cv': self.per_regime_cv,
            
            # Temporal metrics
            'temporal_smoothness': self.temporal_smoothness,
            'regime_persistence': self.regime_persistence,
            
            # Composition metrics
            'n_regimes': self.n_regimes,
            'noise_ratio': self.noise_ratio,
            
            # Balance metrics
            'balance_score': self.balance_score,
            'min_cluster_size_pct': self.min_cluster_size_pct,
            'max_cluster_size_pct': self.max_cluster_size_pct,
            'cluster_size_std': self.cluster_size_std,
            'cluster_size_distribution': self.cluster_size_distribution,
            
            # Model-specific
            'log_likelihood': self.log_likelihood,
            
            # Detailed metrics
            'per_regime_metrics': self.per_regime_metrics,
            'regime_type_per_cluster': self.regime_type_per_cluster,
            'economic_validation': self.economic_validation,
            'economic_interpretation': self.economic_interpretation,
            
            # Aggregate scores
            'predictive_power': self.predictive_power,
            'quality_score': self.quality_score,
            
            # Metadata
            'timestamp': self.timestamp
        }
    
    def is_high_quality(self, 
                        min_silhouette: float = 0.3,
                        max_dbi: float = 2.0,
                        min_ch: float = 50.0,
                        max_noise: float = 0.3) -> bool:
        """
        Check if clustering quality meets minimum thresholds.
        
        Args:
            min_silhouette: Minimum silhouette score
            max_dbi: Maximum Davies-Bouldin Index
            min_ch: Minimum Calinski-Harabasz score
            max_noise: Maximum noise ratio
            
        Returns:
            True if quality meets all thresholds
        """
        checks = []
        
        if self.silhouette_score is not None:
            checks.append(self.silhouette_score >= min_silhouette)
        
        if self.davies_bouldin_score is not None:
            checks.append(self.davies_bouldin_score <= max_dbi)
        
        if self.calinski_harabasz_score is not None:
            checks.append(self.calinski_harabasz_score >= min_ch)
        
        checks.append(self.noise_ratio <= max_noise)
        
        return all(checks) if checks else False


class ClusterQualityAssessor:
    """
    Unified cluster quality assessor for regime/cluster analysis.
    
    This class provides a standardized way to assess cluster quality across
    different clustering approaches. It integrates with BaseStep's artifact
    manager and computes comprehensive quality metrics.
    """
    
    def __init__(self, artifact_manager=None, enable_hardware_optimization=True, enable_vectorization=True):
        """
        Initialize the cluster quality assessor.
        
        Args:
            artifact_manager: Optional artifact manager from BaseStep
            enable_hardware_optimization: Enable hardware optimizations
            enable_vectorization: Enable vectorized computations
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.artifact_manager = artifact_manager
        
        tprint_info("🔧 Initializing ClusterQualityAssessor")
        
        # Initialize hardware manager if available
        self.hardware_manager = None
        if enable_hardware_optimization and HARDWARE_AVAILABLE:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                self.hardware_manager.optimize_for_workload(
                    WorkloadType.DATA_PROCESSING,
                    OptimizationLevel.BALANCED
                )
                tprint_success("✅ Hardware optimization enabled")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware optimization failed: {e}")
                self.hardware_manager = None
        
        # Initialize vectorization manager if available
        self.vectorization_manager = None
        if enable_vectorization and VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_success("✅ Vectorization enabled")
            except Exception as e:
                tprint_warning(f"⚠️ Vectorization initialization failed: {e}")
                self.vectorization_manager = None
    
    @tprint_logged(include_args=False, include_result=False)
    def assess_quality(self,
                      regime_labels: np.ndarray,
                      feature_data: pd.DataFrame,
                      forward_returns: Optional[pd.Series] = None,
                      timestamps: Optional[pd.DatetimeIndex] = None,
                      min_regime_size: int = 10) -> ClusterQualityMetrics:
        """
        Comprehensive cluster quality assessment.
        
        Args:
            regime_labels: Regime/cluster labels (-1 for noise)
            feature_data: Feature data used for clustering
            forward_returns: Optional forward returns for economic validation
            timestamps: Optional timestamps for temporal analysis
            min_regime_size: Minimum regime size to consider
            
        Returns:
            ClusterQualityMetrics object with all computed metrics
        """
        tprint_info("🔍 Starting comprehensive cluster quality assessment")
        
        # Preview input data
        tprint_data_preview(regime_labels, "Regime Labels", max_rows=10)
        tprint_data_preview(feature_data, "Feature Data", max_rows=5, max_cols=5)
        tprint_data_format(feature_data, "Feature Data", check_compatibility=True)
        
        # Initialize metrics object
        metrics = ClusterQualityMetrics()
        
        # Validate inputs
        if len(regime_labels) == 0 or feature_data.empty:
            tprint_warning("⚠️ Empty inputs - cannot assess quality")
            return metrics
        
        # Filter out noise points for core metrics
        non_noise_mask = regime_labels != -1
        
        if np.sum(non_noise_mask) < min_regime_size:
            tprint_warning(f"⚠️ Insufficient non-noise points ({np.sum(non_noise_mask)}) for quality assessment")
            return metrics
        
        # Get clean numeric features
        features_clean = feature_data.select_dtypes(include=[np.number])
        if features_clean.empty:
            tprint_warning("⚠️ No numeric features available for quality assessment")
            return metrics
        
        tprint_data_preview(features_clean, "Clean Numeric Features", max_rows=5)
        
        # Calculate basic statistics
        metrics.n_regimes = len(set(regime_labels[non_noise_mask]))
        metrics.noise_ratio = np.sum(~non_noise_mask) / len(regime_labels)
        
        tprint_info(f"📊 Assessing quality for {metrics.n_regimes} regimes with {metrics.noise_ratio:.1%} noise")
        
        # 1. Silhouette scores
        try:
            with tprint_timer("Silhouette Score Calculation"):
                metrics.silhouette_score, metrics.silhouette_per_cluster = self._calculate_silhouette_scores(
                    regime_labels, features_clean, non_noise_mask
                )
            tprint_success(f"✅ Silhouette score: {metrics.silhouette_score:.4f}")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate silhouette scores: {e}")
        
        # 2. Davies-Bouldin Index
        try:
            with tprint_timer("Davies-Bouldin Index Calculation"):
                metrics.davies_bouldin_score = self._calculate_dbi(
                    regime_labels, features_clean, non_noise_mask
                )
            tprint_success(f"✅ Davies-Bouldin Index: {metrics.davies_bouldin_score:.4f}")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate DBI: {e}")
        
        # 3. Calinski-Harabasz Index
        try:
            with tprint_timer("Calinski-Harabasz Index Calculation"):
                metrics.calinski_harabasz_score = self._calculate_ch(
                    regime_labels, features_clean, non_noise_mask
                )
            tprint_success(f"✅ Calinski-Harabasz Index: {metrics.calinski_harabasz_score:.4f}")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate CH score: {e}")
        
        # 4. Coefficient of variation metrics (with std dev and per-regime)
        try:
            with tprint_timer("CV Metrics Calculation"):
                (metrics.within_regime_cv, metrics.within_regime_cv_std,
                 metrics.between_regime_cv, metrics.between_regime_cv_std,
                 metrics.per_regime_cv) = self._calculate_cv_metrics(
                    regime_labels, features_clean, non_noise_mask
                )
            tprint_success(f"✅ Within CV: {metrics.within_regime_cv:.4f}, Between CV: {metrics.between_regime_cv:.4f}")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate CV metrics: {e}")
        
        # 5. Balance metrics
        try:
            with tprint_timer("Balance Metrics Calculation"):
                (metrics.balance_score, metrics.min_cluster_size_pct,
                 metrics.max_cluster_size_pct, metrics.cluster_size_std,
                 metrics.cluster_size_distribution) = self._calculate_balance_metrics(regime_labels)
            tprint_success(f"✅ Balance score: {metrics.balance_score:.4f}")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate balance metrics: {e}")
        
        # 6. Temporal smoothness and persistence
        if timestamps is not None:
            try:
                with tprint_timer("Temporal Metrics Calculation"):
                    metrics.temporal_smoothness = self._calculate_temporal_smoothness(
                        regime_labels, timestamps
                    )
                    metrics.regime_persistence = self._calculate_regime_persistence(regime_labels)
                tprint_success(f"✅ Temporal smoothness: {metrics.temporal_smoothness:.4f}, Persistence: {metrics.regime_persistence:.2f}")
            except Exception as e:
                tprint_error(f"❌ Failed to calculate temporal metrics: {e}")
        
        # 7. Per-regime metrics (includes regime type detection)
        try:
            with tprint_timer("Per-Regime Metrics Calculation"):
                metrics.per_regime_metrics = self._calculate_per_regime_metrics(
                    regime_labels, features_clean, forward_returns
                )
                
                # Extract regime types from per-regime metrics
                metrics.regime_type_per_cluster = {
                    regime_id: regime_data.get('regime_type', RegimeType.UNKNOWN.value)
                    for regime_id, regime_data in metrics.per_regime_metrics.items()
                }
            tprint_success(f"✅ Calculated metrics for {len(metrics.per_regime_metrics)} regimes")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate per-regime metrics: {e}")
        
        # 8. Economic validation (if forward returns provided)
        if forward_returns is not None:
            try:
                with tprint_timer("Economic Validation"):
                    metrics.economic_validation = self._validate_regime_quality(
                        regime_labels, forward_returns, feature_data
                    )
                tprint_success("✅ Economic validation complete")
            except Exception as e:
                tprint_error(f"❌ Failed to validate regime quality: {e}")
        
        # 8b. Economic interpretation (data-driven insights)
        try:
            with tprint_timer("Economic Interpretation"):
                metrics.economic_interpretation = self._generate_economic_interpretation(
                    metrics.per_regime_metrics, metrics.regime_type_per_cluster
                )
            tprint_success("✅ Economic interpretation generated")
        except Exception as e:
            tprint_error(f"❌ Failed to generate economic interpretation: {e}")
        
        # 9. Predictive power
        if forward_returns is not None and len(forward_returns) > 0:
            try:
                with tprint_timer("Predictive Power Calculation"):
                    metrics.predictive_power = self._calculate_predictive_power(
                        regime_labels, forward_returns
                    )
                tprint_success(f"✅ Predictive power: {metrics.predictive_power:.4f}")
            except Exception as e:
                tprint_error(f"❌ Failed to calculate predictive power: {e}")
        
        # 10. Calculate overall quality score
        try:
            with tprint_timer("Quality Score Calculation"):
                metrics.quality_score = self._calculate_quality_score(metrics)
            tprint_success(f"✅ Overall quality score: {metrics.quality_score:.4f}")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate quality score: {e}")
        
        tprint_success(f"✅ Quality assessment complete - Quality Score: {metrics.quality_score:.3f}")
        
        return metrics
    
    def _calculate_silhouette_scores(self,
                                    regime_labels: np.ndarray,
                                    features: pd.DataFrame,
                                    non_noise_mask: np.ndarray) -> Tuple[float, Dict[int, Dict[str, float]]]:
        """Calculate global and per-cluster silhouette scores."""
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]
        
        if len(set(labels_clean)) < 2:
            return 0.0, {}
        
        # Global silhouette score
        global_silhouette = silhouette_score(features_clean, labels_clean)
        
        # Per-cluster silhouette scores
        silhouette_samples_scores = silhouette_samples(features_clean, labels_clean)
        per_cluster_silhouette = {}
        
        for cluster_id in set(labels_clean):
            cluster_mask = labels_clean == cluster_id
            cluster_scores = silhouette_samples_scores[cluster_mask]
            
            per_cluster_silhouette[int(cluster_id)] = {
                'mean': float(np.mean(cluster_scores)),
                'std': float(np.std(cluster_scores)),
                'min': float(np.min(cluster_scores)),
                'max': float(np.max(cluster_scores))
            }
        
        return global_silhouette, per_cluster_silhouette
    
    def _calculate_dbi(self,
                       regime_labels: np.ndarray,
                       features: pd.DataFrame,
                       non_noise_mask: np.ndarray) -> float:
        """Calculate Davies-Bouldin Index (lower is better)."""
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]
        
        if len(set(labels_clean)) < 2:
            return float('inf')
        
        return davies_bouldin_score(features_clean, labels_clean)
    
    def _calculate_ch(self,
                      regime_labels: np.ndarray,
                      features: pd.DataFrame,
                      non_noise_mask: np.ndarray) -> float:
        """Calculate Calinski-Harabasz Index (higher is better)."""
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]
        
        if len(set(labels_clean)) < 2:
            return 0.0
        
        return calinski_harabasz_score(features_clean, labels_clean)
    
    def _calculate_cv_metrics(self,
                              regime_labels: np.ndarray,
                              features: pd.DataFrame,
                              non_noise_mask: np.ndarray) -> Tuple[float, float, float, float, Dict[int, float]]:
        """
        Calculate within-regime and between-regime coefficient of variation with std dev.
        
        Returns:
            Tuple of (within_regime_cv_mean, within_regime_cv_std, 
                     between_regime_cv_mean, between_regime_cv_std,
                     per_regime_cv_dict)
        """
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]
        
        if len(set(labels_clean)) < 2:
            return 0.0, 0.0, 0.0, 0.0, {}
        
        # Within-regime CV (per cluster)
        within_cvs = []
        per_regime_cv = {}
        
        for cluster_id in set(labels_clean):
            cluster_mask = labels_clean == cluster_id
            cluster_data = features_clean[cluster_mask]
            
            if len(cluster_data) > 1:
                cluster_std = cluster_data.std()
                cluster_mean = cluster_data.mean()
                
                # Safe division with proper handling of zeros
                denominator = np.abs(cluster_mean) + 1e-8
                cv_values = np.divide(
                    cluster_std,
                    denominator,
                    out=np.zeros_like(cluster_std),
                    where=denominator != 0
                )
                
                # Remove infinite or NaN values
                cv_values = cv_values[np.isfinite(cv_values)]
                
                if len(cv_values) > 0:
                    cluster_cv = float(np.mean(cv_values))
                    within_cvs.append(cluster_cv)
                    per_regime_cv[int(cluster_id)] = cluster_cv
        
        # Calculate mean and std dev of within-regime CVs
        within_regime_cv_mean = float(np.mean(within_cvs)) if within_cvs else 0.0
        within_regime_cv_std = float(np.std(within_cvs)) if len(within_cvs) > 1 else 0.0
        
        # Between-regime CV
        cluster_means = []
        for cluster_id in set(labels_clean):
            cluster_mask = labels_clean == cluster_id
            cluster_data = features_clean[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_mean = cluster_data.mean()
                cluster_mean = cluster_mean[np.isfinite(cluster_mean)]
                if len(cluster_mean) > 0:
                    cluster_means.append(cluster_mean)
        
        between_regime_cv_mean = 0.0
        between_regime_cv_std = 0.0
        
        if len(cluster_means) > 1:
            cluster_means_array = np.array(cluster_means)
            
            # Calculate CV for each feature across regimes
            between_cvs = []
            for feature_idx in range(cluster_means_array.shape[1]):
                feature_means = cluster_means_array[:, feature_idx]
                feature_std = np.std(feature_means)
                feature_mean = np.mean(feature_means)
                
                # Safe division
                denominator = np.abs(feature_mean) + 1e-8
                cv = feature_std / denominator
                
                if np.isfinite(cv):
                    between_cvs.append(cv)
            
            between_regime_cv_mean = float(np.mean(between_cvs)) if between_cvs else 0.0
            between_regime_cv_std = float(np.std(between_cvs)) if len(between_cvs) > 1 else 0.0
        
        return within_regime_cv_mean, within_regime_cv_std, between_regime_cv_mean, between_regime_cv_std, per_regime_cv
    
    def _calculate_temporal_smoothness(self,
                                       regime_labels: np.ndarray,
                                       timestamps: pd.DatetimeIndex) -> float:
        """
        Calculate temporal smoothness score.
        
        Higher score means fewer regime transitions (more stable regimes).
        Score is normalized to [0, 1] where 1 is perfectly smooth.
        """
        if len(regime_labels) < 2:
            return 1.0
        
        # Count regime transitions
        regime_changes = np.sum(regime_labels[1:] != regime_labels[:-1])
        max_possible_changes = len(regime_labels) - 1
        
        # Smoothness score: fewer changes = higher smoothness
        smoothness = 1.0 - (regime_changes / max_possible_changes)
        
        return smoothness
    
    def _calculate_regime_persistence(self, regime_labels: np.ndarray) -> float:
        """
        Calculate average regime persistence (how long regimes typically last).
        
        Returns:
            Average number of bars a regime persists
        """
        if len(regime_labels) < 2:
            return float(len(regime_labels))
        
        regime_changes = (regime_labels[1:] != regime_labels[:-1]).astype(int)
        
        # Calculate average duration between changes
        avg_regime_duration = 1.0 / (np.mean(regime_changes) + 1e-8)
        
        return avg_regime_duration
    
    def _calculate_balance_metrics(self,
                                   regime_labels: np.ndarray) -> Tuple[float, float, float, float, List[float]]:
        """
        Calculate cluster balance metrics.
        
        Returns:
            Tuple of (balance_score, min_cluster_size_pct, max_cluster_size_pct,
                     cluster_size_std, cluster_size_distribution)
        """
        unique_labels = np.unique(regime_labels)
        non_noise_labels = unique_labels[unique_labels != -1]
        
        if len(non_noise_labels) < 2:
            return 0.0, 0.0, 0.0, 0.0, []
        
        total_samples = len(regime_labels)
        cluster_sizes = []
        cluster_size_distribution = []
        
        for label in non_noise_labels:
            size = int(np.sum(regime_labels == label))
            size_pct = float(100.0 * size / total_samples)
            cluster_sizes.append(size)
            cluster_size_distribution.append(size_pct)
        
        # Calculate metrics
        min_cluster_size_pct = float(min(cluster_size_distribution))
        max_cluster_size_pct = float(max(cluster_size_distribution))
        cluster_size_std = float(np.std(cluster_sizes))
        
        # Calculate balance score (0-1, higher is better)
        # Perfect balance = all clusters same size (std = 0, score = 1)
        # Highly imbalanced = one cluster dominates (score → 0)
        mean_size = np.mean(cluster_sizes)
        if mean_size > 0:
            # Normalize std by mean to get coefficient of variation
            cv = cluster_size_std / mean_size
            # Convert to score (0-1, lower CV = higher score)
            balance_score = float(1.0 / (1.0 + cv))
        else:
            balance_score = 0.0
        
        return balance_score, min_cluster_size_pct, max_cluster_size_pct, cluster_size_std, cluster_size_distribution
    
    def _detect_regime_type(self, 
                           regime_data: pd.DataFrame,
                           returns: Optional[pd.Series] = None) -> Tuple[RegimeType, Dict[str, float]]:
        """
        Detect regime type based on data characteristics (data-driven).
        
        Args:
            regime_data: Feature data for this regime
            returns: Optional returns series for this regime
            
        Returns:
            Tuple of (RegimeType, metrics_dict with scores for classification)
        """
        metrics = {}
        
        try:
            # Calculate regime characteristics
            if returns is not None and len(returns) > 1:
                # Trend characteristics
                mean_return = returns.mean()
                returns_std = returns.std()
                
                # Trend strength: normalized mean return / std
                trend_strength = abs(mean_return) / (returns_std + 1e-8)
                metrics['trend_strength'] = float(trend_strength)
                
                # Trend persistence: autocorrelation
                if len(returns) > 2:
                    autocorr = returns.autocorr(lag=1)
                    metrics['trend_persistence'] = float(autocorr if not np.isnan(autocorr) else 0.0)
                else:
                    metrics['trend_persistence'] = 0.0
                
                # Mean reversion strength: negative autocorrelation indicates mean reversion
                mean_reversion_score = -metrics['trend_persistence']
                metrics['mean_reversion_strength'] = float(mean_reversion_score)
                
                # Volatility characteristics
                volatility_level = returns_std
                metrics['volatility_level'] = float(volatility_level)
                
                # Volatility clustering: autocorrelation of squared returns
                if len(returns) > 2:
                    squared_returns = returns ** 2
                    vol_clustering = squared_returns.autocorr(lag=1)
                    metrics['volatility_clustering'] = float(vol_clustering if not np.isnan(vol_clustering) else 0.0)
                else:
                    metrics['volatility_clustering'] = 0.0
                
                # Stability: coefficient of variation (inverse)
                cv = volatility_level / (abs(mean_return) + 1e-8)
                metrics['stability_score'] = float(1.0 / (1.0 + cv))
                
                # Classify regime based on dominant characteristic
                # Use data-driven thresholds
                
                # High volatility regime (volatility > 1.5 * median volatility)
                if volatility_level > 0.02:  # 2% daily volatility threshold
                    if metrics['volatility_clustering'] > 0.3:
                        return RegimeType.VOLATILE, metrics
                
                # Trending regime (strong trend + high persistence)
                if trend_strength > 0.5 and metrics['trend_persistence'] > 0.2:
                    return RegimeType.TRENDING, metrics
                
                # Mean reverting regime (negative autocorrelation)
                if metrics['trend_persistence'] < -0.1:
                    return RegimeType.MEAN_REVERTING, metrics
                
                # Stable regime (low volatility + low trend)
                if volatility_level < 0.01 and trend_strength < 0.3:
                    return RegimeType.STABLE, metrics
                
                # Default: determine by strongest signal
                max_score = max(
                    trend_strength,
                    abs(mean_reversion_score),
                    volatility_level * 10,  # Scale volatility for comparison
                    metrics['stability_score']
                )
                
                if max_score == trend_strength:
                    return RegimeType.TRENDING, metrics
                elif max_score == abs(mean_reversion_score):
                    return RegimeType.MEAN_REVERTING, metrics
                elif max_score == volatility_level * 10:
                    return RegimeType.VOLATILE, metrics
                else:
                    return RegimeType.STABLE, metrics
            
            return RegimeType.UNKNOWN, metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to detect regime type: {e}")
            return RegimeType.UNKNOWN, {}
    
    def _calculate_regime_specific_metrics(self,
                                          regime_type: RegimeType,
                                          regime_data: pd.DataFrame,
                                          returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate regime-specific metrics based on detected regime type.
        
        Args:
            regime_type: Detected regime type
            regime_data: Feature data for this regime
            returns: Optional returns series
            
        Returns:
            Dictionary of regime-specific metrics with scores
        """
        specific_metrics = {}
        
        try:
            if returns is None or len(returns) < 2:
                return specific_metrics
            
            if regime_type == RegimeType.TRENDING:
                # Trending regime metrics
                specific_metrics['trend_direction'] = 'bullish' if returns.mean() > 0 else 'bearish'
                specific_metrics['trend_consistency'] = float(
                    np.sum(np.sign(returns) == np.sign(returns.mean())) / len(returns)
                )
                
                # Trend acceleration/deceleration
                if len(returns) > 5:
                    first_half_mean = returns.iloc[:len(returns)//2].mean()
                    second_half_mean = returns.iloc[len(returns)//2:].mean()
                    specific_metrics['trend_acceleration'] = float(
                        (second_half_mean - first_half_mean) / (abs(first_half_mean) + 1e-8)
                    )
                
            elif regime_type == RegimeType.MEAN_REVERTING:
                # Mean reverting regime metrics
                mean_return = returns.mean()
                specific_metrics['reversion_center'] = float(mean_return)
                
                # Reversion speed: how quickly prices return to mean
                deviations = abs(returns - mean_return)
                specific_metrics['reversion_speed'] = float(1.0 / (deviations.mean() + 1e-8))
                
                # Reversion range: typical deviation from mean
                specific_metrics['reversion_range'] = float(deviations.std())
                
            elif regime_type == RegimeType.VOLATILE:
                # Volatile regime metrics
                specific_metrics['volatility_regime'] = 'high'
                
                # Volatility persistence
                if len(returns) > 5:
                    rolling_vol = returns.rolling(window=5).std()
                    vol_autocorr = rolling_vol.autocorr(lag=1)
                    specific_metrics['volatility_persistence'] = float(
                        vol_autocorr if not np.isnan(vol_autocorr) else 0.0
                    )
                
                # Extreme move frequency
                std_dev = returns.std()
                extreme_moves = np.sum(abs(returns) > 2 * std_dev)
                specific_metrics['extreme_move_frequency'] = float(extreme_moves / len(returns))
                
            elif regime_type == RegimeType.STABLE:
                # Stable regime metrics
                specific_metrics['stability_regime'] = 'low_volatility'
                specific_metrics['mean_return'] = float(returns.mean())
                specific_metrics['volatility'] = float(returns.std())
                
                # Stability score
                cv = returns.std() / (abs(returns.mean()) + 1e-8)
                specific_metrics['stability_coefficient'] = float(1.0 / (1.0 + cv))
            
            return specific_metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate regime-specific metrics: {e}")
            return specific_metrics
    
    def _calculate_per_regime_metrics(self,
                                      regime_labels: np.ndarray,
                                      features: pd.DataFrame,
                                      forward_returns: Optional[pd.Series] = None) -> Dict[int, Dict[str, Any]]:
        """Calculate detailed metrics for each regime, including regime type classification."""
        per_regime_metrics = {}
        total_samples = len(regime_labels)
        
        for regime_id in set(regime_labels):
            if regime_id == -1:  # Skip noise
                continue
            
            regime_mask = regime_labels == regime_id
            regime_features = features.iloc[regime_mask].select_dtypes(include=[np.number])
            
            if len(regime_features) == 0:
                continue
            
            # Feature coefficient of variation
            feature_cv = {}
            for col in regime_features.columns:
                if regime_features[col].std() > 0:
                    cv = regime_features[col].std() / (abs(regime_features[col].mean()) + 1e-8)
                    feature_cv[col] = float(cv)
            
            # Calculate regime-specific metrics
            regime_size = int(len(regime_features))
            regime_percentage = float((regime_size / total_samples) * 100)
            
            regime_metrics = {
                # Size and balance
                'size': regime_size,
                'percentage': regime_percentage,
                
                # CV metrics
                'feature_coefficient_of_variation': feature_cv,
                'mean_cv': float(np.mean(list(feature_cv.values()))) if feature_cv else 0.0,
                'std_cv': float(np.std(list(feature_cv.values()))) if feature_cv else 0.0,
                
                # Individual regime balance contribution
                'balance_contribution': float(regime_size / (np.mean([np.sum(regime_labels == r) 
                                                                       for r in set(regime_labels) 
                                                                       if r != -1]) + 1e-8))
            }
            
            # Detect regime type and calculate regime-specific characteristics
            regime_returns = forward_returns[regime_mask] if forward_returns is not None else None
            
            if regime_returns is not None and len(regime_returns) > 0:
                # Detect regime type
                regime_type, classification_scores = self._detect_regime_type(
                    regime_features, regime_returns
                )
                regime_metrics['regime_type'] = regime_type.value
                regime_metrics['classification_scores'] = classification_scores
                
                # Calculate regime-specific metrics based on detected type
                specific_metrics = self._calculate_regime_specific_metrics(
                    regime_type, regime_features, regime_returns
                )
                regime_metrics['regime_specific_metrics'] = specific_metrics
                
                # Add return characteristics
                regime_metrics.update({
                    'mean_return': float(regime_returns.mean()),
                    'volatility': float(regime_returns.std()),
                    'sharpe': float(regime_returns.mean() / (regime_returns.std() + 1e-8)),
                    'skewness': float(regime_returns.skew()) if hasattr(regime_returns, 'skew') else 0.0,
                    'max_drawdown': float(self._compute_max_drawdown(regime_returns))
                })
            else:
                regime_metrics['regime_type'] = RegimeType.UNKNOWN.value
                regime_metrics['classification_scores'] = {}
                regime_metrics['regime_specific_metrics'] = {}
            
            per_regime_metrics[int(regime_id)] = regime_metrics
        
        return per_regime_metrics
    
    def _generate_economic_interpretation(self,
                                          per_regime_metrics: Dict[int, Dict[str, Any]],
                                          regime_type_per_cluster: Optional[Dict[int, str]]) -> Dict[str, Any]:
        """
        Generate data-driven economic interpretation of regimes.
        
        Args:
            per_regime_metrics: Per-regime metrics including returns and characteristics
            regime_type_per_cluster: Regime type classification for each cluster
            
        Returns:
            Dictionary containing economic insights and actionable information
        """
        interpretation = {
            'regime_summary': {},
            'trading_implications': {},
            'risk_characteristics': {},
            'regime_transitions': {},
            'performance_comparison': {}
        }
        
        try:
            if not per_regime_metrics or not regime_type_per_cluster:
                return interpretation
            
            # 1. Regime Summary
            regime_types_count = {}
            for regime_type in regime_type_per_cluster.values():
                regime_types_count[regime_type] = regime_types_count.get(regime_type, 0) + 1
            
            interpretation['regime_summary'] = {
                'total_regimes': len(per_regime_metrics),
                'regime_type_distribution': regime_types_count,
                'dominant_regime': max(regime_types_count.items(), key=lambda x: x[1])[0] if regime_types_count else 'unknown'
            }
            
            # 2. Performance Comparison by Regime Type
            performance_by_type = {}
            
            for regime_id, metrics in per_regime_metrics.items():
                regime_type = metrics.get('regime_type', 'unknown')
                
                if regime_type not in performance_by_type:
                    performance_by_type[regime_type] = {
                        'mean_returns': [],
                        'volatilities': [],
                        'sharpe_ratios': [],
                        'regimes': []
                    }
                
                if 'mean_return' in metrics:
                    performance_by_type[regime_type]['mean_returns'].append(metrics['mean_return'])
                if 'volatility' in metrics:
                    performance_by_type[regime_type]['volatilities'].append(metrics['volatility'])
                if 'sharpe' in metrics:
                    performance_by_type[regime_type]['sharpe_ratios'].append(metrics['sharpe'])
                performance_by_type[regime_type]['regimes'].append(regime_id)
            
            # Aggregate performance statistics
            for regime_type, data in performance_by_type.items():
                interpretation['performance_comparison'][regime_type] = {
                    'avg_return': float(np.mean(data['mean_returns'])) if data['mean_returns'] else 0.0,
                    'avg_volatility': float(np.mean(data['volatilities'])) if data['volatilities'] else 0.0,
                    'avg_sharpe': float(np.mean(data['sharpe_ratios'])) if data['sharpe_ratios'] else 0.0,
                    'num_regimes': len(data['regimes']),
                    'regime_ids': data['regimes']
                }
            
            # 3. Trading Implications (data-driven)
            best_regime = None
            best_sharpe = float('-inf')
            worst_regime = None
            worst_sharpe = float('inf')
            
            for regime_id, metrics in per_regime_metrics.items():
                sharpe = metrics.get('sharpe', 0.0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_regime = (regime_id, metrics)
                if sharpe < worst_sharpe:
                    worst_sharpe = sharpe
                    worst_regime = (regime_id, metrics)
            
            if best_regime:
                regime_id, metrics = best_regime
                interpretation['trading_implications']['most_profitable_regime'] = {
                    'regime_id': regime_id,
                    'regime_type': metrics.get('regime_type', 'unknown'),
                    'sharpe_ratio': metrics.get('sharpe', 0.0),
                    'mean_return': metrics.get('mean_return', 0.0),
                    'volatility': metrics.get('volatility', 0.0),
                    'characteristics': metrics.get('regime_specific_metrics', {})
                }
            
            if worst_regime:
                regime_id, metrics = worst_regime
                interpretation['trading_implications']['least_profitable_regime'] = {
                    'regime_id': regime_id,
                    'regime_type': metrics.get('regime_type', 'unknown'),
                    'sharpe_ratio': metrics.get('sharpe', 0.0),
                    'mean_return': metrics.get('mean_return', 0.0),
                    'volatility': metrics.get('volatility', 0.0),
                    'characteristics': metrics.get('regime_specific_metrics', {})
                }
            
            # 4. Risk Characteristics by Regime Type
            for regime_id, metrics in per_regime_metrics.items():
                regime_type = metrics.get('regime_type', 'unknown')
                
                risk_profile = {
                    'regime_id': regime_id,
                    'volatility': metrics.get('volatility', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'skewness': metrics.get('skewness', 0.0)
                }
                
                # Add regime-specific risk insights
                specific_metrics = metrics.get('regime_specific_metrics', {})
                if regime_type == 'volatile':
                    risk_profile['extreme_move_frequency'] = specific_metrics.get('extreme_move_frequency', 0.0)
                    risk_profile['volatility_persistence'] = specific_metrics.get('volatility_persistence', 0.0)
                elif regime_type == 'trending':
                    risk_profile['trend_consistency'] = specific_metrics.get('trend_consistency', 0.0)
                    risk_profile['trend_direction'] = specific_metrics.get('trend_direction', 'unknown')
                elif regime_type == 'mean_reverting':
                    risk_profile['reversion_speed'] = specific_metrics.get('reversion_speed', 0.0)
                    risk_profile['reversion_range'] = specific_metrics.get('reversion_range', 0.0)
                
                interpretation['risk_characteristics'][f'regime_{regime_id}'] = risk_profile
            
            # 5. Strategy Recommendations (data-driven)
            recommendations = []
            
            # Identify trend-following opportunities
            trending_regimes = [
                (rid, m) for rid, m in per_regime_metrics.items() 
                if m.get('regime_type') == 'trending' and m.get('sharpe', 0) > 0.5
            ]
            if trending_regimes:
                best_trending = max(trending_regimes, key=lambda x: x[1].get('sharpe', 0))
                recommendations.append({
                    'strategy': 'trend_following',
                    'target_regime': best_trending[0],
                    'expected_sharpe': best_trending[1].get('sharpe', 0.0),
                    'confidence': best_trending[1].get('classification_scores', {}).get('trend_persistence', 0.0)
                })
            
            # Identify mean reversion opportunities
            mr_regimes = [
                (rid, m) for rid, m in per_regime_metrics.items() 
                if m.get('regime_type') == 'mean_reverting' and m.get('sharpe', 0) > 0.5
            ]
            if mr_regimes:
                best_mr = max(mr_regimes, key=lambda x: x[1].get('sharpe', 0))
                recommendations.append({
                    'strategy': 'mean_reversion',
                    'target_regime': best_mr[0],
                    'expected_sharpe': best_mr[1].get('sharpe', 0.0),
                    'confidence': abs(best_mr[1].get('classification_scores', {}).get('mean_reversion_strength', 0.0))
                })
            
            # Identify regimes to avoid
            high_risk_regimes = [
                rid for rid, m in per_regime_metrics.items()
                if m.get('max_drawdown', 0) < -0.15 or m.get('sharpe', 0) < -0.5
            ]
            if high_risk_regimes:
                recommendations.append({
                    'strategy': 'risk_avoidance',
                    'target_regimes': high_risk_regimes,
                    'rationale': 'high drawdown or negative sharpe'
                })
            
            interpretation['trading_implications']['strategy_recommendations'] = recommendations
            
            # 6. Regime Stability Insights
            regime_sizes = [m.get('percentage', 0) for m in per_regime_metrics.values()]
            interpretation['regime_transitions']['balance'] = {
                'most_common_regime_pct': float(max(regime_sizes)) if regime_sizes else 0.0,
                'least_common_regime_pct': float(min(regime_sizes)) if regime_sizes else 0.0,
                'size_distribution_std': float(np.std(regime_sizes)) if regime_sizes else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate complete economic interpretation: {e}")
        
        return interpretation
    
    def _validate_regime_quality(self,
                                 regime_labels: np.ndarray,
                                 forward_returns: pd.Series,
                                 feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if discovered regimes are actually predictive.
        
        This is based on the user's provided validate_regime_quality function.
        """
        results = {}
        
        # Per-regime statistics
        for regime_id in np.unique(regime_labels):
            if regime_id == -1:  # Skip noise
                continue
            
            regime_mask = (regime_labels == regime_id)
            regime_returns = forward_returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            results[f'regime_{regime_id}'] = {
                'mean_return': float(regime_returns.mean()),
                'volatility': float(regime_returns.std()),
                'sharpe': float(regime_returns.mean() / (regime_returns.std() + 1e-8)),
                'skewness': float(regime_returns.skew()) if hasattr(regime_returns, 'skew') else 0.0,
                'max_drawdown': float(self._compute_max_drawdown(regime_returns))
            }
            
            # Add feature behavior in this regime (using selected columns)
            numeric_features = feature_data.select_dtypes(include=[np.number])
            for col in ['spread', 'volume', 'volatility']:
                if col in numeric_features.columns:
                    results[f'regime_{regime_id}'][f'avg_{col}'] = float(
                        numeric_features.loc[regime_mask, col].mean()
                    )
        
        # Regime stability
        results['regime_persistence'] = float(self._calculate_regime_persistence(regime_labels))
        
        return results
    
    def _calculate_predictive_power(self,
                                   regime_labels: np.ndarray,
                                   forward_returns: pd.Series) -> float:
        """
        Calculate predictive power: can current regime predict future returns?
        
        Uses Random Forest classifier to predict return sign from regime labels.
        """
        try:
            # Use current regime to predict next period's return sign
            if len(regime_labels) < 10 or len(forward_returns) < 10:
                return 0.0
            
            # Ensure arrays are aligned and valid
            min_len = min(len(regime_labels), len(forward_returns))
            if min_len < 10:
                return 0.0
            
            X = pd.get_dummies(regime_labels[:min_len-1])
            y = (forward_returns[1:min_len] > 0).astype(int).values
            
            if len(X) != len(y):
                return 0.0
            
            # Check if we have enough samples and variation
            if len(y) < 10 or len(set(y)) < 2:
                return 0.0
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_score = cross_val_score(rf, X, y, cv=min(5, len(y) // 2)).mean()
            
            return float(cv_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate predictive power: {e}")
            return 0.0
    
    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown from returns series."""
        try:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            return float(drawdown.min())
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, metrics: ClusterQualityMetrics) -> float:
        """
        Calculate overall composite quality score (0 to 1, higher is better).
        
        Combines multiple metrics into a single score:
        - Silhouette score (weight: 0.20)
        - Davies-Bouldin Index (weight: 0.15)
        - Calinski-Harabasz Index (weight: 0.15)
        - Within/Between CV ratio (weight: 0.15)
        - Balance score (weight: 0.15)
        - Temporal smoothness (weight: 0.10)
        - Noise ratio (weight: 0.10)
        """
        score_components = []
        weights = []
        
        # 1. Silhouette score (normalize to 0-1, already in [-1, 1])
        if metrics.silhouette_score is not None:
            silhouette_normalized = (metrics.silhouette_score + 1) / 2  # Map [-1, 1] to [0, 1]
            score_components.append(silhouette_normalized)
            weights.append(0.20)
        
        # 2. Davies-Bouldin Index (lower is better, normalize inversely)
        if metrics.davies_bouldin_score is not None and not np.isinf(metrics.davies_bouldin_score):
            # DBI typically ranges from 0 to 5+, map to [0, 1] inversely
            dbi_normalized = 1.0 / (1.0 + metrics.davies_bouldin_score)
            score_components.append(dbi_normalized)
            weights.append(0.15)
        
        # 3. Calinski-Harabasz Index (higher is better, normalize)
        if metrics.calinski_harabasz_score is not None:
            # CH typically ranges from 0 to 1000+, use sigmoid-like normalization
            ch_normalized = np.tanh(metrics.calinski_harabasz_score / 100)
            score_components.append(ch_normalized)
            weights.append(0.15)
        
        # 4. CV ratio (higher between/lower within is better)
        if metrics.within_regime_cv is not None and metrics.between_regime_cv is not None:
            # Ideal: low within, high between
            # Ratio of between/within, normalized
            cv_ratio = metrics.between_regime_cv / (metrics.within_regime_cv + 1e-8)
            cv_normalized = np.tanh(cv_ratio)  # Sigmoid-like normalization
            score_components.append(cv_normalized)
            weights.append(0.15)
        
        # 5. Balance score (already in [0, 1])
        if metrics.balance_score is not None:
            score_components.append(metrics.balance_score)
            weights.append(0.15)
        
        # 6. Temporal smoothness (already in [0, 1])
        if metrics.temporal_smoothness is not None:
            score_components.append(metrics.temporal_smoothness)
            weights.append(0.10)
        
        # 7. Noise ratio (lower is better, invert)
        noise_score = 1.0 - metrics.noise_ratio
        score_components.append(noise_score)
        weights.append(0.10)
        
        # Calculate weighted average
        if len(score_components) > 0:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(score_components, weights)) / total_weight
            return float(weighted_score)
        
        return 0.0
    
    def save_metrics(self, metrics: ClusterQualityMetrics, artifact_name: str = "cluster_quality_metrics"):
        """
        Save quality metrics using artifact manager.
        
        Args:
            metrics: ClusterQualityMetrics object
            artifact_name: Name for the artifact
        """
        if self.artifact_manager is None:
            tprint_warning("⚠️ No artifact manager available - cannot save metrics")
            return
        
        try:
            metrics_dict = metrics.to_dict()
            tprint_data_preview(metrics_dict, "Cluster Quality Metrics")
            
            self.artifact_manager.save(
                data=metrics_dict,
                artifact_name=artifact_name,
                artifact_type="data",
                compression="auto"
            )
            tprint_success(f"💾 Saved cluster quality metrics: {artifact_name}")
        except Exception as e:
            tprint_error(f"❌ Failed to save cluster quality metrics: {e}")
    
    def load_metrics(self, artifact_name: str = "cluster_quality_metrics") -> Optional[ClusterQualityMetrics]:
        """
        Load quality metrics from artifact manager.
        
        Args:
            artifact_name: Name of the artifact
            
        Returns:
            ClusterQualityMetrics object or None if not found
        """
        if self.artifact_manager is None:
            tprint_warning("⚠️ No artifact manager available - cannot load metrics")
            return None
        
        try:
            metrics_dict = self.artifact_manager.get_artifact(
                artifact_name=artifact_name,
                artifact_type="data"
            )
            
            if metrics_dict is None:
                return None
            
            tprint_data_preview(metrics_dict, "Loaded Cluster Quality Metrics")
            
            # Reconstruct ClusterQualityMetrics from dict
            return ClusterQualityMetrics(**metrics_dict)
            
        except Exception as e:
            tprint_error(f"❌ Failed to load cluster quality metrics: {e}")
            return None
    
    def generate_markdown_report(self, metrics: ClusterQualityMetrics, 
                                 symbol: str = "UNKNOWN", 
                                 output_dir: str = "outcomes") -> Optional[str]:
        """
        Generate a comprehensive markdown report of cluster quality metrics.
        
        Args:
            metrics: ClusterQualityMetrics object
            symbol: Trading symbol or identifier
            output_dir: Output directory for the report (default: outcomes/)
            
        Returns:
            Path to the generated report file, or None if failed
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cluster_quality_report_{symbol}_{timestamp}.md"
            report_path = output_path / filename
            
            tprint_info(f"📝 Generating markdown report: {report_path}")
            
            # Build markdown content
            md_content = self._build_markdown_content(metrics, symbol)
            
            # Write to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            tprint_success(f"✅ Report generated successfully: {report_path}")
            return str(report_path)
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate markdown report: {e}")
            return None
    
    def _build_markdown_content(self, metrics: ClusterQualityMetrics, symbol: str) -> str:
        """Build the markdown content for the report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# Cluster Quality Assessment Report

**Symbol:** {symbol}  
**Generated:** {timestamp}  
**Quality Score:** {metrics.quality_score:.4f if metrics.quality_score else 'N/A'}

---

## Executive Summary

This report provides a comprehensive assessment of cluster quality for {symbol}.

### Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Number of Regimes** | {metrics.n_regimes} | {'✅' if metrics.n_regimes >= 2 else '⚠️'} |
| **Noise Ratio** | {metrics.noise_ratio:.2%} | {'✅' if metrics.noise_ratio < 0.3 else '⚠️'} |
| **Silhouette Score** | {metrics.silhouette_score:.4f if metrics.silhouette_score else 'N/A'} | {'✅' if metrics.silhouette_score and metrics.silhouette_score > 0.3 else '⚠️'} |
| **Davies-Bouldin Index** | {metrics.davies_bouldin_score:.4f if metrics.davies_bouldin_score else 'N/A'} | {'✅' if metrics.davies_bouldin_score and metrics.davies_bouldin_score < 2.0 else '⚠️'} |
| **Calinski-Harabasz Index** | {metrics.calinski_harabasz_score:.2f if metrics.calinski_harabasz_score else 'N/A'} | {'✅' if metrics.calinski_harabasz_score and metrics.calinski_harabasz_score > 50 else '⚠️'} |
| **Balance Score** | {metrics.balance_score:.4f if metrics.balance_score else 'N/A'} | {'✅' if metrics.balance_score and metrics.balance_score > 0.5 else '⚠️'} |

---

## Clustering Metrics

### Silhouette Analysis
"""
        
        if metrics.silhouette_score is not None:
            md += f"\n**Global Silhouette Score:** {metrics.silhouette_score:.4f}\n\n"
            
            if metrics.silhouette_per_cluster:
                md += "#### Per-Cluster Silhouette Scores\n\n"
                md += "| Cluster | Mean | Std | Min | Max |\n"
                md += "|---------|------|-----|-----|-----|\n"
                
                for cluster_id, scores in sorted(metrics.silhouette_per_cluster.items()):
                    md += f"| {cluster_id} | {scores['mean']:.4f} | {scores['std']:.4f} | {scores['min']:.4f} | {scores['max']:.4f} |\n"
                md += "\n"
        
        md += f"""
### Separation Metrics

- **Davies-Bouldin Index:** {metrics.davies_bouldin_score:.4f if metrics.davies_bouldin_score else 'N/A'} (lower is better)
- **Calinski-Harabasz Index:** {metrics.calinski_harabasz_score:.2f if metrics.calinski_harabasz_score else 'N/A'} (higher is better)

### Coefficient of Variation

- **Within-Regime CV:** {metrics.within_regime_cv:.4f if metrics.within_regime_cv else 'N/A'} ± {metrics.within_regime_cv_std:.4f if metrics.within_regime_cv_std else 'N/A'}
- **Between-Regime CV:** {metrics.between_regime_cv:.4f if metrics.between_regime_cv else 'N/A'} ± {metrics.between_regime_cv_std:.4f if metrics.between_regime_cv_std else 'N/A'}
"""
        
        # Add per-regime CV if available
        if metrics.per_regime_cv:
            md += "\n#### Per-Regime CV Values\n\n"
            md += "| Regime | CV |\n"
            md += "|--------|----|\n"
            for regime_id, cv in sorted(metrics.per_regime_cv.items()):
                md += f"| {regime_id} | {cv:.4f} |\n"
            md += "\n"
        
        md += """
---

## Balance and Distribution

"""
        
        if metrics.balance_score is not None:
            md += f"**Balance Score:** {metrics.balance_score:.4f} (0-1, higher is better)\n\n"
            md += f"- **Smallest Cluster:** {metrics.min_cluster_size_pct:.2f}% of total\n"
            md += f"- **Largest Cluster:** {metrics.max_cluster_size_pct:.2f}% of total\n"
            md += f"- **Cluster Size Std Dev:** {metrics.cluster_size_std:.2f}\n\n"
            
            if metrics.cluster_size_distribution:
                md += "### Cluster Size Distribution\n\n"
                md += "| Cluster Index | Size (%) |\n"
                md += "|---------------|----------|\n"
                for i, size_pct in enumerate(metrics.cluster_size_distribution):
                    md += f"| {i} | {size_pct:.2f}% |\n"
                md += "\n"
        
        # Temporal metrics
        if metrics.temporal_smoothness is not None:
            md += f"""
---

## Temporal Analysis

- **Temporal Smoothness:** {metrics.temporal_smoothness:.4f} (0-1, higher = fewer transitions)
- **Regime Persistence:** {metrics.regime_persistence:.2f} bars (average duration)

"""
        
        # Per-regime metrics
        if metrics.per_regime_metrics:
            md += """
---

## Per-Regime Analysis

"""
            for regime_id, regime_data in sorted(metrics.per_regime_metrics.items()):
                regime_type = regime_data.get('regime_type', 'unknown')
                md += f"""
### Regime {regime_id} ({regime_type})

**Size:** {regime_data.get('size', 'N/A')} samples ({regime_data.get('percentage', 0):.2f}%)

"""
                
                if 'mean_return' in regime_data:
                    md += f"""
**Performance Metrics:**
- Mean Return: {regime_data['mean_return']:.4f}
- Volatility: {regime_data['volatility']:.4f}
- Sharpe Ratio: {regime_data['sharpe']:.4f}
- Skewness: {regime_data.get('skewness', 'N/A')}
- Max Drawdown: {regime_data.get('max_drawdown', 'N/A')}

"""
                
                if 'regime_specific_metrics' in regime_data and regime_data['regime_specific_metrics']:
                    md += "**Regime-Specific Characteristics:**\n\n"
                    for key, value in regime_data['regime_specific_metrics'].items():
                        md += f"- {key}: {value}\n"
                    md += "\n"
        
        # Economic interpretation
        if metrics.economic_interpretation:
            md += """
---

## Economic Interpretation

"""
            interp = metrics.economic_interpretation
            
            if 'regime_summary' in interp:
                summary = interp['regime_summary']
                md += f"""
### Regime Summary

- **Total Regimes:** {summary.get('total_regimes', 'N/A')}
- **Dominant Regime:** {summary.get('dominant_regime', 'N/A')}

"""
                if 'regime_type_distribution' in summary:
                    md += "**Regime Type Distribution:**\n\n"
                    for regime_type, count in summary['regime_type_distribution'].items():
                        md += f"- {regime_type}: {count}\n"
                    md += "\n"
            
            if 'trading_implications' in interp:
                implications = interp['trading_implications']
                md += "\n### Trading Implications\n\n"
                
                if 'most_profitable_regime' in implications:
                    best = implications['most_profitable_regime']
                    md += f"""
**Most Profitable Regime:** {best.get('regime_id', 'N/A')} ({best.get('regime_type', 'N/A')})
- Sharpe Ratio: {best.get('sharpe_ratio', 'N/A')}
- Mean Return: {best.get('mean_return', 'N/A')}
- Volatility: {best.get('volatility', 'N/A')}

"""
                
                if 'strategy_recommendations' in implications:
                    md += "**Strategy Recommendations:**\n\n"
                    for rec in implications['strategy_recommendations']:
                        md += f"- {rec.get('strategy', 'N/A')}: Target Regime {rec.get('target_regime', 'N/A')}\n"
                    md += "\n"
        
        # Predictive power
        if metrics.predictive_power is not None:
            md += f"""
---

## Predictive Power

**Cross-Validation Score:** {metrics.predictive_power:.4f}

This score indicates how well the current regime can predict future return direction.
"""
        
        # Quality assessment
        md += f"""
---

## Quality Assessment

**Overall Quality Score:** {metrics.quality_score:.4f if metrics.quality_score else 'N/A'} / 1.0

"""
        
        # Determine quality level
        if metrics.quality_score:
            if metrics.quality_score >= 0.7:
                quality_level = "Excellent ✅"
                recommendation = "The clustering shows excellent quality. Proceed with confidence."
            elif metrics.quality_score >= 0.5:
                quality_level = "Good ✅"
                recommendation = "The clustering shows good quality. Suitable for most applications."
            elif metrics.quality_score >= 0.3:
                quality_level = "Moderate ⚠️"
                recommendation = "The clustering shows moderate quality. Consider parameter tuning."
            else:
                quality_level = "Poor ❌"
                recommendation = "The clustering shows poor quality. Parameter adjustment recommended."
            
            md += f"""
**Quality Level:** {quality_level}

**Recommendation:** {recommendation}

"""
        
        md += f"""
---

## Report Metadata

- **Generated by:** ClusterQualityAssessor
- **Timestamp:** {metrics.timestamp}
- **Report Version:** 1.0

"""
        
        return md


def create_cluster_quality_assessor(artifact_manager=None, 
                                   enable_hardware_optimization=True,
                                   enable_vectorization=True) -> ClusterQualityAssessor:
    """
    Factory function to create a cluster quality assessor.
    
    Args:
        artifact_manager: Optional artifact manager from BaseStep
        enable_hardware_optimization: Enable hardware optimizations
        enable_vectorization: Enable vectorized computations
        
    Returns:
        ClusterQualityAssessor instance
    """
    return ClusterQualityAssessor(
        artifact_manager=artifact_manager,
        enable_hardware_optimization=enable_hardware_optimization,
        enable_vectorization=enable_vectorization
    )
