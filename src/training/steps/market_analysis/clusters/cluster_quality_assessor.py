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

# Import sklearn metrics
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


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
    
    # Economic validation
    economic_validation: Dict[str, Any] = field(default_factory=dict)
    
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
            'economic_validation': self.economic_validation,
            
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
    
    def __init__(self, artifact_manager=None):
        """
        Initialize the cluster quality assessor.
        
        Args:
            artifact_manager: Optional artifact manager from BaseStep
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.artifact_manager = artifact_manager
    
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
        self.logger.info("Starting comprehensive cluster quality assessment")
        
        # Initialize metrics object
        metrics = ClusterQualityMetrics()
        
        # Validate inputs
        if len(regime_labels) == 0 or feature_data.empty:
            self.logger.warning("Empty inputs - cannot assess quality")
            return metrics
        
        # Filter out noise points for core metrics
        non_noise_mask = regime_labels != -1
        
        if np.sum(non_noise_mask) < min_regime_size:
            self.logger.warning(f"Insufficient non-noise points ({np.sum(non_noise_mask)}) for quality assessment")
            return metrics
        
        # Get clean numeric features
        features_clean = feature_data.select_dtypes(include=[np.number])
        if features_clean.empty:
            self.logger.warning("No numeric features available for quality assessment")
            return metrics
        
        # Calculate basic statistics
        metrics.n_regimes = len(set(regime_labels[non_noise_mask]))
        metrics.noise_ratio = np.sum(~non_noise_mask) / len(regime_labels)
        
        self.logger.info(f"Assessing quality for {metrics.n_regimes} regimes with {metrics.noise_ratio:.1%} noise")
        
        # 1. Silhouette scores
        try:
            metrics.silhouette_score, metrics.silhouette_per_cluster = self._calculate_silhouette_scores(
                regime_labels, features_clean, non_noise_mask
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate silhouette scores: {e}")
        
        # 2. Davies-Bouldin Index
        try:
            metrics.davies_bouldin_score = self._calculate_dbi(
                regime_labels, features_clean, non_noise_mask
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate DBI: {e}")
        
        # 3. Calinski-Harabasz Index
        try:
            metrics.calinski_harabasz_score = self._calculate_ch(
                regime_labels, features_clean, non_noise_mask
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate CH score: {e}")
        
        # 4. Coefficient of variation metrics (with std dev and per-regime)
        try:
            (metrics.within_regime_cv, metrics.within_regime_cv_std,
             metrics.between_regime_cv, metrics.between_regime_cv_std,
             metrics.per_regime_cv) = self._calculate_cv_metrics(
                regime_labels, features_clean, non_noise_mask
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate CV metrics: {e}")
        
        # 5. Balance metrics
        try:
            (metrics.balance_score, metrics.min_cluster_size_pct,
             metrics.max_cluster_size_pct, metrics.cluster_size_std,
             metrics.cluster_size_distribution) = self._calculate_balance_metrics(regime_labels)
        except Exception as e:
            self.logger.warning(f"Failed to calculate balance metrics: {e}")
        
        # 6. Temporal smoothness and persistence
        if timestamps is not None:
            try:
                metrics.temporal_smoothness = self._calculate_temporal_smoothness(
                    regime_labels, timestamps
                )
                metrics.regime_persistence = self._calculate_regime_persistence(regime_labels)
            except Exception as e:
                self.logger.warning(f"Failed to calculate temporal metrics: {e}")
        
        # 7. Per-regime metrics
        try:
            metrics.per_regime_metrics = self._calculate_per_regime_metrics(
                regime_labels, features_clean, forward_returns
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate per-regime metrics: {e}")
        
        # 8. Economic validation (if forward returns provided)
        if forward_returns is not None:
            try:
                metrics.economic_validation = self._validate_regime_quality(
                    regime_labels, forward_returns, feature_data
                )
            except Exception as e:
                self.logger.warning(f"Failed to validate regime quality: {e}")
        
        # 9. Predictive power
        if forward_returns is not None and len(forward_returns) > 0:
            try:
                metrics.predictive_power = self._calculate_predictive_power(
                    regime_labels, forward_returns
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate predictive power: {e}")
        
        # 10. Calculate overall quality score
        try:
            metrics.quality_score = self._calculate_quality_score(metrics)
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {e}")
        
        self.logger.info(f"Quality assessment complete - Quality Score: {metrics.quality_score:.3f}")
        
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
    
    def _calculate_per_regime_metrics(self,
                                      regime_labels: np.ndarray,
                                      features: pd.DataFrame,
                                      forward_returns: Optional[pd.Series] = None) -> Dict[int, Dict[str, Any]]:
        """Calculate detailed metrics for each regime."""
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
            
            # Add return characteristics if available
            if forward_returns is not None and len(forward_returns) > 0:
                regime_returns = forward_returns[regime_mask]
                if len(regime_returns) > 0:
                    regime_metrics.update({
                        'mean_return': float(regime_returns.mean()),
                        'volatility': float(regime_returns.std()),
                        'sharpe': float(regime_returns.mean() / (regime_returns.std() + 1e-8)),
                        'skewness': float(regime_returns.skew()) if hasattr(regime_returns, 'skew') else 0.0,
                        'max_drawdown': float(self._compute_max_drawdown(regime_returns))
                    })
            
            per_regime_metrics[int(regime_id)] = regime_metrics
        
        return per_regime_metrics
    
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
            self.logger.warning("No artifact manager available - cannot save metrics")
            return
        
        try:
            metrics_dict = metrics.to_dict()
            self.artifact_manager.save(
                data=metrics_dict,
                artifact_name=artifact_name,
                artifact_type="data",
                compression="auto"
            )
            self.logger.info(f"Saved cluster quality metrics: {artifact_name}")
        except Exception as e:
            self.logger.error(f"Failed to save cluster quality metrics: {e}")
    
    def load_metrics(self, artifact_name: str = "cluster_quality_metrics") -> Optional[ClusterQualityMetrics]:
        """
        Load quality metrics from artifact manager.
        
        Args:
            artifact_name: Name of the artifact
            
        Returns:
            ClusterQualityMetrics object or None if not found
        """
        if self.artifact_manager is None:
            self.logger.warning("No artifact manager available - cannot load metrics")
            return None
        
        try:
            metrics_dict = self.artifact_manager.get_artifact(
                artifact_name=artifact_name,
                artifact_type="data"
            )
            
            if metrics_dict is None:
                return None
            
            # Reconstruct ClusterQualityMetrics from dict
            return ClusterQualityMetrics(**metrics_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to load cluster quality metrics: {e}")
            return None


def create_cluster_quality_assessor(artifact_manager=None) -> ClusterQualityAssessor:
    """
    Factory function to create a cluster quality assessor.
    
    Args:
        artifact_manager: Optional artifact manager from BaseStep
        
    Returns:
        ClusterQualityAssessor instance
    """
    return ClusterQualityAssessor(artifact_manager=artifact_manager)
