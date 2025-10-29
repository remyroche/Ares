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
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, fields
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


class QualityThresholds:
    """Configuration constants for quality assessment thresholds."""
    # Core clustering quality thresholds
    MIN_SILHOUETTE = 0.3
    MAX_DBI = 2.0
    MIN_CH = 50.0
    MAX_NOISE_RATIO = 0.3
    
    # Quality score thresholds
    QUALITY_EXCELLENT = 0.7
    QUALITY_GOOD = 0.5
    QUALITY_MODERATE = 0.3
    
    # Regime type detection thresholds (Crypto-optimized)
    HIGH_VOLATILITY_THRESHOLD = 0.05  # 5% daily volatility threshold (crypto: Bitcoin/ALTs commonly swing 3–10% daily)
    VOLATILITY_CLUSTERING_THRESHOLD = 0.45  # Crypto volatility clusters tightly & sharply
    TREND_STRENGTH_THRESHOLD = 0.65  # Crypto trends are noisy → require stronger conviction
    TREND_PERSISTENCE_THRESHOLD = 0.35  # Trend needs to hold at least a few candles
    MEAN_REVERSION_THRESHOLD = -0.05  # Mean reversion exists but is much weaker in crypto
    LOW_VOLATILITY_THRESHOLD = 0.025  # 2.5% - Crypto rarely sits at equity-like low vol levels
    LOW_TREND_THRESHOLD = 0.3
    VOLATILITY_SCALE_FACTOR = 10.0  # For volatility comparison scaling
    
    # Economic validation thresholds
    MIN_SHARPE_FOR_STRATEGY = 0.5
    HIGH_DRAWDOWN_THRESHOLD = -0.15
    NEGATIVE_SHARPE_THRESHOLD = -0.5
    
    # Quality score weights
    WEIGHT_SILHOUETTE = 0.20
    WEIGHT_DBI = 0.15
    WEIGHT_CH = 0.15
    WEIGHT_CV_RATIO = 0.15
    WEIGHT_BALANCE = 0.15
    WEIGHT_TEMPORAL_SMOOTHNESS = 0.10
    WEIGHT_NOISE_RATIO = 0.10
    
    # Normalization constants
    CH_NORMALIZATION_DIVISOR = 100.0
    DBI_EPSILON = 1e-8  # Small epsilon for safe division
    
    # Minimum requirements
    MIN_SAMPLES_FOR_PREDICTIVE_POWER = 10
    MIN_SAMPLES_PER_CV_FOLD = 3
    MAX_CV_FOLDS = 5


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
    
    # ENHANCEMENT: I. PREDICTIVE/GENERALIZATION CHECKS
    rolling_predictive_ll: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Rolling log-likelihood on holdout blocks
    one_step_ahead_scores: Optional[np.ndarray] = None  # One-step-ahead predictive densities
    baseline_comparison: Optional[Dict[str, float]] = field(default_factory=dict)  # vs AR(1), constant vol
    delta_ll_across_folds: Optional[List[float]] = None  # ΔLL across folds
    predictive_ll_effect_size: Optional[float] = None  # Effect size vs noise
    
    # DIAGNOSTIC: Median & IQR of predictive LL
    predictive_ll_median: Optional[float] = None
    predictive_ll_iqr: Optional[float] = None  # Interquartile range
    predictive_ll_q25: Optional[float] = None
    predictive_ll_q75: Optional[float] = None
    
    # ENHANCEMENT: II. STABILITY & REPRODUCIBILITY
    refit_stability_ari: Optional[float] = None  # Adjusted Rand Index across refits
    refit_stability_nmi: Optional[float] = None  # Normalized Mutual Information
    refit_stability_median: Optional[float] = None  # Median ARI across runs
    subsample_stability: Optional[Dict[str, float]] = field(default_factory=dict)  # Stability across windows
    transition_matrix_stability: Optional[float] = None  # Transition matrix similarity
    
    # DIAGNOSTIC: ARI across restarts (detailed)
    ari_across_restarts: Optional[List[float]] = None  # All ARI values
    ari_median: Optional[float] = None
    ari_iqr: Optional[float] = None
    ari_q25: Optional[float] = None
    ari_q75: Optional[float] = None
    
    # ENHANCEMENT: III. REGIME OCCUPANCY & PERSISTENCE
    state_occupancy: Optional[Dict[int, float]] = field(default_factory=dict)  # Fraction of time in each state
    tiny_state_count: Optional[int] = None  # States with < 1% occupancy
    expected_state_durations: Optional[Dict[int, float]] = field(default_factory=dict)  # E[D] = 1/(1-p_ii)
    min_expected_duration: Optional[float] = None  # Minimum expected duration
    max_expected_duration: Optional[float] = None  # Maximum expected duration
    duration_quality_flag: Optional[str] = None  # 'good', 'warning', 'poor'
    
    # DIAGNOSTIC: State occupancy distribution (detailed)
    occupancy_distribution: Optional[List[float]] = None  # Sorted occupancies
    occupancy_entropy: Optional[float] = None  # Shannon entropy of distribution
    min_occupancy_pct: Optional[float] = None  # Minimum state occupancy %
    max_occupancy_pct: Optional[float] = None  # Maximum state occupancy %
    
    # ENHANCEMENT: IV. TRANSITION MATRIX SENSIBILITY
    transition_matrix_checks: Optional[Dict[str, Any]] = field(default_factory=dict)
    unrealistic_oscillation_detected: Optional[bool] = None
    transition_interpretability_score: Optional[float] = None  # 0-1, higher = more interpretable
    
    # ENHANCEMENT: V. EMISSION/GEOMETRIC DIAGNOSTICS
    state_conditioned_stats: Optional[Dict[int, Dict[str, float]]] = field(default_factory=dict)
    # mean, std, skew, kurtosis per state
    emission_distinctiveness: Optional[float] = None  # How distinct are emissions?
    umap_separation_score: Optional[float] = None  # Visual separation score
    
    # ENHANCEMENT: VI. POSTERIOR PREDICTIVE CHECKS
    simulated_vs_empirical_moments: Optional[Dict[str, float]] = field(default_factory=dict)
    # return distribution, autocorr, vol clustering, cross-feature corr
    probability_calibration_score: Optional[float] = None  # PIT/CRPS calibration
    pit_histogram_uniformity: Optional[float] = None  # Kolmogorov-Smirnov test
    predictive_density_calibration: Optional[str] = None  # 'well_calibrated', 'too_narrow', 'too_wide'
    
    # DIAGNOSTIC: CRPS and PIT (detailed)
    crps_score: Optional[float] = None  # Continuous Ranked Probability Score
    pit_values: Optional[np.ndarray] = None  # Probability Integral Transform values
    pit_uniformity_pvalue: Optional[float] = None  # KS test p-value
    
    # DIAGNOSTIC: Tail quantiles (simulated vs empirical)
    tail_quantile_comparison: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Contains: q01_empirical, q01_simulated, q05, q95, q99, etc.
    tail_coverage_score: Optional[float] = None  # How well tails are captured (0-1)
    
    # ENHANCEMENT: VII. ECONOMIC UTILITY & ROBUSTNESS
    out_of_sample_sharpe: Optional[float] = None
    out_of_sample_max_drawdown: Optional[float] = None
    strategy_turnover: Optional[float] = None
    transaction_cost_robustness: Optional[Dict[str, float]] = field(default_factory=dict)
    bootstrap_significance: Optional[Dict[str, Any]] = field(default_factory=dict)
    sharpe_uplift_vs_baseline: Optional[float] = None
    economic_utility_score: Optional[float] = None  # Composite economic metric
    
    # DIAGNOSTIC: Median & IQR of economic metrics
    sharpe_across_folds: Optional[List[float]] = None  # Sharpe per fold
    sharpe_median: Optional[float] = None
    sharpe_iqr: Optional[float] = None
    sharpe_q25: Optional[float] = None
    sharpe_q75: Optional[float] = None
    
    turnover_across_folds: Optional[List[float]] = None  # Turnover per fold
    turnover_median: Optional[float] = None
    turnover_iqr: Optional[float] = None
    turnover_q25: Optional[float] = None
    turnover_q75: Optional[float] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @staticmethod
    def _safe_array_to_list(arr: Optional[Union[np.ndarray, List[Any]]]) -> Optional[List[Any]]:
        """
        Safely convert numpy array or list to list for serialization.
        
        Args:
            arr: numpy array, list, or None
            
        Returns:
            List representation or None
        """
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, (list, tuple)):
            try:
                return list(arr)
            except (TypeError, ValueError):
                return None
        return None
    
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
            
            # ENHANCEMENT: I. Predictive/Generalization
            'rolling_predictive_ll': self.rolling_predictive_ll,
            'one_step_ahead_scores': self._safe_array_to_list(self.one_step_ahead_scores),
            'baseline_comparison': self.baseline_comparison,
            'delta_ll_across_folds': self.delta_ll_across_folds,
            'predictive_ll_effect_size': self.predictive_ll_effect_size,
            'predictive_ll_median': self.predictive_ll_median,
            'predictive_ll_iqr': self.predictive_ll_iqr,
            'predictive_ll_q25': self.predictive_ll_q25,
            'predictive_ll_q75': self.predictive_ll_q75,
            
            # ENHANCEMENT: II. Stability & Reproducibility
            'refit_stability_ari': self.refit_stability_ari,
            'refit_stability_nmi': self.refit_stability_nmi,
            'refit_stability_median': self.refit_stability_median,
            'subsample_stability': self.subsample_stability,
            'transition_matrix_stability': self.transition_matrix_stability,
            'ari_across_restarts': self.ari_across_restarts,
            'ari_median': self.ari_median,
            'ari_iqr': self.ari_iqr,
            'ari_q25': self.ari_q25,
            'ari_q75': self.ari_q75,
            
            # ENHANCEMENT: III. Regime Occupancy & Persistence
            'state_occupancy': self.state_occupancy,
            'tiny_state_count': self.tiny_state_count,
            'expected_state_durations': self.expected_state_durations,
            'min_expected_duration': self.min_expected_duration,
            'max_expected_duration': self.max_expected_duration,
            'duration_quality_flag': self.duration_quality_flag,
            'occupancy_distribution': self.occupancy_distribution,
            'occupancy_entropy': self.occupancy_entropy,
            'min_occupancy_pct': self.min_occupancy_pct,
            'max_occupancy_pct': self.max_occupancy_pct,
            
            # ENHANCEMENT: IV. Transition Matrix Sensibility
            'transition_matrix_checks': self.transition_matrix_checks,
            'unrealistic_oscillation_detected': self.unrealistic_oscillation_detected,
            'transition_interpretability_score': self.transition_interpretability_score,
            
            # ENHANCEMENT: V. Emission/Geometric Diagnostics
            'state_conditioned_stats': self.state_conditioned_stats,
            'emission_distinctiveness': self.emission_distinctiveness,
            'umap_separation_score': self.umap_separation_score,
            
            # ENHANCEMENT: VI. Posterior Predictive Checks
            'simulated_vs_empirical_moments': self.simulated_vs_empirical_moments,
            'probability_calibration_score': self.probability_calibration_score,
            'pit_histogram_uniformity': self.pit_histogram_uniformity,
            'predictive_density_calibration': self.predictive_density_calibration,
            'crps_score': self.crps_score,
            'pit_values': self._safe_array_to_list(self.pit_values),
            'pit_uniformity_pvalue': self.pit_uniformity_pvalue,
            'tail_quantile_comparison': self.tail_quantile_comparison,
            'tail_coverage_score': self.tail_coverage_score,
            
            # ENHANCEMENT: VII. Economic Utility & Robustness
            'out_of_sample_sharpe': self.out_of_sample_sharpe,
            'out_of_sample_max_drawdown': self.out_of_sample_max_drawdown,
            'strategy_turnover': self.strategy_turnover,
            'transaction_cost_robustness': self.transaction_cost_robustness,
            'bootstrap_significance': self.bootstrap_significance,
            'sharpe_uplift_vs_baseline': self.sharpe_uplift_vs_baseline,
            'economic_utility_score': self.economic_utility_score,
            'sharpe_across_folds': self.sharpe_across_folds,
            'sharpe_median': self.sharpe_median,
            'sharpe_iqr': self.sharpe_iqr,
            'sharpe_q25': self.sharpe_q25,
            'sharpe_q75': self.sharpe_q75,
            'turnover_across_folds': self.turnover_across_folds,
            'turnover_median': self.turnover_median,
            'turnover_iqr': self.turnover_iqr,
            'turnover_q25': self.turnover_q25,
            'turnover_q75': self.turnover_q75,
            
            # Metadata
            'timestamp': self.timestamp
        }
    
    def is_high_quality(self, 
                        min_silhouette: Optional[float] = None,
                        max_dbi: Optional[float] = None,
                        min_ch: Optional[float] = None,
                        max_noise: Optional[float] = None) -> bool:
        """
        Check if clustering quality meets minimum thresholds.
        
        Args:
            min_silhouette: Minimum silhouette score (defaults to QualityThresholds.MIN_SILHOUETTE)
            max_dbi: Maximum Davies-Bouldin Index (defaults to QualityThresholds.MAX_DBI)
            min_ch: Minimum Calinski-Harabasz score (defaults to QualityThresholds.MIN_CH)
            max_noise: Maximum noise ratio (defaults to QualityThresholds.MAX_NOISE_RATIO)
            
        Returns:
            True if quality meets all thresholds
        """
        # Use defaults from QualityThresholds if not provided
        min_silhouette = min_silhouette if min_silhouette is not None else QualityThresholds.MIN_SILHOUETTE
        max_dbi = max_dbi if max_dbi is not None else QualityThresholds.MAX_DBI
        min_ch = min_ch if min_ch is not None else QualityThresholds.MIN_CH
        max_noise = max_noise if max_noise is not None else QualityThresholds.MAX_NOISE_RATIO
        
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
    
    def _ensure_aligned_data(self,
                             regime_labels: np.ndarray,
                             feature_data: pd.DataFrame,
                             forward_returns: Optional[pd.Series] = None) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.Series]]:
        """
        Ensure regime_labels, feature_data, and forward_returns are properly aligned.
        
        This method validates lengths and handles index misalignment by converting
        to positional indexing if indices don't match.
        
        Args:
            regime_labels: Regime/cluster labels array
            feature_data: Feature DataFrame
            forward_returns: Optional forward returns Series
            
        Returns:
            Tuple of (aligned_regime_labels, aligned_feature_data, aligned_forward_returns)
        """
        # Validate lengths
        if len(regime_labels) != len(feature_data):
            self.logger.warning(
                f"Length mismatch: regime_labels ({len(regime_labels)}) vs "
                f"feature_data ({len(feature_data)}). Using positional indexing."
            )
            # Reset indices to ensure positional alignment
            feature_data = feature_data.reset_index(drop=True)
        
        # Handle forward_returns alignment
        aligned_forward_returns = None
        if forward_returns is not None:
            if len(regime_labels) != len(forward_returns):
                self.logger.warning(
                    f"Length mismatch: regime_labels ({len(regime_labels)}) vs "
                    f"forward_returns ({len(forward_returns)}). Using positional indexing."
                )
                forward_returns = forward_returns.reset_index(drop=True)
            
            # Check if indices match (if both have indices)
            if hasattr(forward_returns, 'index') and hasattr(feature_data, 'index'):
                if not forward_returns.index.equals(feature_data.index):
                    self.logger.warning(
                        "Index mismatch between forward_returns and feature_data. "
                        "Resetting indices for alignment."
                    )
                    forward_returns = forward_returns.reset_index(drop=True)
                    feature_data = feature_data.reset_index(drop=True)
            
            aligned_forward_returns = forward_returns
        
        return regime_labels, feature_data, aligned_forward_returns
    
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
    
    def assess_hmm_regime_quality(
        self,
        regime_labels: np.ndarray,
        feature_data: pd.DataFrame,
        transition_matrix: Optional[np.ndarray] = None,
        hmm_model: Optional[Any] = None,
        forward_returns: Optional[pd.Series] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        timeframe: str = "1h",
        min_regime_size: int = 10,
        run_validators: bool = True
    ) -> ClusterQualityMetrics:
        """
        ENHANCED cluster quality assessment with HMM-specific validators.
        
        This method extends assess_quality() with:
        I. Predictive/generalization checks
        II. Stability & reproducibility
        III. Regime occupancy & persistence
        IV. Transition matrix sensibility
        V. Emission/geometric diagnostics
        VI. Posterior predictive checks
        VII. Economic utility & robustness
        
        Args:
            regime_labels: Cluster/regime assignments
            feature_data: Feature DataFrame
            transition_matrix: HMM transition matrix (if available)
            hmm_model: Fitted HMM model (if available)
            forward_returns: Forward returns for economic validation
            timestamps: Timestamps for temporal analysis
            timeframe: Timeframe string (e.g., '1h', '1d')
            min_regime_size: Minimum regime size
            run_validators: Whether to run comprehensive validators
            
        Returns:
            ClusterQualityMetrics with all enhanced metrics
        """
        tprint_info("🔍 Starting ENHANCED HMM regime quality assessment")
        
        # First, run standard quality assessment
        metrics = self.assess_quality(
            regime_labels=regime_labels,
            feature_data=feature_data,
            forward_returns=forward_returns,
            timestamps=timestamps,
            min_regime_size=min_regime_size
        )
        
        # If validators disabled, return standard metrics
        if not run_validators:
            return metrics
        
        # Initialize HMM validator
        try:
            from .hmm_regime_validators import create_hmm_regime_validator
            validator = create_hmm_regime_validator(timeframe=timeframe)
            tprint_success("✅ HMM regime validator initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Could not initialize HMM validator: {e}")
            return metrics
        
        tprint_info("="*70)
        tprint_info("🔬 Running COMPREHENSIVE HMM regime validators...")
        tprint_info("="*70)
        
        # Prepare data
        data_array = feature_data.select_dtypes(include=[np.number]).values
        
        # III. REGIME OCCUPANCY & PERSISTENCE
        try:
            occupancy_results = validator.regime_occupancy_persistence_validation(
                labels=regime_labels,
                transition_matrix=transition_matrix
            )
            metrics.state_occupancy = occupancy_results.get('state_occupancy', {})
            metrics.tiny_state_count = occupancy_results.get('tiny_state_count', 0)
            metrics.expected_state_durations = occupancy_results.get('expected_durations', {})
            metrics.min_expected_duration = occupancy_results.get('min_expected_duration_days')
            metrics.max_expected_duration = occupancy_results.get('max_expected_duration_days')
            metrics.duration_quality_flag = occupancy_results.get('duration_quality_flag', 'unknown')
            # DIAGNOSTIC: Occupancy distribution
            metrics.occupancy_distribution = occupancy_results.get('occupancy_distribution', [])
            metrics.occupancy_entropy = occupancy_results.get('occupancy_entropy')
            metrics.min_occupancy_pct = occupancy_results.get('min_occupancy_pct')
            metrics.max_occupancy_pct = occupancy_results.get('max_occupancy_pct')
        except Exception as e:
            tprint_error(f"❌ Occupancy validation failed: {e}")
        
        # IV. TRANSITION MATRIX SENSIBILITY
        if transition_matrix is not None:
            try:
                transition_results = validator.transition_matrix_validation(
                    transition_matrix=transition_matrix,
                    labels=regime_labels
                )
                metrics.transition_matrix_checks = transition_results
                metrics.unrealistic_oscillation_detected = transition_results.get('unrealistic_oscillation', False)
                metrics.transition_interpretability_score = transition_results.get('interpretability_score', 0.0)
            except Exception as e:
                tprint_error(f"❌ Transition matrix validation failed: {e}")
        
        # V. EMISSION/GEOMETRIC DIAGNOSTICS
        try:
            emission_results = validator.emission_diagnostics(
                data=feature_data,
                labels=regime_labels
            )
            metrics.state_conditioned_stats = emission_results.get('state_conditioned_stats', {})
            metrics.emission_distinctiveness = emission_results.get('emission_distinctiveness', 0.0)
        except Exception as e:
            tprint_error(f"❌ Emission diagnostics failed: {e}")
        
        # VII. ECONOMIC UTILITY & ROBUSTNESS
        if forward_returns is not None and len(forward_returns) > 0:
            try:
                economic_results = validator.economic_utility_validation(
                    labels=regime_labels,
                    returns=forward_returns
                )
                metrics.out_of_sample_sharpe = economic_results.get('out_of_sample_sharpe')
                metrics.out_of_sample_max_drawdown = economic_results.get('out_of_sample_max_drawdown')
                metrics.strategy_turnover = economic_results.get('strategy_turnover')
                metrics.sharpe_uplift_vs_baseline = economic_results.get('sharpe_uplift')
                metrics.economic_utility_score = economic_results.get('economic_utility_score')
                metrics.bootstrap_significance = {
                    'sharpe_ci': economic_results.get('bootstrap_sharpe_ci'),
                    'significant': economic_results.get('sharpe_significant')
                }
                # DIAGNOSTIC: Sharpe & Turnover across folds (median & IQR)
                metrics.sharpe_across_folds = economic_results.get('sharpe_across_folds')
                metrics.sharpe_median = economic_results.get('sharpe_median')
                metrics.sharpe_iqr = economic_results.get('sharpe_iqr')
                metrics.sharpe_q25 = economic_results.get('sharpe_q25')
                metrics.sharpe_q75 = economic_results.get('sharpe_q75')
                metrics.turnover_across_folds = economic_results.get('turnover_across_folds')
                metrics.turnover_median = economic_results.get('turnover_median')
                metrics.turnover_iqr = economic_results.get('turnover_iqr')
                metrics.turnover_q25 = economic_results.get('turnover_q25')
                metrics.turnover_q75 = economic_results.get('turnover_q75')
            except Exception as e:
                tprint_error(f"❌ Economic utility validation failed: {e}")
        
        # I. PREDICTIVE/GENERALIZATION (requires model)
        if hmm_model is not None:
            try:
                predictive_results = validator.rolling_predictive_ll_validation(
                    model=hmm_model,
                    data=data_array
                )
                metrics.rolling_predictive_ll = predictive_results
                metrics.delta_ll_across_folds = predictive_results.get('delta_ll_across_folds', [])
                metrics.predictive_ll_effect_size = predictive_results.get('effect_size')
                metrics.baseline_comparison = {
                    'mean_delta_ll': predictive_results.get('mean_delta_ll'),
                    'positive_ratio': predictive_results.get('positive_ratio')
                }
                # DIAGNOSTIC: Median & IQR
                metrics.predictive_ll_median = predictive_results.get('predictive_ll_median')
                metrics.predictive_ll_iqr = predictive_results.get('predictive_ll_iqr')
                metrics.predictive_ll_q25 = predictive_results.get('predictive_ll_q25')
                metrics.predictive_ll_q75 = predictive_results.get('predictive_ll_q75')
            except Exception as e:
                tprint_warning(f"⚠️ Predictive validation skipped: {e}")
        
        # VI. POSTERIOR PREDICTIVE CHECKS (requires model with sampling)
        if hmm_model is not None and hasattr(hmm_model, 'sample'):
            try:
                posterior_results = validator.posterior_predictive_check(
                    model=hmm_model,
                    data=data_array
                )
                metrics.simulated_vs_empirical_moments = posterior_results
                metrics.probability_calibration_score = posterior_results.get('calibration_score')
                metrics.predictive_density_calibration = posterior_results.get('calibration_flag', 'unknown')
                # DIAGNOSTIC: CRPS, PIT, Tail quantiles
                metrics.crps_score = posterior_results.get('crps_score')
                metrics.pit_uniformity_pvalue = posterior_results.get('pit_uniformity_pvalue')
                metrics.tail_quantile_comparison = posterior_results.get('tail_quantile_comparison', {})
                metrics.tail_coverage_score = posterior_results.get('tail_coverage_score')
            except Exception as e:
                tprint_warning(f"⚠️ Posterior predictive check skipped: {e}")
        
        tprint_info("="*70)
        tprint_success("✅ COMPREHENSIVE HMM validation complete!")
        tprint_info("="*70)
        
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
                denominator = np.abs(cluster_mean) + QualityThresholds.DBI_EPSILON
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
                denominator = np.abs(feature_mean) + QualityThresholds.DBI_EPSILON
                cv = feature_std / denominator
                
                if np.isfinite(cv):
                    between_cvs.append(cv)
            
            between_regime_cv_mean = float(np.mean(between_cvs)) if between_cvs else 0.0
            between_regime_cv_std = float(np.std(between_cvs)) if len(between_cvs) > 1 else 0.0
        
        return within_regime_cv_mean, within_regime_cv_std, between_regime_cv_mean, between_regime_cv_std, per_regime_cv
    
    def _calculate_temporal_smoothness(self,
                                       regime_labels: np.ndarray,
                                       timestamps: Optional[pd.DatetimeIndex] = None) -> float:
        """
        Calculate temporal smoothness score.
        
        Higher score means fewer regime transitions (more stable regimes).
        Score is normalized to [0, 1] where 1 is perfectly smooth.
        
        Args:
            regime_labels: Regime/cluster labels
            timestamps: Optional timestamps for time-aware analysis (currently not used but
                       validated for future enhancements)
        
        Returns:
            Temporal smoothness score between 0 and 1
        """
        if len(regime_labels) < 2:
            return 1.0
        
        # Validate alignment if timestamps provided
        if timestamps is not None and len(timestamps) != len(regime_labels):
            self.logger.warning(
                f"Timestamps length ({len(timestamps)}) doesn't match "
                f"regime_labels length ({len(regime_labels)}). Using simple smoothness calculation."
            )
        
        # Count regime transitions
        regime_changes = np.sum(regime_labels[1:] != regime_labels[:-1])
        max_possible_changes = len(regime_labels) - 1
        
        if max_possible_changes == 0:
            return 1.0
        
        # Smoothness score: fewer changes = higher smoothness
        smoothness = 1.0 - (regime_changes / max_possible_changes)
        
        return float(smoothness)
    
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
        avg_regime_duration = 1.0 / (np.mean(regime_changes) + QualityThresholds.DBI_EPSILON)
        
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
                trend_strength = abs(mean_return) / (returns_std + QualityThresholds.DBI_EPSILON)
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
                cv = volatility_level / (abs(mean_return) + QualityThresholds.DBI_EPSILON)
                metrics['stability_score'] = float(1.0 / (1.0 + cv))
                
                # Classify regime based on dominant characteristic
                # Use data-driven thresholds from QualityThresholds
                
                # High volatility regime (volatility > threshold)
                if volatility_level > QualityThresholds.HIGH_VOLATILITY_THRESHOLD:
                    if metrics['volatility_clustering'] > QualityThresholds.VOLATILITY_CLUSTERING_THRESHOLD:
                        return RegimeType.VOLATILE, metrics
                
                # Trending regime (strong trend + high persistence)
                if trend_strength > QualityThresholds.TREND_STRENGTH_THRESHOLD and metrics['trend_persistence'] > QualityThresholds.TREND_PERSISTENCE_THRESHOLD:
                    return RegimeType.TRENDING, metrics
                
                # Mean reverting regime (negative autocorrelation)
                if metrics['trend_persistence'] < QualityThresholds.MEAN_REVERSION_THRESHOLD:
                    return RegimeType.MEAN_REVERTING, metrics
                
                # Stable regime (low volatility + low trend)
                if volatility_level < QualityThresholds.LOW_VOLATILITY_THRESHOLD and trend_strength < QualityThresholds.LOW_TREND_THRESHOLD:
                    return RegimeType.STABLE, metrics
                
                # Default: determine by strongest signal
                scaled_volatility = volatility_level * QualityThresholds.VOLATILITY_SCALE_FACTOR
                max_score = max(
                    trend_strength,
                    abs(mean_reversion_score),
                    scaled_volatility,
                    metrics['stability_score']
                )
                
                if max_score == trend_strength:
                    return RegimeType.TRENDING, metrics
                elif max_score == abs(mean_reversion_score):
                    return RegimeType.MEAN_REVERTING, metrics
                elif max_score == scaled_volatility:
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
                        (second_half_mean - first_half_mean) / (abs(first_half_mean) + QualityThresholds.DBI_EPSILON)
                    )
                
            elif regime_type == RegimeType.MEAN_REVERTING:
                # Mean reverting regime metrics
                mean_return = returns.mean()
                specific_metrics['reversion_center'] = float(mean_return)
                
                # Reversion speed: how quickly prices return to mean
                deviations = abs(returns - mean_return)
                specific_metrics['reversion_speed'] = float(1.0 / (deviations.mean() + QualityThresholds.DBI_EPSILON))
                
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
                cv = returns.std() / (abs(returns.mean()) + QualityThresholds.DBI_EPSILON)
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
        
        # OPTIMIZATION: Pre-calculate all cluster sizes once (O(n)) instead of recalculating for each regime (O(n²))
        unique_labels = np.unique(regime_labels)
        non_noise_labels = unique_labels[unique_labels != -1]
        
        # Pre-calculate cluster sizes
        cluster_sizes = {}
        for label in non_noise_labels:
            cluster_sizes[int(label)] = int(np.sum(regime_labels == label))
        
        # Calculate mean cluster size for balance contribution
        if cluster_sizes:
            mean_cluster_size = float(np.mean(list(cluster_sizes.values())))
        else:
            mean_cluster_size = 1.0
        
        # Ensure alignment before processing
        regime_labels, features, forward_returns = self._ensure_aligned_data(
            regime_labels, features, forward_returns
        )
        
        for regime_id in non_noise_labels:
            regime_id = int(regime_id)
            regime_mask = regime_labels == regime_id
            regime_features = features.iloc[regime_mask].select_dtypes(include=[np.number])
            
            if len(regime_features) == 0:
                continue
            
            # Feature coefficient of variation
            feature_cv = {}
            for col in regime_features.columns:
                if regime_features[col].std() > 0:
                    cv = regime_features[col].std() / (abs(regime_features[col].mean()) + QualityThresholds.DBI_EPSILON)
                    feature_cv[col] = float(cv)
            
            # Use pre-calculated cluster size
            regime_size = cluster_sizes[regime_id]
            regime_percentage = float((regime_size / total_samples) * 100)
            
            regime_metrics = {
                # Size and balance
                'size': regime_size,
                'percentage': regime_percentage,
                
                # CV metrics
                'feature_coefficient_of_variation': feature_cv,
                'mean_cv': float(np.mean(list(feature_cv.values()))) if feature_cv else 0.0,
                'std_cv': float(np.std(list(feature_cv.values()))) if feature_cv else 0.0,
                
                # Individual regime balance contribution (using pre-calculated mean)
                'balance_contribution': float(regime_size / (mean_cluster_size + QualityThresholds.DBI_EPSILON))
            }
            
            # Detect regime type and calculate regime-specific characteristics
            # Handle index alignment for forward_returns (already aligned, but ensure positional indexing)
            if forward_returns is not None:
                regime_returns = forward_returns.iloc[regime_mask] if hasattr(forward_returns, 'iloc') else forward_returns[regime_mask]
            else:
                regime_returns = None
            
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
                    'sharpe': float(regime_returns.mean() / (regime_returns.std() + QualityThresholds.DBI_EPSILON)),
                    'skewness': float(regime_returns.skew()) if hasattr(regime_returns, 'skew') else 0.0,
                    'max_drawdown': float(self._compute_max_drawdown(regime_returns))
                })
            else:
                regime_metrics['regime_type'] = RegimeType.UNKNOWN.value
                regime_metrics['classification_scores'] = {}
                regime_metrics['regime_specific_metrics'] = {}
            
            per_regime_metrics[regime_id] = regime_metrics
        
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
                if m.get('regime_type') == 'trending' and m.get('sharpe', 0) > QualityThresholds.MIN_SHARPE_FOR_STRATEGY
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
                if m.get('regime_type') == 'mean_reverting' and m.get('sharpe', 0) > QualityThresholds.MIN_SHARPE_FOR_STRATEGY
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
                if m.get('max_drawdown', 0) < QualityThresholds.HIGH_DRAWDOWN_THRESHOLD or m.get('sharpe', 0) < QualityThresholds.NEGATIVE_SHARPE_THRESHOLD
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
        
        Args:
            regime_labels: Regime/cluster labels (must be aligned with indices)
            forward_returns: Forward returns series (must share index with feature_data)
            feature_data: Feature DataFrame (must share index with forward_returns)
            
        Returns:
            Dictionary of validation results per regime
        """
        results = {}
        
        # Ensure proper alignment before processing
        regime_labels, feature_data, forward_returns = self._ensure_aligned_data(
            regime_labels, feature_data, forward_returns
        )
        
        # Per-regime statistics
        for regime_id in np.unique(regime_labels):
            if regime_id == -1:  # Skip noise
                continue
            
            regime_mask = (regime_labels == regime_id)
            
            # Extract aligned data using positional indexing (iloc)
            regime_returns = forward_returns.iloc[regime_mask] if hasattr(forward_returns, 'iloc') else forward_returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            results[f'regime_{regime_id}'] = {
                'mean_return': float(regime_returns.mean()),
                'volatility': float(regime_returns.std()),
                'sharpe': float(regime_returns.mean() / (regime_returns.std() + QualityThresholds.DBI_EPSILON)),
                'skewness': float(regime_returns.skew()) if hasattr(regime_returns, 'skew') else 0.0,
                'max_drawdown': float(self._compute_max_drawdown(regime_returns))
            }
            
            # Add feature behavior in this regime (using selected columns)
            numeric_features = feature_data.select_dtypes(include=[np.number])
            for col in ['spread', 'volume', 'volatility']:
                if col in numeric_features.columns:
                    # Use positional indexing for consistent alignment
                    col_data = numeric_features.iloc[regime_mask, numeric_features.columns.get_loc(col)]
                    results[f'regime_{regime_id}'][f'avg_{col}'] = float(col_data.mean())
        
        # Regime stability
        results['regime_persistence'] = float(self._calculate_regime_persistence(regime_labels))
        
        return results
    
    def _calculate_predictive_power(self,
                                   regime_labels: np.ndarray,
                                   forward_returns: pd.Series) -> float:
        """
        Calculate predictive power: can current regime predict future returns?
        
        Uses Random Forest classifier to predict return sign from regime labels.
        
        Args:
            regime_labels: Regime/cluster labels
            forward_returns: Forward returns series (must be aligned with regime_labels)
            
        Returns:
            Cross-validation score (0-1) indicating predictive power
        """
        try:
            # Validate input lengths using thresholds from QualityThresholds
            min_samples = QualityThresholds.MIN_SAMPLES_FOR_PREDICTIVE_POWER
            if len(regime_labels) < min_samples or len(forward_returns) < min_samples:
                return 0.0
            
            # Ensure arrays are aligned and valid
            min_len = min(len(regime_labels), len(forward_returns))
            if min_len < min_samples:
                return 0.0
            
            # Ensure alignment: regime_labels[t] predicts forward_returns[t+1]
            # Make sure we have matching lengths for prediction
            max_predictable = min_len - 1
            if max_predictable < min_samples:
                return 0.0
            
            # Extract aligned data: use regime at time t to predict return at t+1
            # Use LabelEncoder for more robust encoding (as suggested in review)
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            X = encoder.fit_transform(regime_labels[:max_predictable]).reshape(-1, 1)
            y = (forward_returns[1:max_predictable + 1] > 0).astype(int).values
            
            # Validate alignment
            if len(X) != len(y):
                self.logger.warning(
                    f"Alignment issue in predictive power: X length={len(X)}, y length={len(y)}"
                )
                return 0.0
            
            # Check if we have enough samples and variation
            if len(y) < min_samples:
                return 0.0
            
            unique_values = len(set(y))
            if unique_values < 2:
                return 0.0
            
            # Calculate safe number of CV folds using thresholds from QualityThresholds
            min_samples_per_fold = QualityThresholds.MIN_SAMPLES_PER_CV_FOLD
            max_folds = min(QualityThresholds.MAX_CV_FOLDS, max(2, len(y) // min_samples_per_fold))
            
            if max_folds < 2:
                return 0.0
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(rf, X, y, cv=max_folds)
            
            if len(cv_scores) == 0:
                return 0.0
            
            return float(cv_scores.mean())
            
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
        
        Combines multiple metrics into a single score using weights from QualityThresholds.
        """
        score_components = []
        weights = []
        
        # 1. Silhouette score (normalize to 0-1, already in [-1, 1])
        if metrics.silhouette_score is not None:
            silhouette_normalized = (metrics.silhouette_score + 1) / 2  # Map [-1, 1] to [0, 1]
            score_components.append(silhouette_normalized)
            weights.append(QualityThresholds.WEIGHT_SILHOUETTE)
        
        # 2. Davies-Bouldin Index (lower is better, normalize inversely)
        if metrics.davies_bouldin_score is not None and not np.isinf(metrics.davies_bouldin_score):
            # DBI typically ranges from 0 to 5+, map to [0, 1] inversely
            dbi_normalized = 1.0 / (1.0 + metrics.davies_bouldin_score)
            score_components.append(dbi_normalized)
            weights.append(QualityThresholds.WEIGHT_DBI)
        
        # 3. Calinski-Harabasz Index (higher is better, normalize)
        if metrics.calinski_harabasz_score is not None:
            # CH typically ranges from 0 to 1000+, use sigmoid-like normalization
            ch_normalized = np.tanh(metrics.calinski_harabasz_score / QualityThresholds.CH_NORMALIZATION_DIVISOR)
            score_components.append(ch_normalized)
            weights.append(QualityThresholds.WEIGHT_CH)
        
        # 4. CV ratio (higher between/lower within is better)
        if metrics.within_regime_cv is not None and metrics.between_regime_cv is not None:
            # Ideal: low within, high between
            # Ratio of between/within, normalized
            cv_ratio = metrics.between_regime_cv / (metrics.within_regime_cv + QualityThresholds.DBI_EPSILON)
            cv_normalized = np.tanh(cv_ratio)  # Sigmoid-like normalization
            score_components.append(cv_normalized)
            weights.append(QualityThresholds.WEIGHT_CV_RATIO)
        
        # 5. Balance score (already in [0, 1])
        if metrics.balance_score is not None:
            score_components.append(metrics.balance_score)
            weights.append(QualityThresholds.WEIGHT_BALANCE)
        
        # 6. Temporal smoothness (already in [0, 1])
        if metrics.temporal_smoothness is not None:
            score_components.append(metrics.temporal_smoothness)
            weights.append(QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS)
        
        # 7. Noise ratio (lower is better, invert)
        noise_score = 1.0 - metrics.noise_ratio
        score_components.append(noise_score)
        weights.append(QualityThresholds.WEIGHT_NOISE_RATIO)
        
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
            
            # Filter to only valid dataclass fields to prevent errors
            valid_fields = {f.name for f in fields(ClusterQualityMetrics)}
            filtered_dict = {
                k: v for k, v in metrics_dict.items() 
                if k in valid_fields
            }
            
            # Reconstruct ClusterQualityMetrics from dict
            return ClusterQualityMetrics(**filtered_dict)
            
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
| **Noise Ratio** | {metrics.noise_ratio:.2%} | {'✅' if metrics.noise_ratio < QualityThresholds.MAX_NOISE_RATIO else '⚠️'} |
| **Silhouette Score** | {metrics.silhouette_score:.4f if metrics.silhouette_score else 'N/A'} | {'✅' if metrics.silhouette_score and metrics.silhouette_score > QualityThresholds.MIN_SILHOUETTE else '⚠️'} |
| **Davies-Bouldin Index** | {metrics.davies_bouldin_score:.4f if metrics.davies_bouldin_score else 'N/A'} | {'✅' if metrics.davies_bouldin_score and metrics.davies_bouldin_score < QualityThresholds.MAX_DBI else '⚠️'} |
| **Calinski-Harabasz Index** | {metrics.calinski_harabasz_score:.2f if metrics.calinski_harabasz_score else 'N/A'} | {'✅' if metrics.calinski_harabasz_score and metrics.calinski_harabasz_score > QualityThresholds.MIN_CH else '⚠️'} |
| **Balance Score** | {metrics.balance_score:.4f if metrics.balance_score else 'N/A'} | {'✅' if metrics.balance_score and metrics.balance_score > QualityThresholds.QUALITY_GOOD else '⚠️'} |

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
            if metrics.quality_score >= QualityThresholds.QUALITY_EXCELLENT:
                quality_level = "Excellent ✅"
                recommendation = "The clustering shows excellent quality. Proceed with confidence."
            elif metrics.quality_score >= QualityThresholds.QUALITY_GOOD:
                quality_level = "Good ✅"
                recommendation = "The clustering shows good quality. Suitable for most applications."
            elif metrics.quality_score >= QualityThresholds.QUALITY_MODERATE:
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
