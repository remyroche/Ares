"""
Unified Cluster Quality Assessor

This module provides a unified, standardized way to assess cluster quality
across different clustering approaches (HDBSCAN, regime clustering, etc.).

It integrates with BaseStep's artifact manager and provides comprehensive
quality metrics including:
- Silhouette scores (global and per-cluster)
- Davies-Bouldin Index (DBI)
- Calinski-Harabasz Index (CH)
- Within/Between regime coefficient of variation (for features and economics)
- Temporal smoothness
- Regime persistence
- Economic validation (including target return analysis)
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
try:
    from src.utils.tprint import (
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
        tprint_debug,
        tprint_timer,
        tprint_logged
    )
except ImportError:
    # Fallback basic logging if tprint is not available
    print("Warning: 'tprint' utilities not found. Using standard logging.")
    logging.basicConfig(level=logging.INFO)
    tprint_info = logging.info
    tprint_warning = logging.warning
    tprint_error = logging.error
    tprint_success = logging.info
    tprint_debug = logging.debug
    tprint_timer = lambda x: (lambda y: (lambda: y))(None) # No-op timer
    tprint_logged = lambda **kwargs: lambda f: f # No-op decorator


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
    # *** NEW: Target return (0.7%) as requested ***
    ECONOMIC_TARGET_RETURN = 0.007 
    
    # Quality score weights
    WEIGHT_TEMPORAL_SMOOTHNESS = 0.30
    WEIGHT_CV_RATIO = 0.30
    WEIGHT_SILHOUETTE = 0.20
    WEIGHT_BALANCE = 0.10
    WEIGHT_NOISE_RATIO = 0.10
    
    # *** MODIFIED: Use 1e-9 for more stability ***
    DBI_EPSILON = 1e-9  
    
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
        within_regime_cv: Avg within-regime CV for features
        within_regime_cv_std: Std dev of within-regime CVs for features
        between_regime_cv: Avg between-regime CV for features
        between_regime_cv_std: Std dev of between-regime CVs for features
        per_regime_cv: Per-regime CV values for features
        economic_cv_metrics: CV metrics for economic outcomes
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
    
    # *** REVERTED: Kept original names for backward compatibility ***
    # Coefficient of variation metrics (with std dev) - for FEATURES
    within_regime_cv: Optional[float] = None
    within_regime_cv_std: Optional[float] = None
    between_regime_cv: Optional[float] = None
    between_regime_cv_std: Optional[float] = None
    per_regime_cv: Optional[Dict[int, float]] = None  # Per-regime CV values
    
    # *** NEW: CV metrics for ECONOMIC outcomes ***
    economic_cv_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # *** NEW: Per-category CV metrics for features ***
    feature_category_cv_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Temporal metrics
    temporal_smoothness: Optional[float] = None
    temporal_smoothness_raw: Optional[float] = None  # Without flip-flop penalty
    flip_flop_ratio: Optional[float] = None  # Ratio of flip-flop transitions
    regime_persistence: Optional[float] = None

    # Enhanced temporal metrics
    regime_duration_distribution: Dict[str, Any] = field(default_factory=dict)
    transition_probability_matrix: Dict[str, Any] = field(default_factory=dict)
    
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
            
            # *** REVERTED: Kept original names ***
            'within_regime_cv': self.within_regime_cv,
            'within_regime_cv_std': self.within_regime_cv_std,
            'between_regime_cv': self.between_regime_cv,
            'between_regime_cv_std': self.between_regime_cv_std,
            'per_regime_cv': self.per_regime_cv,
            
            # *** NEW: Economic CV metrics ***
            'economic_cv_metrics': self.economic_cv_metrics,
            
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
        
        This method validates lengths and handles index misalignment by truncating
        to the minimum length and converting to positional indexing.
        
        Args:
            regime_labels: Regime/cluster labels array
            feature_data: Feature DataFrame
            forward_returns: Optional forward returns Series
            
        Returns:
            Tuple of (aligned_regime_labels, aligned_feature_data, aligned_forward_returns)
        """
        # Determine minimum length across all inputs
        min_length = len(regime_labels)
        lengths = [len(regime_labels), len(feature_data)]
        
        if forward_returns is not None:
            lengths.append(len(forward_returns))
        
        min_length = min(lengths)
        
        # Warn if truncation is needed
        if len(regime_labels) != min_length or len(feature_data) != min_length:
            self.logger.warning(
                f"Length mismatch detected. Truncating to minimum length: {min_length}. "
                f"(regime_labels: {len(regime_labels)}, feature_data: {len(feature_data)}"
                f"{f', forward_returns: {len(forward_returns)}' if forward_returns is not None else ''})"
            )
        
        # Truncate and reset indices for positional alignment
        regime_labels = regime_labels[:min_length]
        feature_data = feature_data.iloc[:min_length].reset_index(drop=True)
        
        # Handle forward_returns alignment
        aligned_forward_returns = None
        if forward_returns is not None:
            forward_returns = forward_returns.iloc[:min_length].reset_index(drop=True)
            aligned_forward_returns = forward_returns
        
        return regime_labels, feature_data, aligned_forward_returns
    
    @tprint_logged(include_args=False, include_result=False)
    def assess_quality(self,
                       regime_labels: np.ndarray,
                       feature_data: pd.DataFrame,
                       forward_returns: Optional[pd.Series] = None,
                       timestamps: Optional[pd.DatetimeIndex] = None,
                       min_regime_size: int = 10,
                       temporal_sensitivity_mode: str = "standard") -> ClusterQualityMetrics:
        """
        Comprehensive cluster quality assessment.

        Args:
            regime_labels: Regime/cluster labels (-1 for noise)
            feature_data: Feature data used for clustering
            forward_returns: Optional forward returns for economic validation
            timestamps: Optional timestamps for temporal analysis
            min_regime_size: Minimum regime size to consider
            temporal_sensitivity_mode: Sensitivity mode for temporal smoothness calculation
                - "standard": Original calculation
                - "exponential_decay": More aggressive transition penalty
                - "weighted_transitions": Weight transitions by regime duration
                - "regime_persistence_focused": Emphasize long regime persistence

        Returns:
            ClusterQualityMetrics object with all computed metrics
        """
        tprint_info("🔍 Starting comprehensive cluster quality assessment")
        
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
        
        # 4. *** REVERTED: Kept original function call ***
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
        
        # 6. Temporal smoothness and persistence (with flip-flop penalty)
        if timestamps is not None:
            try:
                with tprint_timer("Temporal Metrics Calculation"):
                    (metrics.temporal_smoothness,
                     metrics.temporal_smoothness_raw,
                     metrics.flip_flop_ratio) = self._calculate_temporal_smoothness(
                        regime_labels, timestamps, sensitivity_mode=temporal_sensitivity_mode
                    )
                    metrics.regime_persistence = self._calculate_regime_persistence(regime_labels)

                    # Enhanced temporal metrics
                    metrics.regime_duration_distribution = self._calculate_regime_duration_distribution(regime_labels)
                    metrics.transition_probability_matrix = self._calculate_transition_probability_matrix(regime_labels)

                tprint_success(f"✅ Temporal smoothness: {metrics.temporal_smoothness:.4f} (raw: {metrics.temporal_smoothness_raw:.4f}, flip-flop: {metrics.flip_flop_ratio:.3f}), Persistence: {metrics.regime_persistence:.2f}")
                tprint_success(f"✅ Enhanced temporal: Duration stability={metrics.regime_duration_distribution.get('duration_stability_score', 0):.3f}, Transition stability={metrics.transition_probability_matrix.get('transition_stability_score', 0):.3f}")
            except Exception as e:
                tprint_error(f"❌ Failed to calculate temporal metrics: {e}")
        
        # 6b. Per-category CV metrics
        try:
            with tprint_timer("Per-Category CV Metrics Calculation"):
                metrics.feature_category_cv_metrics = self._calculate_cv_metrics_by_category(
                    regime_labels, features_clean, non_noise_mask
                )
                num_categories = len(metrics.feature_category_cv_metrics)
                tprint_success(f"✅ Calculated CV metrics for {num_categories} feature categories")
        except Exception as e:
            tprint_error(f"❌ Failed to calculate per-category CV metrics: {e}")
        
        # 7. Per-regime metrics (includes regime type detection and NEW economic targets)
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
            
        # *** NEW: 7b. Economic Coefficient of Variation ***
        if forward_returns is not None:
            try:
                with tprint_timer("Economic CV Metrics Calculation"):
                    metrics.economic_cv_metrics = self._calculate_economic_cv_metrics(
                        metrics.per_regime_metrics, forward_returns, regime_labels
                    )
                tprint_success("✅ Economic CV metrics complete")
            except Exception as e:
                tprint_error(f"❌ Failed to calculate economic CV metrics: {e}")

        # 8. Economic validation (if forward returns provided)
        if forward_returns is not None:
            try:
                metrics.economic_validation = metrics.per_regime_metrics
                tprint_success("✅ Economic validation populated from per-regime metrics")
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
        run_validators: bool = True,
        temporal_sensitivity_mode: str = "standard"
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
            min_regime_size=min_regime_size,
            temporal_sensitivity_mode=temporal_sensitivity_mode
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

    # *** NEW: Helper function for safe CV calculation ***
    def _calculate_cv(self, x: np.ndarray) -> float:
        """Calculates the Coefficient of Variation (std / |mean|)."""
        if x is None or len(x) < 2:
            return np.nan
        
        mean = np.nanmean(x)
        std = np.nanstd(x)
        
        if np.abs(mean) < QualityThresholds.DBI_EPSILON:
            return np.nan  # Avoid division by zero
        
        return std / np.abs(mean)

    # *** REVERTED: Kept original function name ***
    def _calculate_cv_metrics(self,
                                regime_labels: np.ndarray,
                                features: pd.DataFrame,
                                non_noise_mask: np.ndarray) -> Tuple[float, float, float, float, Dict[int, float]]:
        """
        Calculate within-regime and between-regime coefficient of variation with std dev
        for clustering features.
        
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
                    out=np.zeros_like(cluster_std, dtype=float),
                    where=denominator != 0
                )
                
                # Remove infinite or NaN values
                cv_values = cv_values[np.isfinite(cv_values)]
                
                if len(cv_values) > 0:
                    cluster_cv = float(np.nanmean(cv_values))
                    within_cvs.append(cluster_cv)
                    per_regime_cv[int(cluster_id)] = cluster_cv
        
        # Calculate mean and std dev of within-regime CVs
        within_regime_cv_mean = float(np.nanmean(within_cvs)) if within_cvs else 0.0
        within_regime_cv_std = float(np.nanstd(within_cvs)) if len(within_cvs) > 1 else 0.0
        
        # SAFEGUARD: Ensure minimum within_regime_cv to prevent extreme CV ratios
        # If within-regime CV is too small (overly homogeneous regimes), this indicates
        # potential numerical instability or overfitting
        MIN_WITHIN_CV = 0.01  # Minimum 1% coefficient of variation
        if 0 < within_regime_cv_mean < MIN_WITHIN_CV:
            within_regime_cv_mean = MIN_WITHIN_CV
        
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
                # *** MODIFIED: Use new safe CV helper ***
                cv = self._calculate_cv(feature_means)
                
                if np.isfinite(cv):
                    between_cvs.append(cv)
            
            between_regime_cv_mean = float(np.nanmean(between_cvs)) if between_cvs else 0.0
            between_regime_cv_std = float(np.nanstd(between_cvs)) if len(between_cvs) > 1 else 0.0
        
        return within_regime_cv_mean, within_regime_cv_std, between_regime_cv_mean, between_regime_cv_std, per_regime_cv

    def _auto_detect_feature_categories(self, feature_names: pd.Index) -> Dict[str, List[str]]:
        """
        Auto-detect feature categories from feature names using pattern matching.
        
        Args:
            feature_names: Index or list of feature names
            
        Returns:
            Dict mapping category names to lists of feature names
        """
        categories = {
            'momentum': [],
            'volume': [],
            'volatility': [],
            'spread': [],
            'microstructure': [],
            'price': [],
            'other': []
        }
        
        for feature in feature_names:
            feature_lower = str(feature).lower()
            
            # Momentum indicators
            if any(keyword in feature_lower for keyword in ['rsi', 'macd', 'momentum', 'cci', 'stoch', 'roc', 'trix', 'adx']):
                categories['momentum'].append(feature)
            # Volume indicators
            elif any(keyword in feature_lower for keyword in ['volume', 'obv', 'vwap', 'mfi', 'cmf', 'vpt']):
                categories['volume'].append(feature)
            # Volatility indicators
            elif any(keyword in feature_lower for keyword in ['volatility', 'atr', 'bb', 'bollinger', 'keltner', 'std', 'variance']):
                categories['volatility'].append(feature)
            # Spread/book indicators
            elif any(keyword in feature_lower for keyword in ['spread', 'bid', 'ask', 'depth', 'book']):
                categories['spread'].append(feature)
            # Microstructure
            elif any(keyword in feature_lower for keyword in ['tick', 'trades', 'order', 'imbalance', 'flow']):
                categories['microstructure'].append(feature)
            # Price-based
            elif any(keyword in feature_lower for keyword in ['price', 'close', 'open', 'high', 'low', 'ema', 'sma', 'ma_']):
                categories['price'].append(feature)
            else:
                categories['other'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _calculate_cv_metrics_by_category(self,
                                           regime_labels: np.ndarray,
                                           features: pd.DataFrame,
                                           non_noise_mask: np.ndarray,
                                           feature_categories: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate CV metrics grouped by feature category.
        
        This provides more granular insights into which feature types are most
        discriminative across regimes vs. homogeneous within regimes.
        
        Args:
            regime_labels: Cluster/regime labels
            features: Feature DataFrame
            non_noise_mask: Boolean mask for non-noise samples
            feature_categories: Optional dict mapping category names to lists of feature names.
                               If None, auto-detects categories from feature names.
        
        Returns:
            Dict with structure:
            {
                'momentum': {
                    'within_cv_mean': 0.5,
                    'within_cv_std': 0.1,
                    'between_cv_mean': 1.2,
                    'between_cv_std': 0.3,
                    'cv_ratio': 2.4,
                    'num_features': 5,
                    'features': ['rsi', 'macd', ...]
                },
                'volume': {...},
                ...
            }
        """
        if feature_categories is None:
            # Auto-detect categories from feature names
            feature_categories = self._auto_detect_feature_categories(features.columns)
        
        category_cv_metrics = {}
        
        for category_name, feature_list in feature_categories.items():
            # Filter to features that exist in the DataFrame
            valid_features = [f for f in feature_list if f in features.columns]
            
            if not valid_features:
                continue
            
            # Extract category features
            category_features = features[valid_features]
            
            try:
                # Calculate CV metrics for this category
                within_cv, within_std, between_cv, between_std, per_regime = self._calculate_cv_metrics(
                    regime_labels, category_features, non_noise_mask
                )
                
                # Calculate ratio
                cv_ratio = between_cv / (within_cv + QualityThresholds.DBI_EPSILON)
                
                category_cv_metrics[category_name] = {
                    'within_cv_mean': float(within_cv),
                    'within_cv_std': float(within_std),
                    'between_cv_mean': float(between_cv),
                    'between_cv_std': float(between_std),
                    'cv_ratio': float(cv_ratio),
                    'num_features': len(valid_features),
                    'features': valid_features
                }
                
                tprint_debug(f"  Category '{category_name}': CV ratio={cv_ratio:.3f} ({len(valid_features)} features)")
                
            except Exception as e:
                tprint_warning(f"Failed to calculate CV metrics for category '{category_name}': {e}")
                continue
        
        return category_cv_metrics

    # *** NEW: Function to calculate CV metrics for economic outcomes ***
    def _calculate_economic_cv_metrics(self, 
                                     per_regime_metrics: Dict[int, Dict[str, Any]],
                                     forward_returns: pd.Series,
                                     regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculates CV metrics for the economic relevance results.
        
        This calculates:
        1.  Within-Regime CV: Avg. CV of 1h forward returns *within* each regime.
        2.  Between-Regime CV: CV of the *mean* economic metrics (mean_return, sharpe, etc.)
            *across* regimes.
        3.  Ratio: Between-CV / Within-CV.

        Args:
            per_regime_metrics (Dict): Output from _calculate_per_regime_metrics.
            forward_returns (pd.Series): The raw forward returns timeseries.
            regime_labels (np.ndarray): The raw regime labels timeseries.

        Returns:
            dict: CV metrics for economic results.
        """
        
        if not per_regime_metrics:
            return {}

        metrics_data = {}
        
        # --- 1. Within-Regime CV (based on timeseries data) ---
        # We only calculate this for the raw forward returns, as it measures
        # the compactness of the return distribution within each regime.
        within_cvs = []
        non_noise_labels = [l for l in np.unique(regime_labels) if l != -1]
        
        for label in non_noise_labels:
            regime_mask = (regime_labels == label)
            
            # Ensure indices match between cluster_labels and forward_returns
            if len(regime_mask) != len(forward_returns):
                # Truncate regime_mask to match forward_returns length
                regime_mask_aligned = regime_mask[:len(forward_returns)]
                regime_returns_ts = forward_returns[regime_mask_aligned].values
            else:
                regime_returns_ts = forward_returns[regime_mask].values
            
            within_cvs.append(self._calculate_cv(regime_returns_ts))
        
        avg_within_cv_fwd_return = np.nanmean(within_cvs) if within_cvs else np.nan
        metrics_data['economic_avg_within_cv_fwd_return'] = avg_within_cv_fwd_return

        # --- 2. Between-Regime CV (based on per-regime aggregate metrics) ---
        # This measures the separation of the *average* economic outcomes
        # between regimes.
        try:
            metrics_df = pd.DataFrame.from_dict(per_regime_metrics, orient='index')
        except Exception:
            self.logger.warning("Could not create DataFrame for economic CV metrics.")
            return metrics_data
            
        # Define which economic metrics to compare across regimes
        metrics_to_compare = [
            'mean_return', 'volatility', 'sharpe', 
            'pct_above_target', 'pct_below_neg_target', 'pct_target_hits'
        ]
        
        for col in metrics_to_compare:
            if col in metrics_df.columns:
                metric_values = metrics_df[col].dropna().values
                if len(metric_values) > 1:
                    cv = self._calculate_cv(metric_values)
                    metrics_data[f'economic_between_cv_{col}'] = cv
                else:
                    metrics_data[f'economic_between_cv_{col}'] = np.nan
            else:
                 metrics_data[f'economic_between_cv_{col}'] = np.nan
        
        # --- 3. Ratio ---
        # We can create a ratio for mean_return
        between_mean_return_cv = metrics_data.get('economic_between_cv_mean_return', np.nan)
        
        if not np.isnan(avg_within_cv_fwd_return) and avg_within_cv_fwd_return > QualityThresholds.DBI_EPSILON:
            ratio = between_mean_return_cv / avg_within_cv_fwd_return
            metrics_data['economic_cv_ratio_mean_return'] = ratio
        else:
            metrics_data['economic_cv_ratio_mean_return'] = np.nan
            
        return metrics_data


    def _calculate_temporal_smoothness(self,
                                         regime_labels: np.ndarray,
                                         timestamps: Optional[pd.DatetimeIndex] = None,
                                         flip_flop_weight: float = 1.0,
                                         penalty_mode: str = "effective_transitions",
                                         sensitivity_mode: str = "standard") -> Tuple[float, float, float]:
        """
        Calculate temporal smoothness score with flip-flop penalty.

        Higher score means fewer regime transitions (more stable regimes).
        Score is normalized to [0, 1] where 1 is perfectly smooth.

        Flip-flop penalty: Detects rapid back-and-forth transitions (A→B→A)
        which are particularly undesirable as they indicate instability.

        Args:
            regime_labels: Regime/cluster labels
            timestamps: Optional timestamps for time-aware analysis (currently not used but
                        validated for future enhancements)
            flip_flop_weight: Weight for flip-flop penalty (default 1.0 = count as 2 transitions)
            penalty_mode: How to apply penalty:
                - "effective_transitions": Count each flip-flop as additional transitions (default)
                - "multiplier": Multiply raw smoothness by (1 - flip_flop_ratio * weight)
            sensitivity_mode: How to calculate smoothness for parameter sensitivity:
                - "standard": Original calculation (default)
                - "exponential_decay": Penalize transitions more aggressively
                - "weighted_transitions": Weight transitions by duration
                - "regime_persistence_focused": Emphasize long regime persistence

        Returns:
            Tuple of (smoothness_with_penalty, smoothness_raw, flip_flop_ratio)
        """
        if len(regime_labels) < 2:
            return 1.0, 1.0, 0.0
        
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
            return 1.0, 1.0, 0.0
        
        # Raw smoothness score: fewer changes = higher smoothness
        if sensitivity_mode == "standard":
            smoothness_raw = 1.0 - (regime_changes / max_possible_changes)
        elif sensitivity_mode == "exponential_decay":
            # More aggressive penalty: transitions are exponentially costly
            transition_ratio = regime_changes / max_possible_changes
            smoothness_raw = np.exp(-3.0 * transition_ratio)  # Sharp decay
        elif sensitivity_mode == "weighted_transitions":
            # Weight transitions by their duration context
            transition_weights = self._calculate_transition_weights(regime_labels)
            weighted_transitions = np.sum(transition_weights)
            smoothness_raw = 1.0 - min(1.0, weighted_transitions / max_possible_changes)
        elif sensitivity_mode == "regime_persistence_focused":
            # Emphasize long regimes, penalize short ones heavily
            regime_durations = self._get_regime_durations(regime_labels)
            if len(regime_durations) > 0:
                avg_duration = np.mean(regime_durations)
                duration_score = min(1.0, avg_duration / 50.0)  # Assume 50 is a good duration
                smoothness_raw = duration_score
            else:
                smoothness_raw = 0.0
        else:
            smoothness_raw = 1.0 - (regime_changes / max_possible_changes)
        
        # Detect flip-flop patterns (A→B→A): regime at t-2 equals regime at t, but differs from t-1
        flip_flops = 0.0
        if len(regime_labels) >= 3:
            flip_flops = np.sum(
                (regime_labels[:-2] == regime_labels[2:]) & 
                (regime_labels[:-2] != regime_labels[1:-1])
            )
        
        # Calculate flip-flop ratio
        flip_flop_ratio = flip_flops / max_possible_changes if max_possible_changes > 0 else 0.0
        
        # NEW: Duration-weighted short-lived regime penalty
        # Penalize regimes that are short-lived between longer regimes (broader than 3-step flip-flops)
        # SCALED DOWN: These penalties were too aggressive, zeroing out temporal smoothness
        short_lived_penalty = self._calculate_short_lived_regime_penalty(regime_labels, threshold_hours=3) * 0.3
        
        # NEW: Temporal autocorrelation penalty
        # Penalize regime sequences that look random (low autocorrelation = unstable)
        # SCALED DOWN: These penalties were too aggressive, zeroing out temporal smoothness
        autocorr_penalty = self._calculate_temporal_autocorrelation_penalty(regime_labels) * 0.3
        
        # NEW: BONUSES for "good" regime configurations
        # These reward high-quality temporal behavior (can push score above raw smoothness)
        regime_duration_bonus = self._calculate_regime_duration_bonus(regime_labels)
        low_transition_bonus = self._calculate_low_transition_bonus(regime_labels, smoothness_raw)
        
        # Apply penalty based on mode
        if penalty_mode == "effective_transitions":
            # Each flip-flop counts as additional transitions beyond the already counted ones
            # With weight=1.0: each flip-flop adds 1 extra transition (total 2 transitions)
            # With weight=0.5: each flip-flop adds 0.5 extra transitions (total 1.5 transitions)
            flip_flop_penalty = flip_flop_ratio * flip_flop_weight
        elif penalty_mode == "multiplier":
            # Directly multiply raw smoothness by penalty factor
            flip_flop_penalty = flip_flop_ratio * flip_flop_weight
        else:
            # Default to effective_transitions
            flip_flop_penalty = flip_flop_ratio * flip_flop_weight
        
        # Final smoothness with ALL penalties AND bonuses
        # Penalties subtract from raw score, bonuses add to it
        total_penalties = flip_flop_penalty + short_lived_penalty + autocorr_penalty
        total_bonuses = regime_duration_bonus + low_transition_bonus
        smoothness_final = max(0.0, min(1.0, smoothness_raw - total_penalties + total_bonuses))
        
        return float(smoothness_final), float(smoothness_raw), float(flip_flop_ratio)

    def _calculate_transition_weights(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate weights for transitions based on regime durations."""
        if len(regime_labels) < 2:
            return np.array([])

        # Find transition points
        transitions = np.where(regime_labels[1:] != regime_labels[:-1])[0]

        if len(transitions) == 0:
            return np.array([0.0])

        weights = []
        for i, trans_idx in enumerate(transitions):
            # Weight based on duration of the regime being left
            if i == 0:
                regime_duration = trans_idx + 1
            else:
                regime_duration = trans_idx - transitions[i-1]

            # Shorter regimes get higher transition weights (more disruptive)
            weight = 1.0 / max(1.0, regime_duration / 10.0)
            weights.append(weight)

        return np.array(weights)

    def _get_regime_durations(self, regime_labels: np.ndarray) -> np.ndarray:
        """Extract duration of each regime period."""
        if len(regime_labels) < 1:
            return np.array([])

        durations = []
        current_regime = regime_labels[0]
        current_length = 1

        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_length += 1
            else:
                durations.append(current_length)
                current_regime = regime_labels[i]
                current_length = 1

        # Add the last regime
        durations.append(current_length)

        return np.array(durations)

    def _calculate_short_lived_regime_penalty(self, regime_labels: np.ndarray, threshold_hours: int = 3) -> float:
        """
        Calculate penalty/bonus for regime duration patterns.
        
        PENALTIES for:
        - Short-lived regimes: A(10h)→B(2h)→A(8h) where B is transient noise
        - Sandwiched short regimes between long regimes
        
        BONUSES for:
        - Long-duration regimes (>24h sustained regimes)
        - Very long regimes (>48h = strong structural regimes)
        - Consistent regime durations (stable pattern)
        
        Args:
            regime_labels: Regime sequence
            threshold_hours: Regimes shorter than this are penalized (default 3h)
        
        Returns:
            Net score: negative = penalty, positive = bonus (range: -0.5 to +0.3)
        """
        if len(regime_labels) < 3:
            return 0.0
        
        # Get regime durations and their corresponding regime IDs
        durations = []
        regime_ids = []
        current_regime = regime_labels[0]
        current_length = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_length += 1
            else:
                durations.append(current_length)
                regime_ids.append(current_regime)
                current_regime = regime_labels[i]
                current_length = 1
        
        # Add final regime
        durations.append(current_length)
        regime_ids.append(current_regime)
        
        if len(durations) < 3:
            return 0.0
        
        # PENALTIES and BONUSES: Analyze regime duration patterns with LINEAR functions
        total_penalty = 0.0
        total_bonus = 0.0
        total_internal_regimes = max(1, len(durations) - 2)  # Don't count first/last
        
        for i in range(1, len(durations) - 1):  # Skip first and last (edge effects)
            duration = durations[i]
            prev_duration = durations[i-1]
            next_duration = durations[i+1]
            
            # LINEAR PENALTY for short regimes (continuous, not thresholds)
            if duration < threshold_hours:
                # Base penalty: inversely proportional to duration
                # duration=1h → penalty=1.0, duration=2.9h → penalty≈0.03
                base_penalty = (threshold_hours - duration) / threshold_hours
                
                # Context multiplier: how much longer are neighboring regimes?
                neighbor_avg = (prev_duration + next_duration) / 2.0
                context_multiplier = 1.0 + min(2.0, neighbor_avg / (threshold_hours * 2))
                # If neighbors are 2x threshold (6h+): multiplier = 2.0
                # If neighbors are threshold (3h): multiplier = 1.25
                
                total_penalty += base_penalty * context_multiplier
            
            # LINEAR BONUS for long regimes (continuous scale)
            else:
                # Logarithmic-linear bonus: diminishing returns for very long regimes
                # duration=6h → 0.5, duration=12h → 1.0, duration=24h → 1.5, duration=48h → 2.0, duration=72h → 2.25
                if duration >= 6:
                    # Smooth function: log-linear with saturation
                    normalized_duration = duration / 24.0  # Normalize to 24h
                    bonus = min(2.5, 0.5 + np.log1p(normalized_duration) * 1.5)
                    total_bonus += bonus
        
        # BONUS: Regime duration consistency (continuous function)
        if len(durations) >= 3:
            duration_mean = np.mean(durations)
            duration_std = np.std(durations)
            duration_cv = duration_std / (duration_mean + 1e-9)
            
            # Linear decay: CV=0 → bonus=1.0, CV=1.0 → bonus=0.0
            consistency_bonus = max(0.0, 1.0 - duration_cv)
            total_bonus += consistency_bonus
        
        # BONUS: Clean transition diversity (continuous function)
        unique_transitions = len(set(zip(regime_ids[:-1], regime_ids[1:])))
        total_transitions = len(regime_ids) - 1
        if total_transitions > 0:
            transition_diversity = unique_transitions / total_transitions
            # Linear: diversity=1.0 → bonus=0.5, diversity=0.5 → bonus=0.0
            diversity_bonus = max(0.0, (transition_diversity - 0.5) * 1.0)
            total_bonus += diversity_bonus
        
        # Normalize penalty and bonus
        penalty = total_penalty / total_internal_regimes
        bonus = total_bonus / total_internal_regimes
        
        # Net score: positive bonus - penalty
        # Cap penalty at 0.5, bonus at 0.5 (symmetric)
        net_score = min(0.5, bonus) - min(0.5, penalty)
        
        return -net_score  # Return as penalty (negative = bonus, positive = penalty)

    def _calculate_temporal_autocorrelation_penalty(self, regime_labels: np.ndarray) -> float:
        """
        Calculate penalty/bonus based on temporal autocorrelation of regime sequence.
        
        PENALTIES for:
        - Low autocorrelation (random switching)
        - Negative autocorrelation (alternating pattern)
        
        BONUSES for:
        - Very high autocorrelation (>0.85 = exceptional persistence)
        - Multi-lag autocorrelation (regimes persist across multiple lags)
        
        This catches unstable configurations that pass other metrics but have
        regime sequences that look random/noisy.
        
        Returns:
            Net penalty score (negative values are bonuses)
        """
        if len(regime_labels) < 10:  # Need minimum length for autocorr
            return 0.0
        
        try:
            # Convert regime labels to numeric sequence
            unique_regimes = np.unique(regime_labels)
            regime_to_int = {r: i for i, r in enumerate(unique_regimes)}
            numeric_sequence = np.array([regime_to_int[r] for r in regime_labels])
            
            # Calculate lag-1 autocorrelation
            mean_val = np.mean(numeric_sequence)
            numerator = np.sum((numeric_sequence[:-1] - mean_val) * (numeric_sequence[1:] - mean_val))
            denominator = np.sum((numeric_sequence - mean_val) ** 2)
            
            if denominator > 0:
                autocorr_lag1 = numerator / denominator
            else:
                autocorr_lag1 = 0.0
            
            # Calculate lag-2 autocorrelation (additional stability check)
            autocorr_lag2 = 0.0
            if len(regime_labels) >= 12:
                numerator_lag2 = np.sum((numeric_sequence[:-2] - mean_val) * (numeric_sequence[2:] - mean_val))
                if denominator > 0:
                    autocorr_lag2 = numerator_lag2 / denominator
            
            # LINEAR BONUS for high autocorrelation (continuous function)
            # autocorr=0.7 → bonus=0.0, autocorr=1.0 → bonus=0.25
            bonus = 0.0
            if autocorr_lag1 >= 0.7:
                # Smooth linear bonus above 0.7 threshold
                bonus = (autocorr_lag1 - 0.7) / (1.0 - 0.7) * 0.25  # Scales from 0 to 0.25
            
            # BONUS: Multi-lag persistence (continuous combination)
            if autocorr_lag2 > 0.0:
                # Both lag-1 and lag-2 contribute (weighted average)
                multi_lag_score = 0.7 * autocorr_lag1 + 0.3 * autocorr_lag2
                if multi_lag_score >= 0.7:
                    # Linear bonus: multi_lag=0.7 → 0.0, multi_lag=1.0 → 0.15
                    bonus += (multi_lag_score - 0.7) / (1.0 - 0.7) * 0.15
            
            # LINEAR PENALTY for low autocorrelation (continuous function)
            penalty = 0.0
            
            if autocorr_lag1 < 0.0:
                # Negative autocorr (alternating): linear from 0.30 at -1.0 to 0.20 at 0.0
                penalty = 0.20 + abs(autocorr_lag1) * 0.10
            elif autocorr_lag1 < 0.7:
                # Below good threshold: linear penalty
                # autocorr=0.0 → 0.20, autocorr=0.3 → 0.14, autocorr=0.5 → 0.08, autocorr=0.7 → 0.0
                penalty = (0.7 - autocorr_lag1) / 0.7 * 0.20
            
            # Net score: penalty - bonus (negative = net bonus, positive = net penalty)
            net_score = penalty - bonus
            
            return net_score
            
        except Exception as e:
            # On error, return no penalty
            return 0.0
    
    def _calculate_regime_duration_bonus(self, regime_labels: np.ndarray) -> float:
        """
        Calculate bonus for long-duration, stable regimes.
        
        Rewards:
        - Average regime duration > 5 hours (stable regimes) - MORE GENEROUS
        - Very long regimes (>20 hours = structural regimes) - MORE GENEROUS
        - Consistent durations across all regimes
        
        Returns:
            Bonus score (0.0 to 1.5) - SCALED UP 5x
        """
        if len(regime_labels) < 3:
            return 0.0
        
        # Get regime durations
        durations = []
        current_regime = regime_labels[0]
        current_length = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_length += 1
            else:
                durations.append(current_length)
                current_regime = regime_labels[i]
                current_length = 1
        durations.append(current_length)
        
        if len(durations) == 0:
            return 0.0
        
        durations = np.array(durations)
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        bonus = 0.0
        
        # BONUS 1: Average duration bonus (0.0 to 0.75) - MORE GENEROUS, SCALED 5x
        # avg=5h → 0.10, avg=10h → 0.30, avg=20h → 0.50, avg=30h → 0.65, avg=50h → 0.75
        if avg_duration >= 3:  # Start rewarding at 3h (more generous)
            # More positive linear progression
            bonus += min(0.75, np.log1p(avg_duration / 3.0) * 0.40)
        
        # BONUS 2: Exceptional long regime bonus (0.0 to 0.50) - MORE GENEROUS, SCALED 5x
        # max=20h → 0.15, max=30h → 0.25, max=48h → 0.35, max=72h → 0.42, max=100h → 0.50
        if max_duration >= 15:  # Start rewarding at 15h (more generous)
            bonus += min(0.50, np.log1p(max_duration / 15.0) * 0.30)
        
        # BONUS 3: Duration consistency bonus (0.0 to 0.25) - MORE GENEROUS, SCALED 5x
        # Low coefficient of variation = stable, predictable regime durations
        if len(durations) >= 3:
            duration_std = np.std(durations)
            duration_cv = duration_std / (avg_duration + 1e-9)
            # cv=0.0 → 0.25, cv=0.3 → 0.175, cv=0.5 → 0.125, cv=1.0 → 0.0
            # More positive progression
            consistency_bonus = max(0.0, (1.0 - duration_cv) * 0.25)
            bonus += consistency_bonus
        
        return float(min(1.5, bonus))
    
    def _calculate_low_transition_bonus(self, regime_labels: np.ndarray, raw_smoothness: float) -> float:
        """
        Calculate bonus for configurations with exceptionally few transitions.
        
        Rewards:
        - High raw smoothness (>0.5 = good, >0.7 = exceptional) - MORE GENEROUS
        - Ultra-stable configurations (low transition rate) - MORE GENEROUS
        
        Returns:
            Bonus score (0.0 to 1.0) - SCALED UP 5x
        """
        if len(regime_labels) < 2:
            return 0.0
        
        bonus = 0.0
        
        # BONUS 1: Exceptional smoothness (0.0 to 0.75) - MORE GENEROUS, SCALED 5x
        # smoothness=0.5 → 0.10, smoothness=0.6 → 0.25, smoothness=0.7 → 0.40, 
        # smoothness=0.85 → 0.60, smoothness=0.95 → 0.70, smoothness=1.0 → 0.75
        if raw_smoothness >= 0.4:  # Start rewarding at 0.4 (more generous)
            # More positive linear/exponential progression
            normalized = (raw_smoothness - 0.4) / (1.0 - 0.4)
            bonus += normalized ** 0.7 * 0.75  # More accelerating bonus
        
        # BONUS 2: Ultra-stable bonus (0.0 to 0.25) - MORE GENEROUS, SCALED 5x
        # Very few transitions relative to sequence length
        transitions = np.sum(regime_labels[1:] != regime_labels[:-1])
        total_possible = len(regime_labels) - 1
        transition_rate = transitions / total_possible if total_possible > 0 else 1.0
        
        # transition_rate < 0.2 (less than 20% transitions) = good - MORE GENEROUS
        if transition_rate < 0.25:  # More generous threshold
            # Linear bonus: 0.25 → 0.0, 0.15 → 0.10, 0.05 → 0.20, 0.0 → 0.25
            ultra_stable_bonus = max(0.0, (0.25 - transition_rate) / 0.25 * 0.25)
            bonus += ultra_stable_bonus
        
        return float(min(1.0, bonus))

    def _calculate_regime_duration_distribution(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics about regime duration distribution."""
        durations = self._get_regime_durations(regime_labels)

        if len(durations) == 0:
            return {
                'mean_duration': 0.0,
                'std_duration': 0.0,
                'min_duration': 0,
                'max_duration': 0,
                'duration_stability_score': 0.0,
                'long_regime_ratio': 0.0,
                'short_regime_penalty': 1.0
            }

        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)

        # Duration stability: lower CV (coefficient of variation) is better
        cv_duration = std_duration / mean_duration if mean_duration > 0 else float('inf')
        duration_stability_score = 1.0 / (1.0 + cv_duration)

        # Long regime ratio: fraction of time spent in regimes longer than median
        median_duration = np.median(durations)
        long_regimes = np.sum(durations > median_duration)
        long_regime_ratio = long_regimes / len(durations)

        # Short regime penalty: penalize too many short regimes
        short_regime_threshold = 5  # Very short regimes
        short_regime_ratio = np.sum(durations <= short_regime_threshold) / len(durations)
        short_regime_penalty = 1.0 - min(0.5, short_regime_ratio)  # Cap penalty at 0.5

        return {
            'mean_duration': mean_duration,
            'std_duration': std_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'duration_stability_score': duration_stability_score,
            'long_regime_ratio': long_regime_ratio,
            'short_regime_penalty': short_regime_penalty
        }

    def _calculate_transition_probability_matrix(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate transition probabilities between regimes."""
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)

        if n_regimes <= 1:
            return {
                'transition_matrix': np.array([[1.0]]),
                'transition_entropy': 0.0,
                'regime_stickiness': 1.0,
                'transition_stability_score': 1.0
            }

        # Create transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))

        # Count transitions
        for i in range(len(regime_labels) - 1):
            from_regime = np.where(unique_regimes == regime_labels[i])[0][0]
            to_regime = np.where(unique_regimes == regime_labels[i + 1])[0][0]
            transition_matrix[from_regime, to_regime] += 1

        # Convert to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

        # Calculate transition entropy (lower entropy = more predictable transitions)
        transition_entropies = []
        for i in range(n_regimes):
            row_probs = transition_matrix[i, :]
            if np.sum(row_probs) > 0:
                # Normalize to ensure sum to 1
                row_probs = row_probs / np.sum(row_probs)
                entropy = -np.sum(row_probs * np.log(row_probs + 1e-10))
                transition_entropies.append(entropy)

        avg_transition_entropy = np.mean(transition_entropies) if transition_entropies else 0.0
        max_entropy = np.log(n_regimes)

        # Regime stickiness: how likely regimes are to stay the same
        diagonal_sum = np.sum(np.diag(transition_matrix))
        regime_stickiness = diagonal_sum / n_regimes

        # Transition stability score: combines low entropy and high stickiness
        entropy_score = 1.0 - (avg_transition_entropy / max_entropy)
        transition_stability_score = (entropy_score + regime_stickiness) / 2.0

        return {
            'transition_matrix': transition_matrix,
            'transition_entropy': avg_transition_entropy,
            'regime_stickiness': regime_stickiness,
            'transition_stability_score': transition_stability_score
        }

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
            
            # Add avg feature values (logic moved from _validate_regime_quality)
            features_to_check = ['spread', 'volume', 'volatility', 'momentum']
            for col in features_to_check:
                if col in regime_features.columns:
                    regime_metrics[f'avg_{col}'] = float(regime_features[col].mean())
            # *** END OF NEW BLOCK ***

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
                
                # *** NEW: Add economic target metrics ***
                target_return = QualityThresholds.ECONOMIC_TARGET_RETURN
                pct_above_target = (regime_returns > target_return).mean()
                pct_below_neg_target = (regime_returns < -target_return).mean()
                pct_target_hits = pct_above_target + pct_below_neg_target
                
                # Calculate risk-adjusted metrics
                volatility = regime_returns.std()
                mean_return = regime_returns.mean()
                
                # Risk-adjusted target hits: target hits normalized by volatility
                # Higher value = achieving targets with lower risk
                risk_adj_target_hits = pct_target_hits / (volatility + QualityThresholds.DBI_EPSILON)
                
                # Win rate: proportion of target hits that are positive (long bias)
                win_rate = pct_above_target / (pct_target_hits + QualityThresholds.DBI_EPSILON) if pct_target_hits > 0 else 0.0
                
                # Return per unit volatility (Sharpe-like but using absolute return)
                return_per_vol = abs(mean_return) / (volatility + QualityThresholds.DBI_EPSILON)
                
                # Profit factor approximation: avg winning return vs avg losing return
                winning_returns = regime_returns[regime_returns > 0]
                losing_returns = regime_returns[regime_returns < 0]
                profit_factor = (abs(winning_returns.mean()) / abs(losing_returns.mean()) 
                                if len(losing_returns) > 0 and losing_returns.mean() != 0 else np.nan)
                
                # Add return characteristics
                regime_metrics.update({
                    'mean_return': float(regime_returns.mean()),
                    'volatility': float(volatility),
                    'sharpe': float(mean_return / (volatility + QualityThresholds.DBI_EPSILON)),
                    'skewness': float(regime_returns.skew()) if hasattr(regime_returns, 'skew') else 0.0,
                    'max_drawdown': float(self._compute_max_drawdown(regime_returns)),
                    # Add new economic target metrics
                    'pct_above_target': float(pct_above_target),
                    'pct_below_neg_target': float(pct_below_neg_target),
                    'pct_target_hits': float(pct_target_hits),
                    # *** NEW: Risk-adjusted metrics ***
                    'risk_adj_target_hits': float(risk_adj_target_hits),
                    'win_rate': float(win_rate),
                    'return_per_vol': float(return_per_vol),
                    'profit_factor': float(profit_factor) if not np.isnan(profit_factor) else 0.0,
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
        
        # Log the weights being used for debugging
        tprint_info(f"📊 Using quality score weights: Temporal={QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS:.2f}, CV={QualityThresholds.WEIGHT_CV_RATIO:.2f}, Silhouette={QualityThresholds.WEIGHT_SILHOUETTE:.2f}, Balance={QualityThresholds.WEIGHT_BALANCE:.2f}, Noise={QualityThresholds.WEIGHT_NOISE_RATIO:.2f}")
        
        # 1. CV ratio (higher between/lower within is better) - PRIMARY METRIC
        # *** REVERTED: Kept original field names ***
        if metrics.within_regime_cv is not None and metrics.between_regime_cv is not None:
            # Ideal: low within, high between
            # Ratio of between/within, normalized
            cv_ratio = metrics.between_regime_cv / (metrics.within_regime_cv + QualityThresholds.DBI_EPSILON)
            # ENHANCED: Use log-scaled tanh to prevent CV ratio from dominating the score
            # tanh(log(1 + cv_ratio)) spreads values more evenly across the range
            # This prevents small changes in high CV ratios from being ignored while
            # preventing small changes in low CV ratios from dominating
            cv_normalized = np.tanh(np.log1p(cv_ratio))  # Log-scaled sigmoid normalization
            score_components.append(cv_normalized)
            weights.append(QualityThresholds.WEIGHT_CV_RATIO)
            tprint_info(f"    • CV Ratio: {cv_normalized:.4f} (raw: {cv_ratio:.2f}, weight: {QualityThresholds.WEIGHT_CV_RATIO:.2f})")
        
        # 2. Silhouette score (normalize to 0-1, already in [-1, 1]) - SECONDARY METRIC
        if metrics.silhouette_score is not None:
            silhouette_normalized = (metrics.silhouette_score + 1) / 2  # Map [-1, 1] to [0, 1]
            score_components.append(silhouette_normalized)
            weights.append(QualityThresholds.WEIGHT_SILHOUETTE)
            tprint_info(f"    • Silhouette: {silhouette_normalized:.4f} (weight: {QualityThresholds.WEIGHT_SILHOUETTE:.2f})")
        
        # 3. Temporal smoothness (already in [0, 1])
        if metrics.temporal_smoothness is not None:
            score_components.append(metrics.temporal_smoothness)
            weights.append(QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS)
            tprint_info(f"    • Temporal Smoothness: {metrics.temporal_smoothness:.4f} (weight: {QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS:.2f})")
        
        # 4. Balance score (already in [0, 1])
        if metrics.balance_score is not None:
            score_components.append(metrics.balance_score)
            weights.append(QualityThresholds.WEIGHT_BALANCE)
            tprint_info(f"    • Balance: {metrics.balance_score:.4f} (weight: {QualityThresholds.WEIGHT_BALANCE:.2f})")
        
        # 5. Noise ratio (lower is better, invert)
        noise_score = 1.0 - metrics.noise_ratio
        score_components.append(noise_score)
        weights.append(QualityThresholds.WEIGHT_NOISE_RATIO)
        tprint_info(f"    • Noise Ratio (inverted): {noise_score:.4f} (weight: {QualityThresholds.WEIGHT_NOISE_RATIO:.2f})")
        
        # Calculate weighted average
        if len(score_components) > 0:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(score_components, weights)) / total_weight
            tprint_info(f"📈 Weighted score calculation: {weighted_score:.4f} (total weight: {total_weight:.2f})")
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
            tprint_info(f"📊 Saving cluster quality metrics: {len(metrics_dict)} metrics")
            
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
            
            tprint_info(f"📊 Loaded cluster quality metrics: {len(metrics_dict)} metrics")
            
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
                                 output_dir: str = "outcomes",
                                 method_specific_config: Optional[Dict[str, Any]] = None) -> Optional[str]: # <-- 1. ADDED
        """
        Generate a comprehensive markdown report of cluster quality metrics.
        
        Args:
            metrics: ClusterQualityMetrics object
            symbol: Trading symbol or identifier
            output_dir: Output directory for the report (default: outcomes/)
            method_specific_config: Optional dict of method-specific HPs to include in the report.
            
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
            md_content = self._build_markdown_content(metrics, symbol, method_specific_config)
            
            # Write to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            tprint_success(f"✅ Report generated successfully: {report_path}")
            return str(report_path)
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate markdown report: {e}")
            return None
    
    def _build_markdown_content(self, metrics: ClusterQualityMetrics, symbol: str,
        method_specific_config: Optional[Dict[str, Any]] = None) -> str:
        """Build the markdown content for the report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # *** NEW: Get target return for report ***
        target_pct = QualityThresholds.ECONOMIC_TARGET_RETURN * 100
        
        md = f"""# Cluster Quality Assessment Report

**Symbol:** {symbol}  
**Generated:** {timestamp}  
**Quality Score:** {f'{metrics.quality_score:.4f}' if metrics.quality_score else 'N/A'}

---

## Executive Summary

This report provides a comprehensive assessment of cluster quality for {symbol}.

### Key Metrics

# --- 5. START: NEW MODULAR SECTION ---
        # Dynamically add the method-specific configuration table if provided
        if method_specific_config:
            md += "\n---\n\n## Clustering Method Configuration\n\n"
            md += "| Parameter | Value |\n"
            md += "|---|---|\n"
            for key, value in method_specific_config.items():
                # Format common values nicely
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                md += f"| {key} | {value_str} |\n"
            md += "\n"
        # --- END: NEW MODULAR SECTION ---

        md += """
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

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** {metrics.within_regime_cv:.4f if metrics.within_regime_cv else 'N/A'} ± {metrics.within_regime_cv_std:.4f if metrics.within_regime_cv_std else 'N/A'}
- **Between-Regime CV:** {metrics.between_regime_cv:.4f if metrics.between_regime_cv else 'N/A'} ± {metrics.between_regime_cv_std:.4f if metrics.between_regime_cv_std else 'N/A'}
"""
        
        # Add per-regime feature CV if available
        if metrics.per_regime_cv:
            md += "\n#### Per-Regime CV Values\n\n"
            md += "| Regime | CV |\n"
            md += "|--------|----|\n"
            for regime_id, cv in sorted(metrics.per_regime_cv.items()):
                md += f"| {regime_id} | {cv:.4f} |\n"
            md += "\n"

        # *** NEW: Section for Economic CV ***
        if metrics.economic_cv_metrics:
            md += """
### Economic Coefficient of Variation

- **Avg. Within-Regime CV (fwd_return):** {0:.4f}
- **Between-Regime CV (mean_return):** {1:.4f}
- **CV Ratio (mean_return):** {2:.4f}

""".format(
    metrics.economic_cv_metrics.get('economic_avg_within_cv_fwd_return', 0.0),
    metrics.economic_cv_metrics.get('economic_between_cv_mean_return', 0.0),
    metrics.economic_cv_metrics.get('economic_cv_ratio_mean_return', 0.0)
)
            md += "| Economic Metric | Between-Regime CV |\n"
            md += "|---|---|\n"
            for key, val in sorted(metrics.economic_cv_metrics.items()):
                if key.startswith('economic_between_cv_'):
                    metric_name = key.replace('economic_between_cv_', '')
                    md += f"| {metric_name} | {val:.4f} |\n"
            md += "\n"

        # *** NEW: Section for Per-Category CV ***
        if metrics.feature_category_cv_metrics:
            md += """
### Per-Category Coefficient of Variation

This shows how different feature types (momentum, volume, volatility, etc.) 
contribute to regime discrimination.

"""
            md += "| Category | Within CV | Between CV | Ratio | # Features |\n"
            md += "|----------|-----------|------------|-------|------------|\n"
            for category, cv_data in sorted(metrics.feature_category_cv_metrics.items()):
                within_cv = cv_data.get('within_cv_mean', 0.0)
                within_std = cv_data.get('within_cv_std', 0.0)
                between_cv = cv_data.get('between_cv_mean', 0.0)
                between_std = cv_data.get('between_cv_std', 0.0)
                ratio = cv_data.get('cv_ratio', 0.0)
                num_feats = cv_data.get('num_features', 0)
                md += f"| {category} | {within_cv:.3f} ± {within_std:.3f} | "
                md += f"{between_cv:.3f} ± {between_std:.3f} | {ratio:.3f} | {num_feats} |\n"
            md += "\n**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.\n\n"
        
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
            md += """
---

## Temporal Analysis

"""
            md += f"- **Temporal Smoothness (Penalized):** {metrics.temporal_smoothness:.4f} (0-1, higher = fewer transitions)\n"
            if metrics.temporal_smoothness_raw is not None:
                md += f"- **Temporal Smoothness (Raw):** {metrics.temporal_smoothness_raw:.4f}\n"
            if metrics.flip_flop_ratio is not None:
                md += f"- **Flip-Flop Ratio:** {metrics.flip_flop_ratio:.4f} (rapid back-and-forth transitions)\n"
            if metrics.regime_persistence is not None:
                md += f"- **Regime Persistence:** {metrics.regime_persistence:.2f} bars (average duration)\n"
            md += "\n"
        
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
- Mean Return: {regime_data['mean_return']:.5f}
- Volatility: {regime_data['volatility']:.5f}
- Sharpe Ratio: {regime_data['sharpe']:.4f}
- Skewness: {regime_data.get('skewness', 0.0):.4f}
- Max Drawdown: {regime_data.get('max_drawdown', 0.0):.4f}

**Target-Based Metrics:**
- Pct > {target_pct:.1f}% (Longs): {regime_data.get('pct_above_target', 0.0):.2%}
- Pct < -{target_pct:.1f}% (Shorts): {regime_data.get('pct_below_neg_target', 0.0):.2%}
- Pct Target Hits: {regime_data.get('pct_target_hits', 0.0):.2%}

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: {regime_data.get('risk_adj_target_hits', 0.0):.4f}
- Win Rate (Long Bias): {regime_data.get('win_rate', 0.0):.2%}
- Return per Vol: {regime_data.get('return_per_vol', 0.0):.4f}
- Profit Factor: {regime_data.get('profit_factor', 0.0):.4f}

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

**Overall Quality Score:** {f'{metrics.quality_score:.4f}' if metrics.quality_score else 'N/A'} / 1.0

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
- **Report Version:** 1.2 (Backward Compatible)

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
