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
import csv
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from itertools import combinations

# Import fast temporal metrics if available
try:
    from ..rolling_hmm_clustering.fast_hmm_algorithms import fast_temporal_smoothness
    FAST_TEMPORAL_AVAILABLE = True
except ImportError:
    FAST_TEMPORAL_AVAILABLE = False
    logging.debug("Fast temporal metrics not available")

# Import sklearn metrics
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Import comprehensive temporal score from clustering optimization goals
try:
    from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
        calculate_comprehensive_temporal_score,
        calculate_temporal_smoothness,
        calculate_cv_ratio
    )
    COMPREHENSIVE_TEMPORAL_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_TEMPORAL_AVAILABLE = False
    logging.warning("Comprehensive temporal score functions not available")

# Optional scientific statistics utilities
try:
    from scipy import stats  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    stats = None
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - disabling t-tests/ANOVA in economic gap analysis")

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

# Import regime economic relevance analyzer
try:
    from .regime_economic_relevance_analyzer import create_regime_economic_relevance_analyzer
    ECONOMIC_ANALYZER_AVAILABLE = True
except ImportError:
    ECONOMIC_ANALYZER_AVAILABLE = False
    tprint_warning("Regime economic relevance analyzer not available")

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
    
    # Quality score weights (rebalanced to emphasize CV and Silhouette)
    WEIGHT_TEMPORAL_SMOOTHNESS = 0.12
    WEIGHT_CV_RATIO = 0.50
    WEIGHT_SILHOUETTE = 0.25
    WEIGHT_BALANCE = 0.08
    WEIGHT_NOISE_RATIO = 0.05
    CV_RATIO_SATURATION_POINT = 3.0  # Higher values keep increasing but with diminishing gains
    TEMPORAL_BASELINE = 0.90         # Only ultra-smooth regimes score above this threshold
    TEMPORAL_EXPONENT = 2.2          # Harsher exponent to flatten smoothness gains
    
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

    # NEW: Comprehensive temporal metrics (5-component score)
    comprehensive_temporal_score: Optional[float] = None
    comprehensive_temporal_breakdown: Dict[str, Any] = field(default_factory=dict)

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
    economic_gap_analysis: Dict[str, Any] = field(default_factory=dict)
    transition_insights: Dict[str, Any] = field(default_factory=dict)
    
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
    
    # Economic relevance analysis results
    economic_relevance_analysis: Dict[str, Any] = field(default_factory=dict)
    strategy_performance_metrics: Dict[str, Any] = field(default_factory=dict)
    economic_significance_test: Dict[str, Any] = field(default_factory=dict)
    economic_report_path: Optional[str] = None

    # Information Coefficient (IC) - alpha model evaluation
    information_coefficient: Dict[str, Any] = field(default_factory=dict)
    # Contains: ic_pearson, ic_spearman, ic_t_stat, ic_p_value, ic_mean, ic_std, ic_hit_rate

    # Walk-Forward Validation - time series robustness testing
    walk_forward_validation: Dict[str, Any] = field(default_factory=dict)
    # Contains: overall_accuracy, overall_sharpe, stability, n_windows, window_metrics, degradation

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
            'economic_gap_analysis': self.economic_gap_analysis,
            'transition_insights': self.transition_insights,
            
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
            'timestamp': self.timestamp,
            
            # Economic relevance analysis
            'economic_relevance_analysis': self.economic_relevance_analysis,
            'strategy_performance_metrics': self.strategy_performance_metrics,
            'economic_significance_test': self.economic_significance_test,
            'economic_report_path': self.economic_report_path,
            'information_coefficient': self.information_coefficient,
            'walk_forward_validation': self.walk_forward_validation
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
    
    def __init__(self, artifact_manager=None, enable_hardware_optimization=True, enable_vectorization=True, random_state: Optional[int] = None):
        """
        Initialize the cluster quality assessor.
        
        Args:
            artifact_manager: Optional artifact manager from BaseStep
            enable_hardware_optimization: Enable hardware optimizations
            enable_vectorization: Enable vectorized computations
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.artifact_manager = artifact_manager
        self.random_state = 42 if random_state is None else int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        
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
    
    def __del__(self):
        """Cleanup resources to prevent semaphore leaks."""
        try:
            if hasattr(self, 'hardware_manager') and self.hardware_manager is not None:
                # Close hardware manager if it has a cleanup method
                if hasattr(self.hardware_manager, 'cleanup'):
                    self.hardware_manager.cleanup()
                elif hasattr(self.hardware_manager, 'close'):
                    self.hardware_manager.close()
        except Exception:
            pass  # Ignore errors during cleanup
        
        try:
            if hasattr(self, 'vectorization_manager') and self.vectorization_manager is not None:
                # Close vectorization manager if it has a cleanup method
                if hasattr(self.vectorization_manager, 'cleanup'):
                    self.vectorization_manager.cleanup()
                elif hasattr(self.vectorization_manager, 'close'):
                    self.vectorization_manager.close()
        except Exception:
            pass  # Ignore errors during cleanup
    
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
                       temporal_sensitivity_mode: str = "standard",
                       fast_mode: bool = False,
                       standardize_for_metrics: bool = True) -> ClusterQualityMetrics:
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
            fast_mode: If True, skip expensive O(n²) calculations like silhouette scores (for HPO)

        Returns:
            ClusterQualityMetrics object with all computed metrics
        """
        if fast_mode:
            tprint_info("🔍 Starting FAST cluster quality assessment (HPO mode)")
        else:
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
        
        # Optional standardization for scale-sensitive metrics (DBI/CH/Silhouette/CV)
        features_for_metrics = features_clean
        if standardize_for_metrics:
            try:
                mu = features_clean.mean(axis=0)
                sigma = features_clean.std(axis=0, ddof=0).replace(0, QualityThresholds.DBI_EPSILON)
                features_for_metrics = (features_clean - mu) / sigma
            except Exception as e:
                tprint_warning(f"⚠️ Feature standardization failed, using raw features: {e}")
                features_for_metrics = features_clean
        
        # Calculate basic statistics
        metrics.n_regimes = len(set(regime_labels[non_noise_mask]))
        metrics.noise_ratio = np.sum(~non_noise_mask) / len(regime_labels)

        # 1. Silhouette scores (SKIP in fast mode - O(n²) complexity)
        if not fast_mode:
            try:
                metrics.silhouette_score, metrics.silhouette_per_cluster = self._calculate_silhouette_scores(
                regime_labels, features_for_metrics, non_noise_mask
                )
            except Exception as e:
                tprint_error(f"❌ Silhouette calculation failed: {e}")
        else:
            metrics.silhouette_score = 0.0

        # 2. Davies-Bouldin Index (SKIP in fast mode)
        if not fast_mode:
            try:
                metrics.davies_bouldin_score = self._calculate_dbi(
                    regime_labels, features_for_metrics, non_noise_mask
                )
            except Exception as e:
                tprint_error(f"❌ DBI calculation failed: {e}")
        else:
            metrics.davies_bouldin_score = 0.0

        # 3. Calinski-Harabasz Index
        try:
            metrics.calinski_harabasz_score = self._calculate_ch(
                regime_labels, features_for_metrics, non_noise_mask
            )
        except Exception as e:
            tprint_error(f"❌ CH calculation failed: {e}")

        # 4. CV Metrics
        try:
            (metrics.within_regime_cv, metrics.within_regime_cv_std,
                metrics.between_regime_cv, metrics.between_regime_cv_std,
                metrics.per_regime_cv) = self._calculate_cv_metrics(
                regime_labels, features_for_metrics, non_noise_mask
            )
        except Exception as e:
            tprint_error(f"❌ CV metrics failed: {e}")

        # 5. Balance metrics
        try:
            (metrics.balance_score, metrics.min_cluster_size_pct,
                metrics.max_cluster_size_pct, metrics.cluster_size_std,
                metrics.cluster_size_distribution) = self._calculate_balance_metrics(regime_labels)
        except Exception as e:
            tprint_error(f"❌ Balance metrics failed: {e}")

        # 5b. Entropy of regime distribution (occupancy entropy)
        if metrics.cluster_size_distribution:
            try:
                probs = np.array(metrics.cluster_size_distribution, dtype=float)
                total = float(probs.sum())
                if total > 0.0:
                    probs = probs / total
                    probs = probs[probs > 0]
                    if probs.size > 0:
                        entropy = -float(np.sum(probs * np.log(probs)))
                        max_entropy = float(np.log(len(probs))) if len(probs) > 0 else 1.0
                        metrics.occupancy_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            except Exception as e:
                logger.debug(f"Occupancy entropy calculation failed: {e}", exc_info=True)

        # 6. Temporal metrics
        if timestamps is not None:
            try:
                (metrics.temporal_smoothness,
                    metrics.temporal_smoothness_raw,
                    metrics.flip_flop_ratio) = self._calculate_temporal_smoothness(
                    regime_labels, timestamps, sensitivity_mode=temporal_sensitivity_mode
                )
                metrics.regime_persistence = self._calculate_regime_persistence(regime_labels)
                metrics.regime_duration_distribution = self._calculate_regime_duration_distribution(regime_labels)
                metrics.transition_probability_matrix = self._calculate_transition_probability_matrix(regime_labels)

                # High-level transition and persistence summary
                metrics.transition_insights = self._summarize_transition_insights(
                    metrics.transition_probability_matrix,
                    metrics.regime_duration_distribution,
                    metrics.flip_flop_ratio,
                    metrics.regime_persistence,
                    regime_labels,
                )

                # Comprehensive temporal score
                try:
                    comprehensive_temporal = self._calculate_comprehensive_temporal_metrics(
                        regime_labels,
                        features_clean.values,
                        forward_returns.values if forward_returns is not None else None,
                        target_mean_duration=(5, 20)
                    )
                    if comprehensive_temporal:
                        metrics.comprehensive_temporal_score = comprehensive_temporal.get('composite_temporal_score', 0.0)
                        metrics.comprehensive_temporal_breakdown = comprehensive_temporal
                except Exception as e:
                    logger.debug(f"Comprehensive temporal failed: {e}", exc_info=True)
            except Exception as e:
                tprint_error(f"❌ Temporal metrics failed: {e}")

        # 6b. Per-category CV metrics
        try:
            metrics.feature_category_cv_metrics = self._calculate_cv_metrics_by_category(
                regime_labels, features_for_metrics, non_noise_mask
            )
        except Exception as e:
            tprint_error(f"❌ Per-category CV failed: {e}")

        # 7. Per-regime metrics
        try:
            metrics.per_regime_metrics = self._calculate_per_regime_metrics(
                regime_labels, features_clean, forward_returns
            )
            metrics.regime_type_per_cluster = {
                regime_id: regime_data.get('regime_type', RegimeType.UNKNOWN.value)
                for regime_id, regime_data in metrics.per_regime_metrics.items()
            }
        except Exception as e:
            tprint_error(f"❌ Per-regime metrics failed: {e}")

        if metrics.per_regime_metrics:
            metrics.economic_gap_analysis = self._compute_economic_gap_analysis(
                metrics.per_regime_metrics,
                forward_returns=forward_returns,
                regime_labels=regime_labels
            )
        else:
            metrics.economic_gap_analysis = {}

        # 7b. Economic CV metrics
        if forward_returns is not None:
            try:
                metrics.economic_cv_metrics = self._calculate_economic_cv_metrics(
                    metrics.per_regime_metrics, forward_returns, regime_labels
                )
            except Exception as e:
                tprint_error(f"❌ Economic CV failed: {e}")

        # 7c. Information Coefficient (IC) - alpha model evaluation
        if forward_returns is not None:
            try:
                # Calculate regime labels as alpha scores (0-1 normalized)
                unique_labels = np.unique(regime_labels)
                non_noise_labels = unique_labels[unique_labels != -1]

                # Create alpha scores based on regime quality (higher quality regimes = higher scores)
                alpha_scores = np.zeros_like(regime_labels, dtype=float)
                if non_noise_labels.size > 0 and metrics.per_regime_metrics:
                    # Sort regimes by Sharpe ratio (as proxy for regime quality)
                    regime_sharpes = {}
                    for regime_id in non_noise_labels:
                        regime_data = metrics.per_regime_metrics.get(int(regime_id), {})
                        regime_sharpes[int(regime_id)] = regime_data.get('sharpe', 0.0)

                    # Normalize Sharpe ratios to [0, 1] range
                    sharpe_values = np.array(list(regime_sharpes.values()))
                    if sharpe_values.max() > sharpe_values.min():
                        sharpe_normalized = (sharpe_values - sharpe_values.min()) / (sharpe_values.max() - sharpe_values.min())
                    else:
                        sharpe_normalized = np.ones_like(sharpe_values) * 0.5

                    # Assign alpha scores based on regime quality
                    for idx, regime_id in enumerate(sorted(regime_sharpes.keys())):
                        regime_mask = regime_labels == regime_id
                        alpha_scores[regime_mask] = sharpe_normalized[idx] if idx < len(sharpe_normalized) else 0.5

                # Calculate IC
                if np.any(np.isfinite(alpha_scores)):
                    metrics.information_coefficient = self._calculate_information_coefficient(
                        alpha_scores, forward_returns.values
                    )
                    tprint_info(f"📊 IC Pearson: {metrics.information_coefficient.get('ic_pearson', 0.0):.4f} | "
                               f"IC Hit Rate: {metrics.information_coefficient.get('ic_hit_rate', 0.0):.2%}")
            except Exception as e:
                tprint_error(f"❌ Information Coefficient (IC) failed: {e}")
                metrics.information_coefficient = {}

        # 7d. Walk-Forward Validation (time series robustness)
        if forward_returns is not None and len(feature_data) >= 315:  # Need at least 252 + 63 samples
            try:
                # Use default parameters: 252 trading days for training, 63 for testing
                metrics.walk_forward_validation = self._calculate_walk_forward_validation(
                    regime_labels, feature_data, forward_returns,
                    train_size=252, test_size=63
                )
                if metrics.walk_forward_validation and 'n_windows' in metrics.walk_forward_validation:
                    wfv = metrics.walk_forward_validation
                    tprint_info(f"🔄 Walk-Forward: {wfv.get('n_windows', 0)} windows | "
                               f"Accuracy: {wfv.get('overall_accuracy', 0.0):.2%} | "
                               f"Stability: {wfv.get('stability', 0.0):.4f}")
            except Exception as e:
                tprint_error(f"❌ Walk-Forward Validation failed: {e}")
                metrics.walk_forward_validation = {}

        # 7e. Sub-period stability and stationarity tests (if returns available)
        if forward_returns is not None and SCIPY_AVAILABLE:
            try:
                subsample = self._calculate_subsample_stability(regime_labels, forward_returns)
                if subsample:
                    metrics.subsample_stability = subsample
            except Exception as e:
                self.logger.warning(f"Failed to calculate sub-period stability: {e}")

        # 8. Economic validation
        if forward_returns is not None:
            try:
                metrics.economic_validation = metrics.per_regime_metrics
            except Exception as e:
                tprint_error(f"❌ Economic validation failed: {e}")

        # 9. Predictive power
        # NOTE: This can be computationally heavy (RF + CV on full series).
        # For fast_mode (HPO / interactive diagnostics), skip the expensive
        # calculation and leave predictive_power at a neutral default.
        if forward_returns is not None and len(forward_returns) > 0:
            if fast_mode:
                metrics.predictive_power = 0.0
            else:
                try:
                    metrics.predictive_power = self._calculate_predictive_power(
                        regime_labels, forward_returns, fast_mode=fast_mode
                    )
                except Exception:
                    metrics.predictive_power = 0.0

        # 9. Economic relevance analysis (if forward_returns available)
        if forward_returns is not None and ECONOMIC_ANALYZER_AVAILABLE:
            try:
                economic_results = self.assess_economic_relevance(
                    regime_labels=regime_labels,
                    feature_data=feature_data,
                    forward_returns=forward_returns,
                    timestamps=timestamps
                )
                
                # Store economic results in metrics
                metrics.economic_relevance_analysis = economic_results
                metrics.strategy_performance_metrics = economic_results.get('strategy_performance', {})
                metrics.economic_significance_test = economic_results.get('significance_tests', {})
                metrics.economic_report_path = economic_results.get('economic_report_path')
                
                if economic_results:
                    tprint_success(f"✅ Analyse économique intégrée: {len(economic_results)} sections")
            except Exception as e:
                tprint_error(f"❌ Échec de l'analyse économique: {e}")

        # 10. Calculate overall quality score
        try:
            metrics.quality_score = self._calculate_quality_score(metrics)
        except Exception as e:
            tprint_error(f"❌ Quality score calculation failed: {e}")

        # CONSOLIDATED OUTPUT: Single summary tprint
        cv_ratio = metrics.between_regime_cv / (metrics.within_regime_cv + 1e-8) if metrics.within_regime_cv else 0
        tprint_success(
            f"✅ Quality: {metrics.quality_score:.3f} | "
            f"Regimes: {metrics.n_regimes} | "
            f"CV: {cv_ratio:.2f} (W:{metrics.within_regime_cv:.3f} B:{metrics.between_regime_cv:.3f}) | "
            f"Temporal: {metrics.temporal_smoothness:.3f} | "
            f"Balance: {metrics.balance_score:.3f}"
        )
        
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
        temporal_sensitivity_mode: str = "standard",
        fast_mode: bool = False
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
            timeframe: Timeframe string (e.g., "1h", "15m")
            min_regime_size: Minimum regime size to consider
            run_validators: Whether to run comprehensive validators
            temporal_sensitivity_mode: Temporal smoothness calculation mode
            fast_mode: If True, skip expensive O(n²) calculations for HPO
            
        Returns:
            ClusterQualityMetrics with all enhanced metrics
        """
        # ROBUST TIMEOUT PROTECTION: Add timeout mechanism for HMM quality assessment
        import threading
        import time
        import traceback
        import ctypes
        
        timeout_seconds = 45  # 45 second timeout for HMM quality assessment
        check_interval = 5   # Check every 5 seconds
        
        # Resource-aware timeout adjustment
        if HARDWARE_AVAILABLE:
            try:
                from src.utils.hardware.unified_hardware_manager import get_unified_hardware_manager
                hw_manager = get_unified_hardware_manager()
                cpu_usage = hw_manager.get_cpu_usage()
                memory_pressure = hw_manager.get_memory_pressure()
                
                # Increase timeout under high resource pressure
                if cpu_usage > 90 or memory_pressure > 0.8:
                    timeout_seconds = 60  # Extend timeout under high pressure
                    tprint_warning(f"⚠️ High resource pressure detected (CPU: {cpu_usage:.1f}%, Memory: {memory_pressure:.2f}) - Extending timeout to {timeout_seconds}s")
            except Exception:
                pass  # Fallback to default timeout
        
        # Strategy: Run HMM quality assessment in background thread with monitoring
        result_container = {'metrics': None, 'exception': None, 'completed': False}
        
        def assess_with_timeout():
            try:
                if fast_mode:
                    tprint_info("🔍 Starting FAST HMM regime quality assessment (HPO mode)")
                else:
                    tprint_info("🔍 Starting ENHANCED HMM regime quality assessment")

                # First, run standard quality assessment with fast mode if requested
                if fast_mode:
                    tprint_info("🔍 [HPO] Calling base assess_quality (primary pass)")
                metrics = self.assess_quality(
                    regime_labels=regime_labels,
                    feature_data=feature_data,
                    forward_returns=forward_returns,
                    timestamps=timestamps,
                    min_regime_size=min_regime_size,
                    temporal_sensitivity_mode=temporal_sensitivity_mode,
                    fast_mode=fast_mode
                )
                if fast_mode:
                    tprint_info("✅ [HPO] Finished base assess_quality (primary pass)")
                
                # If validators disabled, return standard metrics
                if not run_validators:
                    result_container['metrics'] = metrics
                    result_container['completed'] = True
                    return
                
                # Initialize HMM validator
                try:
                    from .hmm_regime_validators import create_hmm_regime_validator
                    validator = create_hmm_regime_validator(timeframe=timeframe)
                    tprint_success("✅ HMM regime validator initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Could not initialize HMM validator: {e}")
                    result_container['metrics'] = metrics
                    result_container['completed'] = True
                    return
                
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
                
                # VIII. ECONOMIC RELEVANCE ANALYSIS (NEW)
                # For HPO (fast_mode=True), skip the heavy economic relevance analysis
                # so that permutation-based tests only run for the final winning configuration.
                if fast_mode:
                    tprint_info("ℹ️ Skipping full economic relevance analysis in fast_mode (HPO); it will run only for the winning configuration")
                elif forward_returns is not None and ECONOMIC_ANALYZER_AVAILABLE:
                    try:
                        tprint_info("🔍 Démarrage de l'analyse de pertinence économique HMM...")
                        
                        economic_results = self.assess_economic_relevance(
                            regime_labels=regime_labels,
                            feature_data=feature_data,
                            forward_returns=forward_returns,
                            timestamps=timestamps,
                            predicted_regimes=regime_labels  # Use HMM labels as "predicted"
                        )
                        
                        # Store economic results in metrics
                        metrics.economic_relevance_analysis = economic_results
                        metrics.strategy_performance_metrics = economic_results.get('strategy_performance', {})
                        metrics.economic_significance_test = economic_results.get('significance_tests', {})
                        metrics.economic_report_path = economic_results.get('economic_report_path')
                        
                        if economic_results:
                            tprint_success(f"✅ Analyse économique HMM intégrée: {len(economic_results)} sections")
                    except Exception as e:
                        tprint_error(f"❌ Échec de l'analyse économique HMM: {e}")
                
                tprint_info("="*70)
                tprint_success("✅ COMPREHENSIVE HMM validation complete!")
                tprint_info("="*70)
                
                result_container['metrics'] = metrics
                result_container['completed'] = True
                
            except Exception as e:
                result_container['exception'] = e
                result_container['completed'] = True
                tprint_error(f"❌ HMM quality assessment failed: {e}")
                tprint_debug(f"Full traceback: {traceback.format_exc()}")
        
        # Start quality assessment in background thread
        quality_thread = threading.Thread(target=assess_with_timeout, daemon=True)
        quality_thread.start()
        
        # Monitor thread with timeout
        start_time = time.time()
        while not result_container['completed']:
            if time.time() - start_time > timeout_seconds:
                tprint_error(f"🚨 TIMEOUT: HMM quality assessment exceeded {timeout_seconds}s")
                
                # Try to terminate thread forcefully
                try:
                    # Force thread termination using ctypes
                    thread_id = quality_thread.ident
                    if thread_id:
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(thread_id),
                            ctypes.py_object(SystemError)
                        )
                        if res == 0:
                            tprint_warning("⚠️ Thread termination signal sent")
                        elif res == 1:
                            tprint_warning("⚠️ Thread already terminated")
                        else:
                            tprint_warning("⚠️ Thread termination failed")
                except Exception as e:
                    tprint_warning(f"⚠️ Thread termination failed: {e}")
                
                # Return default metrics on timeout
                if fast_mode:
                    tprint_info("🔍 Starting FAST cluster quality assessment (HPO mode)")
                else:
                    tprint_info("🔍 Starting comprehensive cluster quality assessment")
                
                # Return basic quality assessment without HMM validators
                try:
                    if fast_mode:
                        tprint_info("🔍 [HPO] Calling base assess_quality after TIMEOUT (fallback pass)")
                    default_metrics = self.assess_quality(
                        regime_labels=regime_labels,
                        feature_data=feature_data,
                        forward_returns=forward_returns,
                        timestamps=timestamps,
                        min_regime_size=min_regime_size,
                        temporal_sensitivity_mode=temporal_sensitivity_mode,
                        fast_mode=fast_mode
                    )
                    if fast_mode:
                        tprint_info("✅ [HPO] Finished base assess_quality after TIMEOUT (fallback pass)")
                    tprint_warning(f"⚠️ Returned default quality metrics due to timeout")
                    return default_metrics
                except Exception as e:
                    tprint_error(f"❌ Even default quality assessment failed: {e}")
                    # Return empty metrics as last resort
                    return ClusterQualityMetrics()
            
            # Small delay to prevent busy waiting
            time.sleep(check_interval)
        
        # Check if assessment completed successfully
        if result_container['exception'] is not None:
            tprint_error(f"❌ HMM quality assessment failed with exception: {result_container['exception']}")
            # Return basic quality assessment as fallback
            try:
                fallback_metrics = self.assess_quality(
                    regime_labels=regime_labels,
                    feature_data=feature_data,
                    forward_returns=forward_returns,
                    timestamps=timestamps,
                    min_regime_size=min_regime_size,
                    temporal_sensitivity_mode=temporal_sensitivity_mode,
                    fast_mode=fast_mode
                )
                tprint_warning("⚠️ Returned fallback quality metrics due to exception")
                return fallback_metrics
            except Exception as e:
                tprint_error(f"❌ Fallback quality assessment also failed: {e}")
                return ClusterQualityMetrics()
        
        return result_container['metrics']
    
    def _calculate_silhouette_scores(self,
                                      regime_labels: np.ndarray,
                                      features: pd.DataFrame,
                                      non_noise_mask: np.ndarray) -> Tuple[float, Dict[int, Dict[str, float]]]:
        """Calculate global and per-cluster silhouette scores."""
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]

        # Drop rows with any NaNs to satisfy sklearn requirements
        mask_valid = ~features_clean.isna().any(axis=1)
        features_valid = features_clean.loc[mask_valid]
        labels_valid = labels_clean[mask_valid.to_numpy()]

        # Need at least 2 samples and 2 distinct labels
        if len(features_valid) < 2 or len(set(labels_valid)) < 2:
            return 0.0, {}

        # Optional stratified subsampling for scalability
        MAX_SAMPLES = 10000
        if len(labels_valid) > MAX_SAMPLES:
            # Sample up to max per cluster to keep class balance
            unique_labels, counts = np.unique(labels_valid, return_counts=True)
            per_cluster_cap = max(50, int(MAX_SAMPLES / max(1, len(unique_labels))))
            sample_indices = []
            for lab in unique_labels:
                idx = np.where(labels_valid == lab)[0]
                if len(idx) > per_cluster_cap:
                    chosen = self.rng.choice(idx, size=per_cluster_cap, replace=False)
                else:
                    chosen = idx
                sample_indices.append(chosen)
            sample_indices = np.concatenate(sample_indices)
            features_sub = features_valid.iloc[sample_indices]
            labels_sub = labels_valid[sample_indices]
        else:
            features_sub = features_valid
            labels_sub = labels_valid

        try:
            # Global silhouette score
            global_silhouette = silhouette_score(features_sub, labels_sub)

            # Per-cluster silhouette scores
            silhouette_samples_scores = silhouette_samples(features_sub, labels_sub)
            per_cluster_silhouette: Dict[int, Dict[str, float]] = {}

            for cluster_id in set(labels_sub):
                cluster_mask = labels_sub == cluster_id
                cluster_scores = silhouette_samples_scores[cluster_mask]

                per_cluster_silhouette[int(cluster_id)] = {
                    'mean': float(np.mean(cluster_scores)),
                    'std': float(np.std(cluster_scores)),
                    'min': float(np.min(cluster_scores)),
                    'max': float(np.max(cluster_scores)),
                }

            return global_silhouette, per_cluster_silhouette
        except Exception as exc:
            tprint_warning(f"Silhouette calculation skipped due to error: {exc}")
            return 0.0, {}
    
    def _calculate_dbi(self,
                         regime_labels: np.ndarray,
                         features: pd.DataFrame,
                         non_noise_mask: np.ndarray) -> float:
        """Calculate Davies-Bouldin Index (lower is better)."""
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]

        # Drop rows with any NaNs to satisfy sklearn requirements
        mask_valid = ~features_clean.isna().any(axis=1)
        features_valid = features_clean.loc[mask_valid]
        labels_valid = labels_clean[mask_valid.to_numpy()]  # align with filtered features

        # Need at least 2 samples and 2 distinct labels for a meaningful DBI
        if len(features_valid) < 2 or len(set(labels_valid)) < 2:
            return float("inf")

        try:
            return davies_bouldin_score(features_valid, labels_valid)
        except Exception as exc:
            tprint_warning(f"DBI calculation skipped due to error: {exc}")
            return float("inf")
    
    def _calculate_ch(self,
                        regime_labels: np.ndarray,
                        features: pd.DataFrame,
                        non_noise_mask: np.ndarray) -> float:
        """Calculate Calinski-Harabasz Index (higher is better)."""
        features_clean = features.iloc[non_noise_mask]
        labels_clean = regime_labels[non_noise_mask]

        # Drop rows with any NaNs to satisfy sklearn requirements
        mask_valid = ~features_clean.isna().any(axis=1)
        features_valid = features_clean.loc[mask_valid]
        labels_valid = labels_clean[mask_valid.to_numpy()]

        # Need at least 2 samples and 2 distinct labels for a meaningful CH score
        if len(features_valid) < 2 or len(set(labels_valid)) < 2:
            return 0.0

        try:
            return calinski_harabasz_score(features_valid, labels_valid)
        except Exception as exc:
            tprint_warning(f"CH calculation skipped due to error: {exc}")
            return 0.0

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
                # Ensure numeric and replace infinities with NaN so shapes stay aligned
                cluster_mean = pd.to_numeric(cluster_mean, errors='coerce')
                cluster_mean = cluster_mean.replace([np.inf, -np.inf], np.nan)
                cluster_means.append(cluster_mean)
        
        between_regime_cv_mean = 0.0
        between_regime_cv_std = 0.0
        
        if len(cluster_means) > 1:
            # Build a DataFrame to align feature indices across clusters, then convert to NumPy
            cluster_means_df = pd.DataFrame(cluster_means).astype(float)
            cluster_means_array = cluster_means_df.to_numpy()
            
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
            if any(keyword in feature_lower for keyword in ['rsi', 'macd', 'momentum', 'cci', 'stoch', 'roc', 'trix', 'adx', 'price_momentum', 'momentum_']):
                categories['momentum'].append(feature)
            # Volume indicators
            elif any(keyword in feature_lower for keyword in ['volume', 'obv', 'vwap', 'mfi', 'cmf', 'vpt', 'volume_ratio', 'volume_momentum', 'volume_volatility', 'volume_trend', 'volume_price_corr']):
                categories['volume'].append(feature)
            # Volatility indicators
            elif any(keyword in feature_lower for keyword in ['volatility', 'atr', 'bb', 'bollinger', 'keltner', 'std', 'variance', 'bb_upper', 'bb_lower', 'bb_width']):
                categories['volatility'].append(feature)
            # Spread/book indicators
            elif any(keyword in feature_lower for keyword in ['spread', 'bid', 'ask', 'depth', 'book', 'hl_ratio', 'high_low', 'high_close', 'low_close', 'open_close', 'high_open', 'low_open', 'body_ratio', 'upper_shadow', 'lower_shadow']):
                categories['spread'].append(feature)
            # Microstructure
            elif any(keyword in feature_lower for keyword in ['tick', 'trades', 'order', 'imbalance', 'flow']):
                categories['microstructure'].append(feature)
            # Price-based
            elif any(keyword in feature_lower for keyword in ['price', 'close', 'open', 'high', 'low', 'ema', 'sma', 'ma_', 'returns', 'log_returns', 'price_sma', 'price_above', 'sma_crossover', 'trend_strength']):
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
        """Calculate CV metrics for economic relevance outputs."""
        if not per_regime_metrics:
            return {}

        metrics_data: Dict[str, Any] = {}

        within_cvs: List[float] = []
        non_noise_labels = [l for l in np.unique(regime_labels) if l != -1]

        for label in non_noise_labels:
            regime_mask = (regime_labels == label)
            if len(regime_mask) != len(forward_returns):
                min_length = min(len(regime_mask), len(forward_returns))
                regime_mask_aligned = regime_mask[:min_length]
                forward_returns_aligned = forward_returns[:min_length]
                regime_returns_ts = forward_returns_aligned[regime_mask_aligned].values
            else:
                regime_returns_ts = forward_returns[regime_mask].values

            if len(regime_returns_ts) > 1:
                within_cvs.append(self._calculate_cv(regime_returns_ts))

        avg_within_cv_fwd_return = np.nanmean(within_cvs) if within_cvs else np.nan
        metrics_data['economic_avg_within_cv_fwd_return'] = avg_within_cv_fwd_return

        try:
            metrics_df = pd.DataFrame.from_dict(per_regime_metrics, orient='index')
        except Exception:
            self.logger.warning("Could not create DataFrame for economic CV metrics.")
            return metrics_data

        metrics_to_compare = [
            'mean_return', 'volatility', 'sharpe',
            'pct_above_target', 'pct_below_neg_target', 'pct_target_hits',
            'max_drawdown'  # Economic CV for max drawdown between regimes
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

        between_mean_return_cv = metrics_data.get('economic_between_cv_mean_return', np.nan)
        if not np.isnan(avg_within_cv_fwd_return) and avg_within_cv_fwd_return > QualityThresholds.DBI_EPSILON:
            ratio = between_mean_return_cv / avg_within_cv_fwd_return
            metrics_data['economic_cv_ratio_mean_return'] = ratio
        else:
            metrics_data['economic_cv_ratio_mean_return'] = np.nan

        return metrics_data

    def _calculate_information_coefficient(self,
                                           alpha_scores: np.ndarray,
                                           forward_returns: np.ndarray,
                                           method: str = 'pearson') -> Dict[str, float]:
        """
        Calculate Information Coefficient (IC) between alpha predictions and forward returns.

        Information Coefficient measures the correlation between predicted alpha (alpha_scores)
        and actual forward returns. This is the gold standard for evaluating alpha models.

        Args:
            alpha_scores: Predicted alpha scores/signals (n_samples,)
            forward_returns: Actual forward returns (n_samples,)
            method: 'pearson' (linear correlation) or 'spearman' (rank correlation, more robust)

        Returns:
            Dictionary with IC metrics:
            - ic_pearson: Pearson correlation coefficient
            - ic_spearman: Spearman rank correlation coefficient
            - ic_t_stat: t-statistic for significance testing
            - ic_p_value: p-value for correlation significance
            - ic_mean: Mean IC across all predictions
            - ic_std: Standard deviation of IC
            - ic_hit_rate: Percentage of predictions with correct sign
        """
        ic_metrics: Dict[str, float] = {}

        try:
            # Ensure arrays are properly formatted
            alpha_scores = np.asarray(alpha_scores, dtype=np.float64)
            forward_returns = np.asarray(forward_returns, dtype=np.float64)

            # Align lengths
            min_len = min(len(alpha_scores), len(forward_returns))
            if min_len < 2:
                return {'error': 'Insufficient data for IC calculation'}

            alpha_scores = alpha_scores[:min_len]
            forward_returns = forward_returns[:min_len]

            # Remove NaN and infinite values
            valid_mask = np.isfinite(alpha_scores) & np.isfinite(forward_returns)
            if not valid_mask.any():
                return {'error': 'No valid data for IC calculation'}

            alpha_scores_clean = alpha_scores[valid_mask]
            forward_returns_clean = forward_returns[valid_mask]

            # Calculate Pearson IC
            if len(alpha_scores_clean) >= 2:
                pearson_corr = float(np.corrcoef(alpha_scores_clean, forward_returns_clean)[0, 1])
                ic_metrics['ic_pearson'] = pearson_corr if np.isfinite(pearson_corr) else 0.0

                # Calculate t-statistic for Pearson correlation
                if len(alpha_scores_clean) > 2 and np.isfinite(pearson_corr):
                    t_stat = pearson_corr * np.sqrt(len(alpha_scores_clean) - 2) / np.sqrt(1 - pearson_corr**2 + 1e-8)
                    ic_metrics['ic_t_stat'] = float(t_stat)

                    # Calculate p-value if scipy available
                    if SCIPY_AVAILABLE:
                        try:
                            from scipy.stats import t as t_dist
                            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), len(alpha_scores_clean) - 2))
                            ic_metrics['ic_p_value'] = float(p_value)
                        except Exception:
                            ic_metrics['ic_p_value'] = np.nan
                    else:
                        ic_metrics['ic_p_value'] = np.nan

            # Calculate Spearman IC (rank correlation - more robust to outliers)
            if len(alpha_scores_clean) >= 2 and SCIPY_AVAILABLE:
                try:
                    from scipy.stats import spearmanr
                    spearman_corr, spearman_pval = spearmanr(alpha_scores_clean, forward_returns_clean)
                    ic_metrics['ic_spearman'] = float(spearman_corr) if np.isfinite(spearman_corr) else 0.0
                    ic_metrics['ic_spearman_p_value'] = float(spearman_pval) if np.isfinite(spearman_pval) else np.nan
                except Exception:
                    ic_metrics['ic_spearman'] = 0.0

            # Calculate rolling/daily IC metrics
            # Segment the data into rolling windows to compute rolling IC
            window_size = max(20, len(alpha_scores_clean) // 10)  # At least 10 windows
            if len(alpha_scores_clean) >= window_size * 2:
                rolling_ics = []
                for i in range(0, len(alpha_scores_clean) - window_size, window_size // 2):
                    window_end = min(i + window_size, len(alpha_scores_clean))
                    window_alpha = alpha_scores_clean[i:window_end]
                    window_returns = forward_returns_clean[i:window_end]

                    if len(window_alpha) >= 2:
                        window_corr = np.corrcoef(window_alpha, window_returns)[0, 1]
                        if np.isfinite(window_corr):
                            rolling_ics.append(window_corr)

                if rolling_ics:
                    ic_metrics['ic_mean'] = float(np.mean(rolling_ics))
                    ic_metrics['ic_std'] = float(np.std(rolling_ics))
                    ic_metrics['ic_rolling_count'] = len(rolling_ics)

            # Calculate IC hit rate: proportion of correct sign predictions
            # (predictions and actual returns have the same sign)
            sign_match = (np.sign(alpha_scores_clean) * np.sign(forward_returns_clean)) > 0
            ic_metrics['ic_hit_rate'] = float(np.mean(sign_match))
            ic_metrics['ic_correct_signs'] = int(np.sum(sign_match))
            ic_metrics['ic_total_predictions'] = len(alpha_scores_clean)

            return ic_metrics

        except Exception as e:
            self.logger.warning(f"IC calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_walk_forward_validation(self,
                                          regime_labels: np.ndarray,
                                          features: pd.DataFrame,
                                          forward_returns: pd.Series,
                                          train_size: int = 252,
                                          test_size: int = 63,
                                          step_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Walk-Forward Validation to assess model robustness across time.

        Walk-Forward Validation uses a rolling window approach where:
        1. Train on past data (train_size periods)
        2. Test on future data (test_size periods)
        3. Roll forward by step_size and repeat

        This is more realistic for time series models as it avoids look-ahead bias
        and tests the model's ability to generalize to unseen future data.

        Args:
            regime_labels: Cluster assignments (n_samples,)
            features: Feature matrix (n_samples, n_features)
            forward_returns: Forward returns (n_samples,)
            train_size: Number of samples to use for training (default: 252 ≈ 1 trading year)
            test_size: Number of samples to use for testing (default: 63 ≈ 1 trading quarter)
            step_size: Number of samples to roll forward (default: test_size)

        Returns:
            Dictionary with walk-forward validation results including:
            - overall_accuracy: Mean accuracy across all test periods
            - overall_sharpe: Mean Sharpe ratio across all test periods
            - stability: Standard deviation of accuracies (lower is better/more stable)
            - n_windows: Number of rolling windows tested
            - window_metrics: List of metrics for each window
            - degradation: Performance degradation from training to testing
        """
        wfv_results: Dict[str, Any] = {}

        try:
            # Ensure data alignment
            regime_labels, features, forward_returns = self._ensure_aligned_data(
                regime_labels, features, forward_returns
            )

            n_samples = len(regime_labels)
            if n_samples < train_size + test_size:
                return {
                    'error': f'Insufficient data: {n_samples} samples < required {train_size + test_size}',
                    'n_windows': 0
                }

            # Set default step size
            if step_size is None:
                step_size = test_size

            # Collect window results
            window_metrics: List[Dict[str, Any]] = []
            accuracies: List[float] = []
            sharpes: List[float] = []
            train_accuracies: List[float] = []
            test_accuracies: List[float] = []

            window_idx = 0
            train_start = 0

            while train_start + train_size + test_size <= n_samples:
                train_end = train_start + train_size
                test_end = train_end + test_size

                # Extract training and testing data
                train_mask = np.arange(train_start, train_end)
                test_mask = np.arange(train_end, test_end)

                X_train = features.iloc[train_mask]
                y_train = regime_labels[train_mask]
                X_test = features.iloc[test_mask]
                y_test = regime_labels[test_mask]
                returns_train = forward_returns.iloc[train_mask]
                returns_test = forward_returns.iloc[test_mask]

                # Train a simple regime classifier (using majority vote on features)
                try:
                    # Calculate mean feature values per regime in training set
                    regime_means = {}
                    for regime_id in np.unique(y_train):
                        if regime_id != -1:
                            regime_mask = y_train == regime_id
                            regime_means[int(regime_id)] = X_train.iloc[regime_mask].mean().values

                    # Predict on test set using nearest centroid
                    test_predictions = []
                    for test_idx in range(len(X_test)):
                        test_sample = X_test.iloc[test_idx].values
                        # Find closest regime centroid
                        if regime_means:
                            distances = {
                                regime_id: np.linalg.norm(test_sample - mean)
                                for regime_id, mean in regime_means.items()
                            }
                            predicted_regime = min(distances, key=distances.get)
                        else:
                            predicted_regime = -1
                        test_predictions.append(predicted_regime)

                    test_predictions = np.array(test_predictions)

                    # Calculate accuracy
                    test_accuracy = float(np.mean(test_predictions == y_test))
                    accuracies.append(test_accuracy)
                    test_accuracies.append(test_accuracy)

                    # Calculate training accuracy (mirror nearest-centroid logic used for test set)
                    train_predictions = []
                    for i in range(len(X_train)):
                        train_sample = X_train.iloc[i].values
                        if regime_means:
                            distances = {
                                regime_id: np.linalg.norm(train_sample - mean)
                                for regime_id, mean in regime_means.items()
                            }
                            predicted_regime = min(distances, key=distances.get)
                        else:
                            predicted_regime = -1
                        train_predictions.append(predicted_regime)

                    train_predictions = np.array(train_predictions)
                    train_accuracy = float(np.mean(train_predictions == y_train))
                    train_accuracies.append(train_accuracy)

                    # Calculate Sharpe ratio on test set (regime-aware)
                    regime_returns = {}
                    for regime_id in np.unique(test_predictions):
                        regime_mask = test_predictions == regime_id
                        if regime_mask.any():
                            regime_rets = returns_test.iloc[regime_mask].values
                            if len(regime_rets) > 0:
                                mean_ret = float(np.mean(regime_rets))
                                vol = float(np.std(regime_rets))
                                sharpe = mean_ret / (vol + 1e-8)
                                regime_returns[int(regime_id)] = {
                                    'mean_return': mean_ret,
                                    'volatility': vol,
                                    'sharpe': sharpe,
                                    'n_samples': len(regime_rets)
                                }

                    avg_sharpe = float(np.mean([r['sharpe'] for r in regime_returns.values()])) if regime_returns else 0.0
                    sharpes.append(avg_sharpe)

                    window_metrics.append({
                        'window': window_idx,
                        'train_period': f'{train_start}-{train_end}',
                        'test_period': f'{train_end}-{test_end}',
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'accuracy_degradation': train_accuracy - test_accuracy,
                        'avg_sharpe': avg_sharpe,
                        'regime_returns': regime_returns,
                        'n_test_samples': len(test_predictions)
                    })

                except Exception as e:
                    self.logger.debug(f"Window {window_idx} processing failed: {e}")
                    continue

                # Roll forward
                train_start += step_size
                window_idx += 1

            # Compile results
            if accuracies:
                wfv_results = {
                    'n_windows': len(accuracies),
                    'overall_accuracy': float(np.mean(accuracies)),
                    'overall_accuracy_std': float(np.std(accuracies)),
                    'min_accuracy': float(np.min(accuracies)),
                    'max_accuracy': float(np.max(accuracies)),
                    'overall_sharpe': float(np.mean(sharpes)) if sharpes else 0.0,
                    'overall_sharpe_std': float(np.std(sharpes)) if len(sharpes) > 1 else 0.0,
                    'avg_train_accuracy': float(np.mean(train_accuracies)) if train_accuracies else 0.0,
                    'avg_test_accuracy': float(np.mean(test_accuracies)) if test_accuracies else 0.0,
                    'avg_degradation': float(np.mean([m['accuracy_degradation'] for m in window_metrics])) if window_metrics else 0.0,
                    'stability': float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
                    'window_metrics': window_metrics,
                    'parameters': {
                        'train_size': train_size,
                        'test_size': test_size,
                        'step_size': step_size,
                        'total_samples': n_samples
                    }
                }
            else:
                wfv_results = {
                    'error': 'No valid windows could be processed',
                    'n_windows': 0,
                    'parameters': {
                        'train_size': train_size,
                        'test_size': test_size,
                        'step_size': step_size,
                        'total_samples': n_samples
                    }
                }

            return wfv_results

        except Exception as e:
            self.logger.warning(f"Walk-forward validation failed: {e}")
            return {
                'error': str(e),
                'n_windows': 0
            }

    def _calculate_subsample_stability(
        self,
        regime_labels: np.ndarray,
        forward_returns: pd.Series,
        n_splits: int = 3,
    ) -> Dict[str, Any]:
        """Estimate stability of regime-conditioned returns across sub-periods.

        Splits the time series into contiguous segments and compares
        return distributions and per-regime mean returns between early
        and late segments.
        """
        if n_splits < 2:
            return {}

        if len(regime_labels) < n_splits * 10 or len(forward_returns) < n_splits * 10:
            # Not enough data per segment for meaningful tests
            return {}

        # Align lengths defensively
        n = min(len(regime_labels), len(forward_returns))
        labels = np.asarray(regime_labels[:n])
        returns_aligned = forward_returns.iloc[:n]

        # Build contiguous index segments
        indices = np.array_split(np.arange(n), n_splits)

        segment_stats: List[Dict[str, float]] = []
        for seg_idx in indices:
            seg_rets = returns_aligned.iloc[seg_idx]
            segment_stats.append(
                {
                    'mean_return': float(seg_rets.mean()),
                    'volatility': float(seg_rets.std()),
                    'n_samples': int(len(seg_rets)),
                }
            )

        # Stationarity proxy: KS test between early and late segments
        ks_pvalue: Optional[float]
        try:
            early_rets = returns_aligned.iloc[indices[0]]
            late_rets = returns_aligned.iloc[indices[-1]]
            if len(early_rets) > 0 and len(late_rets) > 0 and stats is not None:
                _, pval = stats.ks_2samp(early_rets.values, late_rets.values)  # type: ignore[attr-defined]
                ks_pvalue = float(pval)
            else:
                ks_pvalue = None
        except Exception:
            ks_pvalue = None

        # Per-regime mean-return shifts between early and late segments
        regime_ids = [int(l) for l in np.unique(labels) if l != -1]
        per_regime_shift: Dict[int, Dict[str, float]] = {}
        if regime_ids:
            all_idx = np.arange(n)
            early_mask_idx = np.isin(all_idx, indices[0])
            late_mask_idx = np.isin(all_idx, indices[-1])

            for rid in regime_ids:
                regime_mask = labels == rid
                early_mask = regime_mask & early_mask_idx
                late_mask = regime_mask & late_mask_idx

                early_mean = float(returns_aligned.iloc[early_mask].mean()) if early_mask.any() else np.nan
                late_mean = float(returns_aligned.iloc[late_mask].mean()) if late_mask.any() else np.nan

                if not np.isnan(early_mean) and not np.isnan(late_mean):
                    per_regime_shift[rid] = {
                        'early_mean_return': early_mean,
                        'late_mean_return': late_mean,
                        'delta_mean_return': late_mean - early_mean,
                    }

        return {
            'n_splits': int(n_splits),
            'segment_stats': segment_stats,
            'ks_pvalue_early_vs_late': ks_pvalue,
            'per_regime_mean_return_shift': per_regime_shift,
        }

    def _compute_economic_gap_analysis(
        self,
        per_regime_metrics: Dict[int, Dict[str, Any]],
        forward_returns: Optional[pd.Series] = None,
        regime_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute pairwise economic gaps and statistical tests between regimes."""
        if not per_regime_metrics:
            return {}

        try:
            summary: Dict[int, Dict[str, float]] = {}
            for regime_id, data in per_regime_metrics.items():
                if not isinstance(data, dict):
                    continue
                rid = int(regime_id)
                summary[rid] = {
                    'regime_type': data.get('regime_type', 'unknown'),
                    'mean_return': float(data.get('mean_return', 0.0) or 0.0),
                    'sharpe': float(data.get('sharpe', 0.0) or 0.0),
                    'volatility': float(data.get('volatility', 0.0) or 0.0),
                    'max_drawdown': float(data.get('max_drawdown', 0.0) or 0.0),
                    'win_rate': float(data.get('win_rate', 0.0) or 0.0),
                    'pct_target_hits': float(data.get('pct_target_hits', 0.0) or 0.0)
                }

            if not summary:
                return {}

            gap_result: Dict[str, Any] = {'per_regime_summary': summary}

            pairwise_diffs: List[Dict[str, float]] = []
            for regime_a, regime_b in combinations(sorted(summary.keys()), 2):
                a = summary[regime_a]
                b = summary[regime_b]
                pairwise_diffs.append({
                    'regime_a': regime_a,
                    'regime_b': regime_b,
                    'mean_return_spread': a['mean_return'] - b['mean_return'],
                    'sharpe_spread': a['sharpe'] - b['sharpe'],
                    'volatility_ratio': self._safe_divide(a['volatility'], b['volatility']),
                    'max_drawdown_spread': a['max_drawdown'] - b['max_drawdown']
                })

            if pairwise_diffs:
                gap_result['pairwise_differences'] = pairwise_diffs

            stats_result = self._run_economic_gap_tests(forward_returns, regime_labels)
            if stats_result:
                gap_result['statistical_tests'] = stats_result

            return gap_result
        except Exception as exc:
            tprint_warning(f"Failed to compute economic gap analysis: {exc}")
            return {}

    def _run_economic_gap_tests(
        self,
        forward_returns: Optional[pd.Series],
        regime_labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Run ANOVA and pairwise t-tests when SciPy and inputs are available."""
        if not SCIPY_AVAILABLE or stats is None or forward_returns is None or regime_labels is None:
            return {}

        returns_map = self._build_regime_returns_map(forward_returns, regime_labels)
        if len(returns_map) < 2:
            return {}

        results: Dict[str, Any] = {}
        samples = [sample for sample in returns_map.values() if sample.size >= 3]

        if len(samples) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*samples)
                results['anova'] = {
                    'statistic': float(f_stat),
                    'p_value': float(p_val),
                    'significant': bool(p_val < 0.05)
                }
            except Exception as exc:
                tprint_warning(f"ANOVA computation failed: {exc}")

        t_tests: List[Dict[str, Any]] = []
        for (regime_a, sample_a), (regime_b, sample_b) in combinations(returns_map.items(), 2):
            if sample_a.size < 3 or sample_b.size < 3:
                continue
            try:
                t_stat, p_val = stats.ttest_ind(sample_a, sample_b, equal_var=False)
                t_tests.append({
                    'regime_a': regime_a,
                    'regime_b': regime_b,
                    'statistic': float(t_stat),
                    'p_value': float(p_val),
                    'cohens_d': self._calculate_cohens_d(sample_a, sample_b),
                    'significant': bool(p_val < 0.05)
                })
            except Exception as exc:
                tprint_warning(f"t-test failed for regimes {regime_a}-{regime_b}: {exc}")

        if t_tests:
            results['t_tests'] = t_tests

        return results

    def _build_regime_returns_map(
        self,
        forward_returns: pd.Series,
        regime_labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Align forward returns with regime labels and group them per regime."""
        returns_map: Dict[int, np.ndarray] = {}
        try:
            returns_array = forward_returns.values if hasattr(forward_returns, 'values') else np.asarray(forward_returns)
        except Exception:
            returns_array = np.asarray(forward_returns)

        labels_array = np.asarray(regime_labels)
        if returns_array.ndim > 1:
            returns_array = returns_array.reshape(-1)

        if returns_array.size == 0 or labels_array.size == 0:
            return returns_map

        min_len = min(len(returns_array), len(labels_array))
        if min_len == 0:
            return returns_map

        returns_array = returns_array[:min_len]
        labels_array = labels_array[:min_len]

        valid_mask = np.isfinite(returns_array)
        returns_array = returns_array[valid_mask]
        labels_array = labels_array[valid_mask]

        for regime in np.unique(labels_array):
            if regime == -1:
                continue
            samples = returns_array[labels_array == regime]
            if samples.size > 0:
                returns_map[int(regime)] = samples.astype(np.float64)

        return returns_map

    @staticmethod
    def _calculate_cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Compute Cohen's d effect size between two samples."""
        if sample_a.size == 0 or sample_b.size == 0:
            return 0.0
        mean_diff = float(np.mean(sample_a) - np.mean(sample_b))
        var_a = np.var(sample_a, ddof=1)
        var_b = np.var(sample_b, ddof=1)
        pooled_std = np.sqrt(((sample_a.size - 1) * var_a + (sample_b.size - 1) * var_b) / max(sample_a.size + sample_b.size - 2, 1))
        if pooled_std == 0:
            return 0.0
        return float(mean_diff / pooled_std)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safely divide two floats."""
        denominator = float(denominator)
        if abs(denominator) < QualityThresholds.DBI_EPSILON:
            return 0.0
        return float(numerator) / denominator

    def _summarize_transition_insights(
        self,
        transition_data: Optional[Dict[str, Any]],
        duration_distribution: Optional[Dict[str, Any]],
        flip_flop_ratio: Optional[float],
        regime_persistence: Optional[float],
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Create a digestible summary of transition dynamics."""
        if not transition_data or 'transition_matrix' not in transition_data:
            return {}

        try:
            matrix = transition_data.get('transition_matrix')
            if matrix is None:
                return {}
            matrix = np.asarray(matrix, dtype=float)
            n_regimes = matrix.shape[0]
            unique_ids = [int(r) for r in sorted(np.unique(regime_labels)) if r != -1]
            if len(unique_ids) != n_regimes:
                unique_ids = list(range(n_regimes))

            diag_info = []
            for idx, regime_id in enumerate(unique_ids):
                diag_prob = float(matrix[idx, idx]) if idx < matrix.shape[0] else 0.0
                diag_info.append({'regime_id': regime_id, 'self_prob': diag_prob})

            hotspots = []
            for i, regime_from in enumerate(unique_ids):
                for j, regime_to in enumerate(unique_ids):
                    if i == j:
                        continue
                    prob = float(matrix[i, j])
                    if prob >= 0.1:
                        hotspots.append({
                            'from': regime_from,
                            'to': regime_to,
                            'probability': prob,
                            'interpretation': 'High-frequency transition'
                        })
            hotspots = sorted(hotspots, key=lambda x: x['probability'], reverse=True)[:5]

            persistence_summary = {
                'average_duration': float((duration_distribution or {}).get('mean_duration', 0.0)),
                'max_duration': float((duration_distribution or {}).get('max_duration', 0.0)),
                'min_duration': float((duration_distribution or {}).get('min_duration', 0.0)),
                'high_persistence_regimes': [info for info in diag_info if info['self_prob'] >= 0.65]
            }

            insights = {
                'persistence_summary': persistence_summary,
                'flip_flop_ratio': float(flip_flop_ratio or 0.0),
                'average_regime_persistence': float(regime_persistence or 0.0),
                'transition_entropy': float((transition_data or {}).get('transition_entropy', 0.0) or 0.0),
                'regime_stickiness': float((transition_data or {}).get('regime_stickiness', 0.0) or 0.0),
                'transition_stability_score': float((transition_data or {}).get('transition_stability_score', 0.0) or 0.0),
                'transition_hotspots': hotspots
            }

            return insights
        except Exception as exc:
            tprint_warning(f"Failed to summarize transition insights: {exc}")
            return {}

    def _calculate_temporal_smoothness(self, regime_labels: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None, flip_flop_weight: float = 1.0, penalty_mode: str = "effective_transitions", sensitivity_mode: str = "standard") -> Tuple[float, float, float]:
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
        
        # Count regime transitions and their magnitude
        regime_changes = np.sum(regime_labels[1:] != regime_labels[:-1])
        max_possible_changes = len(regime_labels) - 1

        # Diagnostic logging
        tprint_debug(f"  [Temporal Smoothness] Transitions: {regime_changes}/{max_possible_changes} ({regime_changes/max_possible_changes*100:.1f}%)")

        if max_possible_changes == 0:
            return 1.0, 1.0, 0.0

        # Raw smoothness score: fewer changes = higher smoothness
        if sensitivity_mode == "standard":
            smoothness_raw = 1.0 - (regime_changes / max_possible_changes)
            tprint_debug(f"  [Temporal Smoothness] Raw smoothness (standard): {smoothness_raw:.6f}")
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
        
        # Detect flip-flop patterns (A→B→A and A→B→B→A): short-lived regime excursions
        if len(regime_labels) < 3:
            flip_flops_1 = 0.0
        else:
            flip_flops_1 = float(np.sum(
                (regime_labels[:-2] == regime_labels[2:]) &
                (regime_labels[:-2] != regime_labels[1:-1])
            ))

        if len(regime_labels) < 4:
            flip_flops_2 = 0.0
        else:
            flip_flops_2 = float(np.sum(
                (regime_labels[:-3] == regime_labels[3:]) &
                (regime_labels[:-3] != regime_labels[1:-2]) &
                (regime_labels[1:-2] == regime_labels[2:-1])
            ))

        flip_flops = flip_flops_1 + flip_flops_2

        # Calculate flip-flop ratio
        flip_flop_ratio = flip_flops / max_possible_changes if max_possible_changes > 0 else 0.0
        tprint_debug(f"  [Temporal Smoothness] Flip-flops: {flip_flops:.0f}, ratio: {flip_flop_ratio:.6f}")
        
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
        base_adjusted = np.clip(smoothness_raw - total_penalties, 0.0, 1.0)

        # Apply bonuses as proportional uplift without saturating at 1.0
        total_bonuses = max(0.0, regime_duration_bonus + low_transition_bonus)
        bonus_factor = np.clip(total_bonuses, 0.0, 1.0)
        max_lift = max(0.0, smoothness_raw - base_adjusted)
        smoothness_final = float(base_adjusted + max_lift * bonus_factor)
        smoothness_final = float(min(smoothness_final, smoothness_raw))

        # Diagnostic logging
        tprint_debug(
            f"  [Temporal Smoothness] Penalties: flip_flop={flip_flop_penalty:.6f}, "
            f"short_lived={short_lived_penalty:.6f}, autocorr={autocorr_penalty:.6f}, "
            f"total={total_penalties:.6f}"
        )
        tprint_debug(
            f"  [Temporal Smoothness] Bonuses: duration={regime_duration_bonus:.6f}, "
            f"low_trans={low_transition_bonus:.6f}, total={total_bonuses:.6f}"
        )
        tprint_debug(
            f"  [Temporal Smoothness] Final: raw={smoothness_raw:.6f}, "
            f"base_adjusted={base_adjusted:.6f}, final={smoothness_final:.6f}"
        )

        return float(smoothness_final), float(smoothness_raw), float(flip_flop_ratio)

    def _calculate_transition_weights(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate weights for transitions based on regime durations."""
        if len(regime_labels) < 2:
            return np.array([])

        # Find transition points
        transitions = np.where(regime_labels[1:] != regime_labels[:-1])[0]

        if len(transitions) == 0:
            return np.array([0.0])

        weights: list[float] = []
        for i, trans_idx in enumerate(transitions):
            # Weight based on duration of the regime being left
            if i == 0:
                regime_duration = trans_idx + 1
            else:
                regime_duration = trans_idx - transitions[i - 1]

            # Shorter regimes get higher transition weights (more disruptive)
            weight = 1.0 / max(1.0, regime_duration / 10.0)
            weights.append(weight)

        return np.array(weights)

    def _get_regime_durations(self, regime_labels: np.ndarray) -> np.ndarray:
        """Extract duration of each regime period."""
        if len(regime_labels) < 1:
            return np.array([])

        durations: list[int] = []
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
                'short_regime_penalty': 1.0,
                'mean_durations': {}
            }

        mean_duration = float(np.mean(durations))
        std_duration = float(np.std(durations))
        min_duration = int(np.min(durations))
        max_duration = int(np.max(durations))

        # Duration stability: lower CV (coefficient of variation) is better
        cv_duration = std_duration / mean_duration if mean_duration > 0 else float('inf')
        duration_stability_score = 1.0 / (1.0 + cv_duration)

        # Long regime ratio: fraction of time spent in regimes longer than median
        median_duration = float(np.median(durations))
        long_regimes = int(np.sum(durations > median_duration))
        long_regime_ratio = long_regimes / float(len(durations))

        # Short regime penalty: penalize too many short regimes
        short_regime_threshold = 5  # Very short regimes
        short_regime_ratio = np.sum(durations <= short_regime_threshold) / float(len(durations))
        short_regime_penalty = 1.0 - min(0.5, float(short_regime_ratio))  # Cap penalty at 0.5

        # Per-regime duration statistics (regime-specific persistence)
        per_regime_stats: Dict[int, Dict[str, float]] = {}
        if len(regime_labels) > 0:
            run_lengths: list[int] = []
            run_regimes: list[int] = []
            current_regime = int(regime_labels[0])
            current_length = 1

            for label in regime_labels[1:]:
                label_int = int(label)
                if label_int == current_regime:
                    current_length += 1
                else:
                    run_lengths.append(current_length)
                    run_regimes.append(current_regime)
                    current_regime = label_int
                    current_length = 1

            run_lengths.append(current_length)
            run_regimes.append(current_regime)

            for regime_id in np.unique(run_regimes):
                regime_id_int = int(regime_id)
                regime_durations = [
                    length for length, rid in zip(run_lengths, run_regimes) if rid == regime_id_int
                ]
                if not regime_durations:
                    continue
                arr = np.asarray(regime_durations, dtype=float)
                per_regime_stats[regime_id_int] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                }

        return {
            'mean_duration': mean_duration,
            'std_duration': std_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'duration_stability_score': float(duration_stability_score),
            'long_regime_ratio': float(long_regime_ratio),
            'short_regime_penalty': float(short_regime_penalty),
            'mean_durations': per_regime_stats,
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

    def _calculate_comprehensive_temporal_metrics(
        self,
        regime_labels: np.ndarray,
        features: np.ndarray,
        returns: Optional[np.ndarray] = None,
        target_mean_duration: Tuple[int, int] = (5, 20)
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive temporal quality score with 5 enhanced metrics.

        Uses the comprehensive temporal score from clustering_optimization_goals.py
        which includes:
        - Basic smoothness (30%): Penalizes rapid switching
        - Duration quality (25%): Encourages tradeable episode lengths (5-20 bars target)
        - Transition predictability (15%): Rewards predictable transitions
        - Regime persistence (15%): Rewards autocorrelation
        - Economic efficiency (15%): Rewards profitable transitions (if returns available)

        Args:
            regime_labels: Regime labels (T,)
            features: Feature matrix (T, D)
            returns: Optional return series (T,)
            target_mean_duration: Target range for mean duration (min, max) in bars

        Returns:
            Dictionary with comprehensive temporal score and components, or None if unavailable
        """
        if not COMPREHENSIVE_TEMPORAL_AVAILABLE:
            self.logger.warning("Comprehensive temporal score functions not available")
            return None

        if len(regime_labels) < 2 or len(features) == 0:
            return None

        try:
            # Call the comprehensive temporal score function from clustering_optimization_goals
            comprehensive_result = calculate_comprehensive_temporal_score(
                labels=regime_labels,
                features=features,
                returns=returns,
                target_mean_duration=target_mean_duration,
                use_jit=True
            )

            return comprehensive_result

        except Exception as e:
            self.logger.warning(f"Failed to calculate comprehensive temporal metrics: {e}")
            self.logger.debug(f"Comprehensive temporal error details: {e}", exc_info=True)
            return None

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
                                      forward_returns: pd.Series,
                                      fast_mode: bool = False) -> float:
        """
        Calculate predictive power: can current regime predict future returns?

        In fast_mode: Uses simple mean separation metric (10-20x faster)
        In normal mode: Uses Logistic Regression with 3-fold CV (balanced speed/accuracy)

        Args:
            regime_labels: Regime/cluster labels
            forward_returns: Forward returns series (must be aligned with regime_labels)
            fast_mode: If True, use fast approximation instead of ML model

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

            # FAST MODE: Use simple mean-based metric (10-20x faster)
            if fast_mode:
                # Calculate mean returns per regime for up/down moves
                regime_return_means = []
                for regime_id in np.unique(X.flatten()):
                    mask = X.flatten() == regime_id
                    if np.sum(mask) >= 10:  # At least 10 samples
                        regime_mean = np.mean(y[mask])
                        regime_return_means.append(abs(regime_mean - 0.5))  # Deviation from random

                if len(regime_return_means) == 0:
                    return 0.0

                # Normalize to 0-1 range (higher separation = better predictive power)
                separation = np.mean(regime_return_means) * 2  # Scale to approximate CV score
                return float(np.clip(separation, 0.0, 1.0))

            # NORMAL MODE: Use Logistic Regression with limited CV (5-10x faster than RF)
            # Calculate safe number of CV folds - use minimum for speed
            min_samples_per_fold = QualityThresholds.MIN_SAMPLES_PER_CV_FOLD
            max_folds = min(3, max(2, len(y) // min_samples_per_fold))  # Cap at 3 folds

            if max_folds < 2:
                return 0.0

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=100, random_state=self.random_state, solver='lbfgs')
            skf = StratifiedKFold(n_splits=max_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(clf, X, y, cv=skf)

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
            # Ratio of between/within, normalized with soft saturation that keeps rewarding higher ratios
            cv_ratio = metrics.between_regime_cv / (metrics.within_regime_cv + QualityThresholds.DBI_EPSILON)
            saturation = max(QualityThresholds.CV_RATIO_SATURATION_POINT, QualityThresholds.DBI_EPSILON)
            cv_normalized = 1.0 - np.exp(-cv_ratio / saturation)
            cv_normalized = float(np.clip(cv_normalized, 0.0, 1.0))
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
            adjusted = metrics.temporal_smoothness - QualityThresholds.TEMPORAL_BASELINE
            if adjusted > 0:
                span = max(1.0 - QualityThresholds.TEMPORAL_BASELINE, QualityThresholds.DBI_EPSILON)
                normalized = np.clip(adjusted / span, 0.0, 1.0)
                temporal_score = float(np.power(normalized, QualityThresholds.TEMPORAL_EXPONENT))
            else:
                temporal_score = 0.0
            score_components.append(temporal_score)
            weights.append(QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS)
            tprint_info(
                f"    • Temporal Smoothness: {metrics.temporal_smoothness:.4f} → {temporal_score:.4f} "
                f"(weight: {QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS:.2f})"
            )
        
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
    
    def assess_economic_relevance(self,
                                regime_labels: np.ndarray,
                                feature_data: pd.DataFrame,
                                forward_returns: pd.Series,
                                timestamps: Optional[pd.DatetimeIndex] = None,
                                predicted_regimes: Optional[np.ndarray] = None,
                                price_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Évalue la pertinence économique des régimes en utilisant RegimeEconomicRelevanceAnalyzer.
        
        Cette méthode analyse si la classification correcte des régimes se traduit par de meilleures
        performances de trading de manière stable et actionnable.
        
        Args:
            regime_labels: Étiquettes de régime/réseau (-1 pour le bruit)
            feature_data: Données de caractéristiques utilisées pour le clustering
            forward_returns: Rendements futurs pour la validation économique
            timestamps: Timestamps optionnels pour l'analyse temporelle
            predicted_regimes: Régimes prédits optionnels pour l'analyse comparative
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse économique
        """
        if not ECONOMIC_ANALYZER_AVAILABLE:
            tprint_warning("⚠️ RegimeEconomicRelevanceAnalyzer non disponible - analyse économique ignorée")
            return {}
        
        try:
            tprint_info("🔍 Démarrage de l'analyse de pertinence économique des régimes")
            
            # Créer une instance de RegimeEconomicRelevanceAnalyzer
            analyzer = create_regime_economic_relevance_analyzer(random_state=self.random_state)
            
            # S'assurer que les données sont alignées
            min_length = min(len(regime_labels), len(feature_data), len(forward_returns))
            if predicted_regimes is not None:
                min_length = min(min_length, len(predicted_regimes))
            
            regime_labels_aligned = regime_labels[:min_length]
            feature_data_aligned = feature_data.iloc[:min_length].reset_index(drop=True)
            forward_returns_aligned = forward_returns.iloc[:min_length].reset_index(drop=True)
            timestamps_aligned = timestamps[:min_length] if timestamps is not None else None
            predicted_regimes_aligned = predicted_regimes[:min_length] if predicted_regimes is not None else None
            
            # Build price series
            if price_series is None:
                # Fallback: attempt to reconstruct pseudo prices from forward returns
                # Start at 1.0 and cumulatively apply returns; this is inferior to true close prices
                prices_aligned = (1.0 + forward_returns_aligned.fillna(0)).cumprod()
            else:
                prices_aligned = price_series.iloc[:min_length].reset_index(drop=True)
                # Align index to features/returns timeline
                prices_aligned.index = forward_returns_aligned.index
            
            # Returns-only evaluation path
            strategies = analyzer.evaluate_strategies(
                prices=forward_returns_aligned.fillna(0.0),
                regime_labels=regime_labels_aligned,
                predicted_regimes=predicted_regimes_aligned,
                returns_input=True
            )
            
            # Effectuer les tests de signification
            # 1) Block-permutation on positions (preferred null)
            try:
                market_returns = forward_returns_aligned.fillna(0.0)
                positions_by_strategy = {name: s.positions for name, s in strategies.items() if name != 'buy_hold'}
                perm_results = analyzer.perform_significance_test(
                    strategies=strategies,
                    test_method='block_permutation',
                    market_returns=market_returns,
                    positions_by_strategy=positions_by_strategy
                )
            except Exception as e:
                tprint_warning(f"⚠️ Position-permutation significance test failed: {e}")
                perm_results = {}
            
            # 2) Bootstrap (MBB) on returns (complementary)
            boot_results = analyzer.perform_significance_test(
                strategies=strategies,
                test_method='bootstrap'
            )
            # Merge results
            significance_results = {
                'position_permutation': perm_results,
                'bootstrap': boot_results
            }
            
            # Générer le rapport économique
            report_path = analyzer.generate_economic_report(
                strategies=strategies,
                significance_results=significance_results,
                output_dir="outcomes"
            )
            
            # Sauvegarder les résultats complets
            results_path = analyzer.save_results(
                strategies=strategies,
                significance_results=significance_results,
                output_dir="outcomes"
            )
            
            # Formater les résultats pour le retour
            economic_results = {
                'strategy_performance': {name: strategy.to_dict() for name, strategy in strategies.items()},
                'significance_tests': significance_results,
                'report_path': report_path,
                'results_path': results_path
            }
            
            tprint_success("✅ Analyse de pertinence économique terminée")
            
            # Extraire et formater les résultats principaux
            formatted_results = {
                'strategy_performance': economic_results.get('strategy_performance', {}),
                'significance_tests': economic_results.get('significance_tests', {}),
                'economic_report_path': economic_results.get('report_path'),
                'regime_mapping': economic_results.get('regime_mapping', {}),
                'performance_comparison': economic_results.get('performance_comparison', {}),
                'economic_interpretation': economic_results.get('economic_interpretation', {})
            }
            
            return formatted_results
            
        except Exception as e:
            tprint_error(f"❌ Échec de l'analyse de pertinence économique: {e}")
            return {}
    
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
                                 method_specific_config: Optional[Dict[str, Any]] = None,
                                 report_prefix: Optional[str] = None) -> Optional[str]: # <-- 1. ADDED
        """
        Generate a comprehensive markdown report of cluster quality metrics.

        Args:
            metrics: ClusterQualityMetrics object
            symbol: Trading symbol or identifier
            output_dir: Output directory for the report (default: outcomes/)
            method_specific_config: Optional dict of method-specific HPs to include in the report.
            report_prefix: Optional prefix for the report filename (default: 'cluster_quality_report')

        Returns:
            Path to the generated report file, or None if failed
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename with datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = report_prefix if report_prefix else "cluster_quality_report"
            filename = f"{prefix}_{symbol}_{timestamp}.md"
            report_path = output_path / filename
            absolute_report_path = report_path.resolve()

            tprint_info(f"📝 Generating markdown report: {absolute_report_path}")
            
            # Build markdown content
            md_content = self._build_markdown_content(metrics, symbol, method_specific_config)
            
            # Write to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            tprint_success(f"✅ Report generated successfully: {absolute_report_path}")
            return str(absolute_report_path)
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate markdown report: {e}")
            return None
    
    def generate_comprehensive_csv_report(self, 
                                         metrics: ClusterQualityMetrics,
                                         all_trials: Optional[List[Dict[str, Any]]] = None,
                                         symbol: str = "UNKNOWN", 
                                         output_dir: str = "outcomes",
                                         method_specific_config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate comprehensive CSV reports with detailed metrics for all trials.
        
        Args:
            metrics: ClusterQualityMetrics object for best trial
            all_trials: List of all trial results with metrics
            symbol: Trading symbol or identifier
            output_dir: Output directory for the reports (default: outcomes/)
            method_specific_config: Optional dict of method-specific HPs
            
        Returns:
            Tuple of (quality_metrics_csv_path, trials_csv_path) or (None, None) if failed
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate 1. Quality Metrics CSV (for best trial)
            quality_csv_path = self._generate_quality_metrics_csv(metrics, symbol, output_path, timestamp, method_specific_config)
            
            # Generate 2. All Trials CSV (if available)
            trials_csv_path = None
            if all_trials:
                trials_csv_path = self._generate_all_trials_csv(all_trials, symbol, output_path, timestamp)
            
            if quality_csv_path:
                tprint_success(f"✅ Comprehensive CSV reports generated:")
                tprint_success(f"   📊 Quality Metrics: {quality_csv_path}")
                if trials_csv_path:
                    tprint_success(f"   📋 All Trials: {trials_csv_path}")
                return quality_csv_path, trials_csv_path
            else:
                return None, None
                
        except Exception as e:
            tprint_error(f"❌ Failed to generate comprehensive CSV reports: {e}")
            return None, None
    
    def _generate_quality_metrics_csv(self, metrics: ClusterQualityMetrics, symbol: str, 
                                     output_path: Path, timestamp: str,
                                     method_specific_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate detailed quality metrics CSV for the best trial."""
        
        try:
            csv_filename = f"cluster_quality_metrics_{symbol}_{timestamp}.csv"
            csv_path = output_path / csv_filename
            
            absolute_csv_path = csv_path.resolve()

            tprint_info(f"📊 Generating detailed quality metrics CSV: {absolute_csv_path}")
            
            # Prepare comprehensive CSV data
            csv_data = []
            
            # Header
            csv_data.append(['Metric Category', 'Metric Name', 'Value', 'Description', 'Interpretation'])
            
            # Core Quality Metrics
            csv_data.append(['Core Quality', 'Composite Quality Score', f"{metrics.quality_score:.6f}", 'Overall clustering quality (0-1, higher is better)', 'Excellent >0.8, Good >0.6, Fair >0.4, Poor <0.4'])
            csv_data.append(['Core Quality', 'Silhouette Score', f"{metrics.silhouette_score:.6f}", 'Cluster separation and cohesion (-1 to 1)', 'Good >0.5, Moderate >0.25, Poor <0.25'])
            csv_data.append(['Core Quality', 'Davies-Bouldin Index', f"{metrics.davies_bouldin_score:.6f}", 'Cluster similarity (lower is better)', 'Excellent <0.5, Good <1.0, Fair <2.0, Poor >2.0'])
            csv_data.append(['Core Quality', 'Calinski-Harabasz Index', f"{metrics.calinski_harabasz_score:.2f}", 'Between-cluster dispersion (higher is better)', 'Context dependent'])
            
            # Enhanced CV Metrics
            csv_data.append(['Feature Distribution', 'Within-Cluster CV', f"{metrics.within_regime_cv:.6f}" if metrics.within_regime_cv is not None else "N/A", 'Average coefficient of variation within clusters', 'Lower values indicate tighter clusters'])
            csv_data.append(['Feature Distribution', 'Between-Cluster CV', f"{metrics.between_regime_cv:.6f}" if metrics.between_regime_cv is not None else "N/A", 'Average coefficient of variation between clusters', 'Higher values indicate better separation'])
            csv_data.append(['Feature Distribution', 'Within-Cluster CV Std', f"{metrics.within_regime_cv_std:.6f}" if metrics.within_regime_cv_std is not None else "N/A", 'Standard deviation of within-cluster CV', 'Lower values indicate more consistent clusters'])
            csv_data.append(['Feature Distribution', 'Between-Cluster CV Std', f"{metrics.between_regime_cv_std:.6f}" if metrics.between_regime_cv_std is not None else "N/A", 'Standard deviation of between-cluster CV', 'Lower values indicate more consistent separation'])
            
            # Per-Regime CV Values
            if metrics.per_regime_cv:
                csv_data.append(['Feature Distribution', 'Per-Regime CV Values', str(metrics.per_regime_cv), 'CV values for each individual regime', 'Shows variation across different regimes'])
            
            # Per-Category CV Metrics
            if metrics.feature_category_cv_metrics:
                csv_data.append(['Feature Distribution', 'Feature Category CV Metrics', str(metrics.feature_category_cv_metrics), 'CV metrics broken down by feature category', 'Reveals which feature categories separate regimes best'])
            
            # Economic CV Metrics
            if metrics.economic_cv_metrics:
                csv_data.append(['Economic Distribution', 'Economic CV Metrics', str(metrics.economic_cv_metrics), 'Coefficient of variation for economic outcomes', 'Shows economic separation between regimes'])
            
            # Cluster Structure Metrics
            csv_data.append(['Cluster Structure', 'Number of Regimes', f"{metrics.n_regimes}", 'Total number of regimes discovered', 'Optimal range depends on data complexity'])
            csv_data.append(['Cluster Structure', 'Noise Ratio', f"{metrics.noise_ratio:.4f}", 'Ratio of noise points (-1 labels)', 'Lower values indicate cleaner clustering'])
            
            # Balance Metrics
            if metrics.balance_score is not None:
                csv_data.append(['Cluster Structure', 'Balance Score', f"{metrics.balance_score:.4f}", 'Cluster size balance (0-1, higher is better)', 'Values >0.8 indicate well-balanced clusters'])
                csv_data.append(['Cluster Structure', 'Smallest Cluster %', f"{metrics.min_cluster_size_pct:.2f}%" if metrics.min_cluster_size_pct is not None else "N/A", 'Smallest cluster as percentage of total', 'Values <5% may indicate noise clusters'])
                csv_data.append(['Cluster Structure', 'Largest Cluster %', f"{metrics.max_cluster_size_pct:.2f}%" if metrics.max_cluster_size_pct is not None else "N/A", 'Largest cluster as percentage of total', 'Values >80% indicate dominance'])
                csv_data.append(['Cluster Structure', 'Cluster Size Std Dev', f"{metrics.cluster_size_std:.4f}" if metrics.cluster_size_std is not None else "N/A", 'Standard deviation of cluster sizes', 'Lower values indicate more balanced clusters'])
            
            # Cluster Size Distribution
            if metrics.cluster_size_distribution:
                csv_data.append(['Cluster Structure', 'Cluster Size Distribution', str(metrics.cluster_size_distribution), 'Size of each cluster as percentage', 'Detailed distribution across all clusters'])
            
            # Temporal Metrics
            csv_data.append(['Temporal Analysis', 'Temporal Smoothness', f"{metrics.temporal_smoothness:.6f}" if metrics.temporal_smoothness is not None else "N/A", 'Regime persistence over time (0-1)', 'High >0.8, Medium >0.6, Low <0.6'])
            if metrics.temporal_smoothness_raw is not None:
                csv_data.append(['Temporal Analysis', 'Temporal Smoothness (Raw)', f"{metrics.temporal_smoothness_raw:.6f}", 'Temporal smoothness without flip-flop penalty', 'Raw measure of regime persistence'])
            if metrics.flip_flop_ratio is not None:
                csv_data.append(['Temporal Analysis', 'Flip-Flop Ratio', f"{metrics.flip_flop_ratio:.4f}", 'Ratio of rapid back-and-forth transitions', 'Lower values indicate more stable regimes'])
            csv_data.append(['Temporal Analysis', 'Regime Persistence', f"{metrics.regime_persistence:.2f}" if metrics.regime_persistence is not None else "N/A", 'Average regime duration in time periods', 'Longer durations indicate more stable regimes'])
            
            # Enhanced Temporal Metrics
            if metrics.regime_duration_distribution:
                csv_data.append(['Temporal Analysis', 'Regime Duration Distribution', str(metrics.regime_duration_distribution), 'Statistical distribution of regime durations', 'Shows stability and predictability of regimes'])
            if metrics.transition_probability_matrix:
                csv_data.append(['Temporal Analysis', 'Transition Probability Matrix', str(metrics.transition_probability_matrix), 'Transition probabilities between regimes', 'Reveals regime switching patterns'])
            
            # Economic Metrics
            if metrics.economic_validation:
                econ_val = metrics.economic_validation
                csv_data.append(['Economic Validation', 'Economic Validation Results', str(econ_val), 'Complete economic validation metrics', 'Includes returns, Sharpe, drawdown, hit rate'])
                
                # Extract individual economic metrics if available
                if isinstance(econ_val, dict):
                    for regime_id, regime_data in econ_val.items():
                        if isinstance(regime_data, dict):
                            regime_metrics = []
                            for key, value in regime_data.items():
                                if key in ['mean_return', 'volatility', 'sharpe', 'max_drawdown', 'hit_rate']:
                                    regime_metrics.append(f"{key}:{value:.4f}")
                            if regime_metrics:
                                csv_data.append(['Economic Validation', f'Regime {regime_id} Metrics', '; '.join(regime_metrics), f'Economic metrics for regime {regime_id}', 'Performance characteristics by regime'])
            
            # Predictive Power
            if metrics.predictive_power is not None:
                csv_data.append(['Predictive Power', 'Predictive Power Score', f"{metrics.predictive_power:.4f}", 'Cross-validation prediction accuracy (0-1)', 'Higher values indicate better predictive capability'])
            
            # Model-Specific Metrics
            if metrics.log_likelihood is not None:
                csv_data.append(['Model Metrics', 'Log Likelihood', f"{metrics.log_likelihood:.2f}", 'Model log-likelihood (higher is better)', 'Measures model fit to data'])
            
            # Enhanced HMM Metrics (if available)
            if metrics.rolling_predictive_ll:
                csv_data.append(['HMM Validation', 'Rolling Predictive LL', str(metrics.rolling_predictive_ll), 'Rolling log-likelihood validation results', 'Assesses model generalization'])
            if metrics.refit_stability_ari is not None:
                csv_data.append(['HMM Validation', 'Refit Stability ARI', f"{metrics.refit_stability_ari:.4f}", 'Adjusted Rand Index across refits', 'Higher values indicate more stable clustering'])
            if metrics.state_occupancy:
                csv_data.append(['HMM Validation', 'State Occupancy', str(metrics.state_occupancy), 'Fraction of time in each state', 'Shows regime dominance patterns'])
            if metrics.expected_state_durations:
                csv_data.append(['HMM Validation', 'Expected State Durations', str(metrics.expected_state_durations), 'Expected duration for each state', 'Predictive regime persistence measure'])
            
            # Economic Relevance Analysis (NEW)
            if metrics.economic_relevance_analysis:
                csv_data.append(['Economic Relevance', 'Economic Analysis Available', 'Yes', 'Economic relevance analysis was performed', 'Provides trading performance insights'])
                
                # Strategy Performance Summary
                if metrics.strategy_performance_metrics:
                    strategy_perf = metrics.strategy_performance_metrics
                    
                    if 'regime_based_strategy' in strategy_perf:
                        regime_strategy = strategy_perf['regime_based_strategy']
                        csv_data.append(['Economic Relevance', 'Regime Strategy Sharpe', f"{regime_strategy.get('sharpe_ratio', 0.0):.4f}", 'Sharpe ratio of regime-based strategy', 'Higher is better'])
                        csv_data.append(['Economic Relevance', 'Regime Strategy Return', f"{regime_strategy.get('total_return', 0.0):.2%}", 'Total return of regime-based strategy', 'Higher is better'])
                        csv_data.append(['Economic Relevance', 'Regime Strategy Max DD', f"{regime_strategy.get('max_drawdown', 0.0):.2%}", 'Maximum drawdown of regime-based strategy', 'Lower is better'])
                        csv_data.append(['Economic Relevance', 'Regime Strategy Win Rate', f"{regime_strategy.get('win_rate', 0.0):.2%}", 'Win rate of regime-based strategy', 'Higher is better'])
                    
                    if 'buy_and_hold' in strategy_perf:
                        bh_strategy = strategy_perf['buy_and_hold']
                        csv_data.append(['Economic Relevance', 'Buy & Hold Sharpe', f"{bh_strategy.get('sharpe_ratio', 0.0):.4f}", 'Sharpe ratio of buy & hold strategy', 'Higher is better'])
                        csv_data.append(['Economic Relevance', 'Buy & Hold Return', f"{bh_strategy.get('total_return', 0.0):.2%}", 'Total return of buy & hold strategy', 'Higher is better'])
                        csv_data.append(['Economic Relevance', 'Buy & Hold Max DD', f"{bh_strategy.get('max_drawdown', 0.0):.2%}", 'Maximum drawdown of buy & hold strategy', 'Lower is better'])
                    
                    if 'performance_comparison' in strategy_perf:
                        comparison = strategy_perf['performance_comparison']
                        csv_data.append(['Economic Relevance', 'Sharpe Uplift vs B&H', f"{comparison.get('sharpe_uplift', 0.0):.2%}", 'Sharpe ratio improvement over buy & hold', 'Positive means outperformance'])
                        csv_data.append(['Economic Relevance', 'Return Uplift vs B&H', f"{comparison.get('return_uplift', 0.0):.2%}", 'Total return improvement over buy & hold', 'Positive means outperformance'])
                        csv_data.append(['Economic Relevance', 'Outperformance Frequency', f"{comparison.get('outperformance_frequency', 0.0):.2%}", 'Frequency of outperforming buy & hold', 'Higher is better'])

                    # Per-regime long/short "only in regime k" strategies
                    regime_strategy_keys = [
                        key for key in strategy_perf.keys()
                        if isinstance(key, str) and key.startswith('regime_')
                    ]
                    if regime_strategy_keys:
                        for key in sorted(regime_strategy_keys):
                            strat = strategy_perf.get(key, {})
                            metrics_dict = strat.get('metrics', {}) if isinstance(strat, dict) else {}
                            name = strat.get('name', key) if isinstance(strat, dict) else key
                            total_ret = metrics_dict.get('total_return', 0.0)
                            sharpe = metrics_dict.get('sharpe_ratio', 0.0)
                            max_dd = metrics_dict.get('max_drawdown', 0.0)

                            csv_data.append([
                                'Economic Relevance',
                                f'{name} - Total Return',
                                f"{total_ret:.4f}",
                                'Total return of per-regime long/short strategy',
                                'Higher is better'
                            ])
                            csv_data.append([
                                'Economic Relevance',
                                f'{name} - Sharpe Ratio',
                                f"{sharpe:.4f}",
                                'Sharpe ratio of per-regime long/short strategy',
                                'Higher is better'
                            ])
                            csv_data.append([
                                'Economic Relevance',
                                f'{name} - Max Drawdown',
                                f"{max_dd:.4f}",
                                'Maximum drawdown of per-regime long/short strategy',
                                'Lower is better'
                            ])
                
                # Significance Tests
                if metrics.economic_significance_test:
                    significance = metrics.economic_significance_test
                    
                    if 'permutation_test' in significance:
                        perm_test = significance['permutation_test']
                        csv_data.append(['Economic Relevance', 'Permutation P-value', f"{perm_test.get('p_value', 1.0):.4f}", 'P-value from permutation test', 'Lower < 0.05 indicates significance'])
                        csv_data.append(['Economic Relevance', 'Permutation Test Statistic', f"{perm_test.get('test_statistic', 0.0):.4f}", 'Test statistic from permutation test', 'Higher absolute value indicates stronger effect'])
                        csv_data.append(['Economic Relevance', 'Permutation Is Significant', str(perm_test.get('is_significant', False)), 'Whether permutation test is significant', 'True indicates statistically significant outperformance'])
                    
                    if 'bootstrap_test' in significance:
                        boot_test = significance['bootstrap_test']
                        csv_data.append(['Economic Relevance', 'Bootstrap CI Lower', f"{boot_test.get('ci_lower', 0.0):.4f}", 'Lower bound of bootstrap confidence interval', '95% confidence interval lower bound'])
                        csv_data.append(['Economic Relevance', 'Bootstrap CI Upper', f"{boot_test.get('ci_upper', 0.0):.4f}", 'Upper bound of bootstrap confidence interval', '95% confidence interval upper bound'])
                        csv_data.append(['Economic Relevance', 'Bootstrap Mean', f"{boot_test.get('bootstrap_mean', 0.0):.4f}", 'Mean of bootstrap distribution', 'Central estimate of performance'])
                        csv_data.append(['Economic Relevance', 'Bootstrap Is Significant', str(boot_test.get('is_significant', False)), 'Whether bootstrap test is significant', 'True indicates statistically significant outperformance'])
                
                # Economic Report Path
                if metrics.economic_report_path:
                    csv_data.append(['Economic Relevance', 'Economic Report Path', metrics.economic_report_path, 'Path to detailed economic report', 'Contains comprehensive economic analysis'])
            
            # Method-Specific Configuration
            if method_specific_config:
                csv_data.append(['Configuration', 'Symbol', symbol, 'Trading symbol or identifier', ''])
                csv_data.append(['Configuration', 'Analysis Timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'When the analysis was performed', ''])
                for param, value in method_specific_config.items():
                    csv_data.append(['Configuration', param, str(value), 'Method-specific hyperparameter', ''])
            
            # Write CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            tprint_success(f"✅ Quality metrics CSV generated: {absolute_csv_path}")
            return str(absolute_csv_path)
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate quality metrics CSV: {e}")
            return None
    
    def _generate_all_trials_csv(self, all_trials: List[Dict[str, Any]], symbol: str, 
                                output_path: Path, timestamp: str) -> Optional[str]:
        """Generate comprehensive CSV with all trial results."""
        
        try:
            csv_filename = f"all_trials_results_{symbol}_{timestamp}.csv"
            csv_path = output_path / csv_filename
            
            absolute_csv_path = csv_path.resolve()

            tprint_info(f"📋 Generating all trials CSV: {absolute_csv_path}")
            
            # Prepare comprehensive CSV data for all trials
            csv_data = []
            
            # Header
            # NOTE: PCA_Components column disabled per user request
            header = ['Trial', 'Rank', 'K', 'Base_Alpha', 'Kappa', 'N_Mixtures', 
                     'Learning_Rate', 'SVI_Iterations', 'ELBO', 'Quality_Score', 'Silhouette_Score', 
                     'Davies_Bouldin_Index', 'Calinski_Harabasz_Index', 'Within_CV', 'Between_CV', 
                     'Within_CV_Std', 'Between_CV_Std', 'Temporal_Smoothness', 'Regime_Persistence', 
                     'Balance_Score', 'N_Regimes', 'Noise_Ratio', 'Predictive_Power']
            
            # Add economic metrics if available
            economic_headers = ['Mean_Return', 'Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'Hit_Rate']
            if all_trials and any('economic_validation' in trial.get('quality_metrics', {}) for trial in all_trials):
                header.extend(economic_headers)
            
            # Add HMM validation metrics if available
            hmm_headers = ['Log_Likelihood', 'Refit_Stability_ARI', 'State_Occupancy_Entropy']
            if all_trials and any(any(key in trial.get('quality_metrics', {}) for key in ['log_likelihood', 'refit_stability_ari', 'occupancy_entropy']) for trial in all_trials):
                header.extend(hmm_headers)
            
            # Add per-regime metrics summary if available
            regime_headers = ['Min_Regime_Size', 'Max_Regime_Size', 'Regime_Size_Std']
            if all_trials and any('cluster_size_distribution' in trial.get('quality_metrics', {}) for trial in all_trials):
                header.extend(regime_headers)
            
            csv_data.append(header)
            
            # Sort trials by quality score (descending) for ranking
            sorted_trials = sorted(all_trials, 
                                 key=lambda x: x.get('quality_metrics', {}).get('quality_score', 0), 
                                 reverse=True)
            
            # Add trial data
            for rank, trial in enumerate(sorted_trials, 1):
                params = trial.get('params', {})
                metrics = trial.get('quality_metrics', {})
                
                # Extract economic summary if available
                econ_summary = self._extract_economic_summary(metrics.get('economic_validation', {}))
                
                row = [
                    trial.get('trial_number', rank),
                    rank,                    params.get('n_components', 'N/A'),
                    'N/A',  # Non applicable pour le HMM roulant
                    params.get('kappa', 'N/A'),
                    params.get('n_components', 'N/A'),
                    # params.get('pca_components', 'N/A'),  # Disabled per user request                    'N/A',  # Non applicable pour le HMM roulant
                    'N/A',  # Non applicable pour le HMM roulant
                    'N/A',  # Non applicable pour le HMM roulant
                    metrics.get('quality_score', 'N/A'),
                    metrics.get('silhouette_score', 'N/A'),
                    metrics.get('davies_bouldin_score', 'N/A'),
                    metrics.get('calinski_harabasz_score', 'N/A'),
                    metrics.get('within_regime_cv', 'N/A'),
                    metrics.get('between_regime_cv', 'N/A'),
                    metrics.get('within_regime_cv_std', 'N/A'),
                    metrics.get('between_regime_cv_std', 'N/A'),
                    metrics.get('temporal_smoothness', 'N/A'),
                    metrics.get('regime_persistence', 'N/A'),
                    metrics.get('balance_score', 'N/A'),
                    metrics.get('n_regimes', 'N/A'),
                    metrics.get('noise_ratio', 'N/A'),
                    metrics.get('predictive_power', 'N/A')
                ]
                
                # Add economic metrics if available
                if 'economic_validation' in metrics:
                    row.extend([
                        econ_summary.get('mean_return', 'N/A'),
                        econ_summary.get('volatility', 'N/A'),
                        econ_summary.get('sharpe', 'N/A'),
                        econ_summary.get('max_drawdown', 'N/A'),
                        econ_summary.get('hit_rate', 'N/A')
                    ])
                else:
                    row.extend(['N/A'] * len(economic_headers))
                
                # Add HMM validation metrics if available
                row.extend([
                    metrics.get('log_likelihood', 'N/A'),
                    metrics.get('refit_stability_ari', 'N/A'),
                    metrics.get('occupancy_entropy', 'N/A')
                ])
                
                # Add regime size metrics if available
                if metrics.get('cluster_size_distribution'):
                    cluster_sizes = metrics['cluster_size_distribution']
                    row.extend([
                        min(cluster_sizes) if cluster_sizes else 'N/A',
                        max(cluster_sizes) if cluster_sizes else 'N/A',
                        float(np.std(cluster_sizes)) if cluster_sizes else 'N/A'
                    ])
                else:
                    row.extend(['N/A'] * len(regime_headers))
                
                csv_data.append(row)
            
            # Write CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            tprint_success(f"✅ All trials CSV generated: {absolute_csv_path}")
            return str(absolute_csv_path)
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate all trials CSV: {e}")
            return None
    
    def _extract_economic_summary(self, economic_validation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract summary economic metrics from validation results.
        
        Args:
            economic_validation: Economic validation results dictionary
            
        Returns:
            Dictionary with summary economic metrics
        """
        summary = {
            'mean_return': 0.0,
            'volatility': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'hit_rate': 0.0
        }
        
        try:
            if not economic_validation:
                return summary
            
            # Calculate weighted averages across all regimes
            total_weight = 0.0
            weighted_metrics = {key: 0.0 for key in summary.keys()}
            
            for regime_id, regime_data in economic_validation.items():
                if isinstance(regime_data, dict):
                    weight = regime_data.get('size', 1.0)  # Use regime size as weight
                    total_weight += weight
                    
                    for metric in weighted_metrics.keys():
                        value = regime_data.get(metric, 0.0)
                        weighted_metrics[metric] += value * weight
            
            # Calculate averages
            if total_weight > 0:
                for metric in summary.keys():
                    summary[metric] = weighted_metrics[metric] / total_weight
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract economic summary: {e}")
        
        return summary
    
    def _get_category_description(self, category: str) -> str:
        """
        Get description for feature category.
        
        Args:
            category: Feature category name
            
        Returns:
            Description of the feature category
        """
        descriptions = {
            'price': 'Price-based features including OHLCV data',
            'volume': 'Volume and flow-based indicators',
            'volatility': 'Volatility measures and risk indicators',
            'momentum': 'Momentum and trend indicators',
            'mean_reversion': 'Mean reversion and oscillation indicators',
            'microstructure': 'Market microstructure features',
            'technical': 'Technical analysis indicators',
            'statistical': 'Statistical and mathematical features',
            'economic': 'Economic and fundamental indicators',
            'sentiment': 'Sentiment and alternative data features'
        }
        return descriptions.get(category, 'Feature category for regime analysis')
    
    def _build_markdown_content(self, metrics: ClusterQualityMetrics, symbol: str,
        method_specific_config: Optional[Dict[str, Any]] = None) -> str:
        """Build the markdown content for the report."""
        # *** NEW: Get target return for report ***
        target_pct = QualityThresholds.ECONOMIC_TARGET_RETURN * 100

        md = (
            "# Cluster Quality Assessment Report\n\n"
            f"**Symbol:** {symbol}  \n"
            f"**Generated:** {metrics.timestamp}\n"
            f"**Data Points:** {getattr(metrics, 'n_samples', 'N/A')}\n"
            f"**Number of Regimes:** {metrics.n_regimes}\n"
            "**Report Version:** 1.3 (Enhanced with Financial Analysis)\n\n"
            f"This report provides a comprehensive assessment of cluster quality for {symbol}.\n\n"
            "### Key Metrics\n\n"
        )

        # --- 5. START: NEW MODULAR SECTION ---
        # Dynamically add the method-specific configuration table if provided
        if method_specific_config:
            md += "\n---\n\n## Clustering Method Configuration\n\n"
            md += "| Parameter | Value |\n"
            md += "|---|---|\n"
            for key, value in method_specific_config.items():
                # Format common values nicely
                if isinstance(value, float):
                    value_str = "{:.4f}".format(value)
                else:
                    value_str = str(value)
                md += "| {} | {} |\n".format(key, value_str)
            md += "\n"
        # --- END: NEW MODULAR SECTION ---

        md += "\n## PCA Feature Analysis\n\n"

        # Add PCA feature information if available from method-specific config
        if method_specific_config and 'pca_components' in method_specific_config:
            pca_components = method_specific_config['pca_components']
            md += """
### Principal Component Analysis Configuration

**Number of PCA Components:** {}

""".format(pca_components)
            
            # Add feature categories if available
            if 'feature_categories' in method_specific_config:
                feature_categories = method_specific_config['feature_categories']
                md += """
### Feature Categories Used in PCA

| Category | Features | Description |
|----------|----------|-------------|
"""
                
                for category, features in feature_categories.items():
                    if isinstance(features, list):
                        feature_list = ', '.join(features[:5])  # Show first 5 features
                        if len(features) > 5:
                            feature_list += " (and {} more)".format(len(features)-5)
                        description = self._get_category_description(category)
                        md += "| {} | {} | {} |\n".format(category, feature_list, description)
                
                md += "\n"
            
            # Add PCA variance explanation if available
            if 'pca_variance_ratio' in method_specific_config:
                variance_ratio = method_specific_config['pca_variance_ratio']
                if isinstance(variance_ratio, list):
                    md += """
### PCA Variance Explained

| Component | Variance Explained | Cumulative Variance |
|-----------|-------------------|-------------------|
"""
                    cumulative = 0.0
                    for i, variance in enumerate(variance_ratio[:10]):  # Show top 10 components
                        cumulative += variance
                        md += "| PC{} | {:.4f} ({:.2f}%) | {:.4f} ({:.2f}%) |\n".format(
                            i+1, variance, variance*100, cumulative, cumulative*100
                        )
                    md += "\n"
            
            # Add feature loadings if available
            if 'pca_feature_loadings' in method_specific_config:
                loadings = method_specific_config['pca_feature_loadings']
                md += """
### Top Feature Loadings by Principal Component

"""
                for pc_idx, component_loadings in enumerate(loadings[:5]):  # Show top 5 components
                    if isinstance(component_loadings, dict):
                        # Sort features by absolute loading
                        sorted_features = sorted(component_loadings.items(), 
                                               key=lambda x: abs(x[1]), reverse=True)[:5]
                        md += "**PC{} Top Features:**\n\n".format(pc_idx+1)
                        for feature, loading in sorted_features:
                            md += "- {}: {:.4f}\n".format(feature, loading)
                        md += "\n"
        
        md += "\n"
        
        # Top Configurations Analysis
        if method_specific_config:
            md += """
---

## Top Configuration Analysis

### Clustering Configuration Parameters

"""
            
            # Add clustering parameters if available
            if 'n_regimes' in method_specific_config:
                n_regimes = method_specific_config['n_regimes']
                md += "- **Number of Regimes (K):** {}\n".format(n_regimes)
            
            if 'stickiness' in method_specific_config:
                stickiness = method_specific_config['stickiness']
                md += "- **HMM Stickiness Parameter:** {:.4f}\n".format(stickiness)
            
            if 'learning_rate' in method_specific_config:
                lr = method_specific_config['learning_rate']
                md += "- **Learning Rate:** {:.6f}\n".format(lr)
            
            if 'convergence_threshold' in method_specific_config:
                conv_thresh = method_specific_config['convergence_threshold']
                md += "- **Convergence Threshold:** {:.8f}\n".format(conv_thresh)
            
            if 'max_iterations' in method_specific_config:
                max_iter = method_specific_config['max_iterations']
                md += "- **Maximum Iterations:** {}\n".format(max_iter)
            
            # Feature selection parameters
            if 'feature_selection' in method_specific_config:
                feat_sel = method_specific_config['feature_selection']
                md += "\n### Feature Selection Configuration\n\n"
                
                if 'n_features' in feat_sel:
                    md += "- **Selected Features:** {}\n".format(feat_sel['n_features'])
                
                if 'selection_method' in feat_sel:
                    method = feat_sel['selection_method']
                    md += "- **Selection Method:** {}\n".format(method)
                
                if 'feature_importance_threshold' in feat_sel:
                    threshold = feat_sel['feature_importance_threshold']
                    md += "- **Importance Threshold:** {:.6f}\n".format(threshold)
                
                if 'top_features' in feat_sel:
                    top_features = feat_sel['top_features']
                    md += "\n**Top {} Selected Features:**\n\n".format(len(top_features))
                    for i, (feature, importance) in enumerate(top_features[:10], 1):
                        md += "{:2d}. {}: {:.6f}\n".format(i, feature, importance)
                    if len(top_features) > 10:
                        md += "... and {} more features\n".format(len(top_features) - 10)
                    md += "\n"
            
            # Auto-tuning results if available
            if 'auto_tuning' in method_specific_config:
                auto_tune = method_specific_config['auto_tuning']
                md += "\n### Auto-Tuning Results\n\n"
                
                if 'best_score' in auto_tune:
                    best_score = auto_tune['best_score']
                    md += f"- **Best Optimization Score:** {best_score:.6f}\n"
                
                if 'total_trials' in auto_tune:
                    total_trials = auto_tune['total_trials']
                    md += f"- **Total Trials Run:** {total_trials}\n"
                
                if 'optimization_time' in auto_tune:
                    opt_time = auto_tune['optimization_time']
                    md += f"- **Optimization Time:** {opt_time:.2f} seconds\n"
                
                if 'parameter_space' in auto_tune:
                    param_space = auto_tune['parameter_space']
                    md += "\n**Optimized Parameter Space:**\n\n"
                    for param, values in param_space.items():
                        if isinstance(values, dict):
                            md += f"- {param}: {values.get('type', 'unknown')} range [{values.get('min', 'N/A')}, {values.get('max', 'N/A')}]\n"
                        else:
                            md += f"- {param}: {values}\n"
                    md += "\n"
                
                if 'top_trials' in auto_tune:
                    top_trials = auto_tune['top_trials']
                    md += "\n**Top 5 Configuration Trials:**\n\n"
                    md += "| Rank | Score | N_Regimes | Stickiness | Learning Rate | PCA Components |\n"
                    md += "|------|-------|------------|------------|---------------|----------------|\n"
                    
                    for i, trial in enumerate(top_trials[:5], 1):
                        score = trial.get('score', 0.0)
                        n_reg = trial.get('n_regimes', 'N/A')
                        stick = trial.get('stickiness', 'N/A')
                        lr = trial.get('learning_rate', 'N/A')
                        pca_comp = trial.get('pca_components', 'N/A')
                        md += f"| {i} | {score:.6f} | {n_reg} | {stick:.4f} | {lr:.6f} | {pca_comp} |\n"
                    md += "\n"
        
        md += """
---

## Clustering Metrics

### Silhouette Analysis
"""
        
        if metrics.silhouette_score is not None:
            md += "\n**Global Silhouette Score:** {:.4f}\n\n".format(metrics.silhouette_score)
            
            if metrics.silhouette_per_cluster:
                md += "#### Per-Cluster Silhouette Scores\n\n"
                md += "| Cluster | Mean | Std | Min | Max |\n"
                md += "|---------|------|-----|-----|-----|\n"
                
                for cluster_id, scores in sorted(metrics.silhouette_per_cluster.items()):
                    md += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                        cluster_id, scores['mean'], scores['std'], scores['min'], scores['max']
                    )
                md += "\n"
        
        # Format Davies-Bouldin Index safely
        dbi_value = "{:.4f}".format(metrics.davies_bouldin_score) if metrics.davies_bouldin_score is not None else 'N/A'
        ch_value = "{:.2f}".format(metrics.calinski_harabasz_score) if metrics.calinski_harabasz_score is not None else 'N/A'

        # Format CV values safely
        within_cv_value = "{:.4f}".format(metrics.within_regime_cv) if metrics.within_regime_cv is not None else 'N/A'
        within_cv_std_value = "{:.4f}".format(metrics.within_regime_cv_std) if metrics.within_regime_cv_std is not None else 'N/A'
        between_cv_value = "{:.4f}".format(metrics.between_regime_cv) if metrics.between_regime_cv is not None else 'N/A'
        between_cv_std_value = "{:.4f}".format(metrics.between_regime_cv_std) if metrics.between_regime_cv_std is not None else 'N/A'

        md += """
### Separation Metrics

- **Davies-Bouldin Index:** """ + str(dbi_value) + """ (lower is better)
- **Calinski-Harabasz Index:** """ + str(ch_value) + """ (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** """ + str(within_cv_value) + """ +/- """ + str(within_cv_std_value) + """
- **Between-Regime CV:** """ + str(between_cv_value) + """ +/- """ + str(between_cv_std_value) + """
"""
        
        # Add per-regime feature CV if available
        if metrics.per_regime_cv:
            md += "\n#### Per-Regime CV Values\n\n"
            md += "| Regime | CV |\n"
            md += "|--------|----|\n"
            for regime_id, cv in sorted(metrics.per_regime_cv.items()):
                md += "| {} | {:.4f} |\n".format(regime_id, cv)
            md += "\n"

        # *** NEW: Section for Economic CV ***
        if metrics.economic_cv_metrics:
            avg_within = metrics.economic_cv_metrics.get('economic_avg_within_cv_fwd_return', 0.0)
            between_mean = metrics.economic_cv_metrics.get('economic_between_cv_mean_return', 0.0)
            cv_ratio = metrics.economic_cv_metrics.get('economic_cv_ratio_mean_return', 0.0)

            md += """
### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** """ + "{:.4f}".format(avg_within) + """
- **Between-Regime CV (Mean Return):** """ + "{:.4f}".format(between_mean) + """
- **CV Ratio (Between/Within):** """ + "{:.4f}".format(cv_ratio) + """

"""
            
            for key, val in sorted(metrics.economic_cv_metrics.items()):
                if key.startswith('economic_between_cv_'):
                    metric_name = key.replace('economic_between_cv_', '')
                    md += "| {} | {:.4f} |\n".format(metric_name, val)
            md += "\n"

        # *** NEW: Economic Gap Analysis (pairwise spreads and tests) ***
        if metrics.economic_gap_analysis:
            gap = metrics.economic_gap_analysis
            md += """
---

## Economic Gap Analysis

"""
            summary = gap.get('per_regime_summary') or {}
            if summary:
                md += "### Per-Regime Snapshot\n\n"
                md += "| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |\n"
                md += "|--------|------|-------------|------------|--------|--------|-----------------|\n"
                for regime_id, row in sorted(summary.items()):
                    md += (
                        f"| {regime_id} | {row.get('regime_type', 'unknown')} | "
                        f"{row.get('mean_return', 0.0):.6f} | {row.get('volatility', 0.0):.6f} | "
                        f"{row.get('sharpe', 0.0):.4f} | {row.get('max_drawdown', 0.0):.2%} | "
                        f"{row.get('pct_target_hits', 0.0):.2%} |\n"
                    )
                md += "\n"

            pairwise = gap.get('pairwise_differences') or []
            if pairwise:
                md += "### Pairwise Economic Spreads\n\n"
                md += (
                    "| Regime A | Regime B | Mean Return Spread | Sharpe Spread | "
                    "Volatility Ratio | Max DD Spread |\n"
                )
                md += "|----------|----------|--------------------|---------------|------------------|---------------|\n"
                for row in pairwise:
                    md += (
                        f"| {row.get('regime_a')} | {row.get('regime_b')} | "
                        f"{row.get('mean_return_spread', 0.0):.6f} | {row.get('sharpe_spread', 0.0):.4f} | "
                        f"{row.get('volatility_ratio', 0.0):.3f} | {row.get('max_drawdown_spread', 0.0):.2%} |\n"
                    )
                md += "\n"

            stats_tests = gap.get('statistical_tests') or {}
            if stats_tests:
                md += "### Statistical Tests (ANOVA / t-tests)\n\n"
                anova = stats_tests.get('anova')
                if anova:
                    md += (
                        f"- **ANOVA F-statistic:** {anova.get('statistic', 0.0):.4f}, "
                        f"p-value={anova.get('p_value', 1.0):.4f} "
                        f"({'significant' if anova.get('significant') else 'ns'})\n"
                    )

                t_tests = stats_tests.get('t_tests') or []
                if t_tests:
                    md += "\n**Pairwise t-tests:**\n\n"
                    md += "| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |\n"
                    md += "|----------|----------|--------|---------|-----------|-------------|\n"
                    for row in t_tests:
                        md += (
                            f"| {row.get('regime_a')} | {row.get('regime_b')} | "
                            f"{row.get('statistic', 0.0):.4f} | {row.get('p_value', 1.0):.4f} | "
                            f"{row.get('cohens_d', 0.0):.3f} | "
                            f"{'Yes' if row.get('significant') else 'No'} |\n"
                        )
                    md += "\n"

        # *** NEW: Section for Per-Category CV ***
        if metrics.feature_category_cv_metrics:
            md += """
### Per-Category Coefficient of Variation


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

            # *** NEW: Transition & Persistence Insights (summary) ***
            if metrics.transition_insights:
                ti = metrics.transition_insights
                md += "### Transition & Persistence Insights\n\n"

                persistence = ti.get('persistence_summary') or {}
                if persistence:
                    md += (
                        f"- **Average Duration:** {persistence.get('average_duration', 0.0):.2f} bars\n"
                        f"- **Max Duration:** {persistence.get('max_duration', 0.0):.2f} bars\n"
                        f"- **Min Duration:** {persistence.get('min_duration', 0.0):.2f} bars\n"
                    )
                    high_pers = persistence.get('high_persistence_regimes') or []
                    if high_pers:
                        md += "- **High-persistence regimes:** "
                        md += ", ".join(
                            f"Regime {r.get('regime_id')} (p_self={r.get('self_prob', 0.0):.2f})"
                            for r in high_pers
                        )
                        md += "\n"

                md += f"- **Flip-flop ratio:** {ti.get('flip_flop_ratio', 0.0):.4f}\n"
                md += f"- **Average regime persistence:** {ti.get('average_regime_persistence', 0.0):.2f} bars\n"
                md += f"- **Transition entropy:** {ti.get('transition_entropy', 0.0):.4f}\n"
                md += f"- **Regime stickiness:** {ti.get('regime_stickiness', 0.0):.4f}\n"
                md += f"- **Transition stability score:** {ti.get('transition_stability_score', 0.0):.4f}\n\n"

                hotspots = ti.get('transition_hotspots') or []
                if hotspots:
                    md += "**Dominant transition hotspots:**\n\n"
                    md += "| From | To | Probability |\n|------|----|-------------|\n"
                    for h in hotspots:
                        md += (
                            f"| {h.get('from')} | {h.get('to')} | "
                            f"{h.get('probability', 0.0):.3f} |\n"
                        )
                    md += "\n"
        
        # Transition Matrix Analysis
        if hasattr(metrics, 'transition_probability_matrix') and metrics.transition_probability_matrix:
            md += """
### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:

"""
            # Create transition matrix table
            transition_matrix = metrics.transition_probability_matrix
            if transition_matrix and 'matrix' in transition_matrix:
                matrix = transition_matrix['matrix']
                regimes = sorted(matrix.keys())
                
                # Header row
                md += "| From \\ To |"
                for regime in regimes:
                    md += f" Regime {regime} |"
                md += "\n|------------|"
                for _ in regimes:
                    md += "-------------|"
                md += "\n"
                
                # Matrix rows
                for from_regime in regimes:
                    md += f"| Regime {from_regime} |"
                    for to_regime in regimes:
                        prob = matrix[from_regime].get(to_regime, 0.0)
                        md += f" {prob:.3f} ({prob*100:.1f}%) |"
                    md += "\n"
                
                md += "\n**Transition Analysis:**\n\n"
                
                # Add transition stability score if available
                if 'transition_stability_score' in transition_matrix:
                    stability = transition_matrix['transition_stability_score']
                    md += f"- **Transition Stability Score:** {stability:.3f} (higher = more stable transitions)\n"
                
                # Find most stable regimes (highest diagonal probabilities)
                diagonal_probs = []
                for regime in regimes:
                    diag_prob = matrix[regime].get(regime, 0.0)
                    diagonal_probs.append((regime, diag_prob))
                
                diagonal_probs.sort(key=lambda x: x[1], reverse=True)
                md += "- **Most Persistent Regimes:**\n"
                for regime, prob in diagonal_probs[:3]:
                    md += f"  - Regime {regime}: {prob:.3f} ({prob*100:.1f}% self-transition)\n"
                
                # Find most common transitions
                all_transitions = []
                for from_regime in regimes:
                    for to_regime in regimes:
                        if from_regime != to_regime:
                            prob = matrix[from_regime].get(to_regime, 0.0)
                            if prob > 0.05:  # Only show transitions > 5%
                                all_transitions.append((from_regime, to_regime, prob))
                
                all_transitions.sort(key=lambda x: x[2], reverse=True)
                if all_transitions:
                    md += "\n- **Most Common Transitions:**\n"
                    for from_reg, to_reg, prob in all_transitions[:5]:
                        md += f"  - Regime {from_reg} → Regime {to_reg}: {prob:.3f} ({prob*100:.1f}%)\n"
                
                md += "\n"
        
        # Regime Duration Analysis
        if hasattr(metrics, 'regime_duration_distribution') and metrics.regime_duration_distribution:
            duration_dist = metrics.regime_duration_distribution
            md += """
### Regime Duration Analysis

"""
            if 'mean_durations' in duration_dist:
                mean_durations = duration_dist['mean_durations']
                md += "**Average Regime Durations:**\n\n"
                md += "| Regime | Mean Duration | Std Duration | Min Duration | Max Duration |\n"
                md += "|--------|---------------|--------------|--------------|--------------|\n"
                
                for regime_id in sorted(mean_durations.keys()):
                    stats = mean_durations[regime_id]
                    md += f"| {regime_id} | {stats.get('mean', 0):.1f} | {stats.get('std', 0):.1f} | "
                    md += f"{stats.get('min', 0):.0f} | {stats.get('max', 0):.0f} |\n"
                
                md += "\n"
            
            if 'duration_stability_score' in duration_dist:
                stability = duration_dist['duration_stability_score']
                md += f"- **Duration Stability Score:** {stability:.3f} (higher = more consistent durations)\n"
            
            md += "\n"
        
        # Per-regime metrics
        if metrics.per_regime_metrics:
            md += """
---

## Per-Regime Analysis

"""
            # Define target percentage for analysis (default 1%)
            target_pct = method_specific_config.get('target_pct', 1.0) if method_specific_config else 1.0
            
            for regime_id, regime_data in sorted(metrics.per_regime_metrics.items()):
                regime_type = regime_data.get('regime_type', 'unknown')
                size_pct = float(regime_data.get('percentage', 0.0))
                regime_size = regime_data.get('size', 'N/A')
                
                md += "### Regime {} ({})\n\n".format(regime_id, regime_type)
                md += "**Size:** {} samples ({:.2f}%)\n\n".format(regime_size, size_pct)
                
                if 'mean_return' in regime_data:
                    md += "**Performance Metrics:**\n"
                    md += "- Mean Return: " + str(regime_data['mean_return']) + "\n"
                    md += "- Volatility: " + str(regime_data['volatility']) + "\n"
                    md += "- Sharpe Ratio: " + str(regime_data['sharpe']) + "\n"
                    md += "- Skewness: " + str(regime_data.get('skewness', 0.0)) + "\n"
                    md += "- Max Drawdown: " + str(regime_data.get('max_drawdown', 0.0)) + "\n\n"
                    
                    md += "**Target-Based Metrics:**\n"
                    md += "- Pct > " + str(target_pct) + "% (Longs): " + str(regime_data.get('pct_above_target', 0.0)) + "\n"
                    md += "- Pct < -" + str(target_pct) + "% (Shorts): " + str(regime_data.get('pct_below_neg_target', 0.0)) + "\n"
                    md += "- Pct Target Hits: " + str(regime_data.get('pct_target_hits', 0.0)) + "\n\n"
                    
                    md += "**Risk-Adjusted Metrics:**\n"
                    md += "- Risk-Adj Target Hits: " + str(regime_data.get('risk_adj_target_hits', 0.0)) + "\n"
                    md += "- Win Rate (Long Bias): " + str(regime_data.get('win_rate', 0.0)) + "\n"
                    md += "- Return per Vol: " + str(regime_data.get('return_per_vol', 0.0)) + "\n"
                    md += "- Profit Factor: " + str(regime_data.get('profit_factor', 0.0)) + "\n\n"
                
                if 'regime_specific_metrics' in regime_data and regime_data['regime_specific_metrics']:
                    md += "**Regime-Specific Characteristics:**\n\n"
                    for key, value in regime_data['regime_specific_metrics'].items():
                        md += "- {}: {}\n".format(key, value)
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
                md += "### Regime Summary\n\n"
                md += "- **Total Regimes:** " + str(summary.get('total_regimes', 'N/A')) + "\n"
                md += "- **Dominant Regime:** " + str(summary.get('dominant_regime', 'N/A')) + "\n\n"
                if 'regime_type_distribution' in summary:
                    md += "**Regime Type Distribution:**\n\n"
                    for regime_type, count in summary['regime_type_distribution'].items():
                        md += "- " + str(regime_type) + ": " + str(count) + "\n"
                    md += "\n"
            
            if 'trading_implications' in interp:
                implications = interp['trading_implications']
                md += "\n### Trading Implications\n\n"
                
                if 'most_profitable_regime' in implications:
                    best = implications['most_profitable_regime']
                    md += "**Most Profitable Regime:** " + str(best.get('regime_id', 'N/A')) + " (" + str(best.get('regime_type', 'N/A')) + ")\n"
                    md += "- Sharpe Ratio: " + str(best.get('sharpe_ratio', 'N/A')) + "\n"
                    md += "- Mean Return: " + str(best.get('mean_return', 'N/A')) + "\n"
                    md += "- Volatility: " + str(best.get('volatility', 'N/A')) + "\n\n"
                
                if 'strategy_recommendations' in implications:
                    md += "**Strategy Recommendations:**\n\n"
                    for rec in implications['strategy_recommendations']:
                        md += "- {}: Target Regime {}\n".format(
                            rec.get('strategy', 'N/A'), 
                            rec.get('target_regime', 'N/A')
                        )
                    md += "\n"
        
        # Predictive power
        if metrics.predictive_power is not None:
            md += """
---

## Predictive Power

**Cross-Validation Accuracy:** """ + str(metrics.predictive_power) + """

This metric indicates how well the clustering can predict regime assignments on unseen data.
"""
        
        # Economic Relevance Analysis (NEW)
        if metrics.economic_relevance_analysis:
            md += """
---

## Economic Relevance Analysis

"""
            economic_analysis = metrics.economic_relevance_analysis
            
            # Strategy Performance Summary
            if 'strategy_performance' in economic_analysis:
                strategy_perf = economic_analysis['strategy_performance']
                md += "### Strategy Performance Summary\n\n"
                
                if 'regime_based_strategy' in strategy_perf:
                    regime_strategy = strategy_perf['regime_based_strategy']
                    md += "**Regime-Based Strategy Performance:**\n"
                    md += f"- Sharpe Ratio: {regime_strategy.get('sharpe_ratio', 'N/A'):.4f}\n"
                    md += f"- Total Return: {regime_strategy.get('total_return', 'N/A'):.2%}\n"
                    md += f"- Max Drawdown: {regime_strategy.get('max_drawdown', 'N/A'):.2%}\n"
                    md += f"- Win Rate: {regime_strategy.get('win_rate', 'N/A'):.2%}\n\n"
                
                if 'buy_and_hold' in strategy_perf:
                    bh_strategy = strategy_perf['buy_and_hold']
                    md += "**Buy & Hold Strategy Performance:**\n"
                    md += f"- Sharpe Ratio: {bh_strategy.get('sharpe_ratio', 'N/A'):.4f}\n"
                    md += f"- Total Return: {bh_strategy.get('total_return', 'N/A'):.2%}\n"
                    md += f"- Max Drawdown: {bh_strategy.get('max_drawdown', 'N/A'):.2%}\n\n"
                
                if 'performance_comparison' in strategy_perf:
                    comparison = strategy_perf['performance_comparison']
                    md += "**Performance Comparison:**\n"
                    md += f"- Sharpe Uplift vs Buy & Hold: {comparison.get('sharpe_uplift', 'N/A'):.2%}\n"
                    md += f"- Return Uplift vs Buy & Hold: {comparison.get('return_uplift', 'N/A'):.2%}\n"
                    md += f"- Outperformance Frequency: {comparison.get('outperformance_frequency', 'N/A'):.2%}\n\n"

                # Per-regime "only in regime k" long/short strategies
                regime_keys = [
                    key for key in strategy_perf.keys()
                    if isinstance(key, str) and key.startswith('regime_')
                ]
                if regime_keys:
                    md += "#### Per-Regime Long/Short Strategies\n\n"
                    for key in sorted(regime_keys):
                        strat = strategy_perf.get(key, {})
                        metrics_dict = strat.get('metrics', {}) if isinstance(strat, dict) else {}
                        name = strat.get('name', key) if isinstance(strat, dict) else key
                        total_ret = metrics_dict.get('total_return', 0.0)
                        sharpe = metrics_dict.get('sharpe_ratio', 0.0)
                        max_dd = metrics_dict.get('max_drawdown', 0.0)
                        md += (
                            f"- {name}: "
                            f"total_return={total_ret:.4f}, "
                            f"sharpe={sharpe:.4f}, "
                            f"max_dd={max_dd:.4f}\n"
                        )
                    md += "\n"
            
            # Significance Tests
            if 'significance_tests' in economic_analysis:
                significance = economic_analysis['significance_tests']
                md += "### Statistical Significance Tests\n\n"
                
                if 'permutation_test' in significance:
                    perm_test = significance['permutation_test']
                    md += "**Permutation Test Results:**\n"
                    md += f"- P-value: {perm_test.get('p_value', 'N/A'):.4f}\n"
                    md += f"- Test Statistic: {perm_test.get('test_statistic', 'N/A'):.4f}\n"
                    md += f"- Is Significant: {'Yes' if perm_test.get('is_significant', False) else 'No'}\n\n"
                
                if 'bootstrap_test' in significance:
                    boot_test = significance['bootstrap_test']
                    md += "**Bootstrap Test Results:**\n"
                    md += f"- Confidence Interval: [{boot_test.get('ci_lower', 'N/A'):.4f}, {boot_test.get('ci_upper', 'N/A'):.4f}]\n"
                    md += f"- Bootstrap Mean: {boot_test.get('bootstrap_mean', 'N/A'):.4f}\n"
                    md += f"- Is Significant: {'Yes' if boot_test.get('is_significant', False) else 'No'}\n\n"
            
            # Regime Mapping
            if 'regime_mapping' in economic_analysis:
                regime_map = economic_analysis['regime_mapping']
                md += "### Economic Regime Mapping\n\n"
                md += "| Regime | Economic Interpretation | Recommended Position |\n"
                md += "|---------|----------------------|----------------------|\n"
                
                for regime_id, mapping_info in regime_map.items():
                    interpretation = mapping_info.get('interpretation', 'Unknown')
                    position = mapping_info.get('recommended_position', 'Neutral')
                    md += f"| {regime_id} | {interpretation} | {position} |\n"
                md += "\n"
            
            # Economic Interpretation
            if 'economic_interpretation' in economic_analysis:
                econ_interp = economic_analysis['economic_interpretation']
                md += "### Economic Interpretation\n\n"
                
                if 'key_insights' in econ_interp:
                    insights = econ_interp['key_insights']
                    md += "**Key Insights:**\n"
                    for insight in insights:
                        md += f"- {insight}\n"
                    md += "\n"
                
                if 'trading_recommendations' in econ_interp:
                    recommendations = econ_interp['trading_recommendations']
                    md += "**Trading Recommendations:**\n"
                    for rec in recommendations:
                        md += f"- {rec}\n"
                    md += "\n"
            
            # Report Path
            if metrics.economic_report_path:
                md += f"**Detailed Economic Report:** {metrics.economic_report_path}\n\n"
        
        # Quality assessment
        md += """
---

## Quality Assessment

**Overall Quality Score:** """ + (f'{metrics.quality_score:.4f}' if metrics.quality_score else 'N/A') + """ / 1.0
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
                recommendation = "The clustering shows poor quality. Re-evaluate approach."
            
            md += "**Quality Level:** " + quality_level + "\n"
            md += "**Recommendation:** " + recommendation + "\n\n"
        
        return md


def create_cluster_quality_assessor(artifact_manager=None, 
                                    enable_hardware_optimization=True,
                                    enable_vectorization=True) -> ClusterQualityAssessor:
    """Factory function to create a cluster quality assessor."""
    return ClusterQualityAssessor(
        artifact_manager=artifact_manager,
        enable_hardware_optimization=enable_hardware_optimization,
        enable_vectorization=enable_vectorization
    )
