"""
Enhanced Regime Discovery Configuration

Production-ready configuration with all safeguards, numerical stability,
causal invariants, and comprehensive parameter management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import os


@dataclass
class RegimeDiscoveryConfig:
    """
    Enhanced configuration for HDBSCAN-based regime discovery with production safeguards.
    
    Includes numerical stability floors, causal invariants, memory guardrails,
    and comprehensive validation parameters.
    """
    
    # ============================================================================
    # CORE HDBSCAN PARAMETERS
    # ============================================================================
    min_cluster_size_pct: float = 0.01  # % of N_eff (after windowing)
    min_cluster_size_floor: int = 12    # Absolute minimum
    min_samples_options: List[Union[int, str]] = field(default_factory=lambda: [None, 'half', 'same'])
    cluster_selection_method_options: List[str] = field(default_factory=lambda: ['eom', 'leaf'])
    cluster_selection_epsilon: float = 0.01
    prediction_data: bool = True
    
    # ============================================================================
    # DIMENSIONALITY REDUCTION
    # ============================================================================
    dim_reduction_mode: str = 'densmap'  # {'pca_only', 'umap', 'densmap'}
    pca_n_components: Union[int, float] = 0.99  # Variance or # components
    pca_whiten: bool = True
    pca_baseline_dims: int = 30  # For pca_only mode
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_n_components: int = 8
    umap_metric: str = 'euclidean'
    umap_densmap: bool = True  # Density preservation
    random_state: int = 42
    
    # ============================================================================
    # FEATURE EXTRACTION (200-300 features)
    # ============================================================================
    lookback_windows: Dict[str, List[int]] = field(default_factory=lambda: {
        'short': [5, 10, 20],
        'medium': [50, 100],
        'long': [200, 500]
    })
    enable_entropy_caching: bool = True
    use_numba_optimization: bool = True
    
    # ============================================================================
    # PREPROCESSING WITH NUMERICAL STABILITY
    # ============================================================================
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    quantile_transformer_output: str = 'normal'
    correlation_threshold: float = 0.95
    mi_threshold: float = 0.9  # Mutual information pruning
    per_asset_fitting: bool = True  # Avoid cross-sectional leakage
    variance_floor: float = 1e-8  # Clamp tiny variances
    min_history_for_asset_fit: int = 1000  # Use frozen default until met
    cold_asset_transformer: str = 'global_median'
    
    # ============================================================================
    # TEMPORAL WINDOWS
    # ============================================================================
    window_size: int = 300
    window_overlap_pct: float = 0.7
    
    # ============================================================================
    # POST-CLUSTERING OPTIMIZATION
    # ============================================================================
    enable_reallocation: bool = True
    change_budget_pct: float = 0.10  # Max 10% moves per pass
    max_optimization_rounds: int = 5
    use_condensed_tree: bool = True  # Tree-aware splits/merges
    
    # ============================================================================
    # NOISE HANDLING WITH CAUSAL INVARIANTS
    # ============================================================================
    noise_handling_mode: str = 'causal_smooth'  # {'keep', 'knn_assign', 'causal_smooth', 'acausal_smooth'}
    smoothing_window: int = 5
    knn_k: int = 5
    min_dwell_bars: int = 5  # Min bars before regime switch (unless prob > 0.9)
    high_confidence_threshold: float = 0.9  # Skip dwell time if exceeded
    cooldown_bars: int = 3  # Cooldown after switch to prevent ping-pong
    
    # ============================================================================
    # OOS ASSIGNMENT WITH AUTO-SWITCH
    # ============================================================================
    oos_low_prob_threshold: float = 0.3
    oos_switch_to_knn_threshold: float = 0.2  # % low-prob triggers kNN
    oos_hysteresis: float = 0.1  # Prevent oscillation at threshold
    log_oos_assignment_mode: bool = True  # Audit trail
    
    # ============================================================================
    # HDBSCAN MEMORY/LATENCY GUARDRAILS
    # ============================================================================
    max_min_span_edges: int = 1_000_000  # Early stop for huge datasets
    enable_chunked_fit: bool = True  # For N > 100k
    chunked_fit_threshold: int = 100_000
    max_hdbscan_seconds: float = 300.0  # 5 min timeout
    
    # ============================================================================
    # REGIME STABILITY ACROSS RETRAINS
    # ============================================================================
    enable_hungarian_matching: bool = True
    hungarian_similarity_threshold: float = 0.7
    split_detection_threshold: float = 0.5  # Distance for split detection
    
    # ============================================================================
    # VALIDATION THRESHOLDS
    # ============================================================================
    target_regime_count: Tuple[int, int] = (3, 7)  # Min, max actionable regimes
    min_dbcv: float = 0.4
    min_silhouette: float = 0.3
    min_stability_index: float = 0.7
    min_regime_duration_bars: int = 20
    max_transition_off_diagonal: float = 0.3
    bootstrap_n_resamples: int = 100
    permutation_n_tests: int = 1000
    
    # ============================================================================
    # ECONOMIC VALIDATION
    # ============================================================================
    interpretable_axes: List[str] = field(default_factory=lambda: [
        'trend_pc', 'vol_pc', 'breadth', 'skew', 'liquidity_stress', 'momentum_strength'
    ])
    min_economic_separation_pct: float = 0.30  # 30% difference in key metrics
    
    # ============================================================================
    # VERSIONING & REPRODUCIBILITY
    # ============================================================================
    schema_version: str = '1.0.0'
    track_provenance: bool = True
    track_library_versions: bool = True  # numpy, numba, umap, hdbscan, blas
    
    # ============================================================================
    # LIVE TRADING SAFETY
    # ============================================================================
    mode_live: bool = False  # Derives all causal defaults
    canary_deployment_days: int = 14
    canary_rollback_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'distribution_kl_divergence': 0.5,
        'duration_collapse_factor': 0.5,
        'noise_explosion_pct': 0.3,
        'sharpe_degradation_pct': 0.2
    })
    
    # ============================================================================
    # DETERMINISM GUARANTEES
    # ============================================================================
    pin_blas_threads: bool = True
    clear_numba_cache_on_version_change: bool = True
    
    # ============================================================================
    # TELEMETRY & MONITORING
    # ============================================================================
    enable_telemetry: bool = True
    telemetry_metrics: List[str] = field(default_factory=lambda: [
        'regime.current', 'regime.noise_pct', 'regime.avg_duration_bars',
        'regime.transition_rate', 'oos.assignment_mode', 'oos.low_prob_rate',
        'validation.dbcv', 'validation.silhouette', 'validation.bootstrap_jaccard_mean',
        'econ.uplift_deflated_sharpe', 'step.latency_ms', 'step.memory_peak_mb'
    ])
    
    def __post_init__(self):
        """Validate configuration and set derived parameters."""
        self._validate_config()
        self._set_derived_parameters()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.mode_live:
            assert self.noise_handling_mode in ['causal_smooth', 'keep'], \
                "mode_live requires causal noise handling"
            assert self.min_dwell_bars >= 3, "min_dwell_bars must be ≥3 in live mode"
            assert self.cooldown_bars >= 1, "cooldown_bars must be ≥1 in live mode"
        
        assert 0 < self.min_cluster_size_pct < 0.5, "min_cluster_size_pct must be in (0, 0.5)"
        assert 0 < self.change_budget_pct <= 0.5, "change_budget_pct must be in (0, 0.5]"
        assert 0 < self.window_overlap_pct < 1, "window_overlap_pct must be in (0, 1)"
        assert 0 < self.high_confidence_threshold <= 1, "high_confidence_threshold must be in (0, 1]"
        
        # Validate dimensionality reduction mode
        valid_modes = {'pca_only', 'umap', 'densmap'}
        assert self.dim_reduction_mode in valid_modes, \
            f"dim_reduction_mode must be in {valid_modes}"
    
    def _set_derived_parameters(self):
        """Set derived parameters based on configuration."""
        # Set noise handling mode based on live mode
        if self.mode_live and self.noise_handling_mode == 'acausal_smooth':
            self.noise_handling_mode = 'causal_smooth'
            print("⚠️ Overriding noise_handling_mode to 'causal_smooth' for live mode")
        
        # Set OOS assignment mode based on live mode
        if self.mode_live:
            self.log_oos_assignment_mode = True  # Always log in live mode
    
    @property
    def is_production_safe(self) -> bool:
        """Validate config is safe for live trading."""
        if self.mode_live:
            assert self.noise_handling_mode in ['causal_smooth', 'keep'], \
                "mode_live requires causal noise handling"
            assert self.min_dwell_bars >= 3, "min_dwell_bars must be ≥3 in live mode"
            assert self.cooldown_bars >= 1, "cooldown_bars must be ≥1 in live mode"
        return True
    
    def get_effective_min_cluster_size(self, n_effective_samples: int) -> int:
        """Calculate effective min_cluster_size based on N_eff."""
        pct_based = max(int(self.min_cluster_size_pct * n_effective_samples), self.min_cluster_size_floor)
        return min(pct_based, n_effective_samples // 2)  # Cap at 50% of samples
    
    def get_min_samples_value(self, min_cluster_size: int) -> Optional[int]:
        """Get min_samples value based on min_cluster_size."""
        if self.min_samples_options[0] is None:
            return None
        elif self.min_samples_options[0] == 'half':
            return max(1, min_cluster_size // 2)
        elif self.min_samples_options[0] == 'same':
            return min_cluster_size
        else:
            return int(self.min_samples_options[0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'min_cluster_size_pct': self.min_cluster_size_pct,
            'min_cluster_size_floor': self.min_cluster_size_floor,
            'dim_reduction_mode': self.dim_reduction_mode,
            'pca_n_components': self.pca_n_components,
            'pca_whiten': self.pca_whiten,
            'umap_n_neighbors': self.umap_n_neighbors,
            'umap_min_dist': self.umap_min_dist,
            'umap_n_components': self.umap_n_components,
            'umap_densmap': self.umap_densmap,
            'random_state': self.random_state,
            'window_size': self.window_size,
            'window_overlap_pct': self.window_overlap_pct,
            'change_budget_pct': self.change_budget_pct,
            'noise_handling_mode': self.noise_handling_mode,
            'min_dwell_bars': self.min_dwell_bars,
            'high_confidence_threshold': self.high_confidence_threshold,
            'mode_live': self.mode_live,
            'schema_version': self.schema_version,
            'created_at': datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RegimeDiscoveryConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        return f"""
Regime Discovery Configuration Summary:
- Mode: {'LIVE' if self.mode_live else 'OFFLINE'}
- Dimensionality Reduction: {self.dim_reduction_mode}
- Min Cluster Size: {self.min_cluster_size_pct:.1%} (floor: {self.min_cluster_size_floor})
- Window Size: {self.window_size} bars (overlap: {self.window_overlap_pct:.1%})
- Noise Handling: {self.noise_handling_mode}
- Min Dwell: {self.min_dwell_bars} bars
- Change Budget: {self.change_budget_pct:.1%}
- Target Regimes: {self.target_regime_count[0]}-{self.target_regime_count[1]}
- Schema Version: {self.schema_version}
"""


@dataclass
class TrainSpanValidator:
    """Ensure consistent train span semantics."""
    
    def __init__(self, data):
        self.wall_clock_span = (data.index[0], data.index[-1])
        self.bar_count = len(data)
        self.intended_horizon_bars = None
        
    def validate_span(self, intended_horizon_days: int, bar_frequency: str = '1h') -> Dict[str, Any]:
        """Assert wall-clock span matches intended horizon."""
        expected_bars = self._compute_expected_bars(intended_horizon_days, bar_frequency)
        
        # Allow 10% tolerance for holidays/weekends
        if abs(self.bar_count - expected_bars) > 0.1 * expected_bars:
            raise ValueError(
                f"Train span mismatch: {self.bar_count} bars vs "
                f"expected {expected_bars} for {intended_horizon_days} days. "
                f"Check for trading hours changes or data gaps."
            )
        
        return {
            'wall_clock_span': self.wall_clock_span,
            'bar_count': self.bar_count,
            'expected_bars': expected_bars,
            'validated': True
        }
    
    def _compute_expected_bars(self, days: int, frequency: str) -> int:
        """Compute expected number of bars for given frequency."""
        if frequency == '1h':
            return days * 24  # 24 hours per day
        elif frequency == '30m':
            return days * 48  # 48 half-hours per day
        elif frequency == '15m':
            return days * 96  # 96 quarters per day
        elif frequency == '1d':
            return days
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")


@dataclass
class DataContractValidator:
    """Validate data meets quality contracts before processing."""
    
    def validate_pre_feature_calc(self, data, config: RegimeDiscoveryConfig) -> Dict[str, Any]:
        """Validate data meets quality contracts before processing."""
        checks = {
            'schema': self._check_schema(data),
            'nan_rate': self._check_nan_rate(data),
            'stationarity': self._check_stationarity(data),
            'effective_sample_size': self._check_effective_n(data, config),
            'correlation_explosion': self._check_feature_correlation(data),
            'outlier_contamination': self._check_outlier_contamination(data)
        }
        
        failures = [k for k, v in checks.items() if not v['passed']]
        
        if failures:
            raise DataContractViolation(f"Failed checks: {failures}", checks)
        
        return checks
    
    def _check_nan_rate(self, data) -> Dict[str, Any]:
        """NaN rate must be <5% per feature."""
        nan_rates = data.isna().mean()
        max_nan_rate = nan_rates.max()
        passed = max_nan_rate < 0.05
        return {
            'passed': passed, 
            'max_nan_rate': max_nan_rate, 
            'features_above_threshold': list(nan_rates[nan_rates >= 0.05].index)
        }
    
    def _check_stationarity(self, data) -> Dict[str, Any]:
        """Returns variance within expected bounds."""
        if 'close' not in data.columns:
            return {'passed': True, 'volatility': None, 'reason': 'No close price column'}
        
        returns = data['close'].pct_change().dropna()
        vol = returns.std()
        # Sanity bounds: daily vol between 0.1% and 20%
        passed = 0.001 < vol < 0.20
        return {'passed': passed, 'volatility': vol}
    
    def _check_effective_n(self, data, config: RegimeDiscoveryConfig) -> Dict[str, Any]:
        """Effective sample size ≥ 5 * min_cluster_size."""
        n_eff = len(data) * (1 - config.window_overlap_pct)
        min_required = 5 * config.min_cluster_size_floor
        passed = n_eff >= min_required
        return {'passed': passed, 'n_eff': n_eff, 'min_required': min_required}
    
    def _check_feature_correlation(self, data) -> Dict[str, Any]:
        """Ensure no near-perfect multicollinearity."""
        import numpy as np
        corr_matrix = data.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        
        if max_corr > 0.98:
            high_corr_pairs = list(zip(*np.where(corr_matrix > 0.98)))
            return {
                'passed': False, 
                'max_correlation': max_corr,
                'high_corr_pairs': high_corr_pairs
            }
        return {'passed': True, 'max_correlation': max_corr}
    
    def _check_outlier_contamination(self, data) -> Dict[str, Any]:
        """Check for extreme values beyond historical bounds."""
        # Simple outlier check: values beyond 5-sigma
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                outliers = np.abs(values - mean_val) > 5 * std_val
                outlier_counts[col] = outliers.sum()
        
        total_outliers = sum(outlier_counts.values())
        outlier_rate = total_outliers / (len(data) * len(numeric_cols))
        
        passed = outlier_rate < 0.01  # Less than 1% outliers
        return {
            'passed': passed,
            'outlier_rate': outlier_rate,
            'outlier_counts': outlier_counts
        }
    
    def _check_schema(self, data) -> Dict[str, Any]:
        """Check data schema consistency."""
        required_columns = ['close', 'volume']  # Minimum required
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        return {
            'passed': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'total_columns': len(data.columns)
        }


class DataContractViolation(Exception):
    """Raised when data contract validation fails."""
    
    def __init__(self, message: str, check_results: Dict[str, Any]):
        super().__init__(message)
        self.check_results = check_results
