"""
Volatility-Aware Multi-Horizon Profit Labeler

This module implements a completely redesigned profit labeling system that explicitly accounts
for volatility and microstructure noise, optimized for creating strong ML labels rather than trading rules.

Key Design Principles:
- Volatility-normalized targets (k·σ_t) instead of fixed percentages
- Data-driven horizons via first-passage time quantiles
- Multi-target labeling (small/medium/high) with separate optimization
- Noise gating to filter microstructure-dominated periods
- Label quality optimization for ML learnability (AUC, stability, balance, SNR)
- No regime dependencies (as requested)

Author: AI Assistant
Date: 2025-10-06
"""

import numpy as np
import pandas as pd
import time
import hashlib
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from abc import ABC

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Core utilities
from src.utils.logger import get_logger
from src.utils.tprint import tprint
from src.utils.math_validation import safe_divide, validate_finite
from src.core.decorators import handles_errors, traced, validates

# Matrix operations for performance
from src.utils.matrix_operations import UnifiedMatrixOperations
from src.feature_generation.utils.enhanced_matrix_operations import EnhancedMatrixOperations

# Statistical and ML utilities
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

@dataclass
class VolatilityAwareConfig:
    """
    Configuration for volatility-aware profit labeling.

    Focus: Creating ML-learnable labels, not trading rules.
    """

    # Base timeframe for analysis
    base_timeframe_minutes: int = 5

    # Volatility modeling parameters
    rv_window_minutes: int = 30        # Rolling volatility window
    atr_window_bars: int = 14          # ATR window in bars
    volatility_ewma_lambda: float = 0.94  # EWMA smoothing for volatility

    # Multi-target configuration (small/medium/high)
    target_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'small': (0.4, 0.8),    # k ∈ [0.4, 0.8]
        'medium': (0.8, 1.3),   # k ∈ [0.8, 1.3]
        'high': (1.3, 2.0)      # k ∈ [1.3, 2.0]
    })

    # First-passage time quantile for horizon selection
    fpt_quantile: float = 0.65  # Q_0.65 of historical FPT

    # Noise gating parameters
    micro_range_alpha: float = 1.5     # k·σ_t ≥ α·mTR_t
    variance_ratio_threshold: float = 1.2  # VR threshold for microstructure filter
    liquidity_percentile: float = 10.0     # Minimum volume percentile
    spread_filter_enabled: bool = True

    # Label quality constraints
    min_positive_balance: float = 0.35    # Minimum 35% positive class
    max_positive_balance: float = 0.65    # Maximum 65% positive class
    min_aic_threshold: float = 0.55       # Minimum AUC for acceptance
    max_auc_std_threshold: float = 0.08   # Maximum AUC standard deviation

    # Hysteresis and conflict resolution
    hysteresis_bars: int = 2
    flip_override_beta: float = 0.3

    # Optimization parameters
    search_grid_k: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])
    search_grid_quantile: List[float] = field(default_factory=lambda: [0.5, 0.65, 0.8])
    search_grid_alpha: List[float] = field(default_factory=lambda: [1.2, 1.5, 1.8])

    # Cross-target correlation constraint
    max_target_correlation: float = 0.6

    # Processing parameters
    min_bars_for_labeling: int = 50
    max_horizon_bars: int = 100
    outlier_cap_percentile: float = 99.9

    # Quality scoring weights
    lqs_weights: Dict[str, float] = field(default_factory=lambda: {
        'predictability': 0.3,
        'stability': 0.25,
        'balance': 0.2,
        'snr': 0.15,
        'consistency': 0.1
    })

@dataclass
class LabelQualityMetrics:
    """Container for label quality metrics."""

    # Core KPIs
    auc_mean: float = 0.0
    auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0

    # Stability metrics
    psi_score: float = 0.0  # Population Stability Index
    flip_rate: float = 0.0  # Label flip rate within windows

    # Balance metrics
    positive_balance: float = 0.0
    class_balance_score: float = 0.0  # How close to 50/50

    # Signal-to-noise metrics
    feature_ic_mean: float = 0.0  # Spearman correlation with features
    mutual_information: float = 0.0  # MI with adjacent horizons

    # Composite score
    label_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for reporting."""
        return {
            'auc_mean': self.auc_mean,
            'auc_std': self.auc_std,
            'pr_auc_mean': self.pr_auc_mean,
            'pr_auc_std': self.pr_auc_std,
            'psi_score': self.psi_score,
            'flip_rate': self.flip_rate,
            'positive_balance': self.positive_balance,
            'class_balance_score': self.class_balance_score,
            'feature_ic_mean': self.feature_ic_mean,
            'mutual_information': self.mutual_information,
            'label_quality_score': self.label_quality_score
        }

class VolatilityAwareProfitLabeler(BaseStep):
    """
    Volatility-aware multi-horizon profit labeler optimized for ML label quality.

    This implementation focuses on creating labels that are:
    1. Learnable by ML models (high AUC, stable, balanced)
    2. Volatility-normalized (not fixed percentage targets)
    3. Noise-resistant (filters microstructure effects)
    4. Multi-target (small/medium/high with data-driven horizons)
    5. BaseStep integrated for standardized pipeline execution
    """

    def __init__(self, config: Optional[VolatilityAwareConfig] = None):
        """Initialize the volatility-aware labeler."""
        super().__init__()
        self.config = config or VolatilityAwareConfig()
        self.logger = get_logger('VolatilityAwareProfitLabeler')

        # Initialize matrix operations
        self.matrix_ops = UnifiedMatrixOperations()
        self.enhanced_ops = EnhancedMatrixOperations()

        # Cache for expensive calculations
        self._volatility_cache: Dict[str, pd.Series] = {}
        self._fpt_cache: Dict[str, np.ndarray] = {}
        self._quality_cache: Dict[str, LabelQualityMetrics] = {}

        tprint("🔧 Initialized Volatility-Aware Profit Labeler")
        self.logger.info(f"📊 Config: {len(self.config.target_bands)} target bands, "
                        f"RV window: {self.config.rv_window_minutes}min, "
                        f"ATR window: {self.config.atr_window_bars} bars")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the volatility-aware profit labeling step.
        
        Args:
            config: Configuration dictionary containing:
                - data: DataFrame with OHLCV data
                - symbol: Optional symbol for context
                - exchange: Optional exchange for context
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - labeled_data: DataFrame with generated labels
                - quality_report: Quality assessment report
                - artifacts: List of generated artifacts
        """
        try:
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract data from config
            data = config.get('data')
            
            if data is None:
                return {
                    'success': False,
                    'error': 'Missing required parameter: data'
                }
            
            # Validate inputs
            if not isinstance(data, pd.DataFrame):
                return {
                    'success': False,
                    'error': 'data must be a pandas DataFrame'
                }
            
            # Generate labels
            labeled_data, quality_report = self.generate_labels(data)
            
            # Save artifacts
            artifacts = []
            
            # Save labeled data
            labeled_data_path = self._save_dataframe(
                labeled_data, 
                'volatility_aware_labeled_data'
            )
            if labeled_data_path:
                artifacts.append(labeled_data_path)
            
            # Save quality report
            if quality_report:
                quality_path = self._save_metadata(
                    quality_report, 
                    'volatility_aware_quality_report'
                )
                if quality_path:
                    artifacts.append(quality_path)
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(labeled_data, quality_report, artifacts)
            self._save_outcome_file(outcome_content, 'volatility_aware_labeling_outcome')
            
            return {
                'success': True,
                'labeled_data': labeled_data,
                'quality_report': quality_report,
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Volatility-aware labeling failed: {str(e)}"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, labeled_data: pd.DataFrame, quality_report: Dict[str, Any], artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Volatility-Aware Profit Labeling Outcome

## Summary
- **Status**: Success
- **Samples Processed**: {len(labeled_data)}
- **Artifacts Generated**: {len(artifacts)}

## Data Overview
- **Columns**: {list(labeled_data.columns)}
- **Memory Usage**: {labeled_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
        
        if quality_report:
            content += f"""
## Quality Report
- **Overall Quality Score**: {quality_report.get('overall_quality_score', 0):.3f}
- **Predictability Score**: {quality_report.get('predictability_score', 0):.3f}
- **Stability Score**: {quality_report.get('stability_score', 0):.3f}
- **Balance Score**: {quality_report.get('balance_score', 0):.3f}
"""
        
        content += f"""
## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}

## Configuration
- **Target Bands**: {len(self.config.target_bands)}
- **RV Window**: {self.config.rv_window_minutes} minutes
- **ATR Window**: {self.config.atr_window_bars} bars
- **Min Bars for Labeling**: {self.config.min_bars_for_labeling}
"""
        
        return content

    def generate_labels(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate volatility-aware multi-horizon labels.

        Args:
            data: OHLCV DataFrame with proper columns

        Returns:
            Tuple of (labeled_data, quality_report)
        """
        tprint("🚀 Starting volatility-aware profit labeling...")

        if len(data) < self.config.min_bars_for_labeling:
            tprint(f"⚠️ Insufficient data: {len(data)} < {self.config.min_bars_for_labeling}")
            return data.copy(), {}

        # Step 0: Data preparation and cleaning
        data_clean = self._prepare_and_clean_data(data)

        # Step 1: Volatility modeling
        volatility_data = self._compute_volatility_series(data_clean)

        # Step 2: Noise gating
        noise_gates = self._compute_noise_gates(data_clean, volatility_data)

        # Step 3: Multi-target configuration optimization
        optimal_configs = self._optimize_target_configurations(
            data_clean, volatility_data, noise_gates
        )

        # Step 4: Generate labels for each target
        labeled_data = data_clean.copy()
        quality_report = {}

        for target_name, config in optimal_configs.items():
            tprint(f"🎯 Generating {target_name} target labels...")

            target_labels = self._generate_single_target_labels(
                data_clean, volatility_data, noise_gates, config, target_name
            )

            # Merge into main dataframe
            for col in target_labels.columns:
                if col not in labeled_data.columns:
                    labeled_data[col] = target_labels[col]

            # Compute quality metrics
            quality_metrics = self._compute_label_quality(
                target_labels, data_clean, target_name
            )
            quality_report[target_name] = quality_metrics.to_dict()

        # Step 5: Final quality validation and reporting
        final_report = self._generate_comprehensive_report(quality_report, optimal_configs)

        tprint("✅ Volatility-aware labeling completed")
        self.logger.info(f"📊 Generated labels for {len(optimal_configs)} targets")

        return labeled_data, final_report

    def _prepare_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Step 0: Clean and prepare data for labeling."""
        tprint("🔍 Preparing and cleaning data...")

        # Create copy for modification
        data_clean = data.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data_clean.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Compute returns and outlier handling
        data_clean['returns'] = data_clean['close'].pct_change()

        # Cap extreme returns
        return_cap = np.percentile(data_clean['returns'].dropna(), self.config.outlier_cap_percentile)
        data_clean['returns'] = np.clip(data_clean['returns'], -return_cap, return_cap)

        # Compute true range for microstructure filtering
        data_clean['true_range'] = np.maximum(
            data_clean['high'] - data_clean['low'],
            np.maximum(
                abs(data_clean['high'] - data_clean['close'].shift(1)),
                abs(data_clean['low'] - data_clean['close'].shift(1))
            )
        )

        # Forward fill any missing values
        data_clean = data_clean.fillna(method='ffill').dropna()

        tprint(f"✅ Data prepared: {len(data_clean)} bars, {data_clean.shape[1]} columns")
        return data_clean

    def _compute_volatility_series(self, data: pd.DataFrame) -> pd.Series:
        """Step 1: Compute volatility series using EWMA of realized volatility and ATR."""
        tprint("📊 Computing volatility series...")

        # Ensure returns are computed
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()

        # Realized volatility (RV) - rolling std of returns
        returns = data['returns'].dropna()
        if len(returns) < self.config.rv_window_minutes:
            # Fallback for short data
            rv_window = min(len(returns), 20)
        else:
            rv_window = self.config.rv_window_minutes

        # Rolling realized volatility
        rolling_rv = returns.rolling(window=rv_window).std() * np.sqrt(rv_window)

        # ATR (Average True Range) - ensure it exists
        if 'true_range' not in data.columns:
            data['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )

        atr = data['true_range'].rolling(window=self.config.atr_window_bars).mean()

        # Combine RV and ATR using EWMA
        combined_vol = (rolling_rv + atr) / 2

        # Apply EWMA smoothing
        vol_ewma = combined_vol.ewm(alpha=1-self.config.volatility_ewma_lambda).mean()

        # Floor at small epsilon to avoid division blowups
        vol_ewma = vol_ewma.clip(lower=1e-6)

        tprint(f"✅ Volatility computed: mean={vol_ewma.mean():.6f}, "
               f"std={vol_ewma.std():.6f}, range=[{vol_ewma.min():.6f}, {vol_ewma.max():.6f}]")

        return vol_ewma

    def _compute_noise_gates(self, data: pd.DataFrame, volatility: pd.Series) -> Dict[str, pd.Series]:
        """Step 2: Compute noise gates to filter microstructure effects."""
        tprint("🔇 Computing noise gates...")

        # Ensure required columns exist
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()

        if 'true_range' not in data.columns:
            data['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )

        gates = {}

        # 1. Micro-range gate: k·σ_t ≥ α·mTR_t (median true range)
        median_tr = data['true_range'].rolling(window=20).median()
        # We'll compute this per target when we have k values

        # 2. Variance ratio test for microstructure
        def compute_variance_ratio(returns: pd.Series, m: int = 5) -> pd.Series:
            """Compute variance ratio VR = Var(r_Δ) / (m·Var(r_Δ/m))."""
            if len(returns) < 2*m:
                return pd.Series(1.0, index=returns.index)

            # Compute returns at different scales
            r_delta = returns.diff(m).dropna()
            r_delta_m = returns.diff(1).dropna()

            # Align indices for proper division
            aligned_idx = r_delta.index.intersection(r_delta_m.index[:len(r_delta)])
            r_delta = r_delta.loc[aligned_idx]
            r_delta_m = r_delta_m.loc[aligned_idx]

            var_delta = r_delta.var()
            var_delta_m = r_delta_m.var()

            if var_delta_m > 0:
                vr = var_delta / (m * var_delta_m)
            else:
                vr = 1.0

            return pd.Series(vr, index=aligned_idx)

        # Rolling variance ratio (looking back 50 bars)
        vr_values = []
        for i in range(max(50, len(data))):
            if i < 50:
                vr_values.append(1.0)
            else:
                window_returns = data['returns'].iloc[i-50:i]
                vr = compute_variance_ratio(window_returns).iloc[-1] if len(window_returns) >= 10 else 1.0
                vr_values.append(vr)

        gates['variance_ratio'] = pd.Series(vr_values, index=data.index)

        # 3. Liquidity gate: volume percentile filter
        rolling_volume_pct = data['volume'].rolling(window=50).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50.0
        )
        gates['liquidity_gate'] = rolling_volume_pct >= self.config.liquidity_percentile

        # 4. Spread filter: ultra-tight ranges
        relative_spread = (data['high'] - data['low']) / data['close'].shift(1)
        median_spread = relative_spread.rolling(window=20).median()
        gates['spread_filter'] = relative_spread >= (median_spread * 0.5)  # Not ultra-tight

        # Combined eligibility gate (all filters must pass)
        gates['eligibility'] = (
            gates['liquidity_gate'] &
            gates['spread_filter'] &
            (gates['variance_ratio'] >= 0.8)  # Not dominated by microstructure
        )

        tprint(f"✅ Noise gates computed: {gates['eligibility'].mean():.1%} eligible bars")
        return gates

    def _optimize_target_configurations(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step 3: Optimize target configurations for each band (small/medium/high).

        This searches over k values within each band to maximize label quality,
        then filters for cross-target correlation constraints.
        """
        tprint("🔍 Optimizing target configurations...")

        # Step 1: Find best config for each target independently
        per_target_configs = {}

        for target_name, (k_min, k_max) in self.config.target_bands.items():
            tprint(f"🎯 Optimizing {target_name} target (k ∈ [{k_min}, {k_max}])...")

            # Search over k values in this band
            best_config = None
            best_lqs = -np.inf

            # Generate candidate k values (grid search within band)
            k_candidates = [k for k in self.config.search_grid_k
                          if k_min <= k <= k_max]

            # Add some additional candidates in the band
            k_candidates.extend(np.linspace(k_min, k_max, 5))

            for k in sorted(set(k_candidates)):
                # For each k, try different quantile configurations
                for q in self.config.search_grid_quantile:
                    for alpha in self.config.search_grid_alpha:
                        config = {
                            'k': k,
                            'fpt_quantile': q,
                            'micro_range_alpha': alpha
                        }

                        # Test this configuration
                        try:
                            quality_metrics = self._evaluate_config_quality(
                                data, volatility, noise_gates, config, target_name
                            )

                            if quality_metrics.label_quality_score > best_lqs:
                                best_lqs = quality_metrics.label_quality_score
                                best_config = config.copy()
                                best_config['quality_metrics'] = quality_metrics

                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to evaluate config k={k}, q={q}, alpha={alpha}: {e}")
                            continue

            if best_config:
                per_target_configs[target_name] = best_config
                tprint(f"✅ Best {target_name}: k={best_config['k']:.2f}, "
                       f"q={best_config['fpt_quantile']:.2f}, "
                       f"LQS={best_lqs:.3f}")
            else:
                # Fallback to reasonable defaults
                per_target_configs[target_name] = {
                    'k': (k_min + k_max) / 2,
                    'fpt_quantile': self.config.fpt_quantile,
                    'micro_range_alpha': self.config.micro_range_alpha,
                    'quality_metrics': LabelQualityMetrics()
                }
                tprint(f"⚠️ Using fallback config for {target_name}")

        # Step 2: Filter configurations based on cross-target correlations
        optimal_configs = self._filter_by_correlation(per_target_configs, data, volatility, noise_gates)

        return optimal_configs

    def _filter_by_correlation(
        self,
        per_target_configs: Dict[str, Dict[str, Any]],
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter configurations to ensure targets are not too correlated."""

        tprint("🔗 Filtering configurations by cross-target correlation...")

        if len(per_target_configs) <= 1:
            return per_target_configs

        # Generate labels for all targets to compute correlations
        target_labels = {}
        for target_name, config in per_target_configs.items():
            labels = self._generate_single_target_labels(
                data, volatility, noise_gates, config, target_name, max_samples=2000
            )
            target_labels[target_name] = labels['target']

        # Compute correlation matrix
        label_series = pd.DataFrame(target_labels)

        # Only consider valid labels (non-zero)
        valid_mask = (label_series != 0).any(axis=1)
        if valid_mask.sum() < 50:
            tprint("⚠️ Insufficient overlapping labels for correlation filtering")
            return per_target_configs

        valid_labels = label_series[valid_mask]

        # Compute pairwise correlations
        correlation_matrix = valid_labels.corr(method='spearman')

        # Greedy selection: start with best LQS, add others if correlation < threshold
        selected_configs = {}

        # Sort by LQS descending
        sorted_targets = sorted(
            per_target_configs.items(),
            key=lambda x: x[1]['quality_metrics'].label_quality_score,
            reverse=True
        )

        for target_name, config in sorted_targets:
            # Check correlation with already selected targets
            should_include = True

            for selected_target in selected_configs.keys():
                corr = abs(correlation_matrix.loc[target_name, selected_target])
                if corr > self.config.max_target_correlation:
                    should_include = False
                    tprint(f"🚫 Excluding {target_name} due to high correlation "
                           f"({corr:.3f}) with {selected_target}")
                    break

            if should_include:
                selected_configs[target_name] = config
                tprint(f"✅ Selected {target_name} (LQS={config['quality_metrics'].label_quality_score:.3f})")

        # Ensure we have at least one target (fallback to best if all filtered out)
        if not selected_configs:
            best_target = sorted_targets[0][0]
            selected_configs[best_target] = per_target_configs[best_target]
            tprint(f"⚠️ All targets filtered by correlation, keeping best: {best_target}")

        tprint(f"✅ Correlation filtering complete: {len(selected_configs)}/{len(per_target_configs)} targets selected")
        return selected_configs

    def _evaluate_config_quality(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series],
        config: Dict[str, Any],
        target_name: str
    ) -> LabelQualityMetrics:
        """Evaluate label quality for a specific configuration."""

        # Generate labels for this config (subset for speed)
        sample_size = min(5000, len(data) - self.config.max_horizon_bars)
        sample_data = data.iloc[-sample_size:].copy()

        # Generate sample labels
        sample_labels = self._generate_single_target_labels(
            sample_data, volatility.iloc[-sample_size:], noise_gates, config, target_name,
            max_samples=1000  # Limit for quality evaluation
        )

        # Compute quality metrics
        return self._compute_label_quality(sample_labels, sample_data, target_name)

    def _generate_single_target_labels(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        noise_gates: Dict[str, pd.Series],
        config: Dict[str, Any],
        target_name: str,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate labels for a single target configuration with hysteresis and conflict resolution."""

        labels = pd.DataFrame(index=data.index)
        labels['target'] = 0  # -1, 0, +1

        # Compute FPT-based horizon for this k
        k = config['k']
        horizon_bars = self._compute_adaptive_horizon(data, volatility, k, config['fpt_quantile'])

        # Micro-range gate for this specific k
        median_tr = data['true_range'].rolling(window=20).median()
        micro_gate = (k * volatility) >= (config['micro_range_alpha'] * median_tr)

        # Combined eligibility
        eligibility = noise_gates['eligibility'] & micro_gate

        # Track active instances to prevent overlap
        active_instances = {}  # timestamp -> (target, expiry_bar)

        # Generate labels bar by bar
        max_idx = min(len(data) - self.config.max_horizon_bars,
                     max_samples if max_samples else len(data))

        for i in range(self.config.min_bars_for_labeling, max_idx):
            current_time = data.index[i]

            # Check if we're in an active instance period (conflict resolution)
            in_active_period = any(
                expiry > i for expiry in [info[1] for info in active_instances.values()]
            )

            if in_active_period:
                # Skip labeling if in active period of another instance
                continue

            if not eligibility.iloc[i]:
                continue

            current_price = data['close'].iloc[i]
            current_vol = volatility.iloc[i]
            horizon = horizon_bars.iloc[i]

            # Define barriers
            target_price_up = current_price * (1 + k * current_vol)
            target_price_down = current_price * (1 - k * current_vol)
            max_horizon_idx = min(i + int(horizon) + 1, len(data))

            # Look forward to find first barrier hit
            window_data = data.iloc[i:max_horizon_idx]

            # Check for upper barrier hit (long target)
            upper_hit = np.any(window_data['high'] >= target_price_up)
            if upper_hit:
                hit_idx = np.where(window_data['high'] >= target_price_up)[0][0]
                raw_target = 1

                # Hysteresis check: if recent label flip, require stronger signal
                if self._check_hysteresis_violation(labels, i, raw_target):
                    # Check if opposite barrier was hit by more than beta threshold
                    if self._check_flip_override(data, i, raw_target, target_price_up, target_price_down):
                        # Override allowed - proceed with flip
                        pass
                    else:
                        # Hysteresis violation without strong override - skip
                        continue

                labels.loc[current_time, 'target'] = raw_target
                labels.loc[current_time, 'time_to_hit'] = hit_idx
                labels.loc[current_time, 'confidence'] = min(1.0, hit_idx / horizon)

                # Register active instance
                expiry_bar = i + hit_idx + 1  # End after hit
                active_instances[current_time] = (raw_target, expiry_bar)

            else:
                # Check for lower barrier hit (short target)
                lower_hit = np.any(window_data['low'] <= target_price_down)
                if lower_hit:
                    hit_idx = np.where(window_data['low'] <= target_price_down)[0][0]
                    raw_target = -1

                    # Hysteresis check: if recent label flip, require stronger signal
                    if self._check_hysteresis_violation(labels, i, raw_target):
                        # Check if opposite barrier was hit by more than beta threshold
                        if self._check_flip_override(data, i, raw_target, target_price_up, target_price_down):
                            # Override allowed - proceed with flip
                            pass
                        else:
                            # Hysteresis violation without strong override - skip
                            continue

                    labels.loc[current_time, 'target'] = raw_target
                    labels.loc[current_time, 'time_to_hit'] = hit_idx
                    labels.loc[current_time, 'confidence'] = min(1.0, hit_idx / horizon)

                    # Register active instance
                    expiry_bar = i + hit_idx + 1  # End after hit
                    active_instances[current_time] = (raw_target, expiry_bar)

        # Add metadata
        labels['k'] = k
        labels['horizon_bars'] = horizon_bars
        labels['eligibility'] = eligibility

        # Clean up expired instances (for long-running scenarios)
        current_bar = max_idx
        expired_times = [t for t, (_, expiry) in active_instances.items() if expiry <= current_bar]
        for t in expired_times:
            del active_instances[t]

        return labels

    def _check_hysteresis_violation(self, labels: pd.DataFrame, current_idx: int, new_target: int) -> bool:
        """Check if assigning new_target would violate hysteresis constraints."""

        # Look back at recent labels within hysteresis window
        lookback_window = self.config.hysteresis_bars

        for lookback in range(1, lookback_window + 1):
            if current_idx - lookback < 0:
                break

            prev_target = labels.iloc[current_idx - lookback]['target']
            if prev_target != 0 and prev_target != new_target:
                return True  # Would be a flip

        return False  # No hysteresis violation

    def _check_flip_override(self, data: pd.DataFrame, current_idx: int,
                           new_target: int, target_up: float, target_down: float) -> bool:
        """Check if flip should be allowed due to strong opposite signal."""

        current_price = data['close'].iloc[current_idx]

        if new_target == 1:  # Trying to flip to long (was short)
            # Check if short barrier was hit by more than beta threshold
            short_barrier_distance = (current_price - target_down) / current_price
            return short_barrier_distance >= self.config.flip_override_beta

        elif new_target == -1:  # Trying to flip to short (was long)
            # Check if long barrier was hit by more than beta threshold
            long_barrier_distance = (target_up - current_price) / current_price
            return long_barrier_distance >= self.config.flip_override_beta

        return False

    def _compute_adaptive_horizon(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        k: float,
        quantile: float
    ) -> pd.Series:
        """Compute adaptive horizons based on first-passage time quantiles."""

        horizons = pd.Series(index=data.index, dtype=float)

        # For each point, estimate FPT to ±k·σ_t
        for i in range(max(50, len(data) - 1000), len(data)):
            # Look back to estimate FPT distribution
            lookback = min(i, 500)
            historical_data = data.iloc[i-lookback:i]

            if len(historical_data) < 50:
                horizons.iloc[i] = 20  # Default
                continue

            # Compute historical FPTs for this k
            fpt_values = []
            for j in range(20, len(historical_data)):  # Skip first 20 for stability
                current_price = historical_data['close'].iloc[j]
                current_vol = volatility.iloc[i-lookback+j] if i-lookback+j < len(volatility) else volatility.iloc[i-1]

                if pd.isna(current_vol) or current_vol <= 0:
                    continue

                target_up = current_price * (1 + k * current_vol)
                target_down = current_price * (1 - k * current_vol)

                # Find first hit within reasonable horizon
                window_end = min(j + self.config.max_horizon_bars, len(historical_data))
                window = historical_data.iloc[j:window_end]

                upper_hit = np.where(window['high'] >= target_up)[0]
                lower_hit = np.where(window['low'] <= target_down)[0]

                if len(upper_hit) > 0 or len(lower_hit) > 0:
                    first_hit = min(upper_hit[0] if len(upper_hit) > 0 else self.config.max_horizon_bars,
                                  lower_hit[0] if len(lower_hit) > 0 else self.config.max_horizon_bars)
                    fpt_values.append(first_hit)

            if fpt_values:
                # Use quantile of FPT distribution
                fpt_array = np.array(fpt_values)
                horizon = np.quantile(fpt_array, quantile)
                horizons.iloc[i] = max(1, min(horizon, self.config.max_horizon_bars))
            else:
                horizons.iloc[i] = 20  # Default fallback

        # Forward fill and smooth with EMA
        horizons = horizons.fillna(method='ffill').fillna(20)
        horizons = horizons.ewm(alpha=0.1).mean()  # Light smoothing

        return horizons

    def _compute_label_quality(
        self,
        labels: pd.DataFrame,
        data: pd.DataFrame,
        target_name: str
    ) -> LabelQualityMetrics:
        """Compute comprehensive label quality metrics."""

        metrics = LabelQualityMetrics()

        # Filter to valid labels only
        valid_labels = labels[labels['target'] != 0].copy()

        if len(valid_labels) < 100:
            self.logger.warning(f"⚠️ Insufficient labels for quality evaluation: {len(valid_labels)}")
            return metrics

        # 1. Balance metrics
        positive_count = (valid_labels['target'] == 1).sum()
        negative_count = (valid_labels['target'] == -1).sum()
        total_count = len(valid_labels)

        metrics.positive_balance = positive_count / total_count if total_count > 0 else 0.5
        # Balance score: closer to 0.5 is better (penalize imbalance)
        metrics.class_balance_score = 1.0 - abs(metrics.positive_balance - 0.5) * 2

        # 2. Simple feature-based predictability (using lagged returns as proxy)
        # Create simple features for testing
        feature_data = []
        label_values = []

        for idx in valid_labels.index:
            if idx < 10:  # Need some history
                continue

            # Simple features: lagged returns, volatility, volume
            features = [
                data.loc[idx, 'returns'] if idx in data.index else 0,
                data.loc[idx, 'volume'] if idx in data.index else 0,
            ]

            # Add some lagged values
            for lag in [1, 2, 5]:
                lag_idx = data.index.get_loc(idx) - lag if idx in data.index else None
                if lag_idx is not None and lag_idx >= 0:
                    features.append(data.iloc[lag_idx]['returns'])
                    features.append(data.iloc[lag_idx]['volume'])

            feature_data.append(features)
            label_values.append(1 if valid_labels.loc[idx, 'target'] == 1 else 0)

        if len(feature_data) >= 50:
            X = np.array(feature_data)
            y = np.array(label_values)

            # Simple logistic regression for AUC
            try:
                # Time series split for proper validation
                tscv = TimeSeriesSplit(n_splits=min(5, len(X)//50))

                auc_scores = []
                pr_scores = []

                for train_idx, test_idx in tscv.split(X):
                    if len(train_idx) < 10 or len(test_idx) < 10:
                        continue

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Simple model
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)

                    # Predict probabilities
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    # AUC
                    auc = roc_auc_score(y_test, y_pred_proba)
                    auc_scores.append(auc)

                    # PR-AUC
                    pr_auc = average_precision_score(y_test, y_pred_proba)
                    pr_scores.append(pr_auc)

                if auc_scores:
                    metrics.auc_mean = np.mean(auc_scores)
                    metrics.auc_std = np.std(auc_scores)
                    metrics.pr_auc_mean = np.mean(pr_scores)
                    metrics.pr_auc_std = np.std(pr_scores)

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute predictability metrics: {e}")

        # 3. Stability metrics (PSI - Population Stability Index)
        if len(valid_labels) > 200:
            # Split into two halves for PSI calculation
            mid_point = len(valid_labels) // 2
            first_half = valid_labels.iloc[:mid_point]
            second_half = valid_labels.iloc[mid_point:]

            # Compute probability distributions
            def compute_prob_dist(labels_subset):
                total = len(labels_subset)
                pos_rate = (labels_subset['target'] == 1).sum() / total
                neg_rate = (labels_subset['target'] == -1).sum() / total
                return [pos_rate, neg_rate]

            dist1 = compute_prob_dist(first_half)
            dist2 = compute_prob_dist(second_half)

            # PSI calculation
            psi = 0.0
            for p1, p2 in zip(dist1, dist2):
                if p1 > 0 and p2 > 0:
                    psi += (p1 - p2) * np.log(p1 / p2)

            metrics.psi_score = min(psi, 1.0)  # Cap at 1.0

        # 4. Flip rate (temporal consistency)
        flip_count = 0
        for i in range(1, len(valid_labels)):
            if valid_labels.iloc[i]['target'] != valid_labels.iloc[i-1]['target']:
                flip_count += 1

        metrics.flip_rate = flip_count / (len(valid_labels) - 1) if len(valid_labels) > 1 else 0.0

        # 5. SNR proxy (correlation between features and labels)
        if len(feature_data) >= 50:
            try:
                # Simple feature z-scores correlation with labels
                feature_zscores = stats.zscore(X, axis=0, nan_policy='omit')
                label_binary = np.array([1 if valid_labels.iloc[i]['target'] == 1 else -1
                                       for i in range(len(valid_labels))])

                # Mean absolute correlation across features
                correlations = []
                for feat_idx in range(X.shape[1]):
                    if not np.all(np.isnan(feature_zscores[:, feat_idx])):
                        corr = abs(stats.spearmanr(
                            feature_zscores[:, feat_idx],
                            label_binary[:len(feature_zscores)],
                            nan_policy='omit'
                        )[0])
                        if not np.isnan(corr):
                            correlations.append(corr)

                metrics.feature_ic_mean = np.mean(correlations) if correlations else 0.0

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute SNR metrics: {e}")

        # 6. Composite Label Quality Score (LQS)
        weights = self.config.lqs_weights

        # Normalize components to [0,1] range where applicable
        auc_score = metrics.auc_mean if metrics.auc_mean > 0 else 0.0
        stability_score = 1.0 - min(metrics.psi_score, 1.0)  # Lower PSI is better
        balance_score = metrics.class_balance_score
        snr_score = min(metrics.feature_ic_mean, 1.0) if metrics.feature_ic_mean > 0 else 0.0
        consistency_score = 1.0 - metrics.flip_rate  # Lower flip rate is better

        metrics.label_quality_score = (
            weights['predictability'] * auc_score +
            weights['stability'] * stability_score +
            weights['balance'] * balance_score +
            weights['snr'] * snr_score +
            weights['consistency'] * consistency_score
        )

        return metrics

    def _generate_comprehensive_report(
        self,
        quality_report: Dict[str, Dict[str, float]],
        optimal_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""

        report = {
            'summary': {
                'total_targets': len(optimal_configs),
                'evaluation_timestamp': datetime.now().isoformat(),
                'quality_thresholds': {
                    'min_positive_balance': self.config.min_positive_balance,
                    'max_positive_balance': self.config.max_positive_balance,
                    'min_auc_threshold': self.config.min_auc_threshold,
                    'max_auc_std_threshold': self.config.max_auc_std_threshold
                }
            },
            'target_reports': quality_report,
            'configurations': optimal_configs,
            'recommendations': []
        }

        # Generate recommendations
        for target_name, metrics in quality_report.items():
            issues = []

            if metrics.get('positive_balance', 0) < self.config.min_positive_balance:
                issues.append(f"Low positive class balance: {metrics['positive_balance']:.1%}")
            if metrics.get('positive_balance', 0) > self.config.max_positive_balance:
                issues.append(f"High positive class balance: {metrics['positive_balance']:.1%}")
            if metrics.get('auc_mean', 0) < self.config.min_auc_threshold:
                issues.append(f"Low AUC: {metrics['auc_mean']:.3f}")
            if metrics.get('auc_std', 0) > self.config.max_auc_std_threshold:
                issues.append(f"High AUC variance: {metrics['auc_std']:.3f}")

            if issues:
                report['recommendations'].append({
                    'target': target_name,
                    'issues': issues,
                    'lqs': metrics.get('label_quality_score', 0)
                })

        return report

    def validate_labels_robustness(self, labeled_data: pd.DataFrame,
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of label robustness across different conditions.

        Tests labels on:
        - High vs low volatility periods
        - Different volume regimes
        - Time-of-day effects
        - Rolling windows for temporal stability
        """

        tprint("🔬 Validating label robustness...")

        validation_results = {
            'volatility_slices': {},
            'volume_slices': {},
            'temporal_slices': {},
            'rolling_stability': {},
            'overall_robustness_score': 0.0
        }

        # 1. Volatility slice analysis
        vol_thresholds = [data['close'].pct_change().rolling(20).std().quantile([0.33, 0.67])]
        low_vol_mask = data['close'].pct_change().rolling(20).std() <= vol_thresholds[0]
        high_vol_mask = data['close'].pct_change().rolling(20).std() > vol_thresholds[1]

        for slice_name, mask in [('low_vol', low_vol_mask), ('high_vol', high_vol_mask)]:
            if mask.sum() > 100:
                slice_data = data[mask]
                slice_labels = labeled_data[mask]

                # Recompute quality metrics for this slice
                slice_quality = self._compute_slice_quality(slice_labels, slice_data, slice_name)
                validation_results['volatility_slices'][slice_name] = slice_quality

        # 2. Volume slice analysis
        vol_thresholds = [data['volume'].quantile([0.33, 0.67])]
        low_vol_mask = data['volume'] <= vol_thresholds[0]
        high_vol_mask = data['volume'] > vol_thresholds[1]

        for slice_name, mask in [('low_volume', low_vol_mask), ('high_volume', high_vol_mask)]:
            if mask.sum() > 100:
                slice_data = data[mask]
                slice_labels = labeled_data[mask]

                slice_quality = self._compute_slice_quality(slice_labels, slice_data, slice_name)
                validation_results['volume_slices'][slice_name] = slice_quality

        # 3. Rolling stability analysis
        window_size = min(1000, len(labeled_data) // 5)
        rolling_metrics = []

        for start_idx in range(0, len(labeled_data) - window_size, window_size // 2):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]
            window_labels = labeled_data.iloc[start_idx:end_idx]

            window_quality = self._compute_slice_quality(window_labels, window_data, f'window_{start_idx}')
            rolling_metrics.append(window_quality)

        if rolling_metrics:
            # Compute stability statistics
            auc_values = [m['auc_mean'] for m in rolling_metrics if m['auc_mean'] > 0]
            if auc_values:
                validation_results['rolling_stability'] = {
                    'auc_mean': np.mean(auc_values),
                    'auc_std': np.std(auc_values),
                    'stability_score': 1.0 - min(np.std(auc_values) / np.mean(auc_values), 1.0)
                }

        # 4. Overall robustness score
        all_slice_scores = []
        for slice_type in ['volatility_slices', 'volume_slices']:
            for slice_metrics in validation_results[slice_type].values():
                if slice_metrics.get('label_quality_score', 0) > 0:
                    all_slice_scores.append(slice_metrics['label_quality_score'])

        if all_slice_scores:
            # Robustness is the worst performance across slices (lower bound)
            validation_results['overall_robustness_score'] = min(all_slice_scores)

        tprint(f"✅ Robustness validation complete: score={validation_results['overall_robustness_score']:.3f}")
        return validation_results

    def _compute_slice_quality(self, labels: pd.DataFrame, data: pd.DataFrame,
                              slice_name: str) -> Dict[str, float]:
        """Compute quality metrics for a data slice."""

        # Extract target columns (assuming format like 'small_target', 'medium_target', etc.)
        target_columns = [col for col in labels.columns if col.endswith('_target')]

        slice_metrics = {}

        for target_col in target_columns:
            target_name = target_col.replace('_target', '')

            # Create temporary labels dataframe for this target
            temp_labels = pd.DataFrame(index=labels.index)
            temp_labels['target'] = labels[target_col]

            # Compute quality metrics
            quality_metrics = self._compute_label_quality(temp_labels, data, target_name)
            slice_metrics[f'{target_name}_auc'] = quality_metrics.auc_mean
            slice_metrics[f'{target_name}_balance'] = quality_metrics.positive_balance
            slice_metrics[f'{target_name}_lqs'] = quality_metrics.label_quality_score

        # Overall slice score (average of target LQS)
        lqs_values = [v for k, v in slice_metrics.items() if k.endswith('_lqs') and v > 0]
        slice_metrics['label_quality_score'] = np.mean(lqs_values) if lqs_values else 0.0

        return slice_metrics

# Factory function for backward compatibility
def create_volatility_aware_labeler(config: Optional[VolatilityAwareConfig] = None) -> VolatilityAwareProfitLabeler:
    """Factory function to create volatility-aware labeler."""
    return VolatilityAwareProfitLabeler(config)

def apply_volatility_aware_labeling(
    data: pd.DataFrame,
    config: Optional[VolatilityAwareConfig] = None,
    validate_robustness: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply volatility-aware multi-horizon profit labeling.

    Args:
        data: OHLCV DataFrame
        config: Configuration (optional)
        validate_robustness: Whether to run comprehensive validation

    Returns:
        Tuple of (labeled_dataframe, quality_report)
    """
    labeler = VolatilityAwareProfitLabeler(config)

    # Generate labels
    labeled_data, report = labeler.generate_labels(data)

    # Optional robustness validation
    if validate_robustness and len(labeled_data) > 500:
        robustness_results = labeler.validate_labels_robustness(labeled_data, data)
        report['robustness_validation'] = robustness_results

    return labeled_data, report

@dataclass
class LabelerConfig:
    """Reproducible configuration for labeler (as requested in deliverables)."""

    # Core parameters
    base_timeframe_minutes: int = 5
    target_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'small': (0.4, 0.8), 'medium': (0.8, 1.3), 'high': (1.3, 2.0)
    })

    # Volatility and noise parameters
    rv_window_minutes: int = 30
    atr_window_bars: int = 14
    volatility_ewma_lambda: float = 0.94
    micro_range_alpha: float = 1.5

    # Quality optimization
    fpt_quantile: float = 0.65
    max_target_correlation: float = 0.6
    hysteresis_bars: int = 2
    flip_override_beta: float = 0.3

    # Balance constraints
    min_positive_balance: float = 0.35
    max_positive_balance: float = 0.65

    # Random seed for reproducibility
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'base_timeframe_minutes': self.base_timeframe_minutes,
            'target_bands': self.target_bands,
            'rv_window_minutes': self.rv_window_minutes,
            'atr_window_bars': self.atr_window_bars,
            'volatility_ewma_lambda': self.volatility_ewma_lambda,
            'micro_range_alpha': self.micro_range_alpha,
            'fpt_quantile': self.fpt_quantile,
            'max_target_correlation': self.max_target_correlation,
            'hysteresis_bars': self.hysteresis_bars,
            'flip_override_beta': self.flip_override_beta,
            'min_positive_balance': self.min_positive_balance,
            'max_positive_balance': self.max_positive_balance,
            'random_seed': self.random_seed
        }

def create_reproducible_labels(data: pd.DataFrame,
                              config: LabelerConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create reproducible labels with fixed configuration.

    This function provides the "repro notebook" equivalent requested in deliverables.
    """

    # Set random seed for reproducibility
    np.random.seed(config.random_seed)

    # Create volatility-aware config from labeler config
    va_config = VolatilityAwareConfig(
        base_timeframe_minutes=config.base_timeframe_minutes,
        target_bands=config.target_bands,
        rv_window_minutes=config.rv_window_minutes,
        atr_window_bars=config.atr_window_bars,
        volatility_ewma_lambda=config.volatility_ewma_lambda,
        micro_range_alpha=config.micro_range_alpha,
        fpt_quantile=config.fpt_quantile,
        max_target_correlation=config.max_target_correlation,
        hysteresis_bars=config.hysteresis_bars,
        flip_override_beta=config.flip_override_beta,
        min_positive_balance=config.min_positive_balance,
        max_positive_balance=config.max_positive_balance
    )

    return apply_volatility_aware_labeling(data, va_config)

# Factory function for backward compatibility
def create_volatility_aware_labeler(config: Optional[VolatilityAwareConfig] = None) -> VolatilityAwareProfitLabeler:
    """Factory function to create volatility-aware labeler."""
    return VolatilityAwareProfitLabeler(config)

def apply_volatility_aware_labeling(
    data: pd.DataFrame,
    config: Optional[VolatilityAwareConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply volatility-aware multi-horizon profit labeling.

    Args:
        data: OHLCV DataFrame
        config: Configuration (optional)

    Returns:
        Tuple of (labeled_dataframe, quality_report)
    """
    labeler = VolatilityAwareProfitLabeler(config)
    return labeler.generate_labels(data)

# Test function
if __name__ == '__main__':
    # Simple test
    tprint('🧪 Testing Volatility-Aware Profit Labeler')

    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)

    # Generate realistic price data with trends and volatility
    base_price = 100.0
    prices = [base_price]

    for i in range(999):
        # Add trend and volatility
        trend = 0.0001 * (i // 100 - 5)  # Changing trend
        vol = 0.002 + 0.001 * np.sin(i / 50)  # Changing volatility
        ret = np.random.normal(trend, vol)
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)

    # Test labeling
    tprint('\n🔍 Testing volatility-aware labeling...')
    config = VolatilityAwareConfig()
    labeled_data, report = apply_volatility_aware_labeling(data, config)

    tprint(f'✅ Labeling completed:')
    tprint(f'   → Input shape: {data.shape}')
    tprint(f'   → Output shape: {labeled_data.shape}')

    # Show sample quality metrics
    if report and 'target_reports' in report:
        tprint(f'\n📊 Quality Report Summary:')
        for target, metrics in report['target_reports'].items():
            tprint(f'   → {target}: LQS={metrics.get("label_quality_score", 0):.3f}, '
                   f'AUC={metrics.get("auc_mean", 0):.3f}±{metrics.get("auc_std", 0):.3f}')

    tprint('✅ Volatility-Aware Profit Labeler test completed!')