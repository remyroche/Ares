"""
Spectral Specialists for Adaptive Event-Driven Labeling (AEDL)

This module transforms traditional specialists into 5-scale spectral versions
optimized for frequency-dependent analysis and cross-scale resonance detection.

Key Features:
- Transform 4 priority specialists to spectral domain
- 5-scale decomposition for each specialist
- Integration with existing causal specialist framework
- Optimized for 2-4 hour trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, asdict

from src.feature_generation.utils.step06_labeling_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
)
from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

# Try to import numba for performance optimizations
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# Import bottleneck for optimized rolling operations
try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False
    tprint_warning("⚠️ bottleneck not available - falling back to pandas (slower)")

# Import scipy for entropy (with guard)
try:
    from scipy.stats import entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_entropy = None


@njit(cache=True)
def _rolling_entropy_numba(returns: np.ndarray, window: int, n_bins: int = 10) -> np.ndarray:
    """Numba-optimized rolling entropy calculation.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        n_bins: Number of histogram bins
        
    Returns:
        Array of rolling entropy values
    """
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = returns[i - window:i]
        
        # Remove NaN values
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) < window // 2:  # Need at least half the window
            continue
            
        # Compute histogram
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        if data_max - data_min < 1e-10:  # Constant data
            result[i] = 0.0
            continue
            
        # Manual histogram computation
        bin_edges = np.linspace(data_min, data_max, n_bins + 1)
        counts = np.zeros(n_bins)
        
        for val in valid_data:
            # Find bin index
            bin_idx = int((val - data_min) / (data_max - data_min) * (n_bins - 1))
            bin_idx = min(bin_idx, n_bins - 1)  # Clamp to valid range
            counts[bin_idx] += 1
        
        # Compute Shannon entropy
        total = np.sum(counts)
        if total > 0:
            entropy_val = 0.0
            for count in counts:
                if count > 0:
                    prob = count / total
                    entropy_val -= prob * np.log(prob + 1e-9)
            result[i] = entropy_val
    
    return result

def _rolling_mad(values: np.ndarray) -> float:
    """Median absolute deviation helper for rolling windows."""
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


@njit
def ewma_numba(arr: np.ndarray, span: int) -> np.ndarray:
    """Numba-optimized exponential weighted moving average."""
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * arr[i] + (1.0 - alpha) * result[i-1]
    return result


@njit
def fast_rolling_entropy_numba(arr: np.ndarray, window: int, bins: int = 10) -> np.ndarray:
    """Numba-optimized rolling entropy calculation."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        
        # Remove NaN values
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) < window // 2:  # Need at least half the window
            continue
            
        # Manual histogram calculation
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
        if max_val - min_val < 1e-10:  # Constant values
            result[i] = 0.0
            continue
            
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        hist = np.zeros(bins)
        
        for val in valid_data:
            bin_idx = int((val - min_val) / (max_val - min_val) * (bins - 1))
            bin_idx = min(max(bin_idx, 0), bins - 1)
            hist[bin_idx] += 1
        
        # Normalize
        hist = hist / hist.sum()
        
        # Calculate entropy
        entropy_val = 0.0
        for h in hist:
            if h > 1e-10:
                entropy_val -= h * np.log(h)
        
        result[i] = entropy_val
    
    return result


def safe_rolling_std(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Optimized rolling std using bottleneck if available."""
    if HAS_BOTTLENECK:
        arr = series.values
        result = bn.move_std(arr, window=window, min_count=min_periods)
        return pd.Series(result, index=series.index)
    else:
        return series.rolling(window, min_periods=min_periods).std()


def safe_rolling_mean(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Optimized rolling mean using bottleneck if available."""
    if HAS_BOTTLENECK:
        arr = series.values
        result = bn.move_mean(arr, window=window, min_count=min_periods)
        return pd.Series(result, index=series.index)
    else:
        return series.rolling(window, min_periods=min_periods).mean()


@dataclass
class TBMConfig:
    """Triple-barrier configuration shared across specialists."""

    profit_take_multiplier: float = 0.015  # Tripled from 0.004
    stop_loss_multiplier: float = 0.006   # Tripled from 0.0025
    time_barrier_minutes: int = 720        # 48 bars * 15m = 720m
    max_lookahead: int = 100               # Cover 48 bars with buffer
    binary_classification: bool = True
    transaction_cost: float = DEFAULT_TRANSACTION_COST

    def merge(self, overrides: Optional[Dict[str, Any]] = None) -> "TBMConfig":
        data = asdict(self)
        if overrides:
            data.update(overrides)
        return TBMConfig(**data)

    def to_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveVolatilityFilterConfig:
    """Adaptive volatility filter parameters."""

    window: int = 48
    floor_multiplier: float = 1.25
    min_vol_rank: float = 0.05
    hard_floor: float = 1e-4
    max_surprise: float = 8.0
    eps: float = 1e-9

    def merge(self, overrides: Optional[Dict[str, Any]] = None) -> "AdaptiveVolatilityFilterConfig":
        data = asdict(self)
        if overrides:
            data.update(overrides)
        return AdaptiveVolatilityFilterConfig(**data)


@dataclass
class SpecialistEventConfig:
    """Event calibration configuration."""

    base_activation_zscore: float = 1.5
    min_coverage: float = 0.04
    max_coverage: float = 0.25
    surprise_scaler: float = 0.75
    min_events: int = 30
    responsiveness_floor: float = 0.05
    correlation_threshold: float = 0.85

    def merge(self, overrides: Optional[Dict[str, Any]] = None) -> "SpecialistEventConfig":
        data = asdict(self)
        if overrides:
            data.update(overrides)
        return SpecialistEventConfig(**data)


class SpectralSpecialists:
    """
    Transform traditional specialists into 5-scale spectral versions.
    
    Priority Specialists for 2-4h trades:
    1. Inventory Specialist (Priority 1) - Dealer exhaustion detection
    2. Volume Specialist (Priority 2) - Micro-surge vs macro-trend resonance
    3. Volatility Specialist (Priority 3) - Dynamic wavelet thresholding
    4. Information Specialist (Causal Addition) - PIN/VPIN informed flow resonance
    """
    
    def __init__(
        self,
        priority_specialists: List[str] = None,
        verbose: bool = True,
        tbm_config: Optional[Dict[str, Any]] = None,
        avf_config: Optional[Dict[str, Any]] = None,
        event_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Spectral Specialists transformer.
        
        Args:
            priority_specialists: List of specialist names to prioritize
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        # Default now includes the new 2026 De Prado specialists
        self.priority_specialists = priority_specialists or [
            'inventory_specialist',
            'volume_specialist', 
            'volatility_specialist',
            'information_specialist',
            'cusum_break_specialist',
            'entropy_specialist',
            'tick_rule_specialist',
            'fractal_efficiency_specialist',
            'liquidity_shock_specialist',
            'gap_specialist',
            'trend_specialist',
            'reversal_specialist',
            'volatility_breakout_specialist',
            'cross_asset_specialist'
        ]
        
        # Specialist descriptions (removed dead key_scales/resonance_pairs fields)
        self.specialist_descriptions = {
            'inventory_specialist': {
                'priority': 1,
                'description': 'Dealer exhaustion detection',
                'role': 'Micro-divergence in 15m wavelet before 4h trend impact'
            },
            'volume_specialist': {
                'priority': 2,
                'description': 'Micro-surge vs macro-trend resonance',
                'role': 'Detect volume micro-surge resonating with macro-trend'
            },
            'volatility_specialist': {
                'priority': 3,
                'description': 'Volatility Z-Score and Shock Detection',
                'role': 'Detects volatility shocks as causal precursors to risk events'
            },
            'information_specialist': {
                'priority': 4,
                'description': 'Price Action and Microstructure Signatures',
                'role': 'Strongest predictor of permanent price moves via PA info'
            },
            'cusum_break_specialist': {
                'priority': 5,
                'description': 'Structural Break Detection (CUSUM)',
                'role': 'Detect regime shifts where underlying process changes'
            },
            'entropy_specialist': {
                'priority': 6,
                'description': 'Market Entropy / Unpredictability',
                'role': 'Measure information content and signal-to-noise breakdown'
            },
            'tick_rule_specialist': {
                'priority': 7,
                'description': 'Aggressor Flow Proxy',
                'role': 'Approximates buy vs sell pressure within bars'
            },
            'fractal_efficiency_specialist': {
                'priority': 8,
                'description': 'Fractal Efficiency (Kaufman/Hurst)',
                'role': 'Distinguish directional trends (clean) from random walks (noisy)'
            },
            'liquidity_shock_specialist': {
                'priority': 9,
                'description': 'Liquidity Shock (Amihud Proxy)',
                'role': 'Detects structural liquidity failures (Price Ease)'
            },
            'gap_specialist': {
                'priority': 10,
                'description': 'Exogenous Shock (Gap)',
                'role': 'Detects overnight/weekend information injection'
            },
            'trend_specialist': {
                'priority': 11,
                'description': 'Trend Persistence (Rolling Returns)',
                'role': 'Captures directional alpha and trend persistence'
            },
            'reversal_specialist': {
                'priority': 12,
                'description': 'Mean Reversion (Oscillator)',
                'role': 'Detects overextended price action and mean-reversion events'
            },
            'volatility_breakout_specialist': {
                'priority': 13,
                'description': 'Volatility Breakout (HL vs Baseline)',
                'role': 'Detects unexpected volatility/range expansion vs baseline'
            },
            'cross_asset_specialist': {
                'priority': 14,
                'description': 'Cross-Asset Resonance (Beta/Lead-Lag)',
                'role': 'Detects systemic shocks and lead-lag relationships across assets'
            }
        }
        
        self.tbm_config = TBMConfig().merge(tbm_config or {})
        self.avf_config = AdaptiveVolatilityFilterConfig().merge(avf_config or {})
        self.event_config = SpecialistEventConfig().merge(event_config or {})
        self._tbm_engine = OptimizedTripleBarrierLabeling(**self.tbm_config.to_kwargs())
        self._reliability_registry: Dict[str, Dict[str, Any]] = {}
        self._last_extracted_specialists: List[str] = []
        self._cached_diversity_report: Dict[str, Any] = {}

        if self.verbose:
            tprint_info("🎯 Spectral Specialists: Initializing...")
            tprint_info(f"   ⚙️ Priority specialists: {len(self.priority_specialists)}")
            for specialist in self.priority_specialists:
                desc = self.specialist_descriptions.get(specialist, {})
                tprint_info(f"      - {specialist}: {desc.get('description', 'N/A')}")
            tprint_success("   ✅ Spectral Specialists: Initialization complete")
    
    def extract_specialist_signals(
        self,
        df_input: pd.DataFrame,
        specialist_configs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Extract raw specialist signals from market data.
        
        Args:
            df_input: Market data with OHLCV and derived features
            specialist_configs: Configuration for each specialist
            
        Returns:
            Dictionary of specialist time series
        """
        df = df_input.copy()
        
        try:
            if self.verbose:
                tprint_info("📊 Extracting raw specialist signals...")
            
            configs = specialist_configs or {}
            asset_context = configs.get("asset_context", {})
            asset_col = asset_context.get("asset_id_col")
            if asset_col is None:
                for candidate in ("asset_id", "asset", "ticker", "symbol"):
                    if candidate in df.columns:
                        asset_col = candidate
                        break
            group_by_asset = bool(asset_context.get("group_by_asset", True))

            if asset_col and group_by_asset and df[asset_col].nunique(dropna=False) > 1:
                if self.verbose:
                    tprint_info(f"   🧩 Grouping specialist extraction by asset ({asset_col})")
                return self._extract_signals_by_asset(df, configs, asset_col)

            # Proactively deduplicate incoming dataframe ONLY for single-asset data (or if grouping skipped)
            # to avoid downstream reindex/join errors
            if df.index.has_duplicates:
                num_dups = df.index.duplicated().sum()
                tprint_warning(f"   ⚠️ Input dataframe contains {num_dups} duplicate timestamps; keeping latest occurrences.")
                df = df.loc[~df.index.duplicated(keep='last')]

            # Early data validation
            df = self._validate_input_data(df)
            
            specialist_signals = {}
            
            # Pre-compute common rolling statistics to speed up specialists
            if self.verbose:
                tprint_info("⚡ Pre-computing rolling statistics...")
            
            # Common rolling windows used by multiple specialists (using bottleneck if available)
            rolling_stats = {}
            if len(df) > 50:
                # Returns statistics
                returns = df['close'].pct_change()
                rolling_stats['returns'] = returns
                
                # Volatility
                rolling_stats['volatility_20'] = safe_rolling_std(returns, 20, min_periods=1)
                rolling_stats['volatility_50'] = safe_rolling_std(returns, 50, min_periods=1)
                rolling_stats['returns_std_20'] = rolling_stats['volatility_20']  # Alias for backward compatibility
                rolling_stats['returns_std_50'] = rolling_stats['volatility_50']  # Alias for backward compatibility
                rolling_stats['returns_std_100'] = safe_rolling_std(returns, 100, min_periods=1)
                rolling_stats['returns_mean_50'] = safe_rolling_mean(returns, 50, min_periods=1)
                rolling_stats['returns_mean_100'] = safe_rolling_mean(returns, 100, min_periods=1)
                
                # Volume statistics
                rolling_stats['volume_ma_20'] = safe_rolling_mean(df['volume'], 20, min_periods=1)
                rolling_stats['volume_ma_50'] = safe_rolling_mean(df['volume'], 50, min_periods=1)
                rolling_stats['vol_20'] = rolling_stats['volume_ma_20'] # Alias for backward compatibility
                rolling_stats['vol_50'] = rolling_stats['volume_ma_50'] # Alias for backward compatibility
                
                # Price statistics
                rolling_stats['close_mean_50'] = safe_rolling_mean(df['close'], 50, min_periods=1)
                rolling_stats['close_std_50'] = safe_rolling_std(df['close'], 50, min_periods=1)
                rolling_stats['close_mean_100'] = safe_rolling_mean(df['close'], 100, min_periods=1)
                rolling_stats['close_std_100'] = safe_rolling_std(df['close'], 100, min_periods=1)
                
                # Advanced statistics
                if 'high' in df.columns and 'low' in df.columns:
                    rolling_stats['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
                
                rolling_stats['vol_rank'] = rolling_stats['volatility_20'].rolling(100).rank(pct=True)
            
            # Helper to safely extract and add signal with validation
            def _add_signal(name, extraction_func):
                if name in self.priority_specialists:
                    try:
                        if self.verbose:
                            tprint_info(f"   🔄 Extracting {name}...")
                        signal = extraction_func(df, rolling_stats)
                        if signal is not None:
                            # Validate signal quality
                            signal_quality = self._validate_signal_quality(signal, name)
                            if signal_quality['is_degenerate']:
                                tprint_warning(f"⚠️ {name} signal is degenerate: {signal_quality['issue']}")
                                # Still include but with warning
                            specialist_signals[name] = signal
                            if self.verbose:
                                tprint_info(f"      ✅ {name}: mean={signal_quality['mean']:.6f}, std={signal_quality['std']:.6f}, nan%={signal_quality['nan_pct']:.2f}%")
                        else:
                            if self.verbose:
                                tprint_warning(f"      ⚠️ {name} returned None")
                    except Exception as e:
                        tprint_error(f"❌ {name} extraction failed: {e}")
            
            _add_signal('inventory_specialist', self._extract_inventory_signal)
            _add_signal('volume_specialist', self._extract_volume_signal)
            _add_signal('volatility_specialist', self._extract_volatility_signal)
            _add_signal('information_specialist', self._extract_information_signal)
            
            # New Specialists
            _add_signal('cusum_break_specialist', self._extract_cusum_break_signal)
            _add_signal('entropy_specialist', self._extract_entropy_signal)
            _add_signal('tick_rule_specialist', self._extract_tick_rule_signal)
            _add_signal('fractal_efficiency_specialist', self._extract_fractal_efficiency_signal)
            _add_signal('liquidity_shock_specialist', self._extract_liquidity_shock_signal)
            _add_signal('gap_specialist', self._extract_gap_signal)
            _add_signal('trend_specialist', self._extract_trend_signal)
            _add_signal('reversal_specialist', self._extract_reversal_signal)
            _add_signal('volatility_breakout_specialist', self._extract_volatility_breakout_signal)

            # Optional: Cross-asset / market-state feature signals (ca__/ms__ prefixes)
            cross_asset_cfg = configs.get("cross_asset", {})
            prefixes = tuple(cross_asset_cfg.get("prefixes", ("ca__", "ms__")))
            max_signals = int(cross_asset_cfg.get("max_signals", 6))
            cross_asset_cols = [
                col for col in df.columns if isinstance(col, str) and col.startswith(prefixes)
            ]
            cross_asset_added = 0
            if cross_asset_cols:
                numeric_cols = [
                    col
                    for col in cross_asset_cols
                    if pd.api.types.is_numeric_dtype(df[col])
                ]
                if numeric_cols:
                    var_rank = df[numeric_cols].var().sort_values(ascending=False)
                    selected_cols = var_rank.head(max_signals).index.tolist()
                    for col in selected_cols:
                        signal = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
                        if signal.notna().sum() == 0:
                            continue
                        mean = signal.mean()
                        std = signal.std()
                        # Check if feature is already normalized (std ≈ 1.0)
                        if 0.9 <= std <= 1.1:  # Already normalized
                            normalized = signal  # Use as-is to preserve natural variance
                        else:
                            normalized = (signal - mean) / (std + 1e-9)  # Normalize if needed
                        name = f"{col}_specialist"
                        if name in specialist_signals:
                            continue
                        signal_quality = self._validate_signal_quality(normalized, name)
                        if signal_quality["is_degenerate"]:
                            tprint_warning(
                                f"⚠️ {name} signal is degenerate: {signal_quality['issue']}"
                            )
                        specialist_signals[name] = normalized
                        cross_asset_added += 1
                if self.verbose and cross_asset_added > 0:
                    tprint_info(
                        f"   🌐 Added {cross_asset_added} cross-asset specialists from {len(cross_asset_cols)} features"
                    )
            
            if self.verbose:
                tprint_success(f"   ✅ Extracted {len(specialist_signals)} specialist signals:")
                for name, signal in specialist_signals.items():
                    tprint_info(f"      - {name}: {len(signal)} samples")
                
                # Overall signal quality report
                self._log_signal_quality_summary(specialist_signals)
            
            self._last_extracted_specialists = list(specialist_signals.keys())
            
            return specialist_signals
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist signal extraction failed: {e}")
            return {}

    def _extract_signals_by_asset(
        self,
        df: pd.DataFrame,
        configs: Dict[str, Dict[str, Any]],
        asset_col: str,
    ) -> Dict[str, pd.Series]:
        df_work = df.copy()
        df_work["_row_order"] = np.arange(len(df_work))
        aggregated: Dict[str, List[pd.Series]] = {}
        asset_configs = dict(configs)
        asset_context = dict(asset_configs.get("asset_context", {}))
        asset_context["group_by_asset"] = False
        asset_context["asset_id_col"] = asset_col
        asset_configs["asset_context"] = asset_context
        n_assets = df_work[asset_col].nunique(dropna=False)

        for _, asset_df in df_work.groupby(asset_col, sort=False):
            asset_order = asset_df["_row_order"].to_numpy()
            asset_df = asset_df.drop(columns=["_row_order"])
            asset_signals = self.extract_specialist_signals(asset_df, asset_configs)
            if not asset_signals:
                continue
            for name, series in asset_signals.items():
                series = series.copy()
                series.index = asset_order
                aggregated.setdefault(name, []).append(series)

        if not aggregated:
            return {}

        full_index = pd.Index(range(len(df_work)), name="_row_order")
        combined_signals: Dict[str, pd.Series] = {}
        for name, parts in aggregated.items():
            combined = pd.concat(parts).sort_index()
            combined = combined.reindex(full_index, fill_value=np.nan)
            combined.index = df.index
            combined_signals[name] = combined

        if self.verbose:
            tprint_info(
                f"   🧩 Asset-grouped extraction produced {len(combined_signals)} signals across {n_assets} assets"
            )

        self._last_extracted_specialists = list(combined_signals.keys())
        return combined_signals

    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input dataframe for common issues."""
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        # Use a local copy to ensure we don't destructively modify the global reference incorrectly
        # and so we can return a clean version.
        df_clean = df.copy()

        # Deduplicate timestamps only when the same asset repeats to avoid cross-asset data loss
        if df_clean.index.has_duplicates:
            has_multi_index = isinstance(df_clean.index, pd.MultiIndex)
            index_names = list(df_clean.index.names) if has_multi_index else [df_clean.index.name]

            # Ensure index levels have names so reset_index columns are well-defined
            if has_multi_index:
                updated_names = []
                for idx, name in enumerate(index_names):
                    if not name:
                        name = f"level_{idx}"
                        df_clean.index = df_clean.index.set_names(name, level=idx)
                    updated_names.append(name)
                index_names = updated_names
            else:
                if not df_clean.index.name:
                    df_clean.index = df_clean.index.rename("timestamp")
                    index_names = ["timestamp"]

            timestamp_name = index_names[0]

            asset_col = None
            asset_from_index = False
            for candidate in ("asset_id", "asset", "ticker", "symbol"):
                if candidate in df_clean.columns:
                    asset_col = candidate
                    break

            if asset_col is None and has_multi_index:
                for candidate in ("asset_id", "asset", "ticker", "symbol"):
                    if candidate in index_names:
                        asset_col = candidate
                        asset_from_index = True
                        break

            if asset_col is not None:
                # Multi-Asset Mode: Deduplicate on (timestamp, asset_id)
                df_reset = df_clean.reset_index()
                subset = [timestamp_name, asset_col]
                dup_mask = df_reset.duplicated(subset=subset, keep="last")
                dup_count = int(dup_mask.sum())

                if dup_count > 0:
                    sample_pairs = df_reset.loc[dup_mask, subset].head(5).values.tolist()
                    tprint_warning(
                        f"⚠️ Input dataframe contains {dup_count} duplicate ({timestamp_name}, {asset_col}) pairs; "
                        f"keeping latest occurrences. Examples: {sample_pairs}"
                    )
                    df_reset.drop_duplicates(subset=subset, keep="last", inplace=True)

                # Restore original index structure
                if has_multi_index:
                    df_clean = df_reset.set_index(index_names)
                else:
                    df_clean = df_reset.set_index(timestamp_name)

                if not asset_from_index and asset_col in df_clean.columns and df_clean.columns.duplicated().any():
                    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

            else:
                # Single-Asset Mode: Deduplicate on index (timestamp)
                dup_mask = df_clean.index.duplicated(keep="last")
                dup_count = int(dup_mask.sum())

                if dup_count > 0:
                    sample_labels = df_clean.index[dup_mask][:5]
                    tprint_warning(
                        f"⚠️ Input dataframe contains {dup_count} duplicate timestamps; keeping latest occurrences. "
                        f"Examples: {list(sample_labels)}"
                    )
                    df_clean = df_clean.loc[~dup_mask].copy()

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for non-finite values
        for col in required_cols:
            non_finite_count = df_clean[col].isna().sum() + np.isinf(df_clean[col]).sum()
            if non_finite_count > 0:
                tprint_warning(f"⚠️ {col} contains {non_finite_count} non-finite values")
        
        # Check data quality
        if len(df_clean) < 100:
            tprint_warning(f"⚠️ Small dataset: {len(df_clean)} rows may cause unreliable signals")
        
        # Check price consistency (on first 500 rows to save time)
        check_head = df_clean.head(500)
        for i in range(len(check_head)):
            row = check_head.iloc[i]
            if not (row['low'] <= row['open'] <= row['high'] and
                    row['low'] <= row['close'] <= row['high']):
                tprint_warning(f"⚠️ Price inconsistency detected at index {i} (checked first 500)")
                break
        
        return df_clean

    def _validate_signal_quality(self, signal: pd.Series, name: str) -> Dict[str, Any]:
        """Validate signal quality and detect degenerate cases."""
        quality = {
            'mean': signal.mean(),
            'std': signal.std(),
            'nan_pct': signal.isna().sum() / len(signal) * 100,
            'is_degenerate': False,
            'issue': None
        }
        
        # Check for zero variance (constant signal)
        if quality['std'] < 1e-10:
            quality['is_degenerate'] = True
            quality['issue'] = 'Zero variance (constant signal)'
        
        # Check for excessive NaN values
        elif quality['nan_pct'] > 50:
            quality['is_degenerate'] = True
            quality['issue'] = f'High NaN percentage: {quality["nan_pct"]:.1f}%'
        
        # Check for extreme values
        elif np.abs(signal).max() > 1e6:
            quality['is_degenerate'] = True
            quality['issue'] = 'Extreme values detected'
        
        # Check for very small signal magnitude
        elif np.abs(signal).max() < 1e-8:
            quality['is_degenerate'] = True
            quality['issue'] = 'Signal magnitude too small'
        
        return quality

    def _log_signal_quality_summary(self, specialist_signals: Dict[str, pd.Series]):
        """Log comprehensive signal quality summary."""
        if not specialist_signals:
            tprint_error("❌ No specialist signals to validate")
            return
        
        tprint_info("📊 Signal Quality Summary:")
        
        degenerate_count = 0
        for name, signal in specialist_signals.items():
            quality = self._validate_signal_quality(signal, name)
            status = "✅ OK" if not quality['is_degenerate'] else "❌ DEGENERATE"
            tprint_info(f"   {name}: {status} (std={quality['std']:.2e}, nan%={quality['nan_pct']:.1f}%)")
            if quality['is_degenerate']:
                degenerate_count += 1
                tprint_warning(f"      Issue: {quality['issue']}")
        
        if degenerate_count > 0:
            tprint_error(f"❌ {degenerate_count}/{len(specialist_signals)} specialist signals are degenerate")
            tprint_error("   This will cause zero resonance in spectral analysis")
        else:
            tprint_success("✅ All specialist signals have acceptable quality")

    def generate_specialist_event_dataset(
        self,
        df: pd.DataFrame,
        specialist_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        tbm_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate fully-labeled specialist events with TBM alignment, AVF filtering,
        surprise scores, and reliability diagnostics.
        """
        if self.verbose:
            tprint_info("🧠 Generating specialist event dataset with TBM + AVF safeguards")

        specialist_signals = self.extract_specialist_signals(df, specialist_configs)
        if not specialist_signals:
            return {}

        tbm_labels = self._label_market_events(df, tbm_overrides)
        market_context = self._prepare_market_context(df)

        specialist_payload: Dict[str, Dict[str, Any]] = {}
        metrics_summary: List[Dict[str, Any]] = []

        for name, signal in specialist_signals.items():
            (
                filtered_signal,
                surprise,
                activation_mask,
                vol_floor,
                avf_metadata,
            ) = self._apply_adaptive_volatility_filter(name, signal, market_context)

            # Pre-allocate arrays for better performance
            n_samples = len(signal)
            direction = np.sign(filtered_signal.values)
            direction[direction == 0] = np.nan
            
            tbm_label_values = tbm_labels["label"].values
            potential_profit_values = tbm_labels["potential_profit_pct"].values
            
            # Compute zone score (vectorized)
            zone_score = self._compute_zone_score(
                surprise, tbm_labels["label"]
            )
            
            # Compute meta_label (vectorized)
            activation_bool = activation_mask.astype(bool).values
            tbm_nonzero = tbm_label_values != 0
            meta_label = np.where(
                activation_bool & tbm_nonzero,
                (direction == tbm_label_values).astype(float),
                np.nan,
            )
            
            # Build DataFrame from dict of arrays (faster than incremental column assignment)
            event_frame = pd.DataFrame({
                "raw_signal": signal.values,
                "filtered_signal": filtered_signal.values,
                "surprise": surprise.values,
                "activation": activation_bool,
                "vol_floor": vol_floor.values,
                "direction": direction,
                "tbm_label": tbm_label_values,
                "potential_profit_pct": potential_profit_values,
                "zone_score": zone_score.values,
                "meta_label": meta_label,
            }, index=signal.index)

            metrics = self._compute_specialist_metrics(name, event_frame, tbm_labels)
            self._reliability_registry[name] = metrics
            metrics_summary.append({"specialist": name, **metrics})

            specialist_payload[name] = {
                "events": event_frame[event_frame["activation"]].copy(),
                "full_frame": event_frame,
                "metrics": metrics,
                "avf_metadata": avf_metadata,
            }

            if self.verbose:
                tprint_info(
                    f"   ↳ {name}: events={metrics.get('event_count', 0)} "
                    f"precision={metrics.get('precision', np.nan):.2f} "
                    f"recall={metrics.get('recall', np.nan):.2f}"
                )

        summary_df = pd.DataFrame(metrics_summary) if metrics_summary else pd.DataFrame()
        if self.verbose and not summary_df.empty:
            tprint_success("   ✅ Specialist event dataset ready (TBM-aligned)")

        self._cached_diversity_report = self._compute_diversity_diagnostics(
            specialist_payload,
            metrics_summary
        )

        return {
            "specialists": specialist_payload,
            "tbm_labels": tbm_labels,
            "summary": summary_df.sort_values("precision", ascending=False)
            if not summary_df.empty
            else summary_df,
            "diversity_diagnostics": self._cached_diversity_report
        }

    def get_reliability_report(self) -> Dict[str, Dict[str, Any]]:
        """Return latest reliability metrics for each specialist."""
        return self._reliability_registry
    
    def get_last_extracted_specialists(self) -> List[str]:
        """Return names of specialists successfully extracted in the last run."""
        return list(self._last_extracted_specialists)
    
    def get_diversity_report(self) -> Dict[str, Any]:
        """Return latest specialist diversity diagnostics."""
        return self._cached_diversity_report

    def _prepare_market_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        if "close" not in df.columns:
            raise ValueError("Market data must include 'close' for volatility context")

        returns = df["close"].pct_change().fillna(0)
        volatility = returns.rolling(self.avf_config.window).std()
        rolling_mad = returns.rolling(self.avf_config.window).apply(_rolling_mad, raw=True)
        vol_rank = volatility.rank(pct=True)
        # Fix for deprecated .mad()
        global_mad = float((returns - returns.mean()).abs().mean()) if not returns.empty else 0.0

        return {
            "returns": returns,
            "volatility": volatility,
            "rolling_mad": rolling_mad,
            "vol_rank": vol_rank,
            "global_mad": global_mad,
        }

    def _apply_adaptive_volatility_filter(
        self,
        specialist_name: str,
        signal: pd.Series,
        market_context: Dict[str, pd.Series],
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, float]]:
        floor = (
            market_context["rolling_mad"] * self.avf_config.floor_multiplier
        ).reindex(signal.index)
        fallback_floor = max(
            market_context["global_mad"] * self.avf_config.floor_multiplier,
            self.avf_config.hard_floor,
        )
        floor = floor.fillna(fallback_floor).clip(lower=self.avf_config.hard_floor)

        base_mask = signal.abs() >= floor
        surprise = signal / (floor + self.avf_config.eps)
        surprise = surprise.clip(-self.avf_config.max_surprise, self.avf_config.max_surprise)

        threshold = self._calibrate_activation_threshold(
            surprise.abs(), base_mask, self.event_config
        )
        activation_mask = base_mask & (surprise.abs() >= threshold)
        filtered_signal = signal.where(activation_mask, 0.0)

        avf_metadata = {
            "floor_median": float(np.nanmedian(floor)),
            "base_coverage": float(base_mask.mean() if len(base_mask) else 0.0),
            "activation_coverage": float(activation_mask.mean() if len(activation_mask) else 0.0),
            "activation_threshold": float(threshold),
        }

        if self.verbose:
            tprint_info(
                f"   • {specialist_name} AVF: coverage={avf_metadata['activation_coverage']:.2%}, "
                f"threshold={avf_metadata['activation_threshold']:.2f}"
            )

        return filtered_signal, surprise, activation_mask, floor, avf_metadata

    def _calibrate_activation_threshold(
        self,
        abs_surprise: pd.Series,
        base_mask: pd.Series,
        event_config: SpecialistEventConfig,
    ) -> float:
        if abs_surprise.empty or base_mask.sum() == 0:
            return event_config.base_activation_zscore

        candidate = event_config.base_activation_zscore
        coverage = (base_mask & (abs_surprise >= candidate)).mean()

        if np.isnan(coverage):
            coverage = 0.0

        abs_surprise_base = abs_surprise.where(base_mask)
        if coverage < event_config.min_coverage:
            quantile_target = max(0.0, 1 - event_config.min_coverage)
            candidate = float(abs_surprise_base.quantile(quantile_target))
        elif coverage > event_config.max_coverage:
            quantile_target = max(0.0, 1 - event_config.max_coverage)
            candidate = float(abs_surprise_base.quantile(quantile_target))

        if np.isnan(candidate) or candidate <= 0:
            candidate = event_config.base_activation_zscore

        return candidate

    def _label_market_events(
        self,
        df: pd.DataFrame,
        tbm_overrides: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

        tbm_cfg = self.tbm_config.merge(tbm_overrides or {})
        tbm_engine = (
            self._tbm_engine
            if tbm_overrides is None
            else OptimizedTripleBarrierLabeling(**tbm_cfg.to_kwargs())
        )
        tbm_result = tbm_engine.apply_triple_barrier_labeling_vectorized(df[required].copy())
        labeled = pd.DataFrame(
            {"label": 0, "potential_profit_pct": 0.0},
            index=df.index,
        )
        if isinstance(tbm_result, pd.DataFrame) and not tbm_result.empty:
            labeled.loc[tbm_result.index, "label"] = tbm_result["label"]
            labeled.loc[tbm_result.index, "potential_profit_pct"] = tbm_result[
                "potential_profit_pct"
            ]

        return labeled

    def _compute_zone_score(self, surprise: pd.Series, tbm_label: pd.Series) -> pd.Series:
        alignment = surprise * tbm_label
        score = 1.0 / (1.0 + np.exp(-(alignment * self.event_config.surprise_scaler)))
        return score.fillna(0.5)

    def _compute_specialist_metrics(
        self,
        specialist_name: str,
        event_frame: pd.DataFrame,
        tbm_labels: pd.DataFrame,
    ) -> Dict[str, Any]:
        activation_mask = event_frame["activation"].astype(bool)
        tbm_event_mask = tbm_labels["label"] != 0
        active = activation_mask & tbm_event_mask

        metrics: Dict[str, Any] = {
            "event_count": int(activation_mask.sum()),
            "coverage": float(activation_mask.mean() if len(activation_mask) else 0.0),
        }

        if active.sum() < self.event_config.min_events:
            metrics.update(
                {
                    "precision": np.nan,
                    "recall": np.nan,
                    "responsiveness": np.nan,
                    "marginal_value": np.nan,
                    "consensus_correlation": np.nan,
                    "avg_zone_score": np.nan,
                    "avg_surprise": np.nan,
                }
            )
            return metrics

        directions = event_frame.loc[active, "direction"]
        realized = tbm_labels.loc[active, "label"]
        profits = tbm_labels.loc[active, "potential_profit_pct"]

        correct = (directions == realized)
        precision = float(correct.mean())
        recall = float(active.sum() / max(tbm_event_mask.sum(), 1))
        responsiveness = float(
            event_frame.loc[active, "surprise"].corr(profits) or 0.0
        )
        marginal_value = float(
            profits.mean() - tbm_labels.loc[tbm_event_mask, "potential_profit_pct"].mean()
        )
        consensus_correlation = float(directions.corr(realized) or 0.0)
        avg_zone_score = float(event_frame.loc[active, "zone_score"].mean())
        avg_surprise = float(event_frame.loc[active, "surprise"].abs().mean())

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "responsiveness": responsiveness,
                "marginal_value": marginal_value,
                "consensus_correlation": consensus_correlation,
                "avg_zone_score": avg_zone_score,
                "avg_surprise": avg_surprise,
            }
        )
        metrics["composite_reliability"] = self._score_specialist_reliability(metrics)

        return metrics

    def _score_specialist_reliability(self, metrics: Dict[str, Any]) -> float:
        """Blend precision/recall/responsiveness into a composite reliability score."""
        precision = float(np.clip(metrics.get("precision", 0.0), 0.0, 1.0) or 0.0)
        recall = float(np.clip(metrics.get("recall", 0.0), 0.0, 1.0) or 0.0)
        responsiveness = float(np.clip(metrics.get("responsiveness", 0.0), 0.0, 1.0) or 0.0)
        consensus_corr = float(np.clip(metrics.get("consensus_correlation", 0.0), -1.0, 1.0) or 0.0)
        avg_zone_score = float(np.clip(metrics.get("avg_zone_score", 0.0), 0.0, 1.0) or 0.0)

        score = (
            0.35 * responsiveness +
            0.30 * precision +
            0.20 * recall +
            0.10 * max(0.0, consensus_corr) +
            0.05 * avg_zone_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def _compute_diversity_diagnostics(
        self,
        specialist_payload: Dict[str, Dict[str, Any]],
        metrics_summary: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute specialist diversity and redundancy diagnostics."""
        diagnostics: Dict[str, Any] = {}
        if not metrics_summary:
            return diagnostics

        metrics_df = pd.DataFrame(metrics_summary)
        if metrics_df.empty:
            return diagnostics

        metrics_df = metrics_df.set_index("specialist", drop=False)
        responsiveness = metrics_df["responsiveness"].astype(float)
        recall = metrics_df["recall"].astype(float)

        low_resp_mask = responsiveness.abs() < self.event_config.responsiveness_floor
        low_resp_specialists = metrics_df.loc[low_resp_mask.fillna(False), "specialist"].tolist()

        resp_recall_corr = None
        corr_sample = metrics_df[["responsiveness", "recall"]].dropna()
        if corr_sample.shape[0] >= 3:
            resp_recall_corr = corr_sample.corr(method="spearman").iloc[0, 1]

        filtered_series: List[pd.Series] = []
        for name, payload in specialist_payload.items():
            full_frame = payload.get("full_frame")
            if full_frame is None:
                continue
            if "filtered_signal" not in full_frame:
                continue
            series = full_frame["filtered_signal"].rename(name)
            filtered_series.append(series)

        correlated_pairs: List[Dict[str, Any]] = []
        if filtered_series:
            filtered_df = pd.concat(filtered_series, axis=1).dropna(how="all")
            if filtered_df.shape[1] > 1:
                corr_matrix = filtered_df.corr(method="spearman")
                for i, col_i in enumerate(corr_matrix.columns):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col_j = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if np.isnan(corr_val):
                            continue
                        if abs(corr_val) >= self.event_config.correlation_threshold:
                            correlated_pairs.append(
                                {
                                    "pair": (col_i, col_j),
                                    "correlation": float(corr_val),
                                }
                            )
                diagnostics["correlation_matrix"] = corr_matrix.round(3).to_dict()

        coverage_stats = {
            "median": float(metrics_df["coverage"].median()),
            "min": float(metrics_df["coverage"].min()),
            "max": float(metrics_df["coverage"].max()),
        }

        diagnostics.update(
            {
                "resp_recall_corr": float(resp_recall_corr) if resp_recall_corr is not None else None,
                "avf_recalibration_candidates": low_resp_specialists,
                "redundant_pairs": correlated_pairs,
                "coverage": coverage_stats,
                "metrics_table": metrics_df[
                    ["precision", "recall", "responsiveness", "coverage", "composite_reliability"]
                ].to_dict("index"),
            }
        )

        if self.verbose and low_resp_specialists:
            tprint_warning(
                f"   ⚠️ AVF recalibration suggested for: {', '.join(low_resp_specialists)}"
            )
        if self.verbose and correlated_pairs:
            formatted = ", ".join(
                f"{a}↔{b} ({corr:.2f})" for (a, b), corr in
                [((pair["pair"][0], pair["pair"][1]), pair["correlation"]) for pair in correlated_pairs[:4]]
            )
            tprint_warning(f"   ⚠️ High specialist correlation detected: {formatted}")

        return diagnostics

    def _extract_trend_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Trend Persistence signal.
        Logic: Rolling return (close - close[-N])
        """
        if self.verbose:
            tprint_info("🏹 Extracting trend specialist signal")
        try:
            if 'close' in df.columns:
                # Rolling return (e.g., 20 bars / 4-5 hours on 15m)
                window = 20
                rolling_return = df['close'].pct_change(window)
                
                # Normalize by rolling volatility to get Sharpe-like trend strength
                if rolling_stats and 'returns_std_20' in rolling_stats:
                    vol = rolling_stats['returns_std_20'] * np.sqrt(window)
                else:
                    vol = df['close'].pct_change().rolling(window).std() * np.sqrt(window)
                
                trend_signal = rolling_return / (vol + 1e-9)
                
                # Z-Score normalize
                if rolling_stats and 'returns_mean_50' in rolling_stats and 'returns_std_50' in rolling_stats:
                    trend_signal = (trend_signal - rolling_stats['returns_mean_50']) / (rolling_stats['returns_std_50'] + 1e-9)
                else:
                    trend_signal = (trend_signal - safe_rolling_mean(trend_signal, 50)) / (safe_rolling_std(trend_signal, 50) + 1e-9)
                
                # DIAGNOSTIC LOGGING (fixed: use rolling_return instead of undefined trend_pressure)
                if self.verbose and trend_signal is not None:
                    sig_std = trend_signal.std()
                    if sig_std < 1e-6:
                        raw_std = rolling_return.std()
                        raw_mean = rolling_return.mean()
                        if self.verbose:
                            tprint_warning(f"      ⚠️ Trend signal low variance: std={sig_std:.6f} (raw_std={raw_std:.6f}, raw_mean={raw_mean:.6f})")
                            tprint_warning(f"         Raw Range: [{rolling_return.min():.4f}, {rolling_return.max():.4f}]")
                
                return trend_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Trend signal extraction failed: {e}")
            return None

    def _extract_reversal_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Mean Reversion signal.
        Logic: Stochastic oscillator proxy - (close - rolling_min)/(rolling_max - rolling_min)
        Centered around 0.
        """
        if self.verbose:
            tprint_info("↩️ Extracting reversal specialist signal")
        try:
            required = ['close', 'high', 'low']
            if all(c in df.columns for c in required):
                window = 20
                roll_low = df['low'].rolling(window).min()
                roll_high = df['high'].rolling(window).max()
                
                # Stochastic %K
                stoch = (df['close'] - roll_low) / (roll_high - roll_low + 1e-9)
                
                # Center around 0.5 -> -0.5 to 0.5
                # But we want "Surprise" -> Reversal pressure.
                # If stoch is high (1.0), reversal pressure is Downative (Sell).
                # If stoch is low (0.0), reversal pressure is Positive (Buy).
                # So we invert: (0.5 - stoch)
                # High Stoch (1.0) -> -0.5 signal (Sell)
                # Low Stoch (0.0) -> +0.5 signal (Buy)
                
                reversal_pressure = 0.5 - stoch
                
                # Normalize
                reversal_signal = (reversal_pressure - reversal_pressure.rolling(50).mean()) / (reversal_pressure.rolling(50).std() + 1e-9)
                
                return reversal_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose: tprint_warning(f"      ⚠️ Reversal signal extraction failed: {e}")
            return None

    def _extract_volatility_breakout_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Volatility Breakout signal.
        Logic: Rolling High-Low / Baseline.
        """
        if self.verbose:
            tprint_info("💥 Extracting volatility breakout specialist signal")
        try:
            required = ['high', 'low', 'close']
            if all(c in df.columns for c in required):
                # Normalized Range (Parkinson proxy input)
                hl_range = (df['high'] - df['low']) / df['close']
                
                # Short and Long windows
                short_window = 10
                long_baseline = 50
                
                short_ma = hl_range.rolling(short_window).mean()
                baseline_ma = hl_range.rolling(long_baseline).mean()
                
                # Breakout Ratio
                breakout_ratio = short_ma / (baseline_ma + 1e-9)
                
                # We want "Surprise" when ratio is high.
                # If ratio > 1.0 -> Vol expansion.
                
                # KEY CHANGE: Directional Breakout
                # Range expansion is only a signal if we know WHICH WAY it broke out.
                direction = np.sign(df['close'].diff())
                vol_break_signal_raw = breakout_ratio * direction
                
                # Normalize
                vol_break_signal = (vol_break_signal_raw - vol_break_signal_raw.rolling(100).mean()) / (vol_break_signal_raw.rolling(100).std() + 1e-9)
                
                return vol_break_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose: tprint_warning(f"      ⚠️ Volatility Breakout signal extraction failed: {e}")
            return None
    
    def _extract_inventory_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """Extract inventory specialist signal (dealer inventory proxy) with temporal weighting."""
        if self.verbose:
            tprint_info("📈 Extracting inventory specialist signal")
        try:
            # Use volume-weighted price changes as inventory proxy
            if 'close' in df.columns and 'volume' in df.columns:
                price_change = df['close'].pct_change()
                # Normalize volume by its moving average to handle daily cycles
                vol_norm = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
                
                # Inventory pressure = price change * normalized volume
                inventory_pressure = price_change * vol_norm
                
                # Apply temporal weighting (EMA) to emphasize recent inventory accumulation
                # Use Numba-optimized EWMA
                inventory_signal_raw = ewma_numba(inventory_pressure.fillna(0).values, 10)
                inventory_signal = pd.Series(inventory_signal_raw, index=df.index)
                
                # Normalize by rolling volatility of the signal
                if rolling_stats and 'returns_mean_50' in rolling_stats and 'returns_std_50' in rolling_stats:
                    inventory_signal = (inventory_signal - rolling_stats['returns_mean_50']) / (rolling_stats['returns_std_50'] + 1e-9)
                else:
                    inventory_signal = (inventory_signal - safe_rolling_mean(inventory_signal, 50)) / \
                                    (safe_rolling_std(inventory_signal, 50) + 1e-9)
                
                if self.verbose and inventory_signal.std() < 1e-6:
                     tprint_warning(f"      ⚠️ Inventory signal low variance: raw_std={inventory_pressure.std():.6f}")

                return inventory_signal.fillna(0)
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Inventory signal extraction failed: {e}")
            return None
    
    def _extract_volume_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """Extract volume specialist signal with volatility normalization and AVF."""
        if self.verbose:
            tprint_info("📊 Extracting volume specialist signal")
        try:
            required = ['open', 'high', 'low', 'close', 'volume']
            if all(c in df.columns for c in required):
                # 1. Volume-Weighted Price Efficiency (Informed Flow)
                # Concept: High volume is only "signal" if it results in efficient price movement.
                # High volume + Low movement = Churn/Noise (Absorption) -> Filtered out
                # High volume + High movement = Informed Breakout -> Signal
                
                # Bar Efficiency (Signed: -1.0 to 1.0)
                # +1.0 = Marubozu Up (Pure Buy Pressure)
                # -1.0 = Marubozu Down (Pure Sell Pressure)
                # ~0.0 = Doji (Indecision/Churn)
                price_range = (df['high'] - df['low'])
                body = (df['close'] - df['open'])
                efficiency = body / (price_range + 1e-9)
                
                # Relative Volume (Log-space to dampen extreme outliers)
                # Use pre-computed volume MA if available
                if rolling_stats and 'volume_ma_20' in rolling_stats:
                    vol_ma = rolling_stats['volume_ma_20']
                else:
                    vol_ma = df['volume'].rolling(20).mean()
                volume_ratio = df['volume'] / (vol_ma + 1e-9)
                log_volume_ratio = np.log1p(volume_ratio)
                
                # Signal: Efficiency amplified by Volume
                # Efficient moves on high volume are the strongest causal events
                volume_signal = efficiency * log_volume_ratio
                
                # Adaptive Volatility Filter (AVF) integration
                # Use pre-computed returns and volatility if available
                if rolling_stats and 'returns' in rolling_stats and 'volatility_20' in rolling_stats:
                    returns = rolling_stats['returns']
                    volatility = rolling_stats['volatility_20']
                else:
                    returns = df['close'].pct_change()
                    volatility = returns.rolling(20).std()
                vol_rank = volatility.rolling(100).rank(pct=True)
                
                # Allow signal if:
                # 1. Volatility is healthy (> 10th percentile)
                # 2. OR Volume is massive (> 3x average) - distinct event
                avf_mask = (vol_rank > 0.1) | (volume_ratio > 3.0)
                
                volume_signal = volume_signal * avf_mask.astype(float)
                
                # Z-score normalization
                volume_signal = (volume_signal - volume_signal.rolling(50).mean()) / \
                               (volume_signal.rolling(50).std() + 1e-9)
                
                # Momentum confirmation (optional but helpful for persistence)
                # Is the signal separating from its recent average?
                sig_mom = volume_signal.diff()
                final_sig = (volume_signal + 0.5 * sig_mom).fillna(0)
                
                if self.verbose and final_sig.std() < 1e-6:
                     tprint_warning(f"      ⚠️ Volume signal low variance: raw_std={volume_signal.std():.6f}")
                     
                return final_sig
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Volume signal extraction failed: {e}")
            return None
    
    def _extract_volatility_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """Extract volatility specialist signal focusing on volatility changes."""
        if self.verbose:
            tprint_info("📈 Extracting volatility specialist signal")
        try:
            if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
                # High-Frequency Realized Volatility Measure (Intraday)
                # Parkinson / Garman-Klass proxy using High-Low
                hl_range = (df['high'] - df['low']) / df['close']
                
                # Realized Volatility (Returns based)
                # Use pre-computed returns and volatility if available
                if rolling_stats and 'returns' in rolling_stats and 'volatility_20' in rolling_stats:
                    returns = rolling_stats['returns']
                    realized_vol = rolling_stats['volatility_20']
                else:
                    returns = df['close'].pct_change()
                    realized_vol = returns.rolling(20).std()
                
                # Volatility Change (Delta Vol)
                # We care about expanding or contracting volatility
                vol_change = realized_vol.diff()
                
                # Range-based surprise
                range_ma = hl_range.rolling(20).mean()
                range_surprise = (hl_range - range_ma) / (range_ma + 1e-9)
                
                # Combined Signal: Volatility Expansion + Intraday Range Expansion
                # KEY CHANGE: Multiply by direction (sign of returns) to make it predictive
                # High Vol + Up = Bullish thrust
                # High Vol + Down = Bearish crash
                raw_mag = vol_change + (range_surprise * realized_vol)
                direction = np.sign(returns)
                
                volatility_signal = raw_mag * direction
                
                # Normalize
                volatility_signal = (volatility_signal - volatility_signal.rolling(50).mean()) / \
                                   (volatility_signal.rolling(50).std() + 1e-9)
                
                if self.verbose and volatility_signal.std() < 1e-6:
                     tprint_warning(f"      ⚠️ Volatility signal low variance: raw_change_std={vol_change.std():.6f}")

                return volatility_signal.fillna(0)
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Volatility signal extraction failed: {e}")
            return None
    
    def _extract_information_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract 'Information' signal using Price Action & Candle Ratios.
        Replaces dead VPIN metric with Microstructure/Price Action features.
        """
        if self.verbose:
            tprint_info("📊 Extracting information (price action) signal")
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                # 1. Candle Ratios
                price_range = df['high'] - df['low']
                body = np.abs(df['close'] - df['open'])
                
                # Trendiness: Body / Range (1.0 = Marubozu, 0.0 = Doji)
                trend_efficiency = body / (price_range + 1e-9)
                
                # Direction
                direction = np.sign(df['close'] - df['open'])
                
                # 2. Wick Rejection (Upper/Lower shadows)
                upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
                lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
                
                # Wick asymmetry (Positive = Selling Pressure/Rejection at top, Negative = Buying at bottom)
                # We interpret "Information" as "Informed Directional Flow"
                # Large Upper Wick = Rejection (Bearish Info)
                # Large Lower Wick = Support (Bullish Info)
                wick_balance = (lower_shadow - upper_shadow) / (price_range + 1e-9)
                
                # 3. Volume Verification
                # Does volume confirm the move?
                if rolling_stats and 'vol_20' in rolling_stats:
                    vol_rel = df['volume'] / (rolling_stats['vol_20'] + 1e-9)
                else:
                    vol_rel = df['volume'] / (safe_rolling_mean(df['volume'], 20, min_periods=1) + 1e-9)
                
                # Combined Price Action Signal
                # Strong body + Volume = Trend
                # Strong Wick + Volume = Rejection (Reversal)
                
                # Signal is directional: Positive = Bullish Info, Negative = Bearish Info
                pa_signal = (trend_efficiency * direction * vol_rel) + (wick_balance * vol_rel)
                
                # Clip raw signal to prevent extreme outliers before normalization
                pa_signal = pa_signal.clip(-10.0, 10.0)
                
                # Normalize using pa_signal's own rolling stats (FIXED: was incorrectly using close stats)
                pa_mean = safe_rolling_mean(pa_signal, 50, min_periods=1)
                pa_std = safe_rolling_std(pa_signal, 50, min_periods=1).replace(0, 1e-9)
                information_signal = (pa_signal - pa_mean) / (pa_std + 1e-9)
                
                # Final safeguard clip
                return information_signal.fillna(0).clip(-20.0, 20.0)
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Information signal extraction failed: {e}")
            return None
    
    def transform_to_spectral(
        self,
        specialist_signals: Dict[str, pd.Series],
        wavelet_engine
    ) -> Dict[str, np.ndarray]:
        """
        Transform specialists to spectral domain using wavelet decomposition.
        
        Args:
            specialist_signals: Raw specialist time series
            wavelet_engine: Wavelet decomposition engine
            
        Returns:
            Dictionary with spectral components
        """
        try:
            if self.verbose:
                tprint_info("🌊 Transforming specialists to spectral domain...")
            
            spectral_components = wavelet_engine.decompose_all_specialists(specialist_signals)
            
            if self.verbose:
                tprint_success(f"   ✅ Spectral transformation complete:")
                tprint_info(f"      - Specialists transformed: {len(specialist_signals)}")
                tprint_info(f"      - Spectral components: {len(spectral_components)}")
                tprint_info(f"      - Scales per specialist: 5")
            
            return spectral_components
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Spectral transformation failed: {e}")
            return {}
    
    def get_specialist_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all priority specialists."""
        if self.verbose:
            tprint_info("📋 Retrieving specialist metadata")
        return {
            name: self.specialist_descriptions.get(name, {})
            for name in self.priority_specialists
        }
    
    def validate_specialist_signals(
        self,
        specialist_signals: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate extracted specialist signals.
        
        Args:
            specialist_signals: Extracted specialist signals
            
        Returns:
            Validation results for each specialist
        """
        try:
            validation_results = {}
            
            for specialist_name, signal in specialist_signals.items():
                if len(signal) == 0:
                    validation_results[specialist_name] = {
                        'valid': False,
                        'error': 'Empty signal'
                    }
                    continue
                
                # Basic validation checks
                nan_count = signal.isna().sum()
                zero_count = (signal == 0).sum()
                signal_std = signal.std()
                signal_mean = signal.mean()
                
                validation_results[specialist_name] = {
                    'valid': True,
                    'length': len(signal),
                    'nan_count': nan_count,
                    'zero_count': zero_count,
                    'mean': signal_mean,
                    'std': signal_std,
                    'quality_score': self._calculate_quality_score(signal)
                }
            
            return validation_results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Signal validation failed: {e}")
            return {}
    
    def _calculate_quality_score(self, signal: pd.Series) -> float:
        """Calculate quality score for a specialist signal."""
        # Removed verbose logging to avoid spam when called in loops
        try:
            # Remove NaN and zeros
            clean_signal = signal.dropna()
            clean_signal = clean_signal[clean_signal != 0]
            
            if len(clean_signal) < 100:
                return 0.0
            
            # Quality metrics
            variance = clean_signal.var()
            autocorr = clean_signal.autocorr(lag=1)
            
            # Higher variance and moderate autocorrelation = better quality
            variance_score = min(variance / 1.0, 1.0)  # Normalize variance
            autocorr_score = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.0
            
            quality_score = 0.6 * variance_score + 0.4 * autocorr_score
            return quality_score
            
        except Exception:
            return 0.0

    def _extract_cusum_break_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract CUSUM structural break signal.
        Detects shifts in the mean of price changes.
        """
        if self.verbose:
            tprint_info("⚡ Extracting CUSUM break specialist signal")
        try:
            if 'close' in df.columns:
                # Use pre-computed returns if available
                if rolling_stats and 'returns' in rolling_stats:
                    returns = rolling_stats['returns'].dropna()
                else:
                    returns = df['close'].pct_change().dropna()
                
                # CUSUM Calculation
                # S[t] = max(0, S[t-1] + y[t] - k) for positive shift
                # We use a simplified two-sided cumulative deviation
                
                # Use pre-computed rolling stats if available
                if rolling_stats and 'close_mean_100' in rolling_stats and 'close_std_100' in rolling_stats:
                    mean_ret = returns.rolling(100).mean()
                    std_ret = returns.rolling(100).std()
                else:
                    mean_ret = returns.rolling(100).mean()
                    std_ret = returns.rolling(100).std()
                
                # Standardized deviation
                z = (returns - mean_ret) / (std_ret + 1e-9)
                
                # Cumulative Sum of Deviations
                # We want to detect *trends* in deviation -> persistent shift
                cusum = z.cumsum()
                
                # Detrend: Remove linear trend to find breaks in the trend
                # Simple way: CUSUM - Moving Average of CUSUM
                break_signal = (cusum - cusum.rolling(50).mean()) / (cusum.rolling(50).std() + 1e-9)
                
                # Re-index to match df
                return break_signal.reindex(df.index).fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ CUSUM signal extraction failed: {e}")
            return None

    def _extract_entropy_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Shannon Entropy signal using Numba-optimized calculation.
        Measures information content / unpredictability.
        """
        if self.verbose:
            tprint_info("🧩 Extracting entropy specialist signal")
        try:
            if 'close' in df.columns:
                # Use pre-computed returns if available
                if rolling_stats and 'returns' in rolling_stats:
                    returns = rolling_stats['returns']
                else:
                    returns = df['close'].pct_change()
                
                # Use Numba-optimized rolling entropy (much faster than pandas apply)
                window = 50
                returns_clean = returns.fillna(0).values
                entropy_sig = fast_rolling_entropy_numba(returns_clean, window=window, bins=10)
                entropy_series = pd.Series(entropy_sig, index=df.index)
                
                # Normalize using bottleneck if available
                # High entropy = High unpredictability
                entropy_signal = (entropy_series - safe_rolling_mean(entropy_series, 100)) / (safe_rolling_std(entropy_series, 100) + 1e-9)
                
                return entropy_signal.fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Entropy signal extraction failed: {e}")
            return None

    def _extract_tick_rule_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Tick Rule proxy signal (Aggressor Flow).
        Approximates net buy/sell pressure.
        """
        if self.verbose:
            tprint_info("🌊 Extracting tick rule specialist signal")
        try:
            required = ['close', 'open', 'volume']
            if all(c in df.columns for c in required):
                # Tick Rule Proxy:
                # Close > Open -> Buy (1)
                # Close < Open -> Sell (-1)
                # Close == Open -> 0 (or prev)
                
                direction = np.sign(df['close'] - df['open'])
                
                # Weight by volume
                signed_volume = direction * df['volume']
                
                # Accumulate (Cumulative Volume Delta - CVD proxy)
                cvd = signed_volume.cumsum()
                
                # Signal is the *divergence* or *acceleration* of CVD
                # We use local trend of CVD
                tick_signal = (cvd - cvd.rolling(20).mean()) / (cvd.rolling(20).std() + 1e-9)
                
                return tick_signal.fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Tick rule signal extraction failed: {e}")
            return None

    def _extract_fractal_efficiency_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Fractal Efficiency (Kaufman) signal with volatility regime filtering.
        Measures trend cleanliness/linearity.
        Logic: Sign(Return) * (Net_Move / Total_Path_Length) * Vol_Regime_Filter
        """
        if self.verbose:
            tprint_info("📏 Extracting fractal efficiency specialist signal")
        try:
            required = ['close']
            if all(c in df.columns for c in required):
                # Fractal Efficiency Ratio (Kaufman)
                # Optimized vectorized implementation using Pandas rolling
                window = 10
                
                # Numba-friendly logic simulation via optimized Pandas
                diffs = df['close'].diff()
                abs_diffs = diffs.abs()
                
                # Efficiency = |Change(N)| / Sum(|Change(1)|..|Change(N)|)
                net_change = df['close'].diff(window)
                path_length = abs_diffs.rolling(window).sum()
                
                # Avoid division by zero
                efficiency = net_change.abs() / (path_length + 1e-9)
                
                # Make Directional: Multiply by sign of the net change
                # Up Trend Efficient = +ve
                # Down Trend Efficient = -ve
                # Choppy/Noise = ~0
                direction = np.sign(net_change)
                directional_efficiency = efficiency * direction
                
                # ENHANCEMENT: Volatility regime filtering
                # High efficiency in low vol (consolidation) is not predictive
                # Weight by volatility rank to focus on meaningful trends
                if rolling_stats and 'vol_rank' in rolling_stats:
                    vol_filter = rolling_stats['vol_rank'].fillna(0.5)  # Neutral if missing
                    # Only keep signals in elevated volatility regimes (>30th percentile)
                    vol_filter = vol_filter.clip(0.3, 1.0)  # Floor at 0.3
                    directional_efficiency = directional_efficiency * vol_filter
                
                # Normalize (Z-Score)
                # We want to detect Anomalous Efficiency (Pure Trends)
                if rolling_stats and 'returns_mean_50' in rolling_stats and 'returns_std_50' in rolling_stats:
                    fractal_signal = (directional_efficiency - rolling_stats['returns_mean_50']) / (rolling_stats['returns_std_50'] + 1e-9)
                else:
                    fractal_signal = (directional_efficiency - safe_rolling_mean(directional_efficiency, 50)) / \
                                    (safe_rolling_std(directional_efficiency, 50) + 1e-9)
                
                return fractal_signal.fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Fractal efficiency extraction failed: {e}")
            return None

    def _extract_liquidity_shock_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Liquidity Shock signal (Amihud Proxy) with log-space normalization.
        Measures Price Impact per Unit of Volume.
        """
        if self.verbose:
            tprint_info("💧 Extracting liquidity shock specialist signal")
        try:
            required = ['close', 'volume']
            if all(c in df.columns for c in required):
                # Amihud Illiquidity: |Return| / (Price * Volume)
                # We want "Price Ease" -> Directional Impact
                
                if rolling_stats and 'returns' in rolling_stats:
                    returns = rolling_stats['returns']
                else:
                    returns = df['close'].pct_change()
                
                # Volume in dollars approx (Volume * Price) or just Volume if FX/Crypto
                # Using Dollar Volume is safer for comparing across price levels
                dollar_volume = df['volume'] * df['close']
                
                # Illiquidity: How much price moves per dollar traded
                # High = Illiquid (Fragile)
                # Low = Liquid (Robust)
                illiquidity = returns.abs() / (dollar_volume + 1e-9)
                
                # ENHANCEMENT: Log-space normalization for heavy-tailed distribution
                # Amihud illiquidity is highly skewed (power-law) - log transform prevents outliers
                illiquidity_log = np.log1p(illiquidity)
                
                # We want to detect SHOCKS in illiquidity that coincide with direction
                # i.e., Price moving easily (thin liquidity) in a direction
                
                # Directional Liquidity Shock = Sign(Return) * Log(Illiquidity)
                liq_shock_raw = np.sign(returns) * illiquidity_log
                
                # Normalize in log-space
                if rolling_stats and 'returns_mean_50' in rolling_stats and 'returns_std_50' in rolling_stats:
                    liq_shock_signal = (liq_shock_raw - rolling_stats['returns_mean_50']) / (rolling_stats['returns_std_50'] + 1e-9)
                else:
                    liq_shock_signal = (liq_shock_raw - safe_rolling_mean(liq_shock_raw, 50)) / \
                                      (safe_rolling_std(liq_shock_raw, 50) + 1e-9)
                                  
                return liq_shock_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Liquidity shock extraction failed: {e}")
            return None

    def _extract_gap_signal(self, df: pd.DataFrame, rolling_stats: Dict[str, pd.Series] = None) -> Optional[pd.Series]:
        """
        Extract Exogenous Gap signal.
        Detects actual gaps in data (time discontinuities or volume bar gaps) combined with price gaps.
        For volume bars, we detect when bars are unusually far apart in time.
        """
        if self.verbose:
            tprint_info("🕳️ Extracting gap specialist signal")
        try:
            required = ['open', 'close']
            if all(c in df.columns for c in required):
                # Detect actual time gaps in the data
                if isinstance(df.index, pd.DatetimeIndex):
                    # Calculate time delta between bars
                    time_deltas = df.index.to_series().diff()
                    
                    # Estimate typical bar frequency (median time delta)
                    typical_delta = time_deltas.median()
                    
                    # Detect gaps: time delta > 2x typical (allows for some variance)
                    # For volume bars, this detects periods of low activity
                    is_gap = time_deltas > (typical_delta * 2.0)
                else:
                    # If no datetime index, assume all bars are continuous
                    is_gap = pd.Series(False, index=df.index)
                
                # Price gap = Open - Prev Close (only meaningful at actual gaps)
                prev_close = df['close'].shift(1)
                price_gap = df['open'] - prev_close
                
                # Use pre-computed volatility if available
                if rolling_stats and 'returns' in rolling_stats and 'volatility_20' in rolling_stats:
                    volatility = rolling_stats['volatility_20']
                else:
                    returns = df['close'].pct_change()
                    volatility = returns.rolling(20).std()
                
                # Standardized price gap (only at actual gaps)
                # For continuous bars, this will be near zero
                gap_sigma = price_gap / (prev_close * volatility + 1e-9)
                
                # Amplify signal at actual gaps, dampen at continuous bars
                gap_multiplier = np.where(is_gap, 3.0, 0.3)  # 10x amplification at gaps
                gap_signal_raw = gap_sigma * gap_multiplier
                
                # Z-score normalize
                gap_signal = (gap_signal_raw - gap_signal_raw.rolling(50).mean()) / \
                            (gap_signal_raw.rolling(50).std() + 1e-9)
                            
                if self.verbose:
                    n_gaps = int(is_gap.sum())
                    if n_gaps > 0:
                        tprint_info(f"      ℹ️ Detected {n_gaps} time gaps in data (typical delta: {typical_delta})")
                    
                return gap_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Gap signal extraction failed: {e}")
            return None


# Convenience functions for quick usage
def quick_spectral_transformation(
    df: pd.DataFrame,
    wavelet_engine,
    priority_specialists: List[str] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Quick spectral transformation for market data."""
    if verbose:
        tprint_info("🚀 Quick spectral transformation")
    spectral_specialists = SpectralSpecialists(priority_specialists, verbose=verbose)
    
    # Extract signals
    specialist_signals = spectral_specialists.extract_specialist_signals(df)
    
    # Transform to spectral
    spectral_components = spectral_specialists.transform_to_spectral(
        specialist_signals, wavelet_engine
    )
    
    return spectral_components



