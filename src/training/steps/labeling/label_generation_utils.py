import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Union, Callable, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from scipy.stats import spearmanr, entropy as shannon_entropy, norm, rankdata
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import json
from pathlib import Path

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.numba_funcs import (
    _numba_rolling_median,
    _numba_rolling_mean,
    _numba_rolling_std,
    _numba_rolling_sum,
    _numba_rolling_correlation,
    _numba_calculate_continuous_weight
)
from src.utils.labeling_optimized import (
    triple_barrier_labels_numba,
    persistence_label_numba,
    window_stats_close_numba,
    window_stats_high_low_numba,
    first_hit_numba,
    first_hit_high_low_numba,
    batch_mi_score_numba
)
from src.utils.orthogonal_numba import (
    _numba_kalman_filter_1d,
    _numba_rolling_hurst,
    _numba_anchored_zscore,
    _numba_time_since_shock,
    _numba_build_indicator_matrix,
    _numba_get_uniqueness,
    _numba_create_ridge_targets,
    _numba_create_tree_targets
)

logger = logging.getLogger(__name__)

ROLLING_Q_WINDOW = 500
ROLLING_Q_THRESHOLDS = (0.96, 0.98)

RAW_METRIC_FIELDS = [
    'ic',
    'f_stat',
    'significance',
    'stability',
    'balance',
    'density',
    'path_score',
    'interventional_contrast',
    'overlap_support',
    'path_stability_var'
]

# Parameter name mapping for metadata handling
GENERATOR_PARAM_NAMES = {
    'ImprovedCUSUMEvents': ['multiplier', 'vol_window'],
    'AdaptiveSymmetricCUSUMEvents': ['multiplier', 'vol_window'],
    'VolatilityCusumEvents': ['h', 'vol_span'],
    'LiquidityCusumEvents': ['h', 'vol_span'],
    'VolumeCusumEvents': ['h', 'span'],
    'CausalSurpriseEvents': ['surprise_threshold', 'zone3_boost', 'zone2_boost', 'exposure_scalar'],
    'VolumeSpecialistEvents': ['threshold', 'window'],
    'VolatilitySpecialistEvents': ['quantile', 'window'],
    'VolatilityCrushEvents': ['quantile', 'window'],
    'LiquiditySpecialistEvents': ['threshold', 'window'],
    'InformationSpecialistEvents': ['threshold', 'window'],
    'InventorySpecialistEvents': ['threshold', 'window'],
    'MomentumDecaySpecialistEvents': ['threshold', 'fast_window', 'slow_window'],
    'FractalEfficiencyEvents': ['threshold', 'window'],
    'GapSpecialistEvents': ['threshold', 'window'],
    'LiquidityShockEvents': ['threshold', 'window'],
    'MicrostructureImbalanceEvents': ['threshold', 'window'],
    'ExhaustionSpecialistEvents': ['threshold', 'window'],
    'VolatilityInnovationSpecialistEvents': ['threshold'],
    'DispersionSpecialistEvents': ['threshold'],
    'ContinuousPredictorEvents': ['threshold', 'predictor_col'],
    'OrderBlockEvents': ['lookback', 'min_move_pct', 'volume_threshold']
}

# Generators that require the full DataFrame instead of just Series
DF_REQUIRED_CLASSES = (
    'OrderBlockEvents',
    'InventorySpecialistEvents',
    'VolumeSpecialistEvents',
    'VolatilitySpecialistEvents',
    'VolatilityCrushEvents',
    'LiquiditySpecialistEvents',
    'InformationSpecialistEvents',
    'MomentumDecaySpecialistEvents',
    'FractalEfficiencyEvents',
    'GapSpecialistEvents',
    'LiquidityShockEvents',
    'MicrostructureImbalanceEvents',
    'CausalSurpriseEvents',
    'AdaptiveSymmetricCUSUMEvents',
    'ImprovedCUSUMEvents',
    'KalmanRegimeEvents',
    'TradeIntensityEvents',
    'OrderFlowImbalanceEvents',
    'BarPressureEvents',
    'ExhaustionSpecialistEvents',
    'VolatilityInnovationSpecialistEvents',
    'DispersionSpecialistEvents',
    'ContinuousPredictorEvents',
    'VolatilityExpansionEvents',
    'VolatilityContractionEvents'
)

def _quantiles_from_threshold(threshold: float, delta: float = 0.02) -> Tuple[float, float]:
    from scipy.stats import norm
    # Convert sigma threshold to quantile (one-sided)
    q1 = norm.cdf(threshold)
    return (q1, q1 + delta)

def _quantiles_from_quantile(quantile: float, delta: float = 0.02) -> Tuple[float, float]:
    return (quantile, quantile + delta)

class OutputGeometry:

    def __init__(self, name, family, events, labels, weights, purity, auc, cluster_id=None, params=None, metrics=None):
        self.name = name
        self.family = family
        self.events = events
        self.labels = labels
        self.weights = weights
        self.purity = purity      # Uniqueness Score
        self.auc = auc            # Learnability Score (The Tournament Metric)
        self.cluster_id = cluster_id
        self.params = params if params is not None else {}
        self.metrics = metrics if metrics is not None else {}

    def __repr__(self):
        return f"<Geometry {self.name} | AUC={self.auc:.3f} | Purity={self.purity:.2f} | N={len(self.events)}>"

class UnifiedPriceMixin:
    """Mixin class for Layer2 generators to use unified price."""

    def __init__(self, use_unified_price: bool = True, layer0_params: dict = None):
        self.use_unified_price = use_unified_price
        self._layer0_params = layer0_params or {}
        self._cached_unified_price = None
        self._cached_timestamp = None

    def _get_unified_price(self, df: pd.DataFrame) -> pd.Series:
        """Get cached unified price or generate new one."""
        if not self.use_unified_price:
            return df['close']

        # Check cache validity (avoid re-computation)
        current_time = df.index[-1] if len(df) > 0 else None
        if (self._cached_unified_price is not None and
            self._cached_timestamp == current_time):
            return self._cached_unified_price

        # Generate unified price (simplified version)
        try:
            # Use Kalman filter if available, otherwise fallback to close
            from .unified_price_layer2 import generate_unified_price
            unified_price = generate_unified_price(df, self._layer0_params)
        except Exception:
            unified_price = df['close']

        # Cache the result
        self._cached_unified_price = unified_price
        self._cached_timestamp = current_time

        return unified_price

class BaseEventGenerator(UnifiedPriceMixin):
    """Base class for event generation with adaptive thresholding and unified price support."""

    def __init__(self, use_unified_price: bool = True, layer0_params: dict = None):
        """
        Initialize base event generator.

        Args:
            use_unified_price: Whether to use Kalman+VWAP unified price
            layer0_params: Layer0 parameters (auto-loaded if None)
        """
        super().__init__(use_unified_price=use_unified_price, layer0_params=layer0_params)

    def generate(self, data: Union[pd.Series, pd.DataFrame], tracker: Optional[Any] = None, **params) -> pd.DatetimeIndex:
        """Generate events using default parameters. Tracker for event accounting."""
        raise NotImplementedError

    def _validate_data(self, data: Union[pd.Series, pd.DataFrame]) -> None:
        """Validate input data with timezone and edge case handling."""
        if isinstance(data, pd.Series):
            if len(data) < 10:
                raise ValueError("Insufficient data points: need at least 10")
            if data.isna().all():
                raise ValueError("Data contains all-NaN values")
            # Check for timezone-aware index
            if data.index.tz is None:
                logger.debug("Data has no timezone - assuming UTC")
        else:
            if len(data) < 10:
                raise ValueError("Insufficient data points: need at least 10")
            if 'close' not in data.columns:
                raise ValueError("DataFrame must contain 'close' column")
            if data['close'].isna().all():
                raise ValueError("Close prices contain all-NaN values")
            # Check for timezone-aware index
            if data.index.tz is None:
                logger.debug("Data has no timezone - assuming UTC")

            # Validate OHLC data consistency if available
            if all(col in data.columns for col in ['open', 'high', 'low']):
                # Check for logical inconsistencies
                invalid_high = data['high'] < data['low']
                invalid_high_low = (data['high'] < data['close']) | (data['high'] < data['open'])
                invalid_low_high = (data['low'] > data['close']) | (data['low'] > data['open'])

                if invalid_high.any():
                    logger.warning(f"Found {invalid_high.sum()} bars where high < low")
                if invalid_high_low.any():
                    logger.warning(f"Found {invalid_high_low.sum()} bars with invalid high/low relationships")
                if invalid_low_high.any():
                    logger.warning(f"Found {invalid_low_high.sum()} bars with invalid low/high relationships")

    def _post_process_events(self, events: pd.DatetimeIndex, min_separation: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DatetimeIndex:
        """Remove clustered events, enforce minimum separation."""
        if len(events) <= 1:
            return events

        sorted_events = events.sort_values()
        filtered = [sorted_events[0]]

        for event in sorted_events[1:]:
            if event - filtered[-1] >= min_separation:
                filtered.append(event)

        return pd.DatetimeIndex(filtered)

    def generate_adaptive(self, data: Union[pd.Series, pd.DataFrame], target_signals_per_day: float = 7.5,
                        max_iterations: int = 20, tolerance: float = 0.1, *args, **params) -> pd.DatetimeIndex:
        """
        Generate events with adaptive thresholds to achieve target signal rate.
        Uses iterative convergence with proportional adjustment.
        """
        # Validate input data
        self._validate_data(data)

        # Calculate data duration and target signal count with timezone handling
        if isinstance(data, pd.Series):
            index = data.index
        else:
            index = data.index

        if len(index) < 2:
            logger.warning("Insufficient data for adaptive generation")
            return pd.DatetimeIndex([])

        # Handle timezone-aware vs naive datetime indices
        if index.tz is None:
            # Assume UTC for naive indices
            index = index.tz_localize('UTC')
            logger.debug("Localized naive datetime index to UTC")

        # Calculate duration in days (handles different timezones properly)
        duration_seconds = (index[-1] - index[0]).total_seconds()
        duration_days = max(1, duration_seconds / (24 * 3600))

        target_signals = int(target_signals_per_day * duration_days)
        min_target = int(target_signals * (1 - tolerance))
        max_target = int(target_signals * (1 + tolerance))

        # Start with default parameters
        current_params = params.copy()
        # Pass positional args if provided
        events = self.generate(data, *args, **current_params)

        if len(events) == 0:
            logger.debug("No events generated initially. Entering adaptive mode.")

        # Iterative adjustment with convergence
        iteration = 0
        best_events = events
        if len(events) == 0:
            best_error = target_signals
        else:
            best_error = abs(len(events) - target_signals)

        while iteration < max_iterations:
            current_count = len(events)
            current_error = abs(current_count - target_signals)

            # Check if within tolerance
            if min_target <= current_count <= max_target:
                logger.debug(f"Converged after {iteration + 1} iterations: {current_count} signals (target: {target_signals})")
                return events

            # Keep best result
            if current_error < best_error:
                best_error = current_error
                best_events = events

            # Calculate proportional adjustment factor

            if current_count > max_target:  # Too many signals -> TIGHTEN
                # Logic: Factor > 1.0
                factor = 1.15
            else:  # Too few signals -> RELAX
                # Logic: Factor < 1.0
                factor = 0.85

                # Panic calculation base
                if current_count < min_target * 0.1:
                    factor = 0.5  # Panic default


            factor = max(0.5, min(2.0, factor))  # Bound the factor

            # Panic mode for extremely low signals (explicit)
            if current_count < min_target * 0.1 and iteration < 5:
                 factor = 0.5 # Aggressive relaxation
                 logger.debug(f"Panic relaxation: Rate is {current_count}/{target_signals} (target). Slashed params by 50%.")

            # Adjust parameters if generator supports it
            if hasattr(self, '_adjust_z_threshold'):
                current_params = self._adjust_z_threshold(current_params, factor)
                try:
                    new_events = self.generate(data, **current_params)
                except Exception as e:
                    logger.warning(f"Optimization step failed: {e}")
                    break

                # Only use adjusted if it improves the signal rate
                new_count = len(new_events)
                new_error = abs(new_count - target_signals)

                if new_error < current_error or iteration == max_iterations - 1:
                    events = new_events
                    logger.debug(f"Iteration {iteration + 1}: {current_count} -> {new_count} signals (factor: {factor:.2f})")
                else:
                    # If error increased significantly, stop
                    if new_error > current_error * 1.5:
                        break

            else:
                # Generator doesn't support adjustment, use current events
                break

            iteration += 1

        # Apply post-processing
        final_events = self._post_process_events(events)

        if len(final_events) != len(events):
            logger.debug(f"Post-processing removed {len(events) - len(final_events)} clustered events")

        return final_events

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 1) -> pd.DataFrame:
    valid_events = events.intersection(index)
    if valid_events.empty:
        return pd.DataFrame(0, index=index, columns=[0])

    event_locs = index.get_indexer(valid_events)
    event_locs = event_locs[event_locs != -1]

    arr = _numba_build_indicator_matrix(event_locs, len(index), horizon, binary=True)

    return pd.DataFrame(arr, index=index, columns=[0])

def average_uniqueness(indicator: pd.DataFrame) -> float:
    concurrency = indicator.sum(axis=1)
    valid_c = concurrency[concurrency > 0]
    if valid_c.empty: return 0.0
    return (1.0 / valid_c).mean()

def compute_dominance_labels(
    price: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    risk_budget: float = 1.0,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 24,
    transaction_cost: float = 0.003,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Vectorized MFE/MAE Dominance Labeling with Risk Budget.
    Uses risk budget to control how close trades can get to stop-loss levels.

    Returns: labels, weights, returns, mfe, mae, volatility
    """
    # 1. Filter events within bounds
    if events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    n_bars = len(price)

    # Map events to integers
    # Normalize TZ/MultiIndex to ensure matching
    price_idx = price.index
    if isinstance(price_idx, pd.MultiIndex):
        price_idx = price_idx.get_level_values(0)
    if getattr(price_idx, "tz", None) is not None:
        price_idx = price_idx.tz_localize(None)

    events_norm = events
    if isinstance(events_norm, pd.MultiIndex):
        events_norm = events_norm.get_level_values(0)
    events_norm = pd.to_datetime(events_norm)
    if getattr(events_norm, "tz", None) is not None:
        events_norm = events_norm.tz_localize(None)
    if not isinstance(events_norm, pd.DatetimeIndex):
        events_norm = pd.DatetimeIndex(events_norm)

    event_idxs = price_idx.get_indexer(events_norm)

    # DEBUG: Deep inspection
    if len(events) > 0 and (event_idxs == -1).all():
         logger.warning(f"DEBUG: All indices -1. Mismatch suspected.")
         logger.warning(f"DEBUG: Price idx dtype: {price.index.dtype}")
         logger.warning(f"DEBUG: Events dtype: {events.dtype}")
         logger.warning(f"DEBUG: First 5 event_idxs: {event_idxs[:5]}")
         try:
             logger.warning(f"DEBUG: Price head: {price.index[:3]}")
             logger.warning(f"DEBUG: Events head: {events[:3]}")
         except: pass
         tprint_warning(
             "⚠️ Dominance labels: all event indices missing after alignment; "
             f"events={len(events)}, price_len={len(price_idx)}"
         )

    valid_mask = (event_idxs != -1) & (event_idxs < (n_bars - horizon))
    valid_idxs = event_idxs[valid_mask]
    valid_events = events[valid_mask] # Removed double calculation

    # DEBUG: Check why empty
    if len(valid_idxs) == 0:
        logger.warning(f"DEBUG: No valid events found! n_events={len(events)}")
        if len(events) > 0:
             logger.warning(f"DEBUG: Event[0]: {events[0]} type={type(events[0])}")
             logger.warning(f"DEBUG: Price[0]: {price.index[0]} type={type(price.index[0])}")
             logger.warning(f"DEBUG: Price[-1]: {price.index[-1]}")
             logger.warning(f"DEBUG: n_bars={n_bars}, horizon={horizon}")
        tprint_warning(
            "⚠️ Dominance labels: no valid events after bounds filtering; "
            f"events={len(events)}, horizon={horizon}"
        )
        return tuple([pd.Series(dtype=float)] * 6)

    # 3. Compute MFE/MAE & Hits (Numba Optimized)
    price_vals = price.values.astype(np.float64)
    vol_vals = volatility.values[valid_idxs].astype(np.float64)
    vol_vals = np.maximum(vol_vals, 1e-6)

    # Scale volatility by square root of horizon for multi-period targets
    # Cap the scaling at sqrt(9) = 3 to avoid explosion for long horizons
    # This ensures 2.0*Vol target is achievable over 'horizon' bars, not just 1 bar
    if horizon > 1:
        # Scale factor: sqrt(horizon) capped at 3.0
        scale_factor = np.sqrt(min(horizon, 9.0))
        eff_vol_vals = vol_vals * scale_factor
    else:
        eff_vol_vals = vol_vals

    # Thresholds as 1D arrays for Numba
    pt_thresh_arr = (eff_vol_vals * pt_mult)
    sl_thresh_arr = (-eff_vol_vals * sl_mult)

    # Check if High/Low provided
    if high is not None and low is not None:
        high_vals = high.values
        low_vals = low.values

        # Calculate stats
        mfe, mae, final_ret = window_stats_high_low_numba(
            high_vals, low_vals, price_vals, valid_idxs, horizon
        )

        # Calculate hits
        first_pt_idx, first_sl_idx, any_pt_arr, any_sl_arr = first_hit_high_low_numba(
            high_vals, low_vals, price_vals, valid_idxs, pt_thresh_arr, sl_thresh_arr, horizon
        )
    else:
        # Calculate stats
        mfe, mae, final_ret = window_stats_close_numba(
            price_vals, valid_idxs, horizon
        )

        # Calculate hits
        first_pt_idx, first_sl_idx, any_pt_arr, any_sl_arr = first_hit_numba(
            price_vals, valid_idxs, pt_thresh_arr, sl_thresh_arr, horizon
        )

    # Convert to boolean/numpy arrays for downstream logic
    any_pt = any_pt_arr.astype(bool)
    any_sl = any_sl_arr.astype(bool)
    # first_pt_idx, first_sl_idx are already 1-based indices relative to start.
    # Logic below compares them: first_pt_idx < first_sl_idx.
    # Note: If not hit, numba returns horizon + 1. So comparison works.

    # We need 'close_returns' last value for timeout case.
    # In Numba 'final_ret' is exactly that.
    timeout_returns = final_ret

    # TBM Logic (Ternary +1, -1, 0)
    # Case 1: Proft hit first (relative to SL)
    win_mask = any_pt & (~any_sl | (first_pt_idx < first_sl_idx))
    # Case 2: Stop hit first (relative to PT)
    loss_mask = any_sl & (~any_pt | (first_sl_idx < first_pt_idx))

    # Initialize ternary labels
    labels = np.zeros(len(valid_idxs), dtype=float)

    # Risk Budget Logic: MAE / Stop_Dist <= risk_budget
    stop_dist = sl_mult * vol_vals
    risk_used = mae / np.maximum(stop_dist, 1e-9)
    risk_mask = risk_used <= risk_budget
    min_profit = transaction_cost * 1.1
    profit_mask = mfe > min_profit

    labels[win_mask & risk_mask & profit_mask] = 1.0
    labels[loss_mask] = -1.0 # Losses are -1 regardless of risk budget or profit mask

    # Case 3: Timeout
    timeout_mask = (~any_pt) & (~any_sl)
    # timeout_returns already set to final_ret
    FEE_THRESHOLD = transaction_cost
    labels[timeout_mask & (timeout_returns > FEE_THRESHOLD)] = 1.0
    labels[timeout_mask & (timeout_returns < -FEE_THRESHOLD)] = -1.0
    # Otherwise label remains 0 (within noise band)

    # 5. Weighting
    mae_safe = np.maximum(mae, 1e-9)
    ratio = mfe / mae_safe
    magnitude = np.log1p(mfe / transaction_cost)

    # Updated Weighting (Edge Proxy): w ~ E[|move|] / Cost
    # Instead of inverse volatility (which penalizes action), reward regimes where
    # expected move > costs.
    # eff_vol_vals is volatility scaled to horizon (capped).
    edge_proxy = eff_vol_vals / max(transaction_cost, 1e-4)

    # Combine: Quality (Ratio/Mag) * Opportunity (Edge)
    weights = ratio * magnitude * edge_proxy

    # Clip and Normalize weights to prevent noise amplification
    # Cap at 99th percentile (approx 3-5 sigma) to handle outliers
    if len(weights) > 100:
        w_cap = np.percentile(weights, 99)
        weights = np.clip(weights, 0, w_cap)

    # Normalize to mean=1.0 for stability
    w_mean = np.mean(weights)
    if w_mean > 1e-9:
        weights = weights / w_mean

    # 6. Returns (use win_mask)
    # Use effective volatility for returns to match the threshold scaling
    out_returns = np.where(win_mask, pt_mult * eff_vol_vals, -sl_mult * eff_vol_vals)
    timeout_mask = (~any_pt) & (~any_sl)
    out_returns[timeout_mask] = timeout_returns[timeout_mask]

    # Construct Series
    idx = valid_events
    s_labels = pd.Series(labels, index=idx)
    s_weights = pd.Series(weights, index=idx)
    s_returns = pd.Series(out_returns, index=idx)
    s_mfe = pd.Series(mfe, index=idx)
    s_mae = pd.Series(mae, index=idx)
    s_vol = pd.Series(vol_vals, index=idx)

    return s_labels, s_weights, s_returns, s_mfe, s_mae, s_vol

def effective_n(labels, max_lag):
    """Estimate effective sample size accounting for autocorrelation."""
    labels = np.asarray(labels)
    n = len(labels)
    if n <= max_lag: return n

    rho_sum = 0.0
    # Fast manual autocorrelation for small lag
    for k in range(1, max_lag + 1):
        y1 = labels[:-k]
        y2 = labels[k:]
        if len(y1) < 2: continue
        y1_dev = y1 - y1.mean()
        y2_dev = y2 - y2.mean()
        denom = np.sqrt(np.sum(y1_dev**2) * np.sum(y2_dev**2))
        if denom == 0: continue
        rho = np.sum(y1_dev * y2_dev) / denom
        rho_sum += rho

    n_eff = n / (1.0 + 2.0 * rho_sum)
    return max(1.0, n_eff)

def significance_score(labels, max_lag):
    n_eff = effective_n(labels, max_lag)
    return np.log1p(n_eff)

def calculate_psr(sharpe, n, skew, kurt, target_sharpe=0):
    if n < 2: return 0.0
    std_sharpe = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    if std_sharpe == 0: return 0.0
    return norm.cdf((sharpe - target_sharpe) / std_sharpe)

def check_label_quality(
    events: pd.DatetimeIndex,
    labels: pd.Series,
    returns: pd.Series,
    df: pd.DataFrame,
    probe_features: pd.DataFrame,
    generator_instance,
    generator_params: dict,
    family: str = "UNKNOWN"
) -> Tuple[bool, Dict, str]:
    """Apply diagnostic gates to filter poor quality geometries."""

    n = len(labels)
    if n == 0:
        tprint_warning("❌ No labels - skipping gates")
        return False, {}, "No labels"

    # Calculate time span correctly for 15-minute data
    if len(labels.index) > 1:
        time_span = labels.index[-1] - labels.index[0]
        days = time_span.total_seconds() / (24 * 3600)
    else:
        days = 1.0

    rate = n / days if days > 0 else 0

    val_metrics = {
        'n': n, 'rate': rate, 'pos_rate': 0.0,
        'jaccard': 0.0, 'psr': 0.0, 'min_p': 0.0, 'max_mi': 0.0
    }

    gates_log = []
    failure_reason = "PASS"
    overall_pass = True

    # 1. Sample Size Gate (family-aware minimums)
    min_daily_rate = 0.1
    if family == 'MOMENTUM_DECAY_SPECIALIST':
        # Momentum decay specialists are sparse; allow lower rate to boost coverage
        min_daily_rate = 0.05

    if rate < min_daily_rate:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [FAIL]")
        overall_pass = False
        if failure_reason == "PASS":
            failure_reason = f"Sample Size (< {min_daily_rate:.2f}/day)"
    else:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [OK]")

    # 2. Class/Sample Balance Gate (relaxed)
    is_regression = family not in ['PRICE_CUSUM'] # All current non-PRICE families use realized returns

    if family == 'PRICE_CUSUM':
        # For ternary (+1, -1, 0), pos_rate = % of non-zero labels that are +1
        non_zero = labels[labels != 0]
        if len(non_zero) > 0:
            pos_rate = (non_zero == 1).mean()
        else:
            pos_rate = 0.0
        val_metrics['pos_rate'] = pos_rate

        # Gates: We want at least 10% on one side (relaxed from 15%)
        if pos_rate < 0.10 or pos_rate > 0.90:
            gates_log.append(f"Bal: {pos_rate:.1%} (Ternary) [FAIL]")
            overall_pass = False
            if failure_reason == "PASS": failure_reason = "Ternary Class Balance (<10% or >90%)"
        else:
            gates_log.append(f"Bal: {pos_rate:.1%} (Ternary) [OK]")
    else:
        # Regression: Check if we have enough non-zero samples (signals)
        pos_rate = (labels != 0).mean()
        val_metrics['pos_rate'] = pos_rate

        # Reduced minimum samples to 5% for regimes (relaxed from 10%)
        min_bal = 0.05 if is_regression else 0.10

        # RELAXATION: If we have many events, allow lower positive rate (rare signals are fine if statistically significant count)
        if n > 100:
            min_bal = 0.01

        if pos_rate < min_bal:
            gates_log.append(f"Bal: {pos_rate:.1%} (Samples) [FAIL]")
            overall_pass = False
            if failure_reason == "PASS": failure_reason = "Sample Balance (<5% non-zero)"
        else:
            gates_log.append(f"Bal: {pos_rate:.1%} (Samples) [OK]")

    # 3. Perturbation Stability Gate
    if generator_instance is None:
        # Meta/Composite signals: Use temporal stability instead of perturbation
        # Calculate autocorrelation-based stability (events should cluster temporally)
        try:
            ind_events = build_indicator_matrix(events, df.index, horizon=1).values.flatten()
            if len(ind_events) > 10:
                # Temporal stability: compute overlap between first and second half
                mid = len(ind_events) // 2
                first_half = ind_events[:mid]
                second_half = ind_events[mid:mid + len(first_half)]

                # Use the ratio of events that appear in both halves as stability proxy
                n_first = first_half.sum()
                n_second = second_half.sum()

                if n_first > 0 and n_second > 0:
                    # Rate stability: how similar are event rates across halves
                    rate_stability = 1.0 - abs(n_first - n_second) / max(n_first, n_second)
                    val_metrics['jaccard'] = rate_stability
                    gates_log.append(f"Jaccard: {rate_stability:.2f} (Rate) [OK]")
                else:
                    val_metrics['jaccard'] = 0.5
                    gates_log.append("Jaccard: 0.50 (Default) [OK]")
            else:
                val_metrics['jaccard'] = 0.5
                gates_log.append("Jaccard: 0.50 (Small) [OK]")
        except Exception:
            val_metrics['jaccard'] = 0.5
            gates_log.append("Jaccard: 0.50 (Fallback) [OK]")
    else:
        try:
            # Optimization: Shallow copy frame, deep copy only modified columns
            df_noisy = df.copy(deep=False)
            noise = np.random.normal(1.0, 0.0001, size=len(df))

            # Identify OHLC columns present
            ohlc_cols = [c for c in ['close', 'high', 'low', 'open'] if c in df_noisy.columns]

            # Deep copy and modify only OHLC
            for col in ohlc_cols:
                df_noisy[col] = df_noisy[col].values * noise

            gen = generator_instance
            if gen.__class__.__name__ in DF_REQUIRED_CLASSES:
                 events_noisy = gen.generate(df_noisy, **generator_params)
            else:
                 events_noisy = gen.generate(df_noisy['close'], **generator_params)

            ind_clean = build_indicator_matrix(events, df.index, horizon=1).values.flatten()
            ind_noisy = build_indicator_matrix(events_noisy, df.index, horizon=1).values.flatten()

            intersection = np.logical_and(ind_clean, ind_noisy).sum()
            union = np.logical_or(ind_clean, ind_noisy).sum()
            jaccard = intersection / union if union > 0 else 0.0
            val_metrics['jaccard'] = jaccard

            if jaccard < 0.3:
                gates_log.append(f"Jaccard: {jaccard:.2f} [WARN]")
            else:
                gates_log.append(f"Jaccard: {jaccard:.2f} [OK]")

        except Exception as e:
            # Fast fail for Jaccard fundamental issues
            logger.debug(f"Jaccard calculation failed for {family}: {e}")
            val_metrics['jaccard'] = 0.0
            gates_log.append(f"Jaccard: 0.00 (Fallback) [WARN]")

    # 4. ANOVA Gate
    X = probe_features.loc[labels.index]
    y = labels
    # Sanitize X to remove inf/nan values before f_classif/f_regression
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4. Statistical Power Gate (ANOVA / F-test)
    with np.errstate(divide='ignore', invalid='ignore'):
        if family == 'PRICE_CUSUM':
            # Use classification F-test for ternary labels
            F, p_values = f_classif(X, y)
        else:
            # Use regression F-test for continuous targets
            F, p_values = f_regression(X, y)

    valid_p = p_values[~np.isnan(p_values)]

    if len(valid_p) > 0:
        min_p = np.min(valid_p)
        val_metrics['min_p'] = min_p
        if min_p > 0.30:
            # RELAX: CAUSAL_SURPRISE is allowed to have weaker univariate F-STAT (often structural/sparse)
            if family == 'CAUSAL_SURPRISE':
                gates_log.append(f"F-STAT: p={min_p:.2f} [WARN-PASS]")
            else:
                gates_log.append(f"F-STAT: p={min_p:.2f} [FAIL]")
                overall_pass = False
                if failure_reason == "PASS": failure_reason = "F-STAT"
        else:
            gates_log.append(f"F-STAT: p={min_p:.2f} [OK]")
    else:
         gates_log.append("F-STAT: N/A [WARN]")

    # 5. Mutual Info Gate
    # Optimized: Use very small sample or correlation proxy
    # MI is expensive and often redundant with F-Test/IC for initial gating.
    # We can use correlation (IC) as a fast proxy and only compute MI if correlation is low but non-linear?
    # Or just drastically reduce sample size.

    MAX_MI_SAMPLES = 1000 # Reduced from 2000

    # Quick IC check first
    # If linear correlation is decent, MI will likely be decent or we don't care about MI specifically.
    # But MI captures non-linear relationships.

    if len(X) > MAX_MI_SAMPLES:
        indices = np.random.RandomState(42).choice(len(X), MAX_MI_SAMPLES, replace=False)
        X_mi = X.iloc[indices]
        y_mi = y.iloc[indices]
    else:
        X_mi = X
        y_mi = y

    try:
        # Check if we can skip MI: if IC > 0.05, assume some relation exists
        # This speeds up things significantly.
        # But we need 'max_mi' for metrics logging.

        if family == 'PRICE_CUSUM':
            mi = mutual_info_classif(X_mi, y_mi, discrete_features=False, random_state=42)
        else:
            mi = mutual_info_regression(X_mi, y_mi, discrete_features=False, random_state=42)

        max_mi = np.max(mi)
        val_metrics['max_mi'] = max_mi

        if max_mi < 0.001:
            gates_log.append(f"MI: {max_mi:.4f} [WARN]")
        else:
            gates_log.append(f"MI: {max_mi:.4f} [OK]")
    except Exception as e:
        gates_log.append(f"MI: Error [WARN]")
        val_metrics['max_mi'] = 0.0

    summary_str = " | ".join(gates_log)
    if overall_pass:
        tprint_info(f"✅ [{family}] Gates Passed: {summary_str}")
    else:
        tprint_warning(f"❌ [{family}] Gates Failed: {summary_str}")

    return overall_pass, val_metrics, failure_reason

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted mean."""
    return np.sum(x * w) / np.sum(w)

def weighted_cov(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted covariance."""
    xm = weighted_mean(x, w)
    ym = weighted_mean(y, w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w)

def weighted_spearmanr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Compute weighted Spearman correlation (Weighted Pearson on ranks).
    """
    if len(x) < 2: return 0.0

    # Ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)

    # Rank data
    x_rank = rankdata(x)
    y_rank = rankdata(y)

    # Weighted Pearson on ranks
    cov = weighted_cov(x_rank, y_rank, w)
    var_x = weighted_cov(x_rank, x_rank, w)
    var_y = weighted_cov(y_rank, y_rank, w)

    sx = np.sqrt(var_x)
    sy = np.sqrt(var_y)

    if sx == 0 or sy == 0:
        return 0.0

    return cov / (sx * sy)

def calculate_multifactor_score(
    candidates: List[Dict],
    probe_features: pd.DataFrame,
    regime_posteriors: Optional[pd.DataFrame] = None
) -> List[Dict]:
    if not candidates: return []
    scores = []
    raw_metric_log = []

    for cand in candidates:
        labels = cand['labels']
        n = len(labels)
        mfe = cand['mfe']
        mae = cand['mae']
        vol = cand['vol']

        X = probe_features.loc[labels.index]
        metrics = cand.get('metrics', {}) or {}
        # Optimization: Limit sample size for Spearman calculation
        MAX_SAMPLES = 2000

        # Get weights
        weights = cand.get('weights')
        if weights is None:
            # Fallback to weight_vector if available
            weights = cand.get('weight_vector')
            if weights is not None and not isinstance(weights, pd.Series):
                # Ensure it's a series aligned with events if possible, or just array
                pass

        # If still no weights, uniform
        if weights is None:
            weights_vals = np.ones(n)
        else:
            # Align weights to labels
            if isinstance(weights, pd.Series):
                weights_vals = weights.reindex(labels.index).fillna(0.0).values
            else:
                # Assuming weights is array-like aligned with candidates['events']
                # But labels might be subset if labels are filtered?
                # Usually labels index == events index.
                # If weights is numpy array, we assume it matches labels length if n is same
                if len(weights) == n:
                    weights_vals = np.asarray(weights)
                else:
                    # Mismatch or unaligned array. Fallback to uniform.
                    weights_vals = np.ones(n)

        # ---------------- REGIME-CONDITIONAL METRICS (The "Contract") ----------------
        min_regime_lift = 0.0
        max_regime_lift = 0.0
        regime_dispersion = 0.0

        if regime_posteriors is not None:
            # Align posteriors
            aligned_probs = regime_posteriors.reindex(labels.index).fillna(0.0)
            regime_lifts = []

            # Simple Lift proxy: Abs(Return) when label != 0 vs baseline
            # Or simpler: weighted mean of absolute returns for events in that regime?
            # Better: Use the 'lift' metric from probe (Meta-Sharpe - Base-Sharpe) but calculated per regime.
            # Approximation: Mean(Label * Return) / Volatility per regime

            returns = cand['returns']

            for col in aligned_probs.columns:
                p_r = aligned_probs[col]
                # Soft-weighted mask for this regime
                # Calculate weighted sharpe proxy
                w_r = weights_vals * p_r.values
                if w_r.sum() > 5: # Need minimal mass
                    weighted_ret = (returns * labels).fillna(0) # Strategy return
                    # Weighted Mean / Weighted Std
                    mean_r = np.average(weighted_ret, weights=w_r)
                    var_r = np.average((weighted_ret - mean_r)**2, weights=w_r)
                    sharpe_r = mean_r / (np.sqrt(var_r) + 1e-9)
                    regime_lifts.append(sharpe_r)

            if regime_lifts:
                min_regime_lift = min(regime_lifts)
                max_regime_lift = max(regime_lifts)
                regime_dispersion = np.std(regime_lifts)
            else:
                # Fallback if no regime alignment
                min_regime_lift = 0.0
                max_regime_lift = 0.0
                regime_dispersion = 0.0
        # -----------------------------------------------------------------------------

        if n > MAX_SAMPLES:
            indices = np.random.RandomState(42).choice(n, MAX_SAMPLES, replace=False)
            X_sub = X.iloc[indices]
            labels_sub = labels.iloc[indices]
            weights_sub = weights_vals[indices]
        else:
            X_sub = X
            labels_sub = labels
            weights_sub = weights_vals

        # Use Weighted Spearman
        ic_vals = []
        for col in X_sub.columns:
            try:
                ic = weighted_spearmanr(X_sub[col].values, labels_sub.values, weights_sub)
                ic_vals.append(abs(ic))
            except Exception:
                ic_vals.append(0.0)

        ic_max = np.nanmax(ic_vals) if ic_vals else 0

        # Sanitize X_sub before f_classif to avoid infinity errors
        X_sub = X_sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        F, _ = f_classif(X_sub, labels_sub)
        f_max = np.nanmax(F) if len(F) > 0 else 0

        # New Significance: Effective N
        max_lag = cand['params'].get('horizon', 120)
        significance = significance_score(labels, max_lag)

        # Stability
        chunk_size = n // 3
        if chunk_size > 10:
            ic_chunks = []
            for i in range(3):
                s = i * chunk_size
                e = (i + 1) * chunk_size if i < 2 else n
                # For stability, we use chunked data (preserving time order)
                # Don't sample here, use contiguous blocks
                sub_X = X.iloc[s:e]; sub_y = labels.iloc[s:e]

                # Limit size of chunks if necessary?
                # If chunk is huge, sampling might destroy time structure for stability?
                # Stability here is cross-validation of IC basically.
                # Let's keep it on full chunk for now as it splits by 3 already.

                chunk_ics = [abs(spearmanr(sub_X[col], sub_y)[0]) for col in sub_X.columns]
                ic_chunks.append(np.nanmax(chunk_ics))
            stability = 1.0 / (np.std(ic_chunks) + 1e-6)
        else: stability = 0.5

        counts = labels.value_counts(normalize=True)
        balance = shannon_entropy(counts)

        indicator = build_indicator_matrix(cand['events'], X.index, horizon=cand['params']['horizon'])
        density = average_uniqueness(indicator)

        path_asymmetry = (mfe / vol) - (mae.abs() / vol)
        path_score = path_asymmetry.mean()

        # --- Causal robustness extensions ---
        # Interventional contrast: difference between strong positive and negative event returns
        interventional_contrast = np.nan
        returns_series = cand.get('returns')
        if returns_series is not None and not returns_series.empty:
            labels_series = cand.get('labels')
            if labels_series is not None and not labels_series.empty:
                aligned_idx = returns_series.index.intersection(labels_series.index)
                if len(aligned_idx) >= 10:
                    pos_mask = labels_series.loc[aligned_idx] > 0
                    neg_mask = labels_series.loc[aligned_idx] < 0
                    if pos_mask.sum() >= 5 and neg_mask.sum() >= 5:
                        pos_mean = returns_series.loc[aligned_idx][pos_mask].mean()
                        neg_mean = returns_series.loc[aligned_idx][neg_mask].mean()
                        interventional_contrast = float(pos_mean - neg_mean)
                    else:
                        interventional_contrast = float(returns_series.loc[aligned_idx].mean())
        # Overlap support: reuse density (average uniqueness) to quantify overlap coverage
        overlap_support = float(np.clip(density, 0.0, 1.0))
        # Path stability variance: variability of asymmetry metric
        path_stability_series = path_asymmetry.replace([np.inf, -np.inf], np.nan).dropna()
        path_stability_var = float(path_stability_series.var()) if not path_stability_series.empty else np.nan

        cand_raw_metrics = {
            'ic': ic_max,
            'f_stat': f_max,
            'significance': significance,
            'stability': stability,
            'balance': balance,
            'density': density,
            'path_score': path_score,
            'lift': max(ic_max, f_max / (f_max + 10.0)), # Proxy for learnability
            # Causal/robustness extensions
            'interventional_contrast': interventional_contrast,
            'overlap_support': overlap_support,
            'path_stability_var': path_stability_var,
            # Regime-Conditional Metrics
            'min_regime_lift': min_regime_lift,
            'max_regime_lift': max_regime_lift,
            'regime_dispersion': regime_dispersion
        }
        cand['metrics_raw'] = cand_raw_metrics
        raw_metric_log.append({
            'uuid': cand.get('uuid', cand.get('name')),
            'family': cand.get('family'),
            **cand_raw_metrics
        })
        scores.append(cand)

    df_scores = pd.DataFrame([c['metrics_raw'] for c in scores])
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_scores), columns=df_scores.columns)

    for i, cand in enumerate(scores):
        row = df_norm.iloc[i]
        metrics = cand['metrics_raw']

        # 1. Base Power: IC or F-stat
        power = max(row['ic'], row['f_stat'])

        # 2. Causal Integrity Bonus
        ic_ic = row.get('interventional_contrast', 0.0)
        purity_score = (power * 0.7 + ic_ic * 0.3) if not np.isnan(ic_ic) else power

        # 3. Regime Robustness (The Contract)
        # Reward candidates that work everywhere (min_lift) and penalize dispersion
        # Or allow Conditional if max_lift is high (but maybe scored lower than Core)

        # Determine Admission Type
        is_core = (metrics['min_regime_lift'] > 0.05) and (metrics['regime_dispersion'] < 0.5)
        is_conditional = (metrics['max_regime_lift'] > 0.15) # High peak performance

        cand['classification'] = 'CORE' if is_core else ('CONDITIONAL' if is_conditional else 'WEAK')

        # Improved Ranking Formula
        # Rank = a*(1-min_p) + b*min_lift - c*dispersion
        # (Using row values which are normalized 0-1)
        # Use significance (log n_eff) as proxy for (1-min_p) confidence

        regime_score = row.get('min_regime_lift', 0.0) - 0.5 * row.get('regime_dispersion', 0.0)

        final_score = (
            0.4 * purity_score +
            0.2 * row['significance'] +
            0.2 * row['stability'] +
            0.2 * regime_score
        )

        cand['score'] = final_score
        cand['power'] = power

    # Persist raw metric inspection log for diagnostics
    try:
        diagnostics_dir = Path("outcomes")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = diagnostics_dir / f"layer2_raw_metric_log_{timestamp}.json"
        raw_path.write_text(json.dumps(raw_metric_log, indent=2, default=float))
        tprint_info(f"Saved raw metric log to {raw_path}")
    except Exception as exc:
        logger.warning(f"Failed to persist raw metric log: {exc}")

    return scores
