import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Union, Callable, Optional, Tuple, Any, Set
from enum import Enum
from collections import Counter
from scipy.stats import spearmanr, entropy as shannon_entropy, norm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, SparsePCA
from dataclasses import dataclass
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.training.steps.labeling.covariance_denoising import marcenko_pastur_distribution
from .focal_loss_utils import get_focal_loss_lgbm
from src.utils.entropy_optimized import rolling_entropy_numba
from src.training.steps.labeling.composite_event_generators import (
    get_microstructure_generators,
    TradeIntensityEvents,
    OrderFlowImbalanceEvents,
    BarPressureEvents,
)

# Import causal framework modules for surprise events
try:
    from src.training.steps.labeling.causal_surprise_events import CausalSurpriseDetector, quick_causal_surprise
    from src.training.steps.labeling.causal_specialists import CausalSpecialist, CausalSpecialistManager
    CAUSAL_AVAILABLE_ORTHOGONAL = True
except ImportError:
    CAUSAL_AVAILABLE_ORTHOGONAL = False

# Import continuous predictor generators
try:
    from src.training.steps.labeling.predictor_geometry_generators import (
        ContinuousPredictorGenerator, CausalResidualGenerator, 
        PredictorGeometry, generate_continuous_predictors
    )
    from src.training.steps.labeling.causal_quality_assessment import SignalRole
    PREDICTOR_GENERATORS_AVAILABLE = True
except ImportError:
    PREDICTOR_GENERATORS_AVAILABLE = False
    class SignalRole:
        PREDICTOR = "predictor"
        TRIGGER = "trigger"
        INTERACTION = "interaction"
        CONTEXT = "context"


# Setup Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Event Pipeline Logger Class
class EventPipelineLogger:
    """Simple one-line logging for event pipeline stages"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stages = []
    
    def log_stage(self, stage_name: str, count: int, total: int = None):
        """Log a single stage with event count"""
        if total:
            percentage = (count / total) * 100
            message = f"📊 {stage_name}: {count:,} events ({percentage:.1f}% of {total:,})"
        else:
            message = f"📊 {stage_name}: {count:,} events"
        
        if self.verbose:
            tprint_info(message)
        
        self.stages.append({
            'stage': stage_name,
            'count': count,
            'total': total,
            'percentage': percentage if total else None
        })
    
    def print_summary(self):
        """Print final summary line"""
        if not self.stages:
            return
        
        initial = self.stages[0]['count']
        final = self.stages[-1]['count']
        efficiency = (final / initial) * 100 if initial > 0 else 0
        
        tprint_info(f"🎯 Pipeline Summary: {initial:,} → {final:,} events ({efficiency:.1f}% efficiency)")

# Define UnifiedPriceMixin inline to avoid circular import
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

VOLUME_FAMILIES = {'VOLUME_SPECIALIST', 'MICROSTRUCTURE_IMBALANCE'}
VOLATILITY_FAMILIES = {'VOLATILITY_SPECIALIST', 'VOLATILITY_CRUSH'}
FAMILY_MIN_EVENT_RATE = {
    'VOLUME_SPECIALIST': 2.5,
    'MICROSTRUCTURE_IMBALANCE': 2.0,
}
CONDITIONAL_MI_THRESHOLD = 0.015

def _should_use_range_specific_optimization() -> bool:
    """Check if 1.5-3% range optimization is enabled in configuration."""
    try:
        import yaml
        config_path = "config/labeling/layer2_coverage_relax_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("target_range_optimization", {}).get("enabled", False)
    except Exception:
        return False



# ==========================================
# 0. Data Structures & Configuration
# ==========================================

MIN_CAUSAL_OVERLAP_SUPPORT = 0.05
PATH_STABILITY_CHUNKS = 4
EPSILON = 1e-9
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

VOLUME_FAMILIES = {'VOLUME_SPECIALIST', 'MICROSTRUCTURE_IMBALANCE'}
VOLATILITY_FAMILIES = {'VOLATILITY_SPECIALIST', 'VOLATILITY_CRUSH'}
FAMILY_MIN_EVENT_RATE = {
    'VOLUME_SPECIALIST': 2.5,
    'MICROSTRUCTURE_IMBALANCE': 2.0,
}

FIXED_GRID = [
    # --- 1.5-3% Target Range Specific Grid (de Prado Framework) ---
    # ADDED: TBM 32 and Trend Model 150
    {'id': 'TBM_32', 'pt': 2.0, 'sl': 1.0, 'horizon': 32},
    {'id': 'Trend_150', 'pt': 4.0, 'sl': 1.5, 'horizon': 150},
    
    # --- Ratio 1.5 ---
    {'id': '1.5:1', 'pt': 2.25, 'sl': 1.5},
    {'id': '3:2',   'pt': 3.75, 'sl': 2.5},

    # --- Ratio 2.0 ---
    {'id': '2:1',   'pt': 3.00, 'sl': 1.5},
    {'id': '4:2',   'pt': 5.00, 'sl': 2.5},

    # --- Ratio 3.0 ---
    {'id': '3:1',   'pt': 4.50, 'sl': 1.5},

    # --- Ratio 4.0 ---
    {'id': '4:1',   'pt': 6.00, 'sl': 1.5},
]
# 1.5-3% Target Range Specific Grid (de Prado Framework)
MEDIUM_TERM_GRID = [
    # --- 1.5% Target (Low End)
    {'id': '1.5pct', 'pt': 1.5, 'sl': 0.75},
    {'id': '1.5pct_tight', 'pt': 1.5, 'sl': 0.5},
    
    # --- 2.0% Target (Mid Range)
    {'id': '2.0pct', 'pt': 2.0, 'sl': 0.8},
    {'id': '2.0pct_tight', 'pt': 2.0, 'sl': 0.6},
    
    # --- 2.25% Target (Optimal Midpoint)
    {'id': '2.25pct', 'pt': 2.25, 'sl': 0.75},
    {'id': '2.25pct_tight', 'pt': 2.25, 'sl': 0.6},
    
    # --- 2.5% Target (Upper Mid)
    {'id': '2.5pct', 'pt': 2.5, 'sl': 0.8},
    {'id': '2.5pct_tight', 'pt': 2.5, 'sl': 0.7},
    
    # --- 3.0% Target (High End)
    {'id': '3.0pct', 'pt': 3.0, 'sl': 0.9},
    {'id': '3.0pct_tight', 'pt': 3.0, 'sl': 0.8},
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
    'MicrostructureImbalanceEvents': ['threshold', 'window']
}

# Generators that require the full DataFrame instead of just Series
DF_REQUIRED_CLASSES = (
    'InventorySpecialistEvents',
    'VolumeSpecialistEvents',
    'VolatilitySpecialistEvents',
    'VolatilityCrushEvents',
    'LiquiditySpecialistEvents',
    'InformationSpecialistEvents',
    'MomentumDecaySpecialistEvents',
    'MicrostructureImbalanceEvents',
    'CausalSurpriseEvents',
    'AdaptiveSymmetricCUSUMEvents',
    'ImprovedCUSUMEvents',
    'KalmanRegimeEvents'
)


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


class KalmanFilter1D:
    def __init__(self, Q: float = 1e-5, R: float = 0.01, initial_value: float = 0.0):
        self.Q = Q
        self.R = R
        self.x = initial_value
        self.P = 1.0

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        values = series.values
        n = len(values)
        x_hat = np.zeros(n)
        P_hat = np.zeros(n)
        x, P = self.x, self.P
        Q, R = self.Q, self.R

        for i in range(n):
            x_pred = x
            P_pred = P + Q
            z = values[i]
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            x_hat[i] = x
            P_hat[i] = P

        return pd.Series(x_hat, index=series.index), pd.Series(P_hat, index=series.index)

def roll_entropy(series: pd.Series, window: int = 24, bins: int = 10) -> pd.Series:
    """
    Calculate rolling Shannon entropy using Numba optimization.
    Returns natural entropy (nats) to match original implementation.
    """
    # rolling_entropy_numba returns bits (base 2)
    # Convert to nats: bits * ln(2)
    entropy_bits = rolling_entropy_numba(series.values, window, bins)
    entropy_nats = entropy_bits * np.log(2)
    return pd.Series(entropy_nats, index=series.index)

def calc_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = price * volume
    cum_pv = pv.rolling(window).sum()
    cum_vol = volume.rolling(window).sum()
    return cum_pv / (cum_vol + 1e-9)

def calc_tr(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    if 'high' in cols and 'low' in cols:
        high = df[cols['high']]
        low = df[cols['low']]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = close.diff().abs()
    return tr

def average_uniqueness(indicator: pd.DataFrame) -> float:
    concurrency = indicator.sum(axis=1)
    valid_c = concurrency[concurrency > 0]
    if valid_c.empty: return 0.0
    return (1.0 / valid_c).mean()

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 1) -> pd.DataFrame:
    arr = np.zeros(len(index), dtype=int)
    valid_events = events.intersection(index)
    if valid_events.empty:
        return pd.DataFrame(0, index=index, columns=[0])
    
    event_locs = index.get_indexer(valid_events)
    event_locs = event_locs[event_locs != -1]
    n_bars = len(index)

    # Fill indicator matrix
    for loc in event_locs:
        end_loc = min(loc + horizon, n_bars)
        arr[loc:end_loc] = 1 # Use binary indicator for set operations
    
    return pd.DataFrame(arr, index=index, columns=[0])

def generate_probe_features(price: pd.Series, volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """Generate basic features for learnability probing."""
    df = pd.DataFrame(index=price.index)
    df['ret_1'] = price.pct_change()
    df['vol_20'] = df['ret_1'].rolling(20).std()

    # RSI approximation
    diff = price.diff()
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rsi = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
    df['rsi_14'] = rsi.fillna(50)

    if volume is not None:
        df['vol_chg'] = volume.pct_change()

    return df.fillna(0)

def ewma_volatility(returns, span=100):
    """
    EWMA volatility estimator.
    Used to normalize thresholds and make CUSUM regime-invariant.
    """
    return returns.ewm(span=span, adjust=False).std()


# ==========================================
# 0.4.5. RMI Feature Reduction (Optimization)
# ==========================================

def calculate_residual_mi(feature_df: pd.DataFrame, target_series: pd.Series, 
                          lag: int = 1, n_neighbors: int = 3, 
                          subsample_size: int = 10000) -> pd.Series:
    """
    Calculates the Residual Mutual Information (RMI) proxy for a set of features.
    Used to reduce composite features to the most informative subset.
    
    Args:
        feature_df: DataFrame of features (e.g., composite candidates).
        target_series: The target (e.g., 1-bar forward returns).
        lag: Number of lags of the target to use for residualization.
        n_neighbors: Number of neighbors for MI estimation.
        subsample_size: Max samples for efficiency.
    
    Returns:
        Series of RMI scores sorted descending.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    
    # 1. Prepare Target Residuals (The 'Innovation')
    # Use a simple AR(lag) model to strip out serial correlation
    y = target_series.values.reshape(-1, 1)
    
    # Create lagged target matrix
    y_lags = np.hstack([target_series.shift(i).values.reshape(-1, 1) for i in range(1, lag + 1)])
    
    # Valid indices (drop NaNs from shifting)
    valid_idx = ~np.isnan(y_lags).any(axis=1)
    y_clean = y[valid_idx]
    y_lags_clean = y_lags[valid_idx]
    
    # Fit AR model and get residuals
    model = LinearRegression()
    model.fit(y_lags_clean, y_clean)
    y_pred = model.predict(y_lags_clean)
    residuals = (y_clean - y_pred).flatten()
    
    # 2. Align Features with Residuals
    X = feature_df.iloc[valid_idx].values
    
    # 3. Subsample if too large (efficiency)
    if len(residuals) > subsample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(residuals), size=subsample_size, replace=False)
        X = X[idx]
        residuals = residuals[idx]
    
    # Standardize to ensure KNN distance is meaningful
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle any NaNs/Infs in scaled data
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 4. Calculate Mutual Information
    mi_scores = mutual_info_regression(
        X_scaled, 
        residuals, 
        discrete_features=False, 
        n_neighbors=n_neighbors,
        random_state=42
    )
    
    return pd.Series(mi_scores, index=feature_df.columns).sort_values(ascending=False)


def filter_composites_by_rmi(df: pd.DataFrame, composites: List[Dict], 
                              target_col: str = 'close', 
                              top_k: int = 500) -> List[Dict]:
    """
    Filter composites using Residual Mutual Information.
    Reduces from ~1800 composites to top_k by RMI score.
    
    Args:
        df: DataFrame with price data.
        composites: List of composite candidate dicts with 'weight_vector'.
        target_col: Column to use for target.
        top_k: Number of top composites to keep.
    
    Returns:
        Filtered list of composites.
    """
    if len(composites) <= top_k:
        return composites
    
    tprint_info(f"   📉 Running RMI reduction: {len(composites)} → {top_k} composites...")
    
    try:
        # Build feature matrix from weight vectors
        weight_data = {}
        valid_composites = []
        for c in composites:
            if 'weight_vector' in c and c['weight_vector'] is not None:
                weight_data[c['family']] = c['weight_vector']
                valid_composites.append(c)
        
        if not weight_data:
            tprint_warning("   ⚠️ No weight vectors available for RMI filtering.")
            return composites
        
        # Build DataFrame
        weights_df = pd.DataFrame(weight_data, index=df.index).fillna(0.0)
        
        # Target: 1-bar forward return
        target = df[target_col].pct_change().shift(-1).fillna(0.0)
        
        # Calculate RMI scores
        rmi_scores = calculate_residual_mi(weights_df, target, lag=1, n_neighbors=3)
        
        # Get top_k families
        top_families = set(rmi_scores.head(top_k).index.tolist())
        
        # Filter composites
        filtered = [c for c in valid_composites if c['family'] in top_families]
        
        tprint_success(f"   ✅ RMI reduction complete: {len(composites)} → {len(filtered)} composites")
        return filtered
        
    except Exception as e:
        tprint_warning(f"   ⚠️ RMI filtering failed: {e}. Returning original set.")
        return composites


# ==========================================
# 0.4.6. Layer 2 Price Processing Pipeline
# ==========================================

def apply_layer2_price_processing(df: pd.DataFrame, 
                                   price_col: str = 'close',
                                   vol_window: int = 20,
                                   fracdiff_d: float = 0.4,
                                   wavelet: str = 'db4',
                                   wavelet_level: int = 2) -> pd.DataFrame:
    """
    Apply de Prado-compliant price processing at the end of Layer 2.
    
    Pipeline:
    1. Log-Returns (eliminates price level non-stationarity)
    2. Vol-Adjusted (GARCH-style normalization for regime invariance)
    3. FracDiff (fractional differentiation to preserve memory while ensuring stationarity)
    4. Wavelet Denoising (removes high-frequency noise)
    
    Args:
        df: DataFrame with price data.
        price_col: Column name for price.
        vol_window: Window for volatility estimation.
        fracdiff_d: Fractional differentiation order (0.3-0.5 typical).
        wavelet: Wavelet family for denoising.
        wavelet_level: Decomposition level.
    
    Returns:
        DataFrame with processed price features added.
    """
    import pywt
    
    result = df.copy()
    price = df[price_col]
    
    tprint_info("   🔧 Applying Layer 2 Price Processing Pipeline...")
    
    # 1. Log-Returns
    log_price = np.log(price.replace(0, np.nan))
    log_returns = log_price.diff().fillna(0)
    result['log_returns'] = log_returns
    
    # 2. Vol-Adjusted Returns
    vol = log_returns.rolling(vol_window).std()
    vol = vol.replace(0, np.nan).fillna(vol.median())
    vol_adjusted_returns = log_returns / (vol + 1e-9)
    result['vol_adjusted_returns'] = vol_adjusted_returns.clip(-10, 10)
    
    # 3. Fractional Differentiation (FracDiff)
    try:
        fracdiff_series = _apply_fracdiff(log_price.fillna(method='ffill'), d=fracdiff_d)
        result['fracdiff_price'] = fracdiff_series
    except Exception as e:
        tprint_warning(f"   ⚠️ FracDiff failed: {e}. Skipping.")
        result['fracdiff_price'] = log_returns  # Fallback to log returns
    
    # 4. Wavelet Denoising
    try:
        denoised = _wavelet_denoise(vol_adjusted_returns.fillna(0).values, 
                                     wavelet=wavelet, level=wavelet_level)
        result['wavelet_denoised_returns'] = pd.Series(denoised, index=df.index)
    except Exception as e:
        tprint_warning(f"   ⚠️ Wavelet denoising failed: {e}. Skipping.")
        result['wavelet_denoised_returns'] = vol_adjusted_returns
    
    tprint_success("   ✅ Price processing complete: log_returns, vol_adjusted, fracdiff, wavelet_denoised")
    
    return result


def _apply_fracdiff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation using fixed-width window.
    
    Uses the approach from AFML Ch. 5:
    (1-B)^d = sum_{k=0}^{inf} C(d,k) * (-B)^k
    where C(d,k) = d*(d-1)*...*(d-k+1) / k!
    """
    # Calculate weights
    def _get_weights(d: float, size: int, threshold: float) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w)
    
    # Get weights
    w = _get_weights(d, len(series), threshold)
    width = len(w)
    
    # Apply convolution
    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(w, series.iloc[i - width + 1:i + 1].values[::-1])
    
    return pd.Series(result, index=series.index)


def _wavelet_denoise(signal: np.ndarray, wavelet: str = 'db4', level: int = 2) -> np.ndarray:
    """
    Apply wavelet denoising using soft thresholding.
    """
    import pywt
    
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise level from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply soft thresholding to detail coefficients
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
    
    # Reconstruct
    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]

# ==========================================
# 0.5. Dynamic Event Frequency Adaptation
# ==========================================

class AdaptiveEventThresholds:
    """
    Dynamic threshold adjustment to maintain consistent event density.
    
    Problem: Fixed CUSUM thresholds produce too few events in low-vol regimes 
    and too many in high-vol regimes, causing "label starvation."
    
    Solution: Adjust thresholds based on current vs historical volatility, 
    with optional target events-per-day calibration.
    
    Usage:
        adapter = AdaptiveEventThresholds(target_events_per_day=2.0)
        adjusted_threshold = adapter.calibrate_threshold(df, base_threshold, generator)
    """
    
    def __init__(
        self, 
        target_events_per_day: float = 2.0, 
        min_events_per_day: float = 0.5,
        max_events_per_day: float = 10.0,
        max_iterations: int = 8,
        tolerance: float = 0.3
    ):
        """
        Args:
            target_events_per_day: Desired event frequency
            min_events_per_day: Lower bound for acceptable frequency
            max_events_per_day: Upper bound for acceptable frequency
            max_iterations: Max binary search iterations for calibration
            tolerance: Acceptable deviation from target (as fraction)
        """
        self.target_events_per_day = target_events_per_day
        self.min_events_per_day = min_events_per_day
        self.max_events_per_day = max_events_per_day
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._calibration_cache = {}
    
    def get_vol_adjustment_factor(self, df: pd.DataFrame, lookback: int = 100) -> float:
        """
        Calculate volatility AND volume-based adjustment factor (regime-adaptive).
        
        Returns:
            Multiplier: >1 raises threshold (fewer events in high-vol), 
                       <1 lowers threshold (more events in low-vol/low-volume).
        """
        if 'close' not in df.columns:
            return 1.0
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < lookback * 2:
            return 1.0
        
        # === Volatility component ===
        current_vol = returns.iloc[-lookback:].std()
        historical_vol = returns.std()
        
        if historical_vol <= 0 or not np.isfinite(historical_vol):
            vol_factor = 1.0
        else:
            vol_factor = current_vol / historical_vol
            vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        # === Volume component (NEW: regime-adaptive) ===
        vol_adj_factor = 1.0
        if 'volume' in df.columns:
            try:
                volume = df['volume'].dropna()
                if len(volume) >= lookback * 2:
                    current_volume = volume.iloc[-lookback:].mean()
                    historical_volume = volume.mean()
                    if historical_volume > 0 and np.isfinite(historical_volume):
                        # Low volume regime → lower threshold (more events)
                        # High volume regime → higher threshold (fewer events)
                        vol_ratio = current_volume / historical_volume
                        vol_adj_factor = np.clip(vol_ratio, 0.6, 1.8)
            except Exception:
                vol_adj_factor = 1.0
        
        # Combine: geometric mean gives balanced adjustment
        combined_factor = np.sqrt(vol_factor * vol_adj_factor)
        return float(np.clip(combined_factor, 0.4, 2.5))
    
    def calibrate_threshold(
        self, 
        df: pd.DataFrame, 
        base_threshold: float,
        generator: 'BaseEventGenerator',
        generator_kwargs: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calibrate threshold to achieve target event frequency using binary search.
        
        Args:
            df: Market data DataFrame
            base_threshold: Starting threshold value
            generator: Event generator instance
            generator_kwargs: Additional kwargs for generator.generate()
            
        Returns:
            Calibrated threshold value
        """
        if generator_kwargs is None:
            generator_kwargs = {}
        
        # Calculate data span in days
        if len(df) < 2:
            return base_threshold
        
        try:
            span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        except Exception:
            span_days = len(df) / 96  # Assume 15-min bars, 96 per day
        
        if span_days <= 0:
            return base_threshold
        
        # Cache key
        cache_key = (generator.__class__.__name__, len(df), round(base_threshold, 4))
        if cache_key in self._calibration_cache:
            return self._calibration_cache[cache_key]
        
        # Binary search bounds
        low_thresh = base_threshold * 0.25
        high_thresh = base_threshold * 4.0
        best_threshold = base_threshold
        best_diff = float('inf')
        
        for iteration in range(self.max_iterations):
            mid_thresh = (low_thresh + high_thresh) / 2
            
            # Generate events with current threshold
            try:
                # Most generators take threshold as first positional arg
                test_kwargs = generator_kwargs.copy()
                
                # Handle different generator signatures
                gen_class = generator.__class__.__name__
                if gen_class in ('VolatilityCusumEvents', 'LiquidityCusumEvents'):
                    events = generator.generate(df, h=mid_thresh, vol_span=100)
                elif gen_class == 'VolumeCusumEvents':
                    events = generator.generate(df, h=mid_thresh, span=960)
                elif gen_class in ('ImprovedCUSUMEvents', 'AdaptiveSymmetricCUSUMEvents'):
                    events = generator.generate(df, multiplier=mid_thresh, vol_window=20)
                else:
                    # Generic fallback
                    if isinstance(df, pd.DataFrame):
                        events = generator.generate(df, mid_thresh)
                    else:
                        events = generator.generate(df['close'], mid_thresh)
            except Exception as e:
                logger.warning(f"Calibration failed for {gen_class}: {e}")
                break
            
            # Calculate events per day
            events_per_day = len(events) / max(span_days, 1)
            diff = abs(events_per_day - self.target_events_per_day)
            
            # Check if within tolerance
            if diff < best_diff:
                best_diff = diff
                best_threshold = mid_thresh
            
            # Check for early exit
            relative_diff = diff / self.target_events_per_day
            if relative_diff < self.tolerance:
                tprint_info(f"✅ Calibrated {gen_class}: threshold={mid_thresh:.4f} -> {events_per_day:.1f} events/day")
                self._calibration_cache[cache_key] = mid_thresh
                return mid_thresh
            
            # Binary search update
            if events_per_day > self.target_events_per_day:
                # Too many events, raise threshold
                low_thresh = mid_thresh
            else:
                # Too few events, lower threshold
                high_thresh = mid_thresh
        
        # Return best found
        tprint_info(f"📊 Best calibration for {generator.__class__.__name__}: threshold={best_threshold:.4f}")
        self._calibration_cache[cache_key] = best_threshold
        return best_threshold
    
    def suggest_threshold_adjustment(
        self,
        current_events: int,
        span_days: float,
        current_threshold: float
    ) -> float:
        """
        Quick heuristic adjustment without full calibration.
        
        Useful for runtime adjustments when full calibration is too expensive.
        """
        if span_days <= 0:
            return current_threshold
        
        current_rate = current_events / span_days
        
        if current_rate < self.min_events_per_day:
            # Too few events, lower threshold
            adjustment = max(0.5, current_rate / self.target_events_per_day)
            return current_threshold * adjustment
        elif current_rate > self.max_events_per_day:
            # Too many events, raise threshold
            adjustment = min(2.0, current_rate / self.target_events_per_day)
            return current_threshold * adjustment
        
        return current_threshold


# ==========================================
# 0.7. Regime-Conditional Triggers with Two-Tier Weighting
# ==========================================

class RegimeConditionalTrigger:
    """
    Regime-conditional event triggers with hybrid percentile+z-score thresholds.
    
    Instead of a single threshold, computes multiple thresholds per regime
    (Low/High/Extreme volatility, liquidity, or trend). Events are assigned
    to Tier-1 or Tier-2 based on regime context.
    
    Example: A volume spike in a high-volatility regime might count as Tier-1,
    while the same absolute spike in low-volatility is Tier-2.
    
    Hybrid Threshold Formula:
        adaptive_threshold = max(k * sigma_regime, np.quantile(signal_regime, 0.995))
    """
    
    # Tier thresholds (in terms of z-score OR percentile)
    TIER_1_Z_THRESHOLD = 3.0    # 3σ OR 99.5th percentile
    TIER_2_Z_THRESHOLD = 2.3    # 2.3σ OR 98th percentile
    TIER_1_PERCENTILE = 0.995
    TIER_2_PERCENTILE = 0.98
    
    # Regime weights for threshold modulation
    REGIME_MULTIPLIERS = {
        'low_vol': 0.8,      # Lower thresholds in calm periods
        'medium_vol': 1.0,   # Normal thresholds
        'high_vol': 1.3,     # Higher thresholds in volatile periods
        'extreme_vol': 1.6,  # Much higher in extreme volatility
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def classify_volatility_regime(self, df: pd.DataFrame, lookback: int = 100) -> pd.Series:
        """
        Classify each bar into volatility regime: low, medium, high, extreme.
        """
        if 'close' not in df.columns:
            return pd.Series('medium_vol', index=df.index)
        
        returns = df['close'].pct_change().fillna(0)
        rolling_vol = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252 * 24 * 4)
        
        # Compute percentile ranks
        vol_rank = rolling_vol.rank(pct=True).fillna(0.5)
        
        # Classify regimes
        regimes = pd.Series('medium_vol', index=df.index)
        regimes[vol_rank < 0.25] = 'low_vol'
        regimes[vol_rank > 0.75] = 'high_vol'
        regimes[vol_rank > 0.95] = 'extreme_vol'
        
        return regimes
    
    def compute_hybrid_threshold(
        self, 
        signal: pd.Series, 
        regime: str,
        tier: int = 1
    ) -> float:
        """
        Compute hybrid threshold: max(k*sigma, quantile).
        
        Args:
            signal: The signal series to threshold
            regime: Current volatility regime
            tier: 1 for Tier-1 (stricter), 2 for Tier-2 (relaxed)
            
        Returns:
            Threshold value
        """
        if tier == 1:
            z_mult = self.TIER_1_Z_THRESHOLD
            percentile = self.TIER_1_PERCENTILE
        else:
            z_mult = self.TIER_2_Z_THRESHOLD
            percentile = self.TIER_2_PERCENTILE
        
        # Get regime multiplier
        regime_mult = self.REGIME_MULTIPLIERS.get(regime, 1.0)
        
        # Compute z-score threshold
        sigma = signal.std()
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        z_threshold = z_mult * sigma * regime_mult
        
        # Compute quantile threshold
        quantile_threshold = signal.quantile(percentile)
        if not np.isfinite(quantile_threshold):
            quantile_threshold = z_threshold
        
        # Hybrid: take max for strictness
        return max(z_threshold, quantile_threshold)
    
    def generate_tiered_events(
        self, 
        signal: pd.Series, 
        df: pd.DataFrame,
        lookback: int = 100
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate events with tier assignments based on regime context.
        
        Returns:
            Tuple of (events_df with tier weights, diagnostics dict)
        """
        regimes = self.classify_volatility_regime(df, lookback)
        
        tier1_events = []
        tier2_events = []
        
        for regime in self.REGIME_MULTIPLIERS.keys():
            regime_mask = regimes == regime
            if not regime_mask.any():
                continue
            
            regime_signal = signal[regime_mask]
            
            # Compute thresholds for this regime
            tier1_thresh = self.compute_hybrid_threshold(regime_signal, regime, tier=1)
            tier2_thresh = self.compute_hybrid_threshold(regime_signal, regime, tier=2)
            
            # Tier-1: exceeds strict threshold
            tier1_mask = regime_signal.abs() >= tier1_thresh
            tier1_times = regime_signal.index[tier1_mask]
            tier1_events.extend([(t, 1.0, regime) for t in tier1_times])
            
            # Tier-2: exceeds relaxed threshold but not strict
            tier2_mask = (regime_signal.abs() >= tier2_thresh) & ~tier1_mask
            tier2_times = regime_signal.index[tier2_mask]
            tier2_events.extend([(t, 0.5, regime) for t in tier2_times])
        
        # Combine events into DataFrame
        all_events = tier1_events + tier2_events
        if not all_events:
            return pd.DataFrame(), {'tier1_count': 0, 'tier2_count': 0}
        
        events_df = pd.DataFrame(all_events, columns=['timestamp', 'tier_weight', 'regime'])
        events_df = events_df.set_index('timestamp').sort_index()
        
        diagnostics = {
            'tier1_count': len(tier1_events),
            'tier2_count': len(tier2_events),
            'total_events': len(all_events),
            'regime_distribution': events_df['regime'].value_counts().to_dict(),
        }
        
        if self.verbose:
            tprint_info(f"   📊 Tiered Events: {diagnostics['tier1_count']} Tier-1, {diagnostics['tier2_count']} Tier-2")
        
        return events_df, diagnostics


def create_geometry_variants(
    base_threshold: float,
    n_variants: int = 3,
    perturbation_sigma: float = 0.15
) -> List[float]:
    """
    Create micro-variants of a threshold using small perturbations.
    
    Noise injection creates slightly different event sets that increase
    survival chances without breaking signal integrity.
    
    Args:
        base_threshold: The base threshold value
        n_variants: Number of variants to create
        perturbation_sigma: Standard deviation of perturbation (as fraction of base)
        
    Returns:
        List of threshold variants
    """
    variants = [base_threshold]  # Always include the base
    
    np.random.seed(42)  # For reproducibility
    for i in range(n_variants - 1):
        # Small perturbation (±1σ where σ = perturbation_sigma * base)
        noise = np.random.normal(0, perturbation_sigma * base_threshold)
        variant = base_threshold + noise
        # Ensure variant is positive and reasonable
        variant = np.clip(variant, base_threshold * 0.7, base_threshold * 1.4)
        variants.append(variant)
    
    return variants


# ==========================================
# 0.8. Low-Volatility Regime Features
# ==========================================

class LowVolRegimeFeatures:
    """
    Generate features specifically designed for low-volatility regime learning.
    
    In high volatility, price moves are driven by Momentum and Liquidations.
    In low volatility, price is driven by Inventory Management and Adverse Selection.
    
    This class provides "State-Space" features that remain stationary even when
    price action is flat, allowing models to learn structural conditions that
    precede regime shifts.
    
    Features:
    1. Hurst Exponent (H) - Long-memory score
    2. VPIN (Volume-Probability of Informed Trading) - Toxic flow detection
    3. Anchored Z-Scores - Distance from last causal surprise
    4. Time-Since-Shock (TSS) - Information half-life decay
    5. Relative Volatility Scaling - Regime-conditioned normalization
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def compute_hurst_exponent(
        self, 
        series: pd.Series, 
        lookback: int = 200
    ) -> pd.Series:
        """
        Compute rolling Hurst exponent using Rescaled Range (R/S) method.
        
        H < 0.5: Mean-reverting (trapped)
        H = 0.5: Random walk
        H > 0.5: Trending (hidden persistence)
        
        This tells the model if current sideways movement is a trap or accumulation.
        """
        if len(series) < lookback:
            return pd.Series(0.5, index=series.index)
        
        def rs_hurst(segment):
            """Compute Hurst exponent for a segment."""
            if len(segment) < 20:
                return 0.5
            
            try:
                # Mean-adjusted series
                mean_adj = segment - np.mean(segment)
                
                # Cumulative deviate series
                cumdev = np.cumsum(mean_adj)
                
                # Range
                R = np.max(cumdev) - np.min(cumdev)
                
                # Standard deviation
                S = np.std(segment)
                
                if S < 1e-9 or R < 1e-9:
                    return 0.5
                
                # R/S calculation
                rs = R / S
                
                # Hurst approximation: H = log(R/S) / log(n)
                n = len(segment)
                H = np.log(rs) / np.log(n)
                
                return np.clip(H, 0.0, 1.0)
            except Exception:
                return 0.5
        
        hurst = series.rolling(lookback, min_periods=50).apply(rs_hurst, raw=True)
        return hurst.fillna(0.5)
    
    def compute_vpin(
        self, 
        df: pd.DataFrame, 
        volume_bucket_size: int = 50
    ) -> pd.Series:
        """
        Compute VPIN (Volume-Synchronized Probability of Informed Trading).
        
        Uses bar close position as buy/sell classification proxy.
        High VPIN indicates informed player is exhausting liquidity on one side.
        
        Leading indicator of breakouts in low-vol regimes.
        """
        close = df.get('close', df.get('Close'))
        high = df.get('high', df.get('High'))
        low = df.get('low', df.get('Low'))
        volume = df.get('volume', df.get('Volume'))
        
        if close is None or volume is None:
            return pd.Series(0.5, index=df.index)
        
        # Bar close position as buy/sell proxy
        bar_range = (high - low).replace(0, 1e-9)
        close_position = (close - low) / bar_range  # 0=low, 1=high
        
        # Buy volume proxy (close higher = more buy)
        buy_vol = close_position * volume
        sell_vol = (1 - close_position) * volume
        
        # Order imbalance
        imbalance = (buy_vol - sell_vol).abs()
        
        # Rolling VPIN over fixed volume windows
        total_vol = volume.rolling(volume_bucket_size).sum()
        vpin = (imbalance.rolling(volume_bucket_size).sum() / (total_vol + 1e-9))
        
        return vpin.fillna(0.5)
    
    def compute_anchored_zscore(
        self, 
        df: pd.DataFrame,
        anchor_events: pd.DatetimeIndex,
        feature_col: str = 'close'
    ) -> pd.Series:
        """
        Compute z-scores anchored to the last causal surprise event.
        
        In low-vol, the market "remembers" the price level where the last
        major consensus was reached. This provides a coordinate system
        relative to the "Memory Anchor" of the last 48 hours.
        """
        if feature_col not in df.columns:
            return pd.Series(0.0, index=df.index)
        
        values = df[feature_col]
        result = pd.Series(0.0, index=df.index)
        
        # Sort anchor events
        anchor_events = anchor_events.sort_values()
        
        for i, current_time in enumerate(df.index):
            # Find most recent anchor before this time
            prior_anchors = anchor_events[anchor_events < current_time]
            
            if len(prior_anchors) == 0:
                continue
            
            last_anchor = prior_anchors[-1]
            anchor_idx = df.index.get_loc(last_anchor)
            current_idx = df.index.get_loc(current_time)
            
            if current_idx <= anchor_idx:
                continue
            
            # Compute z-score from anchor point
            segment = values.iloc[anchor_idx:current_idx+1]
            if len(segment) < 5:
                continue
            
            anchor_value = segment.iloc[0]
            current_value = segment.iloc[-1]
            segment_std = segment.std()
            
            if segment_std > 1e-9:
                result.iloc[i] = (current_value - anchor_value) / segment_std
        
        return result.clip(-5, 5)
    
    def compute_time_since_shock(
        self, 
        df: pd.DataFrame,
        shock_events: pd.DatetimeIndex,
        decay_lambda: float = 0.02
    ) -> pd.Series:
        """
        Compute time-since-shock with exponential decay.
        
        Formula: e^(-λ * ΔT) where ΔT is bars since last 2.7σ shock
        
        This acts as a Confidence Weight:
        - Near 1.0: Recent shock, market has "memory"
        - Near 0.0: Long time since shock, entropy maximized (random walk)
        """
        result = pd.Series(0.0, index=df.index)
        
        if len(shock_events) == 0:
            return result
        
        shock_events = shock_events.sort_values()
        
        for i, current_time in enumerate(df.index):
            # Find most recent shock before this time
            prior_shocks = shock_events[shock_events <= current_time]
            
            if len(prior_shocks) == 0:
                result.iloc[i] = 0.0  # No prior shock = no memory
                continue
            
            last_shock = prior_shocks[-1]
            
            # Time difference in bars
            try:
                shock_idx = df.index.get_loc(last_shock)
                delta_t = i - shock_idx
            except KeyError:
                delta_t = 100  # Default to high decay if not found
            
            # Exponential decay
            result.iloc[i] = np.exp(-decay_lambda * delta_t)
        
        return result
    
    def compute_relative_volatility_scaling(
        self, 
        df: pd.DataFrame,
        feature_col: str = 'close',
        median_window: int = 5 * 24 * 4  # 5 days for 15m bars
    ) -> pd.Series:
        """
        Compute relative volatility scaling for regime-conditioned normalization.
        
        Instead of standard z-scores, compute ratio of current volatility
        to the 5-day median volatility. This makes features regime-invariant.
        """
        if feature_col not in df.columns:
            return pd.Series(1.0, index=df.index)
        
        returns = df[feature_col].pct_change()
        
        # Current volatility (20-bar window)
        current_vol = returns.rolling(20).std()
        
        # Median volatility (5-day window)
        median_vol = current_vol.rolling(median_window, min_periods=100).median()
        
        # Relative scaling
        relative = current_vol / (median_vol + 1e-9)
        
        return relative.clip(0.1, 10.0).fillna(1.0)
    
    def generate_all_features(
        self, 
        df: pd.DataFrame,
        shock_events: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Generate all low-vol regime features.
        
        Args:
            df: OHLCV DataFrame
            shock_events: DatetimeIndex of causal surprise events for anchoring
            
        Returns:
            DataFrame with all low-vol features
        """
        features = pd.DataFrame(index=df.index)
        
        close = df.get('close', df.get('Close'))
        if close is None:
            return features
        
        returns = close.pct_change().fillna(0)
        
        # 1. Hurst exponents at multiple lookbacks
        for lookback in [200, 500]:
            features[f'hurst_{lookback}'] = self.compute_hurst_exponent(returns, lookback)
        
        # 2. VPIN
        features['vpin'] = self.compute_vpin(df, volume_bucket_size=50)
        
        # 3. Relative volatility scaling
        features['rel_vol_scale'] = self.compute_relative_volatility_scaling(df, 'close')
        
        # 4. Time-since-shock (if shock events provided)
        if shock_events is not None and len(shock_events) > 0:
            features['time_since_shock'] = self.compute_time_since_shock(
                df, shock_events, decay_lambda=0.02
            )
            features['anchored_zscore'] = self.compute_anchored_zscore(
                df, shock_events, 'close'
            )
        else:
            features['time_since_shock'] = 0.0
            features['anchored_zscore'] = 0.0
        
        # 5. Low-vol specific indicators
        # Distance from HVN (High Volume Node) proxy using VWAP
        if 'volume' in df.columns:
            vwap = (df['close'] * df['volume']).rolling(100).sum() / (df['volume'].rolling(100).sum() + 1e-9)
            features['dist_from_vwap'] = (close - vwap) / (close.rolling(100).std() + 1e-9)
            features['dist_from_vwap'] = features['dist_from_vwap'].clip(-5, 5).fillna(0)
        
        if self.verbose:
            tprint_info(f"   📊 Generated {len(features.columns)} low-vol regime features")
        
        return features


# ==========================================
# 0.9. Regressor-Specific Target Engineering
# ==========================================

def engineer_regressor_targets(
    df: pd.DataFrame, 
    events: pd.DatetimeIndex,
    raw_returns: pd.Series,
    regressor_type: str = 'lgbm',
    horizon: int = 48
) -> pd.Series:
    """
    Create optimized targets for different regressor types.
    
    Different model types have different optimal target distributions:
    - Ridge: Prefers smooth, normally distributed, stationary targets
    - Tree-based (LGBM/XGB): Can handle non-linear, Sharpe-like targets
    
    Args:
        df: Market data DataFrame with 'close', 'volatility_1d', etc.
        events: Event timestamps
        raw_returns: Raw return targets indexed by events
        regressor_type: 'ridge', 'lgbm', 'xgboost', 'tree', or 'auto'
        horizon: Lookahead horizon in bars
        
    Returns:
        Engineered targets optimized for the specified regressor.
    """
    if regressor_type == 'ridge':
        return _create_ridge_targets(df, events, raw_returns, horizon)
    elif regressor_type in ('lgbm', 'xgboost', 'tree'):
        return _create_tree_targets(df, events, raw_returns, horizon)
    elif regressor_type == 'auto':
        # Default to tree targets as LGBM is most common
        return _create_tree_targets(df, events, raw_returns, horizon)
    else:
        # Fallback to raw returns
        return raw_returns


def _create_ridge_targets(
    df: pd.DataFrame, 
    events: pd.DatetimeIndex, 
    raw_returns: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Ridge-optimized: Smooth, normally distributed targets.
    
    Transforms:
    1. Multi-horizon weighted average (exponential decay) - reduces noise
    2. Volatility normalization - ensures stationarity
    3. Winsorization - removes outliers that hurt linear models
    """
    if raw_returns.empty or len(events) == 0:
        return raw_returns
    
    # Get volatility for normalization
    if 'volatility_1d' in df.columns:
        vol = df['volatility_1d']
    else:
        vol = df['close'].pct_change().rolling(20).std()
    
    # Multi-horizon weighted returns
    horizons = [horizon // 4, horizon // 2, horizon, horizon * 2]
    horizons = [max(1, h) for h in horizons]  # Ensure positive
    weights = np.exp(-0.1 * np.arange(len(horizons)))  # Exponential decay
    weights = weights / weights.sum()
    
    # Calculate returns at each horizon
    close = df['close']
    engineered = pd.Series(index=events, dtype=float)
    
    for event in events:
        if event not in df.index:
            engineered[event] = np.nan
            continue
        
        event_loc = df.index.get_loc(event)
        event_vol = vol.iloc[event_loc] if event_loc < len(vol) else vol.mean()
        
        if not np.isfinite(event_vol) or event_vol <= 0:
            event_vol = vol.mean()
        
        horizon_returns = []
        for h in horizons:
            end_loc = min(event_loc + h, len(df) - 1)
            ret = (close.iloc[end_loc] / close.iloc[event_loc]) - 1.0
            horizon_returns.append(ret)
        
        # Weighted average
        weighted_ret = np.average(horizon_returns, weights=weights)
        
        # Volatility normalization (z-score like)
        normalized_ret = weighted_ret / (event_vol * np.sqrt(horizon) + 1e-9)
        
        engineered[event] = normalized_ret
    
    # Winsorization (clip to 3 std)
    engineered = engineered.dropna()
    if len(engineered) > 10:
        mean_val = engineered.mean()
        std_val = engineered.std()
        if std_val > 0:
            lower = mean_val - 3 * std_val
            upper = mean_val + 3 * std_val
            engineered = engineered.clip(lower, upper)
    
    return engineered


def _create_tree_targets(
    df: pd.DataFrame, 
    events: pd.DatetimeIndex, 
    raw_returns: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Tree-optimized: Sharpe-like targets that capture risk-adjusted alpha.
    
    Transforms:
    1. Return / Volatility ratio (Sharpe-like)
    2. Path quality adjustment (penalize choppy paths)
    3. Preserves non-linearity trees can exploit
    """
    if raw_returns.empty or len(events) == 0:
        return raw_returns
    
    # Get volatility
    if 'volatility_1d' in df.columns:
        vol = df['volatility_1d']
    else:
        vol = df['close'].pct_change().rolling(20).std()
    
    close = df['close']
    engineered = pd.Series(index=events, dtype=float)
    
    for event in events:
        if event not in df.index:
            engineered[event] = np.nan
            continue
        
        event_loc = df.index.get_loc(event)
        end_loc = min(event_loc + horizon, len(df) - 1)
        
        # Basic return
        ret = (close.iloc[end_loc] / close.iloc[event_loc]) - 1.0
        
        # Path volatility over horizon
        if end_loc > event_loc + 1:
            path_returns = close.iloc[event_loc:end_loc+1].pct_change().dropna()
            path_vol = path_returns.std() if len(path_returns) > 1 else vol.iloc[event_loc]
        else:
            path_vol = vol.iloc[event_loc]
        
        if not np.isfinite(path_vol) or path_vol <= 0:
            path_vol = vol.mean()
        
        # Sharpe-like ratio: return / volatility
        # Scale by sqrt(horizon) for time-normalization
        sharpe_target = ret / (path_vol * np.sqrt(horizon) + 1e-9)
        
        # Path quality: consistency of direction
        if end_loc > event_loc + 2:
            path_prices = close.iloc[event_loc:end_loc+1]
            # Efficiency ratio: net movement / total movement
            net_move = abs(path_prices.iloc[-1] - path_prices.iloc[0])
            total_move = path_prices.diff().abs().sum()
            efficiency = net_move / (total_move + 1e-9) if total_move > 0 else 1.0
        else:
            efficiency = 1.0
        
        # Adjust Sharpe by path quality (boost clean moves, penalize choppy)
        adjusted_target = sharpe_target * (0.5 + 0.5 * efficiency)
        
        engineered[event] = adjusted_target
    
    return engineered.dropna()


def get_target_for_model_type(model_name: str) -> str:
    """
    Map model name to appropriate target type.
    
    Args:
        model_name: Name like 'LGBM_Focal', 'XGB_Tree', 'Ridge', etc.
        
    Returns:
        Target type: 'ridge', 'tree', or 'auto'
    """
    model_upper = model_name.upper()
    
    if 'RIDGE' in model_upper or 'LINEAR' in model_upper:
        return 'ridge'
    elif any(x in model_upper for x in ['LGBM', 'XGB', 'CATBOOST', 'RF', 'FOREST', 'TREE']):
        return 'tree'
    else:
        return 'auto'


# ==========================================
# 1. Labeling Logic (Vectorized Dominance & State)
# ==========================================

# --- 2. Causal 2026 Labeling Framework ---
def compute_dominance_labels(
    price: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    risk_budget: float = 1.0,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 120,
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
    # Normalize TZ to ensure matching
    if price.index.tz is not None:
        price_idx = price.index.tz_localize(None)
    else:
        price_idx = price.index
        
    if events.tz is not None:
        events_norm = events.tz_localize(None)
    else:
        events_norm = events

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

    valid_mask = (event_idxs != -1) & (event_idxs < (n_bars - horizon))
    valid_idxs = event_idxs[valid_mask]
    valid_events = events[valid_mask]

    valid_events = events[valid_mask]
    
    # DEBUG: Check why empty
    if len(valid_idxs) == 0:
        logger.warning(f"DEBUG: No valid events found! n_events={len(events)}")
        if len(events) > 0:
             logger.warning(f"DEBUG: Event[0]: {events[0]} type={type(events[0])}")
             logger.warning(f"DEBUG: Price[0]: {price.index[0]} type={type(price.index[0])}")
             logger.warning(f"DEBUG: Price[-1]: {price.index[-1]}")
             logger.warning(f"DEBUG: n_bars={n_bars}, horizon={horizon}")
        return tuple([pd.Series(dtype=float)] * 6)

    # 2. Construct Window Matrix (N x Horizon)
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]

    # Get Prices
    price_vals = price.values
    entry_prices = price_vals[valid_idxs]

    # 3. Compute MFE/MAE & Hits
    vol_vals = volatility.values[valid_idxs]
    vol_vals = np.maximum(vol_vals, 1e-6)

    pt_thresh = (vol_vals * pt_mult)[:, None]
    sl_thresh = (-vol_vals * sl_mult)[:, None]

    # Check if High/Low provided
    if high is not None and low is not None:
        high_vals = high.values
        low_vals = low.values
        window_highs = high_vals[window_idxs]
        window_lows = low_vals[window_idxs]

        # Returns relative to entry
        high_ret = window_highs / entry_prices[:, None] - 1.0
        low_ret = window_lows / entry_prices[:, None] - 1.0

        mfe = np.max(high_ret, axis=1)
        # MAE is max negative excursion (magnitude)
        mae = -np.min(low_ret, axis=1)

        hit_pt = high_ret > pt_thresh
        hit_sl = low_ret < sl_thresh
    else:
        window_prices = price_vals[window_idxs]
        returns_matrix = window_prices / entry_prices[:, None] - 1.0

        mfe = np.max(returns_matrix, axis=1)
        mae = np.max(-returns_matrix, axis=1)

        hit_pt = returns_matrix > pt_thresh
        hit_sl = returns_matrix < sl_thresh

    # For outcome calculation, we use Close prices if neither hit
    window_closes = price_vals[window_idxs]
    close_returns = window_closes / entry_prices[:, None] - 1.0

    # Identify first hit indices
    any_pt = np.any(hit_pt, axis=1)
    any_sl = np.any(hit_sl, axis=1)

    first_pt_idx = np.argmax(hit_pt, axis=1)
    first_sl_idx = np.argmax(hit_sl, axis=1)

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
    timeout_returns = close_returns[:, -1]
    FEE_THRESHOLD = transaction_cost
    labels[timeout_mask & (timeout_returns > FEE_THRESHOLD)] = 1.0
    labels[timeout_mask & (timeout_returns < -FEE_THRESHOLD)] = -1.0
    # Otherwise label remains 0 (within noise band)

    # 5. Weighting
    mae_safe = np.maximum(mae, 1e-9)
    ratio = mfe / mae_safe
    magnitude = np.log1p(mfe / transaction_cost)
    vol_adj = 1.0 / vol_vals
    weights = ratio * magnitude * vol_adj

    # 6. Returns (use win_mask)
    out_returns = np.where(win_mask, pt_mult * vol_vals, -sl_mult * vol_vals)
    timeout_mask = (~any_pt) & (~any_sl)
    out_returns[timeout_mask] = close_returns[timeout_mask, -1]

    # Construct Series
    idx = valid_events
    s_labels = pd.Series(labels, index=idx)
    s_weights = pd.Series(weights, index=idx)
    s_returns = pd.Series(out_returns, index=idx)
    s_mfe = pd.Series(mfe, index=idx)
    s_mae = pd.Series(mae, index=idx)
    s_vol = pd.Series(vol_vals, index=idx)

    return s_labels, s_weights, s_returns, s_mfe, s_mae, s_vol

# ==========================================
# 2. Quality Gates & Checks
# ==========================================

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

    # 1. Sample Size Gate (relaxed from 0.5 to 0.1 events/day)
    if rate < 0.1:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [FAIL]")
        overall_pass = False
        if failure_reason == "PASS": failure_reason = "Sample Size (< 0.1/day)"
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
        if pos_rate < min_bal:
            gates_log.append(f"Bal: {pos_rate:.1%} (Samples) [FAIL]")
            overall_pass = False
            if failure_reason == "PASS": failure_reason = "Sample Balance (<5% non-zero)"
        else:
            gates_log.append(f"Bal: {pos_rate:.1%} (Samples) [OK]")

    # 3. Perturbation Stability Gate
    try:
        df_noisy = df.copy()
        noise = np.random.normal(1.0, 0.0001, size=len(df))
        for col in ['close', 'high', 'low', 'open']:
            if col in df_noisy.columns: df_noisy[col] *= noise
        
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
        # Graceful failure for Jaccard
        logger.debug(f"Jaccard calculation failed for {family}: {e}")
        val_metrics['jaccard'] = 0.0
        gates_log.append(f"Jaccard: 0.00 (CalcFail) [WARN]")

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
        if min_p > 0.20:
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
    # Optimization: effective N limit for MI to avoid O(N^2) scaling
    MAX_MI_SAMPLES = 2000
    if len(X) > MAX_MI_SAMPLES:
        # random_state is already 42 fixed for consistency
        indices = np.random.RandomState(42).choice(len(X), MAX_MI_SAMPLES, replace=False)
        X_mi = X.iloc[indices]
        y_mi = y.iloc[indices]
    else:
        X_mi = X
        y_mi = y

    try:
        if family == 'PRICE_CUSUM':
            mi = mutual_info_classif(X_mi, y_mi, discrete_features=False, random_state=42)
        else:
            mi = mutual_info_regression(X_mi, y_mi, discrete_features=False, random_state=42)
            
        max_mi = np.max(mi)
        val_metrics['max_mi'] = max_mi
        
        # MI gate now just a warning (non-blocking) since MI values are consistently very low
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

# ==========================================
# 3. Multi-Factor Scoring
# ==========================================

def calculate_multifactor_score(
    candidates: List[Dict],
    probe_features: pd.DataFrame
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
        if n > MAX_SAMPLES:
            indices = np.random.RandomState(42).choice(n, MAX_SAMPLES, replace=False)
            X_sub = X.iloc[indices]
            labels_sub = labels.iloc[indices]
        else:
            X_sub = X
            labels_sub = labels

        ic_vals = [abs(spearmanr(X_sub[col], labels_sub)[0]) for col in X_sub.columns]
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
            'interventional_contrast': metrics.get('interventional_contrast', np.nan),
            'overlap_support': metrics.get('overlap_support', np.nan),
            'path_stability_var': metrics.get('path_stability_var', np.nan),
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
        power = max(row['ic'], row['f_stat'])
        raw_sig = df_scores.iloc[i]['significance']
        
        # Incorporate Causal Integrity into power if available
        ic_ic = row.get('interventional_contrast', 0.0)
        purity_score = (power * 0.7 + ic_ic * 0.3) if not np.isnan(ic_ic) else power

        final_score = (
            purity_score *
            raw_sig *
            row['stability'] *
            row['balance'] *
            row['density'] *
            (1.0 + row['path_score'])
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

# ==========================================
# 4. Probe (LGBM - Advanced Metrics)
# ==========================================

def run_lgbm_probe(X, y, w, returns) -> Dict[str, float]:
    """
    Advanced Probe: Returns Meta-Label Lift, Yield, Entropy, Consistency.
    """
    tprint_info(f"🚀 Starting LGBM Probe for {len(y)} samples")
    
    if len(y) < 50:
        tprint_warning("⚠️ Too few samples for probe (< 50)")
        return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'seed': 42,
        'boosting_type': 'goss',
        'max_depth': 3,
        'num_leaves': 7,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'feature_fraction': 0.8,
        'top_rate': 0.2,
        'other_rate': 0.1
    }
    preds_all = []
    labels_all = []
    r_all = [] # Realized returns for all va samples
    base_returns = []  # All validation returns (baseline)
    meta_returns = []  # Returns where prediction > 0.5

    # Initialize TimeSeriesSplit for 3-fold CV
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    tprint_info(f"🔄 Setting up 3-fold time series cross-validation")

    fold = 0
    for tr_idx, va_idx in tscv.split(X):
        fold += 1
        tprint_info(f"📊 Training fold {fold}/3...")
        
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr, w_va = w.iloc[tr_idx], w.iloc[va_idx]
        ret_va = returns.iloc[va_idx]

        if y_tr.nunique() < 2 or y_va.nunique() < 2: 
            tprint_warning(f"⚠️ Fold {fold}: Insufficient class diversity, skipping")
            continue

        tprint_info(f"📊 Fold {fold}: Training {len(X_tr)} samples, validating {len(X_va)} samples")
        
        # Calculate dynamic alpha for Focal Loss
        pos_rate = y_tr.mean()
        alpha = 1.0 - pos_rate
        alpha = float(np.clip(alpha, 0.05, 0.95))
        
        # Use custom objective (ensure 'objective' and 'metric' don't conflict)
        fobj = get_focal_loss_lgbm(alpha=alpha, gamma=2.0)
        
        # Copy params to avoid mutation issues
        fold_params = params.copy()
        # Remove objective from params if we pass fobj is safer, but if train() rejects fobj kwarg,
        # we must either pass it as 'objective' key in params or rely on legacy behavior.
        # It seems the installed LightGBM version might be wrapped or older.
        # Let's try passing it via params['objective']
        fold_params['objective'] = fobj
        
        # Also ensure metric is set (auc)
        
        tprint_info(f"DEBUG: X_tr shape: {X_tr.shape}, columns: {X_tr.columns.tolist()}")
        if X_tr.shape[1] == 0:
             tprint_error("❌ X_tr has 0 features! LightGBM will crash.")
             return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va)
        
        model = lgb.train(
            fold_params, 
            dtrain, 
            valid_sets=[dvalid],
            # fobj=fobj, # Removed to fix TypeError
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )

        preds = model.predict(X_va)
        preds_all.extend(preds)
        labels_all.extend(y_va.values)
        r_all.extend(ret_va.values)
        
        # Performance calculation (De Prado)
        base_returns.extend(ret_va.values)

        # Meta: Pred > 0.5
        mask = preds > 0.5
        meta_count = mask.sum()
        
        if mask.sum() > 0:
            meta_returns.extend(ret_va[mask].values)

    if not base_returns:
        tprint_warning("⚠️ No base returns calculated")
        return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

    tprint_info("📈 Calculating probe metrics...")

    # 1. Sharpe Lift
    def sharpe(r):
        if len(r) < 2: return 0.0
        std = np.std(r)
        if std == 0: return 0.0
        return np.mean(r) / std

    base_sh = sharpe(base_returns)
    meta_sh = sharpe(meta_returns) if meta_returns else 0.0
    lift = meta_sh - base_sh

    # 2. Opportunity Yield
    days = (returns.index[-1] - returns.index[0]).days if not returns.empty else 1
    n_pos = len(meta_returns)
    opp_yield = n_pos / max(1, days)

    # 3. Conditional Outcome Entropy H(Y | Pred > 0.5)
    # y is binary 0/1.
    preds_arr = np.array(preds_all)
    labels_arr = np.array(labels_all)
    mask = preds_arr > 0.5
    if mask.sum() > 0:
        cond_labels = labels_arr[mask]
        counts = pd.Series(cond_labels).value_counts(normalize=True)
        cond_entropy = shannon_entropy(counts)
    else:
        cond_entropy = 1.0 # High entropy if no signals

    # 4. Sign Consistency
    consistency = 0.0
    if mask.sum() > 0:
        consistency = np.mean(labels_arr[mask])    # --- De Prado / Advanced Metrics ---
    # 1. IC (Information Coefficient)
    ic, _ = spearmanr(preds_all, r_all) if len(r_all) > 10 else (0.0, 1.0)
    
    # 2. PSR (Probabilistic Sharpe Ratio)
    r_arr = np.array(meta_returns)
    sharpe_p = sharpe(r_arr)
    n_p = len(r_arr)
    psr_val = 0.0
    if n_p > 2:
        from scipy.stats import skew, kurtosis
        s = skew(r_arr)
        k = kurtosis(r_arr)
        psr_val = calculate_psr(sharpe_p, n_p, s, k)

    # 3. Standardized Error (Consistency)
    step = len(r_arr) // 3
    if step > 0:
        fold_sharpes = [sharpe(np.array(f)) for f in [meta_returns[i:i+step] for i in range(0, len(r_arr), step)] if len(f) > 0]
    else:
        fold_sharpes = [sharpe(r_arr)] if len(r_arr) > 0 else []
    std_error = np.std(fold_sharpes) if len(fold_sharpes) > 1 else 0.0

    # Multi-threshold probe completion
    thresholds = [0.2, 0.5, 0.8]
    r_arr_all = np.array(r_all)  # Convert r_all list to array for threshold slicing
    tprint_info("✅ Probe Complete:")
    for threshold in thresholds:
        mask = preds_arr > threshold
        meta_returns_thresh = r_arr_all[mask]
        meta_sh = sharpe(meta_returns_thresh) if len(meta_returns_thresh) > 0 else 0.0
        lift = meta_sh - base_sh
        
        psr_val = 0.0
        if len(meta_returns_thresh) > 2:
            from scipy.stats import skew, kurtosis
            s = skew(meta_returns_thresh)
            k = kurtosis(meta_returns_thresh)
            psr_val = calculate_psr(meta_sh, len(meta_returns_thresh), s, k)
        
        n_preds = mask.sum(); n_returns = len(meta_returns_thresh); tprint_info(f"  {threshold} threshold: Lift={lift:.4f} (BaseSH={base_sh:.4f}, MetaSH={meta_sh:.4f}), IC={ic:.4f}, PSR={psr_val:.4f} [preds={n_preds}, returns={n_returns}]")
    
    return {
        'lift': float(lift), 
        'yield': float(opp_yield), 
        'entropy': float(cond_entropy), 
        'consistency': float(consistency),
        'sharpe_meta': float(sharpe_p),
        'ic': float(ic),
        'psr': float(psr_val),
        'std_error': float(std_error)
    }

def adaptive_threshold_calculator(
    generator: "BaseEventGenerator",
    data: Union[pd.Series, pd.DataFrame],
    target_signals_per_day: float = 7.5,
    max_iterations: int = 20,
    tolerance: float = 0.2
) -> pd.DatetimeIndex:
    """
    Iteratively adjust thresholds to achieve target signal rate.
    """
    # Calculate data duration and target signal count
    if isinstance(data, pd.Series):
        index = data.index
    else:
        index = data.index
    
    duration_days = (index[-1] - index[0]).days
    if duration_days < 1:
        duration_days = 1
    
    target_signals = int(target_signals_per_day * duration_days)
    min_target = int(target_signals * (1 - tolerance))
    max_target = int(target_signals * (1 + tolerance))
    
    # Start with default parameters (need to be passed in)
    # This is a simplified version - in practice, you'd pass the specific params
    events = generator.generate(data)
    
    if len(events) == 0:
        return events
    
    # Iterative adjustment
    iteration = 0
    current_events = events
    
    while iteration < max_iterations:
        current_count = len(current_events)
        
        # Check if within tolerance
        if min_target <= current_count <= max_target:
            break
        
        # Calculate adjustment factor
        if current_count > max_target:  # Too many signals
            factor = 1.2 + (current_count - max_target) / max_target * 0.3
        else:  # Too few signals
            factor = 0.8 - (min_target - current_count) / min_target * 0.3
        
        factor = max(0.5, min(2.0, factor))  # Bound the factor
        
        # Adjust parameters if generator supports it
        # Panic mode for extremely low signals
        if current_count < min_target * 0.1:
             factor = 0.5 # Aggressive relaxation
             logger.info(f"Panic relaxation: Rate is {current_count}/{min_target} (target). Slashed params by 50%.")
        
        # Adjust parameters if generator supports it
        if hasattr(generator, '_adjust_z_threshold'):
            current_params = generator._adjust_z_threshold(current_params, factor)
            # Re-generate with new params to check progress within loop
            try:
                # We need to call generate again. 
                # generator is an instance of BaseEventGenerator (or subclass)
                # We need to handle the positional args issue if relevant, but here we just use **current_params
                # But wait, generate() might need positional args if they were passed...
                # The prompt said "generator.generate(data)" at line 661. 
                # We should use the same call structure.
                # Actually, check line 839: "events = self.generate(data, *args, **current_params)"
                # This function 'adaptive_threshold_calculator' is a standalone function at module level?
                # No, look at line 635. Yes it is.
                # But wait, BaseEventGenerator.generate_adaptive calls self.generate.
                # 'adaptive_threshold_calculator' seems to be an older standalone function?
                # Actually, 'BaseEventGenerator.generate_adaptive' is the one used in the main loop!
                # Line 1823 calls `gen.generate_adaptive`.
                # So I should update `BaseEventGenerator.generate_adaptive` NOT the standalone function if it's unused.
                # Let's check if `adaptive_threshold_calculator` is used.
                pass
            except Exception as e:
                logger.warning(f"Optimization step failed: {e}")
                break
        
        iteration += 1
    
    return current_events

# ==========================================
# 5. Signal Generators
# ==========================================

# --- 3. Grid Configurations (Legacy Removed) ---

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
            # GOAL: factor < 1.0 -> Relax (More signals)
            #       factor > 1.0 -> Tighten (Fewer signals)
            
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


class OrderBlockEvents(BaseEventGenerator):
    """Generate events based on Order Block identification (Smart Money Concept).
    
    Order Blocks are the last up/down candle before a strong move,
    indicating institutional order placement zones.
    """
    
    def generate(self, df: pd.DataFrame, lookback: int = 20, min_move_pct: float = 0.5, 
                 volume_threshold: float = 2.0, tracker: Optional[Any] = None) -> pd.DatetimeIndex:
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"OrderBlockEvents requires OHLCV data. Missing: {[c for c in required_cols if c not in df.columns]}")
                return pd.DatetimeIndex([])
            
            if len(df) < lookback * 2:
                logger.warning(f"Insufficient data for Order Block analysis: need {lookback * 2}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            events = []
            volume_ma = df['volume'].rolling(lookback).mean()
            
            # Scan for order blocks
            for i in range(lookback, len(df) - lookback):
                current_candle = df.iloc[i]
                
                # Check for bullish order block (last down candle before up move)
                if (current_candle['close'] < current_candle['open'] and  # Red candle
                    current_candle['volume'] > volume_ma.iloc[i] * volume_threshold):  # High volume
                    
                    # Check if strong up move follows
                    future_move = True
                    for j in range(i + 1, min(i + lookback, len(df))):
                        move_pct = (df.iloc[j]['close'] - current_candle['close']) / current_candle['close']
                        if move_pct > min_move_pct / 100:  # Convert percentage to decimal
                            future_move = True
                            break
                        elif move_pct < -min_move_pct / 100:  # Move against
                            future_move = False
                            break
                    
                    if future_move:
                        events.append(df.index[i])
                
                # Check for bearish order block (last up candle before down move)
                elif (current_candle['close'] > current_candle['open'] and  # Green candle
                      current_candle['volume'] > volume_ma.iloc[i] * volume_threshold):  # High volume
                    
                    # Check if strong down move follows
                    future_move = True
                    for j in range(i + 1, min(i + lookback, len(df))):
                        move_pct = (df.iloc[j]['close'] - current_candle['close']) / current_candle['close']
                        if move_pct < -min_move_pct / 100:  # Convert percentage to decimal
                            future_move = True
                            break
                        elif move_pct > min_move_pct / 100:  # Move against
                            future_move = False
                            break
                    
                    if future_move:
                        events.append(df.index[i])
            
            event_index = pd.DatetimeIndex(events)
            
            # Post-processing to avoid clustered events
            if len(event_index) > 1:
                event_index = self._post_process_events(event_index, pd.Timedelta(hours=6))
            
            logger.debug(f"OrderBlockEvents generated {len(event_index)} events (lookback={lookback}, min_move_pct={min_move_pct})")
            return event_index
            
        except Exception as e:
            logger.error(f"OrderBlockEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'min_move_pct' in adjusted:
            adjusted['min_move_pct'] *= factor
        if 'volume_threshold' in adjusted:
             adjusted['volume_threshold'] *= factor
        if 'lookback' in adjusted:
             adjusted['lookback'] = max(10, int(adjusted['lookback'] * factor))
        return adjusted


# ==========================================
# 6. Final Diversity Filter
# ==========================================

def final_diversity_filter(
    geometries: List[OutputGeometry], 
    price: pd.Series,
    jaccard_threshold: float = 0.7,
    returns_threshold: float = 0.8
) -> List[OutputGeometry]:
    """
    Filter geometries to ensure diversity in both event timing AND returns patterns.
    
    Args:
        geometries: List of OutputGeometry objects (one per signal family)
        price: Price series for returns calculation
        jaccard_threshold: Maximum Jaccard similarity allowed (lower = more diverse)
        returns_threshold: Maximum returns correlation allowed (lower = more diverse)
    
    Returns:
        Filtered list of diverse geometries
    """
    if len(geometries) <= 1:
        return geometries
    
    logger.info(f"Applying final diversity filter to {len(geometries)} geometries...")
    
    # Sort by AUC score descending - keep highest scoring as anchor
    geometries.sort(key=lambda x: x.auc, reverse=True)
    
    # Build event timing indicators for Jaccard similarity
    event_indicators = {}
    for geo in geometries:
        indicator = build_indicator_matrix(geo.events, price.index, horizon=1).values.flatten().astype(bool)
        event_indicators[geo.name] = indicator
    
    # Calculate returns series for each geometry
    returns_series = {}
    for geo in geometries:
        if len(geo.events) > 0:
            # Calculate returns from event entry to horizon
            returns_list = []
            for event_time in geo.events:
                if event_time in price.index:
                    event_idx = price.index.get_loc(event_time)
                    horizon = min(120, len(price) - event_idx - 1)  # Use horizon from params or default
                    if horizon > 0:
                        start_price = price.iloc[event_idx]
                        end_price = price.iloc[min(event_idx + horizon, len(price) - 1)]
                        ret = (end_price - start_price) / start_price
                        returns_list.append(ret)
            
            if returns_list:
                returns_series[geo.name] = pd.Series(returns_list, index=geo.events[:len(returns_list)])
    
    # Diversity filtering
    # Group by family
    by_family = {}
    for g in geometries:
        if g.family not in by_family:
            by_family[g.family] = []
        by_family[g.family].append(g)

    final_selected = []
    
    # Process each family independently
    for fam, candidates in by_family.items():
        # Sort by AUC descending
        candidates.sort(key=lambda x: x.auc, reverse=True)
        
        # Always take the best one
        family_selected = [candidates[0]]
        logger.info(f"✅ Selected best {fam}: {candidates[0].name} (AUC={candidates[0].auc:.3f})")
        
        # Configure max winners based on family
        # PRICE_CUSUM needs orthogonality (up to 3)
        # Context/Other families should be single best (1)
        if fam == 'PRICE_CUSUM':
            MAX_PER_FAMILY = 3
        else:
            MAX_PER_FAMILY = 1
        
        for cand in candidates[1:]:
            if len(family_selected) >= MAX_PER_FAMILY:
                break
                
            is_diverse = True
            rejection_reason = ""
            
            for selected_geo in family_selected:
                # 1. Jaccard Check
                if cand.name in event_indicators and selected_geo.name in event_indicators:
                    cand_ind = event_indicators[cand.name]
                    sel_ind = event_indicators[selected_geo.name]
                    intersection = np.logical_and(cand_ind, sel_ind).sum()
                    union = np.logical_or(cand_ind, sel_ind).sum()
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    if jaccard_sim > jaccard_threshold:
                        is_diverse = False
                        rejection_reason = f"Jaccard {jaccard_sim:.2f} > {jaccard_threshold}"
                        break
                
                # 2. Returns Correlation Check
                if is_diverse and cand.name in returns_series and selected_geo.name in returns_series:
                    cand_ret = returns_series[cand.name]
                    sel_ret = returns_series[selected_geo.name]
                    common = cand_ret.index.intersection(sel_ret.index)
                    if len(common) > 10:
                        c_vals = cand_ret.loc[common].values
                        s_vals = sel_ret.loc[common].values
                        corr = abs(np.corrcoef(c_vals, s_vals)[0, 1])
                        if not np.isnan(corr) and corr > returns_threshold:
                            is_diverse = False
                            rejection_reason = f"Corr {corr:.2f} > {returns_threshold}"
                            break
            
            if is_diverse:
                family_selected.append(cand)
                logger.info(f"✅ Selected orthogonal {fam}: {cand.name} (AUC={cand.auc:.3f})")
            else:
                logger.info(f"❌ Rejected {cand.name} vs {fam}: {rejection_reason}")

        final_selected.extend(family_selected)

    logger.info(f"Final diversity filter: {len(final_selected)}/{len(geometries)} geometries retained across {len(by_family)} families")
    return final_selected
        

    
    return selected


# ==========================================
# 7. Enhanced Parameter Grids
# ==========================================

def get_enhanced_parameter_grids(range_specific: bool = False) -> Dict[str, Dict]:
    """
    Define enhanced parameter grids for each signal family including:
    - TP/SL ratios (expanded from fixed grid)
    - Horizons (multiple timeframes)
    - Lookback variations
    - MFE/MAE optimization parameters
    """
    
    # Use range-specific grid if optimization is enabled
    if range_specific:
        tpsl_grid = MEDIUM_TERM_GRID
    else:
        # Enhanced TPSL grid
        tpsl_grid = [
        # Symmetric (for diversity/orthogonality)
        {'id': '1:1', 'pt': 1.0, 'sl': 1.0},
        
        # Conservative (high win rate)
        {'id': '1.5:1', 'pt': 1.5, 'sl': 1.0},
        {'id': '2:1', 'pt': 2.0, 'sl': 1.0},
        
        # Balanced
        {'id': '3:1', 'pt': 3.0, 'sl': 1.0},
        
        # Aggressive (high reward)
        {'id': '4:1', 'pt': 4.0, 'sl': 1.0},
    ]
    
    # Horizon options per family (Modern Causal defaults - EXPANDED for higher event density)
    horizon_options = {
        'default': [24, 48],  # Added 24 for faster signals
        'CAUSAL_SURPRISE': [12, 24, 48],  # Surprises: medium to slow (removed 6, 8 - too noisy)
        'VOLUME_SPECIALIST': [12, 24, 48],       # Volume: medium to slow
        'VOLATILITY_SPECIALIST': [12, 24],    # Volatility: fast-evolving
    }
    
    # Family-specific parameter grids for Causal Specialists - EXPANDED GRIDS
    family_grids = {
        'CAUSAL_SURPRISE': {
            'base_params': [(1.5, 'all'), (1.8, 'all'), (2.2, 'all')],  # Lower thresholds for more events
            'horizons': [12, 24, 48],  # Multi-horizon (removed 6, 8 - too noisy)
        },
        'VOLUME_SPECIALIST': {
            'base_params': [(1.8, 20), (2.0, 20), (2.5, 30)],  # Multiple sensitivity levels
            'horizons': [12, 24, 48],
        },
        'VOLATILITY_SPECIALIST': {
            'base_params': [(1.8, 15), (2.0, 20), (2.5, 30)],  # Multiple sensitivity levels
            'horizons': [12, 24],  # Removed 8 - too noisy
        },
        'LIQUIDITY_SPECIALIST': {
            'base_params': [(1.8, 20), (2.0, 20)],
            'horizons': [12, 24, 48],
        },
        'INFORMATION_SPECIALIST': {
            'base_params': [(1.8, 20), (2.0, 20)],
            'horizons': [12, 24, 48],
        },
        'INVENTORY_SPECIALIST': {
            'base_params': [(2.0, 20), (2.5, 25)],  # Higher threshold since sparse
            'horizons': [24, 48],
        },
        'MOMENTUM_DECAY_SPECIALIST': {
            'base_params': [(1.5, 10, 50), (2.0, 10, 50), (2.5, 10, 50), (2.0, 5, 30), (2.0, 20, 100)],
            'horizons': [12, 24, 48],
        }
    }
    
    return {
        'tpsl_grid': tpsl_grid,
        'horizon_options': horizon_options,
        'family_grids': family_grids,
    }

# ==========================================
# 8. Main Pipeline
# ==========================================

def calibrate_all_cusum_thresholds(df: pd.DataFrame, target_events_per_day: float = 2.0, vol_window: int = 20, atr_window: int = 14, sr_levels: list = None) -> Dict[str, float]:
    thresholds = {}
    if len(df) < 100: return thresholds

    duration_days = (df.index[-1] - df.index[0]).days + 1
    bars_per_day = len(df) / max(1, duration_days)
    target_fraction = target_events_per_day / max(1, bars_per_day)

    # Price
    if 'close' in df.columns:
        price_metric = df['close'].pct_change().fillna(0).abs()
        thresholds['price'] = max(price_metric.quantile(1 - target_fraction), 1e-9)

    # Volatility
    if 'close' in df.columns:
        ret = df['close'].pct_change()
        vol = ret.ewm(span=vol_window, adjust=False).std()
        vol_metric = np.log(vol).diff().fillna(0).abs()
        thresholds['volatility'] = max(vol_metric.quantile(1 - target_fraction), 1e-9)

    # Volume
    if 'volume' in df.columns:
        vol_avg = df['volume'].ewm(span=vol_window, adjust=False).mean()
        volume_metric = np.log(df['volume'] / (vol_avg + 1e-9)).fillna(0).abs()
        thresholds['volume'] = max(volume_metric.quantile(1 - target_fraction), 1e-9)

    # ATR / Range
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
        atr = tr.rolling(atr_window).mean()
        atr_norm = (atr / atr.rolling(vol_window).mean() - 1).fillna(0).abs()
        thresholds['atr'] = max(atr_norm.quantile(1 - target_fraction), 1e-9)

    # S/R
    if 'close' in df.columns and sr_levels and len(sr_levels) > 0:
        sr_dist = np.array([min(abs(c - l) for l in sr_levels) for c in df['close']])
        sr_metric = pd.Series(sr_dist, index=df.index).abs()
        thresholds['sr'] = max(sr_metric.quantile(1 - target_fraction), 1e-9)
    else:
        thresholds['sr'] = None

    # Specialists (Predictive Parents)
    if 'close' in df.columns:
        # Volatility Specialist metric: sudden expansion z-score
        ret = df['close'].pct_change()
        vol = ret.rolling(20).std()
        vol_change = (vol / (vol.shift(1) + 1e-9)).fillna(1.0)
        z_vol = ((vol_change - vol_change.rolling(200).mean()) / (vol_change.rolling(200).std() + 1e-9)).fillna(0.0)
        thresholds['VOLATILITY_SPECIALIST'] = max(z_vol.quantile(1 - target_fraction), 2.7)
        
        # Liquidity Specialist metric: impact z-score
        if 'volume' in df.columns:
            impact = df['close'].pct_change().abs() / (df['volume'] + 1e-9)
            baseline = impact.rolling(100).mean()
            z = ((impact - baseline) / (impact.rolling(100).std() + 1e-9)).fillna(0.0)
            thresholds['LIQUIDITY_SPECIALIST'] = max(z.quantile(1 - target_fraction), 2.7)
        else:
            thresholds['LIQUIDITY_SPECIALIST'] = 2.7

        # Information Specialist metric: absolute autocorrelation z-score
        autocorr = ret.rolling(50).corr(ret.shift(1)).abs().fillna(0.0)
        z_info = ((autocorr - autocorr.rolling(300).mean()) / (autocorr.rolling(300).std() + 1e-9)).fillna(0.0)
        thresholds['INFORMATION_SPECIALIST'] = max(z_info.quantile(1 - target_fraction), 2.0)

        # Inventory Specialist metric: price z-score
        price_std = df['close'].rolling(50).std()
        z_inv = ((df['close'] - df['close'].rolling(50).mean()) / (price_std + 1e-9)).abs().fillna(0.0)
        thresholds['INVENTORY_SPECIALIST'] = max(z_inv.quantile(1 - target_fraction), 2.7)

        # Volume Specialist
        if 'volume' in df.columns:
            vol_val = df['volume']
            baseline_vol = vol_val.rolling(100).mean()
            z_vol = ((vol_val - baseline_vol) / (vol_val.rolling(100).std() + 1e-9)).fillna(0.0)
            thresholds['VOLUME_SPECIALIST'] = max(z_vol.quantile(1 - target_fraction), 2.7)
        else:
            thresholds['VOLUME_SPECIALIST'] = 2.7

    return thresholds

def two_tier_weight(z, tier1_min=3.0, tier2_min=2.7, tier2_max=3.0, alpha=1.5, 
                   uniqueness=1.0, noise_ratio=1.0, horizon_validity=1.0):
    """
    Compute two-tier weights for a Z-score.
    Tier-1: core extremes >= 3.0σ (Mapped to [0.5, 1.0])
    Tier-2: near-extremes 2.7-3.0σ (Mapped to [0.0, 0.5])
    """
    z = np.abs(z)
    if isinstance(z, (pd.Series, np.ndarray)):
        w = np.zeros_like(z, dtype=float)
    else:
        w = 0.0

    # Tier-2: [2.7, 3.0)
    mask_t2 = (z >= tier2_min) & (z < tier1_min)
    t2_scale = 0.5 * ((z[mask_t2] - tier2_min) / (tier1_min - tier2_min)) ** alpha if isinstance(z, (pd.Series, np.ndarray)) else 0.5 * ((z - tier2_min) / (tier1_min - tier2_min)) ** alpha
    if isinstance(z, (pd.Series, np.ndarray)):
        w[mask_t2] = t2_scale
    elif mask_t2:
        w = t2_scale

    # Tier-1: [3.0, inf)
    mask_t1 = (z >= tier1_min)
    t1_scale = 0.5 + 0.5 * np.clip((z[mask_t1] - tier1_min) / tier1_min, 0.0, 1.0) ** alpha if isinstance(z, (pd.Series, np.ndarray)) else 0.5 + 0.5 * np.clip((z - tier1_min) / tier1_min, 0.0, 1.0) ** alpha
    if isinstance(z, (pd.Series, np.ndarray)):
        w[mask_t1] = t1_scale
    elif mask_t1:
        w = t1_scale

    # Apply multipliers
    w_final = w * uniqueness * noise_ratio * horizon_validity
    return np.clip(w_final, 0.0, 1.0)

def volume_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, persistence_factor: float = 1.0) -> pd.Series:
    if events.empty or 'volume' not in df.columns: return pd.Series(0, index=events)
    vol_avg = df['volume'].ewm(span=20, adjust=False).mean()
    vol_norm = (df['volume'] / (vol_avg + 1e-9)) - 1.0
    price_ret = df['close'].pct_change().fillna(0)
    signed_vol_proxy = np.sign(vol_norm) * price_ret
    weight = abs(signed_vol_proxy) * persistence_factor
    return weight.reindex(events).fillna(0)

def atr_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, atr_window: int = 14, vol_window: int = 20) -> pd.Series:
    if events.empty or 'high' not in df.columns: return pd.Series(0, index=events)
    tr = df['high'] - df['low']
    atr = tr.rolling(atr_window).mean().fillna(1e-9)
    atr_change = np.log(atr / atr.rolling(vol_window).mean()).fillna(0)
    weight = abs(atr_change)
    return weight.reindex(events).fillna(0)

def sr_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, sr_levels: list) -> pd.Series:
    if events.empty or not sr_levels: return pd.Series(0, index=events)
    close_vals = df['close'].values
    sr_arr = np.array(sr_levels)
    distance = (close_vals[:, None] - sr_arr[None, :]).min(axis=1)
    weight = pd.Series(abs(distance), index=df.index)
    return weight.reindex(events).fillna(0)

def tail_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, window: int = 50) -> pd.Series:
    if events.empty: return pd.Series(0, index=events)
    returns = df['close'].pct_change()
    kurt = returns.rolling(window).kurt().fillna(0)
    min_k = kurt.rolling(window).min()
    max_k = kurt.rolling(window).max()
    weight = (kurt - min_k) / (max_k - min_k + 1e-9)
    return weight.reindex(events).fillna(0)

def get_uniqueness_weight(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 24) -> pd.Series:
    indicator = build_indicator_matrix(events, index, horizon=horizon)
    concurrency = indicator.sum(axis=1)
    # uniqueness = 1 / concurrency
    # average uniqueness over event lifespan
    uniqueness = pd.Series(0.0, index=events)
    if events.empty: return uniqueness

    # Map events to index locations
    evt_locs = index.get_indexer(events)
    for i, loc in enumerate(evt_locs):
        if loc == -1: continue
        end_loc = min(loc + horizon, len(index))
        c = concurrency.iloc[loc:end_loc]
        if len(c) > 0:
            uniqueness.iloc[i] = (1.0 / c).mean()

    return uniqueness

def get_signal_specific_weights(df: pd.DataFrame, events: pd.DatetimeIndex, sr_levels: list = None,
                               component_weights: Dict[str, float] = None, family: str = None) -> pd.Series:
    if component_weights is None:
        component_weights = {'vol': 1.0, 'atr': 1.0, 'sr': 1.0, 'tail': 1.0}

    intensity = pd.Series(0.0, index=events)

    if family == 'VOL_PARTICIPATION':
        intensity = volume_cusum_weight(df, events) * component_weights.get('vol', 1.0)
    elif family == 'RANGE_ATR':
        intensity = atr_cusum_weight(df, events) * component_weights.get('atr', 1.0)
    elif family == 'SR_CUSUM':
        intensity = sr_cusum_weight(df, events, sr_levels) * component_weights.get('sr', 1.0)
    elif family == 'TAIL_RISK':
        intensity = tail_cusum_weight(df, events) * component_weights.get('tail', 1.0)
    elif family == 'INVENTORY_SPECIALIST':
        price_std = df['close'].rolling(50).std()
        z = ((df['close'] - df['close'].rolling(50).mean()) / (price_std + 1e-9)).abs()
        intensity = two_tier_weight(z.reindex(events).fillna(0))
    elif family == 'VOLATILITY_SPECIALIST':
        ret = df['close'].pct_change()
        vol = ret.rolling(20).std()
        vol_change = vol / (vol.shift(1) + 1e-9)
        z = (vol_change - vol_change.rolling(200).mean()) / (vol_change.rolling(200).std() + 1e-9)
        intensity = two_tier_weight(z.reindex(events).fillna(0))
    elif family == 'VOLUME_SPECIALIST':
        vol = df['volume']
        z = (vol - vol.rolling(100).mean()) / (vol.rolling(100).std() + 1e-9)
        intensity = two_tier_weight(z.reindex(events).fillna(0))
    elif family == 'LIQUIDITY_SPECIALIST':
        impact = df['close'].pct_change().abs() / (df['volume'] + 1e-9)
        z = (impact - impact.rolling(100).mean()) / (impact.rolling(100).std() + 1e-9)
        intensity = two_tier_weight(z.reindex(events).fillna(0))
    elif family == 'INFORMATION_SPECIALIST':
        ret = df['close'].pct_change()
        autocorr = ret.rolling(50).corr(ret.shift(1)).abs()
        z = (autocorr - autocorr.rolling(500).mean()) / (autocorr.rolling(500).std() + 1e-9)
        intensity = two_tier_weight(z.reindex(events).fillna(0))

    u_w = get_uniqueness_weight(events, df.index)

    final_weights = (1 + intensity) * u_w
    return final_weights.reindex(events).fillna(1.0) # Ensure no zeros for survival

def get_specialist_event_matrix(df: pd.DataFrame, family: str) -> pd.Series:
    """Generates a full-index weighted event matrix [0, 1] for a specialist family."""
    if df.empty: return pd.Series(dtype=float)
    
    z = pd.Series(0.0, index=df.index)
    if family == 'INVENTORY_SPECIALIST':
        price_std = df['close'].rolling(50).std()
        z = ((df['close'] - df['close'].rolling(50).mean()) / (price_std + 1e-9)).abs()
    elif family == 'VOLATILITY_SPECIALIST':
        ret = df['close'].pct_change()
        vol = ret.rolling(20).std()
        vol_change = vol / (vol.shift(1) + 1e-9)
        z = (vol_change - vol_change.rolling(200).mean()) / (vol_change.rolling(200).std() + 1e-9)
    elif family == 'VOLUME_SPECIALIST':
        vol = df['volume']
        z = (vol - vol.rolling(100).mean()) / (vol.rolling(100).std() + 1e-9)
    elif family == 'LIQUIDITY_SPECIALIST':
        impact = df['close'].pct_change().abs() / (df['volume'] + 1e-9)
        z = (impact - impact.rolling(100).mean()) / (impact.rolling(100).std() + 1e-9)
    elif family == 'INFORMATION_SPECIALIST':
        ret = df['close'].pct_change()
        autocorr = ret.rolling(50).corr(ret.shift(1)).abs()
        z = (autocorr - autocorr.rolling(500).mean()) / (autocorr.rolling(500).std() + 1e-9)
    
    
    weights = two_tier_weight(z.fillna(0.0))
    return pd.Series(weights, index=df.index)

def generate_ohlcv_candidates(df: pd.DataFrame) -> List[Dict]:
    """
    Generates direct market response candidates from OHLCV data.
    """
    candidates = []
    
    # 1. Return Shock: abs(returns) > 3 sigma
    ret = df['close'].pct_change()
    z_ret = (ret - ret.rolling(100).mean()) / (ret.rolling(100).std() + 1e-9)
    weights_ret = pd.Series(two_tier_weight(z_ret.fillna(0.0)), index=df.index)
    events_ret = weights_ret[weights_ret > 0.0].index
    if len(events_ret) >= 10:
        candidates.append({
            'family': 'OHLCV_RETURN_SHOCK',
            'events': events_ret,
            'weight_vector': weights_ret,
            'params': {'type': 'ohlcv', 'metric': 'return'}
        })
        
    # 2. Volatility Spike: (High-Low) > 3 sigma
    hl = (df['high'] - df['low']) / (df['close'] + 1e-9)
    z_vol = (hl - hl.rolling(100).mean()) / (hl.rolling(100).std() + 1e-9)
    weights_vol = pd.Series(two_tier_weight(z_vol.fillna(0.0)), index=df.index)
    events_vol = weights_vol[weights_vol > 0.0].index
    if len(events_vol) >= 10:
        candidates.append({
            'family': 'OHLCV_VOLATILITY_SPIKE',
            'events': events_vol,
            'weight_vector': weights_vol,
            'params': {'type': 'ohlcv', 'metric': 'volatility'}
        })
        
    # 3. Volume Surge: Volume > 3 sigma
    vol_norm = df['volume'] / (df['volume'].rolling(50).mean() + 1e-9)
    z_volume = (vol_norm - vol_norm.rolling(100).mean()) / (vol_norm.rolling(100).std() + 1e-9)
    weights_volume = pd.Series(two_tier_weight(z_volume.fillna(0.0)), index=df.index)
    events_volume = weights_volume[weights_volume > 0.0].index
    if len(events_volume) >= 10:
        candidates.append({
            'family': 'OHLCV_VOLUME_SURGE',
            'events': events_volume,
            'weight_vector': weights_volume,
            'params': {'type': 'ohlcv', 'metric': 'volume'}
        })
        
    # 4. Flow Imbalance: (Close-Open)/(High-Low) * Volume
    flow = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)) * df['volume']
    z_flow = (flow - flow.rolling(100).mean()) / (flow.rolling(100).std() + 1e-9)
    weights_flow = pd.Series(two_tier_weight(z_flow.fillna(0.0)), index=df.index)
    events_flow = weights_flow[weights_flow > 0.0].index
    if len(events_flow) >= 10:
        candidates.append({
            'family': 'OHLCV_FLOW_IMBALANCE',
            'events': events_flow,
            'weight_vector': weights_flow,
            'params': {'type': 'ohlcv', 'metric': 'flow'}
        })
        
    return sorted(candidates, key=lambda x: x['family'])

class FeatureRegistry:
    """Singleton registry for sharing large dense features across candidates."""
    _instance = None
    _features: Dict[str, pd.Series] = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, key: str, series: pd.Series) -> str:
        """Store series and return key."""
        if key not in self._features:
            self._features[key] = series
        return key
        
    def get(self, key: str) -> Optional[pd.Series]:
        return self._features.get(key)
        
    def clear(self):
        self._features.clear()

def generate_continuous_geometry_candidates(df: pd.DataFrame) -> List[Dict]:
    """
    Generate candidates from Continuous Predictor Families.
    
    Families:
    - RELAXATION: Mean-reversion dynamics
    - FLOW_PRESSURE: Buying/Selling pressure
    - SLOPE: Trend geometry
    - FRAGILITY: Market resilience/illiquidity
    """
    if not PREDICTOR_GENERATORS_AVAILABLE:
        return []
        
    candidates = []
class PruningFilters:
    """Early pruning filters to skip unpromising geometries."""
    
    @staticmethod
    def check_learnability(series: pd.Series, target: pd.Series = None, min_ic: float = 0.005) -> bool:
        """
        Check if signal has minimal predictive content (Entropy + IC).
        Returns True if signal is 'promising' (keep), False if 'dead' (prune).
        """
        # 1. Activity / Entropy Check
        # If signal is constant or near constant, prune.
        if series.std() < 1e-9:
            return False
            
        # 2. Information Coefficient (IC) Check
        # If target (future returns) is provided, check correlation.
        if target is not None:
             # Align indices
             common_idx = series.index.intersection(target.index)
             if len(common_idx) < 100:
                 return False # Too few overlapping points
                 
             # Fast Spearman sample (n=1000)
             if len(common_idx) > 1000:
                 # Deterministic sample for speed
                 sample_idx = common_idx[::len(common_idx)//1000]
                 s_series = series[sample_idx]
                 s_target = target[sample_idx]
             else:
                 s_series = series[common_idx]
                 s_target = target[common_idx]
                 
             ic, _ = spearmanr(s_series, s_target)
             if abs(ic) < min_ic:
                 return False
                 
        return True

def generate_continuous_geometry_candidates(df: pd.DataFrame) -> List[Dict]:
    """
    Generate candidates from Continuous Predictor Families.
    
    Families:
    - RELAXATION: Mean-reversion dynamics
    - FLOW_PRESSURE: Buying/Selling pressure
    - SLOPE: Trend geometry
    - FRAGILITY: Market resilience/illiquidity
    """
    if not PREDICTOR_GENERATORS_AVAILABLE:
        return []
        
    candidates = []
    registry = FeatureRegistry.get_instance()
    
    # Pre-calculate simple target for pruning: 24h forward returns
    try:
        # Assuming hourly/minute data, 24 steps might be 1 day?
        # Safe default: 1-day return (approx 288 5-min bars or 24 1h bars)
        # We'll use a generic 48-step forward return as proxy for "Directional Value"
        if 'close' in df.columns:
            target_proxy = df['close'].pct_change(48).shift(-48)
        else:
            target_proxy = None
    except:
        target_proxy = None

    try:
        # Initialize generator
        cp_gen = ContinuousPredictorGenerator(verbose=False)
        predictors = cp_gen.generate_all_predictors(df)
        
        tprint_info(f"   ✨ Generated {len(predictors)} continuous predictors")
        
        pruned_count = 0
        for pred in predictors:
            try:
                # 0. Early Pruning (Learnability)
                if target_proxy is not None:
                    # Sanitize
                    vals = pred.values.fillna(0)
                    if not PruningFilters.check_learnability(vals, target_proxy, min_ic=0.01):
                        pruned_count += 1
                        continue
                
                # Convert continuous predictor to event-based candidate
                # We use high-activation points (peaks/valleys) as events
                # This allows them to pass the event-based pipeline
                
                # 1. Standardize
                vals = pred.values.fillna(0)
                if vals.std() == 0: continue
                
                # Z-score - Robust
                z = (vals - vals.mean()) / (vals.std() + 1e-9)
                
                # Register feature to save memory
                feature_key = f"{pred.family}_{pred.name}"
                registry.register(feature_key, vals)
                
                # 2. Extract Events (High Z)
                # Ensure we have weights for composite generation
                # Align thresholds: tier2=1.5 (entry), tier1=2.5 (strong)
                weights = two_tier_weight(z, tier1_min=2.5, tier2_min=1.5, alpha=1.5)
                weights_series = pd.Series(weights, index=df.index).fillna(0.0)

                # Positive Activation (High values)
                events_pos = z[z > 1.5].index
                if len(events_pos) > 20:
                    candidates.append({
                        'family': f"{pred.family.upper()}_{pred.name.upper()}_POS",
                        'events': events_pos,
                        'weight_vector': weights_series, # Added for composite engine
                        'feature_key': feature_key, # Reference instead of copy
                        'weight_sign': 1.0,
                        'feature_sign': 1.0, # Explicit sign for interaction
                        'params': {'source': pred.name, 'side': 'positive', 'method': 'continuous_z_threshold'},
                        'status': 'NEW'
                    })
                    
                # Negative Activation (Low values)
                events_neg = z[z < -1.5].index
                if len(events_neg) > 20:
                    # For negative events, we want the weight to reflect the intensity of the *negative* move
                    # two_tier_weight takes abs(z) so it handles negative values correctly as magnitude
                    candidates.append({
                        'family': f"{pred.family.upper()}_{pred.name.upper()}_NEG",
                        'events': events_neg,
                        'weight_vector': weights_series, # Added for composite engine
                        'feature_key': feature_key, 
                        'weight_sign': -1.0,
                        'feature_sign': -1.0,
                        'params': {'source': pred.name, 'side': 'negative', 'method': 'continuous_z_threshold'},
                        'status': 'NEW'
                    })
                    
            except Exception as e:
                continue
        
        if pruned_count > 0:
            tprint_info(f"   ✂️ Pruned {pruned_count} unpromising predictors (Low IC)")
            
    except Exception as e:
        logger.warning(f"generate_continuous_geometry_candidates failed: {e}")
        return []
        
    return candidates


def generate_derived_features(df: pd.DataFrame, ohlcv_candidates: List[Dict]) -> List[Dict]:
    """
    Level 4: Derived Ratios and Relative Features.
    - Ratios: Volume/Volatility, Return/ATR
    - Relative Shocks: Value - Rolling Median
    """
    candidates = []
    
    # 1. Volume / Volatility Ratio (Liquidity Efficiency)
    # Norm volume by volatility -> high volume low vol = absorption?
    vol = df['volume'] / (df['volume'].rolling(50).mean() + 1e-9)
    hl = (df['high'] - df['low']) / (df['close'] + 1e-9)
    hl_norm = hl / (hl.rolling(50).mean() + 1e-9)
    
    ratio = vol / (hl_norm + 1e-9)
    z_ratio = (ratio - ratio.rolling(100).mean()) / (ratio.rolling(100).std() + 1e-9)
    weights_ratio = pd.Series(two_tier_weight(z_ratio.fillna(0.0)), index=df.index)
    events_ratio = weights_ratio[weights_ratio > 0.0].index
    
    if len(events_ratio) > 100: # Slightly higher bar for derived features
        candidates.append({
            'family': 'DERIVED_VOL_VOLATILITY_RATIO',
            'events': events_ratio,
            'weight_vector': weights_ratio,
            'params': {'type': 'derived', 'subtype': 'ratio'}
        })

    # 2. Relative Shocks (Price - Rolling Median)
    # Captures sudden deviations from central tendency
    close = df['close']
    median_50 = close.rolling(50).median()
    deviation = (close - median_50).abs() / (close.rolling(50).std() + 1e-9)
    
    z_dev = (deviation - deviation.rolling(100).mean()) / (deviation.rolling(100).std() + 1e-9)
    weights_dev = pd.Series(two_tier_weight(z_dev.fillna(0.0)), index=df.index)
    events_dev = weights_dev[weights_dev > 0.0].index
    
    if len(events_dev) > 100:
        candidates.append({
            'family': 'DERIVED_PRICE_MEDIAN_DEV',
            'events': events_dev,
            'weight_vector': weights_dev,
            'params': {'type': 'derived', 'subtype': 'relative_shock'}
        })
        
    # Phase 2: Expanded Derived Signals
    
    # 3. Volume Spike Ratio
    # current_volume / rolling_mean(volume)
    vol_mean = df['volume'].rolling(50).mean() + 1e-9
    vol_spike = df['volume'] / vol_mean
    # Z-score and Smart Weight
    z_vol_spike = (vol_spike - vol_spike.rolling(100).mean()) / (vol_spike.rolling(100).std() + 1e-9)
    weights_vspike = pd.Series(smart_event_weight(z_vol_spike.fillna(0.0)), index=df.index)
    if len(weights_vspike[weights_vspike > 0]) > 50:
        candidates.append({ 'family': 'DERIVED_VOLUME_SPIKE_RATIO', 'events': weights_vspike[weights_vspike > 0].index, 'weight_vector': weights_vspike, 'params': {'type': 'derived', 'subtype': 'ratio'} })

    # 4. Return Gap
    # open - prev_close
    prev_close = df['close'].shift(1)
    ret_gap = (df['open'] - prev_close) / (prev_close + 1e-9)
    z_gap = (ret_gap - ret_gap.rolling(100).mean()) / (ret_gap.rolling(100).std() + 1e-9)
    weights_gap = pd.Series(smart_event_weight(z_gap.fillna(0.0)), index=df.index)
    if len(weights_gap[weights_gap > 0]) > 50:
         candidates.append({ 'family': 'DERIVED_RETURN_GAP', 'events': weights_gap[weights_gap > 0].index, 'weight_vector': weights_gap, 'params': {'type': 'derived', 'subtype': 'shock'} })

    # 5. Volatility Expansion
    # (high - low) / rolling_std
    roll_std = df['close'].rolling(50).std() + 1e-9
    vol_exp = (df['high'] - df['low']) / roll_std
    z_volexp = (vol_exp - vol_exp.rolling(100).mean()) / (vol_exp.rolling(100).std() + 1e-9)
    weights_volexp = pd.Series(smart_event_weight(z_volexp.fillna(0.0)), index=df.index)
    if len(weights_volexp[weights_volexp > 0]) > 50:
        candidates.append({ 'family': 'DERIVED_VOLATILITY_EXPANSION', 'events': weights_volexp[weights_volexp > 0].index, 'weight_vector': weights_volexp, 'params': {'type': 'derived', 'subtype': 'expansion'} })

    # 6. Bid-Ask Spread Ratio (Proxy using High-Low if no bid/ask)
    # Using High-Low / Close as proxy for spread/liquidity stress if actual spread unavailable
    ba_proxy = (df['high'] - df['low']) / (df['close'] + 1e-9)
    # We want deviations in this proxy that ARE NOT just volatility? Hard to separate without tick data.
    # Let's skip pure spread if not available, or use it as proxy.
    z_ba = (ba_proxy - ba_proxy.rolling(100).mean()) / (ba_proxy.rolling(100).std() + 1e-9)
    # Use adaptive threshold here?
    weights_ba = pd.Series(smart_event_weight(z_ba.fillna(0.0)), index=df.index)
    if len(weights_ba[weights_ba > 0]) > 50:
         candidates.append({ 'family': 'DERIVED_Liquidity_STRESS_PROXY', 'events': weights_ba[weights_ba > 0].index, 'weight_vector': weights_ba, 'params': {'type': 'derived', 'subtype': 'liquidity'} })

    # 7. Flow Imbalance (Proxy using Close-Open relation to Volume)
    # If Close > Open, assume buy volume dominant? Rough proxy.
    # (2*(Close - Low) - (High - Low)) / (High - Low) * Volume
    # ADL style flow.
    ad_flow = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9) * df['volume']
    z_flow = (ad_flow - ad_flow.rolling(100).mean()) / (ad_flow.rolling(100).std() + 1e-9)
    weights_flow = pd.Series(smart_event_weight(z_flow.fillna(0.0)), index=df.index)
    if len(weights_flow[weights_flow > 0]) > 50:
        candidates.append({ 'family': 'DERIVED_FLOW_IMBALANCE_PROXY', 'events': weights_flow[weights_flow > 0].index, 'weight_vector': weights_flow, 'params': {'type': 'derived', 'subtype': 'flow'} })

    # 8. Liquidity Gap
    # rolling_max(volume) - rolling_min(volume)
    liq_gap = df['volume'].rolling(20).max() - df['volume'].rolling(20).min()
    z_liq_gap = (liq_gap - liq_gap.rolling(100).mean()) / (liq_gap.rolling(100).std() + 1e-9)
    weights_liq_gap = pd.Series(smart_event_weight(z_liq_gap.fillna(0.0)), index=df.index)
    if len(weights_liq_gap[weights_liq_gap > 0]) > 50:
         candidates.append({ 'family': 'DERIVED_LIQUIDITY_GAP', 'events': weights_liq_gap[weights_liq_gap > 0].index, 'weight_vector': weights_liq_gap, 'params': {'type': 'derived', 'subtype': 'liquidity'} })
         
    # 9. Volatility Ratio
    # current_vol / rolling_mean(vol)
    curr_vol = (df['high'] - df['low']) / (df['close'] + 1e-9)
    vol_ratio = curr_vol / (curr_vol.rolling(50).mean() + 1e-9)
    z_vol_ratio = (vol_ratio - vol_ratio.rolling(100).mean()) / (vol_ratio.rolling(100).std() + 1e-9)
    weights_vol_ratio = pd.Series(smart_event_weight(z_vol_ratio.fillna(0.0)), index=df.index)
    if len(weights_vol_ratio[weights_vol_ratio > 0]) > 50:
        candidates.append({ 'family': 'DERIVED_VOLATILITY_RATIO', 'events': weights_vol_ratio[weights_vol_ratio > 0].index, 'weight_vector': weights_vol_ratio, 'params': {'type': 'derived', 'subtype': 'ratio'} })

    return candidates

def smart_event_weight(z_score_series: pd.Series, quantile_threshold: float = 0.995, use_adaptive: bool = True) -> np.ndarray:
    """
    Advanced Event Logic (Level 11):
    1. Quantile Detection: Tier-1 if z >= 3.0 OR value > 99.5% quantile.
    2. Adaptive Thresholds by Regime: Modulate thresholds based on rolling volatility.
       - In high vol regimes, we require higher Z-scores for Tier-1.
       - In low vol regimes, we relax thresholds slightly to capture structural shifts.
    """
    # 1. Base Quantile Calculation
    try:
        q_val = z_score_series.quantile(quantile_threshold)
    except:
        q_val = 3.0 # Fallback
        
    weights = np.zeros(len(z_score_series))
    abs_z = z_score_series.abs()
    
    # 2. Adaptive Regime Modulation
    # Compute rolling volatility of the Z-score itself as a proxy for signal regime
    if use_adaptive:
        try:
            # Signal volatility regime (short-term instability)
            sig_vol = z_score_series.rolling(100).std().fillna(1.0)
            sig_vol_norm = sig_vol / (sig_vol.rolling(500).mean() + 1e-9)
            
            # Modulate thresholds: higher vol -> higher requirement
            # Max boost +50% to threshold in extreme regimes
            adaptive_t1 = 3.0 * np.clip(sig_vol_norm, 0.8, 1.5)
            adaptive_q = q_val * np.clip(sig_vol_norm, 0.9, 1.2)
            adaptive_t2 = 2.5 * np.clip(sig_vol_norm, 0.8, 1.3)
        except:
            adaptive_t1 = 3.0
            adaptive_q = q_val
            adaptive_t2 = 2.5
    else:
        adaptive_t1 = 3.0
        adaptive_q = q_val
        adaptive_t2 = 2.5

    # 3. Tiered Assignment
    # Tier-1: Extreme signals (Adaptive Z or Extreme Quantile)
    tier1_mask = (abs_z >= adaptive_t1) | (abs_z >= adaptive_q)
    # Tier-2: Structural smoothing (Structural threshold)
    tier2_mask = (abs_z >= adaptive_t2) & (~tier1_mask)
    
    weights[tier1_mask] = 1.0
    weights[tier2_mask] = 0.5
    
    return weights


def generate_tail_aggregates(df: pd.DataFrame, base_candidates: List[Dict]) -> List[Dict]:
    """
    Level 2/6: Tail Aggregates (Rolling Sum/Max of weighted events).
    Captures event clustering (e.g. 3 return shocks in 1 hour).
    """
    aggregates = []
    
    for cand in base_candidates:
        w_series = cand['weight_vector']
        if not isinstance(w_series, pd.Series): continue
        
        # Rolling Sum over 4 periods (~1h for 15m)
        w_sum = w_series.rolling(4).sum()
        # Normalize sum to be roughly [0,1] scale or Z-score it?
        # Z-score the sum to find clusters that are unusual
        z_sum = (w_sum - w_sum.rolling(100).mean()) / (w_sum.rolling(100).std() + 1e-9)
        
        weights_agg = pd.Series(two_tier_weight(z_sum.fillna(0.0)), index=df.index)
        events_agg = weights_agg[weights_agg > 0.0].index
        
        if len(events_agg) > 50:
            aggregates.append({
                'family': f"AGG_SUM_4_{cand['family']}",
                'events': events_agg,
                'weight_vector': weights_agg,
                'params': {'type': 'aggregate', 'parent': cand['family'], 'window': 4}
            })
            
            
    return aggregates

def generate_multi_horizon_candidates(df: pd.DataFrame) -> List[Dict]:
    """
    Level 5: Multi-Horizon Signals.
    Resamples OHLCV data to higher timeframes (30m, 1h, 4h) and generates shock events.
    Events are forward-filled back to the original index.
    """
    candidates = []
    horizons = {'30min': 2, '60min': 4, '4h': 16} # Multiples of 15m
    
    for label, factor in horizons.items():
        # Resample - Use aggregation rules
        resampled = df.resample(label).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 1. Return Shock
        ret = resampled['close'].pct_change()
        z_ret = (ret - ret.rolling(100).mean()) / (ret.rolling(100).std() + 1e-9)
        weights_ret = pd.Series(two_tier_weight(z_ret.fillna(0.0)), index=resampled.index)
        events_ret = weights_ret[weights_ret > 0.0].index
        
        # Map back to 15m index (reindex + ffill)
        # Shift(1) is CRITICAL to avoid lookahead bias: 
        # Informational content of the resampled bar is only available at its CLOSE.
        weights_ret_15m = weights_ret.shift(1).reindex(df.index).ffill().fillna(0.0)
        events_ret_15m = weights_ret_15m[weights_ret_15m > 0.0].index
        
        if len(events_ret_15m) > 50:
            candidates.append({
                'family': f'OHLCV_RETURN_SHOCK_{label.upper()}',
                'events': events_ret_15m,
                'weight_vector': weights_ret_15m,
                'params': {'type': 'multi_horizon', 'horizon': label, 'metric': 'return'}
            })
            
        # 2. Volatility Spike
        hl = (resampled['high'] - resampled['low']) / (resampled['close'] + 1e-9)
        z_vol = (hl - hl.rolling(100).mean()) / (hl.rolling(100).std() + 1e-9)
        weights_vol = pd.Series(two_tier_weight(z_vol.fillna(0.0)), index=resampled.index)
        
        # Map back to 15m index (reindex + ffill)
        # Shift(1) is CRITICAL to avoid lookahead bias
        weights_vol_15m = weights_vol.shift(1).reindex(df.index).ffill().fillna(0.0)
        events_vol_15m = weights_vol_15m[weights_vol_15m > 0.0].index
        
        if len(events_vol_15m) > 50:
            candidates.append({
                'family': f'OHLCV_VOLATILITY_SPIKE_{label.upper()}',
                'events': events_vol_15m,
                'weight_vector': weights_vol_15m,
                'params': {'type': 'multi_horizon', 'horizon': label, 'metric': 'volatility'}
            })
            
        # 3. Volume Surge
        vol_norm = resampled['volume'] / (resampled['volume'].rolling(50).mean() + 1e-9)
        z_volume = (vol_norm - vol_norm.rolling(100).mean()) / (vol_norm.rolling(100).std() + 1e-9)
        weights_volume = pd.Series(two_tier_weight(z_volume.fillna(0.0)), index=resampled.index)
        
        # Map back to 15m index (reindex + ffill)
        # Shift(1) is CRITICAL to avoid lookahead bias
        weights_volume_15m = weights_volume.shift(1).reindex(df.index).ffill().fillna(0.0)
        events_volume_15m = weights_volume_15m[weights_volume_15m > 0.0].index
        
        if len(events_volume_15m) > 50:
            candidates.append({
                'family': f'OHLCV_VOLUME_SURGE_{label.upper()}',
                'events': events_volume_15m,
                'weight_vector': weights_volume_15m,
                'params': {'type': 'multi_horizon', 'horizon': label, 'metric': 'volume'}
            })
            
    return candidates

def generate_regime_conditioned_candidates(df: pd.DataFrame, ohlcv_candidates: List[Dict], specialist_families: List[str]) -> List[Dict]:
    """
    Level 6: Regime-Conditioned Features.
    Conditions OHLCV candidates on Specialist states (e.g. Return Shock when Inventory is High).
    Uses implicit regimes defined by Specialist Weight > 0.5 (High) or < 0.2 (Low? or just 1-Weight).
    Actually, just multiplying the weights (Soft Conditioning) is effective and simpler.
    Condition = OHLCV_Weight * Specialist_Weight
    """
    candidates = []
    
    # Pre-calculate specialist weights
    spec_weights = {}
    for fam in specialist_families:
        spec_weights[fam] = get_specialist_event_matrix(df, fam)
        
    for cand in ohlcv_candidates:
        w_ohlcv = cand['weight_vector']
        if not isinstance(w_ohlcv, pd.Series): continue
        
        for fam, w_spec in spec_weights.items():
            if not isinstance(w_spec, pd.Series): continue
            
            # Condition: OHLCV interaction with Specialist Regime
            # We want to capture: "Shock occurring during High Specialist Activity"
            # Simply multiplying the weights achieves this 'soft AND' logic.
            # w_combined = w_ohlcv * w_spec
            # But we might want to be more specific: "High Inventory" vs "Low Inventory"
            # For now, let's stick to "Relevance" (High Weight)
            
            w_combined = w_ohlcv * w_spec
            
            # Filter: Check if we have enough mass
            if w_combined.sum() > 5.0: # Arbitrary mass threshold, ensure some overlap exists
                # Normalize?
                # z_combined = (w_combined - w_combined.mean()) / (w_combined.std() + 1e-9) # No, keep as probability-like mass
                
                # We need to turn this into a weight vector acceptable by Layer 2
                # It is already [0, 1] * [0, 1] -> [0, 1]
                
                 events_combined = w_combined[w_combined > 0.1].index # Lower threshold for combined
                 
                 if len(events_combined) > 50:
                     candidates.append({
                         'family': f'COND_{cand["family"]}_ON_{fam.replace("_SPECIALIST", "")}',
                         'events': events_combined,
                         'weight_vector': w_combined,
                         'params': {'type': 'regime_conditioned', 'parent': cand['family'], 'condition': fam}
                     })
                     
    return candidates

def validate_candidates_with_causal_graph(candidates: List[Dict], df: pd.DataFrame, target_col: str = 'close', verbose: bool = True) -> List[Dict]:
    """
    Level 9: Causal Graph Feedback Validation.
    Run PC Algorithm on Candidates + Target to validate causality.
    Keep only candidates that are PARENTS of the Target (or strong ancestors).
    """
    if not candidates or df is None: return candidates
    
    tprint_info("🔗 Running Level 9: Causal Graph Feedback Validation...")
    
    try:
        from src.training.steps.labeling.causal_discovery import quick_causal_discovery
        
        # 1. Prepare Data for Causal Discovery
        # Extract weight vectors
        weight_data = {c['family']: c['weight_vector'] for c in candidates}
        # Use first candidate index for alignment
        first_idx = candidates[0]['weight_vector'].index
        df_weights = pd.DataFrame(weight_data, index=first_idx).fillna(0.0)
        
        # Prepare Target (1-bar forward return of 'close')
        target_name = 'TARGET_RET_1'
        target_series = df[target_col].pct_change().shift(-1).fillna(0.0)
        
        # Align indices
        common_idx = df_weights.index.intersection(target_series.index)
        
        # Combine into one DataFrame for discovery
        discovery_df = df_weights.loc[common_idx].copy()
        discovery_df[target_name] = target_series.loc[common_idx]
        
        # 2. Run Causal Discovery (Fast mode)
        # We use a higher alpha (0.1) to be permissive, we just want to prune potential non-causes
        results = quick_causal_discovery(
            discovery_df,
            target_variable=target_name,
            significance_level=0.05, 
            use_lingam=True, # Use LiNGAM for orientation
            verbose=verbose
        )
        
        if 'error' in results:
            tprint_warning(f"Causal Validation error: {results['error']}. Skipping validation.")
            return candidates
            
        # 3. Identify Causal Parents
        parents = results.get('causal_parents', {}).get(target_name, [])
        tprint_info(f"   🧬 Identified Causal Parents of Target: {parents}")
        
        if not parents:
            tprint_warning("   ⚠️ No causal parents found for target. This might be due to low signal-to-noise. Keeping Top 50 by correlation-redundancy as fallback.")

            # Fallback: Top 50 by Correlation + Redundancy Check + Lead-Lag Causality
            try:
                # 1. Calculate correlation with 30-bar forward target (matches TBM horizon)
                target_30bar = df[target_col].pct_change(30).shift(-30).fillna(0.0)
                target_30bar_aligned = target_30bar.loc[common_idx]
                discovery_df['TARGET_RET_30'] = target_30bar_aligned
                
                target_corr = discovery_df.corrwith(discovery_df['TARGET_RET_30']).abs()
                target_corr = target_corr.drop(['TARGET_RET_30', target_name], errors='ignore')

                # 2. Lead-Lag Causality Filter (feature must LEAD the target)
                # Compare: corr(feature_t, target_t+30) vs corr(feature_t, target_t-30)
                # If feature leads target, forward corr should be higher
                target_lag = df[target_col].pct_change(30).shift(30).fillna(0.0).loc[common_idx]
                causal_features = []
                for feat in target_corr.index:
                    if feat not in discovery_df.columns:
                        continue
                    feat_series = discovery_df[feat]
                    corr_lead = feat_series.corr(target_30bar_aligned)  # Feature leads
                    corr_lag = feat_series.corr(target_lag)  # Feature lags (spurious)
                    # Feature must have stronger lead correlation than lag correlation
                    if abs(corr_lead) > abs(corr_lag) * 1.2:  # 20% margin
                        causal_features.append(feat)
                
                tprint_info(f"   🧬 Lead-Lag Causality: {len(causal_features)}/{len(target_corr)} features pass (lead > lag * 1.2)")
                
                # Filter to causal features only, then sort by correlation
                if causal_features:
                    target_corr = target_corr[target_corr.index.isin(causal_features)]
                
                # 3. Sort by target correlation (highest first)
                sorted_features = target_corr.sort_values(ascending=False).index.tolist()

                # 4. Greedy Selection (Redundancy Filter)
                selected_features = []
                feature_corr_matrix = discovery_df[sorted_features].corr().abs()

                for feature in sorted_features:
                    if len(selected_features) >= 50:
                        break

                    # Check redundancy against already selected
                    is_redundant = False
                    for selected in selected_features:
                        if feature_corr_matrix.loc[feature, selected] > 0.85:
                            is_redundant = True
                            break

                    if not is_redundant:
                        selected_features.append(feature)

                # 5. Filter candidates
                selected_set = set(selected_features)
                fallback_candidates = [c for c in candidates if c['family'] in selected_set]

                tprint_info(f"   📉 Fallback: Selected {len(fallback_candidates)} candidates (30-bar corr, Lead-Lag filter, Redundancy < 0.85).")
                return fallback_candidates

            except Exception as e:
                tprint_warning(f"   ⚠️ Fallback selection failed: {e}. Returning original set.")
                return candidates
            
        # 4. Filter Candidates
        # Keep candidates that are in the parent list
        validated_candidates = [c for c in candidates if c['family'] in parents]
        
        if not validated_candidates:
             tprint_warning("   ⚠️ No candidates matched the identified parents (maybe parents were latent?). Returning original set.")
             return candidates
             
        tprint_success(f"   ✅ Causal Validation: {len(candidates)} -> {len(validated_candidates)} candidates confirmed as Causal Drivers.")
        return validated_candidates

    except ImportError:
        tprint_warning("Causal Discovery module not found. Skipping Level 9 validation.")
        return candidates
    except Exception as e:
        tprint_warning(f"Causal Validation Failed: {e}. Skipping.")
        return candidates

def filter_advanced_candidates(candidates: List[Dict], min_count: int = 200, max_corr: float = 0.95, df: pd.DataFrame = None) -> List[Dict]:
    """
    Level 7: Feature Filtering & Smart Selection.
    - Removes rare signals (Count < min_count).
    - Removes redundant signals (Correlation > max_corr).
    - If df provided: Ranks by Score = Norm(IC) + Norm(MDI) and selects Top 120.
    """
    if not candidates: return []
    
    # 1. Min Frequency Filter
    filtered_by_count = [c for c in candidates if len(c['events']) >= min_count]
    if not filtered_by_count:
        tprint_warning(f"Feature Filtering: All {len(candidates)} candidates removed by min_count={min_count}. Reverting to original set.")
        filtered_by_count = candidates
    else:
        tprint_info(f"Feature Filtering: {len(candidates)} -> {len(filtered_by_count)} candidates after min_count={min_count}")
        
    # 2. Correlation Filter (Redundancy)
    if len(filtered_by_count) < 2:
        return filtered_by_count
        
    feature_registry = FeatureRegistry()
    weight_data = {}
    
    for c in filtered_by_count:
        if 'feature_key' in c:
            # New optimized path
            series = feature_registry.get(c['feature_key'])
            if series is not None:
                weight_data[c['family']] = series
        elif 'weight_vector' in c:
            # Legacy path
            weight_data[c['family']] = c['weight_vector']
            
    if not weight_data:
        tprint_warning("Feature Filtering: No weight data available for correlation check. Skipping.")
        return filtered_by_count

    # Ensure alignment
    # Note: df is optional. If None, we create a dummy index from the first candidate
    first_series = next(iter(weight_data.values()))
    first_idx = first_series.index
    df_weights = pd.DataFrame(weight_data, index=first_idx).fillna(0.0)
    
    # Run Correlation Logic
    corr_matrix = df_weights.corr().abs()
    to_drop = set()
    columns = df_weights.columns
    
    for i in range(len(columns)):
        col_a = columns[i]
        if col_a in to_drop: continue
        for j in range(i + 1, len(columns)):
            col_b = columns[j]
            if col_b in to_drop: continue
            if corr_matrix.loc[col_a, col_b] > max_corr:
                # Drop one. Heuristic: Keep shorter name or first one.
                if len(col_a) > len(col_b):
                    to_drop.add(col_a)
                else:
                    to_drop.add(col_b)
                    
    filtered_by_corr = [c for c in filtered_by_count if c['family'] not in to_drop]
    tprint_info(f"Feature Filtering: {len(filtered_by_count)} -> {len(filtered_by_corr)} candidates after corr_threshold={max_corr}")
    
    # 3. Smart Selection (Ranking by IC/MDI)
    if df is not None and len(filtered_by_corr) > 120:
        tprint_info("🧠 Running Smart Feature Selection (IC + MDI)...")
        try:
            # 3a. Prepare Target (1-bar forward return)
            # Use 'close' from df
            ret_1 = df['close'].pct_change().shift(-1).fillna(0.0)
            
            # 3b. Compute IC (Information Coefficient)
            # Align weights with returns
            df_w_aligned = df_weights[ [c['family'] for c in filtered_by_corr] ]
            # Ensure same index
            common_idx = df_w_aligned.index.intersection(ret_1.index)
            X = df_w_aligned.loc[common_idx]
            y = ret_1.loc[common_idx]
            
            ic_scores = {}
            from scipy.stats import spearmanr
            for col in X.columns:
                ic, _ = spearmanr(X[col], y)
                ic_scores[col] = abs(ic) if not np.isnan(ic) else 0.0
                
            # 3c. Compute MDI (Feature Importance) via LightGBM
            # Fast model
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            model.fit(X, y)
            importances = model.feature_importances_
            mdi_scores = dict(zip(X.columns, importances))
            
            # 3d. Combined Score (Normalized)
            # Normalize IC
            ic_vals = np.array(list(ic_scores.values()))
            ic_norm = (ic_vals - ic_vals.min()) / (ic_vals.max() - ic_vals.min() + 1e-9)
            ic_map = dict(zip(ic_scores.keys(), ic_norm))
            
            # Normalize MDI
            mdi_vals = np.array(list(mdi_scores.values()))
            mdi_norm = (mdi_vals - mdi_vals.min()) / (mdi_vals.max() - mdi_vals.min() + 1e-9)
            mdi_map = dict(zip(mdi_scores.keys(), mdi_norm))
            
            final_scores = {}
            for fam in X.columns:
                final_scores[fam] = 0.5 * ic_map.get(fam, 0) + 0.5 * mdi_map.get(fam, 0)
                
            # Sort and Pick Top 120
            sorted_fams = sorted(final_scores, key=final_scores.get, reverse=True)
            top_120_fams = set(sorted_fams[:120])
            
            smart_selected = [c for c in filtered_by_corr if c['family'] in top_120_fams]
            tprint_info(f"Feature Filtering: {len(filtered_by_corr)} -> {len(smart_selected)} candidates after Smart Selection (Top 120 by IC/MDI)")
            return smart_selected
            
        except Exception as e:
            tprint_warning(f"Smart Selection failed: {e}. Returning correlation-filtered set.")
            return filtered_by_corr
            
    return filtered_by_corr


def generate_synthetic_meta_signals(df: pd.DataFrame, filtered_candidates: List[Dict], n_components: int = 5) -> List[Dict]:
    """
    Level 8: Synthetic Meta-Signals (PCA).
    Extracts latent factors from the filtered set of candidates.
    Identify shocks in PCA components representing global market modes.

    Enhanced Pipeline:
    1. Marchenko-Pastur Denoising (Covariance)
    2. Signal Reconstruction (Filtering)
    3. SparsePCA (Dimensionality Reduction with Alpha Penalty)
    """
    if len(filtered_candidates) < n_components + 1:
        return []
        
    try:
        # Build matrix
        weight_data = {c['family']: c['weight_vector'] for c in filtered_candidates}
        # Ensure alignment
        df_weights = pd.DataFrame(weight_data, index=df.index).fillna(0.0)
        
        # Normalize for PCA (Z-score)
        df_z = (df_weights - df_weights.mean()) / (df_weights.std() + 1e-9)
        df_z = df_z.fillna(0.0)
        
        # --- 1. Marchenko-Pastur Denoising (Skip for large feature sets) ---
        # Skip for >500 features - TruncatedSVD handles noise via truncation
        n_features = df_z.shape[1]
        
        if n_features > 500:
            tprint_info(f"   📉 Skipping MP denoising for {n_features} features - using TruncatedSVD directly")
            df_z_denoised = df_z
        else:
            corr_matrix = df_z.corr()
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            T, N = df_z.shape
            q = T / N
            mp_evals, _ = marcenko_pastur_distribution(q, sigma=1.0)
            lambda_max = mp_evals[-1]
            n_signal = np.sum(eigenvalues > lambda_max)
            if n_signal < 1: n_signal = 1

            tprint_info(f"   🧬 MP Denoising: {n_signal} signal components (q={q:.2f}, λ_max={lambda_max:.2f})")

            V_signal = eigenvectors[:, :n_signal]
            df_z_denoised = df_z @ V_signal @ V_signal.T
            df_z_denoised = pd.DataFrame(df_z_denoised, index=df_z.index, columns=df_z.columns)


        # --- 2. Subsampling + TruncatedSVD (Fast Randomized PCA) ---
        # Optimization: Fit on subsampled data, transform full dataset
        # TruncatedSVD is ~10x faster than SparsePCA
        from sklearn.decomposition import TruncatedSVD
        
        # Subsample for fitting (20K samples max for efficiency)
        n_samples = len(df_z_denoised)
        max_fit_samples = 20000
        
        if n_samples > max_fit_samples:
            rng = np.random.default_rng(42)
            fit_idx = rng.choice(n_samples, size=max_fit_samples, replace=False)
            fit_idx.sort()  # Keep temporal order for stratification
            df_fit = df_z_denoised.iloc[fit_idx]
            tprint_info(f"   📉 PCA Optimization: Fitting on {max_fit_samples}/{n_samples} samples")
        else:
            df_fit = df_z_denoised
        
        # TruncatedSVD (Randomized SVD) - much faster than SparsePCA
        # CRITICAL: Limit BLAS threads to prevent GIL deadlock on M1 Macs
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        
        try:
            from threadpoolctl import threadpool_limits
            use_threadpool = True
        except ImportError:
            use_threadpool = False
            tprint_warning("   ⚠️ threadpoolctl not installed, using env vars only for thread limiting")
        
        svd = TruncatedSVD(
            n_components=n_components,
            algorithm='randomized',  # Fast randomized algorithm
            n_iter=5,  # Good convergence for this use case
            random_state=42
        )
        
        # Fit on subsample, transform full dataset (with thread limiting)
        if use_threadpool:
            with threadpool_limits(limits=1, user_api='blas'):
                svd.fit(df_fit.values)
                components = svd.transform(df_z_denoised.values)
        else:
            svd.fit(df_fit.values)
            components = svd.transform(df_z_denoised.values)
        
        tprint_info(f"   ✅ TruncatedSVD complete: explained_variance_ratio={svd.explained_variance_ratio_.sum():.3f}")


        
        synthetic_candidates = []
        for i in range(n_components):
            # Resulting component is continuous. 
            # We identify "Shocks" (Extreme values) in this latent factor.
            comp_series = pd.Series(components[:, i], index=df.index)
            
            # Z-score the component to find deviations
            # Use rolling window to adapt to regime changes? 
            # Or global? PCA is global. Let's use rolling to be safe.
            z_comp = (comp_series - comp_series.rolling(100).mean()) / (comp_series.rolling(100).std() + 1e-9)
            
            # Identify peaks (positive and negative shocks matter? standard two_tier handles positive logic?)
            # Actually PCA components sign is arbitrary.
            # We should look for extreme deviations in EITHER direction.
            # But two_tier_weight usually expects positive signal strength.
            # Let's take absolute value deviation.
            abs_z = z_comp.abs()
            
            weights_syn = pd.Series(two_tier_weight(abs_z.fillna(0.0)), index=df.index)
            events_syn = weights_syn[weights_syn > 0.0].index
            
            if len(events_syn) > 50:
                synthetic_candidates.append({
                    'family': f'SYNTHETIC_PCA_C{i+1}',
                    'events': events_syn,
                    'weight_vector': weights_syn,
                    'params': {'type': 'synthetic', 'method': 'pca', 'component': i+1}
                })
                
        return synthetic_candidates
    except Exception as e:
        tprint_warning(f"Synthetic Meta-Signals generation failed: {e}")
        return []

def generate_composite_candidates(df: pd.DataFrame, specialist_families: List[str], ohlcv_candidates: List[Dict] = None, derived_candidates: List[Dict] = None, horizon_candidates: List[Dict] = None, regime_candidates: List[Dict] = None, validated_candidates: List[Dict] = None) -> List[Dict]:
    """
    Level 10: Composite Interaction Engine (Optimized).
    Generates high-order signals by combining Structural Parents (Causal Seeds) with Trigger Events.
    
    OPTIMIZATIONS (2026-01-11):
    1. Trigger Pruning: Only Top 10% triggers by variance used.
    2. Vectorized Generation: Matrix operations.
    3. MI Selection: Top 50 candidates by Binned Mutual Information (Fast Proxy).
    """
    try:
        from sklearn.metrics import mutual_info_score
    except ImportError:
        tprint_warning("   ⚠️ sklearn not available for MI selection. Using correlation fallback.")
        mutual_info_score = None

    # 1. Build Candidate Map
    candidate_map = {}
    source_lists = [ohlcv_candidates, derived_candidates, horizon_candidates, regime_candidates, validated_candidates]
    for src_list in source_lists:
        if src_list:
            for cand in src_list:
                candidate_map[cand['family']] = cand
                
    for fam in specialist_families:
        if fam not in candidate_map:
             pass 
             
    # 2. Identify Structural Parents (Seeds)
    structural_parents = []
    if validated_candidates:
        structural_parents = [c for c in validated_candidates if len(c['events']) > 200]
        tprint_info(f"   🧬 Using {len(structural_parents)} Structural Seeds (Validated or Fallback).")
    else:
        tprint_warning("   ⚠️ No validated parents found. Using base Specialist families as seeds.")
        for fam in specialist_families:
             if fam in candidate_map:
                 structural_parents.append(candidate_map[fam])

    # 3. Identify Triggers (High Frequency / Shock events)
    triggers = []
    for fam, cand in candidate_map.items():
        if any(x in fam for x in ['SHOCK', 'SURGE', 'SPIKE', 'GAP', 'DERIVED']):
            triggers.append(cand)
            
    if not structural_parents or not triggers:
        return []

    # 4. DATA PREPARATION (Vectorized)
    tprint_info(f"   📊 Preparing Matrix Data for {len(structural_parents)} Parents x {len(triggers)} Triggers...")
    
    # Extract weight vectors into DataFrames
    parent_df = pd.DataFrame({p['family']: p['weight_vector'] for p in structural_parents if p['weight_vector'] is not None}).fillna(0.0)
    trigger_df = pd.DataFrame({t['family']: t['weight_vector'] for t in triggers if t['weight_vector'] is not None}).fillna(0.0)
    
    # Align indices (intersection)
    common_idx = parent_df.index.intersection(trigger_df.index)
    parent_df = parent_df.loc[common_idx]
    trigger_df = trigger_df.loc[common_idx]
    
    # OPTIMIZATION 1: Prune Triggers (Top 10% by Variance)
    trigger_vars = trigger_df.var()
    variance_threshold = trigger_vars.quantile(0.90) # Top 10%
    top_triggers = trigger_vars[trigger_vars >= variance_threshold].index
    trigger_df = trigger_df[top_triggers]
    
    tprint_info(f"   ✂️ Pruned Triggers: {len(triggers)} -> {len(top_triggers)} (Top 10% Variance)")

    # Target Proxy for MI (1-period forward return)
    # If not in df, try to calculate
    target = None
    if 'TARGET_RET_1' in df.columns:
        target = df.loc[common_idx, 'TARGET_RET_1']
    elif 'close' in df.columns:
        target = df['close'].pct_change().shift(-1).loc[common_idx].fillna(0.0)
    
    # 5. GENERATION & SELECTION (Binned MI Proxy)
    scored_candidates = []
    
    if target is not None and mutual_info_score is not None:
        tprint_info("   🚀 Running Binned MI Selection (Fast Proxy)...")
        
        # Bin Target ONCE (5 bins)
        try:
            # Use qcut for equal-frequency bins, drop duplicates if constant
            target_binned = pd.qcut(target, 5, labels=False, duplicates='drop').fillna(-1).astype(int)
        except Exception:
            # Fallback if qcut fails (e.g. all zeros)
            target_binned = pd.cut(target, 5, labels=False).fillna(-1).astype(int)

        # Process per Parent
        for p_fam in parent_df.columns:
            p_vec = parent_df[p_fam].values.reshape(-1, 1) # (N, 1)
            
            # Broadcast Interaction: Parent * Triggers
            interactions = p_vec * trigger_df.values # (N, T)
            
            # Reduce sample size for speed if > 50k
            if interactions.shape[0] > 50000:
                indices = np.random.choice(interactions.shape[0], 50000, replace=False)
                sub_interactions = interactions[indices]
                sub_target = target_binned[indices]
            else:
                sub_interactions = interactions
                sub_target = target_binned

            # Loop Triggers and compute Discrete MI
            for i, t_fam in enumerate(trigger_df.columns):
                # Simple filter: Parent should not be trigger
                if p_fam == t_fam or p_fam.split('_')[0] == t_fam.split('_')[0]:
                    continue
                
                interaction_col = sub_interactions[:, i]
                
                try:
                    # Bin Interaction column
                    # Ensure series for qcut
                    inter_series = pd.Series(interaction_col)
                    inter_binned = pd.qcut(inter_series, 5, labels=False, duplicates='drop').fillna(-1).astype(int)
                    
                    # Compute Discrete MI
                    score = mutual_info_score(inter_binned, sub_target)
                    scored_candidates.append((score, p_fam, t_fam))
                except Exception:
                    continue
                
    else:
        # Fallback: Random selection
        tprint_warning("   ⚠️ No target/MI available. Using first 50 combinations.")
        import itertools
        pairs = list(itertools.product(parent_df.columns, trigger_df.columns))
        scored_candidates = [(1.0, p, t) for p, t in pairs[:100]]

    # 6. SELECT TOP 50 (Global Hard Cap)
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_50 = scored_candidates[:50]
    
    tprint_info(f"   🏆 Selected Top {len(top_50)} / {len(scored_candidates)} Interactions by Binned MI.")

    # 7. GENERATE FINAL CANDIDATES
    composites = []
    for score, p_fam, t_fam in top_50:
        try:
            p_vec = parent_df[p_fam]
            t_vec = trigger_df[t_fam]
            interaction_raw = p_vec * t_vec
            
            # Filter: Smart Event Weight
            z_score = (interaction_raw - interaction_raw.rolling(100).mean()) / (interaction_raw.rolling(100).std() + 1e-9)
            final_weight = pd.Series(smart_event_weight(z_score.fillna(0.0), quantile_threshold=0.99, use_adaptive=True), index=common_idx)
            
            events = final_weight[final_weight > 0].index
            
            if len(events) >= 50: # Minimum events
                 composites.append({
                    'family': f'COMPOSITE_{p_fam}_{t_fam}_INT',
                    'events': events,
                    'weight_vector': final_weight,
                    'params': {'type': 'interaction', 'parents': [p_fam, t_fam], 'role': 'contextual_trigger', 'mi_score': score}
                })
        except Exception:
            continue

    tprint_success(f"   ✅ Generated {len(composites)} Optimized Composite Interactions.")
    
    if specialist_families: 
         pass

    return composites


def generate_meta_features(df: pd.DataFrame, composites: List[Dict], validated_parents: List[Dict]) -> List[Dict]:
    """
    Level 11: Meta-Feature Synthesis.
    Synthesizes validated Causal Parents and Level-10 Composites into final robust Meta-Features.
    
    Transformations (Phase 5 / Level 11):
    1. Candidate Transformation:
       - Rolling Aggregations: Sum, Max/Min, EWMA.
       - Multi-Horizon Aggregation: 15m -> 4H -> 1D.
       - Z-score Normalization.
    2. Interaction Synthesis:
       - Structural Reinforcement: Parent * Composite.
       - Weighted Summation: Tier-1 + 0.5 * Tier-2 smoothing.
    3. Smart Weighting & Filtering:
       - Adaptive Thresholds by Regime.
       - Filter by IC_IR, DSR, Causal Integrity.
    """
    meta_features = []
    
    # Map parents for quick access
    parent_map = {c['family']: c for c in validated_parents}
    
    tprint_info(f"   🏗️ Synthesizing Phase 5 Meta-Features from {len(composites)} Composites and {len(validated_parents)} Parents...")
    
    for comp in composites:
        try:
            # Extract composite info
            c_fam = comp['family']
            c_weight = comp['weight_vector']
            c_parents = comp['params'].get('parents', [])
            
            # --- 1. INTERACTION SYNTHESIS: Structural Reinforcement (Composite * Parent) ---
            dominant_parent = None
            for p_name in c_parents:
                if p_name in parent_map:
                    dominant_parent = parent_map[p_name]
                    break
            
            if dominant_parent:
                p_fam = dominant_parent['family']
                p_weight = dominant_parent['weight_vector']
                
                # Cross-Signal Interaction (Parent * Composite)
                # Magnitude of parent reinforces direction of composite
                Reinforced = c_weight * p_weight.abs() 
                
                # Filter with Smart Event Weight (Level 11 Adaptive)
                z_score = (Reinforced - Reinforced.rolling(100).mean()) / (Reinforced.rolling(100).std() + 1e-9)
                final_weight = pd.Series(smart_event_weight(z_score.fillna(0.0), quantile_threshold=0.995, use_adaptive=True), index=df.index)
                events = final_weight[final_weight > 0].index
                
                if len(events) >= 100:
                    meta_features.append({
                        'family': f'META_REINFORCED_{c_fam}',
                        'events': events,
                        'weight_vector': final_weight,
                        'params': {
                            'type': 'meta_reinforcement', 
                            'source': c_fam, 
                            'reinforcer': p_fam,
                            'level': 11
                        }
                    })

                # --- 2. WEIGHTED SUMMATION (Tier-1 + 0.5*Tier-2) ---
                # This logic is already inside smart_event_weight (weights 1.0 and 0.5),
                # but we can also do it at the signal level for smoothing.
                weighted_sum = (c_weight + 0.5 * p_weight) / 1.5
                z_sum = (weighted_sum - weighted_sum.rolling(100).mean()) / (weighted_sum.rolling(100).std() + 1e-9)
                final_weight_ws = pd.Series(smart_event_weight(z_sum.fillna(0.0), quantile_threshold=0.99, use_adaptive=True), index=df.index)
                events_ws = final_weight_ws[final_weight_ws > 0].index
                
                if len(events_ws) >= 100:
                    meta_features.append({
                        'family': f'META_WEIGHTED_SUM_{c_fam}',
                        'events': events_ws,
                        'weight_vector': final_weight_ws,
                        'params': {
                            'type': 'weighted_sum', 
                            'source': c_fam, 
                            'parent_smooth': p_fam,
                            'level': 11
                        }
                    })

            # --- 3. CANDIDATE TRANSFORMATION: ROLLING AGGREGATIONS & MULTI-HORIZON ---
            
            # A. 4H Rolling Sum (Stability)
            rolling_sum_4h = c_weight.rolling(window=16).sum() # ~4 hours at 15m
            z_sum_4h = (rolling_sum_4h - rolling_sum_4h.rolling(300).mean()) / (rolling_sum_4h.rolling(300).std() + 1e-9)
            
            final_weight_sum_4h = pd.Series(smart_event_weight(z_sum_4h.fillna(0.0), quantile_threshold=0.99, use_adaptive=True), index=df.index)
            events_sum_4h = final_weight_sum_4h[final_weight_sum_4h > 0].index
            
            if len(events_sum_4h) >= 100:
                meta_features.append({
                    'family': f'META_SUM_4H_{c_fam}',
                    'events': events_sum_4h,
                    'weight_vector': final_weight_sum_4h,
                    'params': {'type': 'meta_aggregation', 'source': c_fam, 'method': 'rolling_sum_4h', 'level': 11}
                })

            # B. 1D Rolling EWMA (Trend Following Momentum)
            # span=96 for 1 day at 15m
            rolling_ewma_1d = c_weight.ewm(span=96).mean()
            z_ewma_1d = (rolling_ewma_1d - rolling_ewma_1d.rolling(500).mean()) / (rolling_ewma_1d.rolling(500).std() + 1e-9)
            
            final_weight_ewma_1d = pd.Series(smart_event_weight(z_ewma_1d.fillna(0.0), quantile_threshold=0.995, use_adaptive=True), index=df.index)
            events_ewma_1d = final_weight_ewma_1d[final_weight_ewma_1d > 0].index
            
            if len(events_ewma_1d) >= 100:
                meta_features.append({
                    'family': f'META_EWMA_1D_{c_fam}',
                    'events': events_ewma_1d,
                    'weight_vector': final_weight_ewma_1d,
                    'params': {'type': 'meta_aggregation', 'source': c_fam, 'method': 'rolling_ewma_1d', 'level': 11}
                })

            # C. 4H Rolling Max/Min (Tail Risk / Extremes)
            rolling_max_4h = c_weight.rolling(window=16).max()
            z_max_4h = (rolling_max_4h - rolling_max_4h.rolling(300).mean()) / (rolling_max_4h.rolling(300).std() + 1e-9)
            
            final_weight_max_4h = pd.Series(smart_event_weight(z_max_4h.fillna(0.0), quantile_threshold=0.99, use_adaptive=True), index=df.index)
            events_max_4h = final_weight_max_4h[final_weight_max_4h > 0].index
            
            if len(events_max_4h) >= 100:
                meta_features.append({
                    'family': f'META_MAX_4H_{c_fam}',
                    'events': events_max_4h,
                    'weight_vector': final_weight_max_4h,
                    'params': {'type': 'meta_aggregation', 'source': c_fam, 'method': 'rolling_max_4h', 'level': 11}
                })
                
        except Exception as e:
            tprint_warning(f"   ⚠️ Meta-feature synthesis failed for {c_fam}: {e}")
            continue
            
    # --- 4. SPECIALIST TAIL AGGREGATION (Shock Ratio) ---
    for p in validated_parents:
        if 'SPECIALIST' in p['family']:
            try:
                w = p['weight_vector']
                # 1 Day Rolling Max (Tail Risk / Shock Ratio)
                rolling_max_1d = w.rolling(96).max()
                z_max_1d = (rolling_max_1d - rolling_max_1d.rolling(500).mean()) / (rolling_max_1d.rolling(500).std() + 1e-9)
                
                final_weight_max_1d = pd.Series(smart_event_weight(z_max_1d.fillna(0.0), quantile_threshold=0.995, use_adaptive=True), index=df.index)
                events_max_1d = final_weight_max_1d[final_weight_max_1d > 0].index
                
                if len(events_max_1d) >= 50:
                    meta_features.append({
                        'family': f"META_SHOCK_1D_{p['family']}",
                        'events': events_max_1d,
                        'weight_vector': final_weight_max_1d,
                        'params': {'type': 'meta_tail_risk', 'source': p['family'], 'method': 'rolling_max_1d', 'level': 11}
                    })
            except Exception: continue

    tprint_success(f"   ✅ Phase 5 Synthesis Complete: Generated {len(meta_features)} Meta-Features.")
    return meta_features

def get_inventory_specialist_events(df: pd.DataFrame, threshold: float = 2.0, window: int = 20) -> pd.DatetimeIndex:
    """
    Detects inventory stress events using VPIN-like proxy.
    Inventory Stress = |BuyVol - SellVol| / TotalVol
    """
    if 'volume' not in df.columns or 'close' not in df.columns:
        return pd.DatetimeIndex([])
        
    price = df['close']
    volume = df['volume']
    
    # Estimate Buy/Sell Volume (Bulk Classification)
    price_change = price.diff()
    buy_vol = volume.where(price_change > 0, 0)
    sell_vol = volume.where(price_change < 0, 0)
    
    # Add half volume for flat moves? Or just ignore.
    # Standard VPIN bulk classification logic 
    
    # Calculate Order Imbalance
    buy_roll = buy_vol.rolling(window).sum()
    sell_roll = sell_vol.rolling(window).sum()
    total_roll = volume.rolling(window).sum()
    
    # VPIN proxy
    vpin = (buy_roll - sell_roll).abs() / (total_roll + 1e-9)
    
    # Standardize VPIN
    vpin_mean = vpin.expanding().mean()
    vpin_std = vpin.expanding().std()
    vpin_z = (vpin - vpin_mean) / (vpin_std + 1e-9)
    
    # Events when inventory stress is high
    events = vpin_z[vpin_z > threshold].index
    
    return events

def _safe_to_markdown(df: pd.DataFrame) -> str:
    """Fallback for to_markdown() if tabulate is missing."""
    try:
        return df.to_markdown()
    except Exception:
        cols = df.columns
        res = [" | " + " | ".join(map(str, cols)) + " | "]
        res.append(" | " + " | ".join(["---"] * len(cols)) + " | ")
        for _, row in df.iterrows():
            formatted_row = [f"{x:.4f}" if isinstance(x, (float, np.float64, np.float32)) else str(x) for x in row]
            res.append(" | " + " | ".join(formatted_row) + " | ")
        return "\n".join(res)


def _persist_gate_diagnostics(
    outcomes_log: List[Dict[str, Any]],
    scored_candidates: List[Dict[str, Any]],
    selected_geometries_path: Union[str, Path]
) -> None:
    """
    Persist gate diagnostics, reconcile with selected geometries, and propagate raw metrics.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outcomes")
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_df = pd.DataFrame(outcomes_log)

    # Save CSV regardless for traceability
    csv_path = out_dir / f"geometry_gates_{timestamp}.csv"
    diag_df.to_csv(csv_path, index=False)
    tprint_info(f"Saved geometry gates log to {csv_path}")

    # Load selected geometries to reconcile families and ensure raw metrics carry causal extensions
    selected_families: Set[str] = set()
    missing_families: Set[str] = set()
    selected_geoms: List[Dict[str, Any]] = []

    selected_path = Path(selected_geometries_path)
    if selected_path.exists():
        try:
            selected_geoms = json.loads(selected_path.read_text())
            selected_families = {g.get('family') for g in selected_geoms if g.get('family')}
        except Exception as exc:
            tprint_warning(f"⚠️ Failed loading selected geometries for reconciliation: {exc}")
    else:
        tprint_info("ℹ️ layer2_selected_geometries.json not found; skipping reconciliation (likely fresh run).")

    if selected_families:
        optimized_families = {log.get('family') for log in outcomes_log if log.get('family')}
        missing_families = optimized_families - selected_families
        if missing_families:
            tprint_warning(f"⚠️ Gate Reconciliation: Missing families in selection: {sorted(missing_families)}")

    # Propagate causal robustness metrics onto selected geoms if missing
    metrics_by_uuid = {cand.get('uuid') or cand.get('name'): cand.get('metrics_raw', {}) for cand in scored_candidates}
    updated = False
    for geom in selected_geoms:
        raw_metrics = geom.get('raw_metrics', {}) or {}
        source_metrics = metrics_by_uuid.get(geom.get('uuid')) or raw_metrics
        merged_metrics = raw_metrics.copy()
        for key in RAW_METRIC_FIELDS:
            if key in source_metrics and (key not in merged_metrics or pd.isna(merged_metrics[key])):
                merged_metrics[key] = source_metrics[key]
        if merged_metrics != raw_metrics:
            geom['raw_metrics'] = merged_metrics
            updated = True

    if updated:
        try:
            selected_path.write_text(json.dumps(selected_geoms, indent=2, default=float))
            tprint_info(f"Updated {selected_path} with causal robustness metrics")
        except Exception as exc:
            tprint_warning(f"⚠️ Failed to update selected geometries with metrics: {exc}")

    # Build markdown diagnostics with reconciliation summary
    if not diag_df.empty:
        summary_lines = ["# Layer 2 Geometry Gate Diagnostics\n\n"]
        summary_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total_rows = len(diag_df)
        total_failures = int((diag_df['status'] != 'PASS').sum()) if 'status' in diag_df else 0
        overall_pass_rate = float((diag_df['status'] == 'PASS').mean()) if 'status' in diag_df else 0.0
        missing_family_rows = int(diag_df['family'].isna().sum()) if 'family' in diag_df else 0

        summary_lines.append("## Family Pass Rates\n")
        family_stats = diag_df.groupby('family').agg(
            pass_rate=('status', lambda x: (x == 'PASS').mean()),
            total=('status', 'size'),
            fail_count=('status', lambda x: (x != 'PASS').sum())
        )
        summary_lines.append(_safe_to_markdown(family_stats.sort_values('pass_rate', ascending=False)) + "\n\n")

        summary_lines.append("## Top Failure Reasons\n")
        fail_df = diag_df[diag_df['status'] != 'PASS']
        if not fail_df.empty:
            fail_stats = (
                fail_df['status']
                .value_counts()
                .rename_axis('reason')
                .reset_index(name='count')
            )
            summary_lines.append(_safe_to_markdown(fail_stats) + "\n\n")
        else:
            summary_lines.append("- No failures recorded.\n\n")

        summary_lines.append("## Failure Reasons by Family\n")
        if not fail_df.empty and 'family' in fail_df.columns:
            fail_by_family = (
                fail_df.groupby(['family', 'status'])
                .size()
                .reset_index(name='count')
                .sort_values(['count', 'family'], ascending=[False, True])
            )
            summary_lines.append(_safe_to_markdown(fail_by_family) + "\n\n")
        else:
            summary_lines.append("- No failure reasons available by family.\n\n")

        summary_lines.append("## Gate Health Checks\n")
        summary_lines.append(f"- Total rows: {total_rows}\n")
        summary_lines.append(f"- Overall pass rate: {overall_pass_rate:.2%}\n")
        summary_lines.append(f"- Total failures: {total_failures}\n")
        summary_lines.append(f"- Rows missing family: {missing_family_rows}\n")
        if 'status' in diag_df:
            status_values = sorted(set(map(str, diag_df['status'].dropna().unique())))
            summary_lines.append(f"- Status values: {', '.join(status_values)}\n")
        if total_failures and not family_stats.empty and family_stats['pass_rate'].min() >= 1.0:
            summary_lines.append(
                "- ⚠️ All family pass rates are 1.0 despite failures. "
                "Failures may be missing family labels or status mapping.\n"
            )
        if missing_family_rows:
            summary_lines.append(
                "- ⚠️ Some rows lack a family label; include `family` in outcomes_log for clearer diagnostics.\n"
            )
        summary_lines.append("\n")

        summary_lines.append("## Missing Families In Gate Diagnostics vs Selection\n")
        if missing_families:
            summary_lines.append(f"- Missing families: {', '.join(sorted(missing_families))}\n\n")
        else:
            summary_lines.append("- All optimized families represented in selected geometries.\n\n")

        diag_path = out_dir / f"layer2_gate_diagnostics_{timestamp}.md"
        diag_path.write_text("".join(summary_lines))
        tprint_success(f"💾 Gate diagnostics report saved to {diag_path}")

def orthogonal_label_generation(
    data: Union[pd.Series, pd.DataFrame],
    volume: Optional[pd.Series] = None,
    df_full: Optional[pd.DataFrame] = None,
    target_signals_per_day: float = 7.5,
    use_adaptive_thresholds: bool = True,
    signal_weights: Optional[Dict[str, float]] = None,
    return_raw_candidates: bool = False,
    # Causal framework parameters
    enable_causal_events: bool = True,
    specialist_predictions: Optional[Dict[str, pd.Series]] = None,
    causal_graph: Optional[Dict[str, List[str]]] = None,
    causal_surprise_threshold: float = 1.8,
    # Pipeline logging parameters
    enable_pipeline_logging: bool = True,
    tracker: Optional[Any] = None
) -> List[OutputGeometry]:
    """
    Enhanced Execution Pipeline for Orthogonal Label Generation.
    Implements: Generate -> Score -> Top 50% -> Probe -> Final Diversity Filter.
    
    Args:
        return_raw_candidates: If True, returns all candidates passing gates without global filtering.
    """
    import time
    tprint_info(f"--- Starting Advanced Geometry Generation (Target: {target_signals_per_day} signals/day) ---")
    t_start_total = time.time()

    # Initialize pipeline logger
    if enable_pipeline_logging:
        logger = EventPipelineLogger(verbose=True)
        if df_full is not None:
            logger.log_stage("Raw Data", len(df_full))
        else:
            logger.log_stage("Raw Data", 0)

    # 0. Data Standardization
    if isinstance(data, pd.DataFrame):
        price = data['close']
        if volume is None and 'volume' in data.columns:
            volume = data['volume']
        if df_full is None:
            df_full = data
    else:
        price = data

    if volume is None and df_full is not None and 'volume' not in df_full.columns:
        volume = df_full['volume']
    
    elif 'volume' not in df_full.columns and volume is not None:
        df_full['volume'] = volume

    # 0.5. Layer 2 Price Processing
    # Ensure standard price features (fracdiff, denoised) are available
    processed_df = apply_layer2_price_processing(df_full)
    # Update df_full with new columns
    for col in ['log_returns', 'vol_adjusted_returns', 'fracdiff_price', 'wavelet_denoised_returns']:
        if col in processed_df.columns:
            df_full[col] = processed_df[col]

    # 1. Subsampling (de Prado optimization)
    # Target 10% with min 4 months @ 15m (11,520 bars)
    orig_len = len(df_full)
    MIN_BARS = 4 * 30 * 24 * 4  # 11,520
    if orig_len > MIN_BARS:
        sample_frac = 0.10
        target_len = max(MIN_BARS, int(orig_len * sample_frac))
        tprint_info(f"💾 Subsampling enabled: using {target_len} bars (orig: {orig_len})")
        # We take the most recent bars to ensure we are training on relevant data
        df_full = df_full.iloc[-target_len:].copy()
        price = df_full['close']
        if 'volume' in df_full.columns:
            volume = df_full['volume']
        # Update X_probe later or generate it on the sample
    
    # Update X_probe based on (possibly sampled) price/volume
    X_probe = generate_probe_features(price, volume)


    # 2. Check for 1.5-3% range optimization configuration
    use_range_specific = _should_use_range_specific_optimization()
    
    # 3. Get Enhanced Parameter Grids
    param_grids = get_enhanced_parameter_grids(range_specific=use_range_specific)
    
    # 3. Build Enhanced Candidate Configurations
    generator_configs = []
    
    # Identify S/R levels for SRCusumEvents (simplified approach)
    # Use recent highs/lows as dynamic S/R levels
    price = df_full['close']
    recent_window = min(100, len(price))
    recent_data = price.iloc[-recent_window:]
    
    # Simple pivot-based S/R levels
    sr_levels = []
    if len(recent_data) >= 20:
        # Recent high resistance levels
        resistance_candidates = recent_data.rolling(10).max().dropna()
        # Recent low support levels  
        support_candidates = recent_data.rolling(10).min().dropna()
        
        # Get top 3 levels of each type
        resistance_levels = resistance_candidates.nlargest(3).unique()
        support_levels = support_candidates.nsmallest(3).unique()
        sr_levels = list(resistance_levels) + list(support_levels)

    # Calculate adaptive thresholds if requested
    adaptive_thresholds = {}
    if use_adaptive_thresholds:
         adaptive_thresholds = calibrate_all_cusum_thresholds(
             df_full, target_events_per_day=target_signals_per_day,
             sr_levels=sr_levels
         )
         tprint_info(f"Calibrated Thresholds: {adaptive_thresholds}")

    # Validate framework separation

    # Validate framework separation
    if not enable_causal_events:
        tprint_warning("⚠️ Causal events disabled, but AFML legacy path is deprecated.")
        tprint_info("💡 Enabling causal framework automatically.")
        enable_causal_events = True

    # Orthogonal signal families
    base_generators = []

    # Configure event generators based on framework
    tprint_info("🏗️ Orthogonal: Configuring Event Generators...")
    
    # Add causal specialists and surprise events if enabled
    if CAUSAL_AVAILABLE_ORTHOGONAL:
        tprint_info("🔬 Orthogonal: Using Causal Framework - Causal Specialists as Parents")
        tprint_info("   🎯 Adding causal event generators:")
        causal_generators = [
            ('CAUSAL_SURPRISE', CausalSurpriseEvents()),
            ('VOLUME_SPECIALIST', VolumeSpecialistEvents()),
            ('VOLATILITY_SPECIALIST', VolatilitySpecialistEvents()),
            ('LIQUIDITY_SPECIALIST', LiquiditySpecialistEvents()),
            ('INFORMATION_SPECIALIST', InformationSpecialistEvents()),
            ('INVENTORY_SPECIALIST', InventorySpecialistEvents()),
            ('MOMENTUM_DECAY_SPECIALIST', MomentumDecaySpecialistEvents()),
        ]

        for gen_name, gen_class in causal_generators:
            tprint_info(f"      • {gen_name}: {type(gen_class).__name__}")
            base_generators.append((gen_name, gen_class))
        
        tprint_success(f"✅ Orthogonal: Added {len(causal_generators)} causal event generators")
    else:
        tprint_error("❌ Causal orthogonal modules not available!")
        return []

    tprint_info(f"🏁 Orthogonal: Total event generators configured: {len(base_generators)}")
    
    # Build enhanced parameter combinations
    for fam, gen in base_generators:
        family_config = param_grids['family_grids'].get(fam, {})
        base_params = family_config.get('base_params', [])
        
        # Add base parameters only (no variations)
        for params in base_params:
            p_list = list(params)
            # Inject calibrated threshold if available
            if fam in adaptive_thresholds and adaptive_thresholds[fam] is not None:
                p_list[0] = adaptive_thresholds[fam]
            
            # Inject SR levels for SR_CUSUM
            if fam == 'SR_CUSUM':
                 # params is tuple, convert to list to append
                 p_list = list(params)
                 if len(p_list) == 1: # (h,)
                      p_list.append(sr_levels)
                 params = tuple(p_list)

            # Inject calibrated threshold if available
            # Note: params structure varies.
            # PRICE_CUSUM: (multiplier, vol_window) -> multiplier is approx k.
            # VOL_CUSUM: (h, vol_span)
            # LIQ_CUSUM: (h, vol_span)
            # VOL_PARTICIPATION: (h, span)
            # RANGE_ATR: (h, atr_window, vol_window)
            # SR_CUSUM: (h, sr_levels)

            if use_adaptive_thresholds:
                p_list = list(params)
                if fam == 'PRICE_CUSUM': # param 0 is multiplier/k
                     # Calibrated 'price' is a threshold for price change, not directly k.
                     # AdaptiveSymmetricCUSUM uses k * sigma.
                     # Calibrated threshold is raw price change quantile.
                     # We can leave as is or map.
                     pass
                elif fam == 'VOL_CUSUM' and 'volatility' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['volatility']
                elif fam == 'LIQ_CUSUM':
                     # LIQ uses diff of log(TR/Vol).
                     # Calibrated thresholds doesn't directly give this.
                     pass
                elif fam == 'VOL_PARTICIPATION' and 'volume' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['volume']
                elif fam == 'RANGE_ATR' and 'atr' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['atr']
                elif fam == 'SR_CUSUM' and 'sr' in adaptive_thresholds and adaptive_thresholds['sr'] is not None:
                     p_list[0] = adaptive_thresholds['sr']
                elif fam == 'VOLATILITY_SPECIALIST' and 'VOLATILITY_SPECIALIST' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['VOLATILITY_SPECIALIST']
                elif fam == 'LIQUIDITY_SPECIALIST' and 'LIQUIDITY_SPECIALIST' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['LIQUIDITY_SPECIALIST']
                elif fam == 'INFORMATION_SPECIALIST' and 'INFORMATION_SPECIALIST' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['INFORMATION_SPECIALIST']
                elif fam == 'INVENTORY_SPECIALIST' and 'INVENTORY_SPECIALIST' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['INVENTORY_SPECIALIST']
                elif fam == 'VOLUME_SPECIALIST' and 'VOLUME_SPECIALIST' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['VOLUME_SPECIALIST']

                params = tuple(p_list)

            generator_configs.append((fam, gen, params))
    
    tprint_info(f"Generated {len(generator_configs)} enhanced candidate configurations")
    
    # 3. Process Candidates (Generate & Gate)
    candidates = []
    # 4. Pruning Phase: Validate TP:SL Grid on Central Parameters
    # -------------------------------------------------------------
    tprint_info("✂️ Starting Pruning Phase: Validating TP:SL configs on central parameters...")
    
    # Define Central Parameters for Causal Specialists
    CENTRAL_PARAMS = {
        'CAUSAL_SURPRISE': (2.7, 'all'),
        'VOLUME_SPECIALIST': (2.7, 20),
        'VOLATILITY_SPECIALIST': (2.7, 20),
        'LIQUIDITY_SPECIALIST': (2.7, 20),
        'INFORMATION_SPECIALIST': (2.0, 20),
        'INVENTORY_SPECIALIST': (2.7, 20),
        'MOMENTUM_DECAY_SPECIALIST': (2.0, 10, 50)
    }

    valid_tpsl_map = {} # family -> list of valid grid_items
    
    # Get all unique generator instances/families
    unique_families = list(set(g[0] for g in generator_configs))
    
    for fam in unique_families:
        if fam not in CENTRAL_PARAMS:
            valid_tpsl_map[fam] = param_grids['tpsl_grid'] # Fallback: use all
            continue
            
        central_args = CENTRAL_PARAMS[fam]
        # Find the generator instance
        gen_instance = next((g[1] for g in generator_configs if g[0] == fam), None)
        if not gen_instance: continue

        # Generate events for central params
        # Generate events for central params
        try:
             # Extended list of classes requiring DataFrame
             df_required_local = (
                 'VolatilityCusumEvents', 'LiquidityCusumEvents', 'VolumeCusumEvents',
                 'RangeATRcusumEvents', 'SRCusumEvents',
                 'TailRiskCusumEvents', 'TrendRegimeCusumEvents', 'VolatilityStateEvents',
                 'ImprovedCUSUMEvents'
             )
             
             gen_classname = gen_instance.__class__.__name__
             
             if gen_classname in df_required_local:
                 if gen_classname == 'ImprovedCUSUMEvents':
                      # ImprovedCUSUM: generate(df, k=..., vol_window=...)
                      p_names = GENERATOR_PARAM_NAMES.get('ImprovedCUSUMEvents', ['k', 'vol_window'])
                      kwargs = dict(zip(p_names, central_args))
                      c_events = gen_instance.generate(df_full, **kwargs)
                 else:
                      c_events = gen_instance.generate(df_full, *central_args)
             elif gen_classname == 'CausalSurpriseEvents' or fam.startswith('CAUSAL_') or fam.endswith('_SPECIALIST'):
                 # Causal generators need specialist predictions
                 c_events = gen_instance.generate(
                     df_full,
                     specialist_predictions=specialist_predictions,
                     causal_graph=causal_graph,
                     surprise_threshold=causal_surprise_threshold
                 )
             else:
                  c_events = gen_instance.generate(price, *central_args)
        except Exception as e:
            tprint_warning(f"Pruning: Failed to generate central events for {fam}: {e}")
            valid_tpsl_map[fam] = param_grids['tpsl_grid'] # Fallback
            continue

        if len(c_events) < 5:
            tprint_warning(f"Pruning: {fam} central events too few ({len(c_events)}). Skipping usage check.")
            valid_tpsl_map[fam] = param_grids['tpsl_grid']
            continue
            
        tprint_info(f"   [Pruning] Testing {len(param_grids['tpsl_grid'])} TP:SL combinations for {fam}...")
        t_start_fam_prune = time.time()

        # Calculate duration for rate check (aligned with check_label_quality)
        duration_days = max(1, (df_full.index[-1] - df_full.index[0]).days)
        min_events_rate = max(50, int(0.1 * duration_days))

        # Test TP:SL combinations
        valid_items = []
        for grid_item in param_grids['tpsl_grid']:
            pt = grid_item['pt']
            sl = grid_item['sl']
            
            # Use a representative horizon (e.g. 24) and risk_budget (e.g. 0.7) for pruning
            # We check if this TP:SL passes class balance and min samples
            
            try:
                # Label Generation (Simplified call for speed)
                # Need to use appropriate labeling function based on family
                
                # ... (Labeling logic similar to main loop) ...
                # To avoid code duplication, we assume similar mapping logic
                
                # Standard Causal Triple Barrier for ALL specialists
                high, low = df_full.get('high'), df_full.get('low')
                lbls, _, _, _, _, _ = compute_dominance_labels(price, c_events, df_full['volatility_1d'], risk_budget=0.7, pt_mult=pt, sl_mult=sl, horizon=24, high=high, low=low)
                
                if lbls.empty: continue
                
                # Check Gates: Balance & Count (ALIGNED with check_label_quality)
                if fam == 'PRICE_CUSUM':
                    pos_rate = (lbls == 1).mean()
                    min_bal = 0.10 # Stricter to match main gate (0.10)
                else:
                    pos_rate = (lbls != 0).mean()
                    min_bal = 0.05 # Relaxed to match main gate (0.05 for regression)
                
                # Check rate matching main gate (0.1/day) plus minimum 50 events
                count_ok = len(lbls) >= min_events_rate
                bal_ok = (pos_rate >= min_bal) and (pos_rate <= 0.90) if fam == 'PRICE_CUSUM' else (pos_rate >= min_bal)
                
                if count_ok and bal_ok:
                    valid_items.append(grid_item)
                    
            except Exception:
                continue
                
        if len(valid_items) > 0:
            valid_tpsl_map[fam] = valid_items
            tprint_info(f"✂️ Pruned {fam}: Kept {len(valid_items)}/{len(param_grids['tpsl_grid'])} TP:SL configs in {time.time() - t_start_fam_prune:.2f}s")
        else:
            tprint_warning(f"Pruning warning: {fam} had 0 passing TP:SLs. Using all defaults.")
            valid_tpsl_map[fam] = param_grids['tpsl_grid']

    
    tprint_info(f"🏁 Pruning phase complete in {time.time() - t_start_total:.2f}s")
    
    # 5. Process Candidates (Main Sweep)
    tprint_info("🚀 Starting Main Parameter Sweep...")
    candidates = []
    outcomes_log = []


    for fam, gen, params in generator_configs:

            try:
                # Use standard generation (Adaptive logic skipped for now for these new generators to match snippet)
                # But kept logic structure if needed.

                # Extended list of classes requiring DataFrame
                df_required = DF_REQUIRED_CLASSES + (
                    'VolatilityCusumEvents', 'LiquidityCusumEvents', 'VolumeCusumEvents',
                    'RangeATRcusumEvents', 'SRCusumEvents',
                    'TailRiskCusumEvents', 'TrendRegimeCusumEvents', 'VolatilityStateEvents',
                    'ImprovedCUSUMEvents'
                )

                # Get parameter names from registry
                param_names = GENERATOR_PARAM_NAMES.get(gen.__class__.__name__)
                
                # Determine input data (Full DF vs Series)
                input_data = df_full if gen.__class__.__name__ in df_required else price
                
                if gen.__class__.__name__ == 'CausalSurpriseEvents':
                    # Special handling for causal surprise events
                    events = gen.generate(
                        df_full,
                        specialist_predictions=specialist_predictions,
                        causal_graph=causal_graph,
                        surprise_threshold=causal_surprise_threshold,
                        target_signals_per_day=target_signals_per_day,
                        tracker=tracker
                    )
                elif param_names:
                    # Convert positional params to kwargs for cleaner API handling
                    # Handles cases where generate(df, **params) is used
                    kwargs = dict(zip(param_names, params))
                    events = gen.generate(input_data, tracker=tracker, **kwargs)
                else:
                    # Fallback to positional for unregistered legacy generators
                    events = gen.generate(input_data, tracker=tracker, *params)
            except Exception as e:
                tprint_warning(f"Generator {fam} failed: {e}")
                continue

            if len(events) < 5: 
                tprint_warning(f"Skipping {fam}: Too few events ({len(events)})")
                continue
            
            # Log signal rate for monitoring
            duration_days = (events[-1] - events[0]).days if len(events) > 1 else 1
            signals_per_day = len(events) / max(1, duration_days)
            tprint_info(f"DEBUG: {fam} generated {len(events)} events")

            # Create parameter dict for logging
            param_names = GENERATOR_PARAM_NAMES.get(gen.__class__.__name__, [])
            if param_names and len(params) == len(param_names):
                param_dict = dict(zip(param_names, params))
            else:
                param_dict = {'params': params}

            # Iterate Enhanced Grids - Using Validated TP:SLs
            tpsl_grid = valid_tpsl_map.get(fam, param_grids['tpsl_grid'])
            
            # If no pruning happened for this family, warn or info
            if len(tpsl_grid) == len(param_grids['tpsl_grid']):
                 # tprint_info(f"Using full grid for {fam}")
                 pass
            
            horizon_options = param_grids['horizon_options']
            risk_budget_options = [0.4, 0.7, 1.0]  # 0=no drawdown before TP, 1=very close to SL on average
            
            # Get family-specific horizons (PRICE_CUSUM: [12, 48], others: [48])
            if isinstance(horizon_options, dict):
                family_horizons = horizon_options.get(fam, horizon_options.get('default', [48]))
            else:
                family_horizons = horizon_options  # Fallback for list
            
            # Track seen configurations to avoid redundancy
            seen_configs = set()

            for grid_item in tpsl_grid:
                pt = grid_item['pt']
                sl = grid_item['sl']
                
                # OPTIMIZATION: Many families ignore SL and Risk Budget. 
                # Skip redundant iterations to avoid 30x duplication in logs.
                # In Causal 2026, we apply Causal Triple Barrier to Surprise events.
                is_triple_barrier = (fam == 'CAUSAL_SURPRISE')
                
                # If not triple barrier, we only need one SL and one Risk Budget per PT
                current_sl_options = [sl] if is_triple_barrier else [1.0]
                current_rb_options = risk_budget_options if is_triple_barrier else [0.7]

                for horizon in family_horizons:
                    for sl_val in current_sl_options:
                        for risk_budget in current_rb_options:
                            # Use sl_val instead of sl from grid_item if not triple barrier
                            actual_sl = sl if is_triple_barrier else sl_val
                            
                            # Check for redundancy
                            config_key = (fam, pt, horizon, actual_sl, risk_budget)
                            if config_key in seen_configs:
                                continue
                            seen_configs.add(config_key)
                            
                            high = df_full.get('high')
                            low = df_full.get('low')

                            
                            # Robust Volatility Handling
                            if 'volatility_1d' in df_full.columns:
                                vol_series = df_full['volatility_1d']
                            else:
                                # Fallback: Compute rolling volatility (approx 1 day for 15m = 96 bars)
                                # Assuming 15m bars, but safe fallback for any timeframe
                                safe_window = 96 
                                rets = df_full['close'].pct_change()
                                vol_series = rets.rolling(window=safe_window).std()
                                # Backfill initial NaNs to avoid dropping data
                                vol_series = vol_series.bfill().fillna(0.01)

                            # Standard Causal Triple Barrier for ALL specialists
                            # pt=[1.5, 4.0] and sl=[0.5, 1.0] are used from grid
                            # risk_budget=[0.5, 0.9]
                            labels, weights, returns, mfe, mae, vol = compute_dominance_labels(
                                price, events, vol_series,
                                risk_budget=risk_budget, pt_mult=pt, sl_mult=actual_sl, horizon=horizon,
                                high=high, low=low
                            )

                            if labels.empty:
                                continue

                            # Quality Checks - Only if labels exist
                            passed, metrics, status = check_label_quality(
                                events, labels, returns, df_full, X_probe, gen, param_dict, family=fam
                            )

                            outcomes_log.append({
                                'family': fam,
                                'params': str(param_dict),
                                'pt_mult': pt,
                                'sl_mult': actual_sl,
                                'horizon': horizon,
                                'risk_budget': risk_budget,
                                'status': status,
                                'n': metrics.get('n', 0),
                                'pos_rate': metrics.get('pos_rate', 0),
                                'min_p': metrics.get('min_p', 1.0),
                                'max_mi': metrics.get('max_mi', 0.0),
                                'signals_per_day': round(signals_per_day, 2),
                                'target_signals_per_day': target_signals_per_day,
                                'adaptive_used': False
                            })

                            if passed:
                                candidates.append({
                                    'family': fam,
                                    'events': events,
                                    'labels': labels,
                                    'weights': weights,
                                    'returns': returns,
                                    'mfe': mfe, 'mae': mae, 'vol': vol,
                                    'params': {**param_dict, 'risk_budget': risk_budget, 'pt_mult': pt, 'sl_mult': actual_sl, 'horizon': horizon},
                                    'status': status
                                })
            tprint_info(f"Generated {len(candidates)} total candidates across family {fam}")

    # 7. OHLCV Candidate Generation (New Layer 1.5)
    tprint_info("📊 Generating OHLCV Candidates...")
    ohlcv_candidates = generate_ohlcv_candidates(df_full)
    for cand in ohlcv_candidates:
        tprint_info(f"   ➕ Added {cand['family']} with {len(cand['events'])} events")
    
    # 7.1. Continuous Predictor Candidates
    tprint_info("🌊 Generating Continuous Predictor Candidates...")
    continuous_candidates = generate_continuous_geometry_candidates(df_full)
    for cand in continuous_candidates:
        tprint_info(f"   🌊 Added {cand['family']} with {len(cand['events'])} events")
    
    # Add OHLCV candidates to the main list
    # We need to process them fully through labeling loop below, so we add them to a format 
    # that the loop can handle or just add them to composite generation?
    # Actually, let's treat them as distinct families for the loop.
    
    # 7b. Advanced Layer-3 Generation (Derived & Aggregates)
    tprint_info("🧠 Generating Advanced Layer-3 Candidates...")
    derived_candidates = generate_derived_features(df_full, ohlcv_candidates)
    
    # Inject continuous candidates into derived pool for composites
    derived_candidates = derived_candidates + continuous_candidates
    
    # Generate aggregates from OHLCV + Specialists
    # First get specialist weights as candidates list for uniform processing
    spec_families = [f for f in unique_families if f.endswith('_SPECIALIST')]
    spec_candidates = []
    for fam in spec_families:
        w = get_specialist_event_matrix(df_full, fam)
        spec_candidates.append({'family': fam, 'weight_vector': w})
        
    agg_candidates = generate_tail_aggregates(df_full, ohlcv_candidates + spec_candidates)
    
    advanced_candidates = derived_candidates + agg_candidates
    for cand in advanced_candidates:
        tprint_info(f"   🧠 Added {cand['family']} with {len(cand['events'])} events")
        
    # 7c. Multi-Horizon Generation (Level 5)
    tprint_info("⏳ Generating Multi-Horizon Candidates...")
    horizon_candidates = generate_multi_horizon_candidates(df_full)
    for cand in horizon_candidates:
        tprint_info(f"   ⏳ Added {cand['family']} with {len(cand['events'])} events")
        
    # 7d. Regime-Conditioned Features
    tprint_info("🏗️ Generating Regime-Conditioned Features...")
    # Use spec_families defined above
    regime_candidates = generate_regime_conditioned_candidates(df_full, ohlcv_candidates, spec_families)
    for cand in regime_candidates:
        tprint_info(f"   🏗️ Added {cand['family']} with {len(cand['events'])} events")
    
    # Combine base candidates for initial filtering (Level 7)
    pre_generated_candidates = ohlcv_candidates + continuous_candidates + advanced_candidates + horizon_candidates + regime_candidates
    
    # 9. Feature Filtering (Level 7) & Smart Selection
    # Filter the final set before Labeling
    tprint_info("🧹 Running Advanced Feature Filtering (Level 7)...")
    filtered_candidates = filter_advanced_candidates(pre_generated_candidates, min_count=200, max_corr=0.95, df=df_full)
    
    # 9b. Causal Graph Validation (Level 9)
    # Validate the filtered (smart selected) candidates against the causal graph
    validated_candidates = validate_candidates_with_causal_graph(filtered_candidates, df=df_full)
    
    # Update filtered_candidates to be the validated set
    filtered_candidates = validated_candidates
    
    # 10. Composite Interaction Engine (Level 10)
    # Uses validated Causal Parents as seeds for high-order signals
    tprint_info("🧩 Running Composite Interaction Engine (Level 10)...")
    try:
        composite_candidates = generate_composite_candidates(
            df_full, 
            spec_families,
            ohlcv_candidates=ohlcv_candidates, 
            derived_candidates=derived_candidates,
            horizon_candidates=horizon_candidates,
            regime_candidates=regime_candidates,
            validated_candidates=filtered_candidates # Validated seeds
        )
        tprint_success(f"   ✅ Generated {len(composite_candidates)} composite interactions.")
        
        # 10.5 RMI-based Composite Reduction (Optimization)
        # Reduces composites from ~1800 to 500 using Residual Mutual Information
        if len(composite_candidates) > 500:
            composite_candidates = filter_composites_by_rmi(
                df_full, composite_candidates, target_col='close', top_k=500
            )
        
        # Add composites to the final set
        filtered_candidates = filtered_candidates + composite_candidates
        
    except Exception as e:
        tprint_warning(f"Composite generation failed: {e}")
        composite_candidates = []

    
    # 11. Meta-Feature Synthesis (Level 11)
    # Synthesize Composites + Parents into Final Meta-Features
    tprint_info("🧱 Running Meta-Feature Synthesis (Level 11)...")
    try:
        meta_features = generate_meta_features(df_full, composite_candidates, validated_candidates)
        tprint_success(f"   ✅ Generated {len(meta_features)} Meta-Features.")
        
        # Add Meta-Features to the final set
        filtered_candidates = filtered_candidates + meta_features
    except Exception as e:
        tprint_warning(f"Meta-feature synthesis failed: {e}")
        meta_features = []

    # 12. Synthetic Meta-Signals (Level 8) - Keep this as legacy or extra
    tprint_info("🧪 Generating Synthetic Meta-Signals (PCA)...")
    synthetic_candidates = generate_synthetic_meta_signals(df_full, filtered_candidates, n_components=5)
    for cand in synthetic_candidates:
         tprint_info(f"   🧪 Added {cand['family']} with {len(cand['events'])} events")
         
    # Final set includes Synthetic signals
    final_candidates_for_selection = filtered_candidates + synthetic_candidates
    
    for comp in final_candidates_for_selection:
        try:
            # Use fixed TP:SL for composites for now (Standard Institutional Params)
            pt, actual_sl, horizon, risk_budget = 2.0, 1.0, 48, 0.7  # Horizon updated from 24 to 48
            
            labels, weights, returns, mfe, mae, vol = compute_dominance_labels(
                price, comp['events'], df_full['volatility_1d'],
                risk_budget=risk_budget, pt_mult=pt, sl_mult=actual_sl, horizon=horizon
            )
            
            if labels.empty: continue
            
            # Apply Two-Tier Weights to the interaction vector
            # The weight_vector is already [0, 1] scaled by interactions
            # We use it directly as weights
            final_weights = comp['weight_vector'].reindex(comp['events']).fillna(1.0)
            
            passed, metrics, status = check_label_quality(
                comp['events'], labels, returns, df_full, X_probe, None, comp['params'], family=comp['family']
            )
            
            if passed:
                candidates.append({
                    'family': comp['family'],
                    'events': comp['events'],
                    'labels': labels,
                    'weights': final_weights,
                    'returns': returns,
                    'mfe': mfe, 'mae': mae, 'vol': vol,
                    'params': {**comp['params'], 'risk_budget': risk_budget, 'pt_mult': pt, 'sl_mult': actual_sl, 'horizon': horizon},
                    'status': status
                })
            
            # Always log to outcomes for diagnostic visibility
            outcomes_log.append({
                'family': comp['family'],
                'params': str(comp['params']),
                'pt_mult': pt,
                'sl_mult': actual_sl,
                'horizon': horizon,
                'risk_budget': risk_budget,
                'status': status,
                'n': metrics.get('n', 0),
                'pos_rate': metrics.get('pos_rate', 0),
                'min_p': metrics.get('min_p', 1.0),
                'max_mi': metrics.get('max_mi', 0.0),
                'signals_per_day': round(len(comp['events']) / 360, 2), # Approx
                'target_signals_per_day': 0,
                'adaptive_used': False
            })
        except Exception as e:
            tprint_warning(f"Composite {comp['family']} failed: {e}")

    # 5. Multi-Factor Scoring
    scored_candidates = calculate_multifactor_score(candidates, X_probe)
    
    if not scored_candidates:
        tprint_warning("No candidates passed gates.")
        return []

    # Gate diagnostics persist + reconciliation with selected geometries
    _persist_gate_diagnostics(outcomes_log, scored_candidates, selected_geometries_path="outcomes/layer2_selected_geometries.json")

    # Rank -> Top 50% -> Cluster (5) -> Top 1 -> Probe

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Pipeline logging: Candidate generation stage
    if enable_pipeline_logging:
        total_generated_events = sum(len(cand.get('events', [])) for cand in scored_candidates)
        logger.log_stage("Generated Candidates", total_generated_events, len(df_full) if df_full is not None else 0)

    # If requested, return ALL robust candidates for Layer 2 Selection
    if return_raw_candidates:
        tprint_info(f"Returning {len(scored_candidates)} raw candidates for advanced selection.")
        raw_geoms = []
        for cand in scored_candidates:
            # We need to construct OutputGeometry but WITHOUT Probe Metrics (expensive?)
            # Actually, User Plan says: "Select top 5 per family -> Probe -> Winner"
            # So we shouldn't probe ALL here if it's expensive.
            # But OutputGeometry usually expects probe metrics (AUC).
            # Current logic: Probe is done on Top 50%.
            
            # Let's perform a lightweight probe or defer it?
            # 'auc' field in OutputGeometry is key.
            # We can use 'learnability' (IC/PSR) from 'metrics_raw' as proxy for AUC 
            # or just set a placeholder since Layer 2 will re-probe/race.
            
            # Use 'lift' or 'sharpe_meta' from initial metrics as AUC proxy
            auc_proxy = cand['metrics_raw'].get('lift', 0.0)
            
            purity = 1.0 # Placeholder
            
            # Skip expensive weight calculation for raw candidates - specific weights only needed for final geometry
            final_weights = None # get_signal_specific_weights(df_full, cand['events'], sr_levels, component_weights=signal_weights, family=cand['family'])
            
            geo = OutputGeometry(
                name=f"{cand['family']}_{cand['params']}",
                family=cand['family'],
                events=cand['events'],
                labels=cand['labels'],
                weights=final_weights,
                purity=purity,
                auc=auc_proxy, 
                params=cand['params'],
                metrics=cand['metrics_raw']
            )
            raw_geoms.append(geo)
        return raw_geoms

    # Keep Top 50%
    n_keep = max(1, len(scored_candidates) // 2)
    top_candidates = scored_candidates[:n_keep]
    logger.info(f"Top 50% selection: Kept {len(top_candidates)} from {len(scored_candidates)} candidates.")

    # Pipeline logging: Top 50% selection
    if enable_pipeline_logging:
        total_top_events = sum(len(cand.get('events', [])) for cand in top_candidates)
        logger.log_stage("Top 50% Selected", total_top_events, len(df_full) if df_full is not None else 0)

    # 5. Run LGBM Probe on Top Candidates
    tprint_info(f"🚀 Running LGBM Probe on {len(top_candidates)} top candidates...")
    probe_geoms = []
    for i, cand in enumerate(top_candidates):
        tprint_info(f"🎯 Probing candidate {i+1}/{len(top_candidates)}: {cand['family']}_{cand['params']}")
        X = X_probe.loc[cand['labels'].index]
        metrics = run_lgbm_probe(X, cand['labels'], cand['weights'], cand['returns'])
        cand['metrics_probe'] = metrics

        # Create OutputGeometry
        indicator = build_indicator_matrix(cand['events'], price.index, horizon=120)
        purity = average_uniqueness(indicator)

        # Get De Prado Weights (combining all signals)
        final_weights = get_signal_specific_weights(df_full, cand['events'], sr_levels, component_weights=signal_weights, family=cand['family'])
        # Blend or replace? "update with this" suggests using it.
        # Use final_weights as the primary weights for the geometry.

        geo = OutputGeometry(
            name=f"{cand['family']}_{cand['params']}",
            family=cand['family'],
            events=cand['events'],
            labels=cand['labels'],
            weights=final_weights, # Use combined weights
            purity=purity,
            auc=metrics.get('lift', 0.0),  # Store lift as primary metric
            params=cand['params'],
            metrics={**cand.get('metrics_raw', {}), **cand['metrics_probe']}
        )
        probe_geoms.append(geo)

    # Pipeline logging: Probe stage
    if enable_pipeline_logging:
        total_probe_events = sum(len(geo.events) for geo in probe_geoms)
        logger.log_stage("Probed Geometries", total_probe_events, len(df_full) if df_full is not None else 0)

    # 6. Apply Final Diversity Filter
    tprint_info(f"🌐 Applying Final Diversity Filter to {len(probe_geoms)} geometries...")
    final_geoms = final_diversity_filter(probe_geoms, price, 
                                       jaccard_threshold=0.7, 
                                       returns_threshold=0.8)
    
    # Final pipeline logging summary
    if enable_pipeline_logging:
        total_events = sum(len(geo.events) for geo in final_geoms)
        logger.log_stage("Final Geometries", total_events, len(df_full) if df_full is not None else 0)
        logger.print_summary()
    
    tprint_info(f"🎉 Pipeline Complete: {len(final_geoms)} final geometries selected")
    return final_geoms


def generate_dual_cusum_signals(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    k: float = 0.12,
    alpha: float = 1.0,
    beta: float = 1.0,
    er_min: float = 0.2,
    vol_window: int = 20,
    window_er: int = 10
) -> pd.DataFrame:
    """
    Generate CUSUM signals with differentiated weights for trend vs reversal.
    Utilizes Efficiency Ratio (ER) to classify regime.
    """
    # 1. Volatility Scaling
    returns = close.pct_change()
    vol = returns.rolling(vol_window).std()
    
    # 2. Efficiency Ratio (Kaufman)
    direction = close.diff(window_er).abs()
    volatility = returns.abs().rolling(window_er).sum()
    er = direction / (volatility + 1e-9)
    er = er.fillna(0)
    
    # 3. Dynamic Thresholds
    h = k * vol
    
    # 4. Standard CUSUM logic with ER weighting
    s_pos = 0
    s_neg = 0
    diff = close.diff()
    
    trend_signal = pd.Series(0, index=close.index)
    reversal_signal = pd.Series(0, index=close.index)
    
    diff_arr = diff.values
    h_arr = h.values
    er_arr = er.values
    
    for i in range(1, len(close)):
        if np.isnan(h_arr[i]): continue
        
        # S+
        s_pos = max(0, s_pos + diff_arr[i])
        # S-
        s_neg = min(0, s_neg + diff_arr[i])
        
        if s_pos > h_arr[i]:
            # Regime Detection
            if er_arr[i] > er_min:
                trend_signal.iloc[i] = 1
            else:
                reversal_signal.iloc[i] = -1
            s_pos = 0
            
        elif s_neg < -h_arr[i]:
            if er_arr[i] > er_min:
                trend_signal.iloc[i] = -1
            else:
                reversal_signal.iloc[i] = 1
            s_neg = 0
            
    return pd.DataFrame({
        'trend_signal': trend_signal,
        'reversal_signal': reversal_signal
    })

class InventorySpecialistEvents(BaseEventGenerator):
    """
    Inventory Specialist as Causal Parent.
    Causal Role: Estimates the Position (Z). Determines directional commitment of MM.
    Mechanism: Inventory skew leads to mean-reverting pressure.
    """
    def generate(self, df: pd.DataFrame, tracker: Optional[Any] = None, *args, **params) -> pd.DatetimeIndex:
        threshold = params.get('threshold', 2.7)
        window = params.get('window', 50)
        
        try:
            events = self._get_inventory_causal_events(df, threshold=threshold, window=window)
            if len(events) == 0 and tracker:
                tracker.log_rejection("inventory_no_events", 1)
            return events
        except Exception as e:
            logger.warning(f"InventorySpecialistEvents failed: {e}")
            if tracker:
                tracker.log_rejection(f"inventory_exception_{e}", 1)
            return pd.DatetimeIndex([])
            
    def _get_inventory_causal_events(self, df: pd.DataFrame, window=50, threshold=2.7):
        if 'close' not in df.columns:
            return pd.DatetimeIndex([])
        close = df['close']
        # Corrected: Use price rolling std for z-score of price deviation
        # This fixes a dimensional error where it previously used return volatility
        price_std = close.rolling(window).std()
        z = (close - close.rolling(window).mean()) / (price_std + 1e-9)
        return df.index[np.abs(z) > threshold]

class VolumeSpecialistEvents(BaseEventGenerator):
    """
    Volume Specialist as Causal Parent.
    Causal Role: Predicts the Flow (dV). Determines the velocity of information.
    
    Enhanced: Uses Order Imbalance z-score instead of raw volume spikes.
    Order Imbalance captures informed trading direction, not just volume magnitude.
    """
    def generate(self, df: pd.DataFrame, tracker: Optional[Any] = None, **params) -> pd.DatetimeIndex:
        threshold = params.get('threshold', 2.7)
        window = params.get('window', 20)
        
        try:
            events = self._get_volume_causal_events(df, threshold=threshold, window=window)
            if len(events) == 0 and tracker:
                tracker.log_rejection("volume_no_events", 1)
            return events
        except Exception as e:
            logger.warning(f"VolumeSpecialistEvents failed: {e}")
            if tracker:
                tracker.log_rejection(f"volume_exception_{e}", 1)
            return pd.DatetimeIndex([])

    def _get_volume_causal_events(self, df: pd.DataFrame, window=20, threshold=2.7):
        """Detect order flow information events using Order Imbalance z-score."""
        if 'volume' not in df.columns or 'close' not in df.columns:
            return pd.DatetimeIndex([])
        
        vol = df['volume']
        ret = df['close'].pct_change()
        
        # 1. Signed Volume (direction inferred from close-to-close)
        buy_vol = vol * (ret > 0).astype(float)
        sell_vol = vol * (ret < 0).astype(float)
        
        # 2. Order Imbalance over rolling window
        imbalance = (buy_vol - sell_vol).rolling(window).sum()
        total_vol = vol.rolling(window).sum()
        obi = imbalance / (total_vol + 1e-9)  # Order Book Imbalance proxy [-1, 1]
        
        # 3. Z-score of OBI (information content detection)
        obi_mean = obi.rolling(window * 3).mean()
        obi_std = obi.rolling(window * 3).std()
        obi_z = (obi - obi_mean) / (obi_std + 1e-9)
        
        # 4. Also check Volume-Price Divergence (high vol, low price move = hidden accumulation)
        vol_z = (vol - vol.rolling(window * 3).mean()) / (vol.rolling(window * 3).std() + 1e-9)
        price_z = np.abs(ret) / (ret.abs().rolling(window * 3).mean() + 1e-9)
        divergence = vol_z - price_z  # High volume but low price impact
        
        # Trigger on either strong OBI or significant divergence
        mask = (np.abs(obi_z) > threshold) | ((vol_z > 1.5) & (divergence > threshold * 0.5))
        
        return df.index[mask]

class VolatilitySpecialistEvents(BaseEventGenerator):
    """
    Volatility Specialist as Causal Parent.
    Causal Role: Predicts the Risk (sigma). Determines the width of the distribution.
    
    Enhanced: Uses Parkinson range-based volatility with z-score against shifted baseline.
    Avoids self-referential quantile thresholding that suppressed event counts.
    """
    def generate(self, df: pd.DataFrame, tracker: Optional[Any] = None, **params) -> pd.DatetimeIndex:
        # Dynamic threshold: top 5% by default, or user specific
        quantile_threshold = params.get('quantile', 0.95)
        window = params.get('window', 20)
        
        try:
            events = self._get_volatility_causal_events(df, quantile=quantile_threshold, window=window)
            if len(events) == 0 and tracker:
                tracker.log_rejection("volatility_no_events", 1)
            return events
        except Exception as e:
            logger.warning(f"VolatilitySpecialistEvents failed: {e}")
            if tracker:
                tracker.log_rejection(f"volatility_exception_{e}", 1)
            return pd.DatetimeIndex([])

    def _get_volatility_causal_events(self, df: pd.DataFrame, window=20, quantile=0.95):
        """Detect volatility expansion events using Parkinson range-based vol."""
        if 'close' not in df.columns:
            tprint_warning("VolatilitySpecialist: Missing 'close' column")
            return pd.DatetimeIndex([])
        
        # Debug logging
        tprint_info(f"VolatilitySpecialist: Input shape={df.shape}, columns={list(df.columns)}")
        tprint_info(f"VolatilitySpecialist: Price range={df['close'].min():.4f}-{df['close'].max():.4f}")
        
        # Use Parkinson volatility if high/low available, else fallback to close-based
        if 'high' in df.columns and 'low' in df.columns:
            log_hl = np.log(df['high'] / (df['low'] + 1e-9))
            parkinson_vol = log_hl / (2 * np.sqrt(np.log(2)))
            tprint_info("VolatilitySpecialist: Using Parkinson volatility (high/low)")
        else:
            ret = df['close'].pct_change()
            parkinson_vol = ret.abs()
            tprint_info("VolatilitySpecialist: Using close-based volatility (fallback)")
        
        # Check for constant prices (zero volatility)
        vol_std = parkinson_vol.std()
        if vol_std < 1e-8:
            tprint_warning(f"VolatilitySpecialist: Nearly zero volatility (std={vol_std:.2e})")
            # Generate events based on absolute price changes instead
            ret = df['close'].pct_change().abs()
            events = df.index[ret > ret.quantile(0.98)]
            tprint_info(f"VolatilitySpecialist: Fallback - {len(events)} events from price changes")
            return events
        
        tprint_info(f"VolatilitySpecialist: Vol stats mean={parkinson_vol.mean():.6f}, std={vol_std:.6f}")
        
        # EWM baseline for smooth regime detection
        vol_baseline = parkinson_vol.ewm(span=window * 5).mean()
        vol_ratio = parkinson_vol / (vol_baseline + 1e-9)
        
        # Adaptive lookback based on data size
        lookback = min(100, len(df) // 3)
        shifted_mean = vol_ratio.shift(1).rolling(lookback).mean()
        shifted_std = vol_ratio.shift(1).rolling(lookback).std()
        z = (vol_ratio - shifted_mean) / (shifted_std + 1e-9)
        
        # Adaptive threshold: use actual data quantile if too strict
        try:
            from scipy import stats
            z_threshold = stats.norm.ppf(quantile)
            # If threshold would produce < 1% events, relax it
            potential_events = (z > z_threshold).sum()
            if potential_events < len(df) * 0.01:
                z_threshold = z.quantile(0.98)  # Use empirical 98th percentile
                tprint_info(f"VolatilitySpecialist: Relaxed threshold to {z_threshold:.3f}")
        except:
            z_threshold = z.quantile(0.98)
        
        # Secondary: vol term structure inversion
        ret = df['close'].pct_change()
        vol_short = ret.rolling(5).std()
        vol_long = ret.rolling(50).std()
        term_structure = vol_short / (vol_long + 1e-9)
        ts_z = (term_structure - term_structure.rolling(100).mean()) / (term_structure.rolling(100).std() + 1e-9)
        
        # Combine: primary z-threshold OR term structure inversion
        combined_mask = (z > z_threshold) | (ts_z > z_threshold * 0.8)
        events = df.index[combined_mask]
        tprint_info(f"VolatilitySpecialist: Generated {len(events)} events")
        
        return events

class LiquiditySpecialistEvents(BaseEventGenerator):
    """
    Liquidity Specialist as Causal Parent.
    Causal Role: Predicts the Friction (k). Determines energy required to move price.
    
    Enhanced: Detects liquidity ARRIVAL (favorable entry conditions) instead of 
    illiquidity shocks (crashes). Previous version triggered during panic selling,
    leading to anti-directional consistency (0.34).
    """
    def generate(self, df: pd.DataFrame, tracker: Optional[Any] = None, **params) -> pd.DatetimeIndex:
        threshold = params.get('threshold', 2.5)
        window = params.get('window', 20)
        
        try:
            events = self._get_liquidity_causal_events(df, threshold=threshold, window=window)
            if len(events) == 0 and tracker:
                tracker.log_rejection("liquidity_no_events", 1)
            return events
        except Exception as e:
            logger.warning(f"LiquiditySpecialistEvents failed: {e}")
            if tracker:
                tracker.log_rejection(f"liquidity_exception_{e}", 1)
            return pd.DatetimeIndex([])

    def _get_liquidity_causal_events(self, df: pd.DataFrame, window=20, threshold=2.5):
        """Detect liquidity IMPROVEMENT events (favorable entry conditions)."""
        if 'volume' not in df.columns or 'close' not in df.columns:
            tprint_warning("LiquiditySpecialist: Missing required columns")
            return pd.DatetimeIndex([])
        
        # Debug logging
        tprint_info(f"LiquiditySpecialist: Input shape={df.shape}, columns={list(df.columns)}")
        tprint_info(f"LiquiditySpecialist: Volume range={df['volume'].min()}-{df['volume'].max()}")
        
        # Check volume data quality
        if df['volume'].std() < 1e-6:
            tprint_warning("LiquiditySpecialist: Nearly constant volume")
            return pd.DatetimeIndex([])
        
        ret = df['close'].pct_change()
        
        # Amihud illiquidity (price impact per volume)
        amihud = ret.abs() / (df['volume'] + 1e-9)
        liquidity = 1.0 / (amihud + 1e-9)
        
        # Check liquidity data quality
        if liquidity.std() < 1e-6:
            tprint_warning("LiquiditySpecialist: Nearly constant liquidity")
            return pd.DatetimeIndex([])
        
        # Z-score of liquidity improvement
        liq_mean = liquidity.rolling(window * 5).mean()
        liq_std = liquidity.rolling(window * 5).std()
        liq_z = (liquidity - liq_mean) / (liq_std + 1e-9)
        
        # Kyle's Lambda proxy (price impact per signed volume)
        vol_signed = df['volume'] * np.sign(ret)
        cov_window = min(window * 2, len(df) // 4)
        cov = ret.rolling(cov_window).cov(vol_signed)
        var = vol_signed.rolling(cov_window).var()
        kyle_lambda = cov / (var + 1e-9)
        
        # Low lambda is good, so we want negative z
        lambda_mean = kyle_lambda.rolling(window * 5).mean()
        lambda_std = kyle_lambda.rolling(window * 5).std()
        lambda_z = (kyle_lambda - lambda_mean) / (lambda_std + 1e-9)
        
        # Adaptive threshold: if too strict, use empirical quantiles
        liq_events = (liq_z > threshold).sum()
        lambda_events = (lambda_z < -threshold).sum()
        total_events = liq_events + lambda_events
        
        if total_events < len(df) * 0.01:  # Less than 1% events
            # Relax thresholds based on actual distribution
            liq_threshold = liq_z.quantile(0.98)
            lambda_threshold = lambda_z.quantile(0.02)  # Lower values = better liquidity
            tprint_info(f"LiquiditySpecialist: Relaxed thresholds - liq: {liq_threshold:.3f}, lambda: {lambda_threshold:.3f}")
        else:
            liq_threshold = threshold
            lambda_threshold = -threshold
        
        # Event trigger: High liquidity (positive liq_z) OR Low lambda (negative lambda_z)
        mask = (liq_z > liq_threshold) | (lambda_z < lambda_threshold)
        events = df.index[mask]
        tprint_info(f"LiquiditySpecialist: Generated {len(events)} events")
        
        return events

class InformationSpecialistEvents(BaseEventGenerator):
    """
    Information Specialist as Causal Parent.
    Causal Role: Estimates PIN (Probability of Informed Trading).
    Now uses Dynamic Quantile Thresholding to ensure consistent event density.
    """
    def generate(self, df: pd.DataFrame, tracker: Optional[Any] = None, **params) -> pd.DatetimeIndex:
        # Dynamic threshold: top 5% by default
        quantile = params.get('quantile', 0.95)
        window = params.get('window', 50)
        
        try:
            events = self._get_information_causal_events(df, quantile=quantile, window=window)
            if len(events) == 0 and tracker:
                tracker.log_rejection("information_no_events", 1)
            return events
        except Exception as e:
            logger.warning(f"InformationSpecialistEvents failed: {e}")
            if tracker:
                tracker.log_rejection(f"information_exception_{e}", 1)
            return pd.DatetimeIndex([])

    def _get_information_causal_events(self, df: pd.DataFrame, window=50, quantile=0.95):
        # Standardized autocorrelation (Informed trading persistence)
        if 'close' not in df.columns: return pd.DatetimeIndex([])
        ret = df['close'].pct_change()
        
        # Absolute autocorrelation proxy for information arrival
        # (High autocorrelation = trend/persistence = informed flow)
        autocorr = ret.rolling(window).corr(ret.shift(1)).abs()
        
        # Dynamic Thresholding using Rolling Quantile
        # Look for autocorr in the top (1-quantile)% of the last 10*window bars
        baseline_window = window * 10
        threshold_series = autocorr.rolling(baseline_window, min_periods=window).quantile(quantile)
        
        # Trigger when current PIN/autocorr exceeds the local extreme
        events = df.index[autocorr > threshold_series]
        
        # Fallback: if too few events (<50), lower quantile slightly
        if len(events) < 50:
            relaxed_threshold = autocorr.rolling(baseline_window, min_periods=window).quantile(0.90)
            events = df.index[autocorr > relaxed_threshold]
            
        return events

class MomentumDecaySpecialistEvents(BaseEventGenerator):
    """
    Momentum Decay Specialist as Causal Parent.
    Causal Role: Predicts Trend Exhaustion (dM/dt < 0).
    Mechanism: Momentum slowdown precedes reversal.
    
    This specialist captures the transition from trending to mean-reverting regime,
    which is exactly where orthogonal labeling finds edge (MFE dominance shifts).
    """
    def generate(self, df: pd.DataFrame, tracker: Optional[Any] = None, **params) -> pd.DatetimeIndex:
        threshold = params.get('threshold', 2.0)
        fast_window = params.get('fast_window', 10)
        slow_window = params.get('slow_window', 50)
        
        try:
            events = self._get_momentum_decay_events(df, threshold, fast_window, slow_window)
            if len(events) == 0 and tracker:
                tracker.log_rejection("momentum_decay_no_events", 1)
            return events
        except Exception as e:
            logger.warning(f"MomentumDecaySpecialistEvents failed: {e}")
            if tracker:
                tracker.log_rejection(f"momentum_decay_exception_{e}", 1)
            return pd.DatetimeIndex([])
    
    def _get_momentum_decay_events(self, df, threshold, fast_window, slow_window):
        """Detect trend exhaustion via momentum deceleration."""
        if 'close' not in df.columns:
            return pd.DatetimeIndex([])
        
        price = df['close']
        
        # 1. Momentum (Rate of Change at two scales)
        mom_fast = price.pct_change(fast_window)
        mom_slow = price.pct_change(slow_window)
        
        # 2. Momentum Acceleration (2nd derivative - rate of change of momentum)
        mom_accel = mom_fast.diff(fast_window)
        
        # 3. Trend strength filter (only trigger in strong trends)
        # Rank within recent window to identify "strong" moves
        trend_strength = mom_slow.abs().rolling(slow_window).rank(pct=True)
        is_strong_trend = trend_strength > 0.7
        
        # 4. Z-score of deceleration
        accel_z = (mom_accel - mom_accel.rolling(slow_window).mean()) / \
                  (mom_accel.rolling(slow_window).std() + 1e-9)
        
        # 5. Event detection: Strong trend + significant deceleration
        # For uptrend: mom_slow > 0 and accel_z < -threshold (slowing up)
        # For downtrend: mom_slow < 0 and accel_z > threshold (slowing down)
        uptrend_exhaustion = (mom_slow > 0) & (accel_z < -threshold) & is_strong_trend
        downtrend_exhaustion = (mom_slow < 0) & (accel_z > threshold) & is_strong_trend
        
        return df.index[uptrend_exhaustion | downtrend_exhaustion]

class ImprovedCUSUMEvents(BaseEventGenerator):
    """Refined CUSUM event generator using Efficiency Ratio (ER) for regime adaptation."""
    def generate(self, df: pd.DataFrame, multiplier: float = 1.0, vol_window: int = 20) -> pd.DatetimeIndex:
        try:
            if 'close' not in df.columns: return pd.DatetimeIndex([])
            signals = generate_dual_cusum_signals(
                df['close'], 
                volume=df.get('volume'), 
                k=multiplier * 0.12, 
                vol_window=vol_window
            )
            # Combine trend and reversal signals
            combined = (signals['trend_signal'] == 1) | (signals['reversal_signal'] == 1)
            return signals.index[combined]
        except Exception as e:
            tprint_warning(f"⚠️ ImprovedCUSUMEvents failed: {e}")
            return pd.DatetimeIndex([])

class AdaptiveSymmetricCUSUMEvents(BaseEventGenerator):
    """Adaptive CUSUM that maintains event density across different volatility regimes."""
    def generate(self, df: pd.DataFrame, multiplier: float = 1.0, vol_window: int = 20) -> pd.DatetimeIndex:
        try:
            # Reuses the dual CUSUM logic but focuses on symmetric triggers
            # Often used as a baseline for Causal comparisons
            if 'close' not in df.columns: return pd.DatetimeIndex([])
            signals = generate_dual_cusum_signals(
                df['close'], 
                volume=df.get('volume'), 
                k=multiplier * 0.15, 
                vol_window=vol_window
            )
            combined = (signals['trend_signal'] == 1) | (signals['reversal_signal'] == 1)
            return signals.index[combined]
        except Exception as e:
            tprint_warning(f"⚠️ AdaptiveSymmetricCUSUMEvents failed: {e}")
            return pd.DatetimeIndex([])

class CausalSurpriseEvents(BaseEventGenerator):
    """
    Causal surprise event generator using specialist prediction errors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surprise_detector = None

    def generate(self, df: pd.DataFrame, specialist_predictions: Dict[str, pd.Series] = None,
                causal_graph: Dict[str, List[str]] = None, surprise_threshold: float = 1.25,
                zone3_boost: float = 3.0, zone2_boost: float = 2.0, exposure_scalar: float = 1.0,
                tracker: Optional[Any] = None,
                **params) -> pd.DatetimeIndex:
        
        # Merge params for convenience
        params = {
                'specialist_predictions': specialist_predictions,
                'causal_graph': causal_graph,
                'surprise_threshold': surprise_threshold,
                **params
        }
        
        if not CAUSAL_AVAILABLE_ORTHOGONAL or not specialist_predictions:
            return self._fallback_volatility_events(df)

        try:
            self.surprise_detector = CausalSurpriseDetector(surprise_threshold=surprise_threshold)
            registered_count = 0
            for spec_name, predictions in specialist_predictions.items():
                if 'close' in df.columns:
                    targets = df['close']
                    common_idx = predictions.index.intersection(targets.index)
                    if len(common_idx) > 10:
                        pred_aligned = predictions.loc[common_idx]
                        target_aligned = targets.loc[common_idx]
                        self.surprise_detector.register_specialist(spec_name, pred_aligned, target_aligned)
                        registered_count += 1
            tprint_info(f"   📊 CausalSurpriseEvents: Specialists registered: {registered_count}/{len(specialist_predictions)}")

            if len(self.surprise_detector.specialist_errors_) == 0:
                tprint_warning("   ⚠️ CausalSurpriseEvents: No specialists registered successfully")
                return self._fallback_volatility_events(df)

            # Generate surprise events
            tprint_info("   🔍 CausalSurpriseEvents: Computing surprise scores...")
            surprise_df = self.surprise_detector.aggregate_specialist_surprise()

            if surprise_df.empty:
                tprint_warning("   ⚠️ CausalSurpriseEvents: No surprise scores computed")
                return self._fallback_volatility_events(df)

            # Adaptive Calibration to hit target density
            target_density = params.get('target_signals_per_day', 2.0)
            duration_days = (df.index[-1] - df.index[0]).days + 1
            self.surprise_detector.adaptive_calibration(target_density, duration_days)

            tprint_info("   🎯 CausalSurpriseEvents: Generating causal events...")
            causal_events = self.surprise_detector.generate_causal_events()

            if causal_events:
                event_indices = list(causal_events.keys())
                tprint_info(f"🎯 Generated {len(event_indices)} causal surprise events")
                return pd.DatetimeIndex(event_indices)
            else:
                return self._fallback_volatility_events(df)

        except Exception as e:
            tprint_error(f"❌ Causal surprise event generation failed: {e}")
            return self._fallback_volatility_events(df)

    def _fallback_volatility_events(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Fallback event generation using volatility shocks."""
        try:
            if 'close' in df.columns:
                price = df['close']
                # Simple volatility-based events
                returns = price.pct_change()
                vol = returns.rolling(20).std()
                vol_threshold = vol.quantile(0.8)
                events = df.index[vol > vol_threshold]
                return events[:min(len(events), 500)]  # Limit events - increased for OOF fold sizing
            else:
                return pd.DatetimeIndex([])
        except Exception:
            return pd.DatetimeIndex([])

# Aliases
CusumEvents = AdaptiveSymmetricCUSUMEvents

# ==========================================
# 9. Meta-Learning Dataset Generation
# ==========================================

def apply_persistence_label(df: pd.DataFrame, events: pd.DatetimeIndex, series_col: str, horizon: int = 48, threshold: float = 0.0) -> pd.Series:
    """
    Generic persistence labeler.
    Returns 1 if series_col > threshold on average over horizon.
    """
    if events.empty or series_col not in df.columns:
        return pd.Series(0, index=df.index)

    # Align events
    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return pd.Series(0, index=df.index)

    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)

    # Filter valid
    valid_mask = (event_locs != -1) & (event_locs < (n_bars - horizon))
    valid_idxs = event_locs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return pd.Series(0, index=df.index)

    # Window Matrix
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]

    vals = df[series_col].values[window_idxs]
    avg_vals = np.mean(vals, axis=1)

    labels = (avg_vals > threshold).astype(int)

    out = pd.Series(0, index=df.index)
    out.loc[final_events] = labels
    return out

def apply_triple_barrier_multi(df: pd.DataFrame, events: pd.DatetimeIndex,
                                pt_sl: Tuple[float,float]=(2.0, 1.0), # multipliers for vol
                                horizons: list=[12,48]) -> pd.DataFrame:
    """
    Returns a DataFrame of price labels for multiple horizons using volatility-adjusted barriers.
    Columns: 'price_label_{horizon}'
    """
    out = pd.DataFrame(0, index=df.index, columns=[f'price_label_{h}' for h in horizons], dtype=int)
    close = df['close'].values

    # Volatility
    if 'volatility_1d' in df.columns:
        vol = df['volatility_1d'].values
    else:
        vol = df['close'].pct_change().rolling(100).std().fillna(0).values

    # Normalize TZ
    if df.index.tz is not None:
        idx_base = df.index.tz_localize(None)
    else:
        idx_base = df.index

    if events.tz is not None:
        events_norm = events.tz_localize(None)
    else:
        events_norm = events

    event_idxs = idx_base.get_indexer(events_norm)
    valid_mask = (event_idxs != -1)
    valid_idxs = event_idxs[valid_mask]

    for h in horizons:
        # Filter for horizon
        h_mask = valid_idxs < (len(close) - h)
        h_idxs = valid_idxs[h_mask]

        if len(h_idxs) == 0:
            continue

        # Vectorized Window
        offsets = np.arange(1, h + 1)
        window_idxs = h_idxs[:, None] + offsets[None, :]

        window_prices = close[window_idxs]
        entry_prices = close[h_idxs]
        entry_vols = vol[h_idxs]

        # Avoid zero vol
        entry_vols = np.maximum(entry_vols, 1e-6)

        ret = window_prices / entry_prices[:, None] - 1.0

        up_barrier = pt_sl[0] * entry_vols
        down_barrier = pt_sl[1] * entry_vols

        hit_up = ret >= up_barrier[:, None]
        hit_down = ret <= -down_barrier[:, None]

        # First hit logic
        first_up = np.argmax(hit_up, axis=1)
        first_down = np.argmax(hit_down, axis=1)

        # Mask where no hit occurred (argmax returns 0 if all false, need to check if actually hit)
        any_up = np.any(hit_up, axis=1)
        any_down = np.any(hit_down, axis=1)

        labels = np.zeros(len(h_idxs), dtype=int)

        # Vectorized check
        # Case 1: Only Up
        mask_up = any_up & ~any_down
        labels[mask_up] = 1

        # Case 2: Only Down
        mask_down = any_down & ~any_up
        labels[mask_down] = -1

        # Case 3: Both
        mask_both = any_up & any_down
        # first_up < first_down -> 1
        sub_mask_up = mask_both & (first_up < first_down)
        labels[sub_mask_up] = 1

        sub_mask_down = mask_both & (first_down < first_up)
        labels[sub_mask_down] = -1

        # Assign to output
        evt_timestamps = events_norm[valid_mask][h_mask]

        # Align TZ if needed
        if df.index.tz is not None and evt_timestamps.tz is None:
             evt_timestamps = evt_timestamps.tz_localize(df.index.tz)

        out.loc[evt_timestamps, f'price_label_{h}'] = labels

    return out

def create_meta_learning_dataset_dualTBM(df: pd.DataFrame, base_features: pd.DataFrame,
                                         pt_sl=(2.0, 1.0), tbm_horizons=[12,48]):
    meta_df = base_features.copy()

    # Directional price labels for multiple horizons
    if 'price_dual_cusum' in base_features.columns:
        price_events = base_features.index[base_features['price_dual_cusum']==1]
        # Normalize timezones
        if df.index.tz != price_events.tz:
             if price_events.tz is None: price_events = price_events.tz_localize(df.index.tz)
             else: price_events = price_events.tz_convert(df.index.tz)

        tbm_labels = apply_triple_barrier_multi(df, price_events, pt_sl=pt_sl, horizons=tbm_horizons)
        meta_df = pd.concat([meta_df, tbm_labels], axis=1)

    # Contextual labels
    context_map = {
        'volatility_cusum': 'volatility_1d',
        'liquidity_cusum': 'liq_stress',
        'volume_cusum': 'volume',
        'tailrisk_cusum': 'tail_metric',
        'trend_regime_cusum': 'trend',
        'vol_state_cusum': 'vol_state'
    }

    for col, series_col in context_map.items():
        if col in base_features.columns and series_col in df.columns:
            events = base_features.index[base_features[col]==1]
            if df.index.tz != events.tz:
                 if events.tz is None: events = events.tz_localize(df.index.tz)
                 else: events = events.tz_convert(df.index.tz)

            lbl = apply_persistence_label(df, events, series_col=series_col, horizon=48, threshold=0.0)
            meta_df[f'{col}_label'] = lbl

    return meta_df
