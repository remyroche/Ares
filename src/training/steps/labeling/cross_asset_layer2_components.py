"""Cross-asset Layer2 components (panel store, MSV, gating, validation).

Optimized version with:
- Numba JIT for rolling calculations
- Vectorized operations (no iterrows)
- Periodic refit calibration (not per-row)
- LRU caching for expensive computations
- float32 memory optimization
- Pre-allocated DataFrames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from functools import lru_cache
import hashlib
import os

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover
    sm = None
    QuantReg = None
    STATSMODELS_AVAILABLE = False

try:
    from src.utils.fracdiff import FracDiffTransformer
    FRACDIFF_AVAILABLE = True
except Exception:
    FracDiffTransformer = None
    FRACDIFF_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.numba_funcs import (
    _numba_rolling_mean,
    _numba_rolling_std,
    _numba_rolling_correlation,
    _numba_rolling_cov,
    _numba_ewma,
    _numba_rolling_sum
)

NAMESPACE_PREFIXES = ("raw__", "y__", "sa__", "cs__", "ca__", "ms__", "gate__")

# ============================================================================
# Numba-optimized helper functions
# ============================================================================

@jit(nopython=True)
def _numba_rolling_closed_left(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling mean and std with closed-left window (shift by 1).
    Returns (rolling_mean, rolling_std) as float32 arrays.
    """
    n = len(x)
    out_mean = np.zeros(n, dtype=np.float32)
    out_std = np.zeros(n, dtype=np.float32)
    
    if window <= 0 or n <= 1:
        return out_mean, out_std
    
    min_periods = max(2, window // 4)
    s = 0.0
    ss = 0.0
    count = 0
    
    for i in range(1, n):  # Start from 1 (shift by 1)
        # Add element at i-1 (closed left)
        v = x[i - 1]
        if not np.isnan(v):
            s += v
            ss += v * v
            count += 1
        
        # Remove element leaving the window
        if i > window:
            old_v = x[i - window - 1]
            if not np.isnan(old_v):
                s -= old_v
                ss -= old_v * old_v
                count -= 1
        
        if count >= min_periods:
            mean = s / count
            var = (ss / count) - (mean * mean)
            out_mean[i] = mean
            out_std[i] = np.sqrt(max(0.0, var))
    
    return out_mean, out_std


@jit(nopython=True)
def _numba_quantile_partition(arr: np.ndarray, lower: float, upper: float) -> Tuple[float, float]:
    """Compute quantiles with NaN handling.
    
    For small arrays uses sort, for larger could use partition.
    """
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    
    # Filter out NaNs
    valid = np.zeros(n, dtype=np.float64)
    valid_count = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            valid[valid_count] = arr[i]
            valid_count += 1
    
    if valid_count == 0:
        return 0.0, 0.0
    
    valid = valid[:valid_count]
    sorted_arr = np.sort(valid)
    
    low_idx = int(lower * valid_count)
    high_idx = int(upper * valid_count) - 1
    low_idx = max(0, min(low_idx, valid_count - 1))
    high_idx = max(0, min(high_idx, valid_count - 1))
    
    return sorted_arr[low_idx], sorted_arr[high_idx]


@jit(nopython=True)
def _numba_winsorize(arr: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize array, returning clipped copy."""
    n = len(arr)
    if n == 0:
        return arr.copy()
    
    low_val, high_val = _numba_quantile_partition(arr, lower, upper)
    
    out = arr.copy()
    for i in range(n):
        if not np.isnan(out[i]):
            if out[i] < low_val:
                out[i] = low_val
            elif out[i] > high_val:
                out[i] = high_val
    return out


@jit(nopython=True)
def _numba_compute_vpin(close: np.ndarray, high: np.ndarray, low: np.ndarray, 
                        volume: np.ndarray, bucket_size: int) -> np.ndarray:
    """Compute VPIN using Numba for speed."""
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float32)
    
    if n < bucket_size:
        return out
    
    # Pre-compute bar range and close position
    imbalance = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bar_range = high[i] - low[i]
        if bar_range < 1e-9:
            bar_range = 1e-9
        close_pos = (close[i] - low[i]) / bar_range
        buy_vol = close_pos * volume[i]
        sell_vol = (1.0 - close_pos) * volume[i]
        imbalance[i] = abs(buy_vol - sell_vol)
    
    # Rolling sums
    imb_sum = 0.0
    vol_sum = 0.0
    
    for i in range(n):
        imb_sum += imbalance[i]
        vol_sum += volume[i]
        
        if i >= bucket_size:
            imb_sum -= imbalance[i - bucket_size]
            vol_sum -= volume[i - bucket_size]
        
        if i >= bucket_size - 1 and vol_sum > 1e-9:
            out[i] = imb_sum / vol_sum
    
    return out


@jit(nopython=True)
def _numba_batch_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute correlation of each column in X with y. Returns 1D array of correlations."""
    n_rows, n_cols = X.shape
    corrs = np.zeros(n_cols, dtype=np.float32)
    
    # Precompute y stats
    y_mean = 0.0
    y_ss = 0.0
    y_count = 0
    for i in range(n_rows):
        if not np.isnan(y[i]):
            y_mean += y[i]
            y_count += 1
    if y_count == 0:
        return corrs
    y_mean /= y_count
    
    for i in range(n_rows):
        if not np.isnan(y[i]):
            y_ss += (y[i] - y_mean) ** 2
    y_std = np.sqrt(y_ss / y_count) if y_count > 0 else 0.0
    
    if y_std < 1e-9:
        return corrs
    
    for col in range(n_cols):
        x_mean = 0.0
        x_count = 0
        for i in range(n_rows):
            if not np.isnan(X[i, col]) and not np.isnan(y[i]):
                x_mean += X[i, col]
                x_count += 1
        if x_count == 0:
            continue
        x_mean /= x_count
        
        x_ss = 0.0
        xy_sum = 0.0
        for i in range(n_rows):
            if not np.isnan(X[i, col]) and not np.isnan(y[i]):
                x_ss += (X[i, col] - x_mean) ** 2
                xy_sum += (X[i, col] - x_mean) * (y[i] - y_mean)
        
        x_std = np.sqrt(x_ss / x_count) if x_count > 0 else 0.0
        if x_std > 1e-9:
            corrs[col] = xy_sum / (x_count * x_std * y_std)
    
    return corrs


@jit(nopython=True)
def _numba_entropy_histogram(vals: np.ndarray, n_bins: int = 10) -> float:
    """Compute Shannon entropy using histogram binning (proper for percentiles).
    
    This is the correct way to compute entropy on continuous values like percentiles.
    Uses histogram binning instead of treating values as probabilities directly.
    """
    n = len(vals)
    if n == 0:
        return 0.0
    
    # Build histogram
    hist = np.zeros(n_bins, dtype=np.float64)
    for i in range(n):
        if not np.isnan(vals[i]):
            # Clamp to [0, 1] range for percentiles
            v = max(0.0, min(1.0, vals[i]))
            bin_idx = int(v * n_bins)
            bin_idx = min(bin_idx, n_bins - 1)
            hist[bin_idx] += 1.0
    
    # Normalize to probabilities
    total = hist.sum()
    if total < 1e-9:
        return 0.0
    
    entropy = 0.0
    for i in range(n_bins):
        if hist[i] > 0:
            p = hist[i] / total
            entropy -= p * np.log(p + 1e-9)
    
    return entropy


@jit(nopython=True)
def _numba_effective_n_bets(weights: np.ndarray) -> float:
    """Compute effective number of bets: 1 / sum(w^2).
    
    This is a better concentration measure than entropy for portfolio weights.
    Returns N if weights are uniform, 1 if concentrated in single asset.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Normalize weights to sum to 1
    total = 0.0
    for i in range(n):
        if not np.isnan(weights[i]):
            total += abs(weights[i])
    
    if total < 1e-9:
        return 0.0
    
    sum_sq = 0.0
    for i in range(n):
        if not np.isnan(weights[i]):
            w = abs(weights[i]) / total
            sum_sq += w * w
    
    if sum_sq < 1e-9:
        return float(n)
    
    return 1.0 / sum_sq



# ============================================================================
# Caching utilities
# ============================================================================

def _hash_array(arr: np.ndarray) -> str:
    """Create a hash for numpy array for caching purposes."""
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _hash_dataframe_index(df: pd.DataFrame) -> str:
    """Create a hash for DataFrame index."""
    return hashlib.md5(str(df.index.values).encode()).hexdigest()[:16]


def compute_vpin(df: pd.DataFrame, volume_bucket_size: int = 50) -> pd.Series:
    """Compute VPIN using bar close position as buy/sell proxy (Numba-optimized)."""
    close = df.get("close")
    high = df.get("high")
    low = df.get("low")
    volume = df.get("volume")

    if close is None or volume is None or high is None or low is None:
        tprint_warning("[compute_vpin] missing required columns")
        return pd.Series(0.5, index=df.index, dtype=np.float32)

    # Use Numba-optimized version
    close_arr = close.values.astype(np.float64)
    high_arr = high.values.astype(np.float64)
    low_arr = low.values.astype(np.float64)
    vol_arr = volume.fillna(0.0).values.astype(np.float64)
    
    vpin_arr = _numba_compute_vpin(close_arr, high_arr, low_arr, vol_arr, volume_bucket_size)
    return pd.Series(vpin_arr, index=df.index, dtype=np.float32)


@dataclass
class SchemaValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class MarketStateConfig:
    state_instruments: List[str]
    n_components: int = 4
    stability_threshold: float = 0.9
    clustering_method: str = "gmm"
    use_incremental_pca: bool = False  # For streaming updates
    update_frequency: int = 100  # How often to refit PCA/Clustering


@dataclass
class ValidationResult:
    split_name: str
    metrics: Dict[str, float]
    by_asset: Dict[str, Dict[str, float]]
    by_sector: Dict[str, Dict[str, float]]
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "metrics": self.metrics,
            "by_asset": self.by_asset,
            "by_sector": self.by_sector,
            "artifacts": self.artifacts,
        }


@dataclass
class InvarianceReport:
    dispersion: float
    worst_env_pair: Tuple[str, str]
    worst_distance: float
    per_feature_grad_var: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispersion": self.dispersion,
            "worst_env_pair": list(self.worst_env_pair),
            "worst_distance": self.worst_distance,
            "per_feature_grad_var": self.per_feature_grad_var.to_dict(),
        }


@dataclass
class GatingConfig:
    tail_quantile: float = 0.05
    min_tail_sample: int = 30
    beta_exposure_cap: float = 1.5
    max_correlation: float = 0.85
    persistence_bars: int = 3  # Gate must fail N consecutive bars to trigger


class PanelFeatureStore:
    """Immutable panel feature store keyed by (timestamp, ticker).
    
    Optimized to avoid unnecessary copies where possible.
    """

    def __init__(self, panel_df: pd.DataFrame, _skip_copy: bool = False):
        tprint_info("[PanelFeatureStore] init")
        # Allow skipping copy for internal use when we know the df is fresh
        self._panel_df = panel_df if _skip_copy else panel_df.copy()
        if not isinstance(self._panel_df.index, pd.MultiIndex):
            raise ValueError("panel_df must have MultiIndex (timestamp, ticker)")

    def add_features(self, features: pd.DataFrame, prefix: str, allow_overwrite: bool = False) -> "PanelFeatureStore":
        tprint_info(
            f"[PanelFeatureStore] add_features start prefix={prefix} panel_shape={self._panel_df.shape}"
        )
        if not prefix.endswith("__"):
            raise ValueError("prefix must end with '__'")
        if prefix not in NAMESPACE_PREFIXES:
            raise ValueError(f"Unsupported prefix {prefix}. Allowed: {NAMESPACE_PREFIXES}")
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("features must use MultiIndex (timestamp, ticker)")
        if not features.index.equals(self._panel_df.index):
            features = features.reindex(self._panel_df.index)

        # Rename columns in-place on a view to avoid copy
        new_cols = [c if c.startswith(prefix) else f"{prefix}{c}" for c in features.columns]
        overlap = set(new_cols).intersection(self._panel_df.columns)
        if overlap and not allow_overwrite:
            raise ValueError(f"Attempt to overwrite existing columns: {sorted(overlap)[:5]}")

        # Create merged DataFrame efficiently
        merged = self._panel_df.copy()
        for old_col, new_col in zip(features.columns, new_cols):
            merged[new_col] = features[old_col]
        
        tprint_success(
            f"[PanelFeatureStore] add_features done added_cols={len(new_cols)}"
        )
        return PanelFeatureStore(merged, _skip_copy=True)

    @property
    def data(self) -> pd.DataFrame:
        return self._panel_df.copy()
    
    @property
    def data_view(self) -> pd.DataFrame:
        """Return a view (no copy) for read-only operations."""
        return self._panel_df


class PanelDataProcessor:
    """Builds immutable panel data and enforces naming/validation contracts.
    
    Optimized with:
    - Numba JIT for rolling calculations
    - Pre-allocated DataFrames
    - float32 memory optimization
    """

    def __init__(
        self,
        vol_window: int = 20,
        dvol_window: int = 20,
        zscore_window: int = 50,
        enable_zscore: bool = True,
        enable_fracdiff: bool = True,
        fracdiff_d: float = 0.4,
        fracdiff_min_periods: int = 200,
        fracdiff_mode: str = "fixed",
        fracdiff_tolerance: float = 0.01,
    ):
        tprint_info("[PanelDataProcessor] init")
        self.vol_window = vol_window
        self.dvol_window = dvol_window
        self.zscore_window = zscore_window
        self.enable_zscore = enable_zscore
        self.enable_fracdiff = enable_fracdiff
        self.fracdiff_d = fracdiff_d
        self.fracdiff_min_periods = fracdiff_min_periods
        self.fracdiff_mode = fracdiff_mode
        self.fracdiff_tolerance = fracdiff_tolerance
        # Cache for price column resolution
        self._price_col_cache: Dict[Tuple[str, ...], Optional[str]] = {}

    def fit(self, single_asset_data: Dict[str, pd.DataFrame]) -> "PanelDataProcessor":
        tprint_info(f"[PanelDataProcessor] fit start assets={len(single_asset_data)}")
        # No-op fit - schema validation happens in transform
        tprint_success(f"[PanelDataProcessor] fit done")
        return self

    def fit_transform(self, single_asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        tprint_info("[PanelDataProcessor] fit_transform start")
        self.fit(single_asset_data)
        panel = self.transform_to_panel(single_asset_data)
        tprint_success("[PanelDataProcessor] fit_transform done")
        return panel

    def transform_to_panel(self, single_asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        tprint_info(f"[PanelDataProcessor] transform_to_panel start assets={len(single_asset_data)}")
        if not single_asset_data:
            raise ValueError("single_asset_data cannot be empty")

        timestamps = None
        standardized: Dict[str, pd.DataFrame] = {}
        for ticker, df in single_asset_data.items():
            if df is None or df.empty:
                tprint_warning(f"[PanelDataProcessor] Empty data for {ticker}, skipping")
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{ticker} data must have DatetimeIndex")
            df = df.sort_index().loc[~df.index.duplicated(keep="last")].copy()
            # Remove any NaT indices which can break reindex
            df = df.loc[df.index.notnull()]
            df.columns = [c.lower() for c in df.columns]
            df = self._ensure_unique_columns(df, ticker)
            standardized[ticker] = df
            if timestamps is None:
                timestamps = df.index
            else:
                timestamps = timestamps.union(df.index)

        if timestamps is None:
            raise ValueError("No valid ticker data provided")

        # Ensure alignment index has unique, sorted timestamps to avoid reindex errors
        # union() already returns sorted unique index for DatetimeIndex, but we'll be extra safe
        timestamps = timestamps.dropna().unique().sort_values()

        panel_frames = []
        for ticker, df in standardized.items():
            # Robust Alignment with Tolerance (60m)
            # Price columns (State): forward fill with tolerance
            # Volume columns (Flow): zero fill
            price_like_cols = [c for c in df.columns if c in ("open", "high", "low", "close", "vwap") or "px" in c]
            volume_like_cols = [c for c in df.columns if c not in price_like_cols]

            df_prices = df[price_like_cols]
            df_vols = df[volume_like_cols]

            # Use 15m tolerance to allow for reasonable gaps in dollar bars but prevent infinite staleness
            try:
                # Ensure df_prices is sorted and unique before reindex
                df_prices = df_prices.sort_index()
                df_prices = df_prices.loc[~df_prices.index.duplicated(keep='last')]
                df_prices = df_prices.loc[df_prices.index.notnull()]
                
                aligned_prices = df_prices.reindex(timestamps, method='ffill', tolerance=pd.Timedelta('15m'))
            except Exception as e:
                tprint_warning(f"[PanelDataProcessor] Reindex failed for {ticker}: {e}. Attempting fallback with exact reindex.")
                # Fallback: reindex without method if ffill fails
                aligned_prices = df_prices.reindex(timestamps).ffill(limit=1)
            
            # Split volume-like columns into numeric and non-numeric to avoid type errors
            # (e.g. filling string columns with 0)
            vol_numeric = df_vols.select_dtypes(include=[np.number]).columns
            vol_other = df_vols.columns.difference(vol_numeric)
            
            aligned_vols_num = df_vols[vol_numeric].reindex(timestamps, fill_value=0)
            aligned_vols_other = df_vols[vol_other].reindex(timestamps) # Default fill is NaN
            
            aligned_vols = pd.concat([aligned_vols_num, aligned_vols_other], axis=1)
            
            aligned = pd.concat([aligned_prices, aligned_vols], axis=1)

            price_col = self._resolve_price_column(aligned)
            if price_col is None:
                raise ValueError(f"{ticker} missing price column")

            base = pd.DataFrame(index=timestamps)
            # Use aligned price (already ffilled with tolerance)
            base["raw__px"] = aligned[price_col].replace([np.inf, -np.inf], np.nan)
            base.loc[base["raw__px"] <= 0, "raw__px"] = np.nan
            
            # Fill remaining NaNs in raw__px (due to tolerance) if necessary?
            # Ideally we keep them NaN to indicate missing data, but downstream might expect valid prices.
            # However, for 'Raw Price' features, NaNs might propagate. 
            # Given Panel logic usually expects dense, we might fall back to infinite ffill ONLY if strictly needed,
            # but user asked for "margin". So we respect the NaN if it's stale.
            # But "raw__px" is base for everything. If it's NaN, returns will be NaN.
            
            for col in ("open", "high", "low", "close", "volume"):
                if col in aligned.columns:
                    base[f"raw__{col}"] = aligned[col]
            
            if "raw__volume" in base.columns:
                base["raw__volume"] = base["raw__volume"].fillna(0.0)

            base["raw__log_px"] = np.log(base["raw__px"]).ffill()
            returns = base["raw__px"].pct_change().replace([np.inf, -np.inf], np.nan)
            base["y__ret_1"] = returns.shift(-1)

            vol = self._rolling_closed_left(returns, self.vol_window).std()
            base["raw__vol"] = vol
            dvol = self._rolling_closed_left(vol, self.dvol_window).mean()
            base["raw__dvol"] = dvol

            if self.enable_zscore:
                ret_mean = self._rolling_closed_left(returns, self.zscore_window).mean()
                ret_std = self._rolling_closed_left(returns, self.zscore_window).std()
                base["raw__ret_zscore"] = (returns - ret_mean) / (ret_std + 1e-9)

                log_px = base["raw__log_px"]
                lp_mean = self._rolling_closed_left(log_px, self.zscore_window).mean()
                lp_std = self._rolling_closed_left(log_px, self.zscore_window).std()
                base["raw__log_px_zscore"] = (log_px - lp_mean) / (lp_std + 1e-9)

                vol_mean = self._rolling_closed_left(vol, self.zscore_window).mean()
                vol_std = self._rolling_closed_left(vol, self.zscore_window).std()
                base["raw__vol_zscore"] = (vol - vol_mean) / (vol_std + 1e-9)

            if self.enable_fracdiff:
                if not FRACDIFF_AVAILABLE:
                    tprint_warning("[PanelDataProcessor] FracDiff unavailable; skipping fracdiff features")
                elif len(base) < self.fracdiff_min_periods:
                    tprint_warning(
                        f"[PanelDataProcessor] {ticker} insufficient history for fracdiff: {len(base)} rows"
                    )
                else:
                    try:
                        transformer = FracDiffTransformer()
                        log_px = base["raw__log_px"].ffill()
                        if self.fracdiff_mode == "adf":
                            _ = transformer.find_optimal_d(log_px, method="binary_search", tolerance=self.fracdiff_tolerance)
                            fracdiff_series = transformer.transform(log_px)
                        else:
                            fracdiff_series = transformer.fracdiff(log_px, d=self.fracdiff_d, drop_na=False)
                        base["raw__fracdiff_log_px"] = fracdiff_series
                    except Exception as e:
                        tprint_warning(f"[PanelDataProcessor] FracDiff failed for {ticker}: {e}")
                        base["raw__fracdiff_log_px"] = np.nan

            passthrough_cols = [
                col
                for col in aligned.columns
                if col not in {price_col, "open", "high", "low", "close", "volume", "timestamp", "ticker"}
            ]
            for col in passthrough_cols:
                if col in base.columns:
                    continue
                base[col] = aligned[col]

            if "timestamp" in base.columns:
                base = base.drop(columns=["timestamp"])
            if "ticker" in base.columns:
                base = base.drop(columns=["ticker"])
            base["ticker"] = ticker
            base.index.name = "timestamp"
            base = base.reset_index().set_index(["timestamp", "ticker"]).sort_index()
            panel_frames.append(base)

        panel_df = pd.concat(panel_frames).sort_index()
        panel_df = self.enforce_prefix_namespacing(panel_df)
        tprint_success(f"[PanelDataProcessor] transform_to_panel done shape={panel_df.shape}")
        return panel_df

    def _ensure_unique_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Deduplicate column labels to avoid pandas reindex errors on duplicate axes."""
        if df.columns.is_unique:
            return df

        counts: Dict[str, int] = defaultdict(int)
        new_cols: List[str] = []
        dup_names: List[str] = []
        for col in df.columns:
            count = counts[col]
            if count == 0:
                new_cols.append(col)
            else:
                deduped = f"{col}__dup{count}"
                new_cols.append(deduped)
                dup_names.append(col)
            counts[col] += 1

        if dup_names:
            unique_dups = sorted(set(dup_names))
            summary = unique_dups[:5]
            suffix = "..." if len(unique_dups) > 5 else ""
            tprint_warning(
                f"[PanelDataProcessor] Duplicate columns detected for {ticker}: {summary}{suffix}. "
                "Appended __dupN suffixes to ensure uniqueness."
            )

        deduped_df = df.copy()
        deduped_df.columns = new_cols
        return deduped_df

    def validate_schema(self, panel_df: pd.DataFrame) -> SchemaValidationResult:
        tprint_info(f"[PanelDataProcessor] validate_schema start shape={panel_df.shape}")
        errors: List[str] = []
        warnings: List[str] = []
        if not isinstance(panel_df.index, pd.MultiIndex):
            errors.append("panel_df must have MultiIndex (timestamp, ticker)")
        if panel_df.index.has_duplicates:
            errors.append("panel_df index must be unique")

        required = ["raw__px", "y__ret_1", "raw__vol", "raw__dvol"]
        missing = [c for c in required if c not in panel_df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

        if panel_df.isna().mean().max() > 0.5:
            warnings.append("High NaN ratio detected in panel_df")

        result = SchemaValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)
        tprint_success(
            f"[PanelDataProcessor] validate_schema done ok={result.ok} errors={len(errors)} warnings={len(warnings)}"
        )
        return result

    def enforce_prefix_namespacing(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        tprint_info("[PanelDataProcessor] enforce_prefix_namespacing start")
        renamed = panel_df.copy()
        for col in list(renamed.columns):
            if col.startswith(NAMESPACE_PREFIXES) or col == "ticker":
                continue
            if col.startswith("y_"):
                renamed = renamed.rename(columns={col: col.replace("y_", "y__")})
            elif col.startswith("raw_"):
                renamed = renamed.rename(columns={col: col.replace("raw_", "raw__")})
            else:
                renamed = renamed.rename(columns={col: f"raw__{col}"})
        invalid = [
            col for col in renamed.columns if not col.startswith(NAMESPACE_PREFIXES) and col != "ticker"
        ]
        if invalid:
            raise ValueError(f"Columns missing namespace prefixes: {invalid[:5]}")
        tprint_success("[PanelDataProcessor] enforce_prefix_namespacing done")
        return renamed

    def detect_leakage(self, panel_df: pd.DataFrame, label_col: str = "y__ret_1") -> List[str]:
        """Detect potential data leakage using batch correlation (Numba-optimized)."""
        tprint_info(f"[PanelDataProcessor] detect_leakage start label_col={label_col}")
        warnings: List[str] = []
        if label_col not in panel_df.columns:
            msg = f"Missing label column {label_col}"
            tprint_warning(f"[PanelDataProcessor] detect_leakage {msg}")
            return [msg]

        numeric_cols = [
            c
            for c in panel_df.select_dtypes(include=[np.number]).columns
            if c.startswith(("raw__", "ca__", "ms__"))
        ]
        if not numeric_cols:
            tprint_success(f"[PanelDataProcessor] detect_leakage done warnings={len(warnings)}")
            return warnings

        label = panel_df[label_col].fillna(0.0).values.astype(np.float64)
        label_shifted = np.roll(label, 1)
        label_shifted[0] = 0.0
        
        sampled_cols = numeric_cols[: min(20, len(numeric_cols))]
        
        # Use Numba batch correlation for speed
        X = panel_df[sampled_cols].fillna(0.0).values.astype(np.float64)
        corrs_future = _numba_batch_correlation(X, label)
        corrs_past = _numba_batch_correlation(X, label_shifted)
        
        for i, col in enumerate(sampled_cols):
            if corrs_future[i] > corrs_past[i] + 0.1:
                warnings.append(f"Leakage sentinel: {col} corr_future {corrs_future[i]:.3f} > corr_past {corrs_past[i]:.3f}")

        # Fixed shuffle test: shuffle the features, not the index
        # This properly breaks temporal alignment
        np.random.seed(42)
        X_shuffled = X.copy()
        for col_idx in range(X_shuffled.shape[1]):
            np.random.shuffle(X_shuffled[:, col_idx])
        
        shuffled_corrs = _numba_batch_correlation(X_shuffled, label)
        shuffled_max = np.abs(shuffled_corrs).max()
        original_max = np.abs(corrs_future).max()
        
        if shuffled_max >= original_max * 0.7:
            warnings.append("Timestamp perturbation test: predictability did not collapse")

        tprint_success(f"[PanelDataProcessor] detect_leakage done warnings={len(warnings)}")
        return warnings

    def _resolve_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """Resolve price column with caching."""
        col_tuple = tuple(df.columns)
        if col_tuple in self._price_col_cache:
            return self._price_col_cache[col_tuple]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = None
        for col in ("layer0_price", "close", "px", "price", "last", "settle"):
            if col in numeric_cols:
                result = col
                break
        
        self._price_col_cache[col_tuple] = result
        return result

    def _rolling_closed_left_numba(self, series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """Numba-optimized rolling mean and std with closed-left window."""
        arr = series.fillna(0.0).values.astype(np.float64)
        mean_arr, std_arr = _numba_rolling_closed_left(arr, window)
        return (
            pd.Series(mean_arr, index=series.index, dtype=np.float32),
            pd.Series(std_arr, index=series.index, dtype=np.float32)
        )

    @staticmethod
    def _rolling_closed_left(series: pd.Series, window: int) -> pd.core.window.Rolling:
        """Legacy pandas rolling for compatibility."""
        return series.shift(1).rolling(window=window, min_periods=max(2, window // 4))


class MarketStateVector:
    """Continuous PCA components + discrete regime labels/probabilities.
    
    Optimized with:
    - Optional IncrementalPCA for streaming updates
    - Transform result caching
    - float32 output for memory efficiency
    """

    def __init__(self, config: MarketStateConfig):
        tprint_info("[MarketStateVector] init")
        self.config = config
        self.scaler = StandardScaler()
        
        # Use IncrementalPCA if configured (for streaming/large datasets)
        if config.use_incremental_pca:
            self.pca = IncrementalPCA(n_components=config.n_components)
        else:
            self.pca = PCA(n_components=config.n_components, random_state=42)
        
        self.cluster_model: Optional[Any] = None
        self.loadings_: Optional[np.ndarray] = None
        # Transform cache: (data_hash, result)
        self._transform_cache: Optional[Tuple[str, pd.DataFrame]] = None

    def fit(self, state_instruments: pd.DataFrame) -> "MarketStateVector":
        tprint_info(f"[MarketStateVector] fit start shape={state_instruments.shape}")
        # Only fit on numeric data to safeguard against non-numeric leakage
        x_numeric = state_instruments.select_dtypes(include=[np.number])
        if x_numeric.empty:
            raise ValueError("No numeric data available for MarketStateVector fit")
        x = x_numeric.ffill().fillna(0.0).astype(np.float32)
        scaled = self.scaler.fit_transform(x)
        components = self.pca.fit_transform(scaled)
        self.loadings_ = self.pca.components_.copy()

        if self.config.clustering_method == "gmm":
            self.cluster_model = GaussianMixture(n_components=self.config.n_components, random_state=42)
        else:
            from sklearn.cluster import KMeans
            self.cluster_model = KMeans(n_clusters=self.config.n_components, random_state=42)
        self.cluster_model.fit(components)
        
        # Invalidate cache on refit
        self._transform_cache = None
        tprint_success("[MarketStateVector] fit done")
        return self

    def transform(self, state_instruments: pd.DataFrame) -> pd.DataFrame:
        """Transform with caching for repeated calls on same data."""
        tprint_info(f"[MarketStateVector] transform start shape={state_instruments.shape}")
        if self.loadings_ is None:
            raise RuntimeError("MarketStateVector not fitted")
        
        # Check cache
        cache_key = f"{state_instruments.shape}_{state_instruments.index[0]}_{state_instruments.index[-1]}"
        if self._transform_cache is not None:
            cached_key, cached_result = self._transform_cache
            if cached_key == cache_key:
                tprint_success("[MarketStateVector] transform done (cached)")
                return cached_result
        
        x = state_instruments.ffill().fillna(0.0).astype(np.float32)
        scaled = self.scaler.transform(x)
        components = self.pca.transform(scaled)

        result = pd.DataFrame(index=state_instruments.index)
        for i in range(self.config.n_components):
            result[f"ms__pca_{i}"] = components[:, i].astype(np.float32)

        if hasattr(self.cluster_model, "predict_proba"):
            probs = self.cluster_model.predict_proba(components)
            state_id = probs.argmax(axis=1)
            result["ms__state_id"] = state_id.astype(np.int8)
            for i in range(probs.shape[1]):
                result[f"ms__state_prob_{i}"] = probs[:, i].astype(np.float32)
        else:
            state_id = self.cluster_model.predict(components)
            result["ms__state_id"] = state_id.astype(np.int8)
        
        # Cache result
        self._transform_cache = (cache_key, result)
        tprint_success(f"[MarketStateVector] transform done columns={len(result.columns)}")
        return result

    def compute_state(self, state_instruments: pd.DataFrame) -> pd.DataFrame:
        tprint_info("[MarketStateVector] compute_state start")
        self.fit(state_instruments)
        result = self.transform(state_instruments)
        tprint_success("[MarketStateVector] compute_state done")
        return result

    def check_stability(self, loadings_history: List[np.ndarray]) -> bool:
        tprint_info(f"[MarketStateVector] check_stability start history={len(loadings_history)}")
        if len(loadings_history) < 2:
            return True
        prev = loadings_history[-2]
        curr = loadings_history[-1]
        similarities = []
        for i in range(min(prev.shape[0], curr.shape[0])):
            num = np.dot(prev[i], curr[i])
            denom = np.linalg.norm(prev[i]) * np.linalg.norm(curr[i]) + 1e-9
            similarities.append(num / denom)
        min_similarity = float(np.min(similarities)) if similarities else 0.0
        tprint_info(f"[MarketStateVector] Stability min similarity={min_similarity:.3f}")
        ok = min_similarity >= self.config.stability_threshold
        tprint_success(f"[MarketStateVector] check_stability done ok={ok}")
        return ok

    def persist_state(self, version: str, base_dir: str = "artifacts/market_state_vector") -> str:
        tprint_info(f"[MarketStateVector] persist_state start version={version}")
        path = f"{base_dir}/{version}"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/pca_components.npy", self.pca.components_)
        np.save(f"{path}/pca_mean.npy", self.scaler.mean_)
        np.save(f"{path}/pca_scale.npy", self.scaler.scale_)
        if self.cluster_model is not None:
            centers = getattr(self.cluster_model, "means_", getattr(self.cluster_model, "cluster_centers_", None))
            if centers is not None:
                np.save(f"{path}/cluster_centers.npy", centers)
        tprint_success(f"[MarketStateVector] persist_state done path={path}")
        return path


class CrossAssetSurprises:
    """Quantile VPIN spillover and ECT features with tradability filters.
    
    Optimized with:
    - Proper cache invalidation for market returns
    - Numba-accelerated ECT calculations
    - Dimson beta adjustment for illiquid assets
    """

    def __init__(self, quantiles: Optional[List[float]] = None, ect_window: int = 252, 
                 ect_half_life_bounds: Tuple[float, float] = (1.0, 50.0),
                 dimson_lags: int = 1):
        tprint_info("[CrossAssetSurprises] init")
        self.quantiles = quantiles or [0.5, 0.75, 0.9]
        self.ect_window = ect_window
        self.ect_half_life_bounds = ect_half_life_bounds
        self.dimson_lags = dimson_lags  # For Dimson beta adjustment
        self._vpin_models: Dict[str, Dict[float, Any]] = {}
        # Cache with invalidation key
        self._market_returns_cache: Optional[Tuple[str, pd.Series]] = None

    def fit(self, panel_df: pd.DataFrame, state_df: pd.DataFrame) -> "CrossAssetSurprises":
        tprint_info(f"[CrossAssetSurprises] fit start panel_shape={panel_df.shape} state_shape={state_df.shape}")
        if not STATSMODELS_AVAILABLE:
            tprint_warning("statsmodels unavailable; using quantile baselines")
            return self

        for ticker in panel_df.index.get_level_values("ticker").unique():
            slice_df = panel_df.xs(ticker, level="ticker")
            vpin = self._ensure_vpin(slice_df)
            features = state_df.reindex(slice_df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if vpin is None or features.empty:
                continue
            X = sm.add_constant(features, has_constant="add")
            valid = vpin.notna() & np.isfinite(X).all(axis=1)
            X_valid = X.loc[valid]
            y_valid = vpin.loc[valid]
            if len(y_valid) < max(50, len(features.columns) * 2):
                continue
            self._vpin_models.setdefault(ticker, {})
            for q in self.quantiles:
                try:
                    model = QuantReg(y_valid, X_valid).fit(q=q)
                    self._vpin_models[ticker][q] = model
                except Exception:
                    # Store baseline quantile as fallback
                    self._vpin_models[ticker][q] = float(y_valid.quantile(q))
        tprint_success("[CrossAssetSurprises] fit done")
        return self

    def transform(self, panel_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
        """Transform panel data with cross-asset features.

        Uses proper cache invalidation to avoid stale market returns.
        """
        tprint_info(f"[CrossAssetSurprises] transform start panel_shape={panel_df.shape}")
        results = []
        tickers = panel_df.index.get_level_values("ticker").unique()

        # Compute market returns once with proper cache invalidation
        market_returns = self._get_market_returns(panel_df)

        # Pre-compute state_df reindex once (avoid repeated reindex)
        state_df_aligned = state_df.reindex(panel_df.index.get_level_values("timestamp").unique())

        for ticker in tickers:
            slice_df = panel_df.xs(ticker, level="ticker")
            vpin = self._ensure_vpin(slice_df)
            features = state_df_aligned.reindex(slice_df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if vpin is None:
                continue
            X = sm.add_constant(features, has_constant="add") if STATSMODELS_AVAILABLE else features
            out = pd.DataFrame(index=slice_df.index)
            for q in self.quantiles:
                col = f"ca__vpin_spill_q{int(q * 100)}"
                resid_col = f"ca__vpin_spill_resid_q{int(q * 100)}"
                pred = None
                model_or_baseline = self._vpin_models.get(ticker, {}).get(q)
                if model_or_baseline is not None:
                    if isinstance(model_or_baseline, float):
                        # It's a baseline value
                        pred = np.full(len(out), model_or_baseline, dtype=np.float32)
                    else:
                        # It's a fitted model
                        pred = model_or_baseline.predict(X).astype(np.float32)
                if pred is None:
                    pred = np.full(len(out), np.nan, dtype=np.float32)

                pred_series = pd.Series(pred, index=out.index, dtype=np.float32)
                out[col] = pred_series

                resid_values = (vpin.values - pred_series.values).astype(np.float32)
                resid_std = float(np.nanstd(resid_values)) if len(resid_values) else 0.0

                if resid_std < 1e-6:
                    # Fallback 1: center VPIN around its mean to recover variance
                    centered = (vpin.values - np.nanmean(vpin.values)).astype(np.float32)
                    resid_values = centered
                    resid_std = float(np.nanstd(resid_values)) if len(resid_values) else 0.0

                if resid_std < 1e-6:
                    # Fallback 2: use first-difference of VPIN as residual proxy
                    diff_series = vpin.diff().fillna(0.0).astype(np.float32)
                    resid_values = diff_series.values

                resid_values = np.nan_to_num(resid_values, nan=0.0, posinf=0.0, neginf=0.0)
                out[resid_col] = resid_values

            # Compute robust cross-asset features
            robust = self._compute_robust_features(slice_df, market_returns)
            out = pd.concat([out, robust], axis=1)

            ect = self._compute_ect_features(slice_df, state_df_aligned)
            out = pd.concat([out, ect], axis=1)
            out["ticker"] = ticker
            out = out.reset_index().set_index(["timestamp", "ticker"])
            results.append(out)

        if not results:
            tprint_success("[CrossAssetSurprises] transform done empty")
            return pd.DataFrame(index=panel_df.index)
        combined = pd.concat(results).reindex(panel_df.index)
        tprint_success(f"[CrossAssetSurprises] transform done shape={combined.shape}")
        return combined
    
    def _get_market_returns(self, panel_df: pd.DataFrame) -> pd.Series:
        """Get market returns with proper cache invalidation."""
        # Create cache key from panel shape and first/last timestamps
        cache_key = f"{panel_df.shape}_{panel_df.index[0]}_{panel_df.index[-1]}"
        
        if self._market_returns_cache is not None:
            cached_key, cached_returns = self._market_returns_cache
            if cached_key == cache_key:
                return cached_returns
        
        # Compute market returns
        try:
            if "y__ret_1" in panel_df.columns:
                mean_ret = (
                    panel_df["y__ret_1"]
                    .replace([np.inf, -np.inf], np.nan)
                    .groupby(level="timestamp")
                    .mean()
                    .shift(1)
                    .fillna(0.0)
                    .astype(np.float32)
                )
            else:
                mean_ret = pd.Series(0.0, index=panel_df.index.get_level_values("timestamp").unique(), dtype=np.float32)
        except Exception:
            mean_ret = pd.Series(0.0, index=panel_df.index.get_level_values("timestamp").unique(), dtype=np.float32)
        
        self._market_returns_cache = (cache_key, mean_ret)
        return mean_ret

    def fit_transform(self, panel_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
        tprint_info("[CrossAssetSurprises] fit_transform start")
        self.fit(panel_df, state_df)
        result = self.transform(panel_df, state_df)
        tprint_success("[CrossAssetSurprises] fit_transform done")
        return result

    def _compute_robust_features(self, slice_df: pd.DataFrame, market_returns: pd.Series, window: int = 50) -> pd.DataFrame:
        """
        Compute robust cross-asset features: Rolling Beta (with Dimson adjustment), 
        Relative Strength, Lead-Lag, Shock.
        
        Improvements:
        - Dimson beta for illiquid assets (lead/lag adjustment)
        - Winsorized returns for robustness
        - Numba-accelerated calculations
        - float32 output for memory efficiency
        """
        idx = slice_df.index
        out = pd.DataFrame(index=idx)
        
        # 0. Asset Returns (Log Returns preferred for statistical properties)
        if "raw__log_px" in slice_df.columns:
            asset_log_px = slice_df["raw__log_px"].replace([np.inf, -np.inf], np.nan)
            asset_ret = asset_log_px.diff()
        elif "raw__px" in slice_df.columns:
            safe_px = slice_df["raw__px"].where(slice_df["raw__px"] > 0)
            asset_ret = np.log(safe_px).diff()
        elif "close" in slice_df.columns:
            safe_px = slice_df["close"].where(slice_df["close"] > 0)
            asset_ret = np.log(safe_px).diff()
        else:
            return out # Cannot compute

        asset_ret = asset_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mkt_ret = market_returns.reindex(idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Fill NaNs for Numba (0.0 assumption for returns is neutral)
        r_a_vals = asset_ret.fillna(0.0).values.astype(np.float64)
        r_m_vals = mkt_ret.fillna(0.0).values.astype(np.float64)
        
        # Winsorize Returns (1% tails) using Numba
        r_a_win = _numba_winsorize(r_a_vals, 0.01, 0.99)
        r_m_win = _numba_winsorize(r_m_vals, 0.01, 0.99)

        # --- 1. Rolling Beta with Dimson Adjustment ---
        # Standard Beta = Cov(r_a, r_m) / Var(r_m)
        # Dimson Beta = sum of betas on lagged market returns (for illiquid assets)

        def compute_beta_numba(ra, rm, w):
            cov = _numba_rolling_cov(ra, rm, w)
            var = _numba_rolling_cov(rm, rm, w)
            return cov / (var + 1e-9)

        # Standard betas
        beta_short = compute_beta_numba(r_a_win, r_m_win, 24)
        beta_long = compute_beta_numba(r_a_win, r_m_win, 96)
        
        out["ca__beta_short_w24"] = beta_short.astype(np.float32)
        out["ca__beta_long_w96"] = beta_long.astype(np.float32)
        out["ca__beta_shift"] = (beta_short - beta_long).astype(np.float32)
        
        # Dimson beta: sum of contemporaneous + lagged betas
        # This adjusts for thin trading / illiquidity
        if self.dimson_lags > 0:
            dimson_beta = beta_long.copy()
            for lag in range(1, self.dimson_lags + 1):
                # Create lagged market returns
                rm_lagged = np.zeros_like(r_m_win)
                rm_lagged[lag:] = r_m_win[:-lag]
                dimson_beta += compute_beta_numba(r_a_win, rm_lagged, 96)
            out["ca__dimson_beta_w96"] = dimson_beta.astype(np.float32)
        else:
            out["ca__dimson_beta_w96"] = beta_long.astype(np.float32)

        # Downside beta: beta computed only on bars where market return < 0
        mask_down = r_m_win < 0
        r_a_pd_down = pd.Series(np.where(mask_down, r_a_win, np.nan), index=idx)
        r_m_pd_down = pd.Series(np.where(mask_down, r_m_win, np.nan), index=idx)
        downside_beta = (r_a_pd_down.rolling(96, min_periods=20).cov(r_m_pd_down) / 
                        (r_m_pd_down.rolling(96, min_periods=20).var() + 1e-9))
        out["ca__downside_beta_long_w96"] = downside_beta.astype(np.float32)

        # --- 2. Relative Strength / Active Return (vs market) ---
        # active_ret = r_asset - r_mkt
        active_ret_vals = r_a_vals - r_m_vals

        # active_ret_z: 48 bars
        active_mean_48 = _numba_rolling_mean(active_ret_vals, 48)
        active_std_48 = _numba_rolling_std(active_ret_vals, 48)
        out["ca__active_ret_z_w48"] = active_mean_48 / (active_std_48 + 1e-9)

        # active_ret_trend: EWMA span 48 (alpha = 2/(span+1) = 2/49)
        alpha = 2.0 / 49.0
        out["ca__active_ret_trend"] = _numba_ewma(active_ret_vals, alpha, adjust=True)

        # active_ret_mr: (active_ret - EWMA) / rolling_std
        out["ca__active_ret_mr"] = (active_ret_vals - out["ca__active_ret_trend"]) / (active_std_48 + 1e-9)

        # active_ret_voladj: (r_asset - r_mkt) / (σ_asset_48 + ε)
        asset_vol_48 = _numba_rolling_std(r_a_vals, 48)
        out["ca__active_ret_voladj"] = active_ret_vals / (asset_vol_48 + 1e-9)

        # --- 3. Lead-Lag Dynamics ---
        # Lead-Lag Statistic: corr(asset_t, mkt_{t-lag}) - corr(mkt_t, asset_{t-lag})
        # Positive => Market Leads (Asset Follows)
        # Negative => Asset Leads (Market Follows)

        lags = [1, 2, 3, 6, 12]
        ll_window = 48
        max_abs_corr = np.zeros(len(idx))

        for lag in lags:
            # corr(asset_t, mkt_{t-lag})
            mkt_lagged = np.full_like(r_m_win, 0.0)
            mkt_lagged[lag:] = r_m_win[:-lag]
            c1 = _numba_rolling_correlation(r_a_win, mkt_lagged, ll_window)

            # corr(mkt_t, asset_{t-lag})
            asset_lagged = np.full_like(r_a_win, 0.0)
            asset_lagged[lag:] = r_a_win[:-lag]
            c2 = _numba_rolling_correlation(r_m_win, asset_lagged, ll_window)

            diff = c1 - c2

            # Update Max Abs
            mask_update = np.abs(diff) > np.abs(max_abs_corr)
            max_abs_corr[mask_update] = diff[mask_update]

        out["ca__lead_lag_w48"] = max_abs_corr

        # lead_lag_sign_persistence: directional stability of the lead-lag signal
        # Fraction of last 48 bars where sign matched mean sign?
        # Or simply rolling sum of signs magnitude.
        # We use: abs(rolling_sum(sign)) / window. If consistent => 1.0.
        sign_series = np.sign(out["ca__lead_lag_w48"]).values.astype(np.float64)
        out["ca__lead_lag_sign_persistence"] = np.abs(_numba_rolling_mean(sign_series, 48))

        # --- 4. Shock Features ---
        # corr_shock = corr_12(asset,mkt) - corr_96(asset,mkt)
        c12 = _numba_rolling_correlation(r_a_win, r_m_win, 12)
        c96 = _numba_rolling_correlation(r_a_win, r_m_win, 96)
        out["ca__corr_shock"] = c12 - c96

        # vol_shock = (σ_asset_12/σ_asset_96) - (σ_mkt_12/σ_mkt_96)
        std_a_12 = _numba_rolling_std(r_a_win, 12)
        std_a_96 = _numba_rolling_std(r_a_win, 96)
        std_m_12 = _numba_rolling_std(r_m_win, 12)
        std_m_96 = _numba_rolling_std(r_m_win, 96)

        ratio_a = std_a_12 / (std_a_96 + 1e-9)
        ratio_m = std_m_12 / (std_m_96 + 1e-9)
        out["ca__vol_shock"] = ratio_a - ratio_m

        # volume_shock = zscore(log(volume), 96)
        if "raw__volume" in slice_df.columns:
            vol = slice_df["raw__volume"]
        elif "volume" in slice_df.columns:
            vol = slice_df["volume"]
        else:
            vol = None

        if vol is not None:
            # Winsorize/Clean volume (Numpy based)
            vol_vals = vol.values.astype(np.float64)
            # Simple ffill using loop or pandas before conversion?
            # Already have vol as pandas series. Use ffill()
            vol_clean = vol.ffill().fillna(1.0).values.astype(np.float64)

            # Avoid log(0)
            vol_clean[vol_clean <= 0] = 1.0
            log_vol = np.log(vol_clean)

            lv_mean = _numba_rolling_mean(log_vol, 96)
            lv_std = _numba_rolling_std(log_vol, 96)
            out["ca__volume_shock"] = (log_vol - lv_mean) / (lv_std + 1e-9)

            # delta_volume_shock over 12
            # Vectorized diff
            shock_diff = np.full_like(out["ca__volume_shock"], 0.0)
            shock_diff[12:] = out["ca__volume_shock"].values[12:] - out["ca__volume_shock"].values[:-12]
            out["ca__delta_volume_shock_12"] = shock_diff
        else:
            out["ca__volume_shock"] = 0.0
            out["ca__delta_volume_shock_12"] = 0.0

        return out.fillna(0.0)

    def _ensure_vpin(self, df: pd.DataFrame) -> Optional[pd.Series]:
        required_cols = ["raw__close", "raw__high", "raw__low", "raw__volume"]
        if all(col in df.columns for col in required_cols):
            temp = df.rename(columns={
                "raw__close": "close",
                "raw__high": "high",
                "raw__low": "low",
                "raw__volume": "volume",
            })
            return compute_vpin(temp)
        return None

    def _compute_ect_features(self, df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
        """Compute ECT (Error Correction Term) features using Numba-accelerated rolling cov/var.
        
        Improvements:
        - Uses Numba for rolling covariance calculations
        - Better half-life estimation using OU process approximation
        - Variance ratio test as proxy for stationarity p-value
        - float32 output for memory efficiency
        """
        idx = df.index
        out = pd.DataFrame(index=idx)
        if "raw__px" not in df.columns:
            return out
        market_factor = self._resolve_market_factor(state_df, idx)
        if market_factor is None:
            return out

        log_px = np.log(df["raw__px"].replace(0, np.nan)).ffill()
        log_m = market_factor.ffill()
        window = self.ect_window

        # Convert to numpy for Numba
        log_px_arr = log_px.fillna(0.0).values.astype(np.float64)
        log_m_arr = log_m.fillna(0.0).values.astype(np.float64)

        # 1. Rolling Beta & Intercept using Numba
        rolling_cov = _numba_rolling_cov(log_m_arr, log_px_arr, window)
        rolling_var = _numba_rolling_cov(log_m_arr, log_m_arr, window)
        rolling_mean_x = _numba_rolling_mean(log_m_arr, window)
        rolling_mean_y = _numba_rolling_mean(log_px_arr, window)

        beta = rolling_cov / (rolling_var + 1e-9)
        intercept = rolling_mean_y - beta * rolling_mean_x
        
        # Calculate Residuals
        residuals = log_px_arr - (beta * log_m_arr + intercept)
        out["ca__ect_value"] = pd.Series(residuals, index=idx, dtype=np.float32)

        # 2. Half-Life using AR(1) coefficient (Numba-accelerated)
        res_lag = np.zeros_like(residuals)
        res_lag[1:] = residuals[:-1]
        
        ar_cov = _numba_rolling_cov(res_lag, residuals, window)
        ar_var = _numba_rolling_cov(res_lag, res_lag, window)
        phi = ar_cov / (ar_var + 1e-9)
        
        # Half-life = -ln(2) / ln(phi) for mean-reverting process
        phi_clipped = np.clip(phi, 1e-4, 1.0 - 1e-4)
        half_life = -np.log(2) / np.log(phi_clipped)
        
        # Mask invalid phi values
        mask_valid_phi = (phi > 0) & (phi < 1)
        half_life = np.where(mask_valid_phi, half_life, np.nan)
        out["ca__ect_half_life"] = pd.Series(half_life, index=idx, dtype=np.float32)

        # 3. Variance Ratio Test as proxy for stationarity
        # VR(q) = Var(q-period returns) / (q * Var(1-period returns))
        # For mean-reverting series, VR < 1
        q = min(20, window // 10)
        res_diff = np.diff(residuals, prepend=residuals[0])
        var_1 = _numba_rolling_cov(res_diff, res_diff, window)
        
        # q-period differences
        res_diff_q = np.zeros_like(residuals)
        res_diff_q[q:] = residuals[q:] - residuals[:-q]
        var_q = _numba_rolling_cov(res_diff_q, res_diff_q, window)
        
        variance_ratio = var_q / (q * var_1 + 1e-9)
        
        # Map variance ratio to pseudo p-value
        # VR < 0.8 suggests mean reversion (low p-value)
        # VR > 1.2 suggests momentum (high p-value)
        pvalue_proxy = np.where(variance_ratio < 0.8, 0.01,
                       np.where(variance_ratio < 1.0, 0.03,
                       np.where(variance_ratio < 1.2, 0.08, 0.15)))
        
        out["ca__ect_variance_ratio"] = pd.Series(variance_ratio, index=idx, dtype=np.float32)
        out["ca__ect_pvalue"] = pd.Series(pvalue_proxy, index=idx, dtype=np.float32)

        # 4. ECT Active flag
        hl_min, hl_max = self.ect_half_life_bounds
        hl_series = out["ca__ect_half_life"]
        pval_series = out["ca__ect_pvalue"]
        
        active_mask = (
            (hl_series >= hl_min) & (hl_series <= hl_max) & (pval_series <= 0.05)
        )
        out["ca__ect_active"] = active_mask.astype(np.float32).fillna(0.0)
        
        return out

    @staticmethod
    def _resolve_market_factor(state_df: pd.DataFrame, idx: pd.Index) -> Optional[pd.Series]:
        for col in ("ms__pca_0", "ms__pca_1"):
            if col in state_df.columns:
                return state_df[col].reindex(idx).ffill()
        return None


class MetaModelInvariance:
    """Gradient alignment + deterministic pruning + ticker ID block."""

    def enforce_no_ticker_id(self, features: pd.DataFrame) -> pd.DataFrame:
        tprint_info("[MetaModelInvariance] enforce_no_ticker_id start")
        drop_cols = [c for c in features.columns if any(k in c.lower() for k in ["ticker", "symbol", "asset_id", "exchange"])]
        if drop_cols:
            tprint_warning(f"[MetaModelInvariance] Dropping ticker ID features: {drop_cols[:5]}")
        cleaned = features.drop(columns=drop_cols, errors="ignore")
        tprint_success("[MetaModelInvariance] enforce_no_ticker_id done")
        return cleaned

    def compute_gradient_alignment(self, model: Any, features: pd.DataFrame, environments: Dict[str, np.ndarray]) -> InvarianceReport:
        """Compute feature importance alignment across environments.
        
        Note: Uses Ridge coefficients as proxy for feature importance direction,
        not true gradients. For true gradients, would need model-specific implementation
        (e.g., SHAP values or backprop for neural nets).
        
        The coefficients indicate which features the model relies on in each environment.
        High variance across environments suggests the model uses different features
        in different contexts, which may indicate lack of invariance.
        """
        tprint_info(f"[MetaModelInvariance] compute_gradient_alignment start envs={len(environments)}")
        feature_importances = {}
        
        for env_name, mask in environments.items():
            if mask.sum() < 10:
                continue
            X_env = features.loc[mask]
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_env)[:, -1]
            else:
                preds = model.predict(X_env)
            
            # Use Ridge to get linear approximation of feature importance
            # Normalize features for comparable coefficients
            X_scaled = (X_env - X_env.mean()) / (X_env.std() + 1e-9)
            proxy_model = Ridge(alpha=1.0).fit(X_scaled, preds)
            feature_importances[env_name] = proxy_model.coef_

        env_names = list(feature_importances.keys())
        if len(env_names) < 2:
            return InvarianceReport(dispersion=0.0, worst_env_pair=("", ""), worst_distance=0.0, per_feature_grad_var=pd.Series(0.0, index=features.columns))

        # Compute cosine distances between feature importance vectors
        dists = []
        worst_pair = (env_names[0], env_names[1])
        worst_distance = -1.0
        for i, env_i in enumerate(env_names):
            for env_j in env_names[i + 1:]:
                fi = feature_importances[env_i]
                fj = feature_importances[env_j]
                # Cosine distance: 1 - cosine_similarity
                cos_sim = np.dot(fi, fj) / (np.linalg.norm(fi) * np.linalg.norm(fj) + 1e-9)
                dist = 1.0 - float(cos_sim)
                dists.append(dist)
                if dist > worst_distance:
                    worst_distance = dist
                    worst_pair = (env_i, env_j)
        
        # Per-feature variance across environments
        importance_matrix = np.vstack([feature_importances[e] for e in env_names])
        importance_var = pd.Series(np.var(importance_matrix, axis=0), index=features.columns)
        
        report = InvarianceReport(
            dispersion=float(np.mean(dists)),
            worst_env_pair=worst_pair,
            worst_distance=float(worst_distance),
            per_feature_grad_var=importance_var,
        )
        tprint_success("[MetaModelInvariance] compute_gradient_alignment done")
        return report

    def iterative_pruning(self, features: pd.DataFrame, report: InvarianceReport, k_drop: int = 5, max_iter: int = 3, dispersion_target: float = 0.2) -> Tuple[pd.DataFrame, List[str]]:
        tprint_info("[MetaModelInvariance] iterative_pruning start")
        removed: List[str] = []
        current = features
        for _ in range(max_iter):
            if report.dispersion <= dispersion_target:
                break
            drop = report.per_feature_grad_var.sort_values(ascending=False).head(k_drop).index.tolist()
            current = current.drop(columns=drop, errors="ignore")
            removed.extend(drop)
        tprint_success(f"[MetaModelInvariance] iterative_pruning done removed={len(removed)}")
        return current, removed


class CrossAssetPositionSizer:
    """Calibration, percentile ranking, entropy filtering, deterministic Top-K."""

    def __init__(self, calibration_window: int = 250, method: str = "isotonic"):
        tprint_info("[CrossAssetPositionSizer] init")
        self.calibration_window = calibration_window
        self.method = method

    def compute_cross_asset_percentiles(self, scores: pd.DataFrame, labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compute cross-asset percentiles using vectorized groupby.transform.
        
        Uses pd.IndexSlice for proper MultiIndex slicing.
        """
        tprint_info(f"[CrossAssetPositionSizer] compute_cross_asset_percentiles start shape={scores.shape}")
        if not isinstance(scores.index, pd.MultiIndex):
            raise ValueError("scores must be MultiIndex (timestamp, ticker)")
        result = scores.copy()
        result["calibrated_p"] = np.nan
        
        idx = pd.IndexSlice
        for ticker in result.index.get_level_values("ticker").unique():
            sub = result.xs(ticker, level="ticker")
            y = labels.xs(ticker, level="ticker") if labels is not None else None
            calibrated = self._rolling_calibrate(sub["score"], y)
            # Use pd.IndexSlice for proper MultiIndex assignment
            result.loc[idx[:, ticker], "calibrated_p"] = calibrated.values

        result["percentile"] = result.groupby(level="timestamp")["calibrated_p"].rank(pct=True, method="first")
        tprint_success("[CrossAssetPositionSizer] compute_cross_asset_percentiles done")
        return result

    def apply_entropy_filter(self, scores: pd.DataFrame, threshold: float = 1.0, 
                             use_effective_n: bool = True) -> pd.Series:
        """Apply entropy/concentration filter.
        
        Args:
            scores: DataFrame with 'percentile' column
            threshold: For entropy mode, max entropy. For effective_n mode, min effective bets.
            use_effective_n: If True, use effective number of bets instead of entropy.
        """
        tprint_info("[CrossAssetPositionSizer] apply_entropy_filter start")
        if use_effective_n:
            # Effective N: higher is more diversified, filter if below threshold
            eff_n = scores.groupby(level="timestamp")["percentile"].apply(self._effective_n_bets)
            # Default threshold of 1.0 means at least 1 effective bet
            entropy_pass = eff_n >= threshold
        else:
            # Entropy: lower is more concentrated, filter if above threshold
            entropy_vals = scores.groupby(level="timestamp")["percentile"].apply(self._entropy)
            entropy_pass = entropy_vals < threshold
        entropy_pass.index.name = "timestamp"
        tprint_success("[CrossAssetPositionSizer] apply_entropy_filter done")
        return entropy_pass

    def select_top_k(self, scores: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        tprint_info(f"[CrossAssetPositionSizer] select_top_k start k={k}")
        scores = scores.copy()
        scores["rank"] = scores.groupby(level="timestamp")["percentile"].rank(ascending=False, method="first")
        selected = scores[scores["rank"] <= k]
        selected = selected.sort_index()
        tprint_success(f"[CrossAssetPositionSizer] select_top_k done selected={len(selected)}")
        return selected

    def _rolling_calibrate(self, scores: pd.Series, labels: Optional[pd.Series]) -> pd.Series:
        """Rolling calibration with periodic refits (not per-row) for O(N) complexity.
        
        Refits every `refit_interval` rows instead of every row to avoid O(N²) complexity.
        Uses expanding window with periodic model updates.
        """
        tprint_info("[CrossAssetPositionSizer] _rolling_calibrate start")
        if labels is None or labels.isna().all():
            return scores.clip(0.0, 1.0)
        
        n = len(scores)
        calibrated = np.full(n, np.nan, dtype=np.float32)
        min_train = max(20, self.calibration_window // 5)
        refit_interval = max(50, self.calibration_window // 5)  # Refit every ~50 rows
        
        current_model = None
        last_fit_idx = -1
        
        scores_arr = scores.values.astype(np.float64)
        labels_arr = labels.fillna(0.0).values.astype(np.float64)
        
        for i in range(n):
            start = max(0, i - self.calibration_window)
            train_size = i - start
            
            # Warmup period: use raw scores
            if train_size < min_train:
                calibrated[i] = np.clip(scores_arr[i], 0.0, 1.0)
                continue
            
            # Refit model periodically or if no model exists
            need_refit = (current_model is None) or (i - last_fit_idx >= refit_interval)
            
            if need_refit:
                train_scores = scores_arr[start:i]
                train_labels = labels_arr[start:i]
                
                # Check for label variance (need at least 2 classes for Platt)
                unique_labels = np.unique(train_labels[~np.isnan(train_labels)])
                
                if self.method == "platt" and len(unique_labels) >= 2:
                    try:
                        model = LogisticRegression(max_iter=200, solver='lbfgs')
                        model.fit(train_scores.reshape(-1, 1), train_labels)
                        current_model = model
                        last_fit_idx = i
                    except Exception:
                        # Fallback to isotonic if Platt fails
                        pass
                
                if current_model is None or self.method != "platt":
                    try:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(train_scores, train_labels)
                        current_model = iso
                        last_fit_idx = i
                    except Exception:
                        calibrated[i] = np.clip(scores_arr[i], 0.0, 1.0)
                        continue
            
            # Apply current model
            if current_model is not None:
                try:
                    if hasattr(current_model, 'predict_proba'):
                        calibrated[i] = current_model.predict_proba([[scores_arr[i]]])[0, 1]
                    else:
                        calibrated[i] = current_model.predict([scores_arr[i]])[0]
                except Exception:
                    calibrated[i] = np.clip(scores_arr[i], 0.0, 1.0)
            else:
                calibrated[i] = np.clip(scores_arr[i], 0.0, 1.0)
        
        tprint_success("[CrossAssetPositionSizer] _rolling_calibrate done")
        return pd.Series(calibrated, index=scores.index, dtype=np.float32)

    @staticmethod
    def _entropy(series: pd.Series) -> float:
        """Compute Shannon entropy using histogram binning.
        
        For percentiles/continuous values, histogram binning is the correct approach.
        Also computes effective number of bets as alternative concentration measure.
        """
        vals = series.dropna().values.astype(np.float64)
        if len(vals) == 0:
            return 0.0
        # Use proper histogram-based entropy for percentiles
        return float(_numba_entropy_histogram(vals, n_bins=10))
    
    @staticmethod
    def _effective_n_bets(series: pd.Series) -> float:
        """Compute effective number of bets (1/sum(w^2)).
        
        Better concentration measure than entropy for portfolio weights.
        """
        vals = series.dropna().values.astype(np.float64)
        if len(vals) == 0:
            return 0.0
        return float(_numba_effective_n_bets(vals))


class PortfolioConstraints:
    """Tail correlation and beta exposure constraints."""

    def __init__(self, tail_quantile: float = 0.05, min_tail_sample: int = 30, beta_cap: float = 1.5):
        tprint_info("[PortfolioConstraints] init")
        self.tail_quantile = tail_quantile
        self.min_tail_sample = min_tail_sample
        self.beta_cap = beta_cap

    def check_tail_correlation(self, returns: pd.DataFrame, market_returns: pd.Series) -> Tuple[bool, pd.Series]:
        tprint_info("[PortfolioConstraints] check_tail_correlation start")
        tail_mask = market_returns <= market_returns.rolling(252, min_periods=50).quantile(self.tail_quantile)
        if tail_mask.sum() < self.min_tail_sample:
            tprint_success("[PortfolioConstraints] check_tail_correlation done (min sample)")
            return True, pd.Series(index=returns.columns, dtype=float)
        tail_corr = returns.loc[tail_mask].corrwith(market_returns.loc[tail_mask])
        ok = tail_corr.abs().max() <= self.beta_cap
        tprint_success(f"[PortfolioConstraints] check_tail_correlation done ok={ok}")
        return bool(ok), tail_corr

    def check_beta_exposure(self, returns: pd.DataFrame, market_returns: pd.Series) -> Tuple[bool, pd.Series]:
        """Check if portfolio beta exposure is within limits.
        
        Handles both DataFrame and Series inputs for returns.
        """
        tprint_info("[PortfolioConstraints] check_beta_exposure start")
        
        # Handle Series input (single asset)
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name="asset")
        
        # Compute rolling beta for each column
        mkt_var = market_returns.rolling(252, min_periods=50).var()
        
        betas = {}
        for col in returns.columns:
            cov = returns[col].rolling(252, min_periods=50).cov(market_returns)
            betas[col] = cov / (mkt_var + 1e-9)
        
        beta_df = pd.DataFrame(betas)
        
        # Get last valid beta values
        last_beta = beta_df.iloc[-1] if len(beta_df) > 0 else pd.Series(dtype=float)
        
        # Handle case where last_beta might be scalar or have NaNs
        if isinstance(last_beta, (int, float)):
            ok = abs(last_beta) <= self.beta_cap
            last_beta = pd.Series([last_beta])
        else:
            ok = last_beta.abs().max() <= self.beta_cap if len(last_beta) > 0 else True
        
        tprint_success(f"[PortfolioConstraints] check_beta_exposure done ok={ok}")
        return bool(ok), last_beta


class GatingEngine:
    """Central gating engine with reason codes (vectorized, no iterrows).
    
    Features:
    - Temporal persistence to reduce flicker (gate must fail N consecutive bars)
    - Vectorized operations for speed
    """
    
    def __init__(self):
        # Track consecutive failures for persistence
        self._failure_counts: Dict[str, Dict[str, int]] = {}

    def evaluate(self, panel_slice_t: pd.DataFrame, portfolio_state: Dict[str, Any], config: GatingConfig) -> pd.DataFrame:
        """Evaluate gating conditions using vectorized operations.
        
        Includes temporal persistence: gates only trigger after failing
        for `config.persistence_bars` consecutive evaluations.
        """
        tprint_info("[GatingEngine] evaluate start")
        n = len(panel_slice_t)
        result = pd.DataFrame(index=panel_slice_t.index)

        # ECT active gate (per-asset)
        ect_active = panel_slice_t.get("ca__ect_active", pd.Series(True, index=panel_slice_t.index))
        result["gate__ect_active"] = ect_active.fillna(False).astype(bool)

        # Portfolio-level gates (broadcast to all rows)
        entropy_pass = portfolio_state.get("entropy_pass", True)
        tail_ok = portfolio_state.get("tail_corr_pass", True)
        beta_ok = portfolio_state.get("beta_cap_pass", True)
        max_corr_ok = portfolio_state.get("max_corr_pass", True)
        
        # Apply temporal persistence for portfolio-level gates
        persistence = config.persistence_bars
        
        # Track failures and apply persistence
        gate_checks = {
            "entropy": bool(entropy_pass) if entropy_pass is not None else True,
            "tail_corr": bool(tail_ok),
            "beta_cap": bool(beta_ok),
            "max_corr": bool(max_corr_ok),
        }
        
        for gate_name, passes in gate_checks.items():
            if gate_name not in self._failure_counts:
                self._failure_counts[gate_name] = {}
            
            # Use "portfolio" as key for portfolio-level gates
            key = "portfolio"
            if not passes:
                self._failure_counts[gate_name][key] = self._failure_counts[gate_name].get(key, 0) + 1
            else:
                self._failure_counts[gate_name][key] = 0
            
            # Only fail if consecutive failures >= persistence threshold
            effective_pass = passes or (self._failure_counts[gate_name].get(key, 0) < persistence)
            gate_checks[gate_name] = effective_pass
        
        result["gate__entropy_pass"] = gate_checks["entropy"]
        result["gate__tail_corr_pass"] = gate_checks["tail_corr"]
        result["gate__beta_cap_pass"] = gate_checks["beta_cap"]
        result["gate__max_corr_pass"] = gate_checks["max_corr"]

        # Vectorized reason code generation
        reasons = np.full(n, "pass", dtype=object)
        
        ect_fail = ~result["gate__ect_active"].values
        entropy_fail = ~result["gate__entropy_pass"].values
        tail_fail = ~result["gate__tail_corr_pass"].values
        beta_fail = ~result["gate__beta_cap_pass"].values
        corr_fail = ~result["gate__max_corr_pass"].values
        
        for i in range(n):
            parts = []
            if ect_fail[i]:
                parts.append("ect_inactive")
            if entropy_fail[i]:
                parts.append("entropy")
            if tail_fail[i]:
                parts.append("tail_corr")
            if beta_fail[i]:
                parts.append("beta_cap")
            if corr_fail[i]:
                parts.append("max_corr")
            if parts:
                reasons[i] = ",".join(parts)
        
        result["gate__reason_codes"] = reasons
        tprint_success("[GatingEngine] evaluate done")
        return result
    
    def reset_persistence(self):
        """Reset failure counts (e.g., at start of new trading session)."""
        self._failure_counts = {}


class ValidationBattery:
    """LOAO/LOSO/synthetic validation with structured outputs."""

    def __init__(self, base_model: Any):
        tprint_info("[ValidationBattery] init")
        self.base_model = base_model

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC safely, handling edge cases.
        
        Handles:
        - Single class (returns 0.5)
        - Extreme class imbalance (uses try/except)
        - Non-binary labels (binarizes at median)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check for valid data
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.5
        
        unique_labels = np.unique(y_true[~np.isnan(y_true)])
        if len(unique_labels) < 2:
            return 0.5
        
        # Binarize if not already binary
        if not np.all(np.isin(unique_labels, [0, 1])):
            median_val = np.nanmedian(y_true)
            y_true = (y_true > median_val).astype(int)
        
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return 0.5

    @staticmethod
    def _safe_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Brier score safely."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        # Clip predictions to [0, 1]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        # Binarize if needed
        unique_labels = np.unique(y_true[~np.isnan(y_true)])
        if not np.all(np.isin(unique_labels, [0, 1])):
            median_val = np.nanmedian(y_true)
            y_true = (y_true > median_val).astype(int)
        
        try:
            return float(brier_score_loss(y_true, y_pred))
        except Exception:
            return 0.0

    def run_loao_validation(self, features: pd.DataFrame, labels: pd.Series, assets: pd.Series) -> ValidationResult:
        tprint_info("[ValidationBattery] run_loao_validation start")
        by_asset: Dict[str, Dict[str, float]] = {}
        for asset in assets.unique():
            mask = assets == asset
            train_X, train_y = features[~mask], labels[~mask]
            test_X, test_y = features[mask], labels[mask]
            if len(test_y) < 10 or len(train_y) < 30:
                continue
            model = clone(self.base_model)
            model.fit(train_X, train_y)
            preds = model.predict_proba(test_X)[:, -1] if hasattr(model, "predict_proba") else model.predict(test_X)
            by_asset[str(asset)] = {
                "auc": self._safe_auc(test_y.values, preds),
                "brier": self._safe_brier(test_y.values, preds),
            }
        metrics = {
            "auc_mean": float(np.mean([m["auc"] for m in by_asset.values()])) if by_asset else 0.0,
            "brier_mean": float(np.mean([m["brier"] for m in by_asset.values()])) if by_asset else 0.0,
        }
        result = ValidationResult(split_name="LOAO", metrics=metrics, by_asset=by_asset, by_sector={}, artifacts={})
        tprint_success("[ValidationBattery] run_loao_validation done")
        return result

    def run_loso_validation(self, features: pd.DataFrame, labels: pd.Series, sectors: pd.Series) -> ValidationResult:
        tprint_info("[ValidationBattery] run_loso_validation start")
        by_sector: Dict[str, Dict[str, float]] = {}
        for sector in sectors.unique():
            mask = sectors == sector
            train_X, train_y = features[~mask], labels[~mask]
            test_X, test_y = features[mask], labels[mask]
            if len(test_y) < 10 or len(train_y) < 30:
                continue
            model = clone(self.base_model)
            model.fit(train_X, train_y)
            preds = model.predict_proba(test_X)[:, -1] if hasattr(model, "predict_proba") else model.predict(test_X)
            by_sector[str(sector)] = {
                "auc": self._safe_auc(test_y.values, preds),
                "brier": self._safe_brier(test_y.values, preds),
            }
        metrics = {
            "auc_mean": float(np.mean([m["auc"] for m in by_sector.values()])) if by_sector else 0.0,
            "brier_mean": float(np.mean([m["brier"] for m in by_sector.values()])) if by_sector else 0.0,
        }
        result = ValidationResult(split_name="LOSO", metrics=metrics, by_asset={}, by_sector=by_sector, artifacts={})
        tprint_success("[ValidationBattery] run_loso_validation done")
        return result

    def run_synthetic_asset_test(self, features: pd.DataFrame, labels: pd.Series) -> ValidationResult:
        tprint_info("[ValidationBattery] run_synthetic_asset_test start")
        model = clone(self.base_model)
        model.fit(features, labels)
        preds = model.predict_proba(features)[:, -1] if hasattr(model, "predict_proba") else model.predict(features)
        metrics = {
            "auc": self._safe_auc(labels.values, preds),
            "brier": self._safe_brier(labels.values, preds),
        }
        result = ValidationResult(split_name="SYNTHETIC", metrics=metrics, by_asset={}, by_sector={}, artifacts={})
        tprint_success("[ValidationBattery] run_synthetic_asset_test done")
        return result


class CrossAssetChaser:
    """Residual learning utilities for cross-asset corrections."""

    def __init__(self, residual_col: str = "cs__residual"):
        tprint_info("[CrossAssetChaser] init")
        self.residual_col = residual_col

    def compute_peer_residual_momentum(self, panel_df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Compute peer residual momentum with proper index alignment.
        
        Uses transform instead of droplevel to maintain index alignment.
        """
        tprint_info("[CrossAssetChaser] compute_peer_residual_momentum start")
        if self.residual_col not in panel_df.columns:
            raise ValueError(f"Missing residual column {self.residual_col}")
        window = int(config.get("residual_momentum_window", 10))
        
        # Use transform to maintain proper index alignment
        momentum = panel_df.groupby(level="ticker")[self.residual_col].transform(
            lambda x: x.rolling(window=window, min_periods=max(2, window // 2)).mean()
        )
        tprint_success("[CrossAssetChaser] compute_peer_residual_momentum done")
        return momentum

    def compute_relative_volume_clusters(self, panel_df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Compute relative volume clusters with proper index alignment."""
        tprint_info("[CrossAssetChaser] compute_relative_volume_clusters start")
        volume_col = config.get("volume_col", "raw__volume")
        if volume_col not in panel_df.columns:
            raise ValueError(f"Missing volume column {volume_col}")
        window = int(config.get("volume_cluster_window", 20))
        
        # Use transform for proper alignment
        vol = panel_df[volume_col]
        vol_mean = panel_df.groupby(level="ticker")[volume_col].transform(
            lambda x: x.rolling(window).mean()
        )
        vol_std = panel_df.groupby(level="ticker")[volume_col].transform(
            lambda x: x.rolling(window).std()
        )
        zscore = (vol - vol_mean) / (vol_std + 1e-9)
        clusters = pd.qcut(zscore.fillna(0.0), q=3, labels=["low", "mid", "high"], duplicates="drop")
        tprint_success("[CrossAssetChaser] compute_relative_volume_clusters done")
        return clusters

    def validate_incremental_value(self, base_predictions: pd.Series, chaser_predictions: pd.Series, labels: pd.Series) -> bool:
        tprint_info("[CrossAssetChaser] validate_incremental_value start")
        base_corr = np.corrcoef(base_predictions.fillna(0.0), labels.fillna(0.0))[0, 1]
        chaser_corr = np.corrcoef(chaser_predictions.fillna(0.0), labels.fillna(0.0))[0, 1]
        improvement = np.nan_to_num(chaser_corr - base_corr)
        tprint_success(f"[CrossAssetChaser] validate_incremental_value done improvement={improvement:.4f}")
        return improvement > 0.0
