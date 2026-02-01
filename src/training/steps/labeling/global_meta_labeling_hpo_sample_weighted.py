"""Global Multi-Asset Meta-Labeling HPO Sample Weighted Step.

This step orchestrates multi-asset training by:
1. Loading data for all specified assets
2. Adding asset-specific features (asset ID, volatility normalization)
3. Combining data into unified training set
4. Running meta-labeling HPO on combined dataset
5. Storing unified model with asset-specific components

Key Features:
- Multi-asset data loading and combination
- Per-asset volatility normalization
- Asset-specific identification features
- Unified model training with asset context
- Asset-specific model components for inference
- Vectorized cross-asset computations with Numba acceleration
- Caching for expensive multi-asset computations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numba import njit, prange

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_success, tprint_warning, tprint_info, tprint_error

from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
    MetaLabelingHPOSampleWeightedStep
)
from src.training.steps.labeling.feature_engineering_utils import (
    add_market_context_features,
    apply_layer2_price_processing
)


# ============================================================================
# Numba-Optimized Functions for Rolling Beta/Covariance
# ============================================================================

@njit(cache=True)
def _numba_ewma_cov_var(
    asset_rets: np.ndarray,
    market_rets: np.ndarray,
    alpha: float,
) -> tuple:
    """
    Compute EWMA covariance and variance in a single pass.
    O(n) complexity, naturally handles non-stationarity.
    
    Returns:
        (ewma_cov, ewma_var_market) arrays
    """
    n = len(asset_rets)
    ewma_cov = np.empty(n, dtype=np.float64)
    ewma_var = np.empty(n, dtype=np.float64)
    
    ewma_cov[0] = np.nan
    ewma_var[0] = np.nan
    
    if n < 2:
        return ewma_cov, ewma_var
    
    # Initialize with first valid pair
    cov_acc = 0.0
    var_acc = 0.0
    mean_asset = asset_rets[0] if not np.isnan(asset_rets[0]) else 0.0
    mean_market = market_rets[0] if not np.isnan(market_rets[0]) else 0.0
    
    for i in range(1, n):
        a_ret = asset_rets[i]
        m_ret = market_rets[i]
        
        if np.isnan(a_ret) or np.isnan(m_ret):
            ewma_cov[i] = ewma_cov[i-1] if i > 0 else np.nan
            ewma_var[i] = ewma_var[i-1] if i > 0 else np.nan
            continue
        
        # Update means
        delta_asset = a_ret - mean_asset
        delta_market = m_ret - mean_market
        mean_asset = mean_asset + alpha * delta_asset
        mean_market = mean_market + alpha * delta_market
        
        # Update covariance and variance with EWMA
        cov_acc = (1 - alpha) * cov_acc + alpha * delta_asset * delta_market
        var_acc = (1 - alpha) * var_acc + alpha * delta_market * delta_market
        
        ewma_cov[i] = cov_acc
        ewma_var[i] = var_acc
    
    return ewma_cov, ewma_var


@njit(cache=True, parallel=True)
def _numba_rolling_cov_var_batch(
    asset_matrix: np.ndarray,  # Shape: (n_timestamps, n_assets)
    market_rets: np.ndarray,   # Shape: (n_timestamps,)
    window: int,
    min_periods: int,
) -> tuple:
    """
    Batch rolling covariance and variance for all assets.
    Uses parallel processing for multiple assets.
    
    Returns:
        (cov_matrix, var_array) where cov_matrix is (n_timestamps, n_assets)
    """
    n_times, n_assets = asset_matrix.shape
    cov_out = np.full((n_times, n_assets), np.nan, dtype=np.float64)
    var_out = np.full(n_times, np.nan, dtype=np.float64)
    
    # Compute market variance once (same for all assets)
    for t in range(min_periods - 1, n_times):
        start = max(0, t - window + 1)
        m_slice = market_rets[start:t+1]
        valid_m = m_slice[~np.isnan(m_slice)]
        if len(valid_m) >= min_periods:
            var_out[t] = np.var(valid_m)
    
    # Parallel over assets
    for a in prange(n_assets):
        for t in range(min_periods - 1, n_times):
            start = max(0, t - window + 1)
            a_slice = asset_matrix[start:t+1, a]
            m_slice = market_rets[start:t+1]
            
            # Find valid pairs
            valid_mask = ~(np.isnan(a_slice) | np.isnan(m_slice))
            n_valid = np.sum(valid_mask)
            
            if n_valid >= min_periods:
                a_valid = a_slice[valid_mask]
                m_valid = m_slice[valid_mask]
                
                a_mean = np.mean(a_valid)
                m_mean = np.mean(m_valid)
                
                cov_sum = 0.0
                for i in range(n_valid):
                    cov_sum += (a_valid[i] - a_mean) * (m_valid[i] - m_mean)
                cov_out[t, a] = cov_sum / n_valid
    
    return cov_out, var_out


@njit(cache=True)
def _numba_rolling_correlation(
    returns_matrix: np.ndarray,  # Shape: (n_timestamps, n_assets)
    window: int,
    min_periods: int,
) -> np.ndarray:
    """
    Compute average pairwise correlation across assets (correlation regime indicator).
    High correlation = risk-off / contagion regime.
    
    Returns:
        Array of shape (n_timestamps,) with average pairwise correlation
    """
    n_times, n_assets = returns_matrix.shape
    avg_corr = np.full(n_times, np.nan, dtype=np.float64)
    
    if n_assets < 2:
        return avg_corr
    
    for t in range(min_periods - 1, n_times):
        start = max(0, t - window + 1)
        
        corr_sum = 0.0
        n_pairs = 0
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                slice_i = returns_matrix[start:t+1, i]
                slice_j = returns_matrix[start:t+1, j]
                
                # Find valid pairs
                valid_mask = ~(np.isnan(slice_i) | np.isnan(slice_j))
                n_valid = np.sum(valid_mask)
                
                if n_valid >= min_periods:
                    vi = slice_i[valid_mask]
                    vj = slice_j[valid_mask]
                    
                    mi = np.mean(vi)
                    mj = np.mean(vj)
                    
                    cov = 0.0
                    var_i = 0.0
                    var_j = 0.0
                    for k in range(n_valid):
                        di = vi[k] - mi
                        dj = vj[k] - mj
                        cov += di * dj
                        var_i += di * di
                        var_j += dj * dj
                    
                    denom = np.sqrt(var_i * var_j)
                    if denom > 1e-12:
                        corr_sum += cov / denom
                        n_pairs += 1
        
        if n_pairs > 0:
            avg_corr[t] = corr_sum / n_pairs
    
    return avg_corr


class GlobalMetaLabelingHPOSampleWeightedStep(BaseStep):
    """
    Global multi-asset meta-labeling HPO orchestration step.
    
    This step extends the single-asset MetaLabelingHPOSampleWeightedStep
    to handle multiple assets with asset-specific features and normalization.
    
    Key improvements:
    - Vectorized cross-asset computations (no Python loops)
    - EWMA-based beta estimation (O(1) per sample)
    - BTC benchmark proxy option for market factor
    - Lagged market return to prevent label leakage
    - Adaptive quantile thresholds for regime classification
    - Cross-sectional rank and correlation regime features
    - Disk caching for expensive computations
    """
    
    # Configuration constants
    DEFAULT_BETA_HALFLIFE = 120  # ~2.5 days at 15m bars (was 60)
    DEFAULT_BETA_MIN_PERIODS = 60  # Increased from 20
    BETA_CLIP_BOUNDS = (-3.0, 3.0)
    BETA_MIN_COVERAGE = 0.4  # Warn if <40% valid data in beta window
    RESIDUAL_CLIP_SIGMA = 5.0  # Clip residual returns at ±5σ
    CORRELATION_WINDOW = 60
    CORRELATION_MIN_PERIODS = 30

    def __init__(self, step_name: str):
        """Initialize the global meta-labeling step."""
        super().__init__(step_name)
        self.combined_data = None
        self.asset_stats = {}
        self._underlying_step_cache = None  # Cache for MetaLabelingHPOSampleWeightedStep
        self._market_factor_cache = None  # Cache for pre-computed market factor

    def _load_asset_data(self, config: Dict[str, Any], asset: str) -> pd.DataFrame:
        """Load market data for a specific asset."""
        asset_config = config.copy()
        asset_config['symbol'] = f"{asset}USDT"
        
        tprint_info(f"Loading data for {asset}USDT...")
        
        # Use BaseStep's standard data loading
        market_data, _source = self.load_market_data_or_fail(
            asset_config,
            pipeline_state={},
            allow_config_override=True,
            light_mode_filter=True,
        )
        
        if market_data is None or market_data.empty:
            raise ValueError(f"Failed to load market data for {asset}USDT")
        
        tprint_success(f"✅ Loaded {len(market_data)} rows for {asset}USDT from {_source}")
        return market_data

    def _add_asset_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Add asset identifier only. One-hot encoding is done post-concat
        using pd.get_dummies for efficiency.
        """
        # Use assign() to avoid full copy
        return df.assign(asset_id=asset)

    def _add_asset_interaction_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Add asset-specific interaction features.
        Example: ETH_volatility_normalized, ETH_raw_returns, etc.
        Enables model to learn asset-specific feature importance.
        """
        features_to_interact = ['volatility_normalized', 'raw_returns', 'vol_regime_asset']
        for feat in features_to_interact:
            if feat in df.columns:
                df[f"{asset}_{feat}"] = df[feat]
        return df

    def _normalize_volatility_per_asset(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply per-asset volatility normalization respecting causality.
        
        CRITICAL: Uses expanding window statistics to prevent look-ahead bias.
        De Prado Compliance: Standardization must rely only on past data.
        
        IMPROVEMENT: Uses adaptive quantile thresholds from expanding empirical
        distribution instead of static z-score thresholds.
        """
        # Calculate per-asset volatility statistics
        close_series = pd.to_numeric(df['close'], errors='coerce')
        returns = close_series.pct_change().replace([np.inf, -np.inf], np.nan)
        
        # Rolling volatility (timeframe-aware, default 20 bars @ 15m)
        vol_window = self._resolve_vol_window(timeframe)
        rolling_vol = returns.rolling(window=vol_window, min_periods=1).std()
        
        # De Prado Compliance: No look-ahead bias
        # Use expanding window for mean and std of volatility
        vol_expanding = rolling_vol.expanding(min_periods=vol_window)
        vol_mean_exp = vol_expanding.mean()
        vol_std_exp = vol_expanding.std().replace(0, 1.0)
        
        vol_normalized = (rolling_vol - vol_mean_exp) / vol_std_exp
        
        # IMPROVEMENT: Adaptive quantile thresholds from expanding distribution
        # Instead of static ±0.43, use expanding 33rd and 67th percentiles
        vol_q33 = vol_normalized.expanding(min_periods=vol_window).quantile(0.33)
        vol_q67 = vol_normalized.expanding(min_periods=vol_window).quantile(0.67)
        
        # Fallback to static thresholds for early samples
        vol_q33 = vol_q33.fillna(-0.43)
        vol_q67 = vol_q67.fillna(0.43)
        
        vol_regime = np.where(
            vol_normalized > vol_q67, 'high',
            np.where(vol_normalized < vol_q33, 'low', 'medium')
        )
        
        # Store stats for reporting
        self.asset_stats[asset] = {
            'vol_mean': float(vol_mean_exp.iloc[-1]) if len(vol_mean_exp) > 0 else 0.0,
            'vol_std': float(vol_std_exp.iloc[-1]) if len(vol_std_exp) > 0 else 1.0,
            'returns_mean': float(returns.mean()),
            'returns_std': float(returns.std()),
        }
        
        # Use assign() to avoid full copy
        return df.assign(
            raw_returns=returns,
            volatility_normalized=vol_normalized.fillna(0),
            vol_regime_asset=vol_regime,
        )

    @staticmethod
    def _resolve_vol_window(timeframe: Optional[str], default_window: int = 20) -> int:
        """Map timeframe strings to a ~5-hour rolling window in bars."""
        if not timeframe:
            return default_window
        tf = str(timeframe).strip().lower()
        minutes = None
        try:
            if tf.endswith("m"):
                minutes = float(tf[:-1])
            elif tf.endswith("h"):
                minutes = float(tf[:-1]) * 60.0
            elif tf.endswith("d"):
                minutes = float(tf[:-1]) * 1440.0
        except Exception:
            minutes = None

        if minutes is None or not np.isfinite(minutes) or minutes <= 0:
            return default_window

        target_minutes = 300.0  # 5 hours
        window = int(round(target_minutes / minutes))
        return max(5, window)

    @staticmethod
    def _compute_data_hash(df: pd.DataFrame, columns: List[str]) -> str:
        """Compute hash of dataframe subset for cache keying."""
        subset = df[columns].head(1000) if len(df) > 1000 else df[columns]
        data_str = pd.util.hash_pandas_object(subset).sum()
        return hashlib.md5(str(data_str).encode()).hexdigest()[:12]

    def _market_residualize_returns_vectorized(
        self,
        df: pd.DataFrame,
        assets: List[str],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Compute market-residualized returns using vectorized pivot operations.
        
        CRITICAL De Prado Principle: Labels must use residual returns (alpha)
        not raw returns (beta + alpha) to prevent the model from learning
        "buy when market goes up" instead of asset-specific predictive patterns.
        
        IMPROVEMENTS:
        1. Vectorized pivot table operations (no Python loops)
        2. BTC benchmark proxy option instead of equal-weighted
        3. LAGGED market return (t-1) to prevent label leakage
        4. EWMA-based beta for O(1) complexity and non-stationarity handling
        5. Cross-sectional rank features
        6. Correlation regime indicator
        7. Disk caching for expensive computations
        
        Args:
            df: Combined dataframe with all assets
            assets: List of asset identifiers
            config: Configuration dict with optional settings
            
        Returns:
            DataFrame with market-residualized features added
        """
        # Check cache first
        cache_dir = Path(config.get('outcomes_dir', 'outcomes')) / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_cols = ['beta_to_market', 'residual_return', 'market_return', 
                      'momentum_rank_xsec', 'relative_volatility', 'correlation_regime', 'corr_regime_cat']
        cache_key = self._compute_data_hash(df, ['raw_returns', 'asset_id', 'volatility_normalized'])
        cache_path = cache_dir / f"market_residualize_{cache_key}.parquet"
        
        if cache_path.exists() and not config.get('force_recompute', False):
            try:
                cached = pd.read_parquet(cache_path)
                if len(cached) == len(df) and all(c in cached.columns for c in cache_cols):
                    tprint_info("   💾 Loaded cached market residualization features")
                    for col in cache_cols:
                        df[col] = cached[col].values
                    return df
            except Exception as e:
                tprint_warning(f"   ⚠️ Cache load failed: {e}, recomputing...")
        
        tprint_info("   Computing market-residualized returns (vectorized, De Prado compliance)...")
        
        # Reset index locally to avoid label/level ambiguity during pivot/groupby
        original_index = df.index
        df = df.reset_index(drop=True)
        
        # Extract timestamp for mapping
        if 'timestamp' in df.columns:
            ts_values = df['timestamp']
        else:
            tprint_warning("   ⚠️ No timestamp found, skipping market residualization")
            return df
        
        ts_col = 'timestamp'
        asset_col = 'asset_id'
        
        # =====================================================================
        # Step 1: Vectorized market return via pivot table
        # =====================================================================
        tprint_info("   Step 1/5: Computing market returns (vectorized pivot)...")
        
        # Pivot: rows=timestamp, columns=asset, values=raw_returns
        pivot_returns = df.pivot_table(
            index=ts_col,
            columns=asset_col,
            values='raw_returns',
            aggfunc='first'
        )
        
        # Use BTC as benchmark proxy if available and configured
        use_btc_benchmark = config.get('use_btc_benchmark', True)
        use_volume_weighted = config.get('use_volume_weighted_market', False)
        
        if use_btc_benchmark and 'BTC' in pivot_returns.columns:
            tprint_info("   Using BTC as market benchmark proxy")
            market_return_series = pivot_returns['BTC']
        elif use_volume_weighted and 'quote_volume' in df.columns:
            # Volume-weighted market return (respects liquidity/mcap)
            tprint_info("   Using volume-weighted market return")
            pivot_volume = df.pivot_table(
                index=ts_col,
                columns=asset_col,
                values='quote_volume',
                aggfunc='first'
            ).fillna(0)
            # Normalize weights per timestamp
            vol_weights = pivot_volume.div(pivot_volume.sum(axis=1) + 1e-9, axis=0)
            market_return_series = (pivot_returns * vol_weights).sum(axis=1)
        else:
            # Equal-weighted market return
            market_return_series = pivot_returns.mean(axis=1)
        
        # CRITICAL: Lag market return by 1 to prevent label leakage
        # At time t, we use market_return from t-1 to compute residual
        use_lagged_market = config.get('use_lagged_market_return', True)
        if use_lagged_market:
            market_return_lagged = market_return_series.shift(1).fillna(0)
            tprint_info("   Using lagged (t-1) market return to prevent leakage")
        else:
            market_return_lagged = market_return_series
        
        # Map back to original dataframe
        market_return_map = market_return_lagged.to_dict()
        df['market_return'] = ts_values.map(market_return_map).fillna(0).astype(np.float32)
        
        # =====================================================================
        # Step 2: Vectorized EWMA beta computation
        # =====================================================================
        tprint_info("   Step 2/5: Computing EWMA betas (Numba-accelerated)...")
        
        beta_halflife = config.get('beta_halflife', self.DEFAULT_BETA_HALFLIFE)
        alpha = 1.0 - np.exp(np.log(0.5) / beta_halflife)
        
        # Initialize columns
        df['beta_to_market'] = 1.0
        df['residual_return'] = 0.0
        
        # Process each asset using EWMA beta
        for asset in assets:
            if asset not in pivot_returns.columns:
                continue
                
            asset_rets = pivot_returns[asset].values.astype(np.float64)
            market_rets = market_return_lagged.values.astype(np.float64)
            
            # Validate data coverage before beta calculation
            valid_pairs = ~(np.isnan(asset_rets) | np.isnan(market_rets))
            coverage = valid_pairs.sum() / len(valid_pairs) if len(valid_pairs) > 0 else 0
            if coverage < self.BETA_MIN_COVERAGE:
                tprint_warning(f"   ⚠️ Asset {asset} has low data coverage ({coverage:.1%} < {self.BETA_MIN_COVERAGE:.0%}), beta may be unreliable")
            
            # Use Numba EWMA cov/var
            ewma_cov, ewma_var = _numba_ewma_cov_var(asset_rets, market_rets, alpha)
            
            # Beta = Cov / Var
            beta = ewma_cov / (ewma_var + 1e-9)
            beta = np.clip(np.nan_to_num(beta, nan=1.0), *self.BETA_CLIP_BOUNDS)
            
            # Create mapping from timestamp to beta
            beta_series = pd.Series(beta, index=pivot_returns.index)
            
            # Residual return (using lagged market for causality)
            residual = asset_rets - beta * market_rets
            
            # CLIP residual returns to prevent outlier contamination (±5σ)
            residual_std = np.nanstd(residual)
            if residual_std > 0:
                clip_bound = self.RESIDUAL_CLIP_SIGMA * residual_std
                residual = np.clip(residual, -clip_bound, clip_bound)
            
            residual_series = pd.Series(residual, index=pivot_returns.index)
            
            # Map back to dataframe
            asset_mask = df['asset_id'] == asset
            df.loc[asset_mask, 'beta_to_market'] = ts_values[asset_mask].map(beta_series.to_dict()).fillna(1.0).values
            df.loc[asset_mask, 'residual_return'] = ts_values[asset_mask].map(residual_series.to_dict()).fillna(0.0).values
            
            # Store stats
            if asset in self.asset_stats:
                self.asset_stats[asset]['beta_mean'] = float(np.nanmean(beta))
                self.asset_stats[asset]['beta_std'] = float(np.nanstd(beta))
                self.asset_stats[asset]['residual_return_std'] = float(np.nanstd(residual))
                self.asset_stats[asset]['beta_data_coverage'] = float(coverage)
        
        # =====================================================================
        # Step 3: Cross-sectional rank features (De Prado recommended)
        # =====================================================================
        tprint_info("   Step 3/5: Adding cross-sectional rank features...")
        
        # Momentum rank: percentile rank of asset momentum vs peers at each timestamp
        if 'rolling_momentum_20' in df.columns:
            pivot_momentum = df.pivot_table(
                index=ts_col,
                columns=asset_col,
                values='rolling_momentum_20',
                aggfunc='first'
            )
            # Rank across columns (assets) per row (timestamp)
            momentum_ranks = pivot_momentum.rank(axis=1, pct=True)
            
            # Melt back and map
            for asset in assets:
                if asset in momentum_ranks.columns:
                    rank_map = momentum_ranks[asset].to_dict()
                    asset_mask = df['asset_id'] == asset
                    df.loc[asset_mask, 'momentum_rank_xsec'] = ts_values[asset_mask].map(rank_map).fillna(0.5)
        else:
            df['momentum_rank_xsec'] = 0.5
        
        # =====================================================================
        # Step 4: Relative volatility (vectorized)
        # =====================================================================
        tprint_info("   Step 4/5: Computing relative volatility (vectorized)...")
        
        pivot_vol = df.pivot_table(
            index=ts_col,
            columns=asset_col,
            values='volatility_normalized',
            aggfunc='first'
        )
        market_vol = pivot_vol.mean(axis=1)
        
        for asset in assets:
            if asset in pivot_vol.columns:
                rel_vol = pivot_vol[asset] / (market_vol.abs() + 1e-9)
                rel_vol_map = rel_vol.to_dict()
                asset_mask = df['asset_id'] == asset
                df.loc[asset_mask, 'relative_volatility'] = ts_values[asset_mask].map(rel_vol_map).fillna(1.0)
        
        if 'relative_volatility' not in df.columns:
            df['relative_volatility'] = 1.0
        
        # =====================================================================
        # Step 5: Correlation regime indicator (Numba-accelerated)
        # =====================================================================
        tprint_info("   Step 5/5: Computing correlation regime indicator...")
        
        if len(assets) >= 2:
            # Intersect assets with available columns to avoid KeyError
            available_assets = [a for a in assets if a in pivot_returns.columns]
            if len(available_assets) >= 2:
                tprint_info(f"   Step 5/5: Computing correlation regime mapping for {len(available_assets)} assets...")
                # Build returns matrix for correlation computation
                returns_matrix = pivot_returns[available_assets].values.astype(np.float64)
            
            avg_corr = _numba_rolling_correlation(
                returns_matrix,
                window=self.CORRELATION_WINDOW,
                min_periods=self.CORRELATION_MIN_PERIODS,
            )
            
            corr_series = pd.Series(avg_corr, index=pivot_returns.index)
            corr_map = corr_series.to_dict()
            df['correlation_regime'] = ts_values.map(corr_map).fillna(0.5).astype(np.float32)
            
            # Categorize: high correlation = risk-off
            df['corr_regime_cat'] = np.where(
                df['correlation_regime'] > 0.7, 'high_corr',
                np.where(df['correlation_regime'] < 0.3, 'low_corr', 'medium_corr')
            )
        else:
            df['correlation_regime'] = 0.5
            df['corr_regime_cat'] = 'medium_corr'
        
        # Final type optimization
        for col in ['beta_to_market', 'residual_return', 'relative_volatility', 'momentum_rank_xsec']:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        # Restore original index
        df.index = original_index
        tprint_success("   ✅ Vectorized market residualization complete")
        return df

    def _combine_asset_data(
        self,
        asset_dataframes: Dict[str, pd.DataFrame],
        assets: List[str],
        config: Dict[str, Any],
        timeframe: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        force_recompute: bool = False,
        force_context_recompute: bool = False,
    ) -> pd.DataFrame:
        """Combine multiple asset dataframes into unified dataset with caching."""
        tprint_info("Combining multi-asset data...")

        cache_path = None
        context_cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            assets_hash = hash((tuple(sorted(assets)), timeframe)) & 0xFFFFFFFF
            cache_path = cache_dir / f"combined_multi_asset_{assets_hash}.parquet"
            context_cache_path = cache_dir / f"market_context_{assets_hash}.parquet"
            if cache_path.exists() and not force_recompute:
                try:
                    tprint_info(f"   💾 Loading cached combined multi-asset data from {cache_path}")
                    cached_df = pd.read_parquet(cache_path)
                    if isinstance(cached_df.index, pd.MultiIndex) and set(cached_df.index.names) >= {'timestamp', 'asset_id'}:
                        return cached_df.sort_index()
                    if {'timestamp', 'asset_id'} <= set(cached_df.columns):
                        return cached_df.set_index(['timestamp', 'asset_id']).sort_index()
                    tprint_warning(
                        "   ⚠️ Cached combined data missing timestamp/asset_id metadata; rebuilding cache"
                    )
                except Exception as cache_err:
                    tprint_warning(f"   ⚠️ Failed to load cached combined data ({cache_err}); rebuilding")

        combined_dfs = []
        for asset in assets:
            df = asset_dataframes.get(asset)
            if df is None or df.empty:
                continue

            df_proc = apply_layer2_price_processing(
                df,
                price_col='close',
                volume_col='quote_volume' if 'quote_volume' in df.columns else 'volume',
                enable_price_features=True,
            )
            df_proc = self._add_asset_features(df_proc, asset)
            df_proc = self._normalize_volatility_per_asset(df_proc, asset, timeframe=timeframe)
            df_proc = self._add_asset_interaction_features(df_proc, asset)

            for col in df_proc.select_dtypes(include=['float64']).columns:
                df_proc[col] = pd.to_numeric(df_proc[col], downcast='float')

            combined_dfs.append(df_proc)

        if not combined_dfs:
            raise ValueError("No valid asset dataframes to combine")

        combined_df = pd.concat(combined_dfs, ignore_index=False)

        if 'timestamp' not in combined_df.columns:
            combined_df = combined_df.copy()
            combined_df['timestamp'] = combined_df.index

        combined_df = combined_df.sort_values('timestamp')

        if 'asset_id' in combined_df.columns:
            combined_df = combined_df.set_index(['timestamp', 'asset_id'], drop=False)

        combined_df = combined_df.sort_index()
        combined_df = self._market_residualize_returns_vectorized(combined_df, assets, config)

        from src.training.steps.labeling.feature_engineering_utils import add_market_context_features

        combined_df = add_market_context_features(
            combined_df,
            asset_col='asset_id',
            timeframe=timeframe,
            cache_path=context_cache_path,
            force_recompute=force_context_recompute or force_recompute,
        )

        for col in combined_df.select_dtypes(include=['float64']).columns:
            combined_df[col] = combined_df[col].astype(np.float32)

        if cache_path is not None:
            try:
                combined_df.to_parquet(cache_path)
                tprint_info(f"   💾 Cached combined multi-asset data to {cache_path}")
            except Exception as cache_write_err:
                tprint_warning(f"   ⚠️ Failed to cache combined multi-asset data: {cache_write_err}")

        tprint_success(f"✅ Combined {len(combined_dfs)} assets into {len(combined_df)} total rows")
        return combined_df

    def _create_single_asset_config(self, global_config: Dict[str, Any], primary_asset: str) -> Dict[str, Any]:
        """Create a single-asset configuration for the underlying step."""
        single_config = global_config.copy()
        
        # Set symbol to primary asset for compatibility
        single_config['symbol'] = f"{primary_asset}USDT"
        
        # Add multi-asset context
        single_config['multi_asset_mode'] = True
        single_config['all_assets'] = global_config.get('assets', [])
        single_config['asset_stats'] = self.asset_stats
        
        return single_config

    def _load_assets_parallel(self, config: Dict[str, Any], assets: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple assets in parallel using ThreadPoolExecutor.
        I/O bound operation benefits from threading.
        """
        asset_dataframes = {}
        errors = []
        
        max_workers = min(len(assets), 4)  # Limit concurrent I/O
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_asset = {
                executor.submit(self._load_asset_data, config, asset): asset
                for asset in assets
            }
            
            for future in as_completed(future_to_asset):
                asset = future_to_asset[future]
                try:
                    asset_df = future.result()
                    asset_dataframes[asset] = asset_df
                except Exception as e:
                    errors.append((asset, str(e)))
        
        if errors:
            for asset, err in errors:
                tprint_error(f"❌ Failed to load {asset} data: {err}")
        
        return asset_dataframes

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the global multi-asset meta-labeling pipeline.
        
        Optimizations:
        - Parallel asset loading
        - Cached underlying step instance
        - Early memory cleanup
        - Vectorized computations throughout
        
        Args:
            config: Configuration dictionary with assets list
            
        Returns:
            Execution result with combined model and asset-specific components
        """
        outcomes_dir = Path(config.get("outcomes_dir", "outcomes"))
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract assets from configuration
            assets = config.get('assets', [])
            if not assets or len(assets) < 2:
                raise ValueError("At least 2 assets required for multi-asset training")
            
            tprint_info(f"🌍 Starting Global Multi-Asset Meta-Labeling HPO")
            tprint_info(f"Assets: {', '.join(assets)}")
            tprint_info(f"Exchange: {config.get('exchange', 'binance')}")
            tprint_info(f"Timeframe: {config.get('timeframe', '15m')}")
            tprint_info(f"Execution Mode: {config.get('execution_mode', 'light')}")
            tprint_info(f"BTC Benchmark: {config.get('use_btc_benchmark', True)}")
            tprint_info(f"Lagged Market Return: {config.get('use_lagged_market_return', True)}")
            
            # Phase 1: Load data for all assets IN PARALLEL
            tprint_info("Phase 1: Loading multi-asset data (parallel)...")
            asset_dataframes = self._load_assets_parallel(config, assets)
            
            if not asset_dataframes:
                return {"success": False, "error": "Failed to load any asset data"}
            
            # Combine asset data with asset-specific features
            tprint_info("Phase 2: Combining assets with asset-specific features...")
            artifacts_dir = Path(config.get('artifacts_dir', 'artifacts')) / 'global_meta_layer2'
            force_recompute_combined = bool(config.get('force_recompute_combined', False))
            force_recompute_market_context = bool(config.get('force_recompute_market_context', False))

            combined_data = self._combine_asset_data(
                asset_dataframes,
                list(asset_dataframes.keys()),  # Only use successfully loaded assets
                config,
                timeframe=config.get('timeframe', '15m'),
                cache_dir=artifacts_dir,
                force_recompute=force_recompute_combined,
                force_context_recompute=force_recompute_market_context,
            )
            
            # MEMORY CLEANUP: Release raw dataframes after combination
            del asset_dataframes
            gc.collect()
            tprint_info("   Memory: Released raw asset dataframes")
            
            # Create single-asset configuration for underlying step
            primary_asset = assets[0]
            single_asset_config = self._create_single_asset_config(config, primary_asset)

            # Provide cross-asset context to downstream layers
            single_asset_config['assets'] = assets
            
            # Inject combined data into configuration
            single_asset_config['market_data'] = combined_data
            single_asset_config['pooled_market_data_ready'] = True
            
            # CRITICAL: Configure labeling to use residual_return instead of raw returns
            single_asset_config['label_return_column'] = 'residual_return'
            single_asset_config['use_market_residual_labels'] = True
            
            # Run the underlying meta-labeling HPO step on combined data
            tprint_info("Phase 3: Running meta-labeling HPO on combined dataset...")
            
            # Cache underlying step instance for potential reuse in HPO loops
            if self._underlying_step_cache is None:
                self._underlying_step_cache = MetaLabelingHPOSampleWeightedStep("meta_labeling_hpo_sample_weighted")
            
            result = await self._underlying_step_cache.execute(single_asset_config)
            
            if not result.get('success', False):
                tprint_error("❌ Underlying meta-labeling HPO failed")
                return result
            
            # Add multi-asset specific results
            result['multi_asset'] = {
                'assets': assets,
                'asset_stats': self.asset_stats,
                'combined_rows': len(combined_data),
                'primary_asset': primary_asset,
                'execution_mode': config.get('multi_asset_mode', 'global'),
                'use_btc_benchmark': config.get('use_btc_benchmark', True),
                'use_lagged_market_return': config.get('use_lagged_market_return', True),
                'beta_halflife': config.get('beta_halflife', self.DEFAULT_BETA_HALFLIFE),
            }
            
            # Save multi-asset metadata
            metadata_file = outcomes_dir / f"global_meta_labeling_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_file, 'w') as f:
                json.dump(result['multi_asset'], f, indent=2, default=str)
            
            result['metadata_file'] = str(metadata_file)
            
            tprint_success("🌍 Global Multi-Asset Meta-Labeling HPO completed successfully")
            tprint_info(f"Results saved to: {metadata_file}")
            
            return result
            
        except Exception as e:
            error_msg = f"Global multi-asset meta-labeling failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            import traceback
            tprint_error(traceback.format_exc())
            return {"success": False, "error": error_msg}


def register_global_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the global meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("global_meta_labeling_hpo_sample_weighted", GlobalMetaLabelingHPOSampleWeightedStep)
    
    tprint_success("✅ Global meta-labeling HPO sample weighted step registered")


# Auto-register the step
register_global_meta_labeling_hpo_sample_weighted_step()
