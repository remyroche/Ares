"""Composite Event Generators for Higher-Specificity Signals.

This module implements cross-signal composite events that combine multiple
conditions to generate higher-specificity trading signals. These composites
are designed to increase event density while maintaining causal integrity.

Composite Logic Types:
- AND: Both conditions must be true (high precision, low recall)
- OR: Either condition true (high recall, lower precision)
- CONFIRMATION: Primary signal confirmed by secondary (balanced)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from src.utils.numba_funcs import (
    _numba_rolling_mean,
    _numba_rolling_std,
    _numba_rolling_mean_nan_safe,
    _numba_rolling_std_nan_safe,
    _numba_shift
)

logger = logging.getLogger(__name__)


@dataclass
class CompositeSignalConfig:
    """Configuration for a composite signal."""
    name: str
    family: str
    primary_signal: str
    secondary_signal: str
    logic: str  # 'AND', 'OR', 'CONFIRMATION'
    primary_threshold: float = 2.0
    secondary_threshold: float = 2.0
    confirmation_window: int = 3  # bars for confirmation logic


class CompositeEventGenerator:
    """
    Generate composite events from cross-signal combinations.
    
    Implements three types of signal combinations:
    1. AND Logic: Both signals must fire simultaneously
    2. OR Logic: Either signal fires
    3. CONFIRMATION: Primary signal confirmed by secondary within window
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._composite_configs = self._build_default_configs()
    
    def _build_default_configs(self) -> List[CompositeSignalConfig]:
        """Build default composite signal configurations."""
        return [
            # Volume Spike × Trend Regime
            CompositeSignalConfig(
                name="VOLUME_SPIKE_X_TREND",
                family="COMPOSITE_VOLUME_TREND",
                primary_signal="volume_spike",
                secondary_signal="trend_strength",
                logic="AND",
                primary_threshold=2.0,
                secondary_threshold=0.6,
            ),
            # Flow Imbalance × High Volatility
            CompositeSignalConfig(
                name="FLOW_IMBALANCE_X_VOL",
                family="COMPOSITE_FLOW_VOL",
                primary_signal="flow_imbalance",
                secondary_signal="volatility_regime",
                logic="AND",
                primary_threshold=1.8,
                secondary_threshold=0.7,
            ),
            # Return Shock × Volume Confirmation
            CompositeSignalConfig(
                name="RETURN_SHOCK_CONFIRMED",
                family="COMPOSITE_RETURN_VOLUME",
                primary_signal="return_shock",
                secondary_signal="volume_spike",
                logic="CONFIRMATION",
                primary_threshold=2.5,
                secondary_threshold=1.5,
                confirmation_window=3,
            ),
            # Trade Intensity × Volatility Spike
            CompositeSignalConfig(
                name="INTENSITY_X_VOL_SPIKE",
                family="COMPOSITE_INTENSITY_VOL",
                primary_signal="trade_intensity",
                secondary_signal="volatility_spike",
                logic="AND",
                primary_threshold=2.0,
                secondary_threshold=2.0,
            ),
            # Order Flow × Trend Confirmation
            CompositeSignalConfig(
                name="ORDER_FLOW_X_TREND",
                family="COMPOSITE_FLOW_TREND",
                primary_signal="order_flow_imbalance",
                secondary_signal="trend_direction",
                logic="CONFIRMATION",
                primary_threshold=1.8,
                secondary_threshold=0.5,
                confirmation_window=5,
            ),
        ]
    
    def compute_base_signals(self, df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
        """
        Compute all base signals needed for composites from OHLCV data.
        
        Optimized for performance (float32, Numba) and causal safety (rolling rank).

        Args:
            df: DataFrame with OHLCV columns
            freq: Optional frequency string (e.g., '15min'). If None, inferred or default.
            
        Returns:
            DataFrame with base signals as columns
        """
        signals = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        close = df.get('close', df.get('Close', pd.Series(dtype=np.float32)))
        high = df.get('high', df.get('High', pd.Series(dtype=np.float32)))
        low = df.get('low', df.get('Low', pd.Series(dtype=np.float32)))
        volume = df.get('volume', df.get('Volume'))
        
        if close.empty:
            return signals

        # Convert to float32 for performance
        close_vals = close.values.astype(np.float32)
        high_vals = high.values.astype(np.float32) if not high.empty else close_vals
        low_vals = low.values.astype(np.float32) if not low.empty else close_vals

        if volume is not None:
            vol_vals = volume.values.astype(np.float32)
        else:
            vol_vals = None
        
        # === 1. Return-based signals ===
        # Use pandas for pct_change (efficient C impl) but cast to float32
        returns = close.pct_change().fillna(0).astype(np.float32)
        returns_vals = returns.values
        
        # Optimized rolling std using Numba
        # Returns are 0-filled, so standard _numba_rolling_std is fine
        returns_std = _numba_rolling_std(returns_vals, 20)
        
        # Avoid div by zero
        signals['return_shock'] = np.abs(returns_vals) / (returns_std + 1e-9)

        # === 2. Volume-based signals ===
        if vol_vals is not None:
            # Use nan-safe functions instead of pre-filling with 0 (which skews stats)
            vol_mean = _numba_rolling_mean_nan_safe(vol_vals, 20)
            vol_std = _numba_rolling_std_nan_safe(vol_vals, 20)

            # Handle possible NaNs in output of rolling functions (e.g. all NaNs in window)
            vol_mean = np.nan_to_num(vol_mean, nan=0.0)
            vol_std = np.nan_to_num(vol_std, nan=0.0)

            # For signal calculation, fill vol_vals NaNs with 0 temporarily if needed,
            # but ideally we propagate NaNs or handle them.
            # Original code filled with 0.
            vol_vals_clean = np.nan_to_num(vol_vals, nan=0.0)

            signals['volume_spike'] = (vol_vals_clean - vol_mean) / (vol_std + 1e-9)

            # Trade intensity: Volume / True Range (proxy for order book activity)
            # TR = max(H-L, |H-Cp|, |L-Cp|)
            # Simple TR approximation for speed: High - Low (intraday)
            # Full TR requires shift.

            # Fix cyclic shift bug using _numba_shift
            prev_close = _numba_shift(close_vals, 1, fill_value=np.nan)
            if len(prev_close) > 0:
                prev_close[0] = close_vals[0] # Pad first with current close (approximation)

            tr1 = high_vals - low_vals
            tr2 = np.abs(high_vals - prev_close)
            tr3 = np.abs(low_vals - prev_close)

            # Handle potential NaNs from shift (though padded above)
            tr2 = np.nan_to_num(tr2, nan=0.0)
            tr3 = np.nan_to_num(tr3, nan=0.0)

            tr = np.maximum(tr1, np.maximum(tr2, tr3))

            intensity = vol_vals_clean / (tr + 1e-9)

            # Rolling Z-score of intensity
            # Use nan-safe here too just in case
            int_mean = _numba_rolling_mean_nan_safe(intensity, 20)
            int_std = _numba_rolling_std_nan_safe(intensity, 20)

            int_mean = np.nan_to_num(int_mean, nan=0.0)
            int_std = np.nan_to_num(int_std, nan=0.0)

            signals['trade_intensity'] = (intensity - int_mean) / (int_std + 1e-9)
        else:
            signals['volume_spike'] = 0.0
            signals['trade_intensity'] = 0.0
        
        # === 3. Flow Imbalance (OHLCV proxy for order flow) ===
        # High close relative to range suggests buying pressure
        bar_range = (high_vals - low_vals)
        # Close Position: (C - L) / (H - L)
        # Handle zero range
        denom = bar_range + 1e-9
        close_position = (close_vals - low_vals) / denom

        # Clip to [0, 1] to handle data errors
        close_position = np.clip(close_position, 0.0, 1.0)

        signals['flow_imbalance'] = (close_position - 0.5) * 2  # Normalized to [-1, 1]
        
        # Order flow imbalance using volume-weighted bar position
        if vol_vals is not None:
            # vol_vals_clean already defined above
            volume_weighted_position = close_position * vol_vals_clean
            vwp_mean = _numba_rolling_mean_nan_safe(volume_weighted_position, 20)
            vwp_std = _numba_rolling_std_nan_safe(volume_weighted_position, 20)

            vwp_mean = np.nan_to_num(vwp_mean, nan=0.0)
            vwp_std = np.nan_to_num(vwp_std, nan=0.0)

            signals['order_flow_imbalance'] = (volume_weighted_position - vwp_mean) / (vwp_std + 1e-9)
        else:
            signals['order_flow_imbalance'] = signals['flow_imbalance']
        
        # === 4. Volatility-based signals ===
        # Infer frequency scaling
        if freq is None and isinstance(df.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(df.index)
            freq = inferred if inferred else '15min' # Default fallback

        # Estimate bars per year
        # Standard: 15m -> 96/day -> 252 days -> 24192
        # Simple heuristic mapping
        bars_per_year = 252 * 96 # Default 15m
        if freq:
            if 'h' in freq.lower():
                bars_per_year = 252 * 24 # Hourly
            elif 'd' in freq.lower():
                bars_per_year = 252 # Daily
            elif 'm' in freq.lower():
                try:
                    mins = int(''.join(filter(str.isdigit, freq)))
                    bars_per_year = 252 * (1440 / max(1, mins))
                except:
                    pass

        annualization = np.sqrt(bars_per_year)

        # Realized Volatility (Annualized)
        realized_vol = returns_std * annualization

        # Volatility Regime: Rolling Rank (Fix Look-ahead Bias)
        # Replacing global rank(pct=True) with rolling rank
        realized_vol_series = pd.Series(realized_vol, index=df.index)
        signals['volatility_regime'] = realized_vol_series.rolling(window=2000, min_periods=200).rank(pct=True).fillna(0.5)

        # Volatility Spike (Z-score of realized vol)
        # Using longer window for regime baseline (50)
        rv_mean = _numba_rolling_mean(realized_vol, 50)
        rv_std = _numba_rolling_std(realized_vol, 50)
        signals['volatility_spike'] = (realized_vol - rv_mean) / (rv_std + 1e-9)
        
        # === 5. Trend-based signals ===
        # MACD-like trend proxy
        ema_fast = close.ewm(span=8).mean().values.astype(np.float32)
        ema_slow = close.ewm(span=21).mean().values.astype(np.float32)
        trend = ema_fast - ema_slow
        
        # Normalize trend by price volatility
        close_std_50 = _numba_rolling_std(close_vals, 50)
        trend_normalized = trend / (close_std_50 + 1e-9)

        # Trend Strength: Rolling Rank (Fix Look-ahead Bias)
        trend_norm_series = pd.Series(np.abs(trend_normalized), index=df.index)
        signals['trend_strength'] = trend_norm_series.rolling(window=2000, min_periods=200).rank(pct=True).fillna(0.5)

        signals['trend_direction'] = np.sign(trend)

        return signals.fillna(0.0)
    
    def generate_composite_events(
        self, 
        df: pd.DataFrame,
        configs: Optional[List[CompositeSignalConfig]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate composite events for all configured signals.
        
        Args:
            df: OHLCV DataFrame
            configs: Optional list of composite configs (uses defaults if None)
            
        Returns:
            Dict mapping composite name to DataFrame with events
        """
        if configs is None:
            configs = self._composite_configs
        
        # Compute base signals
        base_signals = self.compute_base_signals(df)
        if base_signals.empty:
            return {}
        
        results = {}
        
        for config in configs:
            try:
                events_df = self._generate_single_composite(base_signals, config)
                if not events_df.empty and len(events_df) >= 10:
                    results[config.name] = events_df
                    if self.verbose:
                        logger.info(f"✅ Composite {config.name}: {len(events_df)} events")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"⚠️ Failed to generate {config.name}: {e}")
        
        return results
    
    def _generate_single_composite(
        self, 
        signals: pd.DataFrame, 
        config: CompositeSignalConfig
    ) -> pd.DataFrame:
        """Generate events for a single composite configuration."""
        
        primary = signals.get(config.primary_signal)
        secondary = signals.get(config.secondary_signal)
        
        if primary is None or secondary is None:
            return pd.DataFrame()
        
        if config.logic == "AND":
            # Both must exceed threshold simultaneously
            mask = (primary.abs() >= config.primary_threshold) & \
                   (secondary.abs() >= config.secondary_threshold)
        
        elif config.logic == "OR":
            # Either exceeds threshold
            mask = (primary.abs() >= config.primary_threshold) | \
                   (secondary.abs() >= config.secondary_threshold)
        
        elif config.logic == "CONFIRMATION":
            # Primary fires, confirmed by secondary within window
            primary_fires = primary.abs() >= config.primary_threshold
            secondary_fires = secondary.abs() >= config.secondary_threshold
            
            # Rolling max of secondary signal over confirmation window
            secondary_rolling = secondary_fires.rolling(
                config.confirmation_window, min_periods=1
            ).max().fillna(0)
            
            mask = primary_fires & (secondary_rolling > 0)
        
        else:
            return pd.DataFrame()
        
        # Extract event timestamps
        event_times = signals.index[mask]
        
        if len(event_times) == 0:
            return pd.DataFrame()
        
        # Build events DataFrame
        events_df = pd.DataFrame(index=event_times)
        events_df['family'] = config.family
        events_df['config'] = config.name
        events_df['primary_signal'] = primary.loc[event_times].values
        events_df['secondary_signal'] = secondary.loc[event_times].values
        events_df['composite_strength'] = (
            events_df['primary_signal'].abs() + 
            events_df['secondary_signal'].abs()
        ) / 2
        
        return events_df


# ==========================================
# OHLCV Microstructure Proxy Generators
# ==========================================

class TradeIntensityEvents:
    """
    Detect trade intensity spikes from OHLCV data.
    
    Trade intensity = Volume / True Range (proxy for order book activity)
    High intensity suggests aggressive trading without moving price much,
    often preceding breakouts.
    """
    
    def __init__(self, threshold: float = 2.0, window: int = 20):
        self.threshold = threshold
        self.window = window
    
    def generate(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Generate trade intensity events."""
        close = df.get('close', df.get('Close'))
        high = df.get('high', df.get('High'))
        low = df.get('low', df.get('Low'))
        volume = df.get('volume', df.get('Volume'))
        
        if any(x is None for x in [close, high, low, volume]):
            return pd.DatetimeIndex([])
        
        # Use Numba-optimized path if possible, but keep simple for now to match interface
        # Cast to float32
        close_vals = close.values.astype(np.float32)
        high_vals = high.values.astype(np.float32)
        low_vals = low.values.astype(np.float32)
        vol_vals = volume.values.astype(np.float32)

        # True Range Approximation
        # Fix cyclic shift bug
        prev_close = _numba_shift(close_vals, 1, fill_value=np.nan)
        if len(prev_close) > 0:
            prev_close[0] = close_vals[0]

        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - prev_close)
        tr3 = np.abs(low_vals - prev_close)

        # Handle potential NaNs
        tr2 = np.nan_to_num(tr2, nan=0.0)
        tr3 = np.nan_to_num(tr3, nan=0.0)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Trade intensity
        intensity = vol_vals / (tr + 1e-9)
        
        # Z-score using Numba Nan-Safe
        intensity_mean = _numba_rolling_mean_nan_safe(intensity, self.window)
        intensity_std = _numba_rolling_std_nan_safe(intensity, self.window)

        intensity_mean = np.nan_to_num(intensity_mean, nan=0.0)
        intensity_std = np.nan_to_num(intensity_std, nan=0.0)

        intensity_z = (intensity - intensity_mean) / (intensity_std + 1e-9)
        
        # Events
        mask = np.abs(intensity_z) >= self.threshold
        return df.index[mask]


class OrderFlowImbalanceEvents:
    """
    Detect order flow imbalance from OHLCV data.
    
    Uses bar close position relative to range as proxy for buying/selling pressure.
    Close near high = buying pressure, close near low = selling pressure.
    """
    
    def __init__(self, threshold: float = 2.0, window: int = 20):
        self.threshold = threshold
        self.window = window
    
    def generate(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Generate order flow imbalance events."""
        close = df.get('close', df.get('Close'))
        high = df.get('high', df.get('High'))
        low = df.get('low', df.get('Low'))
        volume = df.get('volume', df.get('Volume'))
        
        if any(x is None for x in [close, high, low]):
            return pd.DatetimeIndex([])
        
        close_vals = close.values.astype(np.float32)
        high_vals = high.values.astype(np.float32)
        low_vals = low.values.astype(np.float32)

        # Bar position: 0 = low, 1 = high
        bar_range = (high_vals - low_vals)
        close_position = (close_vals - low_vals) / (bar_range + 1e-9)
        
        # Clip to [0, 1]
        close_position = np.clip(close_position, 0.0, 1.0)

        # Volume-weighted position for stronger signal
        if volume is not None:
            vol_vals = volume.values.astype(np.float32)
            flow = (close_position - 0.5) * vol_vals
        else:
            flow = close_position - 0.5
        
        # Z-score
        flow_mean = _numba_rolling_mean_nan_safe(flow, self.window)
        flow_std = _numba_rolling_std_nan_safe(flow, self.window)

        flow_mean = np.nan_to_num(flow_mean, nan=0.0)
        flow_std = np.nan_to_num(flow_std, nan=0.0)

        flow_z = (flow - flow_mean) / (flow_std + 1e-9)
        
        # Events: extreme buying or selling pressure
        mask = np.abs(flow_z) >= self.threshold
        return df.index[mask]


class BarPressureEvents:
    """
    Detect bar pressure events from OHLCV data.
    
    Bar pressure = (Close - Open) / Range normalized by volume.
    Strong directional bars with high volume suggest institutional activity.
    """
    
    def __init__(self, threshold: float = 2.0, window: int = 20):
        self.threshold = threshold
        self.window = window
    
    def generate(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Generate bar pressure events."""
        close = df.get('close', df.get('Close'))
        open_ = df.get('open', df.get('Open'))
        high = df.get('high', df.get('High'))
        low = df.get('low', df.get('Low'))
        volume = df.get('volume', df.get('Volume'))
        
        if any(x is None for x in [close, open_, high, low]):
            return pd.DatetimeIndex([])
        
        close_vals = close.values.astype(np.float32)
        open_vals = open_.values.astype(np.float32)
        high_vals = high.values.astype(np.float32)
        low_vals = low.values.astype(np.float32)

        # Directional pressure: (close - open) / range
        bar_range = (high_vals - low_vals)
        direction = (close_vals - open_vals) / (bar_range + 1e-9)
        
        # Volume-weighted for significance
        if volume is not None:
            vol_vals = volume.values.astype(np.float32)
            vol_mean = _numba_rolling_mean_nan_safe(vol_vals, self.window)
            vol_normalized = vol_vals / (vol_mean + 1e-9)
            pressure = direction * vol_normalized
        else:
            pressure = direction
        
        # Z-score
        pressure_mean = _numba_rolling_mean_nan_safe(pressure, self.window)
        pressure_std = _numba_rolling_std_nan_safe(pressure, self.window)

        pressure_mean = np.nan_to_num(pressure_mean, nan=0.0)
        pressure_std = np.nan_to_num(pressure_std, nan=0.0)

        pressure_z = (pressure - pressure_mean) / (pressure_std + 1e-9)
        
        mask = np.abs(pressure_z) >= self.threshold
        return df.index[mask]


class CrossAssetEventGenerator:
    """
    Generate events based on Cross-Asset features (ca__*).
    
    Logic:
    1. Computes Rolling Z-Score for each feature (window=300).
    2. Identifies events exceeding a fixed threshold (e.g. 2.2 sigma).
    
    Triggers:
    1. Lead-Lag: Market leads asset significantly (predictive signal).
    2. Beta Dislocation: Short-term beta diverges from long-term beta.
    3. Shocks: Volatility or Volume shocks relative to market.
    """
    
    def __init__(self, window: int = 300, quantile_threshold: float = 2.2):
        self.window = window
        # Renamed for clarity, though keeping name argument compatible if possible
        # Interpreting quantile_threshold as Z-score threshold now if > 1.0
        # If < 1.0, it was a quantile. 0.97 quantile is approx 1.88 sigma (one-sided) or 2.17 (two-sided normal).
        # We enforce a fixed threshold to avoid look-ahead bias.
        self.threshold_sigma = quantile_threshold if quantile_threshold > 1.0 else 2.2

        self.feature_map = {
            'lead_lag': 'ca__lead_lag_w48',
            'beta_spread': 'ca__beta_shift', 
            'vol_shock': 'ca__vol_shock',
            'volume_shock': 'ca__volume_shock'
        }
    
    def _compute_rolling_z(self, series: pd.Series) -> pd.Series:
        """Compute rolling Z-score using Numba for performance."""
        # Convert to numpy float32 and handle NaNs
        values = series.fillna(0.0).values.astype(np.float32)
        
        # Calculate Rolling Mean and Std using Numba optimized kernels
        mu = _numba_rolling_mean(values, self.window)
        sigma = _numba_rolling_std(values, self.window)
        
        # Compute Z-Score
        z_scores = np.zeros_like(values)
        mask = sigma > 1e-9
        z_scores[mask] = (values[mask] - mu[mask]) / sigma[mask]
        
        return pd.Series(z_scores, index=series.index)

    def generate(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Generate cross-asset events."""
        events = pd.DatetimeIndex([])
        
        for name, col in self.feature_map.items():
            if col not in df.columns:
                # Log warning?
                # logger.warning(f"Feature {col} not found in DataFrame")
                continue
                
            # 1. Compute Rolling Z-Score
            raw_series = df[col]
            z_scores = self._compute_rolling_z(raw_series)
            
            # 2. Trigger Events using Fixed Threshold (No Look-ahead)
            abs_z = z_scores.abs()
            mask = abs_z > self.threshold_sigma
            
            if mask.any():
                events = events.union(df.index[mask])
                
        return events


# Factory function for easy registration
def get_microstructure_generators() -> Dict[str, Any]:
    """Return dictionary of microstructure proxy generators."""
    return {
        'TRADE_INTENSITY': TradeIntensityEvents(threshold=2.0, window=20),
        'ORDER_FLOW_IMBALANCE': OrderFlowImbalanceEvents(threshold=2.0, window=20),
        'BAR_PRESSURE': BarPressureEvents(threshold=2.0, window=20),
        'CROSS_ASSET_SPECIALIST': CrossAssetEventGenerator(),
    }


def get_composite_generator() -> CompositeEventGenerator:
    """Return configured composite event generator."""
    return CompositeEventGenerator(verbose=True)
