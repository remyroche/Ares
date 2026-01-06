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
    
    def compute_base_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all base signals needed for composites from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with base signals as columns
        """
        signals = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        close = df.get('close', df.get('Close', pd.Series(dtype=float)))
        high = df.get('high', df.get('High', pd.Series(dtype=float)))
        low = df.get('low', df.get('Low', pd.Series(dtype=float)))
        open_ = df.get('open', df.get('Open', pd.Series(dtype=float)))
        volume = df.get('volume', df.get('Volume', pd.Series(1, index=df.index)))
        
        if close.empty:
            return signals
        
        # === 1. Return-based signals ===
        returns = close.pct_change().fillna(0)
        returns_std = returns.rolling(20).std().fillna(returns.std())
        signals['return_shock'] = (returns.abs() / (returns_std + 1e-9)).fillna(0)
        
        # === 2. Volume-based signals ===
        vol_mean = volume.rolling(20).mean().fillna(volume.mean())
        vol_std = volume.rolling(20).std().fillna(volume.std())
        signals['volume_spike'] = ((volume - vol_mean) / (vol_std + 1e-9)).fillna(0)
        
        # Trade intensity: Volume / True Range (proxy for order book activity)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        signals['trade_intensity'] = (volume / (tr + 1e-9)).fillna(0)
        intensity_mean = signals['trade_intensity'].rolling(20).mean()
        intensity_std = signals['trade_intensity'].rolling(20).std()
        signals['trade_intensity'] = ((signals['trade_intensity'] - intensity_mean) / (intensity_std + 1e-9)).fillna(0)
        
        # === 3. Flow Imbalance (OHLCV proxy for order flow) ===
        # High close relative to range suggests buying pressure
        bar_range = (high - low)
        close_position = (close - low) / (bar_range + 1e-9)  # 0 = closed at low, 1 = closed at high
        signals['flow_imbalance'] = (close_position - 0.5) * 2  # Normalized to [-1, 1]
        
        # Order flow imbalance using volume-weighted bar position
        volume_weighted_position = close_position * volume
        vwp_mean = volume_weighted_position.rolling(20).mean()
        vwp_std = volume_weighted_position.rolling(20).std()
        signals['order_flow_imbalance'] = ((volume_weighted_position - vwp_mean) / (vwp_std + 1e-9)).fillna(0)
        
        # === 4. Volatility-based signals ===
        realized_vol = returns.rolling(20).std() * np.sqrt(252 * 24 * 4)  # Annualized for 15m bars
        vol_of_vol = realized_vol.rolling(20).std()
        signals['volatility_regime'] = realized_vol.rank(pct=True).fillna(0.5)  # 0=low vol, 1=high vol
        signals['volatility_spike'] = ((realized_vol - realized_vol.rolling(50).mean()) / 
                                        (realized_vol.rolling(50).std() + 1e-9)).fillna(0)
        
        # === 5. Trend-based signals ===
        ema_fast = close.ewm(span=8).mean()
        ema_slow = close.ewm(span=21).mean()
        trend = ema_fast - ema_slow
        trend_normalized = trend / (close.rolling(50).std() + 1e-9)
        signals['trend_strength'] = trend_normalized.abs().rank(pct=True).fillna(0.5)
        signals['trend_direction'] = np.sign(trend).fillna(0)
        
        return signals
    
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
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Trade intensity
        intensity = volume / (tr + 1e-9)
        
        # Z-score
        intensity_mean = intensity.rolling(self.window).mean()
        intensity_std = intensity.rolling(self.window).std()
        intensity_z = (intensity - intensity_mean) / (intensity_std + 1e-9)
        
        # Events
        mask = intensity_z.abs() >= self.threshold
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
        
        # Bar position: 0 = low, 1 = high
        bar_range = (high - low)
        close_position = (close - low) / (bar_range + 1e-9)
        
        # Volume-weighted position for stronger signal
        if volume is not None:
            flow = (close_position - 0.5) * volume
        else:
            flow = close_position - 0.5
        
        # Z-score
        flow_mean = flow.rolling(self.window).mean()
        flow_std = flow.rolling(self.window).std()
        flow_z = (flow - flow_mean) / (flow_std + 1e-9)
        
        # Events: extreme buying or selling pressure
        mask = flow_z.abs() >= self.threshold
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
        
        # Directional pressure: (close - open) / range
        bar_range = (high - low)
        direction = (close - open_) / (bar_range + 1e-9)
        
        # Volume-weighted for significance
        if volume is not None:
            vol_normalized = volume / volume.rolling(self.window).mean()
            pressure = direction * vol_normalized
        else:
            pressure = direction
        
        # Z-score
        pressure_mean = pressure.rolling(self.window).mean()
        pressure_std = pressure.rolling(self.window).std()
        pressure_z = (pressure - pressure_mean) / (pressure_std + 1e-9)
        
        mask = pressure_z.abs() >= self.threshold
        return df.index[mask]


# Factory function for easy registration
def get_microstructure_generators() -> Dict[str, Any]:
    """Return dictionary of microstructure proxy generators."""
    return {
        'TRADE_INTENSITY': TradeIntensityEvents(threshold=2.0, window=20),
        'ORDER_FLOW_IMBALANCE': OrderFlowImbalanceEvents(threshold=2.0, window=20),
        'BAR_PRESSURE': BarPressureEvents(threshold=2.0, window=20),
    }


def get_composite_generator() -> CompositeEventGenerator:
    """Return configured composite event generator."""
    return CompositeEventGenerator(verbose=True)
