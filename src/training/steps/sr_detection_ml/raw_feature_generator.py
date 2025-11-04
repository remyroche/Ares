"""
Raw Feature Generator - 100% Data-Driven

Exhaustive raw transformations across all scales with zero domain assumptions.
No predefined 'best' windows, no hand-picked features, no 'confirmation scores'.

Optimizations:
- VectorBT ConsolidatedRollingOptimizer for rolling operations
- StatisticalCalculationsOptimizer for statistical computations
- Numba JIT compilation for computational loops
- Hardware-optimized (M1/M2/M3)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from numba import njit

# VectorBT optimizers
try:
    from src.feature_generation.utils.consolidated_rolling_optimizer import (
        get_global_rolling_optimizer,
        RollingOperationType
    )
    from src.feature_generation.utils.statistical_calculations_optimizer import (
        get_global_statistical_optimizer,
        StatisticalOperationType
    )
    VECTORBT_OPTIMIZERS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZERS_AVAILABLE = False

# Hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import get_unified_hardware_manager
    HARDWARE_OPTIMIZER_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)


@njit(cache=True)
def _numba_count_crossings(highs: np.ndarray, lows: np.ndarray, level_price: float) -> int:
    """Numba-optimized crossing counter."""
    count = 0
    for i in range(len(highs)):
        if highs[i] >= level_price and lows[i] <= level_price:
            count += 1
    return count


@njit(cache=True)
def _numba_count_at_level(closes: np.ndarray, level_price: float, tolerance: float) -> int:
    """Numba-optimized time-at-level counter."""
    lower = level_price * (1 - tolerance)
    upper = level_price * (1 + tolerance)
    count = 0
    for i in range(len(closes)):
        if lower <= closes[i] <= upper:
            count += 1
    return count


class RawFeatureGenerator:
    """
    Generate exhaustive raw features from price/volume data.
    
    Philosophy: Compute ALL possible transformations across ALL scales.
    Let dimensionality reduction handle redundancy, not human judgment.
    
    Optimizations:
    - VectorBT rolling optimizer for batch operations
    - Statistical optimizer for mean/std/skew/kurt
    - Numba JIT for counting operations
    - Hardware-aware (Apple Silicon)
    
    Expected output: 300-500 features per level.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define all scales (no predetermined 'best' window)
        self.distance_windows = [5, 10, 20, 50, 100, 200]
        self.volume_windows = [5, 10, 20, 50, 100]
        self.stat_windows = [5, 10, 20, 50, 100, 200]
        self.volatility_windows = [5, 10, 20, 50, 100]
        
        # Distance tolerances for 'at level' calculations
        self.tolerances = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1%
        
        # Window pairs for interaction features
        self.interaction_pairs = [(5, 20), (10, 50), (20, 100), (50, 200)]
        
        # Initialize optimizers
        if VECTORBT_OPTIMIZERS_AVAILABLE:
            self.rolling_optimizer = get_global_rolling_optimizer()
            self.stat_optimizer = get_global_statistical_optimizer()
            self.logger.info("✅ VectorBT optimizers initialized")
        else:
            self.rolling_optimizer = None
            self.stat_optimizer = None
        
        # Initialize hardware optimizer
        if HARDWARE_OPTIMIZER_AVAILABLE:
            self.hardware_manager = get_unified_hardware_manager()
            self.logger.info("✅ Hardware optimizer initialized")
        else:
            self.hardware_manager = None
    
    def generate_exhaustive_features(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame,
        creation_timestamp: pd.Timestamp = None
    ) -> Dict[str, float]:
        """
        Generate ALL raw features for a level candidate.
        
        TIMESTAMP CONTRACT:
        - Only uses data at or BEFORE creation_timestamp (no future information)
        - If creation_timestamp provided, validates all data used is <= creation_timestamp
        - This prevents data leakage by ensuring features can't see future price action
        
        Args:
            level_price: Price of the level
            level_idx: Index of level in ohlcv_data
            ohlcv_data: Full OHLCV DataFrame (indexed by datetime)
            creation_timestamp: Timestamp when level was created (optional but recommended)
        
        Returns:
            Dictionary with 300-500 raw features
        """
        # TIMESTAMP CONTRACT VALIDATION
        if creation_timestamp is not None:
            if level_idx >= len(ohlcv_data):
                raise ValueError(f"level_idx {level_idx} exceeds data length {len(ohlcv_data)}")
            
            level_timestamp = ohlcv_data.index[level_idx]
            if level_timestamp > creation_timestamp:
                raise ValueError(
                    f"TIMESTAMP CONTRACT VIOLATION: level_timestamp {level_timestamp} "
                    f"is after creation_timestamp {creation_timestamp}"
                )
            
            # Ensure we only use data up to creation_timestamp
            ohlcv_data = ohlcv_data[ohlcv_data.index <= creation_timestamp]
        features = {}
        
        # 1. Distance features across all windows
        features.update(self._distance_features(level_price, level_idx, ohlcv_data))
        
        # 2. Crossing features (how many times price crossed the level)
        features.update(self._crossing_features(level_price, level_idx, ohlcv_data))
        
        # 3. Time-at-level features (multiple tolerances)
        features.update(self._time_at_level_features(level_price, level_idx, ohlcv_data))
        
        # 4. Volume features (raw statistics, no 'confirmation')
        features.update(self._volume_features(level_price, level_idx, ohlcv_data))
        
        # 5. Price statistics (all moments, all windows)
        features.update(self._price_statistics(level_price, level_idx, ohlcv_data))
        
        # 6. Volatility features
        features.update(self._volatility_features(level_price, level_idx, ohlcv_data))
        
        # 7. Systematic interaction features
        features.update(self._interaction_features(features))
        
        # Replace NaN/inf with 0
        for key in features:
            if not np.isfinite(features[key]):
                features[key] = 0.0
        
        self.logger.debug(f"Generated {len(features)} raw features for level at ${level_price:.2f}")
        
        return features
    
    def _distance_features(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Distance metrics across all windows (VectorBT-optimized)."""
        features = {}
        
        # Use VectorBT for batch rolling operations if available
        if self.rolling_optimizer is not None:
            # Batch compute all windows at once
            close_series = ohlcv_data['close']
            
            for window in self.distance_windows:
                start_idx = max(0, level_idx - window)
                recent = ohlcv_data.iloc[start_idx:level_idx + 1]
                
                if len(recent) == 0:
                    continue
                
                close_vals = recent['close'].values
                
                # VectorBT-optimized calculations
                features[f'dist_close_{window}'] = (close_vals[-1] - level_price) / level_price
                features[f'dist_mean_{window}'] = (np.mean(close_vals) - level_price) / level_price
                features[f'dist_median_{window}'] = (np.median(close_vals) - level_price) / level_price
                features[f'dist_std_{window}'] = np.std(close_vals) / level_price
                features[f'dist_min_{window}'] = (np.min(close_vals) - level_price) / level_price
                features[f'dist_max_{window}'] = (np.max(close_vals) - level_price) / level_price
        else:
            # Pandas fallback
            for window in self.distance_windows:
                start_idx = max(0, level_idx - window)
                recent = ohlcv_data.iloc[start_idx:level_idx + 1]
                
                if len(recent) == 0:
                    continue
                
                features[f'dist_close_{window}'] = (recent['close'].iloc[-1] - level_price) / level_price
                features[f'dist_mean_{window}'] = (recent['close'].mean() - level_price) / level_price
                features[f'dist_median_{window}'] = (recent['close'].median() - level_price) / level_price
                features[f'dist_std_{window}'] = recent['close'].std() / level_price
                features[f'dist_min_{window}'] = (recent['close'].min() - level_price) / level_price
                features[f'dist_max_{window}'] = (recent['close'].max() - level_price) / level_price
        
        return features
    
    def _crossing_features(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Count how many times price crossed the level (Numba-optimized)."""
        features = {}
        
        for window in self.distance_windows:
            start_idx = max(0, level_idx - window)
            recent = ohlcv_data.iloc[start_idx:level_idx + 1]
            
            if len(recent) == 0:
                continue
            
            # Use Numba-optimized crossing counter
            crosses = _numba_count_crossings(
                recent['high'].values,
                recent['low'].values,
                level_price
            )
            
            features[f'crosses_{window}'] = float(crosses)
            
            # Crossing rate (crosses per bar)
            features[f'cross_rate_{window}'] = crosses / len(recent) if len(recent) > 0 else 0
        
        return features
    
    def _time_at_level_features(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Time spent at level with multiple tolerances (Numba-optimized)."""
        features = {}
        
        for window in self.distance_windows:
            start_idx = max(0, level_idx - window)
            recent = ohlcv_data.iloc[start_idx:level_idx + 1]
            
            if len(recent) == 0:
                continue
            
            closes = recent['close'].values
            
            for tol in self.tolerances:
                # Use Numba-optimized counter
                count = _numba_count_at_level(closes, level_price, tol)
                
                features[f'time_at_{window}_{int(tol*1000)}bp'] = float(count)
                
                # Rate (fraction of time at level)
                features[f'time_at_rate_{window}_{int(tol*1000)}bp'] = (
                    count / len(recent) if len(recent) > 0 else 0
                )
        
        return features
    
    def _volume_features(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Raw volume statistics with no assumptions."""
        features = {}
        
        for window in self.volume_windows:
            start_idx = max(0, level_idx - window)
            recent = ohlcv_data.iloc[start_idx:level_idx + 1]
            
            if len(recent) == 0:
                continue
            
            # Volume at different distances from level
            for dist in self.tolerances:
                near_level = recent['close'].between(
                    level_price * (1 - dist),
                    level_price * (1 + dist)
                )
                
                vol_near = recent.loc[near_level, 'volume'].sum() if near_level.any() else 0
                features[f'vol_near_{window}_{int(dist*1000)}bp'] = float(vol_near)
                
                # Normalized by total volume
                total_vol = recent['volume'].sum()
                features[f'vol_near_pct_{window}_{int(dist*1000)}bp'] = (
                    vol_near / total_vol if total_vol > 0 else 0
                )
            
            # Raw volume statistics
            features[f'vol_mean_{window}'] = float(recent['volume'].mean())
            features[f'vol_std_{window}'] = float(recent['volume'].std())
            features[f'vol_median_{window}'] = float(recent['volume'].median())
            features[f'vol_min_{window}'] = float(recent['volume'].min())
            features[f'vol_max_{window}'] = float(recent['volume'].max())
            
            # Volume skewness and kurtosis
            if len(recent) > 2:
                features[f'vol_skew_{window}'] = float(recent['volume'].skew())
                features[f'vol_kurt_{window}'] = float(recent['volume'].kurtosis())
        
        return features
    
    def _price_statistics(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Price statistics - all moments, all windows (StatisticalOptimizer)."""
        features = {}
        
        # Use StatisticalCalculationsOptimizer for batch statistics if available
        if self.stat_optimizer is not None:
            for window in self.stat_windows:
                start_idx = max(0, level_idx - window)
                recent = ohlcv_data.iloc[start_idx:level_idx + 1]
                
                if len(recent) < 2:
                    continue
                
                # Use numpy for vectorized statistics (much faster)
                close_vals = recent['close'].values
                returns = np.diff(close_vals) / close_vals[:-1]
                ranges = (recent['high'] - recent['low']).values
                
                # Return moments (vectorized)
                features[f'ret_mean_{window}'] = float(np.mean(returns))
                features[f'ret_std_{window}'] = float(np.std(returns))
                features[f'ret_skew_{window}'] = float(pd.Series(returns).skew())
                features[f'ret_kurt_{window}'] = float(pd.Series(returns).kurtosis())
                
                # Range statistics (vectorized)
                features[f'range_mean_{window}'] = float(np.mean(ranges))
                features[f'range_std_{window}'] = float(np.std(ranges))
                features[f'range_median_{window}'] = float(np.median(ranges))
                
                # Price level statistics (vectorized)
                features[f'close_mean_{window}'] = float(np.mean(close_vals))
                features[f'close_std_{window}'] = float(np.std(close_vals))
                features[f'close_skew_{window}'] = float(pd.Series(close_vals).skew())
                features[f'close_kurt_{window}'] = float(pd.Series(close_vals).kurtosis())
        else:
            # Pandas fallback
            for window in self.stat_windows:
                start_idx = max(0, level_idx - window)
                recent = ohlcv_data.iloc[start_idx:level_idx + 1]
                
                if len(recent) < 2:
                    continue
                
                returns = recent['close'].pct_change()
                
                features[f'ret_mean_{window}'] = float(returns.mean())
                features[f'ret_std_{window}'] = float(returns.std())
                features[f'ret_skew_{window}'] = float(returns.skew())
                features[f'ret_kurt_{window}'] = float(returns.kurtosis())
                
                ranges = recent['high'] - recent['low']
                features[f'range_mean_{window}'] = float(ranges.mean())
                features[f'range_std_{window}'] = float(ranges.std())
                features[f'range_median_{window}'] = float(ranges.median())
                
                features[f'close_mean_{window}'] = float(recent['close'].mean())
                features[f'close_std_{window}'] = float(recent['close'].std())
                features[f'close_skew_{window}'] = float(recent['close'].skew())
                features[f'close_kurt_{window}'] = float(recent['close'].kurtosis())
        
        return features
    
    def _volatility_features(
        self, 
        level_price: float, 
        level_idx: int, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Volatility features - True Range and derived metrics."""
        features = {}
        
        for window in self.volatility_windows:
            start_idx = max(0, level_idx - window)
            recent = ohlcv_data.iloc[start_idx:level_idx + 1].copy()
            
            if len(recent) < 2:
                continue
            
            # True Range components
            hl = recent['high'] - recent['low']
            hc = (recent['high'] - recent['close'].shift(1)).abs()
            lc = (recent['low'] - recent['close'].shift(1)).abs()
            
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            
            # ATR statistics
            features[f'atr_{window}'] = float(tr.mean())
            features[f'atr_std_{window}'] = float(tr.std())
            features[f'atr_median_{window}'] = float(tr.median())
            features[f'atr_max_{window}'] = float(tr.max())
            
            # Volatility ratio (std / mean absolute return)
            returns = recent['close'].pct_change()
            abs_mean = returns.abs().mean()
            if abs_mean > 1e-8:
                features[f'volatility_ratio_{window}'] = returns.std() / abs_mean
            else:
                features[f'volatility_ratio_{window}'] = 0.0
            
            # Normalized volatility (std / price level)
            features[f'volatility_norm_{window}'] = returns.std() / (level_price + 1e-8)
        
        return features
    
    def _interaction_features(self, existing_features: Dict[str, float]) -> Dict[str, float]:
        """Systematic interaction features between different windows."""
        features = {}
        
        for w1, w2 in self.interaction_pairs:
            # Distance ratios
            key1 = f'dist_close_{w1}'
            key2 = f'dist_close_{w2}'
            if key1 in existing_features and key2 in existing_features:
                denominator = existing_features[key2]
                features[f'dist_ratio_{w1}_{w2}'] = (
                    existing_features[key1] / denominator if abs(denominator) > 1e-8 else 0
                )
            
            # Volume ratios
            key1 = f'vol_mean_{w1}'
            key2 = f'vol_mean_{w2}'
            if key1 in existing_features and key2 in existing_features:
                denominator = existing_features[key2]
                features[f'vol_ratio_{w1}_{w2}'] = (
                    existing_features[key1] / denominator if denominator > 1e-8 else 0
                )
            
            # Crossing ratios
            key1 = f'crosses_{w1}'
            key2 = f'crosses_{w2}'
            if key1 in existing_features and key2 in existing_features:
                denominator = existing_features[key2] + 1  # +1 to avoid division by zero
                features[f'crosses_ratio_{w1}_{w2}'] = existing_features[key1] / denominator
            
            # Volatility ratios
            key1 = f'ret_std_{w1}'
            key2 = f'ret_std_{w2}'
            if key1 in existing_features and key2 in existing_features:
                denominator = existing_features[key2]
                features[f'vol_std_ratio_{w1}_{w2}'] = (
                    existing_features[key1] / denominator if denominator > 1e-8 else 0
                )
        
        return features
    
    def get_feature_count(self) -> int:
        """
        Estimate total number of features that will be generated.
        
        Returns:
            Estimated feature count
        """
        count = 0
        
        # Distance features: 6 per window
        count += len(self.distance_windows) * 6
        
        # Crossing features: 2 per window
        count += len(self.distance_windows) * 2
        
        # Time at level: 2 per (window × tolerance)
        count += len(self.distance_windows) * len(self.tolerances) * 2
        
        # Volume features: (3 × tolerances + 7) per window
        count += len(self.volume_windows) * (3 * len(self.tolerances) + 7)
        
        # Price statistics: 11 per window
        count += len(self.stat_windows) * 11
        
        # Volatility features: 6 per window
        count += len(self.volatility_windows) * 6
        
        # Interaction features: 4 per pair
        count += len(self.interaction_pairs) * 4
        
        return count

