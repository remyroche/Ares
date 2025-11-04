"""
Outcome Target Generator - 100% Data-Driven

Generate ALL possible outcome targets and let AutoML select the best.
No predetermined 'best' horizon or predefined target weights.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class OutcomeTargetGenerator:
    """
    Generate exhaustive outcome targets from future price behavior.
    
    Philosophy: Generate ALL possible targets across ALL horizons.
    Let validation performance determine which target to predict.
    
    Expected output: 50-100 possible targets per level.
    """
    
    def __init__(self, fast_mode: bool = True):
        """
        Initialize target generator.
        
        Args:
            fast_mode: If True, uses reduced target space for speed
                      If False, uses full exhaustive target space
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if fast_mode:
            # FAST MODE: Reduced but still data-driven
            # Focus on most relevant time horizons for SR levels
            self.forward_windows = [10, 20, 50]  # 3 windows instead of 5
            self.breakout_thresholds = [0.02]  # 1 threshold instead of 3
            self.logger.info("⚡ Fast mode: 3 windows, 1 threshold (~40 targets/level)")
        else:
            # FULL MODE: Exhaustive (slow but comprehensive)
            self.forward_windows = [5, 10, 20, 50, 100]
            self.breakout_thresholds = [0.01, 0.02, 0.03]
            self.logger.info("🐌 Full mode: 5 windows, 3 thresholds (~135 targets/level)")
    
    def generate_all_targets(
        self,
        level_price: float,
        level_idx: int,
        future_data: pd.DataFrame,
        creation_timestamp: pd.Timestamp = None
    ) -> Dict[str, float]:
        """
        Generate ALL possible outcome targets from future price behavior.
        
        TIMESTAMP CONTRACT:
        - Only uses data AFTER creation_timestamp (no past information leak)
        - If creation_timestamp provided, validates all data used is > creation_timestamp
        - This prevents reverse leakage by ensuring targets only look forward in time
        
        Args:
            level_price: Price of the level
            level_idx: Index of level in full data
            future_data: Full DataFrame (level at level_idx, future is after)
            creation_timestamp: Timestamp when level was created (optional but recommended)
        
        Returns:
            Dictionary with 50-100 outcome targets
        """
        # TIMESTAMP CONTRACT VALIDATION
        if creation_timestamp is not None:
            if level_idx >= len(future_data):
                raise ValueError(f"level_idx {level_idx} exceeds data length {len(future_data)}")
            
            level_timestamp = future_data.index[level_idx]
            if level_timestamp > creation_timestamp:
                self.logger.warning(
                    f"TIMESTAMP CONTRACT WARNING: level_timestamp {level_timestamp} "
                    f"is after creation_timestamp {creation_timestamp}"
                )
        targets = {}
        
        for window in self.forward_windows:
            # Get future bars
            future_start = level_idx + 1
            future_end = level_idx + 1 + window
            
            if future_end > len(future_data):
                # Not enough future data for this window
                continue
            
            future = future_data.iloc[future_start:future_end]
            
            if len(future) < window:
                continue
            
            # 1. Price reaction magnitudes
            targets.update(self._price_reaction_targets(
                level_price, window, future
            ))
            
            # 2. Touch behavior targets
            targets.update(self._touch_behavior_targets(
                level_price, window, future
            ))
            
            # 3. Reversal metrics
            targets.update(self._reversal_targets(
                level_price, window, future, level_idx, future_data
            ))
            
            # 4. Breakout behavior (multiple thresholds)
            targets.update(self._breakout_targets(
                level_price, window, future
            ))
            
            # 5. Volatility changes
            targets.update(self._volatility_change_targets(
                level_price, window, future, level_idx, future_data
            ))
            
            # 6. Volume changes
            targets.update(self._volume_change_targets(
                level_price, window, future, level_idx, future_data
            ))
        
        # Replace NaN/inf with 0
        for key in targets:
            if not np.isfinite(targets[key]):
                targets[key] = 0.0
        
        self.logger.debug(f"Generated {len(targets)} outcome targets for level at ${level_price:.2f}")
        
        return targets
    
    def _price_reaction_targets(
        self,
        level_price: float,
        window: int,
        future: pd.DataFrame
    ) -> Dict[str, float]:
        """Price reaction magnitude targets."""
        targets = {}
        
        # Maximum upward move
        max_high = future['high'].max()
        targets[f'max_up_{window}'] = (max_high - level_price) / level_price
        
        # Maximum downward move
        min_low = future['low'].min()
        targets[f'max_down_{window}'] = (level_price - min_low) / level_price
        
        # Net move (close to close)
        final_close = future['close'].iloc[-1]
        targets[f'net_move_{window}'] = (final_close - level_price) / level_price
        
        # Absolute maximum move (either direction)
        targets[f'max_abs_move_{window}'] = max(
            abs(targets[f'max_up_{window}']),
            abs(targets[f'max_down_{window}'])
        )
        
        # Average distance from level
        avg_dist = ((future['close'] - level_price).abs() / level_price).mean()
        targets[f'avg_dist_{window}'] = float(avg_dist)
        
        return targets
    
    def _touch_behavior_targets(
        self,
        level_price: float,
        window: int,
        future: pd.DataFrame
    ) -> Dict[str, float]:
        """Touch behavior targets."""
        targets = {}
        
        # Define 'touch' as bar range includes level (within 0.1%)
        touched = (
            (future['high'] >= level_price * 0.999) &
            (future['low'] <= level_price * 1.001)
        )
        
        # Touch count
        touch_count = touched.sum()
        targets[f'touch_count_{window}'] = float(touch_count)
        
        # Binary: was level touched?
        targets[f'touch_binary_{window}'] = float(touched.any())
        
        # Touch rate (touches per bar)
        targets[f'touch_rate_{window}'] = touch_count / len(future) if len(future) > 0 else 0
        
        # Time to first touch (bars, 0 if never touched)
        if touched.any():
            first_touch_idx = touched.idxmax()
            bars_to_touch = future.index.get_loc(first_touch_idx)
            targets[f'bars_to_touch_{window}'] = float(bars_to_touch)
        else:
            targets[f'bars_to_touch_{window}'] = float(window)  # Max if never touched
        
        return targets
    
    def _reversal_targets(
        self,
        level_price: float,
        window: int,
        future: pd.DataFrame,
        level_idx: int,
        full_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Reversal strength targets."""
        targets = {}
        
        # Check if level was touched
        touched = (
            (future['high'] >= level_price * 0.999) &
            (future['low'] <= level_price * 1.001)
        )
        
        if not touched.any():
            # No touch = no reversal data
            targets[f'reversal_mag_{window}'] = 0.0
            targets[f'reversal_dir_{window}'] = 0.0
            targets[f'reversal_strength_{window}'] = 0.0
            return targets
        
        # Get first touch
        first_touch_idx_in_future = touched.idxmax()
        first_touch_loc = future.index.get_loc(first_touch_idx_in_future)
        
        # Price movement after touch
        after_touch = future.iloc[first_touch_loc:]
        
        if len(after_touch) > 1:
            # Magnitude of reversal
            reversal_mag = abs(after_touch['close'].iloc[-1] - after_touch['close'].iloc[0]) / level_price
            targets[f'reversal_mag_{window}'] = float(reversal_mag)
            
            # Direction of reversal
            reversal_dir = np.sign(after_touch['close'].iloc[-1] - after_touch['close'].iloc[0])
            targets[f'reversal_dir_{window}'] = float(reversal_dir)
            
            # Reversal strength (how far it moved from level)
            max_reversal = max(
                abs(after_touch['high'].max() - level_price),
                abs(after_touch['low'].min() - level_price)
            ) / level_price
            targets[f'reversal_strength_{window}'] = float(max_reversal)
        else:
            targets[f'reversal_mag_{window}'] = 0.0
            targets[f'reversal_dir_{window}'] = 0.0
            targets[f'reversal_strength_{window}'] = 0.0
        
        return targets
    
    def _breakout_targets(
        self,
        level_price: float,
        window: int,
        future: pd.DataFrame
    ) -> Dict[str, float]:
        """Breakout behavior targets with multiple thresholds."""
        targets = {}
        
        for threshold in self.breakout_thresholds:
            threshold_pct = int(threshold * 100)
            
            # Check for breakout up
            broke_up = (future['close'] > level_price * (1 + threshold)).any()
            
            # Check for breakout down
            broke_down = (future['close'] < level_price * (1 - threshold)).any()
            
            # Binary: any breakout
            targets[f'break_binary_{window}_{threshold_pct}pct'] = float(broke_up or broke_down)
            
            # Direction: +1 up, -1 down, 0 neither
            if broke_up and not broke_down:
                direction = 1.0
            elif broke_down and not broke_up:
                direction = -1.0
            elif broke_up and broke_down:
                # Both broke - which happened first?
                up_idx = (future['close'] > level_price * (1 + threshold)).idxmax()
                down_idx = (future['close'] < level_price * (1 - threshold)).idxmax()
                direction = 1.0 if future.index.get_loc(up_idx) < future.index.get_loc(down_idx) else -1.0
            else:
                direction = 0.0
            
            targets[f'break_dir_{window}_{threshold_pct}pct'] = direction
            
            # Time to breakout (bars)
            if broke_up or broke_down:
                if broke_up:
                    up_time = future.index.get_loc((future['close'] > level_price * (1 + threshold)).idxmax())
                else:
                    up_time = window
                
                if broke_down:
                    down_time = future.index.get_loc((future['close'] < level_price * (1 - threshold)).idxmax())
                else:
                    down_time = window
                
                targets[f'break_time_{window}_{threshold_pct}pct'] = float(min(up_time, down_time))
            else:
                targets[f'break_time_{window}_{threshold_pct}pct'] = float(window)
        
        return targets
    
    def _volatility_change_targets(
        self,
        level_price: float,
        window: int,
        future: pd.DataFrame,
        level_idx: int,
        full_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Volatility change targets."""
        targets = {}
        
        # Get pre-level volatility
        pre_start = max(0, level_idx - window)
        pre_data = full_data.iloc[pre_start:level_idx]
        
        if len(pre_data) < 2:
            targets[f'vol_change_{window}'] = 0.0
            targets[f'vol_change_abs_{window}'] = 0.0
            return targets
        
        pre_vol = pre_data['close'].pct_change().std()
        post_vol = future['close'].pct_change().std()
        
        # Relative change
        if pre_vol > 1e-8:
            targets[f'vol_change_{window}'] = post_vol / pre_vol
        else:
            targets[f'vol_change_{window}'] = 1.0
        
        # Absolute change
        targets[f'vol_change_abs_{window}'] = post_vol - pre_vol
        
        # Volatility spike (max vs mean)
        targets[f'vol_spike_{window}'] = (
            future['close'].pct_change().abs().max() / 
            (future['close'].pct_change().abs().mean() + 1e-8)
        )
        
        return targets
    
    def _volume_change_targets(
        self,
        level_price: float,
        window: int,
        future: pd.DataFrame,
        level_idx: int,
        full_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Volume change targets."""
        targets = {}
        
        # Get pre-level volume
        pre_start = max(0, level_idx - window)
        pre_data = full_data.iloc[pre_start:level_idx]
        
        if len(pre_data) == 0:
            targets[f'volume_surge_{window}'] = 0.0
            targets[f'volume_surge_abs_{window}'] = 0.0
            return targets
        
        pre_vol_avg = pre_data['volume'].mean()
        post_vol_avg = future['volume'].mean()
        
        # Relative change
        if pre_vol_avg > 1e-8:
            targets[f'volume_surge_{window}'] = post_vol_avg / pre_vol_avg
        else:
            targets[f'volume_surge_{window}'] = 1.0
        
        # Absolute change
        targets[f'volume_surge_abs_{window}'] = post_vol_avg - pre_vol_avg
        
        # Volume spike (max vs mean)
        targets[f'volume_spike_{window}'] = (
            future['volume'].max() / (future['volume'].mean() + 1e-8)
        )
        
        return targets
    
    def get_target_count(self) -> int:
        """
        Estimate total number of targets that will be generated.
        
        Returns:
            Estimated target count
        """
        count = 0
        
        # Price reactions: 5 per window
        count += len(self.forward_windows) * 5
        
        # Touch behavior: 4 per window
        count += len(self.forward_windows) * 4
        
        # Reversal: 3 per window
        count += len(self.forward_windows) * 3
        
        # Breakout: 3 per (window × threshold)
        count += len(self.forward_windows) * len(self.breakout_thresholds) * 3
        
        # Volatility: 3 per window
        count += len(self.forward_windows) * 3
        
        # Volume: 3 per window
        count += len(self.forward_windows) * 3
        
        return count

