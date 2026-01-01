"""Range-Specific Features for 1.5-3% Trading Range

This module implements features specifically designed for predicting
price movements in the 1.5-3% range, following de Prado's framework
for medium-term trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class RangeSpecificFeatures:
    """Range-specific feature engineering for 1.5-3% trading strategies."""
    
    def __init__(self, min_target: float = 0.015, max_target: float = 0.03):
        self.min_target = min_target
        self.max_target = max_target
        self.logger = logger.getChild("RangeSpecificFeatures")
    
    def compute_atr_scaled_distances(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        """Compute distance to key levels scaled by ATR for 1.5-3% range targets."""
        features = pd.DataFrame(index=df.index)
        
        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(lookback).mean()
        
        current_price = df["close"]
        
        # Distance to levels scaled by ATR
        for window in [20, 50, 100]:
            recent_high = df["high"].rolling(window).max()
            recent_low = df["low"].rolling(window).min()
            
            features[f"distance_to_high_{window}_atr"] = (recent_high - current_price) / atr
            features[f"distance_to_low_{window}_atr"] = (current_price - recent_low) / atr
            
            # Normalized distance for 1.5-3% range
            features[f"range_score_high_{window}"] = np.clip(
                (recent_high - current_price) / current_price / self.max_target, 0, 1
            )
            features[f"range_score_low_{window}"] = np.clip(
                (current_price - recent_low) / current_price / self.max_target, 0, 1
            )
        
        return features.fillna(0)
    
    def compute_range_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute probability of reaching 1.5-3% range based on current conditions."""
        features = pd.DataFrame(index=df.index)
        
        returns = df["close"].pct_change()
        volatility = returns.rolling(20).std()
        
        time_horizons = [12, 24, 48]  # 3h, 6h, 12h for 15m data
        
        for horizon in time_horizons:
            expected_move = volatility * np.sqrt(horizon)
            
            # Probability of reaching minimum target (1.5%)
            prob_min_target = 1 - np.exp(-2 * (self.min_target / expected_move) ** 2)
            features[f"prob_min_target_{horizon}"] = np.clip(prob_min_target, 0, 1)
            
            # Probability of reaching maximum target (3%)
            prob_max_target = 1 - np.exp(-2 * (self.max_target / expected_move) ** 2)
            features[f"prob_max_target_{horizon}"] = np.clip(prob_max_target, 0, 1)
        
        return features.fillna(0)
