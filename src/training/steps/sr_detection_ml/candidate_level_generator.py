"""
Candidate Level Generator - 100% Data-Driven

Generates ALL mathematically-defined local extrema with zero filtering.
No heuristics, no thresholds, no judgment about significance.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class DataDrivenLevelGenerator:
    """
    Pure mathematical local extrema generator.
    
    Philosophy: Generate ALL local min/max points and let ML decide which matter.
    No predefined significance thresholds, no filtering based on volume/touches/age.
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize generator.
        
        Args:
            order: How many points on each side to use for comparison.
                   order=1 means immediate neighbors (most granular).
        """
        self.order = order
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_all_candidates(self, ohlcv_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate ALL local extrema from OHLCV data.
        
        Pure mathematical definition: local min/max with no judgment.
        
        Args:
            ohlcv_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                       Index should be datetime
        
        Returns:
            List of candidate levels, each with:
                - price: float
                - idx: int (index in ohlcv_data)
                - type: 'local_high' or 'local_low'
                - timestamp: datetime
        """
        if len(ohlcv_data) < self.order * 2 + 1:
            self.logger.warning(f"Data too short ({len(ohlcv_data)} bars) for order={self.order}")
            return []
        
        candidates = []
        
        # Find local maxima in highs
        highs_idx = argrelextrema(
            ohlcv_data['high'].values, 
            np.greater, 
            order=self.order
        )[0]
        
        # Find local minima in lows
        lows_idx = argrelextrema(
            ohlcv_data['low'].values, 
            np.less, 
            order=self.order
        )[0]
        
        # Add all local highs
        for idx in highs_idx:
            candidates.append({
                'price': float(ohlcv_data['high'].iloc[idx]),
                'idx': int(idx),
                'type': 'local_high',
                'timestamp': ohlcv_data.index[idx]
            })
        
        # Add all local lows
        for idx in lows_idx:
            candidates.append({
                'price': float(ohlcv_data['low'].iloc[idx]),
                'idx': int(idx),
                'type': 'local_low',
                'timestamp': ohlcv_data.index[idx]
            })
        
        self.logger.info(
            f"Generated {len(candidates)} candidates: "
            f"{len(highs_idx)} local highs, {len(lows_idx)} local lows"
        )
        
        return candidates
    
    def get_statistics(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about generated candidates.
        
        Args:
            candidates: List of candidate levels
        
        Returns:
            Dictionary with statistics
        """
        if not candidates:
            return {
                'total': 0,
                'local_highs': 0,
                'local_lows': 0
            }
        
        local_highs = sum(1 for c in candidates if c['type'] == 'local_high')
        local_lows = sum(1 for c in candidates if c['type'] == 'local_low')
        
        prices = [c['price'] for c in candidates]
        
        return {
            'total': len(candidates),
            'local_highs': local_highs,
            'local_lows': local_lows,
            'price_range': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'mean': np.mean(prices) if prices else 0
            }
        }

