"""
Multi-Timeframe SR Level Detector

Real cross-timeframe SR level detection and confirmation.
NOT SIMULATED - detects levels on each TF and finds alignment.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .multi_tf_data_loader import MultiTimeframeDataLoader, get_multi_tf_data_loader

logger = logging.getLogger(__name__)


@dataclass
class MultiTFConfirmation:
    """Single timeframe confirmation for a level."""
    timeframe: str
    price: float
    strength: float
    touches: int
    prominence: float
    age_bars: int


@dataclass
class MultiTFLevel:
    """SR level with multi-timeframe confirmation data."""
    base_price: float
    base_level: any  # SRLevel object from base TF
    confirmation_count: int
    confirmations: List[MultiTFConfirmation] = field(default_factory=list)
    multi_tf_score: float = 0.0
    avg_confirmation_strength: float = 0.0
    weighted_confirmation_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for analysis."""
        return {
            'base_price': self.base_price,
            'confirmation_count': self.confirmation_count,
            'multi_tf_score': self.multi_tf_score,
            'avg_confirmation_strength': self.avg_confirmation_strength,
            'weighted_confirmation_score': self.weighted_confirmation_score,
            'confirmations': [
                {
                    'tf': c.timeframe,
                    'price': c.price,
                    'strength': c.strength,
                    'touches': c.touches
                }
                for c in self.confirmations
            ]
        }


class MultiTimeframeSRDetector:
    """Detects and confirms SR levels across multiple timeframes.
    
    REAL MULTI-TF IMPLEMENTATION:
    1. Loads data from multiple timeframes
    2. Detects SR levels on each TF independently
    3. Finds levels that align across TFs (within tolerance)
    4. Scores based on confirmation strength
    """
    
    def __init__(self, data_loader: Optional[MultiTimeframeDataLoader] = None,
                 alignment_tolerance: float = 0.005):
        """Initialize multi-TF SR detector.
        
        Args:
            data_loader: Data loader instance (creates one if None)
            alignment_tolerance: Price alignment tolerance as fraction (0.005 = 0.5%)
        """
        self.data_loader = data_loader or get_multi_tf_data_loader()
        self.alignment_tolerance = alignment_tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Will be set when SR detector is needed
        self.sr_detector = None
    
    def set_sr_detector(self, detector):
        """Set the SR detector to use for level detection."""
        self.sr_detector = detector
    
    def detect_multi_tf_levels(self, symbol: str, exchange: str, 
                              base_timeframe: str,
                              base_data: pd.DataFrame) -> Dict[str, MultiTFLevel]:
        """Detect SR levels with multi-timeframe confirmation.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            base_timeframe: Base timeframe for detection
            base_data: Data for base timeframe (already loaded)
            
        Returns:
            Dict mapping base level ID to MultiTFLevel with confirmations
        """
        try:
            self.logger.info(f"🌍 Starting real multi-TF SR detection for {symbol} {base_timeframe}")
            
            # Load data from all higher timeframes
            tf_data = self.data_loader.load_multiple_timeframes(
                symbol, exchange, base_timeframe, lookback_days=90
            )
            
            # Add base timeframe data if not already loaded
            if base_timeframe not in tf_data:
                tf_data[base_timeframe] = base_data
            
            # Detect SR levels on each timeframe
            tf_levels = self._detect_levels_on_all_timeframes(tf_data)
            
            if base_timeframe not in tf_levels or len(tf_levels[base_timeframe]) == 0:
                self.logger.warning(f"No base levels found on {base_timeframe}")
                return {}
            
            # Find cross-TF confirmations
            multi_tf_levels = self._find_cross_tf_confirmations(
                tf_levels, base_timeframe
            )
            
            self.logger.info(f"✅ Found {len(multi_tf_levels)} base levels with multi-TF data")
            
            # Log confirmation statistics
            self._log_confirmation_stats(multi_tf_levels)
            
            return multi_tf_levels
            
        except Exception as e:
            self.logger.error(f"Multi-TF detection failed: {e}", exc_info=True)
            return {}
    
    def _detect_levels_on_all_timeframes(self, tf_data: Dict[str, pd.DataFrame]) -> Dict[str, List]:
        """Detect SR levels on each timeframe independently.
        
        Args:
            tf_data: Dict of {timeframe: data_df}
            
        Returns:
            Dict of {timeframe: list_of_levels}
        """
        tf_levels = {}
        
        for tf, data in tf_data.items():
            try:
                if data.empty:
                    self.logger.warning(f"Empty data for {tf}, skipping")
                    continue
                
                self.logger.info(f"  Detecting levels on {tf} ({len(data)} bars)...")
                
                # Detect levels using SR detector
                levels = self._detect_sr_levels_simple(data, tf)
                
                tf_levels[tf] = levels
                self.logger.info(f"  ✓ Found {len(levels)} levels on {tf}")
                
            except Exception as e:
                self.logger.error(f"  ✗ Failed to detect levels on {tf}: {e}")
                tf_levels[tf] = []
        
        return tf_levels
    
    def _detect_sr_levels_simple(self, data: pd.DataFrame, timeframe: str) -> List:
        """Detect SR levels on a single timeframe.
        
        Uses the full SR detector if available, otherwise uses simple detection.
        """
        if self.sr_detector is not None:
            try:
                # Use full SR detector
                result = self.sr_detector.detect_sr_levels(data)
                if isinstance(result, dict) and 'levels' in result:
                    return result['levels']
                return result if isinstance(result, list) else []
            except Exception as e:
                self.logger.warning(f"Full SR detection failed on {timeframe}: {e}")
        
        # Fallback: simple swing high/low detection
        return self._simple_swing_detection(data)
    
    def _simple_swing_detection(self, data: pd.DataFrame, window: int = 20) -> List:
        """Simple swing high/low detection as fallback.
        
        Returns list of price levels with basic attributes.
        """
        levels = []
        
        try:
            # Find swing highs
            for i in range(window, len(data) - window):
                is_high = all(data['high'].iloc[i] >= data['high'].iloc[i-window:i].max()) and \
                         all(data['high'].iloc[i] >= data['high'].iloc[i+1:i+window+1].max())
                
                if is_high:
                    levels.append({
                        'price': data['high'].iloc[i],
                        'type': 'resistance',
                        'strength': 0.5,  # Default
                        'touches': 1,
                        'prominence': 0.5,
                        'age_bars': len(data) - i
                    })
            
            # Find swing lows
            for i in range(window, len(data) - window):
                is_low = all(data['low'].iloc[i] <= data['low'].iloc[i-window:i].min()) and \
                        all(data['low'].iloc[i] <= data['low'].iloc[i+1:i+window+1].min())
                
                if is_low:
                    levels.append({
                        'price': data['low'].iloc[i],
                        'type': 'support',
                        'strength': 0.5,
                        'touches': 1,
                        'prominence': 0.5,
                        'age_bars': len(data) - i
                    })
            
        except Exception as e:
            self.logger.warning(f"Simple swing detection failed: {e}")
        
        return levels
    
    def _find_cross_tf_confirmations(self, tf_levels: Dict[str, List],
                                    base_timeframe: str) -> Dict[str, MultiTFLevel]:
        """Find levels that align across multiple timeframes.
        
        Args:
            tf_levels: Dict of {timeframe: levels}
            base_timeframe: Base TF to use as reference
            
        Returns:
            Dict of {level_id: MultiTFLevel}
        """
        base_levels = tf_levels.get(base_timeframe, [])
        
        if not base_levels:
            return {}
        
        multi_tf_levels = {}
        
        for idx, base_level in enumerate(base_levels):
            level_id = f"{base_timeframe}_{idx}"
            
            # Get base price
            base_price = self._get_level_price(base_level)
            
            # Find confirmations on higher timeframes
            confirmations = []
            
            for tf, levels in tf_levels.items():
                if tf == base_timeframe:
                    continue  # Skip base TF
                
                # Find aligned level on this TF
                confirmation = self._find_aligned_level(base_price, levels, tf)
                if confirmation:
                    confirmations.append(confirmation)
            
            # Calculate multi-TF scores
            multi_tf_score, avg_strength, weighted_score = self._calculate_multi_tf_scores(
                confirmations
            )
            
            # Create MultiTFLevel object
            mtf_level = MultiTFLevel(
                base_price=base_price,
                base_level=base_level,
                confirmation_count=len(confirmations),
                confirmations=confirmations,
                multi_tf_score=multi_tf_score,
                avg_confirmation_strength=avg_strength,
                weighted_confirmation_score=weighted_score
            )
            
            multi_tf_levels[level_id] = mtf_level
        
        return multi_tf_levels
    
    def _get_level_price(self, level) -> float:
        """Extract price from level object (handles different formats)."""
        if isinstance(level, dict):
            return float(level.get('price', 0))
        elif hasattr(level, 'price'):
            return float(level.price)
        else:
            return 0.0
    
    def _find_aligned_level(self, base_price: float, levels: List,
                           timeframe: str) -> Optional[MultiTFConfirmation]:
        """Find level on this TF that aligns with base price.
        
        Args:
            base_price: Price from base TF level
            levels: Levels detected on this TF
            timeframe: Current timeframe
            
        Returns:
            MultiTFConfirmation if aligned level found, None otherwise
        """
        for level in levels:
            level_price = self._get_level_price(level)
            
            # Check alignment
            price_diff_pct = abs(level_price - base_price) / base_price
            
            if price_diff_pct <= self.alignment_tolerance:
                # Found aligned level!
                return MultiTFConfirmation(
                    timeframe=timeframe,
                    price=level_price,
                    strength=self._get_level_attr(level, 'strength', 0.5),
                    touches=self._get_level_attr(level, 'touches', 1),
                    prominence=self._get_level_attr(level, 'prominence', 0.5),
                    age_bars=self._get_level_attr(level, 'age_bars', 0)
                )
        
        return None
    
    def _get_level_attr(self, level, attr: str, default):
        """Get attribute from level (handles dict and object)."""
        if isinstance(level, dict):
            return level.get(attr, default)
        else:
            return getattr(level, attr, default)
    
    def _calculate_multi_tf_scores(self, confirmations: List[MultiTFConfirmation]) -> Tuple[float, float, float]:
        """Calculate multi-TF quality scores.
        
        Returns:
            (multi_tf_score, avg_confirmation_strength, weighted_confirmation_score)
        """
        if not confirmations:
            return 0.0, 0.0, 0.0
        
        # 1. Base score from confirmation count (0-1)
        # More confirmations = better (diminishing returns)
        count_score = 1.0 - np.exp(-len(confirmations) / 2.0)  # Asymptotic to 1.0
        
        # 2. Average confirmation strength
        avg_strength = np.mean([c.strength for c in confirmations])
        
        # 3. Weighted score considering strength + touches
        weighted_scores = []
        for c in confirmations:
            # Weight by strength and touch count
            touch_factor = min(c.touches / 3.0, 1.0)  # Normalize touches
            conf_score = c.strength * 0.7 + touch_factor * 0.3
            weighted_scores.append(conf_score)
        
        weighted_score = np.mean(weighted_scores)
        
        # 4. Final multi-TF score combining all factors
        multi_tf_score = (
            count_score * 0.5 +        # Confirmation count (50%)
            avg_strength * 0.3 +       # Average strength (30%)
            weighted_score * 0.2       # Weighted quality (20%)
        )
        
        return multi_tf_score, avg_strength, weighted_score
    
    def _log_confirmation_stats(self, multi_tf_levels: Dict[str, MultiTFLevel]):
        """Log statistics about multi-TF confirmations."""
        if not multi_tf_levels:
            return
        
        confirmation_counts = [mtf.confirmation_count for mtf in multi_tf_levels.values()]
        scores = [mtf.multi_tf_score for mtf in multi_tf_levels.values()]
        
        self.logger.info(f"📊 Multi-TF Confirmation Statistics:")
        self.logger.info(f"   Total base levels: {len(multi_tf_levels)}")
        self.logger.info(f"   Levels with 0 confirmations: {sum(1 for c in confirmation_counts if c == 0)}")
        self.logger.info(f"   Levels with 1+ confirmations: {sum(1 for c in confirmation_counts if c >= 1)}")
        self.logger.info(f"   Levels with 2+ confirmations: {sum(1 for c in confirmation_counts if c >= 2)}")
        self.logger.info(f"   Avg confirmations: {np.mean(confirmation_counts):.2f}")
        self.logger.info(f"   Avg multi-TF score: {np.mean(scores):.3f}")
        self.logger.info(f"   Score range: [{min(scores):.3f}, {max(scores):.3f}]")


def create_multi_tf_detector(alignment_tolerance: float = 0.005) -> MultiTimeframeSRDetector:
    """Factory function to create multi-TF SR detector."""
    return MultiTimeframeSRDetector(alignment_tolerance=alignment_tolerance)

