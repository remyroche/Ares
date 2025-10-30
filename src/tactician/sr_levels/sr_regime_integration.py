"""
SR Regime Integration Module

PHASE 2: Integrates volatility and trend regime detection into SR pipeline.
Uses existing feature_generation modules (trend.py, volatility.py) to provide
context-aware SR level evaluation.

This module provides:
1. Volatility regime detection (high/medium/low)
2. Trend regime detection (strong_up/weak_up/ranging/weak_down/strong_down)
3. Regime-specific SR level adjustments
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
from enum import Enum

# Import existing feature generators
try:
    from src.feature_generation.categories.trend import TrendFeatureGenerator, ADXGenerator
    from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
    from src.feature_generation.core.feature_generator import FeatureConfig, FeatureCategory
    FEATURE_GEN_AVAILABLE = True
except ImportError:
    FEATURE_GEN_AVAILABLE = False
    TrendFeatureGenerator = None
    ADXGenerator = None
    VolatilityFeatureGenerator = None

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class TrendRegime(Enum):
    """Trend regime classification."""
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    RANGING = "ranging"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"
    UNKNOWN = "unknown"


class SRRegimeDetector:
    """Detects market regimes for SR level evaluation.
    
    PHASE 2 INTEGRATION: Uses existing feature_generation modules.
    """
    
    def __init__(self, lookback_period: int = 20):
        """Initialize regime detector.
        
        Args:
            lookback_period: Window size for regime calculations
        """
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize feature generators if available
        if FEATURE_GEN_AVAILABLE:
            try:
                self.volatility_generator = VolatilityFeatureGenerator(period=lookback_period)
                self.trend_generator = TrendFeatureGenerator()
                self.adx_generator = ADXGenerator(period=14)
                self.logger.info("✅ Feature generators initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize feature generators: {e}")
                self.volatility_generator = None
                self.trend_generator = None
                self.adx_generator = None
        else:
            self.volatility_generator = None
            self.trend_generator = None
            self.adx_generator = None
    
    def detect_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect all market regimes.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Dictionary with regime information:
            {
                'volatility_regime': VolatilityRegime,
                'trend_regime': TrendRegime,
                'volatility_score': float,  # 0-1
                'trend_strength': float,    # 0-1
                'trend_direction': float,   # -1 to 1
                'atr': float,
                'adaptive_window': int
            }
        """
        try:
            results = {}
            
            # Detect volatility regime
            vol_regime, vol_score, atr = self._detect_volatility_regime(data)
            results['volatility_regime'] = vol_regime
            results['volatility_score'] = vol_score
            results['atr'] = atr
            
            # Detect trend regime
            trend_regime, trend_strength, trend_direction = self._detect_trend_regime(data)
            results['trend_regime'] = trend_regime
            results['trend_strength'] = trend_strength
            results['trend_direction'] = trend_direction
            
            # Calculate adaptive window for prominence calculations
            results['adaptive_window'] = self._calculate_adaptive_window(vol_regime, vol_score)
            
            # Log results
            self.logger.info(f"📊 Regime Detection:")
            self.logger.info(f"   Volatility: {vol_regime.value} (score={vol_score:.3f})")
            self.logger.info(f"   Trend: {trend_regime.value} (strength={trend_strength:.3f}, dir={trend_direction:.3f})")
            self.logger.info(f"   Adaptive Window: {results['adaptive_window']} bars")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            return self._get_default_regimes()
    
    def _detect_volatility_regime(self, data: pd.DataFrame) -> Tuple[VolatilityRegime, float, float]:
        """Detect volatility regime using ATR and returns volatility.
        
        Returns:
            (regime, volatility_score, atr_value)
        """
        try:
            # Calculate ATR
            atr = self._calculate_atr(data)
            current_atr = atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else atr.mean()
            
            # Calculate returns volatility
            returns = data['close'].pct_change()
            realized_vol = returns.rolling(self.lookback_period).std().iloc[-1]
            
            # Normalize ATR by price
            atr_pct = current_atr / data['close'].iloc[-1]
            
            # Combined volatility score (0-1 range)
            vol_score = min((atr_pct * 100 + realized_vol) / 2, 1.0)
            
            # Classify regime based on quantiles
            all_volatility = returns.rolling(self.lookback_period).std()
            low_threshold = all_volatility.quantile(0.33)
            high_threshold = all_volatility.quantile(0.67)
            
            if realized_vol < low_threshold:
                regime = VolatilityRegime.LOW
            elif realized_vol > high_threshold:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.MEDIUM
            
            return regime, vol_score, current_atr
            
        except Exception as e:
            self.logger.warning(f"Volatility regime detection failed: {e}")
            return VolatilityRegime.UNKNOWN, 0.5, 0.0
    
    def _detect_trend_regime(self, data: pd.DataFrame) -> Tuple[TrendRegime, float, float]:
        """Detect trend regime using ADX and moving averages.
        
        Returns:
            (regime, trend_strength, trend_direction)
        """
        try:
            # Calculate ADX for trend strength
            adx = self._calculate_adx(data)
            trend_strength = min(adx / 50.0, 1.0)  # Normalize: ADX 50+ = very strong
            
            # Calculate trend direction using multiple timeframes
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            current_price = data['close'].iloc[-1]
            
            # Direction score: -1 (strong down) to +1 (strong up)
            direction_components = []
            
            # 1. Price vs SMA20
            if current_price > sma_20.iloc[-1]:
                direction_components.append(1)
            elif current_price < sma_20.iloc[-1]:
                direction_components.append(-1)
            else:
                direction_components.append(0)
            
            # 2. Price vs SMA50
            if len(sma_50.dropna()) > 0:
                if current_price > sma_50.iloc[-1]:
                    direction_components.append(1)
                elif current_price < sma_50.iloc[-1]:
                    direction_components.append(-1)
                else:
                    direction_components.append(0)
            
            # 3. SMA20 vs SMA50
            if len(sma_50.dropna()) > 0:
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    direction_components.append(1)
                elif sma_20.iloc[-1] < sma_50.iloc[-1]:
                    direction_components.append(-1)
                else:
                    direction_components.append(0)
            
            # Combined direction
            trend_direction = np.mean(direction_components)
            
            # Classify trend regime
            if trend_strength < 0.25:
                # Weak trend = ranging
                regime = TrendRegime.RANGING
            elif trend_direction > 0.5 and trend_strength > 0.5:
                regime = TrendRegime.STRONG_UP
            elif trend_direction > 0:
                regime = TrendRegime.WEAK_UP
            elif trend_direction < -0.5 and trend_strength > 0.5:
                regime = TrendRegime.STRONG_DOWN
            elif trend_direction < 0:
                regime = TrendRegime.WEAK_DOWN
            else:
                regime = TrendRegime.RANGING
            
            return regime, trend_strength, trend_direction
            
        except Exception as e:
            self.logger.warning(f"Trend regime detection failed: {e}")
            return TrendRegime.UNKNOWN, 0.0, 0.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.warning(f"ATR calculation failed: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate +DM and -DM
            high_diff = high.diff()
            low_diff = -low.diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth the directional movements and true range
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return float(adx.iloc[-1]) if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            self.logger.warning(f"ADX calculation failed: {e}")
            return 0.0
    
    def _calculate_adaptive_window(self, vol_regime: VolatilityRegime, vol_score: float) -> int:
        """Calculate adaptive window length for prominence calculations.
        
        High volatility → wider window (more smoothing)
        Low volatility → narrower window (more detail)
        """
        base_window = 20
        
        if vol_regime == VolatilityRegime.HIGH:
            # High volatility: use wider window
            window = int(base_window * (1 + vol_score))
            return min(window, 50)  # Cap at 50
        elif vol_regime == VolatilityRegime.LOW:
            # Low volatility: use narrower window
            window = int(base_window * (1 - vol_score * 0.5))
            return max(window, 10)  # Floor at 10
        else:
            # Medium volatility: use base window
            return base_window
    
    def _get_default_regimes(self) -> Dict[str, Any]:
        """Return default regime information when detection fails."""
        return {
            'volatility_regime': VolatilityRegime.UNKNOWN,
            'trend_regime': TrendRegime.UNKNOWN,
            'volatility_score': 0.5,
            'trend_strength': 0.0,
            'trend_direction': 0.0,
            'atr': 0.0,
            'adaptive_window': 20
        }
    
    def adjust_sr_weights_for_regime(self, regime_info: Dict[str, Any]) -> Dict[str, float]:
        """Adjust SR composite score weights based on regime.
        
        Different regimes require different SR evaluation strategies:
        - High volatility: Emphasize consistency over strength
        - Trending: Support levels more important in uptrend, resistance in downtrend
        - Ranging: Volume and width more important
        
        Args:
            regime_info: Output from detect_regimes()
            
        Returns:
            Dictionary of adjusted weights for composite score
        """
        vol_regime = regime_info['volatility_regime']
        trend_regime = regime_info['trend_regime']
        
        # Default weights
        weights = {
            'strength': 0.30,
            'prominence': 0.25,
            'width': 0.15,
            'volume': 0.15,
            'consistency': 0.10,
            'recency': 0.05
        }
        
        # Adjust for volatility regime
        if vol_regime == VolatilityRegime.HIGH:
            # High volatility: value consistency and volume more
            weights['consistency'] += 0.05
            weights['volume'] += 0.05
            weights['strength'] -= 0.05
            weights['recency'] -= 0.05
        elif vol_regime == VolatilityRegime.LOW:
            # Low volatility: value prominence and width more (zones matter)
            weights['prominence'] += 0.05
            weights['width'] += 0.05
            weights['consistency'] -= 0.05
            weights['volume'] -= 0.05
        
        # Adjust for trend regime
        if trend_regime in [TrendRegime.STRONG_UP, TrendRegime.WEAK_UP]:
            # Uptrend: support levels more important, recent activity matters
            weights['recency'] += 0.03
            weights['consistency'] -= 0.03
        elif trend_regime in [TrendRegime.STRONG_DOWN, TrendRegime.WEAK_DOWN]:
            # Downtrend: resistance levels more important, recent activity matters
            weights['recency'] += 0.03
            weights['consistency'] -= 0.03
        elif trend_regime == TrendRegime.RANGING:
            # Ranging: width and volume crucial (consolidation zones)
            weights['width'] += 0.05
            weights['volume'] += 0.05
            weights['strength'] -= 0.05
            weights['recency'] -= 0.05
        
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        self.logger.debug(f"Adjusted weights for {vol_regime.value}/{trend_regime.value}: {weights}")
        
        return weights


# Convenience function for easy integration
def create_sr_regime_detector(lookback_period: int = 20) -> SRRegimeDetector:
    """Factory function to create SR regime detector."""
    return SRRegimeDetector(lookback_period=lookback_period)

