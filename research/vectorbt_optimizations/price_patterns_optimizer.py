"""
VectorBT Optimized Price Pattern Discovery

This module enhances the price pattern discovery framework with VectorBT capabilities:
- VectorBT technical indicators for pattern recognition
- Enhanced pattern validation using backtesting
- Signal-based pattern effectiveness analysis
- Portfolio-level pattern performance evaluation
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings

# Suppress VectorBT warnings
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)

class VectorBTPricePatternsOptimizer:
    """
    VectorBT-optimized price pattern discovery framework.
    
    This class enhances the existing price pattern discovery with VectorBT capabilities:
    - VectorBT technical indicators for pattern recognition
    - Enhanced pattern validation using backtesting
    - Signal-based pattern effectiveness analysis
    - Portfolio-level pattern performance evaluation
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize VectorBT price patterns optimizer.
        
        Args:
            data: OHLCV data
        """
        self.data = data.copy()
        
        # Ensure proper index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # VectorBT configuration
        vbt.settings.set_theme("dark")
        
        logger.info("✅ VectorBT price patterns optimizer initialized")
    
    def discover_vectorbt_patterns(self) -> Dict[str, pd.Series]:
        """
        Discover price patterns using VectorBT technical indicators.
        
        Returns:
            Dictionary of discovered patterns
        """
        logger.info("🔍 Discovering VectorBT price patterns...")
        
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        patterns = {}
        
        try:
            # Candlestick patterns
            patterns.update(self._discover_candlestick_patterns())
            
            # Technical indicator patterns
            patterns.update(self._discover_technical_patterns(close, high, low, volume))
            
            # Price action patterns
            patterns.update(self._discover_price_action_patterns(close, high, low))
            
            # Volume patterns
            patterns.update(self._discover_volume_patterns(close, volume))
            
            # Momentum patterns
            patterns.update(self._discover_momentum_patterns(close))
            
            # Trend patterns
            patterns.update(self._discover_trend_patterns(close))
            
            # Support/Resistance patterns
            patterns.update(self._discover_support_resistance_patterns(close, high, low))
            
            logger.info(f"✅ Discovered {len(patterns)} VectorBT patterns")
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
            return {}
        
        return patterns
    
    def _discover_candlestick_patterns(self) -> Dict[str, pd.Series]:
        """Discover candlestick patterns using VectorBT."""
        patterns = {}
        
        try:
            open_price = self.data['open']
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            # Basic candlestick patterns
            patterns['doji'] = vbt.DOJI.run(open_price, high, low, close).doji
            patterns['hammer'] = vbt.HAMMER.run(open_price, high, low, close).hammer
            patterns['shooting_star'] = vbt.SHOOTING_STAR.run(open_price, high, low, close).shooting_star
            
            # Custom pattern definitions
            # Bullish engulfing
            patterns['bullish_engulfing'] = (
                (close.shift(1) < open_price.shift(1)) &  # Previous candle bearish
                (close > open_price) &  # Current candle bullish
                (open_price < close.shift(1)) &  # Current open below previous close
                (close > open_price.shift(1))  # Current close above previous open
            ).astype(int)
            
            # Bearish engulfing
            patterns['bearish_engulfing'] = (
                (close.shift(1) > open_price.shift(1)) &  # Previous candle bullish
                (close < open_price) &  # Current candle bearish
                (open_price > close.shift(1)) &  # Current open above previous close
                (close < open_price.shift(1))  # Current close below previous open
            ).astype(int)
            
            # Morning star
            patterns['morning_star'] = (
                (close.shift(2) < open_price.shift(2)) &  # First candle bearish
                (abs(close.shift(1) - open_price.shift(1)) < (high.shift(1) - low.shift(1)) * 0.1) &  # Second candle small
                (close > open_price) &  # Third candle bullish
                (close > (close.shift(2) + open_price.shift(2)) / 2)  # Third candle closes above midpoint of first
            ).astype(int)
            
            # Evening star
            patterns['evening_star'] = (
                (close.shift(2) > open_price.shift(2)) &  # First candle bullish
                (abs(close.shift(1) - open_price.shift(1)) < (high.shift(1) - low.shift(1)) * 0.1) &  # Second candle small
                (close < open_price) &  # Third candle bearish
                (close < (close.shift(2) + open_price.shift(2)) / 2)  # Third candle closes below midpoint of first
            ).astype(int)
            
        except Exception as e:
            logger.error(f"Error discovering candlestick patterns: {e}")
        
        return patterns
    
    def _discover_technical_patterns(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Discover technical indicator patterns."""
        patterns = {}
        
        try:
            # RSI patterns
            rsi = vbt.RSI.run(close).rsi
            patterns['rsi_oversold_bounce'] = (
                (rsi < 30) & (rsi.shift(1) >= 30) & (close > close.shift(1))
            ).astype(int)
            patterns['rsi_overbought_rejection'] = (
                (rsi > 70) & (rsi.shift(1) <= 70) & (close < close.shift(1))
            ).astype(int)
            patterns['rsi_divergence_bullish'] = self._detect_rsi_divergence(close, rsi, bullish=True)
            patterns['rsi_divergence_bearish'] = self._detect_rsi_divergence(close, rsi, bullish=False)
            
            # MACD patterns
            macd = vbt.MACD.run(close)
            patterns['macd_bullish_crossover'] = (
                (macd.macd > macd.signal) & (macd.macd.shift(1) <= macd.signal.shift(1))
            ).astype(int)
            patterns['macd_bearish_crossover'] = (
                (macd.macd < macd.signal) & (macd.macd.shift(1) >= macd.signal.shift(1))
            ).astype(int)
            patterns['macd_histogram_bullish'] = (
                (macd.histogram > 0) & (macd.histogram.shift(1) <= 0)
            ).astype(int)
            patterns['macd_histogram_bearish'] = (
                (macd.histogram < 0) & (macd.histogram.shift(1) >= 0)
            ).astype(int)
            
            # Bollinger Bands patterns
            bb = vbt.BBANDS.run(close)
            patterns['bb_squeeze'] = (bb.width < bb.width.rolling(20).mean()).astype(int)
            patterns['bb_breakout_upper'] = (close > bb.upper).astype(int)
            patterns['bb_breakout_lower'] = (close < bb.lower).astype(int)
            patterns['bb_reversion_upper'] = (
                (close > bb.upper) & (close.shift(1) <= bb.upper.shift(1)) & (close < close.shift(1))
            ).astype(int)
            patterns['bb_reversion_lower'] = (
                (close < bb.lower) & (close.shift(1) >= bb.lower.shift(1)) & (close > close.shift(1))
            ).astype(int)
            
            # Stochastic patterns
            stoch = vbt.STOCH.run(high, low, close)
            patterns['stoch_oversold_bounce'] = (
                (stoch.k < 20) & (stoch.d < 20) & (stoch.k > stoch.k.shift(1))
            ).astype(int)
            patterns['stoch_overbought_rejection'] = (
                (stoch.k > 80) & (stoch.d > 80) & (stoch.k < stoch.k.shift(1))
            ).astype(int)
            patterns['stoch_bullish_crossover'] = (
                (stoch.k > stoch.d) & (stoch.k.shift(1) <= stoch.d.shift(1))
            ).astype(int)
            patterns['stoch_bearish_crossover'] = (
                (stoch.k < stoch.d) & (stoch.k.shift(1) >= stoch.d.shift(1))
            ).astype(int)
            
        except Exception as e:
            logger.error(f"Error discovering technical patterns: {e}")
        
        return patterns
    
    def _discover_price_action_patterns(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Discover price action patterns."""
        patterns = {}
        
        try:
            # Breakout patterns
            resistance = high.rolling(20).max()
            support = low.rolling(20).min()
            
            patterns['resistance_breakout'] = (
                (close > resistance.shift(1)) & (close.shift(1) <= resistance.shift(1))
            ).astype(int)
            patterns['support_breakdown'] = (
                (close < support.shift(1)) & (close.shift(1) >= support.shift(1))
            ).astype(int)
            
            # Double top/bottom patterns
            patterns['double_top'] = self._detect_double_top(close, high)
            patterns['double_bottom'] = self._detect_double_bottom(close, low)
            
            # Head and shoulders patterns
            patterns['head_and_shoulders'] = self._detect_head_and_shoulders(close, high)
            patterns['inverse_head_and_shoulders'] = self._detect_inverse_head_and_shoulders(close, low)
            
            # Triangle patterns
            patterns['ascending_triangle'] = self._detect_ascending_triangle(close, high, low)
            patterns['descending_triangle'] = self._detect_descending_triangle(close, high, low)
            patterns['symmetrical_triangle'] = self._detect_symmetrical_triangle(close, high, low)
            
            # Flag and pennant patterns
            patterns['bull_flag'] = self._detect_bull_flag(close, high, low)
            patterns['bear_flag'] = self._detect_bear_flag(close, high, low)
            
        except Exception as e:
            logger.error(f"Error discovering price action patterns: {e}")
        
        return patterns
    
    def _discover_volume_patterns(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Discover volume-based patterns."""
        patterns = {}
        
        try:
            # Volume spikes
            volume_sma = volume.rolling(20).mean()
            patterns['volume_spike'] = (volume > volume_sma * 2).astype(int)
            patterns['volume_dry_up'] = (volume < volume_sma * 0.5).astype(int)
            
            # Volume trend patterns
            patterns['volume_increasing'] = (volume > volume.shift(1)).astype(int)
            patterns['volume_decreasing'] = (volume < volume.shift(1)).astype(int)
            
            # OBV patterns
            obv = vbt.OBV.run(close, volume).obv
            obv_sma = obv.rolling(20).mean()
            patterns['obv_bullish'] = (obv > obv_sma).astype(int)
            patterns['obv_bearish'] = (obv < obv_sma).astype(int)
            patterns['obv_divergence_bullish'] = self._detect_obv_divergence(close, obv, bullish=True)
            patterns['obv_divergence_bearish'] = self._detect_obv_divergence(close, obv, bullish=False)
            
            # AD patterns
            ad = vbt.AD.run(close, close, close, volume).ad  # Using close for high/low
            ad_sma = ad.rolling(20).mean()
            patterns['ad_bullish'] = (ad > ad_sma).astype(int)
            patterns['ad_bearish'] = (ad < ad_sma).astype(int)
            
            # CMF patterns
            cmf = vbt.CMF.run(close, close, close, volume).cmf
            patterns['cmf_positive'] = (cmf > 0).astype(int)
            patterns['cmf_negative'] = (cmf < 0).astype(int)
            patterns['cmf_bullish_divergence'] = self._detect_cmf_divergence(close, cmf, bullish=True)
            patterns['cmf_bearish_divergence'] = self._detect_cmf_divergence(close, cmf, bullish=False)
            
        except Exception as e:
            logger.error(f"Error discovering volume patterns: {e}")
        
        return patterns
    
    def _discover_momentum_patterns(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Discover momentum patterns."""
        patterns = {}
        
        try:
            # Price momentum
            for period in [5, 10, 20]:
                momentum = close / close.shift(period) - 1
                patterns[f'momentum_bullish_{period}'] = (momentum > 0.02).astype(int)  # 2% gain
                patterns[f'momentum_bearish_{period}'] = (momentum < -0.02).astype(int)  # 2% loss
                patterns[f'momentum_acceleration_{period}'] = (momentum > momentum.shift(1)).astype(int)
                patterns[f'momentum_deceleration_{period}'] = (momentum < momentum.shift(1)).astype(int)
            
            # Rate of change patterns
            for period in [5, 10, 20]:
                roc = close.pct_change(period)
                patterns[f'roc_bullish_{period}'] = (roc > 0.05).astype(int)  # 5% ROC
                patterns[f'roc_bearish_{period}'] = (roc < -0.05).astype(int)  # -5% ROC
            
            # Momentum divergence
            patterns['momentum_divergence_bullish'] = self._detect_momentum_divergence(close, bullish=True)
            patterns['momentum_divergence_bearish'] = self._detect_momentum_divergence(close, bullish=False)
            
        except Exception as e:
            logger.error(f"Error discovering momentum patterns: {e}")
        
        return patterns
    
    def _discover_trend_patterns(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Discover trend patterns."""
        patterns = {}
        
        try:
            # Moving average patterns
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            
            patterns['golden_cross'] = (
                (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
            ).astype(int)
            patterns['death_cross'] = (
                (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
            ).astype(int)
            
            patterns['bullish_alignment'] = (
                (close > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
            ).astype(int)
            patterns['bearish_alignment'] = (
                (close < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
            ).astype(int)
            
            # Trend strength
            patterns['trend_strength_strong'] = (
                (close > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200) &
                ((sma_20 - sma_50) / sma_50 > 0.02)
            ).astype(int)
            patterns['trend_strength_weak'] = (
                (close < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200) &
                ((sma_50 - sma_20) / sma_20 > 0.02)
            ).astype(int)
            
            # Trend reversal patterns
            patterns['trend_reversal_bullish'] = (
                (close < sma_20) & (close.shift(1) >= sma_20.shift(1)) &
                (close > close.shift(1))
            ).astype(int)
            patterns['trend_reversal_bearish'] = (
                (close > sma_20) & (close.shift(1) <= sma_20.shift(1)) &
                (close < close.shift(1))
            ).astype(int)
            
        except Exception as e:
            logger.error(f"Error discovering trend patterns: {e}")
        
        return patterns
    
    def _discover_support_resistance_patterns(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Discover support/resistance patterns."""
        patterns = {}
        
        try:
            # Dynamic support/resistance
            resistance = high.rolling(20).max()
            support = low.rolling(20).min()
            
            patterns['support_bounce'] = (
                (close <= support * 1.001) & (close > close.shift(1))
            ).astype(int)
            patterns['resistance_rejection'] = (
                (close >= resistance * 0.999) & (close < close.shift(1))
            ).astype(int)
            
            # Multiple touches
            patterns['support_multiple_touches'] = self._detect_multiple_touches(close, support, 'support')
            patterns['resistance_multiple_touches'] = self._detect_multiple_touches(close, resistance, 'resistance')
            
            # Break and retest patterns
            patterns['support_break_retest'] = self._detect_break_retest(close, support, 'support')
            patterns['resistance_break_retest'] = self._detect_break_retest(close, resistance, 'resistance')
            
        except Exception as e:
            logger.error(f"Error discovering support/resistance patterns: {e}")
        
        return patterns
    
    def _detect_rsi_divergence(self, close: pd.Series, rsi: pd.Series, bullish: bool = True) -> pd.Series:
        """Detect RSI divergence patterns."""
        # Simplified divergence detection
        if bullish:
            return (
                (close < close.shift(10)) & (rsi > rsi.shift(10)) &
                (rsi < 50) & (rsi.shift(10) < 50)
            ).astype(int)
        else:
            return (
                (close > close.shift(10)) & (rsi < rsi.shift(10)) &
                (rsi > 50) & (rsi.shift(10) > 50)
            ).astype(int)
    
    def _detect_double_top(self, close: pd.Series, high: pd.Series) -> pd.Series:
        """Detect double top pattern."""
        # Simplified double top detection
        peaks = high.rolling(5, center=True).max() == high
        return (peaks & peaks.shift(10)).astype(int)
    
    def _detect_double_bottom(self, close: pd.Series, low: pd.Series) -> pd.Series:
        """Detect double bottom pattern."""
        # Simplified double bottom detection
        troughs = low.rolling(5, center=True).min() == low
        return (troughs & troughs.shift(10)).astype(int)
    
    def _detect_head_and_shoulders(self, close: pd.Series, high: pd.Series) -> pd.Series:
        """Detect head and shoulders pattern."""
        # Simplified H&S detection
        peaks = high.rolling(5, center=True).max() == high
        return (peaks & peaks.shift(5) & peaks.shift(10)).astype(int)
    
    def _detect_inverse_head_and_shoulders(self, close: pd.Series, low: pd.Series) -> pd.Series:
        """Detect inverse head and shoulders pattern."""
        # Simplified inverse H&S detection
        troughs = low.rolling(5, center=True).min() == low
        return (troughs & troughs.shift(5) & troughs.shift(10)).astype(int)
    
    def _detect_ascending_triangle(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect ascending triangle pattern."""
        # Simplified ascending triangle detection
        resistance = high.rolling(20).max()
        support = low.rolling(20).min()
        return (
            (close < resistance) & (close > support) &
            (resistance == resistance.rolling(10).max()) &
            (support > support.shift(10))
        ).astype(int)
    
    def _detect_descending_triangle(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect descending triangle pattern."""
        # Simplified descending triangle detection
        resistance = high.rolling(20).max()
        support = low.rolling(20).min()
        return (
            (close < resistance) & (close > support) &
            (resistance < resistance.shift(10)) &
            (support == support.rolling(10).min())
        ).astype(int)
    
    def _detect_symmetrical_triangle(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect symmetrical triangle pattern."""
        # Simplified symmetrical triangle detection
        resistance = high.rolling(20).max()
        support = low.rolling(20).min()
        return (
            (close < resistance) & (close > support) &
            (resistance < resistance.shift(10)) &
            (support > support.shift(10))
        ).astype(int)
    
    def _detect_bull_flag(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect bull flag pattern."""
        # Simplified bull flag detection
        return (
            (close > close.shift(10)) &  # Uptrend
            (high.rolling(5).max() < high.rolling(10).max()) &  # Flag formation
            (close > close.shift(1))  # Bullish momentum
        ).astype(int)
    
    def _detect_bear_flag(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect bear flag pattern."""
        # Simplified bear flag detection
        return (
            (close < close.shift(10)) &  # Downtrend
            (low.rolling(5).min() > low.rolling(10).min()) &  # Flag formation
            (close < close.shift(1))  # Bearish momentum
        ).astype(int)
    
    def _detect_obv_divergence(self, close: pd.Series, obv: pd.Series, bullish: bool = True) -> pd.Series:
        """Detect OBV divergence patterns."""
        if bullish:
            return (
                (close < close.shift(10)) & (obv > obv.shift(10))
            ).astype(int)
        else:
            return (
                (close > close.shift(10)) & (obv < obv.shift(10))
            ).astype(int)
    
    def _detect_cmf_divergence(self, close: pd.Series, cmf: pd.Series, bullish: bool = True) -> pd.Series:
        """Detect CMF divergence patterns."""
        if bullish:
            return (
                (close < close.shift(10)) & (cmf > cmf.shift(10))
            ).astype(int)
        else:
            return (
                (close > close.shift(10)) & (cmf < cmf.shift(10))
            ).astype(int)
    
    def _detect_momentum_divergence(self, close: pd.Series, bullish: bool = True) -> pd.Series:
        """Detect momentum divergence patterns."""
        momentum = close / close.shift(10) - 1
        if bullish:
            return (
                (close < close.shift(20)) & (momentum > momentum.shift(10))
            ).astype(int)
        else:
            return (
                (close > close.shift(20)) & (momentum < momentum.shift(10))
            ).astype(int)
    
    def _detect_multiple_touches(self, close: pd.Series, level: pd.Series, pattern_type: str) -> pd.Series:
        """Detect multiple touches of support/resistance."""
        if pattern_type == 'support':
            touches = (close <= level * 1.002).astype(int)
        else:  # resistance
            touches = (close >= level * 0.998).astype(int)
        
        return (touches.rolling(20).sum() >= 3).astype(int)
    
    def _detect_break_retest(self, close: pd.Series, level: pd.Series, pattern_type: str) -> pd.Series:
        """Detect break and retest patterns."""
        if pattern_type == 'support':
            break_signal = (close < level.shift(1)).astype(int)
            retest_signal = (close > level).astype(int)
        else:  # resistance
            break_signal = (close > level.shift(1)).astype(int)
            retest_signal = (close < level).astype(int)
        
        return (break_signal.shift(5) & retest_signal).astype(int)
    
    def validate_patterns(self, patterns: Dict[str, pd.Series]) -> Dict[str, Dict[str, Any]]:
        """
        Validate patterns using VectorBT backtesting.
        
        Args:
            patterns: Discovered patterns
            
        Returns:
            Pattern validation results
        """
        logger.info("🔬 Validating patterns with VectorBT backtesting...")
        
        close = self.data['close']
        validation_results = {}
        
        try:
            for pattern_name, pattern_signal in patterns.items():
                if pattern_signal.isna().all() or pattern_signal.sum() == 0:
                    continue
                
                # Create entries and exits
                entries = pattern_signal == 1
                exits = pattern_signal.shift(1) == 1
                
                # Run backtest
                pf = vbt.Portfolio.from_signals(
                    close,
                    entries=entries,
                    exits=exits,
                    init_cash=10000,
                    fees=0.001,
                    freq='1H'
                )
                
                # Extract validation metrics
                validation_results[pattern_name] = {
                    'total_return': pf.total_return(),
                    'sharpe_ratio': pf.sharpe_ratio(),
                    'max_drawdown': pf.max_drawdown(),
                    'win_rate': pf.trades.win_rate(),
                    'profit_factor': pf.trades.profit_factor(),
                    'total_trades': pf.trades.count(),
                    'avg_trade_duration': pf.trades.duration.mean(),
                    'pattern_frequency': pattern_signal.sum() / len(pattern_signal),
                    'pattern_strength': self._calculate_pattern_strength(pattern_signal, close)
                }
            
            logger.info(f"✅ Validated {len(validation_results)} patterns")
            
        except Exception as e:
            logger.error(f"Error validating patterns: {e}")
            return {}
        
        return validation_results
    
    def _calculate_pattern_strength(self, pattern_signal: pd.Series, close: pd.Series) -> float:
        """Calculate pattern strength based on price movement."""
        try:
            pattern_returns = close.pct_change()[pattern_signal == 1]
            if len(pattern_returns) == 0:
                return 0.0
            return abs(pattern_returns.mean())
        except:
            return 0.0
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive VectorBT price pattern analysis.
        
        Returns:
            Complete analysis results
        """
        logger.info("🔬 Running comprehensive VectorBT price pattern analysis...")
        
        # Discover patterns
        patterns = self.discover_vectorbt_patterns()
        
        # Validate patterns
        validation_results = self.validate_patterns(patterns)
        
        # Generate pattern ranking
        pattern_ranking = self._create_pattern_ranking(validation_results)
        
        # Generate summary
        summary = self._generate_pattern_summary(patterns, validation_results)
        
        results = {
            'patterns': patterns,
            'validation_results': validation_results,
            'pattern_ranking': pattern_ranking,
            'summary': summary,
            'data_info': {
                'start_date': self.data.index.min(),
                'end_date': self.data.index.max(),
                'total_periods': len(self.data),
                'price_range': (self.data['close'].min(), self.data['close'].max())
            }
        }
        
        logger.info("✅ Comprehensive VectorBT price pattern analysis completed")
        return results
    
    def _create_pattern_ranking(self, validation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create pattern ranking based on validation results."""
        ranking_data = []
        
        for pattern_name, metrics in validation_results.items():
            ranking_data.append({
                'pattern': pattern_name,
                'sharpe_ratio': abs(metrics.get('sharpe_ratio', 0)),
                'total_return': metrics.get('total_return', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'pattern_frequency': metrics.get('pattern_frequency', 0),
                'pattern_strength': metrics.get('pattern_strength', 0),
                'total_trades': metrics.get('total_trades', 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Calculate composite score
        ranking_df['composite_score'] = (
            ranking_df['sharpe_ratio'] * 0.25 +
            ranking_df['total_return'] * 0.2 +
            ranking_df['win_rate'] * 0.2 +
            ranking_df['profit_factor'] * 0.15 +
            ranking_df['pattern_frequency'] * 0.1 +
            ranking_df['pattern_strength'] * 0.1
        )
        
        return ranking_df.sort_values('composite_score', ascending=False)
    
    def _generate_pattern_summary(self, patterns: Dict[str, pd.Series], 
                                validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate pattern analysis summary."""
        summary = {
            'total_patterns': len(patterns),
            'profitable_patterns': len([r for r in validation_results.values() if r['total_return'] > 0]),
            'best_pattern': None,
            'pattern_categories': {
                'candlestick': len([p for p in patterns.keys() if any(c in p for c in ['doji', 'hammer', 'shooting', 'engulfing', 'star'])]),
                'technical': len([p for p in patterns.keys() if any(t in p for t in ['rsi', 'macd', 'bb', 'stoch'])]),
                'price_action': len([p for p in patterns.keys() if any(pa in p for pa in ['breakout', 'double', 'head', 'triangle', 'flag'])]),
                'volume': len([p for p in patterns.keys() if any(v in p for v in ['volume', 'obv', 'ad', 'cmf'])]),
                'momentum': len([p for p in patterns.keys() if 'momentum' in p or 'roc' in p]),
                'trend': len([p for p in patterns.keys() if 'trend' in p or 'cross' in p]),
                'support_resistance': len([p for p in patterns.keys() if any(sr in p for sr in ['support', 'resistance', 'bounce', 'rejection'])])
            }
        }
        
        # Find best pattern
        if validation_results:
            best_pattern = max(validation_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            summary['best_pattern'] = {
                'name': best_pattern[0],
                'sharpe_ratio': best_pattern[1]['sharpe_ratio'],
                'total_return': best_pattern[1]['total_return'],
                'win_rate': best_pattern[1]['win_rate']
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = "vectorbt_patterns.json"):
        """Save analysis results to file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if key == 'patterns':
                serializable_results[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            elif key == 'pattern_ranking':
                serializable_results[key] = value.to_dict('records')
            else:
                serializable_results[key] = value
        
        import json
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        sample_data.loc[sample_data.index[i], 'high'] = max(sample_data.iloc[i][['open', 'high', 'low', 'close']])
        sample_data.loc[sample_data.index[i], 'low'] = min(sample_data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Run VectorBT pattern analysis
    optimizer = VectorBTPricePatternsOptimizer(sample_data)
    results = optimizer.run_comprehensive_analysis()
    
    # Save results
    optimizer.save_results(results)
    
    print("✅ VectorBT price pattern analysis completed!")
    print(f"Discovered {results['summary']['total_patterns']} patterns")
    print(f"Profitable patterns: {results['summary']['profitable_patterns']}")
    print(f"Best pattern: {results['summary']['best_pattern']}")
    
    # Show top patterns
    top_patterns = results['pattern_ranking'].head(10)
    print("\nTop 10 Patterns:")
    for _, row in top_patterns.iterrows():
        print(f"{row['pattern']}: {row['composite_score']:.4f}")