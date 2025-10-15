"""
Advanced Volatility Impact Research Framework.

This module goes beyond simple volatility clustering to measure how different 
volatility dimensions impact price movement patterns. It focuses on:

1. Volatility Regimes vs Price Pattern Relationships
2. Volatility Asymmetry Effects (leverage effect)
3. Volatility Persistence Impact on Trend Duration
4. Volatility Spillover Effects Across Timeframes
5. Volatility-Volume Interaction Effects
6. Implied vs Realized Volatility Divergence Impact

Key Research Questions:
- How does volatility regime affect price movement patterns beyond just "high vol = big moves"?
- What is the causal relationship between volatility changes and price pattern transitions?
- How do different volatility measures (realized, implied, GARCH, etc.) affect price dynamics?
- What is the economic significance of volatility-based trading signals?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

from src.utils.logger import system_logger


class VolatilityDimension(Enum):
    """Different dimensions of volatility to analyze."""
    REALIZED_VOLATILITY = "realized_volatility"
    GARCH_VOLATILITY = "garch_volatility"
    PARKINSON_VOLATILITY = "parkinson_volatility"  # High-Low based
    GARMAN_KLASS_VOLATILITY = "garman_klass_volatility"  # OHLC based
    VOLATILITY_OF_VOLATILITY = "volatility_of_volatility"
    VOLATILITY_SKEW = "volatility_skew"
    VOLATILITY_PERSISTENCE = "volatility_persistence"
    VOLATILITY_ASYMMETRY = "volatility_asymmetry"  # Leverage effect
    VOLATILITY_CLUSTERING = "volatility_clustering"
    VOLATILITY_SPILLOVER = "volatility_spillover"  # Cross-timeframe effects


class VolatilityImpactType(Enum):
    """Types of volatility impact on price patterns."""
    TREND_PERSISTENCE_MODULATION = "trend_persistence_modulation"
    REVERSAL_SPEED_ACCELERATION = "reversal_speed_acceleration"
    BREAKOUT_PROBABILITY_ENHANCEMENT = "breakout_probability_enhancement"
    MOMENTUM_DECAY_RATE_MODIFICATION = "momentum_decay_rate_modification"
    MEAN_REVERSION_STRENGTH_AMPLIFICATION = "mean_reversion_strength_amplification"
    PRICE_DISCOVERY_EFFICIENCY = "price_discovery_efficiency"
    TAIL_EVENT_CLUSTERING = "tail_event_clustering"
    REGIME_TRANSITION_TRIGGERING = "regime_transition_triggering"


@dataclass
class VolatilityImpactResult:
    """Results from volatility impact analysis."""
    volatility_dimension: VolatilityDimension
    impact_type: VolatilityImpactType
    impact_strength: float  # 0-1 scale
    economic_significance: bool
    causal_evidence: Dict[str, float]
    statistical_tests: Dict[str, float]
    trading_implications: str
    robustness_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    @property
    def is_economically_relevant(self) -> bool:
        """Determine if volatility impact is economically relevant."""
        return (self.economic_significance and 
                self.impact_strength > 0.3 and
                self.causal_evidence.get('granger_p_value', 1.0) < 0.05)


class VolatilityMeasureCalculator:
    """Calculator for various volatility measures."""
    
    def __init__(self):
        self.logger = system_logger.getChild('VolatilityMeasures')
    
    def calculate_all_volatility_measures(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive set of volatility measures."""
        
        measures = {}
        
        # Basic measures
        measures['realized_vol'] = self._realized_volatility(market_data)
        measures['parkinson_vol'] = self._parkinson_volatility(market_data)
        measures['garman_klass_vol'] = self._garman_klass_volatility(market_data)
        
        # Advanced measures
        measures['vol_of_vol'] = self._volatility_of_volatility(measures['realized_vol'])
        measures['vol_skew'] = self._volatility_skew(market_data)
        measures['vol_persistence'] = self._volatility_persistence(measures['realized_vol'])
        measures['vol_asymmetry'] = self._volatility_asymmetry(market_data)
        measures['vol_clustering'] = self._volatility_clustering(measures['realized_vol'])
        
        # Cross-timeframe measures
        measures['vol_spillover'] = self._volatility_spillover(market_data)
        
        return measures
    
    def _realized_volatility(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate realized volatility."""
        returns = market_data['close'].pct_change().fillna(0)
        return returns.rolling(window).std() * np.sqrt(252)
    
    def _parkinson_volatility(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility (high-low based)."""
        if not all(col in market_data.columns for col in ['high', 'low']):
            return pd.Series(0.0, index=market_data.index)
        
        hl_ratio = np.log(market_data['high'] / market_data['low'])
        parkinson = np.sqrt(hl_ratio.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
        return parkinson
    
    def _garman_klass_volatility(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility (OHLC based)."""
        if not all(col in market_data.columns for col in ['high', 'low', 'open', 'close']):
            return pd.Series(0.0, index=market_data.index)
        
        log_hl = np.log(market_data['high'] / market_data['low'])
        log_co = np.log(market_data['close'] / market_data['open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window).mean() * 252)
    
    def _volatility_of_volatility(self, volatility: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility of volatility."""
        return volatility.rolling(window).std()
    
    def _volatility_skew(self, market_data: pd.DataFrame, window: int = 50) -> pd.Series:
        """Calculate volatility skew (asymmetry in volatility distribution)."""
        returns = market_data['close'].pct_change().fillna(0)
        
        # Rolling skewness of absolute returns
        abs_returns = abs(returns)
        return abs_returns.rolling(window).skew()
    
    def _volatility_persistence(self, volatility: pd.Series, max_lag: int = 10) -> pd.Series:
        """Calculate volatility persistence (autocorrelation)."""
        persistence_scores = []
        
        for i in range(len(volatility)):
            if i >= max_lag:
                recent_vol = volatility.iloc[i-max_lag:i]
                if len(recent_vol) > 5:
                    # Calculate autocorrelation at lag 1
                    autocorr = recent_vol.autocorr(1)
                    persistence_scores.append(autocorr if not np.isnan(autocorr) else 0)
                else:
                    persistence_scores.append(0)
            else:
                persistence_scores.append(0)
        
        return pd.Series(persistence_scores, index=volatility.index)
    
    def _volatility_asymmetry(self, market_data: pd.DataFrame, window: int = 50) -> pd.Series:
        """Calculate volatility asymmetry (leverage effect)."""
        returns = market_data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std()
        
        asymmetry_scores = []
        
        for i in range(window, len(returns)):
            recent_returns = returns.iloc[i-window:i]
            recent_vol = volatility.iloc[i-window:i]
            
            # Correlation between returns and future volatility
            future_vol = volatility.iloc[i:i+5].mean() if i+5 < len(volatility) else recent_vol.iloc[-1]
            
            # Separate positive and negative returns
            pos_returns = recent_returns[recent_returns > 0]
            neg_returns = recent_returns[recent_returns < 0]
            
            if len(pos_returns) > 5 and len(neg_returns) > 5:
                pos_vol_impact = abs(pos_returns.corr(recent_vol[recent_returns > 0]))
                neg_vol_impact = abs(neg_returns.corr(recent_vol[recent_returns < 0]))
                
                asymmetry = neg_vol_impact - pos_vol_impact  # Leverage effect
                asymmetry_scores.append(asymmetry if not np.isnan(asymmetry) else 0)
            else:
                asymmetry_scores.append(0)
        
        # Pad beginning
        asymmetry_scores = [0] * window + asymmetry_scores
        return pd.Series(asymmetry_scores, index=market_data.index)
    
    def _volatility_clustering(self, volatility: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility clustering strength."""
        # Measure how often high volatility periods are followed by high volatility
        vol_percentile = volatility.rolling(100).rank(pct=True)
        
        clustering_scores = []
        
        for i in range(window, len(vol_percentile)):
            recent_percentiles = vol_percentile.iloc[i-window:i]
            
            # Count transitions from high vol to high vol
            high_vol_threshold = 0.8
            transitions = 0
            high_to_high = 0
            
            for j in range(len(recent_percentiles) - 1):
                if recent_percentiles.iloc[j] > high_vol_threshold:
                    transitions += 1
                    if recent_percentiles.iloc[j+1] > high_vol_threshold:
                        high_to_high += 1
            
            clustering = high_to_high / transitions if transitions > 0 else 0
            clustering_scores.append(clustering)
        
        # Pad beginning
        clustering_scores = [0] * window + clustering_scores
        return pd.Series(clustering_scores, index=volatility.index)
    
    def _volatility_spillover(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate volatility spillover across timeframes."""
        returns = market_data['close'].pct_change().fillna(0)
        
        # Different timeframe volatilities
        vol_5 = returns.rolling(5).std()
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std()
        
        # Spillover = correlation between short-term and long-term volatility changes
        spillover_scores = []
        window = 50
        
        for i in range(window, len(vol_5)):
            recent_vol_5_changes = vol_5.iloc[i-window:i].diff()
            recent_vol_50_changes = vol_50.iloc[i-window:i].diff()
            
            correlation = recent_vol_5_changes.corr(recent_vol_50_changes)
            spillover_scores.append(abs(correlation) if not np.isnan(correlation) else 0)
        
        # Pad beginning
        spillover_scores = [0] * window + spillover_scores
        return pd.Series(spillover_scores, index=market_data.index)


class VolatilityImpactAnalyzer:
    """Analyzer for volatility impact on price patterns."""
    
    def __init__(self):
        self.logger = system_logger.getChild('VolatilityImpactAnalyzer')
        self.volatility_calculator = VolatilityMeasureCalculator()
    
    def analyze_volatility_impact(self, 
                                market_data: pd.DataFrame,
                                volatility_dimension: VolatilityDimension,
                                impact_type: VolatilityImpactType) -> VolatilityImpactResult:
        """Analyze specific volatility dimension impact on price patterns."""
        
        self.logger.info(f"🌪️ Analyzing {volatility_dimension.value} impact on {impact_type.value}")
        
        # Calculate volatility measures
        volatility_measures = self.volatility_calculator.calculate_all_volatility_measures(market_data)
        
        # Get specific volatility measure
        vol_measure = self._get_volatility_measure(volatility_measures, volatility_dimension)
        
        # Analyze impact on specific pattern
        impact_analysis = self._analyze_specific_impact(market_data, vol_measure, impact_type)
        
        # Causal analysis
        causal_evidence = self._analyze_causal_relationship(market_data, vol_measure, impact_type)
        
        # Statistical tests
        statistical_tests = self._conduct_statistical_tests(market_data, vol_measure, impact_type)
        
        # Economic significance
        economic_significance = self._assess_economic_significance(impact_analysis, statistical_tests)
        
        # Trading implications
        trading_implications = self._generate_trading_implications(
            volatility_dimension, impact_type, impact_analysis
        )
        
        # Robustness tests
        robustness_metrics = self._conduct_robustness_tests(market_data, vol_measure, impact_type)
        
        return VolatilityImpactResult(
            volatility_dimension=volatility_dimension,
            impact_type=impact_type,
            impact_strength=impact_analysis['impact_strength'],
            economic_significance=economic_significance,
            causal_evidence=causal_evidence,
            statistical_tests=statistical_tests,
            trading_implications=trading_implications,
            robustness_metrics=robustness_metrics,
            confidence_intervals={}
        )
    
    def _get_volatility_measure(self, 
                              volatility_measures: Dict[str, pd.Series],
                              volatility_dimension: VolatilityDimension) -> pd.Series:
        """Get specific volatility measure."""
        
        mapping = {
            VolatilityDimension.REALIZED_VOLATILITY: 'realized_vol',
            VolatilityDimension.PARKINSON_VOLATILITY: 'parkinson_vol',
            VolatilityDimension.GARMAN_KLASS_VOLATILITY: 'garman_klass_vol',
            VolatilityDimension.VOLATILITY_OF_VOLATILITY: 'vol_of_vol',
            VolatilityDimension.VOLATILITY_SKEW: 'vol_skew',
            VolatilityDimension.VOLATILITY_PERSISTENCE: 'vol_persistence',
            VolatilityDimension.VOLATILITY_ASYMMETRY: 'vol_asymmetry',
            VolatilityDimension.VOLATILITY_CLUSTERING: 'vol_clustering',
            VolatilityDimension.VOLATILITY_SPILLOVER: 'vol_spillover'
        }
        
        key = mapping.get(volatility_dimension, 'realized_vol')
        return volatility_measures.get(key, pd.Series(0.0, index=volatility_measures['realized_vol'].index))
    
    def _analyze_specific_impact(self, 
                               market_data: pd.DataFrame,
                               vol_measure: pd.Series,
                               impact_type: VolatilityImpactType) -> Dict[str, float]:
        """Analyze specific impact type."""
        
        if impact_type == VolatilityImpactType.TREND_PERSISTENCE_MODULATION:
            return self._analyze_trend_persistence_impact(market_data, vol_measure)
        elif impact_type == VolatilityImpactType.REVERSAL_SPEED_ACCELERATION:
            return self._analyze_reversal_speed_impact(market_data, vol_measure)
        elif impact_type == VolatilityImpactType.BREAKOUT_PROBABILITY_ENHANCEMENT:
            return self._analyze_breakout_probability_impact(market_data, vol_measure)
        elif impact_type == VolatilityImpactType.MOMENTUM_DECAY_RATE_MODIFICATION:
            return self._analyze_momentum_decay_impact(market_data, vol_measure)
        elif impact_type == VolatilityImpactType.MEAN_REVERSION_STRENGTH_AMPLIFICATION:
            return self._analyze_mean_reversion_impact(market_data, vol_measure)
        else:
            return {'impact_strength': 0.0}
    
    def _analyze_trend_persistence_impact(self, 
                                        market_data: pd.DataFrame,
                                        vol_measure: pd.Series) -> Dict[str, float]:
        """Analyze how volatility affects trend persistence."""
        
        prices = market_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Define trend using moving averages
        ma_short = prices.rolling(10).mean()
        ma_long = prices.rolling(50).mean()
        trend_direction = np.where(ma_short > ma_long, 1, -1)
        
        # Analyze trend persistence under different volatility regimes
        vol_percentiles = vol_measure.rolling(100).rank(pct=True)
        
        # High volatility periods (top 30%)
        high_vol_periods = vol_percentiles > 0.7
        # Low volatility periods (bottom 30%)
        low_vol_periods = vol_percentiles < 0.3
        
        trend_durations_high_vol = []
        trend_durations_low_vol = []
        
        # Calculate trend durations for each volatility regime
        for vol_regime, duration_list in [(high_vol_periods, trend_durations_high_vol), 
                                         (low_vol_periods, trend_durations_low_vol)]:
            
            current_trend = None
            current_duration = 0
            
            for i in range(50, len(trend_direction)):
                if vol_regime.iloc[i]:  # In this volatility regime
                    if trend_direction[i] == current_trend:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            duration_list.append(current_duration)
                        current_trend = trend_direction[i]
                        current_duration = 1
                else:
                    # Exiting regime
                    if current_duration > 0:
                        duration_list.append(current_duration)
                        current_duration = 0
                        current_trend = None
        
        # Calculate impact strength
        if trend_durations_high_vol and trend_durations_low_vol:
            avg_duration_high_vol = np.mean(trend_durations_high_vol)
            avg_duration_low_vol = np.mean(trend_durations_low_vol)
            
            # Impact strength = relative difference in trend persistence
            impact_strength = abs(avg_duration_high_vol - avg_duration_low_vol) / max(avg_duration_high_vol, avg_duration_low_vol)
        else:
            impact_strength = 0.0
        
        return {
            'impact_strength': float(min(impact_strength, 1.0)),
            'high_vol_trend_duration': float(np.mean(trend_durations_high_vol)) if trend_durations_high_vol else 0.0,
            'low_vol_trend_duration': float(np.mean(trend_durations_low_vol)) if trend_durations_low_vol else 0.0
        }
    
    def _analyze_reversal_speed_impact(self, 
                                     market_data: pd.DataFrame,
                                     vol_measure: pd.Series) -> Dict[str, float]:
        """Analyze how volatility affects reversal speed."""
        
        prices = market_data['close']
        ma_20 = prices.rolling(20).mean()
        price_deviation = (prices - ma_20) / ma_20
        
        vol_percentiles = vol_measure.rolling(100).rank(pct=True)
        
        # Find reversal events (price crossing moving average)
        reversal_speeds_high_vol = []
        reversal_speeds_low_vol = []
        
        for i in range(20, len(prices) - 10):
            current_deviation = price_deviation.iloc[i]
            vol_regime = vol_percentiles.iloc[i]
            
            # Look for significant deviations
            if abs(current_deviation) > 0.02:  # 2% deviation
                # Look for reversal in next 10 periods
                future_deviations = price_deviation.iloc[i+1:i+11]
                
                # Check if reversal occurs (crossing zero)
                if current_deviation > 0:  # Above MA
                    reversal_points = future_deviations[future_deviations < 0]
                else:  # Below MA
                    reversal_points = future_deviations[future_deviations > 0]
                
                if len(reversal_points) > 0:
                    # Calculate reversal speed (magnitude per period)
                    reversal_periods = np.where(future_deviations.index == reversal_points.index[0])[0][0] + 1
                    reversal_magnitude = abs(current_deviation - reversal_points.iloc[0])
                    reversal_speed = reversal_magnitude / reversal_periods
                    
                    # Classify by volatility regime
                    if vol_regime > 0.7:  # High volatility
                        reversal_speeds_high_vol.append(reversal_speed)
                    elif vol_regime < 0.3:  # Low volatility
                        reversal_speeds_low_vol.append(reversal_speed)
        
        # Calculate impact strength
        if reversal_speeds_high_vol and reversal_speeds_low_vol:
            avg_speed_high_vol = np.mean(reversal_speeds_high_vol)
            avg_speed_low_vol = np.mean(reversal_speeds_low_vol)
            
            impact_strength = abs(avg_speed_high_vol - avg_speed_low_vol) / max(avg_speed_high_vol, avg_speed_low_vol)
        else:
            impact_strength = 0.0
        
        return {
            'impact_strength': float(min(impact_strength, 1.0)),
            'high_vol_reversal_speed': float(np.mean(reversal_speeds_high_vol)) if reversal_speeds_high_vol else 0.0,
            'low_vol_reversal_speed': float(np.mean(reversal_speeds_low_vol)) if reversal_speeds_low_vol else 0.0
        }
    
    def _analyze_breakout_probability_impact(self, 
                                           market_data: pd.DataFrame,
                                           vol_measure: pd.Series) -> Dict[str, float]:
        """Analyze how volatility affects breakout probability."""
        
        prices = market_data['close']
        
        # Bollinger Bands
        ma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = ma_20 + 2 * std_20
        lower_band = ma_20 - 2 * std_20
        
        vol_percentiles = vol_measure.rolling(100).rank(pct=True)
        
        # Analyze breakout probability under different volatility regimes
        breakout_success_high_vol = []
        breakout_success_low_vol = []
        
        for i in range(20, len(prices) - 5):
            current_price = prices.iloc[i]
            vol_regime = vol_percentiles.iloc[i]
            
            # Check if near bands
            near_upper = abs(current_price - upper_band.iloc[i]) / current_price < 0.01
            near_lower = abs(current_price - lower_band.iloc[i]) / current_price < 0.01
            
            if near_upper or near_lower:
                # Look for breakout in next 5 periods
                future_prices = prices.iloc[i+1:i+6]
                
                if near_upper:
                    breakout_success = any(future_prices > upper_band.iloc[i])
                else:
                    breakout_success = any(future_prices < lower_band.iloc[i])
                
                # Classify by volatility regime
                if vol_regime > 0.7:  # High volatility
                    breakout_success_high_vol.append(int(breakout_success))
                elif vol_regime < 0.3:  # Low volatility
                    breakout_success_low_vol.append(int(breakout_success))
        
        # Calculate impact strength
        if breakout_success_high_vol and breakout_success_low_vol:
            prob_high_vol = np.mean(breakout_success_high_vol)
            prob_low_vol = np.mean(breakout_success_low_vol)
            
            impact_strength = abs(prob_high_vol - prob_low_vol)
        else:
            impact_strength = 0.0
        
        return {
            'impact_strength': float(impact_strength),
            'breakout_prob_high_vol': float(np.mean(breakout_success_high_vol)) if breakout_success_high_vol else 0.0,
            'breakout_prob_low_vol': float(np.mean(breakout_success_low_vol)) if breakout_success_low_vol else 0.0
        }
    
    def _analyze_momentum_decay_impact(self, 
                                     market_data: pd.DataFrame,
                                     vol_measure: pd.Series) -> Dict[str, float]:
        """Analyze how volatility affects momentum decay rates."""
        
        returns = market_data['close'].pct_change().fillna(0)
        momentum = returns.rolling(10).mean()
        
        vol_percentiles = vol_measure.rolling(100).rank(pct=True)
        
        # Analyze momentum decay under different volatility regimes
        decay_rates_high_vol = []
        decay_rates_low_vol = []
        
        for i in range(50, len(momentum) - 20):
            current_momentum = momentum.iloc[i]
            vol_regime = vol_percentiles.iloc[i]
            
            # Only consider significant momentum
            if abs(current_momentum) > 0.005:  # 0.5% momentum
                # Track momentum decay over next 20 periods
                future_momentum = momentum.iloc[i+1:i+21]
                
                # Calculate decay rate (how quickly momentum approaches zero)
                decay_rate = 0.0
                for j, future_mom in enumerate(future_momentum):
                    if abs(future_mom) < abs(current_momentum) * 0.5:  # 50% decay
                        decay_rate = 1.0 / (j + 1)  # Faster decay = higher rate
                        break
                
                # Classify by volatility regime
                if vol_regime > 0.7:  # High volatility
                    decay_rates_high_vol.append(decay_rate)
                elif vol_regime < 0.3:  # Low volatility
                    decay_rates_low_vol.append(decay_rate)
        
        # Calculate impact strength
        if decay_rates_high_vol and decay_rates_low_vol:
            avg_decay_high_vol = np.mean(decay_rates_high_vol)
            avg_decay_low_vol = np.mean(decay_rates_low_vol)
            
            impact_strength = abs(avg_decay_high_vol - avg_decay_low_vol) / max(avg_decay_high_vol, avg_decay_low_vol) if max(avg_decay_high_vol, avg_decay_low_vol) > 0 else 0
        else:
            impact_strength = 0.0
        
        return {
            'impact_strength': float(min(impact_strength, 1.0)),
            'momentum_decay_high_vol': float(np.mean(decay_rates_high_vol)) if decay_rates_high_vol else 0.0,
            'momentum_decay_low_vol': float(np.mean(decay_rates_low_vol)) if decay_rates_low_vol else 0.0
        }
    
    def _analyze_mean_reversion_impact(self, 
                                     market_data: pd.DataFrame,
                                     vol_measure: pd.Series) -> Dict[str, float]:
        """Analyze how volatility affects mean reversion strength."""
        
        prices = market_data['close']
        ma_20 = prices.rolling(20).mean()
        price_deviation = (prices - ma_20) / ma_20
        
        vol_percentiles = vol_measure.rolling(100).rank(pct=True)
        
        # Analyze mean reversion strength under different volatility regimes
        reversion_strengths_high_vol = []
        reversion_strengths_low_vol = []
        
        for i in range(20, len(prices) - 10):
            current_deviation = price_deviation.iloc[i]
            vol_regime = vol_percentiles.iloc[i]
            
            # Only consider significant deviations
            if abs(current_deviation) > 0.02:  # 2% deviation
                # Measure reversion over next 10 periods
                future_prices = prices.iloc[i+1:i+11]
                target_price = ma_20.iloc[i]
                current_price = prices.iloc[i]
                
                # Calculate maximum reversion toward mean
                if current_deviation > 0:  # Above mean
                    min_future_price = future_prices.min()
                    reversion_strength = (current_price - min_future_price) / current_price
                else:  # Below mean
                    max_future_price = future_prices.max()
                    reversion_strength = (max_future_price - current_price) / current_price
                
                # Classify by volatility regime
                if vol_regime > 0.7:  # High volatility
                    reversion_strengths_high_vol.append(reversion_strength)
                elif vol_regime < 0.3:  # Low volatility
                    reversion_strengths_low_vol.append(reversion_strength)
        
        # Calculate impact strength
        if reversion_strengths_high_vol and reversion_strengths_low_vol:
            avg_reversion_high_vol = np.mean(reversion_strengths_high_vol)
            avg_reversion_low_vol = np.mean(reversion_strengths_low_vol)
            
            impact_strength = abs(avg_reversion_high_vol - avg_reversion_low_vol) / max(avg_reversion_high_vol, avg_reversion_low_vol)
        else:
            impact_strength = 0.0
        
        return {
            'impact_strength': float(min(impact_strength, 1.0)),
            'reversion_strength_high_vol': float(np.mean(reversion_strengths_high_vol)) if reversion_strengths_high_vol else 0.0,
            'reversion_strength_low_vol': float(np.mean(reversion_strengths_low_vol)) if reversion_strengths_low_vol else 0.0
        }
    
    def _analyze_causal_relationship(self, 
                                   market_data: pd.DataFrame,
                                   vol_measure: pd.Series,
                                   impact_type: VolatilityImpactType) -> Dict[str, float]:
        """Analyze causal relationship between volatility and price patterns."""
        
        # Create target pattern based on impact type
        if impact_type == VolatilityImpactType.TREND_PERSISTENCE_MODULATION:
            returns = market_data['close'].pct_change().fillna(0)
            target = returns.rolling(10).mean()  # Momentum proxy
        elif impact_type == VolatilityImpactType.REVERSAL_SPEED_ACCELERATION:
            prices = market_data['close']
            ma_20 = prices.rolling(20).mean()
            target = (prices - ma_20) / ma_20  # Mean reversion signal
        else:
            target = market_data['close'].pct_change().fillna(0)  # Default to returns
        
        # Align data
        aligned_data = pd.concat([vol_measure, target], axis=1).dropna()
        
        if len(aligned_data) < 100:
            return {'granger_p_value': 1.0, 'correlation': 0.0}
        
        # Simple Granger causality test
        try:
            X = aligned_data.iloc[:, 0].values  # volatility measure
            Y = aligned_data.iloc[:, 1].values  # target pattern
            
            max_lag = min(10, len(X) // 10)
            
            # Create lagged variables
            Y_lagged = np.column_stack([np.roll(Y, i+1) for i in range(max_lag)])
            X_lagged = np.column_stack([np.roll(X, i+1) for i in range(max_lag)])
            
            # Remove initial observations
            Y_current = Y[max_lag:]
            Y_lagged = Y_lagged[max_lag:]
            X_lagged = X_lagged[max_lag:]
            
            # Fit models
            from sklearn.linear_model import LinearRegression
            
            # Restricted model (only Y lags)
            restricted_model = LinearRegression()
            restricted_model.fit(Y_lagged, Y_current)
            rss_restricted = np.sum((Y_current - restricted_model.predict(Y_lagged)) ** 2)
            
            # Unrestricted model (Y lags + X lags)
            unrestricted_features = np.column_stack([Y_lagged, X_lagged])
            unrestricted_model = LinearRegression()
            unrestricted_model.fit(unrestricted_features, Y_current)
            rss_unrestricted = np.sum((Y_current - unrestricted_model.predict(unrestricted_features)) ** 2)
            
            # F-test
            n = len(Y_current)
            k_restricted = max_lag
            k_unrestricted = 2 * max_lag
            
            f_stat = ((rss_restricted - rss_unrestricted) / (k_unrestricted - k_restricted)) / (rss_unrestricted / (n - k_unrestricted - 1))
            
            # P-value
            from scipy.stats import f
            p_value = 1 - f.cdf(f_stat, k_unrestricted - k_restricted, n - k_unrestricted - 1)
            
            # Simple correlation
            correlation, _ = stats.pearsonr(X, Y)
            
            return {
                'granger_p_value': float(p_value),
                'correlation': float(abs(correlation)),
                'f_statistic': float(f_stat)
            }
            
        except Exception as e:
            self.logger.warning(f"Causal analysis failed: {e}")
            return {'granger_p_value': 1.0, 'correlation': 0.0}
    
    def _conduct_statistical_tests(self, 
                                 market_data: pd.DataFrame,
                                 vol_measure: pd.Series,
                                 impact_type: VolatilityImpactType) -> Dict[str, float]:
        """Conduct statistical tests for volatility impact."""
        
        # Get impact analysis
        impact_analysis = self._analyze_specific_impact(market_data, vol_measure, impact_type)
        impact_strength = impact_analysis['impact_strength']
        
        # Bootstrap confidence interval for impact strength
        bootstrap_impacts = []
        n_bootstrap = 100
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_indices = np.random.choice(len(market_data), size=len(market_data), replace=True)
            bootstrap_market_data = market_data.iloc[sample_indices].reset_index(drop=True)
            bootstrap_vol_measure = vol_measure.iloc[sample_indices].reset_index(drop=True)
            
            try:
                bootstrap_impact = self._analyze_specific_impact(
                    bootstrap_market_data, bootstrap_vol_measure, impact_type
                )
                bootstrap_impacts.append(bootstrap_impact['impact_strength'])
            except:
                pass
        
        # Statistical tests
        if bootstrap_impacts:
            # Test against null hypothesis of no impact (impact_strength = 0)
            t_stat, p_value = stats.ttest_1samp(bootstrap_impacts, 0)
            
            # Confidence interval
            ci_lower = np.percentile(bootstrap_impacts, 2.5)
            ci_upper = np.percentile(bootstrap_impacts, 97.5)
        else:
            t_stat, p_value = 0, 1.0
            ci_lower, ci_upper = 0, 0
        
        return {
            'impact_t_statistic': float(t_stat),
            'impact_p_value': float(p_value),
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'bootstrap_mean': float(np.mean(bootstrap_impacts)) if bootstrap_impacts else 0.0,
            'bootstrap_std': float(np.std(bootstrap_impacts)) if bootstrap_impacts else 0.0
        }
    
    def _assess_economic_significance(self, 
                                    impact_analysis: Dict[str, float],
                                    statistical_tests: Dict[str, float]) -> bool:
        """Assess economic significance of volatility impact."""
        
        impact_strength = impact_analysis['impact_strength']
        p_value = statistical_tests.get('impact_p_value', 1.0)
        
        # Economic significance criteria
        criteria = [
            impact_strength > 0.2,  # Minimum 20% impact
            p_value < 0.05,  # Statistical significance
            statistical_tests.get('confidence_interval_lower', 0) > 0.1  # Lower bound > 10%
        ]
        
        return sum(criteria) >= 2  # At least 2 out of 3 criteria
    
    def _generate_trading_implications(self, 
                                     volatility_dimension: VolatilityDimension,
                                     impact_type: VolatilityImpactType,
                                     impact_analysis: Dict[str, float]) -> str:
        """Generate trading implications from volatility impact analysis."""
        
        impact_strength = impact_analysis['impact_strength']
        
        if impact_strength > 0.5:
            strength_desc = "strong"
        elif impact_strength > 0.3:
            strength_desc = "moderate"
        elif impact_strength > 0.1:
            strength_desc = "weak"
        else:
            strength_desc = "negligible"
        
        base_implication = f"{volatility_dimension.value} shows {strength_desc} impact on {impact_type.value}"
        
        if impact_strength > 0.3:
            if impact_type == VolatilityImpactType.TREND_PERSISTENCE_MODULATION:
                return f"{base_implication}. Use volatility regime to adjust trend-following strategy position sizing and holding periods."
            elif impact_type == VolatilityImpactType.REVERSAL_SPEED_ACCELERATION:
                return f"{base_implication}. Use volatility signals to time mean reversion entries and predict reversal speed."
            elif impact_type == VolatilityImpactType.BREAKOUT_PROBABILITY_ENHANCEMENT:
                return f"{base_implication}. Use volatility regime to filter breakout signals and improve success rate."
            elif impact_type == VolatilityImpactType.MOMENTUM_DECAY_RATE_MODIFICATION:
                return f"{base_implication}. Adjust momentum strategy exit timing based on volatility regime."
            elif impact_type == VolatilityImpactType.MEAN_REVERSION_STRENGTH_AMPLIFICATION:
                return f"{base_implication}. Scale mean reversion position sizes based on volatility regime."
            else:
                return f"{base_implication}. Consider incorporating into trading strategy development."
        else:
            return f"{base_implication}. Limited trading utility for this volatility-pattern combination."
    
    def _conduct_robustness_tests(self, 
                                market_data: pd.DataFrame,
                                vol_measure: pd.Series,
                                impact_type: VolatilityImpactType) -> Dict[str, float]:
        """Conduct robustness tests for volatility impact analysis."""
        
        robustness_metrics = {}
        
        # 1. Subsample stability
        if len(market_data) > 500:
            subsample_impacts = []
            n_subsamples = 10
            subsample_size = len(market_data) // 2
            
            for _ in range(n_subsamples):
                start_idx = np.random.randint(0, len(market_data) - subsample_size)
                end_idx = start_idx + subsample_size
                
                subsample_data = market_data.iloc[start_idx:end_idx]
                subsample_vol = vol_measure.iloc[start_idx:end_idx]
                
                try:
                    subsample_impact = self._analyze_specific_impact(
                        subsample_data, subsample_vol, impact_type
                    )
                    subsample_impacts.append(subsample_impact['impact_strength'])
                except:
                    pass
            
            if subsample_impacts:
                robustness_metrics['subsample_stability'] = float(1.0 - np.std(subsample_impacts))
        
        # 2. Time period stability
        if len(market_data) > 1000:
            n_periods = 4
            period_size = len(market_data) // n_periods
            period_impacts = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size
                
                period_data = market_data.iloc[start_idx:end_idx]
                period_vol = vol_measure.iloc[start_idx:end_idx]
                
                try:
                    period_impact = self._analyze_specific_impact(
                        period_data, period_vol, impact_type
                    )
                    period_impacts.append(period_impact['impact_strength'])
                except:
                    pass
            
            if period_impacts:
                robustness_metrics['time_stability'] = float(1.0 - np.std(period_impacts))
        
        return robustness_metrics


# Main orchestrator for comprehensive volatility impact research
class VolatilityImpactResearchOrchestrator:
    """Orchestrator for comprehensive volatility impact research."""
    
    def __init__(self):
        self.logger = system_logger.getChild('VolatilityImpactResearch')
        self.analyzer = VolatilityImpactAnalyzer()
    
    def conduct_comprehensive_volatility_research(self, 
                                                market_data: pd.DataFrame) -> Dict[str, Dict[str, VolatilityImpactResult]]:
        """Conduct comprehensive volatility impact research."""
        
        self.logger.info("🌪️ Starting comprehensive volatility impact research")
        
        # Define research matrix
        volatility_dimensions = [
            VolatilityDimension.REALIZED_VOLATILITY,
            VolatilityDimension.VOLATILITY_OF_VOLATILITY,
            VolatilityDimension.VOLATILITY_PERSISTENCE,
            VolatilityDimension.VOLATILITY_ASYMMETRY,
            VolatilityDimension.VOLATILITY_CLUSTERING
        ]
        
        impact_types = [
            VolatilityImpactType.TREND_PERSISTENCE_MODULATION,
            VolatilityImpactType.REVERSAL_SPEED_ACCELERATION,
            VolatilityImpactType.BREAKOUT_PROBABILITY_ENHANCEMENT,
            VolatilityImpactType.MOMENTUM_DECAY_RATE_MODIFICATION,
            VolatilityImpactType.MEAN_REVERSION_STRENGTH_AMPLIFICATION
        ]
        
        results = {}
        
        for vol_dimension in volatility_dimensions:
            self.logger.info(f"📊 Analyzing {vol_dimension.value}")
            
            dimension_results = {}
            
            for impact_type in impact_types:
                try:
                    result = self.analyzer.analyze_volatility_impact(
                        market_data, vol_dimension, impact_type
                    )
                    dimension_results[impact_type.value] = result
                    
                    if result.is_economically_relevant:
                        self.logger.info(f"   ✅ {impact_type.value}: Economically relevant!")
                    else:
                        self.logger.info(f"   ❌ {impact_type.value}: Not economically relevant")
                        
                except Exception as e:
                    self.logger.error(f"   ⚠️ {impact_type.value} failed: {e}")
                    continue
            
            if dimension_results:
                results[vol_dimension.value] = dimension_results
        
        self.logger.info(f"✅ Volatility impact research completed")
        return results
    
    def generate_volatility_research_report(self, 
                                          research_results: Dict[str, Dict[str, VolatilityImpactResult]]) -> str:
        """Generate comprehensive volatility research report."""
        
        report = []
        report.append("# Volatility Impact Research Report")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        total_tests = sum(len(dimension_results) for dimension_results in research_results.values())
        relevant_tests = sum(
            1 for dimension_results in research_results.values()
            for result in dimension_results.values()
            if result.is_economically_relevant
        )
        
        relevance_rate = (relevant_tests / total_tests * 100) if total_tests > 0 else 0
        
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Volatility-Impact Combinations**: {total_tests}")
        report.append(f"- **Economically Relevant Results**: {relevant_tests}")
        report.append(f"- **Economic Relevance Rate**: {relevance_rate:.1f}%")
        report.append("")
        
        # Key Findings
        report.append("## Key Volatility Impact Findings")
        report.append("")
        
        for vol_dimension, dimension_results in research_results.items():
            relevant_impacts = [
                (impact_type, result) for impact_type, result in dimension_results.items()
                if result.is_economically_relevant
            ]
            
            if relevant_impacts:
                report.append(f"### {vol_dimension.replace('_', ' ').title()}")
                report.append("")
                
                for impact_type, result in relevant_impacts:
                    report.append(f"✅ **{impact_type.replace('_', ' ').title()}**")
                    report.append(f"   - Impact Strength: {result.impact_strength:.3f}")
                    report.append(f"   - Trading Implications: {result.trading_implications}")
                    report.append("")
        
        # Research Insights
        report.append("## Research Insights: Beyond Simple Volatility Clustering")
        report.append("")
        
        if relevance_rate > 50:
            report.append("🎯 **Significant Discovery**: Volatility dimensions show substantial impact on price patterns beyond simple clustering")
            report.append("- Multiple volatility measures provide distinct economic value")
            report.append("- Different volatility regimes require different trading approaches")
            report.append("- Volatility-based market regime identification is economically justified")
        elif relevance_rate > 25:
            report.append("⚠️ **Moderate Discovery**: Some volatility dimensions show economic relevance")
            report.append("- Selective use of volatility measures recommended")
            report.append("- Focus on highest-impact volatility dimensions")
        else:
            report.append("❌ **Limited Discovery**: Traditional volatility clustering may be sufficient")
            report.append("- Advanced volatility measures provide limited additional value")
            report.append("- Consider focusing on simpler volatility-based approaches")
        
        return "\n".join(report)


# Example usage
def run_volatility_impact_research_example():
    """Example of how to run volatility impact research."""
    
    print("Volatility Impact Research Framework")
    print("===================================")
    print()
    print("This framework analyzes how different volatility dimensions impact price patterns:")
    print("1. Trend Persistence Modulation - How volatility affects trend duration")
    print("2. Reversal Speed Acceleration - How volatility affects mean reversion speed")
    print("3. Breakout Probability Enhancement - How volatility affects breakout success")
    print("4. Momentum Decay Rate Modification - How volatility affects momentum persistence")
    print("5. Mean Reversion Strength Amplification - How volatility affects reversion strength")
    print()
    print("Volatility measures analyzed:")
    print("- Realized Volatility (standard)")
    print("- Volatility of Volatility (vol clustering)")
    print("- Volatility Persistence (autocorrelation)")
    print("- Volatility Asymmetry (leverage effect)")
    print("- Volatility Clustering (high vol → high vol probability)")
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = VolatilityImpactResearchOrchestrator()")
    print("results = orchestrator.conduct_comprehensive_volatility_research(market_data)")
    print("report = orchestrator.generate_volatility_research_report(results)")
    print("```")


if __name__ == "__main__":
    run_volatility_impact_research_example()