"""
Market Microstructure Impact Research Framework.

This module provides advanced research methodologies to measure how market 
microstructure dimensions impact price movement patterns and price discovery.

Market microstructure goes beyond simple volume analysis to examine:
1. Order flow dynamics and their impact on price formation
2. Bid-ask spread behavior and liquidity provision
3. Trade size distribution and market impact
4. Information asymmetry effects on price efficiency
5. Market maker vs taker behavior patterns
6. Intraday pattern effects on price discovery

Key Research Questions:
- How do order flow imbalances affect price movement patterns?
- What is the relationship between spread dynamics and price efficiency?
- How does trade size distribution impact momentum vs mean reversion?
- Which microstructure signals predict price pattern transitions?
- How do market depth changes affect breakout probabilities?
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
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings

from src.utils.logger import system_logger


class MicrostructureDimension(Enum):
    """Different microstructure dimensions to analyze."""
    ORDER_FLOW_IMBALANCE = "order_flow_imbalance"
    BID_ASK_SPREAD = "bid_ask_spread"
    MARKET_DEPTH = "market_depth"
    TRADE_SIZE_DISTRIBUTION = "trade_size_distribution"
    PRICE_IMPACT = "price_impact"
    INFORMATION_ASYMMETRY = "information_asymmetry"
    MARKET_MAKER_INVENTORY = "market_maker_inventory"
    TICK_FREQUENCY = "tick_frequency"
    VOLUME_WEIGHTED_PRICE_DEVIATION = "vwap_deviation"
    MICROSTRUCTURE_NOISE = "microstructure_noise"


class MicrostructureImpactType(Enum):
    """Types of microstructure impact on price patterns."""
    PRICE_DISCOVERY_EFFICIENCY = "price_discovery_efficiency"
    MOMENTUM_AMPLIFICATION = "momentum_amplification"
    MEAN_REVERSION_ACCELERATION = "mean_reversion_acceleration"
    BREAKOUT_CONFIRMATION = "breakout_confirmation"
    LIQUIDITY_CRISIS_PREDICTION = "liquidity_crisis_prediction"
    INFORMATION_INCORPORATION_SPEED = "information_incorporation_speed"
    ADVERSE_SELECTION_IMPACT = "adverse_selection_impact"
    MARKET_IMPACT_PERSISTENCE = "market_impact_persistence"
    INTRADAY_PATTERN_FORMATION = "intraday_pattern_formation"
    REGIME_TRANSITION_SIGNALING = "regime_transition_signaling"


@dataclass
class MicrostructureImpactResult:
    """Results from microstructure impact analysis."""
    microstructure_dimension: MicrostructureDimension
    impact_type: MicrostructureImpactType
    impact_strength: float  # 0-1 scale
    predictive_accuracy: float  # 0-1 scale
    economic_significance: bool
    information_content: Dict[str, float]
    statistical_tests: Dict[str, float]
    trading_implications: str
    robustness_metrics: Dict[str, float]
    
    @property
    def is_economically_relevant(self) -> bool:
        """Determine if microstructure impact is economically relevant."""
        return (self.economic_significance and 
                self.impact_strength > 0.25 and
                self.predictive_accuracy > 0.55)


class MicrostructureMetricsCalculator:
    """Calculator for market microstructure metrics."""
    
    def __init__(self):
        self.logger = system_logger.getChild('MicrostructureMetrics')
    
    def calculate_microstructure_metrics(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive microstructure metrics."""
        
        metrics = {}
        
        # Basic microstructure proxies from OHLCV data
        metrics['order_flow_proxy'] = self._calculate_order_flow_proxy(market_data)
        metrics['spread_proxy'] = self._calculate_spread_proxy(market_data)
        metrics['market_depth_proxy'] = self._calculate_market_depth_proxy(market_data)
        metrics['trade_size_proxy'] = self._calculate_trade_size_proxy(market_data)
        metrics['price_impact'] = self._calculate_price_impact(market_data)
        metrics['information_asymmetry'] = self._calculate_information_asymmetry(market_data)
        metrics['vwap_deviation'] = self._calculate_vwap_deviation(market_data)
        metrics['microstructure_noise'] = self._calculate_microstructure_noise(market_data)
        metrics['tick_frequency'] = self._calculate_tick_frequency(market_data)
        
        return metrics
    
    def _calculate_order_flow_proxy(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate order flow imbalance proxy.
        
        Uses the relationship between volume, price change, and intraday patterns
        to estimate buy vs sell pressure.
        """
        if 'volume' not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        returns = market_data['close'].pct_change().fillna(0)
        volume = market_data['volume']
        
        # Order flow proxy: signed volume based on price movement
        # Positive returns suggest more buying pressure
        signed_volume = returns.rolling(5).mean() * volume
        
        # Normalize by rolling average volume
        avg_volume = volume.rolling(50).mean()
        order_flow_imbalance = signed_volume / avg_volume.where(avg_volume > 0, 1)
        
        return order_flow_imbalance.fillna(0)
    
    def _calculate_spread_proxy(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate bid-ask spread proxy.
        
        Uses high-low range as a proxy for bid-ask spread.
        """
        if not all(col in market_data.columns for col in ['high', 'low', 'close']):
            return pd.Series(0.0, index=market_data.index)
        
        # Relative spread: (high - low) / close
        spread_proxy = (market_data['high'] - market_data['low']) / market_data['close']
        
        return spread_proxy.fillna(0)
    
    def _calculate_market_depth_proxy(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate market depth proxy.
        
        Uses volume-to-volatility ratio as a proxy for market depth.
        Higher volume with lower volatility suggests better depth.
        """
        if 'volume' not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        returns = market_data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std()
        volume = market_data['volume']
        
        # Depth proxy: volume / volatility
        # High volume with low volatility = good depth
        depth_proxy = volume / volatility.where(volatility > 0, np.inf)
        depth_proxy = depth_proxy.replace([np.inf, -np.inf], 0)
        
        # Normalize
        depth_proxy = (depth_proxy - depth_proxy.rolling(100).mean()) / depth_proxy.rolling(100).std()
        
        return depth_proxy.fillna(0)
    
    def _calculate_trade_size_proxy(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate trade size distribution proxy.
        
        Uses volume patterns to infer trade size characteristics.
        """
        if 'volume' not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        volume = market_data['volume']
        
        # Trade size proxy: volume volatility
        # High volume volatility suggests heterogeneous trade sizes
        volume_volatility = volume.rolling(20).std()
        avg_volume = volume.rolling(20).mean()
        
        trade_size_proxy = volume_volatility / avg_volume.where(avg_volume > 0, 1)
        
        return trade_size_proxy.fillna(0)
    
    def _calculate_price_impact(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate price impact measure.
        
        Measures how much prices move per unit of volume.
        """
        if 'volume' not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        returns = market_data['close'].pct_change().fillna(0)
        volume = market_data['volume']
        
        # Price impact: |return| / volume
        price_impact = abs(returns) / volume.where(volume > 0, 1)
        
        # Smooth to reduce noise
        price_impact = price_impact.rolling(10).mean()
        
        return price_impact.fillna(0)
    
    def _calculate_information_asymmetry(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate information asymmetry proxy.
        
        Uses the relationship between spreads and adverse selection.
        """
        spread_proxy = self._calculate_spread_proxy(market_data)
        returns = market_data['close'].pct_change().fillna(0)
        
        # Information asymmetry: correlation between spreads and future returns
        asymmetry_scores = []
        window = 50
        
        for i in range(window, len(spread_proxy)):
            recent_spreads = spread_proxy.iloc[i-window:i]
            future_returns = returns.iloc[i:i+5].sum() if i+5 < len(returns) else 0
            
            # High spread followed by significant return suggests information asymmetry
            correlation = recent_spreads.corr(pd.Series([future_returns] * len(recent_spreads)))
            asymmetry_scores.append(abs(correlation) if not np.isnan(correlation) else 0)
        
        # Pad beginning
        asymmetry_scores = [0] * window + asymmetry_scores
        return pd.Series(asymmetry_scores, index=market_data.index)
    
    def _calculate_vwap_deviation(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP deviation.
        
        Measures how far current price is from volume-weighted average price.
        """
        if 'volume' not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
        volume = market_data['volume']
        
        # Rolling VWAP
        window = 20
        vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
        
        # Deviation from VWAP
        vwap_deviation = (market_data['close'] - vwap) / vwap.where(vwap > 0, 1)
        
        return vwap_deviation.fillna(0)
    
    def _calculate_microstructure_noise(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate microstructure noise measure.
        
        Measures the amount of noise in price movements due to microstructure effects.
        """
        returns = market_data['close'].pct_change().fillna(0)
        
        # Microstructure noise: first-order autocorrelation of returns
        # Negative autocorrelation suggests bid-ask bounce and other noise
        noise_scores = []
        window = 50
        
        for i in range(window, len(returns)):
            recent_returns = returns.iloc[i-window:i]
            autocorr = recent_returns.autocorr(1)
            noise_scores.append(abs(autocorr) if not np.isnan(autocorr) else 0)
        
        # Pad beginning
        noise_scores = [0] * window + noise_scores
        return pd.Series(noise_scores, index=market_data.index)
    
    def _calculate_tick_frequency(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate tick frequency proxy.
        
        Uses volume and volatility to infer trading frequency.
        """
        if 'volume' not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        volume = market_data['volume']
        returns = market_data['close'].pct_change().fillna(0)
        
        # Tick frequency proxy: volume / |returns|
        # High volume with small price changes suggests frequent small trades
        tick_freq = volume / abs(returns).where(abs(returns) > 0, 1)
        
        # Normalize
        tick_freq = (tick_freq - tick_freq.rolling(100).mean()) / tick_freq.rolling(100).std()
        
        return tick_freq.fillna(0)


class MicrostructureImpactAnalyzer:
    """Analyzer for microstructure impact on price patterns."""
    
    def __init__(self):
        self.logger = system_logger.getChild('MicrostructureImpactAnalyzer')
        self.metrics_calculator = MicrostructureMetricsCalculator()
    
    def analyze_microstructure_impact(self, 
                                    market_data: pd.DataFrame,
                                    microstructure_dimension: MicrostructureDimension,
                                    impact_type: MicrostructureImpactType) -> MicrostructureImpactResult:
        """Analyze microstructure impact on price patterns."""
        
        self.logger.info(f"🔬 Analyzing {microstructure_dimension.value} impact on {impact_type.value}")
        
        # Calculate microstructure metrics
        microstructure_metrics = self.metrics_calculator.calculate_microstructure_metrics(market_data)
        
        # Get specific microstructure measure
        micro_measure = self._get_microstructure_measure(microstructure_metrics, microstructure_dimension)
        
        # Analyze impact on specific pattern
        impact_analysis = self._analyze_specific_microstructure_impact(
            market_data, micro_measure, impact_type
        )
        
        # Information content analysis
        information_content = self._analyze_information_content(
            market_data, micro_measure, impact_type
        )
        
        # Statistical tests
        statistical_tests = self._conduct_microstructure_statistical_tests(
            market_data, micro_measure, impact_type
        )
        
        # Economic significance
        economic_significance = self._assess_microstructure_economic_significance(
            impact_analysis, information_content, statistical_tests
        )
        
        # Trading implications
        trading_implications = self._generate_microstructure_trading_implications(
            microstructure_dimension, impact_type, impact_analysis
        )
        
        # Robustness tests
        robustness_metrics = self._conduct_microstructure_robustness_tests(
            market_data, micro_measure, impact_type
        )
        
        return MicrostructureImpactResult(
            microstructure_dimension=microstructure_dimension,
            impact_type=impact_type,
            impact_strength=impact_analysis['impact_strength'],
            predictive_accuracy=impact_analysis.get('predictive_accuracy', 0.5),
            economic_significance=economic_significance,
            information_content=information_content,
            statistical_tests=statistical_tests,
            trading_implications=trading_implications,
            robustness_metrics=robustness_metrics
        )
    
    def _get_microstructure_measure(self, 
                                  microstructure_metrics: Dict[str, pd.Series],
                                  microstructure_dimension: MicrostructureDimension) -> pd.Series:
        """Get specific microstructure measure."""
        
        mapping = {
            MicrostructureDimension.ORDER_FLOW_IMBALANCE: 'order_flow_proxy',
            MicrostructureDimension.BID_ASK_SPREAD: 'spread_proxy',
            MicrostructureDimension.MARKET_DEPTH: 'market_depth_proxy',
            MicrostructureDimension.TRADE_SIZE_DISTRIBUTION: 'trade_size_proxy',
            MicrostructureDimension.PRICE_IMPACT: 'price_impact',
            MicrostructureDimension.INFORMATION_ASYMMETRY: 'information_asymmetry',
            MicrostructureDimension.VOLUME_WEIGHTED_PRICE_DEVIATION: 'vwap_deviation',
            MicrostructureDimension.MICROSTRUCTURE_NOISE: 'microstructure_noise',
            MicrostructureDimension.TICK_FREQUENCY: 'tick_frequency'
        }
        
        key = mapping.get(microstructure_dimension, 'order_flow_proxy')
        return microstructure_metrics.get(key, pd.Series(0.0, index=list(microstructure_metrics.values())[0].index))
    
    def _analyze_specific_microstructure_impact(self, 
                                              market_data: pd.DataFrame,
                                              micro_measure: pd.Series,
                                              impact_type: MicrostructureImpactType) -> Dict[str, float]:
        """Analyze specific microstructure impact type."""
        
        if impact_type == MicrostructureImpactType.PRICE_DISCOVERY_EFFICIENCY:
            return self._analyze_price_discovery_efficiency(market_data, micro_measure)
        elif impact_type == MicrostructureImpactType.MOMENTUM_AMPLIFICATION:
            return self._analyze_momentum_amplification(market_data, micro_measure)
        elif impact_type == MicrostructureImpactType.MEAN_REVERSION_ACCELERATION:
            return self._analyze_mean_reversion_acceleration(market_data, micro_measure)
        elif impact_type == MicrostructureImpactType.BREAKOUT_CONFIRMATION:
            return self._analyze_breakout_confirmation(market_data, micro_measure)
        elif impact_type == MicrostructureImpactType.LIQUIDITY_CRISIS_PREDICTION:
            return self._analyze_liquidity_crisis_prediction(market_data, micro_measure)
        else:
            return {'impact_strength': 0.0, 'predictive_accuracy': 0.5}
    
    def _analyze_price_discovery_efficiency(self, 
                                          market_data: pd.DataFrame,
                                          micro_measure: pd.Series) -> Dict[str, float]:
        """Analyze how microstructure affects price discovery efficiency."""
        
        returns = market_data['close'].pct_change().fillna(0)
        
        # Price discovery efficiency: how quickly prices incorporate information
        # Measured by return autocorrelation patterns
        
        # Divide data by microstructure regime
        micro_percentiles = micro_measure.rolling(100).rank(pct=True)
        high_micro_periods = micro_percentiles > 0.7  # Top 30%
        low_micro_periods = micro_percentiles < 0.3   # Bottom 30%
        
        # Calculate autocorrelation patterns for each regime
        high_micro_autocorrs = []
        low_micro_autocorrs = []
        
        window = 50
        for i in range(window, len(returns)):
            if high_micro_periods.iloc[i]:
                recent_returns = returns.iloc[i-window:i]
                autocorr = recent_returns.autocorr(1)
                if not np.isnan(autocorr):
                    high_micro_autocorrs.append(abs(autocorr))
            
            if low_micro_periods.iloc[i]:
                recent_returns = returns.iloc[i-window:i]
                autocorr = recent_returns.autocorr(1)
                if not np.isnan(autocorr):
                    low_micro_autocorrs.append(abs(autocorr))
        
        # Calculate efficiency difference
        if high_micro_autocorrs and low_micro_autocorrs:
            # Lower autocorrelation = more efficient price discovery
            high_micro_efficiency = 1.0 - np.mean(high_micro_autocorrs)
            low_micro_efficiency = 1.0 - np.mean(low_micro_autocorrs)
            
            impact_strength = abs(high_micro_efficiency - low_micro_efficiency)
        else:
            impact_strength = 0.0
        
        return {
            'impact_strength': float(min(impact_strength, 1.0)),
            'high_micro_efficiency': float(high_micro_efficiency) if high_micro_autocorrs else 0.5,
            'low_micro_efficiency': float(low_micro_efficiency) if low_micro_autocorrs else 0.5
        }
    
    def _analyze_momentum_amplification(self, 
                                      market_data: pd.DataFrame,
                                      micro_measure: pd.Series) -> Dict[str, float]:
        """Analyze how microstructure amplifies momentum patterns."""
        
        returns = market_data['close'].pct_change().fillna(0)
        momentum = returns.rolling(10).mean()
        
        # Analyze momentum persistence under different microstructure conditions
        micro_percentiles = micro_measure.rolling(100).rank(pct=True)
        
        momentum_persistence_scores = []
        predictive_accuracy_scores = []
        
        for i in range(50, len(momentum) - 10):
            current_momentum = momentum.iloc[i]
            current_micro = micro_percentiles.iloc[i]
            
            if abs(current_momentum) > 0.005:  # Significant momentum
                # Look at momentum continuation over next 10 periods
                future_momentum = momentum.iloc[i+1:i+11]
                
                # Check if momentum continues in same direction
                same_direction = np.sum(np.sign(future_momentum) == np.sign(current_momentum))
                persistence_score = same_direction / len(future_momentum)
                
                momentum_persistence_scores.append((persistence_score, current_micro))
                
                # Predictive accuracy: does high microstructure signal predict momentum continuation?
                if current_micro > 0.7:  # High microstructure activity
                    predicted_continuation = 1
                else:
                    predicted_continuation = 0
                
                actual_continuation = 1 if persistence_score > 0.6 else 0
                predictive_accuracy_scores.append(predicted_continuation == actual_continuation)
        
        # Calculate impact strength
        if momentum_persistence_scores:
            # Correlation between microstructure activity and momentum persistence
            persistence_values = [score[0] for score in momentum_persistence_scores]
            micro_values = [score[1] for score in momentum_persistence_scores]
            
            correlation, _ = stats.pearsonr(micro_values, persistence_values)
            impact_strength = abs(correlation) if not np.isnan(correlation) else 0
            
            # Predictive accuracy
            predictive_accuracy = np.mean(predictive_accuracy_scores) if predictive_accuracy_scores else 0.5
        else:
            impact_strength = 0.0
            predictive_accuracy = 0.5
        
        return {
            'impact_strength': float(impact_strength),
            'predictive_accuracy': float(predictive_accuracy)
        }
    
    def _analyze_mean_reversion_acceleration(self, 
                                           market_data: pd.DataFrame,
                                           micro_measure: pd.Series) -> Dict[str, float]:
        """Analyze how microstructure accelerates mean reversion."""
        
        prices = market_data['close']
        ma_20 = prices.rolling(20).mean()
        price_deviation = (prices - ma_20) / ma_20
        
        micro_percentiles = micro_measure.rolling(100).rank(pct=True)
        
        reversion_speeds = []
        predictive_accuracy_scores = []
        
        for i in range(20, len(prices) - 10):
            current_deviation = price_deviation.iloc[i]
            current_micro = micro_percentiles.iloc[i]
            
            if abs(current_deviation) > 0.02:  # 2% deviation from mean
                # Look for reversion over next 10 periods
                future_prices = prices.iloc[i+1:i+11]
                target_price = ma_20.iloc[i]
                current_price = prices.iloc[i]
                
                # Calculate reversion speed
                reversion_occurred = False
                reversion_speed = 0.0
                
                for j, future_price in enumerate(future_prices):
                    if current_deviation > 0 and future_price <= target_price:
                        # Reverted below mean
                        reversion_speed = abs(current_deviation) / (j + 1)
                        reversion_occurred = True
                        break
                    elif current_deviation < 0 and future_price >= target_price:
                        # Reverted above mean
                        reversion_speed = abs(current_deviation) / (j + 1)
                        reversion_occurred = True
                        break
                
                reversion_speeds.append((reversion_speed, current_micro))
                
                # Predictive accuracy: does high microstructure activity predict faster reversion?
                predicted_fast_reversion = 1 if current_micro > 0.7 else 0
                actual_fast_reversion = 1 if reversion_speed > np.median([s[0] for s in reversion_speeds[-50:]]) else 0
                predictive_accuracy_scores.append(predicted_fast_reversion == actual_fast_reversion)
        
        # Calculate impact strength
        if reversion_speeds:
            speeds = [speed[0] for speed in reversion_speeds]
            micro_values = [speed[1] for speed in reversion_speeds]
            
            correlation, _ = stats.pearsonr(micro_values, speeds)
            impact_strength = abs(correlation) if not np.isnan(correlation) else 0
            
            predictive_accuracy = np.mean(predictive_accuracy_scores) if predictive_accuracy_scores else 0.5
        else:
            impact_strength = 0.0
            predictive_accuracy = 0.5
        
        return {
            'impact_strength': float(impact_strength),
            'predictive_accuracy': float(predictive_accuracy)
        }
    
    def _analyze_breakout_confirmation(self, 
                                     market_data: pd.DataFrame,
                                     micro_measure: pd.Series) -> Dict[str, float]:
        """Analyze how microstructure confirms breakout patterns."""
        
        prices = market_data['close']
        
        # Bollinger Bands for breakout detection
        ma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = ma_20 + 2 * std_20
        lower_band = ma_20 - 2 * std_20
        
        micro_percentiles = micro_measure.rolling(100).rank(pct=True)
        
        breakout_confirmations = []
        predictive_accuracy_scores = []
        
        for i in range(20, len(prices) - 5):
            current_price = prices.iloc[i]
            current_micro = micro_percentiles.iloc[i]
            
            # Check for potential breakout
            near_upper = abs(current_price - upper_band.iloc[i]) / current_price < 0.01
            near_lower = abs(current_price - lower_band.iloc[i]) / current_price < 0.01
            
            if near_upper or near_lower:
                # Look for breakout confirmation in next 5 periods
                future_prices = prices.iloc[i+1:i+6]
                
                if near_upper:
                    breakout_confirmed = any(future_prices > upper_band.iloc[i])
                else:
                    breakout_confirmed = any(future_prices < lower_band.iloc[i])
                
                breakout_confirmations.append((int(breakout_confirmed), current_micro))
                
                # Predictive accuracy: does high microstructure activity predict breakout confirmation?
                predicted_breakout = 1 if current_micro > 0.7 else 0
                actual_breakout = int(breakout_confirmed)
                predictive_accuracy_scores.append(predicted_breakout == actual_breakout)
        
        # Calculate impact strength
        if breakout_confirmations:
            confirmations = [conf[0] for conf in breakout_confirmations]
            micro_values = [conf[1] for conf in breakout_confirmations]
            
            correlation, _ = stats.pearsonr(micro_values, confirmations)
            impact_strength = abs(correlation) if not np.isnan(correlation) else 0
            
            predictive_accuracy = np.mean(predictive_accuracy_scores) if predictive_accuracy_scores else 0.5
        else:
            impact_strength = 0.0
            predictive_accuracy = 0.5
        
        return {
            'impact_strength': float(impact_strength),
            'predictive_accuracy': float(predictive_accuracy)
        }
    
    def _analyze_liquidity_crisis_prediction(self, 
                                           market_data: pd.DataFrame,
                                           micro_measure: pd.Series) -> Dict[str, float]:
        """Analyze how microstructure predicts liquidity crises."""
        
        returns = market_data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std()
        
        # Define liquidity crisis as periods of extreme volatility
        vol_percentile = volatility.rolling(100).rank(pct=True)
        liquidity_crisis = vol_percentile > 0.95  # Top 5% volatility periods
        
        micro_percentiles = micro_measure.rolling(100).rank(pct=True)
        
        crisis_predictions = []
        
        for i in range(100, len(liquidity_crisis) - 5):
            current_micro = micro_percentiles.iloc[i]
            
            # Look for crisis in next 5 periods
            future_crisis = any(liquidity_crisis.iloc[i+1:i+6])
            
            # High microstructure activity might predict crisis
            predicted_crisis = 1 if current_micro > 0.8 else 0
            actual_crisis = int(future_crisis)
            
            crisis_predictions.append((predicted_crisis, actual_crisis, current_micro))
        
        # Calculate predictive performance
        if crisis_predictions:
            predicted = [pred[0] for pred in crisis_predictions]
            actual = [pred[1] for pred in crisis_predictions]
            micro_values = [pred[2] for pred in crisis_predictions]
            
            # Predictive accuracy
            accuracy = accuracy_score(actual, predicted)
            
            # Impact strength: correlation between microstructure and future crises
            future_crises = [pred[1] for pred in crisis_predictions]
            correlation, _ = stats.pearsonr(micro_values, future_crises)
            impact_strength = abs(correlation) if not np.isnan(correlation) else 0
        else:
            accuracy = 0.5
            impact_strength = 0.0
        
        return {
            'impact_strength': float(impact_strength),
            'predictive_accuracy': float(accuracy)
        }
    
    def _analyze_information_content(self, 
                                   market_data: pd.DataFrame,
                                   micro_measure: pd.Series,
                                   impact_type: MicrostructureImpactType) -> Dict[str, float]:
        """Analyze information content of microstructure measure."""
        
        returns = market_data['close'].pct_change().fillna(0)
        
        information_metrics = {}
        
        # 1. Predictive correlation with future returns
        future_correlations = []
        for lag in [1, 5, 10]:
            future_returns = returns.shift(-lag)
            correlation = micro_measure.corr(future_returns)
            if not np.isnan(correlation):
                future_correlations.append(abs(correlation))
        
        information_metrics['predictive_correlation'] = float(np.mean(future_correlations)) if future_correlations else 0.0
        
        # 2. Information ratio (excess correlation per unit of noise)
        micro_volatility = micro_measure.rolling(20).std()
        if micro_volatility.mean() > 0:
            information_ratio = information_metrics['predictive_correlation'] / micro_volatility.mean()
        else:
            information_ratio = 0.0
        
        information_metrics['information_ratio'] = float(information_ratio)
        
        # 3. Regime-dependent information content
        vol = returns.rolling(20).std()
        vol_percentile = vol.rolling(100).rank(pct=True)
        
        high_vol_correlation = micro_measure[vol_percentile > 0.7].corr(returns[vol_percentile > 0.7])
        low_vol_correlation = micro_measure[vol_percentile < 0.3].corr(returns[vol_percentile < 0.3])
        
        information_metrics['high_vol_info_content'] = float(abs(high_vol_correlation)) if not np.isnan(high_vol_correlation) else 0.0
        information_metrics['low_vol_info_content'] = float(abs(low_vol_correlation)) if not np.isnan(low_vol_correlation) else 0.0
        
        return information_metrics
    
    def _conduct_microstructure_statistical_tests(self, 
                                                 market_data: pd.DataFrame,
                                                 micro_measure: pd.Series,
                                                 impact_type: MicrostructureImpactType) -> Dict[str, float]:
        """Conduct statistical tests for microstructure impact."""
        
        returns = market_data['close'].pct_change().fillna(0)
        
        statistical_tests = {}
        
        # 1. Correlation significance test
        correlation, p_value = stats.pearsonr(micro_measure.dropna(), returns.dropna())
        statistical_tests['correlation'] = float(correlation)
        statistical_tests['correlation_p_value'] = float(p_value)
        
        # 2. Regime difference test
        micro_percentiles = micro_measure.rolling(100).rank(pct=True)
        high_micro_returns = returns[micro_percentiles > 0.7].dropna()
        low_micro_returns = returns[micro_percentiles < 0.3].dropna()
        
        if len(high_micro_returns) > 10 and len(low_micro_returns) > 10:
            t_stat, p_value = stats.ttest_ind(high_micro_returns, low_micro_returns)
            statistical_tests['regime_difference_t_stat'] = float(t_stat)
            statistical_tests['regime_difference_p_value'] = float(p_value)
        
        # 3. Predictive power test (Granger causality simplified)
        try:
            # Simple lag correlation test
            lag_correlations = []
            for lag in [1, 5, 10]:
                future_returns = returns.shift(-lag)
                lag_corr = micro_measure.corr(future_returns)
                if not np.isnan(lag_corr):
                    lag_correlations.append(abs(lag_corr))
            
            statistical_tests['max_lag_correlation'] = float(max(lag_correlations)) if lag_correlations else 0.0
            statistical_tests['avg_lag_correlation'] = float(np.mean(lag_correlations)) if lag_correlations else 0.0
        except:
            statistical_tests['max_lag_correlation'] = 0.0
            statistical_tests['avg_lag_correlation'] = 0.0
        
        return statistical_tests
    
    def _assess_microstructure_economic_significance(self, 
                                                   impact_analysis: Dict[str, float],
                                                   information_content: Dict[str, float],
                                                   statistical_tests: Dict[str, float]) -> bool:
        """Assess economic significance of microstructure impact."""
        
        impact_strength = impact_analysis.get('impact_strength', 0)
        predictive_accuracy = impact_analysis.get('predictive_accuracy', 0.5)
        predictive_correlation = information_content.get('predictive_correlation', 0)
        p_value = statistical_tests.get('correlation_p_value', 1.0)
        
        # Economic significance criteria
        criteria = [
            impact_strength > 0.2,  # Minimum 20% impact strength
            predictive_accuracy > 0.55,  # Better than random prediction
            predictive_correlation > 0.1,  # Meaningful predictive correlation
            p_value < 0.05  # Statistical significance
        ]
        
        return sum(criteria) >= 3  # At least 3 out of 4 criteria
    
    def _generate_microstructure_trading_implications(self, 
                                                    microstructure_dimension: MicrostructureDimension,
                                                    impact_type: MicrostructureImpactType,
                                                    impact_analysis: Dict[str, float]) -> str:
        """Generate trading implications from microstructure analysis."""
        
        impact_strength = impact_analysis.get('impact_strength', 0)
        predictive_accuracy = impact_analysis.get('predictive_accuracy', 0.5)
        
        if impact_strength > 0.4 and predictive_accuracy > 0.6:
            strength_desc = "strong"
        elif impact_strength > 0.25 and predictive_accuracy > 0.55:
            strength_desc = "moderate"
        else:
            strength_desc = "weak"
        
        base_implication = f"{microstructure_dimension.value} shows {strength_desc} impact on {impact_type.value}"
        
        if strength_desc == "strong":
            if impact_type == MicrostructureImpactType.MOMENTUM_AMPLIFICATION:
                return f"{base_implication}. Use microstructure signals to enhance momentum strategy entry timing and position sizing."
            elif impact_type == MicrostructureImpactType.BREAKOUT_CONFIRMATION:
                return f"{base_implication}. Use microstructure activity to confirm breakout signals and reduce false breakouts."
            elif impact_type == MicrostructureImpactType.LIQUIDITY_CRISIS_PREDICTION:
                return f"{base_implication}. Use microstructure deterioration as early warning for liquidity crises and risk management."
            elif impact_type == MicrostructureImpactType.PRICE_DISCOVERY_EFFICIENCY:
                return f"{base_implication}. Use microstructure efficiency measures to optimize trade execution timing."
            else:
                return f"{base_implication}. Strong signal for systematic trading strategy integration."
        elif strength_desc == "moderate":
            return f"{base_implication}. Consider as supporting indicator in multi-signal trading strategies."
        else:
            return f"{base_implication}. Limited direct trading utility, may be useful for market regime identification."
    
    def _conduct_microstructure_robustness_tests(self, 
                                               market_data: pd.DataFrame,
                                               micro_measure: pd.Series,
                                               impact_type: MicrostructureImpactType) -> Dict[str, float]:
        """Conduct robustness tests for microstructure impact analysis."""
        
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
                subsample_micro = micro_measure.iloc[start_idx:end_idx]
                
                try:
                    subsample_impact = self._analyze_specific_microstructure_impact(
                        subsample_data, subsample_micro, impact_type
                    )
                    subsample_impacts.append(subsample_impact['impact_strength'])
                except:
                    pass
            
            if subsample_impacts:
                robustness_metrics['subsample_stability'] = float(1.0 - np.std(subsample_impacts))
        
        # 2. Market regime stability
        returns = market_data['close'].pct_change().fillna(0)
        vol = returns.rolling(20).std()
        vol_percentile = vol.rolling(100).rank(pct=True)
        
        # Test impact in different volatility regimes
        regime_impacts = []
        
        for vol_threshold in [0.3, 0.7]:
            if vol_threshold == 0.3:
                regime_mask = vol_percentile < vol_threshold
            else:
                regime_mask = vol_percentile > vol_threshold
            
            regime_data = market_data[regime_mask]
            regime_micro = micro_measure[regime_mask]
            
            if len(regime_data) > 100:
                try:
                    regime_impact = self._analyze_specific_microstructure_impact(
                        regime_data, regime_micro, impact_type
                    )
                    regime_impacts.append(regime_impact['impact_strength'])
                except:
                    pass
        
        if len(regime_impacts) > 1:
            robustness_metrics['regime_stability'] = float(1.0 - abs(regime_impacts[0] - regime_impacts[1]))
        
        return robustness_metrics


# Main orchestrator for comprehensive microstructure research
class MicrostructureImpactResearchOrchestrator:
    """Orchestrator for comprehensive microstructure impact research."""
    
    def __init__(self):
        self.logger = system_logger.getChild('MicrostructureImpactResearch')
        self.analyzer = MicrostructureImpactAnalyzer()
    
    def conduct_comprehensive_microstructure_research(self, 
                                                    market_data: pd.DataFrame) -> Dict[str, Dict[str, MicrostructureImpactResult]]:
        """Conduct comprehensive microstructure impact research."""
        
        self.logger.info("🔬 Starting comprehensive microstructure impact research")
        
        # Define research matrix
        microstructure_dimensions = [
            MicrostructureDimension.ORDER_FLOW_IMBALANCE,
            MicrostructureDimension.BID_ASK_SPREAD,
            MicrostructureDimension.MARKET_DEPTH,
            MicrostructureDimension.PRICE_IMPACT,
            MicrostructureDimension.INFORMATION_ASYMMETRY
        ]
        
        impact_types = [
            MicrostructureImpactType.PRICE_DISCOVERY_EFFICIENCY,
            MicrostructureImpactType.MOMENTUM_AMPLIFICATION,
            MicrostructureImpactType.MEAN_REVERSION_ACCELERATION,
            MicrostructureImpactType.BREAKOUT_CONFIRMATION,
            MicrostructureImpactType.LIQUIDITY_CRISIS_PREDICTION
        ]
        
        results = {}
        
        for micro_dimension in microstructure_dimensions:
            self.logger.info(f"📊 Analyzing {micro_dimension.value}")
            
            dimension_results = {}
            
            for impact_type in impact_types:
                try:
                    result = self.analyzer.analyze_microstructure_impact(
                        market_data, micro_dimension, impact_type
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
                results[micro_dimension.value] = dimension_results
        
        self.logger.info(f"✅ Microstructure impact research completed")
        return results
    
    def generate_microstructure_research_report(self, 
                                              research_results: Dict[str, Dict[str, MicrostructureImpactResult]]) -> str:
        """Generate comprehensive microstructure research report."""
        
        report = []
        report.append("# Microstructure Impact Research Report")
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
        report.append(f"- **Total Microstructure-Impact Combinations**: {total_tests}")
        report.append(f"- **Economically Relevant Results**: {relevant_tests}")
        report.append(f"- **Economic Relevance Rate**: {relevance_rate:.1f}%")
        report.append("")
        
        # Key Findings
        report.append("## Key Microstructure Impact Findings")
        report.append("")
        
        for micro_dimension, dimension_results in research_results.items():
            relevant_impacts = [
                (impact_type, result) for impact_type, result in dimension_results.items()
                if result.is_economically_relevant
            ]
            
            if relevant_impacts:
                report.append(f"### {micro_dimension.replace('_', ' ').title()}")
                report.append("")
                
                for impact_type, result in relevant_impacts:
                    report.append(f"✅ **{impact_type.replace('_', ' ').title()}**")
                    report.append(f"   - Impact Strength: {result.impact_strength:.3f}")
                    report.append(f"   - Predictive Accuracy: {result.predictive_accuracy:.3f}")
                    report.append(f"   - Trading Implications: {result.trading_implications}")
                    report.append("")
        
        # Research Insights
        report.append("## Research Insights: Beyond Simple Volume Analysis")
        report.append("")
        
        if relevance_rate > 50:
            report.append("🎯 **Significant Discovery**: Microstructure dimensions show substantial impact on price patterns")
            report.append("- Market microstructure provides distinct economic value beyond volume")
            report.append("- Order flow and spread dynamics significantly affect price discovery")
            report.append("- Microstructure-based regime identification is economically justified")
        elif relevance_rate > 25:
            report.append("⚠️ **Moderate Discovery**: Some microstructure dimensions show economic relevance")
            report.append("- Selective use of microstructure measures recommended")
            report.append("- Focus on highest-impact microstructure dimensions")
        else:
            report.append("❌ **Limited Discovery**: Traditional volume analysis may be sufficient")
            report.append("- Advanced microstructure measures provide limited additional value")
            report.append("- Consider focusing on simpler volume-based approaches")
        
        return "\n".join(report)


# Example usage
def run_microstructure_impact_research_example():
    """Example of how to run microstructure impact research."""
    
    print("Microstructure Impact Research Framework")
    print("=======================================")
    print()
    print("This framework analyzes how market microstructure dimensions impact price patterns:")
    print("1. Price Discovery Efficiency - How microstructure affects information incorporation")
    print("2. Momentum Amplification - How order flow amplifies momentum patterns")
    print("3. Mean Reversion Acceleration - How spreads affect reversion speed")
    print("4. Breakout Confirmation - How depth confirms breakout patterns")
    print("5. Liquidity Crisis Prediction - How microstructure predicts liquidity events")
    print()
    print("Microstructure measures analyzed (from OHLCV data):")
    print("- Order Flow Imbalance (buy vs sell pressure proxy)")
    print("- Bid-Ask Spread (high-low range proxy)")
    print("- Market Depth (volume/volatility ratio)")
    print("- Price Impact (return per unit volume)")
    print("- Information Asymmetry (spread-return correlation)")
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = MicrostructureImpactResearchOrchestrator()")
    print("results = orchestrator.conduct_comprehensive_microstructure_research(market_data)")
    print("report = orchestrator.generate_microstructure_research_report(results)")
    print("```")


if __name__ == "__main__":
    run_microstructure_impact_research_example()