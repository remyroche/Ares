"""
Shared Fallback Analysis Module

This module provides unified fallback analysis capabilities for both TAS and NAS components
when primary analysis methods fail or are unavailable.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('FallbackAnalyzer')

@dataclass
class FallbackAnalysisResult:
    """Container for fallback analysis results."""
    signal_direction: str
    confidence_score: float
    market_health_score: float
    volatility_score: float
    liquidation_risk_score: float
    technical_indicators: Dict[str, float]
    analysis_metadata: Dict[str, Any]

class UnifiedFallbackAnalyzer:
    """
    Unified fallback analyzer for both TAS and NAS components.
    
    Provides conservative, rule-based analysis when primary ML methods
    are unavailable or fail.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified fallback analyzer.
        
        Args:
            config: Configuration dictionary for fallback analysis
        """
        self.config = config or {}
        self.logger = logger.getChild('UnifiedFallbackAnalyzer')
        
        # Fallback analysis parameters
        self.min_data_points = self.config.get('min_data_points', 10)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.05)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.02)
        self.confidence_floor = self.config.get('confidence_floor', 0.3)
        self.confidence_ceiling = self.config.get('confidence_ceiling', 0.7)
        
        # Conservative parameters
        self.conservative_mode = self.config.get('conservative_mode', True)
        self.risk_aversion = self.config.get('risk_aversion', 0.8)
        
        # Performance tracking
        self.fallback_analysis_count = 0
        self.analysis_times = []
        
    @handles_errors
    @traced(span_name="fallback_analysis")
    @log_execution_time()
    async def perform_fallback_analysis(
        self,
        market_data: pd.DataFrame,
        analysis_type: str = "both",
        current_position: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> FallbackAnalysisResult:
        """
        Perform fallback analysis when primary methods fail.
        
        Args:
            market_data: Market data DataFrame
            analysis_type: Type of analysis ("nas", "tas", or "both")
            current_position: Current position information
            additional_context: Additional context for analysis
            
        Returns:
            FallbackAnalysisResult: Conservative analysis result
        """
        try:
            if market_data.empty or len(market_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for fallback analysis: {len(market_data)} rows")
                return self._create_conservative_result()
            
            tprint_info(f"🔄 Performing {analysis_type} fallback analysis on {len(market_data)} data points")
            
            # Basic market analysis
            market_metrics = self._analyze_market_conditions(market_data)
            
            # Technical indicators
            technical_indicators = self._calculate_basic_technical_indicators(market_data)
            
            # Signal direction determination
            signal_direction = self._determine_signal_direction(
                market_metrics, technical_indicators, current_position, analysis_type
            )
            
            # Confidence calculation
            confidence_score = self._calculate_fallback_confidence(
                market_metrics, technical_indicators, analysis_type
            )
            
            # Risk assessment
            risk_metrics = self._assess_risk_metrics(market_metrics, technical_indicators)
            
            # Create result
            result = FallbackAnalysisResult(
                signal_direction=signal_direction,
                confidence_score=confidence_score,
                market_health_score=risk_metrics['market_health'],
                volatility_score=risk_metrics['volatility'],
                liquidation_risk_score=risk_metrics['liquidation_risk'],
                technical_indicators=technical_indicators,
                analysis_metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_type': analysis_type,
                    'method': 'fallback',
                    'data_points': len(market_data),
                    'conservative_mode': self.conservative_mode,
                    'additional_context': additional_context or {}
                }
            )
            
            self.fallback_analysis_count += 1
            tprint_success(f"✅ Fallback analysis completed: {signal_direction} (confidence: {confidence_score:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Fallback analysis failed: {e}")
            return self._create_conservative_result()
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze basic market conditions."""
        try:
            if len(market_data) < 20:
                return self._get_default_market_metrics()
            
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Price momentum
            recent_returns = returns[-5:].mean() if len(returns) >= 5 else 0.0
            medium_returns = returns[-10:].mean() if len(returns) >= 10 else 0.0
            long_returns = returns[-20:].mean() if len(returns) >= 20 else 0.0
            
            # Volatility analysis
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            
            # Volume analysis
            volume_ratio = 1.0
            if 'volume' in market_data.columns and len(market_data) >= 20:
                volumes = market_data['volume'].values
                recent_volume = np.mean(volumes[-5:])
                historical_volume = np.mean(volumes[-20:])
                if historical_volume > 0:
                    volume_ratio = recent_volume / historical_volume
            
            return {
                'recent_momentum': recent_returns,
                'medium_momentum': medium_returns,
                'long_momentum': long_returns,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_strength': abs(long_returns),
                'price_stability': 1.0 - min(volatility * 10, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Market conditions analysis failed: {e}")
            return self._get_default_market_metrics()
    
    def _calculate_basic_technical_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators."""
        try:
            if len(market_data) < 20:
                return self._get_default_technical_indicators()
            
            close_prices = market_data['close'].values
            indicators = {}
            
            # Simple RSI
            if len(close_prices) >= 14:
                delta = np.diff(close_prices)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi'] = 100 - (100 / (1 + rs))
                else:
                    indicators['rsi'] = 100.0
            else:
                indicators['rsi'] = 50.0
            
            # Simple MACD
            if len(close_prices) >= 26:
                ema_12 = np.mean(close_prices[-12:])
                ema_26 = np.mean(close_prices[-26:])
                indicators['macd'] = ema_12 - ema_26
            else:
                indicators['macd'] = 0.0
            
            # Moving average position
            if len(close_prices) >= 20:
                ma_20 = np.mean(close_prices[-20:])
                current_price = close_prices[-1]
                indicators['ma_position'] = (current_price - ma_20) / ma_20 if ma_20 > 0 else 0.0
            else:
                indicators['ma_position'] = 0.0
            
            # Bollinger Bands position
            if len(close_prices) >= 20:
                ma_20 = np.mean(close_prices[-20:])
                std_20 = np.std(close_prices[-20:])
                upper_band = ma_20 + 2 * std_20
                lower_band = ma_20 - 2 * std_20
                current_price = close_prices[-1]
                
                if upper_band > lower_band:
                    indicators['bb_position'] = (current_price - lower_band) / (upper_band - lower_band)
                else:
                    indicators['bb_position'] = 0.5
            else:
                indicators['bb_position'] = 0.5
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f"⚠️ Technical indicators calculation failed: {e}")
            return self._get_default_technical_indicators()
    
    def _determine_signal_direction(
        self,
        market_metrics: Dict[str, float],
        technical_indicators: Dict[str, float],
        current_position: Optional[Dict[str, Any]],
        analysis_type: str
    ) -> str:
        """Determine signal direction based on fallback analysis."""
        try:
            # Get key metrics
            recent_momentum = market_metrics.get('recent_momentum', 0.0)
            volatility = market_metrics.get('volatility', 0.02)
            rsi = technical_indicators.get('rsi', 50.0)
            ma_position = technical_indicators.get('ma_position', 0.0)
            
            # Conservative decision making
            if self.conservative_mode:
                # More conservative thresholds
                momentum_threshold = self.momentum_threshold * 1.5
                volatility_threshold = self.volatility_threshold * 0.8
            else:
                momentum_threshold = self.momentum_threshold
                volatility_threshold = self.volatility_threshold
            
            # Check if we have a position
            if current_position:
                # Exit logic for existing positions
                if volatility > volatility_threshold or abs(recent_momentum) < momentum_threshold * 0.5:
                    return 'exit'
                elif (current_position.get('side') == 'long' and recent_momentum < -momentum_threshold) or \
                     (current_position.get('side') == 'short' and recent_momentum > momentum_threshold):
                    return 'exit'
                else:
                    return 'hold'
            else:
                # Entry logic for new positions
                # Check volatility first (safety)
                if volatility > volatility_threshold:
                    return 'hold'
                
                # Check momentum
                if recent_momentum > momentum_threshold:
                    # Additional checks for long signals
                    if rsi < 70 and ma_position > -0.02:  # Not overbought and above MA
                        return 'buy'
                elif recent_momentum < -momentum_threshold:
                    # Additional checks for short signals
                    if rsi > 30 and ma_position < 0.02:  # Not oversold and below MA
                        return 'sell'
                
                return 'hold'
                
        except Exception as e:
            self.logger.warning(f"⚠️ Signal direction determination failed: {e}")
            return 'hold'
    
    def _calculate_fallback_confidence(
        self,
        market_metrics: Dict[str, float],
        technical_indicators: Dict[str, float],
        analysis_type: str
    ) -> float:
        """Calculate confidence score for fallback analysis."""
        try:
            # Base confidence from market conditions
            volatility = market_metrics.get('volatility', 0.02)
            trend_strength = market_metrics.get('trend_strength', 0.0)
            price_stability = market_metrics.get('price_stability', 0.5)
            
            # Technical indicator confidence
            rsi = technical_indicators.get('rsi', 50.0)
            rsi_confidence = 1.0 - abs(rsi - 50.0) / 50.0  # Higher confidence when RSI is neutral
            
            # Volume confidence
            volume_ratio = market_metrics.get('volume_ratio', 1.0)
            volume_confidence = min(volume_ratio, 2.0) / 2.0  # Normalize volume ratio
            
            # Combine confidence factors
            base_confidence = (trend_strength * 0.4 + price_stability * 0.3 + rsi_confidence * 0.2 + volume_confidence * 0.1)
            
            # Apply volatility penalty
            if volatility > self.volatility_threshold:
                volatility_penalty = min((volatility - self.volatility_threshold) * 2, 0.3)
                base_confidence -= volatility_penalty
            
            # Apply risk aversion
            if self.conservative_mode:
                base_confidence *= self.risk_aversion
            
            # Clamp confidence
            confidence = np.clip(base_confidence, self.confidence_floor, self.confidence_ceiling)
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Confidence calculation failed: {e}")
            return self.confidence_floor
    
    def _assess_risk_metrics(
        self,
        market_metrics: Dict[str, float],
        technical_indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess risk metrics for fallback analysis."""
        try:
            volatility = market_metrics.get('volatility', 0.02)
            price_stability = market_metrics.get('price_stability', 0.5)
            
            # Market health based on stability and volatility
            market_health = (price_stability + (1.0 - min(volatility * 10, 1.0))) / 2
            
            # Liquidation risk (simplified)
            liquidation_risk = min(volatility * 2, 0.5)  # Higher volatility = higher liquidation risk
            
            return {
                'market_health': market_health,
                'volatility': volatility,
                'liquidation_risk': liquidation_risk
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Risk assessment failed: {e}")
            return {
                'market_health': 0.5,
                'volatility': 0.02,
                'liquidation_risk': 0.1
            }
    
    def _get_default_market_metrics(self) -> Dict[str, float]:
        """Get default market metrics for insufficient data."""
        return {
            'recent_momentum': 0.0,
            'medium_momentum': 0.0,
            'long_momentum': 0.0,
            'volatility': 0.02,
            'volume_ratio': 1.0,
            'trend_strength': 0.0,
            'price_stability': 0.5
        }
    
    def _get_default_technical_indicators(self) -> Dict[str, float]:
        """Get default technical indicators for insufficient data."""
        return {
            'rsi': 50.0,
            'macd': 0.0,
            'ma_position': 0.0,
            'bb_position': 0.5
        }
    
    def _create_conservative_result(self) -> FallbackAnalysisResult:
        """Create conservative fallback result."""
        return FallbackAnalysisResult(
            signal_direction='hold',
            confidence_score=self.confidence_floor,
            market_health_score=0.5,
            volatility_score=0.02,
            liquidation_risk_score=0.1,
            technical_indicators=self._get_default_technical_indicators(),
            analysis_metadata={
                'analysis_timestamp': datetime.now().isoformat(),
                'method': 'conservative_fallback',
                'insufficient_data': True
            }
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get fallback analysis performance metrics."""
        return {
            'total_analyses': self.fallback_analysis_count,
            'avg_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0,
            'config': {
                'min_data_points': self.min_data_points,
                'conservative_mode': self.conservative_mode,
                'risk_aversion': self.risk_aversion,
                'confidence_floor': self.confidence_floor,
                'confidence_ceiling': self.confidence_ceiling
            }
        }

# Convenience functions
def create_fallback_analyzer(config: Optional[Dict[str, Any]] = None) -> UnifiedFallbackAnalyzer:
    """Create a configured fallback analyzer."""
    return UnifiedFallbackAnalyzer(config)

async def perform_fallback_analysis(
    market_data: pd.DataFrame,
    analysis_type: str = "both",
    current_position: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> FallbackAnalysisResult:
    """Perform fallback analysis with convenience function."""
    analyzer = create_fallback_analyzer(config)
    return await analyzer.perform_fallback_analysis(market_data, analysis_type, current_position)