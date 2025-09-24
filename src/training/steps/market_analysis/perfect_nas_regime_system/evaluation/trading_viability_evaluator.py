"""
Trading Viability Evaluator

Evaluates the trading viability of detected regimes for practical trading decisions.
Focuses on regimes that are actionable for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from ..core.perfect_nas_config import TradingViabilityConfig

logger = logging.getLogger(__name__)

@dataclass
class TradingViabilityResult:
    """Result from trading viability evaluation."""
    overall_score: float
    duration_viability: float
    volatility_viability: float
    volume_viability: float
    trend_viability: float
    liquidity_viability: float
    execution_viability: float
    risk_reward_ratio: float
    trading_frequency: float
    strategy_applicability: float

class TradingViabilityEvaluator:
    """
    Evaluates trading viability of detected regimes.
    
    Considers factors essential for trading:
    - Regime duration (minimum for actionable trades)
    - Volatility levels (tradable range)
    - Volume characteristics (sufficient liquidity)
    - Trend strength (directional opportunities)
    - Liquidity conditions (execution feasibility)
    - Risk-reward characteristics
    """
    
    def __init__(self, config: TradingViabilityConfig):
        """Initialize trading viability evaluator.
        
        Args:
            config: Trading viability configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("✅ Trading Viability Evaluator initialized")
    
    def evaluate(self, market_data: np.ndarray, regime_predictions: np.ndarray, 
                timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate trading viability of regimes.
        
        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Regime predictions
            timestamps: Optional timestamps
            
        Returns:
            Trading viability scores for each regime
        """
        try:
            self.logger.info("📈 Evaluating trading viability...")
            
            # Calculate individual viability metrics
            duration_scores = self._calculate_duration_viability(regime_predictions, timestamps)
            volatility_scores = self._calculate_volatility_viability(market_data, regime_predictions)
            volume_scores = self._calculate_volume_viability(market_data, regime_predictions)
            trend_scores = self._calculate_trend_viability(market_data, regime_predictions)
            liquidity_scores = self._calculate_liquidity_viability(market_data, regime_predictions)
            execution_scores = self._calculate_execution_viability(market_data, regime_predictions)
            risk_reward_scores = self._calculate_risk_reward_ratio(market_data, regime_predictions)
            
            # Calculate weighted trading viability
            viability_scores = (
                duration_scores * 0.2 +
                volatility_scores * 0.15 +
                volume_scores * 0.15 +
                trend_scores * 0.15 +
                liquidity_scores * 0.15 +
                execution_scores * 0.1 +
                risk_reward_scores * 0.1
            )
            
            # Apply viability threshold
            viable_regimes = viability_scores >= self.config.viability_threshold
            
            self.logger.info(f"✅ Trading viability evaluation completed")
            self.logger.info(f"   Mean viability score: {np.mean(viability_scores):.3f}")
            self.logger.info(f"   Viable regimes: {np.sum(viable_regimes)}/{len(regime_predictions)}")
            
            return viability_scores
            
        except Exception as e:
            self.logger.error(f"❌ Trading viability evaluation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_duration_viability(self, regime_predictions: np.ndarray, 
                                   timestamps: Optional[np.ndarray]) -> np.ndarray:
        """Calculate duration viability for each regime."""
        try:
            duration_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_duration = np.sum(regime_mask)
                
                # Check if duration meets minimum requirements
                if regime_duration >= self.config.minimum_regime_duration:
                    # Check if duration is within maximum limits
                    if regime_duration <= self.config.maximum_regime_duration:
                        # Optimal duration range
                        duration_score = 1.0
                    else:
                        # Too long - may indicate regime change needed
                        excess_duration = regime_duration - self.config.maximum_regime_duration
                        duration_score = max(0.3, 1.0 - (excess_duration / self.config.maximum_regime_duration))
                else:
                    # Too short for trading
                    duration_score = regime_duration / self.config.minimum_regime_duration
                
                duration_scores[regime_mask] = min(duration_score, 1.0)
            
            return duration_scores
            
        except Exception as e:
            self.logger.warning(f"Duration viability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_volatility_viability(self, market_data: np.ndarray, 
                                      regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volatility viability for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            volatility_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 2:
                    continue
                
                # Calculate volatility
                returns = np.diff(regime_prices) / regime_prices[:-1]
                volatility = np.std(returns)
                
                # Check if volatility is within tradable range
                if volatility >= self.config.volatility_threshold:
                    # Sufficient volatility for trading
                    if volatility <= 0.1:  # Not too high
                        volatility_score = 1.0
                    else:
                        # Too high volatility - risk management needed
                        volatility_score = max(0.3, 1.0 - (volatility - 0.1) * 2)
                else:
                    # Too low volatility - limited trading opportunities
                    volatility_score = volatility / self.config.volatility_threshold
                
                volatility_scores[regime_mask] = min(volatility_score, 1.0)
            
            return volatility_scores
            
        except Exception as e:
            self.logger.warning(f"Volatility viability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_volume_viability(self, market_data: np.ndarray, 
                                 regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volume viability for each regime."""
        try:
            if market_data.shape[1] < 5:
                # No volume data available
                return np.ones(len(regime_predictions)) * 0.5
            
            volumes = market_data[:, 4]
            volume_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_volumes = volumes[regime_mask]
                
                if len(regime_volumes) < 2:
                    continue
                
                # Calculate volume metrics
                volume_mean = np.mean(regime_volumes)
                volume_consistency = 1.0 / (1.0 + np.std(regime_volumes) / (volume_mean + 1e-8))
                
                # Check if volume meets minimum requirements
                if volume_mean >= self.config.volume_threshold:
                    # Sufficient volume for trading
                    volume_score = min(1.0, volume_consistency)
                else:
                    # Insufficient volume
                    volume_score = volume_mean / self.config.volume_threshold
                
                volume_scores[regime_mask] = min(volume_score, 1.0)
            
            return volume_scores
            
        except Exception as e:
            self.logger.warning(f"Volume viability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_trend_viability(self, market_data: np.ndarray, 
                                 regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate trend viability for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            trend_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate trend strength
                price_changes = np.diff(regime_prices)
                
                if len(price_changes) > 1:
                    trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                else:
                    trend_strength = 0.0
                
                # Calculate trend consistency
                positive_changes = np.sum(price_changes > 0)
                negative_changes = np.sum(price_changes < 0)
                total_changes = len(price_changes)
                
                if total_changes > 0:
                    trend_consistency = max(positive_changes, negative_changes) / total_changes
                else:
                    trend_consistency = 0.5
                
                # Check if trend meets minimum strength requirements
                if trend_strength >= self.config.trend_strength_threshold:
                    # Strong trend for trading
                    trend_score = (trend_strength * 0.6 + trend_consistency * 0.4)
                else:
                    # Weak trend - limited trading opportunities
                    trend_score = trend_strength / self.config.trend_strength_threshold
                
                trend_scores[regime_mask] = min(trend_score, 1.0)
            
            return trend_scores
            
        except Exception as e:
            self.logger.warning(f"Trend viability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_liquidity_viability(self, market_data: np.ndarray, 
                                     regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate liquidity viability for each regime."""
        try:
            if market_data.shape[1] < 5:
                # No volume data available
                return np.ones(len(regime_predictions)) * 0.5
            
            volumes = market_data[:, 4]
            liquidity_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_volumes = volumes[regime_mask]
                
                if len(regime_volumes) < 2:
                    continue
                
                # Calculate liquidity metrics
                volume_mean = np.mean(regime_volumes)
                volume_stability = 1.0 / (1.0 + np.std(regime_volumes) / (volume_mean + 1e-8))
                
                # Calculate volume trend (increasing volume is good for liquidity)
                volume_changes = np.diff(regime_volumes)
                volume_trend = np.mean(volume_changes) / (volume_mean + 1e-8)
                
                # Check liquidity threshold
                if volume_mean >= self.config.liquidity_threshold:
                    # Sufficient liquidity
                    liquidity_score = (volume_stability * 0.6 + max(0, volume_trend) * 0.4)
                else:
                    # Insufficient liquidity
                    liquidity_score = volume_mean / self.config.liquidity_threshold
                
                liquidity_scores[regime_mask] = min(liquidity_score, 1.0)
            
            return liquidity_scores
            
        except Exception as e:
            self.logger.warning(f"Liquidity viability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_execution_viability(self, market_data: np.ndarray, 
                                     regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate execution viability for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            high_prices = market_data[:, 1] if market_data.shape[1] > 1 else market_data[:, 0]
            low_prices = market_data[:, 2] if market_data.shape[1] > 2 else market_data[:, 0]
            close_prices = market_data[:, 3]
            
            execution_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_high = high_prices[regime_mask]
                regime_low = low_prices[regime_mask]
                regime_close = close_prices[regime_mask]
                
                if len(regime_close) < 2:
                    continue
                
                # Calculate execution metrics
                # Price range (spread) - smaller is better for execution
                price_ranges = regime_high - regime_low
                avg_price_range = np.mean(price_ranges)
                price_range_consistency = 1.0 / (1.0 + np.std(price_ranges) / (avg_price_range + 1e-8))
                
                # Price volatility (affects slippage)
                price_changes = np.diff(regime_close)
                price_volatility = np.std(price_changes)
                
                # Execution score (lower volatility and range = better execution)
                execution_score = (
                    price_range_consistency * 0.6 +
                    max(0, 1.0 - price_volatility * 10) * 0.4  # Scale volatility
                )
                
                execution_scores[regime_mask] = min(execution_score, 1.0)
            
            return execution_scores
            
        except Exception as e:
            self.logger.warning(f"Execution viability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_risk_reward_ratio(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate risk-reward ratio for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            risk_reward_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate risk-reward metrics
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Expected return
                expected_return = np.mean(returns)
                
                # Risk (volatility)
                risk = np.std(returns)
                
                # Risk-adjusted return (Sharpe-like ratio)
                if risk > 0:
                    risk_adjusted_return = expected_return / risk
                else:
                    risk_adjusted_return = 0.0
                
                # Maximum drawdown (simplified)
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                # Risk-reward score
                risk_reward_score = (
                    max(0, risk_adjusted_return) * 0.6 +
                    max(0, 1.0 + max_drawdown) * 0.4  # Less negative drawdown is better
                )
                
                risk_reward_scores[regime_mask] = min(risk_reward_score, 1.0)
            
            return risk_reward_scores
            
        except Exception as e:
            self.logger.warning(f"Risk-reward ratio calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def get_trading_strategy_recommendations(self, market_data: np.ndarray, 
                                           regime_predictions: np.ndarray,
                                           viability_scores: np.ndarray) -> Dict[str, Any]:
        """Get trading strategy recommendations for each regime."""
        try:
            recommendations = {
                'strategy_recommendations': {},
                'risk_management': {},
                'position_sizing': {},
                'entry_exit_signals': {}
            }
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_viability = np.mean(viability_scores[regime_mask])
                
                # Strategy recommendation based on viability
                if regime_viability >= 0.8:
                    strategy = "High Confidence Trading"
                    position_size = "Full Position"
                    risk_level = "Low Risk"
                elif regime_viability >= 0.6:
                    strategy = "Moderate Trading"
                    position_size = "Reduced Position"
                    risk_level = "Medium Risk"
                elif regime_viability >= 0.4:
                    strategy = "Conservative Trading"
                    position_size = "Small Position"
                    risk_level = "High Risk"
                else:
                    strategy = "Avoid Trading"
                    position_size = "No Position"
                    risk_level = "Very High Risk"
                
                recommendations['strategy_recommendations'][f'regime_{regime}'] = {
                    'strategy': strategy,
                    'position_size': position_size,
                    'risk_level': risk_level,
                    'viability_score': regime_viability,
                    'confidence': min(regime_viability * 1.2, 1.0)
                }
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Trading strategy recommendations failed: {e}")
            return {}
    
    def get_detailed_trading_analysis(self, market_data: np.ndarray, 
                                    regime_predictions: np.ndarray,
                                    timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get detailed trading analysis for regimes."""
        try:
            analysis = {
                'regime_trading_profiles': {},
                'market_conditions': {},
                'trading_opportunities': {},
                'risk_assessment': {}
            }
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_data = market_data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Trading profile for this regime
                profile = {
                    'regime_id': regime,
                    'duration': len(regime_data),
                    'volatility': np.std(regime_data[:, 3]) if regime_data.shape[1] > 3 else 0.0,
                    'volume_characteristics': np.mean(regime_data[:, 4]) if regime_data.shape[1] > 4 else 1.0,
                    'trend_strength': self._calculate_regime_trend_strength(regime_data),
                    'liquidity_score': self._calculate_regime_liquidity(regime_data),
                    'execution_difficulty': self._calculate_execution_difficulty(regime_data)
                }
                
                analysis['regime_trading_profiles'][f'regime_{regime}'] = profile
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Detailed trading analysis failed: {e}")
            return {}
    
    def _calculate_regime_trend_strength(self, regime_data: np.ndarray) -> float:
        """Calculate trend strength for regime data."""
        try:
            if regime_data.shape[1] < 4 or len(regime_data) < 3:
                return 0.0
            
            prices = regime_data[:, 3]
            price_changes = np.diff(prices)
            
            if len(price_changes) > 1:
                trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                return trend_strength if not np.isnan(trend_strength) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_regime_liquidity(self, regime_data: np.ndarray) -> float:
        """Calculate liquidity score for regime data."""
        try:
            if regime_data.shape[1] < 5 or len(regime_data) < 2:
                return 0.5
            
            volumes = regime_data[:, 4]
            volume_mean = np.mean(volumes)
            volume_stability = 1.0 / (1.0 + np.std(volumes) / (volume_mean + 1e-8))
            
            return min(volume_stability, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_execution_difficulty(self, regime_data: np.ndarray) -> float:
        """Calculate execution difficulty for regime data."""
        try:
            if regime_data.shape[1] < 4 or len(regime_data) < 2:
                return 0.5
            
            high_prices = regime_data[:, 1] if regime_data.shape[1] > 1 else regime_data[:, 0]
            low_prices = regime_data[:, 2] if regime_data.shape[1] > 2 else regime_data[:, 0]
            
            price_ranges = high_prices - low_prices
            range_volatility = np.std(price_ranges) / (np.mean(price_ranges) + 1e-8)
            
            # Higher volatility = more difficult execution
            difficulty = min(range_volatility, 1.0)
            return difficulty
            
        except Exception:
            return 0.5