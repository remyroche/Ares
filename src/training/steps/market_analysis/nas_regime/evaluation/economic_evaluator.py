"""
Economic Significance Evaluator

Evaluates the economic significance of detected regimes for trading and investment decisions.
Focuses on economically relevant market states that impact trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from ..core.perfect_nas_config import EconomicEvaluationConfig

logger = logging.getLogger(__name__)

@dataclass
class EconomicSignificanceResult:
    """Result from economic significance evaluation."""
    overall_score: float
    price_impact_score: float
    volume_significance_score: float
    volatility_impact_score: float
    trend_consistency_score: float
    market_efficiency_score: float
    economic_indicators_score: float
    regime_economic_value: float
    trading_opportunity_score: float
    risk_adjustment_score: float

class EconomicSignificanceEvaluator:
    """
    Evaluates economic significance of detected regimes.
    
    Considers multiple economic factors:
    - Price impact and movement significance
    - Volume and liquidity implications
    - Volatility and risk characteristics
    - Trend consistency and persistence
    - Market efficiency indicators
    - Economic indicator correlations
    """
    
    def __init__(self, config: EconomicEvaluationConfig):
        """Initialize economic significance evaluator.
        
        Args:
            config: Economic evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Economic indicators (placeholder - would be loaded from external data)
        self.economic_indicators = self._load_economic_indicators()
        
        self.logger.info("✅ Economic Significance Evaluator initialized")
    
    def _load_economic_indicators(self) -> Dict[str, np.ndarray]:
        """Load economic indicators (placeholder implementation)."""
        # In a real implementation, this would load actual economic data
        # For now, return placeholder data
        return {
            'gdp_growth': np.random.normal(0.02, 0.01, 1000),
            'inflation_rate': np.random.normal(0.03, 0.005, 1000),
            'interest_rate': np.random.normal(0.05, 0.01, 1000),
            'unemployment_rate': np.random.normal(0.05, 0.01, 1000)
        }
    
    def evaluate(self, market_data: np.ndarray, regime_predictions: np.ndarray, 
                timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate economic significance of regimes.
        
        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Regime predictions
            timestamps: Optional timestamps
            
        Returns:
            Economic significance scores for each regime
        """
        try:
            self.logger.info("💰 Evaluating economic significance...")
            
            # Calculate individual economic metrics
            price_impact_scores = self._calculate_price_impact_significance(market_data, regime_predictions)
            volume_scores = self._calculate_volume_significance(market_data, regime_predictions)
            volatility_scores = self._calculate_volatility_impact(market_data, regime_predictions)
            trend_scores = self._calculate_trend_consistency(market_data, regime_predictions)
            efficiency_scores = self._calculate_market_efficiency(market_data, regime_predictions)
            indicator_scores = self._calculate_economic_indicator_correlation(market_data, regime_predictions, timestamps)
            
            # Calculate weighted economic significance
            economic_scores = (
                price_impact_scores * self.config.price_impact_weight +
                volume_scores * self.config.volume_significance_weight +
                volatility_scores * self.config.volatility_impact_weight +
                trend_scores * self.config.trend_consistency_weight +
                efficiency_scores * self.config.market_efficiency_weight +
                indicator_scores * 0.1  # Additional weight for economic indicators
            )
            
            # Apply significance threshold
            significant_regimes = economic_scores >= self.config.significance_threshold
            
            self.logger.info(f"✅ Economic significance evaluation completed")
            self.logger.info(f"   Mean economic score: {np.mean(economic_scores):.3f}")
            self.logger.info(f"   Significant regimes: {np.sum(significant_regimes)}/{len(regime_predictions)}")
            
            return economic_scores
            
        except Exception as e:
            self.logger.error(f"❌ Economic significance evaluation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_price_impact_significance(self, market_data: np.ndarray, 
                                          regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate price impact significance for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]  # Close prices
            price_impact_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 2:
                    continue
                
                # Calculate price movement magnitude
                price_changes = np.diff(regime_prices)
                price_magnitude = np.mean(np.abs(price_changes))
                
                # Calculate price volatility
                price_volatility = np.std(price_changes)
                
                # Calculate price trend strength
                if len(price_changes) > 1:
                    trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                else:
                    trend_strength = 0.0
                
                # Combine metrics for price impact significance
                price_impact = (price_magnitude * 0.4 + price_volatility * 0.3 + trend_strength * 0.3)
                
                # Normalize to 0-1 range
                price_impact_scores[regime_mask] = min(price_impact, 1.0)
            
            return price_impact_scores
            
        except Exception as e:
            self.logger.warning(f"Price impact calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_volume_significance(self, market_data: np.ndarray, 
                                     regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volume significance for each regime."""
        try:
            if market_data.shape[1] < 5:
                # No volume data available
                return np.ones(len(regime_predictions)) * 0.5
            
            volumes = market_data[:, 4]  # Volume
            volume_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_volumes = volumes[regime_mask]
                
                if len(regime_volumes) < 2:
                    continue
                
                # Calculate volume significance metrics
                volume_mean = np.mean(regime_volumes)
                volume_std = np.std(regime_volumes)
                volume_consistency = 1.0 / (1.0 + volume_std / (volume_mean + 1e-8))
                
                # Calculate volume trend
                volume_changes = np.diff(regime_volumes)
                volume_trend = np.mean(volume_changes) / (volume_mean + 1e-8)
                
                # Calculate volume relative to market average
                market_volume_avg = np.mean(volumes)
                volume_ratio = volume_mean / (market_volume_avg + 1e-8)
                
                # Combine volume significance metrics
                volume_significance = (
                    volume_consistency * 0.4 +
                    abs(volume_trend) * 0.3 +
                    min(volume_ratio, 2.0) * 0.3  # Cap at 2x market average
                )
                
                # Normalize to 0-1 range
                volume_scores[regime_mask] = min(volume_significance, 1.0)
            
            return volume_scores
            
        except Exception as e:
            self.logger.warning(f"Volume significance calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_volatility_impact(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volatility impact for each regime."""
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
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate returns
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Calculate volatility metrics
                volatility = np.std(returns)
                volatility_persistence = self._calculate_volatility_persistence(returns)
                volatility_clustering = self._calculate_volatility_clustering(returns)
                
                # Calculate volatility impact
                volatility_impact = (
                    volatility * 0.4 +
                    volatility_persistence * 0.3 +
                    volatility_clustering * 0.3
                )
                
                # Normalize to 0-1 range
                volatility_scores[regime_mask] = min(volatility_impact, 1.0)
            
            return volatility_scores
            
        except Exception as e:
            self.logger.warning(f"Volatility impact calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_volatility_persistence(self, returns: np.ndarray) -> float:
        """Calculate volatility persistence using GARCH-like approach."""
        try:
            if len(returns) < 5:
                return 0.0
            
            # Calculate rolling volatility
            window_size = min(5, len(returns) // 2)
            rolling_vol = []
            
            for i in range(window_size, len(returns)):
                vol = np.std(returns[i-window_size:i])
                rolling_vol.append(vol)
            
            if len(rolling_vol) < 2:
                return 0.0
            
            # Calculate autocorrelation of volatility
            vol_autocorr = np.corrcoef(rolling_vol[:-1], rolling_vol[1:])[0, 1]
            return abs(vol_autocorr) if not np.isnan(vol_autocorr) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering."""
        try:
            if len(returns) < 5:
                return 0.0
            
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # Calculate autocorrelation of squared returns
            if len(squared_returns) > 1:
                autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                return abs(autocorr) if not np.isnan(autocorr) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_trend_consistency(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate trend consistency for each regime."""
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
                
                # Calculate trend metrics
                price_changes = np.diff(regime_prices)
                
                # Trend direction consistency
                positive_changes = np.sum(price_changes > 0)
                negative_changes = np.sum(price_changes < 0)
                total_changes = len(price_changes)
                
                if total_changes > 0:
                    trend_consistency = max(positive_changes, negative_changes) / total_changes
                else:
                    trend_consistency = 0.5
                
                # Trend strength
                if len(price_changes) > 1:
                    trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                else:
                    trend_strength = 0.0
                
                # Trend persistence
                trend_persistence = self._calculate_trend_persistence(price_changes)
                
                # Combine trend metrics
                trend_score = (
                    trend_consistency * 0.4 +
                    trend_strength * 0.3 +
                    trend_persistence * 0.3
                )
                
                trend_scores[regime_mask] = min(trend_score, 1.0)
            
            return trend_scores
            
        except Exception as e:
            self.logger.warning(f"Trend consistency calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_trend_persistence(self, price_changes: np.ndarray) -> float:
        """Calculate trend persistence."""
        try:
            if len(price_changes) < 3:
                return 0.0
            
            # Calculate trend direction changes
            direction_changes = 0
            for i in range(1, len(price_changes)):
                if (price_changes[i] > 0) != (price_changes[i-1] > 0):
                    direction_changes += 1
            
            # Persistence is inverse of direction changes
            persistence = 1.0 - (direction_changes / (len(price_changes) - 1))
            return max(0.0, persistence)
            
        except Exception:
            return 0.0
    
    def _calculate_market_efficiency(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate market efficiency indicators for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            efficiency_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 5:
                    continue
                
                # Calculate efficiency metrics
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Random walk test (autocorrelation)
                if len(returns) > 1:
                    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    random_walk_score = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                else:
                    random_walk_score = 0.5
                
                # Variance ratio test (simplified)
                variance_ratio = self._calculate_variance_ratio(returns)
                
                # Price discovery efficiency
                price_discovery = self._calculate_price_discovery_efficiency(regime_prices)
                
                # Combine efficiency metrics
                efficiency_score = (
                    random_walk_score * 0.4 +
                    variance_ratio * 0.3 +
                    price_discovery * 0.3
                )
                
                efficiency_scores[regime_mask] = min(efficiency_score, 1.0)
            
            return efficiency_scores
            
        except Exception as e:
            self.logger.warning(f"Market efficiency calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_variance_ratio(self, returns: np.ndarray) -> float:
        """Calculate variance ratio for market efficiency."""
        try:
            if len(returns) < 4:
                return 0.5
            
            # Calculate variance of returns
            var_1 = np.var(returns)
            
            # Calculate variance of 2-period returns
            returns_2 = returns[:-1] + returns[1:]
            var_2 = np.var(returns_2)
            
            # Variance ratio
            if var_1 > 0:
                variance_ratio = var_2 / (2 * var_1)
                return min(variance_ratio, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_price_discovery_efficiency(self, prices: np.ndarray) -> float:
        """Calculate price discovery efficiency."""
        try:
            if len(prices) < 3:
                return 0.5
            
            # Calculate price adjustment speed
            price_changes = np.diff(prices)
            price_volatility = np.std(price_changes)
            price_mean = np.mean(np.abs(price_changes))
            
            # Efficiency is higher when volatility is reasonable relative to mean
            if price_mean > 0:
                efficiency = 1.0 / (1.0 + price_volatility / price_mean)
            else:
                efficiency = 0.5
            
            return min(efficiency, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_economic_indicator_correlation(self, market_data: np.ndarray, 
                                                regime_predictions: np.ndarray,
                                                timestamps: Optional[np.ndarray]) -> np.ndarray:
        """Calculate correlation with economic indicators."""
        try:
            # This is a placeholder implementation
            # In a real system, you would correlate with actual economic data
            
            indicator_scores = np.zeros(len(regime_predictions))
            
            # Simulate economic indicator correlation
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                # Simulate correlation with economic indicators
                # In practice, this would use actual economic data
                correlation_score = np.random.uniform(0.3, 0.8)
                indicator_scores[regime_mask] = correlation_score
            
            return indicator_scores
            
        except Exception as e:
            self.logger.warning(f"Economic indicator correlation calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def get_detailed_economic_analysis(self, market_data: np.ndarray, 
                                     regime_predictions: np.ndarray,
                                     timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get detailed economic analysis for regimes."""
        try:
            analysis = {
                'regime_economic_profiles': {},
                'market_efficiency_analysis': {},
                'economic_indicators_impact': {},
                'trading_opportunities': {},
                'risk_characteristics': {}
            }
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_data = market_data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Economic profile for this regime
                profile = {
                    'regime_id': regime,
                    'duration': len(regime_data),
                    'price_volatility': np.std(regime_data[:, 3]) if regime_data.shape[1] > 3 else 0.0,
                    'volume_characteristics': np.mean(regime_data[:, 4]) if regime_data.shape[1] > 4 else 1.0,
                    'trend_strength': self._calculate_trend_strength(regime_data),
                    'market_efficiency': self._calculate_regime_efficiency(regime_data)
                }
                
                analysis['regime_economic_profiles'][f'regime_{regime}'] = profile
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Detailed economic analysis failed: {e}")
            return {}
    
    def _calculate_trend_strength(self, regime_data: np.ndarray) -> float:
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
    
    def _calculate_regime_efficiency(self, regime_data: np.ndarray) -> float:
        """Calculate market efficiency for regime data."""
        try:
            if regime_data.shape[1] < 4 or len(regime_data) < 3:
                return 0.5
            
            prices = regime_data[:, 3]
            returns = np.diff(prices) / prices[:-1]
            
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                efficiency = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                return min(efficiency, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5