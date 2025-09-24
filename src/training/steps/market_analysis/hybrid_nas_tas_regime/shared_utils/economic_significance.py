"""
Economic Significance Evaluation for Hybrid NAS-TAS Regime Detection.

Provides common economic significance evaluation utilities used by both NAS and TAS regime detection systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EconomicSignificanceConfig:
    """Configuration for economic significance evaluation."""
    price_impact_weight: float = 0.3
    volume_significance_weight: float = 0.2
    volatility_impact_weight: float = 0.2
    trend_consistency_weight: float = 0.15
    market_efficiency_weight: float = 0.1
    economic_indicators_weight: float = 0.05
    significance_threshold: float = 0.5
    min_regime_duration: int = 10
    max_regime_imbalance: float = 0.8


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
    significant_regimes: List[int]
    evaluation_metadata: Dict[str, Any]


class EconomicSignificanceEvaluator:
    """Evaluates economic significance of detected regimes for trading and investment decisions."""
    
    def __init__(self, config: EconomicSignificanceConfig):
        """Initialize the economic significance evaluator.
        
        Args:
            config: Economic significance configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load economic indicators (placeholder implementation)
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
    
    def evaluate(self, market_data: pd.DataFrame, regime_predictions: np.ndarray, 
                timestamps: Optional[np.ndarray] = None) -> EconomicSignificanceResult:
        """Evaluate economic significance of regimes.
        
        Args:
            market_data: Market data DataFrame (OHLCV)
            regime_predictions: Regime predictions array
            timestamps: Optional timestamps array
            
        Returns:
            EconomicSignificanceResult with evaluation results
        """
        try:
            self.logger.info("💰 Evaluating economic significance...")
            start_time = time.time()
            
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
                indicator_scores * self.config.economic_indicators_weight
            )
            
            # Apply significance threshold
            significant_regimes = np.where(economic_scores >= self.config.significance_threshold)[0].tolist()
            
            # Calculate additional metrics
            regime_economic_value = np.mean(economic_scores)
            trading_opportunity_score = self._calculate_trading_opportunity_score(economic_scores, regime_predictions)
            risk_adjustment_score = self._calculate_risk_adjustment_score(market_data, regime_predictions)
            
            processing_time = time.time() - start_time
            
            # Create evaluation metadata
            evaluation_metadata = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'total_regimes': len(np.unique(regime_predictions)),
                'significant_regimes_count': len(significant_regimes),
                'significance_threshold': self.config.significance_threshold,
                'mean_economic_score': float(np.mean(economic_scores)),
                'std_economic_score': float(np.std(economic_scores))
            }
            
            self.logger.info(f"✅ Economic significance evaluation completed")
            self.logger.info(f"   Mean economic score: {np.mean(economic_scores):.3f}")
            self.logger.info(f"   Significant regimes: {len(significant_regimes)}/{len(regime_predictions)}")
            
            return EconomicSignificanceResult(
                overall_score=float(np.mean(economic_scores)),
                price_impact_score=float(np.mean(price_impact_scores)),
                volume_significance_score=float(np.mean(volume_scores)),
                volatility_impact_score=float(np.mean(volatility_scores)),
                trend_consistency_score=float(np.mean(trend_scores)),
                market_efficiency_score=float(np.mean(efficiency_scores)),
                economic_indicators_score=float(np.mean(indicator_scores)),
                regime_economic_value=regime_economic_value,
                trading_opportunity_score=trading_opportunity_score,
                risk_adjustment_score=risk_adjustment_score,
                significant_regimes=significant_regimes,
                evaluation_metadata=evaluation_metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Economic significance evaluation failed: {e}")
            return EconomicSignificanceResult(
                overall_score=0.0,
                price_impact_score=0.0,
                volume_significance_score=0.0,
                volatility_impact_score=0.0,
                trend_consistency_score=0.0,
                market_efficiency_score=0.0,
                economic_indicators_score=0.0,
                regime_economic_value=0.0,
                trading_opportunity_score=0.0,
                risk_adjustment_score=0.0,
                significant_regimes=[],
                evaluation_metadata={'error': str(e)}
            )
    
    def _calculate_price_impact_significance(self, market_data: pd.DataFrame, 
                                          regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate price impact significance for each regime."""
        try:
            if 'close' not in market_data.columns:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data['close'].values
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
    
    def _calculate_volume_significance(self, market_data: pd.DataFrame, 
                                     regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volume significance for each regime."""
        try:
            if 'volume' not in market_data.columns:
                return np.ones(len(regime_predictions)) * 0.5
            
            volumes = market_data['volume'].values
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
    
    def _calculate_volatility_impact(self, market_data: pd.DataFrame, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volatility impact for each regime."""
        try:
            if 'close' not in market_data.columns:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data['close'].values
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
    
    def _calculate_trend_consistency(self, market_data: pd.DataFrame, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate trend consistency for each regime."""
        try:
            if 'close' not in market_data.columns:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data['close'].values
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
    
    def _calculate_market_efficiency(self, market_data: pd.DataFrame, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate market efficiency indicators for each regime."""
        try:
            if 'close' not in market_data.columns:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data['close'].values
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
    
    def _calculate_economic_indicator_correlation(self, market_data: pd.DataFrame, 
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
    
    def _calculate_trading_opportunity_score(self, economic_scores: np.ndarray, 
                                           regime_predictions: np.ndarray) -> float:
        """Calculate trading opportunity score based on economic significance."""
        try:
            # Higher economic significance = better trading opportunities
            mean_score = np.mean(economic_scores)
            score_std = np.std(economic_scores)
            
            # Normalize to 0-1 range
            opportunity_score = min(mean_score / (mean_score + score_std + 1e-8), 1.0)
            
            return float(opportunity_score)
            
        except Exception:
            return 0.5
    
    def _calculate_risk_adjustment_score(self, market_data: pd.DataFrame, 
                                        regime_predictions: np.ndarray) -> float:
        """Calculate risk adjustment score for regime stability."""
        try:
            if 'close' not in market_data.columns:
                return 0.5
            
            close_prices = market_data['close'].values
            
            # Calculate regime stability
            unique_regimes = np.unique(regime_predictions)
            regime_stabilities = []
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) > 1:
                    returns = np.diff(regime_prices) / regime_prices[:-1]
                    stability = 1.0 / (1.0 + np.std(returns))
                    regime_stabilities.append(stability)
            
            if regime_stabilities:
                return float(np.mean(regime_stabilities))
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def get_detailed_economic_analysis(self, market_data: pd.DataFrame, 
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
                    'price_volatility': np.std(regime_data['close']) if 'close' in regime_data.columns else 0.0,
                    'volume_characteristics': np.mean(regime_data['volume']) if 'volume' in regime_data.columns else 1.0,
                    'trend_strength': self._calculate_trend_strength(regime_data),
                    'market_efficiency': self._calculate_regime_efficiency(regime_data)
                }
                
                analysis['regime_economic_profiles'][f'regime_{regime}'] = profile
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Detailed economic analysis failed: {e}")
            return {}
    
    def _calculate_trend_strength(self, regime_data: pd.DataFrame) -> float:
        """Calculate trend strength for regime data."""
        try:
            if 'close' not in regime_data.columns or len(regime_data) < 3:
                return 0.0
            
            prices = regime_data['close'].values
            price_changes = np.diff(prices)
            
            if len(price_changes) > 1:
                trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                return trend_strength if not np.isnan(trend_strength) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_regime_efficiency(self, regime_data: pd.DataFrame) -> float:
        """Calculate market efficiency for regime data."""
        try:
            if 'close' not in regime_data.columns or len(regime_data) < 3:
                return 0.5
            
            prices = regime_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                efficiency = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                return min(efficiency, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5


def create_economic_significance_evaluator(config: EconomicSignificanceConfig) -> EconomicSignificanceEvaluator:
    """Create an economic significance evaluator instance.
    
    Args:
        config: Economic significance configuration
        
    Returns:
        EconomicSignificanceEvaluator instance
    """
    return EconomicSignificanceEvaluator(config)