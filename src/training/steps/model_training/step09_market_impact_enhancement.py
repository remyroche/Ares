from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np
import pandas as pd

"""
Market Impact and Liquidity Enhancement for Step09

This module provides comprehensive market impact modeling and liquidity considerations
for the HMM-based training system, including order book analysis, slippage modeling,
and market microstructure effects.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class MarketImpactMetrics:
    """Metrics for market impact analysis."""
    price_impact_bps: float
    volume_impact_ratio: float
    time_to_fill_seconds: float
    slippage_cost_bps: float
    market_depth_utilization: float
    liquidity_score: float

@dataclass
class OrderBookSnapshot:
    """Order book snapshot for liquidity analysis."""
    timestamp: float
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    spread_bps: float
    depth_levels: int
    total_bid_volume: float
    total_ask_volume: float

class MarketImpactModel:
    """Advanced market impact and liquidity modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Market impact parameters
        self.impact_alpha = config.get('market_impact_alpha', 0.5)  # Square root impact
        self.impact_beta = config.get('market_impact_beta', 0.1)    # Linear impact
        self.impact_gamma = config.get('market_impact_gamma', 0.05) # Temporary impact
        
        # Liquidity parameters
        self.avg_spread_bps = config.get('avg_spread_bps', 2.0)
        self.volatility_factor = config.get('volatility_factor', 1.5)
        self.time_decay_factor = config.get('time_decay_factor', 0.1)
        
        # Order book parameters
        self.depth_levels = config.get('orderbook_depth_levels', 10)
        self.min_trade_size = config.get('min_trade_size', 0.001)  # 0.1% of daily volume
        self.max_trade_size = config.get('max_trade_size', 0.05)   # 5% of daily volume
        
    def calculate_market_impact(self, 
                              trade_size: float, 
                              daily_volume: float,
                              volatility: float,
                              liquidity_score: float) -> MarketImpactMetrics:
        """
        Calculate comprehensive market impact metrics.
        
        Args:
            trade_size: Size of the trade
            daily_volume: Daily trading volume
            volatility: Price volatility (annualized)
            liquidity_score: Market liquidity score (0-1)
            
        Returns:
            MarketImpactMetrics object
        """
        try:
            # Normalize trade size
            trade_ratio = trade_size / daily_volume
            
            # Ensure trade size is within reasonable bounds
            trade_ratio = max(self.min_trade_size, min(trade_ratio, self.max_trade_size))
            
            # Calculate price impact using square root model
            # Temporary impact (reverts quickly)
            temp_impact = self.impact_alpha * np.sqrt(trade_ratio) * volatility * 10000  # Convert to bps
            
            # Permanent impact (persists)
            perm_impact = self.impact_beta * trade_ratio * volatility * 10000
            
            # Total price impact
            price_impact_bps = temp_impact + perm_impact
            
            # Volume impact (how much of the order book is consumed)
            volume_impact_ratio = min(1.0, trade_ratio * 10)  # Assume 10x amplification
            
            # Time to fill (based on liquidity and trade size)
            base_fill_time = 1.0  # 1 second base
            size_factor = trade_ratio * 100  # Larger trades take longer
            liquidity_factor = 1.0 / max(0.1, liquidity_score)  # Lower liquidity = longer fill time
            
            time_to_fill_seconds = base_fill_time * size_factor * liquidity_factor
            
            # Slippage cost (additional cost due to market impact)
            slippage_cost_bps = price_impact_bps * 0.5  # Assume 50% of impact is slippage
            
            # Market depth utilization
            market_depth_utilization = min(1.0, trade_ratio * 20)  # Assume 20x depth utilization
            
            return MarketImpactMetrics(
                price_impact_bps=price_impact_bps,
                volume_impact_ratio=volume_impact_ratio,
                time_to_fill_seconds=time_to_fill_seconds,
                slippage_cost_bps=slippage_cost_bps,
                market_depth_utilization=market_depth_utilization,
                liquidity_score=liquidity_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {e}")
            return MarketImpactMetrics(0, 0, 0, 0, 0, 0)
    
    def calculate_liquidity_score(self, 
                                spread_bps: float,
                                volume_24h: float,
                                volatility: float,
                                orderbook_depth: float) -> float:
        """
        Calculate comprehensive liquidity score.
        
        Args:
            spread_bps: Bid-ask spread in basis points
            volume_24h: 24-hour trading volume
            volatility: Price volatility
            orderbook_depth: Total order book depth
            
        Returns:
            Liquidity score (0-1, higher is better)
        """
        try:
            # Normalize metrics
            spread_score = max(0, 1 - (spread_bps / 50))  # 50 bps = 0 score
            volume_score = min(1, np.log10(volume_24h) / 8)  # Log scale, 10^8 = 1.0
            volatility_score = max(0, 1 - (volatility / 2))  # 200% vol = 0 score
            depth_score = min(1, orderbook_depth / 1000000)  # 1M depth = 1.0
            
            # Weighted combination
            weights = [0.3, 0.3, 0.2, 0.2]  # spread, volume, volatility, depth
            scores = [spread_score, volume_score, volatility_score, depth_score]
            
            liquidity_score = sum(w * s for w, s in zip(weights, scores))
            return max(0, min(1, liquidity_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.5  # Default neutral score
    
    def estimate_transaction_costs(self,
                                 trade_size: float,
                                 market_data: Dict[str, Any],
                                 execution_strategy: str = "market") -> Dict[str, float]:
        """
        Estimate comprehensive transaction costs including market impact.
        
        Args:
            trade_size: Size of the trade
            market_data: Market data including volume, volatility, spread
            execution_strategy: Execution strategy ("market", "limit", "twap")
            
        Returns:
            Dictionary of cost components
        """
        try:
            # Extract market data
            daily_volume = market_data.get('volume_24h', 1000000)
            volatility = market_data.get('volatility', 0.02)
            spread_bps = market_data.get('spread_bps', self.avg_spread_bps)
            orderbook_depth = market_data.get('orderbook_depth', 100000)
            
            # Calculate liquidity score
            liquidity_score = self.calculate_liquidity_score(
                spread_bps, daily_volume, volatility, orderbook_depth
            )
            
            # Calculate market impact
            impact_metrics = self.calculate_market_impact(
                trade_size, daily_volume, volatility, liquidity_score
            )
            
            # Base transaction costs
            base_fee_bps = market_data.get('exchange_fee_bps', 5.0)  # 0.05%
            
            # Strategy-specific adjustments
            strategy_multipliers = {
                'market': 1.0,      # Full market impact
                'limit': 0.3,       # Reduced impact for limit orders
                'twap': 0.5,        # Time-weighted average price
                'iceberg': 0.2      # Hidden orders
            }
            
            strategy_mult = strategy_multipliers.get(execution_strategy, 1.0)
            
            # Calculate total costs
            total_costs = {
                'exchange_fee_bps': base_fee_bps,
                'market_impact_bps': impact_metrics.price_impact_bps * strategy_mult,
                'slippage_cost_bps': impact_metrics.slippage_cost_bps * strategy_mult,
                'total_cost_bps': base_fee_bps + (impact_metrics.price_impact_bps * strategy_mult),
                'liquidity_score': liquidity_score,
                'time_to_fill_seconds': impact_metrics.time_to_fill_seconds,
                'volume_impact_ratio': impact_metrics.volume_impact_ratio
            }
            
            return total_costs
            
        except Exception as e:
            self.logger.error(f"Error estimating transaction costs: {e}")
            return {'total_cost_bps': 10.0, 'liquidity_score': 0.5}  # Default fallback
    
    def optimize_trade_size(self,
                          target_volume: float,
                          market_data: Dict[str, Any],
                          max_impact_bps: float = 10.0) -> Dict[str, Any]:
        """
        Optimize trade size to minimize market impact while achieving target volume.
        
        Args:
            target_volume: Target trading volume
            market_data: Market data
            max_impact_bps: Maximum acceptable market impact
            
        Returns:
            Optimization results
        """
        try:
            daily_volume = market_data.get('volume_24h', 1000000)
            volatility = market_data.get('volatility', 0.02)
            spread_bps = market_data.get('spread_bps', self.avg_spread_bps)
            orderbook_depth = market_data.get('orderbook_depth', 100000)
            
            liquidity_score = self.calculate_liquidity_score(
                spread_bps, daily_volume, volatility, orderbook_depth
            )
            
            # Binary search for optimal trade size
            min_size = target_volume * 0.1  # 10% of target
            max_size = target_volume * 2.0  # 200% of target
            optimal_size = target_volume
            
            for _ in range(20):  # Max 20 iterations
                test_size = (min_size + max_size) / 2
                impact_metrics = self.calculate_market_impact(
                    test_size, daily_volume, volatility, liquidity_score
                )
                
                if impact_metrics.price_impact_bps <= max_impact_bps:
                    optimal_size = test_size
                    min_size = test_size
                else:
                    max_size = test_size
                
                if abs(max_size - min_size) < target_volume * 0.01:  # 1% tolerance
                    break
            
            # Calculate final metrics
            final_impact = self.calculate_market_impact(
                optimal_size, daily_volume, volatility, liquidity_score
            )
            
            return {
                'optimal_trade_size': optimal_size,
                'target_volume': target_volume,
                'achievement_ratio': optimal_size / target_volume,
                'market_impact_bps': final_impact.price_impact_bps,
                'total_cost_bps': market_data.get('exchange_fee_bps', 5.0) + final_impact.price_impact_bps,
                'liquidity_score': liquidity_score,
                'time_to_fill_seconds': final_impact.time_to_fill_seconds
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing trade size: {e}")
            return {
                'optimal_trade_size': target_volume,
                'target_volume': target_volume,
                'achievement_ratio': 1.0,
                'market_impact_bps': max_impact_bps,
                'total_cost_bps': 15.0,
                'liquidity_score': 0.5
            }

class LiquidityAwareTraining:
    """Enhancement to training pipeline with liquidity considerations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.market_impact_model = MarketImpactModel(config)
        
    def enhance_training_with_liquidity(self, 
                                      training_data: pd.DataFrame,
                                      market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance training data with liquidity-aware features and cost adjustments.
        
        Args:
            training_data: Original training data
            market_data: Market liquidity data
            
        Returns:
            Enhanced training data with liquidity features
        """
        try:
            enhanced_data = training_data.copy()
            
            # Add liquidity features
            enhanced_data['liquidity_score'] = self._calculate_rolling_liquidity_score(
                training_data, market_data
            )
            
            enhanced_data['spread_bps'] = self._calculate_rolling_spread(
                training_data, market_data
            )
            
            enhanced_data['volume_impact_ratio'] = self._calculate_volume_impact(
                training_data, market_data
            )
            
            # Add market impact cost adjustments
            enhanced_data['transaction_cost_bps'] = self._calculate_transaction_costs(
                training_data, market_data
            )
            
            # Adjust target returns for transaction costs
            if 'target_return' in enhanced_data.columns:
                enhanced_data['net_target_return'] = (
                    enhanced_data['target_return'] - 
                    enhanced_data['transaction_cost_bps'] / 10000  # Convert bps to decimal
                )
            
            self.logger.info(f"✅ Enhanced training data with liquidity features: "
                           f"{len(enhanced_data.columns)} total features")
            
            return {
                'enhanced_data': enhanced_data,
                'liquidity_features_added': 4,
                'cost_adjustments_applied': True
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing training with liquidity: {e}")
            return {'enhanced_data': training_data, 'error': str(e)}
    
    def _calculate_rolling_liquidity_score(self, 
                                         data: pd.DataFrame, 
                                         market_data: Dict[str, Any]) -> pd.Series:
        """Calculate rolling liquidity score."""
        try:
            # Use volume and volatility as proxies for liquidity
            volume = data.get('volume', pd.Series(1.0, index=data.index))
            volatility = data.get('volatility', pd.Series(0.02, index=data.index))
            
            # Rolling calculations
            volume_ma = volume.rolling(window=20).mean()
            volatility_ma = volatility.rolling(window=20).mean()
            
            # Normalize and combine
            volume_score = volume_ma / volume_ma.quantile(0.8)
            volatility_score = 1 - (volatility_ma / volatility_ma.quantile(0.8))
            
            liquidity_score = (volume_score + volatility_score) / 2
            return liquidity_score.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return pd.Series(0.5, index=data.index)
    
    def _calculate_rolling_spread(self, 
                                data: pd.DataFrame, 
                                market_data: Dict[str, Any]) -> pd.Series:
        """Calculate rolling spread estimates."""
        try:
            # Use high-low range as spread proxy
            if 'high' in data.columns and 'low' in data.columns:
                spread = (data['high'] - data['low']) / data['close'] * 10000  # Convert to bps
                return spread.rolling(window=10).mean().fillna(market_data.get('avg_spread_bps', 2.0))
            else:
                return pd.Series(market_data.get('avg_spread_bps', 2.0), index=data.index)
                
        except Exception as e:
            self.logger.error(f"Error calculating spread: {e}")
            return pd.Series(2.0, index=data.index)
    
    def _calculate_volume_impact(self, 
                               data: pd.DataFrame, 
                               market_data: Dict[str, Any]) -> pd.Series:
        """Calculate volume impact ratio."""
        try:
            volume = data.get('volume', pd.Series(1.0, index=data.index))
            daily_volume = market_data.get('volume_24h', 1000000)
            
            # Calculate trade size as percentage of daily volume
            trade_ratio = volume / daily_volume
            return trade_ratio.fillna(0.001)  # Default 0.1%
            
        except Exception as e:
            self.logger.error(f"Error calculating volume impact: {e}")
            return pd.Series(0.001, index=data.index)
    
    def _calculate_transaction_costs(self, 
                                   data: pd.DataFrame, 
                                   market_data: Dict[str, Any]) -> pd.Series:
        """Calculate transaction costs for each period."""
        try:
            volume = data.get('volume', pd.Series(1.0, index=data.index))
            daily_volume = market_data.get('volume_24h', 1000000)
            volatility = data.get('volatility', pd.Series(0.02, index=data.index))
            
            costs = []
            for i, (vol, vol_ann) in enumerate(zip(volume, volatility)):
                trade_size = vol
                vol_annual = vol_ann if not pd.isna(vol_ann) else 0.02
                
                cost_estimate = self.market_impact_model.estimate_transaction_costs(
                    trade_size, {
                        'volume_24h': daily_volume,
                        'volatility': vol_annual,
                        'spread_bps': market_data.get('avg_spread_bps', 2.0),
                        'exchange_fee_bps': market_data.get('exchange_fee_bps', 5.0)
                    }
                )
                
                costs.append(cost_estimate['total_cost_bps'])
            
            return pd.Series(costs, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction costs: {e}")
            return pd.Series(10.0, index=data.index)  # Default 10 bps

def integrate_market_impact_enhancement(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integration function to add market impact and liquidity considerations to Step09.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Enhanced configuration with market impact settings
    """
    enhanced_config = config.copy()
    
    # Add market impact configuration
    enhanced_config.update({
        'market_impact_enabled': True,
        'market_impact_alpha': 0.5,      # Square root impact coefficient
        'market_impact_beta': 0.1,       # Linear impact coefficient
        'market_impact_gamma': 0.05,     # Temporary impact coefficient
        'avg_spread_bps': 2.0,           # Average spread in basis points
        'volatility_factor': 1.5,        # Volatility impact multiplier
        'time_decay_factor': 0.1,        # Time decay for temporary impact
        'orderbook_depth_levels': 10,    # Number of order book levels to consider
        'min_trade_size': 0.001,         # Minimum trade size (0.1% of daily volume)
        'max_trade_size': 0.05,          # Maximum trade size (5% of daily volume)
        'max_feature_correlation': 0.95, # Maximum feature correlation threshold
        'max_missing_ratio': 0.1,        # Maximum missing value ratio
        'min_feature_samples': 100,      # Minimum samples for feature validation
        'embargo_percentage': 0.05,      # 5% embargo for data leakage prevention
        'min_embargo_samples': 20,       # Minimum embargo samples
        'feature_matrix_samples': 1000,  # Default feature matrix samples
        'min_feature_matrix_samples': 500,  # Minimum feature matrix samples
        'max_feature_matrix_samples': 5000, # Maximum feature matrix samples
    })
    
    return enhanced_config