# src/analyst/unified_regime_classifier_sr_optimized.py
"""
S/R-Focused Unified Regime Classifier with Optimized Relevance Weights

This version includes:
1. All S/R detection from sr_focused version
2. Optimized relevance scoring weights
3. Dynamic weight adjustment based on market conditions
"""

from src.analyst.unified_regime_classifier_sr_focused import UnifiedRegimeClassifierSRFocused
from src.analyst.sr_relevance_optimizer import SRRelevanceOptimizer
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from src.utils.logger import system_logger


class UnifiedRegimeClassifierSROptimized(UnifiedRegimeClassifierSRFocused):
    """
    Enhanced S/R classifier with optimized relevance scoring weights.
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ):
        # Initialize parent class
        super().__init__(config, exchange, symbol)
        
        # Initialize optimizer
        self.relevance_optimizer = SRRelevanceOptimizer(config)
        
        # Optimization settings
        self.enable_weight_optimization = config.get('enable_weight_optimization', True)
        self.optimization_frequency = config.get('optimization_frequency_hours', 24)
        self.min_data_for_optimization = config.get('min_data_for_optimization', 1000)
        
        # Dynamic adjustment settings
        self.enable_dynamic_adjustment = config.get('enable_dynamic_adjustment', True)
        self.market_regime_lookback = config.get('market_regime_lookback', 100)
        
        # Optimization state
        self.last_optimization_time = None
        self.optimized_weights = None
        self.optimization_results = []
        
        # Override default weights with optimized ones if available
        self._load_optimized_weights()
        
    async def initialize(self) -> bool:
        """Initialize with optimization capabilities."""
        success = await super().initialize()
        
        if success and self.enable_weight_optimization:
            self.logger.info("S/R Weight Optimization enabled")
            
            # Schedule periodic optimization
            if hasattr(self, '_start_optimization_scheduler'):
                asyncio.create_task(self._optimization_scheduler())
        
        return success
    
    async def classify_location(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced classification with optimized weights and optional re-optimization.
        """
        # Check if we need to optimize weights
        if self._should_optimize_weights(features_df):
            await self._optimize_weights_async(features_df)
        
        # Get current market regime for dynamic adjustment
        market_regime = self._detect_market_regime(features_df)
        volatility_percentile = self._calculate_volatility_percentile(features_df)
        
        # Get adjusted weights if dynamic adjustment is enabled
        if self.enable_dynamic_adjustment and self.optimized_weights:
            current_weights = self.relevance_optimizer.dynamic_weight_adjustment(
                market_regime, volatility_percentile
            )
            # Update the config temporarily
            original_weights = self.sr_detection_config['relevance_weights'].copy()
            self.sr_detection_config['relevance_weights'] = current_weights
        
        # Perform classification with optimized weights
        result = await super().classify_location(features_df)
        
        # Add optimization info to result
        result['weight_optimization'] = {
            'weights_used': self.sr_detection_config['relevance_weights'].copy(),
            'market_regime': market_regime,
            'volatility_percentile': volatility_percentile,
            'last_optimization': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'optimization_enabled': self.enable_weight_optimization
        }
        
        # Restore original weights if we adjusted them
        if self.enable_dynamic_adjustment and self.optimized_weights:
            self.sr_detection_config['relevance_weights'] = original_weights
        
        return result
    
    def _should_optimize_weights(self, features_df: pd.DataFrame) -> bool:
        """Check if weights should be re-optimized."""
        if not self.enable_weight_optimization:
            return False
        
        if len(features_df) < self.min_data_for_optimization:
            return False
        
        if self.last_optimization_time is None:
            return True
        
        hours_since_optimization = (
            datetime.now() - self.last_optimization_time
        ).total_seconds() / 3600
        
        return hours_since_optimization >= self.optimization_frequency
    
    async def _optimize_weights_async(self, features_df: pd.DataFrame):
        """Asynchronously optimize weights."""
        try:
            self.logger.info("Starting weight optimization...")
            
            # Prepare historical data for optimization
            historical_data, detected_levels, outcomes = await self._prepare_optimization_data(features_df)
            
            # Run optimization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            optimized_weights = await loop.run_in_executor(
                None,
                self.relevance_optimizer.optimize_weights,
                historical_data,
                detected_levels,
                outcomes
            )
            
            # Update weights
            self.optimized_weights = optimized_weights
            self.sr_detection_config['relevance_weights'] = optimized_weights
            self.last_optimization_time = datetime.now()
            
            # Store results
            self.optimization_results.append({
                'timestamp': self.last_optimization_time,
                'weights': optimized_weights,
                'data_size': len(features_df)
            })
            
            # Save optimized weights
            self._save_optimized_weights()
            
            self.logger.info(f"Weight optimization complete: {optimized_weights}")
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")
    
    async def _prepare_optimization_data(self, features_df: pd.DataFrame):
        """Prepare data for optimization."""
        # Detect S/R levels using current weights
        returns_data = self._calculate_returns_data(features_df)
        detected_levels = self._detect_sr_levels_using_returns(features_df, returns_data)
        
        # Add component scores to each level for optimization
        for level in detected_levels:
            level['component_scores'] = self._calculate_component_scores(level, features_df, returns_data)
        
        # Generate outcomes for validation
        outcomes = self._generate_historical_outcomes(features_df, detected_levels)
        
        return features_df, detected_levels, outcomes
    
    def _calculate_component_scores(self, level: Dict, df: pd.DataFrame, returns_data: Dict) -> Dict[str, float]:
        """Calculate individual component scores for a level."""
        scores = {}
        
        # Return magnitude score
        avg_return_mag = level.get('avg_return_magnitude', 0)
        scores['return_magnitude'] = min(1.0, avg_return_mag / 0.05)  # Normalize to 5%
        
        # Touch count score
        touches = level.get('total_touches', 1)
        scores['touch_count'] = min(1.0, touches / 10)  # Normalize to 10 touches
        
        # Recency score
        if 'source_levels' in level:
            indices = []
            for src in level['source_levels']:
                if 'index' in src:
                    indices.append(src['index'])
            if indices:
                most_recent = max(indices)
                age = len(df) - most_recent
                scores['recency'] = max(0, 1 - age / 100)  # Decay over 100 periods
            else:
                scores['recency'] = 0.5
        else:
            scores['recency'] = 0.5
        
        # Volume confirmation score
        volume_ratio = level.get('volume_ratio', 1.0)
        scores['volume_confirmation'] = min(1.0, volume_ratio / 2)  # Normalize to 2x average
        
        # Success rate score
        if 'validation' in level:
            scores['success_rate'] = level['validation']['success_rate']
        else:
            scores['success_rate'] = 0.5
        
        return scores
    
    def _generate_historical_outcomes(self, df: pd.DataFrame, detected_levels: List[Dict]) -> pd.DataFrame:
        """Generate historical outcomes for optimization."""
        outcomes = []
        
        for i in range(100, len(df) - 10):  # Need future data for outcomes
            current_price = df['close'].iloc[i]
            
            # Find if price approached any S/R level
            for level in detected_levels:
                level_price = level['price']
                tolerance = 0.002  # 0.2%
                
                if abs(current_price - level_price) / level_price <= tolerance:
                    # Check outcome
                    future_prices = df['close'].iloc[i:i+10]
                    
                    if level['type'] == 'support':
                        bounced = future_prices.min() >= level_price * 0.995
                        max_gain = (future_prices.max() - current_price) / current_price
                    else:
                        broke_through = future_prices.max() >= level_price * 1.005
                        max_gain = (current_price - future_prices.min()) / current_price
                    
                    outcomes.append({
                        'timestamp': df.index[i],
                        'level_price': level_price,
                        'approach_price': current_price,
                        'level_type': level['type'],
                        'bounced': bounced if level['type'] == 'support' else not broke_through,
                        'broke_through': not bounced if level['type'] == 'support' else broke_through,
                        'max_gain': max_gain
                    })
        
        return pd.DataFrame(outcomes)
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime for dynamic weight adjustment."""
        if len(df) < self.market_regime_lookback:
            return 'unknown'
        
        recent_data = df.iloc[-self.market_regime_lookback:]
        returns = recent_data['close'].pct_change().dropna()
        
        # Calculate regime indicators
        trend_strength = abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # High/Low analysis
        high_low_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
        
        # Classify regime
        if trend_strength > 0.5:
            return 'trending'
        elif volatility > 0.3:  # 30% annualized volatility
            return 'volatile'
        elif high_low_range < 0.05:  # 5% range
            return 'ranging'
        else:
            return 'normal'
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current volatility percentile."""
        if len(df) < 252:  # Need 1 year of data
            return 0.5
        
        returns = df['close'].pct_change().dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Calculate percentile
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol)
        
        return percentile
    
    def _save_optimized_weights(self):
        """Save optimized weights to file."""
        import json
        import os
        
        weights_file = os.path.join(
            self.config.get('model_dir', 'models'),
            f'sr_weights_{self.exchange}_{self.symbol}.json'
        )
        
        try:
            os.makedirs(os.path.dirname(weights_file), exist_ok=True)
            
            data = {
                'weights': self.optimized_weights,
                'timestamp': self.last_optimization_time.isoformat(),
                'exchange': self.exchange,
                'symbol': self.symbol,
                'optimization_history': self.optimization_results[-10:]  # Keep last 10
            }
            
            with open(weights_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved optimized weights to {weights_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save weights: {e}")
    
    def _load_optimized_weights(self):
        """Load previously optimized weights if available."""
        import json
        import os
        
        weights_file = os.path.join(
            self.config.get('model_dir', 'models'),
            f'sr_weights_{self.exchange}_{self.symbol}.json'
        )
        
        try:
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                
                self.optimized_weights = data['weights']
                self.sr_detection_config['relevance_weights'] = self.optimized_weights
                self.last_optimization_time = datetime.fromisoformat(data['timestamp'])
                
                self.logger.info(f"Loaded optimized weights from {weights_file}")
                self.logger.info(f"Weights: {self.optimized_weights}")
                
        except Exception as e:
            self.logger.warning(f"Could not load optimized weights: {e}")
    
    async def _optimization_scheduler(self):
        """Background task to periodically optimize weights."""
        while True:
            try:
                await asyncio.sleep(self.optimization_frequency * 3600)  # Convert hours to seconds
                
                # Get latest data
                # This would need to be implemented based on your data source
                # latest_data = await self.get_latest_market_data()
                
                # For now, we'll skip the scheduled optimization
                self.logger.info("Scheduled optimization check...")
                
            except Exception as e:
                self.logger.error(f"Optimization scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed report on weight optimization."""
        report = self.relevance_optimizer.get_optimization_report()
        
        # Add current state
        report['current_weights'] = self.sr_detection_config['relevance_weights']
        report['optimization_enabled'] = self.enable_weight_optimization
        report['dynamic_adjustment_enabled'] = self.enable_dynamic_adjustment
        report['optimization_frequency_hours'] = self.optimization_frequency
        
        return report