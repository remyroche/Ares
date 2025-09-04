"""
Limited Market Microstructure Features Extraction
Extracts maximum value from available market data without multi-level order book
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.core.decorators.errors import handles_errors


class LimitedMicrostructureFeatures:
    """
    Limited Market Microstructure Features Extraction.
    Extracts maximum value from available market data without multi-level order book.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Limited Microstructure Features system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('LimitedMicrostructure')
        
        # Configuration
        self.microstructure_config = config.get('microstructure_features', {})
        self.available_data_types = ['bid', 'ask', 'last_price', 'volume', 'high', 'low', 'open', 'close']
        
        # Feature storage
        self.feature_history: deque = deque(maxlen=1000)  # Last 1000 feature sets
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
        
        # Feature calculation parameters
        self.volatility_window = self.microstructure_config.get('volatility_window', 20)
        self.volume_window = self.microstructure_config.get('volume_window', 10)
        self.price_window = self.microstructure_config.get('price_window', 5)
        
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='microstructure features initialization')
    async def initialize(self) -> bool:
        """
        Initialize the Limited Microstructure Features system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Limited Microstructure Features system...")
            
            # Initialize feature statistics
            for data_type in self.available_data_types:
                self.feature_statistics[data_type] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                }
            
            self.logger.info("✅ Limited Microstructure Features system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Limited Microstructure Features initialization failed: {e}")
            return False
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='microstructure feature extraction')
    async def extract_features(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract microstructure features from available market data.
        
        Args:
            market_data: Current market data (bid, ask, last_price, volume, etc.)
            historical_data: Historical data for context (optional)
            
        Returns:
            Dict: Extracted microstructure features
        """
        try:
            # Validate input data
            if not self._validate_market_data(market_data):
                self.logger.error("Invalid market data provided")
                return None
            
            # Extract basic features (always available)
            basic_features = self._extract_basic_features(market_data)
            
            # Extract enhanced features (if data is available)
            enhanced_features = self._extract_enhanced_features(market_data, historical_data)
            
            # Extract time-based features
            time_features = self._extract_time_features(market_data)
            
            # Extract volume-based features
            volume_features = self._extract_volume_features(market_data, historical_data)
            
            # Extract price-based features
            price_features = self._extract_price_features(market_data, historical_data)
            
            # Combine all features
            all_features = {
                **basic_features,
                **enhanced_features,
                **time_features,
                **volume_features,
                **price_features
            }
            
            # Store features for statistics
            self._update_feature_statistics(all_features)
            self.feature_history.append(all_features)
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error extracting microstructure features: {e}")
            return None
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Validate market data has required fields"""
        
        required_fields = ['bid', 'ask', 'last_price']
        return all(field in market_data for field in required_fields)
    
    def _extract_basic_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic features that are always available"""
        
        bid = market_data['bid']
        ask = market_data['ask']
        last_price = market_data['last_price']
        
        # Calculate basic spread features
        spread = ask - bid
        mid_price = (bid + ask) / 2
        spread_percentage = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        # Price position relative to spread
        if spread > 0:
            price_position = (last_price - bid) / spread
        else:
            price_position = 0.5  # Middle if no spread
        
        return {
            'bid': bid,
            'ask': ask,
            'last_price': last_price,
            'spread': spread,
            'mid_price': mid_price,
            'spread_percentage': spread_percentage,
            'price_position': price_position,
            'spread_tightness': 1.0 / (1.0 + spread_percentage)  # Higher = tighter spread
        }
    
    def _extract_enhanced_features(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Extract enhanced features if additional data is available"""
        
        enhanced_features = {}
        
        # Volume imbalance (if bid/ask volumes available)
        if 'bid_volume' in market_data and 'ask_volume' in market_data:
            bid_volume = market_data['bid_volume']
            ask_volume = market_data['ask_volume']
            
            if bid_volume + ask_volume > 0:
                volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            else:
                volume_imbalance = 0
            
            enhanced_features.update({
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_imbalance': volume_imbalance,
                'total_volume': bid_volume + ask_volume
            })
        
        # VWAP deviation (if VWAP available)
        if 'vwap' in market_data:
            vwap = market_data['vwap']
            last_price = market_data['last_price']
            
            if vwap > 0:
                vwap_deviation = ((last_price - vwap) / vwap) * 100
            else:
                vwap_deviation = 0
            
            enhanced_features.update({
                'vwap': vwap,
                'vwap_deviation': vwap_deviation,
                'price_vs_vwap': last_price / vwap if vwap > 0 else 1.0
            })
        
        # High/Low features (if available)
        if 'high' in market_data and 'low' in market_data:
            high = market_data['high']
            low = market_data['low']
            last_price = market_data['last_price']
            
            if high - low > 0:
                price_range = high - low
                price_position_in_range = (last_price - low) / (high - low)
            else:
                price_range = 0
                price_position_in_range = 0.5
            
            enhanced_features.update({
                'high': high,
                'low': low,
                'price_range': price_range,
                'price_position_in_range': price_position_in_range,
                'range_percentage': (price_range / last_price) * 100 if last_price > 0 else 0
            })
        
        return enhanced_features
    
    def _extract_time_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time-based features"""
        
        current_time = datetime.now()
        
        # Time since last trade (if available)
        if 'last_trade_time' in market_data:
            last_trade_time = market_data['last_trade_time']
            if isinstance(last_trade_time, str):
                last_trade_time = datetime.fromisoformat(last_trade_time)
            
            time_since_last_trade = (current_time - last_trade_time).total_seconds()
        else:
            time_since_last_trade = 0
        
        # Time-based features
        time_features = {
            'timestamp': current_time,
            'time_since_last_trade': time_since_last_trade,
            'hour_of_day': current_time.hour,
            'day_of_week': current_time.weekday(),
            'is_market_hours': True,  # Always true for 24/7 crypto trading
            'is_weekend': current_time.weekday() >= 5,
            'trading_session': self._get_trading_session(current_time)
        }
        
        return time_features
    
    def _extract_volume_features(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Extract volume-based features"""
        
        volume_features = {}
        
        # Current volume
        if 'volume' in market_data:
            current_volume = market_data['volume']
            volume_features['current_volume'] = current_volume
            
            # Volume velocity (if historical data available)
            if historical_data is not None and 'volume' in historical_data.columns:
                recent_volumes = historical_data['volume'].tail(self.volume_window)
                if len(recent_volumes) > 1:
                    avg_volume = recent_volumes.mean()
                    volume_velocity = current_volume / avg_volume if avg_volume > 0 else 1.0
                    volume_trend = self._calculate_trend(recent_volumes)
                    
                    volume_features.update({
                        'volume_velocity': volume_velocity,
                        'volume_trend': volume_trend,
                        'avg_volume': avg_volume,
                        'volume_anomaly': 1.0 if volume_velocity > 2.0 else 0.0  # High volume anomaly
                    })
        
        return volume_features
    
    def _extract_price_features(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Extract price-based features"""
        
        price_features = {}
        
        # Current price features
        last_price = market_data['last_price']
        price_features['last_price'] = last_price
        
        # Price volatility (if historical data available)
        if historical_data is not None and 'last_price' in historical_data.columns:
            recent_prices = historical_data['last_price'].tail(self.volatility_window)
            if len(recent_prices) > 1:
                price_volatility = recent_prices.std()
                price_trend = self._calculate_trend(recent_prices)
                price_momentum = self._calculate_momentum(recent_prices)
                
                price_features.update({
                    'price_volatility': price_volatility,
                    'price_trend': price_trend,
                    'price_momentum': price_momentum,
                    'volatility_percentage': (price_volatility / last_price) * 100 if last_price > 0 else 0
                })
        
        # Price change features
        if historical_data is not None and 'last_price' in historical_data.columns:
            if len(historical_data) > 0:
                previous_price = historical_data['last_price'].iloc[-1]
                price_change = last_price - previous_price
                price_change_percentage = (price_change / previous_price) * 100 if previous_price > 0 else 0
                
                price_features.update({
                    'price_change': price_change,
                    'price_change_percentage': price_change_percentage,
                    'price_direction': 1 if price_change > 0 else -1 if price_change < 0 else 0
                })
        
        return price_features
    
    def _calculate_trend(self, data: pd.Series) -> float:
        """Calculate trend using linear regression slope"""
        
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = data.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by mean value
        mean_value = np.mean(y)
        normalized_slope = slope / mean_value if mean_value > 0 else 0
        
        return normalized_slope
    
    def _calculate_momentum(self, data: pd.Series) -> float:
        """Calculate momentum as rate of change"""
        
        if len(data) < 2:
            return 0.0
        
        # Calculate momentum as percentage change over the period
        start_value = data.iloc[0]
        end_value = data.iloc[-1]
        
        if start_value > 0:
            momentum = ((end_value - start_value) / start_value) * 100
        else:
            momentum = 0.0
        
        return momentum
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during market hours (24/7 for crypto)"""
        
        # 24/7 trading for crypto markets - no market hours restrictions
        return True
    
    def _get_trading_session(self, timestamp: datetime) -> str:
        """Get trading session based on time of day (24/7 crypto)"""
        
        hour = timestamp.hour
        
        # Define trading sessions for 24/7 crypto markets
        if 0 <= hour < 6:
            return 'asian_overnight'
        elif 6 <= hour < 12:
            return 'asian_morning'
        elif 12 <= hour < 18:
            return 'european_afternoon'
        elif 18 <= hour < 24:
            return 'us_evening'
        else:
            return 'unknown'
    
    def _update_feature_statistics(self, features: Dict[str, Any]) -> None:
        """Update feature statistics for normalization"""
        
        for feature_name, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if feature_name not in self.feature_statistics:
                    self.feature_statistics[feature_name] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min': float('inf'),
                        'max': float('-inf'),
                        'count': 0
                    }
                
                stats = self.feature_statistics[feature_name]
                stats['count'] += 1
                
                # Update min/max
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
                
                # Update mean (running average)
                old_mean = stats['mean']
                stats['mean'] = old_mean + (value - old_mean) / stats['count']
                
                # Update standard deviation (simplified)
                if stats['count'] > 1:
                    variance = ((stats['count'] - 1) * stats['std']**2 + (value - old_mean) * (value - stats['mean'])) / stats['count']
                    stats['std'] = np.sqrt(variance) if variance > 0 else 0
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='feature normalization')
    def normalize_features(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize features using calculated statistics"""
        
        try:
            normalized_features = {}
            
            for feature_name, value in features.items():
                if feature_name in self.feature_statistics:
                    stats = self.feature_statistics[feature_name]
                    
                    if stats['std'] > 0:
                        # Z-score normalization
                        normalized_value = (value - stats['mean']) / stats['std']
                    else:
                        normalized_value = 0.0
                    
                    normalized_features[f"{feature_name}_normalized"] = normalized_value
                else:
                    normalized_features[feature_name] = value
            
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {e}")
            return None
    
    def create_trading_signals(self, features: Dict[str, Any]) -> Dict[str, int]:
        """Create trading signals from microstructure features"""
        
        signals = {}
        
        # Spread-based signals
        if 'spread_percentage' in features:
            spread_pct = features['spread_percentage']
            if spread_pct < 0.01:  # Very tight spread
                signals['tight_spread'] = 1
            elif spread_pct > 0.05:  # Wide spread
                signals['wide_spread'] = 1
            else:
                signals['normal_spread'] = 1
        
        # Volume-based signals
        if 'volume_velocity' in features:
            volume_velocity = features['volume_velocity']
            if volume_velocity > 1.5:
                signals['high_volume'] = 1
            elif volume_velocity < 0.5:
                signals['low_volume'] = 1
            else:
                signals['normal_volume'] = 1
        
        # Price position signals
        if 'price_position_in_range' in features:
            price_pos = features['price_position_in_range']
            if price_pos > 0.8:
                signals['near_high'] = 1
            elif price_pos < 0.2:
                signals['near_low'] = 1
            else:
                signals['mid_range'] = 1
        
        # Volatility signals
        if 'volatility_percentage' in features:
            volatility_pct = features['volatility_percentage']
            if volatility_pct > 1.0:  # High volatility
                signals['high_volatility'] = 1
            elif volatility_pct < 0.1:  # Low volatility
                signals['low_volatility'] = 1
            else:
                signals['normal_volatility'] = 1
        
        # Time-based signals (24/7 trading)
        if 'trading_session' in features:
            session = features['trading_session']
            signals[f'session_{session}'] = 1
        
        # Always market hours for 24/7 crypto
        signals['market_hours'] = 1
        
        return signals
    
    async def get_timeframe_specific_features(
        self,
        market_data: Dict[str, Any],
        timeframe: str,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract timeframe-specific features optimized for high-frequency trading.
        
        Args:
            market_data: Current market data
            timeframe: Trading timeframe ('5m', '15m', '30m', '1h')
            historical_data: Historical data for context
            
        Returns:
            Dict: Timeframe-specific features
        """
        try:
            # Get base features
            base_features = await self.extract_features(market_data, historical_data)
            if not base_features:
                return None
            
            # Add timeframe-specific adjustments
            timeframe_features = base_features.copy()
            
            # Adjust parameters based on timeframe
            if timeframe == '5m':
                # 5m timeframe: Higher sensitivity, faster response
                timeframe_features['timeframe_multiplier'] = 1.5
                timeframe_features['sensitivity_level'] = 'high'
                timeframe_features['response_speed'] = 'fast'
                
                # Adjust volatility window for 5m
                if historical_data is not None and 'last_price' in historical_data.columns:
                    recent_prices = historical_data['last_price'].tail(12)  # 1 hour of 5m data
                    if len(recent_prices) > 1:
                        short_volatility = recent_prices.std()
                        timeframe_features['short_term_volatility'] = short_volatility
                        timeframe_features['volatility_5m'] = short_volatility
                
            elif timeframe == '15m':
                # 15m timeframe: Balanced approach
                timeframe_features['timeframe_multiplier'] = 1.0
                timeframe_features['sensitivity_level'] = 'medium'
                timeframe_features['response_speed'] = 'medium'
                
            elif timeframe == '30m':
                # 30m timeframe: Lower sensitivity, more stable
                timeframe_features['timeframe_multiplier'] = 0.8
                timeframe_features['sensitivity_level'] = 'low'
                timeframe_features['response_speed'] = 'slow'
                
            elif timeframe == '1h':
                # 1h timeframe: Lowest sensitivity, most stable
                timeframe_features['timeframe_multiplier'] = 0.6
                timeframe_features['sensitivity_level'] = 'very_low'
                timeframe_features['response_speed'] = 'very_slow'
            
            # Add timeframe-specific trading signals
            timeframe_signals = self._create_timeframe_specific_signals(timeframe_features, timeframe)
            timeframe_features.update(timeframe_signals)
            
            return timeframe_features
            
        except Exception as e:
            self.logger.error(f"Error extracting timeframe-specific features: {e}")
            return None
    
    def _create_timeframe_specific_signals(
        self,
        features: Dict[str, Any],
        timeframe: str
    ) -> Dict[str, int]:
        """Create timeframe-specific trading signals"""
        
        signals = {}
        
        # Base signals
        base_signals = self.create_trading_signals(features)
        signals.update(base_signals)
        
        # Timeframe-specific signals
        if timeframe == '5m':
            # High-frequency signals for 5m
            if 'spread_percentage' in features:
                if features['spread_percentage'] < 0.005:  # Very tight spread for 5m
                    signals['ultra_tight_spread'] = 1
                elif features['spread_percentage'] > 0.02:  # Wide spread for 5m
                    signals['wide_spread_5m'] = 1
            
            if 'volume_velocity' in features:
                if features['volume_velocity'] > 2.0:  # High volume for 5m
                    signals['high_volume_5m'] = 1
                elif features['volume_velocity'] < 0.3:  # Low volume for 5m
                    signals['low_volume_5m'] = 1
            
            # 5m-specific volatility signals
            if 'volatility_5m' in features:
                if features['volatility_5m'] > 0.001:  # High volatility for 5m
                    signals['high_volatility_5m'] = 1
                elif features['volatility_5m'] < 0.0001:  # Low volatility for 5m
                    signals['low_volatility_5m'] = 1
        
        elif timeframe == '15m':
            # Standard signals for 15m
            signals['timeframe_15m'] = 1
            
        elif timeframe == '30m':
            # Longer-term signals for 30m
            signals['timeframe_30m'] = 1
            
        elif timeframe == '1h':
            # Long-term signals for 1h
            signals['timeframe_1h'] = 1
        
        return signals
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of extracted features"""
        
        return {
            'system_status': 'active',
            'available_data_types': self.available_data_types,
            'supported_timeframes': ['5m', '15m', '30m', '1h'],
            'feature_count': len(self.feature_statistics),
            'history_length': len(self.feature_history),
            'feature_statistics': self.feature_statistics,
            'last_updated': datetime.now()
        }