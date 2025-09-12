"""
SR Feature Integration for Enhanced Feature Engineering

This module integrates SR-specific features into the existing feature engineering pipeline,
focusing only on SR proximity and strength features to avoid redundancy with existing
price, volume, momentum, and technical indicator features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from src.utils.logger import system_logger

class SRFeatureIntegration:
    """
    Integrates SR-specific features into existing feature engineering pipeline.
    Focuses only on SR proximity and strength features to avoid redundancy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SR feature integration."""
        self.config = config or {}
        self.logger = system_logger.getChild('SRFeatureIntegration')
        
        # SR feature configuration
        self.sr_config = self.config.get('sr_features', {
            'proximity_threshold': 0.05,  # 5% threshold for proximity calculations
            'strength_weights': {
                'touch_count': 0.4,
                'volume_confirmation': 0.3,
                'time_decay': 0.2,
                'confluence': 0.1
            }
        })
    
    def extract_sr_proximity_features(
        self, 
        current_price: float, 
        sr_levels: List[Dict[str, Any]],
        previous_balance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract SR proximity features.
        
        Args:
            current_price: Current market price
            sr_levels: List of SR levels with price and type information
            previous_balance: Previous SR balance for delta calculation
            
        Returns:
            Dictionary of proximity features
        """
        try:
            features = {}
            
            # Separate support and resistance levels
            support_levels = [l for l in sr_levels if l.get('level_type', '').lower() == 'support']
            resistance_levels = [l for l in sr_levels if l.get('level_type', '').lower() == 'resistance']
            
            # Calculate proximity to nearest support
            if support_levels:
                support_prices = [l.get('price', 0) for l in support_levels]
                # Find support levels below current price
                valid_supports = [p for p in support_prices if p < current_price]
                if valid_supports:
                    nearest_support = max(valid_supports)  # Highest support below price
                    support_distance = (current_price - nearest_support) / current_price
                    features['support_proximity'] = min(support_distance, 1.0)
                    features['nearest_support_strength'] = float(
                        next(l.get('strength', 0.5) for l in support_levels 
                             if l.get('price') == nearest_support)
                    )
                else:
                    features['support_proximity'] = 1.0
                    features['nearest_support_strength'] = 0.0
            else:
                features['support_proximity'] = 1.0
                features['nearest_support_strength'] = 0.0
            
            # Calculate proximity to nearest resistance
            if resistance_levels:
                resistance_prices = [l.get('price', 0) for l in resistance_levels]
                # Find resistance levels above current price
                valid_resistances = [p for p in resistance_prices if p > current_price]
                if valid_resistances:
                    nearest_resistance = min(valid_resistances)  # Lowest resistance above price
                    resistance_distance = (nearest_resistance - current_price) / current_price
                    features['resistance_proximity'] = min(resistance_distance, 1.0)
                    features['nearest_resistance_strength'] = float(
                        next(l.get('strength', 0.5) for l in resistance_levels 
                             if l.get('price') == nearest_resistance)
                    )
                else:
                    features['resistance_proximity'] = 1.0
                    features['nearest_resistance_strength'] = 0.0
            else:
                features['resistance_proximity'] = 1.0
                features['nearest_resistance_strength'] = 0.0
            
            # SR balance (ratio of support to total levels)
            total_levels = len(support_levels) + len(resistance_levels)
            features['sr_balance'] = len(support_levels) / total_levels if total_levels > 0 else 0.5
            
            # SR zone width (distance between nearest support and resistance)
            if support_levels and resistance_levels:
                support_prices = [l.get('price', 0) for l in support_levels]
                resistance_prices = [l.get('price', 0) for l in resistance_levels]
                
                # Find closest support and resistance to current price
                supports_below = [p for p in support_prices if p < current_price]
                resistances_above = [p for p in resistance_prices if p > current_price]
                
                if supports_below and resistances_above:
                    closest_support = max(supports_below)
                    closest_resistance = min(resistances_above)
                    zone_width = (closest_resistance - closest_support) / current_price
                    features['sr_zone_width'] = min(zone_width, 1.0)
                else:
                    features['sr_zone_width'] = 1.0
            else:
                features['sr_zone_width'] = 1.0
            
            # Level counts
            features['total_support_levels'] = float(len(support_levels))
            features['total_resistance_levels'] = float(len(resistance_levels))
            
            # NEW: Distance × Strength to nearest SR level
            nearest_distance = min(features['support_proximity'], features['resistance_proximity'])
            nearest_strength = max(features['nearest_support_strength'], features['nearest_resistance_strength'])
            features['nearest_level_distance_strength'] = nearest_distance * nearest_strength
            
            # NEW: SR balance delta (rate of change)
            if previous_balance is not None:
                features['sr_balance_delta'] = features['sr_balance'] - previous_balance
            else:
                features['sr_balance_delta'] = 0.0
            
            # NEW: Price position in SR zone
            if features['sr_zone_width'] > 0:
                zone_start = current_price - (features['support_proximity'] * current_price)
                zone_end = current_price + (features['resistance_proximity'] * current_price)
                if zone_end > zone_start:
                    features['price_position_in_zone'] = (current_price - zone_start) / (zone_end - zone_start)
                else:
                    features['price_position_in_zone'] = 0.5
            else:
                features['price_position_in_zone'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting SR proximity features: {e}")
            # Return default values
            return {
                'support_proximity': 1.0,
                'resistance_proximity': 1.0,
                'nearest_support_strength': 0.0,
                'nearest_resistance_strength': 0.0,
                'sr_balance': 0.5,
                'sr_zone_width': 1.0,
                'total_support_levels': 0.0,
                'total_resistance_levels': 0.0,
                'nearest_level_distance_strength': 0.0,
                'sr_balance_delta': 0.0,
                'price_position_in_zone': 0.5
            }
    
    def extract_sr_strength_features(self, sr_levels: List[Dict[str, Any]], current_price: float) -> Dict[str, float]:
        """
        Extract trading-focused SR strength features.
        
        Args:
            sr_levels: List of SR levels with strength information
            current_price: Current market price for context
            
        Returns:
            Dictionary of trading-relevant strength features
        """
        try:
            features = {}
            
            if not sr_levels:
                return self._get_default_strength_features()
            
            # Separate support and resistance levels
            support_levels = [l for l in sr_levels if l.get('level_type', '').lower() == 'support']
            resistance_levels = [l for l in sr_levels if l.get('level_type', '').lower() == 'resistance']
            
            # Overall strength (most important for trading)
            all_strengths = [l.get('strength', 0.5) for l in sr_levels]
            features['overall_sr_strength'] = float(np.mean(all_strengths))
            
            # Support vs Resistance strength ratio (market bias indicator)
            if support_levels and resistance_levels:
                support_strengths = [l.get('strength', 0.5) for l in support_levels]
                resistance_strengths = [l.get('strength', 0.5) for l in resistance_levels]
                avg_support_strength = np.mean(support_strengths)
                avg_resistance_strength = np.mean(resistance_strengths)
                
                # Ratio: >1 means stronger support (bullish bias), <1 means stronger resistance (bearish bias)
                if avg_resistance_strength > 0:
                    features['support_resistance_strength_ratio'] = float(avg_support_strength / avg_resistance_strength)
                else:
                    features['support_resistance_strength_ratio'] = 1.0
            else:
                features['support_resistance_strength_ratio'] = 1.0
            
            # Nearest level strength ratio (local vs global strength)
            nearest_level = self._find_nearest_level(sr_levels, current_price)
            if nearest_level:
                nearest_strength = nearest_level.get('strength', 0.5)
                if features['overall_sr_strength'] > 0:
                    features['nearest_level_strength_ratio'] = float(nearest_strength / features['overall_sr_strength'])
                else:
                    features['nearest_level_strength_ratio'] = 1.0
            else:
                features['nearest_level_strength_ratio'] = 1.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting SR strength features: {e}")
            return self._get_default_strength_features()
    
    def _find_nearest_level(self, sr_levels: List[Dict[str, Any]], current_price: float) -> Optional[Dict[str, Any]]:
        """Find the nearest SR level to current price."""
        if not sr_levels:
            return None
        
        nearest_level = None
        min_distance = float('inf')
        
        for level in sr_levels:
            level_price = level.get('price', 0)
            distance = abs(current_price - level_price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
        
        return nearest_level
    
    def extract_sr_trading_features(self, sr_levels: List[Dict[str, Any]], current_price: float, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract additional trading-relevant SR features.
        
        Args:
            sr_levels: List of SR levels
            current_price: Current market price
            market_data: Market data for context
            
        Returns:
            Dictionary of trading features
        """
        try:
            features = {}
            
            if not sr_levels or market_data.empty:
                return self._get_default_trading_features()
            
            # SR level density (levels per price range)
            price_range = current_price * 0.1  # 10% price range
            levels_in_range = [l for l in sr_levels if abs(l.get('price', 0) - current_price) <= price_range]
            features['sr_level_density'] = float(len(levels_in_range) / max(1, price_range / current_price))
            
            # Remove ML-decision features - let ML models decide based on raw data
            # Instead, provide raw proximity and strength data for ML to process
            
            # Confluence score (levels clustering around current price)
            confluence_range = current_price * 0.02  # 2% range
            confluence_levels = [l for l in sr_levels if abs(l.get('price', 0) - current_price) <= confluence_range]
            features['sr_confluence_score'] = float(len(confluence_levels) / max(1, len(sr_levels)))
            
            # Time since last touch (if available in market data)
            if 'timestamp' in market_data.columns:
                # Simplified: assume more recent data = fresher levels
                features['sr_time_since_last_touch'] = 0.1  # Placeholder
            else:
                features['sr_time_since_last_touch'] = 0.5
            
            # Trend alignment with strength consideration
            if len(market_data) > 1:
                recent_trend = (market_data['close'].iloc[-1] - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]
                if recent_trend > 0:
                    # Uptrend: check if approaching resistance (with strength weighting)
                    resistances_above = [l for l in sr_levels if l.get('level_type', '').lower() == 'resistance' and l.get('price', 0) > current_price]
                    if resistances_above:
                        # Weight by strength and proximity
                        weighted_alignment = sum(
                            l.get('strength', 0.5) * (1 - abs(l.get('price', 0) - current_price) / current_price)
                            for l in resistances_above
                        ) / len(resistances_above)
                        features['sr_trend_alignment'] = float(weighted_alignment)
                    else:
                        features['sr_trend_alignment'] = 0.0  # No resistance above = perfect alignment
                else:
                    # Downtrend: check if approaching support (with strength weighting)
                    supports_below = [l for l in sr_levels if l.get('level_type', '').lower() == 'support' and l.get('price', 0) < current_price]
                    if supports_below:
                        # Weight by strength and proximity
                        weighted_alignment = sum(
                            l.get('strength', 0.5) * (1 - abs(l.get('price', 0) - current_price) / current_price)
                            for l in supports_below
                        ) / len(supports_below)
                        features['sr_trend_alignment'] = float(weighted_alignment)
                    else:
                        features['sr_trend_alignment'] = 0.0  # No support below = perfect alignment
            else:
                features['sr_trend_alignment'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting SR trading features: {e}")
            return self._get_default_trading_features()
    
    def _get_default_trading_features(self) -> Dict[str, float]:
        """Get default trading features when no data is available."""
        return {
            'sr_level_density': 0.0,
            'sr_confluence_score': 0.0,
            'sr_time_since_last_touch': 0.5,
            'sr_trend_alignment': 0.5
        }
    
    def _get_default_strength_features(self) -> Dict[str, float]:
        """Get default strength features when no SR levels are available."""
        return {
            'overall_sr_strength': 0.0,
            'support_resistance_strength_ratio': 1.0,
            'nearest_level_strength_ratio': 1.0
        }
    
    def integrate_sr_features_into_pipeline(
        self, 
        existing_features: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate SR features into existing feature pipeline.
        
        Args:
            existing_features: Existing feature set from pipeline
            pipeline_state: Pipeline state containing SR data
            
        Returns:
            Enhanced feature set with SR features
        """
        try:
            # Extract SR data from pipeline state
            sr_levels = pipeline_state.get('sr_levels', [])
            market_data = pipeline_state.get('market_data')
            
            if not sr_levels or market_data is None or market_data.empty:
                self.logger.warning("No SR data available for feature integration")
                return existing_features
            
            # Get current price
            current_price = float(market_data['close'].iloc[-1]) if 'close' in market_data.columns else 0.0
            
            # Extract SR-specific features
            proximity_features = self.extract_sr_proximity_features(current_price, sr_levels)
            strength_features = self.extract_sr_strength_features(sr_levels, current_price)
            trading_features = self.extract_sr_trading_features(sr_levels, current_price, market_data)
            
            # Combine SR features
            sr_features = {**proximity_features, **strength_features, **trading_features}
            
            # Add prefix to avoid naming conflicts
            prefixed_sr_features = {
                f"sr_{key}": value for key, value in sr_features.items()
            }
            
            # Merge with existing features
            enhanced_features = {**existing_features, **prefixed_sr_features}
            
            # Log integration results
            self.logger.info(f"SR features integrated: {len(prefixed_sr_features)} features added")
            self.logger.info(f"Total features: {len(existing_features)} -> {len(enhanced_features)}")
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Error integrating SR features: {e}")
            return existing_features
    
    def get_sr_feature_names(self) -> List[str]:
        """Get list of SR feature names for documentation and validation."""
        proximity_features = [
            'sr_support_proximity', 'sr_resistance_proximity',
            'sr_nearest_support_strength', 'sr_nearest_resistance_strength',
            'sr_sr_balance', 'sr_sr_zone_width',
            'sr_total_support_levels', 'sr_total_resistance_levels',
            'sr_nearest_level_distance_strength', 'sr_balance_delta', 'sr_price_position_in_zone'
        ]
        
        strength_features = [
            'sr_overall_sr_strength', 'sr_support_resistance_strength_ratio', 'sr_nearest_level_strength_ratio'
        ]
        
        trading_features = [
            'sr_level_density', 'sr_confluence_score', 'sr_time_since_last_touch', 'sr_trend_alignment'
        ]
        
        return proximity_features + strength_features + trading_features