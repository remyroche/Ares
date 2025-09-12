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
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Extract SR proximity features.
        
        Args:
            current_price: Current market price
            sr_levels: List of SR levels with price and type information
            
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
                'total_resistance_levels': 0.0
            }
    
    def extract_sr_strength_features(self, sr_levels: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract SR strength features.
        
        Args:
            sr_levels: List of SR levels with strength information
            
        Returns:
            Dictionary of strength features
        """
        try:
            features = {}
            
            if not sr_levels:
                return self._get_default_strength_features()
            
            # Separate support and resistance levels
            support_levels = [l for l in sr_levels if l.get('level_type', '').lower() == 'support']
            resistance_levels = [l for l in sr_levels if l.get('level_type', '').lower() == 'resistance']
            
            # Calculate support strength metrics
            if support_levels:
                support_strengths = [l.get('strength', 0.5) for l in support_levels]
                features['avg_support_strength'] = float(np.mean(support_strengths))
                features['max_support_strength'] = float(np.max(support_strengths))
                features['min_support_strength'] = float(np.min(support_strengths))
            else:
                features['avg_support_strength'] = 0.0
                features['max_support_strength'] = 0.0
                features['min_support_strength'] = 0.0
            
            # Calculate resistance strength metrics
            if resistance_levels:
                resistance_strengths = [l.get('strength', 0.5) for l in resistance_levels]
                features['avg_resistance_strength'] = float(np.mean(resistance_strengths))
                features['max_resistance_strength'] = float(np.max(resistance_strengths))
                features['min_resistance_strength'] = float(np.min(resistance_strengths))
            else:
                features['avg_resistance_strength'] = 0.0
                features['max_resistance_strength'] = 0.0
                features['min_resistance_strength'] = 0.0
            
            # Overall strength metrics
            all_strengths = [l.get('strength', 0.5) for l in sr_levels]
            features['overall_sr_strength'] = float(np.mean(all_strengths))
            features['strength_variance'] = float(np.var(all_strengths))
            features['strength_std'] = float(np.std(all_strengths))
            
            # Strongest and weakest level types
            if all_strengths:
                max_strength_idx = np.argmax(all_strengths)
                min_strength_idx = np.argmin(all_strengths)
                
                strongest_level = sr_levels[max_strength_idx]
                weakest_level = sr_levels[min_strength_idx]
                
                features['strongest_level_type'] = 1.0 if strongest_level.get('level_type', '').lower() == 'support' else 0.0
                features['weakest_level_type'] = 1.0 if weakest_level.get('level_type', '').lower() == 'support' else 0.0
            else:
                features['strongest_level_type'] = 0.5
                features['weakest_level_type'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting SR strength features: {e}")
            return self._get_default_strength_features()
    
    def _get_default_strength_features(self) -> Dict[str, float]:
        """Get default strength features when no SR levels are available."""
        return {
            'avg_support_strength': 0.0,
            'max_support_strength': 0.0,
            'min_support_strength': 0.0,
            'avg_resistance_strength': 0.0,
            'max_resistance_strength': 0.0,
            'min_resistance_strength': 0.0,
            'overall_sr_strength': 0.0,
            'strength_variance': 0.0,
            'strength_std': 0.0,
            'strongest_level_type': 0.5,
            'weakest_level_type': 0.5
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
            strength_features = self.extract_sr_strength_features(sr_levels)
            
            # Combine SR features
            sr_features = {**proximity_features, **strength_features}
            
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
            'sr_total_support_levels', 'sr_total_resistance_levels'
        ]
        
        strength_features = [
            'sr_avg_support_strength', 'sr_max_support_strength', 'sr_min_support_strength',
            'sr_avg_resistance_strength', 'sr_max_resistance_strength', 'sr_min_resistance_strength',
            'sr_overall_sr_strength', 'sr_strength_variance', 'sr_strength_std',
            'sr_strongest_level_type', 'sr_weakest_level_type'
        ]
        
        return proximity_features + strength_features