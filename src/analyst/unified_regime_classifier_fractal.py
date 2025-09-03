# src/analyst/unified_regime_classifier_fractal.py
"""
Unified Regime Classifier - Fractal Location-Based Version

This version focuses exclusively on location-based classification with fractal granularity.
Regime classification is handled by HMM models in the training pipeline.
"""

from src.core.decorators import handles_errors
import os
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import CONFIG
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
import logging
import asyncio
from src.core.decorators import (
    validates as comprehensive_data_validation,
    validates as validate_data_quality,
    traced as with_tracing_span,
)


class UnifiedRegimeClassifierFractal:
    """
    Fractal Location-Based Classifier
    
    Focuses exclusively on identifying price location relative to support/resistance levels
    with multiple timeframe (fractal) analysis for enhanced granularity.
    
    Location Types:
    - STRONG_SUPPORT / STRONG_RESISTANCE: Major levels across multiple timeframes
    - SUPPORT_[TIMEFRAME] / RESISTANCE_[TIMEFRAME]: Timeframe-specific levels
    - BREAKOUT_SUPPORT / BREAKOUT_RESISTANCE: Breaking through levels
    - RETEST_SUPPORT / RETEST_RESISTANCE: Retesting broken levels
    - CONSOLIDATION_SUPPORT / CONSOLIDATION_RESISTANCE: Consolidating near levels
    - OPEN_RANGE: No significant levels nearby
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("UnifiedRegimeClassifierFractal")
        self.exchange = exchange
        self.symbol = symbol
        
        # Fractal timeframes for multi-timeframe analysis
        self.fractal_timeframes = self.config.get("fractal_timeframes", [
            {"name": "1m", "periods": 60, "weight": 0.1},     # Micro structure
            {"name": "5m", "periods": 12, "weight": 0.15},    # Short-term
            {"name": "15m", "periods": 4, "weight": 0.2},     # Medium-term
            {"name": "1h", "periods": 1, "weight": 0.25},     # Base timeframe
            {"name": "4h", "periods": 0.25, "weight": 0.2},   # Macro structure
            {"name": "1d", "periods": 0.042, "weight": 0.1},  # Daily structure
        ])
        
        # Location classification parameters
        self.proximity_threshold = self.config.get("proximity_threshold", 0.002)  # 0.2% from level
        self.breakout_threshold = self.config.get("breakout_threshold", 0.005)   # 0.5% beyond level
        self.strong_level_min_timeframes = self.config.get("strong_level_min_timeframes", 3)
        self.min_touches_for_validity = self.config.get("min_touches_for_validity", 2)
        self.volume_confirmation_threshold = self.config.get("volume_confirmation_threshold", 1.5)
        
        # S/R Predictor for enhanced level detection
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        self.sr_predictor = None
        
        # Enhanced S/R configuration
        self.sr_config = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_proximity_threshold": self.proximity_threshold,
                "breakout_confidence_threshold": 0.7,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.4,
                "max_sr_levels": 20,  # More levels for fractal analysis
                "sr_lookback_periods": 200,
                "volume_weight": 0.6,
                "price_weight": 0.4,
                "use_optimized_params": True,
                
                # Enhanced strength calculation
                "strength_calculation": {
                    "enable_enhanced_strength": True,
                    "touch_count_lookback": 100,
                    "bounce_rate_threshold": 0.015,
                    "isolation_distance_threshold": 0.03,
                    "age_decay_factor": 0.98
                },
                
                # DBSCAN clustering for level grouping
                "dbscan_clustering": {
                    "enable_dbscan_clustering": True,
                    "eps": 0.008,
                    "min_samples": 3,
                    "enable_noise_filtering": True
                }
            }
        }
        
        # Model components
        self.scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        
        # Training status
        self.trained = False
        self.last_training_time = None
        
    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid data for location classification"),
            AttributeError: (False, "Missing required attributes"),
        },
        default_return=False,
        context="classifier initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the fractal location classifier."""
        try:
            self.logger.info("Initializing Fractal Location Classifier...")
            
            # Initialize S/R Predictor if enabled
            if self.enable_sr_integration:
                self.sr_predictor = SRBreakoutPredictor(
                    self.global_config,
                    self.exchange,
                    self.symbol
                )
                sr_init_success = await self.sr_predictor.initialize()
                if not sr_init_success:
                    self.logger.warning("Failed to initialize S/R Predictor, will use basic analysis")
                    self.sr_predictor = None
            
            self.logger.info("✅ Fractal Location Classifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier: {e}")
            return False
    
    async def classify_location(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify current price location using fractal analysis.
        
        Returns:
            Dict containing:
            - primary_location: Main location classification
            - fractal_locations: Location per timeframe
            - nearby_levels: List of nearby S/R levels with details
            - location_strength: Confidence in location classification
            - action_bias: Suggested action based on location
        """
        if features_df.empty or len(features_df) < 200:
            return self._get_default_classification()
        
        try:
            # Get current price and recent price action
            current_price = features_df['close'].iloc[-1]
            recent_high = features_df['high'].iloc[-20:].max()
            recent_low = features_df['low'].iloc[-20:].min()
            current_volume = features_df['volume'].iloc[-1]
            avg_volume = features_df['volume'].iloc[-50:].mean()
            
            # Perform fractal analysis across timeframes
            fractal_analysis = await self._analyze_fractal_levels(features_df)
            
            # Get nearby levels from all timeframes
            all_levels = self._consolidate_fractal_levels(fractal_analysis)
            
            # Classify location based on consolidated levels
            location_result = self._classify_price_location(
                current_price, 
                all_levels, 
                recent_high, 
                recent_low,
                current_volume / avg_volume if avg_volume > 0 else 1.0
            )
            
            # Add fractal details
            location_result['fractal_analysis'] = fractal_analysis
            location_result['timestamp'] = datetime.now().isoformat()
            
            return location_result
            
        except Exception as e:
            self.logger.error(f"Error in location classification: {e}")
            return self._get_default_classification()
    
    async def _analyze_fractal_levels(self, features_df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze S/R levels across multiple timeframes."""
        fractal_analysis = {}
        
        for tf_config in self.fractal_timeframes:
            tf_name = tf_config['name']
            periods = int(tf_config['periods'] * len(features_df)) if tf_config['periods'] < 1 else int(tf_config['periods'])
            
            if periods > len(features_df):
                periods = len(features_df)
            
            # Get data window for this timeframe
            tf_data = features_df.iloc[-periods:] if periods > 0 else features_df
            
            # Analyze S/R levels for this timeframe
            if self.sr_predictor and self.enable_sr_integration:
                sr_levels = await self._get_enhanced_sr_levels(tf_data, tf_name)
            else:
                sr_levels = self._get_basic_sr_levels(tf_data, tf_name)
            
            fractal_analysis[tf_name] = {
                'levels': sr_levels,
                'weight': tf_config['weight'],
                'periods_analyzed': len(tf_data)
            }
        
        return fractal_analysis
    
    async def _get_enhanced_sr_levels(self, data: pd.DataFrame, timeframe: str) -> Dict[str, List[Dict]]:
        """Get S/R levels using enhanced S/R predictor."""
        try:
            current_price = data['close'].iloc[-1]
            sr_context = await self.sr_predictor.get_sr_context(data, current_price)
            
            support_levels = []
            resistance_levels = []
            
            # Process support levels
            for level in sr_context.get('support_levels', []):
                if isinstance(level, dict):
                    support_levels.append({
                        'price': level.get('price', 0),
                        'strength': level.get('enhanced_strength', 0.5),
                        'touches': level.get('touches', 0),
                        'timeframe': timeframe,
                        'type': 'support',
                        'distance_pct': abs(current_price - level.get('price', 0)) / current_price
                    })
            
            # Process resistance levels
            for level in sr_context.get('resistance_levels', []):
                if isinstance(level, dict):
                    resistance_levels.append({
                        'price': level.get('price', 0),
                        'strength': level.get('enhanced_strength', 0.5),
                        'touches': level.get('touches', 0),
                        'timeframe': timeframe,
                        'type': 'resistance',
                        'distance_pct': abs(current_price - level.get('price', 0)) / current_price
                    })
            
            return {
                'support': sorted(support_levels, key=lambda x: x['price'], reverse=True),
                'resistance': sorted(resistance_levels, key=lambda x: x['price'])
            }
            
        except Exception as e:
            self.logger.warning(f"Enhanced S/R analysis failed for {timeframe}: {e}")
            return self._get_basic_sr_levels(data, timeframe)
    
    def _get_basic_sr_levels(self, data: pd.DataFrame, timeframe: str) -> Dict[str, List[Dict]]:
        """Get basic S/R levels using simple pivot analysis."""
        current_price = data['close'].iloc[-1]
        
        # Calculate pivot points
        pivot_high = data['high'].rolling(20).max()
        pivot_low = data['low'].rolling(20).min()
        
        # Find local maxima/minima
        support_levels = []
        resistance_levels = []
        
        for i in range(20, len(data) - 1):
            # Resistance levels (local highs)
            if data['high'].iloc[i] == pivot_high.iloc[i]:
                resistance_levels.append({
                    'price': data['high'].iloc[i],
                    'strength': 0.5,
                    'touches': 1,
                    'timeframe': timeframe,
                    'type': 'resistance',
                    'distance_pct': abs(current_price - data['high'].iloc[i]) / current_price
                })
            
            # Support levels (local lows)
            if data['low'].iloc[i] == pivot_low.iloc[i]:
                support_levels.append({
                    'price': data['low'].iloc[i],
                    'strength': 0.5,
                    'touches': 1,
                    'timeframe': timeframe,
                    'type': 'support',
                    'distance_pct': abs(current_price - data['low'].iloc[i]) / current_price
                })
        
        # Remove duplicates and sort
        support_levels = self._remove_duplicate_levels(support_levels)
        resistance_levels = self._remove_duplicate_levels(resistance_levels)
        
        return {
            'support': sorted(support_levels, key=lambda x: x['price'], reverse=True),
            'resistance': sorted(resistance_levels, key=lambda x: x['price'])
        }
    
    def _remove_duplicate_levels(self, levels: List[Dict], tolerance: float = 0.001) -> List[Dict]:
        """Remove duplicate levels within tolerance."""
        if not levels:
            return []
        
        unique_levels = []
        for level in sorted(levels, key=lambda x: x['price']):
            is_duplicate = False
            for unique_level in unique_levels:
                if abs(level['price'] - unique_level['price']) / level['price'] < tolerance:
                    # Merge strength and touches
                    unique_level['strength'] = max(unique_level['strength'], level['strength'])
                    unique_level['touches'] += level['touches']
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_levels.append(level.copy())
        
        return unique_levels
    
    def _consolidate_fractal_levels(self, fractal_analysis: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Consolidate levels from all timeframes with weighting."""
        all_support = []
        all_resistance = []
        
        for tf_name, tf_data in fractal_analysis.items():
            weight = tf_data['weight']
            
            # Add weighted support levels
            for level in tf_data['levels'].get('support', []):
                weighted_level = level.copy()
                weighted_level['weighted_strength'] = level['strength'] * weight
                weighted_level['timeframes'] = [tf_name]
                all_support.append(weighted_level)
            
            # Add weighted resistance levels
            for level in tf_data['levels'].get('resistance', []):
                weighted_level = level.copy()
                weighted_level['weighted_strength'] = level['strength'] * weight
                weighted_level['timeframes'] = [tf_name]
                all_resistance.append(weighted_level)
        
        # Cluster nearby levels
        clustered_support = self._cluster_levels(all_support)
        clustered_resistance = self._cluster_levels(all_resistance)
        
        return {
            'support': sorted(clustered_support, key=lambda x: x['price'], reverse=True),
            'resistance': sorted(clustered_resistance, key=lambda x: x['price'])
        }
    
    def _cluster_levels(self, levels: List[Dict], cluster_threshold: float = 0.002) -> List[Dict]:
        """Cluster nearby levels and combine their properties."""
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if level belongs to current cluster
            cluster_center = np.mean([l['price'] for l in current_cluster])
            if abs(level['price'] - cluster_center) / cluster_center <= cluster_threshold:
                current_cluster.append(level)
            else:
                # Process completed cluster
                clusters.append(self._merge_cluster(current_cluster))
                current_cluster = [level]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(self._merge_cluster(current_cluster))
        
        return clusters
    
    def _merge_cluster(self, cluster: List[Dict]) -> Dict:
        """Merge levels in a cluster into a single level."""
        merged_level = {
            'price': np.mean([l['price'] for l in cluster]),
            'strength': sum([l.get('weighted_strength', l['strength']) for l in cluster]),
            'touches': sum([l['touches'] for l in cluster]),
            'timeframes': list(set(sum([l.get('timeframes', [l['timeframe']]) for l in cluster], []))),
            'type': cluster[0]['type'],
            'cluster_size': len(cluster)
        }
        
        # Mark as strong level if present in multiple timeframes
        merged_level['is_strong'] = len(merged_level['timeframes']) >= self.strong_level_min_timeframes
        
        return merged_level
    
    def _classify_price_location(
        self, 
        current_price: float, 
        all_levels: Dict[str, List[Dict]], 
        recent_high: float,
        recent_low: float,
        volume_ratio: float
    ) -> Dict[str, Any]:
        """Classify price location based on consolidated levels."""
        
        # Find nearest levels
        nearest_support = self._find_nearest_level(current_price, all_levels['support'], 'below')
        nearest_resistance = self._find_nearest_level(current_price, all_levels['resistance'], 'above')
        
        # Calculate distances
        support_distance = abs(current_price - nearest_support['price']) / current_price if nearest_support else float('inf')
        resistance_distance = abs(current_price - nearest_resistance['price']) / current_price if nearest_resistance else float('inf')
        
        # Determine primary location
        location_type = "OPEN_RANGE"
        location_details = {}
        action_bias = "NEUTRAL"
        location_strength = 0.5
        
        # Check for breakout conditions
        if nearest_resistance and current_price > nearest_resistance['price']:
            if (current_price - nearest_resistance['price']) / current_price > self.breakout_threshold:
                if volume_ratio > self.volume_confirmation_threshold:
                    location_type = "BREAKOUT_RESISTANCE"
                    action_bias = "BULLISH"
                    location_strength = min(0.9, 0.5 + nearest_resistance['strength'])
                else:
                    location_type = "FALSE_BREAKOUT_RESISTANCE"
                    action_bias = "BEARISH"
                    location_strength = 0.3
            elif resistance_distance <= self.proximity_threshold:
                location_type = "RETEST_RESISTANCE"
                action_bias = "NEUTRAL_BEARISH"
                location_strength = 0.6
        
        elif nearest_support and current_price < nearest_support['price']:
            if (nearest_support['price'] - current_price) / current_price > self.breakout_threshold:
                if volume_ratio > self.volume_confirmation_threshold:
                    location_type = "BREAKOUT_SUPPORT"
                    action_bias = "BEARISH"
                    location_strength = min(0.9, 0.5 + nearest_support['strength'])
                else:
                    location_type = "FALSE_BREAKOUT_SUPPORT"
                    action_bias = "BULLISH"
                    location_strength = 0.3
            elif support_distance <= self.proximity_threshold:
                location_type = "RETEST_SUPPORT"
                action_bias = "NEUTRAL_BULLISH"
                location_strength = 0.6
        
        # Check for proximity conditions
        elif support_distance <= self.proximity_threshold:
            if nearest_support.get('is_strong', False):
                location_type = "STRONG_SUPPORT"
                location_strength = min(0.9, nearest_support['strength'])
            else:
                timeframe = nearest_support['timeframes'][0] if nearest_support.get('timeframes') else "1h"
                location_type = f"SUPPORT_{timeframe.upper()}"
                location_strength = nearest_support['strength']
            action_bias = "BULLISH"
            
        elif resistance_distance <= self.proximity_threshold:
            if nearest_resistance.get('is_strong', False):
                location_type = "STRONG_RESISTANCE"
                location_strength = min(0.9, nearest_resistance['strength'])
            else:
                timeframe = nearest_resistance['timeframes'][0] if nearest_resistance.get('timeframes') else "1h"
                location_type = f"RESISTANCE_{timeframe.upper()}"
                location_strength = nearest_resistance['strength']
            action_bias = "BEARISH"
        
        # Check for consolidation
        elif support_distance <= self.proximity_threshold * 2 and resistance_distance <= self.proximity_threshold * 2:
            if abs(recent_high - recent_low) / current_price < 0.02:  # 2% range
                location_type = "CONSOLIDATION_RANGE"
                action_bias = "NEUTRAL"
                location_strength = 0.5
        
        # Build location details
        location_details = {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': support_distance * 100,
            'resistance_distance_pct': resistance_distance * 100,
            'volume_confirmation': volume_ratio > self.volume_confirmation_threshold,
            'price_range_pct': (recent_high - recent_low) / current_price * 100
        }
        
        # Get nearby levels for context
        nearby_levels = self._get_nearby_levels(current_price, all_levels, radius=0.02)  # 2% radius
        
        return {
            'primary_location': location_type,
            'location_strength': location_strength,
            'action_bias': action_bias,
            'location_details': location_details,
            'nearby_levels': nearby_levels,
            'fractal_locations': {}  # Will be added by calling method
        }
    
    def _find_nearest_level(self, price: float, levels: List[Dict], direction: str) -> Optional[Dict]:
        """Find nearest level above or below price."""
        if not levels:
            return None
        
        if direction == 'below':
            valid_levels = [l for l in levels if l['price'] <= price]
            return max(valid_levels, key=lambda x: x['price']) if valid_levels else None
        else:  # above
            valid_levels = [l for l in levels if l['price'] >= price]
            return min(valid_levels, key=lambda x: x['price']) if valid_levels else None
    
    def _get_nearby_levels(self, price: float, all_levels: Dict[str, List[Dict]], radius: float = 0.02) -> List[Dict]:
        """Get all levels within radius of current price."""
        nearby = []
        
        for level_type, levels in all_levels.items():
            for level in levels:
                distance = abs(level['price'] - price) / price
                if distance <= radius:
                    level_info = level.copy()
                    level_info['distance_pct'] = distance * 100
                    level_info['relative_position'] = 'above' if level['price'] > price else 'below'
                    nearby.append(level_info)
        
        # Sort by distance
        return sorted(nearby, key=lambda x: x['distance_pct'])
    
    def _get_default_classification(self) -> Dict[str, Any]:
        """Return default classification when analysis fails."""
        return {
            'primary_location': 'OPEN_RANGE',
            'location_strength': 0.5,
            'action_bias': 'NEUTRAL',
            'location_details': {},
            'nearby_levels': [],
            'fractal_locations': {},
            'fractal_analysis': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Insufficient data or analysis failed'
        }
    
    def get_location_features(self, classification: Dict[str, Any]) -> pd.Series:
        """
        Convert location classification to features for ML models.
        
        Returns a Series with binary features for each location type.
        """
        location_features = {}
        
        # All possible location types
        location_types = [
            'STRONG_SUPPORT', 'STRONG_RESISTANCE',
            'SUPPORT_1M', 'SUPPORT_5M', 'SUPPORT_15M', 'SUPPORT_1H', 'SUPPORT_4H', 'SUPPORT_1D',
            'RESISTANCE_1M', 'RESISTANCE_5M', 'RESISTANCE_15M', 'RESISTANCE_1H', 'RESISTANCE_4H', 'RESISTANCE_1D',
            'BREAKOUT_SUPPORT', 'BREAKOUT_RESISTANCE',
            'FALSE_BREAKOUT_SUPPORT', 'FALSE_BREAKOUT_RESISTANCE',
            'RETEST_SUPPORT', 'RETEST_RESISTANCE',
            'CONSOLIDATION_RANGE', 'OPEN_RANGE'
        ]
        
        # Initialize all features to 0
        for loc_type in location_types:
            location_features[f'location_{loc_type.lower()}'] = 0
        
        # Set the current location to 1
        primary_location = classification.get('primary_location', 'OPEN_RANGE')
        location_features[f'location_{primary_location.lower()}'] = 1
        
        # Add continuous features
        location_features['location_strength'] = classification.get('location_strength', 0.5)
        
        # Add distance features
        details = classification.get('location_details', {})
        location_features['support_distance_pct'] = details.get('support_distance_pct', 100.0)
        location_features['resistance_distance_pct'] = details.get('resistance_distance_pct', 100.0)
        location_features['volume_confirmed'] = int(details.get('volume_confirmation', False))
        location_features['price_range_pct'] = details.get('price_range_pct', 0.0)
        
        # Add nearby levels count
        nearby_levels = classification.get('nearby_levels', [])
        location_features['nearby_support_count'] = len([l for l in nearby_levels if l.get('type') == 'support'])
        location_features['nearby_resistance_count'] = len([l for l in nearby_levels if l.get('type') == 'resistance'])
        location_features['nearby_strong_levels'] = len([l for l in nearby_levels if l.get('is_strong', False)])
        
        return pd.Series(location_features)