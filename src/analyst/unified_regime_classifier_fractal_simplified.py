# src/analyst/unified_regime_classifier_fractal_simplified.py
"""
Simplified Unified Regime Classifier - Fractal Location-Based Version

This version focuses exclusively on:
1. Distance from S/R levels (how far)
2. Strength of S/R levels (how strong)

Fractal analysis is used to quantify these two aspects across multiple timeframes.
"""

from src.core.decorators import handles_errors
import os
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import CONFIG
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
import logging
import asyncio
from src.core.decorators import (
from src.core.decorators.errors import handles_errors
    validates as validate_data_quality,
    traced as with_tracing_span,
)


class UnifiedRegimeClassifierFractal:
    """
    Simplified Fractal Location Classifier
    
    Focuses on two key aspects:
    1. Distance from nearest S/R levels (normalized by ATR or percentage)
    2. Strength of those S/R levels (based on touches, volume, multi-timeframe confirmation)
    
    The fractal analysis aggregates S/R information across timeframes to provide
    robust distance and strength metrics.
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
        
        # Fractal timeframes for multi-timeframe S/R analysis
        self.fractal_timeframes = self.config.get("fractal_timeframes", [
            {"name": "1m", "periods": 60, "weight": 0.1},
            {"name": "5m", "periods": 12, "weight": 0.15},
            {"name": "15m", "periods": 4, "weight": 0.2},
            {"name": "1h", "periods": 1, "weight": 0.25},
            {"name": "4h", "periods": 0.25, "weight": 0.2},
            {"name": "1d", "periods": 0.042, "weight": 0.1},
        ])
        
        # Distance and strength parameters
        self.distance_normalization = self.config.get("distance_normalization", "percentage")  # "percentage" or "atr"
        self.min_strength_threshold = self.config.get("min_strength_threshold", 0.3)
        self.max_relevant_distance = self.config.get("max_relevant_distance", 0.05)  # 5% max distance
        
        # S/R Predictor for enhanced level detection
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        self.sr_predictor = None
        
        # Enhanced S/R configuration
        self.sr_config = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.3,
                "max_sr_levels": 10,
                "sr_lookback_periods": 200,
                "volume_weight": 0.6,
                "price_weight": 0.4,
                "use_optimized_params": True,
                
                "strength_calculation": {
                    "enable_enhanced_strength": True,
                    "touch_count_lookback": 100,
                    "bounce_rate_threshold": 0.015,
                    "isolation_distance_threshold": 0.03,
                    "age_decay_factor": 0.98
                },
                
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
            self.logger.info("Initializing Simplified Fractal Location Classifier...")
            
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
            
            self.logger.info("✅ Simplified Fractal Location Classifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier: {e}")
            return False
    
    async def classify_location(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify current price location based on distance and strength of S/R levels.
        
        Returns:
            Dict containing:
            - support_distance: Normalized distance to nearest support (positive = above support)
            - resistance_distance: Normalized distance to nearest resistance (negative = below resistance)
            - support_strength: Strength of nearest support (0-1)
            - resistance_strength: Strength of nearest resistance (0-1)
            - combined_location_score: Overall location score (-1 to 1, negative = near support, positive = near resistance)
            - location_quality: Quality of the location analysis based on S/R clarity
        """
        if features_df.empty or len(features_df) < 200:
            return self._get_default_classification()
        
        try:
            # Get current price and calculate ATR for normalization
            current_price = features_df['close'].iloc[-1]
            atr = self._calculate_atr(features_df)
            
            # Perform fractal S/R analysis
            fractal_sr_data = await self._analyze_fractal_sr_levels(features_df)
            
            # Aggregate S/R levels with strength weighting
            aggregated_levels = self._aggregate_sr_levels(fractal_sr_data, current_price)
            
            # Calculate distances and strengths
            location_metrics = self._calculate_location_metrics(
                current_price, 
                aggregated_levels, 
                atr
            )
            
            # Add metadata
            location_metrics['timestamp'] = datetime.now().isoformat()
            location_metrics['current_price'] = current_price
            location_metrics['atr'] = atr
            
            return location_metrics
            
        except Exception as e:
            self.logger.error(f"Error in location classification: {e}")
            return self._get_default_classification()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for distance normalization."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR as EMA of TR
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1]
        
        return atr
    
    async def _analyze_fractal_sr_levels(self, features_df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze S/R levels across multiple timeframes."""
        fractal_sr_data = {}
        
        for tf_config in self.fractal_timeframes:
            tf_name = tf_config['name']
            periods = int(tf_config['periods'] * len(features_df)) if tf_config['periods'] < 1 else int(tf_config['periods'])
            
            if periods > len(features_df):
                periods = len(features_df)
            
            # Get data window for this timeframe
            tf_data = features_df.iloc[-periods:] if periods > 0 else features_df
            
            # Get S/R levels for this timeframe
            if self.sr_predictor and self.enable_sr_integration:
                sr_levels = await self._get_enhanced_sr_levels(tf_data, tf_name)
            else:
                sr_levels = self._get_basic_sr_levels(tf_data, tf_name)
            
            fractal_sr_data[tf_name] = {
                'support_levels': sr_levels['support'],
                'resistance_levels': sr_levels['resistance'],
                'weight': tf_config['weight']
            }
        
        return fractal_sr_data
    
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
                        'timeframe': timeframe
                    })
            
            # Process resistance levels
            for level in sr_context.get('resistance_levels', []):
                if isinstance(level, dict):
                    resistance_levels.append({
                        'price': level.get('price', 0),
                        'strength': level.get('enhanced_strength', 0.5),
                        'touches': level.get('touches', 0),
                        'timeframe': timeframe
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
        # Calculate pivot points
        pivot_high = data['high'].rolling(20).max()
        pivot_low = data['low'].rolling(20).min()
        
        support_levels = []
        resistance_levels = []
        
        for i in range(20, len(data) - 1):
            # Resistance levels (local highs)
            if data['high'].iloc[i] == pivot_high.iloc[i]:
                resistance_levels.append({
                    'price': data['high'].iloc[i],
                    'strength': 0.5,
                    'touches': 1,
                    'timeframe': timeframe
                })
            
            # Support levels (local lows)
            if data['low'].iloc[i] == pivot_low.iloc[i]:
                support_levels.append({
                    'price': data['low'].iloc[i],
                    'strength': 0.5,
                    'touches': 1,
                    'timeframe': timeframe
                })
        
        return {
            'support': sorted(support_levels, key=lambda x: x['price'], reverse=True),
            'resistance': sorted(resistance_levels, key=lambda x: x['price'])
        }
    
    def _aggregate_sr_levels(self, fractal_sr_data: Dict[str, Dict], current_price: float) -> Dict[str, List[Dict]]:
        """Aggregate S/R levels across timeframes with strength weighting."""
        all_support = []
        all_resistance = []
        
        # Collect all levels with timeframe weighting
        for tf_name, tf_data in fractal_sr_data.items():
            weight = tf_data['weight']
            
            for level in tf_data['support_levels']:
                if level['price'] < current_price:  # Only consider support below current price
                    all_support.append({
                        'price': level['price'],
                        'raw_strength': level['strength'],
                        'weighted_strength': level['strength'] * weight,
                        'touches': level['touches'],
                        'timeframes': [tf_name]
                    })
            
            for level in tf_data['resistance_levels']:
                if level['price'] > current_price:  # Only consider resistance above current price
                    all_resistance.append({
                        'price': level['price'],
                        'raw_strength': level['strength'],
                        'weighted_strength': level['strength'] * weight,
                        'touches': level['touches'],
                        'timeframes': [tf_name]
                    })
        
        # Cluster nearby levels
        support_clustered = self._cluster_levels(all_support, current_price)
        resistance_clustered = self._cluster_levels(all_resistance, current_price)
        
        # Sort by distance from current price
        support_clustered.sort(key=lambda x: current_price - x['price'])
        resistance_clustered.sort(key=lambda x: x['price'] - current_price)
        
        return {
            'support': support_clustered,
            'resistance': resistance_clustered
        }
    
    def _cluster_levels(self, levels: List[Dict], current_price: float, cluster_threshold: float = 0.002) -> List[Dict]:
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
        # Weighted average price based on strength
        total_weight = sum(l['weighted_strength'] for l in cluster)
        if total_weight > 0:
            weighted_price = sum(l['price'] * l['weighted_strength'] for l in cluster) / total_weight
        else:
            weighted_price = np.mean([l['price'] for l in cluster])
        
        # Aggregate properties
        merged_level = {
            'price': weighted_price,
            'strength': sum(l['weighted_strength'] for l in cluster),  # Sum of weighted strengths
            'raw_strength': np.mean([l['raw_strength'] for l in cluster]),
            'touches': sum(l['touches'] for l in cluster),
            'timeframe_count': len(set(sum([l['timeframes'] for l in cluster], []))),
            'cluster_size': len(cluster)
        }
        
        # Normalize strength to 0-1 range (assuming max possible is sum of all timeframe weights)
        max_possible_strength = sum(tf['weight'] for tf in self.fractal_timeframes)
        merged_level['normalized_strength'] = min(1.0, merged_level['strength'] / max_possible_strength)
        
        return merged_level
    
    def _calculate_location_metrics(
        self, 
        current_price: float, 
        aggregated_levels: Dict[str, List[Dict]], 
        atr: float
    ) -> Dict[str, Any]:
        """Calculate distance and strength metrics for current location."""
        
        # Get nearest levels
        nearest_support = aggregated_levels['support'][0] if aggregated_levels['support'] else None
        nearest_resistance = aggregated_levels['resistance'][0] if aggregated_levels['resistance'] else None
        
        # Calculate distances
        if self.distance_normalization == "atr":
            # Normalize by ATR
            support_distance = ((current_price - nearest_support['price']) / atr) if nearest_support else float('inf')
            resistance_distance = ((nearest_resistance['price'] - current_price) / atr) if nearest_resistance else float('inf')
        else:
            # Normalize by percentage
            support_distance = ((current_price - nearest_support['price']) / current_price) if nearest_support else 1.0
            resistance_distance = ((nearest_resistance['price'] - current_price) / current_price) if nearest_resistance else 1.0
        
        # Get strengths
        support_strength = nearest_support['normalized_strength'] if nearest_support else 0.0
        resistance_strength = nearest_resistance['normalized_strength'] if nearest_resistance else 0.0
        
        # Calculate combined location score
        # Negative = closer to support, Positive = closer to resistance
        if support_distance == float('inf') and resistance_distance == float('inf'):
            combined_score = 0.0
        elif support_distance == float('inf'):
            combined_score = 1.0  # Only resistance exists
        elif resistance_distance == float('inf'):
            combined_score = -1.0  # Only support exists
        else:
            # Weight by inverse distance and strength
            support_weight = (1.0 / (support_distance + 0.001)) * support_strength
            resistance_weight = (1.0 / (resistance_distance + 0.001)) * resistance_strength
            total_weight = support_weight + resistance_weight
            
            if total_weight > 0:
                combined_score = (resistance_weight - support_weight) / total_weight
            else:
                combined_score = 0.0
        
        # Calculate location quality based on S/R clarity
        location_quality = self._calculate_location_quality(aggregated_levels, support_strength, resistance_strength)
        
        return {
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'support_strength': support_strength,
            'resistance_strength': resistance_strength,
            'combined_location_score': combined_score,
            'location_quality': location_quality,
            'nearest_support_price': nearest_support['price'] if nearest_support else None,
            'nearest_resistance_price': nearest_resistance['price'] if nearest_resistance else None,
            'support_details': {
                'touches': nearest_support['touches'] if nearest_support else 0,
                'timeframe_count': nearest_support['timeframe_count'] if nearest_support else 0,
                'cluster_size': nearest_support['cluster_size'] if nearest_support else 0
            } if nearest_support else None,
            'resistance_details': {
                'touches': nearest_resistance['touches'] if nearest_resistance else 0,
                'timeframe_count': nearest_resistance['timeframe_count'] if nearest_resistance else 0,
                'cluster_size': nearest_resistance['cluster_size'] if nearest_resistance else 0
            } if nearest_resistance else None
        }
    
    def _calculate_location_quality(
        self, 
        aggregated_levels: Dict[str, List[Dict]], 
        support_strength: float, 
        resistance_strength: float
    ) -> float:
        """
        Calculate quality of location analysis based on:
        - Strength of nearby levels
        - Number of levels identified
        - Clarity of S/R structure
        """
        # Base quality on strength of nearest levels
        strength_quality = (support_strength + resistance_strength) / 2.0
        
        # Bonus for having multiple levels
        support_count = len(aggregated_levels['support'])
        resistance_count = len(aggregated_levels['resistance'])
        level_count_bonus = min(0.3, (support_count + resistance_count) * 0.05)
        
        # Bonus for balanced S/R structure
        balance_bonus = 0.1 if support_count > 0 and resistance_count > 0 else 0.0
        
        quality = min(1.0, strength_quality + level_count_bonus + balance_bonus)
        
        return quality
    
    def _get_default_classification(self) -> Dict[str, Any]:
        """Return default classification when analysis fails."""
        return {
            'support_distance': 1.0,
            'resistance_distance': 1.0,
            'support_strength': 0.0,
            'resistance_strength': 0.0,
            'combined_location_score': 0.0,
            'location_quality': 0.0,
            'nearest_support_price': None,
            'nearest_resistance_price': None,
            'timestamp': datetime.now().isoformat(),
            'error': 'Insufficient data or analysis failed'
        }
    
    def get_location_features(self, classification: Dict[str, Any]) -> pd.Series:
        """
        Convert location classification to features for ML models.
        
        Returns continuous features based on distance and strength.
        """
        location_features = {
            # Distance features (normalized)
            'support_distance': classification.get('support_distance', 1.0),
            'resistance_distance': classification.get('resistance_distance', 1.0),
            
            # Strength features
            'support_strength': classification.get('support_strength', 0.0),
            'resistance_strength': classification.get('resistance_strength', 0.0),
            
            # Combined metrics
            'combined_location_score': classification.get('combined_location_score', 0.0),
            'location_quality': classification.get('location_quality', 0.0),
            
            # Additional detail features
            'support_touches': classification.get('support_details', {}).get('touches', 0) if classification.get('support_details') else 0,
            'resistance_touches': classification.get('resistance_details', {}).get('touches', 0) if classification.get('resistance_details') else 0,
            'support_timeframes': classification.get('support_details', {}).get('timeframe_count', 0) if classification.get('support_details') else 0,
            'resistance_timeframes': classification.get('resistance_details', {}).get('timeframe_count', 0) if classification.get('resistance_details') else 0,
            
            # Relative position features
            'distance_ratio': classification.get('support_distance', 1.0) / (classification.get('resistance_distance', 1.0) + 0.001),
            'strength_ratio': classification.get('support_strength', 0.0) / (classification.get('resistance_strength', 0.0) + 0.001),
        }
        
        return pd.Series(location_features)