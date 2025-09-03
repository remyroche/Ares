"""Ensemble S/R Predictor Module.

This module combines multiple S/R detection methods for more robust level identification.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from datetime import datetime

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger


@dataclass
class EnsembleSRLevel:
    """S/R level with ensemble confidence."""
    price: float
    strength: float
    type: str  # 'support' or 'resistance'
    method_votes: Dict[str, float]  # Method name -> confidence
    ensemble_confidence: float
    metadata: Dict[str, Any]


class BaseSRMethod(ABC):
    """Base class for S/R detection methods."""
    
    @abstractmethod
    def identify_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify S/R levels using this method."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this method."""
        pass


class OptimizedSRMethod(BaseSRMethod):
    """Our optimized S/R detection method."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimizedSRMethod")
        
        # Import the strength optimizer
        from src.tactician.sr_strength_optimizer import SRLevelIdentifier
        self.identifier = SRLevelIdentifier(config)
    
    def identify_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify levels using optimized parameters."""
        strong_levels = self.identifier.identify_strong_sr_levels(market_data)
        
        return [{
            'price': level.price,
            'strength': level.strength,
            'type': level.type,
            'metadata': {
                'touches': level.touch_count,
                'age_bars': level.age_bars,
                'bounce_ratio': level.avg_bounce_ratio
            }
        } for level in strong_levels]
    
    def get_method_name(self) -> str:
        return "optimized_strength"


class ClassicalSRMethod(BaseSRMethod):
    """Classical technical analysis S/R detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("ClassicalSRMethod")
        self.lookback = config.get("classical_sr_lookback", 100)
    
    def identify_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify levels using classical pivot points and swing highs/lows."""
        levels = []
        
        # Pivot points
        pivots = self._calculate_pivot_points(market_data)
        levels.extend(pivots)
        
        # Swing highs/lows
        swings = self._find_swing_points(market_data)
        levels.extend(swings)
        
        # Psychological levels (round numbers)
        psych_levels = self._find_psychological_levels(market_data)
        levels.extend(psych_levels)
        
        return levels
    
    def _calculate_pivot_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate daily pivot points."""
        levels = []
        
        # Use last complete day
        high = data['high'].iloc[-24:].max()  # Assuming hourly data
        low = data['low'].iloc[-24:].min()
        close = data['close'].iloc[-1]
        
        # Pivot point
        pp = (high + low + close) / 3
        
        # Support and resistance levels
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        
        levels.extend([
            {'price': pp, 'strength': 0.7, 'type': 'pivot', 'metadata': {'pivot_type': 'pp'}},
            {'price': r1, 'strength': 0.6, 'type': 'resistance', 'metadata': {'pivot_type': 'r1'}},
            {'price': s1, 'strength': 0.6, 'type': 'support', 'metadata': {'pivot_type': 's1'}},
            {'price': r2, 'strength': 0.5, 'type': 'resistance', 'metadata': {'pivot_type': 'r2'}},
            {'price': s2, 'strength': 0.5, 'type': 'support', 'metadata': {'pivot_type': 's2'}}
        ])
        
        return levels
    
    def _find_swing_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find swing highs and lows."""
        levels = []
        window = 10
        
        # Swing highs (resistance)
        for i in range(window, len(data) - window):
            if all(data['high'].iloc[i] >= data['high'].iloc[i-window:i]) and \
               all(data['high'].iloc[i] >= data['high'].iloc[i+1:i+window+1]):
                levels.append({
                    'price': data['high'].iloc[i],
                    'strength': 0.7,
                    'type': 'resistance',
                    'metadata': {'swing_type': 'high', 'index': i}
                })
        
        # Swing lows (support)
        for i in range(window, len(data) - window):
            if all(data['low'].iloc[i] <= data['low'].iloc[i-window:i]) and \
               all(data['low'].iloc[i] <= data['low'].iloc[i+1:i+window+1]):
                levels.append({
                    'price': data['low'].iloc[i],
                    'strength': 0.7,
                    'type': 'support',
                    'metadata': {'swing_type': 'low', 'index': i}
                })
        
        return levels
    
    def _find_psychological_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find psychological round number levels."""
        levels = []
        current_price = data['close'].iloc[-1]
        
        # Determine price scale
        if current_price > 1000:
            round_interval = 100
        elif current_price > 100:
            round_interval = 10
        elif current_price > 10:
            round_interval = 1
        else:
            round_interval = 0.1
        
        # Find nearby round numbers
        base = int(current_price / round_interval) * round_interval
        
        for i in range(-3, 4):
            level_price = base + i * round_interval
            if level_price > 0:
                level_type = 'support' if level_price < current_price else 'resistance'
                levels.append({
                    'price': level_price,
                    'strength': 0.5,
                    'type': level_type,
                    'metadata': {'psychological': True}
                })
        
        return levels
    
    def get_method_name(self) -> str:
        return "classical_ta"


class VolumeProfileSRMethod(BaseSRMethod):
    """Volume profile based S/R detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VolumeProfileSRMethod")
        self.n_bins = config.get("volume_profile_bins", 50)
    
    def identify_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify high volume nodes as S/R levels."""
        
        # Calculate volume profile
        price_range = market_data['high'].max() - market_data['low'].min()
        bin_size = price_range / self.n_bins
        
        # Create price bins
        bins = np.linspace(
            market_data['low'].min(),
            market_data['high'].max(),
            self.n_bins + 1
        )
        
        # Calculate volume at each price level
        volume_profile = np.zeros(self.n_bins)
        
        for i in range(len(market_data)):
            low = market_data['low'].iloc[i]
            high = market_data['high'].iloc[i]
            volume = market_data['volume'].iloc[i]
            
            # Distribute volume across price range
            low_bin = np.searchsorted(bins, low)
            high_bin = np.searchsorted(bins, high)
            
            if low_bin == high_bin:
                volume_profile[min(low_bin, self.n_bins-1)] += volume
            else:
                for bin_idx in range(low_bin, min(high_bin + 1, self.n_bins)):
                    volume_profile[bin_idx] += volume / (high_bin - low_bin + 1)
        
        # Find high volume nodes
        levels = []
        threshold = np.percentile(volume_profile, 70)
        current_price = market_data['close'].iloc[-1]
        
        for i in range(1, self.n_bins - 1):
            if volume_profile[i] > threshold and \
               volume_profile[i] > volume_profile[i-1] and \
               volume_profile[i] > volume_profile[i+1]:
                
                price = (bins[i] + bins[i+1]) / 2
                strength = min(volume_profile[i] / volume_profile.max(), 1.0)
                level_type = 'support' if price < current_price else 'resistance'
                
                levels.append({
                    'price': price,
                    'strength': strength,
                    'type': level_type,
                    'metadata': {
                        'volume_node': True,
                        'volume_ratio': volume_profile[i] / volume_profile.mean()
                    }
                })
        
        return levels
    
    def get_method_name(self) -> str:
        return "volume_profile"


class FractalSRMethod(BaseSRMethod):
    """Fractal-based S/R detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("FractalSRMethod")
    
    def identify_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify S/R using Williams fractals."""
        levels = []
        
        # Find fractal highs (resistance)
        for i in range(2, len(market_data) - 2):
            if (market_data['high'].iloc[i] > market_data['high'].iloc[i-1] and
                market_data['high'].iloc[i] > market_data['high'].iloc[i-2] and
                market_data['high'].iloc[i] > market_data['high'].iloc[i+1] and
                market_data['high'].iloc[i] > market_data['high'].iloc[i+2]):
                
                # Check how many times price respected this level
                touches = self._count_level_touches(
                    market_data, market_data['high'].iloc[i], 'resistance', i
                )
                
                if touches >= 2:
                    levels.append({
                        'price': market_data['high'].iloc[i],
                        'strength': min(touches / 10, 1.0),
                        'type': 'resistance',
                        'metadata': {'fractal': True, 'touches': touches}
                    })
        
        # Find fractal lows (support)
        for i in range(2, len(market_data) - 2):
            if (market_data['low'].iloc[i] < market_data['low'].iloc[i-1] and
                market_data['low'].iloc[i] < market_data['low'].iloc[i-2] and
                market_data['low'].iloc[i] < market_data['low'].iloc[i+1] and
                market_data['low'].iloc[i] < market_data['low'].iloc[i+2]):
                
                touches = self._count_level_touches(
                    market_data, market_data['low'].iloc[i], 'support', i
                )
                
                if touches >= 2:
                    levels.append({
                        'price': market_data['low'].iloc[i],
                        'strength': min(touches / 10, 1.0),
                        'type': 'support',
                        'metadata': {'fractal': True, 'touches': touches}
                    })
        
        return levels
    
    def _count_level_touches(self, data: pd.DataFrame, level_price: float, 
                             level_type: str, start_idx: int) -> int:
        """Count how many times price touched this level."""
        touches = 1  # Initial touch
        threshold = 0.002  # 0.2%
        
        for i in range(start_idx + 1, len(data)):
            if level_type == 'resistance':
                if abs(data['high'].iloc[i] - level_price) / level_price < threshold:
                    touches += 1
            else:
                if abs(data['low'].iloc[i] - level_price) / level_price < threshold:
                    touches += 1
        
        return touches
    
    def get_method_name(self) -> str:
        return "fractal"


class EnsembleSRPredictor:
    """Combines multiple S/R detection methods for robust level identification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnsembleSRPredictor")
        
        # Initialize all methods
        self.methods = [
            OptimizedSRMethod(config),
            ClassicalSRMethod(config),
            VolumeProfileSRMethod(config),
            FractalSRMethod(config)
        ]
        
        # Method weights (can be optimized)
        self.method_weights = self._load_method_weights()
        
        # Ensemble configuration
        self.min_method_agreement = config.get("min_method_agreement", 2)
        self.price_clustering_threshold = config.get("price_clustering_threshold", 0.002)
    
    def _load_method_weights(self) -> Dict[str, float]:
        """Load optimized method weights."""
        default_weights = {
            "optimized_strength": 0.35,
            "classical_ta": 0.25,
            "volume_profile": 0.25,
            "fractal": 0.15
        }
        
        try:
            weights_file = os.path.join(
                self.config.get("model_save_path", "models"),
                "ensemble_sr_weights.json"
            )
            
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    return data.get("weights", default_weights)
        except Exception as e:
            self.logger.error(f"Error loading ensemble weights: {e}")
        
        return default_weights
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=[],
        context="ensemble SR prediction"
    )
    @traced(span_name="EnsembleSR.predict")
    async def identify_ensemble_levels(
        self,
        market_data: pd.DataFrame
    ) -> List[EnsembleSRLevel]:
        """
        Identify S/R levels using ensemble of methods.
        
        Args:
            market_data: Historical market data
            
        Returns:
            List of ensemble S/R levels
        """
        try:
            # Collect levels from all methods
            all_levels = {}
            
            for method in self.methods:
                method_name = method.get_method_name()
                levels = method.identify_levels(market_data)
                all_levels[method_name] = levels
                
                self.logger.info(f"Method {method_name} found {len(levels)} levels")
            
            # Cluster and combine levels
            ensemble_levels = self._combine_levels(all_levels)
            
            # Sort by ensemble confidence
            ensemble_levels.sort(key=lambda x: x.ensemble_confidence, reverse=True)
            
            self.logger.info(f"Ensemble identified {len(ensemble_levels)} strong S/R levels")
            
            return ensemble_levels
            
        except Exception as e:
            self.logger.error(f"Error in ensemble S/R prediction: {e}")
            return []
    
    def _combine_levels(self, all_levels: Dict[str, List[Dict]]) -> List[EnsembleSRLevel]:
        """Combine levels from different methods."""
        ensemble_levels = []
        
        # Flatten all levels with method tags
        tagged_levels = []
        for method_name, levels in all_levels.items():
            for level in levels:
                tagged_levels.append({
                    'method': method_name,
                    'price': level['price'],
                    'strength': level['strength'],
                    'type': level.get('type', 'unknown'),
                    'metadata': level.get('metadata', {})
                })
        
        # Sort by price for clustering
        tagged_levels.sort(key=lambda x: x['price'])
        
        # Cluster nearby levels
        clusters = []
        current_cluster = []
        
        for level in tagged_levels:
            if not current_cluster:
                current_cluster.append(level)
            else:
                # Check if close to cluster
                cluster_price = np.mean([l['price'] for l in current_cluster])
                if abs(level['price'] - cluster_price) / cluster_price < self.price_clustering_threshold:
                    current_cluster.append(level)
                else:
                    # Process current cluster
                    if len(current_cluster) >= self.min_method_agreement:
                        ensemble_level = self._create_ensemble_level(current_cluster)
                        ensemble_levels.append(ensemble_level)
                    current_cluster = [level]
        
        # Don't forget last cluster
        if current_cluster and len(current_cluster) >= self.min_method_agreement:
            ensemble_level = self._create_ensemble_level(current_cluster)
            ensemble_levels.append(ensemble_level)
        
        return ensemble_levels
    
    def _create_ensemble_level(self, cluster: List[Dict]) -> EnsembleSRLevel:
        """Create ensemble level from cluster of method predictions."""
        
        # Calculate weighted average price
        weighted_price = 0
        total_weight = 0
        method_votes = {}
        
        for level in cluster:
            method_name = level['method']
            method_weight = self.method_weights.get(method_name, 0.25)
            
            weighted_price += level['price'] * level['strength'] * method_weight
            total_weight += level['strength'] * method_weight
            
            method_votes[method_name] = level['strength']
        
        avg_price = weighted_price / total_weight if total_weight > 0 else np.mean([l['price'] for l in cluster])
        
        # Determine type (majority vote)
        type_votes = {}
        for level in cluster:
            level_type = level.get('type', 'unknown')
            type_votes[level_type] = type_votes.get(level_type, 0) + 1
        
        level_type = max(type_votes, key=type_votes.get)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(cluster, method_votes)
        
        # Aggregate metadata
        metadata = {
            'cluster_size': len(cluster),
            'methods_agreed': list(method_votes.keys()),
            'price_spread': max(l['price'] for l in cluster) - min(l['price'] for l in cluster)
        }
        
        return EnsembleSRLevel(
            price=avg_price,
            strength=ensemble_confidence,
            type=level_type,
            method_votes=method_votes,
            ensemble_confidence=ensemble_confidence,
            metadata=metadata
        )
    
    def _calculate_ensemble_confidence(
        self,
        cluster: List[Dict],
        method_votes: Dict[str, float]
    ) -> float:
        """Calculate ensemble confidence score."""
        
        # Base confidence from number of methods agreeing
        agreement_score = len(method_votes) / len(self.methods)
        
        # Weighted average of method strengths
        weighted_strength = 0
        total_weight = 0
        
        for method_name, strength in method_votes.items():
            weight = self.method_weights.get(method_name, 0.25)
            weighted_strength += strength * weight
            total_weight += weight
        
        avg_strength = weighted_strength / total_weight if total_weight > 0 else 0.5
        
        # Price consistency score
        prices = [l['price'] for l in cluster]
        price_std = np.std(prices) / np.mean(prices) if prices else 0
        consistency_score = 1 - min(price_std * 100, 1)  # Penalize high variance
        
        # Combine scores
        ensemble_confidence = (
            agreement_score * 0.4 +
            avg_strength * 0.4 +
            consistency_score * 0.2
        )
        
        return min(ensemble_confidence, 1.0)
    
    def optimize_method_weights(
        self,
        historical_data: pd.DataFrame,
        actual_reversals: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Optimize method weights based on historical performance."""
        
        # Track method performance
        method_scores = {method.get_method_name(): [] for method in self.methods}
        
        # Test each method's predictions against actual reversals
        for method in self.methods:
            method_name = method.get_method_name()
            levels = method.identify_levels(historical_data)
            
            # Score based on proximity to actual reversals
            for reversal in actual_reversals:
                reversal_price = reversal['price']
                reversal_type = reversal['type']
                
                # Find closest predicted level
                matching_levels = [l for l in levels if l.get('type') == reversal_type]
                if matching_levels:
                    closest = min(matching_levels, 
                                key=lambda x: abs(x['price'] - reversal_price))
                    
                    # Score based on proximity
                    distance = abs(closest['price'] - reversal_price) / reversal_price
                    if distance < 0.005:  # Within 0.5%
                        score = 1.0
                    elif distance < 0.01:  # Within 1%
                        score = 0.5
                    else:
                        score = 0.0
                    
                    method_scores[method_name].append(score)
        
        # Calculate average scores
        avg_scores = {}
        for method_name, scores in method_scores.items():
            avg_scores[method_name] = np.mean(scores) if scores else 0.0
        
        # Normalize to create weights
        total_score = sum(avg_scores.values())
        if total_score > 0:
            optimized_weights = {
                method: score / total_score 
                for method, score in avg_scores.items()
            }
        else:
            # Equal weights if no clear winner
            optimized_weights = {
                method: 1.0 / len(self.methods) 
                for method in avg_scores.keys()
            }
        
        # Save optimized weights
        self._save_optimized_weights(optimized_weights)
        
        return optimized_weights
    
    def _save_optimized_weights(self, weights: Dict[str, float]) -> None:
        """Save optimized weights to file."""
        try:
            weights_file = os.path.join(
                self.config.get("model_save_path", "models"),
                "ensemble_sr_weights.json"
            )
            
            os.makedirs(os.path.dirname(weights_file), exist_ok=True)
            
            with open(weights_file, 'w') as f:
                json.dump({
                    "weights": weights,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            self.logger.info(f"Saved optimized ensemble weights to {weights_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble weights: {e}")