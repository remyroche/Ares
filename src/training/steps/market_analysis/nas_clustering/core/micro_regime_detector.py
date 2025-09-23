"""
Micro-regime detector for subtle market changes.

This module provides micro-regime detection capabilities for short-term trading,
identifying subtle market changes that may not be captured by standard regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import talib

logger = logging.getLogger(__name__)


class MicroRegimeType(Enum):
    """Micro-regime types for subtle market changes."""
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    TREND_CHANGE = "trend_change"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class MicroRegimeResult:
    """Result of micro-regime detection."""
    micro_regimes: np.ndarray
    micro_regime_types: List[MicroRegimeType]
    micro_regime_scores: np.ndarray
    micro_regime_metadata: Dict[str, Any]
    detection_accuracy: float
    execution_time: float


class MicroRegimeDetector:
    """Detector for micro-regimes in short-term trading."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize micro-regime detector.
        
        Args:
            config: Micro-regime detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Micro-regime detection settings
        self.sensitivity = config.get('micro_regime_sensitivity', 0.7)
        self.min_duration = config.get('min_micro_regime_duration', 3)  # 3 periods
        self.max_duration = config.get('max_micro_regime_duration', 15)  # 15 periods
        self.enable_micro_regime_detection = config.get('enable_micro_regime_detection', True)
        
        # Micro-regime types to detect
        self.micro_regime_types = config.get('micro_regime_types', [
            MicroRegimeType.BREAKOUT,
            MicroRegimeType.CONSOLIDATION,
            MicroRegimeType.REVERSAL,
            MicroRegimeType.ACCELERATION,
            MicroRegimeType.VOLUME_SPIKE,
            MicroRegimeType.VOLATILITY_SPIKE
        ])
        
        # Detection thresholds
        self.breakout_threshold = config.get('breakout_threshold', 0.02)  # 2% price change
        self.consolidation_threshold = config.get('consolidation_threshold', 0.005)  # 0.5% price change
        self.volume_spike_threshold = config.get('volume_spike_threshold', 2.0)  # 2x average volume
        self.volatility_spike_threshold = config.get('volatility_spike_threshold', 1.5)  # 1.5x average volatility
        
        self.logger.info(f"✅ Micro-regime detector initialized with sensitivity {self.sensitivity}")
    
    def detect_micro_regimes(self, data: np.ndarray, timestamps: np.ndarray,
                            features: Optional[np.ndarray] = None) -> MicroRegimeResult:
        """Detect micro-regimes in market data.
        
        Args:
            data: Market data (OHLCV)
            timestamps: Timestamps array
            features: Optional pre-computed features
            
        Returns:
            MicroRegimeResult with detected micro-regimes
        """
        import time
        start_time = time.time()
        
        try:
            if not self.enable_micro_regime_detection:
                return self._create_empty_result(timestamps, time.time() - start_time)
            
            # Extract micro-regime features
            micro_features = self._extract_micro_regime_features(data, features)
            
            # Detect micro-regimes using multiple methods
            micro_regimes = self._detect_micro_regimes_clustering(micro_features)
            micro_regime_types = self._classify_micro_regime_types(data, micro_regimes)
            micro_regime_scores = self._calculate_micro_regime_scores(data, micro_regimes)
            
            # Validate micro-regimes
            validated_regimes = self._validate_micro_regimes(
                micro_regimes, micro_regime_types, micro_regime_scores
            )
            
            # Calculate detection accuracy
            detection_accuracy = self._calculate_detection_accuracy(
                data, validated_regimes
            )
            
            execution_time = time.time() - start_time
            
            # Create result
            result = MicroRegimeResult(
                micro_regimes=validated_regimes['regimes'],
                micro_regime_types=validated_regimes['types'],
                micro_regime_scores=validated_regimes['scores'],
                micro_regime_metadata={
                    'sensitivity': self.sensitivity,
                    'min_duration': self.min_duration,
                    'max_duration': self.max_duration,
                    'detection_method': 'clustering',
                    'feature_count': micro_features.shape[1] if micro_features is not None else 0
                },
                detection_accuracy=detection_accuracy,
                execution_time=execution_time
            )
            
            self.logger.info(f"✅ Micro-regime detection completed: {len(np.unique(validated_regimes['regimes']))} micro-regimes in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Micro-regime detection failed: {e}")
            return self._create_empty_result(timestamps, execution_time)
    
    def _extract_micro_regime_features(self, data: np.ndarray, 
                                     features: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract features for micro-regime detection."""
        try:
            if features is not None:
                return features
            
            if data.shape[1] < 4:
                return np.array([])
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            micro_features = []
            
            # Price-based micro-features
            if len(close_price) >= 3:
                # Price acceleration (second derivative)
                price_acceleration = np.diff(np.diff(close_price))
                micro_features.append(price_acceleration)
                
                # Price jerk (third derivative)
                price_jerk = np.diff(price_acceleration)
                micro_features.append(price_jerk)
            
            # Volume-based micro-features
            if len(volume) >= 3:
                # Volume acceleration
                volume_acceleration = np.diff(np.diff(volume))
                micro_features.append(volume_acceleration)
                
                # Volume ratio
                volume_ma = talib.SMA(volume, timeperiod=5)
                volume_ratio = volume / volume_ma
                micro_features.append(volume_ratio)
            
            # Volatility-based micro-features
            if len(close_price) >= 5:
                # Micro-volatility
                micro_volatility = np.abs(np.diff(close_price, n=2))
                micro_features.append(micro_volatility)
                
                # Volatility acceleration
                vol_acceleration = np.diff(micro_volatility)
                micro_features.append(vol_acceleration)
            
            # Trend-based micro-features
            if len(close_price) >= 5:
                # Micro-trend changes
                micro_trend = np.diff(close_price, n=2)
                micro_features.append(micro_trend)
                
                # Trend acceleration
                trend_acceleration = np.diff(micro_trend)
                micro_features.append(trend_acceleration)
            
            # Range-based micro-features
            if len(high_price) >= 3:
                # Range changes
                range_changes = np.diff(high_price - low_price)
                micro_features.append(range_changes)
                
                # Range acceleration
                range_acceleration = np.diff(range_changes)
                micro_features.append(range_acceleration)
            
            if micro_features:
                # Pad arrays to same length
                max_length = max(len(f) for f in micro_features)
                padded_features = []
                for feature in micro_features:
                    if len(feature) < max_length:
                        padded = np.pad(feature, (0, max_length - len(feature)), mode='edge')
                    else:
                        padded = feature[:max_length]
                    padded_features.append(padded)
                
                return np.column_stack(padded_features)
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime feature extraction failed: {e}")
            return np.array([])
    
    def _detect_micro_regimes_clustering(self, features: np.ndarray) -> np.ndarray:
        """Detect micro-regimes using clustering."""
        try:
            if features.size == 0:
                return np.array([])
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use DBSCAN for micro-regime detection
            dbscan = DBSCAN(
                eps=self.sensitivity * 0.5,  # Adjust epsilon based on sensitivity
                min_samples=self.min_duration,
                metric='euclidean'
            )
            
            micro_regimes = dbscan.fit_predict(features_scaled)
            
            # Handle noise points (labeled as -1)
            noise_mask = micro_regimes == -1
            if np.any(noise_mask):
                # Assign noise points to nearest clusters
                unique_regimes = np.unique(micro_regimes[micro_regimes != -1])
                if len(unique_regimes) > 0:
                    # Use K-means for noise points
                    kmeans = KMeans(n_clusters=len(unique_regimes), random_state=42)
                    noise_regimes = kmeans.fit_predict(features_scaled[noise_mask])
                    micro_regimes[noise_mask] = noise_regimes
            
            return micro_regimes
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime clustering failed: {e}")
            return np.array([])
    
    def _classify_micro_regime_types(self, data: np.ndarray, 
                                   micro_regimes: np.ndarray) -> List[MicroRegimeType]:
        """Classify micro-regime types based on market data."""
        try:
            if len(micro_regimes) == 0:
                return []
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            micro_regime_types = []
            unique_regimes = np.unique(micro_regimes)
            
            for regime_id in unique_regimes:
                regime_mask = micro_regimes == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) == 0:
                    micro_regime_types.append(MicroRegimeType.CONSOLIDATION)
                    continue
                
                # Analyze regime characteristics
                regime_close = regime_data[:, 3]
                regime_high = regime_data[:, 1]
                regime_low = regime_data[:, 2]
                regime_volume = regime_data[:, 4] if regime_data.shape[1] > 4 else np.ones(len(regime_close))
                
                # Calculate regime metrics
                price_change = (regime_close[-1] - regime_close[0]) / regime_close[0]
                price_volatility = np.std(regime_close)
                volume_ratio = np.mean(regime_volume) / np.mean(volume)
                range_ratio = (np.max(regime_high) - np.min(regime_low)) / np.mean(regime_close)
                
                # Classify regime type
                regime_type = self._classify_regime_type(
                    price_change, price_volatility, volume_ratio, range_ratio
                )
                
                micro_regime_types.append(regime_type)
            
            return micro_regime_types
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime type classification failed: {e}")
            return [MicroRegimeType.CONSOLIDATION] * len(np.unique(micro_regimes))
    
    def _classify_regime_type(self, price_change: float, price_volatility: float,
                            volume_ratio: float, range_ratio: float) -> MicroRegimeType:
        """Classify a single regime type based on characteristics."""
        try:
            # Breakout detection
            if abs(price_change) > self.breakout_threshold:
                if price_change > 0:
                    return MicroRegimeType.BREAKOUT
                else:
                    return MicroRegimeType.REVERSAL
            
            # Volume spike detection
            if volume_ratio > self.volume_spike_threshold:
                return MicroRegimeType.VOLUME_SPIKE
            
            # Volatility spike detection
            if price_volatility > self.volatility_spike_threshold:
                return MicroRegimeType.VOLATILITY_SPIKE
            
            # Acceleration detection
            if abs(price_change) > self.consolidation_threshold:
                if price_change > 0:
                    return MicroRegimeType.ACCELERATION
                else:
                    return MicroRegimeType.DECELERATION
            
            # Default to consolidation
            return MicroRegimeType.CONSOLIDATION
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime type classification failed: {e}")
            return MicroRegimeType.CONSOLIDATION
    
    def _calculate_micro_regime_scores(self, data: np.ndarray, 
                                     micro_regimes: np.ndarray) -> np.ndarray:
        """Calculate scores for micro-regimes."""
        try:
            if len(micro_regimes) == 0:
                return np.array([])
            
            scores = np.zeros(len(micro_regimes))
            unique_regimes = np.unique(micro_regimes)
            
            for regime_id in unique_regimes:
                regime_mask = micro_regimes == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Calculate regime score based on characteristics
                regime_close = regime_data[:, 3]
                regime_volume = regime_data[:, 4] if regime_data.shape[1] > 4 else np.ones(len(regime_close))
                
                # Price change score
                price_change = abs((regime_close[-1] - regime_close[0]) / regime_close[0])
                price_score = min(price_change / self.breakout_threshold, 1.0)
                
                # Volume score
                volume_ratio = np.mean(regime_volume) / np.mean(data[:, 4] if data.shape[1] > 4 else np.ones(len(data)))
                volume_score = min(volume_ratio / self.volume_spike_threshold, 1.0)
                
                # Volatility score
                volatility = np.std(regime_close)
                volatility_score = min(volatility / (np.std(data[:, 3]) * self.volatility_spike_threshold), 1.0)
                
                # Combined score
                regime_score = (price_score + volume_score + volatility_score) / 3.0
                scores[regime_mask] = regime_score
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime score calculation failed: {e}")
            return np.zeros(len(micro_regimes))
    
    def _validate_micro_regimes(self, micro_regimes: np.ndarray,
                              micro_regime_types: List[MicroRegimeType],
                              micro_regime_scores: np.ndarray) -> Dict[str, Any]:
        """Validate micro-regimes based on duration and quality."""
        try:
            if len(micro_regimes) == 0:
                return {
                    'regimes': np.array([]),
                    'types': [],
                    'scores': np.array([])
                }
            
            # Filter regimes by duration
            valid_regimes = []
            valid_types = []
            valid_scores = []
            
            unique_regimes = np.unique(micro_regimes)
            
            for i, regime_id in enumerate(unique_regimes):
                regime_mask = micro_regimes == regime_id
                regime_duration = np.sum(regime_mask)
                
                # Check duration constraints
                if self.min_duration <= regime_duration <= self.max_duration:
                    # Check quality score
                    regime_score = np.mean(micro_regime_scores[regime_mask])
                    if regime_score > 0.3:  # Minimum quality threshold
                        valid_regimes.append(regime_id)
                        valid_types.append(micro_regime_types[i])
                        valid_scores.append(regime_score)
            
            # Create validated arrays
            validated_regimes = np.zeros_like(micro_regimes)
            validated_scores = np.zeros_like(micro_regime_scores)
            
            for i, regime_id in enumerate(valid_regimes):
                regime_mask = micro_regimes == regime_id
                validated_regimes[regime_mask] = i
                validated_scores[regime_mask] = valid_scores[i]
            
            return {
                'regimes': validated_regimes,
                'types': valid_types,
                'scores': validated_scores
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime validation failed: {e}")
            return {
                'regimes': micro_regimes,
                'types': micro_regime_types,
                'scores': micro_regime_scores
            }
    
    def _calculate_detection_accuracy(self, data: np.ndarray, 
                                   validated_regimes: Dict[str, Any]) -> float:
        """Calculate micro-regime detection accuracy."""
        try:
            if len(validated_regimes['regimes']) == 0:
                return 0.0
            
            # Use silhouette score as accuracy metric
            if len(np.unique(validated_regimes['regimes'])) > 1:
                # Extract features for accuracy calculation
                micro_features = self._extract_micro_regime_features(data)
                if micro_features.size > 0:
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(micro_features)
                    accuracy = silhouette_score(features_scaled, validated_regimes['regimes'])
                    return max(0.0, accuracy)  # Ensure non-negative
            
            return 0.5  # Default accuracy for single regime
            
        except Exception as e:
            self.logger.warning(f"⚠️ Detection accuracy calculation failed: {e}")
            return 0.0
    
    def _create_empty_result(self, timestamps: np.ndarray, execution_time: float) -> MicroRegimeResult:
        """Create empty result for failed detection."""
        return MicroRegimeResult(
            micro_regimes=np.zeros(len(timestamps)),
            micro_regime_types=[],
            micro_regime_scores=np.zeros(len(timestamps)),
            micro_regime_metadata={'error': 'Detection failed'},
            detection_accuracy=0.0,
            execution_time=execution_time
        )