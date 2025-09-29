"""
Micro Regime Detector

Implementation for micro regime detection in NAS clustering.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class MicroRegimeType(Enum):
    """Types of micro regimes."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class MicroRegimeConfig:
    """Configuration for micro regime detection."""
    window_size: int = 20
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.01
    breakout_threshold: float = 0.05
    min_samples: int = 10


class MicroRegimeDetector:
    """Micro Regime Detector for fine-grained regime analysis."""
    
    def __init__(self, config: MicroRegimeConfig):
        """Initialize micro regime detector.
        
        Args:
            config: Micro regime detection configuration
        """
        self.config = config
        self.regime_history = []
        self.regime_transitions = []
        self.detection_metrics = {}
        
    def detect_regimes(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict:
        """Detect micro regimes in data.
        
        Args:
            data: Input data
            timestamps: Optional timestamps for data points
            
        Returns:
            Dictionary containing regime detection results
        """
        start_time = time.time()
        
        try:
            # Detect regimes
            regimes = self._detect_regime_sequence(data)
            
            # Analyze regime transitions
            transitions = self._analyze_regime_transitions(regimes)
            
            # Calculate regime statistics
            statistics = self._calculate_regime_statistics(regimes)
            
            # Record detection
            detection_record = {
                'regimes': regimes,
                'transitions': transitions,
                'statistics': statistics,
                'detection_time': time.time() - start_time,
                'timestamp': time.time()
            }
            self.regime_history.append(detection_record)
            
            return {
                'regimes': regimes,
                'transitions': transitions,
                'statistics': statistics,
                'detection_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'detection_time': time.time() - start_time
            }
    
    def _detect_regime_sequence(self, data: np.ndarray) -> List[Dict]:
        """Detect sequence of micro regimes."""
        regimes = []
        window_size = self.config.window_size
        
        for i in range(window_size, len(data)):
            window_data = data[i-window_size:i]
            
            # Calculate regime indicators
            volatility = np.std(window_data)
            trend = np.mean(np.diff(window_data))
            price_change = data[i] - data[i-window_size]
            
            # Classify regime
            regime_type = self._classify_regime(volatility, trend, price_change)
            
            regime_info = {
                'index': i,
                'regime_type': regime_type,
                'volatility': volatility,
                'trend': trend,
                'price_change': price_change,
                'window_data': window_data
            }
            
            regimes.append(regime_info)
        
        return regimes
    
    def _classify_regime(self, volatility: float, trend: float, price_change: float) -> MicroRegimeType:
        """Classify regime based on indicators."""
        # High volatility regime
        if volatility > self.config.volatility_threshold:
            if abs(price_change) > self.config.breakout_threshold:
                return MicroRegimeType.BREAKOUT
            else:
                return MicroRegimeType.HIGH_VOLATILITY
        
        # Low volatility regime
        elif volatility < self.config.volatility_threshold / 2:
            if abs(trend) < self.config.trend_threshold / 2:
                return MicroRegimeType.SIDEWAYS
            else:
                return MicroRegimeType.LOW_VOLATILITY
        
        # Trending regimes
        elif trend > self.config.trend_threshold:
            return MicroRegimeType.TRENDING_UP
        elif trend < -self.config.trend_threshold:
            return MicroRegimeType.TRENDING_DOWN
        
        # Reversal detection
        elif abs(price_change) > self.config.breakout_threshold:
            return MicroRegimeType.REVERSAL
        
        # Default to sideways
        else:
            return MicroRegimeType.SIDEWAYS
    
    def _analyze_regime_transitions(self, regimes: List[Dict]) -> List[Dict]:
        """Analyze transitions between regimes."""
        transitions = []
        
        for i in range(1, len(regimes)):
            current_regime = regimes[i]['regime_type']
            previous_regime = regimes[i-1]['regime_type']
            
            if current_regime != previous_regime:
                transition = {
                    'from_regime': previous_regime,
                    'to_regime': current_regime,
                    'transition_index': i,
                    'transition_time': regimes[i]['index'],
                    'volatility_change': regimes[i]['volatility'] - regimes[i-1]['volatility'],
                    'trend_change': regimes[i]['trend'] - regimes[i-1]['trend']
                }
                transitions.append(transition)
        
        return transitions
    
    def _calculate_regime_statistics(self, regimes: List[Dict]) -> Dict:
        """Calculate statistics for detected regimes."""
        if not regimes:
            return {}
        
        # Count regimes
        regime_counts = {}
        for regime in regimes:
            regime_type = regime['regime_type']
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        # Calculate regime durations
        regime_durations = {}
        current_regime = regimes[0]['regime_type']
        duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i]['regime_type'] == current_regime:
                duration += 1
            else:
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(duration)
                current_regime = regimes[i]['regime_type']
                duration = 1
        
        # Add final duration
        if current_regime not in regime_durations:
            regime_durations[current_regime] = []
        regime_durations[current_regime].append(duration)
        
        # Calculate average durations
        avg_durations = {}
        for regime_type, durations in regime_durations.items():
            avg_durations[regime_type] = np.mean(durations)
        
        # Calculate volatility statistics
        volatilities = [regime['volatility'] for regime in regimes]
        trends = [regime['trend'] for regime in regimes]
        
        return {
            'regime_counts': regime_counts,
            'regime_percentages': {
                regime: count / len(regimes) * 100 
                for regime, count in regime_counts.items()
            },
            'avg_durations': avg_durations,
            'volatility_stats': {
                'mean': np.mean(volatilities),
                'std': np.std(volatilities),
                'min': np.min(volatilities),
                'max': np.max(volatilities)
            },
            'trend_stats': {
                'mean': np.mean(trends),
                'std': np.std(trends),
                'min': np.min(trends),
                'max': np.max(trends)
            }
        }
    
    def predict_next_regime(self, data: np.ndarray) -> MicroRegimeType:
        """Predict next regime based on recent data."""
        if len(data) < self.config.window_size:
            return MicroRegimeType.SIDEWAYS
        
        # Use recent data for prediction
        recent_data = data[-self.config.window_size:]
        
        # Calculate indicators
        volatility = np.std(recent_data)
        trend = np.mean(np.diff(recent_data))
        price_change = recent_data[-1] - recent_data[0]
        
        # Classify regime
        return self._classify_regime(volatility, trend, price_change)
    
    def get_regime_history(self) -> List[Dict]:
        """Get regime detection history."""
        return self.regime_history
    
    def get_regime_transitions(self) -> List[Dict]:
        """Get regime transitions."""
        return self.regime_transitions
    
    def get_detection_metrics(self) -> Dict:
        """Get detection metrics."""
        return self.detection_metrics
    
    def get_regime_summary(self, data: np.ndarray) -> Dict:
        """Get summary of regimes in data."""
        regimes = self._detect_regime_sequence(data)
        transitions = self._analyze_regime_transitions(regimes)
        statistics = self._calculate_regime_statistics(regimes)
        
        return {
            'total_regimes': len(regimes),
            'regime_transitions': len(transitions),
            'statistics': statistics,
            'most_common_regime': max(statistics.get('regime_counts', {}), 
                                    key=statistics.get('regime_counts', {}).get, 
                                    default=None)
        }
