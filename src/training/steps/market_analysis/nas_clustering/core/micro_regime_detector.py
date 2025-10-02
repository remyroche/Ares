"""
Micro Regime Detector

Implementation for micro regime detection in NAS clustering.
"""

print("🔍 [MICRO_REGIME_DETECTOR] Loading Micro Regime Detector module")
print("🔍 [MICRO_REGIME_DETECTOR] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/micro_regime_detector.py")
print("🔍 [MICRO_REGIME_DETECTOR] Purpose: Implementation for micro regime detection in NAS clustering")
print("🔍 [MICRO_REGIME_DETECTOR] Status: Starting module import")

import numpy as np
print("🔍 [MICRO_REGIME_DETECTOR] ✓ NumPy imported successfully")

from typing import Dict, List, Any, Optional, Tuple
print("🔍 [MICRO_REGIME_DETECTOR] ✓ Typing imports completed")

from dataclasses import dataclass
print("🔍 [MICRO_REGIME_DETECTOR] ✓ Dataclasses imported successfully")

from enum import Enum
print("🔍 [MICRO_REGIME_DETECTOR] ✓ Enum imported successfully")

import time
print("🔍 [MICRO_REGIME_DETECTOR] ✓ Time module imported successfully")

print("🔍 [MICRO_REGIME_DETECTOR] All imports completed successfully")


class MicroRegimeType(Enum):
    """Types of micro regimes."""
    print("🔍 [MICRO_REGIME_TYPE] Defining MicroRegimeType enum")
    HIGH_VOLATILITY = "high_volatility"
    print("🔍 [MICRO_REGIME_TYPE] ✓ HIGH_VOLATILITY defined")
    LOW_VOLATILITY = "low_volatility"
    print("🔍 [MICRO_REGIME_TYPE] ✓ LOW_VOLATILITY defined")
    TRENDING_UP = "trending_up"
    print("🔍 [MICRO_REGIME_TYPE] ✓ TRENDING_UP defined")
    TRENDING_DOWN = "trending_down"
    print("🔍 [MICRO_REGIME_TYPE] ✓ TRENDING_DOWN defined")
    SIDEWAYS = "sideways"
    print("🔍 [MICRO_REGIME_TYPE] ✓ SIDEWAYS defined")
    BREAKOUT = "breakout"
    print("🔍 [MICRO_REGIME_TYPE] ✓ BREAKOUT defined")
    REVERSAL = "reversal"
    print("🔍 [MICRO_REGIME_TYPE] ✓ REVERSAL defined")
    print("🔍 [MICRO_REGIME_TYPE] All regime types defined successfully")


@dataclass
class MicroRegimeConfig:
    """Configuration for micro regime detection."""
    print("🔍 [MICRO_REGIME_CONFIG] Defining MicroRegimeConfig dataclass")
    window_size: int = 20
    print("🔍 [MICRO_REGIME_CONFIG] ✓ window_size set to 20")
    volatility_threshold: float = 0.02
    print("🔍 [MICRO_REGIME_CONFIG] ✓ volatility_threshold set to 0.02")
    trend_threshold: float = 0.01
    print("🔍 [MICRO_REGIME_CONFIG] ✓ trend_threshold set to 0.01")
    breakout_threshold: float = 0.05
    print("🔍 [MICRO_REGIME_CONFIG] ✓ breakout_threshold set to 0.05")
    min_samples: int = 10
    print("🔍 [MICRO_REGIME_CONFIG] ✓ min_samples set to 10")
    print("🔍 [MICRO_REGIME_CONFIG] All configuration parameters defined successfully")


class MicroRegimeDetector:
    """Micro Regime Detector for fine-grained regime analysis."""
    
    def __init__(self, config: MicroRegimeConfig):
        """Initialize micro regime detector.
        
        Args:
            config: Micro regime detection configuration
        """
        print("🔍 [MICRO_REGIME_DETECTOR_INIT] Initializing MicroRegimeDetector")
        print(f"🔍 [MICRO_REGIME_DETECTOR_INIT] Config received: {config}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_INIT] Config type: {type(config)}")
        
        self.config = config
        print("🔍 [MICRO_REGIME_DETECTOR_INIT] ✓ Config assigned to self.config")
        
        self.regime_history = []
        print("🔍 [MICRO_REGIME_DETECTOR_INIT] ✓ regime_history initialized as empty list")
        
        self.regime_transitions = []
        print("🔍 [MICRO_REGIME_DETECTOR_INIT] ✓ regime_transitions initialized as empty list")
        
        self.detection_metrics = {}
        print("🔍 [MICRO_REGIME_DETECTOR_INIT] ✓ detection_metrics initialized as empty dict")
        
        print("🔍 [MICRO_REGIME_DETECTOR_INIT] Initialization complete!")
        
    def detect_regimes(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict:
        """Detect micro regimes in data.
        
        Args:
            data: Input data
            timestamps: Optional timestamps for data points
            
        Returns:
            Dictionary containing regime detection results
        """
        print("🔍 [MICRO_REGIME_DETECTOR_DETECT] Starting regime detection")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data shape: {data.shape}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data type: {type(data)}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data dtype: {data.dtype}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data min: {np.min(data):.6f}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data max: {np.max(data):.6f}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data mean: {np.mean(data):.6f}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Data std: {np.std(data):.6f}")
        
        if timestamps is not None:
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Timestamps provided - shape: {timestamps.shape}")
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Timestamp range: {np.min(timestamps)} to {np.max(timestamps)}")
        else:
            print("🔍 [MICRO_REGIME_DETECTOR_DETECT] No timestamps provided")
        
        start_time = time.time()
        print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Start time recorded: {start_time}")
        
        try:
            print("🔍 [MICRO_REGIME_DETECTOR_DETECT] Starting try block")
            # Detect regimes
            print("🔍 [MICRO_REGIME_DETECTOR_DETECT] Detecting regime sequence...")
            regimes = self._detect_regime_sequence(data)
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] ✓ Regime sequence detected - {len(regimes)} regimes found")
            
            # Analyze regime transitions
            print("🔍 [MICRO_REGIME_DETECTOR_DETECT] Analyzing regime transitions...")
            transitions = self._analyze_regime_transitions(regimes)
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] ✓ Transitions analyzed - {len(transitions)} transitions found")
            
            # Calculate regime statistics
            print("🔍 [MICRO_REGIME_DETECTOR_DETECT] Calculating regime statistics...")
            statistics = self._calculate_regime_statistics(regimes)
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] ✓ Statistics calculated: {statistics}")
            
            # Record detection
            detection_time = time.time() - start_time
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Detection time: {detection_time:.4f}s")
            
            detection_record = {
                'regimes': regimes,
                'transitions': transitions,
                'statistics': statistics,
                'detection_time': detection_time,
                'timestamp': time.time()
            }
            print("🔍 [MICRO_REGIME_DETECTOR_DETECT] Creating detection record...")
            self.regime_history.append(detection_record)
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] ✓ Detection record added to history (total: {len(self.regime_history)})")
            
            result = {
                'regimes': regimes,
                'transitions': transitions,
                'statistics': statistics,
                'detection_time': detection_time
            }
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] ✓ Detection completed successfully")
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Result: {result}")
            return result
            
        except Exception as e:
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] ❌ Exception occurred: {e}")
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Exception type: {type(e)}")
            detection_time = time.time() - start_time
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Detection time before error: {detection_time:.4f}s")
            
            error_result = {
                'error': str(e),
                'detection_time': detection_time
            }
            print(f"🔍 [MICRO_REGIME_DETECTOR_DETECT] Returning error result: {error_result}")
            return error_result
    
    def _detect_regime_sequence(self, data: np.ndarray) -> List[Dict]:
        """Detect sequence of micro regimes."""
        print("🔍 [MICRO_REGIME_DETECTOR_SEQUENCE] Starting regime sequence detection")
        print(f"🔍 [MICRO_REGIME_DETECTOR_SEQUENCE] Data length: {len(data)}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_SEQUENCE] Window size: {self.config.window_size}")
        
        regimes = []
        window_size = self.config.window_size
        print(f"🔍 [MICRO_REGIME_DETECTOR_SEQUENCE] Processing {len(data) - window_size} windows")
        
        for i in range(window_size, len(data)):
            if i % 100 == 0:  # Print progress every 100 iterations
                print(f"🔍 [MICRO_REGIME_DETECTOR_SEQUENCE] Processing window {i-window_size+1}/{len(data)-window_size}")
            
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
        
        print(f"🔍 [MICRO_REGIME_DETECTOR_SEQUENCE] ✓ Sequence detection complete - {len(regimes)} regimes found")
        return regimes
    
    def _classify_regime(self, volatility: float, trend: float, price_change: float) -> MicroRegimeType:
        """Classify regime based on indicators."""
        print(f"🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] Classifying regime - volatility: {volatility:.6f}, trend: {trend:.6f}, price_change: {price_change:.6f}")
        print(f"🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] Thresholds - volatility: {self.config.volatility_threshold}, trend: {self.config.trend_threshold}, breakout: {self.config.breakout_threshold}")
        
        # High volatility regime
        if volatility > self.config.volatility_threshold:
            print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] High volatility detected")
            if abs(price_change) > self.config.breakout_threshold:
                print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as BREAKOUT")
                return MicroRegimeType.BREAKOUT
            else:
                print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as HIGH_VOLATILITY")
                return MicroRegimeType.HIGH_VOLATILITY
        
        # Low volatility regime
        elif volatility < self.config.volatility_threshold / 2:
            print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] Low volatility detected")
            if abs(trend) < self.config.trend_threshold / 2:
                print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as SIDEWAYS")
                return MicroRegimeType.SIDEWAYS
            else:
                print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as LOW_VOLATILITY")
                return MicroRegimeType.LOW_VOLATILITY
        
        # Trending regimes
        elif trend > self.config.trend_threshold:
            print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as TRENDING_UP")
            return MicroRegimeType.TRENDING_UP
        elif trend < -self.config.trend_threshold:
            print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as TRENDING_DOWN")
            return MicroRegimeType.TRENDING_DOWN
        
        # Reversal detection
        elif abs(price_change) > self.config.breakout_threshold:
            print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Classified as REVERSAL")
            return MicroRegimeType.REVERSAL
        
        # Default to sideways
        else:
            print("🔍 [MICRO_REGIME_DETECTOR_CLASSIFY] ✓ Default classification: SIDEWAYS")
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
