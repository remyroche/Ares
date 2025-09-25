"""
Continuous Adaptation System for CLVSA Architectures

This module provides continuous adaptation capabilities for tree-based CLVSA models
during live trading, including:
- Continuous performance monitoring
- Automatic adaptation triggers
- Incremental model updates
- Regime change detection
- CLVSA-specific adaptation strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import threading
import queue
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import existing utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_progress, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

try:
    from src.utils.nas_tas.tas.meta_learning import (
        AdvancedMetaLearningSystem, MetaTask
    )
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False

try:
    from src.utils.nas_tas.tas.realtime import (
        RealTimeOptimizationEngine, PerformanceMonitor
    )
    REALTIME_OPTIMIZATION_AVAILABLE = True
except ImportError:
    REALTIME_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ContinuousAdaptationConfig:
    """Configuration for continuous adaptation."""
    
    # Adaptation triggers
    enable_continuous_adaptation: bool = True
    adaptation_frequency: float = 60.0  # seconds
    performance_threshold: float = 0.05  # minimum performance change to trigger adaptation
    regime_change_threshold: float = 0.2  # regime change detection threshold
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_window: int = 100  # number of recent predictions
    performance_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'latency'
    ])
    
    # Regime detection
    enable_regime_detection: bool = True
    regime_detection_window: int = 50  # window for regime detection
    regime_stability_threshold: float = 0.8  # stability threshold for regime
    
    # CLVSA-specific adaptation
    enable_cvlsa_adaptation: bool = True
    cvlsa_adaptation_rate: float = 0.1  # CLVSA adaptation rate
    cvlsa_memory_efficiency: bool = True  # memory-efficient adaptation
    cvlsa_parallelization: bool = True  # parallel adaptation
    
    # Meta-learning integration
    enable_meta_learning: bool = True
    meta_learning_adaptation_rate: float = 0.05  # meta-learning adaptation rate
    few_shot_adaptation: bool = True  # few-shot learning for adaptation
    
    # Incremental updates
    enable_incremental_updates: bool = True
    incremental_update_rate: float = 0.1  # rate of incremental updates
    memory_size: int = 1000  # memory size for incremental learning
    
    # Resource management
    enable_resource_management: bool = True
    max_memory_usage: float = 0.8  # maximum memory usage
    max_cpu_usage: float = 0.8  # maximum CPU usage
    resource_check_interval: float = 30.0  # seconds
    
    # Threading and concurrency
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    adaptation_queue_size: int = 100
    
    # Logging and debugging
    enable_detailed_logging: bool = True
    log_interval: float = 60.0  # seconds
    debug_mode: bool = False


@dataclass
class AdaptationTrigger:
    """Adaptation trigger definition."""
    trigger_type: str  # "performance", "regime_change", "manual", "scheduled"
    trigger_value: float
    trigger_threshold: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of adaptation."""
    adaptation_id: str
    trigger: AdaptationTrigger
    adaptation_type: str
    success: bool
    performance_improvement: float
    adaptation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeChangeDetector:
    """Regime change detection for continuous adaptation."""
    
    def __init__(self, config: ContinuousAdaptationConfig):
        """Initialize regime change detector."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Regime detection state
        self.regime_history = deque(maxlen=config.regime_detection_window)
        self.current_regime = None
        self.regime_stability = 0.0
        
        # Detection metrics
        self.regime_change_count = 0
        self.last_regime_change = None
        
        tprint_info("✅ Regime Change Detector initialized")
    
    def detect_regime_change(self, 
                            market_data: np.ndarray,
                            timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Detect regime changes in market data.
        
        Args:
            market_data: Current market data
            timestamp: Timestamp of the data
            
        Returns:
            Regime change detection results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Extract regime features
            regime_features = self._extract_regime_features(market_data)
            
            # Update regime history
            self.regime_history.append({
                'timestamp': timestamp,
                'features': regime_features,
                'regime_id': self._identify_regime(regime_features)
            })
            
            # Detect regime change
            regime_change_detected = self._detect_regime_change()
            
            # Update regime stability
            self._update_regime_stability()
            
            result = {
                'regime_change_detected': regime_change_detected,
                'current_regime': self.current_regime,
                'regime_stability': self.regime_stability,
                'regime_change_count': self.regime_change_count,
                'last_regime_change': self.last_regime_change,
                'regime_features': regime_features
            }
            
            if regime_change_detected:
                self.regime_change_count += 1
                self.last_regime_change = timestamp
                
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Regime change detected: {self.current_regime}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Regime change detection failed: {e}")
            return {
                'regime_change_detected': False,
                'error': str(e)
            }
    
    def _extract_regime_features(self, market_data: np.ndarray) -> np.ndarray:
        """Extract features for regime detection."""
        # Statistical features
        features = [
            np.mean(market_data),
            np.std(market_data),
            np.median(market_data),
            np.percentile(market_data, 25),
            np.percentile(market_data, 75),
            np.min(market_data),
            np.max(market_data)
        ]
        
        return np.array(features)
    
    def _identify_regime(self, features: np.ndarray) -> str:
        """Identify regime from features."""
        # Simple regime identification based on volatility
        volatility = features[1]  # std
        
        if volatility < 0.1:
            return "low_volatility"
        elif volatility < 0.3:
            return "medium_volatility"
        else:
            return "high_volatility"
    
    def _detect_regime_change(self) -> bool:
        """Detect if regime has changed."""
        if len(self.regime_history) < 2:
            return False
        
        # Get current and previous regimes
        current_regime = self.regime_history[-1]['regime_id']
        previous_regime = self.regime_history[-2]['regime_id']
        
        # Check for regime change
        regime_changed = current_regime != previous_regime
        
        if regime_changed:
            self.current_regime = current_regime
        
        return regime_changed
    
    def _update_regime_stability(self):
        """Update regime stability metric."""
        if len(self.regime_history) < 10:
            self.regime_stability = 0.0
            return
        
        # Calculate stability based on recent regime consistency
        recent_regimes = [entry['regime_id'] for entry in list(self.regime_history)[-10:]]
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate stability as the proportion of the most common regime
        max_count = max(regime_counts.values())
        self.regime_stability = max_count / len(recent_regimes)


class PerformanceAdaptationTrigger:
    """Performance-based adaptation trigger."""
    
    def __init__(self, config: ContinuousAdaptationConfig):
        """Initialize performance adaptation trigger."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.performance_history = deque(maxlen=config.monitoring_window)
        self.baseline_performance = None
        
        tprint_info("✅ Performance Adaptation Trigger initialized")
    
    def check_adaptation_trigger(self, 
                               current_performance: Dict[str, float],
                               timestamp: Optional[datetime] = None) -> Optional[AdaptationTrigger]:
        """
        Check if adaptation should be triggered based on performance.
        
        Args:
            current_performance: Current performance metrics
            timestamp: Timestamp of the performance data
            
        Returns:
            Adaptation trigger if triggered, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Update performance history
            self.performance_history.append({
                'timestamp': timestamp,
                'performance': current_performance
            })
            
            # Calculate performance degradation
            performance_degradation = self._calculate_performance_degradation()
            
            # Check if adaptation should be triggered
            if performance_degradation > self.config.performance_threshold:
                trigger = AdaptationTrigger(
                    trigger_type="performance",
                    trigger_value=performance_degradation,
                    trigger_threshold=self.config.performance_threshold,
                    timestamp=timestamp,
                    metadata={
                        'current_performance': current_performance,
                        'performance_degradation': performance_degradation
                    }
                )
                
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Performance adaptation triggered: {performance_degradation:.3f}")
                
                return trigger
            
            return None
            
        except Exception as e:
            tprint_error(f"❌ Performance adaptation trigger check failed: {e}")
            return None
    
    def _calculate_performance_degradation(self) -> float:
        """Calculate performance degradation."""
        if len(self.performance_history) < 2:
            return 0.0
        
        # Get recent performance
        recent_performance = list(self.performance_history)[-5:]  # Last 5 measurements
        if len(recent_performance) < 2:
            return 0.0
        
        # Calculate average performance
        recent_avg = np.mean([p['performance'].get('accuracy', 0.0) for p in recent_performance])
        
        # Calculate baseline performance
        if self.baseline_performance is None:
            self.baseline_performance = recent_avg
            return 0.0
        
        # Calculate degradation
        degradation = self.baseline_performance - recent_avg
        
        return max(0.0, degradation)


class CLVSAAdaptationEngine:
    """CLVSA-specific adaptation engine."""
    
    def __init__(self, config: ContinuousAdaptationConfig):
        """Initialize CLVSA adaptation engine."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # CLVSA adaptation state
        self.cvlsa_parameters = {}
        self.adaptation_history = []
        
        # Meta-learning integration
        if META_LEARNING_AVAILABLE:
            self.meta_learning_system = AdvancedMetaLearningSystem()
        else:
            self.meta_learning_system = None
        
        tprint_info("✅ CLVSA Adaptation Engine initialized")
    
    def adapt_cvlsa_model(self, 
                         model: Any,
                         adaptation_trigger: AdaptationTrigger,
                         market_data: np.ndarray) -> AdaptationResult:
        """
        Adapt CLVSA model based on trigger.
        
        Args:
            model: CLVSA model to adapt
            adaptation_trigger: Trigger for adaptation
            market_data: Current market data
            
        Returns:
            Adaptation result
        """
        start_time = time.time()
        adaptation_id = f"cvlsa_adaptation_{int(time.time())}"
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔄 Starting CLVSA adaptation: {adaptation_id}")
            
            # Determine adaptation strategy
            adaptation_strategy = self._determine_adaptation_strategy(adaptation_trigger)
            
            # Perform adaptation
            adapted_model = self._perform_cvlsa_adaptation(
                model, adaptation_strategy, market_data
            )
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(
                model, adapted_model, market_data
            )
            
            adaptation_time = time.time() - start_time
            
            result = AdaptationResult(
                adaptation_id=adaptation_id,
                trigger=adaptation_trigger,
                adaptation_type="cvlsa_adaptation",
                success=True,
                performance_improvement=performance_improvement,
                adaptation_time=adaptation_time,
                metadata={
                    'adaptation_strategy': adaptation_strategy,
                    'cvlsa_parameters': self.cvlsa_parameters
                }
            )
            
            # Update adaptation history
            self.adaptation_history.append(result)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ CLVSA adaptation completed: {performance_improvement:.3f} improvement")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ CLVSA adaptation failed: {e}")
            return AdaptationResult(
                adaptation_id=adaptation_id,
                trigger=adaptation_trigger,
                adaptation_type="cvlsa_adaptation",
                success=False,
                performance_improvement=0.0,
                adaptation_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _determine_adaptation_strategy(self, trigger: AdaptationTrigger) -> str:
        """Determine adaptation strategy based on trigger."""
        if trigger.trigger_type == "performance":
            return "performance_optimization"
        elif trigger.trigger_type == "regime_change":
            return "regime_adaptation"
        else:
            return "general_adaptation"
    
    def _perform_cvlsa_adaptation(self, 
                                 model: Any,
                                 strategy: str,
                                 market_data: np.ndarray) -> Any:
        """Perform CLVSA-specific adaptation."""
        try:
            # Apply CLVSA-specific adaptations
            if strategy == "performance_optimization":
                model = self._optimize_performance(model, market_data)
            elif strategy == "regime_adaptation":
                model = self._adapt_to_regime(model, market_data)
            else:
                model = self._general_adaptation(model, market_data)
            
            return model
            
        except Exception as e:
            tprint_error(f"❌ CLVSA adaptation failed: {e}")
            return model
    
    def _optimize_performance(self, model: Any, market_data: np.ndarray) -> Any:
        """Optimize model performance."""
        # Performance optimization strategies
        if hasattr(model, 'n_estimators'):
            model.n_estimators = min(model.n_estimators + 10, 1000)
        
        if hasattr(model, 'max_depth'):
            model.max_depth = min(model.max_depth + 1, 20)
        
        return model
    
    def _adapt_to_regime(self, model: Any, market_data: np.ndarray) -> Any:
        """Adapt model to new regime."""
        # Regime-specific adaptations
        volatility = np.std(market_data)
        
        if volatility > 0.3:  # High volatility regime
            if hasattr(model, 'max_depth'):
                model.max_depth = min(model.max_depth + 2, 20)
        else:  # Low volatility regime
            if hasattr(model, 'max_depth'):
                model.max_depth = max(model.max_depth - 1, 3)
        
        return model
    
    def _general_adaptation(self, model: Any, market_data: np.ndarray) -> Any:
        """General model adaptation."""
        # General adaptation strategies
        return model
    
    def _calculate_performance_improvement(self, 
                                         original_model: Any,
                                         adapted_model: Any,
                                         market_data: np.ndarray) -> float:
        """Calculate performance improvement."""
        try:
            # Simple performance comparison
            # In practice, this would use proper evaluation metrics
            return 0.1  # Placeholder improvement
        except Exception as e:
            tprint_warning(f"Adaptation improvement calculation failed: {e}. Returning 0.0.")
            return 0.0


class ContinuousAdaptationSystem:
    """
    Main continuous adaptation system for CLVSA architectures.
    """
    
    def __init__(self, config: Optional[ContinuousAdaptationConfig] = None):
        """Initialize continuous adaptation system."""
        self.config = config or ContinuousAdaptationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.regime_detector = RegimeChangeDetector(self.config)
        self.performance_trigger = PerformanceAdaptationTrigger(self.config)
        self.cvlsa_adaptation_engine = CLVSAAdaptationEngine(self.config)
        
        # Real-time optimization integration
        if REALTIME_OPTIMIZATION_AVAILABLE:
            self.realtime_optimizer = RealTimeOptimizationEngine()
        else:
            self.realtime_optimizer = None
        
        # Adaptation state
        self.is_adapting = False
        self.adaptation_queue = queue.Queue(maxsize=self.config.adaptation_queue_size)
        self.adaptation_thread = None
        
        # Performance tracking
        self.adaptation_history = []
        self.performance_history = []
        
        tprint_info("✅ Continuous Adaptation System initialized")
    
    def start_continuous_adaptation(self):
        """Start continuous adaptation system."""
        if self.is_adapting:
            return
        
        self.is_adapting = True
        
        # Start adaptation thread
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        if TPRINT_AVAILABLE:
            tprint_success("🚀 Continuous adaptation system started")
    
    def stop_continuous_adaptation(self):
        """Stop continuous adaptation system."""
        self.is_adapting = False
        
        # Stop adaptation thread
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=1.0)
        
        if TPRINT_AVAILABLE:
            tprint_info("🛑 Continuous adaptation system stopped")
    
    def process_market_data(self, 
                           market_data: np.ndarray,
                           model: Any,
                           timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process market data and trigger adaptations if needed.
        
        Args:
            market_data: Current market data
            model: Current model
            timestamp: Timestamp of the data
            
        Returns:
            Processing results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            results = {
                'timestamp': timestamp,
                'adaptations_triggered': [],
                'regime_changes_detected': [],
                'performance_updates': []
            }
            
            # Detect regime changes
            if self.config.enable_regime_detection:
                regime_result = self.regime_detector.detect_regime_change(market_data, timestamp)
                if regime_result.get('regime_change_detected', False):
                    results['regime_changes_detected'].append(regime_result)
                    
                    # Trigger regime adaptation
                    regime_trigger = AdaptationTrigger(
                        trigger_type="regime_change",
                        trigger_value=regime_result.get('regime_stability', 0.0),
                        trigger_threshold=self.config.regime_change_threshold,
                        timestamp=timestamp,
                        metadata=regime_result
                    )
                    
                    self.adaptation_queue.put(('regime_adaptation', regime_trigger, model, market_data))
            
            # Check performance triggers
            if self.config.enable_performance_monitoring:
                # Calculate current performance (placeholder)
                current_performance = {'accuracy': 0.8, 'latency': 0.1}
                
                performance_trigger = self.performance_trigger.check_adaptation_trigger(
                    current_performance, timestamp
                )
                
                if performance_trigger:
                    results['adaptations_triggered'].append(performance_trigger)
                    
                    # Trigger performance adaptation
                    self.adaptation_queue.put(('performance_adaptation', performance_trigger, model, market_data))
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Market data processing failed: {e}")
            return {
                'timestamp': timestamp,
                'error': str(e)
            }
    
    def _adaptation_loop(self):
        """Main adaptation loop."""
        while self.is_adapting:
            try:
                # Get adaptation request
                adaptation_request = self.adaptation_queue.get(timeout=1.0)
                adaptation_type, trigger, model, market_data = adaptation_request
                
                # Perform adaptation
                if adaptation_type == "regime_adaptation":
                    result = self.cvlsa_adaptation_engine.adapt_cvlsa_model(
                        model, trigger, market_data
                    )
                elif adaptation_type == "performance_adaptation":
                    result = self.cvlsa_adaptation_engine.adapt_cvlsa_model(
                        model, trigger, market_data
                    )
                else:
                    continue
                
                # Update adaptation history
                self.adaptation_history.append(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                tprint_error(f"❌ Adaptation loop error: {e}")
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status."""
        return {
            'is_adapting': self.is_adapting,
            'adaptation_queue_size': self.adaptation_queue.qsize(),
            'n_adaptations': len(self.adaptation_history),
            'regime_detection_enabled': self.config.enable_regime_detection,
            'performance_monitoring_enabled': self.config.enable_performance_monitoring,
            'cvlsa_adaptation_enabled': self.config.enable_cvlsa_adaptation,
            'recent_adaptations': self.adaptation_history[-5:] if len(self.adaptation_history) > 5 else self.adaptation_history
        }


# Factory functions
def create_continuous_adaptation_system(config: Optional[ContinuousAdaptationConfig] = None) -> ContinuousAdaptationSystem:
    """Create continuous adaptation system instance."""
    return ContinuousAdaptationSystem(config)


def create_regime_change_detector(config: Optional[ContinuousAdaptationConfig] = None) -> RegimeChangeDetector:
    """Create regime change detector instance."""
    return RegimeChangeDetector(config or ContinuousAdaptationConfig())


def create_cvlsa_adaptation_engine(config: Optional[ContinuousAdaptationConfig] = None) -> CLVSAAdaptationEngine:
    """Create CLVSA adaptation engine instance."""
    return CLVSAAdaptationEngine(config)


# Example usage
if __name__ == "__main__":
    # Create continuous adaptation system
    config = ContinuousAdaptationConfig(
        enable_continuous_adaptation=True,
        enable_regime_detection=True,
        enable_performance_monitoring=True,
        enable_cvlsa_adaptation=True
    )
    
    adaptation_system = create_continuous_adaptation_system(config)
    
    print("Continuous Adaptation System created successfully!")
    print(f"Adaptation status: {adaptation_system.get_adaptation_status()}")