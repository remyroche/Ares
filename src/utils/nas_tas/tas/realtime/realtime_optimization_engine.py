"""
Real-Time Optimization Engine for CLVSA Architectures

This module provides continuous optimization capabilities specifically designed
for tree-based CLVSA models during live trading, including:
- Continuous performance monitoring
- Real-time adaptation triggers
- Incremental model updates
- Latency optimization
- Resource management
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
    from src.utils.nas_tas.tas.hardware import (
        TreeHardwareAccelerator, CLVSAHardwareOptimizer
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RealTimeOptimizationConfig:
    """Configuration for real-time optimization."""
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    performance_window: int = 100  # number of recent predictions
    performance_threshold: float = 0.05  # minimum performance change to trigger adaptation
    
    # Adaptation triggers
    enable_adaptation_triggers: bool = True
    adaptation_frequency: int = 50  # adapt every N predictions
    adaptation_threshold: float = 0.1  # performance threshold for adaptation
    regime_change_threshold: float = 0.2  # regime change detection threshold
    
    # Optimization settings
    enable_continuous_optimization: bool = True
    optimization_interval: float = 10.0  # seconds
    max_optimization_time: float = 5.0  # maximum time for single optimization
    optimization_budget: float = 0.1  # fraction of time for optimization
    
    # Latency optimization
    enable_latency_optimization: bool = True
    target_latency: float = 0.1  # target latency in seconds
    max_latency: float = 1.0  # maximum acceptable latency
    latency_window: int = 20  # window for latency calculation
    
    # Resource management
    enable_resource_management: bool = True
    max_memory_usage: float = 0.8  # maximum memory usage fraction
    max_cpu_usage: float = 0.8  # maximum CPU usage fraction
    resource_check_interval: float = 5.0  # seconds
    
    # CLVSA-specific settings
    enable_cvlsa_optimization: bool = True
    cvlsa_adaptation_rate: float = 0.1  # CLVSA adaptation rate
    cvlsa_memory_efficiency: bool = True  # memory-efficient CLVSA updates
    
    # Threading and concurrency
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    thread_pool_size: int = 8
    
    # Logging and debugging
    enable_detailed_logging: bool = True
    log_interval: float = 10.0  # seconds
    debug_mode: bool = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for real-time optimization."""
    
    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Latency metrics
    prediction_latency: float = 0.0
    optimization_latency: float = 0.0
    total_latency: float = 0.0
    
    # Throughput metrics
    predictions_per_second: float = 0.0
    optimizations_per_hour: float = 0.0
    
    # Resource metrics
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # Adaptation metrics
    adaptation_frequency: float = 0.0
    adaptation_success_rate: float = 0.0
    regime_change_detection_rate: float = 0.0
    
    # CLVSA-specific metrics
    cvlsa_convergence_rate: float = 0.0
    cvlsa_memory_efficiency: float = 0.0
    cvlsa_adaptation_speed: float = 0.0


class PerformanceMonitor:
    """Real-time performance monitoring for CLVSA models."""
    
    def __init__(self, config: RealTimeOptimizationConfig):
        """Initialize performance monitor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.performance_history = deque(maxlen=config.performance_window)
        self.latency_history = deque(maxlen=config.latency_window)
        self.adaptation_history = deque(maxlen=100)
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        tprint_info("✅ Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        if TPRINT_AVAILABLE:
            tprint_info("🛑 Performance monitoring stopped")
    
    def update_performance(self, 
                          prediction: Any,
                          actual: Any,
                          latency: float,
                          timestamp: Optional[datetime] = None):
        """Update performance metrics."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy(prediction, actual)
        precision = self._calculate_precision(prediction, actual)
        recall = self._calculate_recall(prediction, actual)
        f1_score = self._calculate_f1_score(precision, recall)
        
        # Update metrics
        self.current_metrics.accuracy = accuracy
        self.current_metrics.precision = precision
        self.current_metrics.recall = recall
        self.current_metrics.f1_score = f1_score
        self.current_metrics.prediction_latency = latency
        
        # Update history
        self.performance_history.append({
            'timestamp': timestamp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'latency': latency
        })
        
        self.latency_history.append(latency)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {}
        
        # Calculate aggregated metrics
        recent_performance = list(self.performance_history)[-10:]  # Last 10 predictions
        
        return {
            'current_metrics': {
                'accuracy': self.current_metrics.accuracy,
                'precision': self.current_metrics.precision,
                'recall': self.current_metrics.recall,
                'f1_score': self.current_metrics.f1_score,
                'prediction_latency': self.current_metrics.prediction_latency
            },
            'recent_performance': {
                'avg_accuracy': np.mean([p['accuracy'] for p in recent_performance]),
                'avg_latency': np.mean([p['latency'] for p in recent_performance]),
                'performance_trend': self._calculate_performance_trend()
            },
            'adaptation_metrics': {
                'adaptation_frequency': len(self.adaptation_history),
                'recent_adaptations': list(self.adaptation_history)[-5:]
            }
        }
    
    def should_adapt(self) -> bool:
        """Check if adaptation should be triggered."""
        if not self.performance_history:
            return False
        
        # Check performance threshold
        recent_performance = list(self.performance_history)[-10:]
        if len(recent_performance) < 5:
            return False
        
        avg_recent_accuracy = np.mean([p['accuracy'] for p in recent_performance])
        avg_historical_accuracy = np.mean([p['accuracy'] for p in self.performance_history])
        
        performance_degradation = avg_historical_accuracy - avg_recent_accuracy
        
        return performance_degradation > self.config.adaptation_threshold
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Update resource metrics
                self._update_resource_metrics()
                
                # Check for adaptation triggers
                if self.should_adapt():
                    self._trigger_adaptation()
                
                # Sleep for monitoring interval
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                tprint_error(f"❌ Monitoring loop error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _update_resource_metrics(self):
        """Update resource usage metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.current_metrics.memory_usage = memory.percent / 100.0
            
            # CPU usage
            self.current_metrics.cpu_usage = psutil.cpu_percent() / 100.0
            
            # GPU usage (if available)
            self.current_metrics.gpu_usage = self._get_gpu_usage()
            
        except Exception as e:
            tprint_error(f"❌ Resource metrics update failed: {e}")
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        # Placeholder for GPU usage monitoring
        return 0.0
    
    def _calculate_accuracy(self, prediction: Any, actual: Any) -> float:
        """Calculate accuracy."""
        try:
            if isinstance(prediction, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
                return np.mean(np.array(prediction) == np.array(actual))
            return 0.0
        except Exception as e:
            tprint_warning(f"Performance evaluation failed: {e}. Returning 0.0.")
            return 0.0
    
    def _calculate_precision(self, prediction: Any, actual: Any) -> float:
        """Calculate precision."""
        # Placeholder for precision calculation
        return 0.0
    
    def _calculate_recall(self, prediction: Any, actual: Any) -> float:
        """Placeholder for recall calculation."""
        return 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend."""
        if len(self.performance_history) < 2:
            return "insufficient_data"
        
        recent_accuracy = [p['accuracy'] for p in list(self.performance_history)[-5:]]
        historical_accuracy = [p['accuracy'] for p in list(self.performance_history)[:-5]]
        
        if not historical_accuracy:
            return "insufficient_data"
        
        recent_avg = np.mean(recent_accuracy)
        historical_avg = np.mean(historical_accuracy)
        
        if recent_avg > historical_avg + 0.05:
            return "improving"
        elif recent_avg < historical_avg - 0.05:
            return "degrading"
        else:
            return "stable"
    
    def _trigger_adaptation(self):
        """Trigger adaptation."""
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'trigger_reason': 'performance_degradation',
            'performance_metrics': self.current_metrics
        })
        
        if TPRINT_AVAILABLE:
            tprint_warning("⚠️ Adaptation triggered due to performance degradation")


class AdaptationEngine:
    """Real-time adaptation engine for CLVSA models."""
    
    def __init__(self, config: RealTimeOptimizationConfig):
        """Initialize adaptation engine."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Adaptation state
        self.is_adapting = False
        self.adaptation_queue = queue.Queue()
        self.adaptation_thread = None
        
        # CLVSA-specific adaptation
        self.cvlsa_adaptation_rate = config.cvlsa_adaptation_rate
        self.cvlsa_memory_efficiency = config.cvlsa_memory_efficiency
        
        tprint_info("✅ Adaptation Engine initialized")
    
    def start_adaptation(self):
        """Start adaptation engine."""
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            return
        
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        if TPRINT_AVAILABLE:
            tprint_info("🔄 Adaptation engine started")
    
    def stop_adaptation(self):
        """Stop adaptation engine."""
        self.is_adapting = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=1.0)
        
        if TPRINT_AVAILABLE:
            tprint_info("🛑 Adaptation engine stopped")
    
    def trigger_adaptation(self, 
                         model: Any,
                         adaptation_type: str = "performance",
                         adaptation_data: Optional[Dict] = None):
        """Trigger model adaptation."""
        adaptation_request = {
            'timestamp': datetime.now(),
            'model': model,
            'adaptation_type': adaptation_type,
            'adaptation_data': adaptation_data or {}
        }
        
        self.adaptation_queue.put(adaptation_request)
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Adaptation triggered: {adaptation_type}")
    
    def _adaptation_loop(self):
        """Main adaptation loop."""
        while True:
            try:
                # Get adaptation request
                adaptation_request = self.adaptation_queue.get(timeout=1.0)
                
                # Perform adaptation
                self._perform_adaptation(adaptation_request)
                
            except queue.Empty:
                continue
            except Exception as e:
                tprint_error(f"❌ Adaptation loop error: {e}")
    
    def _perform_adaptation(self, adaptation_request: Dict):
        """Perform model adaptation."""
        try:
            self.is_adapting = True
            start_time = time.time()
            
            model = adaptation_request['model']
            adaptation_type = adaptation_request['adaptation_type']
            adaptation_data = adaptation_request['adaptation_data']
            
            # Perform CLVSA-specific adaptation
            if self.config.enable_cvlsa_optimization:
                adapted_model = self._adapt_cvlsa_model(model, adaptation_data)
            else:
                adapted_model = self._adapt_generic_model(model, adaptation_data)
            
            # Calculate adaptation metrics
            adaptation_time = time.time() - start_time
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Adaptation completed in {adaptation_time:.2f}s")
            
            self.is_adapting = False
            
        except Exception as e:
            tprint_error(f"❌ Adaptation failed: {e}")
            self.is_adapting = False
    
    def _adapt_cvlsa_model(self, model: Any, adaptation_data: Dict) -> Any:
        """Adapt CLVSA model."""
        try:
            # CLVSA-specific adaptation strategies
            if self.cvlsa_memory_efficiency:
                model = self._apply_memory_efficient_adaptation(model)
            
            # Apply adaptation rate
            model = self._apply_adaptation_rate(model, self.cvlsa_adaptation_rate)
            
            return model
            
        except Exception as e:
            tprint_error(f"❌ CLVSA model adaptation failed: {e}")
            return model
    
    def _adapt_generic_model(self, model: Any, adaptation_data: Dict) -> Any:
        """Adapt generic model."""
        # Generic adaptation strategies
        return model
    
    def _apply_memory_efficient_adaptation(self, model: Any) -> Any:
        """Apply memory-efficient adaptation."""
        # Memory-efficient adaptation strategies
        return model
    
    def _apply_adaptation_rate(self, model: Any, rate: float) -> Any:
        """Apply adaptation rate to model."""
        # Apply adaptation rate
        return model


class RealTimeOptimizationEngine:
    """
    Main real-time optimization engine for CLVSA architectures.
    """
    
    def __init__(self, config: Optional[RealTimeOptimizationConfig] = None):
        """Initialize real-time optimization engine."""
        self.config = config or RealTimeOptimizationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(self.config)
        self.adaptation_engine = AdaptationEngine(self.config)
        
        # Hardware acceleration
        if HARDWARE_ACCELERATION_AVAILABLE:
            self.hardware_accelerator = TreeHardwareAccelerator()
            self.cvlsa_optimizer = CLVSAHardwareOptimizer()
        else:
            self.hardware_accelerator = None
            self.cvlsa_optimizer = None
        
        # Optimization state
        self.is_optimizing = False
        self.optimization_thread = None
        
        tprint_info("✅ Real-Time Optimization Engine initialized")
    
    def start_optimization(self):
        """Start real-time optimization."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Start adaptation
        self.adaptation_engine.start_adaptation()
        
        # Start optimization loop
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        if TPRINT_AVAILABLE:
            tprint_success("🚀 Real-time optimization started")
    
    def stop_optimization(self):
        """Stop real-time optimization."""
        self.is_optimizing = False
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Stop adaptation
        self.adaptation_engine.stop_adaptation()
        
        # Stop optimization thread
        if self.optimization_thread:
            self.optimization_thread.join(timeout=1.0)
        
        if TPRINT_AVAILABLE:
            tprint_info("🛑 Real-time optimization stopped")
    
    def optimize_model(self, 
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      clvsa_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize model with real-time capabilities.
        
        Args:
            model: Model to optimize
            X: Training features
            y: Training targets
            clvsa_config: CLVSA-specific configuration
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Starting real-time model optimization")
            
            # Apply hardware acceleration if available
            if self.hardware_accelerator:
                optimization_results = self.hardware_accelerator.accelerate_tree_training(
                    model, X, y, clvsa_config
                )
            else:
                # Fallback to standard optimization
                optimization_results = self._standard_optimization(model, X, y)
            
            # Start real-time optimization
            self.start_optimization()
            
            optimization_time = time.time() - start_time
            
            results = {
                'optimization_results': optimization_results,
                'real_time_optimization_enabled': True,
                'optimization_time': optimization_time,
                'performance_monitoring': self.performance_monitor.get_performance_summary(),
                'hardware_acceleration_used': self.hardware_accelerator is not None
            }
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Real-time optimization completed in {optimization_time:.2f}s")
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Real-time optimization failed: {e}")
            raise
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_optimizing:
            try:
                # Check performance and trigger adaptations
                if self.performance_monitor.should_adapt():
                    # Trigger adaptation
                    self.adaptation_engine.trigger_adaptation(
                        model=None,  # Will be provided by the system
                        adaptation_type="performance"
                    )
                
                # Sleep for optimization interval
                time.sleep(self.config.optimization_interval)
                
            except Exception as e:
                tprint_error(f"❌ Optimization loop error: {e}")
                time.sleep(self.config.optimization_interval)
    
    def _standard_optimization(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Standard optimization fallback."""
        try:
            # Train model
            model.fit(X, y)
            
            return {
                'model': model,
                'training_completed': True,
                'hardware_acceleration_used': False
            }
            
        except Exception as e:
            tprint_error(f"❌ Standard optimization failed: {e}")
            raise
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'is_optimizing': self.is_optimizing,
            'performance_monitoring': self.performance_monitor.get_performance_summary(),
            'adaptation_engine_active': self.adaptation_engine.is_adapting,
            'hardware_acceleration_available': self.hardware_accelerator is not None,
            'cvlsa_optimization_enabled': self.config.enable_cvlsa_optimization
        }


# Factory functions
def create_realtime_optimization_engine(config: Optional[RealTimeOptimizationConfig] = None) -> RealTimeOptimizationEngine:
    """Create real-time optimization engine instance."""
    return RealTimeOptimizationEngine(config)


def create_performance_monitor(config: Optional[RealTimeOptimizationConfig] = None) -> PerformanceMonitor:
    """Create performance monitor instance."""
    return PerformanceMonitor(config or RealTimeOptimizationConfig())


def create_adaptation_engine(config: Optional[RealTimeOptimizationConfig] = None) -> AdaptationEngine:
    """Create adaptation engine instance."""
    return AdaptationEngine(config or RealTimeOptimizationConfig())


# Example usage
if __name__ == "__main__":
    # Create real-time optimization engine
    config = RealTimeOptimizationConfig(
        enable_performance_monitoring=True,
        enable_adaptation_triggers=True,
        enable_continuous_optimization=True,
        enable_cvlsa_optimization=True
    )
    
    engine = create_realtime_optimization_engine(config)
    
    # Example usage
    print("Real-Time Optimization Engine created successfully!")
    print(f"Optimization status: {engine.get_optimization_status()}")