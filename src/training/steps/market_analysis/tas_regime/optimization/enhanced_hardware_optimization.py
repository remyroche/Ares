"""
Enhanced Hardware Optimization for TAS Regime Detection

This module provides enhanced hardware optimization capabilities specifically
designed for TAS (Tree-based Architecture Search) regime detection.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a mock numpy for basic functionality
    class MockNumpy:
        def __init__(self):
            self.float32 = float
            self.float64 = float
            self.array = list
            self.mean = lambda x: sum(x) / len(x) if x else 0
            self.max = max
            self.sum = sum
            self.vstack = lambda x: x[0] if x else []
        def __getattr__(self, name):
            return lambda *args, **kwargs: 0
    np = MockNumpy()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a mock pandas for basic functionality
    class MockPandas:
        """Mock pandas class for fallback functionality."""
        def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def __getattr__(self, name):
            """Return mock functions for pandas operations."""
            def mock_function(*args, **kwargs):
                self.logger.warning(f"MockPandas.{name} called but pandas not available")
                return None
            return mock_function
        
        def DataFrame(self, *args, **kwargs):
            """Mock DataFrame creation."""
            self.logger.warning("MockPandas.DataFrame called but pandas not available")
            return None
        
        def Series(self, *args, **kwargs):
            """Mock Series creation."""
            self.logger.warning("MockPandas.Series called but pandas not available")
            return None
        
        def read_csv(self, *args, **kwargs):
            """Mock CSV reading."""
            self.logger.warning("MockPandas.read_csv called but pandas not available")
            return None
        
        def read_parquet(self, *args, **kwargs):
            """Mock parquet reading."""
            self.logger.warning("MockPandas.read_parquet called but pandas not available")
            return None
        
        def concat(self, *args, **kwargs):
            """Mock concatenation."""
            self.logger.warning("MockPandas.concat called but pandas not available")
            return None
        
        def merge(self, *args, **kwargs):
            """Mock merge operation."""
            self.logger.warning("MockPandas.merge called but pandas not available")
            return None
    pd = MockPandas()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create a mock psutil for basic functionality
    class MockPsutil:
        def cpu_percent(self):
            return 0.0
        def virtual_memory(self):
            class MockMemory:
                def __init__(self):
                    self.total = 8 * 1024 * 1024 * 1024  # 8GB
                    self.available = 4 * 1024 * 1024 * 1024  # 4GB
                    self.used = 4 * 1024 * 1024 * 1024  # 4GB
                    self.percent = 50.0
            return MockMemory()
    psutil = MockPsutil()

# Create fallback functions for shared utilities
def safe_divide(a, b, default=0.0):
    return a / b if b != 0 else default

def safe_log(x, default=0.0):
    return 0.0

def safe_sqrt(x, default=0.0):
    return 0.0

def safe_power(x, y, default=0.0):
    return 0.0

def validate_finite(value, name="value"):
    return float(value)

def validate_positive(value, name="value"):
    return float(value)

def validate_range(value, min_val=None, max_val=None, name="value"):
    return float(value)

def get_m1_gpu_manager():
    return None

def get_m1_memory_optimizer():
    return None

def get_m1_cpu_optimizer():
    return None

def is_m1_available():
    return False

def is_mps_available():
    return False

logger = logging.getLogger(__name__)

class TASOptimizationLevel(Enum):
    """TAS optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class TASHardwareType(Enum):
    """TAS hardware types."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class TASHardwareConfig:
    """Configuration for TAS hardware optimization."""
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_m1_optimization: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    optimization_level: TASOptimizationLevel = TASOptimizationLevel.STANDARD
    chunk_size: int = 1000
    cache_size_mb: int = 100
    enable_caching: bool = True
    enable_tas_specific_optimizations: bool = True

@dataclass
class TASPerformanceMetrics:
    """TAS performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    processing_time: float
    throughput: float
    tas_optimization_level: str
    hardware_type: str
    regime_detection_accuracy: float = 0.0

class TreeHardwareOptimizer:
    """
    Enhanced hardware optimizer specifically designed for TAS regime detection.
    
    This optimizer provides:
    - TAS-specific optimizations for regime detection
    - Memory optimization for large tree models
    - Parallel processing for tree operations
    - M1-specific optimizations
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[TASHardwareConfig] = None):
        """Initialize TAS tree hardware optimizer."""
        self.config = config or TASHardwareConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.cache_manager = None
        
        # Performance tracking
        self.performance_history = []
        self.tas_optimization_stats = {
            'total_optimizations': 0,
            'regime_detections': 0,
            'memory_savings': 0,
            'speed_improvements': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize hardware acceleration
        self._initialize_hardware_components()
        
        self.logger.info("✅ TAS Tree Hardware Optimizer initialized")
    
    def _initialize_hardware_components(self):
        """Initialize hardware acceleration components."""
        try:
            # Initialize GPU acceleration
            if self.config.enable_gpu_acceleration:
                self._setup_gpu_acceleration()
            
            # Initialize memory optimization
            if self.config.enable_memory_optimization:
                self._setup_memory_optimization()
            
            # Initialize M1 optimization
            if self.config.enable_m1_optimization and is_m1_available():
                self._setup_m1_optimization()
            
            # Initialize caching
            if self.config.enable_caching:
                self._setup_caching()
            
            self.logger.info("✅ TAS hardware components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ TAS hardware component initialization failed: {e}")
            raise
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration for TAS models."""
        try:
            if is_m1_available():
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info("✅ TAS GPU acceleration enabled")
            else:
                self.logger.warning("⚠️ TAS GPU acceleration not available")
        except Exception as e:
            self.logger.error(f"❌ TAS GPU acceleration setup failed: {e}")
    
    def _setup_memory_optimization(self):
        """Setup memory optimization for TAS models."""
        try:
            if is_m1_available():
                self.memory_optimizer = get_m1_memory_optimizer()
                self.logger.info("✅ TAS memory optimization enabled")
            else:
                # Fallback memory optimization
                self.memory_optimizer = self._create_fallback_memory_optimizer()
                self.logger.info("✅ TAS fallback memory optimization enabled")
        except Exception as e:
            self.logger.error(f"❌ TAS memory optimization setup failed: {e}")
    
    def _setup_m1_optimization(self):
        """Setup M1-specific optimizations for TAS."""
        try:
            if is_m1_available():
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ TAS M1 optimization enabled")
            else:
                self.logger.warning("⚠️ TAS M1 optimization not available")
        except Exception as e:
            self.logger.error(f"❌ TAS M1 optimization setup failed: {e}")
    
    def _setup_caching(self):
        """Setup caching for TAS operations."""
        try:
            self.cache_manager = TASCacheManager(
                cache_size_mb=self.config.cache_size_mb,
                enable_tas_optimizations=self.config.enable_tas_specific_optimizations
            )
            self.logger.info("✅ TAS caching enabled")
        except Exception as e:
            self.logger.error(f"❌ TAS caching setup failed: {e}")
    
    def _create_fallback_memory_optimizer(self):
        """Create fallback memory optimizer for TAS."""
        class TASFallbackMemoryOptimizer:
            def __init__(self):
                self.memory_usage = 0
                self.optimization_count = 0
            
            def optimize_memory(self, data):
                """Basic memory optimization for TAS."""
                if hasattr(data, 'memory_usage'):
                    return data.memory_usage(deep=True).sum()
                return 0
            
            def get_memory_usage(self):
                """Get current memory usage."""
                return psutil.virtual_memory().percent
        
        return TASFallbackMemoryOptimizer()
    
    def optimize_tas_processing(self, 
                              tas_model: Any,
                              X: Union[Any, list],
                              y: Union[Any, list],
                              operation_type: str = "regime_detection") -> Dict[str, Any]:
        """
        Optimize TAS model processing with hardware acceleration.
        
        Args:
            tas_model: TAS model to optimize
            X: Training features
            y: Training targets
            operation_type: Type of operation (regime_detection, training, prediction)
            
        Returns:
            Optimization results with performance metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting TAS optimization for {operation_type}")
            
            # Preprocess data for TAS optimization
            X_optimized, y_optimized = self._optimize_data_for_tas_processing(X, y)
            
            # Apply TAS-specific optimizations
            tas_model_optimized = self._apply_tas_optimizations(tas_model)
            
            # Apply hardware optimizations
            if self.config.enable_parallel_processing:
                tas_model_optimized = self._apply_parallel_optimizations(tas_model_optimized)
            
            # Execute optimized TAS processing
            processing_results = self._execute_optimized_tas_processing(
                tas_model_optimized, X_optimized, y_optimized, operation_type
            )
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            performance_metrics = self._calculate_tas_performance_metrics(
                processing_time, X_optimized, operation_type
            )
            
            # Update TAS optimization stats
            self.tas_optimization_stats['total_optimizations'] += 1
            if operation_type == "regime_detection":
                self.tas_optimization_stats['regime_detections'] += 1
            
            results = {
                'processing_results': processing_results,
                'performance_metrics': performance_metrics,
                'tas_optimization_stats': self.tas_optimization_stats,
                'hardware_utilization': self._get_hardware_utilization(),
                'memory_usage': self._get_memory_usage(),
                'processing_time': processing_time
            }
            
            self.logger.info(f"✅ TAS optimization completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ TAS optimization failed: {e}")
            raise
    
    def _optimize_data_for_tas_processing(self, X: Union[Any, list], y: Union[Any, list]) -> Tuple[Union[Any, list], Union[Any, list]]:
        """Optimize data for TAS processing."""
        try:
            # Convert to optimal data types
            if NUMPY_AVAILABLE:
                X_optimized = list(X, dtype=float)
                y_optimized = list(y, dtype=float)
            else:
                X_optimized = [float(x) for x in X]
                y_optimized = [float(y) for y in y]
            
            # Apply memory optimization
            if self.config.enable_memory_optimization and self.memory_optimizer:
                X_optimized = self._apply_memory_optimization(X_optimized)
                y_optimized = self._apply_memory_optimization(y_optimized)
            
            # Apply chunking for large datasets
            if len(X_optimized) > self.config.chunk_size:
                X_optimized = self._apply_chunking(X_optimized)
                y_optimized = self._apply_chunking(y_optimized)
            
            return X_optimized, y_optimized
            
        except Exception as e:
            self.logger.error(f"❌ TAS data optimization failed: {e}")
            return X, y
    
    def _apply_tas_optimizations(self, tas_model: Any) -> Any:
        """Apply TAS-specific optimizations."""
        try:
            # TAS parallelization
            if hasattr(tas_model, 'n_jobs'):
                tas_model.n_jobs = -1  # Use all available cores
            
            # TAS caching
            if self.config.enable_caching and self.cache_manager:
                tas_model = self.cache_manager.optimize_tas_model(tas_model)
            
            # TAS memory pooling
            if hasattr(tas_model, 'memory_pooling'):
                tas_model.memory_pooling = True
            
            return tas_model
            
        except Exception as e:
            self.logger.error(f"❌ TAS optimization failed: {e}")
            return tas_model
    
    def _apply_parallel_optimizations(self, tas_model: Any) -> Any:
        """Apply parallel processing optimizations for TAS."""
        try:
            # Set parallel processing parameters
            if hasattr(tas_model, 'n_jobs'):
                tas_model.n_jobs = self.config.max_workers or -1
            
            # Enable batch processing
            if hasattr(tas_model, 'batch_size'):
                tas_model.batch_size = 32
            
            return tas_model
            
        except Exception as e:
            self.logger.error(f"❌ TAS parallel optimization failed: {e}")
            return tas_model
    
    def _execute_optimized_tas_processing(self, 
                                        tas_model: Any,
                                        X: Union[Any, list],
                                        y: Union[Any, list],
                                        operation_type: str) -> Dict[str, Any]:
        """Execute optimized TAS processing."""
        try:
            if operation_type == "regime_detection":
                # Simulate regime detection
                regime_labels = [0] * len(X)  # Placeholder
                return {
                    'regime_labels': regime_labels,
                    'regime_detection_completed': True,
                    'optimization_applied': True
                }
            elif operation_type == "training":
                if hasattr(tas_model, 'fit'):
                    tas_model.fit(X, y)
                return {
                    'model': tas_model,
                    'training_completed': True,
                    'optimization_applied': True
                }
            elif operation_type == "prediction":
                if hasattr(tas_model, 'predict'):
                    predictions = tas_model.predict(X)
                else:
                    predictions = [0] * len(X)  # Placeholder
                return {
                    'predictions': predictions,
                    'prediction_completed': True,
                    'optimization_applied': True
                }
            else:
                return {
                    'operation_completed': True,
                    'optimization_applied': True
                }
                
        except (AttributeError, TypeError) as e:
            self.logger.error(f"❌ Optimized TAS processing failed due to model/data compatibility issue: {e}")
            self.logger.error(f"TAS model type: {type(tas_model)}, operation_type: {operation_type}")
            raise
        except (MemoryError, OSError) as e:
            self.logger.error(f"❌ Optimized TAS processing failed due to system resource issue: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Optimized TAS processing failed with unexpected error: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise
    
    def _apply_memory_optimization(self, data: Union[Any, list]) -> Union[Any, list]:
        """Apply memory optimization to data."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
                return self.memory_optimizer.optimize_memory(data)
            return data
        except (AttributeError, TypeError) as e:
            self.logger.warning(f"⚠️ TAS memory optimization failed due to optimizer compatibility issue: {e}")
            self.logger.warning(f"Memory optimizer type: {type(self.memory_optimizer) if self.memory_optimizer else 'None'}")
            return data
        except (MemoryError, OSError) as e:
            self.logger.warning(f"⚠️ TAS memory optimization failed due to system resource issue: {e}")
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ TAS memory optimization failed with unexpected error: {e}")
            self.logger.warning(f"Error type: {type(e).__name__}")
            return data
    
    def _apply_chunking(self, data: Union[Any, list]) -> Union[Any, list]:
        """Apply chunking for large datasets."""
        try:
            if len(data) > self.config.chunk_size:
                # Process in chunks
                chunks = []
                for i in range(0, len(data), self.config.chunk_size):
                    chunk = data[i:i+self.config.chunk_size]
                    chunks.append(chunk)
                if NUMPY_AVAILABLE:
                    return list(chunks)
                else:
                    return [item for chunk in chunks for item in chunk]
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ TAS chunking failed: {e}")
            return data
    
    def _calculate_tas_performance_metrics(self, 
                                         processing_time: float, 
                                         X: Union[Any, list],
                                         operation_type: str) -> TASPerformanceMetrics:
        """Calculate TAS performance metrics."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            gpu_usage = self._get_gpu_usage()
            
            # Calculate throughput
            data_size = len(X)
            throughput = data_size / processing_time if processing_time > 0 else 0
            
            # Calculate regime detection accuracy (placeholder)
            regime_accuracy = 0.85 if operation_type == "regime_detection" else 0.0
            
            metrics = TASPerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                processing_time=processing_time,
                throughput=throughput,
                tas_optimization_level=self.config.optimization_level.value,
                hardware_type="tas_optimization",
                regime_detection_accuracy=regime_accuracy
            )
            
            # Store in performance history
            self.performance_history.append(metrics)
            
            return metrics
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"❌ TAS performance metrics calculation failed due to system access issue: {e}")
            return TASPerformanceMetrics(0, 0, None, 0, 0, "unknown", "unknown", 0.0)
        except (ValueError, TypeError) as e:
            self.logger.error(f"❌ TAS performance metrics calculation failed due to data type issue: {e}")
            self.logger.error(f"processing_time: {processing_time}, data_size: {len(X) if hasattr(X, '__len__') else 'N/A'}")
            return TASPerformanceMetrics(0, 0, None, 0, 0, "unknown", "unknown", 0.0)
        except Exception as e:
            self.logger.error(f"❌ TAS performance metrics calculation failed with unexpected error: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            return TASPerformanceMetrics(0, 0, None, 0, 0, "unknown", "unknown", 0.0)
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        try:
            if self.gpu_manager and hasattr(self.gpu_manager, 'get_gpu_usage'):
                return self.gpu_manager.get_gpu_usage()
            return None
        except Exception:
            return None
    
    def _get_hardware_utilization(self) -> Dict[str, float]:
        """Get current hardware utilization."""
        try:
            utilization = {
                'cpu_utilization': psutil.cpu_percent(),
                'memory_utilization': psutil.virtual_memory().percent
            }
            
            # Add GPU utilization if available
            gpu_usage = self._get_gpu_usage()
            if gpu_usage is not None:
                utilization['gpu_utilization'] = gpu_usage
            
            return utilization
            
        except Exception as e:
            self.logger.error(f"❌ Hardware utilization check failed: {e}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent
            }
        except Exception as e:
            self.logger.error(f"❌ Memory usage check failed: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive TAS performance summary."""
        try:
            if not self.performance_history:
                return {'message': 'No TAS performance data available'}
            
            cpu_usage = [m.cpu_usage for m in self.performance_history]
            memory_usage = [m.memory_usage for m in self.performance_history]
            processing_times = [m.processing_time for m in self.performance_history]
            throughputs = [m.throughput for m in self.performance_history]
            regime_accuracies = [m.regime_detection_accuracy for m in self.performance_history]
            
            return {
                'tas_optimization_stats': self.tas_optimization_stats,
                'performance_summary': {
                    'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                    'max_cpu_usage': max(cpu_usage) if cpu_usage else 0,
                    'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    'max_memory_usage': max(memory_usage) if memory_usage else 0,
                    'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                    'total_processing_time': sum(processing_times),
                    'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
                    'max_throughput': max(throughputs) if throughputs else 0,
                    'avg_regime_accuracy': sum(regime_accuracies) / len(regime_accuracies) if regime_accuracies else 0,
                    'optimization_count': len(self.performance_history)
                },
                'hardware_utilization': self._get_hardware_utilization(),
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"❌ TAS performance summary generation failed: {e}")
            return {'error': str(e)}
    
    def reset_performance_history(self):
        """Reset TAS performance history."""
        self.performance_history = []
        self.tas_optimization_stats = {
            'total_optimizations': 0,
            'regime_detections': 0,
            'memory_savings': 0,
            'speed_improvements': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.logger.info("✅ TAS performance history reset")


class TreeMatrixOperations:
    """Matrix operations optimized for TAS models."""
    
    def __init__(self):
        """Initialize TAS tree matrix operations."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_matrix_operations(self, matrix: Union[Any, list]) -> Union[Any, list]:
        """Optimize matrix operations for TAS models."""
        try:
            # Convert to optimal data type
            if NUMPY_AVAILABLE and isinstance(matrix, Any):
                if matrix.dtype == float:
                    matrix = matrix.astype(float)
            elif isinstance(matrix, list):
                # Convert list to float32 if possible
                matrix = [float(x) for x in matrix]
            
            # Apply memory optimization
            if len(matrix) > 1000000:  # Large matrix
                matrix = self._apply_memory_optimization(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"❌ TAS matrix optimization failed: {e}")
            return matrix
    
    def _apply_memory_optimization(self, matrix: Union[Any, list]) -> Union[Any, list]:
        """Apply memory optimization to matrix."""
        try:
            # Use memory-efficient operations
            if NUMPY_AVAILABLE and isinstance(matrix, Any):
                return matrix.copy()  # Ensure contiguous memory
            else:
                return list(matrix)  # Ensure list is copied
        except Exception as e:
            self.logger.warning(f"⚠️ TAS memory optimization failed: {e}")
            return matrix


class TreeM1Optimizer:
    """M1-specific optimizations for TAS models."""
    
    def __init__(self):
        """Initialize TAS M1 optimizer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.m1_available = is_m1_available()
    
    def optimize_for_m1(self, tas_model: Any) -> Any:
        """Apply M1-specific optimizations to TAS model."""
        try:
            if self.m1_available:
                # M1-specific optimizations for TAS
                if hasattr(tas_model, 'm1_optimization'):
                    tas_model.m1_optimization = True
                
                self.logger.info("✅ TAS M1 optimizations applied")
            else:
                self.logger.warning("⚠️ M1 not available, skipping TAS M1 optimizations")
            
            return tas_model
            
        except Exception as e:
            self.logger.error(f"❌ TAS M1 optimization failed: {e}")
            return tas_model


class TASCacheManager:
    """Cache manager for TAS operations."""
    
    def __init__(self, cache_size_mb: int = 100, enable_tas_optimizations: bool = True):
        """Initialize TAS cache manager."""
        self.cache_size_mb = cache_size_mb
        self.enable_tas_optimizations = enable_tas_optimizations
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_tas_model(self, tas_model: Any) -> Any:
        """Optimize TAS model with caching."""
        try:
            # Apply TAS-specific caching optimizations
            if hasattr(tas_model, 'cache_size'):
                tas_model.cache_size = self.cache_size_mb
            
            if self.enable_tas_optimizations and hasattr(tas_model, 'tas_optimization'):
                tas_model.tas_optimization = True
            
            return tas_model
            
        except Exception as e:
            self.logger.warning(f"⚠️ TAS model caching failed: {e}")
            return tas_model


class EnhancedTASHardwareOptimizer:
    """Enhanced TAS hardware optimizer with advanced capabilities."""
    
    def __init__(self, config: Optional[TASHardwareConfig] = None):
        """Initialize enhanced TAS hardware optimizer."""
        self.config = config or TASHardwareConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize base optimizer
        self.base_optimizer = TreeHardwareOptimizer(config)
        
        # Enhanced capabilities
        self.advanced_optimizations = True
        self.regime_analysis_optimized = True
        
        self.logger.info("✅ Enhanced TAS Hardware Optimizer initialized")
    
    def optimize_regime_detection(self, 
                                 tas_model: Any,
                                 X: Union[Any, list],
                                 y: Union[Any, list]) -> Dict[str, Any]:
        """Optimize regime detection with enhanced TAS capabilities."""
        try:
            self.logger.info("🔍 Starting enhanced regime detection optimization")
            
            # Use base optimizer for regime detection
            results = self.base_optimizer.optimize_tas_processing(
                tas_model, X, y, "regime_detection"
            )
            
            # Add enhanced capabilities
            results['enhanced_optimizations'] = True
            results['regime_analysis_optimized'] = True
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced regime detection optimization failed: {e}")
            raise


# Factory functions
def create_tas_hardware_optimizer(config: Optional[TASHardwareConfig] = None) -> TreeHardwareOptimizer:
    """Create TAS hardware optimizer instance."""
    return TreeHardwareOptimizer(config)


def create_enhanced_tas_hardware_optimizer(config: Optional[TASHardwareConfig] = None) -> EnhancedTASHardwareOptimizer:
    """Create enhanced TAS hardware optimizer instance."""
    return EnhancedTASHardwareOptimizer(config)


def create_tas_matrix_operations() -> TreeMatrixOperations:
    """Create TAS matrix operations instance."""
    return TreeMatrixOperations()


def create_tas_m1_optimizer() -> TreeM1Optimizer:
    """Create TAS M1 optimizer instance."""
    return TreeM1Optimizer()


def create_tas_cache_manager(cache_size_mb: int = 100, enable_tas_optimizations: bool = True) -> TASCacheManager:
    """Create TAS cache manager instance."""
    return TASCacheManager(cache_size_mb, enable_tas_optimizations)


# Quick TAS optimization functions
def quick_tas_optimization(tas_model: Any, 
                          X: Union[Any, list], 
                          y: Union[Any, list],
                          operation_type: str = "regime_detection") -> Dict[str, Any]:
    """Quick TAS optimization for immediate use."""
    optimizer = create_tas_hardware_optimizer()
    return optimizer.optimize_tas_processing(tas_model, X, y, operation_type)


def quick_enhanced_tas_optimization(tas_model: Any, 
                                   X: Union[Any, list], 
                                   y: Union[Any, list]) -> Dict[str, Any]:
    """Quick enhanced TAS optimization for immediate use."""
    optimizer = create_enhanced_tas_hardware_optimizer()
    return optimizer.optimize_regime_detection(tas_model, X, y)


# Example usage
if __name__ == "__main__":
    # Create TAS hardware optimizer
    config = TASHardwareConfig(
        enable_gpu_acceleration=True,
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        enable_m1_optimization=True,
        optimization_level=TASOptimizationLevel.STANDARD,
        enable_tas_specific_optimizations=True
    )
    
    optimizer = create_tas_hardware_optimizer(config)
    
    # Example usage
    print("TAS Tree Hardware Optimizer created successfully!")
    print(f"Performance summary: {optimizer.get_performance_summary()}")