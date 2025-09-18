# Hardware-specific optimization utilities
# Advanced hardware optimization system for Apple Silicon

__all__ = [
    # Core hardware optimizers
    'M1CPUOptimizer', 'M1GPUManager', 'M1MemoryOptimizer',
    
    # Advanced hardware optimizers
    'AdvancedM1CPUOptimizer', 'EnhancedM1GPUManager', 'AdvancedM1MemoryOptimizer',
    
    # Unified hardware management
    'UnifiedHardwareManager', 'HardwareConfig', 'WorkloadType', 'OptimizationLevel',
    
    # Adaptive optimization
    'AdaptiveOptimizationEngine', 'OptimizationTarget', 'LearningAlgorithm',
    
    # Convenience functions
    'get_unified_hardware_manager', 'get_advanced_cpu_optimizer', 
    'get_enhanced_gpu_manager', 'get_advanced_memory_optimizer',
    'get_adaptive_optimization_engine',
    
    # Optimization functions
    'optimize_for_workload', 'optimize_for_workload_adaptive',
    'optimize_dataframe_advanced', 'record_performance_adaptive',
    
    # Legacy compatibility
    'get_m1_cpu_optimizer', 'm1_gpu_manager', 'm1_memory_optimizer'
]

# Import core components
from .m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer
from .m1_gpu_utils import M1GPUManager, m1_gpu_manager
from .m1_memory_optimizer import M1MemoryOptimizer, m1_memory_optimizer

# Legacy compatibility aliases
m1_cpu_optimizer = get_m1_cpu_optimizer()

# Import advanced components
try:
    from .advanced_cpu_optimizer import AdvancedM1CPUOptimizer, get_advanced_cpu_optimizer
    from .enhanced_gpu_manager import EnhancedM1GPUManager, get_enhanced_gpu_manager
    from .advanced_memory_optimizer import AdvancedM1MemoryOptimizer, get_advanced_memory_optimizer
    from .unified_hardware_manager import (
        UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel,
        get_unified_hardware_manager, optimize_for_workload
    )
    from .adaptive_optimization_engine import (
        AdaptiveOptimizationEngine, OptimizationTarget, LearningAlgorithm,
        get_adaptive_optimization_engine, optimize_for_workload_adaptive,
        record_performance_adaptive
    )
    from .advanced_memory_optimizer import optimize_dataframe_advanced
    
    # Set availability flags
    ADVANCED_CPU_AVAILABLE = True
    ENHANCED_GPU_AVAILABLE = True
    ADVANCED_MEMORY_AVAILABLE = True
    UNIFIED_MANAGER_AVAILABLE = True
    ADAPTIVE_ENGINE_AVAILABLE = True
    
except ImportError as e:
    # Set availability flags to False if imports fail
    ADVANCED_CPU_AVAILABLE = False
    ENHANCED_GPU_AVAILABLE = False
    ADVANCED_MEMORY_AVAILABLE = False
    UNIFIED_MANAGER_AVAILABLE = False
    ADAPTIVE_ENGINE_AVAILABLE = False
    
    # Create placeholder functions
    def get_advanced_cpu_optimizer():
        raise ImportError("Advanced CPU Optimizer not available")
    
    def get_enhanced_gpu_manager():
        raise ImportError("Enhanced GPU Manager not available")
    
    def get_advanced_memory_optimizer():
        raise ImportError("Advanced Memory Optimizer not available")
    
    def get_unified_hardware_manager():
        raise ImportError("Unified Hardware Manager not available")
    
    def get_adaptive_optimization_engine():
        raise ImportError("Adaptive Optimization Engine not available")
    
    def optimize_for_workload(*args, **kwargs):
        raise ImportError("Unified Hardware Manager not available")
    
    def optimize_for_workload_adaptive(*args, **kwargs):
        raise ImportError("Adaptive Optimization Engine not available")
    
    def optimize_dataframe_advanced(*args, **kwargs):
        raise ImportError("Advanced Memory Optimizer not available")
    
    def record_performance_adaptive(*args, **kwargs):
        raise ImportError("Adaptive Optimization Engine not available")

# Version information
__version__ = "2.0.0"
__author__ = "Hardware Optimization Team"
__description__ = "Advanced hardware optimization system for Apple Silicon"

# Feature availability
FEATURES = {
    'basic_cpu_optimization': True,
    'basic_gpu_management': True,
    'basic_memory_optimization': True,
    'advanced_cpu_optimization': ADVANCED_CPU_AVAILABLE,
    'enhanced_gpu_acceleration': ENHANCED_GPU_AVAILABLE,
    'advanced_memory_management': ADVANCED_MEMORY_AVAILABLE,
    'unified_hardware_management': UNIFIED_MANAGER_AVAILABLE,
    'adaptive_optimization': ADAPTIVE_ENGINE_AVAILABLE
}

def get_feature_status():
    """Get status of all hardware optimization features."""
    return FEATURES.copy()

def get_available_features():
    """Get list of available features."""
    return [feature for feature, available in FEATURES.items() if available]

def is_feature_available(feature_name: str) -> bool:
    """Check if a specific feature is available."""
    return FEATURES.get(feature_name, False)
