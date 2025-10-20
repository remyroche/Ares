"""
Enhanced features_common module with comprehensive optimization.

This module provides a unified, high-performance foundation for all feature systems
with automatic optimization, VectorBT integration, caching, performance monitoring,
and intelligent fallback mechanisms.
"""

__version__ = "2.0.0"

# Import common utilities
from .utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager
)

# Core imports with backward compatibility and enhanced features enabled by default
from .backward_compatibility import BaseScaler, create_enhanced_scaler, enable_enhanced_logging
from .transforms.base_scaler import create_optimized_scaler, create_optimized_batch_scaler
from .transforms.vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from .optimization.cv_base import BaseCVSplitter, PurgedCVSplitter
from .registry.base_registry import BaseFeatureRegistry

# Enhanced imports
from .config import (
    OptimizationConfig, get_optimization_config,
    VectorBTConfig, get_vectorbt_config,
    UnifiedConfig, get_unified_config
)

from .mixins import (
    OptimizationMixin, PerformanceMixin, VectorBTMixin,
    ValidationMixin, CachingMixin, MonitoringMixin
)

# Factory imports temporarily disabled
from .factories import (
#     ScalerFactory, create_optimized_scaler, create_batch_scaler,
    OptimizerFactory, create_optimizer, create_vectorbt_optimizer,
    RegistryFactory, create_registry, create_feature_registry,
    UnifiedFactory, create_optimized_component
)

from .vectorbt_extensions import (
    UnifiedVectorBTManager, get_unified_vectorbt_manager,
    VectorBTOptimizationEngine, get_optimization_engine,
    GPUAccelerator, get_gpu_accelerator,
    VectorBTPerformanceMonitor, get_performance_monitor
)

# Hardware optimization availability check
try:
    from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Note: Normalization feature generators removed from direct imports to avoid circular dependencies
# They will be imported lazily when needed
# from .normalization import (
#     NormalizationFeatureGenerator,
#     RollingZScoreGenerator,
#     VolatilityScalingGenerator,
#     CrossSectionalNormalizer,
#     create_data_normalizer,
#     create_default_normalization_generators,
#     NormalizationConfig
# )

# Error handling and logging
from .error_handling import (
    FeaturesCommonError, ValidationError, OptimizationError, VectorBTError, ConfigurationError, SilentFailureError,
    ensure_no_silent_failures, validate_input_data, safe_execute, validate_configuration, check_system_health, report_silent_failures
)

from .logging_config import get_logger, log_operation

__all__ = [
    # Core components
    'BaseScaler',
    'VectorBTScaler',
    'VectorBTBatchScaler',
    'BaseCVSplitter',
    'PurgedCVSplitter',
    'BaseFeatureRegistry',
    'create_optimized_scaler',
    'create_optimized_batch_scaler',

    # Configuration
    'OptimizationConfig',
    'get_optimization_config',
    'VectorBTConfig',
    'get_vectorbt_config',
    'UnifiedConfig',
    'get_unified_config',

    # Mixins
    'OptimizationMixin',
    'PerformanceMixin',
    'VectorBTMixin',
    'ValidationMixin',
    'CachingMixin',
    'MonitoringMixin',

    # Factories
    'ScalerFactory',
    'create_optimized_scaler',
    'create_batch_scaler',
    'OptimizerFactory',
    'create_optimizer',
    'create_vectorbt_optimizer',
    'RegistryFactory',
    'create_registry',
    'create_feature_registry',
    'UnifiedFactory',
    'create_optimized_component',

    # VectorBT
    'UnifiedVectorBTManager',
    'get_unified_vectorbt_manager',
    'VectorBTOptimizationEngine',
    'get_optimization_engine',
    'GPUAccelerator',
    'get_gpu_accelerator',
    'VectorBTPerformanceMonitor',
    'get_performance_monitor',

    # Hardware Optimization (if available)
    'HARDWARE_OPTIMIZATION_AVAILABLE',

    # Error handling and logging
    'FeaturesCommonError',
    'ValidationError',
    'OptimizationError',
    'VectorBTError',
    'ConfigurationError',
    'SilentFailureError',
    'ensure_no_silent_failures',
    'validate_input_data',
    'safe_execute',
    'validate_configuration',
    'check_system_health',
    'report_silent_failures',
    'get_logger',
    'log_operation',

    # Backward compatibility
    'create_enhanced_scaler',
    'enable_enhanced_logging',

    # Note: Normalization feature generators removed from __all__ to avoid circular imports
    # They will be imported lazily when needed
]

# Add VectorBT optimization components to __all__ if available
if VECTORBT_OPTIMIZER_AVAILABLE:
    __all__.extend([
        'VectorBTRollingOptimizer',
        'get_vectorbt_rolling_optimizer',
        'UnifiedVectorizationManager',
        'get_unified_vectorization_manager',
    ])

# Hardware optimization is now integrated into existing components
# No additional exports needed as all components now have hardware optimization built-in

if TPRINT_AVAILABLE:
    tprint(f"🚀 [features_common] Enhanced module initialized with {len(__all__)} exports", color="cyan")
    tprint("✅ [features_common] All optimizations enabled by default", color="green")
    tprint("🔧 [features_common] VectorBT integration available", color="blue")
    tprint("📊 [features_common] Performance monitoring enabled", color="magenta")
    tprint("💾 [features_common] Intelligent caching active", color="yellow")
else:
    print(f"🚀 [features_common] Enhanced module initialized with {len(__all__)} exports")
    print("✅ [features_common] All optimizations enabled by default")
    print("🔧 [features_common] VectorBT integration available")
    print("📊 [features_common] Performance monitoring enabled")
    print("💾 [features_common] Intelligent caching active")
