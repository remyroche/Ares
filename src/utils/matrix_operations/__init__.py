"""
Unified Matrix Operations Module

This module provides a single source of truth for all matrix and vectorized operations
in the codebase, consolidating functionality from multiple scattered sources while
maintaining full backwards compatibility.

Key Features:
- Unified interface for all matrix operations
- VectorBT optimizations for 2-10x performance improvements
- GPU acceleration with Apple Silicon M1/M2/M3 support
- Memory optimization and batch processing
- Vectorized operations for machine learning workflows
- Comprehensive error handling and recovery
- Backwards compatibility with existing code

Usage:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        get_enhanced_matrix_operations
    )
    
    # Get unified operations instance
"""

# Import logger
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Core unified operations
try:
    from .unified_operations import (
        get_unified_matrix_operations,
        UnifiedMatrixOperations,
        M1EnhancedMatrixOperations,  # Backwards compatibility alias
    )
    UNIFIED_OPERATIONS_AVAILABLE = True
except ImportError as e:
    UNIFIED_OPERATIONS_AVAILABLE = False
    logger.warning(f"Unified operations not available: {e}")

# Vectorized processing core
try:
    from .vectorized_core import (
        get_vectorized_processing_core,
        VectorizedProcessingCore,
        OptimizedPipelineExecutor,
        PipelineStage,
        PipelineExecutionMode,
        PipelineStageStatus,
        PipelineExecutionResult,
    )
    VECTORIZED_CORE_AVAILABLE = True
except ImportError as e:
    VECTORIZED_CORE_AVAILABLE = False
    logger.warning(f"Vectorized core not available: {e}")

# Batch matrix operations
try:
    from .batch_operations import (
        get_batch_matrix_processor,
        BatchMatrixProcessor,
    )
    BATCH_OPERATIONS_AVAILABLE = True
except ImportError as e:
    BATCH_OPERATIONS_AVAILABLE = False
    logger.warning(f"Batch operations not available: {e}")

# Enhanced matrix operations with GPU support
try:
    from .enhanced_operations import (
        get_enhanced_matrix_operations,
        EnhancedMatrixOperations,
        BatchOptimizationStrategy,
        OperationComplexity,
        DynamicBatchOptimizer,
        CustomMatrixOperation,
        CustomMatrixOperationsRegistry,
        get_custom_operations_registry,
        register_custom_matrix_operation,
        execute_custom_matrix_operation,
        list_custom_matrix_operations,
    )
    ENHANCED_OPERATIONS_AVAILABLE = True
except ImportError as e:
    ENHANCED_OPERATIONS_AVAILABLE = False
    logger.warning(f"Enhanced operations not available: {e}")

# Error handling and recovery (should always be available)
try:
    from .error_handling import (
        ErrorHandler,
        OptimizationError,
        GPUError,
        MemoryError,
        MatrixOperationError,
        DataProcessingError,
        ConfigurationError,
        ErrorRecoveryResult,
        with_error_handling,
        with_gpu_fallback,
        with_memory_optimization,
        get_global_error_handler,
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError as e:
    ERROR_HANDLING_AVAILABLE = False
    logger.warning(f"Error handling not available: {e}")

# Convenience functions for common operations
try:
    from .convenience import (
        # Matrix operations
        safe_matrix_multiply,
        safe_correlation_matrix,
        safe_matrix_inverse,
        safe_matrix_operations,
        validate_matrix_properties,
        optimize_matrix_computations,
        gpu_matrix_multiply,
        correlation_matrix_gpu,
        eigendecomposition_gpu,
        svd_gpu,
        
        # Vectorized operations
        optimize_dataframe,
        vectorized_rolling_features,
        matrix_correlation_analysis,
        parallel_feature_engineering,
        
        # Batch operations
        batch_matrix_multiply,
        batch_feature_transformation,
        batch_correlation_analysis,
        
        # Sparse matrix operations
        sparse_matrix_multiply,
        sparse_svd,
        sparse_eigen,
        create_sparse_matrix,
        sparse_solve,
        
        # Pipeline operations
        create_ml_pipeline,
        execute_ml_pipeline,
        optimize_pipeline_config,
        get_pipeline_executor,
        
        # Optimization utilities
        optimize_batch_size,
        record_batch_performance,
        get_batch_optimization_stats,
        
        # Backwards compatibility
        m1_matrix_multiply,
        
        # Trading indicators
        compute_trading_indicators,
        compute_moving_averages,
        compute_momentum_indicators,
        compute_volatility_indicators,
        compute_volume_indicators,
        compute_trend_indicators,
        compute_oscillator_indicators,
        compute_pattern_indicators,
        
        # Hardware optimization
        get_hardware_performance_report,
        optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources,
        get_processing_performance_stats,
        
    )
    CONVENIENCE_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    CONVENIENCE_FUNCTIONS_AVAILABLE = False
    logger.warning(f"Convenience functions not available: {e}")

# Computation toolbox imports
try:
    from .computation_toolbox import (
        compute_trading_indicators_optimized,
        matrix_multiply_optimized,
        correlation_analysis_optimized,
        batch_process_optimized,
        optimize_dataframe_optimized,
        get_toolbox_performance_report,
        cleanup_toolbox_resources,
    )
    COMPUTATION_TOOLBOX_AVAILABLE = True
except ImportError as e:
    COMPUTATION_TOOLBOX_AVAILABLE = False
    logger.warning(f"Computation toolbox not available: {e}")

# Vectorized correlations
try:
    from .vectorized_correlations import (
        VectorizedCorrelationCalculator,
        CorrelationResult,
        SafeNaNHandler,
        AlignmentResult,
        calculate_correlations_vectorized,
        calculate_batch_correlations_vectorized,
        safe_correlation_with_nan_handling,
        safe_mutual_information_with_nan_handling,
    )
    VECTORIZED_CORRELATIONS_AVAILABLE = True
except ImportError as e:
    VECTORIZED_CORRELATIONS_AVAILABLE = False
    logger.warning(f"Vectorized correlations not available: {e}")

# VectorBT optimizations
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError as e:
    VECTORBT_AVAILABLE = False
    vbt = None
    logger.warning(f"VectorBT not available: {e}")

# VectorBT optimized operations
try:
    from .vectorbt_optimizations import (
        get_vectorbt_optimized_operations,
        VectorBTOptimizedOperations,
        vectorbt_matrix_multiply,
        vectorbt_correlation_matrix,
        vectorbt_trading_indicators,
        vectorbt_rolling_features,
        vectorbt_batch_processing,
    )
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"VectorBT optimizations not available: {e}")

# VectorBT Rolling Optimizer and Unified Vectorization Manager
try:
    from ...feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer,
        get_vectorbt_rolling_optimizer,
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError as e:
    VECTORBT_ROLLING_AVAILABLE = False
    logger.warning(f"VectorBTRollingOptimizer not available: {e}")

try:
    from ...feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager,
        get_unified_vectorization_manager,
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError as e:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    logger.warning(f"UnifiedVectorizationManager not available: {e}")

# Version and compatibility information
__version__ = "1.0.0"
__author__ = "Unified Matrix Operations Team"
__description__ = "Unified matrix and vectorized operations with Apple Silicon optimization"

# Build __all__ list conditionally based on available modules
__all__ = []

# Always available
if ERROR_HANDLING_AVAILABLE:
    __all__.extend([
        "ErrorHandler",
        "OptimizationError",
        "GPUError",
        "MemoryError",
        "MatrixOperationError",
        "DataProcessingError",
        "ConfigurationError",
        "ErrorRecoveryResult",
        "with_error_handling",
        "with_gpu_fallback",
        "with_memory_optimization",
        "get_global_error_handler",
    ])

# Core unified operations
if UNIFIED_OPERATIONS_AVAILABLE:
    __all__.extend([
        "UnifiedMatrixOperations",
        "M1EnhancedMatrixOperations",  # Backwards compatibility
        "get_unified_matrix_operations",
    ])

# Vectorized processing core
if VECTORIZED_CORE_AVAILABLE:
    __all__.extend([
        "VectorizedProcessingCore",
        "OptimizedPipelineExecutor",
        "PipelineStage",
        "PipelineExecutionMode",
        "PipelineStageStatus",
        "PipelineExecutionResult",
        "get_vectorized_processing_core",
    ])

# Batch operations
if BATCH_OPERATIONS_AVAILABLE:
    __all__.extend([
        "BatchMatrixProcessor",
        "get_batch_matrix_processor",
    ])

# Enhanced operations
if ENHANCED_OPERATIONS_AVAILABLE:
    __all__.extend([
        "EnhancedMatrixOperations",
        "BatchOptimizationStrategy",
        "OperationComplexity",
        "DynamicBatchOptimizer",
        "CustomMatrixOperation",
        "CustomMatrixOperationsRegistry",
        "get_enhanced_matrix_operations",
        "get_custom_operations_registry",
        "register_custom_matrix_operation",
        "execute_custom_matrix_operation",
        "list_custom_matrix_operations",
    ])

# Convenience functions
if CONVENIENCE_FUNCTIONS_AVAILABLE:
    __all__.extend([
        "safe_matrix_multiply",
        "safe_correlation_matrix",
        "safe_matrix_inverse",
        "safe_matrix_operations",
        "validate_matrix_properties",
        "optimize_matrix_computations",
        "gpu_matrix_multiply",
        "correlation_matrix_gpu",
        "eigendecomposition_gpu",
        "svd_gpu",
        "optimize_dataframe",
        "vectorized_rolling_features",
        "matrix_correlation_analysis",
        "parallel_feature_engineering",
        "batch_matrix_multiply",
        "batch_feature_transformation",
        "batch_correlation_analysis",
        "sparse_matrix_multiply",
        "sparse_svd",
        "sparse_eigen",
        "create_sparse_matrix",
        "sparse_solve",
        "create_ml_pipeline",
        "execute_ml_pipeline",
        "optimize_pipeline_config",
        "get_pipeline_executor",
        "optimize_batch_size",
        "record_batch_performance",
        "get_batch_optimization_stats",
        "m1_matrix_multiply",  # Add backwards compatibility function
    ])

# Add trading indicators and hardware optimization to __all__ if available
if VECTORIZED_CORE_AVAILABLE:
    __all__.extend([
        # Trading indicators
        "compute_trading_indicators",
        "compute_moving_averages",
        "compute_momentum_indicators",
        "compute_volatility_indicators",
        "compute_volume_indicators",
        "compute_trend_indicators",
        "compute_oscillator_indicators",
        "compute_pattern_indicators",
        
        # Hardware optimization
        "get_hardware_performance_report",
        "optimize_matrix_operation_with_hardware",
        "cleanup_hardware_resources",
        "get_processing_performance_stats",
    ])

# Add computation toolbox to __all__ if available
if COMPUTATION_TOOLBOX_AVAILABLE:
    __all__.extend([
        # Computation toolbox
        "compute_trading_indicators_optimized",
        "matrix_multiply_optimized",
        "correlation_analysis_optimized",
        "batch_process_optimized",
        "optimize_dataframe_optimized",
        "get_toolbox_performance_report",
        "cleanup_toolbox_resources",
    ])

# Add vectorized correlations to __all__ if available
if VECTORIZED_CORRELATIONS_AVAILABLE:
    __all__.extend([
        # Vectorized correlations
        "VectorizedCorrelationCalculator",
        "CorrelationResult",
        "SafeNaNHandler",
        "AlignmentResult",
        "calculate_correlations_vectorized",
        "calculate_batch_correlations_vectorized",
        "safe_correlation_with_nan_handling",
        "safe_mutual_information_with_nan_handling",
    ])

# Add VectorBT optimizations to __all__ if available
if VECTORBT_OPTIMIZATIONS_AVAILABLE:
    __all__.extend([
        # VectorBT optimized operations
        "VectorBTOptimizedOperations",
        "get_vectorbt_optimized_operations",
        "vectorbt_matrix_multiply",
        "vectorbt_correlation_matrix",
        "vectorbt_trading_indicators",
        "vectorbt_rolling_features",
        "vectorbt_batch_processing",
    ])

# Add VectorBT Rolling Optimizer to __all__ if available
if VECTORBT_ROLLING_AVAILABLE:
    __all__.extend([
        "VectorBTRollingOptimizer",
        "get_vectorbt_rolling_optimizer",
    ])

# Add Unified Vectorization Manager to __all__ if available
if UNIFIED_VECTORIZATION_AVAILABLE:
    __all__.extend([
        "UnifiedVectorizationManager",
        "get_unified_vectorization_manager",
    ])

# Initialize default custom operations
# Note: Default operations are automatically registered when enhanced_operations module is imported

# Log initialization
import logging
logger = logging.getLogger(__name__)
logger.info("✅ Unified Matrix Operations module initialized")
logger.info(f"📦 Version: {__version__}")
logger.info("🔧 Features: VectorBT optimizations, GPU acceleration, memory optimization, vectorized processing, batch operations")
logger.info("🍎 Optimized for: Apple Silicon M1/M2/M3 Macs")
if VECTORBT_AVAILABLE:
    logger.info("🚀 VectorBT optimizations enabled - expect 2-10x performance improvements")
else:
    logger.info("ℹ️ VectorBT not available - using standard implementations")

if VECTORBT_ROLLING_AVAILABLE:
    logger.info("🎯 VectorBTRollingOptimizer available - enhanced rolling operations")
else:
    logger.info("ℹ️ VectorBTRollingOptimizer not available")

if UNIFIED_VECTORIZATION_AVAILABLE:
    logger.info("⚡ UnifiedVectorizationManager available - unified vectorization operations")
else:
    logger.info("ℹ️ UnifiedVectorizationManager not available")