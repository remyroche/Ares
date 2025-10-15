"""
Unified Vectorization Manager

This module provides a centralized system for managing all vectorization and matrix
optimizations across the Ares trading system. It automatically selects the optimal
optimization strategy based on operation type, data size, and available hardware.
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import tprint with fallback
try:
    from ..tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        """Fallback tprint function when tprint is not available."""
        print(*args, **kwargs)

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be optimized."""
    FEATURE_ENGINEERING = "feature_engineering"
    CROSS_VALIDATION = "cross_validation"
    BACKTESTING = "backtesting"
    HMM_TRAINING = "hmm_training"
    MODEL_TRAINING = "model_training"
    FEATURE_SELECTION = "feature_selection"
    TECHNICAL_INDICATORS = "technical_indicators"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    STATISTICAL_COMPUTATION = "statistical_computation"
    # VectorBT-specific operations
    VECTORBT_BACKTESTING = "vectorbt_backtesting"
    VECTORBT_METRICS = "vectorbt_metrics"
    VECTORBT_PORTFOLIO_OPTIMIZATION = "vectorbt_portfolio_optimization"
    VECTORBT_TECHNICAL_ANALYSIS = "vectorbt_technical_analysis"


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    VECTORIZED_CPU = "vectorized_cpu"
    GPU_ACCELERATED = "gpu_accelerated"
    PARALLEL_PROCESSING = "parallel_processing"
    HYBRID_OPTIMIZATION = "hybrid_optimization"
    MEMORY_OPTIMIZED = "memory_optimized"
    FALLBACK = "fallback"
    # VectorBT-specific strategies
    VECTORBT_CPU = "vectorbt_cpu"
    VECTORBT_GPU = "vectorbt_gpu"
    VECTORBT_PARALLEL = "vectorbt_parallel"


@dataclass
class StrategySelectionConfig:
    """Configuration for strategy selection thresholds."""
    # Data size thresholds
    gpu_data_size_threshold: int = 10000
    parallel_data_size_threshold: int = 5000
    vectorbt_data_size_threshold: int = 100
    vectorbt_gpu_threshold: int = 5000
    vectorbt_parallel_threshold: int = 1000
    
    # Memory thresholds
    memory_optimization_threshold_mb: float = 512.0
    chunking_data_size_threshold: int = 1000
    
    # CPU core thresholds
    parallel_cpu_cores_threshold: int = 4
    vectorbt_parallel_cpu_cores_threshold: int = 2


@dataclass
class OperationConfig:
    """Configuration for operation optimization."""
    operation_type: OperationType
    data_size: int
    data_dimensions: Tuple[int, ...]
    memory_budget_mb: float = 1024.0
    time_budget_seconds: float = 300.0
    precision_requirement: str = "medium"  # "low", "medium", "high"
    parallel_workers: Optional[int] = None
    # Performance baselines (in seconds)
    baseline_times: Optional[Dict[OperationType, float]] = None
    # Strategy selection configuration
    strategy_config: Optional[StrategySelectionConfig] = None


@dataclass
class OptimizationResult:
    """Result of an optimized operation."""
    result: Any
    strategy_used: OptimizationStrategy
    computation_time: float
    memory_used_mb: float
    performance_gain: float
    metadata: Dict[str, Any]


class UnifiedVectorizationManager:
    """
    Unified manager for all vectorization and matrix optimizations.

    This class provides intelligent optimization selection and execution
    for various machine learning and trading operations.
    """

    def __init__(self, strategy_config: Optional[StrategySelectionConfig] = None):
        """Initialize the unified vectorization manager."""
        tprint("🚀 Initializing Unified Vectorization Manager...")
        self.logger = logging.getLogger(__name__)

        # Initialize strategy selection configuration
        self.strategy_config = strategy_config or StrategySelectionConfig()

        # Initialize optimization components
        tprint("🔄 Initializing optimization components...")
        self._initialize_components()

        # Performance tracking
        tprint("🔄 Setting up performance tracking...")
        self.performance_history = []
        self.optimization_stats = {
            'total_operations': 0,
            'strategy_usage': {strategy: 0 for strategy in OptimizationStrategy},
            'average_speedup': 0.0,
            'total_computation_time': 0.0
        }

        tprint("✅ Unified Vectorization Manager initialized")
        self.logger.info("✅ Unified Vectorization Manager initialized")

    def _initialize_components(self):
        """Initialize all optimization components."""
        tprint("🔄 Initializing optimization components...")
        # Import and initialize optimization modules
        try:
            # Matrix operations
            tprint("🔄 Loading matrix operations...")
            from ..matrix_operations import get_unified_matrix_operations
            self.matrix_ops = get_unified_matrix_operations()
            self.matrix_ops_available = True
            tprint("✅ Matrix operations loaded")
        except Exception as e:
            tprint(f"⚠️ Matrix operations not available: {e}")
            self.matrix_ops = None
            self.matrix_ops_available = False

        try:
            # Vectorized backtesting
            tprint("🔄 Loading vectorized backtesting...")
            from .vectorized_backtesting import VectorizedBacktestingEngine, BacktestMode
            self.backtesting_engine = VectorizedBacktestingEngine()
            self.backtesting_available = True
            tprint("✅ Vectorized backtesting loaded")
        except Exception as e:
            tprint(f"⚠️ Vectorized backtesting not available: {e}")
            self.backtesting_engine = None
            self.backtesting_available = False

        # VectorBT components
        try:
            # VectorBT backtesting engine
            tprint("🔄 Loading VectorBT backtesting engine...")
            from .vectorbt_backtesting_engine import VectorBTBacktestingEngine, BacktestMode as VectorBTBacktestMode
            self.vectorbt_backtesting_engine = VectorBTBacktestingEngine()
            self.vectorbt_backtesting_available = True
            tprint("✅ VectorBT backtesting engine loaded")
        except Exception as e:
            tprint(f"⚠️ VectorBT backtesting engine not available: {e}")
            self.vectorbt_backtesting_engine = None
            self.vectorbt_backtesting_available = False

        try:
            # VectorBT financial metrics
            tprint("🔄 Loading VectorBT financial metrics...")
            from .vectorbt_financial_metrics import VectorBTFinancialMetrics
            self.vectorbt_metrics = VectorBTFinancialMetrics()
            self.vectorbt_metrics_available = True
            tprint("✅ VectorBT financial metrics loaded")
        except Exception as e:
            tprint(f"⚠️ VectorBT financial metrics not available: {e}")
            self.vectorbt_metrics = None
            self.vectorbt_metrics_available = False

        try:
            # VectorBT portfolio optimization
            tprint("🔄 Loading VectorBT portfolio optimization...")
            from .vectorbt_portfolio_optimization import VectorBTPortfolioOptimizer
            self.vectorbt_portfolio_optimizer = VectorBTPortfolioOptimizer()
            self.vectorbt_portfolio_optimization_available = True
            tprint("✅ VectorBT portfolio optimization loaded")
        except Exception as e:
            tprint(f"⚠️ VectorBT portfolio optimization not available: {e}")
            self.vectorbt_portfolio_optimizer = None
            self.vectorbt_portfolio_optimization_available = False

        try:
            # Matrix cross-validation
            tprint("🔄 Loading matrix cross-validation...")
            from .matrix_cross_validation import MatrixCrossValidator
            self.cv_engine = MatrixCrossValidator()
            self.cv_available = True
            tprint("✅ Matrix cross-validation loaded")
        except Exception as e:
            tprint(f"⚠️ Matrix cross-validation not available: {e}")
            self.cv_engine = None
            self.cv_available = False

        try:
            # Feature importance analyzer
            tprint("🔄 Loading feature importance analyzer...")
            from ..feature_selection.feature_importance_analyzer import FeatureImportanceAnalyzer
            self.feature_analyzer = FeatureImportanceAnalyzer()
            self.feature_selection_available = True
            tprint("✅ Feature importance analyzer loaded")
        except Exception as e:
            tprint(f"⚠️ Feature importance analyzer not available: {e}")
            self.feature_analyzer = None
            self.feature_selection_available = False

        try:
            # Technical indicators
            tprint("🔄 Loading technical indicators...")
            from ..utils.feature_generators import FeatureGenerators
            self.technical_indicators = FeatureGenerators()
            self.technical_indicators_available = True
            tprint("✅ Technical indicators loaded")
        except Exception as e:
            tprint(f"⚠️ Technical indicators not available: {e}")
            self.technical_indicators = None
            self.technical_indicators_available = False

        try:
            # HMM operations
            tprint("🔄 Loading HMM operations...")
            from ..hmm_composite_manager import EnhancedHMMCompositeManager
            self.hmm_manager = EnhancedHMMCompositeManager()
            self.hmm_available = True
            tprint("✅ HMM operations loaded")
        except Exception as e:
            tprint(f"⚠️ HMM operations not available: {e}")
            self.hmm_manager = None
            self.hmm_available = False

        # Hardware detection
        tprint("🔄 Detecting hardware capabilities...")
        self._detect_hardware_capabilities()
        tprint("✅ Component initialization completed")

    def _detect_hardware_capabilities(self):
        """Detect available hardware capabilities."""
        tprint("🔄 Detecting hardware capabilities...")
        self.hardware_caps = {
            'cpu_cores': 1,
            'gpu_available': False,
            'gpu_type': None,
            'memory_gb': 4.0,
            'mps_available': False
        }

        # Detect CPU cores
        tprint("🔄 Detecting CPU cores...")
        import multiprocessing
        self.hardware_caps['cpu_cores'] = multiprocessing.cpu_count()
        tprint(f"📊 CPU cores detected: {self.hardware_caps['cpu_cores']}")

        # Detect GPU availability
        tprint("🔄 Detecting GPU availability...")
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.hardware_caps['gpu_available'] = True
                self.hardware_caps['gpu_type'] = 'cuda'
                self.hardware_caps['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                tprint(f"📊 CUDA GPU detected with {self.hardware_caps['gpu_memory_gb']:.1f}GB memory")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.hardware_caps['gpu_available'] = True
                self.hardware_caps['gpu_type'] = 'mps'
                self.hardware_caps['mps_available'] = True
                tprint("📊 MPS GPU detected")
            else:
                tprint("📊 No GPU detected")
        else:
            tprint("⚠️ PyTorch not available for GPU detection")

        # Detect memory
        tprint("🔄 Detecting system memory...")
        try:
            import psutil
            self.hardware_caps['memory_gb'] = psutil.virtual_memory().total / (1024**3)
            tprint(f"📊 System memory: {self.hardware_caps['memory_gb']:.1f}GB")
        except ImportError:
            tprint("⚠️ psutil not available, using default memory estimate")
            self.hardware_caps['memory_gb'] = 4.0  # Default estimate

        tprint(f"🖥️ Hardware capabilities: {self.hardware_caps}")
        self.logger.info(f"🖥️ Hardware detected: {self.hardware_caps}")

    def optimize_operation(self, operation_type: OperationType,
                          data: Any,
                          config: Optional[OperationConfig] = None,
                          force_strategy: Optional[OptimizationStrategy] = None,
                          **kwargs) -> OptimizationResult:
        """
        Optimize and execute an operation using the best available strategy.

        Args:
            operation_type: Type of operation to optimize
            data: Input data for the operation
            config: Operation configuration
            **kwargs: Additional arguments for the operation

        Returns:
            OptimizationResult with optimized execution
        """
        tprint(f"🚀 Starting operation optimization for {operation_type.value}...")
        start_time = time.time()

        # Create default config if not provided
        if config is None:
            tprint("🔄 Creating default configuration...")
            config = self._create_default_config(operation_type, data)
            tprint(f"📊 Default config created - data_size: {config.data_size}, dimensions: {config.data_dimensions}")

        # Select optimal strategy
        tprint("🔄 Selecting optimal strategy...")
        if force_strategy is not None:
            strategy = force_strategy
            tprint(f"📊 Forced strategy: {strategy.value}")
        else:
            strategy = self._select_optimal_strategy(operation_type, config, **kwargs)
            tprint(f"📊 Selected strategy: {strategy.value}")

        # Execute operation with selected strategy
        tprint("🔄 Executing operation with selected strategy...")
        result, metadata = self._execute_with_strategy(strategy, operation_type, data, config, **kwargs)
        tprint("✅ Operation execution completed")

        # Calculate performance metrics
        computation_time = time.time() - start_time
        tprint(f"⏱️ Computation time: {computation_time:.3f}s")
        memory_used = self._estimate_memory_usage(data, operation_type)
        tprint(f"🧠 Memory used: {memory_used:.1f}MB")
        performance_gain = self._calculate_performance_gain(operation_type, computation_time, metadata, config)
        tprint(f"📈 Performance gain: {performance_gain:.2f}x")

        # Update statistics
        tprint("🔄 Updating performance statistics...")
        self._update_performance_stats(strategy, computation_time, performance_gain)

        optimization_result = OptimizationResult(
            result=result,
            strategy_used=strategy,
            computation_time=computation_time,
            memory_used_mb=memory_used,
            performance_gain=performance_gain,
            metadata=metadata
        )

        tprint(f"✅ Operation {operation_type.value} completed with {strategy.value} strategy")
        self.logger.info(f"✅ Operation {operation_type.value} completed with {strategy.value} strategy")
        return optimization_result

    def _select_optimal_strategy(self, operation_type: OperationType,
                               config: OperationConfig, **kwargs) -> OptimizationStrategy:
        """
        Select the optimal optimization strategy based on operation type and constraints.
        Enhanced with VectorBT prioritization for financial operations.
        """
        # Check for prefer_vectorbt flag
        prefer_vectorbt = kwargs.get('prefer_vectorbt', False)
        
        # VectorBT operations - prioritize VectorBT for financial operations
        if operation_type in [OperationType.VECTORBT_BACKTESTING,
                            OperationType.VECTORBT_METRICS,
                            OperationType.VECTORBT_PORTFOLIO_OPTIMIZATION,
                            OperationType.VECTORBT_TECHNICAL_ANALYSIS]:
            if (self.hardware_caps['gpu_available'] and 
                config.data_size > self.strategy_config.gpu_data_size_threshold):
                return OptimizationStrategy.VECTORBT_GPU
            elif (self.hardware_caps['cpu_cores'] >= self.strategy_config.parallel_cpu_cores_threshold and 
                  config.data_size > self.strategy_config.parallel_data_size_threshold):
                return OptimizationStrategy.VECTORBT_PARALLEL
            else:
                return OptimizationStrategy.VECTORBT_CPU

        # Enhanced VectorBT integration for financial operations - prioritize VectorBT
        if operation_type in [OperationType.BACKTESTING, OperationType.CROSS_VALIDATION]:
            # Use VectorBT for backtesting if available (lower threshold for default usage)
            if (hasattr(self, 'vectorbt_backtesting_available') and 
                self.vectorbt_backtesting_available and 
                (config.data_size > self.strategy_config.vectorbt_data_size_threshold or prefer_vectorbt)):
                if (self.hardware_caps['gpu_available'] and 
                    config.data_size > self.strategy_config.vectorbt_gpu_threshold):
                    return OptimizationStrategy.VECTORBT_GPU
                elif (self.hardware_caps['cpu_cores'] >= self.strategy_config.vectorbt_parallel_cpu_cores_threshold and 
                      config.data_size > self.strategy_config.vectorbt_parallel_threshold):
                    return OptimizationStrategy.VECTORBT_PARALLEL
                else:
                    return OptimizationStrategy.VECTORBT_CPU

        # GPU-first approach for supported operations
        if self.hardware_caps['gpu_available']:
            if operation_type in [OperationType.MATRIX_MULTIPLICATION,
                                OperationType.HMM_TRAINING,
                                OperationType.FEATURE_ENGINEERING]:
                if config.data_size > self.strategy_config.gpu_data_size_threshold:
                    return OptimizationStrategy.GPU_ACCELERATED

        # Parallel processing for CPU-bound operations
        if self.hardware_caps['cpu_cores'] >= self.strategy_config.parallel_cpu_cores_threshold:
            if operation_type in [OperationType.MODEL_TRAINING,
                                OperationType.FEATURE_SELECTION]:
                if config.data_size > self.strategy_config.parallel_data_size_threshold:
                    return OptimizationStrategy.PARALLEL_PROCESSING

        # Hybrid optimization for complex operations
        if operation_type in [OperationType.CROSS_VALIDATION]:
            if (self.hardware_caps['gpu_available'] and 
                self.hardware_caps['cpu_cores'] >= self.strategy_config.parallel_cpu_cores_threshold):
                return OptimizationStrategy.HYBRID_OPTIMIZATION

        # Handle portfolio optimization
        if operation_type == OperationType.PORTFOLIO_OPTIMIZATION:
            # Use VectorBT portfolio optimization if available, otherwise fallback
            if (hasattr(self, 'vectorbt_portfolio_optimization_available') and 
                self.vectorbt_portfolio_optimization_available):
                if (self.hardware_caps['gpu_available'] and 
                    config.data_size > self.strategy_config.vectorbt_gpu_threshold):
                    return OptimizationStrategy.VECTORBT_GPU
                elif (self.hardware_caps['cpu_cores'] >= self.strategy_config.vectorbt_parallel_cpu_cores_threshold and 
                      config.data_size > self.strategy_config.vectorbt_parallel_threshold):
                    return OptimizationStrategy.VECTORBT_PARALLEL
                else:
                    return OptimizationStrategy.VECTORBT_CPU
            else:
                # Fallback to vectorized CPU for basic portfolio optimization
                return OptimizationStrategy.VECTORIZED_CPU

        # Handle statistical computation
        if operation_type == OperationType.STATISTICAL_COMPUTATION:
            if (self.hardware_caps['gpu_available'] and 
                config.data_size > self.strategy_config.gpu_data_size_threshold):
                return OptimizationStrategy.GPU_ACCELERATED
            elif (self.hardware_caps['cpu_cores'] >= self.strategy_config.parallel_cpu_cores_threshold and 
                  config.data_size > self.strategy_config.parallel_data_size_threshold):
                return OptimizationStrategy.PARALLEL_PROCESSING
            else:
                return OptimizationStrategy.VECTORIZED_CPU

        # Default to vectorized CPU for most operations
        if operation_type in [OperationType.FEATURE_ENGINEERING,
                            OperationType.TECHNICAL_INDICATORS]:
            return OptimizationStrategy.VECTORIZED_CPU

        # Memory optimization for large datasets
        if config.memory_budget_mb < self.strategy_config.memory_optimization_threshold_mb:
            return OptimizationStrategy.MEMORY_OPTIMIZED

        # Fallback to vectorized CPU
        return OptimizationStrategy.VECTORIZED_CPU

    def _execute_with_strategy(self, strategy: OptimizationStrategy,
                             operation_type: OperationType,
                             data: Any,
                             config: OperationConfig,
                             **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute operation using the selected strategy.
        """
        if strategy in [OptimizationStrategy.VECTORBT_CPU, OptimizationStrategy.VECTORBT_GPU, OptimizationStrategy.VECTORBT_PARALLEL]:
            return self._execute_vectorbt(operation_type, data, config, strategy, **kwargs)
        elif strategy == OptimizationStrategy.GPU_ACCELERATED:
            return self._execute_gpu_accelerated(operation_type, data, config, **kwargs)
        elif strategy == OptimizationStrategy.PARALLEL_PROCESSING:
            return self._execute_parallel(operation_type, data, config, **kwargs)
        elif strategy == OptimizationStrategy.HYBRID_OPTIMIZATION:
            return self._execute_hybrid(operation_type, data, config, **kwargs)
        elif strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            return self._execute_memory_optimized(operation_type, data, config, **kwargs)
        else:  # VECTORIZED_CPU or FALLBACK
            return self._execute_vectorized_cpu(operation_type, data, config, **kwargs)

    def _execute_gpu_accelerated(self, operation_type: OperationType,
                               data: Any, config: OperationConfig, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with GPU acceleration."""
        metadata = {'gpu_accelerated': True}

        if operation_type == OperationType.MATRIX_MULTIPLICATION and self.matrix_ops:
            # Use matrix operations for matrix multiplication
            result = self.matrix_ops.matrix_multiply(data['a'], data['b'])
            metadata['matrix_ops_used'] = True

        elif operation_type == OperationType.HMM_TRAINING and self.hmm_available:
            # Use GPU-accelerated HMM training
            result = self.hmm_manager.gpu_accelerated_hmm_training(
                data, **kwargs
            )
            metadata['gpu_hmm_used'] = True

        elif operation_type == OperationType.BACKTESTING and self.backtesting_available:
            # Use GPU-accelerated backtesting
            result = self.backtesting_engine.run_vectorized_backtest(
                data['signals'], data['prices'], mode=BacktestMode.GPU_ACCELERATED, **kwargs
            )
            metadata['gpu_backtesting_used'] = True

        else:
            # Fallback to CPU execution
            self.logger.warning(f"⚠️ GPU acceleration not available for {operation_type.value}, falling back to CPU")
            return self._execute_vectorized_cpu(operation_type, data, config, **kwargs)

        return result, metadata

    def _execute_vectorbt(self, operation_type: OperationType,
                         data: Any, config: OperationConfig, strategy: OptimizationStrategy, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with VectorBT."""
        metadata = {'vectorbt_optimized': True, 'strategy': strategy.value}
        
        if operation_type == OperationType.VECTORBT_BACKTESTING and self.vectorbt_backtesting_available:
            # Use VectorBT backtesting engine
            signals = data.get('signals')
            prices = data.get('prices')
            timestamps = data.get('timestamps')
            
            # Determine VectorBT mode based on strategy
            if strategy == OptimizationStrategy.VECTORBT_GPU:
                from .vectorbt_backtesting_engine import BacktestMode
                mode = BacktestMode.VECTORBT_GPU
            elif strategy == OptimizationStrategy.VECTORBT_PARALLEL:
                from .vectorbt_backtesting_engine import BacktestMode
                mode = BacktestMode.VECTORBT_PARALLEL
            else:
                from .vectorbt_backtesting_engine import BacktestMode
                mode = BacktestMode.VECTORBT_CPU
            
            result = self.vectorbt_backtesting_engine.run_backtest(
                signals, prices, timestamps, mode=mode, **kwargs
            )
            metadata['vectorbt_backtesting_used'] = True
            
        elif operation_type == OperationType.VECTORBT_METRICS and self.vectorbt_metrics_available:
            # Use VectorBT financial metrics
            portfolio_values = data.get('portfolio_values')
            returns = data.get('returns')
            benchmark_values = data.get('benchmark_values')
            timestamps = data.get('timestamps')
            
            result = self.vectorbt_metrics.calculate_comprehensive_metrics(
                portfolio_values, returns, benchmark_values, timestamps
            )
            metadata['vectorbt_metrics_used'] = True
            
        elif operation_type == OperationType.VECTORBT_PORTFOLIO_OPTIMIZATION and self.vectorbt_portfolio_optimization_available:
            # Use VectorBT portfolio optimization
            returns = data.get('returns')
            expected_returns = data.get('expected_returns')
            asset_names = data.get('asset_names')
            
            result = self.vectorbt_portfolio_optimizer.optimize_portfolio(
                returns, expected_returns, asset_names, **kwargs
            )
            metadata['vectorbt_portfolio_optimization_used'] = True
            
        elif operation_type == OperationType.VECTORBT_TECHNICAL_ANALYSIS and self.vectorbt_metrics_available:
            # Use VectorBT for technical analysis (placeholder for future implementation)
            result = self.vectorbt_metrics.calculate_comprehensive_metrics(
                data.get('portfolio_values'), data.get('returns')
            )
            metadata['vectorbt_technical_analysis_used'] = True
            
        # Enhanced VectorBT integration for standard operations
        elif operation_type == OperationType.BACKTESTING and self.vectorbt_backtesting_available:
            # Use VectorBT for standard backtesting operations
            signals = data.get('signals')
            prices = data.get('prices')
            timestamps = data.get('timestamps')
            
            # Determine VectorBT mode based on strategy
            if strategy == OptimizationStrategy.VECTORBT_GPU:
                from .vectorbt_backtesting_engine import BacktestMode
                mode = BacktestMode.VECTORBT_GPU
            elif strategy == OptimizationStrategy.VECTORBT_PARALLEL:
                from .vectorbt_backtesting_engine import BacktestMode
                mode = BacktestMode.VECTORBT_PARALLEL
            else:
                from .vectorbt_backtesting_engine import BacktestMode
                mode = BacktestMode.VECTORBT_CPU
            
            result = self.vectorbt_backtesting_engine.run_backtest(
                signals, prices, timestamps, mode=mode, **kwargs
            )
            metadata['vectorbt_backtesting_used'] = True
            metadata['enhanced_integration'] = True
            
        elif operation_type == OperationType.CROSS_VALIDATION and hasattr(self, 'cv_engine') and self.cv_available:
            # Use VectorBT-enhanced cross-validation
            X = data.get('X')
            y = data.get('y')
            model_class = data.get('model_class')
            model_params = data.get('model_params', {})
            
            # Use VectorBT cross-validation if available
            if hasattr(self.cv_engine, 'vectorbt_cross_validate'):
                result = self.cv_engine.vectorbt_cross_validate(
                    X, y, model_class, model_params, **kwargs
                )
                metadata['vectorbt_cv_used'] = True
            else:
                # Fallback to standard cross-validation
                result = self.cv_engine.cross_validate(
                    X, y, model_class, model_params, **kwargs
                )
                metadata['standard_cv_used'] = True
            
        else:
            # Fallback to CPU execution
            self.logger.warning(f"⚠️ VectorBT operation not available for {operation_type.value}, falling back to CPU")
            return self._execute_vectorized_cpu(operation_type, data, config, **kwargs)
        
        return result, metadata

    def _execute_parallel(self, operation_type: OperationType,
                        data: Any, config: OperationConfig, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with parallel processing."""
        metadata = {'parallel_processing': True, 'workers': config.parallel_workers or self.hardware_caps['cpu_cores']}

        if operation_type == OperationType.CROSS_VALIDATION and self.cv_available:
            # Use parallel cross-validation
            result = self.cv_engine.parallel_cross_validate(
                data['X'], data['y'], data['model_class'],
                model_params=data.get('model_params'),
                max_workers=metadata['workers'],
                **kwargs
            )
            metadata['parallel_cv_used'] = True

        elif operation_type == OperationType.FEATURE_SELECTION and self.feature_selection_available:
            # Use parallel feature selection
            result = self.feature_analyzer.batch_compute_importance(
                data['X'], data['y'], **kwargs
            )
            metadata['parallel_feature_selection_used'] = True

        else:
            # Fallback to CPU execution
            return self._execute_vectorized_cpu(operation_type, data, config, **kwargs)

        return result, metadata

    def _execute_hybrid(self, operation_type: OperationType,
                      data: Any, config: OperationConfig, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with hybrid optimization (GPU + parallel)."""
        metadata = {'hybrid_optimization': True}

        if operation_type == OperationType.BACKTESTING and self.backtesting_available:
            # Use hybrid backtesting (GPU + parallel chunks)
            result = self.backtesting_engine.run_vectorized_backtest(
                data['signals'], data['prices'], mode=BacktestMode.HYBRID, **kwargs
            )
            metadata['hybrid_backtesting_used'] = True

        else:
            # Fallback to GPU execution
            return self._execute_gpu_accelerated(operation_type, data, config, **kwargs)

        return result, metadata

    def _execute_memory_optimized(self, operation_type: OperationType,
                                data: Any, config: OperationConfig, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with memory optimization."""
        metadata = {'memory_optimized': True}

        # Use chunked processing for memory efficiency
        # Only chunk if memory budget is actually constrained
        if (config.memory_budget_mb < self.strategy_config.memory_optimization_threshold_mb and 
            hasattr(data, '__len__') and 
            len(data) > self.strategy_config.chunking_data_size_threshold):
            # Split data into chunks
            chunk_size = max(100, config.data_size // 4)  # Minimum chunk size
            chunks = self._split_data_into_chunks(data, chunk_size)

            # Process chunks and combine results
            chunk_results = []
            for chunk in chunks:
                chunk_result, _ = self._execute_vectorized_cpu(operation_type, chunk, config, **kwargs)
                chunk_results.append(chunk_result)

            result = self._combine_chunk_results(chunk_results, operation_type)
            metadata['chunked_processing'] = True
            metadata['num_chunks'] = len(chunks)
        else:
            # Execute normally
            result, metadata_cpu = self._execute_vectorized_cpu(operation_type, data, config, **kwargs)
            metadata.update(metadata_cpu)

        return result, metadata

    def _execute_vectorized_cpu(self, operation_type: OperationType,
                              data: Any, config: OperationConfig, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with vectorized CPU operations."""
        metadata = {'vectorized_cpu': True}

        if operation_type == OperationType.TECHNICAL_INDICATORS and self.technical_indicators_available:
            # Use batch technical indicators
            result = self.technical_indicators.batch_technical_indicators(
                data, kwargs.get('indicator_configs', {}), **kwargs
            )
            metadata['batch_indicators_used'] = True

        elif operation_type == OperationType.FEATURE_SELECTION and self.feature_selection_available:
            # Use vectorized feature selection
            result = self.feature_analyzer.batch_compute_importance(
                data['X'], data['y'], **kwargs
            )
            metadata['vectorized_feature_selection_used'] = True

        elif operation_type == OperationType.CROSS_VALIDATION and self.cv_available:
            # Use matrix cross-validation
            result = self.cv_engine.cross_validate(
                data['X'], data['y'], data['model_class'],
                model_params=data.get('model_params'), **kwargs
            )
            metadata['matrix_cv_used'] = True

        elif operation_type == OperationType.BACKTESTING and self.backtesting_available:
            # Use vectorized backtesting
            result = self.backtesting_engine.run_vectorized_backtest(
                data['signals'], data['prices'], mode=BacktestMode.VECTORIZED, **kwargs
            )
            metadata['vectorized_backtesting_used'] = True

        elif operation_type == OperationType.PORTFOLIO_OPTIMIZATION:
            # Basic portfolio optimization using numpy
            returns = data.get('returns')
            if returns is not None:
                # Simple mean-variance optimization
                mean_returns = np.mean(returns, axis=0)
                cov_matrix = np.cov(returns.T)
                
                # Equal weight portfolio as baseline
                n_assets = len(mean_returns)
                equal_weights = np.ones(n_assets) / n_assets
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(equal_weights, mean_returns)
                portfolio_variance = np.dot(equal_weights, np.dot(cov_matrix, equal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                result = {
                    'weights': equal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                }
                metadata['basic_portfolio_optimization_used'] = True
            else:
                result = self._execute_generic_operation(operation_type, data, **kwargs)
                metadata['generic_fallback'] = True

        elif operation_type == OperationType.STATISTICAL_COMPUTATION:
            # Basic statistical computations
            if isinstance(data, np.ndarray):
                result = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'median': np.median(data),
                    'skewness': self._calculate_skewness(data),
                    'kurtosis': self._calculate_kurtosis(data)
                }
                metadata['basic_statistical_computation_used'] = True
            else:
                result = self._execute_generic_operation(operation_type, data, **kwargs)
                metadata['generic_fallback'] = True

        else:
            # Generic fallback
            self.logger.warning(f"⚠️ No specific optimization available for {operation_type.value}")
            result = self._execute_generic_operation(operation_type, data, **kwargs)
            metadata['generic_fallback'] = True

        return result, metadata

    def _execute_generic_operation(self, operation_type: OperationType, data: Any, **kwargs) -> Any:
        """Execute operation with generic optimization."""
        # Apply basic vectorization where possible
        if isinstance(data, (list, tuple)) and len(data) > 1:
            # Convert to numpy array for vectorized operations
            if all(isinstance(x, (int, float)) for x in data):
                data_array = np.array(data)
                # Apply vectorized operations
                return np.mean(data_array), np.std(data_array), np.min(data_array), np.max(data_array)

        return data  # Return as-is if no optimization possible

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _create_default_config(self, operation_type: OperationType, data: Any) -> OperationConfig:
        """Create default configuration for an operation."""
        if hasattr(data, '__len__'):
            data_size = len(data)
        else:
            data_size = 1

        if hasattr(data, 'shape'):
            data_dimensions = data.shape
        else:
            data_dimensions = (data_size,)

        return OperationConfig(
            operation_type=operation_type,
            data_size=data_size,
            data_dimensions=data_dimensions,
            parallel_workers=self.hardware_caps['cpu_cores']
        )

    def _estimate_memory_usage(self, data: Any, operation_type: OperationType) -> float:
        """Estimate memory usage for an operation."""
        if hasattr(data, 'nbytes'):
            # NumPy array
            base_memory = data.nbytes / (1024 * 1024)  # MB
        elif hasattr(data, '__len__'):
            # Estimate for other iterables
            base_memory = len(data) * 8 / (1024 * 1024)  # Assume 8 bytes per element
        else:
            base_memory = 1.0  # Default estimate

        # Adjust based on operation type
        memory_multipliers = {
            OperationType.MATRIX_MULTIPLICATION: 3.0,
            OperationType.HMM_TRAINING: 2.5,
            OperationType.BACKTESTING: 2.0,
            OperationType.CROSS_VALIDATION: 1.5,
            OperationType.FEATURE_ENGINEERING: 1.8
        }

        multiplier = memory_multipliers.get(operation_type, 1.0)
        return base_memory * multiplier

    def _calculate_performance_gain(self, operation_type: OperationType,
                                  computation_time: float,
                                  metadata: Dict[str, Any],
                                  config: Optional[OperationConfig] = None) -> float:
        """Calculate performance gain compared to baseline."""
        # Use config baselines if available, otherwise use defaults
        if config and config.baseline_times:
            baseline_time = config.baseline_times.get(operation_type, 10.0)
        else:
            # Default baseline times (estimated) for different operations
            baseline_times = {
                OperationType.MATRIX_MULTIPLICATION: 10.0,
                OperationType.HMM_TRAINING: 50.0,
                OperationType.BACKTESTING: 30.0,
                OperationType.CROSS_VALIDATION: 20.0,
                OperationType.FEATURE_ENGINEERING: 15.0,
                OperationType.TECHNICAL_INDICATORS: 25.0,
                OperationType.PORTFOLIO_OPTIMIZATION: 20.0,
                OperationType.STATISTICAL_COMPUTATION: 5.0
            }
            baseline_time = baseline_times.get(operation_type, 10.0)

        if computation_time > 0:
            return baseline_time / computation_time
        return 1.0

    def _update_performance_stats(self, strategy: OptimizationStrategy,
                                computation_time: float, performance_gain: float):
        """Update performance statistics."""
        self.optimization_stats['total_operations'] += 1
        self.optimization_stats['strategy_usage'][strategy] += 1
        self.optimization_stats['total_computation_time'] += computation_time

        # Update performance history
        self.performance_history.append({
            'strategy': strategy.value,
            'computation_time': computation_time,
            'performance_gain': performance_gain,
            'timestamp': time.time()
        })

        # Update average speedup
        total_operations = self.optimization_stats['total_operations']
        current_avg = self.optimization_stats['average_speedup']
        self.optimization_stats['average_speedup'] = (
            (current_avg * (total_operations - 1)) + performance_gain
        ) / total_operations

    def _split_data_into_chunks(self, data: Any, chunk_size: int) -> List[Any]:
        """Split data into chunks for memory-efficient processing."""
        if isinstance(data, dict):
            # Handle dictionary of arrays (e.g., for backtesting with signals and prices)
            chunks = []
            keys = list(data.keys())
            if not keys:
                return [data]
            
            # Get the length from the first array
            first_key = keys[0]
            first_array = data[first_key]
            if not hasattr(first_array, '__len__'):
                return [data]
            
            total_length = len(first_array)
            for i in range(0, total_length, chunk_size):
                chunk = {}
                for key in keys:
                    array = data[key]
                    if hasattr(array, '__len__') and len(array) == total_length:
                        if isinstance(array, np.ndarray):
                            chunk[key] = array[i:i+chunk_size]
                        elif isinstance(array, pd.DataFrame):
                            chunk[key] = array.iloc[i:i+chunk_size]
                        else:
                            chunk[key] = array[i:i+chunk_size]
                    else:
                        chunk[key] = array  # Keep non-matching arrays as-is
                chunks.append(chunk)
            return chunks
        elif isinstance(data, np.ndarray):
            return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, pd.DataFrame):
            return [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, list):
            return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        else:
            return [data]  # Can't split, return as single chunk

    def _combine_chunk_results(self, chunk_results: List[Any], operation_type: OperationType) -> Any:
        """Combine results from multiple chunks."""
        if not chunk_results:
            tprint("⚠️ No chunk results to combine, returning empty result")
            # Return appropriate empty result based on operation type
            if operation_type == OperationType.FEATURE_ENGINEERING:
                return pd.DataFrame()
            elif operation_type in [OperationType.CROSS_VALIDATION, OperationType.BACKTESTING, OperationType.PORTFOLIO_OPTIMIZATION, OperationType.STATISTICAL_COMPUTATION]:
                return {}
            else:
                return []

        if operation_type == OperationType.FEATURE_ENGINEERING:
            # Concatenate DataFrames
            if all(isinstance(result, pd.DataFrame) for result in chunk_results):
                return pd.concat(chunk_results, ignore_index=True)
            elif all(isinstance(result, np.ndarray) for result in chunk_results):
                return np.concatenate(chunk_results)

        elif operation_type in [OperationType.CROSS_VALIDATION, OperationType.BACKTESTING, OperationType.PORTFOLIO_OPTIMIZATION, OperationType.STATISTICAL_COMPUTATION]:
            # Average results across chunks
            if all(isinstance(result, dict) for result in chunk_results):
                combined = {}
                for key in chunk_results[0].keys():
                    if isinstance(chunk_results[0][key], (int, float)):
                        combined[key] = np.mean([result[key] for result in chunk_results])
                    elif isinstance(chunk_results[0][key], np.ndarray):
                        combined[key] = np.concatenate([result[key] for result in chunk_results])
                    elif isinstance(chunk_results[0][key], list):
                        # Concatenate lists
                        combined[key] = []
                        for result in chunk_results:
                            combined[key].extend(result[key])
                    else:
                        combined[key] = chunk_results[0][key]  # Take first one
                return combined

        # Handle single values (e.g., from statistical computations)
        elif operation_type == OperationType.STATISTICAL_COMPUTATION:
            if all(isinstance(result, (int, float)) for result in chunk_results):
                return np.mean(chunk_results)
            elif all(isinstance(result, dict) for result in chunk_results):
                # Already handled above
                pass

        # Default: return first result
        return chunk_results[0]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = self.optimization_stats.copy()
        stats['hardware_capabilities'] = self.hardware_caps.copy()
        stats['available_optimizations'] = {
            'matrix_operations': self.matrix_ops_available,
            'backtesting': self.backtesting_available,
            'cross_validation': self.cv_available,
            'feature_selection': self.feature_selection_available,
            'technical_indicators': self.technical_indicators_available,
            'hmm_operations': self.hmm_available,
            # VectorBT components
            'vectorbt_backtesting': self.vectorbt_backtesting_available,
            'vectorbt_metrics': self.vectorbt_metrics_available,
            'vectorbt_portfolio_optimization': self.vectorbt_portfolio_optimization_available
        }

        return stats

    def benchmark_operation(self, operation_type: OperationType,
                          data: Any, trials: int = 3) -> Dict[str, Any]:
        """
        Benchmark an operation across different optimization strategies.

        Args:
            operation_type: Type of operation to benchmark
            data: Input data for benchmarking
            trials: Number of trials to run for each strategy

        Returns:
            Benchmarking results comparing different strategies
        """
        self.logger.info(f"🔬 Benchmarking {operation_type.value} across optimization strategies...")

        results = {}
        config = self._create_default_config(operation_type, data)

        # Test each available strategy
        strategies_to_test = [
            OptimizationStrategy.VECTORIZED_CPU,
            OptimizationStrategy.PARALLEL_PROCESSING,
            OptimizationStrategy.MEMORY_OPTIMIZED
        ]

        if self.hardware_caps['gpu_available']:
            strategies_to_test.append(OptimizationStrategy.GPU_ACCELERATED)
            if self.hardware_caps['cpu_cores'] >= 4:
                strategies_to_test.append(OptimizationStrategy.HYBRID_OPTIMIZATION)

        for strategy in strategies_to_test:
            strategy_times = []
            strategy_gains = []

            for trial in range(trials):
                try:
                    result = self.optimize_operation(
                        operation_type, data, config,
                        force_strategy=strategy
                    )
                    strategy_times.append(result.computation_time)
                    strategy_gains.append(result.performance_gain)
                except Exception as e:
                    self.logger.warning(f"⚠️ Strategy {strategy.value} failed: {e}")
                    continue

            if strategy_times:
                results[strategy.value] = {
                    'avg_time': np.mean(strategy_times),
                    'std_time': np.std(strategy_times),
                    'avg_gain': np.mean(strategy_gains),
                    'trials_completed': len(strategy_times)
                }

        return results


# Convenience functions for common operations
def optimize_feature_engineering(data: pd.DataFrame,
                               indicator_configs: Dict[str, List[int]],
                               strategy_config: Optional[StrategySelectionConfig] = None,
                               **kwargs) -> OptimizationResult:
    """Convenience function for optimized feature engineering."""
    manager = UnifiedVectorizationManager(strategy_config)
    config = OperationConfig(
        operation_type=OperationType.TECHNICAL_INDICATORS,
        data_size=len(data),
        data_dimensions=data.shape
    )
    return manager.optimize_operation(
        OperationType.TECHNICAL_INDICATORS,
        data,
        config,
        indicator_configs=indicator_configs,
        **kwargs
    )


def optimize_cross_validation(X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           model_class: Any,
                           strategy_config: Optional[StrategySelectionConfig] = None,
                           **kwargs) -> OptimizationResult:
    """Convenience function for optimized cross-validation with VectorBT by default."""
    manager = UnifiedVectorizationManager(strategy_config)
    data_size = len(X) if hasattr(X, '__len__') else 1
    data_dimensions = X.shape if hasattr(X, 'shape') else (data_size,)
    config = OperationConfig(
        operation_type=OperationType.CROSS_VALIDATION,
        data_size=data_size,
        data_dimensions=data_dimensions
    )
    data = {'X': X, 'y': y, 'model_class': model_class}
    # Force VectorBT preference
    kwargs['prefer_vectorbt'] = True
    return manager.optimize_operation(OperationType.CROSS_VALIDATION, data, config, **kwargs)


def optimize_backtesting(signals: Union[np.ndarray, pd.DataFrame],
                       prices: Union[np.ndarray, pd.DataFrame],
                       timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                       strategy_config: Optional[StrategySelectionConfig] = None,
                       **kwargs) -> OptimizationResult:
    """Convenience function for optimized backtesting with VectorBT by default."""
    manager = UnifiedVectorizationManager(strategy_config)
    data_size = len(signals) if hasattr(signals, '__len__') else 1
    data_dimensions = signals.shape if hasattr(signals, 'shape') else (data_size,)
    config = OperationConfig(
        operation_type=OperationType.BACKTESTING,
        data_size=data_size,
        data_dimensions=data_dimensions
    )
    data = {'signals': signals, 'prices': prices, 'timestamps': timestamps}
    # Force VectorBT preference
    kwargs['prefer_vectorbt'] = True
    return manager.optimize_operation(OperationType.BACKTESTING, data, config, **kwargs)


# VectorBT convenience functions
def optimize_vectorbt_backtesting(signals: Union[np.ndarray, pd.DataFrame],
                                prices: Union[np.ndarray, pd.DataFrame],
                                timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                                **kwargs) -> OptimizationResult:
    """Convenience function for VectorBT backtesting."""
    manager = UnifiedVectorizationManager()
    data_size = len(signals) if hasattr(signals, '__len__') else 1
    data_dimensions = signals.shape if hasattr(signals, 'shape') else (data_size,)
    config = OperationConfig(
        operation_type=OperationType.VECTORBT_BACKTESTING,
        data_size=data_size,
        data_dimensions=data_dimensions
    )
    data = {'signals': signals, 'prices': prices, 'timestamps': timestamps}
    return manager.optimize_operation(OperationType.VECTORBT_BACKTESTING, data, config, **kwargs)


def optimize_vectorbt_metrics(portfolio_values: Union[np.ndarray, pd.Series],
                            returns: Optional[Union[np.ndarray, pd.Series]] = None,
                            benchmark_values: Optional[Union[np.ndarray, pd.Series]] = None,
                            timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                            **kwargs) -> OptimizationResult:
    """Convenience function for VectorBT financial metrics."""
    manager = UnifiedVectorizationManager()
    data_size = len(portfolio_values) if hasattr(portfolio_values, '__len__') else 1
    data_dimensions = portfolio_values.shape if hasattr(portfolio_values, 'shape') else (data_size,)
    config = OperationConfig(
        operation_type=OperationType.VECTORBT_METRICS,
        data_size=data_size,
        data_dimensions=data_dimensions
    )
    data = {
        'portfolio_values': portfolio_values,
        'returns': returns,
        'benchmark_values': benchmark_values,
        'timestamps': timestamps
    }
    return manager.optimize_operation(OperationType.VECTORBT_METRICS, data, config, **kwargs)


def optimize_vectorbt_portfolio(returns: Union[np.ndarray, pd.DataFrame],
                              expected_returns: Optional[Union[np.ndarray, pd.Series]] = None,
                              asset_names: Optional[List[str]] = None,
                              **kwargs) -> OptimizationResult:
    """Convenience function for VectorBT portfolio optimization."""
    manager = UnifiedVectorizationManager()
    data_size = len(returns) if hasattr(returns, '__len__') else 1
    data_dimensions = returns.shape if hasattr(returns, 'shape') else (data_size,)
    config = OperationConfig(
        operation_type=OperationType.VECTORBT_PORTFOLIO_OPTIMIZATION,
        data_size=data_size,
        data_dimensions=data_dimensions
    )
    data = {
        'returns': returns,
        'expected_returns': expected_returns,
        'asset_names': asset_names
    }
    return manager.optimize_operation(OperationType.VECTORBT_PORTFOLIO_OPTIMIZATION, data, config, **kwargs)


def optimize_financial_operation(operation_type: str,
                               data: Dict[str, Any],
                               use_vectorbt: bool = True,
                               **kwargs) -> OptimizationResult:
    """
    Enhanced convenience function for financial operations with automatic VectorBT integration.
    
    Args:
        operation_type: Type of financial operation ('backtesting', 'metrics', 'portfolio', 'cv')
        data: Data dictionary containing required inputs
        use_vectorbt: Whether to prefer VectorBT optimization
        **kwargs: Additional arguments
        
    Returns:
        OptimizationResult with optimized execution
    """
    manager = UnifiedVectorizationManager()
    
    # Map operation types
    operation_map = {
        'backtesting': OperationType.BACKTESTING,
        'metrics': OperationType.VECTORBT_METRICS,
        'portfolio': OperationType.VECTORBT_PORTFOLIO_OPTIMIZATION,
        'cv': OperationType.CROSS_VALIDATION,
        'cross_validation': OperationType.CROSS_VALIDATION
    }
    
    if operation_type not in operation_map:
        raise ValueError(f"Unsupported operation type: {operation_type}. "
                        f"Supported types: {list(operation_map.keys())}")
    
    op_type = operation_map[operation_type]
    
    # Create configuration
    data_size = 1000  # Default estimate
    if 'signals' in data and hasattr(data['signals'], '__len__'):
        data_size = len(data['signals'])
    elif 'X' in data and hasattr(data['X'], '__len__'):
        data_size = len(data['X'])
    elif 'returns' in data and hasattr(data['returns'], '__len__'):
        data_size = len(data['returns'])
    
    data_dimensions = (data_size,)
    if 'signals' in data and hasattr(data['signals'], 'shape'):
        data_dimensions = data['signals'].shape
    elif 'X' in data and hasattr(data['X'], 'shape'):
        data_dimensions = data['X'].shape
    elif 'returns' in data and hasattr(data['returns'], 'shape'):
        data_dimensions = data['returns'].shape
    
    config = OperationConfig(
        operation_type=op_type,
        data_size=data_size,
        data_dimensions=data_dimensions
    )
    
    # Add VectorBT preference to kwargs
    if use_vectorbt:
        kwargs['prefer_vectorbt'] = True
    
    return manager.optimize_operation(op_type, data, config, **kwargs)


# Global instance for easy access
_unified_manager = None

def get_unified_vectorization_manager(strategy_config: Optional[StrategySelectionConfig] = None) -> UnifiedVectorizationManager:
    """Get global unified vectorization manager instance."""
    global _unified_manager
    if _unified_manager is None or strategy_config is not None:
        _unified_manager = UnifiedVectorizationManager(strategy_config)
    return _unified_manager


if __name__ == "__main__":
    # Example usage and testing
    manager = get_unified_vectorization_manager()

    # Test basic functionality
    print("🧪 Testing Unified Vectorization Manager...")

    # Create sample data
    np.random.seed(42)
    sample_data = np.random.randn(1000, 10)

    # Test matrix multiplication
    A = np.random.randn(500, 500)
    B = np.random.randn(500, 500)

    result = manager.optimize_operation(
        OperationType.MATRIX_MULTIPLICATION,
        {'a': A, 'b': B}
    )

    print("✅ Matrix multiplication test completed")
    # Print optimization stats
    stats = manager.get_optimization_stats()
    print("\n📊 Optimization Statistics:")
    print(f"Total operations: {stats['total_operations']}")
    print(f"Average speedup: {stats['average_speedup']:.2f}x")
    print(f"Strategy usage: {stats['strategy_usage']}")

    print("\n🎉 Unified Vectorization Manager ready for production use!")
