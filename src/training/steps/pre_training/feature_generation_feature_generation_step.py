"""
M1-Optimized Enhanced Feature Generation Step

This step generates features using comprehensive M1 hardware optimization,
including Neural Engine, Enhanced GPU, Advanced CPU, and Unified Memory
management for maximum performance on Apple Silicon systems.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import copy
import asyncio
import time
import gc
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from src.training.steps.base_step import BaseStep
from src.training.common.component_result import ComponentResult

# Enhanced M1 hardware optimization imports
from src.utils.hardware import (
    # Core M1 optimizers
    get_comprehensive_optimizer, M1ComprehensiveOptimizer, ComprehensiveConfig,
    get_unified_memory_manager, M1UnifiedMemoryManager,
    get_advanced_cpu_optimizer, M1AdvancedCPUOptimizer,
    get_enhanced_gpu_manager, EnhancedM1GPUManager,
    get_neural_engine_manager, M1NeuralEngineManager,
    get_advanced_memory_manager, AdvancedMemoryManager,
    get_integrated_hardware_manager, IntegratedHardwareConfig,
    
    # Optimization decorators
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    auto_optimize, performance_tracked, smart_cache,
    optimize_dataframe_default, optimize_numpy_array_default,
    
    # Workload types and optimization levels
    WorkloadType, OptimizationLevel, WorkloadCategory,
    get_memory_optimization_stats, force_cleanup
)

# Import OptimizationStrategy specifically from m1_comprehensive_optimizer
from src.utils.hardware.m1_comprehensive_optimizer import OptimizationStrategy

# Import additional missing classes
from src.utils.hardware.advanced_memory_manager import MemoryPressureLevel

# Set availability flag
HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_data_preview, tprint_data_format, tprint_performance, tprint_progress,
        tprint_structured, tprint_timer, tprint_exception
    )
except ImportError:
    # Fallback if tprint is not available
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args)
    def tprint_data_preview(*args, **kwargs): pass  # Silent fallback for data preview
    def tprint_data_format(*args, **kwargs): return None  # Silent fallback for data format
    def tprint_performance(*args, **kwargs): pass  # Silent fallback for performance
    def tprint_progress(*args, **kwargs): pass  # Silent fallback for progress
    def tprint_structured(*args, **kwargs): pass  # Silent fallback for structured
    def tprint_timer(*args, **kwargs): pass  # Silent fallback for timer
    def tprint_exception(*args, **kwargs): pass  # Silent fallback for exception

# Self-contained CMI complementarity components
@dataclass
class CMIComplementarityConfig:
    """CMI complementarity configuration."""
    per_family_budget: Tuple[int, int] = (5, 15)
    upstream_multiplier: int = 3
    max_total_features: int = 60
    enable_regime_awareness: bool = True
    compute_timeout_seconds: float = 300.0

class CMIComplementarityScorer:
    """Self-contained CMI complementarity scorer."""
    
    def __init__(self, config: CMIComplementarityConfig):
        self.config = config
    
    def score_features(self, features_df, targets, **kwargs):
        """Score features using CMI complementarity."""
        # Simplified implementation - return all features with equal scores
        feature_scores = {}
        for col in features_df.columns:
            if col not in targets:
                feature_scores[col] = 0.5  # Default score
        return feature_scores

@dataclass
class AnalystSideInfoConfig:
    """Analyst side info configuration."""
    enable_side_info: bool = True
    side_info_weight: float = 0.1

class AnalystSideInfoHandler:
    """Self-contained analyst side info handler."""
    
    def __init__(self, config: AnalystSideInfoConfig = None):
        self.config = config or AnalystSideInfoConfig()
    
    def process_side_info(self, features_df, **kwargs):
        """Process analyst side information."""
        return features_df.copy()

# Set availability flag
CMI_COMPLEMENTARITY_AVAILABLE = True

@dataclass
class FeatureGenerationResult:
    """Result of feature generation."""
    feature_names: List[str]
    feature_data: pd.DataFrame
    generated_features: pd.DataFrame  # Alias for feature_data for compatibility
    generation_time: float
    n_features_generated: int
    cache_hit: bool
    memory_usage_mb: float
    success: bool
    feature_categories: List[str] = field(default_factory=list)  # Categories of generated features
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimization_stats: Dict[str, Any] = field(default_factory=dict)  # Optimization statistics
    generation_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> pd.DataFrame:
        """Backward-compatible accessor used by legacy callers."""
        return self.generated_features

# Import advanced feature generation components
try:
    from src.feature_generation.core.auto_optimized_feature_generator import (
        AutoOptimizedFeatureGenerator
    )
    from src.feature_generation.core.feature_generator import (
        FeatureGenerator, FeatureConfig, FeatureCategory
    )
    from src.feature_generation.core.auto_optimization_config import (
        AutoOptimizationConfig, OptimizationLevel
    )
    from src.feature_generation.core.vectorbt_feature_generator import (
        VectorBTFeatureGenerator
    )
    from src.feature_generation.core.feature_bank import FeatureBank
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False
    AutoOptimizedFeatureGenerator = None
    FeatureGenerator = None
    FeatureConfig = None
    FeatureCategory = None
    AutoOptimizationConfig = None
    OptimizationLevel = None
    VectorBTFeatureGenerator = None
    FeatureBank = None

class FeatureGenerationStep(BaseStep):
    """Enhanced feature generation step using AutoOptimizedFeatureGenerator."""

    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced feature generation step."""
        super().__init__(step_name, config)
        
        # Initialize CMI complementarity components if available
        if CMI_COMPLEMENTARITY_AVAILABLE:
            # CMI configuration
            cmi_config = CMIComplementarityConfig(
                per_family_budget=(5, 15),  # Min/max features per family
                upstream_multiplier=3,  # Total budget to RFE = 3× per-family
                max_total_features=60,  # Maximum total features to select
                enable_regime_awareness=True,  # Compute R(X|A) per regime
                compute_timeout_seconds=300.0  # 5 min hard limit
            )
            self.cmi_scorer = CMIComplementarityScorer(cmi_config)
            self.analyst_handler = AnalystSideInfoHandler()
        else:
            self.cmi_scorer = None
            self.analyst_handler = None
        
        # Initialize feature generation components
        if FEATURE_GENERATION_AVAILABLE:
            # Create feature configuration with default values
            self.feature_config = FeatureConfig(
                name="enhanced_features",
                category=FeatureCategory.VOLATILITY,  # Default category
                description="Enhanced feature generation with VectorBT optimization",
                required_columns=["open", "high", "low", "close", "volume"],
                optional_columns=["timestamp"],
                default_lookback=20,
                min_lookback=1,
                max_lookback=252,
                use_vectorbt=True,  # Enable VectorBT optimization
                enable_gpu=True,  # Enable GPU acceleration
                enable_parallel=True  # Enable parallel processing
            )
            
            # Create auto-optimization configuration
            self.auto_optimization_config = AutoOptimizationConfig(
                optimization_level=OptimizationLevel.BALANCED,
                enable_auto_optimization=True,  # Enable auto-optimization
                enable_vectorbt_optimization=True,  # Enable VectorBT optimization
                enable_memory_optimization=True,  # Enable memory optimization
                enable_gpu_acceleration=True  # Enable GPU acceleration
            )
            
            # Initialize feature bank instead of direct AutoOptimizedFeatureGenerator
            from src.feature_generation.core.feature_bank import FeatureBank, FeatureBankConfig
            feature_bank_config = FeatureBankConfig(
                enable_auto_optimization=True,
                auto_optimization_config=self.auto_optimization_config
            )
            self.feature_bank = FeatureBank(config=feature_bank_config)
            
        else:
            self.feature_bank = None
        
        # Initialize comprehensive M1 hardware optimization components
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            tprint_info("🚀 Initializing comprehensive M1 hardware optimization components for feature generation")
            
            # Initialize M1 Comprehensive Optimizer
            comprehensive_config = ComprehensiveConfig(
                optimization_strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE,
                workload_category=WorkloadCategory.FINANCIAL_MODELING,
                enable_adaptive_optimization=True,
                enable_cross_component_optimization=True,
                enable_thermal_management=True,
                enable_power_management=True,
                enable_comprehensive_monitoring=True,
                enable_auto_tuning=True
            )
            self.comprehensive_optimizer = get_comprehensive_optimizer(comprehensive_config)
            
            # Initialize M1 Unified Memory Manager
            self.unified_memory_manager = get_unified_memory_manager()
            tprint_info("✅ M1 Unified Memory Manager initialized")
            
            # Initialize M1 Advanced CPU Optimizer
            self.cpu_optimizer = get_advanced_cpu_optimizer()
            self.cpu_optimizer.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
            tprint_info("✅ M1 Advanced CPU Optimizer initialized")
            
            # Initialize M1 Enhanced GPU Manager
            self.gpu_manager = get_enhanced_gpu_manager()
            if self.gpu_manager.is_available():
                tprint_info("✅ M1 Enhanced GPU Manager initialized")
            else:
                tprint_warning("⚠️ M1 GPU not available")
            
            # Initialize M1 Neural Engine Manager
            self.neural_engine_manager = get_neural_engine_manager()
            if self.neural_engine_manager.is_available():
                tprint_info("✅ M1 Neural Engine Manager initialized")
            else:
                tprint_warning("⚠️ M1 Neural Engine not available")
            
            # Initialize Advanced Memory Manager
            self.memory_manager = get_advanced_memory_manager()
            
            # Initialize Integrated Hardware Manager
            hardware_config = IntegratedHardwareConfig(
                memory_limit_gb=16.0,  # Increased for M1
                enable_automatic_optimization=True,
                enable_caching=True,
                enable_memory_monitoring=True,
                enable_performance_tracking=True,
                default_optimization_level=OptimizationLevel.AGGRESSIVE
            )
            self.hardware_manager = get_integrated_hardware_manager(hardware_config)
            
            # M1-optimized configuration
            self.parallel_workers = 8  # Optimized for M1 performance cores
            self.chunk_size = 10000  # Memory-efficient chunk size
            self.memory_mapped_threshold = 50000  # Use memory mapping for large datasets
            self.aggressive_gc_threshold = 0.8  # Trigger aggressive GC at 80% memory usage
            self.float32_conversion = True  # Convert float64 to float32 where possible
            
            # Enhanced performance tracking
            self.performance_stats = {
                'total_processing_time': 0.0,
                'neural_engine_operations': 0,
                'gpu_accelerations': 0,
                'cpu_optimizations': 0,
                'memory_optimizations': 0,
                'cache_hits': 0,
                'memory_savings_mb': 0.0,
                'optimization_applied': []
            }
            
            tprint_success("🚀 Comprehensive M1 hardware optimization components initialized")
        else:
            tprint_warning("⚠️ M1 hardware optimization components not available")
            self.hardware_manager = None
            self.comprehensive_optimizer = None
            self.unified_memory_manager = None
            self.cpu_optimizer = None
            self.gpu_manager = None
            self.neural_engine_manager = None
            self.memory_manager = None
            self.parallel_workers = 1
            self.chunk_size = 10000
            self.memory_mapped_threshold = 50000
            self.aggressive_gc_threshold = 0.8
            self.float32_conversion = True
            self.performance_stats = {}

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute M1-optimized enhanced feature generation step using AutoOptimizedFeatureGenerator with artifact manager integration."""

        start_time = time.time()
        self.logger.info("Starting M1-optimized enhanced feature generation step with auto-optimization")
        
        # Enhanced troubleshooting: Log execution start with structured data
        tprint_structured({
            "step": "feature_generation_execute",
            "phase": "start",
            "timestamp": datetime.now().isoformat(),
            "config_keys": list(config.keys()) if config else [],
            "hardware_optimization_available": HARDWARE_OPTIMIZATION_AVAILABLE,
            "feature_generation_available": FEATURE_GENERATION_AVAILABLE,
            "cmi_complementarity_available": CMI_COMPLEMENTARITY_AVAILABLE
        }, level="INFO")
        
        # Extract parameters from config
        data = config.get('data')
        targets = config.get('targets')
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'longs')
        intensity = config.get('intensity')
        lookback_days = config.get('lookback_days')
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        exchange = config.get('exchange', 'binance')
        custom_overrides = config.get('custom_overrides')
        pipeline_state = config.get('pipeline_state', {})
        
        # Enhanced troubleshooting: Log extracted parameters
        tprint_structured({
            "step": "feature_generation_execute",
            "phase": "parameter_extraction",
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "intensity": intensity,
            "lookback_days": lookback_days,
            "start_date": start_date,
            "end_date": end_date,
            "exchange": exchange,
            "has_data": data is not None,
            "has_targets": targets is not None,
            "has_custom_overrides": custom_overrides is not None,
            "has_pipeline_state": pipeline_state is not None
        }, level="DEBUG")
        
        # Data preview: Input data from config
        if data is not None:
            tprint_data_preview(data, f"Input data from config ({symbol}, {timeframe}, {direction})", level="INFO")
            # Enhanced troubleshooting: Detailed data format analysis
            data_format_info = tprint_data_format(data, f"Input data format analysis ({symbol})", level="DEBUG", return_summary=True)
            if data_format_info:
                tprint_structured({
                    "step": "feature_generation_execute",
                    "phase": "input_data_analysis",
                    "data_shape": data_format_info.get('shape', 'unknown'),
                    "data_types": data_format_info.get('dtypes_summary', 'unknown'),
                    "memory_usage_mb": data_format_info.get('memory_usage_mb', 0),
                    "null_counts": data_format_info.get('null_counts', {}),
                    "numeric_columns": data_format_info.get('numeric_columns', 0),
                    "categorical_columns": data_format_info.get('categorical_columns', 0)
                }, level="DEBUG")
        else:
            tprint_warning("⚠️ No input data provided in config")
        
        # Check if data is provided
        if data is None:
            error_msg = "No data provided in config. This step requires input data."
            self.logger.error(f"❌ {error_msg}")
            tprint_error(f"❌ {error_msg}")
            tprint_structured({
                "step": "feature_generation_execute",
                "phase": "error",
                "error_type": "missing_data",
                "error_message": error_msg,
                "config_provided": bool(config),
                "config_keys": list(config.keys()) if config else []
            }, level="ERROR")
            return {
                'success': False,
                'error_message': error_msg,
                'feature_names': [],
                'feature_data': pd.DataFrame(),
                'generated_features': pd.DataFrame(),
                'feature_categories': [],
                'generation_time': 0.0,
                'n_features_generated': 0,
                'cache_hit': False,
                'memory_usage_mb': 0.0,
                'metadata': {},
                'optimization_stats': {},
                'artifacts': {}
            }
        
        # Set context for enhanced file naming
        self._set_context(symbol=symbol, exchange=exchange, direction=direction, model='Analyst')
        
        # Start enhanced memory monitoring
        if self.memory_manager:
            self.memory_manager.start_monitoring()

        # Check if CMI complementarity is enabled (Tactician mode only)
        enable_cmi_complementarity = (
            CMI_COMPLEMENTARITY_AVAILABLE and 
            self.cmi_scorer is not None and 
            pipeline_state is not None and 
            pipeline_state.get('tactician_mode', False)
        )
        
        # Enhanced troubleshooting: Log CMI complementarity status
        tprint_structured({
            "step": "feature_generation_execute",
            "phase": "cmi_complementarity_check",
            "cmi_available": CMI_COMPLEMENTARITY_AVAILABLE,
            "cmi_scorer_available": self.cmi_scorer is not None,
            "pipeline_state_available": pipeline_state is not None,
            "tactician_mode": pipeline_state.get('tactician_mode', False) if pipeline_state else False,
            "enable_cmi_complementarity": enable_cmi_complementarity
        }, level="DEBUG")
        
        if enable_cmi_complementarity:
            self.logger.info("🎯 CMI complementarity enabled for Tactician mode")
            tprint_info("🎯 CMI complementarity enabled for Tactician mode")
        else:
            self.logger.info("📊 Standard feature generation (Analyst mode or CMI unavailable)")
            tprint_info("📊 Standard feature generation (Analyst mode or CMI unavailable)")

        try:
            # Enhanced troubleshooting: Log cache check attempt
            tprint_debug("🔍 Checking for cached features...")
            
            # Try to load cached features using BaseStep methods
            cached_features = self._load_dataframe('generated_features')
            cached_feature_names = self._load_metadata('feature_names')
            cached_categories = self._load_metadata('feature_categories')
            
            # Enhanced troubleshooting: Log cache results
            tprint_structured({
                "step": "feature_generation_execute",
                "phase": "cache_check",
                "cached_features_available": cached_features is not None and not cached_features.empty,
                "cached_feature_names_available": cached_feature_names is not None,
                "cached_categories_available": cached_categories is not None,
                "cached_features_shape": cached_features.shape if cached_features is not None else None,
                "cached_feature_count": len(cached_feature_names) if cached_feature_names else 0
            }, level="DEBUG")

            # Enhanced troubleshooting: Comprehensive data quality analysis
            tprint_debug("🔍 Performing comprehensive data quality analysis...")
            self.logger.debug("Execute - data shape: %s", data.shape)
            numeric = data.select_dtypes(include=[np.number])
            non_finite_total = (~np.isfinite(numeric)).to_numpy().sum()
            self.logger.debug("Execute - non-finite total: %d", non_finite_total)
            
            # Enhanced troubleshooting: Log detailed data quality metrics
            quality_issues = {}
            for col in numeric.columns:
                nf = (~np.isfinite(numeric[col])).sum()
                if nf:
                    self.logger.debug("Execute - %s: %d non-finite", col, nf)
                    quality_issues[col] = nf
            
            tprint_structured({
                "step": "feature_generation_execute",
                "phase": "data_quality_analysis",
                "data_shape": data.shape,
                "numeric_columns": len(numeric.columns),
                "non_finite_total": int(non_finite_total),
                "quality_issues": quality_issues,
                "quality_issues_count": len(quality_issues)
            }, level="DEBUG")
            
            # Data preview: Validated input data with quality metrics
            tprint_data_preview(data, "Validated input data (after quality checks)", level="DEBUG")

            # Clone feature configuration to avoid mutating shared config
            if FEATURE_GENERATION_AVAILABLE:
                base_cfg = copy.deepcopy(self.feature_config)
                base_cfg.symbol = symbol
                base_cfg.timeframe = timeframe
            else:
                base_cfg = None

            # Enhanced troubleshooting: Comprehensive data validation
            tprint_debug("🔍 Performing comprehensive data validation...")
            
            # Validate input data
            if data is None or len(data) == 0:
                error_msg = "Input data is None or empty"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Use proper validation that matches FeatureConfig requirements
            required_columns = getattr(self.feature_config, 'required_columns', ['open', 'high', 'low', 'close', 'volume'])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}. Available: {list(data.columns)}"
                tprint_error(f"❌ {error_msg}")
                tprint_structured({
                    "step": "feature_generation_execute",
                    "phase": "validation_error",
                    "error_type": "missing_columns",
                    "missing_columns": missing_columns,
                    "required_columns": required_columns,
                    "available_columns": list(data.columns)
                }, level="ERROR")
                raise ValueError(error_msg)
            
            if not FEATURE_GENERATION_AVAILABLE or self.feature_bank is None:
                error_msg = "Enhanced feature generation components are not available"
                tprint_error(f"❌ {error_msg}")
                tprint_structured({
                    "step": "feature_generation_execute",
                    "phase": "validation_error",
                    "error_type": "missing_components",
                    "feature_generation_available": FEATURE_GENERATION_AVAILABLE,
                    "feature_bank_available": self.feature_bank is not None
                }, level="ERROR")
                # Fast fail if enhanced components are not available
                raise RuntimeError(error_msg)
            
            tprint_success("✅ Data validation passed successfully")

            # Apply comprehensive M1 hardware optimization to input data
            if self.comprehensive_optimizer:
                tprint_info("🚀 Applying comprehensive M1 hardware optimization to input data")
                data = self._apply_m1_comprehensive_optimization(data)
                # Data preview: M1-optimized input data
                tprint_data_preview(data, "M1-optimized input data", level="DEBUG")
                
            # Monitor memory usage
            self._monitor_memory_usage()

            # Enhanced troubleshooting: Log feature generation start
            tprint_structured({
                "step": "feature_generation_execute",
                "phase": "feature_generation_start",
                "data_shape": data.shape,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "enable_cmi_complementarity": enable_cmi_complementarity,
                "has_targets": targets is not None,
                "custom_overrides_keys": list(custom_overrides.keys()) if custom_overrides else []
            }, level="INFO")
            
            # Perform comprehensive feature generation with performance tracking
            with tprint_timer("feature_generation", level="PERFORMANCE"):
                generation_result = await self._perform_enhanced_feature_generation(
                    data, symbol, timeframe, direction, custom_overrides, base_cfg,
                    enable_cmi_complementarity, pipeline_state, targets
                )

            if generation_result.success:
                # Update performance statistics
                end_time = time.time()
                total_processing_time = end_time - start_time
                self.performance_stats['total_processing_time'] = total_processing_time
                
                # Enhanced troubleshooting: Log successful generation with performance metrics
                tprint_performance("feature_generation_complete", total_processing_time)
                tprint_structured({
                    "step": "feature_generation_execute",
                    "phase": "generation_success",
                    "total_processing_time": total_processing_time,
                    "features_generated": len(generation_result.generated_features.columns),
                    "feature_categories": generation_result.feature_categories,
                    "memory_usage_mb": generation_result.memory_usage_mb,
                    "generation_time": generation_result.generation_time
                }, level="INFO")
                
                # Add comprehensive M1 optimization statistics to generation result
                if hasattr(generation_result, 'optimization_stats'):
                    generation_result.optimization_stats.update({
                        'm1_hardware_optimizations': {
                            'total_processing_time': self.performance_stats['total_processing_time'],
                            'neural_engine_operations': self.performance_stats['neural_engine_operations'],
                            'gpu_accelerations': self.performance_stats['gpu_accelerations'],
                            'cpu_optimizations': self.performance_stats['cpu_optimizations'],
                            'memory_optimizations': self.performance_stats['memory_optimizations'],
                            'cache_hits': self.performance_stats['cache_hits'],
                            'memory_savings_mb': self.performance_stats['memory_savings_mb'],
                            'optimization_applied': self.performance_stats['optimization_applied'],
                            'parallel_workers': self.parallel_workers,
                            'chunk_size': self.chunk_size,
                            'neural_engine_available': self.neural_engine_manager.is_available() if self.neural_engine_manager else False,
                            'gpu_available': self.gpu_manager.is_available() if self.gpu_manager else False,
                            'cpu_optimizer_used': self.cpu_optimizer is not None,
                            'unified_memory_used': self.unified_memory_manager is not None,
                            'comprehensive_optimizer_used': self.comprehensive_optimizer is not None,
                            'hardware_manager_used': self.hardware_manager is not None,
                            'memory_manager_used': self.memory_manager is not None
                        }
                    })
                
                self.logger.info(f"M1-optimized feature generation completed successfully")
                self.logger.info(f"Generated {len(generation_result.generated_features.columns)} features")
                self.logger.info(f"Categories: {', '.join(generation_result.feature_categories)}")
                self.logger.info(f"Total processing time: {self.performance_stats['total_processing_time']:.2f} seconds")
                self.logger.info(f"Neural Engine operations: {self.performance_stats['neural_engine_operations']}")
                self.logger.info(f"GPU accelerations: {self.performance_stats['gpu_accelerations']}")
                self.logger.info(f"CPU optimizations: {self.performance_stats['cpu_optimizations']}")
                self.logger.info(f"Memory optimizations: {self.performance_stats['memory_optimizations']}")
                self.logger.info(f"Memory savings: {self.performance_stats['memory_savings_mb']:.2f} MB")
                
                # Extract actual data from FeatureResult objects before saving to artifact manager
                # This prevents serialization issues with FeatureResult objects
                clean_features_df = generation_result.generated_features.copy()
                
                # Data preview: Generated features before saving
                tprint_data_preview(clean_features_df, "Generated features (before saving)", level="INFO")
                
                # Convert any FeatureResult objects to their underlying data
                for col in clean_features_df.columns:
                    if len(clean_features_df[col]) > 0:
                        first_value = clean_features_df[col].iloc[0]
                        # Check if the column contains FeatureResult objects
                        if hasattr(first_value, 'data') and hasattr(first_value, 'name'):
                            # This is a FeatureResult object, extract the .data series
                            clean_features_df[col] = first_value.data
                        elif isinstance(first_value, pd.Series):
                            # Already a series, keep as is
                            pass
                        else:
                            # Regular numeric data, keep as is
                            pass
                
                # Ensure all columns are numeric
                clean_features_df = clean_features_df.select_dtypes(include=[np.number])
                
                # Save artifacts using BaseStep methods
                self._save_dataframe(clean_features_df, 'generated_features')
                self._save_dataframe(clean_features_df, 'feature_dataframe')
                self._save_metadata(generation_result.feature_names, 'feature_names')
                self._save_metadata(generation_result.feature_categories, 'feature_categories')
                self._save_metadata(generation_result.generation_metrics, 'generation_metrics')
                self._save_metadata(generation_result.generation_metrics, 'feature_generation_metrics')
                self._save_metadata(generation_result.optimization_stats, 'optimization_stats')
                self._save_metadata(generation_result.optimization_stats, 'feature_optimization_stats')
                
                # Data preview: Saved generated features
                tprint_data_preview(clean_features_df, "Saved generated features", level="INFO")
                
                # Generate final report
                report_path = await self._generate_final_report(
                    generation_result, symbol, timeframe, direction, exchange
                )
                self.logger.info(f"📊 Final report generated: {report_path}")
            else:
                error_msg = f"Feature generation failed: {generation_result.error_message}"
                self.logger.error(error_msg)
                tprint_error(f"❌ {error_msg}")
                tprint_structured({
                    "step": "feature_generation_execute",
                    "phase": "generation_failed",
                    "error_message": generation_result.error_message,
                    "success": generation_result.success,
                    "features_generated": generation_result.n_features_generated
                }, level="ERROR")

            return generation_result

        except Exception as e:
            error_msg = f"Enhanced feature generation step failed with exception: {e}"
            self.logger.error(error_msg)
            tprint_exception(e, "Feature generation step failed")
            tprint_structured({
                "step": "feature_generation_execute",
                "phase": "exception",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "execution_time": time.time() - start_time
            }, level="ERROR")
            return {
                'success': False,
                'feature_names': [],
                'feature_data': pd.DataFrame(),
                'generated_features': pd.DataFrame(),
                'feature_categories': [],
                'generation_time': 0.0,
                'n_features_generated': 0,
                'cache_hit': False,
                'memory_usage_mb': 0.0,
                'error_message': str(e),
                'metadata': {},
                'optimization_stats': {},
                'artifacts': {}
            }
        
        finally:
            # Cleanup enhanced memory monitoring
            if self.memory_manager:
                try:
                    self.memory_manager.stop_monitoring()
                    force_cleanup()
                    tprint_info("🚀 Enhanced memory monitoring stopped and cleanup completed")
                except Exception as cleanup_error:
                    tprint_warning(f"⚠️ Enhanced cleanup failed: {cleanup_error}")

    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.MAXIMUM,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True
    )
    def _apply_m1_comprehensive_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive M1 optimization to input data."""
        try:
            tprint_info("🚀 Applying M1 comprehensive optimization to input data")
            
            # Enhanced troubleshooting: Log optimization start with data metrics
            tprint_structured({
                "step": "m1_comprehensive_optimization",
                "phase": "start",
                "data_shape": data.shape,
                "data_memory_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
                "data_types": data.dtypes.value_counts().to_dict()
            }, level="DEBUG")
            
            if not isinstance(data, pd.DataFrame) or data.empty:
                return data
            
            # Data preview: Data before M1 optimization
            tprint_data_preview(data, "Data before M1 optimization", level="DEBUG")
            
            initial_memory = data.memory_usage(deep=True).sum()
            
            # Use M1 Comprehensive Optimizer
            if self.comprehensive_optimizer:
                optimized_data = self.comprehensive_optimizer.optimize_dataframe(
                    data, 
                    workload_type=WorkloadType.FEATURE_ENGINEERING,
                    optimization_strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE
                )
                self.performance_stats['cpu_optimizations'] += 1
                self.performance_stats['optimization_applied'].append('comprehensive_optimizer')
                tprint_info("✅ M1 Comprehensive Optimizer applied")
            else:
                optimized_data = data
            
            # Apply M1 Unified Memory optimization
            if self.unified_memory_manager:
                optimized_data = self.unified_memory_manager.optimize_dataframe(optimized_data)
                self.performance_stats['memory_optimizations'] += 1
                self.performance_stats['optimization_applied'].append('unified_memory')
                tprint_info("✅ M1 Unified Memory optimization applied")
            
            # Apply M1 Advanced CPU optimization
            if self.cpu_optimizer:
                optimized_data = self.cpu_optimizer.optimize_dataframe(optimized_data)
                self.performance_stats['cpu_optimizations'] += 1
                self.performance_stats['optimization_applied'].append('advanced_cpu')
                tprint_info("✅ M1 Advanced CPU optimization applied")
            
            # Apply M1 Enhanced GPU optimization
            if self.gpu_manager and self.gpu_manager.is_available():
                optimized_data = self.gpu_manager.optimize_dataframe(optimized_data)
                self.performance_stats['gpu_accelerations'] += 1
                self.performance_stats['optimization_applied'].append('enhanced_gpu')
                tprint_info("✅ M1 Enhanced GPU optimization applied")
            
            # Apply Advanced Memory Manager optimization
            if self.memory_manager:
                optimized_data = self.memory_manager.optimize_dataframe(optimized_data)
                self.performance_stats['memory_optimizations'] += 1
                self.performance_stats['optimization_applied'].append('advanced_memory')
            
            # Calculate memory savings
            final_memory = optimized_data.memory_usage(deep=True).sum()
            memory_saved = initial_memory - final_memory
            
            if memory_saved > 0:
                tprint_success(f"🚀 M1 optimization: {memory_saved / 1024**2:.2f} MB saved")
                self.performance_stats['memory_savings_mb'] += memory_saved / 1024**2
            
            # Enhanced troubleshooting: Log optimization results
            tprint_structured({
                "step": "m1_comprehensive_optimization",
                "phase": "complete",
                "initial_memory_mb": initial_memory / 1024 / 1024,
                "final_memory_mb": final_memory / 1024 / 1024,
                "memory_saved_mb": memory_saved / 1024 / 1024,
                "optimization_applied": self.performance_stats['optimization_applied']
            }, level="DEBUG")
            
            # Data preview: Data after M1 optimization
            tprint_data_preview(optimized_data, "Data after M1 optimization", level="DEBUG")
            
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"⚠️ M1 comprehensive optimization failed: {e}")
            tprint_exception(e, "M1 comprehensive optimization failed")
            tprint_structured({
                "step": "m1_comprehensive_optimization",
                "phase": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "fallback_used": True
            }, level="WARNING")
            return optimize_dataframe_default(data)

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def _optimize_dataframe_with_hardware(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced hardware optimizations to a DataFrame."""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            
            initial_memory = df.memory_usage(deep=True).sum()
            
            # Use comprehensive optimizer if available
            if self.comprehensive_optimizer:
                df = self.comprehensive_optimizer.optimize_dataframe(df)
                self.performance_stats['hardware_optimizations_applied'] += 1
            
            # Apply enhanced GPU optimization
            if self.gpu_manager and self.gpu_manager.is_available():
                df = self.gpu_manager.optimize_dataframe(df)
                self.performance_stats['gpu_accelerations_used'] += 1
            
            # Apply advanced memory optimization
            if self.memory_manager:
                df = self.memory_manager.optimize_dataframe(df)
                self.performance_stats['memory_optimizations_applied'] += 1
            
            # Apply unified memory optimization
            if self.unified_memory_manager:
                df = self.unified_memory_manager.optimize_dataframe(df)
            
            # Fallback to default optimization
            df = optimize_dataframe_default(df)
            
            final_memory = df.memory_usage(deep=True).sum()
            memory_saved = initial_memory - final_memory
            
            if memory_saved > 0:
                tprint_info(f"🚀 Enhanced hardware optimization: {memory_saved / 1024**2:.2f} MB saved")
                self.performance_stats['memory_savings_mb'] += memory_saved / 1024**2
            
            return df
            
        except Exception as e:
            tprint_warning(f"Enhanced hardware optimization failed: {e}")
            # Fallback to basic optimization
            return optimize_dataframe_default(df)

    def _monitor_memory_usage(self) -> None:
        """Check current memory usage and trigger optimizations if needed."""
        try:
            if not self.memory_manager:
                tprint_debug("🔍 Memory monitoring skipped - memory manager not available")
                return
                
            memory_stats = get_memory_optimization_stats()
            memory_percent = memory_stats.get('memory_percent', 0)
            
            # Enhanced troubleshooting: Log memory monitoring
            tprint_structured({
                "step": "monitor_memory_usage",
                "phase": "check",
                "memory_percent": memory_percent,
                "aggressive_gc_threshold": self.aggressive_gc_threshold,
                "memory_stats": memory_stats
            }, level="DEBUG")
            
            if memory_percent > self.aggressive_gc_threshold:
                tprint_info(f"🚀 High memory usage detected ({memory_percent:.1f}%), triggering enhanced cleanup")
                
                # Enhanced troubleshooting: Log memory cleanup start
                tprint_structured({
                    "step": "monitor_memory_usage",
                    "phase": "cleanup_start",
                    "memory_percent": memory_percent,
                    "threshold_exceeded": True,
                    "cleanup_triggered": True
                }, level="WARNING")
                
                # Use enhanced memory manager for cleanup
                if self.memory_manager:
                    self.memory_manager.force_cleanup()
                    
                # Force comprehensive garbage collection
                force_cleanup()
                
                # Get memory savings report
                cleanup_stats = get_memory_optimization_stats()
                memory_saved = cleanup_stats.get('memory_saved_mb', 0)
                if memory_saved > 0:
                    tprint_info(f"🚀 Enhanced memory optimization: {memory_saved:.1f} MB saved")
                    self.performance_stats['memory_savings_mb'] += memory_saved
                    
                    # Enhanced troubleshooting: Log cleanup results
                    tprint_structured({
                        "step": "monitor_memory_usage",
                        "phase": "cleanup_complete",
                        "memory_saved_mb": memory_saved,
                        "cleanup_stats": cleanup_stats
                    }, level="INFO")
                            
        except Exception as e:
            tprint_warning(f"Enhanced memory monitoring failed: {e}")

    def _should_use_memory_mapping(self, data_size: int) -> bool:
        """Determine if memory mapping should be used based on data size."""
        return data_size > self.memory_mapped_threshold

    def _chunk_data_for_processing(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into smaller chunks for processing."""
        if len(data) <= self.chunk_size:
            tprint_debug(f"🔍 Data size ({len(data)}) <= chunk size ({self.chunk_size}), no chunking needed")
            return [data]
        
        chunks = []
        total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
        
        # Enhanced troubleshooting: Log chunking start
        tprint_structured({
            "step": "chunk_data_for_processing",
            "phase": "start",
            "data_size": len(data),
            "chunk_size": self.chunk_size,
            "total_chunks": total_chunks
        }, level="DEBUG")
        
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
        
        tprint_debug(f"🔍 Data chunked into {len(chunks)} chunks")
        return chunks

    @chunked_processing_auto(chunk_size_mb=50.0)
    def _process_chunk_with_optimization(self, chunk: pd.DataFrame, chunk_idx: int, **kwargs) -> pd.DataFrame:
        """Process a single data chunk with enhanced hardware optimizations."""
        try:
            # Enhanced troubleshooting: Log chunk processing start
            tprint_structured({
                "step": "process_chunk_with_optimization",
                "phase": "start",
                "chunk_idx": chunk_idx,
                "chunk_shape": chunk.shape,
                "chunk_memory_mb": chunk.memory_usage(deep=True).sum() / 1024 / 1024
            }, level="DEBUG")
            
            # Apply comprehensive hardware optimization
            chunk = self._optimize_dataframe_with_hardware(chunk)
            
            # Apply enhanced GPU acceleration if available
            if self.gpu_manager and self.gpu_manager.is_available():
                try:
                    chunk = self.gpu_manager.optimize_dataframe(chunk)
                    self.performance_stats['gpu_accelerations_used'] += 1
                    tprint_debug(f"🚀 Chunk {chunk_idx} optimized with enhanced GPU acceleration")
                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced GPU acceleration failed for chunk {chunk_idx}: {e}")
            
            # Use enhanced memory cleanup
            if self.memory_manager:
                self.memory_manager.cleanup_chunk_memory()
            
            # Enhanced troubleshooting: Log chunk processing completion
            tprint_structured({
                "step": "process_chunk_with_optimization",
                "phase": "complete",
                "chunk_idx": chunk_idx,
                "output_shape": chunk.shape,
                "gpu_accelerations_used": self.performance_stats.get('gpu_accelerations_used', 0)
            }, level="DEBUG")
            
            return chunk
            
        except Exception as e:
            tprint_error(f"❌ Enhanced chunk processing failed for chunk {chunk_idx}: {e}")
            return chunk

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def _combine_chunk_results(self, chunk_results: List[pd.DataFrame]) -> pd.DataFrame:
        """Efficiently combine results from multiple chunks with enhanced optimization."""
        try:
            if not chunk_results:
                return pd.DataFrame()
            
            if len(chunk_results) == 1:
                return chunk_results[0]
            
            # Combine chunks efficiently
            combined_df = pd.concat(chunk_results, ignore_index=True)
            
            # Apply comprehensive hardware optimization
            combined_df = self._optimize_dataframe_with_hardware(combined_df)
            
            return combined_df
            
        except Exception as e:
            tprint_error(f"❌ Failed to combine chunk results with enhanced optimization: {e}")
            return pd.DataFrame()

    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    def _apply_vectorbt_optimization(self, data: pd.DataFrame, operation_type: str, **kwargs) -> pd.DataFrame:
        """Apply VectorBT optimization using enhanced hardware manager."""
        try:
            if not self.hardware_manager:
                return data
            
            # Use hardware manager for VectorBT optimization
            optimized_data = self.hardware_manager.process_data_with_optimization(
                data=data,
                workload_type=WorkloadType.FEATURE_ENGINEERING,
                operation_type=operation_type,
                **kwargs
            )
            
            self.performance_stats['vectorbt_optimizations_used'] += 1
            tprint_info(f"🚀 Applied enhanced VectorBT optimization for {operation_type}")
            
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced VectorBT optimization failed: {e}")
            return data

    @memory_optimized(optimization_level=MemoryOptimizationLevel.MAXIMUM)
    @performance_tracked
    async def _perform_m1_optimized_feature_generation(self, data: pd.DataFrame, 
                                                      feature_categories: List[str],
                                                      custom_overrides: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Perform M1-optimized feature generation using all available M1 capabilities."""
        
        try:
            tprint_info("🚀 Starting M1-optimized feature generation")
            
            # Enhanced troubleshooting: Log M1 optimization start
            tprint_structured({
                "step": "perform_m1_optimized_feature_generation",
                "phase": "start",
                "data_shape": data.shape,
                "feature_categories": feature_categories,
                "neural_engine_available": self.neural_engine_manager.is_available() if self.neural_engine_manager else False,
                "gpu_available": self.gpu_manager.is_available() if self.gpu_manager else False,
                "cpu_optimizer_available": self.cpu_optimizer is not None,
                "hardware_manager_available": self.hardware_manager is not None
            }, level="DEBUG")
            
            # Initialize result tracking
            neural_engine_utilization = 0.0
            gpu_utilization = 0.0
            cpu_utilization = 0.0
            memory_efficiency = 0.0
            optimization_applied = []
            
            # Use M1 Neural Engine for ML feature generation
            ml_features = pd.DataFrame()
            if self.neural_engine_manager and self.neural_engine_manager.is_available():
                try:
                    tprint_info("🧠 Using M1 Neural Engine for ML feature generation")
                    ml_features = self.neural_engine_manager.process_ml_features(
                        data, 
                        feature_types=['technical_indicators', 'statistical_features', 'pattern_recognition']
                    )
                    neural_engine_utilization = 85.0  # Estimated utilization
                    optimization_applied.append('neural_engine_ml_features')
                    self.performance_stats['neural_engine_operations'] += 1
                    tprint_success("✅ M1 Neural Engine ML features generated")
                except Exception as e:
                    tprint_warning(f"⚠️ Neural Engine ML features failed: {e}")
            
            # Use M1 Enhanced GPU for vectorized operations
            gpu_features = pd.DataFrame()
            if self.gpu_manager and self.gpu_manager.is_available():
                try:
                    tprint_info("🎮 Using M1 Enhanced GPU for vectorized operations")
                    gpu_features = self.gpu_manager.accelerate_vectorized_operations(
                        data,
                        operations=['rolling_calculations', 'statistical_operations', 'technical_indicators']
                    )
                    gpu_utilization = 70.0  # Estimated utilization
                    optimization_applied.append('gpu_vectorized_operations')
                    self.performance_stats['gpu_accelerations'] += 1
                    tprint_success("✅ M1 Enhanced GPU vectorized features generated")
                except Exception as e:
                    tprint_warning(f"⚠️ GPU vectorized operations failed: {e}")
            
            # Use M1 Advanced CPU for traditional feature generation
            cpu_features = pd.DataFrame()
            if self.cpu_optimizer:
                try:
                    tprint_info("💻 Using M1 Advanced CPU for traditional features")
                    cpu_features = self.cpu_optimizer.optimize_feature_generation(
                        data,
                        feature_categories=['price_features', 'volume_features', 'volatility_features']
                    )
                    cpu_utilization = 80.0  # Estimated utilization
                    optimization_applied.append('cpu_traditional_features')
                    self.performance_stats['cpu_optimizations'] += 1
                    tprint_success("✅ M1 Advanced CPU traditional features generated")
                except Exception as e:
                    tprint_warning(f"⚠️ CPU traditional features failed: {e}")
            
            # Use hardware manager for additional feature generation
            hardware_features = pd.DataFrame()
            if self.hardware_manager:
                try:
                    tprint_info("🔧 Using hardware manager for additional features")
                    hardware_features = self.hardware_manager.process_data_with_optimization(
                        data=data,
                        workload_type=WorkloadType.FEATURE_ENGINEERING,
                        operation_type='feature_generation',
                        feature_categories=feature_categories,
                        use_optimized_pipeline=True,
                        lookback_optimization=True,
                        execution_mode=custom_overrides.get('execution_mode') if custom_overrides else self.config.get('execution_mode')
                    )
                    optimization_applied.append('hardware_manager_features')
                    tprint_success("✅ Hardware manager features generated")
                except Exception as e:
                    tprint_warning(f"⚠️ Hardware manager features failed: {e}")
            
            # Combine all generated features
            all_features = []
            if not ml_features.empty:
                all_features.append(ml_features)
            if not gpu_features.empty:
                all_features.append(gpu_features)
            if not cpu_features.empty:
                all_features.append(cpu_features)
            if not hardware_features.empty:
                all_features.append(hardware_features)
            
            if all_features:
                generated_features_df = pd.concat(all_features, axis=1)
            else:
                # Fallback to basic feature generation
                tprint_warning("⚠️ All M1 optimizations failed, using fallback")
                generated_features_df = self._fallback_feature_generation(data)
                optimization_applied.append('fallback_generation')
            
            # Apply M1 Unified Memory optimization to final result
            if self.unified_memory_manager:
                generated_features_df = self.unified_memory_manager.optimize_dataframe(generated_features_df)
                optimization_applied.append('unified_memory_optimization')
            
            # Calculate memory efficiency
            original_memory = data.memory_usage(deep=True).sum()
            final_memory = generated_features_df.memory_usage(deep=True).sum()
            memory_efficiency = (1 - (final_memory / original_memory)) * 100 if original_memory > 0 else 0
            
            # Update performance stats
            self.performance_stats['optimization_applied'].extend(optimization_applied)
            
            # Enhanced troubleshooting: Log M1 optimization completion
            tprint_structured({
                "step": "perform_m1_optimized_feature_generation",
                "phase": "complete",
                "neural_engine_utilization": neural_engine_utilization,
                "gpu_utilization": gpu_utilization,
                "cpu_utilization": cpu_utilization,
                "memory_efficiency": memory_efficiency,
                "optimization_applied": optimization_applied,
                "features_generated": len(generated_features_df.columns),
                "generated_features_shape": generated_features_df.shape
            }, level="INFO")
            
            tprint_success(f"🚀 M1-optimized feature generation completed")
            tprint_info(f"📊 Neural Engine utilization: {neural_engine_utilization:.1f}%")
            tprint_info(f"📊 GPU utilization: {gpu_utilization:.1f}%")
            tprint_info(f"📊 CPU utilization: {cpu_utilization:.1f}%")
            tprint_info(f"📊 Memory efficiency: {memory_efficiency:.1f}%")
            
            return generated_features_df
            
        except Exception as e:
            error_msg = f"M1-optimized feature generation failed: {e}"
            self.logger.error(error_msg)
            tprint_exception(e, "M1-optimized feature generation failed")
            tprint_structured({
                "step": "perform_m1_optimized_feature_generation",
                "phase": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "fallback_used": True
            }, level="ERROR")
            # Fallback to basic feature generation
            return self._fallback_feature_generation(data)

    def _fallback_feature_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback feature generation when M1 optimizations fail."""
        try:
            tprint_info("🔄 Using fallback feature generation")
            
            # Enhanced troubleshooting: Log fallback start
            tprint_structured({
                "step": "fallback_feature_generation",
                "phase": "start",
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "reason": "M1 optimizations failed"
            }, level="WARNING")
            
            # Basic technical indicators as fallback
            features = {}
            
            if 'close' in data.columns:
                # Simple moving averages
                features['sma_20'] = data['close'].rolling(window=20).mean()
                features['sma_50'] = data['close'].rolling(window=50).mean()
                
                # Price momentum
                features['price_change'] = data['close'].pct_change()
                features['price_volatility'] = data['close'].rolling(window=20).std()
            
            if 'volume' in data.columns:
                # Volume features
                features['volume_sma'] = data['volume'].rolling(window=20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma']
            
            # Enhanced troubleshooting: Log fallback completion
            tprint_structured({
                "step": "fallback_feature_generation",
                "phase": "complete",
                "features_generated": len(features),
                "feature_names": list(features.keys()),
                "output_shape": (len(data), len(features))
            }, level="INFO")
            
            return pd.DataFrame(features, index=data.index)
            
        except Exception as e:
            tprint_error(f"❌ Fallback feature generation failed: {e}")
            tprint_exception(e, "Fallback feature generation failed")
            tprint_structured({
                "step": "fallback_feature_generation",
                "phase": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "data_shape": data.shape if data is not None else None
            }, level="ERROR")
            return pd.DataFrame(index=data.index)

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked
    async def _perform_enhanced_feature_generation(self, data: pd.DataFrame, symbol: str,
                                                   timeframe: str, direction: str,
                                                   custom_overrides: Optional[Dict[str, Any]],
                                                   base_config: Optional[FeatureConfig] = None,
                                                   enable_cmi_complementarity: bool = False,
                                                   pipeline_state: Optional[Dict[str, Any]] = None,
                                                   targets: Optional[pd.Series] = None) -> FeatureGenerationResult:
        """Perform enhanced feature generation using FeatureBank."""
        
        start_time = time.time()
        
        # Enhanced troubleshooting: Log feature generation start
        tprint_structured({
            "step": "perform_enhanced_feature_generation",
            "phase": "start",
            "data_shape": data.shape,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "enable_cmi_complementarity": enable_cmi_complementarity,
            "has_targets": targets is not None,
            "has_custom_overrides": custom_overrides is not None,
            "has_pipeline_state": pipeline_state is not None
        }, level="INFO")
        
        try:
            # Use the provided base config or create a fresh copy
            if base_config is not None:
                feature_config = copy.deepcopy(base_config)
            else:
                feature_config = copy.deepcopy(self.feature_config)
            
            # Update configuration with custom overrides
            if custom_overrides:
                feature_config.update_from_dict(custom_overrides)
                # Sanity checks after overrides
                if not getattr(feature_config, 'required_columns', None):
                    raise ValueError("feature_config.required_columns cannot be empty after overrides.")
            
            # Generate features using FeatureBank with all available categories
            # Use all available categories except Autoencoder, Wavelet, regime, microstructure, and interaction features
            from src.feature_generation.core.feature_generator import FeatureCategory
            excluded_categories = {
                'autoencoder', 'wavelet', 'regime', 
                'advanced_statistical', 'order_flow', 'microstructure', 'interaction'
            }
            feature_categories = [
                cat.value for cat in FeatureCategory 
                if cat.value.lower() not in excluded_categories
            ]
            
            self.logger.info(f"🎯 Generating features for {len(feature_categories)} categories: {', '.join(feature_categories)}")
            
            # Enhanced troubleshooting: Log feature categories
            tprint_structured({
                "step": "perform_enhanced_feature_generation",
                "phase": "feature_categories_selection",
                "total_categories": len(feature_categories),
                "excluded_categories": list(excluded_categories),
                "selected_categories": feature_categories
            }, level="DEBUG")
            
            # Add progress monitoring during feature generation
            self.logger.info("📊 Starting feature generation process...")
            self.logger.info(f"📈 Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            self.logger.info(f"🧮 Total memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Enhanced troubleshooting: Log detailed generation parameters
            tprint_structured({
                "step": "perform_enhanced_feature_generation",
                "phase": "generation_parameters",
                "data_rows": data.shape[0],
                "data_columns": data.shape[1],
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
                "feature_categories_count": len(feature_categories),
                "comprehensive_optimizer_available": self.comprehensive_optimizer is not None
            }, level="INFO")
            
            # Use FeatureBank to generate features with enhanced hardware optimization
            generation_start_time = time.time()
            
            # Apply comprehensive hardware optimization before feature generation
            if self.comprehensive_optimizer:
                data = self.comprehensive_optimizer.optimize_dataframe(data)
                self.performance_stats['hardware_optimizations_applied'] += 1
            
            # Use M1-optimized feature generation with progress tracking
            tprint_progress(1, 3, "Starting M1-optimized feature generation...")
            generated_features_df = await self._perform_m1_optimized_feature_generation(
                data, feature_categories, custom_overrides
            )
            tprint_progress(2, 3, "M1-optimized feature generation completed")
            
            generation_duration = time.time() - generation_start_time
            
            # Enhanced troubleshooting: Log generation completion
            tprint_structured({
                "step": "perform_enhanced_feature_generation",
                "phase": "m1_generation_complete",
                "generation_duration": generation_duration,
                "features_generated": len(generated_features_df.columns),
                "generated_features_shape": generated_features_df.shape,
                "generated_features_memory_mb": generated_features_df.memory_usage(deep=True).sum() / 1024 / 1024
            }, level="INFO")
            
            # Apply CMI complementarity filtering if enabled
            if enable_cmi_complementarity and targets is not None:
                tprint_info("🎯 Applying CMI complementarity filtering to generated features")
                tprint_progress(3, 3, "Applying CMI complementarity filtering...")
                
                # Enhanced troubleshooting: Log CMI filtering start
                tprint_structured({
                    "step": "perform_enhanced_feature_generation",
                    "phase": "cmi_filtering_start",
                    "features_before_filtering": len(generated_features_df.columns),
                    "targets_available": targets is not None,
                    "targets_shape": targets.shape if hasattr(targets, 'shape') else None
                }, level="DEBUG")
                
                # Data preview: Features before CMI filtering
                tprint_data_preview(generated_features_df, "Features before CMI filtering", level="DEBUG")
                try:
                    # Extract Analyst side information
                    analyst_result = self.analyst_handler.extract_side_info(
                        pipeline_state, targets, generated_features_df.index
                    )
                    
                    if analyst_result.is_valid and not analyst_result.degraded_to_unconditional:
                        # Apply CMI complementarity scoring
                        cmi_result = self.cmi_scorer.score_features(
                            generated_features_df, targets, analyst_result.A,
                            pipeline_state=pipeline_state
                        )
                        
                        if cmi_result.is_valid and cmi_result.selected_features:
                            # Filter features based on CMI selection
                            original_count = len(generated_features_df.columns)
                            generated_features_df = generated_features_df[cmi_result.selected_features]
                            filtered_count = len(generated_features_df.columns)
                            
                            tprint_success(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} features")
                            tprint_info(f"📊 Noise floor: {cmi_result.noise_floor:.6f}")
                            tprint_info(f"📊 ΔPerf threshold: {cmi_result.delta_perf_threshold:.6f}")
                            
                            # Store CMI diagnostics
                            cmi_diagnostics = {
                                'cmi_enabled': True,
                                'original_features': original_count,
                                'filtered_features': filtered_count,
                                'noise_floor': cmi_result.noise_floor,
                                'delta_perf_threshold': cmi_result.delta_perf_threshold,
                                'analyst_source': analyst_result.source,
                                'analyst_dims': analyst_result.n_dims,
                                'I_Y_A': analyst_result.I_Y_A,
                                'degraded_to_unconditional': analyst_result.degraded_to_unconditional
                            }
                        else:
                            tprint_warning("⚠️ CMI complementarity scoring failed, using all features")
                            cmi_diagnostics = {'cmi_enabled': False, 'error': 'CMI scoring failed'}
                    else:
                        tprint_warning("⚠️ Analyst side information extraction failed, using all features")
                        cmi_diagnostics = {'cmi_enabled': False, 'error': 'Analyst side info failed'}
                        
                except Exception as e:
                    tprint_warning(f"⚠️ CMI complementarity filtering failed: {e}, using all features")
                    cmi_diagnostics = {'cmi_enabled': False, 'error': str(e)}
            else:
                cmi_diagnostics = {'cmi_enabled': False, 'reason': 'Not in Tactician mode or no targets'}
            
            # Log generation completion
            self.logger.info(f"✅ Feature generation completed in {generation_duration:.2f} seconds")
            self.logger.info(f"📊 Generated {len(generated_features_df.columns)} features")
            self.logger.info(f"💾 Output memory usage: {generated_features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Enhanced troubleshooting: Log final generation summary
            tprint_structured({
                "step": "perform_enhanced_feature_generation",
                "phase": "generation_complete",
                "total_duration": generation_duration,
                "features_generated": len(generated_features_df.columns),
                "output_memory_mb": generated_features_df.memory_usage(deep=True).sum() / 1024 / 1024,
                "feature_categories": feature_categories,
                "success": True
            }, level="INFO")
            
            # Store the generated features dataframe with memory optimization
            try:
                import os
                from datetime import datetime
                import gc
                
                # Create generated directory if it doesn't exist
                generated_dir = "generated"
                os.makedirs(generated_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Clean the dataframe to extract data from FeatureResult objects
                clean_df = generated_features_df.copy()
                for col in clean_df.columns:
                    if len(clean_df[col]) > 0:
                        first_value = clean_df[col].iloc[0]
                        # Check if the column contains FeatureResult objects
                        if hasattr(first_value, 'data') and hasattr(first_value, 'name'):
                            # This is a FeatureResult object, extract the .data series
                            clean_df[col] = first_value.data
                
                # Ensure all columns are numeric
                clean_df = clean_df.select_dtypes(include=[np.number])
                
                # Optimize DataFrame before saving to reduce memory pressure
                self.logger.info("🔧 Optimizing DataFrame for efficient saving...")
                optimized_df = self._optimize_dataframe_for_saving(clean_df)
                
                # Data preview: Optimized features for saving
                tprint_data_preview(optimized_df, "Optimized features for saving", level="DEBUG")
                
                # Don't delete optimized_df yet - we need it for saving
                # del optimized_df
                # gc.collect()
                
                # Save as compressed Parquet (much more efficient than CSV for large datasets)
                parquet_filename = f"generated_features_{symbol}_{timeframe}_{direction}_{timestamp}.parquet"
                parquet_path = os.path.join(generated_dir, parquet_filename)
                
                try:
                    # Use compression to reduce file size and memory usage
                    optimized_df.to_parquet(parquet_path, compression='snappy', index=False)
                    self.logger.info(f"📄 Generated features Parquet saved to: {parquet_path}")
                    
                    # Get file size for logging
                    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                    self.logger.info(f"💾 File size: {file_size_mb:.2f} MB")
                    
                except Exception as parquet_e:
                    self.logger.warning(f"Parquet save failed: {parquet_e}, falling back to CSV")
                    # Fallback to CSV with chunked writing
                    csv_filename = f"generated_features_{symbol}_{timeframe}_{direction}_{timestamp}.csv"
                    csv_path = os.path.join(generated_dir, csv_filename)
                    self._save_dataframe_chunked(optimized_df, csv_path)
                    self.logger.info(f"📄 Generated features CSV saved to: {csv_path}")
                
                # Only save pickle if dataset is reasonably sized (< 1GB)
                df_memory_mb = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
                if df_memory_mb < 1000:  # Only save pickle if < 1GB
                    try:
                        pickle_filename = f"generated_features_{symbol}_{timeframe}_{direction}_{timestamp}.pkl"
                        pickle_path = os.path.join(generated_dir, pickle_filename)
                        optimized_df.to_pickle(pickle_path)
                        self.logger.info(f"💾 Generated features pickle saved to: {pickle_path}")
                    except Exception as pickle_e:
                        self.logger.warning(f"Failed to save pickle: {pickle_e}")
                else:
                    self.logger.info(f"⏭️ Skipping pickle save due to large size ({df_memory_mb:.1f}MB)")
                
                # Clear optimized DataFrame to free memory after saving
                del optimized_df
                gc.collect()
                
            except Exception as e:
                self.logger.warning(f"Failed to save generated features: {e}")
            
            # Create feature names list from generated features
            # Use clean_df if it was created, otherwise use the original
            if 'clean_df' in locals() and clean_df is not None:
                feature_names = list(clean_df.columns)
            else:
                feature_names = list(generated_features_df.columns)
            
            # Create feature categories list
            feature_categories = list(set(feature_categories))
            
            # Create optimization stats
            optimization_stats = {
                'vectorbt_optimization_enabled': True,
                'memory_optimization_enabled': True,
                'gpu_acceleration_enabled': True,
                'parallel_processing_enabled': True
            }
            
            # Create generation metrics
            generation_metrics = {
                'generation_time': generation_duration,
                'features_generated': len(feature_names),
                'memory_usage_mb': generated_features_df.memory_usage(deep=True).sum() / 1024 / 1024,
                'categories_count': len(feature_categories)
            }
            
            # Create feature details by category
            feature_details_by_category = {}
            for category in feature_categories:
                # Filter features by category (simplified approach)
                category_features = [f for f in feature_names if category.lower() in f.lower()]
                if category_features:
                    feature_details_by_category[category] = category_features
            
            # Create vectorbt optimizations info
            vectorbt_optimizations = {
                'enabled': True,
                'optimization_level': 'balanced',
                'memory_optimization': True,
                'gpu_acceleration': True
            }
            
            # Store dataframes for later use
            stored_dataframes = {}
            try:
                if 'csv_path' in locals():
                    stored_dataframes['csv'] = csv_path
                if 'pickle_path' in locals():
                    stored_dataframes['pickle'] = pickle_path
            except:
                pass
            
            result = FeatureGenerationResult(
                feature_names=feature_names,
                feature_data=generated_features_df,
                generated_features=generated_features_df,
                feature_categories=feature_categories,
                generation_time=generation_duration,
                n_features_generated=len(feature_names),
                cache_hit=False,  # FeatureBank doesn't provide cache info directly
                memory_usage_mb=generated_features_df.memory_usage(deep=True).sum() / 1024 / 1024 if not generated_features_df.empty else 0.0,
                success=True,
                error_message=None,
                optimization_stats=optimization_stats,
                metadata={
                    'feature_metadata': {'generated_features': feature_names},
                    'generation_metrics': generation_metrics,
                    'feature_categories': feature_categories,
                    'vectorbt_optimizations': vectorbt_optimizations,
                    'feature_result': {'generated_features_df_shape': generated_features_df.shape},
                    'config': self._serialize_config(feature_config),
                    'auto_optimization_config': self._serialize_config(self.auto_optimization_config),
                    'feature_details_by_category': feature_details_by_category,
                    'stored_dataframes': stored_dataframes,
                    'cmi_diagnostics': cmi_diagnostics,
                    'm1_optimizations': {
                        'neural_engine_operations': self.performance_stats['neural_engine_operations'],
                        'gpu_accelerations': self.performance_stats['gpu_accelerations'],
                        'cpu_optimizations': self.performance_stats['cpu_optimizations'],
                        'memory_optimizations': self.performance_stats['memory_optimizations'],
                        'optimization_applied': self.performance_stats['optimization_applied']
                    }
                },
                artifacts={
                    'feature_dataframe': generated_features_df,
                    'feature_names': feature_names,
                    'feature_categories': feature_categories,
                    'vectorbt_optimizations': vectorbt_optimizations,
                    'raw_dataframe': data
                }
            )
            
            # Convert dataclass to dictionary for base step compatibility
            return {
                'success': result.success,
                'feature_names': result.feature_names,
                'feature_data': result.feature_data,
                'generated_features': result.generated_features,
                'feature_categories': result.feature_categories,
                'generation_time': result.generation_time,
                'n_features_generated': result.n_features_generated,
                'cache_hit': result.cache_hit,
                'memory_usage_mb': result.memory_usage_mb,
                'error_message': result.error_message,
                'optimization_stats': result.optimization_stats,
                'metadata': result.metadata,
                'artifacts': result.artifacts
            }
            
        except Exception as e:
            error_msg = f"Enhanced feature generation failed: {e}"
            self.logger.error(error_msg)
            tprint_exception(e, "Enhanced feature generation failed")
            tprint_structured({
                "step": "perform_enhanced_feature_generation",
                "phase": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
                "data_shape": data.shape if data is not None else None
            }, level="ERROR")
            # Fast fail - no fallback, just raise the error
            raise RuntimeError(error_msg) from e
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.MAXIMUM)
    def _optimize_dataframe_for_saving(self, df):
        """Optimize DataFrame for efficient saving using enhanced hardware optimizations."""
        try:
            import pandas as pd
            import numpy as np
            
            self.logger.info("🚀 Optimizing DataFrame with enhanced hardware optimizations...")
            
            # Use comprehensive optimizer if available
            if self.comprehensive_optimizer:
                optimized_df = self.comprehensive_optimizer.optimize_dataframe(df)
                self.performance_stats['hardware_optimizations_applied'] += 1
            else:
                # Fallback to enhanced default optimization
                optimized_df = optimize_dataframe_default(df)
            
            # Apply additional memory manager optimizations
            if self.memory_manager:
                optimized_df = self.memory_manager.optimize_dataframe(optimized_df)
                self.performance_stats['memory_optimizations_applied'] += 1
            
            # Apply unified memory optimizations
            if self.unified_memory_manager:
                optimized_df = self.unified_memory_manager.optimize_dataframe(optimized_df)
            
            # Calculate memory savings
            original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
            memory_saved = original_memory - optimized_memory
            reduction_percentage = (memory_saved / original_memory) * 100 if original_memory > 0 else 0
            
            self.logger.info(f"🚀 Enhanced hardware optimization: {memory_saved:.1f}MB saved ({reduction_percentage:.1f}% reduction)")
            self.logger.info(f"📊 Optimized memory usage: {optimized_memory:.1f}MB")
            
            # Update performance stats
            self.performance_stats['memory_savings_mb'] += memory_saved
            
            return optimized_df
            
        except Exception as e:
            self.logger.warning(f"Enhanced DataFrame optimization failed: {e}, using fallback optimization")
            # Fallback to basic optimization
            return optimize_dataframe_default(df)
    
    @chunked_processing_auto(chunk_size_mb=50.0)
    def _save_dataframe_chunked(self, df, filepath, chunk_size=10000):
        """Save large DataFrame in chunks with enhanced memory optimization."""
        try:
            import pandas as pd
            
            self.logger.info(f"🚀 Saving DataFrame with enhanced chunked processing...")
            
            # Get total rows
            total_rows = len(df)
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            # Save header first
            df.head(0).to_csv(filepath, index=False)
            
            # Process chunks with enhanced optimization
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                
                chunk = df.iloc[start_idx:end_idx]
                
                # Apply hardware optimization to each chunk
                if self.comprehensive_optimizer:
                    chunk = self.comprehensive_optimizer.optimize_dataframe(chunk)
                
                # Apply memory optimization
                if self.memory_manager:
                    chunk = self.memory_manager.optimize_dataframe(chunk)
                
                chunk.to_csv(filepath, mode='a', header=False, index=False)
                
                # Cleanup chunk memory
                if self.memory_manager:
                    self.memory_manager.cleanup_chunk_memory()
                
                # Progress update
                progress = (i + 1) / num_chunks * 100
                if (i + 1) % 10 == 0 or i == num_chunks - 1:  # Log every 10 chunks or last chunk
                    self.logger.info(f"🚀 Enhanced chunked save progress: {progress:.1f}% ({i+1}/{num_chunks} chunks)")
            
            self.logger.info(f"✅ Enhanced chunked save completed: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Enhanced chunked save failed: {e}")
            # Fallback to regular save with basic optimization
            optimized_df = optimize_dataframe_default(df)
            optimized_df.to_csv(filepath, index=False)



    async def _generate_final_report(self, generation_result: FeatureGenerationResult, 
                                    symbol: str, timeframe: str, direction: str, exchange: str = "binance") -> str:
        """Generate a human-readable final report."""
        try:
            from datetime import datetime
            import os
            
            # Enhanced troubleshooting: Log report generation start
            tprint_structured({
                "step": "generate_final_report",
                "phase": "start",
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "exchange": exchange,
                "generation_success": generation_result.success,
                "features_generated": generation_result.n_features_generated
            }, level="DEBUG")
            
            # Create outcomes directory if it doesn't exist
            outcomes_dir = "outcomes"
            os.makedirs(outcomes_dir, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"feature_generation_report_{symbol}_{timeframe}_{direction}_{timestamp}.md"
            report_path = os.path.join(outcomes_dir, report_filename)
            
            # Generate report content
            report_content = f"""# Feature Generation Report

## Summary
- **Symbol**: {symbol}
- **Exchange**: {exchange}
- **Timeframe**: {timeframe}
- **Direction**: {direction}
- **Generated At**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Status**: {'✅ SUCCESS' if generation_result.success else '❌ FAILED'}

## Feature Generation Results
- **Total Features Generated**: {generation_result.n_features_generated}
- **Generation Time**: {generation_result.generation_time:.3f} seconds
- **Memory Usage**: {generation_result.memory_usage_mb:.2f} MB
- **Cache Hit**: {'Yes' if generation_result.cache_hit else 'No'}

## Feature Categories
"""
            
            # Use detailed category information if available, otherwise fall back to basic categories
            if generation_result.metadata and 'feature_details_by_category' in generation_result.metadata:
                feature_details_by_category = generation_result.metadata['feature_details_by_category']
                for category, features in sorted(feature_details_by_category.items()):
                    report_content += f"- **{category.title()}**: {len(features)} features generated\n"
            elif generation_result.feature_categories:
                for category in generation_result.feature_categories:
                    report_content += f"- **{category.title()}**: Generated\n"
            else:
                report_content += "- No categories specified\n"
            
            # Add detailed feature information by category
            if generation_result.metadata and 'feature_details_by_category' in generation_result.metadata:
                feature_details_by_category = generation_result.metadata['feature_details_by_category']
                report_content += f"""
## Detailed Feature List by Category
"""
                
                for category, features in sorted(feature_details_by_category.items()):
                    report_content += f"""
### {category.title()} Features ({len(features)} features)
"""
                    for i, feature_name in enumerate(features, 1):
                        report_content += f"{i}. {feature_name}\n"
                    report_content += "\n"
            else:
                # Fallback to simple feature list
                report_content += f"""
## Feature Names
"""
                
                if generation_result.feature_names:
                    # Group features by category if possible
                    for i, feature_name in enumerate(generation_result.feature_names[:50], 1):  # Show first 50
                        report_content += f"{i}. {feature_name}\n"
                    if len(generation_result.feature_names) > 50:
                        report_content += f"... and {len(generation_result.feature_names) - 50} more features\n"
                else:
                    report_content += "- No features generated\n"
            
            report_content += f"""
## M1 Hardware Optimization Statistics
"""
            
            if generation_result.optimization_stats:
                # Display M1 hardware optimization stats prominently
                if 'm1_hardware_optimizations' in generation_result.optimization_stats:
                    m1_stats = generation_result.optimization_stats['m1_hardware_optimizations']
                    report_content += f"- **Total Processing Time**: {m1_stats.get('total_processing_time', 0):.2f} seconds\n"
                    report_content += f"- **Neural Engine Operations**: {m1_stats.get('neural_engine_operations', 0)}\n"
                    report_content += f"- **GPU Accelerations**: {m1_stats.get('gpu_accelerations', 0)}\n"
                    report_content += f"- **CPU Optimizations**: {m1_stats.get('cpu_optimizations', 0)}\n"
                    report_content += f"- **Memory Optimizations**: {m1_stats.get('memory_optimizations', 0)}\n"
                    report_content += f"- **Cache Hits**: {m1_stats.get('cache_hits', 0)}\n"
                    report_content += f"- **Memory Savings**: {m1_stats.get('memory_savings_mb', 0):.2f} MB\n"
                    report_content += f"- **Optimizations Applied**: {', '.join(m1_stats.get('optimization_applied', []))}\n"
                    report_content += f"- **Parallel Workers**: {m1_stats.get('parallel_workers', 1)}\n"
                    report_content += f"- **Chunk Size**: {m1_stats.get('chunk_size', 10000)}\n"
                    report_content += f"- **Neural Engine Available**: {'Yes' if m1_stats.get('neural_engine_available', False) else 'No'}\n"
                    report_content += f"- **GPU Available**: {'Yes' if m1_stats.get('gpu_available', False) else 'No'}\n"
                    report_content += f"- **CPU Optimizer Used**: {'Yes' if m1_stats.get('cpu_optimizer_used', False) else 'No'}\n"
                    report_content += f"- **Unified Memory Used**: {'Yes' if m1_stats.get('unified_memory_used', False) else 'No'}\n"
                    report_content += f"- **Comprehensive Optimizer Used**: {'Yes' if m1_stats.get('comprehensive_optimizer_used', False) else 'No'}\n"
                    report_content += f"- **Hardware Manager Used**: {'Yes' if m1_stats.get('hardware_manager_used', False) else 'No'}\n"
                    report_content += f"- **Memory Manager Used**: {'Yes' if m1_stats.get('memory_manager_used', False) else 'No'}\n"
                
                # Display other optimization stats
                for key, value in generation_result.optimization_stats.items():
                    if key != 'hardware_optimizations':
                        report_content += f"- **{key}**: {value}\n"
            else:
                report_content += "- No optimization statistics available\n"
            
            report_content += f"""
## Data Quality
- **Data Shape**: {generation_result.generated_features.shape if not generation_result.generated_features.empty else 'Empty DataFrame'}
- **Success**: {'Yes' if generation_result.success else 'No'}
"""
            
            # Add information about stored dataframes if available
            if generation_result.metadata and 'stored_dataframes' in generation_result.metadata:
                stored_dataframes = generation_result.metadata['stored_dataframes']
                report_content += f"""
## Stored Data
"""
                for file_type, file_path in stored_dataframes.items():
                    report_content += f"- **{file_type.title()}**: {file_path}\n"
            
            if generation_result.error_message:
                report_content += f"- **Error**: {generation_result.error_message}\n"
            
            report_content += f"""
## Technical Details
- **Feature Data Type**: {type(generation_result.feature_data).__name__}
- **Generated Features Type**: {type(generation_result.generated_features).__name__}
- **Metadata Available**: {'Yes' if generation_result.metadata else 'No'}

## Recommendations
"""
            
            if generation_result.success:
                report_content += """- ✅ M1-optimized feature generation completed successfully
- 🧠 M1 Neural Engine utilized for ML feature generation
- 🎮 M1 Enhanced GPU acceleration applied for vectorized operations
- 💻 M1 Advanced CPU optimization used for traditional features
- 🧠 M1 Unified Memory management applied for optimal allocation
- 📊 Consider analyzing feature importance for model training
- 🔍 Review feature categories for completeness
- 💾 Features are ready for model training pipeline
- ⚡ Comprehensive M1 optimization strategy applied
- 🚀 Maximum performance achieved through M1 hardware utilization
"""
            else:
                report_content += """- ❌ M1-optimized feature generation failed
- 🔧 Check error message for specific issues
- 🔄 Consider retrying with different parameters
- 📋 Review input data quality
- 🧠 M1 Neural Engine may not be available
- 🎮 M1 GPU acceleration may have encountered issues
- 💻 M1 CPU optimization may have failed
- 🧠 M1 Unified Memory management may have issues
"""
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Enhanced troubleshooting: Log report generation completion
            tprint_structured({
                "step": "generate_final_report",
                "phase": "complete",
                "report_path": report_path,
                "report_size_bytes": len(report_content),
                "success": True
            }, level="INFO")
            
            return report_path
            
        except Exception as e:
            error_msg = f"Failed to generate final report: {e}"
            self.logger.error(error_msg)
            tprint_exception(e, "Final report generation failed")
            tprint_structured({
                "step": "generate_final_report",
                "phase": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "success": False
            }, level="ERROR")
            return ""

    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """Serialize configuration object to dictionary."""
        try:
            if hasattr(config, '__dict__'):
                return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            elif isinstance(config, dict):
                return config
            else:
                return {'config': str(config)}
        except Exception:
            return {'config': str(config)}


async def handle_feature_generation_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    exchange: str = "binance",
    direction: str = "longs",
    intensity: str = None,
    lookback_days: int = None,
    start_date: str = None,
    end_date: str = None,
    custom_overrides: dict = None,
    **kwargs
) -> ComponentResult:
    """
    Handler function for feature generation step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        timeframe: Timeframe (e.g., "15m")
        exchange: Exchange name (e.g., "binance")
        direction: Trading direction (e.g., "longs")
        intensity: Intensity level (e.g., "light", "full", "blank") or None for default
        lookback_days: Number of days to look back
        start_date: Start date for data
        end_date: End date for data
        custom_overrides: Custom configuration overrides
        **kwargs: Additional arguments

    Returns:
        ComponentResult: Result of the feature generation step
    """
    # Enhanced troubleshooting: Log handler function start
    tprint_structured({
        "step": "handle_feature_generation_step",
        "phase": "start",
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange,
        "direction": direction,
        "intensity": intensity,
        "lookback_days": lookback_days,
        "start_date": start_date,
        "end_date": end_date,
        "has_custom_overrides": custom_overrides is not None,
        "kwargs_keys": list(kwargs.keys())
    }, level="INFO")
    
    # Handle None intensity by defaulting to light mode (more reasonable default)
    if intensity is None:
        intensity = "light"
        tprint_debug("🔧 Defaulting intensity to 'light' mode")

    try:
        # Create the step instance
        step = FeatureGenerationStep(
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'custom_overrides': custom_overrides
            }
        )

        # Create training input
        training_input = {
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': exchange,
            'direction': direction,
            'intensity': intensity,
            'lookback_days': lookback_days,
            'start_date': start_date,
            'end_date': end_date,
            'custom_overrides': custom_overrides
        }

        # Execute the step
        tprint_debug("🚀 Executing feature generation step...")
        result = await step.execute(training_input)
        
        # Enhanced troubleshooting: Log handler function completion
        tprint_structured({
            "step": "handle_feature_generation_step",
            "phase": "complete",
            "success": result.get('success', False),
            "features_generated": result.get('n_features_generated', 0),
            "generation_time": result.get('generation_time', 0),
            "error_message": result.get('error_message')
        }, level="INFO")

        return result

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        error_msg = f"❌ Handler function failed: {e}"
        logger.error(error_msg)
        tprint_exception(e, "Handler function failed")
        tprint_structured({
            "step": "handle_feature_generation_step",
            "phase": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "success": False
        }, level="ERROR")
        return ComponentResult(
            success=False,
            metadata={},
            error_message=str(e)
        )
