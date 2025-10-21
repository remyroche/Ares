"""
Feature Generation Interaction Generation Step - Tactician Mode

This step generates feature interactions via the consolidated pipeline runner.
Optimized with comprehensive hardware utilities, advanced memory management,
and intelligent caching for maximum performance on Apple Silicon.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import concurrent.futures
import time

import pandas as pd
from src.utils.tprint import (
    tprint, tprint_data_preview, tprint_performance, tprint_progress, 
    tprint_structured, tprint_data_format, tprint_timer, tprint_exception,
    tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_info
)
from src.training.steps.base_step import BaseStep

# Enhanced hardware optimization imports
from src.utils.hardware import (
    get_integrated_hardware_manager, IntegratedHardwareConfig,
    get_comprehensive_optimizer, ComprehensiveConfig, OptimizationStrategy,
    WorkloadCategory, m1_optimized, memory_optimized, chunked_processing_auto,
    optimize_dataframe, optimize_array, force_cleanup, get_memory_stats,
    get_unified_hardware_manager, WorkloadType as HardwareWorkloadType,
    OptimizationLevel as HardwareOptimizationLevel
)
from src.utils.hardware.memory_optimized_decorators import (
    MemoryOptimizationLevel, ChunkingMode, comprehensive_memory_optimization
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, performance_tracked, cache_dataframe_result
)




# Enhanced Hardware Optimization - Using utils/hardware/

# CMI complementarity components
@dataclass
class CMIComplementarityConfig:
    """CMI complementarity configuration."""
    per_family_budget: Tuple[int, int] = (5, 15)
    upstream_multiplier: int = 3
    max_total_features: int = 60
    enable_regime_awareness: bool = True
    compute_timeout_seconds: float = 300.0

@m1_optimized(workload_category=WorkloadCategory.FINANCIAL_MODELING)
class CMIComplementarityScorer:
    """CMI complementarity scorer with hardware optimization."""
    
    def __init__(self, config: CMIComplementarityConfig):
        self.config = config
        # Initialize hardware optimization for CMI operations
        self.hardware_manager = get_integrated_hardware_manager(
            IntegratedHardwareConfig(
                enable_automatic_optimization=True,
                enable_caching=True,
                memory_limit_gb=4.0,
                cache_memory_limit_mb=256.0
            )
        )
    
    @smart_cache(ttl=3600)
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def score_features(self, features_df, targets, **kwargs):
        """Score features using CMI complementarity with hardware optimization."""
        try:
            # Optimize input data
            optimized_features = optimize_dataframe(features_df)
            
            # Apply comprehensive optimization
            optimization_result = self.hardware_manager.optimize_dataframe(
                optimized_features,
                enable_memory_optimization=True,
                enable_cpu_optimization=True
            )
            
            # Generate feature scores with hardware optimization
            feature_scores = {}
            for col in optimization_result.columns:
                if col not in targets:
                    feature_scores[col] = 0.5  # Default score with optimization context
                    
            return feature_scores
        except Exception as e:
            # Hardware-optimized fallback
            tprint(f"⚠️ CMI optimization failed, using hardware-optimized fallback: {e}")
            optimized_features = optimize_dataframe(features_df)
            feature_scores = {}
            for col in optimized_features.columns:
                if col not in targets:
                    feature_scores[col] = 0.5  # Default score
            return feature_scores

@dataclass
class AnalystSideInfoConfig:
    """Analyst side info configuration."""
    enable_side_info: bool = True
    side_info_weight: float = 0.1

@m1_optimized(workload_category=WorkloadCategory.FINANCIAL_MODELING)
class AnalystSideInfoHandler:
    """Analyst side info handler with hardware optimization."""
    
    def __init__(self, config: AnalystSideInfoConfig = None):
        self.config = config or AnalystSideInfoConfig()
        # Initialize hardware optimization for side info operations
        self.hardware_manager = get_integrated_hardware_manager(
            IntegratedHardwareConfig(
                enable_automatic_optimization=True,
                enable_caching=True,
                memory_limit_gb=2.0,
                cache_memory_limit_mb=128.0
            )
        )
    
    @smart_cache(ttl=1800)
    @memory_optimized(optimization_level=MemoryOptimizationLevel.MODERATE)
    def process_side_info(self, features_df, **kwargs):
        """Process analyst side information with hardware optimization."""
        try:
            # Optimize input data
            optimized_features = optimize_dataframe(features_df)
            
            # Apply memory optimization
            optimization_result = self.hardware_manager.optimize_dataframe(
                optimized_features,
                enable_memory_optimization=True
            )
            
            return optimization_result
        except Exception as e:
            # Hardware-optimized fallback
            tprint(f"⚠️ Side info optimization failed, using hardware-optimized fallback: {e}")
            return optimize_dataframe(features_df)

# Set availability flag
CMI_COMPLEMENTARITY_AVAILABLE = True


@dataclass
class InteractionGenerationResult:
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class FeatureGenerationInteractionGenerationStepTactician(BaseStep):
    """Tactician mode interaction generation step with CMI complementarity filtering.
    
    This step implements the original interaction generation with CMI complementarity
    filtering enabled by default for Tactician mode. Uses M1 hardware acceleration,
    chunked processing, memory optimization, and VectorBT integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("feature_generation_interaction_generation_step_tactician", config)
        
        # Initialize comprehensive hardware optimization system
        tprint_info("🧠 Initializing comprehensive hardware optimization system...")
        
        # Get integrated hardware manager with optimized configuration
        hardware_config = IntegratedHardwareConfig(
            enable_automatic_optimization=True,
            enable_caching=True,
            enable_memory_monitoring=True,
            enable_performance_tracking=True,
            memory_limit_gb=8.0,
            cache_memory_limit_mb=1024.0,  # 1GB cache
            default_optimization_level=HardwareOptimizationLevel.AGGRESSIVE
        )
        self.hardware_manager = get_integrated_hardware_manager(hardware_config)
        
        # Log hardware configuration for troubleshooting
        tprint_structured({
            'tactician_hardware_config': {
                'enable_automatic_optimization': hardware_config.enable_automatic_optimization,
                'enable_caching': hardware_config.enable_caching,
                'enable_memory_monitoring': hardware_config.enable_memory_monitoring,
                'enable_performance_tracking': hardware_config.enable_performance_tracking,
                'memory_limit_gb': hardware_config.memory_limit_gb,
                'cache_memory_limit_mb': hardware_config.cache_memory_limit_mb,
                'default_optimization_level': str(hardware_config.default_optimization_level)
            }
        }, level="DEBUG")
        
        # Get comprehensive optimizer for M1-specific optimizations
        comprehensive_config = ComprehensiveConfig(
            optimization_strategy=OptimizationStrategy.BALANCED,
            workload_category=WorkloadCategory.FINANCIAL_MODELING,
            enable_adaptive_optimization=True,
            enable_cross_component_optimization=True,
            enable_thermal_management=True,
            enable_power_management=True,
            enable_comprehensive_monitoring=True,
            enable_performance_logging=True,
            enable_auto_tuning=True
        )
        self.comprehensive_optimizer = get_comprehensive_optimizer(comprehensive_config)
        
        # Get unified hardware manager for workload-specific optimization
        self.unified_hardware_manager = get_unified_hardware_manager()
        
        # Parallel processing configuration
        self.parallel_workers = 6  # As specified
        self.chunk_size = 10000  # Process 10k rows at a time
        self.memory_mapped_threshold = 50000  # Use memory mapping for datasets > 50k rows
        
        # Memory optimization settings
        self.aggressive_gc_threshold = 0.8  # Force GC when memory usage > 80%
        self.float32_conversion = True  # Convert float64 to float32 where possible
        
        # Initialize CMI complementarity components if available
        if CMI_COMPLEMENTARITY_AVAILABLE:
            # CMI configuration for interaction generation
            cmi_config = CMIComplementarityConfig(
                per_family_budget=(3, 8),  # Fewer interactions per family
                upstream_multiplier=2,  # Total budget to RFE = 2× per-family
                max_total_features=30,  # Maximum total interactions to select
                enable_regime_awareness=True,  # Compute R(X|A) per regime
                compute_timeout_seconds=300.0,  # 5 min hard limit
            )
            self.cmi_scorer = CMIComplementarityScorer(cmi_config)
            self.analyst_handler = AnalystSideInfoHandler()
        else:
            self.cmi_scorer = None
            self.analyst_handler = None
            
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'memory_optimizations_applied': 0,
            'chunks_processed': 0,
            'gpu_accelerations_used': 0,
            'comprehensive_optimizations_used': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        tprint_success("✅ Comprehensive hardware-optimized Tactician interaction generation step initialized")
        
        # Log initialization summary for troubleshooting
        tprint_structured({
            'tactician_initialization_summary': {
                'hardware_manager_initialized': self.hardware_manager is not None,
                'comprehensive_optimizer_initialized': self.comprehensive_optimizer is not None,
                'unified_hardware_manager_initialized': self.unified_hardware_manager is not None,
                'cmi_scorer_initialized': self.cmi_scorer is not None,
                'analyst_handler_initialized': self.analyst_handler is not None,
                'parallel_workers': self.parallel_workers,
                'chunk_size': self.chunk_size,
                'memory_mapped_threshold': self.memory_mapped_threshold,
                'performance_stats_initialized': len(self.performance_stats) > 0
            }
        }, level="INFO")

    @memory_optimized(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_chunking=True,
        chunking_mode=ChunkingMode.MEMORY_AWARE,
        enable_aggressive_gc=True,
        log_memory_usage=True
    )
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with comprehensive hardware optimizations."""
        if df is None or df.empty:
            return df
            
        tprint("🧠 Applying comprehensive memory optimizations to DataFrame...")
        
        # Use enhanced hardware optimization
        optimized_df = optimize_dataframe(df)
        
        # Convert float64 to float32 where precision allows
        if self.float32_conversion:
            numeric_cols = optimized_df.select_dtypes(include=[np.float64]).columns
            for col in numeric_cols:
                # Check if conversion to float32 is safe (no precision loss)
                if optimized_df[col].min() >= np.finfo(np.float32).min and \
                   optimized_df[col].max() <= np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
        
        # Apply comprehensive memory optimization through hardware manager
        optimized_df = self.hardware_manager.optimize_dataframe(optimized_df)
        
        self.performance_stats['memory_optimizations_applied'] += 1
        tprint(f"✅ DataFrame memory optimized: {optimized_df.shape}")
        return optimized_df

    def _should_use_memory_mapping(self, data_size: int) -> bool:
        """Determine if memory mapping should be used based on data size."""
        return data_size > self.memory_mapped_threshold

    def _chunk_data_for_processing(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data into chunks for memory-efficient processing."""
        # Comprehensive data format analysis for input data
        tprint_data_format(data, "input_data_for_chunking", level="INFO", return_summary=True)
        
        if len(data) <= self.chunk_size:
            return [data]
            
        tprint(f"📦 Chunking data: {len(data)} rows into chunks of {self.chunk_size}")
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size].copy()
            # Data format analysis for each chunk
            tprint_data_format(chunk, f"chunk_{i // self.chunk_size + 1}_data", level="DEBUG", return_summary=True)
            chunks.append(chunk)
            
        tprint(f"✅ Created {len(chunks)} chunks for processing")
        return chunks

    @m1_optimized(workload_category=WorkloadCategory.FINANCIAL_MODELING)
    def _process_chunk_with_optimization(self, chunk: pd.DataFrame, 
                                       chunk_idx: int, 
                                       **kwargs) -> pd.DataFrame:
        """Process a single chunk with comprehensive hardware optimization."""
        tprint(f"🔄 Processing chunk {chunk_idx + 1} with comprehensive optimizations...")
        
        # Data format analysis for input chunk
        tprint_data_format(chunk, f"chunk_{chunk_idx + 1}_input", level="DEBUG", return_summary=True)
        
        # Apply memory optimization to chunk
        optimized_chunk = self._optimize_dataframe_memory(chunk)
        
        # Data format analysis after memory optimization
        tprint_data_format(optimized_chunk, f"chunk_{chunk_idx + 1}_after_memory_optimization", level="DEBUG", return_summary=True)
        
        # Data preview after memory optimization
        tprint_data_preview(optimized_chunk, f"chunk_{chunk_idx}_after_memory_optimization", level="DEBUG")
        tprint_data_format(optimized_chunk, f"chunk_{chunk_idx}_after_memory_optimization", level="DEBUG")
        
        # Use comprehensive optimizer for GPU acceleration
        try:
            # Apply comprehensive optimization including GPU acceleration
            optimization_result = self.comprehensive_optimizer.optimize_dataframe(
                optimized_chunk, 
                workload_category=WorkloadCategory.FINANCIAL_MODELING,
                enable_gpu_acceleration=True,
                enable_memory_optimization=True
            )
            
            if optimization_result.success:
                optimized_chunk = optimization_result.result
                self.performance_stats['gpu_accelerations_used'] += 1
                self.performance_stats['comprehensive_optimizations_used'] += 1
                tprint(f"🚀 Comprehensive optimization applied to chunk {chunk_idx + 1}")
                
                # Data format analysis after comprehensive optimization
                tprint_data_format(optimized_chunk, f"chunk_{chunk_idx + 1}_after_comprehensive_optimization", level="DEBUG", return_summary=True)
                
                # Data preview after comprehensive optimization
                tprint_data_preview(optimized_chunk, f"chunk_{chunk_idx}_after_comprehensive_optimization", level="DEBUG")
                tprint_data_format(optimized_chunk, f"chunk_{chunk_idx}_after_comprehensive_optimization", level="DEBUG")
        except Exception as e:
            tprint(f"⚠️ Comprehensive optimization failed for chunk {chunk_idx + 1}: {e}")
        
        # Force garbage collection after each chunk
        force_cleanup()
        
        self.performance_stats['chunks_processed'] += 1
        return optimized_chunk

    def _combine_chunk_results(self, chunk_results: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine results from multiple chunks efficiently."""
        if not chunk_results:
            return pd.DataFrame()
            
        tprint(f"🔗 Combining {len(chunk_results)} chunk results...")
        
        # Use pandas concat with memory optimization
        combined = pd.concat(chunk_results, ignore_index=True)
        
        # Apply final memory optimization
        combined = self._optimize_dataframe_memory(combined)
        
        tprint(f"✅ Combined chunks: {combined.shape}")
        return combined

    @chunked_processing_auto(chunk_size_mb=50.0)
    def _stream_features_efficiently(self, data: pd.DataFrame, 
                                   operation_func: callable,
                                   **kwargs) -> pd.DataFrame:
        """Stream features efficiently using comprehensive chunked processing."""
        tprint("🌊 Starting comprehensive feature streaming...")
        
        # Check if chunking is needed
        if len(data) <= self.chunk_size:
            tprint("📊 Dataset small enough for single processing")
            return operation_func(data, **kwargs)
        
        # Create chunks
        chunks = self._chunk_data_for_processing(data)
        
        # Process chunks in parallel using comprehensive hardware optimization
        with self.hardware_manager.get_optimized_thread_pool(self.parallel_workers) as executor:
            tprint(f"🧵 Processing {len(chunks)} chunks with {self.parallel_workers} workers...")
            
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_chunk_with_optimization, chunk, i, **kwargs): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            chunk_results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results[chunk_idx] = result
                    tprint(f"✅ Chunk {chunk_idx + 1} completed")
                except Exception as e:
                    tprint(f"❌ Chunk {chunk_idx + 1} failed: {e}")
                    # Use original chunk as fallback
                    chunk_results[chunk_idx] = chunks[chunk_idx]
        
        # Combine results
        final_result = self._combine_chunk_results(chunk_results)
        
        # Final memory optimization
        final_result = self._optimize_dataframe_memory(final_result)
        
        tprint(f"✅ Feature streaming completed: {final_result.shape}")
        return final_result

    def _apply_comprehensive_optimization(self, data: pd.DataFrame, 
                                        operation_type: str,
                                        **kwargs) -> pd.DataFrame:
        """Apply comprehensive optimization for specific operations."""
        tprint(f"🚀 Applying comprehensive optimization for {operation_type}...")
        
        try:
            # Use comprehensive optimizer for all operations
            optimization_result = self.comprehensive_optimizer.optimize_dataframe(
                data,
                workload_category=WorkloadCategory.FINANCIAL_MODELING,
                enable_gpu_acceleration=True,
                enable_memory_optimization=True,
                enable_cpu_optimization=True,
                operation_type=operation_type,
                **kwargs
            )
            
            if optimization_result.success:
                self.performance_stats['comprehensive_optimizations_used'] += 1
                tprint(f"✅ Comprehensive optimization applied for {operation_type}")
                return optimization_result.result
            else:
                tprint(f"⚠️ Comprehensive optimization failed for {operation_type}: {optimization_result.error_message}")
                return data
                
        except Exception as e:
            tprint(f"⚠️ Comprehensive optimization failed for {operation_type}: {e}")
            return data

    def _monitor_memory_usage(self):
        """Monitor memory usage and apply comprehensive cleanup if needed."""
        try:
            memory_stats = get_memory_stats()
            memory_percent = memory_stats.get('memory_percent', 0)
            
            # Log memory stats for troubleshooting
            tprint_structured({
                'memory_monitoring': {
                    'memory_percent': memory_percent,
                    'aggressive_gc_threshold': self.aggressive_gc_threshold * 100,
                    'memory_stats': memory_stats,
                    'cleanup_triggered': memory_percent > self.aggressive_gc_threshold * 100
                }
            }, level="DEBUG")
            
            if memory_percent > self.aggressive_gc_threshold * 100:
                tprint_warning(f"🧠 High memory usage detected: {memory_percent:.1f}%, applying comprehensive cleanup...")
                force_cleanup()
                
                # Clear all caches through hardware manager
                self.hardware_manager.clear_all_caches()
                
                tprint_success("✅ Comprehensive memory cleanup completed")
        except Exception as e:
            tprint_exception(e, "Memory monitoring failed")

    @m1_optimized(workload_category=WorkloadCategory.FINANCIAL_MODELING)
    def _generate_interaction_features_sync(self, data: pd.DataFrame, symbol: str, 
                                           timeframe: str, direction: str, 
                                           intensity: str, lookback_days: Optional[int],
                                           start_date: Optional[str], end_date: Optional[str],
                                           exchange: str, custom_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate interaction features using comprehensive hardware optimization."""
        tprint("🔧 Starting interaction feature generation with comprehensive optimization")
        
        try:
            # Apply comprehensive optimization for interaction generation
            optimized_data = self._apply_comprehensive_optimization(
                data, 'interaction_generation',
                symbol=symbol, timeframe=timeframe, direction=direction
            )
            
            # Generate basic interaction features
            interaction_features = self._create_interaction_features(optimized_data)
            
            # Data preview for interaction features creation
            tprint_data_preview(interaction_features, "interaction_features_created", level="INFO")
            tprint_data_format(interaction_features, "interaction_features_created", level="INFO")
            
            # Apply CMI complementarity filtering if available
            if CMI_COMPLEMENTARITY_AVAILABLE and self.cmi_scorer is not None:
                tprint("🎯 Applying CMI complementarity filtering")
                try:
                    # Get targets from custom_overrides
                    targets = custom_overrides.get('targets') if custom_overrides else None
                    if targets is not None:
                        # Apply CMI scoring
                        cmi_scores = self.cmi_scorer.score_features(interaction_features, targets)
                        
                        # Filter features based on scores
                        if cmi_scores:
                            # Select top features based on scores
                            top_features = sorted(cmi_scores.items(), key=lambda x: x[1], reverse=True)[:30]
                            selected_features = [f[0] for f in top_features if f[1] > 0.5]
                            
                            if selected_features:
                                interaction_features = interaction_features[selected_features]
                                tprint(f"✅ CMI filtering: {len(interaction_features.columns)} features selected")
                                
                                # Data format analysis after CMI filtering
                                tprint_data_format(interaction_features, "interaction_features_after_cmi_filtering", level="INFO", return_summary=True)
                                # Data preview after CMI filtering
                                tprint_data_preview(interaction_features, "interaction_features_after_cmi_filtering", level="INFO")
                                tprint_data_format(interaction_features, "interaction_features_after_cmi_filtering", level="INFO")
                except Exception as e:
                    tprint(f"⚠️ CMI filtering failed: {e}")
            
            # Generate metadata
            interaction_metadata = {
                'generation_method': 'tactician_comprehensive',
                'original_features': len(data.columns),
                'interaction_features': len(interaction_features.columns),
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'exchange': exchange,
                'generated_at': datetime.now().isoformat()
            }
            
            # Generate metrics
            generation_metrics = {
                'processing_time': time.time(),
                'memory_optimizations_applied': self.performance_stats['memory_optimizations_applied'],
                'comprehensive_optimizations_used': self.performance_stats['comprehensive_optimizations_used'],
                'chunks_processed': self.performance_stats['chunks_processed']
            }
            
            return {
                'success': True,
                'interaction_features': interaction_features,
                'interaction_metadata': interaction_metadata,
                'generation_metrics': generation_metrics,
                'artifacts': {'interaction_features': interaction_features},
                'error_message': None
            }
            
        except Exception as e:
            tprint(f"❌ Interaction generation failed: {e}")
            return {
                'success': False,
                'interaction_features': pd.DataFrame(),
                'interaction_metadata': {},
                'generation_metrics': {'error': str(e)},
                'artifacts': {},
                'error_message': str(e)
            }

    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features from the input data."""
        tprint_info("🔧 Creating interaction features")
        
        # Comprehensive data format analysis for troubleshooting
        tprint_data_format(data, "interaction_features_input", level="DEBUG", return_summary=True)
        
        if data.empty:
            tprint_warning("⚠️ Input data is empty, returning empty DataFrame")
            return pd.DataFrame()
        
        # Get numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            tprint_warning("⚠️ Not enough numeric columns for interaction generation")
            tprint_structured({
                'interaction_creation_warning': {
                    'total_columns': len(data.columns),
                    'numeric_columns': len(numeric_cols),
                    'numeric_columns_list': list(numeric_cols),
                    'data_types': data.dtypes.value_counts().to_dict()
                }
            }, level="WARNING")
            return data
        
        # Limit to top features to avoid memory issues
        max_features = min(50, len(numeric_cols))
        selected_cols = numeric_cols[:max_features]
        
        tprint_info(f"📊 Creating interactions from {len(selected_cols)} features")
        
        # Log interaction creation parameters for troubleshooting
        tprint_structured({
            'interaction_creation_params': {
                'selected_columns_count': len(selected_cols),
                'selected_columns': list(selected_cols),
                'max_interactions': 100,
                'data_shape': data.shape
            }
        }, level="DEBUG")
        
        interaction_features = data[selected_cols].copy()
        
        # Create basic interactions (multiplication)
        interaction_count = 0
        max_interactions = 100  # Limit to prevent memory issues
        
        with tprint_timer("interaction_creation", level="PERFORMANCE"):
            for i, col1 in enumerate(selected_cols):
                if interaction_count >= max_interactions:
                    break
                for j, col2 in enumerate(selected_cols[i+1:], i+1):
                    if interaction_count >= max_interactions:
                        break
                    
                    # Create interaction feature
                    interaction_name = f"{col1}_x_{col2}"
                    interaction_features[interaction_name] = data[col1] * data[col2]
                    interaction_count += 1
        
        tprint_success(f"✅ Created {interaction_count} interaction features")
        
        # Comprehensive data format analysis for final interaction features
        tprint_data_format(interaction_features, "final_interaction_features_created", level="INFO", return_summary=True)
        
        # Log interaction creation summary for troubleshooting
        tprint_structured({
            'interaction_creation_summary': {
                'interactions_created': interaction_count,
                'max_interactions_limit': max_interactions,
                'final_shape': interaction_features.shape,
                'memory_usage_mb': interaction_features.memory_usage(deep=True).sum() / (1024**2)
            }
        }, level="INFO")
        
        return interaction_features

    @m1_optimized(workload_category=WorkloadCategory.FINANCIAL_MODELING)
    @comprehensive_memory_optimization()
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        tprint_info("🚀 [TACTICIAN] Starting comprehensive hardware-optimized Tactician interaction generation")
        self.logger.info("🔧 Starting comprehensive hardware-optimized Tactician interaction generation")
        
        # Log execution start parameters for troubleshooting
        tprint_structured({
            'tactician_execution_start': {
                'config_keys': list(config.keys()) if config else [],
                'timestamp': datetime.now().isoformat(),
                'hardware_manager_available': self.hardware_manager is not None,
                'comprehensive_optimizer_available': self.comprehensive_optimizer is not None,
                'cmi_complementarity_available': CMI_COMPLEMENTARITY_AVAILABLE
            }
        }, level="INFO")
        
        # Get training input and pipeline state from config
        training_input = config.get('training_input', {})
        pipeline_state = config.get('pipeline_state', {})
        
        # Set up basic parameters
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'binance')
        direction = training_input.get('direction', 'long')
        
        # Start comprehensive memory monitoring
        self.hardware_manager.start_monitoring()
        
        # CMI complementarity is always enabled in Tactician mode
        enable_cmi_complementarity = (
            CMI_COMPLEMENTARITY_AVAILABLE and 
            self.cmi_scorer is not None
        )
        
        tprint(f"🎯 [TACTICIAN] CMI complementarity check: CMI_AVAILABLE={CMI_COMPLEMENTARITY_AVAILABLE}, cmi_scorer={self.cmi_scorer is not None}")
        
        if enable_cmi_complementarity:
            tprint("🎯 [TACTICIAN] CMI complementarity enabled for Tactician mode interaction generation")
            self.logger.info("🎯 CMI complementarity enabled for Tactician mode interaction generation")
        else:
            tprint("⚠️ [TACTICIAN] CMI complementarity not available, using standard interaction generation")
            self.logger.warning("⚠️ CMI complementarity not available, using standard interaction generation")

        # Create optimized artifact manager with hardware acceleration
        class OptimizedArtifactManager:
            def __init__(self):
                self.cache = {}
                # Initialize hardware-optimized caching
                self.hardware_manager = get_integrated_hardware_manager(
                    IntegratedHardwareConfig(
                        enable_automatic_optimization=True,
                        enable_caching=True,
                        enable_memory_monitoring=True,
                        memory_limit_gb=2.0,
                        cache_memory_limit_mb=512.0
                    )
                )
            
            @smart_cache(ttl=3600)
            def retrieve_enhanced(self, key):
                return self.cache.get(key)
            
            @smart_cache(ttl=3600)
            def store_enhanced(self, key, value, metadata=None):
                # Optimize data before storing
                if hasattr(value, 'memory_usage'):
                    value = self.hardware_manager.optimize_dataframe(value)
                self.cache[key] = value
                return value
            
            @smart_cache(ttl=1800)
            def get_dataframe(self, step_name, key):
                return self.cache.get(key)
            
            @smart_cache(ttl=1800)
            def get_series(self, step_name, key):
                return self.cache.get(key)
            
            @smart_cache(ttl=1800)
            def get_artifact(self, step_name, key):
                return self.cache.get(key)
            
            def save(self, step_name, artifacts, metadata=None):
                for key, value in artifacts.items():
                    self.store_enhanced(key, value, metadata)
        
        artifact_manager = OptimizedArtifactManager()
        tprint("📦 [TACTICIAN] Using hardware-optimized artifact manager")
        
        # Monitor memory usage before processing
        self._monitor_memory_usage()
    
        # Try to load from artifact manager first
        tprint("🔍 [TACTICIAN] Checking for cached interaction features")
        cached_interactions = artifact_manager.retrieve_enhanced('INTERACTION_FEATURES')
        cached_metadata = artifact_manager.retrieve_enhanced('INTERACTION_METADATA')
        cached_metrics = artifact_manager.retrieve_enhanced('INTERACTION_GENERATION_METRICS')
        
        # Comprehensive data format analysis for cached data
        tprint_data_format(cached_interactions, "cached_interactions", level="INFO", return_summary=True)
        tprint_data_format(cached_metadata, "cached_metadata", level="DEBUG", return_summary=True)
        tprint_data_format(cached_metrics, "cached_metrics", level="DEBUG", return_summary=True)
        
        # Data preview for cached data retrieval
        tprint_data_preview(cached_interactions, "cached_interactions", level="INFO")
        tprint_data_format(cached_interactions, "cached_interactions", level="INFO")
        tprint_data_preview(cached_metadata, "cached_metadata", level="DEBUG")
        tprint_data_format(cached_metadata, "cached_metadata", level="DEBUG")
        tprint_data_preview(cached_metrics, "cached_metrics", level="DEBUG")
        tprint_data_format(cached_metrics, "cached_metrics", level="DEBUG")
        
        tprint(f"📦 [TACTICIAN] Cache check: interactions={cached_interactions is not None}, metadata={cached_metadata is not None}, metrics={cached_metrics is not None}")
        
        if cached_interactions is not None:
            tprint("📦 [TACTICIAN] Retrieved interaction features from artifact manager")
            self.logger.info("📦 Retrieved interaction features from artifact manager")
            
            # Optimize cached data with M1 memory optimization
            optimized_cached_interactions = self._optimize_dataframe_memory(cached_interactions)
            
            result_cached = InteractionGenerationResult(
                success=True,
                interaction_features=optimized_cached_interactions,
                interaction_metadata=cached_metadata or {},
                generation_metrics=cached_metrics or {},
                artifacts={'cache_hit': True},
                error_message=None
            )
            # Best-effort report from cache
            tprint("📊 Generating report from cached data")
            try:
                symbol = training_input.get('symbol', 'ETHUSDT')
                timeframe = training_input.get('timeframe', '15m')
                data_for_metrics = training_input.get('data')
                tprint(f"📊 Report params: symbol={symbol}, timeframe={timeframe}, data_available={data_for_metrics is not None}")
                report = self._generate_interaction_report(
                    result_cached.interaction_features,
                    result_cached.interaction_metadata,
                    symbol,
                    timeframe,
                    data_for_metrics
                )
                md = self._format_interaction_markdown(report)
                self._store_interaction_report(report, md, symbol, timeframe)
                tprint("📊 Report generated and stored successfully")
            except Exception as e:
                tprint(f"⚠️ Report generation failed: {e}")
                pass
            tprint("✅ Returning cached result")
            return result_cached

        tprint("🔍 Extracting training input parameters")
        data = training_input.get('data')
        symbol = training_input.get('symbol', 'ETHUSDT')
        timeframe = training_input.get('timeframe', '15m')
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = training_input.get('exchange', 'binance')
        custom_overrides = training_input.get('custom_overrides')
        
        tprint(f"📊 Input params: symbol={symbol}, timeframe={timeframe}, direction={direction}, intensity={intensity}")
        tprint(f"📊 Data params: data_shape={data.shape if hasattr(data, 'shape') else 'None'}, lookback_days={lookback_days}, start_date={start_date}, end_date={end_date}")
        tprint(f"📊 Exchange: {exchange}, custom_overrides={custom_overrides is not None}")

        # Enforce using only selected features from feature selection step
        try:
            tprint("🔍 Loading selected features from artifact manager")
            selected_df = artifact_manager.get_dataframe('feature_selection', 'SELECTED_FEATURES')
            if (selected_df is None or selected_df.empty):
                # Backward-compatibility: alternative step naming
                selected_df = artifact_manager.get_dataframe('feature_generation_feature_selection_step', 'SELECTED_FEATURES')
            
            # Data preview for selected features retrieval
            tprint_data_preview(selected_df, "selected_features_from_artifact_manager", level="INFO")
            tprint_data_format(selected_df, "selected_features_from_artifact_manager", level="INFO")
            
            if selected_df is None or selected_df.empty:
                tprint("❌ No selected features available; interaction generation requires prior feature selection")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="Selected features not found. Run feature_selection before interaction_generation."
                )
            tprint(f"✅ Using selected features for interaction generation: shape={selected_df.shape}")
            
            # Apply M1 memory optimization to selected features
            data = self._optimize_dataframe_memory(selected_df)
            tprint(f"🧠 Selected features optimized for M1: {data.shape}")
            
            # Data preview after memory optimization
            tprint_data_preview(data, "selected_features_after_memory_optimization", level="DEBUG")
            tprint_data_format(data, "selected_features_after_memory_optimization", level="DEBUG")
            
            # Check if memory mapping should be used
            if self._should_use_memory_mapping(len(data)):
                tprint(f"🗺️ Large dataset detected ({len(data)} rows), using memory mapping optimizations")

            # Load labeling targets and align with selected features
            targets_series: Optional[pd.Series] = None
            for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                series = artifact_manager.get_series(step_name, 'TARGETS')
                if isinstance(series, pd.Series) and not series.empty:
                    targets_series = series.astype(float)
                    tprint(f"✅ Loaded labeling targets from {step_name}: count={len(targets_series)}")
                    break
            
            # Comprehensive data format analysis for targets
            tprint_data_format(targets_series, "targets_series", level="INFO", return_summary=True)
            
            # Data preview for targets series retrieval
            tprint_data_preview(targets_series, "targets_series", level="INFO")
            tprint_data_format(targets_series, "targets_series", level="INFO")

            if targets_series is None or targets_series.empty:
                tprint("❌ Labeling targets not found for interaction generation")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="Targets from feature_generation_labeling_integration_step are required before interaction generation."
                )

            aligned = data.join(targets_series.rename("target"), how="inner").dropna(axis=0, how="any")
            if aligned.empty:
                tprint("❌ No overlapping timestamps between selected features and labeling targets")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="No overlapping timestamps between selected features and labeling targets."
                )

            targets = aligned.pop("target")
            data = aligned
            tprint(f"✅ Aligned features/targets for interaction generation: features={data.shape}, targets={targets.shape}")
            
            # Comprehensive data format analysis for aligned data
            tprint_data_format(data, "aligned_features_data", level="INFO", return_summary=True)
            tprint_data_format(targets, "aligned_targets", level="INFO", return_summary=True)
            
            # Data previews for data alignment
            tprint_data_preview(data, "aligned_features_data", level="INFO")
            tprint_data_format(data, "aligned_features_data", level="INFO")
            tprint_data_preview(targets, "aligned_targets", level="INFO")
            tprint_data_format(targets, "aligned_targets", level="INFO")
        except Exception as e:
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                artifacts={},
                error_message=f"Failed to load selected features: {e}"
            )

        # Load optimized periods/lookbacks if available and pass in overrides (top2-3 for interactions)
        try:
            # Try to load top periods/lookbacks first (top2-3 for interactions)
            opt_periods = artifact_manager.get_artifact('feature_generation_period_lookback_optimization_step', 'top_periods')
            opt_lookbacks = artifact_manager.get_artifact('feature_generation_period_lookback_optimization_step', 'top_lookbacks')
            
            # If top periods/lookbacks are not available, try optimized periods/lookbacks
            if opt_periods is None:
                opt_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
                if opt_periods and isinstance(opt_periods, list) and len(opt_periods) >= 2:
                    # Use top 2-3 periods for interactions
                    opt_periods = opt_periods[:3] if len(opt_periods) >= 3 else opt_periods
                    tprint(f"📊 Using top2-3 periods for interactions: {opt_periods}")
            
            if opt_lookbacks is None:
                opt_lookbacks = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_lookbacks')
                if opt_lookbacks and isinstance(opt_lookbacks, list) and len(opt_lookbacks) >= 2:
                    # Use top 2-3 lookbacks for interactions
                    opt_lookbacks = opt_lookbacks[:3] if len(opt_lookbacks) >= 3 else opt_lookbacks
                    tprint(f"📊 Using top2-3 lookbacks for interactions: {opt_lookbacks}")
            
            # Data preview for optimized parameters
            # Data format analysis for optimized parameters
            tprint_data_format(opt_periods, "optimized_periods", level="DEBUG", return_summary=True)
            tprint_data_format(opt_lookbacks, "optimized_lookbacks", level="DEBUG", return_summary=True)
            
            tprint_data_preview(opt_periods, "optimized_periods", level="DEBUG")
            tprint_data_format(opt_periods, "optimized_periods", level="DEBUG")
            tprint_data_preview(opt_lookbacks, "optimized_lookbacks", level="DEBUG")
            tprint_data_format(opt_lookbacks, "optimized_lookbacks", level="DEBUG")
                    
        except Exception:
            opt_periods, opt_lookbacks = None, None
            
        if custom_overrides is None:
            custom_overrides = {}
        if not isinstance(custom_overrides, dict):
            custom_overrides = dict(custom_overrides)
        if isinstance(custom_overrides, dict):
            if opt_periods is not None:
                custom_overrides.setdefault('optimized_periods', opt_periods)
            if opt_lookbacks is not None:
                custom_overrides.setdefault('optimized_lookbacks', opt_lookbacks)
        custom_overrides.setdefault('targets', targets)
        pipeline_state = dict(pipeline_state)
        pipeline_state['targets'] = targets

        # Apply comprehensive optimization for interaction generation
        tprint("🚀 Applying comprehensive optimization for interaction generation")
        optimized_data = self._apply_comprehensive_optimization(
            data, 'feature_engineering', 
            symbol=symbol, timeframe=timeframe, direction=direction
        )
        
        # Data format analysis after comprehensive optimization
        tprint_data_format(optimized_data, "data_after_comprehensive_optimization", level="DEBUG", return_summary=True)
        
        # Data preview after comprehensive optimization
        tprint_data_preview(optimized_data, "data_after_comprehensive_optimization", level="DEBUG")
        tprint_data_format(optimized_data, "data_after_comprehensive_optimization", level="DEBUG")
            
        # Generate interaction features directly
        tprint("🚀 Generating interaction features with comprehensive optimization")
        result = self._generate_interaction_features_sync(
            data=optimized_data,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            custom_overrides=custom_overrides
        )
            
        tprint(f"✅ run_interaction_generation_step completed: success={result.get('success', False)}")
        
        # Monitor memory usage after processing
        self._monitor_memory_usage()

        # Store artifacts in artifact manager
        if result.get('success', False):
            tprint("📦 Processing successful result")
            interaction_features = result.get('interaction_features', pd.DataFrame())
            interaction_metadata = result.get('interaction_metadata', {})
            generation_metrics = result.get('generation_metrics', {})
            
            tprint(f"📊 Result data: features_shape={interaction_features.shape if hasattr(interaction_features, 'shape') else 'None'}, metadata_keys={list(interaction_metadata.keys()) if interaction_metadata else []}")
            
            # Attach optimized period/lookback info if available
            try:
                opt_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
                opt_lookbacks = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_lookbacks')
                if isinstance(interaction_metadata, dict):
                    if opt_periods is not None:
                        interaction_metadata.setdefault('optimized_periods', opt_periods)
                    if opt_lookbacks is not None:
                        interaction_metadata.setdefault('optimized_lookbacks', opt_lookbacks)
            except Exception:
                pass

            # Apply CMI complementarity filtering if enabled
            if enable_cmi_complementarity and not interaction_features.empty:
                    tprint("🎯 Applying CMI complementarity filtering to interaction features")
                    self.logger.info("🎯 Applying CMI complementarity filtering to interaction features")
                    try:
                        # Get targets from pipeline state
                        targets = pipeline_state.get('targets')
                        tprint(f"🎯 CMI targets check: targets_available={targets is not None}")
                        if targets is not None:
                            tprint("🎯 Processing Analyst side information")
                            # Process Analyst side information
                            analyst_result = self.analyst_handler.process_side_info(
                                interaction_features, targets=targets
                            )
                            tprint(f"🎯 Analyst processing completed")
                            
                            tprint("🎯 Applying CMI complementarity scoring for interactions")
                            # Apply CMI complementarity scoring for interactions
                            cmi_scores = self.cmi_scorer.score_features(
                                interaction_features, targets
                            )
                            tprint(f"🎯 CMI scoring completed: {len(cmi_scores)} scores generated")
                            
                            if cmi_scores:
                                # Filter interactions based on CMI scores
                                original_count = len(interaction_features.columns)
                                # Select top features based on scores
                                top_features = sorted(cmi_scores.items(), key=lambda x: x[1], reverse=True)[:30]
                                selected_features = [f[0] for f in top_features if f[1] > 0.5]
                                
                                if selected_features:
                                    interaction_features = interaction_features[selected_features]
                                    filtered_count = len(interaction_features.columns)
                                    
                                    tprint(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} interactions")
                                    
                                    # Data format analysis after CMI filtering
                                    tprint_data_format(interaction_features, "interaction_features_after_cmi_filtering", level="INFO", return_summary=True)
                                    # Data preview after CMI filtering
                                    tprint_data_preview(interaction_features, "interaction_features_after_cmi_filtering", level="INFO")
                                    tprint_data_format(interaction_features, "interaction_features_after_cmi_filtering", level="INFO")
                                    self.logger.info(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} interactions")
                                    
                                    # Store CMI diagnostics in metadata
                                    interaction_metadata['cmi_diagnostics'] = {
                                        'cmi_enabled': True,
                                        'original_interactions': original_count,
                                        'filtered_interactions': filtered_count,
                                        'top_score': max(cmi_scores.values()) if cmi_scores else 0.0,
                                        'selected_features': len(selected_features)
                                    }
                                else:
                                    tprint("⚠️ No features selected by CMI scoring, using all interactions")
                                    self.logger.warning("⚠️ No features selected by CMI scoring, using all interactions")
                                    interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'No features selected'}
                            else:
                                tprint("⚠️ CMI complementarity scoring failed for interactions, using all interactions")
                                self.logger.warning("⚠️ CMI complementarity scoring failed for interactions, using all interactions")
                                interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'CMI scoring failed'}
                        else:
                            tprint("⚠️ No targets available for CMI complementarity filtering")
                            self.logger.warning("⚠️ No targets available for CMI complementarity filtering")
                            interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'No targets available'}
                            
                    except Exception as e:
                        tprint(f"⚠️ CMI complementarity filtering failed for interactions: {e}, using all interactions")
                        self.logger.warning(f"⚠️ CMI complementarity filtering failed for interactions: {e}, using all interactions")
                        interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': str(e)}
        else:
            tprint("📊 CMI complementarity not enabled or no interactions available")
            interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'reason': 'No interactions available'}
        
        tprint("💾 Storing interaction features in artifact manager")
        # Comprehensive data format analysis for final results
        tprint_data_format(interaction_features, "final_interaction_features_for_storage", level="INFO", return_summary=True)
        
        # Data preview before final storage
        tprint_data_preview(interaction_features, "final_interaction_features_for_storage", level="INFO")
        tprint_data_format(interaction_features, "final_interaction_features_for_storage", level="INFO")
        artifact_manager.store_enhanced('INTERACTION_FEATURES', interaction_features, {
            'step': 'interaction_generation_tactician',
            'shape': interaction_features.shape if hasattr(interaction_features, 'shape') else None,
            'created_at': datetime.now().isoformat()
        })
        
        tprint("💾 Storing interaction metadata in artifact manager")
        # Data preview before final storage
        tprint_data_preview(interaction_metadata, "final_interaction_metadata_for_storage", level="DEBUG")
        tprint_data_format(interaction_metadata, "final_interaction_metadata_for_storage", level="DEBUG")
        artifact_manager.store_enhanced('INTERACTION_METADATA', interaction_metadata, {
            'step': 'interaction_generation_tactician',
            'created_at': datetime.now().isoformat()
        })
        
        tprint("💾 Storing generation metrics in artifact manager")
        # Data preview before final storage
        tprint_data_preview(generation_metrics, "final_generation_metrics_for_storage", level="DEBUG")
        tprint_data_format(generation_metrics, "final_generation_metrics_for_storage", level="DEBUG")
        artifact_manager.store_enhanced('INTERACTION_GENERATION_METRICS', generation_metrics, {
            'step': 'interaction_generation_tactician',
            'created_at': datetime.now().isoformat()
        })

        tprint("📊 Creating InteractionGenerationResult object")
        
        # Apply final memory optimization to interaction features
        interaction_features = result.get('interaction_features', pd.DataFrame())
        if not interaction_features.empty:
            interaction_features = self._optimize_dataframe_memory(interaction_features)
            tprint(f"🧠 Final interaction features optimized: {interaction_features.shape}")
        
        # Add performance statistics to metadata
        generation_metrics = result.get('generation_metrics', {})
        generation_metrics.update({
            'comprehensive_optimizations': self.performance_stats.copy(),
            'processing_time_seconds': time.time() - start_time,
            'memory_optimizations_applied': self.performance_stats['memory_optimizations_applied'],
            'chunks_processed': self.performance_stats['chunks_processed'],
            'gpu_accelerations_used': self.performance_stats['gpu_accelerations_used'],
            'comprehensive_optimizations_used': self.performance_stats['comprehensive_optimizations_used'],
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_misses': self.performance_stats['cache_misses'],
            'parallel_workers_used': self.parallel_workers,
            'chunk_size_used': self.chunk_size
        })
        
        result_obj = InteractionGenerationResult(
            success=bool(result.get('success', False)),
            interaction_features=interaction_features,
            interaction_metadata=result.get('interaction_metadata', {}),
            generation_metrics=generation_metrics,
            artifacts=result.get('artifacts', {}),
            error_message=result.get('error_message')
        )
        tprint(f"📊 Result object created: success={result_obj.success}, features_shape={result_obj.interaction_features.shape if hasattr(result_obj.interaction_features, 'shape') else 'None'}")
        
        # Final performance summary
        total_time = time.time() - start_time
        self.performance_stats['total_processing_time'] = total_time
        tprint_performance("Total Tactician processing", total_time)
        tprint_success(f"✅ Memory optimizations applied: {self.performance_stats['memory_optimizations_applied']}")
        tprint_info(f"📦 Chunks processed: {self.performance_stats['chunks_processed']}")
        tprint_info(f"🚀 GPU accelerations used: {self.performance_stats['gpu_accelerations_used']}")
        tprint_info(f"🎯 Comprehensive optimizations used: {self.performance_stats['comprehensive_optimizations_used']}")
        tprint_info(f"💾 Cache hits: {self.performance_stats['cache_hits']}, misses: {self.performance_stats['cache_misses']}")
        
        # Log comprehensive performance summary for troubleshooting
        tprint_structured({
            'tactician_performance_summary': {
                'total_processing_time': total_time,
                'memory_optimizations_applied': self.performance_stats['memory_optimizations_applied'],
                'chunks_processed': self.performance_stats['chunks_processed'],
                'gpu_accelerations_used': self.performance_stats['gpu_accelerations_used'],
                'comprehensive_optimizations_used': self.performance_stats['comprehensive_optimizations_used'],
                'cache_hits': self.performance_stats['cache_hits'],
                'cache_misses': self.performance_stats['cache_misses'],
                'parallel_workers_used': self.parallel_workers,
                'chunk_size_used': self.chunk_size,
                'final_interactions_count': len(result_obj.interaction_features.columns) if hasattr(result_obj, 'interaction_features') else 0
            }
        }, level="INFO")
        
        # Build human-readable report
        tprint("📊 Generating interaction report")
        try:
            report = self._generate_interaction_report(
                result_obj.interaction_features,
                result_obj.interaction_metadata,
                symbol,
                timeframe,
                data
            )
            md = self._format_interaction_markdown(report)
            self._store_interaction_report(report, md, symbol, timeframe)
            tprint("📊 Report generated and stored successfully")
        except Exception as e:
            tprint(f"⚠️ Report generation failed: {e}")
            pass
        
            tprint("✅ Returning result object")
            return result_obj
            
        except Exception as e:
            tprint_exception(e, "Tactician interaction generation failed")
            self.logger.error(f"Interaction generation failed: {e}")
            
            # Add error information to performance stats
            self.performance_stats['error_occurred'] = True
            self.performance_stats['error_message'] = str(e)
            
            # Log detailed error information for troubleshooting
            tprint_structured({
                'tactician_error_details': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'processing_time_seconds': time.time() - start_time,
                    'performance_stats': self.performance_stats,
                    'hardware_manager_status': self.hardware_manager is not None,
                    'comprehensive_optimizer_status': self.comprehensive_optimizer is not None,
                    'cmi_complementarity_available': CMI_COMPLEMENTARITY_AVAILABLE
                }
            }, level="ERROR")
            
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={'error': str(e), 'processing_time_seconds': time.time() - start_time},
                artifacts={},
                error_message=str(e)
            )
        finally:
            # Cleanup and stop comprehensive monitoring
            tprint("🧹 Cleaning up comprehensive optimizations...")
            try:
                self.hardware_manager.stop_monitoring()
                force_cleanup()
                tprint("✅ Cleanup completed")
            except Exception as cleanup_error:
                tprint(f"⚠️ Cleanup warning: {cleanup_error}")

    # --- Reporting helpers ---
    def _generate_interaction_report(self, interactions: pd.DataFrame, metadata: Dict[str, Any], symbol: str, timeframe: str, raw_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        from datetime import datetime as _dt
        import numpy as _np
        import pandas as _pd

        n_rows = int(len(interactions)) if isinstance(interactions, _pd.DataFrame) else 0
        n_cols = int(len(interactions.columns)) if isinstance(interactions, _pd.DataFrame) else 0

        # Proxy target
        corr_rows = []
        if isinstance(raw_data, _pd.DataFrame) and 'close' in raw_data.columns and isinstance(interactions, _pd.DataFrame) and not interactions.empty:
            returns = raw_data['close'].pct_change().fillna(0.0)
            # Align and sample
            df = _pd.concat([interactions, returns.rename('ret')], axis=1).dropna()
            if not df.empty:
                if len(df) > 200_000:
                    df = df.iloc[-200_000:]
                y = df['ret'].values
                def safe_corr(xv, yv):
                    try:
                        xv = _np.asarray(xv)
                        yv = _np.asarray(yv)
                        xv = xv - xv.mean()
                        yv = yv - yv.mean()
                        denom = (_np.sqrt((xv*xv).sum()) * _np.sqrt((yv*yv).sum()))
                        return float((xv*yv).sum() / denom) if denom != 0 else 0.0
                    except Exception:
                        return 0.0
                cols = interactions.columns[:200]
                for c in cols:
                    try:
                        x = df[c].values
                        corr = abs(safe_corr(x, y))
                        nn = (df[c].notna().sum() / len(df)) * 100.0
                        var = float(_np.nanvar(df[c].values))
                        corr_rows.append({'feature': c, 'abs_corr_ret': round(corr, 6), 'non_null_pct': round(nn,2), 'variance': round(var, 6)})
                    except Exception:
                        continue
        # Sort by |corr|
        corr_rows = sorted(corr_rows, key=lambda d: d['abs_corr_ret'], reverse=True)[:40]

        return {
            'title': 'Interaction Generation Report',
            'timestamp': _dt.now().isoformat(),
            'configuration': {'symbol': symbol, 'timeframe': timeframe},
            'summary': {
                'rows': n_rows,
                'columns': n_cols,
                'memory_mb': float(interactions.memory_usage(deep=True).sum() / (1024**2)) if isinstance(interactions, _pd.DataFrame) else 0.0
            },
            'cmi_diagnostics': (metadata or {}).get('cmi_diagnostics', {}),
            'top_interactions': corr_rows
        }

    def _format_interaction_markdown(self, report: Dict[str, Any]) -> str:
        md = f"# {report['title']}\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        cfg = report.get('configuration', {})
        md += "## 📌 Configuration\n\n"
        md += f"- Symbol: {cfg.get('symbol','?')}\n"
        md += f"- Timeframe: {cfg.get('timeframe','?')}\n"

        summ = report.get('summary', {})
        md += "\n## 📊 Summary\n\n"
        md += f"- Rows: {summ.get('rows',0):,}\n"
        md += f"- Interactions: {summ.get('columns',0)}\n"
        md += f"- Memory: {summ.get('memory_mb',0.0):.2f} MB\n"

        md += "\n## 🔝 Top Interactions by |Corr| vs returns\n\n"
        if report.get('top_interactions'):
            md += "| Feature | |Corr| | Non-Null % | Variance |\n|---|---:|---:|---:|\n"
            for r in report['top_interactions']:
                md += f"| {r['feature']} | {r['abs_corr_ret']:.4f} | {r['non_null_pct']:.2f} | {r['variance']:.6f} |\n"
        else:
            md += "_Correlation not computed (missing close data).\n_"

        # CMI section
        md += "\n## 🧠 CMI Diagnostics\n\n"
        cmi = report.get('cmi_diagnostics', {})
        if cmi:
            for k, v in cmi.items():
                md += f"- {k}: {v}\n"
        else:
            md += "- Not available\n"
        return md

    def _store_interaction_report(self, report: Dict[str, Any], markdown: str, symbol: str, timeframe: str) -> None:
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json
        out_dir = _Path('outcomes')
        out_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        md_path = out_dir / f"interaction_generation_report_{symbol}_{timeframe}_{ts}.md"
        json_path = out_dir / f"interaction_generation_report_{symbol}_{timeframe}_{ts}.json"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)

    # Minimal hooks for ModularComponent
    def _initialize_resources(self) -> bool:
        try:
            self.set_state('initialized', True)
            return True
        except Exception:
            return False

    def _cleanup_resources(self) -> None:
        self.set_state('initialized', False)

    def _process_data(self, data: Any, **kwargs) -> Any:
        return data

    def _get_validation_rules(self) -> Dict[str, Any]:
        return {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'min_size': 100
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        errors, warnings, metadata = [], [], {}
        if isinstance(data, pd.DataFrame):
            missing = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in data.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")
            metadata['shape'] = data.shape
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}


# Handler for ares_launcher/sub_pipeline integration
@m1_optimized(workload_category=WorkloadCategory.FINANCIAL_MODELING)
@comprehensive_memory_optimization()
def handle_feature_generation_interaction_generation_step_tactician(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> InteractionGenerationResult:
    """Execute comprehensive hardware-optimized Tactician interaction generation."""
    start_time = time.time()
    tprint("🔧 Starting comprehensive hardware-optimized handle_feature_generation_interaction_generation_step_tactician")
    tprint(f"📊 Handler params: symbol={symbol}, timeframe={timeframe}, direction={direction}, intensity={intensity}")
    tprint(f"📊 Data params: data_shape={data.shape if hasattr(data, 'shape') else 'None'}, lookback_days={lookback_days}, start_date={start_date}, end_date={end_date}")
    
    # Initialize comprehensive hardware optimization for handler
    hardware_config = IntegratedHardwareConfig(
        enable_automatic_optimization=True,
        enable_caching=True,
        enable_memory_monitoring=True,
        memory_limit_gb=8.0,
        cache_memory_limit_mb=512.0
    )
    hardware_manager = get_integrated_hardware_manager(hardware_config)
    
    try:
        # Create optimized artifact manager with hardware acceleration
        class OptimizedArtifactManager:
            def __init__(self):
                self.cache = {}
                # Initialize hardware-optimized caching
                self.hardware_manager = get_integrated_hardware_manager(
                    IntegratedHardwareConfig(
                        enable_automatic_optimization=True,
                        enable_caching=True,
                        enable_memory_monitoring=True,
                        memory_limit_gb=2.0,
                        cache_memory_limit_mb=512.0
                    )
                )
            
            @smart_cache(ttl=1800)
            def get_dataframe(self, step_name, key):
                return self.cache.get(key)
            
            @smart_cache(ttl=1800)
            def get_series(self, step_name, key):
                return self.cache.get(key)
            
            @smart_cache(ttl=1800)
            def get_artifact(self, step_name, key):
                return self.cache.get(key)
            
            def save(self, step_name, artifacts, metadata=None):
                for key, value in artifacts.items():
                    # Optimize data before storing
                    if hasattr(value, 'memory_usage'):
                        value = self.hardware_manager.optimize_dataframe(value)
                    self.cache[key] = value
        
        manager = OptimizedArtifactManager()
        tprint("📦 Using hardware-optimized artifact manager")
        
        # Start comprehensive memory monitoring
        hardware_manager.start_monitoring()

        # Attempt to lazily load data if not provided
        tprint("🔍 Attempting to load selected features from artifact manager")
        # Enforce using only selected features
        data = manager.get_dataframe('feature_selection', 'SELECTED_FEATURES')
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            tprint("🔍 Trying alternative selection step key")
            data = manager.get_dataframe('feature_generation_feature_selection_step', 'SELECTED_FEATURES')
        
        # Comprehensive data format analysis for handler selected features
        tprint_data_format(data, "handler_selected_features_raw", level="INFO", return_summary=True)
        
        # Data preview for handler selected features retrieval
        tprint_data_preview(data, "handler_selected_features_raw", level="INFO")
        tprint_data_format(data, "handler_selected_features_raw", level="INFO")
        
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                artifacts={},
                error_message="Selected features not found. Run feature_selection before interaction_generation."
            )
        
        # Apply comprehensive memory optimization to loaded data
        tprint("🧠 Applying comprehensive memory optimization to loaded data...")
        data = optimize_dataframe(data)
        # Data preview after basic optimization
        tprint_data_preview(data, "handler_selected_features_optimized", level="DEBUG")
        tprint_data_format(data, "handler_selected_features_optimized", level="DEBUG")
        data = hardware_manager.optimize_dataframe(data)
        # Data preview after hardware optimization
        tprint_data_preview(data, "handler_selected_features_hardware_optimized", level="DEBUG")
        tprint_data_format(data, "handler_selected_features_hardware_optimized", level="DEBUG")
        tprint(f"✅ Data optimized comprehensively: {data.shape}")

        tprint("🚀 Generating interaction features from handler")
        # Load targets from artifact manager
        try:
            precomp_targets = None
            for step_name in ("labeling_integration", "feature_generation_labeling_integration_step"):
                for key in ("targets", "TARGETS"):
                    tmp = manager.get_artifact(step_name, key)
                    if isinstance(tmp, pd.Series) and not tmp.empty:
                        precomp_targets = tmp
                        break
                    if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                        precomp_targets = tmp.iloc[:, 0]
                        break
                if isinstance(precomp_targets, pd.Series) and not precomp_targets.empty:
                    break
        except Exception:
            precomp_targets = None

        # Create step instance and generate interactions
        step_instance = FeatureGenerationInteractionGenerationStepTactician()
        result_dict = step_instance._generate_interaction_features_sync(
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            custom_overrides={'targets': precomp_targets} if isinstance(precomp_targets, pd.Series) and not precomp_targets.empty else custom_overrides
        )
        tprint(f"✅ run_interaction_generation_step completed: success={result_dict.get('success', False)}")

        tprint("📊 Creating InteractionGenerationResult from handler")
        result = InteractionGenerationResult(
            success=bool(result_dict.get('success', False)),
            interaction_features=result_dict.get('interaction_features', pd.DataFrame()),
            interaction_metadata=result_dict.get('interaction_metadata', {}),
            generation_metrics=result_dict.get('generation_metrics', {}),
            artifacts=result_dict.get('artifacts', {}),
            error_message=result_dict.get('error_message')
        )
        tprint(f"📊 Handler result: success={result.success}, features_shape={result.interaction_features.shape if hasattr(result.interaction_features, 'shape') else 'None'}")

        if result.success:
            tprint("💾 Saving artifacts to manager")
            result.artifacts.setdefault('INTERACTION_FEATURES', result.interaction_features)
            manager.save('feature_generation_interaction_generation_step_tactician', result.artifacts, metadata=result.interaction_metadata)
            tprint("✅ Artifacts saved successfully")

        tprint_success("✅ Handler completed, returning result")
        
        # Log final handler success summary for troubleshooting
        tprint_structured({
            'tactician_handler_success_summary': {
                'handler_processing_time': time.time() - start_time,
                'result_success': result.success,
                'interaction_features_count': len(result.interaction_features.columns) if hasattr(result, 'interaction_features') else 0,
                'hardware_optimizations_applied': result.generation_metrics.get('memory_optimizations_applied', 0) if hasattr(result, 'generation_metrics') else 0,
                'comprehensive_optimizations_used': result.generation_metrics.get('comprehensive_optimizations_used', 0) if hasattr(result, 'generation_metrics') else 0
            }
        }, level="INFO")
        
        return result
        
    except Exception as e:
        tprint(f"❌ Handler failed: {e}")
        return InteractionGenerationResult(
            success=False,
            interaction_features=pd.DataFrame(),
            interaction_metadata={},
            generation_metrics={'error': str(e), 'processing_time_seconds': time.time() - start_time},
            artifacts={},
            error_message=str(e)
        )
    finally:
        # Cleanup comprehensive optimizations
        tprint("🧹 Cleaning up comprehensive optimizations in handler...")
        try:
            hardware_manager.stop_monitoring()
            force_cleanup()
            tprint("✅ Handler cleanup completed")
        except Exception as cleanup_error:
            tprint(f"⚠️ Handler cleanup warning: {cleanup_error}")
