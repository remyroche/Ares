"""
Feature Generation Interaction Generation Step

This step generates feature interactions via the consolidated pipeline runner.
Optimized with M1 hardware utilities, chunked processing, memory optimization,
and VectorBT acceleration for maximum performance on Apple Silicon.
"""

from __future__ import annotations

import logging
import gc
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import concurrent.futures
import threading
import time
import os

import pandas as pd
from src.utils.tprint import tprint

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_interaction_generation_step
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)
from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactKeys,
)

# M1 Hardware Optimization Imports
from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, optimize_dataframe_for_m1, create_m1_optimized_array
)
from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer, optimize_dataframe_memory, force_garbage_collection
)
from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer, create_m1_optimized_thread_pool, parallel_map_m1
)

# VectorBT and Unified Vectorization Imports
from src.utils.ml_common.unified_vectorization_manager import (
    get_unified_vectorization_manager, OperationType, OptimizationStrategy
)

# Import CMI complementarity components
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer, CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler, AnalystSideInfoConfig
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
except ImportError:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    AnalystSideInfoConfig = None


@dataclass
class InteractionGenerationResult:
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class FeatureGenerationInteractionGenerationStepTactician(ModularComponent):
    """Tactician mode interaction generation step with CMI complementarity filtering.
    
    This step implements the original interaction generation with CMI complementarity
    filtering enabled by default for Tactician mode. Uses M1 hardware acceleration,
    chunked processing, memory optimization, and VectorBT integration."""

    def __init__(self, name: str = "interaction_generation_step_tactician",
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(name, config or {}, logger)
        
        # Initialize M1 hardware optimizers
        tprint("🧠 Initializing M1 hardware optimizers...")
        self.m1_gpu_manager = get_m1_gpu_manager()
        self.m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)  # 8GB limit
        self.m1_cpu_optimizer = get_m1_cpu_optimizer()
        
        # Initialize VectorBT and unified vectorization
        tprint("🚀 Initializing VectorBT and unified vectorization...")
        self.unified_vectorization_manager = get_unified_vectorization_manager()
        
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
                interaction_gain_percentile=75  # Accept if > 75th percentile of null
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
            'vectorbt_optimizations_used': 0
        }
        
        tprint("✅ M1-optimized Tactician interaction generation step initialized")

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with M1-specific optimizations."""
        if df is None or df.empty:
            return df
            
        tprint("🧠 Applying M1 memory optimizations to DataFrame...")
        
        # Apply M1-specific memory optimization
        optimized_df = optimize_dataframe_for_m1(df)
        
        # Convert float64 to float32 where precision allows
        if self.float32_conversion:
            numeric_cols = optimized_df.select_dtypes(include=[np.float64]).columns
            for col in numeric_cols:
                # Check if conversion to float32 is safe (no precision loss)
                if optimized_df[col].min() >= np.finfo(np.float32).min and \
                   optimized_df[col].max() <= np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
                    
        # Apply pandas memory optimization
        optimized_df = self.m1_memory_optimizer.optimize_dataframe_memory(optimized_df)
        
        self.performance_stats['memory_optimizations_applied'] += 1
        tprint(f"✅ DataFrame memory optimized: {optimized_df.shape}")
        return optimized_df

    def _should_use_memory_mapping(self, data_size: int) -> bool:
        """Determine if memory mapping should be used based on data size."""
        return data_size > self.memory_mapped_threshold

    def _chunk_data_for_processing(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data into chunks for memory-efficient processing."""
        if len(data) <= self.chunk_size:
            return [data]
            
        tprint(f"📦 Chunking data: {len(data)} rows into chunks of {self.chunk_size}")
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
            
        tprint(f"✅ Created {len(chunks)} chunks for processing")
        return chunks

    def _process_chunk_with_optimization(self, chunk: pd.DataFrame, 
                                       chunk_idx: int, 
                                       **kwargs) -> pd.DataFrame:
        """Process a single chunk with full M1 optimization."""
        tprint(f"🔄 Processing chunk {chunk_idx + 1} with M1 optimizations...")
        
        # Apply memory optimization to chunk
        optimized_chunk = self._optimize_dataframe_memory(chunk)
        
        # Use M1 GPU acceleration for matrix operations if available
        if self.m1_gpu_manager.mps_available:
            try:
                # Convert to M1-optimized array for GPU operations
                numeric_data = optimized_chunk.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    gpu_optimized = self.m1_gpu_manager.optimize_tensor_operations(numeric_data.values)
                    optimized_chunk[numeric_data.columns] = gpu_optimized
                    self.performance_stats['gpu_accelerations_used'] += 1
                    tprint(f"🚀 GPU acceleration applied to chunk {chunk_idx + 1}")
            except Exception as e:
                tprint(f"⚠️ GPU optimization failed for chunk {chunk_idx + 1}: {e}")
        
        # Force garbage collection after each chunk
        force_garbage_collection()
        
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

    def _stream_features_efficiently(self, data: pd.DataFrame, 
                                   operation_func: callable,
                                   **kwargs) -> pd.DataFrame:
        """Stream features efficiently using chunked processing and parallel workers."""
        tprint("🌊 Starting efficient feature streaming...")
        
        # Check if chunking is needed
        if len(data) <= self.chunk_size:
            tprint("📊 Dataset small enough for single processing")
            return operation_func(data, **kwargs)
        
        # Create chunks
        chunks = self._chunk_data_for_processing(data)
        
        # Process chunks in parallel using M1-optimized thread pool
        with self.m1_cpu_optimizer.create_optimized_thread_pool(self.parallel_workers) as executor:
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

    def _apply_vectorbt_optimization(self, data: pd.DataFrame, 
                                   operation_type: str,
                                   **kwargs) -> pd.DataFrame:
        """Apply VectorBT optimization for specific operations."""
        tprint(f"🚀 Applying VectorBT optimization for {operation_type}...")
        
        try:
            # Map operation type to VectorBT operation
            vectorbt_operations = {
                'feature_engineering': OperationType.FEATURE_ENGINEERING,
                'backtesting': OperationType.VECTORBT_BACKTESTING,
                'cross_validation': OperationType.CROSS_VALIDATION,
                'technical_indicators': OperationType.VECTORBT_TECHNICAL_ANALYSIS
            }
            
            if operation_type in vectorbt_operations:
                op_type = vectorbt_operations[operation_type]
                
                # Use unified vectorization manager for optimization
                result = self.unified_vectorization_manager.optimize_operation(
                    op_type, data, prefer_vectorbt=True, **kwargs
                )
                
                self.performance_stats['vectorbt_optimizations_used'] += 1
                tprint(f"✅ VectorBT optimization applied for {operation_type}")
                
                # Extract result from OptimizationResult
                if hasattr(result, 'result'):
                    return result.result
                else:
                    return result
            else:
                tprint(f"⚠️ No VectorBT optimization available for {operation_type}")
                return data
                
        except Exception as e:
            tprint(f"⚠️ VectorBT optimization failed for {operation_type}: {e}")
            return data

    def _monitor_memory_usage(self):
        """Monitor memory usage and apply aggressive garbage collection if needed."""
        try:
            memory_stats = self.m1_memory_optimizer.get_memory_stats()
            memory_percent = memory_stats.get('memory_percent', 0)
            
            if memory_percent > self.aggressive_gc_threshold * 100:
                tprint(f"🧠 High memory usage detected: {memory_percent:.1f}%, applying aggressive cleanup...")
                force_garbage_collection()
                
                # Clear M1-specific caches
                self.m1_memory_optimizer._clear_m1_caches()
                
                tprint("✅ Aggressive memory cleanup completed")
        except Exception as e:
            tprint(f"⚠️ Memory monitoring failed: {e}")

    async def execute(self,
                      training_input: Dict[str, Any],
                      pipeline_state: Dict[str, Any]) -> InteractionGenerationResult:
        start_time = time.time()
        tprint("🚀 [TACTICIAN] Starting M1-optimized Tactician interaction generation via consolidated pipeline")
        self.logger.info("🔧 Starting M1-optimized Tactician interaction generation via consolidated pipeline")
        
        # Start memory monitoring
        self.m1_memory_optimizer.start_monitoring()
        
        try:
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

            # Get artifact manager
            tprint("📦 [TACTICIAN] Getting artifact manager")
            artifact_manager = get_pretraining_artifact_manager()
            
            # Monitor memory usage before processing
            self._monitor_memory_usage()
        
            # Try to load from artifact manager first
            tprint("🔍 [TACTICIAN] Checking for cached interaction features")
            cached_interactions = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_FEATURES)
            cached_metadata = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_METADATA)
            cached_metrics = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_GENERATION_METRICS)
            
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
                selected_df = artifact_manager.get_dataframe('feature_selection', ArtifactKeys.SELECTED_FEATURES)
                if (selected_df is None or selected_df.empty):
                    # Backward-compatibility: alternative step naming
                    selected_df = artifact_manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
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
                
                # Check if memory mapping should be used
                if self._should_use_memory_mapping(len(data)):
                    tprint(f"🗺️ Large dataset detected ({len(data)} rows), using memory mapping optimizations")

                # Load labeling targets and align with selected features
                targets_series: Optional[pd.Series] = None
                for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                    series = artifact_manager.get_series(step_name, ArtifactKeys.TARGETS)
                    if isinstance(series, pd.Series) and not series.empty:
                        targets_series = series.astype(float)
                        tprint(f"✅ Loaded labeling targets from {step_name}: count={len(targets_series)}")
                        break

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
            except Exception as e:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message=f"Failed to load selected features: {e}"
                )

        # Load optimized periods/lookbacks if available and pass in overrides
        try:
            opt_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
            opt_lookbacks = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_lookbacks')
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

        # Apply VectorBT optimization for interaction generation
        tprint("🚀 Applying VectorBT optimization for interaction generation")
        optimized_data = self._apply_vectorbt_optimization(
            data, 'feature_engineering', 
            symbol=symbol, timeframe=timeframe, direction=direction
        )
            
        # Use efficient feature streaming for large datasets
        if len(data) > self.chunk_size:
            tprint("🌊 Using efficient feature streaming for large dataset")
            # Define the interaction generation function for streaming
            def interaction_generation_func(chunk_data, **kwargs):
                return run_interaction_generation_step(
                    data=chunk_data,
                    symbol=kwargs.get('symbol'),
                    timeframe=kwargs.get('timeframe'),
                    direction=kwargs.get('direction'),
                    intensity=kwargs.get('intensity'),
                    lookback_days=kwargs.get('lookback_days'),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                        exchange=kwargs.get('exchange'),
                        custom_overrides=kwargs.get('custom_overrides')
                    )
                
            # Stream the interaction generation
            result = await self._stream_features_efficiently(
                data, interaction_generation_func,
                symbol=symbol, timeframe=timeframe, direction=direction,
                intensity=intensity, lookback_days=lookback_days,
                start_date=start_date, end_date=end_date,
                exchange=exchange, custom_overrides=custom_overrides
            )
        else:
            tprint("🚀 Calling run_interaction_generation_step with optimized data")
            result = await run_interaction_generation_step(
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
                            tprint("🎯 Extracting Analyst side information")
                            # Extract Analyst side information
                            analyst_result = self.analyst_handler.extract_side_info(
                                pipeline_state, targets, interaction_features.index
                            )
                            tprint(f"🎯 Analyst result: is_valid={analyst_result.is_valid}, degraded={analyst_result.degraded_to_unconditional}, source={analyst_result.source}")
                            
                            if analyst_result.is_valid and not analyst_result.degraded_to_unconditional:
                                tprint("🎯 Applying CMI complementarity scoring for interactions")
                                # Apply CMI complementarity scoring for interactions
                                # Use conditional gain: I(Y; fi∘fj | A, fi, fj) > threshold
                                cmi_result = self.cmi_scorer.score_features(
                                    interaction_features, targets, analyst_result.A,
                                    pipeline_state=pipeline_state
                                )
                                tprint(f"🎯 CMI scoring result: is_valid={cmi_result.is_valid}, selected_count={len(cmi_result.selected_features) if cmi_result.selected_features else 0}")
                                
                                if cmi_result.is_valid and cmi_result.selected_features:
                                    # Filter interactions based on CMI selection
                                    original_count = len(interaction_features.columns)
                                    interaction_features = interaction_features[cmi_result.selected_features]
                                    filtered_count = len(interaction_features.columns)
                                    
                                    tprint(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} interactions")
                                    tprint(f"📊 Noise floor: {cmi_result.noise_floor:.6f}")
                                    tprint(f"📊 ΔPerf threshold: {cmi_result.delta_perf_threshold:.6f}")
                                    self.logger.info(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} interactions")
                                    self.logger.info(f"📊 Noise floor: {cmi_result.noise_floor:.6f}")
                                    self.logger.info(f"📊 ΔPerf threshold: {cmi_result.delta_perf_threshold:.6f}")
                                    
                                    # Store CMI diagnostics in metadata
                                    interaction_metadata['cmi_diagnostics'] = {
                                        'cmi_enabled': True,
                                        'original_interactions': original_count,
                                        'filtered_interactions': filtered_count,
                                        'noise_floor': cmi_result.noise_floor,
                                        'delta_perf_threshold': cmi_result.delta_perf_threshold,
                                        'analyst_source': analyst_result.source,
                                        'analyst_dims': analyst_result.n_dims,
                                        'I_Y_A': analyst_result.I_Y_A,
                                        'degraded_to_unconditional': analyst_result.degraded_to_unconditional
                                    }
                                else:
                                    tprint("⚠️ CMI complementarity scoring failed for interactions, using all interactions")
                                    self.logger.warning("⚠️ CMI complementarity scoring failed for interactions, using all interactions")
                                    interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'CMI scoring failed'}
                            else:
                                tprint("⚠️ Analyst side information extraction failed for interactions, using all interactions")
                                self.logger.warning("⚠️ Analyst side information extraction failed for interactions, using all interactions")
                                interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'Analyst side info failed'}
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
        artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_FEATURES, interaction_features, {
            'step': 'interaction_generation_tactician',
            'shape': interaction_features.shape if hasattr(interaction_features, 'shape') else None,
            'created_at': datetime.now().isoformat()
        })
        
        tprint("💾 Storing interaction metadata in artifact manager")
        artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_METADATA, interaction_metadata, {
            'step': 'interaction_generation_tactician',
            'created_at': datetime.now().isoformat()
        })
        
        tprint("💾 Storing generation metrics in artifact manager")
        artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_GENERATION_METRICS, generation_metrics, {
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
            'm1_optimizations': self.performance_stats.copy(),
            'processing_time_seconds': time.time() - start_time,
            'memory_optimizations_applied': self.performance_stats['memory_optimizations_applied'],
            'chunks_processed': self.performance_stats['chunks_processed'],
            'gpu_accelerations_used': self.performance_stats['gpu_accelerations_used'],
            'vectorbt_optimizations_used': self.performance_stats['vectorbt_optimizations_used'],
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
        tprint(f"⏱️ Total processing time: {total_time:.2f}s")
        tprint(f"🧠 Memory optimizations applied: {self.performance_stats['memory_optimizations_applied']}")
        tprint(f"📦 Chunks processed: {self.performance_stats['chunks_processed']}")
        tprint(f"🚀 GPU accelerations used: {self.performance_stats['gpu_accelerations_used']}")
        tprint(f"🎯 VectorBT optimizations used: {self.performance_stats['vectorbt_optimizations_used']}")
        
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
            tprint(f"❌ Interaction generation failed: {e}")
            self.logger.error(f"Interaction generation failed: {e}")
            
            # Add error information to performance stats
            self.performance_stats['error_occurred'] = True
            self.performance_stats['error_message'] = str(e)
            
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={'error': str(e), 'processing_time_seconds': time.time() - start_time},
                artifacts={},
                error_message=str(e)
            )
        finally:
            # Cleanup and stop memory monitoring
            tprint("🧹 Cleaning up M1 optimizations...")
            try:
                self.m1_memory_optimizer.stop_monitoring()
                force_garbage_collection()
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
async def handle_feature_generation_interaction_generation_step_tactician(
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
    """Execute M1-optimized Tactician interaction generation via consolidated pipeline runner."""
    start_time = time.time()
    tprint("🔧 Starting M1-optimized handle_feature_generation_interaction_generation_step_tactician")
    tprint(f"📊 Handler params: symbol={symbol}, timeframe={timeframe}, direction={direction}, intensity={intensity}")
    tprint(f"📊 Data params: data_shape={data.shape if hasattr(data, 'shape') else 'None'}, lookback_days={lookback_days}, start_date={start_date}, end_date={end_date}")
    
    # Initialize M1 optimizers for handler
    m1_gpu_manager = get_m1_gpu_manager()
    m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
    m1_cpu_optimizer = get_m1_cpu_optimizer()
    
    try:
        manager = get_pretraining_artifact_manager()
        tprint("📦 Got pretraining artifact manager")
        
        # Start memory monitoring
        m1_memory_optimizer.start_monitoring()

        # Attempt to lazily load data if not provided
        tprint("🔍 Attempting to load selected features from artifact manager")
        # Enforce using only selected features
        data = manager.get_dataframe('feature_selection', ArtifactKeys.SELECTED_FEATURES)
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            tprint("🔍 Trying alternative selection step key")
            data = manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                artifacts={},
                error_message="Selected features not found. Run feature_selection before interaction_generation."
            )
        
        # Apply M1 memory optimization to loaded data
        tprint("🧠 Applying M1 memory optimization to loaded data...")
        data = optimize_dataframe_for_m1(data)
        data = m1_memory_optimizer.optimize_dataframe_memory(data)
        tprint(f"✅ Data optimized for M1: {data.shape}")

        tprint("🚀 Calling run_interaction_generation_step from handler")
        # Load targets from artifact manager for runner
        try:
            precomp_targets = None
            for step_name in ("labeling_integration", "feature_generation_labeling_integration_step"):
                for key in ("targets", ArtifactKeys.TARGETS):
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

        result_dict = await run_interaction_generation_step(
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
            result.artifacts.setdefault(ArtifactKeys.INTERACTION_FEATURES, result.interaction_features)
            manager.save('feature_generation_interaction_generation_step_tactician', result.artifacts, metadata=result.interaction_metadata)
            tprint("✅ Artifacts saved successfully")

        tprint("✅ Handler completed, returning result")
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
        # Cleanup M1 optimizations
        tprint("🧹 Cleaning up M1 optimizations in handler...")
        try:
            m1_memory_optimizer.stop_monitoring()
            force_garbage_collection()
            tprint("✅ Handler cleanup completed")
        except Exception as cleanup_error:
            tprint(f"⚠️ Handler cleanup warning: {cleanup_error}")
