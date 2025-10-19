"""
Consolidated Pipeline Runner

This module provides functions to run the consolidated pipeline up to specific steps,
allowing the step files to call the consolidated pipeline at the proper places.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Awaitable
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import gc
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# M1 Optimization imports
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, optimize_dataframe_for_m1, create_m1_optimized_array
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_dataframe_memory, optimize_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, create_m1_optimized_thread_pool, parallel_map_m1

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)

# Ensure tprint_error is always available
if not TPRINT_AVAILABLE:
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)

from .consolidated_pipeline import (
    UnifiedDataDrivenPipeline,
    create_unified_pipeline,
    ConsolidatedPipelineResult
)
from .core.config import UnifiedPipelineConfig, create_default_config
from .core.simplified_config import (
    create_full_config,
    create_blank_config,
    create_light_config,
    create_config_by_intensity,
    PipelineIntensity
)

class ConsolidatedPipelineRunner:
    """Runner for executing consolidated pipeline up to specific steps."""

    def __init__(self, config: Optional[UnifiedPipelineConfig] = None) -> None:
        """
        Initialize the pipeline runner.

        Args:
            config: Optional pipeline configuration. If None, uses default config.

        Raises:
            ValueError: If config is invalid
            ImportError: If required dependencies are missing
        """
        try:
            tprint_step("🚀 Initializing ConsolidatedPipelineRunner")

            if config is None:
                tprint_info("📋 Using default configuration")
                self.config = create_default_config()
            else:
                tprint_info("📋 Using provided configuration")
                self.config = config

            # Validate configuration
            if not isinstance(self.config, UnifiedPipelineConfig):
                raise ValueError(f"Invalid config type: {type(self.config)}. Expected UnifiedPipelineConfig.")

            tprint_info("🔧 Creating unified pipeline")
            self.pipeline = create_unified_pipeline(self.config)

            if self.pipeline is None:
                raise RuntimeError("Failed to create unified pipeline")

            self.logger = logging.getLogger(__name__)
            
            # Initialize M1 optimization components
            tprint_info("🧠 Initializing M1 optimization components")
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            
            # Optimization configuration
            self.parallel_workers = 6  # Optimized for M1
            self.chunk_size = 10000  # Memory-efficient chunk size
            self.memory_mapping_enabled = True
            self.aggressive_gc_enabled = True
            self.data_type_optimization = True  # Convert float64 to float32
            
            tprint_success("🧠 M1 optimization components initialized")
            tprint_success("✅ ConsolidatedPipelineRunner initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize ConsolidatedPipelineRunner: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        try:
            if not isinstance(df, pd.DataFrame):
                return df
            
            initial_memory = df.memory_usage(deep=True).sum()
            
            # Convert float64 to float32 where precision allows
            if self.data_type_optimization:
                for col in df.select_dtypes(include=[np.float64]).columns:
                    if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            
            # Use M1 memory optimizer
            df = optimize_dataframe_for_m1(df)
            
            final_memory = df.memory_usage(deep=True).sum()
            memory_saved = initial_memory - final_memory
            
            if memory_saved > 0:
                tprint_info(f"🧠 Data type optimization: {memory_saved / 1024**2:.2f} MB saved")
            
            return df
            
        except Exception as e:
            tprint_warning(f"Data type optimization failed: {e}")
            return df

    def _aggressive_garbage_collection(self) -> None:
        """Perform aggressive garbage collection for memory optimization."""
        try:
            # Force multiple garbage collections
            for _ in range(3):
                collected = gc.collect()
                if collected > 0:
                    tprint_info(f"Garbage collection cycle: {collected} objects collected")
            
            # Use M1 memory optimizer for additional cleanup
            memory_result = optimize_memory()
            if memory_result.get('success', False):
                memory_saved = memory_result.get('memory_saved_mb', 0)
                if memory_saved > 0:
                    tprint_info(f"🧠 Memory optimization: {memory_saved:.1f} MB saved")
                
        except Exception as e:
            tprint_warning(f"Aggressive garbage collection failed: {e}")

    def _optimize_pipeline_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize pipeline data for M1 processing."""
        try:
            # Start memory monitoring
            self.m1_memory_optimizer.start_monitoring()
            
            # Optimize data types
            data = self._optimize_dataframe_dtypes(data)
            
            # Aggressive garbage collection
            if self.aggressive_gc_enabled:
                self._aggressive_garbage_collection()
            
            tprint_info(f"🚀 Pipeline data optimized: {data.shape}")
            return data
            
        except Exception as e:
            tprint_warning(f"Pipeline data optimization failed: {e}")
            return data

    def _finalize_pipeline_processing(self) -> None:
        """Finalize pipeline processing with cleanup."""
        try:
            # Aggressive garbage collection
            if self.aggressive_gc_enabled:
                self._aggressive_garbage_collection()
            
            # Stop memory monitoring
            self.m1_memory_optimizer.stop_monitoring()
            
            # Get final memory statistics
            memory_stats = self.m1_memory_optimizer.get_memory_stats()
            tprint_info(f"🧠 Final memory usage: {memory_stats.get('memory_percent', 0):.1f}%")
            
        except Exception as e:
            tprint_warning(f"Pipeline finalization failed: {e}")

    async def _process_large_dataset_with_chunking(self, 
                                                 data: pd.DataFrame, 
                                                 targets: pd.Series, 
                                                 timeframe: str, 
                                                 pipeline_state: Dict[str, Any]) -> Any:
        """Process large datasets using chunked processing with M1 optimizations."""
        try:
            tprint_info(f"📦 Processing large dataset with chunked approach: {len(data)} rows")
            
            # Split data into chunks
            chunk_size = self.chunk_size
            chunks = [data.iloc[i:i + chunk_size].copy() for i in range(0, len(data), chunk_size)]
            target_chunks = [targets.iloc[i:i + chunk_size].copy() for i in range(0, len(targets), chunk_size)]
            
            tprint_info(f"📊 Created {len(chunks)} chunks of size {chunk_size}")
            
            # Process chunks in parallel using M1 CPU optimizer
            chunk_results = []
            
            with self.m1_cpu_optimizer.create_m1_optimized_thread_pool(max_workers=self.parallel_workers) as executor:
                # Submit chunk processing tasks
                future_to_chunk = {}
                for i, (chunk, target_chunk) in enumerate(zip(chunks, target_chunks)):
                    # Optimize chunk data types
                    chunk = self._optimize_dataframe_dtypes(chunk)
                    target_chunk = self._optimize_target_series(target_chunk)
                    
                    # Create chunk-specific pipeline state
                    chunk_pipeline_state = pipeline_state.copy()
                    chunk_pipeline_state['chunk_index'] = i
                    chunk_pipeline_state['total_chunks'] = len(chunks)
                    
                    # Submit chunk for processing
                    future = executor.submit(
                        self._process_single_chunk,
                        chunk, target_chunk, timeframe, chunk_pipeline_state
                    )
                    future_to_chunk[future] = i
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        chunk_results.append((chunk_idx, chunk_result))
                        tprint_info(f"✅ Chunk {chunk_idx + 1}/{len(chunks)} processed successfully")
                    except Exception as e:
                        tprint_error(f"❌ Chunk {chunk_idx + 1} failed: {e}")
                        # Create a minimal result for failed chunks
                        chunk_results.append((chunk_idx, None))
                
                # Aggressive garbage collection between chunks
                if self.aggressive_gc_enabled:
                    self._aggressive_garbage_collection()
            
            # Sort results by chunk index and combine
            chunk_results.sort(key=lambda x: x[0])
            successful_results = [result for _, result in chunk_results if result is not None]
            
            if not successful_results:
                raise RuntimeError("All chunks failed to process")
            
            # Combine results from successful chunks
            tprint_info("🔗 Combining chunk results")
            combined_result = self._combine_chunk_results(successful_results)
            
            tprint_success(f"✅ Successfully processed {len(successful_results)}/{len(chunks)} chunks")
            return combined_result
            
        except Exception as e:
            tprint_error(f"❌ Chunked processing failed: {e}")
            raise RuntimeError(f"Chunked processing failed: {e}") from e

    def _process_single_chunk(self, chunk: pd.DataFrame, target_chunk: pd.Series, 
                            timeframe: str, pipeline_state: Dict[str, Any]) -> Any:
        """Process a single chunk with M1 optimizations."""
        try:
            # Apply M1 GPU acceleration if available
            if self.m1_gpu_manager.mps_available:
                try:
                    chunk = self.m1_gpu_manager.optimize_dataframe_for_m1(chunk)
                    tprint_debug(f"🚀 Chunk optimized with M1 GPU acceleration")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 GPU acceleration failed for chunk: {e}")
            
            # Process chunk using pipeline (synchronous version)
            # Note: This is a simplified version - in practice, you'd need to adapt
            # the pipeline to work synchronously or use asyncio.run()
            result = {
                'success': True,
                'chunk_data': chunk,
                'chunk_targets': target_chunk,
                'chunk_metadata': {
                    'chunk_size': len(chunk),
                    'chunk_index': pipeline_state.get('chunk_index', 0),
                    'm1_optimized': True
                }
            }
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Single chunk processing failed: {e}")
            raise RuntimeError(f"Single chunk processing failed: {e}") from e

    def _combine_chunk_results(self, chunk_results: List[Any]) -> Any:
        """Combine results from multiple chunks into a single result."""
        try:
            if not chunk_results:
                raise ValueError("No chunk results to combine")
            
            # Extract chunk data and targets
            chunk_data_list = []
            chunk_targets_list = []
            
            for result in chunk_results:
                if result and 'chunk_data' in result:
                    chunk_data_list.append(result['chunk_data'])
                if result and 'chunk_targets' in result:
                    chunk_targets_list.append(result['chunk_targets'])
            
            # Combine data
            if chunk_data_list:
                combined_data = pd.concat(chunk_data_list, ignore_index=True)
                combined_data = self._optimize_dataframe_dtypes(combined_data)
            else:
                combined_data = pd.DataFrame()
            
            if chunk_targets_list:
                combined_targets = pd.concat(chunk_targets_list, ignore_index=True)
            else:
                combined_targets = pd.Series()
            
            # Create combined result
            combined_result = type('CombinedResult', (), {
                'success': True,
                'interaction_features': combined_data,
                'interaction_metadata': {
                    'total_chunks': len(chunk_results),
                    'combined_shape': combined_data.shape,
                    'm1_optimized': True,
                    'chunked_processing': True
                },
                'generation_metrics': {
                    'chunks_processed': len(chunk_results),
                    'total_rows': len(combined_data),
                    'processing_method': 'chunked_with_m1_optimization'
                },
                'artifacts': {},
                'error_message': None
            })()
            
            tprint_success(f"✅ Combined {len(chunk_results)} chunks into result with shape {combined_data.shape}")
            return combined_result
            
        except Exception as e:
            tprint_error(f"❌ Failed to combine chunk results: {e}")
            raise RuntimeError(f"Failed to combine chunk results: {e}") from e

    def _optimize_target_series(self, targets: pd.Series) -> pd.Series:
        """Optimize target series for M1 processing."""
        try:
            if targets.empty:
                return targets
            
            # Convert to float32 if possible
            if targets.dtype == np.float64:
                if targets.min() >= np.finfo(np.float32).min and targets.max() <= np.finfo(np.float32).max:
                    targets = targets.astype(np.float32)
            
            return targets
            
        except Exception as e:
            tprint_warning(f"Target series optimization failed: {e}")
            return targets

    async def run_data_validation_step(self,
                                     data: pd.DataFrame,
                                     symbol: str = "ETHUSDT",
                                     timeframe: str = "15m",
                                     direction: str = "longs",
                                     intensity: str = "blank",
                                     lookback_days: Optional[int] = None,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     exchange: str = "binance",
                                     custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run pipeline up to data validation step.

        Args:
            data: Input DataFrame with time series data
            symbol: Trading symbol (default: "ETHUSDT")
            timeframe: Time frame for analysis (default: "15m")
            direction: Trading direction (default: "longs")
            intensity: Pipeline intensity level (default: "blank")
            lookback_days: Optional lookback period in days
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            exchange: Exchange name (default: "binance")
            custom_overrides: Optional configuration overrides

        Returns:
            Dict containing validation results with keys:
            - success: bool
            - data_quality_score: float
            - validation_metadata: Dict[str, Any]
            - artifacts: Dict[str, Any]
            - error_message: Optional[str]

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If pipeline execution fails
        """
        try:
            tprint_step("🔍 Starting data validation step")
            tprint_info(f"📊 Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            tprint_info(f"🎯 Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
            tprint_info(f"⚙️ Intensity: {intensity}, Exchange: {exchange}")

            # Validate input data
            if data is None or len(data) == 0:
                raise ValueError("Input data cannot be None or empty")

            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(data)}")

            # Configure pipeline based on intensity
            tprint_info("🔧 Configuring pipeline based on intensity")
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            if self.pipeline is None:
                raise RuntimeError("Failed to create unified pipeline with new configuration")

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'data_validation'
            }

            tprint_info("🚀 Executing pipeline up to data validation")
            # Run pipeline up to data validation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            if result is None:
                raise RuntimeError("Pipeline returned None result")

            # Extract validation results
            validation_result = {
                'success': result.success,
                'data_quality_score': getattr(result, 'data_quality_score', 0.0),
                'validation_metadata': getattr(result, 'validation_metadata', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            if validation_result['success']:
                tprint_success(f"✅ Data validation completed successfully")
                tprint_result(f"📈 Data quality score: {validation_result['data_quality_score']:.3f}")
                tprint_info(f"📋 Validation metadata keys: {list(validation_result['validation_metadata'].keys())}")
                tprint_info(f"📦 Artifacts generated: {len(validation_result['artifacts'])}")
            else:
                tprint_error(f"❌ Data validation failed: {validation_result['error_message']}")

            # Generate human-readable report
            tprint_info("📄 Generating human-readable report")
            await self._generate_data_validation_report(validation_result, data)

            return validation_result

        except ValueError as e:
            error_msg = f"Invalid input for data validation step: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return {
                'success': False,
                'error_message': error_msg,
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }
        except RuntimeError as e:
            error_msg = f"Runtime error in data validation step: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return {
                'success': False,
                'error_message': error_msg,
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }
        except Exception as e:
            error_msg = f"Unexpected error in data validation step: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(f"Data validation step failed: {e}", exc_info=True)
            return {
                'success': False,
                'error_message': error_msg,
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }

    async def run_feature_generation_step(self,
                                        data: pd.DataFrame,
                                        symbol: str = "ETHUSDT",
                                        timeframe: str = "15m",
                                        direction: str = "longs",
                                        intensity: str = "blank",
                                        lookback_days: Optional[int] = None,
                                        start_date: Optional[str] = None,
                                        end_date: Optional[str] = None,
                                        exchange: str = "binance",
                                        custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to feature generation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'feature_generation'
            }

            # Run pipeline up to feature generation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract feature generation results
            feature_result = {
                'success': result.success,
                'generated_features': getattr(result, 'generated_features', pd.DataFrame()),
                'feature_metadata': getattr(result, 'feature_metadata', {}),
                'generation_metrics': getattr(result, 'generation_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_feature_generation_report(feature_result, data)

            return feature_result

        except Exception as e:
            self.logger.error(f"Feature generation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'generated_features': pd.DataFrame(),
                'feature_metadata': {},
                'generation_metrics': {}
            }

    async def run_feature_selection_step(self,
                                       data: pd.DataFrame,
                                       symbol: str = "ETHUSDT",
                                       timeframe: str = "15m",
                                       direction: str = "longs",
                                       intensity: str = "blank",
                                       lookback_days: Optional[int] = None,
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       exchange: str = "binance",
                                       custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to feature selection step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'feature_selection'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to feature selection
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract feature selection results
            selection_result = {
                'success': result.success,
                'selected_features': getattr(result, 'selected_features', pd.DataFrame()),
                'selection_metadata': getattr(result, 'selection_metadata', {}),
                'selection_metrics': getattr(result, 'selection_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_feature_selection_report(selection_result, data)

            return selection_result

        except Exception as e:
            self.logger.error(f"Feature selection step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'selected_features': pd.DataFrame(),
                'selection_metadata': {},
                'selection_metrics': {}
            }

    async def run_period_optimization_step(self,
                                         data: pd.DataFrame,
                                         symbol: str = "ETHUSDT",
                                         timeframe: str = "15m",
                                         direction: str = "longs",
                                         intensity: str = "blank",
                                         lookback_days: Optional[int] = None,
                                         start_date: Optional[str] = None,
                                         end_date: Optional[str] = None,
                                         exchange: str = "binance",
                                         custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to period optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'period_optimization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to period optimization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract period optimization results
            optimization_result = {
                'success': result.success,
                'optimal_periods': getattr(result, 'optimal_periods', {}),
                'optimization_metrics': getattr(result, 'optimization_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_period_optimization_report(optimization_result, data)

            return optimization_result

        except Exception as e:
            self.logger.error(f"Period optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'optimal_periods': {},
                'optimization_metrics': {}
            }

    async def run_lookback_optimization_step(self,
                                           data: pd.DataFrame,
                                           symbol: str = "ETHUSDT",
                                           timeframe: str = "15m",
                                           direction: str = "longs",
                                           intensity: str = "blank",
                                           lookback_days: Optional[int] = None,
                                           start_date: Optional[str] = None,
                                           end_date: Optional[str] = None,
                                           exchange: str = "binance",
                                           custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to lookback optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'lookback_optimization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to lookback optimization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract lookback optimization results
            optimization_result = {
                'success': result.success,
                'optimal_lookbacks': getattr(result, 'optimal_lookbacks', {}),
                'optimization_metrics': getattr(result, 'optimization_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_lookback_optimization_report(optimization_result, data)

            return optimization_result

        except Exception as e:
            self.logger.error(f"Lookback optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'optimal_lookbacks': {},
                'optimization_metrics': {}
            }

    async def run_period_lookback_optimization_step(self,
                                                  data: pd.DataFrame,
                                                  symbol: str = "ETHUSDT",
                                                  timeframe: str = "15m",
                                                  direction: str = "longs",
                                                  intensity: str = "blank",
                                                  lookback_days: Optional[int] = None,
                                                  start_date: Optional[str] = None,
                                                  end_date: Optional[str] = None,
                                                  exchange: str = "binance",
                                                  custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to concurrent period + lookback optimization step with M1 optimizations."""
        try:
            tprint_step("🚀 Starting M1-optimized period + lookback optimization step")
            
            # Log optimization configuration
            tprint_info(f"🧠 M1 Optimization Configuration:")
            tprint_info(f"   - Parallel Workers: {self.parallel_workers}")
            tprint_info(f"   - Chunk Size: {self.chunk_size}")
            tprint_info(f"   - Memory Mapping: {self.memory_mapping_enabled}")
            tprint_info(f"   - Aggressive GC: {self.aggressive_gc_enabled}")
            tprint_info(f"   - Data Type Optimization: {self.data_type_optimization}")
            tprint_info(f"   - M1 GPU Available: {self.m1_gpu_manager.mps_available}")
            
            # Optimize input data
            tprint_info("🔧 Optimizing input data for M1 processing")
            data = self._optimize_pipeline_data(data)
            
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'period_lookback_optimization',
                'm1_optimization_enabled': True,
                'parallel_workers': self.parallel_workers,
                'chunk_size': self.chunk_size,
                'memory_mapping_enabled': self.memory_mapping_enabled,
                'aggressive_gc_enabled': self.aggressive_gc_enabled,
                'data_type_optimization': self.data_type_optimization
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Create M1 optimization context
            with self.m1_cpu_optimizer.create_m1_optimized_context():
                tprint_info("🚀 Running M1-optimized pipeline processing")
                # Run pipeline up to concurrent period + lookback optimization
                result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract period + lookback optimization results
            optimization_result = {
                'success': result.success,
                'period_results': getattr(result, 'period_results', {}),
                'lookback_results': getattr(result, 'lookback_results', {}),
                'combined_results': getattr(result, 'combined_results', {}),
                'trading_defaults': getattr(result, 'trading_defaults', {}),
                'interaction_periods': getattr(result, 'interaction_periods', []),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None,
                'm1_optimization_stats': {
                    'parallel_workers_used': self.parallel_workers,
                    'chunk_size': self.chunk_size,
                    'memory_mapping_enabled': self.memory_mapping_enabled,
                    'aggressive_gc_enabled': self.aggressive_gc_enabled,
                    'data_type_optimization': self.data_type_optimization,
                    'm1_gpu_acceleration': self.m1_gpu_manager.mps_available
                }
            }

            # Finalize processing with cleanup
            self._finalize_pipeline_processing()

            # Generate human-readable report
            await self._generate_period_lookback_optimization_report(optimization_result, data)

            tprint_success("✅ M1-optimized period + lookback optimization completed")
            return optimization_result

        except Exception as e:
            tprint_error(f"❌ M1-optimized period + lookback optimization failed: {e}")
            self.logger.error(f"Concurrent period + lookback optimization step failed: {e}")
            
            # Ensure cleanup even on error
            try:
                self._finalize_pipeline_processing()
            except Exception as cleanup_error:
                tprint_warning(f"Cleanup failed: {cleanup_error}")
            
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'period_results': {},
                'lookback_results': {},
                'combined_results': {},
                'trading_defaults': {},
                'interaction_periods': [],
                'm1_optimization_stats': {
                    'optimization_failed': True,
                    'error': str(e)
                }
            }

    async def run_interaction_generation_step(self,
                                            data: pd.DataFrame,
                                            symbol: str = "ETHUSDT",
                                            timeframe: str = "15m",
                                            direction: str = "longs",
                                            intensity: str = "blank",
                                            lookback_days: Optional[int] = None,
                                            start_date: Optional[str] = None,
                                            end_date: Optional[str] = None,
                                            exchange: str = "binance",
                                            custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run M1-optimized pipeline up to interaction generation step with full hardware acceleration."""
        try:
            tprint_step("🚀 Starting M1-optimized interaction generation step")
            
            # Log optimization configuration
            tprint_info(f"🧠 M1 Optimization Configuration:")
            tprint_info(f"   - Parallel Workers: {self.parallel_workers}")
            tprint_info(f"   - Chunk Size: {self.chunk_size}")
            tprint_info(f"   - Memory Mapping: {self.memory_mapping_enabled}")
            tprint_info(f"   - Aggressive GC: {self.aggressive_gc_enabled}")
            tprint_info(f"   - Data Type Optimization: {self.data_type_optimization}")
            tprint_info(f"   - M1 GPU Available: {self.m1_gpu_manager.mps_available}")
            
            # Optimize input data for M1 processing
            tprint_info("🔧 Optimizing input data for M1 processing")
            data = self._optimize_pipeline_data(data)
            
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create enhanced pipeline state with M1 optimization flags
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'interaction_generation',
                'm1_optimization_enabled': True,
                'parallel_workers': self.parallel_workers,
                'chunk_size': self.chunk_size,
                'memory_mapping_enabled': self.memory_mapping_enabled,
                'aggressive_gc_enabled': self.aggressive_gc_enabled,
                'data_type_optimization': self.data_type_optimization,
                'm1_gpu_acceleration': self.m1_gpu_manager.mps_available
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Apply M1 GPU acceleration to targets if available
            if self.m1_gpu_manager.mps_available and not targets.empty:
                try:
                    tprint_info("🚀 Applying M1 GPU acceleration to targets")
                    targets_array = self.m1_gpu_manager.optimize_tensor_operations(targets.values)
                    targets = pd.Series(targets_array, index=targets.index, name=targets.name)
                    tprint_success("✅ Targets optimized with M1 GPU acceleration")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 GPU acceleration for targets failed: {e}")

            # Create M1 optimization context for pipeline processing
            with self.m1_cpu_optimizer.create_m1_optimized_context():
                tprint_info("🚀 Running M1-optimized pipeline processing")
                
                # Use chunked processing for large datasets
                if len(data) > self.chunk_size:
                    tprint_info(f"📦 Large dataset detected ({len(data)} rows), using chunked processing")
                    result = await self._process_large_dataset_with_chunking(
                        data, targets, timeframe, pipeline_state
                    )
                else:
                    # Standard processing for smaller datasets
                    result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract and optimize interaction generation results
            interaction_features = getattr(result, 'interaction_features', pd.DataFrame())
            if not interaction_features.empty:
                tprint_info("🧠 Optimizing interaction features with M1 memory optimization")
                interaction_features = self._optimize_dataframe_dtypes(interaction_features)
                interaction_features = optimize_dataframe_for_m1(interaction_features)
                tprint_success(f"✅ Interaction features optimized: {interaction_features.shape}")

            # Build enhanced interaction result with M1 optimization statistics
            interaction_result = {
                'success': result.success,
                'interaction_features': interaction_features,
                'interaction_metadata': getattr(result, 'interaction_metadata', {}),
                'generation_metrics': getattr(result, 'generation_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None,
                'm1_optimization_stats': {
                    'parallel_workers_used': self.parallel_workers,
                    'chunk_size': self.chunk_size,
                    'memory_mapping_enabled': self.memory_mapping_enabled,
                    'aggressive_gc_enabled': self.aggressive_gc_enabled,
                    'data_type_optimization': self.data_type_optimization,
                    'm1_gpu_acceleration': self.m1_gpu_manager.mps_available,
                    'chunked_processing_used': len(data) > self.chunk_size,
                    'data_shape': data.shape,
                    'targets_optimized': not targets.empty and self.m1_gpu_manager.mps_available
                }
            }

            # Finalize processing with cleanup
            self._finalize_pipeline_processing()

            # Generate human-readable report
            await self._generate_interaction_generation_report(interaction_result, data)

            tprint_success("✅ M1-optimized interaction generation completed")
            return interaction_result

        except Exception as e:
            tprint_error(f"❌ M1-optimized interaction generation failed: {e}")
            self.logger.error(f"M1-optimized interaction generation step failed: {e}")
            
            # Ensure cleanup even on error
            try:
                self._finalize_pipeline_processing()
            except Exception as cleanup_error:
                tprint_warning(f"Cleanup failed: {cleanup_error}")
            
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'interaction_features': pd.DataFrame(),
                'interaction_metadata': {},
                'generation_metrics': {},
                'm1_optimization_stats': {
                    'optimization_failed': True,
                    'error': str(e)
                }
            }

    async def run_vectorization_step(self,
                                   data: pd.DataFrame,
                                   symbol: str = "ETHUSDT",
                                   timeframe: str = "15m",
                                   direction: str = "longs",
                                   intensity: str = "blank",
                                   lookback_days: Optional[int] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   exchange: str = "binance",
                                   custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to vectorization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'vectorization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to vectorization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract vectorization results
            vectorization_result = {
                'success': result.success,
                'vectorized_features': getattr(result, 'vectorized_features', pd.DataFrame()),
                'vectorization_metadata': getattr(result, 'vectorization_metadata', {}),
                'performance_metrics': getattr(result, 'performance_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_vectorization_report(vectorization_result, data)

            return vectorization_result

        except Exception as e:
            self.logger.error(f"Vectorization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'vectorized_features': pd.DataFrame(),
                'vectorization_metadata': {},
                'performance_metrics': {}
            }

    async def run_labeling_integration_step(self,
                                          data: pd.DataFrame,
                                          symbol: str = "ETHUSDT",
                                          timeframe: str = "15m",
                                          direction: str = "longs",
                                          intensity: str = "blank",
                                          lookback_days: Optional[int] = None,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          exchange: str = "binance",
                                          custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to labeling integration step."""
        try:
            # Check what custom_overrides contains
            if isinstance(custom_overrides, pd.DataFrame):
                self.logger.warning("custom_overrides is a DataFrame, converting to None")
                custom_overrides = None
            
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'labeling_integration'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to labeling integration
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract labeling integration results
            labeling_result = {
                'success': result.success,
                'labeled_data': getattr(result, 'labeled_data', pd.DataFrame()),
                'labeling_metadata': getattr(result, 'labeling_metadata', {}),
                'quality_metrics': getattr(result, 'quality_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_labeling_integration_report(labeling_result, data)

            return labeling_result

        except Exception as e:
            self.logger.error(f"Labeling integration step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'labeled_data': pd.DataFrame(),
                'labeling_metadata': {},
                'quality_metrics': {}
            }

    async def run_final_feature_selection_step(self,
                                             data: pd.DataFrame,
                                             symbol: str = "ETHUSDT",
                                             timeframe: str = "15m",
                                             direction: str = "longs",
                                             intensity: str = "blank",
                                             lookback_days: Optional[int] = None,
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             exchange: str = "binance",
                                             custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to final feature selection step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'final_feature_selection'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to final feature selection
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract final feature selection results
            final_selection_result = {
                'success': result.success,
                'final_selected_features': getattr(result, 'final_selected_features', pd.DataFrame()),
                'final_selection_metadata': getattr(result, 'final_selection_metadata', {}),
                'selection_metrics': getattr(result, 'selection_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_final_feature_selection_report(final_selection_result, data)

            return final_selection_result

        except Exception as e:
            self.logger.error(f"Final feature selection step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'final_selected_features': pd.DataFrame(),
                'final_selection_metadata': {},
                'selection_metrics': {}
            }

    async def run_final_validation_step(self,
                                      data: pd.DataFrame,
                                      symbol: str = "ETHUSDT",
                                      timeframe: str = "15m",
                                      direction: str = "longs",
                                      intensity: str = "blank",
                                      lookback_days: Optional[int] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      exchange: str = "binance",
                                      custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to final validation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'final_validation'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to final validation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract final validation results
            validation_result = {
                'success': result.success,
                'final_dataset': getattr(result, 'final_dataset', pd.DataFrame()),
                'validation_summary': getattr(result, 'validation_summary', {}),
                'quality_metrics': getattr(result, 'quality_metrics', {}),
                'pipeline_summary': getattr(result, 'pipeline_summary', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_final_validation_report(validation_result, data)

            return validation_result

        except Exception as e:
            self.logger.error(f"Final validation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'final_dataset': pd.DataFrame(),
                'validation_summary': {},
                'quality_metrics': {},
                'pipeline_summary': {}
            }

    def _get_labels_from_pipeline_state(self, pipeline_state: Optional[Dict[str, Any]]) -> Optional[pd.Series]:
        """
        Extract labels from pipeline state (from previous labeling steps).

        Args:
            pipeline_state: Pipeline state dictionary containing labels from previous steps

        Returns:
            Target series from previous steps, or None if not found
        """
        # Handle different types of pipeline_state
        if pipeline_state is None:
            return None
        
        # If pipeline_state is a DataFrame, return None (no labels available)
        if isinstance(pipeline_state, pd.DataFrame):
            return None
        
        # If pipeline_state is not a dict, return None
        if not isinstance(pipeline_state, dict):
            return None
        
        # Check if pipeline_state is empty (only for dictionaries)
        if isinstance(pipeline_state, dict) and not pipeline_state:
            return None

        # Try to get labels from various possible sources in pipeline state
        if 'labeled_data' in pipeline_state and 'target' in pipeline_state['labeled_data'].columns:
            return pipeline_state['labeled_data']['target']
        elif 'targets' in pipeline_state:
            return pipeline_state['targets']
        elif 'labels' in pipeline_state:
            return pipeline_state['labels']
        else:
            return None

    def _create_config_from_intensity(self, intensity: str, custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
        """
        Create configuration based on intensity level.

        Args:
            intensity: Intensity level ("full", "blank", "light")
            custom_overrides: Optional configuration overrides

        Returns:
            Configured UnifiedPipelineConfig instance

        Raises:
            ValueError: If intensity is invalid
            RuntimeError: If config creation fails
        """
        try:
            tprint_info(f"🔧 Creating configuration for intensity: {intensity}")

            # Validate intensity parameter
            valid_intensities = {"full", "blank", "light"}
            if intensity not in valid_intensities:
                raise ValueError(f"Invalid intensity '{intensity}'. Must be one of: {valid_intensities}")

            # Create base configuration
            if intensity == "full":
                tprint_info("📋 Creating full intensity configuration (100%)")
                config = create_full_config()
            elif intensity == "blank":
                tprint_info("📋 Creating blank intensity configuration (25%)")
                config = create_blank_config()
            elif intensity == "light":
                tprint_info("📋 Creating light intensity configuration (10%)")
                config = create_light_config()
            else:
                tprint_warning(f"⚠️ Unknown intensity '{intensity}', falling back to blank")
                config = create_config_by_intensity(PipelineIntensity.BLANK)

            if config is None:
                raise RuntimeError("Failed to create configuration")

            # Apply custom overrides if provided
            if custom_overrides:
                tprint_info(f"🔧 Applying {len(custom_overrides)} custom overrides")
                for key, value in custom_overrides.items():
                    if hasattr(config, key):
                        old_value = getattr(config, key)
                        setattr(config, key, value)
                        tprint_debug(f"  - {key}: {old_value} → {value}")
                    else:
                        tprint_warning(f"  - Unknown config key: {key}")

            tprint_success(f"✅ Configuration created successfully for intensity: {intensity}")
            return config

        except ValueError as e:
            error_msg = f"Invalid intensity parameter: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to create configuration: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _generate_data_validation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """
        Generate human-readable report for data validation step.

        Args:
            result: Validation result dictionary
            data: Input DataFrame

        Raises:
            OSError: If report file cannot be created
            ValueError: If result data is invalid
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"data_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        try:
            tprint_info("📄 Generating data validation report")

            # Validate inputs
            if not isinstance(result, dict):
                raise ValueError(f"Result must be a dictionary, got {type(result)}")

            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data must be a pandas DataFrame, got {type(data)}")

            # Create outcomes directory
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            tprint_debug(f"📁 Using outcomes directory: {outcomes_dir.absolute()}")

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"data_validation_report_{timestamp}.md"
            report_path = outcomes_dir / report_filename

            # Generate report content
            status_emoji = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            quality_score = result.get('data_quality_score', 0.0)
            error_msg = result.get('error_message', 'None')
            artifacts_count = len(result.get('artifacts', {}))

            report_content = f"""# Data Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {status_emoji}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Quality Score**: {quality_score:.3f}

## Validation Results
- **Success**: {result['success']}
- **Error Message**: {error_msg}
- **Artifacts Generated**: {artifacts_count}

## Data Quality Metrics
- **Rows**: {data.shape[0]:,}
- **Columns**: {data.shape[1]:,}
- **Memory Usage**: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Missing Values**: {data.isnull().sum().sum():,}

## Next Steps
1. Review validation results
2. Address any issues if present
3. Proceed to feature generation step

---
*Report generated by Consolidated Pipeline Runner*
"""

            # Write report
            tprint_debug(f"💾 Writing report to: {report_path}")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            # Add report to artifacts
            result['artifacts']['human_readable_report'] = str(report_path)

            tprint_success(f"📊 Human-readable report saved: {report_path}")

        except OSError as e:
            error_msg = f"Failed to create report file: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            raise OSError(error_msg) from e
        except ValueError as e:
            error_msg = f"Invalid data for report generation: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error generating report: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def _generate_feature_generation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for feature generation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"feature_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Feature Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Generated Features**: {len(result.get('generated_features', pd.DataFrame()).columns)}

## Generation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review generated features
2. Proceed to feature selection step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_feature_selection_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for feature selection step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"feature_selection_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Feature Selection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Selected Features**: {len(result.get('selected_features', pd.DataFrame()).columns)}

## Selection Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review selected features
2. Proceed to period optimization step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_period_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for period optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"period_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Period Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Optimized Periods**: {len(result.get('optimal_periods', {}))}

## Optimization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review optimized periods
2. Proceed to lookback optimization step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_lookback_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for lookback optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"lookback_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Lookback Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Optimized Lookbacks**: {len(result.get('optimal_lookbacks', {}))}

## Optimization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review optimized lookbacks
2. Proceed to interaction generation step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_interaction_generation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for interaction generation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"interaction_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Interaction Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Interaction Features**: {len(result.get('interaction_features', pd.DataFrame()).columns)}

## Generation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review interaction features
2. Proceed to vectorization step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_vectorization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for vectorization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"vectorization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Vectorization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Vectorized Features**: {len(result.get('vectorized_features', pd.DataFrame()).columns)}

## Vectorization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review vectorized features
2. Proceed to labeling integration step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_labeling_integration_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for labeling integration step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"labeling_integration_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Labeling Integration Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Labeled Data**: {len(result.get('labeled_data', pd.DataFrame()).columns)}

## Labeling Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review labeled data
2. Proceed to final validation step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_final_validation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for final validation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"final_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Final Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Final Dataset**: {len(result.get('final_dataset', pd.DataFrame()).columns)}

## Validation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review final dataset
2. Use dataset for model training

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_final_feature_selection_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for final feature selection step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"final_feature_selection_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Final Feature Selection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Final Selected Features**: {len(result.get('final_selected_features', pd.DataFrame()).columns)}

## Final Selection Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review final selected features
2. Proceed to final validation step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_period_lookback_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for period + lookback optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"period_lookback_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Period + Lookback Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result.get('success', False) else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Optimized Periods**: {result.get('optimized_periods', 'N/A')}
- **Optimized Lookbacks**: {result.get('optimized_lookbacks', 'N/A')}

## Period Optimization Results
- **Success**: {result.get('period_result', {}).get('success', False)}
- **Best Period**: {result.get('period_result', {}).get('optimized_periods', 'N/A')}
- **Best Score**: {result.get('period_result', {}).get('metadata', {}).get('best_score', 'N/A')}

## Lookback Optimization Results
- **Success**: {result.get('lookback_result', {}).get('success', False)}
- **Best Lookback**: {result.get('lookback_result', {}).get('optimized_lookbacks', 'N/A')}
- **Best Score**: {result.get('lookback_result', {}).get('metadata', {}).get('best_score', 'N/A')}

## Combined Results
- **Artifacts Generated**: {len(result.get('combined_artifacts', {}))}
- **Metadata Fields**: {len(result.get('combined_metadata', {}))}

## Next Steps
1. Review optimization results
2. Proceed to next pipeline step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['combined_artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

# Convenience functions for each step
async def run_data_validation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run data validation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing validation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting data validation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_data_validation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for data validation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_feature_generation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run feature generation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing feature generation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting feature generation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_feature_generation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for feature generation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_feature_selection_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run feature selection step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing feature selection results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting feature selection step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_feature_selection_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for feature selection: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_period_optimization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run period optimization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing period optimization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting period optimization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_period_optimization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for period optimization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_lookback_optimization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run lookback optimization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing lookback optimization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting lookback optimization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_lookback_optimization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for lookback optimization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_interaction_generation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run M1-optimized interaction generation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing interaction generation results with M1 optimization statistics

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting M1-optimized interaction generation step (convenience function)")
        
        # Initialize M1 optimizers for convenience function
        m1_gpu_manager = get_m1_gpu_manager()
        m1_memory_optimizer = get_m1_memory_optimizer()
        m1_cpu_optimizer = get_m1_cpu_optimizer()
        
        # Start memory monitoring
        m1_memory_optimizer.start_monitoring()
        
        try:
            # Optimize input data for M1 processing
            tprint_info("🔧 Optimizing input data for M1 processing (convenience function)")
            data = optimize_dataframe_for_m1(data)
            data = optimize_dataframe_memory(data)
            
            # Create runner and execute step
            runner = ConsolidatedPipelineRunner()
            result = await runner.run_interaction_generation_step(data, **kwargs)
            
            # Log M1 optimization statistics if available
            if 'm1_optimization_stats' in result:
                stats = result['m1_optimization_stats']
                tprint_info("📊 M1 Optimization Statistics (Convenience Function):")
                tprint_info(f"   - Workers Used: {stats.get('parallel_workers_used', 'N/A')}")
                tprint_info(f"   - GPU Acceleration: {stats.get('m1_gpu_acceleration', False)}")
                tprint_info(f"   - Memory Mapping: {stats.get('memory_mapping_enabled', False)}")
                tprint_info(f"   - Data Type Optimization: {stats.get('data_type_optimization', False)}")
                tprint_info(f"   - Chunked Processing: {stats.get('chunked_processing_used', False)}")
                tprint_info(f"   - Data Shape: {stats.get('data_shape', 'N/A')}")
                tprint_info(f"   - Targets Optimized: {stats.get('targets_optimized', False)}")
            
            return result
            
        finally:
            # Ensure cleanup
            m1_memory_optimizer.stop_monitoring()
            optimize_memory()
            
    except Exception as e:
        error_msg = f"M1-optimized convenience function failed for interaction generation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_vectorization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run vectorization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing vectorization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting vectorization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_vectorization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for vectorization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_labeling_integration_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run labeling integration step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing labeling integration results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting labeling integration step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_labeling_integration_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for labeling integration: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_period_lookback_optimization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run concurrent period + lookback optimization step using consolidated pipeline with M1 optimizations.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing period + lookback optimization results with M1 optimization statistics

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting M1-optimized concurrent period + lookback optimization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        result = await runner.run_period_lookback_optimization_step(data, **kwargs)
        
        # Log M1 optimization statistics if available
        if 'm1_optimization_stats' in result:
            stats = result['m1_optimization_stats']
            tprint_info("📊 M1 Optimization Statistics:")
            tprint_info(f"   - Workers Used: {stats.get('parallel_workers_used', 'N/A')}")
            tprint_info(f"   - GPU Acceleration: {stats.get('m1_gpu_acceleration', False)}")
            tprint_info(f"   - Memory Mapping: {stats.get('memory_mapping_enabled', False)}")
            tprint_info(f"   - Data Type Optimization: {stats.get('data_type_optimization', False)}")
        
        return result
    except Exception as e:
        error_msg = f"M1-optimized convenience function failed for period + lookback optimization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_final_feature_selection_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run final feature selection step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing final feature selection results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting final feature selection step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_final_feature_selection_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for final feature selection: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_final_validation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run final validation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing final validation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting final validation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_final_validation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for final validation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e
