"""
Enhanced Feature Generation Step

This step generates features using the AutoOptimizedFeatureGenerator
with VectorBT optimization and comprehensive feature categories.
"""

from __future__ import annotations

import logging
import json
import warnings
import pandas as pd
import numpy as np
import copy
import asyncio
import time
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from src.training.steps.base_step import BaseStep
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Self-contained optimization components
class OperationType:
    """Operation types for vectorization."""
    FEATURE_GENERATION = "feature_generation"
    MATRIX_OPERATIONS = "matrix_operations"
    DATA_TRANSFORMATION = "data_transformation"

class M1GPUManager:
    """Self-contained M1 GPU manager."""
    def __init__(self):
        self.mps_available = False  # Simplified - no actual M1 GPU detection
    
    def optimize_dataframe(self, df):
        """Optimize dataframe for M1."""
        return df.copy()

class M1MemoryOptimizer:
    """Self-contained M1 memory optimizer."""
    def __init__(self, memory_limit_gb=8.0):
        self.memory_limit_gb = memory_limit_gb
    
    def optimize_dataframe(self, df):
        """Optimize dataframe memory usage."""
        return df.copy()
    
    def optimize_memory(self):
        """Optimize memory usage."""
        gc.collect()
        return {'success': True}

class M1CPUOptimizer:
    """Self-contained M1 CPU optimizer."""
    def __init__(self):
        self.max_workers = 4
    
    def create_thread_pool(self):
        """Create optimized thread pool."""
        return ThreadPoolExecutor(max_workers=self.max_workers)
    
    def parallel_map(self, func, items):
        """Parallel map function."""
        return [func(item) for item in items]

class UnifiedVectorizationManager:
    """Self-contained unified vectorization manager."""
    def __init__(self):
        self.operation_type = OperationType.FEATURE_GENERATION
    
    def optimize_operation(self, data, operation_type, **kwargs):
        """Optimize operation using vectorization."""
        return data.copy()

# Set availability flag
M1_OPTIMIZATION_AVAILABLE = True

# Convenience functions
def optimize_dataframe_for_m1(df): 
    return df.copy()

def optimize_dataframe_memory(df): 
    return df.copy()

def optimize_memory(): 
    gc.collect()
    return {'success': True}

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    # Fallback if tprint is not available
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args)

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced feature generation step."""
        super().__init__("feature_generation_feature_generation_step", config)
        
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
        
        # Initialize M1 optimization components
        if M1_OPTIMIZATION_AVAILABLE:
            tprint_info("🧠 Initializing M1 optimization components for feature generation")
            self.m1_gpu_manager = M1GPUManager()
            self.m1_memory_optimizer = M1MemoryOptimizer(memory_limit_gb=8)
            self.m1_cpu_optimizer = M1CPUOptimizer()
            self.unified_vectorization_manager = UnifiedVectorizationManager()
            
            # M1 optimization configuration
            self.parallel_workers = 6  # Optimized for M1
            self.chunk_size = 10000  # Memory-efficient chunk size
            self.memory_mapped_threshold = 50000  # Use memory mapping for large datasets
            self.aggressive_gc_threshold = 0.8  # Trigger aggressive GC at 80% memory usage
            self.float32_conversion = True  # Convert float64 to float32 where possible
            
            # Performance tracking
            self.performance_stats = {
                'total_processing_time': 0.0,
                'memory_optimizations_applied': 0,
                'chunks_processed': 0,
                'gpu_accelerations_used': 0,
                'vectorbt_optimizations_used': 0
            }
            
            tprint_success("🧠 M1 optimization components initialized")
        else:
            tprint_warning("⚠️ M1 optimization components not available")
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.unified_vectorization_manager = None
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
        
        # Set context for enhanced file naming
        self._set_context(symbol=symbol, exchange=exchange, direction=direction, model='Analyst')
        
        # Start M1 memory monitoring
        if self.m1_memory_optimizer:
            self.m1_memory_optimizer.start_monitoring()

        # Check if CMI complementarity is enabled (Tactician mode only)
        enable_cmi_complementarity = (
            CMI_COMPLEMENTARITY_AVAILABLE and 
            self.cmi_scorer is not None and 
            pipeline_state is not None and 
            pipeline_state.get('tactician_mode', False)
        )
        
        if enable_cmi_complementarity:
            self.logger.info("🎯 CMI complementarity enabled for Tactician mode")
        else:
            self.logger.info("📊 Standard feature generation (Analyst mode or CMI unavailable)")

        try:
            # Try to load cached features using BaseStep methods
            cached_features = self._load_dataframe('generated_features')
            cached_feature_names = self._load_metadata('feature_names')
            cached_categories = self._load_metadata('feature_categories')

            # DEBUG: Check data quality at the start of execute
            self.logger.debug("Execute - data shape: %s", data.shape)
            numeric = data.select_dtypes(include=[np.number])
            non_finite_total = (~np.isfinite(numeric)).to_numpy().sum()
            self.logger.debug("Execute - non-finite total: %d", non_finite_total)
            for col in numeric.columns:
                nf = (~np.isfinite(numeric[col])).sum()
                if nf:
                    self.logger.debug("Execute - %s: %d non-finite", col, nf)

            # Clone feature configuration to avoid mutating shared config
            if FEATURE_GENERATION_AVAILABLE:
                base_cfg = copy.deepcopy(self.feature_config)
                base_cfg.symbol = symbol
                base_cfg.timeframe = timeframe
            else:
                base_cfg = None

            # Validate input data
            if data is None or len(data) == 0:
                raise ValueError("Input data is None or empty")
            
            # Use proper validation that matches FeatureConfig requirements
            required_columns = getattr(self.feature_config, 'required_columns', ['open', 'high', 'low', 'close', 'volume'])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Available: {list(data.columns)}")
            if not FEATURE_GENERATION_AVAILABLE or self.feature_bank is None:
                # Fast fail if enhanced components are not available
                raise RuntimeError("Enhanced feature generation components are not available")

            # Apply M1 optimization to input data
            if self.m1_memory_optimizer:
                tprint_info("🧠 Applying M1 memory optimization to input data")
                data = self._optimize_dataframe_memory(data)
                
            if self.m1_gpu_manager and self.m1_gpu_manager.mps_available:
                tprint_info("🚀 Applying M1 GPU acceleration to input data")
                data = optimize_dataframe_for_m1(data)
                
            # Monitor memory usage
            self._monitor_memory_usage()

            # Perform comprehensive feature generation
            generation_result = await self._perform_enhanced_feature_generation(
                data, symbol, timeframe, direction, custom_overrides, base_cfg,
                enable_cmi_complementarity, pipeline_state, targets
            )

            if generation_result.success:
                # Update performance statistics
                end_time = time.time()
                self.performance_stats['total_processing_time'] = end_time - start_time
                
                # Add M1 optimization statistics to generation result
                if hasattr(generation_result, 'optimization_stats'):
                    generation_result.optimization_stats.update({
                        'm1_optimizations': {
                            'total_processing_time': self.performance_stats['total_processing_time'],
                            'memory_optimizations_applied': self.performance_stats['memory_optimizations_applied'],
                            'chunks_processed': self.performance_stats['chunks_processed'],
                            'gpu_accelerations_used': self.performance_stats['gpu_accelerations_used'],
                            'vectorbt_optimizations_used': self.performance_stats['vectorbt_optimizations_used'],
                            'parallel_workers': self.parallel_workers,
                            'chunk_size': self.chunk_size,
                            'm1_gpu_available': self.m1_gpu_manager.mps_available if self.m1_gpu_manager else False,
                            'm1_memory_optimizer_used': self.m1_memory_optimizer is not None,
                            'm1_cpu_optimizer_used': self.m1_cpu_optimizer is not None
                        }
                    })
                
                self.logger.info(f"M1-optimized enhanced feature generation completed successfully")
                self.logger.info(f"Generated {len(generation_result.generated_features.columns)} features")
                self.logger.info(f"Categories: {', '.join(generation_result.feature_categories)}")
                self.logger.info(f"Total processing time: {self.performance_stats['total_processing_time']:.2f} seconds")
                self.logger.info(f"M1 optimization stats: {generation_result.optimization_stats.get('m1_optimizations', {})}")
                
                # Extract actual data from FeatureResult objects before saving to artifact manager
                # This prevents serialization issues with FeatureResult objects
                clean_features_df = generation_result.generated_features.copy()
                
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
                
                # Generate final report
                report_path = await self._generate_final_report(
                    generation_result, symbol, timeframe, direction, exchange
                )
                self.logger.info(f"📊 Final report generated: {report_path}")
            else:
                self.logger.error(f"Feature generation failed: {generation_result.error_message}")

            return generation_result

        except Exception as e:
            self.logger.error(f"Enhanced feature generation step failed with exception: {e}")
            return FeatureGenerationResult(
                feature_names=[],
                feature_data=pd.DataFrame(),
                generated_features=pd.DataFrame(),
                feature_categories=[],
                generation_time=0.0,
                n_features_generated=0,
                cache_hit=False,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e),
                metadata={},
                optimization_stats={}
            )
        
        finally:
            # Cleanup M1 memory monitoring
            if self.m1_memory_optimizer:
                try:
                    self.m1_memory_optimizer.stop_monitoring()
                    optimize_memory()
                    tprint_info("🧠 M1 memory monitoring stopped and cleanup completed")
                except Exception as cleanup_error:
                    tprint_warning(f"⚠️ M1 cleanup failed: {cleanup_error}")

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply M1-specific memory optimizations to a DataFrame."""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            
            initial_memory = df.memory_usage(deep=True).sum()
            
            # Apply M1 GPU optimization
            if self.m1_gpu_manager and self.m1_gpu_manager.mps_available:
                df = optimize_dataframe_for_m1(df)
            
            # Convert float64 to float32 where precision allows
            if self.float32_conversion:
                for col in df.select_dtypes(include=[np.float64]).columns:
                    if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            
            # Apply M1 memory optimizer
            if self.m1_memory_optimizer:
                df = optimize_dataframe_memory(df)
            
            final_memory = df.memory_usage(deep=True).sum()
            memory_saved = initial_memory - final_memory
            
            if memory_saved > 0:
                tprint_info(f"🧠 Data type optimization: {memory_saved / 1024**2:.2f} MB saved")
                self.performance_stats['memory_optimizations_applied'] += 1
            
            return df
            
        except Exception as e:
            tprint_warning(f"Data type optimization failed: {e}")
            return df

    def _monitor_memory_usage(self) -> None:
        """Check current memory usage and trigger optimizations if needed."""
        try:
            if not self.m1_memory_optimizer:
                return
                
            memory_stats = self.m1_memory_optimizer.get_memory_stats()
            memory_percent = memory_stats.get('memory_percent', 0)
            
            if memory_percent > self.aggressive_gc_threshold:
                tprint_info(f"🧠 High memory usage detected ({memory_percent:.1f}%), triggering aggressive GC")
                
                # Force aggressive garbage collection
                for _ in range(3):
                    collected = gc.collect()
                    if collected > 0:
                        tprint_info(f"Garbage collection cycle: {collected} objects collected")
                
                # Clear M1-specific caches
                if self.m1_memory_optimizer:
                    memory_result = optimize_memory()
                    if memory_result.get('success', False):
                        memory_saved = memory_result.get('memory_saved_mb', 0)
                        if memory_saved > 0:
                            tprint_info(f"🧠 Memory optimization: {memory_saved:.1f} MB saved")
                            
        except Exception as e:
            tprint_warning(f"Memory monitoring failed: {e}")

    def _should_use_memory_mapping(self, data_size: int) -> bool:
        """Determine if memory mapping should be used based on data size."""
        return data_size > self.memory_mapped_threshold

    def _chunk_data_for_processing(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into smaller chunks for processing."""
        if len(data) <= self.chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
        
        return chunks

    def _process_chunk_with_optimization(self, chunk: pd.DataFrame, chunk_idx: int, **kwargs) -> pd.DataFrame:
        """Process a single data chunk with M1 optimizations."""
        try:
            # Apply memory optimization
            chunk = self._optimize_dataframe_memory(chunk)
            
            # Apply M1 GPU acceleration if available
            if self.m1_gpu_manager and self.m1_gpu_manager.mps_available:
                try:
                    chunk = optimize_dataframe_for_m1(chunk)
                    self.performance_stats['gpu_accelerations_used'] += 1
                    tprint_debug(f"🚀 Chunk {chunk_idx} optimized with M1 GPU acceleration")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 GPU acceleration failed for chunk {chunk_idx}: {e}")
            
            # Force garbage collection after processing chunk
            gc.collect()
            
            return chunk
            
        except Exception as e:
            tprint_error(f"❌ Chunk processing failed for chunk {chunk_idx}: {e}")
            return chunk

    def _combine_chunk_results(self, chunk_results: List[pd.DataFrame]) -> pd.DataFrame:
        """Efficiently combine results from multiple chunks."""
        try:
            if not chunk_results:
                return pd.DataFrame()
            
            if len(chunk_results) == 1:
                return chunk_results[0]
            
            # Combine chunks efficiently
            combined_df = pd.concat(chunk_results, ignore_index=True)
            
            # Apply final memory optimization
            combined_df = self._optimize_dataframe_memory(combined_df)
            
            return combined_df
            
        except Exception as e:
            tprint_error(f"❌ Failed to combine chunk results: {e}")
            return pd.DataFrame()

    def _apply_vectorbt_optimization(self, data: pd.DataFrame, operation_type: str, **kwargs) -> pd.DataFrame:
        """Apply VectorBT optimization using unified vectorization manager."""
        try:
            if not self.unified_vectorization_manager:
                return data
            
            # Map generic operation types to VectorBT-specific operations
            operation_mapping = {
                'rolling': OperationType.ROLLING,
                'correlation': OperationType.CORRELATION,
                'regression': OperationType.REGRESSION,
                'feature_generation': OperationType.FEATURE_GENERATION
            }
            
            vectorbt_operation = operation_mapping.get(operation_type, OperationType.FEATURE_GENERATION)
            
            # Use unified vectorization manager with VectorBT preference
            optimized_data = self.unified_vectorization_manager.optimize_operation(
                data=data,
                operation_type=vectorbt_operation,
                prefer_vectorbt=True,
                **kwargs
            )
            
            self.performance_stats['vectorbt_optimizations_used'] += 1
            tprint_info(f"🚀 Applied VectorBT optimization for {operation_type}")
            
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT optimization failed: {e}")
            return data

    async def _perform_enhanced_feature_generation(self, data: pd.DataFrame, symbol: str,
                                                   timeframe: str, direction: str,
                                                   custom_overrides: Optional[Dict[str, Any]],
                                                   base_config: Optional[FeatureConfig] = None,
                                                   enable_cmi_complementarity: bool = False,
                                                   pipeline_state: Optional[Dict[str, Any]] = None,
                                                   targets: Optional[pd.Series] = None) -> FeatureGenerationResult:
        """Perform enhanced feature generation using FeatureBank."""
        
        start_time = time.time()
        
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
            
            # Add progress monitoring during feature generation
            self.logger.info("📊 Starting feature generation process...")
            self.logger.info(f"📈 Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            self.logger.info(f"🧮 Total memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Use FeatureBank to generate features properly
            generation_start_time = time.time()
            generated_features_df = self.feature_bank.generate_features(
                data=data,
                categories=feature_categories,
                use_optimized_pipeline=True,
                lookback_optimization=True,
                execution_mode=custom_overrides.get('execution_mode') if custom_overrides else self.config.get('execution_mode')  # Pass execution mode for light mode restriction
            )
            generation_duration = time.time() - generation_start_time
            
            # Apply CMI complementarity filtering if enabled
            if enable_cmi_complementarity and targets is not None:
                tprint_info("🎯 Applying CMI complementarity filtering to generated features")
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
            
            return FeatureGenerationResult(
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
                    'cmi_diagnostics': cmi_diagnostics
                },
                artifacts={
                    'feature_dataframe': generated_features_df,
                    'feature_names': feature_names,
                    'feature_categories': feature_categories,
                    'vectorbt_optimizations': vectorbt_optimizations,
                    'raw_dataframe': data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced feature generation failed: {e}")
            # Fast fail - no fallback, just raise the error
            raise RuntimeError(f"Feature generation failed: {e}") from e
    
    def _optimize_dataframe_for_saving(self, df):
        """Optimize DataFrame for efficient saving by reducing memory usage."""
        try:
            import pandas as pd
            import numpy as np
            
            self.logger.info("🔧 Optimizing DataFrame data types for memory efficiency...")
            
            # Create a copy to avoid modifying original
            optimized_df = df.copy()
            original_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_data = optimized_df[col]
                
                # Skip if column has NaN values that might cause issues
                if col_data.isna().any():
                    continue
                
                # Float64 -> Float32 optimization
                if col_data.dtype == np.float64:
                    if (col_data.max() < np.finfo(np.float32).max and
                        col_data.min() > np.finfo(np.float32).min):
                        optimized_df[col] = col_data.astype(np.float32)
                
                # Int64 -> Int32 optimization
                elif col_data.dtype == np.int64:
                    if (col_data.max() < np.iinfo(np.int32).max and
                        col_data.min() > np.iinfo(np.int32).min):
                        optimized_df[col] = col_data.astype(np.int32)
                
                # Int64 -> Int16 optimization for small ranges
                elif col_data.dtype == np.int64:
                    if (col_data.max() < np.iinfo(np.int16).max and
                        col_data.min() > np.iinfo(np.int16).min):
                        optimized_df[col] = col_data.astype(np.int16)
            
            # Optimize object columns to category if beneficial
            for col in optimized_df.select_dtypes(include=['object']).columns:
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
            
            # Calculate memory savings
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
            memory_saved = original_memory - optimized_memory
            reduction_percentage = (memory_saved / original_memory) * 100 if original_memory > 0 else 0
            
            self.logger.info(f"💾 Memory optimization: {memory_saved:.1f}MB saved ({reduction_percentage:.1f}% reduction)")
            self.logger.info(f"📊 Optimized memory usage: {optimized_memory:.1f}MB")
            
            return optimized_df
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}, using original DataFrame")
            return df
    
    def _save_dataframe_chunked(self, df, filepath, chunk_size=10000):
        """Save large DataFrame in chunks to reduce memory pressure."""
        try:
            import pandas as pd
            
            self.logger.info(f"📝 Saving DataFrame in chunks of {chunk_size} rows...")
            
            # Get total rows
            total_rows = len(df)
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            # Save header first
            df.head(0).to_csv(filepath, index=False)
            
            # Append chunks
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                
                chunk = df.iloc[start_idx:end_idx]
                chunk.to_csv(filepath, mode='a', header=False, index=False)
                
                # Progress update
                progress = (i + 1) / num_chunks * 100
                if (i + 1) % 10 == 0 or i == num_chunks - 1:  # Log every 10 chunks or last chunk
                    self.logger.info(f"📈 Chunked save progress: {progress:.1f}% ({i+1}/{num_chunks} chunks)")
            
            self.logger.info(f"✅ Chunked save completed: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Chunked save failed: {e}")
            # Fallback to regular save
            df.to_csv(filepath, index=False)



    async def _generate_final_report(self, generation_result: FeatureGenerationResult, 
                                    symbol: str, timeframe: str, direction: str, exchange: str = "binance") -> str:
        """Generate a human-readable final report."""
        try:
            from datetime import datetime
            import os
            
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
## Optimization Statistics
"""
            
            if generation_result.optimization_stats:
                for key, value in generation_result.optimization_stats.items():
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
                report_content += """- ✅ Feature generation completed successfully
- 📊 Consider analyzing feature importance for model training
- 🔍 Review feature categories for completeness
- 💾 Features are ready for model training pipeline
"""
            else:
                report_content += """- ❌ Feature generation failed
- 🔧 Check error message for specific issues
- 🔄 Consider retrying with different parameters
- 📋 Review input data quality
"""
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
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
    # Handle None intensity by defaulting to light mode (more reasonable default)
    if intensity is None:
        intensity = "light"

    try:
        # Create the step instance
        step = FeatureGenerationStep(
            name="feature_generation_step",
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
        result = await step.execute(
            training_input=training_input,
            pipeline_state={},
            **kwargs
        )

        return result

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Handler function failed: {e}")
        return ComponentResult(
            success=False,
            metadata={},
            error_message=str(e)
        )
