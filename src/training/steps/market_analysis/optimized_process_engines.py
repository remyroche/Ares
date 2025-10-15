"""
Optimized Process Engines for Market Analysis

This module provides optimized engines for all major market analysis processes:
1. Feature Lookback Optimization
2. Interactive Feature Generation  
3. Multi-Horizon Profit Labeler
4. Final Feature Selection

Each engine implements:
- Vectorized operations using matrix operations
- Intelligent caching for repeated calculations
- Chunking for large dataset processing
- Hardware acceleration for M1/M2/M3 chips
- Parallel processing where applicable
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import hashlib
import time
from contextlib import nullcontext
from enum import Enum

# Import optimization utilities
try:
    from src.utils.matrix_operations import (
        get_vectorized_processing_core,
        get_hardware_optimized_processor,
        hardware_optimized,
        optimize_matrix_operation,
        vectorized_rolling_features,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_advanced_cpu_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_memory_optimizer,
        optimize_for_workload
    )
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False

from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.tprint import tprint

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

class ProcessType(Enum):
    """Types of processes that can be optimized."""
    INTERACTIVE_FEATURE_GENERATION = "interactive_feature_generation"
    MULTI_HORIZON_PROFIT_LABELER = "multi_horizon_profit_labeler"
    FINAL_FEATURE_SELECTION = "final_feature_selection"

class BaseOptimizedProcessEngine:
    """Base class for all optimized process engines."""
    
    def __init__(self, process_type: ProcessType, use_hardware_accel: bool = True, cache_size: int = 1000):
        self.process_type = process_type
        self.use_hardware_accel = use_hardware_accel and HARDWARE_ACCEL_AVAILABLE
        self.cache_size = cache_size
        
        # Initialize hardware components
        self.hardware_manager = None
        self.vectorized_core = None
        self.cpu_optimizer = None
        self.memory_optimizer = None
        self.matrix_ops = None
        
        # Caching
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize hardware acceleration
        self._initialize_hardware_components()
    
    def _initialize_hardware_components(self):
        """Initialize hardware acceleration components."""
        if self.use_hardware_accel:
            try:
                if HARDWARE_ACCEL_AVAILABLE:
                    self.hardware_manager = get_unified_hardware_manager()
                    self.cpu_optimizer = get_advanced_cpu_optimizer()
                    self.memory_optimizer = get_advanced_memory_optimizer()
                
                if MATRIX_OPS_AVAILABLE:
                    self.vectorized_core = get_vectorized_processing_core()
                
                # Initialize matrix operations
                self.matrix_ops = UnifiedMatrixOperations()
                    
                tprint(f"✅ {self.process_type.value} engine initialized with hardware acceleration")
            except Exception as e:
                tprint(f"⚠️ Hardware acceleration initialization failed for {self.process_type.value}: {e}")
                self.use_hardware_accel = False
    
    def _create_cache_key(self, data: Any, params: Dict[str, Any] = None) -> str:
        """Create a cache key for the given data and parameters."""
        try:
            # Create hash from data and parameters
            data_hash = hashlib.md5(str(data).encode()).hexdigest()
            params_hash = hashlib.md5(str(params or {}).encode()).hexdigest()
            return f"{self.process_type.value}_{data_hash}_{params_hash}"
        except Exception:
            return f"{self.process_type.value}_{hash(str(data))}_{hash(str(params))}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache."""
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        return None
    
    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in cache."""
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            'process_type': self.process_type.value,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

class OptimizedFeatureLookbackEngine(BaseOptimizedProcessEngine):
    """Optimized engine for feature lookback optimization."""
    
    def __init__(self, use_hardware_accel: bool = True, cache_size: int = 1000):
        super().__init__(ProcessType.FEATURE_LOOKBACK_OPTIMIZATION, use_hardware_accel, cache_size)
    
    def optimize_feature_lookbacks(self, features_df: pd.DataFrame, 
                                 target_column: str,
                                 lookback_ranges: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Optimize feature lookback periods using vectorized operations and caching.
        
        Args:
            features_df: DataFrame with features
            target_column: Target column for optimization
            lookback_ranges: List of (min, max) lookback ranges to test
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(features_df, {
                'target_column': target_column,
                'lookback_ranges': lookback_ranges
            })
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                tprint(f"📋 Using cached lookback optimization results for {target_column}")
                return cached_result
            
            # Use vectorized operations for optimization
            if self.use_hardware_accel and self.vectorized_core:
                result = self._optimize_with_matrix_ops(features_df, target_column, lookback_ranges)
            else:
                result = self._optimize_basic(features_df, target_column, lookback_ranges)
            
            # Cache the result
            self._store_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            tprint(f"⚠️ Feature lookback optimization failed: {e}")
            return {'error': str(e), 'optimized_features': {}}
    
    def _optimize_with_matrix_ops(self, features_df: pd.DataFrame, target_column: str, 
                                lookback_ranges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Optimize using matrix operations."""
        try:
            # Use vectorized core for preprocessing
            if self.vectorized_core:
                features_optimized = self.vectorized_core.optimize_dataframe_for_processing(features_df)
            else:
                features_optimized = features_df
            
            # Use hardware-optimized workload processing
            if self.hardware_manager:
                workload_config = {
                    'workload_type': 'interactive_feature_generation',
                    'data_size': len(features_df),
                    'complexity': 'medium',
                    'memory_intensive': False
                }
                
                # Optimize for lookback workload
                optimized_config = optimize_for_workload(workload_config)
                
                # Process with hardware optimization
                with self.cpu_optimizer.optimized_execution_context() if self.cpu_optimizer else nullcontext():
                    return self._process_lookback_optimization_with_hardware(features_optimized, target_column, lookback_ranges)
            else:
                return self._optimize_basic(features_optimized, target_column, lookback_ranges)
                
        except Exception as e:
            tprint(f"⚠️ Matrix operations lookback optimization failed: {e}")
            return self._optimize_basic(features_df, target_column, lookback_ranges)
    
    def _process_lookback_optimization_with_hardware(self, features_df: pd.DataFrame, 
                                                   target_column: str, 
                                                   lookback_ranges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Process lookback optimization with hardware acceleration."""
        # Use chunking for large datasets
        if len(features_df) > 10000 and self.memory_optimizer:
            return self._optimize_lookback_chunked(features_df, target_column, lookback_ranges)
        else:
            return self._optimize_basic(features_df, target_column, lookback_ranges)
    
    def _optimize_lookback_chunked(self, features_df: pd.DataFrame, target_column: str, 
                                 lookback_ranges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Optimize lookback using chunking for large datasets."""
        try:
            # Use memory-optimized chunking
            chunks = self.memory_optimizer.chunk_series(pd.Series(range(len(features_df))), chunk_size=5000)
            
            optimization_results = []
            
            for chunk in chunks:
                if len(chunk) < 100:  # Skip small chunks
                    continue
                
                chunk_features = features_df.iloc[chunk]
                chunk_result = self._optimize_basic(chunk_features, target_column, lookback_ranges)
                optimization_results.append(chunk_result)
            
            # Aggregate results
            return self._aggregate_lookback_results(optimization_results)
            
        except Exception as e:
            tprint(f"⚠️ Chunked lookback optimization failed: {e}")
            return self._optimize_basic(features_df, target_column, lookback_ranges)
    
    def _optimize_basic(self, features_df: pd.DataFrame, target_column: str, 
                       lookback_ranges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Basic lookback optimization using vectorized operations."""
        try:
            if lookback_ranges is None:
                lookback_ranges = [(5, 20), (10, 50), (20, 100)]
            
            optimized_features = {}
            optimization_metrics = {}
            
            # Use matrix operations for correlation calculations
            for feature in features_df.columns:
                if feature == target_column:
                    continue
                
                best_score = -1
                best_lookback = None
                
                for min_lookback, max_lookback in lookback_ranges:
                    # Vectorized correlation calculation
                    correlations = []
                    
                    for lookback in range(min_lookback, max_lookback + 1):
                        if lookback >= len(features_df):
                            continue
                        
                        # Calculate rolling correlation using vectorized operations
                        feature_series = features_df[feature].rolling(lookback).corr(features_df[target_column])
                        correlation = feature_series.dropna().mean()
                        
                        if not np.isnan(correlation):
                            correlations.append((lookback, abs(correlation)))
                    
                    # Find best correlation
                    if correlations:
                        best_lookback, best_score = max(correlations, key=lambda x: x[1])
                
                if best_lookback is not None:
                    optimized_features[feature] = best_lookback
                    optimization_metrics[feature] = {
                        'best_lookback': best_lookback,
                        'correlation_score': best_score
                    }
            
            return {
                'optimized_features': optimized_features,
                'optimization_metrics': optimization_metrics,
                'total_features_optimized': len(optimized_features)
            }
            
        except Exception as e:
            tprint(f"⚠️ Basic lookback optimization failed: {e}")
            return {'optimized_features': {}, 'optimization_metrics': {}}
    
    def _aggregate_lookback_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from chunked optimization."""
        try:
            all_optimized_features = {}
            all_metrics = {}
            
            for result in results:
                if 'optimized_features' in result:
                    all_optimized_features.update(result['optimized_features'])
                if 'optimization_metrics' in result:
                    all_metrics.update(result['optimization_metrics'])
            
            return {
                'optimized_features': all_optimized_features,
                'optimization_metrics': all_metrics,
                'total_features_optimized': len(all_optimized_features)
            }
            
        except Exception as e:
            tprint(f"⚠️ Result aggregation failed: {e}")
            return {'optimized_features': {}, 'optimization_metrics': {}}

class OptimizedInteractiveFeatureEngine(BaseOptimizedProcessEngine):
    """Optimized engine for interactive feature generation."""
    
    def __init__(self, use_hardware_accel: bool = True, cache_size: int = 1000):
        super().__init__(ProcessType.INTERACTIVE_FEATURE_GENERATION, use_hardware_accel, cache_size)
    
    def generate_interactive_features(self, base_features: pd.DataFrame,
                            interaction_features: List[Tuple[str, str]] = None,
                            polynomial_features: List[str] = None,
                            cross_timeframe_features: Dict[str, List[int]] = None) -> Dict[str, Any]:
        """
        Generate interactive features using vectorized operations and matrix operations.
        
        Args:
            base_features: Base feature DataFrame
            interaction_features: List of feature pairs for interaction
            polynomial_features: List of features for polynomial expansion
            cross_timeframe_features: Dictionary of features and timeframes
            
        Returns:
            Dictionary with generated features
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(base_features, {
                'interaction_features': interaction_features,
                'polynomial_features': polynomial_features,
                'cross_timeframe_features': cross_timeframe_features
            })
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                tprint("📋 Using cached interactive feature generation results")
                return cached_result
            
            # Use vectorized operations for feature generation
            if self.use_hardware_accel and self.vectorized_core:
                result = self._generate_with_matrix_ops(base_features, interaction_features, 
                                                      polynomial_features, cross_timeframe_features)
            else:
                result = self._generate_basic(base_features, interaction_features, 
                                            polynomial_features, cross_timeframe_features)
            
            # Cache the result
            self._store_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            tprint(f"⚠️ Interactive feature generation failed: {e}")
            return {'error': str(e), 'generated_features': pd.DataFrame()}
    
    def _generate_with_matrix_ops(self, base_features: pd.DataFrame,
                                interaction_features: List[Tuple[str, str]],
                                polynomial_features: List[str],
                                cross_timeframe_features: Dict[str, List[int]]) -> Dict[str, Any]:
        """Generate features using matrix operations."""
        try:
            # Use vectorized core for preprocessing
            if self.vectorized_core:
                features_optimized = self.vectorized_core.optimize_dataframe_for_processing(base_features)
            else:
                features_optimized = base_features
            
            # Use hardware-optimized workload processing
            if self.hardware_manager:
                workload_config = {
                    'workload_type': 'pid_feature_generation',
                    'data_size': len(base_features),
                    'complexity': 'high',
                    'memory_intensive': True
                }
                
                # Optimize for interactive generation workload
                optimized_config = optimize_for_workload(workload_config)
                
                # Process with hardware optimization
                with self.cpu_optimizer.optimized_execution_context() if self.cpu_optimizer else nullcontext():
                    return self._process_pid_generation_with_hardware(features_optimized, interaction_features, 
                                                                    polynomial_features, cross_timeframe_features)
            else:
                return self._generate_basic(features_optimized, interaction_features, 
                                          polynomial_features, cross_timeframe_features)
                
        except Exception as e:
            tprint(f"⚠️ Matrix operations interactive generation failed: {e}")
            return self._generate_basic(base_features, interaction_features, 
                                      polynomial_features, cross_timeframe_features)
    
    def _process_pid_generation_with_hardware(self, base_features: pd.DataFrame,
                                            interaction_features: List[Tuple[str, str]],
                                            polynomial_features: List[str],
                                            cross_timeframe_features: Dict[str, List[int]]) -> Dict[str, Any]:
        """Process interactive generation with hardware acceleration."""
        # Use chunking for large datasets
        if len(base_features) > 5000 and self.memory_optimizer:
            return self._generate_pid_chunked(base_features, interaction_features, 
                                            polynomial_features, cross_timeframe_features)
        else:
            return self._generate_basic(base_features, interaction_features, 
                                      polynomial_features, cross_timeframe_features)
    
    def _generate_pid_chunked(self, base_features: pd.DataFrame,
                            interaction_features: List[Tuple[str, str]],
                            polynomial_features: List[str],
                            cross_timeframe_features: Dict[str, List[int]]) -> Dict[str, Any]:
        """Generate interactive features using chunking for large datasets."""
        try:
            # Use memory-optimized chunking
            chunks = self.memory_optimizer.chunk_series(pd.Series(range(len(base_features))), chunk_size=2000)
            
            generated_features = []
            
            for chunk in chunks:
                if len(chunk) < 100:  # Skip small chunks
                    continue
                
                chunk_features = base_features.iloc[chunk]
                chunk_result = self._generate_basic(chunk_features, interaction_features, 
                                                  polynomial_features, cross_timeframe_features)
                generated_features.append(chunk_result)
            
            # Aggregate results
            return self._aggregate_pid_results(generated_features)
            
        except Exception as e:
            tprint(f"⚠️ Chunked interactive generation failed: {e}")
            return self._generate_basic(base_features, interaction_features, 
                                      polynomial_features, cross_timeframe_features)
    
    def _generate_basic(self, base_features: pd.DataFrame,
                       interaction_features: List[Tuple[str, str]],
                       polynomial_features: List[str],
                       cross_timeframe_features: Dict[str, List[int]]) -> Dict[str, Any]:
        """Basic interactive feature generation using vectorized operations."""
        try:
            generated_features = base_features.copy()
            feature_metadata = {}
            
            # Generate interaction features using matrix operations
            if interaction_features:
                interaction_results = self._generate_interaction_features(base_features, interaction_features)
                generated_features = pd.concat([generated_features, interaction_results['features']], axis=1)
                feature_metadata.update(interaction_results['metadata'])
            
            # Generate polynomial features using vectorized operations
            if polynomial_features:
                polynomial_results = self._generate_polynomial_features(base_features, polynomial_features)
                generated_features = pd.concat([generated_features, polynomial_results['features']], axis=1)
                feature_metadata.update(polynomial_results['metadata'])
            
            # Generate cross-timeframe features using matrix operations
            if cross_timeframe_features:
                cross_timeframe_results = self._generate_cross_timeframe_features(base_features, cross_timeframe_features)
                generated_features = pd.concat([generated_features, cross_timeframe_results['features']], axis=1)
                feature_metadata.update(cross_timeframe_results['metadata'])
            
            return {
                'generated_features': generated_features,
                'feature_metadata': feature_metadata,
                'total_features_generated': len(generated_features.columns) - len(base_features.columns)
            }
            
        except Exception as e:
            tprint(f"⚠️ Basic interactive generation failed: {e}")
            return {'generated_features': base_features.copy(), 'feature_metadata': {}}
    
    def _generate_interaction_features(self, base_features: pd.DataFrame, 
                                     interaction_features: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Generate interaction features using matrix operations."""
        try:
            interaction_df = pd.DataFrame(index=base_features.index)
            metadata = {}
            
            for feature1, feature2 in interaction_features:
                if feature1 in base_features.columns and feature2 in base_features.columns:
                    # Use vectorized operations for interaction
                    interaction = base_features[feature1] * base_features[feature2]
                    interaction_name = f"{feature1}_x_{feature2}"
                    interaction_df[interaction_name] = interaction
                    
                    metadata[interaction_name] = {
                        'type': 'interaction',
                        'base_features': [feature1, feature2],
                        'generation_method': 'vectorized_multiplication'
                    }
            
            return {
                'features': interaction_df,
                'metadata': metadata
            }
            
        except Exception as e:
            tprint(f"⚠️ Interaction feature generation failed: {e}")
            return {'features': pd.DataFrame(index=base_features.index), 'metadata': {}}
    
    def _generate_polynomial_features(self, base_features: pd.DataFrame, 
                                    polynomial_features: List[str]) -> Dict[str, Any]:
        """Generate polynomial features using vectorized operations."""
        try:
            polynomial_df = pd.DataFrame(index=base_features.index)
            metadata = {}
            
            for feature in polynomial_features:
                if feature in base_features.columns:
                    # Generate polynomial features (degree 2 and 3)
                    feature_data = base_features[feature]
                    
                    # Square (degree 2)
                    square_name = f"{feature}_squared"
                    polynomial_df[square_name] = feature_data ** 2
                    metadata[square_name] = {
                        'type': 'polynomial',
                        'base_feature': feature,
                        'degree': 2,
                        'generation_method': 'vectorized_power'
                    }
                    
                    # Cube (degree 3)
                    cube_name = f"{feature}_cubed"
                    polynomial_df[cube_name] = feature_data ** 3
                    metadata[cube_name] = {
                        'type': 'polynomial',
                        'base_feature': feature,
                        'degree': 3,
                        'generation_method': 'vectorized_power'
                    }
            
            return {
                'features': polynomial_df,
                'metadata': metadata
            }
            
        except Exception as e:
            tprint(f"⚠️ Polynomial feature generation failed: {e}")
            return {'features': pd.DataFrame(index=base_features.index), 'metadata': {}}
    
    def _generate_cross_timeframe_features(self, base_features: pd.DataFrame, 
                                         cross_timeframe_features: Dict[str, List[int]]) -> Dict[str, Any]:
        """Generate cross-timeframe features using matrix operations."""
        try:
            cross_timeframe_df = pd.DataFrame(index=base_features.index)
            metadata = {}
            
            for feature, timeframes in cross_timeframe_features.items():
                if feature in base_features.columns:
                    feature_data = base_features[feature]
                    
                    for timeframe in timeframes:
                        # Generate rolling statistics using vectorized operations
                        if timeframe < len(feature_data):
                            # Rolling mean
                            mean_name = f"{feature}_mean_{timeframe}"
                            cross_timeframe_df[mean_name] = feature_data.rolling(timeframe).mean()
                            metadata[mean_name] = {
                                'type': 'cross_timeframe',
                                'base_feature': feature,
                                'timeframe': timeframe,
                                'statistic': 'mean',
                                'generation_method': 'vectorized_rolling'
                            }
                            
                            # Rolling std
                            std_name = f"{feature}_std_{timeframe}"
                            cross_timeframe_df[std_name] = feature_data.rolling(timeframe).std()
                            metadata[std_name] = {
                                'type': 'cross_timeframe',
                                'base_feature': feature,
                                'timeframe': timeframe,
                                'statistic': 'std',
                                'generation_method': 'vectorized_rolling'
                            }
            
            return {
                'features': cross_timeframe_df,
                'metadata': metadata
            }
            
        except Exception as e:
            tprint(f"⚠️ Cross-timeframe feature generation failed: {e}")
            return {'features': pd.DataFrame(index=base_features.index), 'metadata': {}}
    
    def _aggregate_interactive_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from chunked interactive generation."""
        try:
            all_features = []
            all_metadata = {}
            
            for result in results:
                if 'generated_features' in result:
                    all_features.append(result['generated_features'])
                if 'feature_metadata' in result:
                    all_metadata.update(result['feature_metadata'])
            
            # Concatenate all features
            combined_features = pd.concat(all_features, axis=1) if all_features else pd.DataFrame()
            
            return {
                'generated_features': combined_features,
                'feature_metadata': all_metadata,
                'total_features_generated': len(combined_features.columns)
            }
            
        except Exception as e:
            tprint(f"⚠️ Interactive result aggregation failed: {e}")
            return {'generated_features': pd.DataFrame(), 'feature_metadata': {}}

class OptimizedMultiHorizonEngine(BaseOptimizedProcessEngine):
    """Optimized engine for multi-horizon profit labeling."""
    
    def __init__(self, use_hardware_accel: bool = True, cache_size: int = 1000):
        super().__init__(ProcessType.MULTI_HORIZON_PROFIT_LABELER, use_hardware_accel, cache_size)
    
    def generate_multi_horizon_labels(self, market_data: pd.DataFrame,
                                    horizons: List[int] = None,
                                    profit_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate multi-horizon profit labels using vectorized operations and caching.
        
        Args:
            market_data: DataFrame with OHLCV data
            horizons: List of horizon periods
            profit_thresholds: Dictionary of profit thresholds
            
        Returns:
            Dictionary with labeling results
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(market_data, {
                'horizons': horizons,
                'profit_thresholds': profit_thresholds
            })
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                tprint("📋 Using cached multi-horizon labeling results")
                return cached_result
            
            # Use vectorized operations for labeling
            if self.use_hardware_accel and self.vectorized_core:
                result = self._label_with_matrix_ops(market_data, horizons, profit_thresholds)
            else:
                result = self._label_basic(market_data, horizons, profit_thresholds)
            
            # Cache the result
            self._store_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            tprint(f"⚠️ Multi-horizon labeling failed: {e}")
            return {'error': str(e), 'labels': pd.DataFrame()}
    
    def _label_with_matrix_ops(self, market_data: pd.DataFrame,
                             horizons: List[int],
                             profit_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Generate labels using matrix operations."""
        try:
            # Use vectorized core for preprocessing
            if self.vectorized_core:
                data_optimized = self.vectorized_core.optimize_dataframe_for_processing(market_data)
            else:
                data_optimized = market_data
            
            # Use hardware-optimized workload processing
            if self.hardware_manager:
                workload_config = {
                    'workload_type': 'multi_horizon_labeling',
                    'data_size': len(market_data),
                    'complexity': 'medium',
                    'memory_intensive': False
                }
                
                # Optimize for labeling workload
                optimized_config = optimize_for_workload(workload_config)
                
                # Process with hardware optimization
                with self.cpu_optimizer.optimized_execution_context() if self.cpu_optimizer else nullcontext():
                    return self._process_labeling_with_hardware(data_optimized, horizons, profit_thresholds)
            else:
                return self._label_basic(data_optimized, horizons, profit_thresholds)
                
        except Exception as e:
            tprint(f"⚠️ Matrix operations labeling failed: {e}")
            return self._label_basic(market_data, horizons, profit_thresholds)
    
    def _process_labeling_with_hardware(self, market_data: pd.DataFrame,
                                      horizons: List[int],
                                      profit_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Process labeling with hardware acceleration."""
        # Use chunking for large datasets
        if len(market_data) > 10000 and self.memory_optimizer:
            return self._label_chunked(market_data, horizons, profit_thresholds)
        else:
            return self._label_basic(market_data, horizons, profit_thresholds)
    
    def _label_chunked(self, market_data: pd.DataFrame,
                      horizons: List[int],
                      profit_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Generate labels using chunking for large datasets."""
        try:
            # Use memory-optimized chunking
            chunks = self.memory_optimizer.chunk_series(pd.Series(range(len(market_data))), chunk_size=5000)
            
            labeling_results = []
            
            for chunk in chunks:
                if len(chunk) < 100:  # Skip small chunks
                    continue
                
                chunk_data = market_data.iloc[chunk]
                chunk_result = self._label_basic(chunk_data, horizons, profit_thresholds)
                labeling_results.append(chunk_result)
            
            # Aggregate results
            return self._aggregate_labeling_results(labeling_results)
            
        except Exception as e:
            tprint(f"⚠️ Chunked labeling failed: {e}")
            return self._label_basic(market_data, horizons, profit_thresholds)
    
    def _label_basic(self, market_data: pd.DataFrame,
                    horizons: List[int],
                    profit_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Basic multi-horizon labeling using vectorized operations."""
        try:
            if horizons is None:
                horizons = [5, 10, 20, 50]
            
            if profit_thresholds is None:
                profit_thresholds = {'min_profit': 0.01, 'max_profit': 0.05}
            
            labels_df = pd.DataFrame(index=market_data.index)
            labeling_metadata = {}
            
            # Calculate returns using vectorized operations
            if 'close' in market_data.columns:
                close_prices = market_data['close']
                
                for horizon in horizons:
                    # Calculate future returns using vectorized operations
                    future_returns = close_prices.shift(-horizon) / close_prices - 1
                    
                    # Generate labels using vectorized operations
                    labels = np.where(
                        future_returns >= profit_thresholds['max_profit'], 1,  # Profit
                        np.where(
                            future_returns <= -profit_thresholds['min_profit'], -1,  # Loss
                            0  # Hold
                        )
                    )
                    
                    label_name = f"label_h{horizon}"
                    labels_df[label_name] = labels
                    
                    labeling_metadata[label_name] = {
                        'horizon': horizon,
                        'profit_threshold': profit_thresholds['max_profit'],
                        'loss_threshold': profit_thresholds['min_profit'],
                        'generation_method': 'vectorized_thresholding'
                    }
            
            return {
                'labels': labels_df,
                'labeling_metadata': labeling_metadata,
                'total_labels_generated': len(labels_df.columns)
            }
            
        except Exception as e:
            tprint(f"⚠️ Basic labeling failed: {e}")
            return {'labels': pd.DataFrame(index=market_data.index), 'labeling_metadata': {}}
    
    def _aggregate_labeling_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from chunked labeling."""
        try:
            all_labels = []
            all_metadata = {}
            
            for result in results:
                if 'labels' in result:
                    all_labels.append(result['labels'])
                if 'labeling_metadata' in result:
                    all_metadata.update(result['labeling_metadata'])
            
            # Concatenate all labels
            combined_labels = pd.concat(all_labels, axis=0) if all_labels else pd.DataFrame()
            
            return {
                'labels': combined_labels,
                'labeling_metadata': all_metadata,
                'total_labels_generated': len(combined_labels.columns)
            }
            
        except Exception as e:
            tprint(f"⚠️ Labeling result aggregation failed: {e}")
            return {'labels': pd.DataFrame(), 'labeling_metadata': {}}

class OptimizedFeatureSelectionEngine(BaseOptimizedProcessEngine):
    """Optimized engine for final feature selection."""
    
    def __init__(self, use_hardware_accel: bool = True, cache_size: int = 1000):
        super().__init__(ProcessType.FINAL_FEATURE_SELECTION, use_hardware_accel, cache_size)
    
    def select_features(self, features_df: pd.DataFrame,
                       target_column: str,
                       selection_stages: List[int] = None) -> Dict[str, Any]:
        """
        Perform multi-stage feature selection using parallel processing and caching.
        
        Args:
            features_df: DataFrame with features
            target_column: Target column for selection
            selection_stages: List of target feature counts for each stage
            
        Returns:
            Dictionary with selection results
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(features_df, {
                'target_column': target_column,
                'selection_stages': selection_stages
            })
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                tprint("📋 Using cached feature selection results")
                return cached_result
            
            # Use parallel processing for selection stages
            if self.use_hardware_accel and self.cpu_optimizer:
                result = self._select_with_parallel_processing(features_df, target_column, selection_stages)
            else:
                result = self._select_basic(features_df, target_column, selection_stages)
            
            # Cache the result
            self._store_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            tprint(f"⚠️ Feature selection failed: {e}")
            return {'error': str(e), 'selected_features': pd.DataFrame()}
    
    def _select_with_parallel_processing(self, features_df: pd.DataFrame,
                                       target_column: str,
                                       selection_stages: List[int]) -> Dict[str, Any]:
        """Perform feature selection using parallel processing."""
        try:
            if selection_stages is None:
                selection_stages = [100, 80, 60]
            
            # Use parallel processing for different selection methods
            with ThreadPoolExecutor(max_workers=min(4, len(selection_stages))) as executor:
                futures = []
                
                for stage, target_count in enumerate(selection_stages):
                    future = executor.submit(
                        self._select_stage_parallel, 
                        features_df, target_column, stage, target_count
                    )
                    futures.append(future)
                
                # Collect results
                stage_results = []
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        stage_results.append(result)
                    except Exception as e:
                        tprint(f"⚠️ Parallel selection stage failed: {e}")
                        stage_results.append(None)
            
            # Aggregate results
            return self._aggregate_selection_results(stage_results)
            
        except Exception as e:
            tprint(f"⚠️ Parallel selection failed: {e}")
            return self._select_basic(features_df, target_column, selection_stages)
    
    def _select_stage_parallel(self, features_df: pd.DataFrame, target_column: str, 
                             stage: int, target_count: int) -> Dict[str, Any]:
        """Perform a single selection stage in parallel."""
        try:
            # Use matrix operations for correlation calculations
            if self.matrix_ops:
                correlations = self.matrix_ops.calculate_pairwise_similarities(
                    features_df.drop(columns=[target_column]).values, method='euclidean'
                )
            else:
                correlations = features_df.corr()
            
            # Select features based on correlation with target
            target_correlations = correlations[target_column].abs().sort_values(ascending=False)
            selected_features = target_correlations.head(target_count).index.tolist()
            
            return {
                'stage': stage,
                'target_count': target_count,
                'selected_features': selected_features,
                'correlation_scores': target_correlations.head(target_count).to_dict()
            }
            
        except Exception as e:
            tprint(f"⚠️ Selection stage {stage} failed: {e}")
            return None
    
    def _select_basic(self, features_df: pd.DataFrame,
                     target_column: str,
                     selection_stages: List[int]) -> Dict[str, Any]:
        """Basic feature selection using vectorized operations."""
        try:
            if selection_stages is None:
                selection_stages = [100, 80, 60]
            
            selection_results = {}
            current_features = features_df.copy()
            
            for stage, target_count in enumerate(selection_stages):
                if target_count >= len(current_features.columns):
                    continue
                
                # Calculate correlations using vectorized operations
                correlations = current_features.corr()[target_column].abs()
                correlations = correlations.drop(target_column).sort_values(ascending=False)
                
                # Select top features
                selected_features = correlations.head(target_count).index.tolist()
                selected_features.append(target_column)  # Always include target
                
                # Update current features for next stage
                current_features = current_features[selected_features]
                
                selection_results[f'stage_{stage}'] = {
                    'target_count': target_count,
                    'selected_features': selected_features,
                    'correlation_scores': correlations.head(target_count).to_dict()
                }
            
            return {
                'selection_results': selection_results,
                'final_features': current_features,
                'total_stages_completed': len(selection_results)
            }
            
        except Exception as e:
            tprint(f"⚠️ Basic selection failed: {e}")
            return {'selection_results': {}, 'final_features': features_df}
    
    def _aggregate_selection_results(self, stage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from parallel selection stages."""
        try:
            selection_results = {}
            final_features = None
            
            for result in stage_results:
                if result is not None:
                    stage = result['stage']
                    selection_results[f'stage_{stage}'] = {
                        'target_count': result['target_count'],
                        'selected_features': result['selected_features'],
                        'correlation_scores': result['correlation_scores']
                    }
                    
                    # Use the last stage's features as final features
                    if final_features is None:
                        final_features = result['selected_features']
            
            return {
                'selection_results': selection_results,
                'final_features': final_features,
                'total_stages_completed': len(selection_results)
            }
            
        except Exception as e:
            tprint(f"⚠️ Selection result aggregation failed: {e}")
            return {'selection_results': {}, 'final_features': []}

# Factory function to create optimized engines
def create_optimized_engine(process_type: ProcessType, **kwargs) -> BaseOptimizedProcessEngine:
    """Factory function to create optimized engines."""
    engines = {
        ProcessType.FEATURE_LOOKBACK_OPTIMIZATION: OptimizedFeatureLookbackEngine,
        ProcessType.INTERACTIVE_FEATURE_GENERATION: OptimizedInteractiveFeatureEngine,
        ProcessType.MULTI_HORIZON_PROFIT_LABELER: OptimizedMultiHorizonEngine,
        ProcessType.FINAL_FEATURE_SELECTION: OptimizedFeatureSelectionEngine
    }
    
    engine_class = engines.get(process_type)
    if engine_class is None:
        raise ValueError(f"Unknown process type: {process_type}")
    
    return engine_class(**kwargs)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
