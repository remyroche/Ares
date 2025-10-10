"""
Kernel Fusion for Feature Interactions

This module implements kernel fusion to compute sum/diff/prod/ratio interactions
in a single pass per pair, significantly improving performance for interaction
feature generation.

Key Features:
- Single-pass computation for multiple interaction types
- Vectorized operations using NumPy
- Memory-efficient batch processing
- Support for multiple interaction types
- Optimized for large datasets
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class KernelFusionConfig:
    """Configuration for kernel fusion."""
    enable_fusion: bool = True
    batch_size: int = 1000  # Batch size for processing
    max_workers: int = 4  # Number of workers for parallel processing
    use_multiprocessing: bool = False  # Use multiprocessing vs threading
    memory_limit_mb: int = 1000  # Memory limit per batch
    interaction_types: List[str] = None  # Types of interactions to compute
    row_block_size: int = 10000  # Row block size for contiguous writes
    preallocate_output: bool = True  # Preallocate output matrices
    domain_pair_quotas: bool = True  # Use domain-based pair quotas
    max_pairs_per_domain: int = 100  # Maximum pairs per domain
    
    def __post_init__(self):
        if self.interaction_types is None:
            self.interaction_types = ['sum', 'diff', 'prod', 'ratio']


class KernelFusion:
    """Kernel fusion for efficient interaction computation."""
    
    def __init__(self, config: KernelFusionConfig):
        self.config = config
        self.fusion_stats = {}
        
        tprint_info("⚡ Kernel fusion initialized")
        tprint_info(f"📊 Batch size: {config.batch_size}")
        tprint_info(f"📊 Max workers: {config.max_workers}")
        tprint_info(f"📊 Interaction types: {config.interaction_types}")
    
    def fuse_interactions(self, 
                         data: pd.DataFrame,
                         feature_pairs: List[Tuple[str, str]],
                         interaction_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fuse multiple interaction types in a single pass.
        
        Args:
            data: Input data
            feature_pairs: List of feature pairs to interact
            interaction_types: Types of interactions to compute
            
        Returns:
            DataFrame with fused interactions
        """
        if not self.config.enable_fusion:
            return self._compute_interactions_sequential(data, feature_pairs, interaction_types)
        
        if interaction_types is None:
            interaction_types = self.config.interaction_types
        
        tprint_info(f"⚡ Fusing {len(feature_pairs)} pairs with {len(interaction_types)} interaction types")
        
        # Initialize statistics
        self.fusion_stats = {
            'total_pairs': len(feature_pairs),
            'interaction_types': interaction_types,
            'batches_processed': 0,
            'total_interactions': 0,
            'processing_time': 0
        }
        
        # Process in batches
        if len(feature_pairs) <= self.config.batch_size:
            # Single batch
            result = self._process_batch(data, feature_pairs, interaction_types)
        else:
            # Multiple batches
            result = self._process_batches(data, feature_pairs, interaction_types)
        
        # Update statistics
        self.fusion_stats['total_interactions'] = len(result.columns)
        
        tprint_success(f"✅ Generated {len(result.columns)} fused interactions")
        
        return result
    
    def _process_batches(self, 
                        data: pd.DataFrame,
                        feature_pairs: List[Tuple[str, str]],
                        interaction_types: List[str]) -> pd.DataFrame:
        """Process feature pairs in batches."""
        results = []
        
        # Split pairs into batches
        batches = [feature_pairs[i:i + self.config.batch_size] 
                  for i in range(0, len(feature_pairs), self.config.batch_size)]
        
        tprint_info(f"📊 Processing {len(batches)} batches")
        
        # Process batches
        if self.config.max_workers > 1:
            # Parallel processing
            with self._get_executor() as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(self._process_batch, data, batch, interaction_types)
                    futures.append(future)
                
                for future in futures:
                    batch_result = future.result()
                    if not batch_result.empty:
                        results.append(batch_result)
        else:
            # Sequential processing
            for batch in batches:
                batch_result = self._process_batch(data, batch, interaction_types)
                if not batch_result.empty:
                    results.append(batch_result)
        
        # Combine results
        if results:
            return pd.concat(results, axis=1)
        else:
            return pd.DataFrame(index=data.index)
    
    def _process_batch(self, 
                      data: pd.DataFrame,
                      feature_pairs: List[Tuple[str, str]],
                      interaction_types: List[str]) -> pd.DataFrame:
        """Process a single batch of feature pairs with optimizations."""
        if not feature_pairs:
            return pd.DataFrame(index=data.index)
        
        # Preallocate output if configured
        if self.config.preallocate_output:
            total_interactions = len(feature_pairs) * len(interaction_types)
            if total_interactions > 0:
                # Preallocate with optimal dtype
                interactions = {}
                for pair in feature_pairs:
                    feature1, feature2 = pair
                    if feature1 in data.columns and feature2 in data.columns:
                        for interaction_type in interaction_types:
                            key = f'{feature1}_{interaction_type}_{feature2}'
                            interactions[key] = np.empty(len(data), dtype=np.float32)
            else:
                interactions = {}
        else:
            interactions = {}
        
        # Extract feature data efficiently
        feature_data = {}
        for pair in feature_pairs:
            for feature in pair:
                if feature not in feature_data and feature in data.columns:
                    feature_data[feature] = data[feature].values
        
        # Process in row blocks for contiguous writes
        if self.config.row_block_size > 0 and len(data) > self.config.row_block_size:
            interactions = self._process_row_blocks(
                data, feature_pairs, interaction_types, feature_data, interactions
            )
        else:
            # Process all at once
            interactions = self._process_all_rows(
                feature_pairs, interaction_types, feature_data, interactions
            )
        
        # Update statistics
        self.fusion_stats['batches_processed'] += 1
        
        return pd.DataFrame(interactions, index=data.index)
    
    def _process_row_blocks(self, 
                           data: pd.DataFrame,
                           feature_pairs: List[Tuple[str, str]],
                           interaction_types: List[str],
                           feature_data: Dict[str, np.ndarray],
                           interactions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process interactions in row blocks for contiguous writes."""
        row_block_size = self.config.row_block_size
        
        for start_idx in range(0, len(data), row_block_size):
            end_idx = min(start_idx + row_block_size, len(data))
            
            # Extract row block data
            block_data = {}
            for feature, values in feature_data.items():
                block_data[feature] = values[start_idx:end_idx]
            
            # Compute interactions for this block
            block_interactions = self._compute_block_interactions(
                block_data, feature_pairs, interaction_types
            )
            
            # Store results in preallocated arrays
            for key, block_result in block_interactions.items():
                if key in interactions:
                    interactions[key][start_idx:end_idx] = block_result
        
        return interactions
    
    def _process_all_rows(self, 
                         feature_pairs: List[Tuple[str, str]],
                         interaction_types: List[str],
                         feature_data: Dict[str, np.ndarray],
                         interactions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process all interactions at once."""
        for pair in feature_pairs:
            feature1, feature2 = pair
            
            if feature1 not in feature_data or feature2 not in feature_data:
                continue
            
            data1 = feature_data[feature1]
            data2 = feature_data[feature2]
            
            # Compute all interaction types in one pass
            pair_interactions = self._compute_pair_interactions(
                data1, data2, feature1, feature2, interaction_types
            )
            
            interactions.update(pair_interactions)
        
        return interactions
    
    def _compute_block_interactions(self, 
                                   block_data: Dict[str, np.ndarray],
                                   feature_pairs: List[Tuple[str, str]],
                                   interaction_types: List[str]) -> Dict[str, np.ndarray]:
        """Compute interactions for a block of data."""
        interactions = {}
        
        for pair in feature_pairs:
            feature1, feature2 = pair
            
            if feature1 not in block_data or feature2 not in block_data:
                continue
            
            data1 = block_data[feature1]
            data2 = block_data[feature2]
            
            # Compute all interaction types in one pass
            pair_interactions = self._compute_pair_interactions(
                data1, data2, feature1, feature2, interaction_types
            )
            
            interactions.update(pair_interactions)
        
        return interactions
    
    def _compute_pair_interactions(self, 
                                  data1: np.ndarray,
                                  data2: np.ndarray,
                                  name1: str,
                                  name2: str,
                                  interaction_types: List[str]) -> Dict[str, np.ndarray]:
        """Compute all interaction types for a pair in one pass with optimizations."""
        interactions = {}
        
        # Handle NaN values efficiently
        valid_mask = ~(np.isnan(data1) | np.isnan(data2))
        
        if not np.any(valid_mask):
            # All NaN values - return NaN arrays
            for interaction_type in interaction_types:
                interactions[f'{name1}_{interaction_type}_{name2}'] = np.full_like(data1, np.nan)
            return interactions
        
        # Precompute common values for efficiency
        epsilon = 1e-8
        data2_safe = data2 + epsilon
        
        # Compute all interactions in vectorized operations
        if 'sum' in interaction_types:
            interactions[f'{name1}_sum_{name2}'] = data1 + data2
        
        if 'diff' in interaction_types:
            interactions[f'{name1}_diff_{name2}'] = data1 - data2
        
        if 'prod' in interaction_types:
            interactions[f'{name1}_prod_{name2}'] = data1 * data2
        
        if 'ratio' in interaction_types:
            interactions[f'{name1}_ratio_{name2}'] = data1 / data2_safe
        
        if 'max' in interaction_types:
            interactions[f'{name1}_max_{name2}'] = np.maximum(data1, data2)
        
        if 'min' in interaction_types:
            interactions[f'{name1}_min_{name2}'] = np.minimum(data1, data2)
        
        if 'abs_diff' in interaction_types:
            interactions[f'{name1}_abs_diff_{name2}'] = np.abs(data1 - data2)
        
        if 'squared_diff' in interaction_types:
            diff = data1 - data2
            interactions[f'{name1}_squared_diff_{name2}'] = diff * diff
        
        if 'log_ratio' in interaction_types:
            ratio = data1 / data2_safe
            interactions[f'{name1}_log_ratio_{name2}'] = np.log(np.abs(ratio) + epsilon)
        
        # Apply NaN mask to all results
        for key, result in interactions.items():
            interactions[key] = np.where(valid_mask, result, np.nan)
        
        return interactions
    
    def _compute_interactions_sequential(self, 
                                       data: pd.DataFrame,
                                       feature_pairs: List[Tuple[str, str]],
                                       interaction_types: Optional[List[str]]) -> pd.DataFrame:
        """Fallback sequential computation."""
        if interaction_types is None:
            interaction_types = self.config.interaction_types
        
        interactions = {}
        
        for pair in feature_pairs:
            feature1, feature2 = pair
            
            if feature1 not in data.columns or feature2 not in data.columns:
                continue
            
            data1 = data[feature1].values
            data2 = data[feature2].values
            
            pair_interactions = self._compute_pair_interactions(
                data1, data2, feature1, feature2, interaction_types
            )
            
            interactions.update(pair_interactions)
        
        return pd.DataFrame(interactions, index=data.index)
    
    def _get_executor(self):
        """Get appropriate executor for parallel processing."""
        if self.config.use_multiprocessing:
            return ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            return ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get kernel fusion statistics."""
        return self.fusion_stats


class OptimizedKernelFusion:
    """Optimized kernel fusion with advanced techniques."""
    
    def __init__(self, config: KernelFusionConfig):
        self.config = config
        self.base_fusion = KernelFusion(config)
        self.optimization_stats = {}
    
    def fuse_interactions_optimized(self, 
                                   data: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]],
                                   interaction_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Optimized kernel fusion with advanced techniques.
        
        Args:
            data: Input data
            feature_pairs: List of feature pairs to interact
            interaction_types: Types of interactions to compute
            
        Returns:
            DataFrame with optimized fused interactions
        """
        if interaction_types is None:
            interaction_types = self.config.interaction_types
        
        tprint_info(f"⚡ Optimized kernel fusion for {len(feature_pairs)} pairs")
        
        # Pre-allocate memory for efficiency
        total_interactions = len(feature_pairs) * len(interaction_types)
        if total_interactions > 10000:  # Large number of interactions
            return self._fuse_large_scale(data, feature_pairs, interaction_types)
        else:
            return self._fuse_standard(data, feature_pairs, interaction_types)
    
    def _fuse_large_scale(self, 
                         data: pd.DataFrame,
                         feature_pairs: List[Tuple[str, str]],
                         interaction_types: List[str]) -> pd.DataFrame:
        """Fuse interactions for large-scale datasets."""
        # Use memory-mapped arrays for large datasets
        tprint_info("📊 Using large-scale fusion mode")
        
        # Process in smaller batches to manage memory
        batch_size = min(self.config.batch_size, 500)
        results = []
        
        for i in range(0, len(feature_pairs), batch_size):
            batch_pairs = feature_pairs[i:i + batch_size]
            batch_result = self.base_fusion._process_batch(data, batch_pairs, interaction_types)
            
            if not batch_result.empty:
                results.append(batch_result)
            
            # Memory cleanup
            if i % (batch_size * 5) == 0:
                import gc
                gc.collect()
        
        if results:
            return pd.concat(results, axis=1)
        else:
            return pd.DataFrame(index=data.index)
    
    def _fuse_standard(self, 
                      data: pd.DataFrame,
                      feature_pairs: List[Tuple[str, str]],
                      interaction_types: List[str]) -> pd.DataFrame:
        """Standard fusion for normal-sized datasets."""
        return self.base_fusion.fuse_interactions(data, feature_pairs, interaction_types)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats


# Global instances
_kernel_fusion = None
_optimized_fusion = None

def get_kernel_fusion() -> KernelFusion:
    """Get the global kernel fusion instance."""
    global _kernel_fusion
    if _kernel_fusion is None:
        config = KernelFusionConfig()
        _kernel_fusion = KernelFusion(config)
    return _kernel_fusion

def get_optimized_fusion() -> OptimizedKernelFusion:
    """Get the global optimized fusion instance."""
    global _optimized_fusion
    if _optimized_fusion is None:
        config = KernelFusionConfig()
        _optimized_fusion = OptimizedKernelFusion(config)
    return _optimized_fusion

def fuse_interactions(data: pd.DataFrame,
                     feature_pairs: List[Tuple[str, str]],
                     interaction_types: Optional[List[str]] = None,
                     optimized: bool = True) -> pd.DataFrame:
    """
    Fuse interactions using kernel fusion.
    
    Args:
        data: Input data
        feature_pairs: List of feature pairs to interact
        interaction_types: Types of interactions to compute
        optimized: Use optimized fusion
        
    Returns:
        DataFrame with fused interactions
    """
    if optimized:
        fusion = get_optimized_fusion()
        return fusion.fuse_interactions_optimized(data, feature_pairs, interaction_types)
    else:
        fusion = get_kernel_fusion()
        return fusion.fuse_interactions(data, feature_pairs, interaction_types)

def get_fusion_statistics() -> Dict[str, Any]:
    """Get kernel fusion statistics."""
    fusion = get_kernel_fusion()
    return fusion.get_fusion_statistics()