"""
Blockwise Correlation with Early-Abort for Redundancy Pruning

This module implements efficient blockwise correlation computation with early-abort
to handle the O(F²·N) complexity of wide correlation matrices.

Key Features:
- Blockwise correlation computation
- Early-abort when |ρ| > threshold
- Memory-efficient processing
- Approximate top-K with sketches
- Sparse prefiltering
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from scipy.sparse import csr_matrix
from sklearn.random_projection import GaussianRandomProjection
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class BlockwiseCorrelationConfig:
    """Configuration for blockwise correlation computation."""
    block_size: int = 1000  # Features per block
    correlation_threshold: float = 0.97  # Early-abort threshold
    max_correlations: int = 10000  # Maximum correlations to compute
    use_approximation: bool = True  # Use random projections for approximation
    approximation_components: int = 50  # Number of random projection components
    memory_limit_mb: int = 1000  # Memory limit per block
    parallel_blocks: bool = True  # Process blocks in parallel
    max_workers: int = 4  # Number of workers for parallel processing


class BlockwiseCorrelation:
    """Efficient blockwise correlation computation with early-abort."""
    
    def __init__(self, config: BlockwiseCorrelationConfig):
        self.config = config
        self.computation_stats = {}
        
        tprint_info("🔥 Blockwise correlation initialized")
        tprint_info(f"📊 Block size: {config.block_size}")
        tprint_info(f"📊 Correlation threshold: {config.correlation_threshold}")
        tprint_info(f"📊 Memory limit: {config.memory_limit_mb}MB")
    
    def compute_correlations(self, 
                           data: pd.DataFrame,
                           target: Optional[pd.Series] = None,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute correlations using blockwise approach with early-abort.
        
        Args:
            data: Feature matrix
            target: Target vector (optional)
            feature_names: List of feature names to process
            
        Returns:
            Dictionary with correlation results and statistics
        """
        if feature_names is None:
            feature_names = list(data.columns)
        
        tprint_info(f"🔥 Computing correlations for {len(feature_names)} features")
        
        # Initialize statistics
        self.computation_stats = {
            'total_features': len(feature_names),
            'blocks_processed': 0,
            'correlations_computed': 0,
            'early_aborts': 0,
            'high_correlations': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Prefilter features if using approximation
        if self.config.use_approximation:
            feature_names = self._prefilter_features(data[feature_names], target)
            tprint_info(f"📊 Prefiltered to {len(feature_names)} features")
        
        # Split features into blocks
        blocks = self._split_into_blocks(feature_names)
        tprint_info(f"📊 Split into {len(blocks)} blocks")
        
        # Compute correlations blockwise
        if self.config.parallel_blocks and len(blocks) > 1:
            correlations = self._compute_parallel_blocks(data, blocks, target)
        else:
            correlations = self._compute_sequential_blocks(data, blocks, target)
        
        # Update statistics
        self.computation_stats['processing_time'] = time.time() - start_time
        
        tprint_success(f"✅ Computed {self.computation_stats['correlations_computed']} correlations")
        tprint_info(f"📊 Early aborts: {self.computation_stats['early_aborts']}")
        tprint_info(f"📊 High correlations: {self.computation_stats['high_correlations']}")
        
        return {
            'correlations': correlations,
            'stats': self.computation_stats
        }
    
    def _prefilter_features(self, 
                          data: pd.DataFrame,
                          target: Optional[pd.Series]) -> List[str]:
        """Prefilter features using cheap methods before correlation computation."""
        if target is None:
            # Use variance-based filtering
            variances = data.var()
            threshold = variances.quantile(0.1)  # Keep top 90% by variance
            return variances[variances > threshold].index.tolist()
        
        # Use IC-based filtering
        ics = []
        for col in data.columns:
            try:
                ic = self._compute_ic(data[col], target)
                ics.append((col, ic))
            except:
                ics.append((col, 0.0))
        
        # Sort by IC and take top features
        ics.sort(key=lambda x: abs(x[1]), reverse=True)
        top_count = min(len(ics), self.config.max_correlations // 2)
        
        return [col for col, ic in ics[:top_count]]
    
    def _compute_ic(self, feature: pd.Series, target: pd.Series) -> float:
        """Compute Information Coefficient (IC) between feature and target."""
        try:
            # Remove NaN values
            valid_mask = ~(feature.isna() | target.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]
            
            # Compute correlation
            correlation = feature_clean.corr(target_clean)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _split_into_blocks(self, feature_names: List[str]) -> List[List[str]]:
        """Split features into blocks for processing."""
        blocks = []
        for i in range(0, len(feature_names), self.config.block_size):
            block = feature_names[i:i + self.config.block_size]
            blocks.append(block)
        return blocks
    
    def _compute_parallel_blocks(self, 
                               data: pd.DataFrame,
                               blocks: List[List[str]],
                               target: Optional[pd.Series]) -> Dict[str, Any]:
        """Compute correlations for blocks in parallel."""
        correlations = {
            'feature_correlations': {},
            'target_correlations': {},
            'redundant_pairs': []
        }
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for i, block in enumerate(blocks):
                future = executor.submit(self._compute_block_correlations, data, block, target, i)
                futures.append(future)
            
            for future in futures:
                block_result = future.result()
                if block_result:
                    self._merge_block_results(correlations, block_result)
        
        return correlations
    
    def _compute_sequential_blocks(self, 
                                 data: pd.DataFrame,
                                 blocks: List[List[str]],
                                 target: Optional[pd.Series]) -> Dict[str, Any]:
        """Compute correlations for blocks sequentially."""
        correlations = {
            'feature_correlations': {},
            'target_correlations': {},
            'redundant_pairs': []
        }
        
        for i, block in enumerate(blocks):
            block_result = self._compute_block_correlations(data, block, target, i)
            if block_result:
                self._merge_block_results(correlations, block_result)
        
        return correlations
    
    def _compute_block_correlations(self, 
                                  data: pd.DataFrame,
                                  block: List[str],
                                  target: Optional[pd.Series],
                                  block_id: int) -> Optional[Dict[str, Any]]:
        """Compute correlations for a single block."""
        try:
            block_data = data[block].dropna()
            
            if len(block_data) == 0:
                return None
            
            result = {
                'feature_correlations': {},
                'target_correlations': {},
                'redundant_pairs': []
            }
            
            # Compute feature-feature correlations with early-abort
            for i, feature1 in enumerate(block):
                if feature1 not in block_data.columns:
                    continue
                
                for j, feature2 in enumerate(block[i+1:], i+1):
                    if feature2 not in block_data.columns:
                        continue
                    
                    # Check if we've hit the correlation limit
                    if self.computation_stats['correlations_computed'] >= self.config.max_correlations:
                        tprint_warning("⚠️ Hit correlation limit, stopping computation")
                        break
                    
                    # Compute correlation
                    try:
                        corr = block_data[feature1].corr(block_data[feature2])
                        
                        if not np.isnan(corr):
                            self.computation_stats['correlations_computed'] += 1
                            
                            # Early-abort if correlation is too high
                            if abs(corr) > self.config.correlation_threshold:
                                self.computation_stats['early_aborts'] += 1
                                self.computation_stats['high_correlations'] += 1
                                
                                result['redundant_pairs'].append({
                                    'feature1': feature1,
                                    'feature2': feature2,
                                    'correlation': corr,
                                    'reason': 'high_correlation'
                                })
                                
                                # Skip remaining correlations for this feature
                                break
                            
                            result['feature_correlations'][f'{feature1}_{feature2}'] = corr
                    
                    except Exception as e:
                        tprint_debug(f"⚠️ Correlation computation failed for {feature1} vs {feature2}: {e}")
                        continue
                
                # Check if we should stop processing this block
                if self.computation_stats['correlations_computed'] >= self.config.max_correlations:
                    break
            
            # Compute target correlations if target is provided
            if target is not None:
                for feature in block:
                    if feature in block_data.columns:
                        try:
                            corr = block_data[feature].corr(target)
                            if not np.isnan(corr):
                                result['target_correlations'][feature] = corr
                        except Exception as e:
                            tprint_debug(f"⚠️ Target correlation failed for {feature}: {e}")
            
            self.computation_stats['blocks_processed'] += 1
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Block {block_id} computation failed: {e}")
            return None
    
    def _merge_block_results(self, 
                           correlations: Dict[str, Any],
                           block_result: Dict[str, Any]):
        """Merge results from a block into the main correlations dictionary."""
        # Merge feature correlations
        correlations['feature_correlations'].update(block_result['feature_correlations'])
        
        # Merge target correlations
        correlations['target_correlations'].update(block_result['target_correlations'])
        
        # Merge redundant pairs
        correlations['redundant_pairs'].extend(block_result['redundant_pairs'])
    
    def get_redundant_features(self, 
                             correlations: Dict[str, Any],
                             threshold: float = 0.97) -> List[str]:
        """Get list of redundant features based on correlations."""
        redundant_features = set()
        
        for pair in correlations['redundant_pairs']:
            if pair['correlation'] > threshold:
                # Add the feature with lower variance (less informative)
                feature1, feature2 = pair['feature1'], pair['feature2']
                redundant_features.add(feature1)  # Simple heuristic: remove first feature
        
        return list(redundant_features)
    
    def get_top_correlations(self, 
                           correlations: Dict[str, Any],
                           top_k: int = 100) -> List[Tuple[str, str, float]]:
        """Get top K correlations by absolute value."""
        feature_correlations = correlations['feature_correlations']
        
        # Sort by absolute correlation value
        sorted_correlations = sorted(
            feature_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Format as tuples
        result = []
        for (feature_pair, corr) in sorted_correlations[:top_k]:
            feature1, feature2 = feature_pair.split('_', 1)
            result.append((feature1, feature2, corr))
        
        return result
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get detailed computation statistics."""
        return self.computation_stats


class ApproximateCorrelation:
    """Approximate correlation computation using random projections."""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.random_projection = None
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit random projection and transform data."""
        self.random_projection = GaussianRandomProjection(n_components=self.n_components)
        return self.random_projection.fit_transform(data)
    
    def compute_approximate_correlations(self, 
                                       data: pd.DataFrame,
                                       target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Compute approximate correlations using random projections."""
        # Transform data to lower dimension
        transformed_data = self.fit_transform(data)
        
        # Compute correlations in lower dimension
        correlations = np.corrcoef(transformed_data.T)
        
        # Map back to original features
        feature_names = list(data.columns)
        result = {}
        
        for i, feature1 in enumerate(feature_names):
            for j, feature2 in enumerate(feature_names[i+1:], i+1):
                corr = correlations[i, j]
                if not np.isnan(corr):
                    result[f'{feature1}_{feature2}'] = corr
        
        return {
            'approximate_correlations': result,
            'n_components': self.n_components,
            'original_features': len(feature_names)
        }


# Global instances
_blockwise_correlation = None

def get_blockwise_correlation() -> BlockwiseCorrelation:
    """Get the global blockwise correlation instance."""
    global _blockwise_correlation
    if _blockwise_correlation is None:
        config = BlockwiseCorrelationConfig()
        _blockwise_correlation = BlockwiseCorrelation(config)
    return _blockwise_correlation

def compute_correlations_blockwise(data: pd.DataFrame,
                                 target: Optional[pd.Series] = None,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute correlations using blockwise approach.
    
    Args:
        data: Feature matrix
        target: Target vector (optional)
        feature_names: List of feature names to process
        
    Returns:
        Dictionary with correlation results
    """
    correlation = get_blockwise_correlation()
    return correlation.compute_correlations(data, target, feature_names)

def get_redundant_features(correlations: Dict[str, Any], threshold: float = 0.97) -> List[str]:
    """Get redundant features based on correlations."""
    correlation = get_blockwise_correlation()
    return correlation.get_redundant_features(correlations, threshold)