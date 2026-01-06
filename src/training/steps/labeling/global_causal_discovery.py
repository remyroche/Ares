"""
Global Causal Discovery System

Implements global causal discovery optimization:
- Discover causal graph once per family/data batch
- Reuse results for all candidates in the family
- Significant performance improvement (5-10x speedup)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import time
import hashlib
from functools import lru_cache

# Import enhanced causal components
try:
    from .enhanced_causal_discovery import enhanced_causal_discovery
    from .structural_causal_model import StructuralCausalModel
    from .causal_quality_metrics import CausalQualityMetrics
except ImportError as e:
    warnings.warn(f"Could not import enhanced causal components: {e}")

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class GlobalCausalDiscovery:
    """
    Global causal discovery manager for performance optimization.
    
    Discovers causal graphs once per family/data batch and reuses
    results for all candidates, providing 5-10x speedup.
    """
    
    def __init__(self, verbose: bool = True, cache_size_limit: int = 10):
        """
        Initialize Global Causal Discovery.
        
        Args:
            verbose: Whether to print progress information
            cache_size_limit: Maximum number of cached discoveries
        """
        self.verbose = verbose
        self.cache_size_limit = cache_size_limit
        
        # Global discovery cache
        self.discovery_cache_ = {}
        self.cache_access_order_ = []
        
        # Statistics
        self.cache_hits_ = 0
        self.cache_misses_ = 0
        self.total_discoveries_ = 0
        
    def get_data_fingerprint(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Generate a unique fingerprint for the data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            String fingerprint for caching
        """
        # Create a hash based on data shape and key statistics
        data_info = {
            'X_shape': X.shape,
            'y_shape': y.shape,
            'X_columns': list(X.columns),
            'X_dtypes': X.dtypes.to_dict(),
            'X_mean': X.mean().to_dict(),
            'y_mean': float(y.mean()),
            'y_std': float(y.std())
        }
        
        # Create hash
        data_str = str(sorted(data_info.items()))
        fingerprint = hashlib.md5(data_str.encode()).hexdigest()
        
        return fingerprint
    
    def discover_or_reuse(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        target_features: int = 100,
        n_bootstrap: int = 50,
        causal_quality_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Discover causal graph or reuse cached results.
        
        Args:
            X: Feature matrix
            y: Target variable
            target_features: Number of features for MDI pre-pruning
            n_bootstrap: Number of bootstrap samples
            causal_quality_thresholds: Thresholds for quality metrics
            
        Returns:
            Dictionary with discovery results
        """
        # Generate data fingerprint
        fingerprint = self.get_data_fingerprint(X, y)
        
        # Check cache
        if fingerprint in self.discovery_cache_:
            self.cache_hits_ += 1
            if self.verbose:
                tprint_info(f"🎯 Global Discovery: Using cached results (cache hit #{self.cache_hits_})")
            
            # Update access order
            self.cache_access_order_.remove(fingerprint)
            self.cache_access_order_.append(fingerprint)
            
            return self.discovery_cache_[fingerprint]
        
        # Cache miss - run discovery
        self.cache_misses_ += 1
        self.total_discoveries_ += 1
        
        if self.verbose:
            tprint_info(f"🔍 Global Discovery: Running discovery (cache miss #{self.cache_misses_})")
        
        start_time = time.time()
        
        try:
            # Run enhanced causal discovery
            discovery_results = enhanced_causal_discovery(
                X, y,
                target_features=target_features,
                n_bootstrap=n_bootstrap,
                verbose=False
            )
            
            # Fit structural causal models
            if self.verbose:
                tprint_info("   🧠 Fitting structural causal models...")
            
            scm = StructuralCausalModel(verbose=False)
            scm.fit_structural_equations(X, discovery_results.get('consensus_graph', {}))
            
            # Initialize causal quality metrics
            if self.verbose:
                tprint_info("   🔍 Initializing causal quality metrics...")
            
            quality_metrics = CausalQualityMetrics(
                causal_graph=discovery_results.get('consensus_graph', {}),
                scm=scm,
                quality_thresholds=causal_quality_thresholds or {},
                verbose=False
            )
            
            # Combine results
            global_results = {
                'discovery_results': discovery_results,
                'structural_causal_model': scm,
                'causal_quality_metrics': quality_metrics,
                'consensus_graph': discovery_results.get('consensus_graph', {}),
                'fingerprint': fingerprint,
                'discovery_time': time.time() - start_time,
                'cache_status': 'fresh'
            }
            
            # Cache the results
            self._cache_results(fingerprint, global_results)
            
            if self.verbose:
                n_edges = sum(len(parents) for parents in global_results['consensus_graph'].values())
                tprint_success(f"✅ Global Discovery: Complete!")
                tprint_info(f"   📊 Graph: {len(global_results['consensus_graph'])} nodes, {n_edges} edges")
                tprint_info(f"   📊 Time: {global_results['discovery_time']:.2f}s")
                tprint_info(f"   📊 Cache size: {len(self.discovery_cache_)}/{self.cache_size_limit}")
            
            return global_results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Global Discovery failed: {e}")
            
            # Return empty results
            return {
                'discovery_results': {},
                'structural_causal_model': None,
                'causal_quality_metrics': None,
                'consensus_graph': {},
                'fingerprint': fingerprint,
                'discovery_time': time.time() - start_time,
                'cache_status': 'error',
                'error': str(e)
            }
    
    def _cache_results(self, fingerprint: str, results: Dict[str, Any]):
        """Cache discovery results with LRU eviction."""
        # Remove oldest if cache is full
        if len(self.discovery_cache_) >= self.cache_size_limit:
            oldest_fingerprint = self.cache_access_order_.pop(0)
            del self.discovery_cache_[oldest_fingerprint]
            
            if self.verbose:
                tprint_info(f"   🗑️ Evicted oldest cache entry: {oldest_fingerprint[:8]}")
        
        # Add new results
        self.discovery_cache_[fingerprint] = results
        self.cache_access_order_.append(fingerprint)
    
    def assess_candidate_fast(
        self,
        candidate: Any,
        X: pd.DataFrame,
        y: pd.Series,
        global_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Fast assessment using pre-computed global results.
        
        Args:
            candidate: Geometry candidate
            X: Feature matrix
            y: Target variable
            global_results: Pre-computed global discovery results
            
        Returns:
            Assessment results
        """
        try:
            # Use pre-computed causal quality metrics
            quality_metrics = global_results.get('causal_quality_metrics')
            
            if quality_metrics is None:
                return {
                    'final_status': 'FAILED',
                    'survival_status': 'FAILED',
                    'causal_quality_status': 'FAILED',
                    'Layer2Score': 0.0,
                    'error': 'No global quality metrics available'
                }
            
            # Run fast assessment
            assessment = quality_metrics.assess_geometry_causal_quality(candidate, X, y)
            
            # Add global discovery info
            assessment['global_discovery_used'] = True
            assessment['global_fingerprint'] = global_results.get('fingerprint')
            assessment['cache_status'] = global_results.get('cache_status')
            
            return assessment
            
        except Exception as e:
            return {
                'final_status': 'FAILED',
                'survival_status': 'FAILED',
                'causal_quality_status': 'FAILED',
                'Layer2Score': 0.0,
                'error': f'Fast assessment failed: {e}'
            }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits_ + self.cache_misses_
        hit_rate = self.cache_hits_ / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits_,
            'cache_misses': self.cache_misses_,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'total_discoveries': self.total_discoveries_,
            'cache_size': len(self.discovery_cache_),
            'cache_size_limit': self.cache_size_limit,
            'speedup_estimate': 1.0 + (hit_rate * 9.0)  # Estimate 10x speedup for cache hits
        }
    
    def clear_cache(self):
        """Clear the discovery cache."""
        self.discovery_cache_.clear()
        self.cache_access_order_.clear()
        
        if self.verbose:
            tprint_info("🗑️ Global discovery cache cleared")
    
    def preload_cache(
        self,
        families_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        target_features: int = 100,
        n_bootstrap: int = 50
    ):
        """
        Preload cache with discoveries for multiple families.
        
        Args:
            families_data: Dictionary mapping family names to (X, y) tuples
            target_features: Number of features for MDI pre-pruning
            n_bootstrap: Number of bootstrap samples
        """
        if self.verbose:
            tprint_info(f"🚀 Preloading cache with {len(families_data)} families...")
        
        start_time = time.time()
        
        for family_name, (X, y) in families_data.items():
            if self.verbose:
                tprint_info(f"   📊 Preloading {family_name}...")
            
            self.discover_or_reuse(X, y, target_features, n_bootstrap)
        
        preload_time = time.time() - start_time
        
        if self.verbose:
            tprint_success(f"✅ Cache preloading complete ({preload_time:.2f}s)")


# Convenience function for quick usage
def get_global_discovery_manager(verbose: bool = True, cache_size_limit: int = 10) -> GlobalCausalDiscovery:
    """
    Get or create a global discovery manager instance.
    
    Args:
        verbose: Whether to print progress information
        cache_size_limit: Maximum number of cached discoveries
        
    Returns:
        GlobalCausalDiscovery instance
    """
    # Simple singleton pattern for now
    if not hasattr(get_global_discovery_manager, '_instance'):
        get_global_discovery_manager._instance = GlobalCausalDiscovery(
            verbose=verbose, 
            cache_size_limit=cache_size_limit
        )
    
    return get_global_discovery_manager._instance
