"""
Optimized SHAP Computation Engine

Implements incremental SHAP computation, sampling approximation, early stopping,
and M1-optimized SHAP calculations for feature interaction generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from src.utils.tprint import tprint

# SHAP and LightGBM imports
try:
    import lightgbm as lgb
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    lgb = None
    shap = None
    warnings.warn("SHAP/LightGBM not available for optimized computation")

# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class SHAPConfig:
    """Configuration for optimized SHAP computation."""
    
    # Incremental computation
    enable_incremental: bool = True
    batch_size: int = 10
    early_stopping_threshold: float = 0.001
    
    # Sampling approximation
    enable_sampling: bool = True
    max_samples: int = 1000
    sampling_strategy: str = "random"  # "random", "stratified", "importance"
    
    # Performance optimization
    enable_gpu_acceleration: bool = True
    enable_parallel_computation: bool = True
    max_workers: int = 4
    
    # Memory management
    enable_memory_optimization: bool = True
    chunk_size: int = 1000
    gc_frequency: int = 50

class IncrementalSHAPComputer:
    """Incremental SHAP computation with early stopping."""
    
    def __init__(self, config: SHAPConfig):
        self.config = config
        self.logger = logger.getChild('IncrementalSHAPComputer')
        
        # State tracking
        self.computed_shap_values = []
        self.feature_importance_history = []
        self.computation_time = 0.0
        
        # Early stopping
        self.convergence_threshold = config.early_stopping_threshold
        self.convergence_window = 5  # Check convergence over last N batches
        
    def compute_incremental_shap(self, model: Any, X: np.ndarray, 
                               feature_names: List[str],
                               max_features: Optional[int] = None) -> np.ndarray:
        """Compute SHAP values incrementally with early stopping."""
        if not SHAP_AVAILABLE:
            tprint("❌ [SHAP] SHAP not available for incremental computation")
            return np.array([])
        
        tprint(f"🔄 [SHAP] Starting incremental SHAP computation for {X.shape[1]} features")
        
        if max_features is None:
            max_features = X.shape[1]
        
        max_features = min(max_features, X.shape[1])
        
        explainer = shap.TreeExplainer(model)
        shap_values = []
        
        start_time = time.time()
        
        for i in range(0, max_features, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, max_features)
            batch_features = list(range(i, batch_end))
            
            tprint(f"📊 [SHAP] Computing batch {i//self.config.batch_size + 1}: features {i}-{batch_end-1}")
            
            # Compute SHAP for batch
            batch_X = X[:, batch_features]
            batch_shap = explainer.shap_values(batch_X)
            
            # Handle multi-output case
            if isinstance(batch_shap, list):
                batch_shap = batch_shap[0]  # Take first output
            
            shap_values.append(batch_shap)
            
            # Early stopping check
            if self._should_early_stop(batch_shap):
                tprint(f"🛑 [SHAP] Early stopping triggered at feature {batch_end-1}")
                break
        
        # Combine results
        if shap_values:
            combined_shap = np.concatenate(shap_values, axis=1)
            self.computation_time = time.time() - start_time
            
            tprint(f"✅ [SHAP] Incremental computation completed in {self.computation_time:.2f}s")
            tprint(f"📊 [SHAP] Final SHAP matrix shape: {combined_shap.shape}")
            
            return combined_shap
        
        return np.array([])
    
    def _should_early_stop(self, batch_shap: np.ndarray) -> bool:
        """Check if early stopping should be triggered."""
        if len(self.computed_shap_values) < self.convergence_window:
            self.computed_shap_values.append(batch_shap)
            return False
        
        # Update history
        self.computed_shap_values.append(batch_shap)
        if len(self.computed_shap_values) > self.convergence_window:
            self.computed_shap_values.pop(0)
        
        # Check convergence
        recent_values = self.computed_shap_values[-self.convergence_window:]
        mean_abs_values = [np.mean(np.abs(values)) for values in recent_values]
        
        # Check if values are becoming negligible
        if len(mean_abs_values) >= 2:
            recent_change = abs(mean_abs_values[-1] - mean_abs_values[-2])
            if recent_change < self.convergence_threshold:
                return True
        
        return False

class SamplingSHAPComputer:
    """SHAP computation with sampling approximation."""
    
    def __init__(self, config: SHAPConfig):
        self.config = config
        self.logger = logger.getChild('SamplingSHAPComputer')
        
    def compute_sampling_shap(self, model: Any, X: np.ndarray, 
                            feature_names: List[str],
                            y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values using sampling approximation."""
        if not SHAP_AVAILABLE:
            tprint("❌ [SHAP] SHAP not available for sampling computation")
            return np.array([]), np.array([])
        
        original_size = len(X)
        
        if original_size <= self.config.max_samples:
            tprint(f"📊 [SHAP] Dataset size ({original_size}) <= max_samples ({self.config.max_samples}), using full dataset")
            return self._compute_full_shap(model, X, feature_names)
        
        tprint(f"🔄 [SHAP] Sampling dataset from {original_size} to {self.config.max_samples} samples")
        
        # Select samples based on strategy
        if self.config.sampling_strategy == "random":
            indices = self._random_sampling(X)
        elif self.config.sampling_strategy == "stratified":
            indices = self._stratified_sampling(X, y)
        elif self.config.sampling_strategy == "importance":
            indices = self._importance_sampling(model, X, y)
        else:
            indices = self._random_sampling(X)
        
        # Sample data
        X_sample = X[indices]
        y_sample = y[indices] if y is not None else None
        
        tprint(f"✅ [SHAP] Selected {len(indices)} samples using {self.config.sampling_strategy} strategy")
        
        # Compute SHAP on sample
        return self._compute_full_shap(model, X_sample, feature_names)
    
    def _random_sampling(self, X: np.ndarray) -> np.ndarray:
        """Random sampling strategy."""
        n_samples = min(self.config.max_samples, len(X))
        return np.random.choice(len(X), n_samples, replace=False)
    
    def _stratified_sampling(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Stratified sampling strategy."""
        if y is None:
            return self._random_sampling(X)
        
        # Simple stratified sampling by quantiles
        n_quantiles = 5
        quantiles = np.quantile(y, np.linspace(0, 1, n_quantiles + 1))
        
        indices = []
        samples_per_quantile = self.config.max_samples // n_quantiles
        
        for i in range(n_quantiles):
            mask = (y >= quantiles[i]) & (y < quantiles[i + 1])
            quantile_indices = np.where(mask)[0]
            
            if len(quantile_indices) > 0:
                n_samples = min(samples_per_quantile, len(quantile_indices))
                selected = np.random.choice(quantile_indices, n_samples, replace=False)
                indices.extend(selected)
        
        return np.array(indices)
    
    def _importance_sampling(self, model: Any, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Importance-based sampling strategy."""
        if y is None:
            return self._random_sampling(X)
        
        # Get feature importance from model
        try:
            if hasattr(model, 'feature_importance'):
                importance = model.feature_importance(importance_type='gain')
            else:
                # Fallback to variance
                importance = np.var(X, axis=0)
            
            # Weight samples by importance
            sample_weights = np.mean(X * importance, axis=1)
            
            # Sample based on weights
            n_samples = min(self.config.max_samples, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False, p=sample_weights/sample_weights.sum())
            
            return indices
            
        except Exception as e:
            tprint(f"⚠️ [SHAP] Importance sampling failed: {e}, falling back to random sampling")
            return self._random_sampling(X)
    
    def _compute_full_shap(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute full SHAP values."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first output
        
        return shap_values, X

class GPUAcceleratedSHAPComputer:
    """GPU-accelerated SHAP computation using PyTorch."""
    
    def __init__(self, config: SHAPConfig):
        self.config = config
        self.logger = logger.getChild('GPUAcceleratedSHAPComputer')
        
        # GPU availability
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            # Check for MPS (Metal Performance Shaders) on M1
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                tprint("✅ [SHAP] M1 GPU (MPS) available for SHAP computation")
                return True
            
            # Check for CUDA
            if torch.cuda.is_available():
                tprint("✅ [SHAP] CUDA GPU available for SHAP computation")
                return True
            
            return False
            
        except Exception as e:
            tprint(f"⚠️ [SHAP] GPU check failed: {e}")
            return False
    
    def compute_gpu_accelerated_shap(self, model: Any, X: np.ndarray, 
                                   feature_names: List[str]) -> np.ndarray:
        """Compute SHAP values with GPU acceleration."""
        if not self.gpu_available or not self.config.enable_gpu_acceleration:
            tprint("⚠️ [SHAP] GPU acceleration not available, using CPU")
            return self._compute_cpu_shap(model, X, feature_names)
        
        try:
            tprint("🚀 [SHAP] GPU-accelerated SHAP computation")
            
            # Convert to PyTorch tensor
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'
            
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            
            # For now, fall back to CPU SHAP computation
            # GPU-accelerated SHAP would require custom implementation
            tprint("⚠️ [SHAP] GPU-accelerated SHAP not yet implemented, using CPU")
            return self._compute_cpu_shap(model, X, feature_names)
            
        except Exception as e:
            tprint(f"❌ [SHAP] GPU-accelerated SHAP failed: {e}")
            return self._compute_cpu_shap(model, X, feature_names)
    
    def _compute_cpu_shap(self, model: Any, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fallback CPU SHAP computation."""
        if not SHAP_AVAILABLE:
            return np.array([])
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values

class OptimizedSHAPComputer:
    """Main optimized SHAP computer combining all optimization techniques."""
    
    def __init__(self, config: Optional[SHAPConfig] = None):
        self.config = config or SHAPConfig()
        self.logger = logger.getChild('OptimizedSHAPComputer')
        
        # Initialize components
        self.incremental_computer = IncrementalSHAPComputer(self.config)
        self.sampling_computer = SamplingSHAPComputer(self.config)
        self.gpu_computer = GPUAcceleratedSHAPComputer(self.config)
        
        # Performance tracking
        self.computation_stats = {
            'total_computations': 0,
            'total_time': 0.0,
            'gpu_accelerations': 0,
            'early_stops': 0,
            'sampling_approximations': 0
        }
        
        tprint("🚀 [SHAP] Optimized SHAP Computer initialized")
    
    def compute_optimized_shap(self, model: Any, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             computation_mode: str = "adaptive") -> Dict[str, Any]:
        """Compute SHAP values using optimized methods."""
        if not SHAP_AVAILABLE:
            tprint("❌ [SHAP] SHAP not available")
            return {'error': 'SHAP not available'}
        
        tprint(f"🔄 [SHAP] Optimized SHAP computation: {X.shape}, mode: {computation_mode}")
        
        start_time = time.time()
        self.computation_stats['total_computations'] += 1
        
        try:
            # Determine computation strategy
            if computation_mode == "adaptive":
                strategy = self._determine_optimal_strategy(X, y, feature_names)
            else:
                strategy = computation_mode
            
            tprint(f"📊 [SHAP] Selected strategy: {strategy}")
            
            # Execute computation
            if strategy == "incremental":
                shap_values = self.incremental_computer.compute_incremental_shap(
                    model, X, feature_names
                )
            elif strategy == "sampling":
                shap_values, X_sample = self.sampling_computer.compute_sampling_shap(
                    model, X, feature_names, y
                )
                self.computation_stats['sampling_approximations'] += 1
            elif strategy == "gpu":
                shap_values = self.gpu_computer.compute_gpu_accelerated_shap(
                    model, X, feature_names
                )
                self.computation_stats['gpu_accelerations'] += 1
            else:  # Default to full computation
                shap_values = self._compute_full_shap(model, X, feature_names)
            
            computation_time = time.time() - start_time
            self.computation_stats['total_time'] += computation_time
            
            # Calculate interaction centrality
            interaction_centrality = self._calculate_interaction_centrality(shap_values)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(shap_values)
            
            result = {
                'shap_values': shap_values,
                'interaction_centrality': interaction_centrality,
                'stability_metrics': stability_metrics,
                'computation_time': computation_time,
                'strategy_used': strategy,
                'feature_names': feature_names
            }
            
            tprint(f"✅ [SHAP] Optimized computation completed in {computation_time:.2f}s")
            return result
            
        except Exception as e:
            tprint(f"❌ [SHAP] Optimized computation failed: {e}")
            return {'error': str(e)}
    
    def _determine_optimal_strategy(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str]) -> str:
        """Determine optimal computation strategy based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Large dataset -> use sampling
        if n_samples > self.config.max_samples:
            return "sampling"
        
        # Many features -> use incremental
        if n_features > 50:
            return "incremental"
        
        # GPU available and small dataset -> use GPU
        if self.gpu_computer.gpu_available and n_samples < 5000:
            return "gpu"
        
        # Default to full computation
        return "full"
    
    def _compute_full_shap(self, model: Any, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Compute full SHAP values."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values
    
    def _calculate_interaction_centrality(self, shap_values: np.ndarray) -> np.ndarray:
        """Calculate interaction centrality for each feature."""
        if shap_values.size == 0:
            return np.array([])
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # For interaction centrality, we could compute pairwise interactions
        # For now, return mean absolute values as a proxy
        return mean_abs_shap
    
    def _calculate_stability_metrics(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate stability metrics for SHAP values."""
        if shap_values.size == 0:
            return {}
        
        # Coefficient of variation across features
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        std_shap = np.std(np.abs(shap_values), axis=0)
        cv = np.mean(std_shap / (mean_shap + 1e-8))
        
        return {
            'coefficient_of_variation': cv,
            'mean_absolute_shap': np.mean(mean_shap),
            'std_absolute_shap': np.mean(std_shap)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.computation_stats.copy()
        
        if stats['total_computations'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_computations']
        else:
            stats['average_time'] = 0.0
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        tprint("🧹 [SHAP] Cleaning up SHAP computer")
        
        # Clear computation history
        self.incremental_computer.computed_shap_values.clear()
        self.incremental_computer.feature_importance_history.clear()
        
        # Clear GPU cache
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        tprint("✅ [SHAP] SHAP computer cleanup completed")
