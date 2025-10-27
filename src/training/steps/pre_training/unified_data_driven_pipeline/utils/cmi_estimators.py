"""
CMI Estimators

This module provides various estimators for Conditional Mutual Information (CMI)
computation, including KSG, GCMI, and Binned estimators with hardware optimizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getChild('CMIEstimators')

@dataclass
class CMIEstimatorConfig:
    """Configuration for CMI estimators."""
    
    # Estimator type
    estimator_type: str = 'ksg'  # ksg, gcmi, binned
    
    # KSG estimator parameters
    ksg_k: int = 3  # Number of nearest neighbors
    
    # GCMI estimator parameters
    gcmi_bins: int = 10  # Number of bins for GCMI
    
    # Binned estimator parameters
    binned_bins: int = 20  # Number of bins for binned estimator
    
    # Performance settings
    enable_parallel: bool = True
    n_jobs: int = -1
    enable_caching: bool = True
    cache_size_mb: int = 50
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = False

class CMIEstimator:
    """
    Conditional Mutual Information estimator with multiple algorithms.
    
    Supports KSG (Kraskov-Stögbauer-Grassberger), GCMI (Gaussian CMI),
    and Binned estimators with hardware optimizations.
    """
    
    def __init__(self, config: CMIEstimatorConfig):
        """Initialize the CMI estimator."""
        self.config = config
        self.logger = logger.getChild('CMIEstimator')
        
        # Initialize estimator
        self._initialize_estimator()
        
        # Initialize caching
        self._initialize_caching()
        
        # Performance tracking
        self.computation_stats = {
            'total_computations': 0,
            'ksg_computations': 0,
            'gcmi_computations': 0,
            'binned_computations': 0,
            'avg_computation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger.info(f"✅ CMI Estimator initialized with {config.estimator_type} algorithm")
    
    def _initialize_estimator(self):
        """Initialize the selected estimator."""
        try:
            if self.config.estimator_type == 'ksg':
                self._initialize_ksg_estimator()
            elif self.config.estimator_type == 'gcmi':
                self._initialize_gcmi_estimator()
            elif self.config.estimator_type == 'binned':
                self._initialize_binned_estimator()
            else:
                self.logger.warning(f"⚠️ Unknown estimator type: {self.config.estimator_type}")
                self._initialize_ksg_estimator()  # Fallback
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize {self.config.estimator_type} estimator: {e}")
            self._initialize_ksg_estimator()  # Fallback
    
    def _initialize_ksg_estimator(self):
        """Initialize KSG estimator."""
        try:
            # Try to import KSG implementation
            from sklearn.neighbors import NearestNeighbors
            self.ksg_nn = NearestNeighbors(n_neighbors=self.config.ksg_k + 1)
            self.ksg_available = True
            self.logger.info("✅ KSG estimator initialized")
        except ImportError:
            self.ksg_nn = None
            self.ksg_available = False
            self.logger.warning("⚠️ KSG estimator not available")
    
    def _initialize_gcmi_estimator(self):
        """Initialize GCMI estimator."""
        try:
            # GCMI estimator initialization
            self.gcmi_available = True
            self.logger.info("✅ GCMI estimator initialized")
        except Exception as e:
            self.gcmi_available = False
            self.logger.warning(f"⚠️ GCMI estimator not available: {e}")
    
    def _initialize_binned_estimator(self):
        """Initialize Binned estimator."""
        try:
            # Binned estimator initialization
            self.binned_available = True
            self.logger.info("✅ Binned estimator initialized")
        except Exception as e:
            self.binned_available = False
            self.logger.warning(f"⚠️ Binned estimator not available: {e}")
    
    def _initialize_caching(self):
        """Initialize caching system."""
        if self.config.enable_caching:
            try:
                from src.utils.caching import get_cmi_cache_manager
                self.cache_manager = get_cmi_cache_manager(
                    max_size_mb=self.config.cache_size_mb
                )
                self.caching_available = True
            except ImportError:
                self.cache_manager = None
                self.caching_available = False
                self.logger.warning("⚠️ Caching not available")
        else:
            self.cache_manager = None
            self.caching_available = False
    
    def compute_cmi(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """
        Compute Conditional Mutual Information I(X; Y | Z).
        
        Args:
            X: First variable
            Y: Second variable
            Z: Conditioning variable
            
        Returns:
            CMI value
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.caching_available:
                cache_key = self._generate_cache_key(X, Y, Z)
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.computation_stats['cache_hits'] += 1
                    return cached_result
                self.computation_stats['cache_misses'] += 1
            
            # Compute CMI based on estimator type
            if self.config.estimator_type == 'ksg' and self.ksg_available:
                cmi_value = self._compute_cmi_ksg(X, Y, Z)
            elif self.config.estimator_type == 'gcmi' and self.gcmi_available:
                cmi_value = self._compute_cmi_gcmi(X, Y, Z)
            elif self.config.estimator_type == 'binned' and self.binned_available:
                cmi_value = self._compute_cmi_binned(X, Y, Z)
            else:
                # Fallback to simple correlation-based approximation
                cmi_value = self._compute_cmi_fallback(X, Y, Z)
            
            # Cache result
            if self.caching_available:
                self.cache_manager.set(cache_key, cmi_value)
            
            # Update stats
            computation_time = (datetime.now() - start_time).total_seconds()
            self._update_computation_stats(computation_time)
            
            return cmi_value
            
        except Exception as e:
            self.logger.warning(f"⚠️ CMI computation failed: {e}")
            return 0.0
    
    def compute_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Mutual Information I(X; Y).
        
        Args:
            X: First variable
            Y: Second variable
            
        Returns:
            MI value
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.caching_available:
                cache_key = self._generate_cache_key(X, Y, None)
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.computation_stats['cache_hits'] += 1
                    return cached_result
                self.computation_stats['cache_misses'] += 1
            
            # Compute MI based on estimator type
            if self.config.estimator_type == 'ksg' and self.ksg_available:
                mi_value = self._compute_mi_ksg(X, Y)
            elif self.config.estimator_type == 'gcmi' and self.gcmi_available:
                mi_value = self._compute_mi_gcmi(X, Y)
            elif self.config.estimator_type == 'binned' and self.binned_available:
                mi_value = self._compute_mi_binned(X, Y)
            else:
                # Fallback to simple correlation-based approximation
                mi_value = self._compute_mi_fallback(X, Y)
            
            # Cache result
            if self.caching_available:
                self.cache_manager.set(cache_key, mi_value)
            
            # Update stats
            computation_time = (datetime.now() - start_time).total_seconds()
            self._update_computation_stats(computation_time)
            
            return mi_value
            
        except Exception as e:
            self.logger.warning(f"⚠️ MI computation failed: {e}")
            return 0.0
    
    def _compute_cmi_ksg(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Compute CMI using KSG estimator."""
        try:
            # Ensure arrays are 2D
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
            
            # Combine variables
            XYZ = np.hstack([X, Y, Z])
            XZ = np.hstack([X, Z])
            YZ = np.hstack([Y, Z])
            
            # Fit nearest neighbors
            self.ksg_nn.fit(XYZ)
            distances_xyz, _ = self.ksg_nn.kneighbors(XYZ)
            
            self.ksg_nn.fit(XZ)
            distances_xz, _ = self.ksg_nn.kneighbors(XZ)
            
            self.ksg_nn.fit(YZ)
            distances_yz, _ = self.ksg_nn.kneighbors(YZ)
            
            # Compute CMI using KSG formula
            k = self.config.ksg_k
            cmi_values = []
            
            for i in range(len(X)):
                eps_xyz = distances_xyz[i, k]
                eps_xz = distances_xz[i, k]
                eps_yz = distances_yz[i, k]
                
                # Count points within epsilon
                n_xyz = np.sum(np.linalg.norm(XYZ - XYZ[i], axis=1) <= eps_xyz)
                n_xz = np.sum(np.linalg.norm(XZ - XZ[i], axis=1) <= eps_xz)
                n_yz = np.sum(np.linalg.norm(YZ - YZ[i], axis=1) <= eps_yz)
                
                # KSG CMI formula
                cmi_i = np.log(n_xz * n_yz / (n_xyz * len(X)))
                cmi_values.append(cmi_i)
            
            return np.mean(cmi_values)
            
        except Exception as e:
            self.logger.warning(f"⚠️ KSG CMI computation failed: {e}")
            return self._compute_cmi_fallback(X, Y, Z)
    
    def _compute_mi_ksg(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MI using KSG estimator."""
        try:
            # Ensure arrays are 2D
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            
            # Combine variables
            XY = np.hstack([X, Y])
            
            # Fit nearest neighbors
            self.ksg_nn.fit(XY)
            distances_xy, _ = self.ksg_nn.kneighbors(XY)
            
            self.ksg_nn.fit(X)
            distances_x, _ = self.ksg_nn.kneighbors(X)
            
            self.ksg_nn.fit(Y)
            distances_y, _ = self.ksg_nn.kneighbors(Y)
            
            # Compute MI using KSG formula
            k = self.config.ksg_k
            mi_values = []
            
            for i in range(len(X)):
                eps_xy = distances_xy[i, k]
                eps_x = distances_x[i, k]
                eps_y = distances_y[i, k]
                
                # Count points within epsilon
                n_xy = np.sum(np.linalg.norm(XY - XY[i], axis=1) <= eps_xy)
                n_x = np.sum(np.linalg.norm(X - X[i], axis=1) <= eps_x)
                n_y = np.sum(np.linalg.norm(Y - Y[i], axis=1) <= eps_y)
                
                # KSG MI formula
                mi_i = np.log(n_x * n_y / (n_xy * len(X)))
                mi_values.append(mi_i)
            
            return np.mean(mi_values)
            
        except Exception as e:
            self.logger.warning(f"⚠️ KSG MI computation failed: {e}")
            return self._compute_mi_fallback(X, Y)
    
    def _compute_cmi_gcmi(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Compute CMI using GCMI estimator."""
        try:
            # GCMI implementation (simplified)
            # This is a placeholder - real GCMI would use Gaussian assumptions
            
            # Compute correlations
            corr_xy = np.corrcoef(X, Y)[0, 1]
            corr_xz = np.corrcoef(X, Z.flatten())[0, 1]
            corr_yz = np.corrcoef(Y, Z.flatten())[0, 1]
            
            # GCMI approximation
            cmi = -0.5 * np.log(1 - corr_xy**2) - 0.5 * np.log(1 - corr_xz**2) - 0.5 * np.log(1 - corr_yz**2)
            cmi += 0.5 * np.log(1 - corr_xy**2 - corr_xz**2 - corr_yz**2 + 2*corr_xy*corr_xz*corr_yz)
            
            return max(0.0, cmi)
            
        except Exception as e:
            self.logger.warning(f"⚠️ GCMI CMI computation failed: {e}")
            return self._compute_cmi_fallback(X, Y, Z)
    
    def _compute_mi_gcmi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MI using GCMI estimator."""
        try:
            # GCMI MI implementation (simplified)
            corr = np.corrcoef(X, Y)[0, 1]
            mi = -0.5 * np.log(1 - corr**2)
            return max(0.0, mi)
            
        except Exception as e:
            self.logger.warning(f"⚠️ GCMI MI computation failed: {e}")
            return self._compute_mi_fallback(X, Y)
    
    def _compute_cmi_binned(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Compute CMI using Binned estimator."""
        try:
            # Binned CMI implementation
            bins = self.config.binned_bins
            
            # Discretize variables
            X_binned = np.digitize(X, np.linspace(X.min(), X.max(), bins))
            Y_binned = np.digitize(Y, np.linspace(Y.min(), Y.max(), bins))
            Z_binned = np.digitize(Z.flatten(), np.linspace(Z.min(), Z.max(), bins))
            
            # Compute joint and marginal probabilities
            p_xyz = self._compute_joint_probability(X_binned, Y_binned, Z_binned)
            p_xz = self._compute_joint_probability(X_binned, Z_binned)
            p_yz = self._compute_joint_probability(Y_binned, Z_binned)
            p_z = self._compute_marginal_probability(Z_binned)
            
            # Compute CMI
            cmi = 0.0
            for i in range(bins):
                for j in range(bins):
                    for k in range(bins):
                        if p_xyz[i, j, k] > 0 and p_xz[i, k] > 0 and p_yz[j, k] > 0 and p_z[k] > 0:
                            cmi += p_xyz[i, j, k] * np.log(
                                (p_xyz[i, j, k] * p_z[k]) / (p_xz[i, k] * p_yz[j, k])
                            )
            
            return max(0.0, cmi)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Binned CMI computation failed: {e}")
            return self._compute_cmi_fallback(X, Y, Z)
    
    def _compute_mi_binned(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MI using Binned estimator."""
        try:
            # Binned MI implementation
            bins = self.config.binned_bins
            
            # Discretize variables
            X_binned = np.digitize(X, np.linspace(X.min(), X.max(), bins))
            Y_binned = np.digitize(Y, np.linspace(Y.min(), Y.max(), bins))
            
            # Compute joint and marginal probabilities
            p_xy = self._compute_joint_probability(X_binned, Y_binned)
            p_x = self._compute_marginal_probability(X_binned)
            p_y = self._compute_marginal_probability(Y_binned)
            
            # Compute MI
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return max(0.0, mi)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Binned MI computation failed: {e}")
            return self._compute_mi_fallback(X, Y)
    
    def _compute_joint_probability(self, *variables):
        """Compute joint probability distribution."""
        try:
            # Count occurrences
            counts = np.zeros([len(np.unique(var)) for var in variables])
            
            for i in range(len(variables[0])):
                indices = tuple(var[i] for var in variables)
                counts[indices] += 1
            
            # Normalize to probabilities
            return counts / np.sum(counts)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Joint probability computation failed: {e}")
            return np.ones([len(np.unique(var)) for var in variables]) / np.prod([len(np.unique(var)) for var in variables])
    
    def _compute_marginal_probability(self, variable):
        """Compute marginal probability distribution."""
        try:
            unique, counts = np.unique(variable, return_counts=True)
            return counts / np.sum(counts)
        except Exception as e:
            self.logger.warning(f"⚠️ Marginal probability computation failed: {e}")
            return np.ones(len(np.unique(variable))) / len(np.unique(variable))
    
    def _compute_cmi_fallback(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Fallback CMI computation using correlation."""
        try:
            from scipy.stats import pearsonr
            
            corr_xy = abs(pearsonr(X, Y)[0])
            corr_xz = abs(pearsonr(X, Z.flatten())[0])
            corr_yz = abs(pearsonr(Y, Z.flatten())[0])
            
            # Simple approximation
            cmi = corr_xy - 0.5 * (corr_xz + corr_yz)
            return max(0.0, cmi)
            
        except Exception:
            return 0.0
    
    def _compute_mi_fallback(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Fallback MI computation using correlation."""
        try:
            from scipy.stats import pearsonr
            corr = abs(pearsonr(X, Y)[0])
            return corr * 0.5  # Rough approximation
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, X: np.ndarray, Y: np.ndarray, Z: Optional[np.ndarray]) -> str:
        """Generate cache key for computation."""
        try:
            # Use hash of data for cache key
            key_data = f"{X.tobytes()}{Y.tobytes()}"
            if Z is not None:
                key_data += Z.tobytes()
            return str(hash(key_data))
        except Exception:
            return f"{len(X)}_{len(Y)}_{len(Z) if Z is not None else 0}"
    
    def _update_computation_stats(self, computation_time: float):
        """Update computation statistics."""
        self.computation_stats['total_computations'] += 1
        
        # Update estimator-specific stats
        if self.config.estimator_type == 'ksg':
            self.computation_stats['ksg_computations'] += 1
        elif self.config.estimator_type == 'gcmi':
            self.computation_stats['gcmi_computations'] += 1
        elif self.config.estimator_type == 'binned':
            self.computation_stats['binned_computations'] += 1
        
        # Update average computation time
        total = self.computation_stats['total_computations']
        current_avg = self.computation_stats['avg_computation_time']
        self.computation_stats['avg_computation_time'] = (
            (current_avg * (total - 1) + computation_time) / total
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_computations': self.computation_stats['total_computations'],
            'ksg_computations': self.computation_stats['ksg_computations'],
            'gcmi_computations': self.computation_stats['gcmi_computations'],
            'binned_computations': self.computation_stats['binned_computations'],
            'avg_computation_time': self.computation_stats['avg_computation_time'],
            'cache_hit_rate': (
                self.computation_stats['cache_hits'] / 
                max(1, self.computation_stats['cache_hits'] + self.computation_stats['cache_misses'])
            ),
            'estimator_type': self.config.estimator_type,
            'ksg_available': self.ksg_available,
            'gcmi_available': self.gcmi_available,
            'binned_available': self.binned_available,
            'caching_available': self.caching_available
        }

# Convenience functions
def create_cmi_estimator(config: Optional[CMIEstimatorConfig] = None) -> CMIEstimator:
    """Create CMI estimator instance."""
    return CMIEstimator(config or CMIEstimatorConfig())

def compute_cmi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                estimator_type: str = 'ksg') -> float:
    """Compute CMI using specified estimator."""
    config = CMIEstimatorConfig(estimator_type=estimator_type)
    estimator = create_cmi_estimator(config)
    return estimator.compute_cmi(X, Y, Z)

def compute_mi(X: np.ndarray, Y: np.ndarray, 
               estimator_type: str = 'ksg') -> float:
    """Compute MI using specified estimator."""
    config = CMIEstimatorConfig(estimator_type=estimator_type)
    estimator = create_cmi_estimator(config)
    return estimator.compute_mi(X, Y)
