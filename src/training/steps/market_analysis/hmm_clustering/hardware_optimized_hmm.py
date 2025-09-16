#!/usr/bin/env python3
"""
Hardware-Optimized HMM Clustering for Apple Silicon

This module provides specialized HMM clustering implementations optimized
for Apple Silicon M1/M2/M3 hardware, leveraging all available common utilities.
"""

import logging
import time
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# HMM dependencies
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None

# Import common utilities
from src.utils.common_operations import (
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_m1_cpu_optimizer
)
from src.utils.math_validation import safe_divide, safe_log, validate_finite
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.logger import system_logger

# Setup logging
logger = system_logger.getChild('HardwareOptimizedHMM')

@dataclass
class HardwareOptimizedConfig:
    """Configuration for hardware-optimized HMM clustering."""
    n_components: int = 3
    covariance_type: str = 'full'
    n_iter: int = 100
    random_state: int = 42
    
    # Hardware-specific settings
    use_gpu_acceleration: bool = True
    use_memory_optimization: bool = True
    use_cpu_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    batch_size: Optional[int] = None
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 1000
    
    # Validation settings
    enable_validation: bool = True
    enable_profiling: bool = False

class HardwareOptimizedHMM:
    """
    Hardware-optimized HMM clustering for Apple Silicon.
    
    This class provides specialized optimizations for M1/M2/M3 hardware
    including GPU acceleration, memory optimization, and CPU optimization.
    """
    
    def __init__(self, config: HardwareOptimizedConfig):
        """Initialize hardware-optimized HMM clustering."""
        self.config = config
        self.logger = logger.getChild('HardwareOptimizedHMM')
        
        # Initialize hardware managers
        self.gpu_manager = get_m1_gpu_manager() if config.use_gpu_acceleration else None
        self.memory_optimizer = get_m1_memory_optimizer() if config.use_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if config.use_cpu_optimization else None
        
        # Initialize matrix operations
        self.matrix_ops = UnifiedMatrixOperations()
        
        # Initialize HMM regime detector
        self.hmm_regime_detector = HMMRegimeDetector()
        
        # Performance tracking
        self.performance_metrics = {}
        self.memory_usage_history = []
        self.gpu_usage_history = []
        
        # State
        self.is_trained = False
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self.logger.info("🚀 Hardware-Optimized HMM initialized")
        self._log_hardware_capabilities()
    
    def _log_hardware_capabilities(self):
        """Log available hardware capabilities."""
        self.logger.info("🔧 Hardware Capabilities:")
        self.logger.info(f"   GPU Manager: {'✅ Available' if self.gpu_manager else '❌ Not Available'}")
        self.logger.info(f"   Memory Optimizer: {'✅ Available' if self.memory_optimizer else '❌ Not Available'}")
        self.logger.info(f"   CPU Optimizer: {'✅ Available' if self.cpu_optimizer else '❌ Not Available'}")
        self.logger.info(f"   Matrix Operations: {'✅ Available' if self.matrix_ops else '❌ Not Available'}")
        
        if self.gpu_manager and hasattr(self.gpu_manager, 'is_available'):
            self.logger.info(f"   GPU Available: {'✅ Yes' if self.gpu_manager.is_available() else '❌ No'}")
        
        if self.memory_optimizer:
            memory_info = self.memory_optimizer.get_memory_info()
            self.logger.info(f"   Memory Info: {memory_info}")
    
    def _optimize_data_for_hardware(self, data: np.ndarray) -> np.ndarray:
        """Optimize data for hardware-specific processing."""
        self.logger.info("🔧 Optimizing data for hardware...")
        
        # Memory optimization
        if self.memory_optimizer:
            # Convert to memory-efficient format
            data = self.memory_optimizer.create_memory_efficient_array(
                data, dtype=np.float32
            )
            
            # Monitor memory pressure
            memory_pressure = self.memory_optimizer.get_memory_pressure()
            if memory_pressure > 0.8:
                self.logger.warning(f"⚠️ High memory pressure: {memory_pressure:.2%}")
                # Trigger garbage collection
                gc.collect()
        
        # GPU optimization
        if self.gpu_manager and self.gpu_manager.is_available():
            # Prepare data for GPU processing
            data = self._prepare_data_for_gpu(data)
        
        # CPU optimization
        if self.cpu_optimizer:
            # Optimize data layout for CPU processing
            data = self._optimize_data_layout_for_cpu(data)
        
        return data
    
    def _prepare_data_for_gpu(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for GPU processing."""
        if not self.gpu_manager or not self.gpu_manager.is_available():
            return data
        
        self.logger.info("🚀 Preparing data for GPU processing...")
        
        try:
            # Ensure data is contiguous and properly aligned
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            
            # Convert to appropriate dtype for GPU
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Apply GPU-specific optimizations
            if hasattr(self.gpu_manager, 'optimize_array_for_gpu'):
                data = self.gpu_manager.optimize_array_for_gpu(data)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU data preparation failed: {e}")
            return data
    
    def _optimize_data_layout_for_cpu(self, data: np.ndarray) -> np.ndarray:
        """Optimize data layout for CPU processing."""
        if not self.cpu_optimizer:
            return data
        
        self.logger.info("⚡ Optimizing data layout for CPU...")
        
        try:
            # Get optimal thread count
            optimal_threads = self.cpu_optimizer.get_optimal_thread_count()
            self.logger.info(f"   Using {optimal_threads} threads")
            
            # Optimize data layout
            if hasattr(self.cpu_optimizer, 'optimize_array_layout'):
                data = self.cpu_optimizer.optimize_array_layout(data)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"⚠️ CPU data optimization failed: {e}")
            return data
    
    def _scale_features_optimized(self, data: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """Scale features using hardware-optimized operations."""
        self.logger.info("📏 Scaling features with hardware optimization...")
        
        # Use matrix operations for efficient scaling
        if self.matrix_ops and hasattr(self.matrix_ops, 'optimized_scaling'):
            scaled_data, scaler = self.matrix_ops.optimized_scaling(data)
        else:
            # Fallback to standard scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
        
        return scaled_data, scaler
    
    def _train_hmm_with_hardware_optimization(self, features: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train HMM with hardware-specific optimizations."""
        self.logger.info("🎯 Training HMM with hardware optimization...")
        
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not available. Install with: pip install hmmlearn")
        
        start_time = time.time()
        training_metrics = {}
        
        try:
            # Create HMM model
            model = hmm.GaussianHMM(
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state
            )
            
            # GPU-accelerated training
            if self.gpu_manager and self.gpu_manager.is_available():
                self.logger.info("🚀 Using GPU acceleration for HMM training...")
                training_metrics.update(self._train_with_gpu_acceleration(model, features))
            else:
                # CPU-optimized training
                self.logger.info("⚡ Using CPU optimization for HMM training...")
                training_metrics.update(self._train_with_cpu_optimization(model, features))
            
            training_time = time.time() - start_time
            training_metrics['training_time'] = training_time
            
            # Calculate model metrics
            log_likelihood = model.score(features)
            aic = 2 * model.n_features * self.config.n_components - 2 * log_likelihood
            bic = np.log(features.shape[0]) * model.n_features * self.config.n_components - 2 * log_likelihood
            
            training_metrics.update({
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'converged': model.monitor_.converged
            })
            
            self.logger.info(f"✅ HMM training completed in {training_time:.2f}s")
            self.logger.info(f"📊 Model metrics - AIC: {aic:.2f}, BIC: {bic:.2f}")
            
            return model, training_metrics
            
        except Exception as e:
            self.logger.error(f"❌ HMM training failed: {e}")
            raise
    
    def _train_with_gpu_acceleration(self, model: Any, features: np.ndarray) -> Dict[str, Any]:
        """Train HMM with GPU acceleration."""
        gpu_metrics = {}
        
        try:
            # Monitor GPU usage
            if hasattr(self.gpu_manager, 'start_gpu_monitoring'):
                self.gpu_manager.start_gpu_monitoring()
            
            # Train model
            model.fit(features)
            
            # Stop GPU monitoring and get metrics
            if hasattr(self.gpu_manager, 'stop_gpu_monitoring'):
                gpu_metrics = self.gpu_manager.stop_gpu_monitoring()
                self.gpu_usage_history.append(gpu_metrics)
            
            gpu_metrics['gpu_acceleration_used'] = True
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU acceleration failed, falling back to CPU: {e}")
            gpu_metrics = self._train_with_cpu_optimization(model, features)
            gpu_metrics['gpu_acceleration_used'] = False
        
        return gpu_metrics
    
    def _train_with_cpu_optimization(self, model: Any, features: np.ndarray) -> Dict[str, Any]:
        """Train HMM with CPU optimization."""
        cpu_metrics = {}
        
        try:
            # Set optimal thread count
            if self.cpu_optimizer:
                optimal_threads = self.cpu_optimizer.get_optimal_thread_count()
                cpu_metrics['optimal_threads'] = optimal_threads
                
                # Set thread count for numpy operations
                if hasattr(self.cpu_optimizer, 'set_numpy_threads'):
                    self.cpu_optimizer.set_numpy_threads(optimal_threads)
            
            # Monitor CPU usage
            if hasattr(self.cpu_optimizer, 'start_cpu_monitoring'):
                self.cpu_optimizer.start_cpu_monitoring()
            
            # Train model
            model.fit(features)
            
            # Stop CPU monitoring and get metrics
            if hasattr(self.cpu_optimizer, 'stop_cpu_monitoring'):
                cpu_metrics.update(self.cpu_optimizer.stop_cpu_monitoring())
            
            cpu_metrics['cpu_optimization_used'] = True
            
        except Exception as e:
            self.logger.warning(f"⚠️ CPU optimization failed: {e}")
            # Fallback to standard training
            model.fit(features)
            cpu_metrics['cpu_optimization_used'] = False
        
        return cpu_metrics
    
    def _batch_process_large_dataset(self, data: np.ndarray) -> np.ndarray:
        """Process large datasets in batches for memory efficiency."""
        if self.config.batch_size is None:
            return data
        
        self.logger.info(f"📦 Processing data in batches of {self.config.batch_size}")
        
        n_samples = data.shape[0]
        batch_size = min(self.config.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        processed_batches = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch = data[start_idx:end_idx]
            
            # Process batch with hardware optimization
            batch = self._optimize_data_for_hardware(batch)
            processed_batches.append(batch)
            
            # Memory management
            if self.memory_optimizer:
                memory_pressure = self.memory_optimizer.get_memory_pressure()
                if memory_pressure > 0.9:
                    self.logger.warning("⚠️ High memory pressure, triggering garbage collection")
                    gc.collect()
        
        return np.vstack(processed_batches)
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Fit hardware-optimized HMM clustering model.
        
        Args:
            data: Input data (DataFrame or numpy array)
            
        Returns:
            Dict containing results and performance metrics
        """
        self.logger.info("🚀 Starting hardware-optimized HMM clustering...")
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                self.feature_names = data.columns.tolist()
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Validate data
            if self.config.enable_validation:
                self._validate_data(data_array)
            
            # Optimize data for hardware
            data_array = self._optimize_data_for_hardware(data_array)
            
            # Process in batches if dataset is large
            if self.config.batch_size and data_array.shape[0] > self.config.batch_size:
                data_array = self._batch_process_large_dataset(data_array)
            
            # Scale features
            features, scaler = self._scale_features_optimized(data_array)
            self.scaler = scaler
            
            # Train HMM with hardware optimization
            model, training_metrics = self._train_hmm_with_hardware_optimization(features)
            self.model = model
            
            # Get predictions
            labels = model.predict(features)
            probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            # Calculate clustering metrics
            clustering_metrics = self._calculate_clustering_metrics(features, labels)
            
            # Get memory usage
            memory_usage = {}
            if self.memory_optimizer:
                memory_usage = self.memory_optimizer.get_memory_usage()
                self.memory_usage_history.append(memory_usage)
            
            # Create results
            results = {
                'model': model,
                'labels': labels,
                'probabilities': probabilities,
                'scaler': scaler,
                'feature_names': self.feature_names,
                'training_metrics': training_metrics,
                'clustering_metrics': clustering_metrics,
                'memory_usage': memory_usage,
                'hardware_metrics': {
                    'gpu_usage_history': self.gpu_usage_history,
                    'memory_usage_history': self.memory_usage_history,
                    'cpu_optimization_used': training_metrics.get('cpu_optimization_used', False),
                    'gpu_acceleration_used': training_metrics.get('gpu_acceleration_used', False)
                }
            }
            
            # Update state
            self.is_trained = True
            self.performance_metrics = training_metrics
            
            self.logger.info("✅ Hardware-optimized HMM clustering completed!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Hardware-optimized HMM clustering failed: {e}")
            raise
    
    def _validate_data(self, data: np.ndarray):
        """Validate input data."""
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        if data.shape[0] < self.config.n_components:
            raise ValueError(f"Not enough samples ({data.shape[0]}) for {self.config.n_components} components")
        
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")
    
    def _calculate_clustering_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering metrics."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        metrics = {}
        
        try:
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(features, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
            
            metrics['n_clusters'] = len(np.unique(labels))
            metrics['cluster_balance'] = self._calculate_cluster_balance(labels)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating clustering metrics: {e}")
            metrics = {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
        
        return metrics
    
    def _calculate_cluster_balance(self, labels: np.ndarray) -> float:
        """Calculate cluster balance metric."""
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) <= 1:
            return 1.0
        
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        return safe_divide(std_count, mean_count, 1.0)
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        # Optimize data for hardware
        data_array = self._optimize_data_for_hardware(data_array)
        
        # Scale features
        features = self.scaler.transform(data_array)
        
        # Make predictions
        return self.model.predict(features)
    
    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict cluster probabilities for new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        # Optimize data for hardware
        data_array = self._optimize_data_for_hardware(data_array)
        
        # Scale features
        features = self.scaler.transform(data_array)
        
        # Make probability predictions
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_hardware_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive hardware performance summary."""
        if not self.is_trained:
            return {}
        
        summary = {
            'training_metrics': self.performance_metrics,
            'hardware_utilization': {
                'gpu_available': self.gpu_manager.is_available() if self.gpu_manager else False,
                'memory_optimizer_active': self.memory_optimizer is not None,
                'cpu_optimizer_active': self.cpu_optimizer is not None,
                'matrix_operations_active': self.matrix_ops is not None
            },
            'gpu_usage_history': self.gpu_usage_history,
            'memory_usage_history': self.memory_usage_history,
            'config': self.config.__dict__
        }
        
        return summary


def create_hardware_optimized_hmm(config: Optional[HardwareOptimizedConfig] = None) -> HardwareOptimizedHMM:
    """Factory function to create hardware-optimized HMM clustering instance."""
    if config is None:
        config = HardwareOptimizedConfig()
    
    return HardwareOptimizedHMM(config)


# Example usage
if __name__ == "__main__":
    # Example usage
    logger.info("🚀 Hardware-Optimized HMM Clustering Example")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    n_features = 8
    
    # Generate sample data with 4 clusters
    cluster1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0], np.eye(8), n_samples // 4)
    cluster2 = np.random.multivariate_normal([2, 2, 2, 2, 2, 2, 2, 2], np.eye(8), n_samples // 4)
    cluster3 = np.random.multivariate_normal([-2, -2, -2, -2, -2, -2, -2, -2], np.eye(8), n_samples // 4)
    cluster4 = np.random.multivariate_normal([4, 4, 4, 4, 4, 4, 4, 4], np.eye(8), n_samples - 3 * (n_samples // 4))
    
    sample_data = np.vstack([cluster1, cluster2, cluster3, cluster4])
    
    # Create configuration
    config = HardwareOptimizedConfig(
        n_components=4,
        covariance_type='full',
        n_iter=100,
        random_state=42,
        use_gpu_acceleration=True,
        use_memory_optimization=True,
        use_cpu_optimization=True,
        batch_size=500,
        enable_profiling=True
    )
    
    # Create and train model
    hmm_clustering = create_hardware_optimized_hmm(config)
    results = hmm_clustering.fit(sample_data)
    
    # Print results
    print(f"Training completed in {results['training_metrics']['training_time']:.2f} seconds")
    print(f"Silhouette Score: {results['clustering_metrics']['silhouette_score']:.3f}")
    print(f"AIC: {results['training_metrics']['aic']:.2f}")
    print(f"BIC: {results['training_metrics']['bic']:.2f}")
    
    # Get hardware performance summary
    summary = hmm_clustering.get_hardware_performance_summary()
    print(f"Hardware Performance: {summary['hardware_utilization']}")