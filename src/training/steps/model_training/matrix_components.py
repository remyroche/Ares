"""Matrix operation components for enhanced matrix operations step.

This module contains specialized components for matrix computations,
GPU acceleration, and optimization.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from src.utils.logger import system_logger


class MatrixProcessor:
    """Handles matrix computations with GPU acceleration support."""
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 1000):
        """Initialize matrix processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for processing
        """
        self.logger = system_logger.getChild("MatrixProcessor")
        self.batch_size = batch_size
        
        # Check GPU availability
        self.device = self._setup_device(use_gpu)
        self.logger.info(f"✅ Matrix processor initialized with device: {self.device}")
        
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """Setup computation device (CPU/GPU/MPS).
        
        Args:
            use_gpu: Whether to use GPU
            
        Returns:
            Torch device
        """
        if not use_gpu:
            return torch.device("cpu")
        
        # Check for CUDA
        if torch.cuda.is_available():
            self.logger.info("🎮 CUDA GPU detected")
            return torch.device("cuda")
        
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            self.logger.info("🍎 Apple MPS detected")
            return torch.device("mps")
        
        # Fallback to CPU
        self.logger.warning("⚠️ No GPU available, using CPU")
        return torch.device("cpu")
    
    async def compute_correlation_matrix(
        self, 
        data: pd.DataFrame
    ) -> np.ndarray:
        """Compute correlation matrix using GPU if available.
        
        Args:
            data: Feature data
            
        Returns:
            Correlation matrix
        """
        try:
            # Convert to tensor
            data_tensor = torch.tensor(
                data.values, 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Standardize data
            mean = data_tensor.mean(dim=0)
            std = data_tensor.std(dim=0)
            standardized = (data_tensor - mean) / (std + 1e-8)
            
            # Compute correlation matrix
            n_samples = standardized.shape[0]
            corr_matrix = torch.matmul(
                standardized.T, 
                standardized
            ) / (n_samples - 1)
            
            # Convert back to numpy
            return corr_matrix.cpu().numpy()
            
        except Exception as e:
            self.logger.warning(f"GPU computation failed: {e}, using CPU")
            # Fallback to pandas
            return data.corr().values
    
    async def compute_covariance_matrix(
        self, 
        data: pd.DataFrame
    ) -> np.ndarray:
        """Compute covariance matrix using GPU if available.
        
        Args:
            data: Feature data
            
        Returns:
            Covariance matrix
        """
        try:
            # Convert to tensor
            data_tensor = torch.tensor(
                data.values, 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Center the data
            mean = data_tensor.mean(dim=0)
            centered = data_tensor - mean
            
            # Compute covariance matrix
            n_samples = centered.shape[0]
            cov_matrix = torch.matmul(
                centered.T, 
                centered
            ) / (n_samples - 1)
            
            # Convert back to numpy
            return cov_matrix.cpu().numpy()
            
        except Exception as e:
            self.logger.warning(f"GPU computation failed: {e}, using CPU")
            # Fallback to pandas
            return data.cov().values
    
    def compute_eigendecomposition(
        self, 
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigendecomposition of a matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            # Convert to tensor
            matrix_tensor = torch.tensor(
                matrix, 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Compute eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix_tensor)
            
            # Sort by eigenvalue magnitude (descending)
            indices = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]
            
            # Convert back to numpy
            return (
                eigenvalues.cpu().numpy(), 
                eigenvectors.cpu().numpy()
            )
            
        except Exception as e:
            self.logger.warning(f"GPU computation failed: {e}, using CPU")
            # Fallback to numpy
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            indices = np.argsort(eigenvalues)[::-1]
            return eigenvalues[indices], eigenvectors[:, indices]
    
    def compute_matrix_factorization(
        self, 
        matrix: np.ndarray, 
        n_components: int
    ) -> Dict[str, np.ndarray]:
        """Compute matrix factorization (PCA-like).
        
        Args:
            matrix: Input matrix
            n_components: Number of components to keep
            
        Returns:
            Dictionary with factorization results
        """
        # Compute eigendecomposition
        eigenvalues, eigenvectors = self.compute_eigendecomposition(matrix)
        
        # Keep top n_components
        n_components = min(n_components, len(eigenvalues))
        
        return {
            "components": eigenvectors[:, :n_components],
            "explained_variance": eigenvalues[:n_components],
            "explained_variance_ratio": eigenvalues[:n_components] / eigenvalues.sum()
        }


class DiverseLookbackIntegrator:
    """Integrates with diverse lookback optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize lookback integrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("DiverseLookbackIntegrator")
        self.optimizer = None
        
        # Try to import diverse lookback optimizer
        try:
            from src.training.diverse_lookback_optimizer import DiverseLookbackOptimizer
            self.optimizer = DiverseLookbackOptimizer(config)
            self.logger.info("✅ Diverse lookback optimizer loaded")
        except ImportError:
            self.logger.warning("⚠️ Diverse lookback optimizer not available")
    
    async def optimize_lookback_periods(
        self, 
        data: pd.DataFrame, 
        features: List[str]
    ) -> Dict[str, Any]:
        """Optimize lookback periods for features.
        
        Args:
            data: Training data
            features: List of features
            
        Returns:
            Optimization results
        """
        if self.optimizer:
            try:
                # Group features by type
                feature_groups = self._group_features_by_type(features)
                
                # Optimize each group
                optimized_periods = {}
                
                for group_name, group_features in feature_groups.items():
                    if group_features:
                        periods = await self.optimizer.optimize_feature_group(
                            data, 
                            group_features
                        )
                        optimized_periods[group_name] = periods
                
                return {
                    "optimized_periods": optimized_periods,
                    "feature_groups": feature_groups,
                    "method": "diverse_lookback"
                }
                
            except Exception as e:
                self.logger.error(f"Optimization failed: {e}")
                return self._get_default_periods()
        else:
            return self._get_default_periods()
    
    def _group_features_by_type(self, features: List[str]) -> Dict[str, List[str]]:
        """Group features by their type.
        
        Args:
            features: List of feature names
            
        Returns:
            Dictionary of feature groups
        """
        groups = {
            "price": [],
            "volume": [],
            "technical": [],
            "volatility": [],
            "momentum": [],
            "other": []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(x in feature_lower for x in ["price", "sma", "ema", "close", "open"]):
                groups["price"].append(feature)
            elif any(x in feature_lower for x in ["volume", "obv", "vpt"]):
                groups["volume"].append(feature)
            elif any(x in feature_lower for x in ["rsi", "macd", "stoch", "bb"]):
                groups["technical"].append(feature)
            elif any(x in feature_lower for x in ["volatility", "atr", "std"]):
                groups["volatility"].append(feature)
            elif any(x in feature_lower for x in ["momentum", "roc", "rate"]):
                groups["momentum"].append(feature)
            else:
                groups["other"].append(feature)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _get_default_periods(self) -> Dict[str, Any]:
        """Get default lookback periods.
        
        Returns:
            Default period configuration
        """
        return {
            "optimized_periods": {
                "price": [10, 20, 50],
                "volume": [10, 20],
                "technical": [14, 21],
                "volatility": [20, 50],
                "momentum": [10, 20]
            },
            "method": "default"
        }


class MatrixOptimizer:
    """Optimizes matrix operations and memory usage."""
    
    def __init__(self, optimization_level: str = "high"):
        """Initialize matrix optimizer.
        
        Args:
            optimization_level: Level of optimization (low, medium, high)
        """
        self.optimization_level = optimization_level
        self.logger = system_logger.getChild("MatrixOptimizer")
        
        # Set optimization parameters
        self.params = self._get_optimization_params(optimization_level)
        
    def _get_optimization_params(self, level: str) -> Dict[str, Any]:
        """Get optimization parameters based on level.
        
        Args:
            level: Optimization level
            
        Returns:
            Optimization parameters
        """
        params = {
            "low": {
                "use_float32": False,
                "chunk_processing": False,
                "compression": False,
                "cache_intermediates": True
            },
            "medium": {
                "use_float32": True,
                "chunk_processing": True,
                "compression": False,
                "cache_intermediates": True,
                "chunk_size": 5000
            },
            "high": {
                "use_float32": True,
                "chunk_processing": True,
                "compression": True,
                "cache_intermediates": False,
                "chunk_size": 1000
            }
        }
        
        return params.get(level, params["medium"])
    
    def optimize_matrix_computation(
        self, 
        matrix_func: callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Optimize a matrix computation.
        
        Args:
            matrix_func: Function to optimize
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if self.params["chunk_processing"]:
            return self._chunk_processing(matrix_func, *args, **kwargs)
        else:
            return matrix_func(*args, **kwargs)
    
    def _chunk_processing(
        self, 
        matrix_func: callable, 
        data: np.ndarray, 
        *args, 
        **kwargs
    ) -> np.ndarray:
        """Process matrix computation in chunks.
        
        Args:
            matrix_func: Function to apply
            data: Input data
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed result
        """
        chunk_size = self.params.get("chunk_size", 1000)
        n_samples = data.shape[0]
        
        if n_samples <= chunk_size:
            return matrix_func(data, *args, **kwargs)
        
        # Process in chunks
        results = []
        for i in range(0, n_samples, chunk_size):
            chunk = data[i:i + chunk_size]
            result = matrix_func(chunk, *args, **kwargs)
            results.append(result)
        
        # Combine results
        return np.vstack(results) if results else np.array([])
    
    def compress_matrix(self, matrix: np.ndarray, threshold: float = 0.01) -> Dict[str, Any]:
        """Compress matrix by removing small values.
        
        Args:
            matrix: Input matrix
            threshold: Threshold for small values
            
        Returns:
            Compressed matrix data
        """
        if not self.params.get("compression", False):
            return {"matrix": matrix, "compressed": False}
        
        # Create sparse representation
        mask = np.abs(matrix) > threshold
        sparse_matrix = matrix * mask
        
        # Calculate compression ratio
        n_nonzero = np.count_nonzero(sparse_matrix)
        n_total = matrix.size
        compression_ratio = 1 - (n_nonzero / n_total)
        
        return {
            "matrix": sparse_matrix,
            "compressed": True,
            "compression_ratio": compression_ratio,
            "threshold": threshold
        }
    
    def estimate_memory_usage(self, shape: Tuple[int, ...], dtype: str = "float64") -> Dict[str, float]:
        """Estimate memory usage for a matrix.
        
        Args:
            shape: Matrix shape
            dtype: Data type
            
        Returns:
            Memory usage estimates in MB
        """
        dtype_sizes = {
            "float64": 8,
            "float32": 4,
            "int64": 8,
            "int32": 4
        }
        
        if self.params.get("use_float32", False) and dtype == "float64":
            dtype = "float32"
        
        bytes_per_element = dtype_sizes.get(dtype, 8)
        total_elements = np.prod(shape)
        memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)
        
        return {
            "estimated_memory_mb": memory_mb,
            "dtype": dtype,
            "shape": shape,
            "optimization_applied": self.params.get("use_float32", False)
        }