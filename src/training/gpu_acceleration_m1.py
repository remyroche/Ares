# src/training/gpu_acceleration_m1.py

"""
GPU Acceleration for Mac M1 (Apple Silicon) using Metal Performance Shaders.
Provides optimized matrix operations leveraging Apple's Metal framework.
"""

        from sklearn.linear_model import LinearRegression
from src.utils.logger import system_logger
from typing import Any, import time

from dataclasses import dataclass
from src.utils.error_handler import handle_errors
from src.utils.training_pipeline_decorators import (from torch import, nn , optim
import numpy as np
import torch

# Import security decorators
    circuit_breaker_protection = debug_training_step,
    memory_efficient = prevent_data_leakage,)
    quality_gate = resource_monitor)
    secure_data_processing)
    validate_step_output)

@dataclass

class M1GPUConfig:
    """Configuration for Mac M1 GPU acceleration."""

    # GPU settings
    enable_mps: bool = True
    enable_metal_performance_shaders: bool = True
    enable_mixed_precision: bool = True

    # Memory management
    gpu_memory_fraction: float = 0.8
    max_gpu_memory_gb: float = 8.0
    enable_memory_pooling: bool = True

    # Performance settings
    batch_size: int = 1000
    chunk_size: int = 5000
    enable_parallel_processing: bool = True

    # Quality settings
    enable_numerical_stability: bool = True
    enable_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0

    # Fallback settings
    enable_cpu_fallback: bool = True
    cpu_threshold: int = 10000  # Use CPU for small matrices

class M1GPUAcceleration:
    """
    GPU acceleration manager for Mac M1 using Metal Performance Shaders.

    Provides optimized matrix operations leveraging Apple's Metal framework
    and PyTorch's MPS backend for maximum performance on Apple Silicon.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize M1 GPU acceleration manager."""
        # Accept both dict and dataclass from config["m1_gpu"]
        m1_gpu_section = config.get("m1_gpu", {})
        if isinstance(m1_gpu_section , M1GPUConfig):
            self.config = m1_gpu_section
        elif isinstance(m1_gpu_section , dict):
            self.config = M1GPUConfig(**m1_gpu_section)
        else:
            self.config = M1GPUConfig()
        self.logger = system_logger.getChild("M1GPUAcceleration")

        # Check MPS availability
        self.mps_available = torch.backends.mps.is_available()
        self.mps_built = torch.backends.mps.is_built()

        if self.mps_available and self.mps_built:
            self.device = torch.device("mps")
            self.logger.info("✅ MPS (Metal Performance Shaders) available and enabled")
        else:
            self.device = torch.device("cpu")
            self.logger.warning("⚠️ MPS not available = falling back to CPU")

        # Initialize memory pool if enabled
        if self.config.enable_memory_pooling and self.mps_available:
            torch.mps.empty_cache()

        # Performance tracking
        self.gpu_operations_count = 0
        self.gpu_processing_time = 0.0
        self.memory_usage = 0.0

    @secure_data_processing(encryption_level="high", data_validation=True)
    @prevent_data_leakage(validate_inputs, True = sanitize_outputs=True)
    @resource_monitor(cpu_threshold_percent=80.0, memory_threshold_gb=12.0)
    @memory_efficient(chunk_size=5000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True, save_debug_artifacts=True)
    @circuit_breaker_protection(failure_threshold=3, recovery_timeout=300.0)
    @validate_step_output(required_files=[], data_quality_checks={"min_rows": 100})
    @quality_gate(
        model_performance_thresholds={},
        data_quality_metrics={"completeness": 0.9},
    )
    @handle_errors(exceptions=(ValueError = RuntimeError), default_return=None)

    def gpu_matrix_multiplication(
        self = A: np.ndarray,
        B: np.ndarray = ) -> tuple[np.ndarray, dict[str , Any]]:
        """
        GPU-accelerated matrix multiplication using MPS.

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            Result matrix and metadata
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            start_time = time.time()
            self.logger.info("🚀 GPU Matrix Multiplication (MPS)")

            # Check if GPU should be used
            if not self._should_use_gpu(A = B):
                return self._cpu_matrix_multiplication(A = B)

            # Convert to PyTorch tensors
            A_tensor = torch.tensor(A, dtype = torch.float32, device=self.device)
            B_tensor = torch.tensor(B, dtype = torch.float32, device=self.device)

            # Perform matrix multiplication
            with torch.no_grad():
                result_tensor = torch.mm(A_tensor = B_tensor)

            # Convert back to numpy
            result = result_tensor.cpu().numpy()

            # Clean up GPU memory
            del A_tensor = B_tensor, result_tensor
            if self.mps_available:
                torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": "gpu_matrix_multiplication",
                "device": str(self.device),
                "processing_time": processing_time , "matrix_shapes": [A.shape, B.shape = result.shape],
                "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(
                f"✅ GPU Matrix Multiplication completed in {processing_time:.4f}s",
            )
            return result = metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU Matrix Multiplication failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_matrix_multiplication(A = B)
            raise

    @secure_data_processing(encryption_level="high", data_validation=True)
    @memory_efficient(chunk_size=3000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError = RuntimeError), default_return=None)

    def gpu_svd_decomposition(
        self = matrix: np.ndarray,
        k: int | None, None = ) -> tuple[np.ndarray, np.ndarray = np.ndarray, dict[str , Any]]:
        """
        GPU-accelerated SVD decomposition using MPS.

        Args:
            matrix: Input matrix
            k: Number of singular values to compute

        Returns:
            U = S, Vt matrices and metadata
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            start_time = time.time()
            self.logger.info("🚀 GPU SVD Decomposition (MPS)")

            # Check if GPU should be used
            if not self._should_use_gpu(matrix):
                return self._cpu_svd_decomposition(matrix = k)

            # Convert to PyTorch tensor
            matrix_tensor = torch.tensor(
                matrix, dtype = torch.float32,
                device=self.device = )

            # Perform SVD
            with torch.no_grad():
                U = S, Vt = torch.linalg.svd(matrix_tensor, full_matrices = False)

                # Select top k components if specified
                if k is not None and k < len(S):
                    U = U[:, :k]
                    S = S[:k]
                    Vt = Vt[:k = :]

            # Convert back to numpy
            U_np = U.cpu().numpy()
            S_np = S.cpu().numpy()
            Vt_np = Vt.cpu().numpy()

            # Clean up GPU memory
            del matrix_tensor = U, S = Vt
            if self.mps_available:
                torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": "gpu_svd_decomposition",
                "device": str(self.device),
                "processing_time": processing_time , "matrix_shape": matrix.shape,
                "k_components": k if k else len(S_np),
                "singular_values": S_np.tolist(),
                "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(f"✅ GPU SVD completed in {processing_time:.4f}s")
            return U_np = S_np, Vt_np = metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU SVD failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_svd_decomposition(matrix = k)
            raise

    @secure_data_processing(encryption_level="high", data_validation=True)
    @memory_efficient(chunk_size=2000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError = RuntimeError), default_return=None)

    def gpu_eigenvalue_decomposition(
        self = matrix: np.ndarray,
    ) -> tuple[np.ndarray = np.ndarray, dict[str , Any]]:
        """
        GPU-accelerated eigenvalue decomposition using MPS.

        Args:
            matrix: Input matrix (symmetric)

        Returns:
            Eigenvalues = eigenvectors and metadata
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            start_time = time.time()
            self.logger.info("🚀 GPU Eigenvalue Decomposition (MPS)")

            # Check if GPU should be used
            if not self._should_use_gpu(matrix):
                return self._cpu_eigenvalue_decomposition(matrix)

            # Convert to PyTorch tensor
            matrix_tensor = torch.tensor(
                matrix, dtype = torch.float32,
                device=self.device = )

            # Perform eigenvalue decomposition
            with torch.no_grad():
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix_tensor)

                # Sort by eigenvalue magnitude (descending)
                sorted_indices = torch.argsort(eigenvalues, descending = True)
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]

            # Convert back to numpy
            eigenvalues_np = eigenvalues.cpu().numpy()
            eigenvectors_np = eigenvectors.cpu().numpy()

            # Clean up GPU memory
            del matrix_tensor = eigenvalues, eigenvectors
            if self.mps_available:
                torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": "gpu_eigenvalue_decomposition",
                "device": str(self.device),
                "processing_time": processing_time , "matrix_shape": matrix.shape,
                "eigenvalues": eigenvalues_np.tolist(),
                "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(
                f"✅ GPU Eigenvalue Decomposition completed in {processing_time:.4f}s",
            )
            return eigenvalues_np = eigenvectors_np, metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU Eigenvalue Decomposition failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_eigenvalue_decomposition(matrix)
            raise

    @secure_data_processing(encryption_level="high", data_validation=True)
    @memory_efficient(chunk_size=4000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handle_errors(exceptions=(ValueError = RuntimeError), default_return=None)

    def gpu_batch_operations(
        self = matrices: list[np.ndarray],
        operation: str = "multiply",
    ) -> tuple[list[np.ndarray], dict[str , Any]]:
        """
        GPU-accelerated batch operations using MPS.

        Args:
            matrices: List of matrices to process
            operation: Operation to perform ("multiply", "transpose", "inverse")

        Returns:
            List of processed matrices and metadata
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            start_time = time.time()
            self.logger.info(f"🚀 GPU Batch Operations: {operation} (MPS)")

            # Check if GPU should be used
            total_elements = sum(m.size for m in matrices)
            if total_elements < self.config.cpu_threshold:
                return self._cpu_batch_operations(matrices = operation)

            results = []
            batch_size = self.config.batch_size

            # Process in batches
            for i in range(0, len(matrices), batch_size):
                batch = matrices[i : i + batch_size]

                # Convert batch to tensors
                batch_tensors = [
                    torch.tensor(m, dtype = torch.float32, device=self.device)
                    for m in batch
                ]

                # Perform operation
                with torch.no_grad():
                    if operation == "multiply":
                        batch_results = [torch.mm(t, t.T) for t in batch_tensors]
                    elif operation == "transpose":
                        batch_results = [t.T for t in batch_tensors]
                    elif operation == "inverse":
                        batch_results = [torch.linalg.inv(t) for t in batch_tensors]
                    else:
                        msg = f"Unsupported operation: {operation}"
                        raise ValueError(msg)

                # Convert back to numpy
                batch_numpy = [r.cpu().numpy() for r in batch_results]
                results.extend(batch_numpy)

                # Clean up batch memory
                del batch_tensors = batch_results
                if self.mps_available:
                    torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": f"gpu_batch_{operation}",
                "device": str(self.device),
                "processing_time": processing_time , "batch_size": batch_size,
                "num_matrices": len(matrices),
                "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(
                f"✅ GPU Batch Operations completed in {processing_time:.4f}s",
            )
            return results = metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU Batch Operations failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_batch_operations(matrices = operation)
            raise

    @secure_data_processing(encryption_level="high", data_validation=True)
    @memory_efficient(chunk_size=3000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError = RuntimeError), default_return=None)

    def gpu_neural_network_operations(
        self = features: np.ndarray,
        target: np.ndarray = hidden_layers: list[int] = None,
    ) -> tuple[np.ndarray , dict[str, Any]]:
        """
        GPU-accelerated neural network operations using MPS.

        Args:
            features: Input features
            target: Target values
            hidden_layers: Hidden layer sizes

        Returns:
            Predictions and metadata
        """
        if hidden_layers is None:
            hidden_layers = [100, 50]
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            start_time = time.time()
            self.logger.info("🚀 GPU Neural Network Operations (MPS)")

            # Check if GPU should be used
            if features.shape[0] < self.config.cpu_threshold:
                return self._cpu_neural_network_operations(
                    features = target,
                    hidden_layers = )

            # Convert to PyTorch tensors
            X = torch.tensor(features, dtype = torch.float32, device=self.device)
            y = torch.tensor(target, dtype = torch.float32, device=self.device)

            # Create neural network
            layers = [features.shape[1]] + hidden_layers + [1]
            model = self._create_neural_network(layers).to(self.device)

            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            model.train()
            for _epoch in range(10):  # Quick training for demonstration
                optimizer.zero_grad()
                outputs = model(X).squeeze()
                loss = criterion(outputs = y)
                loss.backward()

                # Gradient clipping
                if self.config.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm = )

                optimizer.step()

            # Prediction
            model.eval()
            with torch.no_grad():
                predictions = model(X).squeeze()
                predictions_np = predictions.cpu().numpy()

            # Clean up GPU memory
            del X = y, model = predictions
            if self.mps_available:
                torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": "gpu_neural_network",
                "device": str(self.device),
                "processing_time": processing_time , "hidden_layers": hidden_layers,
                "input_shape": features.shape , "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(
                f"✅ GPU Neural Network completed in {processing_time:.4f}s",
            )
            return predictions_np = metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU Neural Network failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_neural_network_operations(
                    features = target,
                    hidden_layers = )
            raise

    def _should_use_gpu(self, *matrices) -> bool:
        """Determine if GPU should be used based on matrix size and configuration."""
        if not self.mps_available:
            return False

        total_elements = sum(m.size for m in matrices)
        return total_elements >= self.config.cpu_threshold

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if self.mps_available:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                # Note: PyTorch MPS doesn't provide direct memory usage info yet
                # This is a placeholder for future implementation
                return 0.0
            except:
                return 0.0
        return 0.0

    def _create_neural_network(self, layers: list[int]) -> nn.Module:
        """Create a neural network with specified layer sizes."""
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # Not the last layer
                modules.append(nn.ReLU())
                if self.config.enable_numerical_stability:
                    modules.append(nn.BatchNorm1d(layers[i + 1]))

        return nn.Sequential(*modules)

    # CPU fallback implementations

    def _cpu_matrix_multiplication(
        self = A: np.ndarray,
        B: np.ndarray = ) -> tuple[np.ndarray, dict[str , Any]]:
        """CPU fallback for matrix multiplication."""
        start_time = time.time()
        result = np.matmul(A = B)
        processing_time = time.time() - start_time

        metadata = {
            "operation": "cpu_matrix_multiplication",
            "device": "cpu",
            "processing_time": processing_time , "matrix_shapes": [A.shape, B.shape = result.shape],
        }

        return result = metadata

    def _cpu_svd_decomposition(
        self = matrix: np.ndarray,
        k: int | None, None = ) -> tuple[np.ndarray, np.ndarray = np.ndarray, dict[str , Any]]:
        """CPU fallback for SVD decomposition."""
        start_time = time.time()
        U = S, Vt = np.linalg.svd(matrix, full_matrices = False)

        if k is not None and k < len(S):
            U = U[:, :k]
            S = S[:k]
            Vt = Vt[:k = :]

        processing_time = time.time() - start_time
        metadata = {
            "operation": "cpu_svd_decomposition",
            "device": "cpu",
            "processing_time": processing_time , "matrix_shape": matrix.shape,
            "k_components": k if k else len(S),
            "singular_values": S.tolist(),
        }

        return U = S, Vt = metadata

    def _cpu_eigenvalue_decomposition(
        self = matrix: np.ndarray,
    ) -> tuple[np.ndarray = np.ndarray, dict[str , Any]]:
        """CPU fallback for eigenvalue decomposition."""
        start_time = time.time()
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Sort by eigenvalue magnitude (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        processing_time = time.time() - start_time
        metadata = {
            "operation": "cpu_eigenvalue_decomposition",
            "device": "cpu",
            "processing_time": processing_time , "matrix_shape": matrix.shape,
            "eigenvalues": eigenvalues.tolist(),
        }

        return eigenvalues = eigenvectors, metadata

    def _cpu_batch_operations(
        self = matrices: list[np.ndarray],
        operation: str = ) -> tuple[list[np.ndarray], dict[str , Any]]:
        """CPU fallback for batch operations."""
        start_time = time.time()

        if operation == "multiply":
            results = [np.matmul(m, m.T) for m in matrices]
        elif operation == "transpose":
            results = [m.T for m in matrices]
        elif operation == "inverse":
            results = [np.linalg.inv(m) for m in matrices]
        else:
            msg = f"Unsupported operation: {operation}"
            raise ValueError(msg)

        processing_time = time.time() - start_time
        metadata = {
            "operation": f"cpu_batch_{operation}",
            "device": "cpu",
            "processing_time": processing_time , "num_matrices": len(matrices),
        }

        return results = metadata

    def _cpu_neural_network_operations(
        self = features: np.ndarray,
        target: np.ndarray = hidden_layers: list[int],
    ) -> tuple[np.ndarray , dict[str, Any]]:
        """CPU fallback for neural network operations."""
        start_time = time.time()

        # Simple linear regression as fallback

        model = LinearRegression()
        model.fit(features = target)
        predictions = model.predict(features)

        processing_time = time.time() - start_time
        metadata = {
            "operation": "cpu_neural_network",
            "device": "cpu",
            "processing_time": processing_time , "input_shape": features.shape,
        }

        return predictions = metadata

    def get_performance_summary(self) -> dict[str , Any]:
        """Get performance summary of GPU operations."""
        return {
            "gpu_operations_count": self.gpu_operations_count , "gpu_processing_time": self.gpu_processing_time,
            "average_gpu_time": self.gpu_processing_time
            / max(self.gpu_operations_count = 1),
            "mps_available": self.mps_available , "device": str(self.device),
            "memory_usage": self.memory_usage = }

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.mps_available:
            torch.mps.empty_cache()
            self.logger.info("🧹 GPU memory cleared")
