from __future__ import annotations

from src.core.decorators import cached as cached_src_core_decorators
from src.core.decorators import log_call, validates
from src.core.domain import quality_gate as quality_gate_src_core_domain
from src.core.domain import secure_data_processing

# src/training/gpu_acceleration_m1.py


"""
GPU Acceleration for Mac M1 (Apple Silicon) using Metal Performance Shaders.
Provides optimized matrix operations leveraging Apple's Metal framework.
"""

import time
from typing import Any

import numpy as np
import torch

from src.core.decorators import handles_errors as handles_errors_src_core_decorators
from src.utils.logger import system_logger


class GPUAccelerationM1:
    """GPU acceleration for M1 Mac using MPS (Metal Performance Shaders)."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize GPU acceleration.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.logger = system_logger.getChild("GPUAccelerationM1")

        # GPU configuration
        self.mps_available = torch.backends.mps.is_available()
        self.device = torch.device("mps" if self.mps_available else "cpu")

        # Performance tracking
        self.gpu_operations_count = 0
        self.gpu_processing_time = 0.0

        # Configuration
        self.enable_cpu_fallback = self.config.get("enable_cpu_fallback", True)
        self.memory_threshold = self.config.get("memory_threshold", 0.8)

        self.logger.info(
            f"GPU Acceleration initialized - MPS available: {self.mps_available}",
        )

    @validates(required_files=[], data_quality_checks={"min_rows": 100})
    @quality_gate(
        model_performance_thresholds={},
        data_quality_metrics={"completeness": 0.9},
    )
    @handles_errors(fallback=None)
    def gpu_matrix_multiplication(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        GPU-accelerated matrix multiplication using MPS.

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            Result matrix and metadata
        """
        try:
            start_time = time.time()
            self.logger.info("🚀 GPU Matrix Multiplication (MPS)")

            # Check if GPU should be used
            if not self._should_use_gpu(A, B):
                return self._cpu_matrix_multiplication(A, B)

            # Convert to PyTorch tensors
            A_tensor = torch.tensor(A, dtype=torch.float32, device=self.device)
            B_tensor = torch.tensor(B, dtype=torch.float32, device=self.device)

            # Perform matrix multiplication
            with torch.no_grad():
                result_tensor = torch.mm(A_tensor, B_tensor)

            # Convert back to numpy
            result = result_tensor.cpu().numpy()

            # Clean up GPU memory
            del A_tensor, B_tensor, result_tensor
            if self.mps_available:
                torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": "gpu_matrix_multiplication",
                "device": str(self.device),
                "processing_time": processing_time,
                "matrix_shapes": [A.shape, B.shape, result.shape],
                "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(
                f"✅ GPU Matrix Multiplication completed in {processing_time:.4f}s",
            )
            return result, metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU Matrix Multiplication failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_matrix_multiplication(A, B)
            raise

    @secure_data_processing(encryption_level="high", data_validation=True)
    @cached(chunk_size=3000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handles_errors(fallback=None)
    def gpu_svd_decomposition(
        self,
        matrix: np.ndarray,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        GPU-accelerated SVD decomposition using MPS.

        Args:
            matrix: Input matrix
            k: Number of singular values to compute

        Returns:
            U, S, Vt matrices and metadata
        """
        try:
            start_time = time.time()
            self.logger.info("🚀 GPU SVD Decomposition (MPS)")

            # Check if GPU should be used
            if not self._should_use_gpu(matrix):
                return self._cpu_svd_decomposition(matrix, k)

            # Convert to PyTorch tensor
            matrix_tensor = torch.tensor(
                matrix, dtype=torch.float32, device=self.device
            )

            # Perform SVD decomposition
            with torch.no_grad():
                U, S, Vt = torch.linalg.svd(matrix_tensor, full_matrices=False)

                # Truncate if k is specified
                if k is not None and k < len(S):
                    U = U[:, :k]
                    S = S[:k]
                    Vt = Vt[:k, :]

            # Convert back to numpy
            U_np = U.cpu().numpy()
            S_np = S.cpu().numpy()
            Vt_np = Vt.cpu().numpy()

            # Clean up GPU memory
            del matrix_tensor, U, S, Vt
            if self.mps_available:
                torch.mps.empty_cache()

            processing_time = time.time() - start_time
            metadata = {
                "operation": "gpu_svd_decomposition",
                "device": str(self.device),
                "processing_time": processing_time,
                "matrix_shape": matrix.shape,
                "k": k,
                "gpu_memory_used": self._get_gpu_memory_usage(),
            }

            self.gpu_operations_count += 1
            self.gpu_processing_time += processing_time

            self.logger.info(
                f"✅ GPU SVD Decomposition completed in {processing_time:.4f}s",
            )
            return U_np, S_np, Vt_np, metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU SVD Decomposition failed: {e}")
            if self.config.enable_cpu_fallback:
                self.logger.info("🔄 Falling back to CPU implementation")
                return self._cpu_svd_decomposition(matrix, k)
            raise

    @handles_errors(fallback=False)
    def _should_use_gpu(self, *matrices: np.ndarray) -> bool:
        """Check if GPU should be used for the given matrices.

        Args:
            *matrices: Matrices to check

        Returns:
            bool: True if GPU should be used, False otherwise

        """
        try:
            # Check if MPS is available
            if not self.mps_available:
                return False

            # Check matrix sizes (GPU is more efficient for larger matrices)
            total_elements = sum(matrix.size for matrix in matrices)
            if total_elements < 10000:  # Small matrices are faster on CPU
                return False

            # Check memory usage
            return not self._get_gpu_memory_usage() > self.memory_threshold

        except Exception as e:
            self.logger.warning(f"Error checking GPU usage: {e}")
            return False

    def _cpu_matrix_multiplication(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """CPU fallback for matrix multiplication.

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            Result matrix and metadata

        """
        start_time = time.time()
        result = np.matmul(A, B)
        processing_time = time.time() - start_time

        metadata = {
            "operation": "cpu_matrix_multiplication",
            "device": "cpu",
            "processing_time": processing_time,
            "matrix_shapes": [A.shape, B.shape, result.shape],
        }

        return result, metadata

    def _cpu_svd_decomposition(
        self,
        matrix: np.ndarray,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """CPU fallback for SVD decomposition.

        Args:
            matrix: Input matrix
            k: Number of singular values to compute

        Returns:
            U, S, Vt matrices and metadata

        """
        start_time = time.time()
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

        # Truncate if k is specified
        if k is not None and k < len(S):
            U = U[:, :k]
            S = S[:k]
            Vt = Vt[:k, :]

        processing_time = time.time() - start_time

        metadata = {
            "operation": "cpu_svd_decomposition",
            "device": "cpu",
            "processing_time": processing_time,
            "matrix_shape": matrix.shape,
            "k": k,
        }

        return U, S, Vt, metadata

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage.

        Returns:
            float: Memory usage as a fraction of total memory

        """
        try:
            if self.mps_available:
                # MPS doesn't provide direct memory usage info
                # Return a conservative estimate
                return 0.5
            return 0.0
        except Exception:
            return 0.0

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            dict: Performance statistics

        """
        return {
            "gpu_operations_count": self.gpu_operations_count,
            "gpu_processing_time": self.gpu_processing_time,
            "mps_available": self.mps_available,
            "device": str(self.device),
        }
