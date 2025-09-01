# src/config/m1_gpu_config.py

"""
Configuration for Mac M1 GPU acceleration and optimization.
Provides comprehensive settings for Metal Performance Shaders (MPS) integration.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class M1GPUConfig:
    """Configuration for Mac M1 GPU acceleration."""

    # GPU settings (DEFAULT: ALL ENABLED)
    enable_mps: bool = True
    enable_metal_performance_shaders: bool = True
    enable_mixed_precision: bool = True

    # Memory management (DEFAULT: OPTIMIZED)
    gpu_memory_fraction: float = 0.8
    max_gpu_memory_gb: float = 8.0
    enable_memory_pooling: bool = True
    enable_memory_cleanup: bool = True

    # Performance settings (DEFAULT: OPTIMIZED)
    batch_size: int = 1000
    chunk_size: int = 5000
    enable_parallel_processing: bool = True
    enable_batch_processing: bool = True

    # Quality settings (DEFAULT: ALL ENABLED)
    enable_numerical_stability: bool = True
    enable_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    enable_batch_norm: bool = True

    # Fallback settings (DEFAULT: ENABLED FOR RELIABILITY)
    enable_cpu_fallback: bool = True
    cpu_threshold: int = 10000  # Use CPU for small matrices
    enable_automatic_fallback: bool = True

    # Optimization settings (DEFAULT: ALL ENABLED)
    enable_tensor_cores: bool = True
    enable_fusion_operations: bool = True
    enable_memory_optimization: bool = True
    enable_compute_optimization: bool = True


@dataclass
class M1MatrixOperationsConfig:
    """Configuration for M1-optimized matrix operations."""

    # Matrix factorization settings (DEFAULT: ALL ENABLED)
    enable_gpu_svd: bool = True
    enable_gpu_eigenvalue: bool = True
    enable_gpu_matrix_multiply: bool = True
    enable_gpu_batch_operations: bool = True

    # Neural network settings (DEFAULT: ALL ENABLED)
    enable_gpu_neural_networks: bool = True
    enable_gpu_training: bool = True
    enable_gpu_inference: bool = True

    # Performance thresholds (DEFAULT: OPTIMIZED)
    min_matrix_size_for_gpu: int = 100
    min_batch_size_for_gpu: int = 50
    max_gpu_memory_usage: float = 0.8

    # Quality thresholds (DEFAULT: HIGH PRECISION)
    numerical_precision: float = 1e-6
    convergence_tolerance: float = 1e-8
    max_iterations: int = 1000


@dataclass
class M1SecurityConfig:
    """Configuration for M1 GPU security."""

    # Data security (DEFAULT: ALL ENABLED)
    enable_gpu_data_encryption: bool = True
    enable_memory_isolation: bool = True
    enable_secure_computation: bool = True

    # Monitoring (DEFAULT: ALL ENABLED)
    enable_gpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_monitoring: bool = True

    # Quality gates (DEFAULT: ALL ENABLED)
    enable_gpu_quality_gates: bool = True
    enable_result_validation: bool = True
    enable_error_detection: bool = True





def validate_m1_config(config: dict[str, Any]) -> bool:
    """Validate M1 GPU configuration settings."""

    try:
        # Check required sections
        required_sections = ["m1_gpu", "m1_matrix_operations", "m1_security"]
        for section in required_sections:
            if section not in config:
                msg = f"Missing required configuration section: {section}"
                raise ValueError(msg)

        # Validate GPU settings
        gpu_config = config["m1_gpu"]
        if (
            gpu_config["gpu_memory_fraction"] <= 0
            or gpu_config["gpu_memory_fraction"] > 1
        ):
            msg = "gpu_memory_fraction must be between 0 and 1"
            raise ValueError(msg)
        if gpu_config["max_gpu_memory_gb"] <= 0:
            msg = "max_gpu_memory_gb must be positive"
            raise ValueError(msg)
        if gpu_config["batch_size"] <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if gpu_config["chunk_size"] <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)

        # Validate matrix operations settings
        matrix_config = config["m1_matrix_operations"]
        if matrix_config["min_matrix_size_for_gpu"] <= 0:
            msg = "min_matrix_size_for_gpu must be positive"
            raise ValueError(msg)
        if matrix_config["min_batch_size_for_gpu"] <= 0:
            msg = "min_batch_size_for_gpu must be positive"
            raise ValueError(msg)
        if (
            matrix_config["max_gpu_memory_usage"] <= 0
            or matrix_config["max_gpu_memory_usage"] > 1
        ):
            msg = "max_gpu_memory_usage must be between 0 and 1"
            raise ValueError(msg)

        # Validate quality settings
        quality_config = config["m1_gpu"]
        if quality_config["gradient_clip_norm"] <= 0:
            msg = "gradient_clip_norm must be positive"
            raise ValueError(msg)
        if quality_config["cpu_threshold"] <= 0:
            msg = "cpu_threshold must be positive"
            raise ValueError(msg)

        return True

    except Exception as e:
        print(f"M1 configuration validation failed: {e}")
        return False


