# src/config/m1_gpu_config.py

"""
Configuration for Mac M1 GPU acceleration and optimization.
Provides comprehensive settings for Metal Performance Shaders (MPS) integration.
"""

from typing import Any
from dataclasses import dataclass


import @dataclass
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


def get_m1_gpu_config() -> dict[str, Any]:
    pass
    pass
    """Get comprehensive configuration for M1 GPU acceleration."""

    return {
        "m1_gpu": M1GPUConfig(),
        "m1_matrix_operations": M1MatrixOperationsConfig(),
        "m1_security": M1SecurityConfig(),
        # GPU acceleration settings
        "gpu_acceleration": {
            "enable_mps": True,
            "enable_metal_performance_shaders": True,
            "enable_mixed_precision": True,
            "enable_tensor_cores": True,
            "enable_fusion_operations": True,
            "enable_memory_optimization": True,
            "enable_compute_optimization": True,
        },
        # Matrix operations settings
        "matrix_operations": {
            "enable_gpu_svd": True,
            "enable_gpu_eigenvalue": True,
            "enable_gpu_matrix_multiply": True,
            "enable_gpu_batch_operations": True,
            "enable_gpu_neural_networks": True,
            "enable_gpu_training": True,
            "enable_gpu_inference": True,
            "min_matrix_size_for_gpu": 100,
            "min_batch_size_for_gpu": 50,
            "max_gpu_memory_usage": 0.8,
        },
        # Performance optimization settings
        "performance_optimization": {
            "batch_size": 1000,
            "chunk_size": 5000,
            "enable_parallel_processing": True,
            "enable_batch_processing": True,
            "enable_memory_pooling": True,
            "enable_memory_cleanup": True,
            "cpu_threshold": 10000,
            "enable_automatic_fallback": True,
        },
        # Quality and stability settings
        "quality_stability": {
            "enable_numerical_stability": True,
            "enable_gradient_clipping": True,
            "gradient_clip_norm": 1.0,
            "enable_batch_norm": True,
            "numerical_precision": 1e-6,
            "convergence_tolerance": 1e-8,
            "max_iterations": 1000,
        },
        # Security settings
        "security": {
            "enable_gpu_data_encryption": True,
            "enable_memory_isolation": True,
            "enable_secure_computation": True,
            "enable_gpu_monitoring": True,
            "enable_memory_monitoring": True,
            "enable_performance_monitoring": True,
            "enable_gpu_quality_gates": True,
            "enable_result_validation": True,
            "enable_error_detection": True,
        },
        # Memory management settings
        "memory_management": {
            "gpu_memory_fraction": 0.8,
            "max_gpu_memory_gb": 8.0,
            "enable_memory_pooling": True,
            "enable_memory_cleanup": True,
            "enable_memory_isolation": True,
            "enable_memory_monitoring": True,
        },
        # Fallback and error handling
        "fallback_error_handling": {
            "enable_cpu_fallback": True,
            "enable_automatic_fallback": True,
            "cpu_threshold": 10000,
            "enable_error_detection": True,
            "enable_result_validation": True,
            "max_retry_attempts": 3,
            "retry_delay_seconds": 1.0,
        },
    }


def get_optimized_m1_config(optimization_target: str = "performance") -> dict[str, Any]:
    pass
    pass
    """Get M1 configuration optimized for specific target."""

    base_config = get_m1_gpu_config()

    if optimization_target == "performance":
    pass
    pass
        # Optimize for maximum performance
        base_config["performance_optimization"]["batch_size"] = 2000
        base_config["performance_optimization"]["chunk_size"] = 10000
        base_config["performance_optimization"]["cpu_threshold"] = 5000
        base_config["memory_management"]["gpu_memory_fraction"] = 0.9
        base_config["matrix_operations"]["min_matrix_size_for_gpu"] = 50
        base_config["matrix_operations"]["min_batch_size_for_gpu"] = 25

    elif optimization_target == "memory":
        # Optimize for memory efficiency
        base_config["performance_optimization"]["batch_size"] = 500
        base_config["performance_optimization"]["chunk_size"] = 2000
        base_config["memory_management"]["gpu_memory_fraction"] = 0.6
        base_config["memory_management"]["enable_memory_cleanup"] = True
        base_config["performance_optimization"]["enable_memory_pooling"] = True
        base_config["matrix_operations"]["min_matrix_size_for_gpu"] = 200
        base_config["matrix_operations"]["min_batch_size_for_gpu"] = 100

    elif optimization_target == "accuracy":
        # Optimize for accuracy
        base_config["quality_stability"]["numerical_precision"] = 1e-8
        base_config["quality_stability"]["convergence_tolerance"] = 1e-10
        base_config["quality_stability"]["max_iterations"] = 2000
        base_config["quality_stability"]["enable_numerical_stability"] = True
        base_config["quality_stability"]["enable_gradient_clipping"] = True
        base_config["quality_stability"]["gradient_clip_norm"] = 0.5

    elif optimization_target == "stability":
        # Optimize for stability
        base_config["fallback_error_handling"]["enable_cpu_fallback"] = True
        base_config["fallback_error_handling"]["enable_automatic_fallback"] = True
        base_config["fallback_error_handling"]["max_retry_attempts"] = 5
        base_config["security"]["enable_gpu_quality_gates"] = True
        base_config["security"]["enable_result_validation"] = True
        base_config["quality_stability"]["enable_numerical_stability"] = True

    return base_config


def get_minimal_m1_config() -> dict[str, Any]:
    pass
    pass
    """Get minimal M1 configuration for basic GPU operations."""

    config = get_m1_gpu_config()

    # Disable advanced features
    config["gpu_acceleration"]["enable_tensor_cores"] = False
    config["gpu_acceleration"]["enable_fusion_operations"] = False
    config["matrix_operations"]["enable_gpu_neural_networks"] = False
    config["matrix_operations"]["enable_gpu_training"] = False
    config["performance_optimization"]["enable_parallel_processing"] = False
    config["performance_optimization"]["enable_batch_processing"] = False

    # Reduce memory usage
    config["memory_management"]["gpu_memory_fraction"] = 0.5
    config["memory_management"]["max_gpu_memory_gb"] = 4.0

    # Increase CPU threshold
    config["performance_optimization"]["cpu_threshold"] = 50000
    config["matrix_operations"]["min_matrix_size_for_gpu"] = 500
    config["matrix_operations"]["min_batch_size_for_gpu"] = 200

    return config


def validate_m1_config(config: dict[str, Any]) -> bool:
    pass
    pass
    """Validate M1 GPU configuration settings."""

    try:
        # Check required sections
    except Exception as e:
        pass
    except Exception as e:
        pass
        required_sections = ["m1_gpu", "m1_matrix_operations", "m1_security"]
        for section in required_sections:
    pass
    pass
            if section not in config:
    pass
    pass
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
    pass
    pass
            msg = "max_gpu_memory_gb must be positive"
            raise ValueError(msg)
        if gpu_config["batch_size"] <= 0:
    pass
    pass
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if gpu_config["chunk_size"] <= 0:
    pass
    pass
            msg = "chunk_size must be positive"
            raise ValueError(msg)

        # Validate matrix operations settings
        matrix_config = config["m1_matrix_operations"]
        if matrix_config["min_matrix_size_for_gpu"] <= 0:
    pass
    pass
            msg = "min_matrix_size_for_gpu must be positive"
            raise ValueError(msg)
        if matrix_config["min_batch_size_for_gpu"] <= 0:
    pass
    pass
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
    pass
    pass
            msg = "gradient_clip_norm must be positive"
            raise ValueError(msg)
        if quality_config["cpu_threshold"] <= 0:
    pass
    pass
            msg = "cpu_threshold must be positive"
            raise ValueError(msg)

        return True

    except Exception as e:
        print(f"M1 configuration validation failed: {e}")
        return False


def get_default_m1_config() -> dict[str, Any]:
    pass
    pass
    """Get default M1 configuration for GPU acceleration."""
    return get_m1_gpu_config()


def get_production_m1_config() -> dict[str, Any]:
    pass
    pass
    """Get production-ready M1 configuration."""

    config = get_m1_gpu_config()

    # Production optimizations
    config["m1_gpu"]["enable_memory_cleanup"] = True
    config["m1_gpu"]["enable_automatic_fallback"] = True
    config["m1_gpu"]["cpu_threshold"] = 5000

    # Security enhancements
    config["m1_security"]["enable_gpu_data_encryption"] = True
    config["m1_security"]["enable_memory_isolation"] = True
    config["m1_security"]["enable_gpu_quality_gates"] = True
    config["m1_security"]["enable_result_validation"] = True

    # Quality enhancements
    config["m1_gpu"]["enable_numerical_stability"] = True
    config["m1_gpu"]["enable_gradient_clipping"] = True
    config["m1_matrix_operations"]["numerical_precision"] = 1e-7
    config["m1_matrix_operations"]["convergence_tolerance"] = 1e-9

    return config
