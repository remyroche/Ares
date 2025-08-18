# src/config/enhanced_matrix_config.py

"""
Configuration for enhanced matrix operations integration in the training pipeline.
Provides comprehensive settings for GPU acceleration and matrix operations.
"""

from typing import Dict, Any
from src.config.m1_gpu_config import get_m1_gpu_config, get_optimized_m1_config


def get_enhanced_matrix_training_config() -> Dict[str, Any]:
    """
    Get comprehensive configuration for enhanced matrix operations in training pipeline.
    
    Returns:
        dict: Complete configuration for enhanced matrix operations
    """
    
    # Get base M1 GPU configuration
    m1_config = get_m1_gpu_config()
    
    # Enhanced matrix operations configuration
    enhanced_matrix_config = {
        # Enable enhanced matrix operations
        "enable_enhanced_matrix_operations": True,
        
        # Matrix optimization modes
        "matrix_optimization_mode": "performance",  # "performance", "memory", "accuracy", "stability"
        "model_training_optimization_mode": "accuracy",  # For model training steps
        
        # GPU acceleration settings
        "enable_gpu_acceleration": True,
        "enable_mps": True,
        "enable_metal_performance_shaders": True,
        "enable_mixed_precision": True,
        
        # Memory management
        "gpu_memory_fraction": 0.8,
        "max_gpu_memory_gb": 8.0,
        "enable_memory_pooling": True,
        "enable_memory_cleanup": True,
        
        # Performance settings
        "batch_size": 1000,
        "chunk_size": 5000,
        "enable_parallel_processing": True,
        "enable_batch_processing": True,
        
        # Quality settings
        "enable_numerical_stability": True,
        "enable_gradient_clipping": True,
        "gradient_clip_norm": 1.0,
        "enable_batch_norm": True,
        
        # Fallback settings
        "enable_cpu_fallback": True,
        "cpu_threshold": 10000,
        "enable_automatic_fallback": True,
        
        # Matrix operations settings
        "enable_gpu_svd": True,
        "enable_gpu_eigenvalue": True,
        "enable_gpu_matrix_multiply": True,
        "enable_gpu_batch_operations": True,
        "enable_gpu_neural_networks": True,
        "enable_gpu_training": True,
        "enable_gpu_inference": True,
        
        # Performance thresholds
        "min_matrix_size_for_gpu": 100,
        "min_batch_size_for_gpu": 50,
        "max_gpu_memory_usage": 0.8,
        
        # Quality thresholds
        "numerical_precision": 1e-6,
        "convergence_tolerance": 1e-8,
        "max_iterations": 1000,
        
        # Security settings
        "enable_gpu_data_encryption": True,
        "enable_memory_isolation": True,
        "enable_secure_computation": True,
        "enable_gpu_monitoring": True,
        "enable_memory_monitoring": True,
        "enable_performance_monitoring": True,
        "enable_gpu_quality_gates": True,
        "enable_result_validation": True,
        "enable_error_detection": True,
        
        # Integration settings
        "enable_step_2_5_enhancement": True,  # Enhanced matrix operations after feature engineering
        "enable_step_5_5_enhancement": True,  # Enhanced matrix operations before model training
        
        # Logging and monitoring
        "enable_enhanced_logging": True,
        "enable_performance_tracking": True,
        "enable_quality_monitoring": True,
        
        # Error handling
        "max_retry_attempts": 3,
        "retry_delay_seconds": 1.0,
        "enable_graceful_degradation": True,
    }
    
    # Merge M1 GPU configuration
    enhanced_matrix_config.update(m1_config)
    
    return enhanced_matrix_config


def get_optimized_enhanced_matrix_config(optimization_target: str = "performance") -> Dict[str, Any]:
    """
    Get optimized configuration for enhanced matrix operations.
    
    Args:
        optimization_target: Optimization target ("performance", "memory", "accuracy", "stability")
        
    Returns:
        dict: Optimized configuration
    """
    
    base_config = get_enhanced_matrix_training_config()
    
    if optimization_target == "performance":
        # Optimize for maximum performance
        base_config.update({
            "matrix_optimization_mode": "performance",
            "model_training_optimization_mode": "performance",
            "batch_size": 2000,
            "chunk_size": 10000,
            "cpu_threshold": 5000,
            "gpu_memory_fraction": 0.9,
            "min_matrix_size_for_gpu": 50,
            "min_batch_size_for_gpu": 25,
            "enable_mixed_precision": True,
            "enable_parallel_processing": True,
            "enable_batch_processing": True,
        })
        
    elif optimization_target == "memory":
        # Optimize for memory efficiency
        base_config.update({
            "matrix_optimization_mode": "memory",
            "model_training_optimization_mode": "memory",
            "batch_size": 500,
            "chunk_size": 2000,
            "gpu_memory_fraction": 0.6,
            "enable_memory_cleanup": True,
            "enable_memory_pooling": True,
            "min_matrix_size_for_gpu": 200,
            "min_batch_size_for_gpu": 100,
            "enable_memory_monitoring": True,
        })
        
    elif optimization_target == "accuracy":
        # Optimize for accuracy
        base_config.update({
            "matrix_optimization_mode": "accuracy",
            "model_training_optimization_mode": "accuracy",
            "numerical_precision": 1e-8,
            "convergence_tolerance": 1e-10,
            "max_iterations": 2000,
            "enable_numerical_stability": True,
            "enable_gradient_clipping": True,
            "gradient_clip_norm": 0.5,
            "enable_result_validation": True,
        })
        
    elif optimization_target == "stability":
        # Optimize for stability
        base_config.update({
            "matrix_optimization_mode": "stability",
            "model_training_optimization_mode": "stability",
            "enable_cpu_fallback": True,
            "enable_automatic_fallback": True,
            "max_retry_attempts": 5,
            "enable_gpu_quality_gates": True,
            "enable_result_validation": True,
            "enable_numerical_stability": True,
            "enable_error_detection": True,
            "enable_graceful_degradation": True,
        })
    
    return base_config


def get_production_enhanced_matrix_config() -> Dict[str, Any]:
    """
    Get production-ready configuration for enhanced matrix operations.
    
    Returns:
        dict: Production configuration
    """
    
    config = get_enhanced_matrix_training_config()
    
    # Production optimizations
    config.update({
        "enable_memory_cleanup": True,
        "enable_automatic_fallback": True,
        "cpu_threshold": 5000,
        
        # Security enhancements
        "enable_gpu_data_encryption": True,
        "enable_memory_isolation": True,
        "enable_gpu_quality_gates": True,
        "enable_result_validation": True,
        
        # Quality enhancements
        "enable_numerical_stability": True,
        "enable_gradient_clipping": True,
        "numerical_precision": 1e-7,
        "convergence_tolerance": 1e-9,
        
        # Monitoring enhancements
        "enable_enhanced_logging": True,
        "enable_performance_tracking": True,
        "enable_quality_monitoring": True,
        "enable_gpu_monitoring": True,
        "enable_memory_monitoring": True,
        "enable_performance_monitoring": True,
        
        # Error handling
        "enable_error_detection": True,
        "enable_graceful_degradation": True,
        "max_retry_attempts": 3,
    })
    
    return config


def get_minimal_enhanced_matrix_config() -> Dict[str, Any]:
    """
    Get minimal configuration for enhanced matrix operations.
    
    Returns:
        dict: Minimal configuration
    """
    
    config = get_enhanced_matrix_training_config()
    
    # Disable advanced features
    config.update({
        "enable_gpu_acceleration": False,
        "enable_mps": False,
        "enable_metal_performance_shaders": False,
        "enable_gpu_neural_networks": False,
        "enable_gpu_training": False,
        "enable_parallel_processing": False,
        "enable_batch_processing": False,
        
        # Reduce memory usage
        "gpu_memory_fraction": 0.5,
        "max_gpu_memory_gb": 4.0,
        
        # Increase CPU threshold
        "cpu_threshold": 50000,
        "min_matrix_size_for_gpu": 500,
        "min_batch_size_for_gpu": 200,
        
        # Disable advanced features
        "enable_step_2_5_enhancement": False,
        "enable_step_5_5_enhancement": False,
    })
    
    return config


def validate_enhanced_matrix_config(config: Dict[str, Any]) -> bool:
    """
    Validate enhanced matrix operations configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    
    try:
        # Check required settings
        required_settings = [
            "enable_enhanced_matrix_operations",
            "matrix_optimization_mode",
            "model_training_optimization_mode",
            "enable_gpu_acceleration",
            "batch_size",
            "chunk_size",
            "cpu_threshold",
        ]
        
        for setting in required_settings:
            if setting not in config:
                print(f"Missing required setting: {setting}")
                return False
        
        # Validate optimization modes
        valid_modes = ["performance", "memory", "accuracy", "stability"]
        if config["matrix_optimization_mode"] not in valid_modes:
            print(f"Invalid matrix_optimization_mode: {config['matrix_optimization_mode']}")
            return False
        
        if config["model_training_optimization_mode"] not in valid_modes:
            print(f"Invalid model_training_optimization_mode: {config['model_training_optimization_mode']}")
            return False
        
        # Validate numeric settings
        if config["batch_size"] <= 0:
            print("batch_size must be positive")
            return False
        
        if config["chunk_size"] <= 0:
            print("chunk_size must be positive")
            return False
        
        if config["cpu_threshold"] <= 0:
            print("cpu_threshold must be positive")
            return False
        
        if config["gpu_memory_fraction"] <= 0 or config["gpu_memory_fraction"] > 1:
            print("gpu_memory_fraction must be between 0 and 1")
            return False
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def get_default_enhanced_matrix_config() -> Dict[str, Any]:
    """
    Get default configuration for enhanced matrix operations.
    
    Returns:
        dict: Default configuration
    """
    return get_enhanced_matrix_training_config()


def get_enhanced_matrix_config_for_training_type(training_type: str) -> Dict[str, Any]:
    """
    Get configuration optimized for specific training type.
    
    Args:
        training_type: Type of training ("quick", "standard", "thorough", "production")
        
    Returns:
        dict: Optimized configuration
    """
    
    if training_type == "quick":
        return get_optimized_enhanced_matrix_config("performance")
    elif training_type == "standard":
        return get_enhanced_matrix_training_config()
    elif training_type == "thorough":
        return get_optimized_enhanced_matrix_config("accuracy")
    elif training_type == "production":
        return get_production_enhanced_matrix_config()
    else:
        return get_enhanced_matrix_training_config()