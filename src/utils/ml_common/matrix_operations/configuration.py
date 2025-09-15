"""
Unified Configuration System for Matrix Operations

This module provides a single configuration system that consolidates all
matrix and vector operation settings from across the codebase.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Try to import existing configurations
try:
    from ...config.enhanced_matrix_config import get_enhanced_matrix_training_config
    from ...config.matrix_diverse_lookback_config import get_matrix_diverse_lookback_config
    from ...config.m1_gpu_config import get_m1_gpu_config
    EXISTING_CONFIGS_AVAILABLE = True
except ImportError:
    EXISTING_CONFIGS_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    """Optimization targets."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    BALANCED = "balanced"

class HardwareProfile(Enum):
    """Hardware profiles."""
    M1_MAC = "m1_mac"
    M2_MAC = "m2_mac"
    M3_MAC = "m3_mac"
    INTEL_MAC = "intel_mac"
    LINUX_CPU = "linux_cpu"
    LINUX_GPU = "linux_gpu"
    WINDOWS_CPU = "windows_cpu"
    WINDOWS_GPU = "windows_gpu"
    AUTO = "auto"

@dataclass
class HardwareCapabilities:
    """Hardware capabilities detection."""
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_type: Optional[str]
    mps_available: bool
    cuda_available: bool
    platform: str

class UnifiedConfiguration:
    """
    Unified configuration system for all matrix and vector operations.
    
    This class consolidates configuration from:
    - enhanced_matrix_config.py
    - matrix_diverse_lookback_config.py
    - m1_gpu_config.py
    - Environment variables
    - Hardware auto-detection
    """
    
    @classmethod
    def create_optimal_config(cls, 
                             optimization_target: str = "balanced",
                             hardware_profile: str = "auto",
                             data_profile: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create optimal configuration based on requirements and hardware.
        
        Args:
            optimization_target: Optimization target ("performance", "memory", "accuracy", "balanced")
            hardware_profile: Hardware profile ("auto", "m1_mac", "m2_mac", etc.)
            data_profile: Data characteristics for optimization
        
        Returns:
            Unified configuration dictionary
        """
        logger.info(f"🔧 Creating optimal configuration: {optimization_target} on {hardware_profile}")
        
        # Detect hardware capabilities
        hardware_caps = cls._detect_hardware_capabilities()
        
        # Load base configuration
        base_config = cls._load_base_configuration()
        
        # Apply optimization target
        optimized_config = cls._apply_optimization_target(base_config, optimization_target)
        
        # Apply hardware-specific optimizations
        hardware_config = cls._apply_hardware_optimizations(optimized_config, hardware_caps)
        
        # Apply data-specific optimizations
        if data_profile:
            data_config = cls._apply_data_optimizations(hardware_config, data_profile)
        else:
            data_config = hardware_config
        
        # Merge with existing configurations if available
        if EXISTING_CONFIGS_AVAILABLE:
            existing_config = cls._merge_existing_configurations(data_config)
        else:
            existing_config = data_config
        
        # Apply environment overrides
        final_config = cls._apply_environment_overrides(existing_config)
        
        logger.info("✅ Optimal configuration created successfully")
        return final_config
    
    @classmethod
    def _detect_hardware_capabilities(cls) -> HardwareCapabilities:
        """Detect hardware capabilities."""
        import platform
        
        # CPU cores
        if PSUTIL_AVAILABLE:
            cpu_cores = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            # Fallback values
            cpu_cores = 4
            memory_gb = 8.0
        
        # Platform
        platform_name = platform.system().lower()
        
        # GPU detection
        gpu_available = False
        gpu_type = None
        mps_available = False
        cuda_available = False
        
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_available = True
                gpu_type = "mps"
                mps_available = True
            elif torch.cuda.is_available():
                gpu_available = True
                gpu_type = "cuda"
                cuda_available = True
        except ImportError:
            pass
        
        # Detect M1/M2/M3 Macs
        if platform_name == "darwin":
            try:
                import subprocess
                result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
                if 'arm' in result.stdout.lower():
                    if gpu_available and mps_available:
                        # Could be M1, M2, or M3 - we'll use MPS availability as indicator
                        pass
            except:
                pass
        
        return HardwareCapabilities(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_type=gpu_type,
            mps_available=mps_available,
            cuda_available=cuda_available,
            platform=platform_name
        )
    
    @classmethod
    def _load_base_configuration(cls) -> Dict[str, Any]:
        """Load base configuration from existing modules."""
        base_config = {
            # Core settings
            'enable_gpu': True,
            'enable_vectorization': True,
            'enable_parallel_processing': True,
            'enable_memory_optimization': True,
            'optimization_mode': 'balanced',
            
            # Matrix operations
            'matrix_operations': {
                'batch_size': 1000,
                'chunk_size': 5000,
                'gpu_threshold': 5000,
                'precision': 'float32',
                'enable_tiling': True,
                'tile_size': 1000
            },
            
            # Vectorization
            'vectorization': {
                'chunk_size': 50000,
                'max_memory_gb': 8.0,
                'enable_pipeline_execution': True,
                'max_concurrent_stages': 4,
                'enable_async_execution': True
            },
            
            # Cross-validation
            'cross_validation': {
                'n_splits': 5,
                'enable_parallel': True,
                'max_workers': 4,
                'enable_gpu_acceleration': True
            },
            
            # Memory management
            'memory': {
                'chunk_size_mb': 256,
                'max_memory_percent': 0.7,
                'enable_cleanup': True,
                'enable_pooling': True
            },
            
            # Performance
            'performance': {
                'enable_monitoring': True,
                'enable_profiling': True,
                'enable_auto_optimization': True,
                'log_level': 'INFO'
            }
        }
        
        # Merge with existing configurations if available
        if EXISTING_CONFIGS_AVAILABLE:
            try:
                # Enhanced matrix config
                enhanced_config = get_enhanced_matrix_training_config()
                base_config.update(enhanced_config)
                
                # M1 GPU config
                m1_config = get_m1_gpu_config()
                base_config.update(m1_config)
                
                # Matrix diverse lookback config
                matrix_config = get_matrix_diverse_lookback_config()
                base_config.update(matrix_config)
                
            except Exception as e:
                logger.warning(f"⚠️ Error loading existing configurations: {e}")
        
        return base_config
    
    @classmethod
    def _apply_optimization_target(cls, config: Dict[str, Any], target: str) -> Dict[str, Any]:
        """Apply optimization target settings."""
        config['optimization_target'] = target
        
        if target == "performance":
            config.update({
                'matrix_operations': {
                    **config.get('matrix_operations', {}),
                    'batch_size': 2000,
                    'chunk_size': 10000,
                    'gpu_threshold': 5000,
                    'precision': 'float32'
                },
                'vectorization': {
                    **config.get('vectorization', {}),
                    'chunk_size': 100000,
                    'max_concurrent_stages': 8
                },
                'cross_validation': {
                    **config.get('cross_validation', {}),
                    'max_workers': 8
                },
                'memory': {
                    **config.get('memory', {}),
                    'chunk_size_mb': 512,
                    'max_memory_percent': 0.9
                }
            })
            
        elif target == "memory":
            config.update({
                'matrix_operations': {
                    **config.get('matrix_operations', {}),
                    'batch_size': 500,
                    'chunk_size': 2000,
                    'gpu_threshold': 10000,
                    'precision': 'float16'
                },
                'vectorization': {
                    **config.get('vectorization', {}),
                    'chunk_size': 25000,
                    'max_memory_gb': 4.0
                },
                'memory': {
                    **config.get('memory', {}),
                    'chunk_size_mb': 128,
                    'max_memory_percent': 0.5,
                    'enable_cleanup': True
                }
            })
            
        elif target == "accuracy":
            config.update({
                'matrix_operations': {
                    **config.get('matrix_operations', {}),
                    'precision': 'float64',
                    'batch_size': 1000,
                    'chunk_size': 5000
                },
                'vectorization': {
                    **config.get('vectorization', {}),
                    'chunk_size': 50000
                },
                'cross_validation': {
                    **config.get('cross_validation', {}),
                    'n_splits': 10
                }
            })
        
        return config
    
    @classmethod
    def _apply_hardware_optimizations(cls, config: Dict[str, Any], 
                                    hardware_caps: HardwareCapabilities) -> Dict[str, Any]:
        """Apply hardware-specific optimizations."""
        
        # GPU settings
        if hardware_caps.gpu_available:
            config['enable_gpu'] = True
            if hardware_caps.mps_available:
                config['gpu_type'] = 'mps'
                config['enable_mps'] = True
            elif hardware_caps.cuda_available:
                config['gpu_type'] = 'cuda'
                config['enable_cuda'] = True
        else:
            config['enable_gpu'] = False
            config['gpu_threshold'] = float('inf')  # Never use GPU
        
        # CPU optimization
        config['max_workers'] = min(hardware_caps.cpu_cores, 8)
        config['vectorization']['max_concurrent_stages'] = min(hardware_caps.cpu_cores, 8)
        
        # Memory optimization
        memory_gb = hardware_caps.memory_gb
        if memory_gb < 8:
            # Low memory system
            config['memory']['chunk_size_mb'] = 128
            config['memory']['max_memory_percent'] = 0.5
            config['matrix_operations']['batch_size'] = 500
        elif memory_gb < 16:
            # Medium memory system
            config['memory']['chunk_size_mb'] = 256
            config['memory']['max_memory_percent'] = 0.7
            config['matrix_operations']['batch_size'] = 1000
        else:
            # High memory system
            config['memory']['chunk_size_mb'] = 512
            config['memory']['max_memory_percent'] = 0.8
            config['matrix_operations']['batch_size'] = 2000
        
        # Platform-specific optimizations
        if hardware_caps.platform == "darwin" and hardware_caps.mps_available:
            # M1/M2/M3 Mac optimizations
            config['enable_metal_performance_shaders'] = True
            config['enable_unified_memory'] = True
            config['matrix_operations']['precision'] = 'float32'  # MPS works best with float32
        
        return config
    
    @classmethod
    def _apply_data_optimizations(cls, config: Dict[str, Any], 
                                data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data-specific optimizations."""
        
        # Data size optimizations
        if 'n_samples' in data_profile:
            n_samples = data_profile['n_samples']
            if n_samples > 100000:
                # Large dataset
                config['matrix_operations']['batch_size'] = min(2000, config['matrix_operations']['batch_size'])
                config['memory']['chunk_size_mb'] = min(256, config['memory']['chunk_size_mb'])
            elif n_samples < 10000:
                # Small dataset
                config['matrix_operations']['batch_size'] = max(500, config['matrix_operations']['batch_size'])
        
        # Feature count optimizations
        if 'n_features' in data_profile:
            n_features = data_profile['n_features']
            if n_features > 1000:
                # High-dimensional data
                config['matrix_operations']['gpu_threshold'] = 1000  # Use GPU earlier
                config['vectorization']['chunk_size'] = min(25000, config['vectorization']['chunk_size'])
        
        # Data type optimizations
        if 'dtype' in data_profile:
            dtype = data_profile['dtype']
            if dtype in ['float16', 'int16']:
                config['matrix_operations']['precision'] = 'float16'
            elif dtype in ['float64', 'int64']:
                config['matrix_operations']['precision'] = 'float64'
        
        return config
    
    @classmethod
    def _merge_existing_configurations(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge with existing configuration modules."""
        try:
            # Enhanced matrix config
            enhanced_config = get_enhanced_matrix_training_config()
            config.update(enhanced_config)
            
            # M1 GPU config
            m1_config = get_m1_gpu_config()
            config.update(m1_config)
            
            # Matrix diverse lookback config
            matrix_config = get_matrix_diverse_lookback_config()
            config.update(matrix_config)
            
        except Exception as e:
            logger.warning(f"⚠️ Error merging existing configurations: {e}")
        
        return config
    
    @classmethod
    def _apply_environment_overrides(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        
        # GPU settings
        if os.getenv('ARES_DISABLE_GPU', '').lower() in ['true', '1', 'yes']:
            config['enable_gpu'] = False
        
        if os.getenv('ARES_FORCE_GPU', '').lower() in ['true', '1', 'yes']:
            config['enable_gpu'] = True
            config['gpu_threshold'] = 0
        
        # Memory settings
        memory_limit_gb = os.getenv('ARES_MEMORY_LIMIT_GB')
        if memory_limit_gb:
            try:
                limit_gb = float(memory_limit_gb)
                config['memory']['max_memory_gb'] = limit_gb
                config['memory']['max_memory_percent'] = 0.8
            except ValueError:
                pass
        
        # CPU settings
        max_workers = os.getenv('ARES_MAX_WORKERS')
        if max_workers:
            try:
                workers = int(max_workers)
                config['max_workers'] = workers
                config['vectorization']['max_concurrent_stages'] = workers
            except ValueError:
                pass
        
        # Precision settings
        precision = os.getenv('ARES_PRECISION_POLICY')
        if precision in ['float16', 'float32', 'float64']:
            config['matrix_operations']['precision'] = precision
        
        # Logging settings
        log_level = os.getenv('ARES_LOG_LEVEL', 'INFO')
        config['performance']['log_level'] = log_level
        
        return config
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration."""
        return cls.create_optimal_config("balanced", "auto")
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance-optimized configuration."""
        return cls.create_optimal_config("performance", "auto")
    
    @classmethod
    def get_memory_config(cls) -> Dict[str, Any]:
        """Get memory-optimized configuration."""
        return cls.create_optimal_config("memory", "auto")
    
    @classmethod
    def get_accuracy_config(cls) -> Dict[str, Any]:
        """Get accuracy-optimized configuration."""
        return cls.create_optimal_config("accuracy", "auto")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration and return issues."""
        issues = []
        
        # Check required fields
        required_fields = ['enable_gpu', 'enable_vectorization', 'enable_parallel_processing']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Check numeric ranges
        if 'matrix_operations' in config:
            matrix_ops = config['matrix_operations']
            if 'batch_size' in matrix_ops and matrix_ops['batch_size'] <= 0:
                issues.append("batch_size must be positive")
            if 'chunk_size' in matrix_ops and matrix_ops['chunk_size'] <= 0:
                issues.append("chunk_size must be positive")
        
        if 'memory' in config:
            memory = config['memory']
            if 'max_memory_percent' in memory:
                percent = memory['max_memory_percent']
                if not (0 < percent <= 1):
                    issues.append("max_memory_percent must be between 0 and 1")
        
        return len(issues) == 0, issues
    
    @classmethod
    def optimize_config_for_data(cls, config: Dict[str, Any], 
                               data_shape: Tuple[int, ...],
                               data_type: str = "float32") -> Dict[str, Any]:
        """Optimize configuration for specific data characteristics."""
        
        optimized_config = config.copy()
        
        # Data size optimization
        total_elements = 1
        for dim in data_shape:
            total_elements *= dim
        
        if total_elements > 10_000_000:  # Large data
            optimized_config['matrix_operations']['batch_size'] = min(1000, optimized_config['matrix_operations']['batch_size'])
            optimized_config['memory']['chunk_size_mb'] = min(128, optimized_config['memory']['chunk_size_mb'])
        elif total_elements < 100_000:  # Small data
            optimized_config['matrix_operations']['batch_size'] = max(100, optimized_config['matrix_operations']['batch_size'])
            optimized_config['enable_gpu'] = False  # GPU overhead not worth it
        
        # Data type optimization
        if data_type in ['float16', 'int16']:
            optimized_config['matrix_operations']['precision'] = 'float16'
        elif data_type in ['float64', 'int64']:
            optimized_config['matrix_operations']['precision'] = 'float64'
        
        return optimized_config