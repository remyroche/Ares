"""
Hardware Optimizer for Regime Detection Systems.

This module provides hardware optimization utilities that can be used by both
NAS and TAS regime detection systems to optimize performance based on available
hardware resources.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from src.utils.logger import system_logger


@dataclass
class HardwareConfig:
    """Configuration for hardware optimization."""
    max_memory_gb: float = 8.0
    max_cores: int = 4
    enable_gpu: bool = False
    gpu_memory_gb: float = 0.0
    optimization_target: str = 'balanced'  # 'speed', 'memory', 'balanced'
    adaptive_scaling: bool = True
    memory_safety_margin: float = 0.1  # 10% safety margin


@dataclass
class HardwareProfile:
    """Profile of available hardware."""
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    architecture: str
    cache_sizes: Dict[str, int]


class HardwareOptimizer:
    """
    Hardware optimizer for regime detection systems.

    This class provides hardware-aware optimization that can be used by both
    NAS and TAS systems to optimize performance based on available hardware
    resources and constraints.
    """

    def __init__(self, config: HardwareConfig):
        """
        Initialize the hardware optimizer.

        Args:
            config: Hardware optimization configuration
        """
        self.logger = system_logger.getChild('HardwareOptimizer')
        self.config = config

        # Hardware profile detection
        self.hardware_profile = self._detect_hardware_profile()

        # Optimization state
        self.current_optimizations = {}
        self.performance_metrics = {}

        self.logger.info("✅ Hardware Optimizer initialized"
        self.logger.info(f"   CPU cores: {self.hardware_profile.cpu_cores}")
        self.logger.info(f"   Memory: {self.hardware_profile.memory_gb".1f"} GB")
        self.logger.info(f"   GPU available: {self.hardware_profile.gpu_available}")
        self.logger.info(f"   Optimization target: {config.optimization_target}")

    def _detect_hardware_profile(self) -> HardwareProfile:
        """
        Detect available hardware resources.

        Returns:
            Hardware profile
        """
        try:
            import psutil
            import platform

            # CPU information
            cpu_cores = psutil.cpu_count(logical=False) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # GPU information (simplified detection)
            gpu_available = False
            gpu_memory_gb = 0.0

            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_available = True
                    gpu_memory_gb = gpus[0].memoryTotal / 1024  # Convert to GB
            except ImportError:
                # GPU detection not available
                pass

            # Architecture information
            architecture = platform.machine()
            cache_sizes = self._detect_cache_sizes()

            profile = HardwareProfile(
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                gpu_available=gpu_available,
                gpu_memory_gb=gpu_memory_gb,
                architecture=architecture,
                cache_sizes=cache_sizes
            )

            self.logger.info(f"📊 Hardware profile detected: {cpu_cores} cores, {memory_gb".1f"} GB RAM")
            if gpu_available:
                self.logger.info(f"🖥️ GPU available: {gpu_memory_gb".1f"} GB")

            return profile

        except Exception as e:
            self.logger.warning(f"⚠️ Hardware detection failed, using defaults: {e}")

            # Fallback profile
            return HardwareProfile(
                cpu_cores=4,
                memory_gb=8.0,
                gpu_available=False,
                gpu_memory_gb=0.0,
                architecture='x86_64',
                cache_sizes={'L1': 32768, 'L2': 262144, 'L3': 8388608}
            )

    def _detect_cache_sizes(self) -> Dict[str, int]:
        """
        Detect CPU cache sizes.

        Returns:
            Dictionary of cache sizes
        """
        try:
            # This is a simplified cache detection
            # In practice, you would use more sophisticated detection methods

            cache_sizes = {
                'L1': 32 * 1024,      # 32 KB L1 cache (typical)
                'L2': 256 * 1024,     # 256 KB L2 cache (typical)
                'L3': 8 * 1024 * 1024 # 8 MB L3 cache (typical)
            }

            return cache_sizes

        except Exception as e:
            self.logger.warning(f"⚠️ Cache size detection failed: {e}")
            return {'L1': 32768, 'L2': 262144, 'L3': 8388608}

    def optimize_for_hardware(self, algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize algorithm configuration for available hardware.

        Args:
            algorithm_config: Original algorithm configuration

        Returns:
            Hardware-optimized configuration
        """
        try:
            self.logger.info("🔧 Optimizing configuration for hardware")

            optimized_config = algorithm_config.copy()

            # Memory optimization
            optimized_config = self._optimize_memory_usage(optimized_config)

            # CPU optimization
            optimized_config = self._optimize_cpu_usage(optimized_config)

            # GPU optimization
            if self.hardware_profile.gpu_available:
                optimized_config = self._optimize_gpu_usage(optimized_config)

            # Cache optimization
            optimized_config = self._optimize_cache_usage(optimized_config)

            self.logger.info(f"✅ Hardware optimization completed for {self.config.optimization_target} target")
            return optimized_config

        except Exception as e:
            self.logger.error(f"❌ Hardware optimization failed: {e}")
            return algorithm_config

    def _optimize_memory_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration for memory usage.

        Args:
            config: Algorithm configuration

        Returns:
            Memory-optimized configuration
        """
        try:
            optimized_config = config.copy()

            # Calculate available memory
            available_memory = self.hardware_profile.memory_gb * (1 - self.config.memory_safety_margin)

            # Adjust batch sizes based on memory
            if 'batch_size' in config:
                max_batch_size = self._estimate_max_batch_size(available_memory)
                optimized_config['batch_size'] = min(config['batch_size'], max_batch_size)

            # Adjust data loading based on memory
            if 'preload_data' in config:
                if available_memory < 4.0:  # Low memory
                    optimized_config['preload_data'] = False
                    optimized_config['chunk_size'] = min(config.get('chunk_size', 1000), 500)

            # Adjust model complexity based on memory
            if 'max_complexity' in config:
                memory_factor = available_memory / 8.0  # Assuming 8GB is baseline
                optimized_config['max_complexity'] = int(config['max_complexity'] * memory_factor)

            self.logger.debug(f"💾 Memory optimization: available={available_memory".1f"} GB")
            return optimized_config

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return config

    def _estimate_max_batch_size(self, available_memory_gb: float) -> int:
        """
        Estimate maximum batch size based on available memory.

        Args:
            available_memory_gb: Available memory in GB

        Returns:
            Maximum recommended batch size
        """
        try:
            # Rough estimation: assume each sample uses ~1KB of memory
            # This is highly dependent on the specific algorithm and data
            max_samples = int(available_memory_gb * 1024 * 1024 * 0.5)  # Use 50% of memory for batch

            # Common batch sizes: 16, 32, 64, 128, 256, 512, 1024
            batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
            max_batch_size = max([bs for bs in batch_sizes if bs <= max_samples] + [16])

            return max_batch_size

        except Exception as e:
            self.logger.warning(f"⚠️ Batch size estimation failed: {e}")
            return 32  # Safe default

    def _optimize_cpu_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration for CPU usage.

        Args:
            config: Algorithm configuration

        Returns:
            CPU-optimized configuration
        """
        try:
            optimized_config = config.copy()

            # Adjust parallel processing based on CPU cores
            if 'n_jobs' in config or 'parallel' in config:
                max_cores = self.hardware_profile.cpu_cores
                if 'n_jobs' in config:
                    optimized_config['n_jobs'] = min(config['n_jobs'], max_cores)
                if 'parallel' in config and config['parallel']:
                    optimized_config['n_jobs'] = max_cores

            # Adjust population sizes for evolutionary algorithms
            if 'population_size' in config:
                cpu_factor = self.hardware_profile.cpu_cores / 4  # Assuming 4 cores is baseline
                optimized_config['population_size'] = int(config['population_size'] * cpu_factor)

            # Adjust iteration counts based on optimization target
            if self.config.optimization_target == 'speed':
                if 'max_iterations' in config:
                    optimized_config['max_iterations'] = int(config['max_iterations'] * 0.7)
                if 'n_generations' in config:
                    optimized_config['n_generations'] = int(config['n_generations'] * 0.7)

            self.logger.debug(f"🖥️ CPU optimization: cores={self.hardware_profile.cpu_cores}")
            return optimized_config

        except Exception as e:
            self.logger.warning(f"⚠️ CPU optimization failed: {e}")
            return config

    def _optimize_gpu_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration for GPU usage.

        Args:
            config: Algorithm configuration

        Returns:
            GPU-optimized configuration
        """
        try:
            optimized_config = config.copy()

            # Enable GPU acceleration if available
            if 'use_gpu' in config:
                optimized_config['use_gpu'] = True

            # Adjust batch size for GPU memory
            if 'batch_size' in config:
                gpu_memory_factor = self.hardware_profile.gpu_memory_gb / 8.0  # Assuming 8GB is baseline
                optimized_config['batch_size'] = int(config['batch_size'] * gpu_memory_factor)

            # Adjust model complexity for GPU
            if 'max_complexity' in config:
                gpu_factor = self.hardware_profile.gpu_memory_gb / 8.0
                optimized_config['max_complexity'] = int(config['max_complexity'] * gpu_factor)

            self.logger.debug(f"🖥️ GPU optimization: memory={self.hardware_profile.gpu_memory_gb".1f"} GB")
            return optimized_config

        except Exception as e:
            self.logger.warning(f"⚠️ GPU optimization failed: {e}")
            return config

    def _optimize_cache_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration for cache usage.

        Args:
            config: Algorithm configuration

        Returns:
            Cache-optimized configuration
        """
        try:
            optimized_config = config.copy()

            # Optimize data chunk sizes based on L3 cache
            l3_cache_size = self.hardware_profile.cache_sizes.get('L3', 8 * 1024 * 1024)
            optimal_chunk_size = l3_cache_size // (1024 * 8)  # Assuming 8KB per sample

            if 'chunk_size' in config:
                optimized_config['chunk_size'] = min(config['chunk_size'], optimal_chunk_size)

            # Adjust matrix operation parameters based on cache
            if 'block_size' in config:
                # Use cache-friendly block sizes
                cache_block_size = int(np.sqrt(l3_cache_size / 8))  # For float64 matrices
                optimized_config['block_size'] = cache_block_size

            self.logger.debug(f"💾 Cache optimization: L3={l3_cache_size / (1024*1024)".1f"} MB")
            return optimized_config

        except Exception as e:
            self.logger.warning(f"⚠️ Cache optimization failed: {e}")
            return config

    def get_hardware_recommendations(self) -> Dict[str, Any]:
        """
        Get hardware-specific recommendations.

        Returns:
            Dictionary of hardware recommendations
        """
        try:
            recommendations = {
                'memory_recommendations': self._get_memory_recommendations(),
                'cpu_recommendations': self._get_cpu_recommendations(),
                'gpu_recommendations': self._get_gpu_recommendations(),
                'overall_assessment': self._get_overall_assessment()
            }

            return recommendations

        except Exception as e:
            self.logger.warning(f"⚠️ Hardware recommendations generation failed: {e}")
            return {'error': str(e)}

    def _get_memory_recommendations(self) -> Dict[str, Any]:
        """
        Get memory-specific recommendations.

        Returns:
            Memory recommendations
        """
        try:
            available_memory = self.hardware_profile.memory_gb
            recommendations = {}

            if available_memory < 4.0:
                recommendations['level'] = 'LOW'
                recommendations['suggestions'] = [
                    'Use smaller batch sizes',
                    'Avoid preloading large datasets',
                    'Consider data sampling for training'
                ]
            elif available_memory < 8.0:
                recommendations['level'] = 'MODERATE'
                recommendations['suggestions'] = [
                    'Optimize memory usage in algorithms',
                    'Use chunked data loading'
                ]
            else:
                recommendations['level'] = 'HIGH'
                recommendations['suggestions'] = [
                    'Can use larger models and batch sizes',
                    'Consider preloading data for faster training'
                ]

            recommendations['available_memory_gb'] = available_memory
            recommendations['max_recommended_batch_size'] = self._estimate_max_batch_size(available_memory)

            return recommendations

        except Exception as e:
            self.logger.warning(f"⚠️ Memory recommendations failed: {e}")
            return {'level': 'UNKNOWN', 'error': str(e)}

    def _get_cpu_recommendations(self) -> Dict[str, Any]:
        """
        Get CPU-specific recommendations.

        Returns:
            CPU recommendations
        """
        try:
            cpu_cores = self.hardware_profile.cpu_cores
            recommendations = {}

            if cpu_cores <= 2:
                recommendations['level'] = 'LOW'
                recommendations['suggestions'] = [
                    'Limit parallel processing',
                    'Use sequential algorithms when possible'
                ]
            elif cpu_cores <= 4:
                recommendations['level'] = 'MODERATE'
                recommendations['suggestions'] = [
                    'Enable moderate parallel processing'
                ]
            else:
                recommendations['level'] = 'HIGH'
                recommendations['suggestions'] = [
                    'Maximize parallel processing',
                    'Use multi-threaded algorithms'
                ]

            recommendations['cpu_cores'] = cpu_cores
            recommendations['max_parallel_jobs'] = cpu_cores

            return recommendations

        except Exception as e:
            self.logger.warning(f"⚠️ CPU recommendations failed: {e}")
            return {'level': 'UNKNOWN', 'error': str(e)}

    def _get_gpu_recommendations(self) -> Dict[str, Any]:
        """
        Get GPU-specific recommendations.

        Returns:
            GPU recommendations
        """
        try:
            recommendations = {}

            if not self.hardware_profile.gpu_available:
                recommendations['available'] = False
                recommendations['suggestions'] = [
                    'Consider CPU-based algorithms',
                    'GPU acceleration not available'
                ]
            else:
                gpu_memory = self.hardware_profile.gpu_memory_gb
                recommendations['available'] = True
                recommendations['memory_gb'] = gpu_memory

                if gpu_memory < 4.0:
                    recommendations['level'] = 'LOW'
                    recommendations['suggestions'] = [
                        'Use GPU for inference only',
                        'Limited memory for training'
                    ]
                elif gpu_memory < 8.0:
                    recommendations['level'] = 'MODERATE'
                    recommendations['suggestions'] = [
                        'Use GPU for training with moderate batch sizes'
                    ]
                else:
                    recommendations['level'] = 'HIGH'
                    recommendations['suggestions'] = [
                        'Full GPU acceleration available',
                        'Can use large models and batch sizes'
                    ]

            return recommendations

        except Exception as e:
            self.logger.warning(f"⚠️ GPU recommendations failed: {e}")
            return {'available': False, 'error': str(e)}

    def _get_overall_assessment(self) -> Dict[str, Any]:
        """
        Get overall hardware assessment.

        Returns:
            Overall assessment
        """
        try:
            memory_level = self._get_memory_recommendations()['level']
            cpu_level = self._get_cpu_recommendations()['level']
            gpu_info = self._get_gpu_recommendations()

            # Simple scoring system
            level_scores = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
            memory_score = level_scores.get(memory_level, 2)
            cpu_score = level_scores.get(cpu_level, 2)
            gpu_score = 3 if gpu_info.get('available', False) else 1

            overall_score = (memory_score + cpu_score + gpu_score) / 3

            if overall_score < 1.5:
                overall_level = 'LOW'
            elif overall_score < 2.5:
                overall_level = 'MODERATE'
            else:
                overall_level = 'HIGH'

            assessment = {
                'overall_level': overall_level,
                'component_levels': {
                    'memory': memory_level,
                    'cpu': cpu_level,
                    'gpu': gpu_info.get('level', 'NONE')
                },
                'recommendations': self._get_optimization_recommendations(overall_level)
            }

            return assessment

        except Exception as e:
            self.logger.warning(f"⚠️ Overall assessment failed: {e}")
            return {'overall_level': 'UNKNOWN', 'error': str(e)}

    def _get_optimization_recommendations(self, level: str) -> List[str]:
        """
        Get optimization recommendations based on hardware level.

        Args:
            level: Hardware level

        Returns:
            List of recommendations
        """
        try:
            if level == 'LOW':
                return [
                    'Focus on memory-efficient algorithms',
                    'Use smaller model sizes',
                    'Consider algorithm simplification',
                    'Implement data sampling strategies'
                ]
            elif level == 'MODERATE':
                return [
                    'Balance between performance and resource usage',
                    'Use moderate batch sizes and model complexity',
                    'Enable parallel processing where beneficial'
                ]
            else:  # HIGH
                return [
                    'Maximize performance with available resources',
                    'Use largest feasible model sizes',
                    'Enable all optimization features',
                    'Consider ensemble methods'
                ]

        except Exception as e:
            self.logger.warning(f"⚠️ Optimization recommendations failed: {e}")
            return ['Use default settings']

    def monitor_performance(self, algorithm_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Monitor algorithm performance and provide metrics.

        Args:
            algorithm_config: Current algorithm configuration

        Returns:
            Performance metrics
        """
        try:
            import time
            import psutil

            # Get current process information
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            metrics = {
                'memory_usage_mb': memory_info.rss / (1024 * 1024),
                'memory_percent': process.memory_percent(),
                'cpu_percent': cpu_percent,
                'num_threads': process.num_threads(),
                'timestamp': time.time()
            }

            self.performance_metrics = metrics
            self.logger.debug(f"📊 Performance metrics: memory={metrics['memory_usage_mb']".1f"} MB, CPU={metrics['cpu_percent']".1f"}%")

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Performance monitoring failed: {e}")
            return {}