"""
Dynamic Memory Allocator for Intelligent Resource Management.

This module provides intelligent, dynamic memory allocation based on:
- System resources (total memory, CPU cores, GPU memory)
- Workload characteristics (data size, processing intensity)
- Real-time memory pressure and usage patterns
- User preferences and constraints
"""

import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class SystemTier(Enum):
    """System performance tiers based on resources."""
    ENTERPRISE = "enterprise"      # 64GB+ RAM, 16+ cores
    HIGH_END = "high_end"         # 32GB+ RAM, 8+ cores
    MID_RANGE = "mid_range"       # 16GB+ RAM, 4+ cores
    STANDARD = "standard"         # 8GB+ RAM, 2+ cores
    LOW_END = "low_end"          # <8GB RAM, <2 cores

class WorkloadType(Enum):
    """Workload types for memory allocation optimization."""
    LIGHT = "light"              # Small datasets, simple operations
    MODERATE = "moderate"        # Medium datasets, standard ML
    HEAVY = "heavy"              # Large datasets, complex ML
    EXTREME = "extreme"          # Very large datasets, deep learning
    STREAMING = "streaming"      # Continuous data processing

@dataclass
class SystemResources:
    """System resource information."""
    total_memory_gb: float
    available_memory_gb: float
    cpu_cores: int
    cpu_frequency_mhz: float
    gpu_memory_gb: Optional[float] = None
    gpu_cores: Optional[int] = None
    system_tier: SystemTier = SystemTier.STANDARD
    is_m1_chip: bool = False
    is_ssd: bool = True

@dataclass
class MemoryAllocation:
    """Memory allocation configuration."""
    cache_memory_mb: float
    processing_memory_mb: float
    buffer_memory_mb: float
    total_allocated_mb: float
    allocation_strategy: str
    adaptive_scaling: bool = True
    pressure_thresholds: Dict[str, float] = None

class DynamicMemoryAllocator:
    """Intelligent dynamic memory allocator."""
    
    def __init__(self):
        self.logger = logger.getChild('DynamicMemoryAllocator')
        self.system_resources = self._detect_system_resources()
        self.memory_usage_history = []
        self.allocation_history = []
        self.adaptive_factors = {
            'memory_pressure': 1.0,
            'workload_intensity': 1.0,
            'success_rate': 1.0,
            'performance_score': 1.0
        }
        self.lock = threading.RLock()
        
        tprint_success("✅ Dynamic Memory Allocator initialized")
        self.logger.info(f"System detected: {self.system_resources.system_tier.value} "
                        f"({self.system_resources.total_memory_gb:.1f}GB RAM, "
                        f"{self.system_resources.cpu_cores} cores)")
    
    def _detect_system_resources(self) -> SystemResources:
        """Detect system resources and capabilities."""
        try:
            # Memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            
            # CPU information
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency_mhz = cpu_freq.max if cpu_freq else 0
            
            # GPU detection
            gpu_memory_gb = None
            gpu_cores = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_cores = torch.cuda.get_device_properties(0).multi_processor_count
            except ImportError:
                pass
            
            # M1 chip detection
            is_m1_chip = False
            try:
                import platform
                if platform.processor() == 'arm':
                    is_m1_chip = True
            except:
                pass
            
            # SSD detection
            is_ssd = True
            try:
                for disk in psutil.disk_partitions():
                    if disk.mountpoint == '/':
                        # This is a simplified check
                        is_ssd = True
                        break
            except:
                pass
            
            # Determine system tier
            system_tier = self._determine_system_tier(
                total_memory_gb, cpu_cores, gpu_memory_gb
            )
            
            return SystemResources(
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                cpu_cores=cpu_cores,
                cpu_frequency_mhz=cpu_frequency_mhz,
                gpu_memory_gb=gpu_memory_gb,
                gpu_cores=gpu_cores,
                system_tier=system_tier,
                is_m1_chip=is_m1_chip,
                is_ssd=is_ssd
            )
            
        except Exception as e:
            self.logger.error(f"Failed to detect system resources: {e}")
            # Fallback to conservative defaults
            return SystemResources(
                total_memory_gb=8.0,
                available_memory_gb=6.0,
                cpu_cores=4,
                cpu_frequency_mhz=2000,
                system_tier=SystemTier.STANDARD
            )
    
    def _determine_system_tier(self, total_memory_gb: float, cpu_cores: int, 
                              gpu_memory_gb: Optional[float]) -> SystemTier:
        """Determine system performance tier."""
        if total_memory_gb >= 64 and cpu_cores >= 16:
            return SystemTier.ENTERPRISE
        elif total_memory_gb >= 32 and cpu_cores >= 8:
            return SystemTier.HIGH_END
        elif total_memory_gb >= 16 and cpu_cores >= 4:
            return SystemTier.MID_RANGE
        elif total_memory_gb >= 8 and cpu_cores >= 2:
            return SystemTier.STANDARD
        else:
            return SystemTier.LOW_END
    
    def get_optimal_allocation(self, workload_type: WorkloadType = WorkloadType.MODERATE,
                             data_size_mb: Optional[float] = None,
                             user_preferences: Optional[Dict[str, Any]] = None) -> MemoryAllocation:
        """Get optimal memory allocation based on system and workload."""
        with self.lock:
            # Base allocation based on system tier
            base_allocation = self._get_base_allocation()
            
            # Adjust for workload type
            workload_factor = self._get_workload_factor(workload_type)
            
            # Adjust for data size
            data_factor = self._get_data_size_factor(data_size_mb)
            
            # Apply adaptive factors
            adaptive_factor = self._calculate_adaptive_factor()
            
            # Calculate final allocation
            total_factor = workload_factor * data_factor * adaptive_factor
            
            # Apply user preferences
            if user_preferences:
                total_factor *= user_preferences.get('memory_usage_factor', 1.0)
            
            # Calculate memory allocation
            cache_memory_mb = base_allocation['cache'] * total_factor
            processing_memory_mb = base_allocation['processing'] * total_factor
            buffer_memory_mb = base_allocation['buffer'] * total_factor
            
            # Ensure reasonable bounds
            max_total_mb = self.system_resources.total_memory_gb * 1024 * 0.8  # Max 80% of total memory
            total_allocated = cache_memory_mb + processing_memory_mb + buffer_memory_mb
            
            if total_allocated > max_total_mb:
                scale_factor = max_total_mb / total_allocated
                cache_memory_mb *= scale_factor
                processing_memory_mb *= scale_factor
                buffer_memory_mb *= scale_factor
                total_allocated = max_total_mb
            
            # Set pressure thresholds based on allocation
            pressure_thresholds = {
                'low': 0.6,
                'medium': 0.75,
                'high': 0.85,
                'critical': 0.95
            }
            
            allocation = MemoryAllocation(
                cache_memory_mb=cache_memory_mb,
                processing_memory_mb=processing_memory_mb,
                buffer_memory_mb=buffer_memory_mb,
                total_allocated_mb=total_allocated,
                allocation_strategy=f"{workload_type.value}_{self.system_resources.system_tier.value}",
                adaptive_scaling=True,
                pressure_thresholds=pressure_thresholds
            )
            
            # Store allocation history
            self.allocation_history.append({
                'timestamp': time.time(),
                'allocation': allocation,
                'workload_type': workload_type.value,
                'data_size_mb': data_size_mb,
                'factors': {
                    'workload': workload_factor,
                    'data_size': data_factor,
                    'adaptive': adaptive_factor,
                    'total': total_factor
                }
            })
            
            # Keep only last 100 allocations
            if len(self.allocation_history) > 100:
                self.allocation_history = self.allocation_history[-100:]
            
            tprint_info(f"Dynamic allocation: {total_allocated:.0f}MB total "
                       f"(Cache: {cache_memory_mb:.0f}MB, Processing: {processing_memory_mb:.0f}MB, "
                       f"Buffer: {buffer_memory_mb:.0f}MB)")
            
            return allocation
    
    def _get_base_allocation(self) -> Dict[str, float]:
        """Get base memory allocation based on system tier."""
        tier = self.system_resources.system_tier
        
        if tier == SystemTier.ENTERPRISE:
            return {
                'cache': self.system_resources.total_memory_gb * 1024 * 0.20,  # 20%
                'processing': self.system_resources.total_memory_gb * 1024 * 0.30,  # 30%
                'buffer': self.system_resources.total_memory_gb * 1024 * 0.10   # 10%
            }
        elif tier == SystemTier.HIGH_END:
            return {
                'cache': self.system_resources.total_memory_gb * 1024 * 0.25,  # 25%
                'processing': self.system_resources.total_memory_gb * 1024 * 0.35,  # 35%
                'buffer': self.system_resources.total_memory_gb * 1024 * 0.10   # 10%
            }
        elif tier == SystemTier.MID_RANGE:
            return {
                'cache': self.system_resources.total_memory_gb * 1024 * 0.30,  # 30%
                'processing': self.system_resources.total_memory_gb * 1024 * 0.40,  # 40%
                'buffer': self.system_resources.total_memory_gb * 1024 * 0.10   # 10%
            }
        elif tier == SystemTier.STANDARD:
            return {
                'cache': self.system_resources.total_memory_gb * 1024 * 0.35,  # 35%
                'processing': self.system_resources.total_memory_gb * 1024 * 0.45,  # 45%
                'buffer': self.system_resources.total_memory_gb * 1024 * 0.10   # 10%
            }
        else:  # LOW_END
            return {
                'cache': self.system_resources.total_memory_gb * 1024 * 0.40,  # 40%
                'processing': self.system_resources.total_memory_gb * 1024 * 0.50,  # 50%
                'buffer': self.system_resources.total_memory_gb * 1024 * 0.10   # 10%
            }
    
    def _get_workload_factor(self, workload_type: WorkloadType) -> float:
        """Get memory allocation factor based on workload type."""
        factors = {
            WorkloadType.LIGHT: 0.5,
            WorkloadType.MODERATE: 1.0,
            WorkloadType.HEAVY: 1.5,
            WorkloadType.EXTREME: 2.0,
            WorkloadType.STREAMING: 0.8
        }
        return factors.get(workload_type, 1.0)
    
    def _get_data_size_factor(self, data_size_mb: Optional[float]) -> float:
        """Get memory allocation factor based on data size."""
        if data_size_mb is None:
            return 1.0
        
        # Scale based on data size relative to available memory
        available_memory_mb = self.system_resources.available_memory_gb * 1024
        data_ratio = data_size_mb / available_memory_mb
        
        if data_ratio < 0.1:
            return 0.8  # Small data, less memory needed
        elif data_ratio < 0.3:
            return 1.0  # Normal data size
        elif data_ratio < 0.6:
            return 1.3  # Large data, more memory needed
        else:
            return 1.6  # Very large data, significantly more memory needed
    
    def _calculate_adaptive_factor(self) -> float:
        """Calculate adaptive factor based on historical performance."""
        if not self.allocation_history:
            return 1.0
        
        # Calculate average performance score from recent allocations
        recent_allocations = self.allocation_history[-10:]  # Last 10 allocations
        
        # Simple adaptive logic based on success rate
        success_count = 0
        for allocation_record in recent_allocations:
            # This would be enhanced with actual performance metrics
            success_count += 1  # Simplified for now
        
        success_rate = success_count / len(recent_allocations)
        
        # Adjust factor based on success rate
        if success_rate > 0.9:
            return 1.1  # Increase allocation slightly
        elif success_rate > 0.7:
            return 1.0  # Keep current allocation
        else:
            return 0.9  # Decrease allocation
    
    def update_memory_usage(self, used_memory_mb: float, pressure_level: str):
        """Update memory usage for adaptive learning."""
        with self.lock:
            self.memory_usage_history.append({
                'timestamp': time.time(),
                'used_memory_mb': used_memory_mb,
                'pressure_level': pressure_level
            })
            
            # Keep only last 1000 records
            if len(self.memory_usage_history) > 1000:
                self.memory_usage_history = self.memory_usage_history[-1000:]
            
            # Update adaptive factors based on pressure
            if pressure_level == 'critical':
                self.adaptive_factors['memory_pressure'] *= 0.9
            elif pressure_level == 'high':
                self.adaptive_factors['memory_pressure'] *= 0.95
            elif pressure_level == 'low':
                self.adaptive_factors['memory_pressure'] *= 1.05
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system-specific recommendations for optimal performance."""
        recommendations = {
            'system_tier': self.system_resources.system_tier.value,
            'total_memory_gb': self.system_resources.total_memory_gb,
            'recommended_cache_mb': 0,
            'recommended_processing_mb': 0,
            'optimization_tips': []
        }
        
        # Get optimal allocation for different workload types
        light_allocation = self.get_optimal_allocation(WorkloadType.LIGHT)
        heavy_allocation = self.get_optimal_allocation(WorkloadType.HEAVY)
        
        recommendations['recommended_cache_mb'] = light_allocation.cache_memory_mb
        recommendations['recommended_processing_mb'] = light_allocation.processing_memory_mb
        
        # Add optimization tips based on system
        if self.system_resources.is_m1_chip:
            recommendations['optimization_tips'].append(
                "M1 chip detected: Enable M1-specific optimizations for better performance"
            )
        
        if self.system_resources.gpu_memory_gb:
            recommendations['optimization_tips'].append(
                f"GPU detected: {self.system_resources.gpu_memory_gb:.1f}GB VRAM available for acceleration"
            )
        
        if self.system_resources.system_tier == SystemTier.LOW_END:
            recommendations['optimization_tips'].append(
                "Low-end system: Consider using chunking and streaming for large datasets"
            )
        elif self.system_resources.system_tier == SystemTier.ENTERPRISE:
            recommendations['optimization_tips'].append(
                "Enterprise system: Can handle large datasets in memory, consider disabling chunking"
            )
        
        return recommendations
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get statistics about memory allocations."""
        if not self.allocation_history:
            return {'message': 'No allocation history available'}
        
        recent_allocations = self.allocation_history[-50:]  # Last 50 allocations
        
        cache_allocations = [a['allocation'].cache_memory_mb for a in recent_allocations]
        processing_allocations = [a['allocation'].processing_memory_mb for a in recent_allocations]
        total_allocations = [a['allocation'].total_allocated_mb for a in recent_allocations]
        
        return {
            'total_allocations': len(self.allocation_history),
            'recent_allocations': len(recent_allocations),
            'average_cache_mb': np.mean(cache_allocations),
            'average_processing_mb': np.mean(processing_allocations),
            'average_total_mb': np.mean(total_allocations),
            'max_total_mb': np.max(total_allocations),
            'min_total_mb': np.min(total_allocations),
            'adaptive_factors': self.adaptive_factors.copy()
        }

# Global instance
_global_allocator: Optional[DynamicMemoryAllocator] = None

def get_dynamic_allocator() -> DynamicMemoryAllocator:
    """Get or create the global dynamic memory allocator."""
    global _global_allocator
    
    if _global_allocator is None:
        _global_allocator = DynamicMemoryAllocator()
    
    return _global_allocator

def get_optimal_memory_allocation(workload_type: WorkloadType = WorkloadType.MODERATE,
                                data_size_mb: Optional[float] = None,
                                user_preferences: Optional[Dict[str, Any]] = None) -> MemoryAllocation:
    """Get optimal memory allocation for given workload and data size."""
    allocator = get_dynamic_allocator()
    return allocator.get_optimal_allocation(workload_type, data_size_mb, user_preferences)

def get_system_recommendations() -> Dict[str, Any]:
    """Get system-specific recommendations."""
    allocator = get_dynamic_allocator()
    return allocator.get_system_recommendations()

def update_memory_usage(used_memory_mb: float, pressure_level: str):
    """Update memory usage for adaptive learning."""
    allocator = get_dynamic_allocator()
    allocator.update_memory_usage(used_memory_mb, pressure_level)