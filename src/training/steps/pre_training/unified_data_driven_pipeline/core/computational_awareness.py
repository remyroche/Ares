"""
Computational Awareness Framework for Feature Selection

This module provides computational awareness and resource management for feature selection,
ensuring that methods are selected and executed based on available computational resources
and alignment with financial objectives.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class ComputationalTier(Enum):
    """Computational resource tiers for method selection."""
    MINIMAL = "minimal"      # < 2GB RAM, < 2 CPU cores
    STANDARD = "standard"    # 2-8GB RAM, 2-4 CPU cores
    HIGH = "high"           # 8-32GB RAM, 4-8 CPU cores
    ENTERPRISE = "enterprise" # > 32GB RAM, > 8 CPU cores


class MethodComplexity(Enum):
    """Feature selection method complexity levels."""
    LOW = "low"         # O(n) - correlation, variance
    MEDIUM = "medium"   # O(n²) - mutual information, basic mRMR
    HIGH = "high"       # O(n³) - RFE, advanced mRMR
    VERY_HIGH = "very_high"  # O(n⁴) - ensemble methods, evolutionary


@dataclass
class ComputationalProfile:
    """Profile of computational requirements for a feature selection method."""
    method_name: str
    complexity: MethodComplexity
    memory_usage_mb: float
    cpu_cores_required: int
    estimated_time_seconds: float
    vectorbt_optimized: bool
    parallelizable: bool
    gpu_required: bool = False
    financial_objective_alignment: float = 1.0  # 0-1 score


@dataclass
class SystemResources:
    """Current system resource availability."""
    available_memory_gb: float
    available_cpu_cores: int
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_available: bool
    gpu_memory_gb: float = 0.0


@dataclass
class ComputationalConstraints:
    """Constraints for computational resource usage."""
    max_memory_gb: float
    max_cpu_cores: int
    max_execution_time_seconds: float
    memory_safety_margin: float = 0.2  # 20% safety margin
    cpu_safety_margin: float = 0.1     # 10% safety margin


class ComputationalAwarenessManager:
    """Manages computational awareness for feature selection."""
    
    def __init__(self, constraints: Optional[ComputationalConstraints] = None):
        """Initialize computational awareness manager."""
        self.constraints = constraints or self._get_default_constraints()
        self.logger = logger.getChild('ComputationalAwarenessManager')
        
        # Define computational profiles for different methods
        self.method_profiles = self._initialize_method_profiles()
        
        # Performance tracking
        self.performance_history = {
            'method_executions': {},
            'resource_usage': {},
            'success_rates': {}
        }
        
        tprint_info("🧠 Computational Awareness Manager initialized")
    
    def _get_default_constraints(self) -> ComputationalConstraints:
        """Get default computational constraints based on system resources."""
        try:
            # Get system resources
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_cores = psutil.cpu_count()
            
            # Set conservative constraints
            return ComputationalConstraints(
                max_memory_gb=memory_gb * 0.7,  # Use 70% of available memory
                max_cpu_cores=max(1, cpu_cores - 1),  # Leave 1 core free
                max_execution_time_seconds=300.0  # 5 minutes max
            )
        except Exception as e:
            tprint_warning(f"Could not detect system resources: {e}")
            return ComputationalConstraints(
                max_memory_gb=4.0,
                max_cpu_cores=2,
                max_execution_time_seconds=300.0
            )
    
    def _initialize_method_profiles(self) -> Dict[str, ComputationalProfile]:
        """Initialize computational profiles for feature selection methods."""
        return {
            # Standard methods
            'correlation': ComputationalProfile(
                method_name='correlation',
                complexity=MethodComplexity.LOW,
                memory_usage_mb=100,
                cpu_cores_required=1,
                estimated_time_seconds=1.0,
                vectorbt_optimized=False,
                parallelizable=True,
                financial_objective_alignment=0.6
            ),
            'mutual_info': ComputationalProfile(
                method_name='mutual_info',
                complexity=MethodComplexity.MEDIUM,
                memory_usage_mb=200,
                cpu_cores_required=2,
                estimated_time_seconds=5.0,
                vectorbt_optimized=False,
                parallelizable=True,
                financial_objective_alignment=0.7
            ),
            'rfe': ComputationalProfile(
                method_name='rfe',
                complexity=MethodComplexity.HIGH,
                memory_usage_mb=500,
                cpu_cores_required=2,
                estimated_time_seconds=30.0,
                vectorbt_optimized=False,
                parallelizable=False,
                financial_objective_alignment=0.8
            ),
            'lasso': ComputationalProfile(
                method_name='lasso',
                complexity=MethodComplexity.MEDIUM,
                memory_usage_mb=300,
                cpu_cores_required=2,
                estimated_time_seconds=10.0,
                vectorbt_optimized=False,
                parallelizable=True,
                financial_objective_alignment=0.8
            ),
            
            # Enhanced methods
            'improved_mrmr': ComputationalProfile(
                method_name='improved_mrmr',
                complexity=MethodComplexity.HIGH,
                memory_usage_mb=400,
                cpu_cores_required=2,
                estimated_time_seconds=15.0,
                vectorbt_optimized=False,
                parallelizable=True,
                financial_objective_alignment=0.9
            ),
            'vectorbt_mrmr': ComputationalProfile(
                method_name='vectorbt_mrmr',
                complexity=MethodComplexity.MEDIUM,
                memory_usage_mb=300,
                cpu_cores_required=4,
                estimated_time_seconds=8.0,
                vectorbt_optimized=True,
                parallelizable=True,
                financial_objective_alignment=0.9
            ),
            'vectorbt_rfe': ComputationalProfile(
                method_name='vectorbt_rfe',
                complexity=MethodComplexity.HIGH,
                memory_usage_mb=600,
                cpu_cores_required=4,
                estimated_time_seconds=20.0,
                vectorbt_optimized=True,
                parallelizable=True,
                financial_objective_alignment=0.85
            ),
            'vectorbt_lasso': ComputationalProfile(
                method_name='vectorbt_lasso',
                complexity=MethodComplexity.MEDIUM,
                memory_usage_mb=400,
                cpu_cores_required=4,
                estimated_time_seconds=12.0,
                vectorbt_optimized=True,
                parallelizable=True,
                financial_objective_alignment=0.85
            ),
            'enhanced_ensemble': ComputationalProfile(
                method_name='enhanced_ensemble',
                complexity=MethodComplexity.VERY_HIGH,
                memory_usage_mb=800,
                cpu_cores_required=4,
                estimated_time_seconds=45.0,
                vectorbt_optimized=False,
                parallelizable=True,
                financial_objective_alignment=0.95
            ),
            'enhanced_advanced': ComputationalProfile(
                method_name='enhanced_advanced',
                complexity=MethodComplexity.VERY_HIGH,
                memory_usage_mb=1000,
                cpu_cores_required=6,
                estimated_time_seconds=60.0,
                vectorbt_optimized=False,
                parallelizable=True,
                financial_objective_alignment=0.95
            )
        }
    
    def get_current_resources(self) -> SystemResources:
        """Get current system resource availability."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return SystemResources(
                available_memory_gb=(memory.available / (1024**3)),
                available_cpu_cores=psutil.cpu_count(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                gpu_available=self._check_gpu_availability(),
                gpu_memory_gb=self._get_gpu_memory()
            )
        except Exception as e:
            tprint_warning(f"Could not get system resources: {e}")
            return SystemResources(
                available_memory_gb=4.0,
                available_cpu_cores=2,
                cpu_usage_percent=50.0,
                memory_usage_percent=50.0,
                gpu_available=False
            )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
    
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        return 0.0
    
    def select_optimal_methods(self, 
                             data_shape: Tuple[int, int],
                             available_methods: List[str],
                             financial_objectives: List[str],
                             time_constraint: Optional[float] = None) -> List[str]:
        """
        Select optimal feature selection methods based on computational constraints.
        
        Args:
            data_shape: (n_samples, n_features) shape of the data
            available_methods: List of available method names
            financial_objectives: List of financial objectives to optimize
            time_constraint: Maximum time constraint in seconds
            
        Returns:
            List of recommended method names in order of preference
        """
        n_samples, n_features = data_shape
        current_resources = self.get_current_resources()
        
        # Calculate data complexity factor
        data_complexity = self._calculate_data_complexity(n_samples, n_features)
        
        # Filter methods based on constraints
        feasible_methods = []
        for method_name in available_methods:
            if method_name not in self.method_profiles:
                continue
                
            profile = self.method_profiles[method_name]
            
            # Check memory constraint
            required_memory_gb = profile.memory_usage_mb / 1024
            if required_memory_gb > current_resources.available_memory_gb * (1 - self.constraints.memory_safety_margin):
                tprint_debug(f"❌ {method_name}: Insufficient memory ({required_memory_gb:.1f}GB required)")
                continue
            
            # Check CPU constraint
            if profile.cpu_cores_required > current_resources.available_cpu_cores * (1 - self.constraints.cpu_safety_margin):
                tprint_debug(f"❌ {method_name}: Insufficient CPU cores ({profile.cpu_cores_required} required)")
                continue
            
            # Check time constraint
            estimated_time = profile.estimated_time_seconds * data_complexity
            if time_constraint and estimated_time > time_constraint:
                tprint_debug(f"❌ {method_name}: Exceeds time constraint ({estimated_time:.1f}s estimated)")
                continue
            
            # Check GPU requirement
            if profile.gpu_required and not current_resources.gpu_available:
                tprint_debug(f"❌ {method_name}: GPU required but not available")
                continue
            
            feasible_methods.append((method_name, profile))
        
        if not feasible_methods:
            tprint_warning("⚠️ No feasible methods found, using fallback")
            return ['correlation']  # Fallback to simplest method
        
        # Score methods based on multiple criteria
        scored_methods = []
        for method_name, profile in feasible_methods:
            score = self._calculate_method_score(
                profile, data_complexity, financial_objectives, current_resources
            )
            scored_methods.append((method_name, score))
        
        # Sort by score (higher is better)
        scored_methods.sort(key=lambda x: x[1], reverse=True)
        
        recommended_methods = [method for method, score in scored_methods]
        
        tprint_info(f"🧠 Recommended methods: {recommended_methods[:3]}")
        return recommended_methods
    
    def _calculate_data_complexity(self, n_samples: int, n_features: int) -> float:
        """Calculate data complexity factor."""
        # Base complexity on data size
        base_complexity = (n_samples * n_features) / (1000 * 100)  # Normalize to reasonable scale
        
        # Adjust for feature-to-sample ratio
        if n_features > n_samples * 0.5:  # High-dimensional data
            base_complexity *= 1.5
        elif n_features < n_samples * 0.01:  # Low-dimensional data
            base_complexity *= 0.7
        
        return max(1.0, base_complexity)
    
    def _calculate_method_score(self, 
                              profile: ComputationalProfile,
                              data_complexity: float,
                              financial_objectives: List[str],
                              resources: SystemResources) -> float:
        """Calculate a score for a method based on multiple criteria."""
        score = 0.0
        
        # Financial objective alignment (40% weight)
        objective_alignment = profile.financial_objective_alignment
        score += objective_alignment * 0.4
        
        # Computational efficiency (30% weight)
        efficiency = self._calculate_efficiency_score(profile, data_complexity)
        score += efficiency * 0.3
        
        # Resource utilization (20% weight)
        resource_utilization = self._calculate_resource_utilization_score(profile, resources)
        score += resource_utilization * 0.2
        
        # VectorBT optimization bonus (10% weight)
        if profile.vectorbt_optimized:
            score += 0.1
        
        return score
    
    def _calculate_efficiency_score(self, profile: ComputationalProfile, data_complexity: float) -> float:
        """Calculate efficiency score based on time and memory usage."""
        # Normalize time efficiency (lower is better)
        time_efficiency = 1.0 / (profile.estimated_time_seconds * data_complexity + 1)
        
        # Normalize memory efficiency (lower is better)
        memory_efficiency = 1.0 / (profile.memory_usage_mb / 1000 + 1)
        
        return (time_efficiency + memory_efficiency) / 2
    
    def _calculate_resource_utilization_score(self, profile: ComputationalProfile, resources: SystemResources) -> float:
        """Calculate how well the method utilizes available resources."""
        # CPU utilization (optimal is 70-90%)
        cpu_utilization = min(profile.cpu_cores_required / resources.available_cpu_cores, 1.0)
        cpu_score = 1.0 - abs(cpu_utilization - 0.8)  # Peak at 80% utilization
        
        # Memory utilization (optimal is 50-80%)
        memory_utilization = (profile.memory_usage_mb / 1024) / resources.available_memory_gb
        memory_score = 1.0 - abs(memory_utilization - 0.65)  # Peak at 65% utilization
        
        return (cpu_score + memory_score) / 2
    
    def monitor_execution(self, method_name: str, execution_func: Callable) -> Any:
        """Monitor the execution of a feature selection method."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        try:
            tprint_debug(f"🧠 Executing {method_name} with monitoring")
            result = execution_func()
            
            # Record successful execution
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_used = end_memory - start_memory
            
            self._record_execution(method_name, execution_time, memory_used, success=True)
            
            tprint_success(f"✅ {method_name} completed in {execution_time:.2f}s, used {memory_used:.1f}MB")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution(method_name, execution_time, 0, success=False)
            tprint_error(f"❌ {method_name} failed after {execution_time:.2f}s: {e}")
            raise
    
    def _record_execution(self, method_name: str, execution_time: float, memory_used: float, success: bool):
        """Record execution statistics."""
        if method_name not in self.performance_history['method_executions']:
            self.performance_history['method_executions'][method_name] = []
        
        self.performance_history['method_executions'][method_name].append({
            'execution_time': execution_time,
            'memory_used': memory_used,
            'success': success,
            'timestamp': time.time()
        })
        
        # Update success rate
        executions = self.performance_history['method_executions'][method_name]
        success_rate = sum(1 for ex in executions if ex['success']) / len(executions)
        self.performance_history['success_rates'][method_name] = success_rate
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all methods."""
        summary = {
            'method_performance': {},
            'resource_utilization': {},
            'recommendations': {}
        }
        
        for method_name, executions in self.performance_history['method_executions'].items():
            if not executions:
                continue
                
            successful_executions = [ex for ex in executions if ex['success']]
            if not successful_executions:
                continue
            
            avg_time = np.mean([ex['execution_time'] for ex in successful_executions])
            avg_memory = np.mean([ex['memory_used'] for ex in successful_executions])
            success_rate = self.performance_history['success_rates'][method_name]
            
            summary['method_performance'][method_name] = {
                'avg_execution_time': avg_time,
                'avg_memory_usage': avg_memory,
                'success_rate': success_rate,
                'total_executions': len(executions)
            }
        
        return summary
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for method selection based on performance history."""
        recommendations = {
            'preferred_methods': [],
            'avoid_methods': [],
            'resource_optimization': {}
        }
        
        for method_name, perf in self.performance_history['method_executions'].items():
            if not perf:
                continue
            
            success_rate = self.performance_history['success_rates'][method_name]
            avg_time = np.mean([ex['execution_time'] for ex in perf if ex['success']])
            
            if success_rate > 0.8 and avg_time < 30:  # High success rate and fast
                recommendations['preferred_methods'].append(method_name)
            elif success_rate < 0.5 or avg_time > 120:  # Low success rate or slow
                recommendations['avoid_methods'].append(method_name)
        
        return recommendations


# Convenience functions
def create_computational_awareness_manager(constraints: Optional[ComputationalConstraints] = None) -> ComputationalAwarenessManager:
    """Create a computational awareness manager with optional constraints."""
    return ComputationalAwarenessManager(constraints)


def get_optimal_methods_for_data(data_shape: Tuple[int, int],
                                available_methods: List[str],
                                financial_objectives: List[str],
                                constraints: Optional[ComputationalConstraints] = None) -> List[str]:
    """Get optimal methods for given data and constraints."""
    manager = create_computational_awareness_manager(constraints)
    return manager.select_optimal_methods(data_shape, available_methods, financial_objectives)