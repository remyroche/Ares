"""
Advanced Decorators Module
Provides enhanced decorators for performance monitoring, model validation, data pipeline management,
caching, adaptive resource allocation, and comprehensive validation.
"""

import functools
import time
import psutil
import gc
import os
import json
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union, Type
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from collections import OrderedDict
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning, critical, success


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    STRICT = "strict"
    SILENT = "silent"


class PerformanceLevel(Enum):
    """Performance monitoring levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    PROFILING = "profiling"
    MEMORY_TRACKING = "memory_tracking"
    CPU_TRACKING = "cpu_tracking"


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    gc_collections: int
    function_name: str
    timestamp: datetime


class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.logger = system_logger.getChild("PerformanceMonitor")
        
    def start_monitoring(self, function_name: str) -> Dict[str, Any]:
        """Start performance monitoring."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        start_cpu = psutil.cpu_percent()
        start_gc = gc.get_count()
        
        return {
            "start_time": start_time,
            "start_memory": start_memory,
            "start_cpu": start_cpu,
            "start_gc": start_gc,
            "function_name": function_name
        }
    
    def end_monitoring(self, start_metrics: Dict[str, Any]) -> PerformanceMetrics:
        """End performance monitoring and return metrics."""
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        end_cpu = psutil.cpu_percent()
        end_gc = gc.get_count()
        
        execution_time = end_time - start_metrics["start_time"]
        memory_usage = end_memory - start_metrics["start_memory"]
        cpu_usage = end_cpu - start_metrics["start_cpu"]
        gc_collections = sum(end_gc) - sum(start_metrics["start_gc"])
        
        # Get peak memory usage
        process = psutil.Process()
        peak_memory = process.memory_info().rss / 1024 / 1024
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            peak_memory_mb=peak_memory,
            gc_collections=gc_collections,
            function_name=start_metrics["function_name"],
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def log_metrics(self, metrics: PerformanceMetrics, level: PerformanceLevel = PerformanceLevel.BASIC):
        """Log performance metrics."""
        if level == PerformanceLevel.BASIC:
            self.logger.info(f"⏱️ {metrics.function_name}: {metrics.execution_time:.2f}s")
        elif level == PerformanceLevel.DETAILED:
            self.logger.info(
                f"📊 {metrics.function_name}: {metrics.execution_time:.2f}s, "
                f"Memory: {metrics.memory_usage_mb:.1f}MB, CPU: {metrics.cpu_usage_percent:.1f}%"
            )
        elif level == PerformanceLevel.PROFILING:
            self.logger.info(
                f"🔍 {metrics.function_name}: {metrics.execution_time:.2f}s, "
                f"Peak Memory: {metrics.peak_memory_mb:.1f}MB, "
                f"GC Collections: {metrics.gc_collections}"
            )


class ModelValidator:
    """Model validation utility."""
    
    def __init__(self):
        self.logger = system_logger.getChild("ModelValidator")
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_model_performance(
        self, 
        model, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    ) -> Dict[str, float]:
        """Validate model performance."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = model.predict(X_val)
            results = {}
            
            if "accuracy" in metrics:
                results["accuracy"] = accuracy_score(y_val, y_pred)
            if "precision" in metrics:
                results["precision"] = precision_score(y_val, y_pred, average='weighted')
            if "recall" in metrics:
                results["recall"] = recall_score(y_val, y_pred, average='weighted')
            if "f1" in metrics:
                results["f1"] = f1_score(y_val, y_pred, average='weighted')
            
            return results
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {}
    
    def check_overfitting(
        self, 
        train_score: float, 
        val_score: float, 
        threshold: float = 0.1
    ) -> bool:
        """Check for overfitting."""
        overfitting = (train_score - val_score) > threshold
        if overfitting:
            self.logger.warning(f"⚠️ Potential overfitting detected: train={train_score:.3f}, val={val_score:.3f}")
        return overfitting
    
    def check_underfitting(
        self, 
        train_score: float, 
        val_score: float, 
        threshold: float = 0.6
    ) -> bool:
        """Check for underfitting."""
        underfitting = train_score < threshold
        if underfitting:
            self.logger.warning(f"⚠️ Potential underfitting detected: train={train_score:.3f}")
        return underfitting


class DataPipelineManager:
    """Data pipeline management utility."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.logger = system_logger.getChild("DataPipelineManager")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self, 
        step_name: str, 
        data: Any, 
        metadata: Dict[str, Any] = None
    ) -> str:
        """Save pipeline checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{step_name}.pkl")
        metadata = metadata or {}
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name
        })
        
        checkpoint_data = {
            "data": data,
            "metadata": metadata
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Load pipeline checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{step_name}.pkl")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24):
        """Clean up old checkpoints."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_time:
                    os.remove(filepath)
                    self.logger.info(f"🗑️ Cleaned up old checkpoint: {filename}")


class IntelligentCache:
    """Intelligent caching utility."""
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.logger = system_logger.getChild("IntelligentCache")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        # Create a stable representation of arguments
        args_repr = str(sorted(args))
        kwargs_repr = str(sorted(kwargs.items()))
        cache_string = f"{func_name}:{args_repr}:{kwargs_repr}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check TTL
            if datetime.now() - entry["timestamp"] > timedelta(hours=self.ttl_hours):
                del self.cache[cache_key]
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            self.logger.debug(f"✅ Cache hit: {cache_key}")
            return entry["value"]
        
        self.logger.debug(f"❌ Cache miss: {cache_key}")
        return None
    
    def set(self, cache_key: str, value: Any):
        """Set value in cache."""
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            "value": value,
            "timestamp": datetime.now()
        }
        self.logger.debug(f"💾 Cached: {cache_key}")


class AdaptiveResourceManager:
    """Adaptive resource allocation utility."""
    
    def __init__(self):
        self.logger = system_logger.getChild("AdaptiveResourceManager")
        self.resource_history: List[Dict[str, float]] = []
    
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        return {
            "memory_percent": memory.percent,
            "cpu_percent": cpu,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "timestamp": time.time()
        }
    
    def should_scale_down(self, threshold: float = 0.8) -> bool:
        """Check if resources should be scaled down."""
        resources = self.get_current_resources()
        return resources["memory_percent"] > threshold or resources["cpu_percent"] > threshold
    
    def optimize_batch_size(self, current_batch_size: int, memory_usage: float) -> int:
        """Optimize batch size based on memory usage."""
        if memory_usage > 0.8:  # High memory usage
            return max(1, current_batch_size // 2)
        elif memory_usage < 0.4:  # Low memory usage
            return current_batch_size * 2
        else:
            return current_batch_size


class ComprehensiveValidator:
    """Comprehensive validation utility."""
    
    def __init__(self):
        self.logger = system_logger.getChild("ComprehensiveValidator")
        self.validation_results: List[Dict[str, Any]] = []
    
    def validate_data_quality(self, data: Any) -> Dict[str, Any]:
        """Validate data quality."""
        results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        if isinstance(data, pd.DataFrame):
            # Check for NaN values
            nan_count = data.isnull().sum().sum()
            if nan_count > 0:
                results["warnings"].append(f"Found {nan_count} NaN values")
            
            # Check for infinite values
            if data.select_dtypes(include=[np.number]).size > 0:
                inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
                if inf_count > 0:
                    results["warnings"].append(f"Found {inf_count} infinite values")
            
            # Check for constant columns
            constant_cols = data.columns[data.nunique() <= 1].tolist()
            if constant_cols:
                results["warnings"].append(f"Found constant columns: {constant_cols}")
            
            # Check data types
            object_cols = data.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                results["warnings"].append(f"Found object columns: {object_cols}")
        
        return results
    
    def validate_model_quality(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Validate model quality."""
        results = {
            "is_valid": True,
            "metrics": {},
            "issues": []
        }
        
        try:
            # Basic predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results["metrics"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }
            
            # Check for poor performance
            if results["metrics"]["accuracy"] < 0.5:
                results["issues"].append("Low accuracy detected")
                results["is_valid"] = False
            
        except Exception as e:
            results["issues"].append(f"Model validation failed: {e}")
            results["is_valid"] = False
        
        return results
    
    def validate_pipeline_quality(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline quality."""
        results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check required keys
        required_keys = ["step_name", "timestamp", "status"]
        for key in required_keys:
            if key not in pipeline_state:
                results["issues"].append(f"Missing required key: {key}")
                results["is_valid"] = False
        
        # Check status
        if "status" in pipeline_state and pipeline_state["status"] == "FAILED":
            results["issues"].append("Pipeline step failed")
            results["is_valid"] = False
        
        return results


# Global instances
_performance_monitor = PerformanceMonitor()
_model_validator = ModelValidator()
_pipeline_manager = DataPipelineManager()
_intelligent_cache = IntelligentCache()
_adaptive_manager = AdaptiveResourceManager()
_comprehensive_validator = ComprehensiveValidator()


# Performance Monitoring Decorators
def performance_monitor(
    enable_profiling: bool = True,
    enable_memory_tracking: bool = True,
    enable_cpu_tracking: bool = True,
    save_profile_data: bool = False,
    level: PerformanceLevel = PerformanceLevel.DETAILED
):
    """Decorator for comprehensive performance monitoring."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_metrics = _performance_monitor.start_monitoring(func.__name__)
            
            try:
                result = await func(*args, **kwargs)
                metrics = _performance_monitor.end_monitoring(start_metrics)
                _performance_monitor.log_metrics(metrics, level)
                
                if save_profile_data:
                    # Save profile data to file
                    profile_data = {
                        "function_name": func.__name__,
                        "execution_time": metrics.execution_time,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "cpu_usage_percent": metrics.cpu_usage_percent,
                        "peak_memory_mb": metrics.peak_memory_mb,
                        "timestamp": metrics.timestamp.isoformat()
                    }
                    
                    profile_file = f"profiles/{func.__name__}_{int(time.time())}.json"
                    os.makedirs("profiles", exist_ok=True)
                    with open(profile_file, 'w') as f:
                        json.dump(profile_data, f, indent=2)
                
                return result
            except Exception as e:
                metrics = _performance_monitor.end_monitoring(start_metrics)
                _performance_monitor.log_metrics(metrics, level)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_metrics = _performance_monitor.start_monitoring(func.__name__)
            
            try:
                result = func(*args, **kwargs)
                metrics = _performance_monitor.end_monitoring(start_metrics)
                _performance_monitor.log_metrics(metrics, level)
                
                if save_profile_data:
                    # Save profile data to file
                    profile_data = {
                        "function_name": func.__name__,
                        "execution_time": metrics.execution_time,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "cpu_usage_percent": metrics.cpu_usage_percent,
                        "peak_memory_mb": metrics.peak_memory_mb,
                        "timestamp": metrics.timestamp.isoformat()
                    }
                    
                    profile_file = f"profiles/{func.__name__}_{int(time.time())}.json"
                    os.makedirs("profiles", exist_ok=True)
                    with open(profile_file, 'w') as f:
                        json.dump(profile_data, f, indent=2)
                
                return result
            except Exception as e:
                metrics = _performance_monitor.end_monitoring(start_metrics)
                _performance_monitor.log_metrics(metrics, level)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Model Validation Decorators
def model_validation(
    check_overfitting: bool = True,
    check_underfitting: bool = True,
    validation_metrics: List[str] = ["accuracy", "precision", "recall", "f1"],
    cross_validation_folds: int = 5,
    overfitting_threshold: float = 0.1,
    underfitting_threshold: float = 0.6
):
    """Decorator for comprehensive model validation."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Extract model and validation data from result
            if isinstance(result, dict) and "model" in result:
                model = result["model"]
                X_val = result.get("X_val")
                y_val = result.get("y_val")
                X_train = result.get("X_train")
                y_train = result.get("y_train")
                
                if X_val is not None and y_val is not None:
                    # Validate model performance
                    val_metrics = _model_validator.validate_model_performance(
                        model, X_val, y_val, validation_metrics
                    )
                    
                    if X_train is not None and y_train is not None:
                        train_metrics = _model_validator.validate_model_performance(
                            model, X_train, y_train, ["accuracy"]
                        )
                        
                        if check_overfitting and "accuracy" in train_metrics and "accuracy" in val_metrics:
                            _model_validator.check_overfitting(
                                train_metrics["accuracy"], 
                                val_metrics["accuracy"], 
                                overfitting_threshold
                            )
                        
                        if check_underfitting and "accuracy" in train_metrics:
                            _model_validator.check_underfitting(
                                train_metrics["accuracy"], 
                                underfitting_threshold
                            )
                    
                    # Add validation results to return value
                    if isinstance(result, dict):
                        result["validation_metrics"] = val_metrics
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract model and validation data from result
            if isinstance(result, dict) and "model" in result:
                model = result["model"]
                X_val = result.get("X_val")
                y_val = result.get("y_val")
                X_train = result.get("X_train")
                y_train = result.get("y_train")
                
                if X_val is not None and y_val is not None:
                    # Validate model performance
                    val_metrics = _model_validator.validate_model_performance(
                        model, X_val, y_val, validation_metrics
                    )
                    
                    if X_train is not None and y_train is not None:
                        train_metrics = _model_validator.validate_model_performance(
                            model, X_train, y_train, ["accuracy"]
                        )
                        
                        if check_overfitting and "accuracy" in train_metrics and "accuracy" in val_metrics:
                            _model_validator.check_overfitting(
                                train_metrics["accuracy"], 
                                val_metrics["accuracy"], 
                                overfitting_threshold
                            )
                        
                        if check_underfitting and "accuracy" in train_metrics:
                            _model_validator.check_underfitting(
                                train_metrics["accuracy"], 
                                underfitting_threshold
                            )
                    
                    # Add validation results to return value
                    if isinstance(result, dict):
                        result["validation_metrics"] = val_metrics
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Data Pipeline Decorators
def pipeline_checkpoint(
    save_intermediate_results: bool = True,
    checkpoint_frequency: int = 1000,
    enable_rollback: bool = True,
    checkpoint_dir: str = "checkpoints"
):
    """Decorator for data pipeline checkpointing."""
    
    def decorator(func: Callable) -> Callable:
        call_count = 0
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Try to load checkpoint
            checkpoint_data = _pipeline_manager.load_checkpoint(func.__name__)
            if checkpoint_data and enable_rollback:
                system_logger.info(f"📂 Using checkpoint for {func.__name__}")
                return checkpoint_data["data"]
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Save checkpoint if conditions are met
            if save_intermediate_results and call_count % checkpoint_frequency == 0:
                metadata = {
                    "call_count": call_count,
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                _pipeline_manager.save_checkpoint(func.__name__, result, metadata)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Try to load checkpoint
            checkpoint_data = _pipeline_manager.load_checkpoint(func.__name__)
            if checkpoint_data and enable_rollback:
                system_logger.info(f"📂 Using checkpoint for {func.__name__}")
                return checkpoint_data["data"]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Save checkpoint if conditions are met
            if save_intermediate_results and call_count % checkpoint_frequency == 0:
                metadata = {
                    "call_count": call_count,
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                _pipeline_manager.save_checkpoint(func.__name__, result, metadata)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Caching Decorators
def intelligent_caching(
    cache_intermediate_results: bool = True,
    cache_validation_data: bool = True,
    cache_model_artifacts: bool = True,
    cache_ttl_hours: int = 24,
    max_cache_size: int = 100
):
    """Decorator for intelligent caching."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _intelligent_cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = _intelligent_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result if conditions are met
            if cache_intermediate_results:
                _intelligent_cache.set(cache_key, result)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _intelligent_cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = _intelligent_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result if conditions are met
            if cache_intermediate_results:
                _intelligent_cache.set(cache_key, result)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Adaptive Decorators
def adaptive_resource_allocation(
    dynamic_memory_allocation: bool = True,
    adaptive_batch_sizes: bool = True,
    resource_scaling_threshold: float = 0.8
):
    """Decorator for adaptive resource allocation."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check current resources
            resources = _adaptive_manager.get_current_resources()
            
            # Adjust batch size if needed
            if adaptive_batch_sizes and "batch_size" in kwargs:
                current_batch_size = kwargs["batch_size"]
                optimized_batch_size = _adaptive_manager.optimize_batch_size(
                    current_batch_size, resources["memory_percent"] / 100
                )
                kwargs["batch_size"] = optimized_batch_size
                
                if optimized_batch_size != current_batch_size:
                    system_logger.info(f"🔄 Adjusted batch size: {current_batch_size} -> {optimized_batch_size}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Log resource usage
            if dynamic_memory_allocation:
                system_logger.info(
                    f"📊 Resource usage: Memory {resources['memory_percent']:.1f}%, "
                    f"CPU {resources['cpu_percent']:.1f}%"
                )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check current resources
            resources = _adaptive_manager.get_current_resources()
            
            # Adjust batch size if needed
            if adaptive_batch_sizes and "batch_size" in kwargs:
                current_batch_size = kwargs["batch_size"]
                optimized_batch_size = _adaptive_manager.optimize_batch_size(
                    current_batch_size, resources["memory_percent"] / 100
                )
                kwargs["batch_size"] = optimized_batch_size
                
                if optimized_batch_size != current_batch_size:
                    system_logger.info(f"🔄 Adjusted batch size: {current_batch_size} -> {optimized_batch_size}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log resource usage
            if dynamic_memory_allocation:
                system_logger.info(
                    f"📊 Resource usage: Memory {resources['memory_percent']:.1f}%, "
                    f"CPU {resources['cpu_percent']:.1f}%"
                )
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Comprehensive Validation Decorators
def comprehensive_validation(
    data_quality_checks: bool = True,
    model_quality_checks: bool = True,
    pipeline_quality_checks: bool = True,
    output_validation: bool = True,
    validation_level: ValidationLevel = ValidationLevel.WARNING
):
    """Decorator for comprehensive validation."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function
            result = await func(*args, **kwargs)
            
            validation_issues = []
            
            # Data quality validation
            if data_quality_checks:
                for arg in args:
                    if isinstance(arg, (pd.DataFrame, np.ndarray)):
                        data_quality = _comprehensive_validator.validate_data_quality(arg)
                        if not data_quality["is_valid"]:
                            validation_issues.extend(data_quality["issues"])
                        if data_quality["warnings"]:
                            for warning in data_quality["warnings"]:
                                system_logger.warning(f"⚠️ Data quality warning: {warning}")
            
            # Model quality validation
            if model_quality_checks and isinstance(result, dict) and "model" in result:
                model = result["model"]
                X_val = result.get("X_val")
                y_val = result.get("y_val")
                
                if X_val is not None and y_val is not None:
                    model_quality = _comprehensive_validator.validate_model_quality(model, X_val, y_val)
                    if not model_quality["is_valid"]:
                        validation_issues.extend(model_quality["issues"])
            
            # Pipeline quality validation
            if pipeline_quality_checks:
                pipeline_state = {
                    "step_name": func.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "status": "SUCCESS" if not validation_issues else "FAILED"
                }
                pipeline_quality = _comprehensive_validator.validate_pipeline_quality(pipeline_state)
                if not pipeline_quality["is_valid"]:
                    validation_issues.extend(pipeline_quality["issues"])
            
            # Output validation
            if output_validation:
                output_quality = _comprehensive_validator.validate_data_quality(result)
                if not output_quality["is_valid"]:
                    validation_issues.extend(output_quality["issues"])
            
            # Handle validation issues based on level
            if validation_issues:
                if validation_level == ValidationLevel.ERROR:
                    raise ValueError(f"Validation failed: {validation_issues}")
                elif validation_level == ValidationLevel.WARNING:
                    for issue in validation_issues:
                        system_logger.warning(f"⚠️ Validation issue: {issue}")
                elif validation_level == ValidationLevel.CRITICAL:
                    system_logger.critical(f"🚨 Critical validation issues: {validation_issues}")
                    raise ValueError(f"Critical validation failed: {validation_issues}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            validation_issues = []
            
            # Data quality validation
            if data_quality_checks:
                for arg in args:
                    if isinstance(arg, (pd.DataFrame, np.ndarray)):
                        data_quality = _comprehensive_validator.validate_data_quality(arg)
                        if not data_quality["is_valid"]:
                            validation_issues.extend(data_quality["issues"])
                        if data_quality["warnings"]:
                            for warning in data_quality["warnings"]:
                                system_logger.warning(f"⚠️ Data quality warning: {warning}")
            
            # Model quality validation
            if model_quality_checks and isinstance(result, dict) and "model" in result:
                model = result["model"]
                X_val = result.get("X_val")
                y_val = result.get("y_val")
                
                if X_val is not None and y_val is not None:
                    model_quality = _comprehensive_validator.validate_model_quality(model, X_val, y_val)
                    if not model_quality["is_valid"]:
                        validation_issues.extend(model_quality["issues"])
            
            # Pipeline quality validation
            if pipeline_quality_checks:
                pipeline_state = {
                    "step_name": func.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "status": "SUCCESS" if not validation_issues else "FAILED"
                }
                pipeline_quality = _comprehensive_validator.validate_pipeline_quality(pipeline_state)
                if not pipeline_quality["is_valid"]:
                    validation_issues.extend(pipeline_quality["issues"])
            
            # Output validation
            if output_validation:
                output_quality = _comprehensive_validator.validate_data_quality(result)
                if not output_quality["is_valid"]:
                    validation_issues.extend(output_quality["issues"])
            
            # Handle validation issues based on level
            if validation_issues:
                if validation_level == ValidationLevel.ERROR:
                    raise ValueError(f"Validation failed: {validation_issues}")
                elif validation_level == ValidationLevel.WARNING:
                    for issue in validation_issues:
                        system_logger.warning(f"⚠️ Validation issue: {issue}")
                elif validation_level == ValidationLevel.CRITICAL:
                    system_logger.critical(f"🚨 Critical validation issues: {validation_issues}")
                    raise ValueError(f"Critical validation failed: {validation_issues}")
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Export all decorators
__all__ = [
    "performance_monitor",
    "model_validation",
    "pipeline_checkpoint",
    "intelligent_caching",
    "adaptive_resource_allocation",
    "comprehensive_validation",
    "PerformanceLevel",
    "ValidationLevel",
    "PerformanceMonitor",
    "ModelValidator",
    "DataPipelineManager",
    "IntelligentCache",
    "AdaptiveResourceManager",
    "ComprehensiveValidator",
]