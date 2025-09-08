"""
Optimized Step17 Implementation with Utility Integration

This implementation integrates all the specified utility modules:
- src/utils/common_operations.py
- src/utils/math_validation.py  
- src/utils/parquet_utils.py
- src/core/decorators/
- src/core/errors/
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import psutil
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

# Import utility modules
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    format_datetime, safe_dict_get, safe_dict_items, safe_append,
    safe_extend, get_logger, setup_basic_logging, generate_hash,
    generate_cache_key, safe_deepcopy, safe_copy, validate_dataframe,
    validate_numeric_range, safe_sleep, safe_gather, create_async_task,
    timed_operation, format_bytes, chunked_iterable, parallel_map
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, MathValidationError
)

from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

# Import core decorators and errors
from src.core.decorators import (
    handles_errors, error_boundary, retry, timeout, circuit_breaker,
    log_call, log_execution_time, traced, cached, memoize,
    validate_dataframe as validate_df_decorator, validates
)

from src.core.errors import (
    AppError, ValidationError, NotFoundError, TimeoutError,
    ServiceUnavailableError, BusinessRuleError, DataIntegrityError,
    ErrorCode, ErrorMapper, error_mapper
)

# Import existing utilities
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.logger import system_logger
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Custom Exceptions extending core errors
class Step17ValidationError(ValidationError):
    """Custom exception for step17 validation errors."""
    def __init__(self, message: str, error_code: str = "STEP17_VALIDATION_ERROR"):
        super().__init__(message)
        self.error_code = error_code

class Step17ResourceError(ServiceUnavailableError):
    """Custom exception for step17 resource errors."""
    def __init__(self, message: str, error_code: str = "STEP17_RESOURCE_ERROR"):
        super().__init__(message)
        self.error_code = error_code

class Step17OptimizationError(BusinessRuleError):
    """Custom exception for step17 optimization errors."""
    def __init__(self, message: str, error_code: str = "STEP17_OPTIMIZATION_ERROR"):
        super().__init__(message)
        self.error_code = error_code

# Data Classes
@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class OptimizationMetrics:
    """Optimization performance metrics."""
    convergence_score: float
    optimization_efficiency: float
    parameter_stability: float
    objective_improvement: float
    memory_usage: float
    cpu_usage: float

@dataclass
class ParameterGroup:
    """Parameter group for intelligent grouping."""
    name: str
    parameters: List[str]
    correlation_strength: float
    importance_score: float
    optimization_priority: int

# Thread-safe Configuration Manager using utility functions
class ThreadSafeConfigManager:
    """Thread-safe configuration manager for step17."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._config = {}
        self._cache = {}
        self._cache_lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    async def update_config(self, params: Dict[str, Any], categories: List[str]) -> None:
        """Thread-safe configuration update."""
        async with self._lock:
            for category in categories:
                category_params = {k: v for k, v in params.items() if k in self._get_search_space(category)}
                if category_params:
                    self._config[category] = category_params
                    # Invalidate cache for this category
                    await self._invalidate_cache(category)
    
    async def get_config(self, category: str) -> Dict[str, Any]:
        """Thread-safe configuration retrieval."""
        async with self._lock:
            return safe_dict_get(self._config, category, {}).copy()
    
    async def _invalidate_cache(self, category: str) -> None:
        """Invalidate cache for a specific category."""
        with self._cache_lock:
            keys_to_remove = [k for k in self._cache.keys() if category in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
    
    def _get_search_space(self, category: str) -> Dict[str, Any]:
        """Get search space for a category."""
        # This would integrate with your existing search space logic
        return {}

# Memory Management Context using utility functions
@asynccontextmanager
async def memory_efficient_context(max_memory_gb: float = 4.0):
    """Context manager for memory-efficient operations."""
    initial_memory = psutil.virtual_memory().used / (1024**3)
    logger = get_logger(__name__)
    
    try:
        logger.debug(f"🔧 Starting memory-efficient context (limit: {max_memory_gb}GB)")
        yield
    finally:
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage
        final_memory = psutil.virtual_memory().used / (1024**3)
        memory_increase = final_memory - initial_memory
        
        if memory_increase > max_memory_gb:
            logger.warning(f"⚠️ High memory usage detected: {memory_increase:.2f}GB")
        else:
            logger.debug(f"✅ Memory usage within limits: {memory_increase:.2f}GB")

# Parameter Result Cache using utility functions
class ParameterResultCache:
    """Cache for parameter evaluation results."""
    
    def __init__(self, max_size: int = 10000):
        self._cache = {}
        self._max_size = max_size
        self._lock = threading.Lock()
        self._access_times = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger = get_logger(__name__)
    
    def _generate_cache_key(self, category: str, params: Dict[str, Any], calibration_hash: str) -> str:
        """Generate cache key for parameters using utility function."""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{category}:{params_str}:{calibration_hash}"
        return generate_hash(combined, 'md5')
    
    def get(self, category: str, params: Dict[str, Any], calibration_hash: str) -> Optional[float]:
        """Get cached result."""
        key = self._generate_cache_key(category, params, calibration_hash)
        
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                self._cache_hits += 1
                return self._cache[key]
        
        self._cache_misses += 1
        return None
    
    def set(self, category: str, params: Dict[str, Any], calibration_hash: str, result: float) -> None:
        """Set cached result."""
        key = self._generate_cache_key(category, params, calibration_hash)
        
        with self._lock:
            # Implement LRU eviction
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            self._cache[key] = result
            self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._cache.pop(lru_key, None)
        self._access_times.pop(lru_key, None)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = safe_divide(self._cache_hits, total_requests, 0.0) if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'max_size': self._max_size
        }

# Advanced Optimization Strategies using utility functions
class AdvancedOptimizationStrategies:
    """Advanced optimization strategies for step17."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.optimization_history = []
        self.performance_metrics = []
    
    @math_safe
    async def implement_adaptive_sampling(self, study: optuna.Study) -> None:
        """Implement adaptive sampling based on convergence."""
        if len(study.trials) > 50:
            # Switch to more aggressive sampling
            study.sampler = NSGAIISampler(population_size=50)
            self.logger.info("🔄 Switched to NSGA-II sampler for better convergence")
    
    @math_safe
    async def implement_early_stopping(self, study: optuna.Study) -> bool:
        """Implement intelligent early stopping."""
        if len(study.trials) < 20:
            return False
        
        recent_trials = study.trials[-10:]
        if not recent_trials:
            return False
        
        # Check for stagnation using safe math operations
        best_values = [t.value for t in recent_trials if t.value is not None]
        if len(best_values) < 5:
            return False
        
        improvement = safe_divide(max(best_values) - min(best_values), 1.0, 0.0)
        stagnation_threshold = 0.001
        
        if improvement < stagnation_threshold:
            self.logger.warning(f"⏹️ Early stopping triggered: improvement {improvement:.6f} < {stagnation_threshold}")
            return True
        
        return False
    
    @math_safe
    async def implement_parameter_pruning(self, study: optuna.Study) -> List[str]:
        """Identify and prune low-impact parameters."""
        if len(study.trials) < 30:
            return []
        
        # Analyze parameter importance using safe math operations
        importance_scores = {}
        for trial in study.trials:
            if trial.value is None:
                continue
            
            for param_name, param_value in trial.params.items():
                if param_name not in importance_scores:
                    importance_scores[param_name] = []
                importance_scores[param_name].append(trial.value)
        
        # Calculate variance for each parameter
        low_impact_params = []
        for param_name, values in importance_scores.items():
            if len(values) < 5:
                continue
            
            # Use safe math operations
            mean_val = safe_mean(values)
            variance = safe_divide(sum((v - mean_val) ** 2 for v in values), len(values), 0.0)
            
            if variance < 0.01:  # Low impact threshold
                low_impact_params.append(param_name)
        
        if low_impact_params:
            self.logger.info(f"✂️ Identified low-impact parameters for pruning: {low_impact_params}")
        
        return low_impact_params

# Intelligent Parameter Grouper using utility functions
class IntelligentParameterGrouper:
    """Intelligent parameter grouping based on correlation analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.parameter_groups = {}
        self.correlation_matrix = {}
    
    @math_safe
    async def analyze_parameter_correlations(self, optimization_history: List[Dict]) -> Dict[str, List[str]]:
        """Analyze parameter correlations to create optimal groups."""
        if len(optimization_history) < 50:
            return self._get_default_groups()
        
        # Extract parameter values and outcomes
        param_data = {}
        outcomes = []
        
        for trial in optimization_history:
            if trial.get('value') is None:
                continue
            
            outcomes.append(trial['value'])
            for param_name, param_value in trial.get('params', {}).items():
                if param_name not in param_data:
                    param_data[param_name] = []
                param_data[param_name].append(param_value)
        
        # Calculate correlations using safe math operations
        correlations = {}
        for param_name, values in param_data.items():
            if len(values) != len(outcomes):
                continue
            
            # Use safe correlation calculation
            correlation = self._safe_correlation(values, outcomes)
            correlations[param_name] = abs(correlation) if not np.isnan(correlation) else 0
        
        # Group parameters by correlation strength
        return self._create_correlation_groups(correlations)
    
    def _safe_correlation(self, x: List[float], y: List[float]) -> float:
        """Safely calculate correlation coefficient."""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            # Use safe math operations
            mean_x = safe_mean(x)
            mean_y = safe_mean(y)
            
            numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
            sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
            
            denominator = safe_sqrt(sum_sq_x * sum_sq_y, 1.0)
            return safe_divide(numerator, denominator, 0.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating correlation: {e}")
            return 0.0
    
    def _create_correlation_groups(self, correlations: Dict[str, float]) -> Dict[str, List[str]]:
        """Create parameter groups based on correlation analysis."""
        # Sort parameters by correlation strength
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Create groups
        groups = {
            'high_impact': [],
            'medium_impact': [],
            'low_impact': []
        }
        
        for param_name, correlation in sorted_params:
            if correlation > 0.3:
                groups['high_impact'] = safe_append(groups['high_impact'], param_name)
            elif correlation > 0.1:
                groups['medium_impact'] = safe_append(groups['medium_impact'], param_name)
            else:
                groups['low_impact'] = safe_append(groups['low_impact'], param_name)
        
        self.logger.info(f"📊 Created parameter groups: {len(groups['high_impact'])} high, {len(groups['medium_impact'])} medium, {len(groups['low_impact'])} low impact")
        
        return groups
    
    def _get_default_groups(self) -> Dict[str, List[str]]:
        """Get default parameter groups."""
        return {
            'high_impact': ['base_entry_threshold', 'kelly_multiplier', 'stop_loss_atr_multiplier'],
            'medium_impact': ['analyst_confidence_threshold', 'tactician_confidence_threshold', 'max_position_size'],
            'low_impact': ['learning_rate', 'n_estimators', 'max_depth']
        }

# Resource Validator using utility functions
class ResourceValidator:
    """Validator for system resources."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @handles_errors(
        default_return=ValidationResult(is_valid=False, errors=["Resource validation failed"]),
        context="ResourceValidator.validate_resources"
    )
    async def validate_resources(self) -> ValidationResult:
        """Validate system resources."""
        errors = []
        warnings = []
        
        # Check memory using safe operations
        memory = psutil.virtual_memory()
        available_memory_gb = safe_divide(memory.available, 1024**3, 0.0)
        
        if available_memory_gb < 2.0:
            errors = safe_append(errors, f"Insufficient memory: {available_memory_gb:.2f}GB available, 2GB required")
        elif available_memory_gb < 4.0:
            warnings = safe_append(warnings, f"Low memory: {available_memory_gb:.2f}GB available, 4GB recommended")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            warnings = safe_append(warnings, f"Limited CPU cores: {cpu_count}, 2+ recommended")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_space_gb = safe_divide(disk_usage.free, 1024**3, 0.0)
        
        if free_space_gb < 5.0:
            errors = safe_append(errors, f"Insufficient disk space: {free_space_gb:.2f}GB available, 5GB required")
        elif free_space_gb < 10.0:
            warnings = safe_append(warnings, f"Low disk space: {free_space_gb:.2f}GB available, 10GB recommended")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            warnings = safe_append(warnings, f"High CPU usage: {cpu_percent}%, may impact performance")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# Input Validator using utility functions
class InputValidator:
    """Validator for step17 inputs."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @handles_errors(
        default_return=ValidationResult(is_valid=False, errors=["Input validation failed"]),
        context="InputValidator.validate_training_input"
    )
    async def validate_training_input(self, training_input: Dict[str, Any]) -> ValidationResult:
        """Validate training input parameters."""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['symbol', 'exchange', 'data_dir']
        for field in required_fields:
            if field not in training_input:
                errors = safe_append(errors, f"Missing required field: {field}")
            elif not training_input[field]:
                errors = safe_append(errors, f"Empty required field: {field}")
        
        # Validate symbol format
        if 'symbol' in training_input:
            symbol = training_input['symbol']
            if not isinstance(symbol, str) or len(symbol) < 3:
                errors = safe_append(errors, f"Invalid symbol format: {symbol}")
        
        # Validate exchange
        if 'exchange' in training_input:
            exchange = training_input['exchange']
            valid_exchanges = ['BINANCE', 'COINBASE', 'KRAKEN', 'BITFINEX']
            if exchange not in valid_exchanges:
                warnings = safe_append(warnings, f"Unrecognized exchange: {exchange}")
        
        # Validate data directory using utility functions
        if 'data_dir' in training_input:
            data_dir = training_input['data_dir']
            if not safe_file_exists(data_dir):
                errors = safe_append(errors, f"Data directory does not exist: {data_dir}")
            elif not os.access(data_dir, os.R_OK):
                errors = safe_append(errors, f"Data directory not readable: {data_dir}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @handles_errors(
        default_return=ValidationResult(is_valid=False, errors=["Pipeline validation failed"]),
        context="InputValidator.validate_pipeline_state"
    )
    async def validate_pipeline_state(self, pipeline_state: Dict[str, Any]) -> ValidationResult:
        """Validate pipeline state."""
        errors = []
        warnings = []
        
        # Check for required previous step results
        required_steps = ['calibration_results', 'model_parameters']
        for step in required_steps:
            if step not in pipeline_state:
                errors = safe_append(errors, f"Missing required pipeline state: {step}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# Result Validator using utility functions
class ResultValidator:
    """Validator for optimization results."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @handles_errors(
        default_return=ValidationResult(is_valid=False, errors=["Result validation failed"]),
        context="ResultValidator.validate_optimization_results"
    )
    async def validate_optimization_results(self, results: Dict[str, Any]) -> ValidationResult:
        """Validate optimization results."""
        errors = []
        warnings = []
        
        if not results:
            errors = safe_append(errors, "Empty optimization results")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check result structure
        expected_categories = [
            'confidence', 'position_sizing', 'leverage', 'tpsl', 
            'ensemble', 'sr', 'two_tier', 'technical_indicators', 
            'system_monitoring', 'training_optimization', 
            'regime_transitions', 'signal_aggregation'
        ]
        
        for category in expected_categories:
            if category not in results:
                warnings = safe_append(warnings, f"Missing optimization results for category: {category}")
                continue
            
            result = results[category]
            if not result or 'best_params' not in result:
                errors = safe_append(errors, f"Invalid result structure for category: {category}")
                continue
            
            # Validate parameter values using utility functions
            param_validation = await self._validate_parameter_values(result['best_params'])
            if not param_validation.is_valid:
                for error in param_validation.errors:
                    errors = safe_append(errors, f"{category}: {error}")
        
        # Check convergence
        convergence_validation = await self._validate_convergence(results)
        if not convergence_validation.is_valid:
            errors = safe_extend(errors, convergence_validation.errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @handles_errors(
        default_return=ValidationResult(is_valid=False, errors=["Parameter validation failed"]),
        context="ResultValidator._validate_parameter_values"
    )
    async def _validate_parameter_values(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate parameter values using utility functions."""
        errors = []
        
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                # Use safe math validation
                try:
                    validate_finite(param_value, param_name)
                    
                    # Check for extreme values
                    if abs(param_value) > 1000:
                        errors = safe_append(errors, f"Extreme parameter value for {param_name}: {param_value}")
                    
                    # Check for zero values (might indicate issues)
                    if param_value == 0:
                        errors = safe_append(errors, f"Zero parameter value for {param_name}")
                    
                    # Check for negative values (if not expected)
                    if param_value < 0 and "threshold" not in param_name.lower():
                        errors = safe_append(errors, f"Negative parameter value for {param_name}: {param_value}")
                        
                except MathValidationError as e:
                    errors = safe_append(errors, f"Invalid parameter {param_name}: {e}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @handles_errors(
        default_return=ValidationResult(is_valid=False, errors=["Convergence validation failed"]),
        context="ResultValidator._validate_convergence"
    )
    async def _validate_convergence(self, results: Dict[str, Any]) -> ValidationResult:
        """Validate optimization convergence using utility functions."""
        errors = []
        
        for category, result in results.items():
            if 'convergence_score' in result:
                score = result['convergence_score']
                try:
                    validate_range(score, 0.0, 1.0, f"convergence_score_{category}")
                    if score < 0.5:
                        errors = safe_append(errors, f"Low convergence score for {category}: {score}")
                except MathValidationError as e:
                    errors = safe_append(errors, f"Invalid convergence score for {category}: {e}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)