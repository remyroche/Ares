"""
Step05 Optimized Validation Module

This module provides optimized validation capabilities with shared caching,
batch processing, and comprehensive logging for Step05 labeling operations.
"""

import pandas as pd
import numpy as np
import hashlib
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import logging

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates

logger = system_logger.getChild('Step05OptimizedValidation')


@dataclass
class ValidationCache:
    """Cache for validation results to avoid redundant computation."""
    data_hash: str
    validation_type: str
    result: Any
    timestamp: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0


@dataclass
class BatchValidationResult:
    """Result of batch validation operations."""
    passed: bool
    score: float
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    details: Dict[str, Any]
    computation_time: float
    cache_hits: int = 0
    cache_misses: int = 0


class Step05OptimizedValidator:
    """Optimized validator with caching, batch processing, and comprehensive logging."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.validation_cache: Dict[str, ValidationCache] = {}
        self.validation_history = []
        self.performance_stats = {
            'total_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_computation_time': 0.0,
            'avg_computation_time': 0.0
        }
        
        self.logger.info("🚀 Initializing Step05 Optimized Validator")
        self.logger.info(f"📊 Cache size limit: {self.config.get('cache_size_limit', 1000)}")
        self.logger.info(f"⏱️ Cache TTL: {self.config.get('cache_ttl_hours', 24)} hours")
    
    def _generate_data_hash(self, data: pd.DataFrame, validation_type: str) -> str:
        """Generate hash for data to enable caching."""
        try:
            # Create a hash based on data shape, dtypes, and sample values
            hash_components = [
                str(data.shape),
                str(data.dtypes.to_dict()),
                str(data.index.name),
                str(data.columns.tolist()),
                validation_type
            ]
            
            # Add sample of data for content-based hashing
            if len(data) > 0:
                sample_size = min(100, len(data))
                sample_data = data.iloc[::max(1, len(data) // sample_size)]
                hash_components.append(str(pd.util.hash_pandas_object(sample_data).sum()))
            
            hash_string = "|".join(hash_components)
            return hashlib.md5(hash_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"❌ Error generating data hash: {e}")
            return f"error_hash_{int(time.time())}"
    
    def _get_cached_result(self, data_hash: str, validation_type: str) -> Optional[Any]:
        """Get cached validation result if available and not expired."""
        try:
            cache_key = f"{data_hash}_{validation_type}"
            
            if cache_key not in self.validation_cache:
                self.performance_stats['cache_misses'] += 1
                return None
            
            cache_entry = self.validation_cache[cache_key]
            
            # Check if cache entry is expired
            cache_age = datetime.now() - cache_entry.timestamp
            ttl_hours = self.config.get('cache_ttl_hours', 24)
            
            if cache_age.total_seconds() > ttl_hours * 3600:
                self.logger.info(f"🗑️ Cache entry expired for {validation_type} (age: {cache_age})")
                del self.validation_cache[cache_key]
                self.performance_stats['cache_misses'] += 1
                return None
            
            self.performance_stats['cache_hits'] += 1
            self.logger.debug(f"✅ Cache hit for {validation_type} (age: {cache_age})")
            return cache_entry.result
            
        except Exception as e:
            self.logger.error(f"❌ Error accessing cache: {e}")
            self.performance_stats['cache_misses'] += 1
            return None
    
    def _cache_result(self, data_hash: str, validation_type: str, result: Any, computation_time: float):
        """Cache validation result."""
        try:
            cache_key = f"{data_hash}_{validation_type}"
            
            # Limit cache size
            max_cache_size = self.config.get('cache_size_limit', 1000)
            if len(self.validation_cache) >= max_cache_size:
                # Remove oldest entries
                oldest_key = min(self.validation_cache.keys(), 
                               key=lambda k: self.validation_cache[k].timestamp)
                del self.validation_cache[oldest_key]
                self.logger.info(f"🗑️ Removed oldest cache entry: {oldest_key}")
            
            self.validation_cache[cache_key] = ValidationCache(
                data_hash=data_hash,
                validation_type=validation_type,
                result=result,
                computation_time=computation_time
            )
            
            self.logger.debug(f"💾 Cached result for {validation_type} (computation time: {computation_time:.3f}s)")
            
        except Exception as e:
            self.logger.error(f"❌ Error caching result: {e}")
    
    @traced(span_name='fast_fail_validation')
    @validates()
    @handles_errors()
    def fast_fail_validation(self, data: pd.DataFrame, file_path: Optional[Path] = None) -> bool:
        """
        Fast fail validation to prevent expensive operations on invalid data.
        
        Args:
            data: DataFrame to validate
            file_path: Optional file path for file-based validation
            
        Returns:
            True if data passes fast fail checks, False otherwise
        """
        start_time = time.time()
        self.logger.info("⚡ Starting fast fail validation...")
        
        try:
            # Check 1: Basic data structure
            if data is None or data.empty:
                self.logger.error("❌ FAST FAIL: Data is None or empty")
                return False
            
            self.logger.info(f"📊 Data shape: {data.shape}")
            self.logger.info(f"📋 Data columns: {list(data.columns)}")
            
            # Check 2: Required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"❌ FAST FAIL: Missing required columns: {missing_columns}")
                self.logger.error(f"📋 Available columns: {list(data.columns)}")
                return False
            
            self.logger.info("✅ Required columns present")
            
            # Check 3: Data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    self.logger.error(f"❌ FAST FAIL: Column '{col}' is not numeric (type: {data[col].dtype})")
                    return False
            
            self.logger.info("✅ Data types valid")
            
            # Check 4: Basic OHLC relationships
            ohlc_errors = 0
            ohlc_errors += (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            ohlc_errors += (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            ohlc_errors += (data['high'] < data['low']).sum()
            
            if ohlc_errors > 0:
                self.logger.error(f"❌ FAST FAIL: Found {ohlc_errors} OHLC relationship errors")
                self.logger.error("🔍 Sample OHLC errors:")
                error_mask = (data['high'] < data[['open', 'close']].max(axis=1)) | \
                           (data['low'] > data[['open', 'close']].min(axis=1)) | \
                           (data['high'] < data['low'])
                error_samples = data[error_mask].head(5)
                for idx, row in error_samples.iterrows():
                    self.logger.error(f"   Row {idx}: O={row['open']:.4f}, H={row['high']:.4f}, L={row['low']:.4f}, C={row['close']:.4f}")
                return False
            
            self.logger.info("✅ OHLC relationships valid")
            
            # Check 5: Null values
            null_counts = data[required_columns].isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                self.logger.warning(f"⚠️ Found {total_nulls} null values:")
                for col, count in null_counts.items():
                    if count > 0:
                        self.logger.warning(f"   {col}: {count} nulls")
                
                # Allow up to 5% null values
                null_percentage = total_nulls / (len(data) * len(required_columns))
                if null_percentage > 0.05:
                    self.logger.error(f"❌ FAST FAIL: Too many null values ({null_percentage:.1%})")
                    return False
            
            self.logger.info("✅ Null value check passed")
            
            # Check 6: File-based validation
            if file_path:
                if not self._validate_file_fast_fail(file_path):
                    return False
            
            # Check 7: System resources
            if not self._check_system_resources_fast_fail():
                return False
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ Fast fail validation passed in {elapsed_time:.3f}s")
            
            return True
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Fast fail validation failed after {elapsed_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return False
    
    def _validate_file_fast_fail(self, file_path: Path) -> bool:
        """Validate file for fast fail checks."""
        try:
            self.logger.info(f"📁 Validating file: {file_path}")
            
            # Check file existence
            if not file_path.exists():
                self.logger.error(f"❌ FAST FAIL: File does not exist: {file_path}")
                return False
            
            # Check file size
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            self.logger.info(f"📏 File size: {file_size_mb:.2f} MB")
            
            if file_size == 0:
                self.logger.error("❌ FAST FAIL: File is empty")
                return False
            
            # Check for extremely large files
            max_file_size_mb = self.config.get('max_file_size_mb', 1000)
            if file_size_mb > max_file_size_mb:
                self.logger.warning(f"⚠️ Large file detected ({file_size_mb:.2f} MB > {max_file_size_mb} MB)")
                self.logger.warning("💡 Consider using chunked processing for large files")
            
            # Check file age
            file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
            max_file_age_days = self.config.get('max_file_age_days', 7)
            
            if file_age.days > max_file_age_days:
                self.logger.warning(f"⚠️ Stale data detected (age: {file_age.days} days > {max_file_age_days} days)")
                self.logger.warning("💡 Consider refreshing data source")
            
            self.logger.info(f"📅 File age: {file_age.days} days, {file_age.seconds // 3600} hours")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ File validation failed: {e}")
            return False
    
    def _check_system_resources_fast_fail(self) -> bool:
        """Check system resources for fast fail validation."""
        try:
            self.logger.info("🖥️ Checking system resources...")
            
            # Check available memory
            memory_info = psutil.virtual_memory()
            available_memory_gb = memory_info.available / (1024**3)
            total_memory_gb = memory_info.total / (1024**3)
            memory_percent = memory_info.percent
            
            self.logger.info(f"💾 Memory: {available_memory_gb:.1f}GB available / {total_memory_gb:.1f}GB total ({memory_percent:.1f}% used)")
            
            min_available_memory_gb = self.config.get('min_available_memory_gb', 2.0)
            if available_memory_gb < min_available_memory_gb:
                self.logger.error(f"❌ FAST FAIL: Insufficient memory ({available_memory_gb:.1f}GB < {min_available_memory_gb}GB)")
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.logger.info(f"🖥️ CPU usage: {cpu_percent:.1f}%")
            
            max_cpu_percent = self.config.get('max_cpu_percent', 95)
            if cpu_percent > max_cpu_percent:
                self.logger.warning(f"⚠️ High CPU usage ({cpu_percent:.1f}% > {max_cpu_percent}%)")
                self.logger.warning("💡 Consider reducing concurrent operations")
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            total_space_gb = disk_usage.total / (1024**3)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            self.logger.info(f"💿 Disk: {free_space_gb:.1f}GB free / {total_space_gb:.1f}GB total ({disk_percent:.1f}% used)")
            
            min_free_space_gb = self.config.get('min_free_space_gb', 5.0)
            if free_space_gb < min_free_space_gb:
                self.logger.warning(f"⚠️ Low disk space ({free_space_gb:.1f}GB < {min_free_space_gb}GB)")
                self.logger.warning("💡 Consider cleaning up temporary files")
            
            self.logger.info("✅ System resource checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System resource check failed: {e}")
            return False
    
    @traced(span_name='batch_validate_all')
    @validates()
    @handles_errors()
    def batch_validate_all(self, data: pd.DataFrame, barrier_params: Dict[str, Any]) -> BatchValidationResult:
        """
        Perform all validations in a single batch operation with caching.
        
        Args:
            data: DataFrame to validate
            barrier_params: Triple barrier parameters
            
        Returns:
            BatchValidationResult with all validation results
        """
        start_time = time.time()
        self.logger.info("🔄 Starting batch validation with caching...")
        
        warnings = []
        errors = []
        recommendations = []
        details = {}
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Generate data hash for caching
            data_hash = self._generate_data_hash(data, "batch_validation")
            self.logger.info(f"🔑 Data hash: {data_hash[:16]}...")
            
            # 1. Data Integrity Validation
            integrity_result = self._validate_data_integrity_cached(data, data_hash)
            if integrity_result:
                details['data_integrity'] = integrity_result
                if not integrity_result.get('passed', False):
                    errors.extend(integrity_result.get('errors', []))
                warnings.extend(integrity_result.get('warnings', []))
                recommendations.extend(integrity_result.get('recommendations', []))
                cache_hits += 1 if integrity_result.get('cached', False) else 0
                cache_misses += 1 if not integrity_result.get('cached', False) else 0
            
            # 2. Lookahead Bias Validation
            bias_result = self._validate_lookahead_bias_cached(data, barrier_params, data_hash)
            if bias_result:
                details['lookahead_bias'] = bias_result
                if bias_result.get('bias_detected', False):
                    errors.append("Lookahead bias detected")
                warnings.extend(bias_result.get('recommendations', []))
                cache_hits += 1 if bias_result.get('cached', False) else 0
                cache_misses += 1 if not bias_result.get('cached', False) else 0
            
            # 3. Temporal Consistency Validation
            temporal_result = self._validate_temporal_consistency_cached(data, data_hash)
            if temporal_result:
                details['temporal_consistency'] = temporal_result
                if not temporal_result.get('passed', False):
                    errors.extend(temporal_result.get('errors', []))
                warnings.extend(temporal_result.get('warnings', []))
                cache_hits += 1 if temporal_result.get('cached', False) else 0
                cache_misses += 1 if not temporal_result.get('cached', False) else 0
            
            # 4. OHLC Validation
            ohlc_result = self._validate_ohlc_comprehensive_cached(data, data_hash)
            if ohlc_result:
                details['ohlc_validation'] = ohlc_result
                if not ohlc_result.get('passed', False):
                    errors.extend(ohlc_result.get('errors', []))
                warnings.extend(ohlc_result.get('warnings', []))
                cache_hits += 1 if ohlc_result.get('cached', False) else 0
                cache_misses += 1 if not ohlc_result.get('cached', False) else 0
            
            # Calculate overall score
            scores = []
            for result in details.values():
                if isinstance(result, dict) and 'score' in result:
                    scores.append(result['score'])
            
            overall_score = np.mean(scores) if scores else 0.0
            passed = len(errors) == 0 and overall_score > 0.7
            
            computation_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_validations'] += 1
            self.performance_stats['cache_hits'] += cache_hits
            self.performance_stats['cache_misses'] += cache_misses
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_validations']
            )
            
            result = BatchValidationResult(
                passed=passed,
                score=overall_score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                details=details,
                computation_time=computation_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
            self.logger.info(f"✅ Batch validation completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Cache performance: {cache_hits} hits, {cache_misses} misses")
            self.logger.info(f"📈 Overall score: {overall_score:.3f}, Passed: {passed}")
            
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Batch validation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            return BatchValidationResult(
                passed=False,
                score=0.0,
                warnings=[],
                errors=[f"Batch validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"],
                details={'error': str(e)},
                computation_time=computation_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
    
    def _validate_data_integrity_cached(self, data: pd.DataFrame, data_hash: str) -> Optional[Dict[str, Any]]:
        """Validate data integrity with caching."""
        validation_type = "data_integrity"
        
        # Check cache first
        cached_result = self._get_cached_result(data_hash, validation_type)
        if cached_result is not None:
            cached_result['cached'] = True
            return cached_result
        
        # Perform validation
        start_time = time.time()
        try:
            self.logger.info("🔍 Validating data integrity...")
            
            warnings = []
            errors = []
            recommendations = []
            details = {}
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
                recommendations.append("Ensure all OHLCV data is present")
            
            details['missing_columns'] = missing_columns
            
            # Check data completeness
            if not data.empty:
                completeness = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
                details['completeness'] = completeness
                
                if completeness < 0.95:
                    warnings.append(f"Data completeness is {completeness:.1%}, below 95% threshold")
                    recommendations.append("Review data collection process for missing values")
            
            # Check for duplicate rows
            duplicates = data.duplicated().sum()
            details['duplicates'] = duplicates
            if duplicates > 0:
                warnings.append(f"Found {duplicates} duplicate rows")
                recommendations.append("Remove duplicate records")
            
            # Calculate score
            score = 1.0
            score -= len(missing_columns) * 0.2
            score -= (1.0 - details.get('completeness', 1.0)) * 0.5
            score -= min(duplicates * 0.01, 0.3)
            score = max(score, 0.0)
            
            passed = len(errors) == 0 and score > 0.8
            
            result = {
                'passed': passed,
                'score': score,
                'warnings': warnings,
                'errors': errors,
                'recommendations': recommendations,
                'details': details,
                'cached': False
            }
            
            computation_time = time.time() - start_time
            self._cache_result(data_hash, validation_type, result, computation_time)
            
            self.logger.info(f"✅ Data integrity validation completed in {computation_time:.3f}s")
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Data integrity validation failed: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'warnings': [],
                'errors': [f"Data integrity validation failed: {str(e)}"],
                'recommendations': ["Fix data integrity issues"],
                'details': {'error': str(e)},
                'cached': False
            }
    
    def _validate_lookahead_bias_cached(self, data: pd.DataFrame, barrier_params: Dict[str, Any], data_hash: str) -> Optional[Dict[str, Any]]:
        """Validate lookahead bias with caching."""
        validation_type = "lookahead_bias"
        
        # Check cache first
        cached_result = self._get_cached_result(data_hash, validation_type)
        if cached_result is not None:
            cached_result['cached'] = True
            return cached_result
        
        # Perform validation
        start_time = time.time()
        try:
            self.logger.info("🔍 Validating lookahead bias...")
            
            max_lookahead = barrier_params.get('max_lookahead', 100)
            bias_indicators = []
            
            # Check for information leakage
            if 'label' in data.columns and 'close' in data.columns:
                future_returns = data['close'].pct_change().shift(-1)
                label_correlation = data['label'].corr(future_returns)
                
                if abs(label_correlation) > 0.3:
                    bias_indicators.append(f"High label-future return correlation: {label_correlation:.3f}")
            
            # Check for perfect timing
            perfect_timing = 0
            for i in range(len(data) - max_lookahead):
                if pd.isna(data['label'].iloc[i]) or data['label'].iloc[i] == 0:
                    continue
                
                future_window = data['close'].iloc[i+1:i+max_lookahead+1]
                if len(future_window) > 0:
                    max_future_move = future_window.pct_change().abs().max()
                    if max_future_move > 0.005:
                        if (data['label'].iloc[i] == 1 and future_window.pct_change().max() > 0.002) or \
                           (data['label'].iloc[i] == -1 and future_window.pct_change().min() < -0.002):
                            perfect_timing += 1
            
            if perfect_timing > len(data) * 0.1:
                bias_indicators.append(f"High perfect timing rate: {perfect_timing/len(data):.1%}")
            
            bias_detected = len(bias_indicators) > 0
            bias_score = min(len(bias_indicators) * 0.3, 1.0)
            
            result = {
                'bias_detected': bias_detected,
                'bias_score': bias_score,
                'temporal_violations': 0,
                'future_data_leakage': bias_detected,
                'recommendations': bias_indicators,
                'details': {'bias_indicators': bias_indicators},
                'cached': False
            }
            
            computation_time = time.time() - start_time
            self._cache_result(data_hash, validation_type, result, computation_time)
            
            self.logger.info(f"✅ Lookahead bias validation completed in {computation_time:.3f}s")
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Lookahead bias validation failed: {e}")
            return {
                'bias_detected': True,
                'bias_score': 1.0,
                'temporal_violations': 0,
                'future_data_leakage': True,
                'recommendations': [f"Lookahead bias validation failed: {str(e)}"],
                'details': {'error': str(e)},
                'cached': False
            }
    
    def _validate_temporal_consistency_cached(self, data: pd.DataFrame, data_hash: str) -> Optional[Dict[str, Any]]:
        """Validate temporal consistency with caching."""
        validation_type = "temporal_consistency"
        
        # Check cache first
        cached_result = self._get_cached_result(data_hash, validation_type)
        if cached_result is not None:
            cached_result['cached'] = True
            return cached_result
        
        # Perform validation
        start_time = time.time()
        try:
            self.logger.info("🔍 Validating temporal consistency...")
            
            warnings = []
            errors = []
            recommendations = []
            details = {}
            
            # Check for gaps in time series
            if hasattr(data.index, 'to_pydatetime'):
                time_diffs = data.index.to_series().diff()
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else None
                
                if expected_interval:
                    gaps = time_diffs[time_diffs > expected_interval * 2]
                    if len(gaps) > 0:
                        warnings.append(f"Found {len(gaps)} time gaps in data")
                        details['time_gaps'] = len(gaps)
            
            # Check for duplicate timestamps
            if data.index.duplicated().any():
                errors.append("Duplicate timestamps found")
                details['duplicate_timestamps'] = data.index.duplicated().sum()
            
            # Check temporal ordering
            if hasattr(data.index, 'to_pydatetime'):
                ordering_violations = 0
                for i in range(1, len(data.index)):
                    if data.index[i] < data.index[i-1]:
                        ordering_violations += 1
                
                if ordering_violations > 0:
                    errors.append(f"Found {ordering_violations} temporal ordering violations")
                    details['ordering_violations'] = ordering_violations
            
            score = 1.0
            score -= len(errors) * 0.2
            score -= len(warnings) * 0.1
            score = max(score, 0.0)
            
            passed = len(errors) == 0
            
            result = {
                'passed': passed,
                'score': score,
                'warnings': warnings,
                'errors': errors,
                'recommendations': recommendations,
                'details': details,
                'cached': False
            }
            
            computation_time = time.time() - start_time
            self._cache_result(data_hash, validation_type, result, computation_time)
            
            self.logger.info(f"✅ Temporal consistency validation completed in {computation_time:.3f}s")
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Temporal consistency validation failed: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'warnings': [],
                'errors': [f"Temporal consistency validation failed: {str(e)}"],
                'recommendations': ["Fix temporal consistency issues"],
                'details': {'error': str(e)},
                'cached': False
            }
    
    def _validate_ohlc_comprehensive_cached(self, data: pd.DataFrame, data_hash: str) -> Optional[Dict[str, Any]]:
        """Validate OHLC data comprehensively with caching."""
        validation_type = "ohlc_comprehensive"
        
        # Check cache first
        cached_result = self._get_cached_result(data_hash, validation_type)
        if cached_result is not None:
            cached_result['cached'] = True
            return cached_result
        
        # Perform validation
        start_time = time.time()
        try:
            self.logger.info("🔍 Validating OHLC data comprehensively...")
            
            warnings = []
            errors = []
            recommendations = []
            details = {}
            
            # Basic OHLC relationships
            high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            hl_violations = (data['high'] < data['low']).sum()
            
            total_ohlc_errors = high_violations + low_violations + low_violations
            details['ohlc_errors'] = {
                'high_violations': int(high_violations),
                'low_violations': int(low_violations),
                'hl_violations': int(hl_violations),
                'total': int(total_ohlc_errors)
            }
            
            if total_ohlc_errors > 0:
                errors.append(f"Found {total_ohlc_errors} OHLC relationship errors")
                recommendations.append("Review price data for invalid OHLC relationships")
            
            # Price movement validation
            price_changes = data['close'].pct_change().abs()
            extreme_moves = (price_changes > 0.1).sum()
            details['extreme_moves'] = int(extreme_moves)
            
            if extreme_moves > len(data) * 0.01:
                warnings.append(f"High number of extreme price moves: {extreme_moves}")
                recommendations.append("Review data for potential outliers or data quality issues")
            
            # Volume-price relationship
            if 'volume' in data.columns:
                volume_price_corr = data['volume'].corr(data['close'].pct_change().abs())
                details['volume_price_correlation'] = float(volume_price_corr) if not pd.isna(volume_price_corr) else 0.0
                
                if volume_price_corr < 0.3:
                    warnings.append("Low volume-price correlation detected")
                    recommendations.append("Review volume data quality")
            
            # Calculate score
            score = 1.0
            score -= min(total_ohlc_errors * 0.01, 0.5)
            score -= min(extreme_moves * 0.001, 0.3)
            score = max(score, 0.0)
            
            passed = len(errors) == 0 and score > 0.7
            
            result = {
                'passed': passed,
                'score': score,
                'warnings': warnings,
                'errors': errors,
                'recommendations': recommendations,
                'details': details,
                'cached': False
            }
            
            computation_time = time.time() - start_time
            self._cache_result(data_hash, validation_type, result, computation_time)
            
            self.logger.info(f"✅ OHLC comprehensive validation completed in {computation_time:.3f}s")
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ OHLC comprehensive validation failed: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'warnings': [],
                'errors': [f"OHLC comprehensive validation failed: {str(e)}"],
                'recommendations': ["Fix OHLC data issues"],
                'details': {'error': str(e)},
                'cached': False
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'performance_stats': self.performance_stats.copy(),
            'cache_size': len(self.validation_cache),
            'cache_hit_rate': (
                self.performance_stats['cache_hits'] / 
                max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
            ),
            'avg_computation_time': self.performance_stats['avg_computation_time']
        }
    
    def clear_cache(self):
        """Clear validation cache."""
        cache_size = len(self.validation_cache)
        self.validation_cache.clear()
        self.logger.info(f"🗑️ Cleared validation cache ({cache_size} entries)")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            'cache_size': len(self.validation_cache),
            'cache_entries': [
                {
                    'key': key,
                    'validation_type': entry.validation_type,
                    'timestamp': entry.timestamp.isoformat(),
                    'computation_time': entry.computation_time
                }
                for key, entry in self.validation_cache.items()
            ]
        }