#!/usr/bin/env python3
"""
Enhanced Common Operations Utility Module

This module provides enhanced commonly used operations with comprehensive:
1. Data formatting and standardization
2. Data analysis and validation
3. Data access protection and security
4. Error handling and recovery
5. Performance monitoring and optimization
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager, contextmanager

# Import base common operations
from .common_operations import (
    get_current_datetime,
    format_datetime,
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
)

# Import framework components
from .data_formatting_framework import DataFormattingFramework, DataFormat
from .data_quality_framework import DataQualityFramework
from .security_framework import SecurityFramework
from .logger import system_logger

logger = system_logger.getChild("EnhancedCommonOperations")


class DataAccessManager:
    """Manager for secure data access with comprehensive protection."""
    
    def __init__(self):
        self.logger = system_logger.getChild("DataAccessManager")
        self.security = SecurityFramework()
        self.data_quality = DataQualityFramework()
        self.access_log = []
        self.access_lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the data access manager."""
        try:
            await self.security.initialize()
            await self.data_quality.initialize()
            self.logger.info("✅ Data access manager initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize data access manager: {e}")
            return False
    
    async def secure_data_read(
        self,
        file_path: Union[str, Path],
        operation_type: str = "data_read",
        validate_format: bool = True,
        encrypt_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Securely read data with comprehensive validation and protection."""
        start_time = time.time()
        
        try:
            # Validate file path
            if not safe_file_exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check access permissions
            access_check = await self.security.validate_data_access(
                operation_type=operation_type,
                data_context="read",
                file_path=str(file_path)
            )
            
            if not access_check.get('allowed', True):
                raise PermissionError(f"Access denied: {access_check.get('reason')}")
            
            # Log access
            await self._log_data_access("read", str(file_path), operation_type)
            
            # Read data based on file extension
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.json':
                data = safe_json_load(file_path)
            elif file_path.suffix.lower() in ['.parquet', '.pq']:
                data = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
            
            # Validate data format if requested
            if validate_format:
                validation_result = await self.data_quality.validate_data_quality(
                    data=data,
                    data_type="read_data",
                    file_path=str(file_path)
                )
                
                if not validation_result.get('valid', True):
                    self.logger.warning(f"⚠️ Data quality validation failed: {validation_result.get('warnings')}")
            
            # Encrypt sensitive data if requested
            if encrypt_sensitive:
                data = await self.security.encrypt_sensitive_data(data, operation_type)
            
            duration = time.time() - start_time
            self.logger.info(f"✅ Secure data read completed: {file_path} in {duration:.3f}s")
            
            return {
                'success': True,
                'data': data,
                'file_path': str(file_path),
                'operation_type': operation_type,
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Secure data read failed: {file_path} - {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'operation_type': operation_type,
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def secure_data_write(
        self,
        data: Any,
        file_path: Union[str, Path],
        operation_type: str = "data_write",
        validate_format: bool = True,
        backup_existing: bool = True
    ) -> Dict[str, Any]:
        """Securely write data with comprehensive validation and protection."""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            # Check access permissions
            access_check = await self.security.validate_data_access(
                operation_type=operation_type,
                data_context="write",
                file_path=str(file_path)
            )
            
            if not access_check.get('allowed', True):
                raise PermissionError(f"Access denied: {access_check.get('reason')}")
            
            # Validate data format if requested
            if validate_format:
                validation_result = await self.data_quality.validate_data_quality(
                    data=data,
                    data_type="write_data",
                    file_path=str(file_path)
                )
                
                if not validation_result.get('valid', True):
                    self.logger.warning(f"⚠️ Data quality validation failed: {validation_result.get('warnings')}")
            
            # Backup existing file if requested
            if backup_existing and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{int(time.time())}")
                file_path.rename(backup_path)
                self.logger.info(f"📦 Created backup: {backup_path}")
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data based on file extension
            if file_path.suffix.lower() == '.json':
                safe_json_dump(data, file_path, indent=2)
            elif file_path.suffix.lower() in ['.parquet', '.pq']:
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path, index=False)
                else:
                    raise ValueError("Data must be a DataFrame for parquet format")
            elif file_path.suffix.lower() == '.csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False)
                else:
                    raise ValueError("Data must be a DataFrame for CSV format")
            elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # Write as text
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            
            # Log access
            await self._log_data_access("write", str(file_path), operation_type)
            
            duration = time.time() - start_time
            self.logger.info(f"✅ Secure data write completed: {file_path} in {duration:.3f}s")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'operation_type': operation_type,
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Secure data write failed: {file_path} - {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'operation_type': operation_type,
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _log_data_access(self, access_type: str, file_path: str, operation_type: str) -> None:
        """Log data access for audit trail."""
        with self.access_lock:
            access_record = {
                'access_type': access_type,
                'file_path': file_path,
                'operation_type': operation_type,
                'timestamp': get_current_datetime().isoformat(),
                'user_id': 'system',  # In a real system, this would be the actual user
                'session_id': threading.current_thread().ident
            }
            
            self.access_log.append(access_record)
            
            # Keep only last 10000 access records
            if len(self.access_log) > 10000:
                self.access_log = self.access_log[-10000:]
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log for audit purposes."""
        with self.access_lock:
            return self.access_log.copy()


class DataAnalysisManager:
    """Manager for comprehensive data analysis with validation and protection."""
    
    def __init__(self):
        self.logger = system_logger.getChild("DataAnalysisManager")
        self.data_formatter = DataFormattingFramework()
        self.data_quality = DataQualityFramework()
        self.analysis_cache = {}
        self.cache_lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the data analysis manager."""
        try:
            await self.data_quality.initialize()
            self.logger.info("✅ Data analysis manager initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize data analysis manager: {e}")
            return False
    
    async def analyze_dataframe(
        self,
        df: pd.DataFrame,
        analysis_type: str = "comprehensive",
        cache_results: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive DataFrame analysis with validation."""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(df, analysis_type)
            
            # Check cache if enabled
            if cache_results:
                cached_result = await self._get_cached_analysis(cache_key)
                if cached_result:
                    self.logger.info(f"📊 Using cached analysis for {analysis_type}")
                    return cached_result
            
            # Validate DataFrame
            validation_result = await self.data_quality.validate_data_quality(
                data=df,
                data_type="dataframe_analysis"
            )
            
            if not validation_result.get('valid', True):
                self.logger.warning(f"⚠️ DataFrame validation failed: {validation_result.get('warnings')}")
            
            # Perform analysis based on type
            if analysis_type == "comprehensive":
                analysis_result = await self._comprehensive_analysis(df)
            elif analysis_type == "statistical":
                analysis_result = await self._statistical_analysis(df)
            elif analysis_type == "quality":
                analysis_result = await self._quality_analysis(df)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Add metadata
            analysis_result.update({
                'analysis_type': analysis_type,
                'dataframe_shape': df.shape,
                'columns': list(df.columns),
                'duration': time.time() - start_time,
                'timestamp': get_current_datetime().isoformat(),
                'validation_result': validation_result
            })
            
            # Cache results if enabled
            if cache_results:
                await self._cache_analysis(cache_key, analysis_result)
            
            duration = time.time() - start_time
            self.logger.info(f"✅ DataFrame analysis completed: {analysis_type} in {duration:.3f}s")
            
            return analysis_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ DataFrame analysis failed: {analysis_type} - {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': analysis_type,
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive DataFrame analysis."""
        try:
            analysis = {
                'basic_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'index_type': str(type(df.index))
                },
                'data_quality': {
                    'null_counts': df.isnull().sum().to_dict(),
                    'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
                    'duplicate_rows': df.duplicated().sum(),
                    'duplicate_percentage': (df.duplicated().sum() / len(df) * 100)
                },
                'statistical_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
                'categorical_info': {},
                'numeric_info': {}
            }
            
            # Analyze categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                analysis['categorical_info'][col] = {
                    'unique_count': df[col].nunique(),
                    'unique_values': df[col].unique().tolist()[:10],  # First 10 unique values
                    'most_common': df[col].value_counts().head(5).to_dict()
                }
            
            # Analyze numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                analysis['numeric_info'][col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'outliers': self._detect_outliers(df[col])
                }
            
            return analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Comprehensive analysis failed: {e}")
            return {'error': str(e)}
    
    async def _statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on DataFrame."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'error': 'No numeric columns found for statistical analysis'}
            
            analysis = {
                'correlation_matrix': numeric_df.corr().to_dict(),
                'covariance_matrix': numeric_df.cov().to_dict(),
                'descriptive_stats': numeric_df.describe().to_dict(),
                'distribution_info': {}
            }
            
            # Analyze distributions
            for col in numeric_df.columns:
                analysis['distribution_info'][col] = {
                    'is_normal': self._test_normality(numeric_df[col]),
                    'skewness': numeric_df[col].skew(),
                    'kurtosis': numeric_df[col].kurtosis(),
                    'quartiles': numeric_df[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            
            return analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Statistical analysis failed: {e}")
            return {'error': str(e)}
    
    async def _quality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality analysis."""
        try:
            analysis = {
                'completeness': {
                    'total_cells': df.size,
                    'missing_cells': df.isnull().sum().sum(),
                    'completeness_percentage': (1 - df.isnull().sum().sum() / df.size) * 100
                },
                'uniqueness': {
                    'duplicate_rows': df.duplicated().sum(),
                    'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                    'unique_rows': len(df) - df.duplicated().sum()
                },
                'validity': {},
                'consistency': {}
            }
            
            # Check validity for each column
            for col in df.columns:
                analysis['validity'][col] = {
                    'null_count': df[col].isnull().sum(),
                    'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                    'unique_count': df[col].nunique(),
                    'data_type': str(df[col].dtype)
                }
            
            return analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Quality analysis failed: {e}")
            return {'error': str(e)}
    
    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> Dict[str, Any]:
        """Detect outliers in a series."""
        try:
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = series[z_scores > 3]
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            return {
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100,
                'indices': outliers.index.tolist(),
                'values': outliers.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_normality(self, series: pd.Series) -> bool:
        """Test if a series follows normal distribution."""
        try:
            from scipy import stats
            _, p_value = stats.normaltest(series.dropna())
            return p_value > 0.05  # If p > 0.05, we cannot reject the null hypothesis of normality
        except:
            return False
    
    def _generate_cache_key(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Generate cache key for DataFrame analysis."""
        try:
            # Create hash from DataFrame content and analysis type
            content_hash = hashlib.md5(
                f"{df.shape}_{df.columns.tolist()}_{analysis_type}_{df.iloc[0].to_string()}".encode()
            ).hexdigest()
            return f"analysis_{content_hash}"
        except:
            return f"analysis_{int(time.time())}"
    
    async def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        with self.cache_lock:
            return self.analysis_cache.get(cache_key)
    
    async def _cache_analysis(self, cache_key: str, analysis_result: Dict[str, Any]) -> None:
        """Cache analysis result."""
        with self.cache_lock:
            self.analysis_cache[cache_key] = analysis_result
            
            # Keep only last 1000 cached analyses
            if len(self.analysis_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.analysis_cache.keys())[:100]
                for key in oldest_keys:
                    del self.analysis_cache[key]


class PerformanceMonitor:
    """Monitor and optimize performance of operations."""
    
    def __init__(self):
        self.logger = system_logger.getChild("PerformanceMonitor")
        self.performance_metrics = {}
        self.metrics_lock = threading.Lock()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        start_memory = 0
        
        try:
            # Get initial memory usage
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except:
                pass
            
            self.logger.info(f"📊 Starting performance monitoring: {operation_name}")
            
            yield
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = 0
            memory_delta = 0
            
            try:
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
            except:
                pass
            
            # Record metrics
            self._record_metrics(operation_name, duration, start_memory, end_memory, memory_delta, True)
            
            self.logger.info(f"📊 Operation completed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB")
            
        except Exception as e:
            # Calculate metrics for failed operation
            duration = time.time() - start_time
            end_memory = 0
            memory_delta = 0
            
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
            except:
                pass
            
            # Record metrics for failed operation
            self._record_metrics(operation_name, duration, start_memory, end_memory, memory_delta, False, str(e))
            
            self.logger.error(f"📊 Operation failed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB - {e}")
            raise
    
    @asynccontextmanager
    async def monitor_async_operation(self, operation_name: str):
        """Async context manager for monitoring operation performance."""
        start_time = time.time()
        start_memory = 0
        
        try:
            # Get initial memory usage
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except:
                pass
            
            self.logger.info(f"📊 Starting async performance monitoring: {operation_name}")
            
            yield
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = 0
            memory_delta = 0
            
            try:
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
            except:
                pass
            
            # Record metrics
            self._record_metrics(operation_name, duration, start_memory, end_memory, memory_delta, True)
            
            self.logger.info(f"📊 Async operation completed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB")
            
        except Exception as e:
            # Calculate metrics for failed operation
            duration = time.time() - start_time
            end_memory = 0
            memory_delta = 0
            
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
            except:
                pass
            
            # Record metrics for failed operation
            self._record_metrics(operation_name, duration, start_memory, end_memory, memory_delta, False, str(e))
            
            self.logger.error(f"📊 Async operation failed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB - {e}")
            raise
    
    def _record_metrics(self, operation_name: str, duration: float, start_memory: float, end_memory: float, memory_delta: float, success: bool, error: str = None) -> None:
        """Record performance metrics."""
        with self.metrics_lock:
            if operation_name not in self.performance_metrics:
                self.performance_metrics[operation_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0,
                    'total_memory_delta': 0.0,
                    'avg_memory_delta': 0.0,
                    'last_execution': None,
                    'errors': []
                }
            
            metrics = self.performance_metrics[operation_name]
            metrics['total_executions'] += 1
            metrics['total_duration'] += duration
            metrics['total_memory_delta'] += memory_delta
            metrics['last_execution'] = get_current_datetime().isoformat()
            
            if success:
                metrics['successful_executions'] += 1
            else:
                metrics['failed_executions'] += 1
                if error:
                    metrics['errors'].append({
                        'error': error,
                        'timestamp': get_current_datetime().isoformat()
                    })
                    # Keep only last 100 errors
                    if len(metrics['errors']) > 100:
                        metrics['errors'] = metrics['errors'][-100:]
            
            # Update calculated metrics
            metrics['avg_duration'] = metrics['total_duration'] / metrics['total_executions']
            metrics['avg_memory_delta'] = metrics['total_memory_delta'] / metrics['total_executions']
            metrics['min_duration'] = min(metrics['min_duration'], duration)
            metrics['max_duration'] = max(metrics['max_duration'], duration)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations."""
        with self.metrics_lock:
            return {
                'total_operations': len(self.performance_metrics),
                'operations': self.performance_metrics.copy(),
                'timestamp': get_current_datetime().isoformat()
            }
    
    def get_operation_metrics(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific operation."""
        with self.metrics_lock:
            return self.performance_metrics.get(operation_name)


# Global instances for easy access
data_access_manager = DataAccessManager()
data_analysis_manager = DataAnalysisManager()
performance_monitor = PerformanceMonitor()


# Convenience functions
async def secure_read_data(file_path: Union[str, Path], operation_type: str = "data_read") -> Dict[str, Any]:
    """Convenience function for secure data reading."""
    return await data_access_manager.secure_data_read(file_path, operation_type)


async def secure_write_data(data: Any, file_path: Union[str, Path], operation_type: str = "data_write") -> Dict[str, Any]:
    """Convenience function for secure data writing."""
    return await data_access_manager.secure_data_write(data, file_path, operation_type)


async def analyze_dataframe(df: pd.DataFrame, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Convenience function for DataFrame analysis."""
    return await data_analysis_manager.analyze_dataframe(df, analysis_type)


def monitor_operation(operation_name: str):
    """Convenience function for operation monitoring."""
    return performance_monitor.monitor_operation(operation_name)


def monitor_async_operation(operation_name: str):
    """Convenience function for async operation monitoring."""
    return performance_monitor.monitor_async_operation(operation_name)