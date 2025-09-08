#!/usr/bin/env python3
"""Fast Fail Validation System for Step03.

This module provides comprehensive fast fail validation mechanisms including:
1. Early data validation with extensive logging
2. Resource availability checks
3. HMM convergence validation
4. Financial metric validation
5. System health checks
6. Configuration validation
"""

import asyncio
import logging
import os
import psutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self, passed: bool, message: str, level: ValidationLevel = ValidationLevel.INFO, 
                 details: Optional[Dict[str, Any]] = None):
        self.passed = passed
        self.message = message
        self.level = level
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def __bool__(self) -> bool:
        return self.passed
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} [{self.level.value.upper()}] {self.message}"

@dataclass
class ValidationConfig:
    """Configuration for validation system."""
    min_available_memory_gb: float = 2.0
    min_disk_space_gb: float = 5.0
    max_cpu_usage_percent: float = 90.0
    min_data_rows: int = 100
    max_data_rows: int = 10000000
    required_columns: List[str] = None
    max_file_size_mb: float = 10000.0
    min_file_size_mb: float = 0.1
    enable_extensive_logging: bool = True
    validation_timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['open', 'high', 'low', 'close', 'volume']

class FastFailValidator:
    """Fast fail validation system with extensive logging."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(f"{__name__}.FastFailValidator")
        self.validation_history = []
        self.failure_count = 0
        self.warning_count = 0
        
        # Setup extensive logging
        if self.config.enable_extensive_logging:
            self._setup_extensive_logging()
    
    def _setup_extensive_logging(self) -> None:
        """Setup extensive logging for troubleshooting."""
        # Create detailed log formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Add file handler for validation logs
        log_file = Path("logs") / f"step03_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info("🔍 Extensive validation logging enabled")
        self.logger.info(f"📝 Validation logs will be saved to: {log_file}")
    
    def _log_validation_result(self, result: ValidationResult, context: str = "") -> None:
        """Log validation result with extensive details."""
        self.validation_history.append({
            'result': result,
            'context': context,
            'timestamp': datetime.now()
        })
        
        if not result.passed:
            self.failure_count += 1
            if result.level == ValidationLevel.WARNING:
                self.warning_count += 1
        
        # Log based on level
        if result.level == ValidationLevel.CRITICAL:
            self.logger.critical(f"🚨 CRITICAL VALIDATION FAILURE: {result.message}")
            if result.details:
                self.logger.critical(f"   Details: {json.dumps(result.details, indent=2)}")
        elif result.level == ValidationLevel.ERROR:
            self.logger.error(f"❌ VALIDATION ERROR: {result.message}")
            if result.details:
                self.logger.error(f"   Details: {json.dumps(result.details, indent=2)}")
        elif result.level == ValidationLevel.WARNING:
            self.logger.warning(f"⚠️ VALIDATION WARNING: {result.message}")
            if result.details:
                self.logger.warning(f"   Details: {json.dumps(result.details, indent=2)}")
        else:
            self.logger.info(f"✅ VALIDATION PASSED: {result.message}")
            if result.details:
                self.logger.debug(f"   Details: {json.dumps(result.details, indent=2)}")
    
    async def validate_system_resources(self) -> ValidationResult:
        """Validate system resource availability."""
        self.logger.info("🔍 Validating system resources...")
        
        try:
            # Check memory availability
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            if available_memory_gb < self.config.min_available_memory_gb:
                result = ValidationResult(
                    passed=False,
                    message=f"Insufficient memory: {available_memory_gb:.1f}GB available, {self.config.min_available_memory_gb}GB required",
                    level=ValidationLevel.CRITICAL,
                    details={
                        'available_memory_gb': available_memory_gb,
                        'required_memory_gb': self.config.min_available_memory_gb,
                        'total_memory_gb': memory.total / (1024**3),
                        'memory_usage_percent': memory.percent
                    }
                )
                self._log_validation_result(result, "system_resources")
                return result
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            available_disk_gb = disk_usage.free / (1024**3)
            
            if available_disk_gb < self.config.min_disk_space_gb:
                result = ValidationResult(
                    passed=False,
                    message=f"Insufficient disk space: {available_disk_gb:.1f}GB available, {self.config.min_disk_space_gb}GB required",
                    level=ValidationLevel.CRITICAL,
                    details={
                        'available_disk_gb': available_disk_gb,
                        'required_disk_gb': self.config.min_disk_space_gb,
                        'total_disk_gb': disk_usage.total / (1024**3),
                        'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100
                    }
                )
                self._log_validation_result(result, "system_resources")
                return result
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.max_cpu_usage_percent:
                result = ValidationResult(
                    passed=False,
                    message=f"High CPU usage: {cpu_percent:.1f}% > {self.config.max_cpu_usage_percent}%",
                    level=ValidationLevel.WARNING,
                    details={
                        'cpu_usage_percent': cpu_percent,
                        'max_cpu_usage_percent': self.config.max_cpu_usage_percent,
                        'cpu_count': psutil.cpu_count()
                    }
                )
                self._log_validation_result(result, "system_resources")
                return result
            
            # All checks passed
            result = ValidationResult(
                passed=True,
                message="System resources validation passed",
                level=ValidationLevel.INFO,
                details={
                    'available_memory_gb': available_memory_gb,
                    'available_disk_gb': available_disk_gb,
                    'cpu_usage_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count()
                }
            )
            self._log_validation_result(result, "system_resources")
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                message=f"Failed to validate system resources: {str(e)}",
                level=ValidationLevel.CRITICAL,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self._log_validation_result(result, "system_resources")
            return result
    
    async def validate_data_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate data file with extensive checks."""
        file_path = Path(file_path)
        self.logger.info(f"🔍 Validating data file: {file_path}")
        
        try:
            # Check file existence
            if not file_path.exists():
                result = ValidationResult(
                    passed=False,
                    message=f"Data file does not exist: {file_path}",
                    level=ValidationLevel.CRITICAL,
                    details={'file_path': str(file_path)}
                )
                self._log_validation_result(result, "data_file")
                return result
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024**2)
            
            if file_size_mb < self.config.min_file_size_mb:
                result = ValidationResult(
                    passed=False,
                    message=f"File too small: {file_size_mb:.2f}MB < {self.config.min_file_size_mb}MB",
                    level=ValidationLevel.ERROR,
                    details={
                        'file_size_mb': file_size_mb,
                        'min_file_size_mb': self.config.min_file_size_mb
                    }
                )
                self._log_validation_result(result, "data_file")
                return result
            
            if file_size_mb > self.config.max_file_size_mb:
                result = ValidationResult(
                    passed=False,
                    message=f"File too large: {file_size_mb:.2f}MB > {self.config.max_file_size_mb}MB",
                    level=ValidationLevel.WARNING,
                    details={
                        'file_size_mb': file_size_mb,
                        'max_file_size_mb': self.config.max_file_size_mb
                    }
                )
                self._log_validation_result(result, "data_file")
                return result
            
            # Check file format
            if not file_path.suffix.lower() in ['.parquet', '.csv', '.json']:
                result = ValidationResult(
                    passed=False,
                    message=f"Unsupported file format: {file_path.suffix}",
                    level=ValidationLevel.ERROR,
                    details={'file_extension': file_path.suffix}
                )
                self._log_validation_result(result, "data_file")
                return result
            
            # Try to read file header
            try:
                if file_path.suffix.lower() == '.parquet':
                    # Read just the schema for parquet files
                    df_sample = pd.read_parquet(file_path, nrows=0)
                elif file_path.suffix.lower() == '.csv':
                    df_sample = pd.read_csv(file_path, nrows=0)
                else:
                    # For JSON, we'll need to read a small sample
                    with open(file_path, 'r') as f:
                        sample_data = json.load(f)
                        if isinstance(sample_data, list) and len(sample_data) > 0:
                            df_sample = pd.DataFrame(sample_data[:1])
                        else:
                            df_sample = pd.DataFrame()
                
                # Check required columns
                missing_columns = set(self.config.required_columns) - set(df_sample.columns)
                if missing_columns:
                    result = ValidationResult(
                        passed=False,
                        message=f"Missing required columns: {missing_columns}",
                        level=ValidationLevel.CRITICAL,
                        details={
                            'missing_columns': list(missing_columns),
                            'available_columns': list(df_sample.columns),
                            'required_columns': self.config.required_columns
                        }
                    )
                    self._log_validation_result(result, "data_file")
                    return result
                
                # Check data types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                type_issues = []
                for col in numeric_columns:
                    if col in df_sample.columns:
                        if not pd.api.types.is_numeric_dtype(df_sample[col]):
                            type_issues.append(f"{col}: {df_sample[col].dtype}")
                
                if type_issues:
                    result = ValidationResult(
                        passed=False,
                        message=f"Invalid data types for numeric columns: {type_issues}",
                        level=ValidationLevel.ERROR,
                        details={'type_issues': type_issues}
                    )
                    self._log_validation_result(result, "data_file")
                    return result
                
            except Exception as e:
                result = ValidationResult(
                    passed=False,
                    message=f"Failed to read file header: {str(e)}",
                    level=ValidationLevel.CRITICAL,
                    details={'error': str(e), 'error_type': type(e).__name__}
                )
                self._log_validation_result(result, "data_file")
                return result
            
            # All checks passed
            result = ValidationResult(
                passed=True,
                message=f"Data file validation passed: {file_path.name}",
                level=ValidationLevel.INFO,
                details={
                    'file_path': str(file_path),
                    'file_size_mb': file_size_mb,
                    'columns': list(df_sample.columns),
                    'data_types': {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
                }
            )
            self._log_validation_result(result, "data_file")
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                message=f"Failed to validate data file: {str(e)}",
                level=ValidationLevel.CRITICAL,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self._log_validation_result(result, "data_file")
            return result
    
    async def validate_data_quality(self, data: pd.DataFrame, context: str = "") -> ValidationResult:
        """Validate data quality with extensive checks."""
        self.logger.info(f"🔍 Validating data quality for {context}")
        
        try:
            issues = []
            details = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2)
            }
            
            # Check data size
            if len(data) < self.config.min_data_rows:
                issues.append(f"Too few rows: {len(data)} < {self.config.min_data_rows}")
                details['row_count_issue'] = True
            
            if len(data) > self.config.max_data_rows:
                issues.append(f"Too many rows: {len(data)} > {self.config.max_data_rows}")
                details['row_count_issue'] = True
            
            # Check for missing values
            missing_counts = data.isnull().sum()
            high_missing_columns = missing_counts[missing_counts > len(data) * 0.1].to_dict()
            
            if high_missing_columns:
                issues.append(f"High missing values in columns: {high_missing_columns}")
                details['high_missing_columns'] = high_missing_columns
            
            # Check for duplicate rows
            duplicate_count = data.duplicated().sum()
            if duplicate_count > len(data) * 0.05:  # More than 5% duplicates
                issues.append(f"High duplicate count: {duplicate_count} ({duplicate_count/len(data)*100:.1f}%)")
                details['duplicate_issue'] = True
            
            # Check for infinite values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            infinite_counts = {}
            for col in numeric_columns:
                infinite_count = np.isinf(data[col]).sum()
                if infinite_count > 0:
                    infinite_counts[col] = infinite_count
            
            if infinite_counts:
                issues.append(f"Infinite values found: {infinite_counts}")
                details['infinite_values'] = infinite_counts
            
            # Check for extreme outliers
            outlier_issues = []
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    if outliers > len(data) * 0.01:  # More than 1% outliers
                        outlier_issues.append(f"{col}: {outliers} outliers")
            
            if outlier_issues:
                issues.append(f"High outlier count: {outlier_issues}")
                details['outlier_issues'] = outlier_issues
            
            # Check price consistency
            price_consistency_issues = []
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= max(open, close)
                high_issues = (data['high'] < np.maximum(data['open'], data['close'])).sum()
                if high_issues > 0:
                    price_consistency_issues.append(f"High < max(open,close): {high_issues} rows")
                
                # Low should be <= min(open, close)
                low_issues = (data['low'] > np.minimum(data['open'], data['close'])).sum()
                if low_issues > 0:
                    price_consistency_issues.append(f"Low > min(open,close): {low_issues} rows")
            
            if price_consistency_issues:
                issues.append(f"Price consistency issues: {price_consistency_issues}")
                details['price_consistency_issues'] = price_consistency_issues
            
            # Determine result level
            if issues:
                if any('critical' in issue.lower() or 'missing' in issue.lower() for issue in issues):
                    level = ValidationLevel.CRITICAL
                elif any('high' in issue.lower() or 'extreme' in issue.lower() for issue in issues):
                    level = ValidationLevel.ERROR
                else:
                    level = ValidationLevel.WARNING
                
                result = ValidationResult(
                    passed=False,
                    message=f"Data quality issues found: {'; '.join(issues)}",
                    level=level,
                    details=details
                )
            else:
                result = ValidationResult(
                    passed=True,
                    message="Data quality validation passed",
                    level=ValidationLevel.INFO,
                    details=details
                )
            
            self._log_validation_result(result, f"data_quality_{context}")
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                message=f"Failed to validate data quality: {str(e)}",
                level=ValidationLevel.CRITICAL,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self._log_validation_result(result, f"data_quality_{context}")
            return result
    
    async def validate_hmm_convergence(self, hmm_model: Any) -> ValidationResult:
        """Validate HMM model convergence and quality."""
        self.logger.info("🔍 Validating HMM model convergence...")
        
        try:
            issues = []
            details = {}
            
            # Check if model has convergence attribute
            if not hasattr(hmm_model, 'converged_'):
                result = ValidationResult(
                    passed=False,
                    message="HMM model does not have convergence information",
                    level=ValidationLevel.ERROR,
                    details={'model_type': type(hmm_model).__name__}
                )
                self._log_validation_result(result, "hmm_convergence")
                return result
            
            # Check convergence
            if not hmm_model.converged_:
                issues.append("Model did not converge")
                details['converged'] = False
            else:
                details['converged'] = True
            
            # Check transition matrix
            if hasattr(hmm_model, 'transmat_'):
                transmat = hmm_model.transmat_
                
                # Check for negative probabilities
                negative_probs = (transmat < 0).sum()
                if negative_probs > 0:
                    issues.append(f"Negative transition probabilities: {negative_probs} values")
                    details['negative_transitions'] = negative_probs
                
                # Check for probabilities > 1
                invalid_probs = (transmat > 1).sum()
                if invalid_probs > 0:
                    issues.append(f"Invalid transition probabilities > 1: {invalid_probs} values")
                    details['invalid_transitions'] = invalid_probs
                
                # Check for NaN values
                nan_probs = np.isnan(transmat).sum()
                if nan_probs > 0:
                    issues.append(f"NaN transition probabilities: {nan_probs} values")
                    details['nan_transitions'] = nan_probs
                
                # Check for infinite values
                inf_probs = np.isinf(transmat).sum()
                if inf_probs > 0:
                    issues.append(f"Infinite transition probabilities: {inf_probs} values")
                    details['inf_transitions'] = inf_probs
                
                # Check row sums (should be close to 1)
                row_sums = transmat.sum(axis=1)
                invalid_sums = np.abs(row_sums - 1) > 1e-6
                if invalid_sums.any():
                    issues.append(f"Invalid transition matrix row sums: {invalid_sums.sum()} rows")
                    details['invalid_row_sums'] = invalid_sums.sum()
                    details['row_sum_details'] = row_sums[invalid_sums].tolist()
                
                details['transition_matrix_shape'] = transmat.shape
                details['transition_matrix_stats'] = {
                    'min': float(transmat.min()),
                    'max': float(transmat.max()),
                    'mean': float(transmat.mean()),
                    'std': float(transmat.std())
                }
            
            # Check emission parameters
            if hasattr(hmm_model, 'means_'):
                means = hmm_model.means_
                nan_means = np.isnan(means).sum()
                if nan_means > 0:
                    issues.append(f"NaN emission means: {nan_means} values")
                    details['nan_means'] = nan_means
                
                inf_means = np.isinf(means).sum()
                if inf_means > 0:
                    issues.append(f"Infinite emission means: {inf_means} values")
                    details['inf_means'] = inf_means
            
            # Check number of iterations
            if hasattr(hmm_model, 'n_iter_'):
                details['iterations'] = hmm_model.n_iter_
                if hmm_model.n_iter_ >= 1000:
                    issues.append(f"High iteration count: {hmm_model.n_iter_}")
                    details['high_iterations'] = True
            
            # Determine result level
            if issues:
                if any('NaN' in issue or 'infinite' in issue.lower() for issue in issues):
                    level = ValidationLevel.CRITICAL
                elif any('negative' in issue.lower() or 'invalid' in issue.lower() for issue in issues):
                    level = ValidationLevel.ERROR
                else:
                    level = ValidationLevel.WARNING
                
                result = ValidationResult(
                    passed=False,
                    message=f"HMM convergence issues: {'; '.join(issues)}",
                    level=level,
                    details=details
                )
            else:
                result = ValidationResult(
                    passed=True,
                    message="HMM convergence validation passed",
                    level=ValidationLevel.INFO,
                    details=details
                )
            
            self._log_validation_result(result, "hmm_convergence")
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                message=f"Failed to validate HMM convergence: {str(e)}",
                level=ValidationLevel.CRITICAL,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self._log_validation_result(result, "hmm_convergence")
            return result
    
    async def validate_financial_metrics(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Validate financial metric calculations."""
        self.logger.info("🔍 Validating financial metrics...")
        
        try:
            issues = []
            details = {
                'total_metrics': len(metrics),
                'metric_names': list(metrics.keys())
            }
            
            for metric_name, value in metrics.items():
                metric_issues = []
                
                # Check data type
                if not isinstance(value, (int, float, np.number)):
                    metric_issues.append(f"Invalid type: {type(value)}")
                    details[f'{metric_name}_type'] = type(value).__name__
                
                # Check for NaN
                if isinstance(value, (int, float, np.number)) and np.isnan(value):
                    metric_issues.append("NaN value")
                    details[f'{metric_name}_nan'] = True
                
                # Check for infinite values
                if isinstance(value, (int, float, np.number)) and np.isinf(value):
                    metric_issues.append("Infinite value")
                    details[f'{metric_name}_infinite'] = True
                
                # Check for reasonable ranges based on metric type
                if isinstance(value, (int, float, np.number)) and not (np.isnan(value) or np.isinf(value)):
                    if 'ratio' in metric_name.lower():
                        if value < -10 or value > 10:
                            metric_issues.append(f"Extreme ratio value: {value}")
                            details[f'{metric_name}_extreme'] = True
                    
                    elif 'percent' in metric_name.lower() or 'pct' in metric_name.lower():
                        if value < -100 or value > 1000:
                            metric_issues.append(f"Unreasonable percentage: {value}")
                            details[f'{metric_name}_unreasonable'] = True
                    
                    elif 'return' in metric_name.lower():
                        if abs(value) > 1:  # More than 100% return
                            metric_issues.append(f"Extreme return: {value}")
                            details[f'{metric_name}_extreme'] = True
                    
                    elif 'volatility' in metric_name.lower():
                        if value < 0 or value > 1:
                            metric_issues.append(f"Invalid volatility: {value}")
                            details[f'{metric_name}_invalid'] = True
                
                if metric_issues:
                    issues.append(f"{metric_name}: {'; '.join(metric_issues)}")
                    details[f'{metric_name}_issues'] = metric_issues
            
            # Determine result level
            if issues:
                if any('NaN' in issue or 'infinite' in issue for issue in issues):
                    level = ValidationLevel.CRITICAL
                elif any('extreme' in issue or 'invalid' in issue for issue in issues):
                    level = ValidationLevel.ERROR
                else:
                    level = ValidationLevel.WARNING
                
                result = ValidationResult(
                    passed=False,
                    message=f"Financial metric issues: {'; '.join(issues)}",
                    level=level,
                    details=details
                )
            else:
                result = ValidationResult(
                    passed=True,
                    message="Financial metrics validation passed",
                    level=ValidationLevel.INFO,
                    details=details
                )
            
            self._log_validation_result(result, "financial_metrics")
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                message=f"Failed to validate financial metrics: {str(e)}",
                level=ValidationLevel.CRITICAL,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self._log_validation_result(result, "financial_metrics")
            return result
    
    async def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration parameters."""
        self.logger.info("🔍 Validating configuration...")
        
        try:
            issues = []
            details = {'config_keys': list(config.keys())}
            
            # Check required parameters
            required_params = ['symbol', 'exchange', 'timeframe']
            missing_params = [param for param in required_params if param not in config]
            if missing_params:
                issues.append(f"Missing required parameters: {missing_params}")
                details['missing_required'] = missing_params
            
            # Validate symbol format
            if 'symbol' in config:
                symbol = config['symbol']
                if not isinstance(symbol, str) or len(symbol) < 3:
                    issues.append(f"Invalid symbol format: {symbol}")
                    details['invalid_symbol'] = symbol
            
            # Validate exchange
            if 'exchange' in config:
                exchange = config['exchange']
                valid_exchanges = ['BINANCE', 'COINBASE', 'KRAKEN', 'BITFINEX']
                if exchange not in valid_exchanges:
                    issues.append(f"Unsupported exchange: {exchange}")
                    details['invalid_exchange'] = exchange
            
            # Validate timeframe
            if 'timeframe' in config:
                timeframe = config['timeframe']
                valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
                if timeframe not in valid_timeframes:
                    issues.append(f"Unsupported timeframe: {timeframe}")
                    details['invalid_timeframe'] = timeframe
            
            # Validate numeric parameters
            numeric_params = ['n_trials', 'timeout_minutes', 'cv_folds', 'random_state']
            for param in numeric_params:
                if param in config:
                    value = config[param]
                    if not isinstance(value, (int, float)) or value <= 0:
                        issues.append(f"Invalid {param}: {value}")
                        details[f'invalid_{param}'] = value
            
            # Determine result level
            if issues:
                if any('missing' in issue.lower() for issue in issues):
                    level = ValidationLevel.CRITICAL
                else:
                    level = ValidationLevel.ERROR
                
                result = ValidationResult(
                    passed=False,
                    message=f"Configuration issues: {'; '.join(issues)}",
                    level=level,
                    details=details
                )
            else:
                result = ValidationResult(
                    passed=True,
                    message="Configuration validation passed",
                    level=ValidationLevel.INFO,
                    details=details
                )
            
            self._log_validation_result(result, "configuration")
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                message=f"Failed to validate configuration: {str(e)}",
                level=ValidationLevel.CRITICAL,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self._log_validation_result(result, "configuration")
            return result
    
    async def comprehensive_validation(self, config: Dict[str, Any], 
                                     data_files: List[Path],
                                     data: Optional[pd.DataFrame] = None,
                                     hmm_model: Optional[Any] = None,
                                     financial_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationResult]:
        """Perform comprehensive validation of all components."""
        self.logger.info("🔍 Starting comprehensive validation...")
        
        results = {}
        
        # System resources
        results['system_resources'] = await self.validate_system_resources()
        
        # Configuration
        results['configuration'] = await self.validate_configuration(config)
        
        # Data files
        for i, file_path in enumerate(data_files):
            results[f'data_file_{i}'] = await self.validate_data_file(file_path)
        
        # Data quality
        if data is not None:
            results['data_quality'] = await self.validate_data_quality(data)
        
        # HMM model
        if hmm_model is not None:
            results['hmm_convergence'] = await self.validate_hmm_convergence(hmm_model)
        
        # Financial metrics
        if financial_metrics is not None:
            results['financial_metrics'] = await self.validate_financial_metrics(financial_metrics)
        
        # Summary
        critical_failures = [r for r in results.values() if r.level == ValidationLevel.CRITICAL and not r.passed]
        error_failures = [r for r in results.values() if r.level == ValidationLevel.ERROR and not r.passed]
        warning_failures = [r for r in results.values() if r.level == ValidationLevel.WARNING and not r.passed]
        
        if critical_failures:
            overall_result = ValidationResult(
                passed=False,
                message=f"Critical validation failures: {len(critical_failures)}",
                level=ValidationLevel.CRITICAL,
                details={
                    'critical_failures': len(critical_failures),
                    'error_failures': len(error_failures),
                    'warning_failures': len(warning_failures),
                    'total_validations': len(results)
                }
            )
        elif error_failures:
            overall_result = ValidationResult(
                passed=False,
                message=f"Error validation failures: {len(error_failures)}",
                level=ValidationLevel.ERROR,
                details={
                    'critical_failures': len(critical_failures),
                    'error_failures': len(error_failures),
                    'warning_failures': len(warning_failures),
                    'total_validations': len(results)
                }
            )
        else:
            overall_result = ValidationResult(
                passed=True,
                message="All validations passed",
                level=ValidationLevel.INFO,
                details={
                    'critical_failures': len(critical_failures),
                    'error_failures': len(error_failures),
                    'warning_failures': len(warning_failures),
                    'total_validations': len(results)
                }
            )
        
        results['overall'] = overall_result
        self._log_validation_result(overall_result, "comprehensive_validation")
        
        self.logger.info(f"✅ Comprehensive validation completed: {len(results)} validations performed")
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        return {
            'total_validations': len(self.validation_history),
            'failures': self.failure_count,
            'warnings': self.warning_count,
            'success_rate': (len(self.validation_history) - self.failure_count) / len(self.validation_history) if self.validation_history else 0,
            'recent_validations': self.validation_history[-10:] if self.validation_history else []
        }

# Global instance
_fast_fail_validator = None

def get_fast_fail_validator(config: Optional[ValidationConfig] = None) -> FastFailValidator:
    """Get or create global fast fail validator instance."""
    global _fast_fail_validator
    if _fast_fail_validator is None:
        _fast_fail_validator = FastFailValidator(config)
    return _fast_fail_validator