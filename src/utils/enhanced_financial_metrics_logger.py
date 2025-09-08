"""
Enhanced Financial Metrics Logger with Per-HMM Regime Logging and Fail-Fast Validation

This module extends the existing financial_metrics_logger to provide:
1. Enhanced per-HMM regime logging for steps after HMM-based data splitting
2. Fail-fast validation to prevent empty running or important degradation
3. Comprehensive regime-specific metrics tracking
4. Automatic regime detection and validation
"""

import logging
import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

# Import the base financial metrics logger
try:
    from src.utils.financial_metrics_logger import (
        FinancialMetricsLogger, 
        FinancialMetric, 
        TradingPerformanceMetrics,
        get_financial_metrics_logger
    )
    BASE_LOGGER_AVAILABLE = True
except ImportError:
    BASE_LOGGER_AVAILABLE = False
    FinancialMetricsLogger = None
    FinancialMetric = None
    TradingPerformanceMetrics = None
    get_financial_metrics_logger = None

# Import the main logger for fallback
try:
    from src.utils.logger import system_logger, get_logger
except ImportError:
    system_logger = None
    get_logger = lambda name: logging.getLogger(name)


@dataclass
class RegimeValidationResult:
    """Result of regime validation checks."""
    is_valid: bool
    regime_count: int
    regime_ids: List[str]
    missing_regimes: List[str]
    empty_regimes: List[str]
    validation_errors: List[str]
    quality_score: float


@dataclass
class FailFastValidationResult:
    """Result of fail-fast validation checks."""
    should_fail: bool
    failure_reason: Optional[str]
    warnings: List[str]
    critical_issues: List[str]
    degradation_detected: bool
    empty_running_detected: bool


class EnhancedFinancialMetricsLogger:
    """
    Enhanced financial metrics logger with per-HMM regime logging and fail-fast validation.
    
    Features:
    - Per-HMM regime logging for steps after HMM-based data splitting
    - Fail-fast validation to prevent empty running or important degradation
    - Automatic regime detection and validation
    - Comprehensive regime-specific metrics tracking
    - Integration with existing financial_metrics_logger
    """
    
    def __init__(self, 
                 log_dir: str = "logs/financial_metrics",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_csv: bool = True,
                 enable_json: bool = True,
                 fail_fast_enabled: bool = True,
                 regime_validation_enabled: bool = True,
                 min_regime_samples: int = 100,
                 max_regime_imbalance: float = 0.8):
        """
        Initialize the enhanced financial metrics logger.
        
        Args:
            log_dir: Directory for financial metrics logs
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_csv: Enable CSV export
            enable_json: Enable JSON export
            fail_fast_enabled: Enable fail-fast validation
            regime_validation_enabled: Enable regime validation
            min_regime_samples: Minimum samples required per regime
            max_regime_imbalance: Maximum allowed regime imbalance ratio
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_csv = enable_csv
        self.enable_json = enable_json
        self.fail_fast_enabled = fail_fast_enabled
        self.regime_validation_enabled = regime_validation_enabled
        self.min_regime_samples = min_regime_samples
        self.max_regime_imbalance = max_regime_imbalance
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize base logger if available
        if BASE_LOGGER_AVAILABLE:
            self.base_logger = get_financial_metrics_logger()
        else:
            self.base_logger = None
        
        # Initialize enhanced logger
        self._setup_enhanced_logger()
        
        # Regime tracking
        self.regime_registry = {}
        self.regime_validation_history = []
        self.fail_fast_history = []
        
        # Fallback to main logger if available
        self.fallback_logger = system_logger.getChild('EnhancedFinancialMetrics') if system_logger else None
    
    def _setup_enhanced_logger(self):
        """Setup the enhanced financial metrics logger."""
        # Enhanced financial metrics logger
        self.logger = logging.getLogger('EnhancedFinancialMetrics')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with timestamp
        if self.enable_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f'enhanced_financial_metrics_{timestamp}.log'
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def validate_regime_data(self, 
                           data: pd.DataFrame, 
                           regime_column: str = 'composite_cluster_id',
                           step_name: str = 'unknown') -> RegimeValidationResult:
        """
        Validate regime data for fail-fast behavior.
        
        Args:
            data: DataFrame containing regime data
            regime_column: Name of the regime column
            step_name: Name of the current step
            
        Returns:
            RegimeValidationResult with validation details
        """
        validation_errors = []
        warnings = []
        
        try:
            # Check if regime column exists
            if regime_column not in data.columns:
                validation_errors.append(f"Regime column '{regime_column}' not found in data")
                return RegimeValidationResult(
                    is_valid=False,
                    regime_count=0,
                    regime_ids=[],
                    missing_regimes=[],
                    empty_regimes=[],
                    validation_errors=validation_errors,
                    quality_score=0.0
                )
            
            # Get regime data
            regime_data = data[regime_column].dropna()
            
            if regime_data.empty:
                validation_errors.append("No valid regime data found")
                return RegimeValidationResult(
                    is_valid=False,
                    regime_count=0,
                    regime_ids=[],
                    missing_regimes=[],
                    empty_regimes=[],
                    validation_errors=validation_errors,
                    quality_score=0.0
                )
            
            # Get unique regimes
            unique_regimes = regime_data.unique()
            regime_ids = [str(regime) for regime in unique_regimes]
            regime_count = len(unique_regimes)
            
            # Check minimum regime count
            if regime_count < 2:
                validation_errors.append(f"Insufficient regime diversity: only {regime_count} regimes found")
            
            # Check regime sample sizes
            regime_counts = regime_data.value_counts()
            empty_regimes = []
            small_regimes = []
            
            for regime_id in regime_ids:
                count = regime_counts.get(regime_id, 0)
                if count == 0:
                    empty_regimes.append(regime_id)
                elif count < self.min_regime_samples:
                    small_regimes.append(regime_id)
                    warnings.append(f"Regime {regime_id} has only {count} samples (minimum: {self.min_regime_samples})")
            
            # Check regime imbalance
            if len(regime_counts) > 0:
                max_count = regime_counts.max()
                min_count = regime_counts.min()
                imbalance_ratio = min_count / max_count if max_count > 0 else 0
                
                if imbalance_ratio < (1 - self.max_regime_imbalance):
                    warnings.append(f"Severe regime imbalance detected: ratio {imbalance_ratio:.3f} (max allowed: {1 - self.max_regime_imbalance:.3f})")
            
            # Calculate quality score
            quality_score = 1.0
            if validation_errors:
                quality_score -= len(validation_errors) * 0.3
            if warnings:
                quality_score -= len(warnings) * 0.1
            if empty_regimes:
                quality_score -= len(empty_regimes) * 0.2
            if small_regimes:
                quality_score -= len(small_regimes) * 0.1
            
            quality_score = max(0.0, quality_score)
            
            # Determine if valid
            is_valid = len(validation_errors) == 0 and len(empty_regimes) == 0
            
            result = RegimeValidationResult(
                is_valid=is_valid,
                regime_count=regime_count,
                regime_ids=regime_ids,
                missing_regimes=[],
                empty_regimes=empty_regimes,
                validation_errors=validation_errors,
                quality_score=quality_score
            )
            
            # Store validation history
            self.regime_validation_history.append({
                'timestamp': datetime.now().isoformat(),
                'step_name': step_name,
                'result': result,
                'warnings': warnings
            })
            
            return result
            
        except Exception as e:
            validation_errors.append(f"Regime validation failed: {str(e)}")
            return RegimeValidationResult(
                is_valid=False,
                regime_count=0,
                regime_ids=[],
                missing_regimes=[],
                empty_regimes=[],
                validation_errors=validation_errors,
                quality_score=0.0
            )
    
    def validate_fail_fast_conditions(self, 
                                    data: pd.DataFrame,
                                    step_name: str,
                                    expected_regimes: Optional[List[str]] = None,
                                    min_data_quality: float = 0.7,
                                    check_empty_running: bool = True,
                                    check_degradation: bool = True) -> FailFastValidationResult:
        """
        Perform fail-fast validation to prevent empty running or important degradation.
        
        Args:
            data: DataFrame to validate
            step_name: Name of the current step
            expected_regimes: List of expected regime IDs
            min_data_quality: Minimum required data quality score
            check_empty_running: Whether to check for empty running conditions
            check_degradation: Whether to check for performance degradation
            
        Returns:
            FailFastValidationResult with validation details
        """
        warnings = []
        critical_issues = []
        should_fail = False
        failure_reason = None
        empty_running_detected = False
        degradation_detected = False
        
        try:
            # Check for empty data
            if data is None or data.empty:
                should_fail = True
                failure_reason = "Empty or None data provided"
                critical_issues.append("Data is empty or None")
                empty_running_detected = True
            
            # Check for regime data if this is a post-HMM step
            if step_name and 'step0' in step_name.lower() and int(step_name.split('step')[1][:2]) > 8:
                regime_validation = self.validate_regime_data(data, step_name=step_name)
                
                if not regime_validation.is_valid:
                    should_fail = True
                    failure_reason = f"Regime validation failed: {', '.join(regime_validation.validation_errors)}"
                    critical_issues.extend(regime_validation.validation_errors)
                
                if regime_validation.quality_score < min_data_quality:
                    should_fail = True
                    failure_reason = f"Data quality too low: {regime_validation.quality_score:.3f} < {min_data_quality}"
                    critical_issues.append(f"Data quality score {regime_validation.quality_score:.3f} below threshold {min_data_quality}")
                    degradation_detected = True
                
                # Check for expected regimes
                if expected_regimes:
                    missing_regimes = set(expected_regimes) - set(regime_validation.regime_ids)
                    if missing_regimes:
                        should_fail = True
                        failure_reason = f"Missing expected regimes: {list(missing_regimes)}"
                        critical_issues.append(f"Missing expected regimes: {list(missing_regimes)}")
            
            # Check for data quality issues
            if data is not None and not data.empty:
                # Check for excessive NaN values
                nan_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                if nan_ratio > 0.5:
                    should_fail = True
                    failure_reason = f"Excessive NaN values: {nan_ratio:.3f}"
                    critical_issues.append(f"NaN ratio {nan_ratio:.3f} exceeds threshold 0.5")
                    degradation_detected = True
                
                # Check for constant columns
                constant_columns = []
                for col in data.columns:
                    if data[col].nunique() <= 1:
                        constant_columns.append(col)
                
                if len(constant_columns) > len(data.columns) * 0.3:
                    should_fail = True
                    failure_reason = f"Too many constant columns: {len(constant_columns)}/{len(data.columns)}"
                    critical_issues.append(f"Too many constant columns: {len(constant_columns)}/{len(data.columns)}")
                    degradation_detected = True
                
                # Check for empty running conditions
                if check_empty_running:
                    # Check if all values are the same
                    if data.nunique().sum() <= len(data.columns):
                        should_fail = True
                        failure_reason = "Empty running detected: insufficient data variation"
                        critical_issues.append("Empty running: insufficient data variation")
                        empty_running_detected = True
                    
                    # Check for suspiciously small datasets
                    if len(data) < 10:
                        should_fail = True
                        failure_reason = f"Dataset too small: {len(data)} samples"
                        critical_issues.append(f"Dataset too small: {len(data)} samples")
                        empty_running_detected = True
            
            # Check for degradation patterns
            if check_degradation and len(self.fail_fast_history) > 0:
                recent_failures = [h for h in self.fail_fast_history[-5:] if h['should_fail']]
                if len(recent_failures) >= 3:
                    should_fail = True
                    failure_reason = "Performance degradation detected: multiple recent failures"
                    critical_issues.append("Performance degradation: multiple recent failures")
                    degradation_detected = True
            
            result = FailFastValidationResult(
                should_fail=should_fail,
                failure_reason=failure_reason,
                warnings=warnings,
                critical_issues=critical_issues,
                degradation_detected=degradation_detected,
                empty_running_detected=empty_running_detected
            )
            
            # Store fail-fast history
            self.fail_fast_history.append({
                'timestamp': datetime.now().isoformat(),
                'step_name': step_name,
                'result': result
            })
            
            return result
            
        except Exception as e:
            should_fail = True
            failure_reason = f"Fail-fast validation error: {str(e)}"
            critical_issues.append(f"Validation error: {str(e)}")
            
            return FailFastValidationResult(
                should_fail=should_fail,
                failure_reason=failure_reason,
                warnings=[],
                critical_issues=critical_issues,
                degradation_detected=True,
                empty_running_detected=True
            )
    
    def log_financial_metric_with_regime_validation(self,
                                                  symbol: str,
                                                  exchange: str,
                                                  timeframe: str,
                                                  metric_name: str,
                                                  metric_value: float,
                                                  metric_type: str,
                                                  step_name: str,
                                                  regime_id: Optional[str] = None,
                                                  additional_data: Optional[Dict[str, Any]] = None,
                                                  data: Optional[pd.DataFrame] = None,
                                                  expected_regimes: Optional[List[str]] = None) -> bool:
        """
        Log financial metric with regime validation and fail-fast checks.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric
            step_name: Training step name
            regime_id: Market regime identifier
            additional_data: Additional context data
            data: DataFrame for validation (optional)
            expected_regimes: Expected regime IDs (optional)
            
        Returns:
            True if logging succeeded, False if fail-fast conditions triggered
        """
        with self._lock:
            try:
                # Perform fail-fast validation if enabled
                if self.fail_fast_enabled and data is not None:
                    fail_fast_result = self.validate_fail_fast_conditions(
                        data=data,
                        step_name=step_name,
                        expected_regimes=expected_regimes
                    )
                    
                    if fail_fast_result.should_fail:
                        self.logger.error(f"🚨 FAIL-FAST TRIGGERED for {step_name}")
                        self.logger.error(f"   Reason: {fail_fast_result.failure_reason}")
                        for issue in fail_fast_result.critical_issues:
                            self.logger.error(f"   Critical Issue: {issue}")
                        
                        # Log the failure as a financial metric
                        if self.base_logger:
                            self.base_logger.log_financial_metric(
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                metric_name="fail_fast_triggered",
                                metric_value=1.0,
                                metric_type="risk",
                                step_name=step_name,
                                regime_id=regime_id,
                                additional_data={
                                    'failure_reason': fail_fast_result.failure_reason,
                                    'critical_issues': fail_fast_result.critical_issues,
                                    'degradation_detected': fail_fast_result.degradation_detected,
                                    'empty_running_detected': fail_fast_result.empty_running_detected
                                }
                            )
                        
                        return False
                
                # Log the metric using base logger
                if self.base_logger:
                    self.base_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        metric_type=metric_type,
                        step_name=step_name,
                        regime_id=regime_id,
                        additional_data=additional_data
                    )
                
                # Log regime-specific metrics if regime_id is provided
                if regime_id and self.regime_validation_enabled:
                    self._log_regime_specific_metrics(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        step_name=step_name,
                        regime_id=regime_id,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        metric_type=metric_type
                    )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to log financial metric with regime validation: {e}")
                return False
    
    def _log_regime_specific_metrics(self,
                                   symbol: str,
                                   exchange: str,
                                   timeframe: str,
                                   step_name: str,
                                   regime_id: str,
                                   metric_name: str,
                                   metric_value: float,
                                   metric_type: str) -> None:
        """Log regime-specific metrics and tracking."""
        try:
            # Track regime usage
            if regime_id not in self.regime_registry:
                self.regime_registry[regime_id] = {
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'metric_count': 0,
                    'steps_used': set()
                }
            
            self.regime_registry[regime_id]['last_seen'] = datetime.now().isoformat()
            self.regime_registry[regime_id]['metric_count'] += 1
            self.regime_registry[regime_id]['steps_used'].add(step_name)
            
            # Log regime tracking metric
            if self.base_logger:
                self.base_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name=f"regime_{regime_id}_usage_count",
                    metric_value=float(self.regime_registry[regime_id]['metric_count']),
                    metric_type="regime",
                    step_name=step_name,
                    regime_id=regime_id,
                    additional_data={
                        'regime_tracking': True,
                        'steps_used': list(self.regime_registry[regime_id]['steps_used'])
                    }
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to log regime-specific metrics: {e}")
    
    def log_per_regime_metrics(self,
                              symbol: str,
                              exchange: str,
                              timeframe: str,
                              step_name: str,
                              regime_metrics: Dict[str, Dict[str, Any]],
                              data: Optional[pd.DataFrame] = None) -> bool:
        """
        Log metrics for multiple regimes with validation.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            step_name: Training step name
            regime_metrics: Dictionary of regime_id -> metrics
            data: DataFrame for validation (optional)
            
        Returns:
            True if all logging succeeded, False if any fail-fast conditions triggered
        """
        success = True
        
        try:
            # Validate regime data if provided
            if data is not None and self.regime_validation_enabled:
                regime_validation = self.validate_regime_data(data, step_name=step_name)
                
                if not regime_validation.is_valid:
                    self.logger.error(f"🚨 Regime validation failed for {step_name}")
                    for error in regime_validation.validation_errors:
                        self.logger.error(f"   Error: {error}")
                    
                    if self.fail_fast_enabled:
                        return False
            
            # Log metrics for each regime
            for regime_id, metrics in regime_metrics.items():
                for metric_name, metric_value in metrics.items():
                    metric_success = self.log_financial_metric_with_regime_validation(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name=f"regime_{regime_id}_{metric_name}",
                        metric_value=float(metric_value),
                        metric_type="regime",
                        step_name=step_name,
                        regime_id=str(regime_id),
                        data=data
                    )
                    
                    if not metric_success:
                        success = False
            
            # Log regime summary metrics
            if self.base_logger:
                self.base_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="total_regimes_processed",
                    metric_value=float(len(regime_metrics)),
                    metric_type="regime",
                    step_name=step_name
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to log per-regime metrics: {e}")
            return False
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of regime usage and validation history."""
        try:
            return {
                'regime_registry': self.regime_registry,
                'validation_history': self.regime_validation_history[-10:],  # Last 10 validations
                'fail_fast_history': self.fail_fast_history[-10:],  # Last 10 fail-fast checks
                'total_regimes_tracked': len(self.regime_registry),
                'total_validations': len(self.regime_validation_history),
                'total_fail_fast_checks': len(self.fail_fast_history)
            }
        except Exception as e:
            self.logger.error(f"Failed to get regime summary: {e}")
            return {}
    
    def close(self) -> None:
        """Close the enhanced financial metrics logger and clean up resources."""
        with self._lock:
            try:
                # Close base logger if available
                if self.base_logger and hasattr(self.base_logger, 'close'):
                    self.base_logger.close()
                
                # Clear handlers
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
                
                self.logger.info("🔒 Enhanced financial metrics logger closed successfully")
                
            except Exception as e:
                if self.fallback_logger:
                    self.fallback_logger.error(f"Error closing enhanced financial metrics logger: {e}")


# Global instance
_enhanced_financial_metrics_logger: Optional[EnhancedFinancialMetricsLogger] = None


def get_enhanced_financial_metrics_logger() -> EnhancedFinancialMetricsLogger:
    """Get the global enhanced financial metrics logger instance."""
    global _enhanced_financial_metrics_logger
    if _enhanced_financial_metrics_logger is None:
        _enhanced_financial_metrics_logger = EnhancedFinancialMetricsLogger()
    return _enhanced_financial_metrics_logger


def setup_enhanced_financial_metrics_logging(log_dir: str = "logs/financial_metrics", **kwargs) -> EnhancedFinancialMetricsLogger:
    """Setup the global enhanced financial metrics logger."""
    global _enhanced_financial_metrics_logger
    _enhanced_financial_metrics_logger = EnhancedFinancialMetricsLogger(log_dir=log_dir, **kwargs)
    return _enhanced_financial_metrics_logger


@contextmanager
def enhanced_financial_metrics_context(step_name: str, symbol: str, exchange: str, timeframe: str, 
                                     data: Optional[pd.DataFrame] = None, expected_regimes: Optional[List[str]] = None):
    """Context manager for enhanced financial metrics logging within a training step."""
    logger = get_enhanced_financial_metrics_logger()
    
    try:
        # Perform initial validation
        if data is not None and logger.fail_fast_enabled:
            fail_fast_result = logger.validate_fail_fast_conditions(
                data=data,
                step_name=step_name,
                expected_regimes=expected_regimes
            )
            
            if fail_fast_result.should_fail:
                logger.logger.error(f"🚨 FAIL-FAST TRIGGERED at step start for {step_name}")
                logger.logger.error(f"   Reason: {fail_fast_result.failure_reason}")
                raise RuntimeError(f"Fail-fast validation failed: {fail_fast_result.failure_reason}")
        
        # Log step start
        if logger.base_logger:
            logger.base_logger.log_step_start(step_name, symbol, exchange, timeframe)
        
        yield logger
        
        # Log step end
        if logger.base_logger:
            logger.base_logger.log_step_end(step_name, symbol, exchange, timeframe, success=True)
            
    except Exception as e:
        # Log step end with error
        if logger.base_logger:
            logger.base_logger.log_step_end(step_name, symbol, exchange, timeframe, success=False, error_message=str(e))
        raise


# Convenience functions for enhanced operations
def log_regime_metric_with_validation(symbol: str, exchange: str, timeframe: str, step_name: str, 
                                    regime_id: str, metric_name: str, metric_value: float, 
                                    metric_type: str = "regime", data: Optional[pd.DataFrame] = None) -> bool:
    """Log a regime-specific metric with validation."""
    logger = get_enhanced_financial_metrics_logger()
    return logger.log_financial_metric_with_regime_validation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_type=metric_type,
        step_name=step_name,
        regime_id=regime_id,
        data=data
    )


def validate_and_log_regime_data(symbol: str, exchange: str, timeframe: str, step_name: str,
                               data: pd.DataFrame, regime_column: str = 'composite_cluster_id') -> bool:
    """Validate regime data and log validation results."""
    logger = get_enhanced_financial_metrics_logger()
    
    # Perform validation
    validation_result = logger.validate_regime_data(data, regime_column, step_name)
    
    # Log validation results
    logger.log_financial_metric_with_regime_validation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="regime_validation_quality_score",
        metric_value=validation_result.quality_score,
        metric_type="quality",
        step_name=step_name,
        data=data
    )
    
    logger.log_financial_metric_with_regime_validation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="regime_count",
        metric_value=float(validation_result.regime_count),
        metric_type="regime",
        step_name=step_name,
        data=data
    )
    
    return validation_result.is_valid


# Export main classes and functions
__all__ = [
    'EnhancedFinancialMetricsLogger',
    'RegimeValidationResult',
    'FailFastValidationResult',
    'get_enhanced_financial_metrics_logger',
    'setup_enhanced_financial_metrics_logging',
    'enhanced_financial_metrics_context',
    'log_regime_metric_with_validation',
    'validate_and_log_regime_data'
]