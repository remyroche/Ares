#!/usr/bin/env python3
"""
Common Utilities for Backtesting Pipeline

This module provides common utilities for data operations, error handling,
and pipeline management in the backtesting system.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import concurrent.futures
from functools import wraps

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
    safe_sleep,
    safe_gather,
)
from src.utils.compat import handle_errors


class OperationStatus(str, Enum):
    """Status of pipeline operations."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class OperationResult:
    """Result of a pipeline operation."""
    operation_id: str
    status: OperationStatus
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class DataOperationUtilities:
    """Utilities for data operations in the backtesting pipeline."""
    
    @staticmethod
    @handle_errors(exceptions=(Exception,), default_return=None)
    def load_price_data(
        file_path: Union[str, Path],
        symbol: str,
        exchange: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load and validate price data from file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logging.error(f"Price data file not found: {file_path}")
                return None
            
            # Load data based on file extension
            if file_path.suffix.lower() == '.parquet':
                data = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            else:
                logging.error(f"Unsupported file format: {file_path.suffix}")
                return None
            
            # Ensure timestamp column is datetime
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            
            # Filter by date range if specified
            if start_date and "timestamp" in data.columns:
                start_dt = pd.to_datetime(start_date)
                data = data[data["timestamp"] >= start_dt]
            
            if end_date and "timestamp" in data.columns:
                end_dt = pd.to_datetime(end_date)
                data = data[data["timestamp"] <= end_dt]
            
            # Sort by timestamp
            if "timestamp" in data.columns:
                data = data.sort_values("timestamp").reset_index(drop=True)
            
            logging.info(f"Loaded {len(data)} rows of price data for {symbol} on {exchange}")
            return data
            
        except Exception as e:
            logging.exception(f"Error loading price data: {e}")
            return None
    
    @staticmethod
    @handle_errors(exceptions=(Exception,), default_return=None)
    def save_processed_data(
        data: pd.DataFrame,
        file_path: Union[str, Path],
        format: str = "parquet"
    ) -> bool:
        """Save processed data to file."""
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            if format.lower() == "parquet":
                data.to_parquet(file_path, index=False)
            elif format.lower() == "csv":
                data.to_csv(file_path, index=False)
            else:
                logging.error(f"Unsupported save format: {format}")
                return False
            
            logging.info(f"Saved {len(data)} rows to {file_path}")
            return True
            
        except Exception as e:
            logging.exception(f"Error saving processed data: {e}")
            return False
    
    @staticmethod
    @handle_errors(exceptions=(Exception,), default_return=None)
    def validate_data_continuity(
        data: pd.DataFrame,
        expected_interval: str = "1min",
        max_gap_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """Validate data continuity and identify gaps."""
        try:
            if "timestamp" not in data.columns or len(data) < 2:
                return {"valid": False, "error": "Insufficient timestamp data"}
            
            # Calculate expected interval
            expected_delta = pd.Timedelta(expected_interval)
            
            # Calculate actual intervals
            time_diffs = data["timestamp"].diff().dropna()
            
            # Identify gaps
            large_gaps = time_diffs > expected_delta * 2
            gap_count = large_gaps.sum()
            gap_ratio = gap_count / len(time_diffs)
            
            # Calculate statistics
            stats = {
                "valid": gap_ratio <= max_gap_ratio,
                "total_rows": len(data),
                "expected_interval": expected_interval,
                "gap_count": int(gap_count),
                "gap_ratio": float(gap_ratio),
                "max_gap_ratio": max_gap_ratio,
                "date_range": {
                    "start": data["timestamp"].min().isoformat(),
                    "end": data["timestamp"].max().isoformat()
                },
                "interval_stats": {
                    "mean": str(time_diffs.mean()),
                    "median": str(time_diffs.median()),
                    "std": str(time_diffs.std())
                }
            }
            
            if gap_count > 0:
                gap_locations = data.index[large_gaps].tolist()
                stats["gap_locations"] = gap_locations[:10]  # First 10 gaps
            
            return stats
            
        except Exception as e:
            logging.exception(f"Error validating data continuity: {e}")
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    @handle_errors(exceptions=(Exception,), default_return=None)
    def resample_data(
        data: pd.DataFrame,
        target_frequency: str = "1H",
        method: str = "ohlc"
    ) -> Optional[pd.DataFrame]:
        """Resample data to target frequency."""
        try:
            if "timestamp" not in data.columns:
                logging.error("Timestamp column not found for resampling")
                return None
            
            # Set timestamp as index
            data_indexed = data.set_index("timestamp")
            
            # Resample based on method
            if method.lower() == "ohlc":
                # OHLC resampling for price data
                if all(col in data.columns for col in ["open", "high", "low", "close"]):
                    resampled = data_indexed[["open", "high", "low", "close", "volume"]].resample(target_frequency).agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum"
                    })
                else:
                    logging.error("OHLC columns not found for resampling")
                    return None
            else:
                # Simple resampling
                resampled = data_indexed.resample(target_frequency).mean()
            
            # Reset index
            resampled = resampled.reset_index()
            
            # Remove rows with all NaN values
            resampled = resampled.dropna()
            
            logging.info(f"Resampled data from {len(data)} to {len(resampled)} rows at {target_frequency}")
            return resampled
            
        except Exception as e:
            logging.exception(f"Error resampling data: {e}")
            return None


class ErrorHandlingUtilities:
    """Utilities for error handling and recovery in the backtesting pipeline."""
    
    @staticmethod
    def create_error_context(
        operation_name: str,
        symbol: str,
        exchange: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized error context."""
        return {
            "operation": operation_name,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": format_datetime(get_current_datetime()),
            "additional_info": additional_info or {}
        }
    
    @staticmethod
    def log_error_with_context(
        error: Exception,
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Log error with standardized context."""
        if logger is None:
            logger = logging.getLogger(__name__)
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        logger.error(f"Error in {context.get('operation', 'unknown')}: {error_info}")
    
    @staticmethod
    @contextmanager
    def error_recovery_context(
        operation_name: str,
        symbol: str,
        exchange: str,
        fallback_value: Any = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Context manager for error recovery with retries."""
        context = ErrorHandlingUtilities.create_error_context(
            operation_name, symbol, exchange
        )
        
        for attempt in range(max_retries + 1):
            try:
                yield context
                return
            except Exception as e:
                context["attempt"] = attempt + 1
                context["max_retries"] = max_retries
                
                ErrorHandlingUtilities.log_error_with_context(e, context)
                
                if attempt < max_retries:
                    logging.info(f"Retrying {operation_name} (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logging.error(f"All retry attempts failed for {operation_name}")
                    if fallback_value is not None:
                        logging.info(f"Using fallback value for {operation_name}")
                        yield context
                        return
                    raise e
    
    @staticmethod
    def safe_execute_with_fallback(
        func: Callable,
        fallback_func: Optional[Callable] = None,
        fallback_value: Any = None,
        *args,
        **kwargs
    ) -> Any:
        """Safely execute function with fallback options."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Error executing {func.__name__}: {e}")
            
            if fallback_func is not None:
                try:
                    logging.info(f"Attempting fallback function: {fallback_func.__name__}")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logging.exception(f"Fallback function also failed: {fallback_error}")
            
            if fallback_value is not None:
                logging.info(f"Using fallback value for {func.__name__}")
                return fallback_value
            
            raise e


class PipelineManagementUtilities:
    """Utilities for managing the backtesting pipeline execution."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.operation_queue = queue.Queue()
        self.results = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def execute_operation(
        self,
        operation_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute a single operation with tracking."""
        start_time = format_datetime(get_current_datetime())
        
        try:
            self.logger.info(f"Starting operation: {operation_id}")
            
            # Execute the function
            result_data = func(*args, **kwargs)
            
            end_time = format_datetime(get_current_datetime())
            duration = (get_current_datetime() - pd.to_datetime(start_time)).total_seconds()
            
            result = OperationResult(
                operation_id=operation_id,
                status=OperationStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                result_data=result_data
            )
            
            self.logger.info(f"Completed operation: {operation_id} in {duration:.2f}s")
            return result
            
        except Exception as e:
            end_time = format_datetime(get_current_datetime())
            duration = (get_current_datetime() - pd.to_datetime(start_time)).total_seconds()
            
            result = OperationResult(
                operation_id=operation_id,
                status=OperationStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
            
            self.logger.error(f"Failed operation: {operation_id} - {e}")
            return result
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def execute_operations_parallel(
        self,
        operations: List[Tuple[str, Callable, tuple, dict]]
    ) -> Dict[str, OperationResult]:
        """Execute multiple operations in parallel."""
        self.logger.info(f"Executing {len(operations)} operations in parallel")
        
        # Submit all operations
        future_to_operation = {}
        for operation_id, func, args, kwargs in operations:
            future = self.executor.submit(
                self.execute_operation,
                operation_id,
                func,
                *args,
                **kwargs
            )
            future_to_operation[future] = operation_id
        
        # Collect results
        results = {}
        for future in concurrent.futures.as_completed(future_to_operation):
            operation_id = future_to_operation[future]
            try:
                result = future.result()
                results[operation_id] = result
            except Exception as e:
                self.logger.exception(f"Error in parallel execution for {operation_id}: {e}")
                results[operation_id] = OperationResult(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    start_time=format_datetime(get_current_datetime()),
                    error_message=str(e)
                )
        
        return results
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def execute_operations_sequential(
        self,
        operations: List[Tuple[str, Callable, tuple, dict]],
        stop_on_error: bool = True
    ) -> Dict[str, OperationResult]:
        """Execute multiple operations sequentially."""
        self.logger.info(f"Executing {len(operations)} operations sequentially")
        
        results = {}
        for operation_id, func, args, kwargs in operations:
            result = self.execute_operation(operation_id, func, *args, **kwargs)
            results[operation_id] = result
            
            if stop_on_error and result.status == OperationStatus.FAILED:
                self.logger.error(f"Stopping sequential execution due to failure in {operation_id}")
                break
        
        return results
    
    def get_execution_summary(self, results: Dict[str, OperationResult]) -> Dict[str, Any]:
        """Get summary of execution results."""
        total_operations = len(results)
        completed = sum(1 for r in results.values() if r.status == OperationStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == OperationStatus.FAILED)
        
        total_duration = sum(
            r.duration_seconds for r in results.values() 
            if r.duration_seconds is not None
        )
        
        return {
            "total_operations": total_operations,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_operations if total_operations > 0 else 0.0,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / total_operations if total_operations > 0 else 0.0
        }
    
    def save_execution_report(
        self,
        results: Dict[str, OperationResult],
        output_path: Union[str, Path]
    ) -> bool:
        """Save execution report to file."""
        try:
            output_path = Path(output_path)
            ensure_directory(output_path.parent)
            
            report = {
                "execution_summary": self.get_execution_summary(results),
                "operation_results": {k: v.to_dict() for k, v in results.items()},
                "timestamp": format_datetime(get_current_datetime())
            }
            
            safe_json_dump(report, output_path, indent=2)
            self.logger.info(f"Execution report saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error saving execution report: {e}")
            return False
    
    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class ConfigurationUtilities:
    """Utilities for configuration management in the backtesting pipeline."""
    
    @staticmethod
    @handle_errors(exceptions=(Exception,), default_return=None)
    def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logging.error(f"Configuration file not found: {config_path}")
                return None
            
            if config_path.suffix.lower() == '.json':
                return safe_json_load(config_path)
            else:
                logging.error(f"Unsupported configuration format: {config_path.suffix}")
                return None
                
        except Exception as e:
            logging.exception(f"Error loading configuration: {e}")
            return None
    
    @staticmethod
    @handle_errors(exceptions=(Exception,), default_return=False)
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
        """Save configuration to file."""
        try:
            config_path = Path(config_path)
            ensure_directory(config_path.parent)
            
            if config_path.suffix.lower() == '.json':
                safe_json_dump(config, config_path, indent=2)
            else:
                logging.error(f"Unsupported configuration format: {config_path.suffix}")
                return False
            
            logging.info(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            logging.exception(f"Error saving configuration: {e}")
            return False
    
    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge configuration dictionaries with override precedence."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigurationUtilities.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(
        config: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate configuration structure."""
        errors = []
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required configuration key: {key}")
        
        # Check optional keys (if provided)
        if optional_keys:
            for key in optional_keys:
                if key not in config:
                    logging.warning(f"Optional configuration key not found: {key}")
        
        return len(errors) == 0, errors


class LoggingUtilities:
    """Utilities for logging in the backtesting pipeline."""
    
    @staticmethod
    def setup_pipeline_logging(
        log_dir: Union[str, Path] = "logs/backtesting",
        log_level: str = "INFO",
        log_format: Optional[str] = None
    ) -> logging.Logger:
        """Setup logging for the backtesting pipeline."""
        log_dir = Path(log_dir)
        ensure_directory(log_dir)
        
        # Create logger
        logger = logging.getLogger("backtesting_pipeline")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        
        # File handler
        log_file = log_dir / f"backtesting_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"Backtesting pipeline logging initialized - Log file: {log_file}")
        return logger
    
    @staticmethod
    def log_operation_start(
        logger: logging.Logger,
        operation_name: str,
        symbol: str,
        exchange: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the start of an operation."""
        info = {
            "operation": operation_name,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": format_datetime(get_current_datetime())
        }
        if additional_info:
            info.update(additional_info)
        
        logger.info(f"Starting {operation_name} for {symbol} on {exchange} - {info}")
    
    @staticmethod
    def log_operation_end(
        logger: logging.Logger,
        operation_name: str,
        symbol: str,
        exchange: str,
        success: bool,
        duration_seconds: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the end of an operation."""
        status = "SUCCESS" if success else "FAILED"
        info = {
            "operation": operation_name,
            "symbol": symbol,
            "exchange": exchange,
            "status": status,
            "timestamp": format_datetime(get_current_datetime())
        }
        if duration_seconds is not None:
            info["duration_seconds"] = duration_seconds
        if additional_info:
            info.update(additional_info)
        
        if success:
            logger.info(f"Completed {operation_name} for {symbol} on {exchange} - {info}")
        else:
            logger.error(f"Failed {operation_name} for {symbol} on {exchange} - {info}")