# src/utils/error_handler.py
import sys
import logging
import json
import traceback
from datetime import datetime
import os
import asyncio # Added import for asyncio
from typing import List, Dict, Any, Union

from src.utils.logger import system_logger
# Import both managers for type hinting, but use the one passed via db_manager_instance
# Removed FirestoreManager import as we're using SQLite only
from src.database.sqlite_manager import SQLiteManager
from src.config import CONFIG # Import CONFIG for error log file path

class AresExceptionHandler:
    """
    A centralized exception handler for the Ares trading bot.
    It logs unhandled exceptions to the system logger, a dedicated JSON file,
    and optionally to a database (Firestore or SQLite).
    """
    def __init__(self):
        self.logger = system_logger.getChild('ExceptionHandler') # Fixed: Use imported logger
        self.error_log_file = os.path.join("logs", CONFIG.get("ERROR_LOG_FILE", "ares_errors.jsonl")) # Using .jsonl for JSON Lines format
        os.makedirs(os.path.dirname(self.error_log_file), exist_ok=True)
        self._original_excepthook = sys.excepthook
        # db_manager is now passed from main.py
        self.db_manager: Union[FirestoreManager, SQLiteManager, None] = None # Will be set by set_db_manager

    def set_db_manager(self, db_manager: Union[FirestoreManager, SQLiteManager, None]):
        """Sets the database manager instance for logging exceptions to DB."""
        self.db_manager = db_manager
        self.logger.info("Database manager set for exception handler.")

    def _log_exception(self, exc_type: Any, exc_value: Any, exc_traceback: Any): # Fixed: Type hints
        """
        Internal method to log the exception details.
        """
        timestamp = datetime.utcnow().isoformat()
        traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        error_record = {
            "timestamp": timestamp,
            "level": "CRITICAL",
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": traceback_str,
            "context": "Unhandled Exception" # Default context, can be expanded
        }

        # Log to system logger (which goes to console and main log file)
        self.logger.critical(f"Unhandled exception caught: {exc_type.__name__}: {exc_value}", exc_info=True)

        # Log to dedicated JSON Lines file
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(json.dumps(error_record) + "\n")
            self.logger.info(f"Unhandled exception logged to {self.error_log_file}")
        except Exception as e:
            self.logger.error(f"Failed to write exception to dedicated log file: {e}", exc_info=True)

        # Optionally log to DB
        if self.db_manager: # Fixed: Use self.db_manager
            try:
                # DBs have document/record size limits, so truncate traceback if very long
                db_record = error_record.copy()
                if len(db_record["traceback"]) > 50000: # Arbitrary limit, adjust as needed
                    db_record["traceback"] = db_record["traceback"][:50000] + "\n... (truncated)"
                
                # Run DB operation in a non-blocking way (e.g., in a separate task)
                # Use add_document for error logs
                asyncio.create_task(self.db_manager.add_document("ares_unhandled_exceptions", db_record, is_public=False))
                self.logger.info("Unhandled exception sent to DB.")
            except Exception as e:
                self.logger.error(f"Failed to send exception to DB: {e}", exc_info=True)

        # Call the original excepthook to maintain default Python behavior (e.g., for IDEs)
        self._original_excepthook(exc_type, exc_value, exc_traceback)

    def register(self):
        """Registers this handler as the default sys.excepthook."""
        sys.excepthook = self._log_exception
        self.logger.info("Ares custom exception handler registered.")

    def unregister(self):
        """Unregisters the custom handler and restores the original."""
        sys.excepthook = self._original_excepthook
        self.logger.info("Ares custom exception handler unregistered.")

# Global instance of the exception handler
ares_exception_handler = AresExceptionHandler()

def register_global_exception_handler(db_manager: Union[FirestoreManager, SQLiteManager, None] = None): # Fixed: Accept db_manager
    """Convenience function to register the global exception handler."""
    ares_exception_handler.set_db_manager(db_manager) # Set the db_manager
    ares_exception_handler.register()

def unregister_global_exception_handler():
    """Convenience function to unregister the global exception handler."""
    ares_exception_handler.unregister()

def get_logged_exceptions(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Reads recent exceptions from the dedicated error log file.
    """
    exceptions = []
    try:
        with open(ares_exception_handler.error_log_file, 'r') as f:
            # Read lines in reverse to get most recent first, then reverse again
            for line in reversed(f.readlines()):
                try:
                    exceptions.append(json.loads(line))
                    if len(exceptions) >= limit:
                        break
                except json.JSONDecodeError:
                    ares_exception_handler.system_logger.warning(f"Skipping malformed line in error log: {line.strip()}")
        return list(reversed(exceptions)) # Return in chronological order
    except FileNotFoundError:
        ares_exception_handler.system_logger.info("Error log file not found.")
        return []
    except Exception as e:
        ares_exception_handler.system_logger.error(f"Error reading error log file: {e}", exc_info=True)
        return []


# Enhanced error handling utilities for comprehensive error management
import functools
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import Optional, Callable, Any, Type, Tuple

class ErrorRecoveryStrategies:
    """Common error recovery strategies for different types of operations."""
    
    @staticmethod
    def safe_dict_access(data: dict, keys: Union[str, List[str]], default: Any = None) -> Any:
        """Safely access nested dictionary keys with fallback."""
        if isinstance(keys, str):
            keys = [keys]
        
        current = data
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def safe_dataframe_access(df: pd.DataFrame, column: str, default: Any = None) -> Any:
        """Safely access DataFrame columns with fallback."""
        try:
            if column in df.columns:
                return df[column]
            return default
        except (KeyError, AttributeError):
            return default
    
    @staticmethod
    def safe_numeric_operation(operation: Callable, *args, default: float = 0.0, **kwargs) -> float:
        """Safely perform numeric operations with NaN/inf handling."""
        try:
            result = operation(*args, **kwargs)
            if np.isnan(result) or np.isinf(result):
                return default
            return result
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            return default
    
    @staticmethod
    def safe_type_conversion(value: Any, target_type: Type, default: Any = None) -> Any:
        """Safely convert types with fallback."""
        try:
            return target_type(value)
        except (ValueError, TypeError, OverflowError):
            return default


def handle_errors(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    default_return: Any = None,
    log_errors: bool = True,
    recovery_strategy: Optional[Callable] = None,
    context: str = ""
):
    """
    Decorator for comprehensive error handling with recovery strategies.
    
    Args:
        exceptions: Tuple of exception types to catch
        default_return: Default value to return on error
        log_errors: Whether to log errors
        recovery_strategy: Optional recovery function to call on error
        context: Additional context for logging
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    system_logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'context': context,
                            'args': str(args)[:200],  # Truncate for logging
                            'kwargs': str(kwargs)[:200],
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )
                
                if recovery_strategy:
                    try:
                        return recovery_strategy(*args, **kwargs)
                    except Exception as recovery_error:
                        if log_errors:
                            system_logger.error(f"Recovery strategy failed: {recovery_error}", exc_info=True)
                
                return default_return
        return wrapper
    return decorator


def handle_data_processing_errors(
    default_return: Any = None,
    log_errors: bool = True,
    context: str = ""
):
    """
    Decorator specifically for data processing operations that might encounter
    NaN, inf, or division by zero errors.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Check for NaN or inf values in numpy arrays or pandas objects
                if isinstance(result, (np.ndarray, pd.Series, pd.DataFrame)):
                    if isinstance(result, np.ndarray):
                        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                            system_logger.warning(f"NaN or inf values detected in {func.__name__} result")
                            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                    elif isinstance(result, (pd.Series, pd.DataFrame)):
                        if result.isnull().any().any() if isinstance(result, pd.DataFrame) else result.isnull().any():
                            system_logger.warning(f"NaN values detected in {func.__name__} result")
                            result = result.fillna(0)
                
                return result
            except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                if log_errors:
                    system_logger.error(
                        f"Data processing error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'context': context,
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger.error(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'context': context,
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )
                return default_return
        return wrapper
    return decorator


def handle_file_operations(
    default_return: Any = None,
    log_errors: bool = True,
    context: str = ""
):
    """
    Decorator for file operations with comprehensive error handling.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                if log_errors:
                    system_logger.error(f"File not found in {func.__name__}: {str(e)}", exc_info=True)
                return default_return
            except PermissionError as e:
                if log_errors:
                    system_logger.error(f"Permission denied in {func.__name__}: {str(e)}", exc_info=True)
                return default_return
            except OSError as e:
                if log_errors:
                    system_logger.error(f"OS error in {func.__name__}: {str(e)}", exc_info=True)
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
                return default_return
        return wrapper
    return decorator


def handle_network_operations(
    max_retries: int = 3,
    default_return: Any = None,
    log_errors: bool = True,
    context: str = ""
):
    """
    Decorator for network operations with retry logic.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import aiohttp
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        system_logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}) in {func.__name__}: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        if log_errors:
                            system_logger.error(f"Network operation failed after {max_retries} attempts in {func.__name__}: {e}", exc_info=True)
                        return default_return
                except Exception as e:
                    if log_errors:
                        system_logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    return default_return
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            import requests
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        system_logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}) in {func.__name__}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        if log_errors:
                            system_logger.error(f"Network operation failed after {max_retries} attempts in {func.__name__}: {e}", exc_info=True)
                        return default_return
                except Exception as e:
                    if log_errors:
                        system_logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    return default_return
            
        # Return async wrapper if function is async, sync wrapper otherwise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


def handle_type_conversions(
    default_return: Any = None,
    log_errors: bool = True,
    context: str = ""
):
    """
    Decorator for type conversion operations that might fail.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError, OverflowError) as e:
                if log_errors:
                    system_logger.error(
                        f"Type conversion error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'context': context,
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger.error(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'context': context,
                            'error_type': type(e).__name__
                        },
                        exc_info=True
                    )
                return default_return
        return wrapper
    return decorator


@contextmanager
def error_context(
    operation_name: str,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    default_return: Any = None,
    log_errors: bool = True
):
    """
    Context manager for error handling in code blocks.
    
    Usage:
        with error_context("database_operation", default_return={}):
            result = some_database_operation()
    """
    try:
        yield
    except exceptions as e:
        if log_errors:
            system_logger.error(
                f"Error in {operation_name}: {str(e)}",
                extra={
                    'operation': operation_name,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
        return default_return


def safe_file_operation(operation: Callable, *args, **kwargs) -> Any:
    """Wrapper for safe file operations with comprehensive error handling."""
    try:
        return operation(*args, **kwargs)
    except FileNotFoundError as e:
        system_logger.error(f"File not found: {e}", exc_info=True)
        return None
    except PermissionError as e:
        system_logger.error(f"Permission denied: {e}", exc_info=True)
        return None
    except OSError as e:
        system_logger.error(f"OS error during file operation: {e}", exc_info=True)
        return None
    except Exception as e:
        system_logger.error(f"Unexpected error in file operation: {e}", exc_info=True)
        return None


async def safe_network_operation(operation: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
    """Wrapper for safe network operations with retry logic."""
    import aiohttp
    
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                return operation(*args, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                system_logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                system_logger.error(f"Network operation failed after {max_retries} attempts: {e}", exc_info=True)
                return None
        except Exception as e:
            system_logger.error(f"Unexpected error in network operation: {e}", exc_info=True)
            return None


def safe_database_operation(operation: Callable, *args, **kwargs) -> Any:
    """Wrapper for safe database operations."""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        system_logger.error(f"Database operation failed: {e}", exc_info=True)
        return None


def safe_dataframe_operation(operation: Callable, *args, **kwargs) -> Any:
    """Wrapper for safe DataFrame operations with NaN handling."""
    try:
        result = operation(*args, **kwargs)
        
        # Handle NaN values in result
        if isinstance(result, pd.DataFrame):
            result = result.fillna(0)
        elif isinstance(result, pd.Series):
            result = result.fillna(0)
        elif isinstance(result, np.ndarray):
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
        return result
    except Exception as e:
        system_logger.error(f"DataFrame operation failed: {e}", exc_info=True)
        return None


def safe_numeric_operation(operation: Callable, *args, **kwargs) -> Any:
    """Wrapper for safe numeric operations with division by zero and overflow handling."""
    try:
        result = operation(*args, **kwargs)
        
        # Handle special numeric values
        if isinstance(result, (int, float)):
            if np.isnan(result) or np.isinf(result):
                return 0.0
        elif isinstance(result, np.ndarray):
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
        return result
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as e:
        system_logger.error(f"Numeric operation failed: {e}", exc_info=True)
        return 0.0
    except Exception as e:
        system_logger.error(f"Unexpected error in numeric operation: {e}", exc_info=True)
        return 0.0
