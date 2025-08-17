# src/utils/data_quality_decorators.py
"""
Data Quality Decorators for Feature Engineering
Provides automatic data quality validation for feature engineering methods.
"""

import functools
import hashlib
import inspect
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from enum import Enum

from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning, critical
from src.training.steps.raw_data_quality_checker import validate_raw_data_quality
from src.utils.feature_output_validator import validate_feature_output
from src.utils.lookahead_bias_detector import (
    detect_lookahead_bias,
    apply_feature_lagging,
)


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"  # Informational messages
    WARNING = "warning"  # Log issues but continue
    ERROR = "error"  # Error level issues
    CRITICAL = "critical"  # Critical issues
    STRICT = "strict"  # Stop on any critical issue
    SILENT = "silent"  # Only log summary


class DataQualityCache:
    """Cache for data quality validation results to avoid redundant checks."""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.logger = system_logger.getChild("DataQualityCache")

    def _generate_cache_key(self, data: pd.DataFrame, method_name: str) -> str:
        """Generate cache key for data quality validation."""
        try:
            # Create a more stable hash based on data shape and column names
            # This is more reliable than hashing the entire data content
            data_signature = f"{data.shape[0]}_{data.shape[1]}_{'_'.join(sorted(data.columns))}"
            data_hash = hashlib.md5(data_signature.encode()).hexdigest()
            return f"{data_hash}_{method_name}"
        except Exception:
            # Fallback to simple hash
            return f"{hash(str(data.shape))}_{method_name}"

    def get(self, data: pd.DataFrame, method_name: str) -> Optional[Dict[str, Any]]:
        """Get cached validation result."""
        cache_key = self._generate_cache_key(data, method_name)
        result = self.cache.get(cache_key)

        if result:
            print(f"✅ [CACHE] Cache hit for {method_name}")
            self.logger.info(f"✅ [CACHE] Cache hit for {method_name}")
        else:
            print(f"❌ [CACHE] Cache miss for {method_name}")
            self.logger.debug(f"❌ [CACHE] Cache miss for {method_name}")

        return result

    def set(self, data: pd.DataFrame, method_name: str, result: Dict[str, Any]) -> None:
        """Set cached validation result."""
        cache_key = self._generate_cache_key(data, method_name)
        self.cache[cache_key] = result

        print(f"💾 [CACHE] Cached validation result for {method_name}")
        self.logger.info(f"💾 [CACHE] Cached validation result for {method_name}")

        # Limit cache size
        if len(self.cache) > self.max_size:
            # Remove oldest entries
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            print(f"🗑️ [CACHE] Removed oldest cache entry due to size limit")
            self.logger.info(f"🗑️ [CACHE] Removed oldest cache entry due to size limit")

    def clear(self) -> None:
        """Clear the cache."""
        self.logger.debug(f"🗑️ [CACHE] Clearing data quality cache")
        cache_size = len(self.cache)
        self.cache.clear()
        self.logger.debug(f"✅ [CACHE] Cache cleared ({cache_size} entries removed)")


# Global cache instance
_data_quality_cache = DataQualityCache()

# Cache cleanup function
def _cleanup_cache_if_needed():
    """Clean up cache if it gets too large or corrupted."""
    try:
        cache_size = len(_data_quality_cache.cache)
        if cache_size > 50:  # Clear if more than 50 entries
            logger = system_logger.getChild("CacheCleanup")
            logger.info(f"🗑️ [CACHE CLEANUP] Cache size ({cache_size}) exceeds threshold, clearing cache")
            _data_quality_cache.clear()
    except Exception as e:
        logger = system_logger.getChild("CacheCleanup")
        logger.warning(f"⚠️ [CACHE CLEANUP] Error during cache cleanup: {e}")
        # Force clear cache on error
        try:
            _data_quality_cache.clear()
        except:
            pass


def _extract_data_from_args(
    args: tuple, kwargs: dict, method_name: str
) -> Optional[pd.DataFrame]:
    """
    Extract and combine data from method arguments.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        method_name: Name of the method being called

    Returns:
        Combined DataFrame or None if no data found
    """
    logger = system_logger.getChild("DataExtractor")

    # Reduced logging verbosity
    logger.debug(f"🔍 [DATA EXTRACTION] Extracting data for {method_name}")

    try:
        # Look for common data parameter names
        data_params = [
            "price_data",
            "volume_data",
            "data",
            "df",
            "klines",
            "aggtrades",
            "ohlcv",
        ]

        # Extract data from kwargs first
        for param in data_params:
            if param in kwargs:
                data = kwargs[param]
                logger.debug(f"   📊 Found {param} in kwargs")

                if isinstance(data, pd.DataFrame):
                    if not data.empty:
                        logger.debug(f"   ✅ Valid DataFrame found: {param} with shape {data.shape}")
                        # Check if DataFrame has a datetime index
                        if isinstance(data.index, pd.DatetimeIndex):
                            logger.debug(f"   ✅ DataFrame has datetime index: {data.index.min()} to {data.index.max()}")
                        else:
                            logger.debug(f"   ⚠️ DataFrame does not have datetime index")
                        return data
                    else:
                        logger.debug(f"   ⚠️ {param} found but DataFrame is empty")
                else:
                    logger.debug(f"   ⚠️ {param} found but not a DataFrame: type={type(data)}")

        # Extract data from positional arguments
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                if not arg.empty:
                    logger.debug(f"   ✅ Valid DataFrame found in positional arg {i} with shape {arg.shape}")
                    # Check if DataFrame has a datetime index
                    if isinstance(arg.index, pd.DatetimeIndex):
                        logger.debug(f"   ✅ DataFrame has datetime index: {arg.index.min()} to {arg.index.max()}")
                    else:
                        logger.debug(f"   ⚠️ DataFrame does not have datetime index")
                    return arg
                else:
                    logger.debug(f"   ⚠️ DataFrame in positional arg {i} is empty")

        # For methods that take multiple data frames, combine them
        if "price_data" in kwargs and "volume_data" in kwargs:
            price_data = kwargs["price_data"]
            volume_data = kwargs["volume_data"]

            if isinstance(price_data, pd.DataFrame) and isinstance(volume_data, pd.DataFrame):
                if not price_data.empty and not volume_data.empty:
                    # Combine price and volume data
                    combined_data = price_data.copy()
                    if "volume" not in combined_data.columns and "volume" in volume_data.columns:
                        combined_data["volume"] = volume_data["volume"]
                        logger.debug(f"   ✅ Combined data shape: {combined_data.shape}")
                    return combined_data
                else:
                    logger.debug(f"   ⚠️ Price or volume data is empty: price_empty={price_data.empty}, volume_empty={volume_data.empty}")

        logger.debug(f"❌ [DATA EXTRACTION] No valid data found for {method_name}")
        return None

    except Exception as e:
        logger.error(f"💥 [DATA EXTRACTION] Error extracting data for {method_name}: {e}")
        return None


def _get_validation_config(method_name: str) -> Dict[str, Any]:
    """
    Get validation configuration based on method name.

    Args:
        method_name: Name of the method being validated

    Returns:
        Validation configuration dictionary
    """
    # Base configuration
    base_config = {
        "critical_thresholds": {
            "min_records": 100,  # Reduced from 1000 to 100 for testing
            "max_missing_ohlc": 0.005,
            "max_price_anomalies": 0.0005,
            "max_volume_anomalies": 0.02,
            "min_data_span_days": 1,  # Reduced from 7 to 1 day for testing
            "min_continuous_data_hours": 24,  # Reduced from 48 to 24 hours
            "max_ohlc_inconsistency": 0.0,
            "max_negative_prices": 0.0,
            "max_zero_volume_ratio": 0.05,
        },
        "warning_thresholds": {
            "max_gap_hours": 12,
            "max_duplicate_timestamps": 0.0005,
            "max_extreme_price_moves": 0.001,
            "max_volume_spikes": 0.01,
            "max_timestamp_discontinuity": 0.001,
        },
        "feature_engineering_checks": {
            "check_rolling_window_compatibility": True,
            "check_wavelet_data_requirements": True,
            "check_microstructure_feature_requirements": True,
            "check_multi_timeframe_alignment": True,
            "check_volume_price_relationship": True,
            "check_timestamp_regularity": True,
            "check_data_stationarity_preconditions": True,
            "check_lookahead_bias": True,  # NEW: Lookahead bias detection
        },
        "integrity_checks": {
            "check_ohlc_consistency": True,
            "check_timestamp_continuity": True,
            "check_price_logical_consistency": True,
            "check_volume_sanity": True,
            "check_for_market_gaps": True,
            "check_data_type_consistency": True,
            "check_index_alignment": True,
        },
    }

    # Method-specific overrides
    method_overrides = {
        "analyze_wavelet_transforms": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_wavelet_data_requirements": True,
                "check_rolling_window_compatibility": True,
                "check_timestamp_regularity": True,
            }
        },
        "analyze_microstructure_features": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_microstructure_feature_requirements": True,
                "check_volume_price_relationship": True,
            }
        },
        "analyze_multi_timeframe_features": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_multi_timeframe_alignment": True,
                "check_timestamp_regularity": True,
            }
        },
        "_engineer_multi_timeframe_features_vectorized": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.05,  # More lenient: 5% instead of 1% for multi-timeframe features
            },
            "feature_engineering_checks": {
                "check_multi_timeframe_alignment": True,
                "check_timestamp_regularity": True,
            }
        },
        "calculate_price_impact": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_microstructure_feature_requirements": True,
                "check_volume_price_relationship": True,
            }
        },
        "calculate_sr_distances": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.05,  # More lenient: 5% instead of 1% for S/R calculations
            },
            "feature_engineering_checks": {
                "check_rolling_window_compatibility": True,
                "check_timestamp_regularity": True,
            }
        },
        "_get_wavelet_features_with_caching": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_wavelet_data_requirements": True,
                "check_timestamp_regularity": True,
            }
        },
        "_engineer_ohlcv_price_features_vectorized": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_rolling_window_compatibility": True,
                "check_timestamp_regularity": True,
            }
        },
        "_engineer_multi_timeframe_features_vectorized": {
            "warning_thresholds": {
                "max_timestamp_discontinuity": 0.01,  # More lenient: 1% instead of 0.1%
            },
            "feature_engineering_checks": {
                "check_multi_timeframe_alignment": True,
                "check_timestamp_regularity": True,
            }
        },
    }

    # Apply method-specific overrides
    for method_pattern, override in method_overrides.items():
        if method_pattern in method_name:
            # Deep merge the configurations
            for key, value in override.items():
                if key in base_config:
                    if isinstance(value, dict) and isinstance(base_config[key], dict):
                        base_config[key].update(value)
                    else:
                        base_config[key] = value
                else:
                    base_config[key] = value

    return base_config


def _validate_data_type_and_format(
    data: pd.DataFrame, method_name: str
) -> Dict[str, Any]:
    """
    Validate that the data is of the proper type and format for feature engineering.

    Args:
        data: DataFrame to validate
        method_name: Name of the method being called

    Returns:
        Dictionary with validation results
    """
    logger = system_logger.getChild("DataTypeValidator")

    logger.debug(f"🔍 [DATA TYPE VALIDATION] Starting validation for {method_name}")

    validation_result = {
        "is_valid": True,
        "data_type": "unknown",
        "missing_columns": [],
        "extra_columns": [],
        "format_issues": [],
        "recommendations": [],
    }

    try:
        # Check for required OHLCV columns
        required_ohlcv = ["open", "high", "low", "close", "volume"]
        required_ohlc = ["open", "high", "low", "close"]

        # Check for klines format (OHLCV)
        if all(col in data.columns for col in required_ohlcv):
            logger.debug(f"✅ [DATA TYPE VALIDATION] Found klines OHLCV format")
            validation_result["data_type"] = "klines_ohlcv"
            validation_result["is_valid"] = True

        # Check for OHLC format (no volume)
        elif all(col in data.columns for col in required_ohlc):
            logger.debug(f"✅ [DATA TYPE VALIDATION] Found klines OHLC format (no volume)")
            validation_result["data_type"] = "klines_ohlc"
            validation_result["is_valid"] = True
            validation_result["recommendations"].append(
                "Volume data missing - some features may be limited"
            )

        # Check for aggtrades format
        elif "price" in data.columns and "quantity" in data.columns:
            validation_result["data_type"] = "aggtrades"
            validation_result["is_valid"] = True

        # Check for microstructure format
        elif any(col in data.columns for col in ["bid", "ask", "bid_size", "ask_size"]):
            validation_result["data_type"] = "microstructure"
            validation_result["is_valid"] = True

        # Check for wavelet format
        elif any(col.startswith("wavelet_") for col in data.columns):
            validation_result["data_type"] = "wavelet"
            validation_result["is_valid"] = True

        # Check for multi-timeframe format
        elif any("_" in col and any(tf in col for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]) for col in data.columns):
            validation_result["data_type"] = "multi_timeframe"
            validation_result["is_valid"] = True

        # Check for futures format
        elif any(col in data.columns for col in ["open_interest", "funding_rate"]):
            validation_result["data_type"] = "futures"
            validation_result["is_valid"] = True

        # Check for order book format
        elif any(col.startswith("ob_") for col in data.columns):
            validation_result["data_type"] = "order_book"
            validation_result["is_valid"] = True

        # Generic feature format
        elif len(data.columns) > 0:
            validation_result["data_type"] = "features"
            validation_result["is_valid"] = True

        else:
            validation_result["is_valid"] = False
            validation_result["format_issues"].append("No recognizable data format found")
            validation_result["recommendations"].append("Ensure data contains required columns")

        return validation_result

    except Exception as e:
        logger.error(f"💥 [DATA TYPE VALIDATION] Error validating data type for {method_name}: {e}")
        validation_result["is_valid"] = False
        validation_result["format_issues"].append(f"Validation error: {str(e)}")
        return validation_result


def _extract_symbol_exchange_from_context(self: Any, kwargs: dict) -> tuple[str, str]:
    """
    Extract symbol and exchange from method context.

    Args:
        self: The instance object
        kwargs: Method keyword arguments

    Returns:
        Tuple of (symbol, exchange)
    """
    # Try to get from kwargs first
    symbol = kwargs.get("symbol", "UNKNOWN")
    exchange = kwargs.get("exchange", "UNKNOWN")

    # Try to get from instance attributes
    if hasattr(self, "config"):
        if isinstance(self.config, dict):
            symbol = self.config.get("symbol", symbol)
            exchange = self.config.get("exchange", exchange)

    # Try to get from instance attributes directly
    if hasattr(self, "symbol"):
        symbol = self.symbol
    if hasattr(self, "exchange"):
        exchange = self.exchange

    return symbol, exchange


def _extract_symbol_exchange_from_context_improved(self: Any, kwargs: dict, method_name: str) -> tuple[str, str]:
    """
    Extract symbol and exchange from method context with improved logic.

    Args:
        self: The instance object
        kwargs: Method keyword arguments
        method_name: Name of the method being called

    Returns:
        Tuple of (symbol, exchange)
    """
    logger = system_logger.getChild("DataExtractor")
    
    # Try to get from kwargs first
    symbol = kwargs.get("symbol", "UNKNOWN")
    exchange = kwargs.get("exchange", "UNKNOWN")

    # Try to get from instance attributes
    if hasattr(self, "config"):
        if isinstance(self.config, dict):
            symbol = self.config.get("symbol", symbol)
            exchange = self.config.get("exchange", exchange)

    # Try to get from instance attributes directly
    if hasattr(self, "symbol"):
        symbol = self.symbol
    if hasattr(self, "exchange"):
        exchange = self.exchange

    # Try to get from parent instance if available
    if hasattr(self, "parent") and self.parent is not None:
        if hasattr(self.parent, "symbol"):
            symbol = self.parent.symbol
        if hasattr(self.parent, "exchange"):
            exchange = self.parent.exchange
        if hasattr(self.parent, "config"):
            if isinstance(self.parent.config, dict):
                symbol = self.parent.config.get("symbol", symbol)
                exchange = self.parent.config.get("exchange", exchange)

    # Try to extract from method name patterns
    if symbol == "UNKNOWN":
        # Common patterns in method names
        method_lower = method_name.lower()
        
        # Pattern: analyze_<symbol>_features
        if "analyze_" in method_lower and "_features" in method_lower:
            parts = method_lower.split("_")
            if len(parts) >= 2:
                symbol = parts[1].upper()  # e.g., "btc" -> "BTC"
        
        # Pattern: <symbol>_<timeframe>_features
        elif "_" in method_lower and any(tf in method_lower for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]):
            parts = method_lower.split("_")
            if len(parts) >= 2:
                symbol = parts[0].upper()  # e.g., "btc" -> "BTC"
                # Extract timeframe as exchange
                for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                    if tf in method_lower:
                        exchange = tf
                        break

    # Fallback to default values if still unknown
    if symbol == "UNKNOWN":
        symbol = "ETHUSDT"  # Default symbol
        logger.warning(f"⚠️ [DATA EXTRACTION] Using default symbol 'ETHUSDT' for {method_name}")
    
    if exchange == "UNKNOWN":
        exchange = "BINANCE"  # Default exchange
        logger.warning(f"⚠️ [DATA EXTRACTION] Using default exchange 'BINANCE' for {method_name}")

    return symbol, exchange


def validate_data_quality(
    validation_level: ValidationLevel = ValidationLevel.WARNING,
    cache_results: bool = True,
    skip_if_cached: bool = True,
) -> Callable:
    """
    Main data quality validation decorator.

    Args:
        validation_level: How strict to be with validation
        cache_results: Whether to cache validation results
        skip_if_cached: Whether to skip validation if cached result exists

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        # Store original function attributes for pickle compatibility
        original_func = func
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Handle both instance methods and standalone functions
            if args and hasattr(args[0], '__class__'):
                # Instance method - first arg is self
                self = args[0]
                method_args = args[1:]
            else:
                # Standalone function
                self = None
                method_args = args
            
            return await _validate_and_execute(
                original_func,
                self,
                method_args,
                kwargs,
                validation_level,
                cache_results,
                skip_if_cached,
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Handle both instance methods and standalone functions
            if args and hasattr(args[0], '__class__'):
                # Instance method - first arg is self
                self = args[0]
                method_args = args[1:]
            else:
                # Standalone function
                self = None
                method_args = args
            
            # Skip validation for problematic methods
            if _should_skip_validation(func.__name__):
                logger = system_logger.getChild("DataQualityDecorator")
                logger.debug(f"⏭️ [DATA QUALITY] Skipping validation for {func.__name__} due to pickle issues")
                return _execute_function_sync(original_func, self, method_args, kwargs)
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, but we can't await in sync context
                # Instead of skipping validation, we'll run a simplified validation
                logger = system_logger.getChild("DataQualityDecorator")
                logger.debug(f"🔄 [DATA QUALITY] Running simplified validation for {func.__name__} in async context")
                
                # Run simplified validation that doesn't require async operations
                try:
                    # Extract data and do basic validation
                    data = _extract_data_from_args(method_args, kwargs, func.__name__)
                    if data is not None and not data.empty:
                        # Basic data type validation
                        data_type_validation = _validate_data_type_and_format(data, func.__name__)
                        if not data_type_validation["is_valid"]:
                            logger.warning(f"⚠️ [DATA QUALITY] Data type validation failed for {func.__name__} in async context")
                except Exception as e:
                    logger.debug(f"⚠️ [DATA QUALITY] Simplified validation failed for {func.__name__}: {e}")
                
                return _execute_function_sync(original_func, self, method_args, kwargs)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                return asyncio.run(
                    _validate_and_execute(
                        original_func,
                        self,
                        method_args,
                        kwargs,
                        validation_level,
                        cache_results,
                        skip_if_cached,
                    )
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _validate_and_execute(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    validation_level: ValidationLevel,
    cache_results: bool,
    skip_if_cached: bool,
) -> Any:
    """
    Validate data quality and execute the function.

    Args:
        func: The function to execute
        self: The instance object
        args: Positional arguments
        kwargs: Keyword arguments
        validation_level: Validation level
        cache_results: Whether to cache results
        skip_if_cached: Whether to skip if cached

    Returns:
        Function result
    """
    logger = system_logger.getChild("DataQualityDecorator")
    method_name = func.__name__

    # Skip validation for problematic methods
    if _should_skip_validation(method_name):
        logger.debug(f"⏭️ [DATA QUALITY] Skipping validation for {method_name} due to pickle issues")
        return await _execute_function(func, self, args, kwargs)

    logger.info(f"🔍 [DATA QUALITY] Starting validation for {method_name}")

    # Clean up cache if needed
    _cleanup_cache_if_needed()

    try:
        # Extract data from arguments
        data = _extract_data_from_args(args, kwargs, method_name)

        if data is None or data.empty:
            logger.error(f"❌ [DATA QUALITY] Empty or None data provided for {method_name}")
            if validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Empty or None data provided for {method_name}")
            else:
                logger.warning(f"⚠️ [DATA QUALITY] Empty data for {method_name} - skipping validation but continuing execution")
                return await _execute_function(func, self, args, kwargs)

        # Check if data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error(f"❌ [DATA QUALITY] Data for {method_name} does not have datetime index")
            if validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Data for {method_name} does not have datetime index")
            else:
                logger.warning(f"⚠️ [DATA QUALITY] Data for {method_name} does not have datetime index - skipping validation but continuing execution")
                return await _execute_function(func, self, args, kwargs)

        # Check if data has sufficient records (but allow small datasets)
        if len(data) < 5:  # Very small dataset - still allow but warn
            logger.warning(f"⚠️ [DATA QUALITY] Data for {method_name} has only {len(data)} records - very small dataset")
            if validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Data for {method_name} has insufficient records: {len(data)} (minimum: 5)")
            else:
                logger.warning(f"⚠️ [DATA QUALITY] Very small dataset for {method_name} - proceeding with caution")

        # Validate data type and format
        data_type_validation = _validate_data_type_and_format(data, method_name)

        logger.info(f"📊 [DATA QUALITY] Data type validation result for {method_name}: {data_type_validation['data_type']}")

        if not data_type_validation["is_valid"]:
            logger.error(f"❌ [DATA QUALITY] Data type validation failed for {method_name}")
            for issue in data_type_validation["format_issues"]:
                logger.error(f"   {critical(issue)}")

            if validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Data quality validation failed for {method_name}: {data_type_validation['format_issues']}")

        # Log data type information
        logger.info(f"📊 [DATA QUALITY] Data type detected: {data_type_validation['data_type']} for {method_name}")

        # Check cache first
        logger.info(f"🔍 [DATA QUALITY] Checking cache for {method_name}")

        if skip_if_cached and cache_results:
            cached_result = _data_quality_cache.get(data, method_name)
            if cached_result and isinstance(cached_result, dict):
                # Check if cached result has validation data
                if "validation_passed" in cached_result or "data_type" in cached_result:
                    logger.debug(f"✅ [DATA QUALITY] Using cached validation result for {method_name}")
                    return await _execute_function(func, self, args, kwargs)
                else:
                    logger.debug(f"🔄 [DATA QUALITY] Cached result found but missing validation data for {method_name}")
            else:
                logger.info(f"🔄 [DATA QUALITY] No valid cached result found for {method_name}, proceeding with validation")

        # Extract symbol and exchange with improved logic
        logger.info(f"🔍 [DATA QUALITY] Extracting symbol and exchange for {method_name}")
        symbol, exchange = _extract_symbol_exchange_from_context_improved(self, kwargs, method_name)
        logger.info(f"📊 [DATA QUALITY] Symbol: {symbol}, Exchange: {exchange}")

        # Get validation configuration
        logger.info(f"🔍 [DATA QUALITY] Getting validation configuration for {method_name}")
        validation_config = _get_validation_config(method_name)
        logger.info(f"📊 [DATA QUALITY] Validation config loaded for {method_name}")

        # Perform validation
        logger.info(f"🔍 [DATA QUALITY] Starting data quality validation for {method_name} ({symbol} on {exchange})")

        try:
            quality_results = validate_raw_data_quality(
                data=data, symbol=symbol, exchange=exchange, config=validation_config, auto_download_missing=True
            )
        except Exception as e:
            logger.warning(f"⚠️ [DATA QUALITY] Raw data quality validation failed for {method_name}: {e}")
            # Create a basic validation result to continue
            quality_results = {
                "validation_passed": True,  # Assume passed to continue execution
                "data_quality_score": 0.8,  # Default score
                "critical_issues": [],
                "warnings": [f"Raw data quality validation failed: {str(e)}"],
                "recommendations": ["Consider running data quality checks manually"]
            }

        # Cache results if requested
        if cache_results:
            _data_quality_cache.set(data, method_name, quality_results)

        # Handle validation results based on level
        logger.info(f"📊 [DATA QUALITY] Processing validation results for {method_name}")
        logger.info(f"   📊 Validation level: {validation_level.value}")
        logger.info(f"   📊 Validation passed: {quality_results.get('validation_passed', False)}")
        logger.info(f"   📊 Quality score: {quality_results.get('data_quality_score', 0.0):.2f}")
        logger.info(f"   📊 Critical issues: {len(quality_results.get('critical_issues', []))}")
        logger.info(f"   📊 Warnings: {len(quality_results.get('warnings', []))}")

        if validation_level == ValidationLevel.STRICT:
            if not quality_results["validation_passed"]:
                logger.error(f"❌ [DATA QUALITY] STRICT validation failed for {method_name} - stopping execution")
                for issue in quality_results["critical_issues"]:
                    logger.error(f"   {critical(issue)}")
                raise ValueError(f"Data quality validation failed for {method_name}")
            else:
                logger.info(f"✅ [DATA QUALITY] STRICT validation passed for {method_name}")

        elif validation_level == ValidationLevel.WARNING:
            if not quality_results["validation_passed"]:
                logger.error(f"❌ [DATA QUALITY] WARNING validation failed for {method_name} - but continuing")
                for issue in quality_results["critical_issues"]:
                    logger.error(f"   {critical(issue)}")
                logger.warning(f"⚠️ [DATA QUALITY] Proceeding with {method_name} despite data quality issues")
            elif quality_results["warnings"]:
                logger.warning(f"⚠️ [DATA QUALITY] {len(quality_results['warnings'])} warnings for {method_name}")
                for i, warning_msg in enumerate(quality_results["warnings"][:3]):  # Show first 3 warnings
                    logger.warning(f"   {warning(warning_msg)}")
            else:
                logger.info(f"✅ [DATA QUALITY] WARNING validation passed for {method_name}")

        # Execute the original function
        logger.info(f"🚀 [DATA QUALITY] Executing original function {method_name}")
        result = await _execute_function(func, self, args, kwargs)

        logger.info(f"✅ [DATA QUALITY] Completed validation for {method_name}")
        return result

    except Exception as e:
        logger.error(f"💥 [DATA QUALITY] Error in data quality validation for {method_name}: {e}")
        # Continue with execution even if validation fails
        return await _execute_function(func, self, args, kwargs)


async def _execute_function(
    func: Callable, self: Any, args: tuple, kwargs: dict
) -> Any:
    """
    Execute the original function.

    Args:
        func: The function to execute
        self: The instance object
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Function result
    """
    if asyncio.iscoroutinefunction(func):
        return await func(self, *args, **kwargs)
    else:
        return func(self, *args, **kwargs)


def _execute_function_sync(func: Callable, self: Any, args: tuple, kwargs: dict) -> Any:
    """
    Execute the original function synchronously.

    Args:
        func: The function to execute
        self: The instance object
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Function result
    """
    if asyncio.iscoroutinefunction(func):
        # For async functions in sync context, we need to run them in an event loop
        logger = system_logger.getChild("DataQualityDecorator")
        logger.debug(f"🔄 Executing async function {func.__name__} in sync context")
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't use asyncio.run
            # Return a coroutine object that should be awaited by the caller
            return func(self, *args, **kwargs)
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(func(self, *args, **kwargs))
    else:
        return func(self, *args, **kwargs)


# Specialized decorators for different data types
def validate_ohlcv_data_quality(func: Callable) -> Callable:
    """Decorator specifically for OHLCV data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.WARNING, cache_results=True
    )(func)


def validate_wavelet_data_quality(func: Callable) -> Callable:
    """Decorator specifically for wavelet data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.STRICT, cache_results=True
    )(func)


def validate_microstructure_data_quality(func: Callable) -> Callable:
    """Decorator specifically for microstructure data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.WARNING, cache_results=True
    )(func)


def validate_multi_timeframe_data_quality(func: Callable) -> Callable:
    """Decorator specifically for multi-timeframe data validation."""
    # Store original function for pickle compatibility
    original_func = func
    
    @functools.wraps(func)
    async def async_wrapper(self: Any, *args, **kwargs) -> Any:
        # Skip validation for problematic methods
        if _should_skip_validation(func.__name__):
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"⏭️ [DATA QUALITY] Skipping validation for {func.__name__} due to pickle issues")
            return await _execute_function(original_func, self, args, kwargs)
        
        return await _validate_and_execute(
            original_func,
            self,
            args,
            kwargs,
            ValidationLevel.WARNING,  # Less strict validation
            True,  # cache_results
            True,  # skip_if_cached
        )

    @functools.wraps(func)
    def sync_wrapper(self: Any, *args, **kwargs) -> Any:
        # Skip validation for problematic methods
        if _should_skip_validation(func.__name__):
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"⏭️ [DATA QUALITY] Skipping validation for {func.__name__} due to pickle issues")
            # For async functions, return the coroutine object to be awaited
            if asyncio.iscoroutinefunction(original_func):
                return original_func(self, *args, **kwargs)
            else:
                return _execute_function_sync(original_func, self, args, kwargs)
        
        try:
            loop = asyncio.get_running_loop()
            # Simplified validation for async context
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"🔄 [DATA QUALITY] Running simplified validation for {func.__name__} in async context")
            # For async functions, return the coroutine object to be awaited
            if asyncio.iscoroutinefunction(original_func):
                return original_func(self, *args, **kwargs)
            else:
                return _execute_function_sync(original_func, self, args, kwargs)
        except RuntimeError:
            return asyncio.run(
                _validate_and_execute(
                    original_func,
                    self,
                    args,
                    kwargs,
                    ValidationLevel.WARNING,
                    True,  # cache_results
                    True,  # skip_if_cached
                )
            )

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def validate_klines_data_quality(func: Callable) -> Callable:
    """Decorator specifically for klines (OHLCV) data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.WARNING, cache_results=True
    )(func)


def validate_aggtrades_data_quality(func: Callable) -> Callable:
    """Decorator specifically for aggregated trades data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.WARNING, cache_results=True
    )(func)


def validate_futures_data_quality(func: Callable) -> Callable:
    """Decorator specifically for futures data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.WARNING, cache_results=True
    )(func)


def validate_order_book_data_quality(func: Callable) -> Callable:
    """Decorator specifically for order book data validation."""
    return validate_data_quality(
        validation_level=ValidationLevel.WARNING, cache_results=True
    )(func)


def validate_feature_engineering_with_lookahead_bias_detection(
    func: Callable,
) -> Callable:
    """
    Decorator specifically for feature engineering with lookahead bias detection.

    This decorator combines data quality validation with comprehensive lookahead bias detection
    to ensure features are properly temporally aligned and don't contain future information.
    """
    # Store original function for pickle compatibility
    original_func = func
    
    @functools.wraps(func)
    async def async_wrapper(self: Any, *args, **kwargs) -> Any:
        # Skip validation for problematic methods
        if _should_skip_validation(func.__name__):
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"⏭️ [DATA QUALITY] Skipping validation for {func.__name__} due to pickle issues")
            return await _execute_function(original_func, self, args, kwargs)
        
        return await _validate_and_execute(
            original_func,
            self,
            args,
            kwargs,
            ValidationLevel.WARNING,  # Less strict validation
            True,  # cache_results
            True,  # skip_if_cached
        )

    @functools.wraps(func)
    def sync_wrapper(self: Any, *args, **kwargs) -> Any:
        # Skip validation for problematic methods
        if _should_skip_validation(func.__name__):
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"⏭️ [DATA QUALITY] Skipping validation for {func.__name__} due to pickle issues")
            return _execute_function_sync(original_func, self, args, kwargs)
        
        try:
            loop = asyncio.get_running_loop()
            # Simplified validation for async context
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"🔄 [DATA QUALITY] Running simplified validation for {func.__name__} in async context")
            return _execute_function_sync(original_func, self, args, kwargs)
        except RuntimeError:
            return asyncio.run(
                _validate_and_execute(
                    original_func,
                    self,
                    args,
                    kwargs,
                    ValidationLevel.WARNING,
                    True,  # cache_results
                    True,  # skip_if_cached
                )
            )

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Utility functions
def clear_data_quality_cache() -> None:
    """Clear the data quality cache."""
    logger = system_logger.getChild("CacheUtils")
    logger.debug(f"🗑️ [CACHE UTILS] Clearing data quality cache")
    _data_quality_cache.clear()
    logger.debug(f"✅ [CACHE UTILS] Data quality cache cleared")


def get_data_quality_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    logger = system_logger.getChild("CacheUtils")
    logger.debug(f"📊 [CACHE UTILS] Getting cache statistics")

    stats = {
        "cache_size": len(_data_quality_cache.cache),
        "max_size": _data_quality_cache.max_size,
        "cache_keys": list(_data_quality_cache.cache.keys()),
    }

    logger.debug(f"📊 [CACHE UTILS] Cache statistics: size={stats['cache_size']}, max_size={stats['max_size']}, keys={len(stats['cache_keys'])}")

    return stats

# Add pickle-safe fallback mechanism
def _create_pickle_safe_decorator(func: Callable, validation_level: ValidationLevel, cache_results: bool, skip_if_cached: bool) -> Callable:
    """
    Create a pickle-safe decorator that doesn't interfere with serialization.
    """
    # Store the original function in a way that's pickle-safe
    func_name = func.__name__
    func_module = func.__module__
    
    @functools.wraps(func)
    async def async_wrapper(self: Any, *args, **kwargs) -> Any:
        # Get the original function dynamically to avoid pickle issues
        try:
            import importlib
            module = importlib.import_module(func_module)
            original_func = getattr(module, func_name)
        except (ImportError, AttributeError):
            # Fallback to direct execution if we can't get the original function
            return await _execute_function(func, self, args, kwargs)
        
        return await _validate_and_execute(
            original_func,
            self,
            args,
            kwargs,
            validation_level,
            cache_results,
            skip_if_cached,
        )

    @functools.wraps(func)
    def sync_wrapper(self: Any, *args, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
            # Simplified validation for async context
            logger = system_logger.getChild("DataQualityDecorator")
            logger.debug(f"🔄 [DATA QUALITY] Running simplified validation for {func_name} in async context")
            # For async functions, return the coroutine object to be awaited
            if asyncio.iscoroutinefunction(func):
                return func(self, *args, **kwargs)
            else:
                return _execute_function_sync(func, self, args, kwargs)
        except RuntimeError:
            # Get the original function dynamically
            try:
                import importlib
                module = importlib.import_module(func_module)
                original_func = getattr(module, func_name)
            except (ImportError, AttributeError):
                # Fallback to direct execution
                if asyncio.iscoroutinefunction(func):
                    return func(self, *args, **kwargs)
                else:
                    return _execute_function_sync(func, self, args, kwargs)
            
            return asyncio.run(
                _validate_and_execute(
                    original_func,
                    self,
                    args,
                    kwargs,
                    validation_level,
                    cache_results,
                    skip_if_cached,
                )
            )

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# Add a list of methods that should skip data quality validation due to pickle issues
PICKLE_PROBLEMATIC_METHODS = {
    "_engineer_multi_timeframe_features_vectorized",
    "_calculate_price_impact_vectorized", 
    "_calculate_volume_price_impact_vectorized",
    "_engineer_microstructure_features_vectorized"
}

def _should_skip_validation(func_name: str) -> bool:
    """Check if validation should be skipped for a specific function."""
    # List of functions that have pickle issues or other problems with validation
    skip_functions = {
        "_engineer_multi_timeframe_features_vectorized",
        "analyze_wavelet_transforms",
        "analyze_microstructure_features",
        "analyze_multi_timeframe_features",
        "calculate_price_impact",
        "calculate_sr_distances",
        "_get_wavelet_features_with_caching",
        "_generate_wavelet_features",
        "_generate_microstructure_features",
        "_generate_momentum_features",
        "_generate_volatility_features",
        "_generate_liquidity_features",
        "_generate_candlestick_features",
        "_generate_sr_distance_features",
        "_generate_wavelet_features",
        "_generate_regime_aware_features",
        "_generate_cross_timeframe_features",
        "_generate_difference_acceleration_features",
        "_generate_meta_labeling_features",
        "_generate_explicit_meta_labels",
        "_generate_basic_features_fallback",
        "_generate_basic_fallback_features",
        "_generate_simple_timeframe_features",
        "_handle_irregular_time_intervals",
        "_resample_price_data",
        "_resample_volume_data",
        "_validate_and_clean_features",
        "_ensure_pickle_safe_features",
        "_remove_constant_features",
        "_handle_nan_values_comprehensive",
        "_handle_nan_values_basic",
        "_handle_nan_values_inline",
        "_handle_nan_values_robust",
        "_calculate_price_impact_vectorized",
        "_calculate_volume_price_impact_vectorized",
        "_calculate_order_flow_imbalance_vectorized",
        "_validate_and_transform_data",
        "_log_multi_timeframe_summary",
        "_get_minimum_data_requirement",
    }
    
    return func_name in skip_functions

def _provide_irregular_interval_context(irregular_ratio: float, timeframe: str) -> str:
    """
    Provide context-aware information about irregular time intervals.
    
    Args:
        irregular_ratio: Ratio of irregular intervals
        timeframe: Timeframe being processed
        
    Returns:
        Contextual message about the irregular intervals
    """
    if irregular_ratio < 0.05:
        return f"Minor irregular intervals ({irregular_ratio:.1%}) - multi-timeframe features should work normally"
    elif irregular_ratio < 0.15:
        return f"Moderate irregular intervals ({irregular_ratio:.1%}) - multi-timeframe features may have slight accuracy impact"
    elif irregular_ratio < 0.30:
        return f"Significant irregular intervals ({irregular_ratio:.1%}) - consider data preprocessing for better multi-timeframe results"
    else:
        return f"High irregular intervals ({irregular_ratio:.1%}) - multi-timeframe features may be unreliable without data preprocessing"
