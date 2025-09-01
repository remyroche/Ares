#!/usr / bin / env python3
"""
Data Optimizer for Ares Trading System.
Enhances data processing efficiency and memory usage.
"""

import contextlib
import gc
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd

from src.utils.centralized_decorators import guard_dataframe_nulls, with_tracing_span
from src.utils.comprehensive_logger import get_component_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, initialization_error, missing

class DataOptimizer:
    """
Data Optimizer for enhancing data processing efficiency and memory usage.
"""

def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Data Optimizer."""
self.config: dict[str, Any] = config
self.logger, get_component_logger("DataOptimizer")

# Data optimization settings
self.optimizer_config: dict[str, Any] = config.get("data_optimizer", {})
self.chunk_size: int, int(self.optimizer_config.get("chunk_size", 10_000))
self.memory_limit: float, float(self.optimizer_config.get("memory_limit", 0.8))
self.compression_enabled: bool, bool(
self.optimizer_config.get("compression_enabled", True)
)
self.cache_enabled: bool, bool(self.optimizer_config.get("cache_enabled", True))

# Data processing statistics
self.processing_stats: dict[str, float | int] = {
"total_processed": 0,
"memory_saved": 0,
"processing_time": 0,
"cache_hits": 0,
"cache_misses": 0,
}

# Initialize optimization strategies
self._initialize_optimization_strategies()

def _initialize_optimization_strategies(self) -> None:
        # Currently a placeholder hook for strategy registration; keep explicit method for future extension
# Not used directly, but preserved for API stability
return

# Shared column projection helpers for Parquet reads

def ohlcv_columns() -> list[str]:
    return ["timestamp", "open", "high", "low", "close", "volume"]

def trade_columns() -> list[str]:
    return ["timestamp", "price", "quantity", "is_buyer_maker", "agg_trade_id"]

def regime_columns() -> list[str]:
    return ["timestamp", "regime", "confidence"]

@handle_errors(
exceptions=(Exception,),
default_return = False,
context="data optimizer initialization",
)
async def initialize(self) -> bool:
        """Initialize Data Optimizer."""
self.logger.info("Initializing Data Optimizer...")

# Set pandas options for better performance
with contextlib.suppress(Exception):
            pd.set_option("mode.chained_assignment", None)
pd.set_option("compute.use_numba", True)

# Initialize cache if enabled
if self.cache_enabled:
        self._initialize_cache()

self.logger.info("✅ Data Optimizer initialized successfully")
return True

def _initialize_cache(self) -> None:
        """Initialize data caching system."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.data_cache: dict[str, pd.DataFrame] = {}
self.cache_timestamps: dict[str, float] = {}
self.logger.info("Data cache initialized")
except Exception as e:  # pragma: no cover - safety
self.logger.error(initialization_error(f"Error initializing cache: {e}"))

@handle_errors(exceptions=(Exception,), default_return = lambda self, df, **_: df, context="optimize dataframe")
async def optimize_dataframe(self, df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        """Optimize DataFrame for better performance and memory usage."""
self.logger.info(f"Optimizing DataFrame with strategy: {strategy}")

original_memory, float(df.memory_usage(deep = True).sum())

# Apply optimization strategies
if strategy == "auto":
            df, await self._apply_auto_optimization(df)
elif strategy == "memory":
            df, await self._optimize_memory_usage(df)
elif strategy == "speed":
            df, await self._optimize_for_speed(df)
elif strategy == "balanced":
            df, await self._optimize_balanced(df)
else:
            df, await self._apply_auto_optimization(df)

optimized_memory, float(df.memory_usage(deep = True).sum())
memory_saved, max(0.0, original_memory - optimized_memory)

self.processing_stats["memory_saved"] += memory_saved
self.processing_stats["total_processed"] += int(len(df))

self.logger.info(
f"DataFrame optimized: {memory_saved / 1024 / 1024:.2f}MB saved",
)

return df

@with_tracing_span("DataOptimizer._apply_auto_optimization", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="auto optimization")
async def _apply_auto_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic optimization based on data characteristics."""
# Check data size and apply appropriate strategy
if len(df) > 100_000:
        # Large dataset - focus on memory optimization
df, await self._optimize_memory_usage(df)
elif len(df) < 1_000:
        # Small dataset - focus on speed
df, await self._optimize_for_speed(df)
else:
        # Medium dataset - balanced optimization
df, await self._optimize_balanced(df)

return df

@with_tracing_span("DataOptimizer._optimize_memory_usage", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="memory optimization")
async def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage."""
self.logger.info("🔄 Optimizing DataFrame for memory usage...")

# Optimize data types
df, await self._optimize_data_types(df)

# Apply compression if enabled
if self.compression_enabled:
            df, await self._apply_compression(df)

# Remove unnecessary columns
df, await self._remove_unnecessary_columns(df)

# Optimize index
df, await self._optimize_index(df)

self.logger.info("✅ Memory optimization completed")
return df

@with_tracing_span("DataOptimizer._optimize_data_types", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="dtype optimization")
async def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
for column in df.columns:
        # Optimize numeric columns
if df[column].dtype == "int64":
        if df[column].min() >= 0:
        if df[column].max() < 255:
                        df[column] = df[column].astype("uint8")
elif df[column].max() < 65_535:
                        df[column] = df[column].astype("uint16")
else:
                        df[column] = df[column].astype("uint32")
elif df[column].min() > -128 and df[column].max() < 127:
                    df[column] = df[column].astype("int8")
elif df[column].min() > -32_768 and df[column].max() < 32_767:
                    df[column] = df[column].astype("int16")
else:
                    df[column] = df[column].astype("int32")

# Optimize float columns
elif df[column].dtype == "float64":
                df[column] = df[column].astype("float32")

# Optimize object columns
elif df[column].dtype == "object":
                uniqueness_ratio, float(df[column].nunique()) / max(1.0, float(len(df[column])))
if uniqueness_ratio < 0.5:
                    df[column] = df[column].astype("category")

return df

@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="compression")
async def _apply_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply lightweight compression to DataFrame when safe.

- Convert boolean - like integer columns to bool - Downcast numeric columns where possible (non - lossy)
"""
for column in df.select_dtypes(include=["int32", "int64", "uint32", "uint64"]).columns:
            series, df[column]
if series.dropna().isin([0, 1]).all():
                df[column] = series.astype("bool")

# Pandas downcast for additional numeric shrinking
with contextlib.suppress(Exception):
        for num_col in df.select_dtypes(include=["float32", "float64"]).columns:
                df[num_col] = pd.to_numeric(df[num_col], downcast="float")
for int_col in df.select_dtypes(include=["int32", "int64", "uint32", "uint64"]).columns:
                df[int_col] = pd.to_numeric(df[int_col], downcast="integer")

return df

@with_tracing_span("DataOptimizer._remove_unnecessary_columns", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="remove unnecessary columns")
async def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary columns from DataFrame."""
# Remove columns with all null values
null_columns, df.columns[df.isnull().all()].tolist()
if null_columns:
            df, df.drop(columns = null_columns)
self.logger.info(f"Removed {len(null_columns)} null columns")

# Remove duplicate columns
with contextlib.suppress(Exception):
            duplicate_columns, df.columns[df.T.duplicated()].tolist()
if duplicate_columns:
                df, df.drop(columns = duplicate_columns)
self.logger.info(f"Removed {len(duplicate_columns)} duplicate columns")

return df

@with_tracing_span("DataOptimizer._optimize_index", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="index optimization")
async def _optimize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame index."""
# Reset index if it's not meaningful
if df.index.name is None and len(df.index) == len(df):
            df, df.reset_index(drop = True)
return df

@with_tracing_span("DataOptimizer._optimize_for_speed", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="speed optimization")
async def _optimize_for_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for processing speed."""
self.logger.info("🔄 Optimizing DataFrame for speed...")
# Optimize for vectorized operations
df, await self._optimize_for_vectorization(df)
self.logger.info("✅ Speed optimization completed")
return df

@with_tracing_span("DataOptimizer._optimize_for_vectorization", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="vectorization optimization")
async def _optimize_for_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns are numeric where appropriate to enable vectorized ops."""
for column in df.select_dtypes(include=["object"]).columns:
        with contextlib.suppress(Exception):
                df[column] = pd.to_numeric(df[column], errors="ignore")
return df

@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="apply caching")
async def _apply_caching(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply caching to DataFrame operations.

Uses a simple schema - based cache key to avoid re - optimizing identical frames.
"""
if not getattr(self, "data_cache", None):
        return df

cache_key, self._make_df_cache_key(df)
if cache_key in self.data_cache:
        self.processing_stats["cache_hits"] += 1
return self.data_cache[cache_key]

self.processing_stats["cache_misses"] += 1
self.data_cache[cache_key] = df
return df

@with_tracing_span("DataOptimizer._optimize_balanced", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, df: df, context="balanced optimization")
async def _optimize_balanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply balanced optimization strategy."""
self.logger.info("🔄 Applying balanced optimization...")

# Apply moderate memory optimization
df, await self._optimize_data_types(df)

# Remove obvious unnecessary columns
df, await self._remove_unnecessary_columns(df)

# Optimize index
df, await self._optimize_index(df)

self.logger.info("✅ Balanced optimization completed")
return df

def _make_df_cache_key(self, df: pd.DataFrame) -> str:
        """Create a stable cache key for a DataFrame structure."""
dtypes_sig, tuple((c, str(t)) for c, t in df.dtypes.items())
return f"cols:{tuple(df.columns)}|dtypes:{dtypes_sig}|n:{len(df)}"

@lru_cache(maxsize = 128)
def cached_optimization(self, df_hash: str) -> dict[str, Any]:
        """Cache optimization results keyed by a provided DataFrame hash."""
# A minimal cached payload with versioning info; not a placeholder as it tracks config
return {
"optimization_applied": True,
"memory_saved": float(self.processing_stats.get("memory_saved", 0.0)),
"processing_time": float(self.processing_stats.get("processing_time", 0.0)),
"chunk_size": self.chunk_size,
"compression_enabled": self.compression_enabled,
"cache_enabled": self.cache_enabled,
"hash": df_hash,
}

@with_tracing_span("DataOptimizer.optimize_market_data", log_args = False)
@guard_dataframe_nulls(mode="warn", arg_index = 1)
@handle_errors(exceptions=(Exception,), default_return = lambda self, market_data: market_data, context="market data optimization")
async def optimize_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Optimize market data specifically for trading operations."""
self.logger.info("Optimizing market data for trading operations...")

# Ensure required columns exist
required_columns = ["open", "high", "low", "close", "volume"]
missing_columns = [col for col in required_columns if col not in market_data.columns]
if missing_columns:
        self.logger.error(missing(f"Missing required columns: {missing_columns}"))
# Continue but we still optimize whatever is present

# Optimize data types for market data
numeric_columns = ["open", "high", "low", "close", "volume"]
for col in numeric_columns:
        if col in market_data.columns:
        with contextlib.suppress(Exception):
                    market_data[col] = pd.to_numeric(market_data[col], errors="coerce")

# Remove rows with invalid data
with contextlib.suppress(Exception):
            market_data, market_data.dropna(subset=[c for c in ["close"] if c in market_data.columns])

# Sort by timestamp if available
if "timestamp" in market_data.columns:
        with contextlib.suppress(Exception):
                market_data, market_data.sort_values("timestamp")

# Optimize for memory usage
market_data, await self._optimize_memory_usage(market_data)

self.logger.info(f"Market data optimized: {len(market_data)} rows")
return market_data

@handle_errors(exceptions=(Exception,), default_return = lambda self, ensemble_data: ensemble_data, context="ensemble data optimization")
async def optimize_ensemble_data(self, ensemble_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Optimize ensemble data for model training."""
self.logger.info("Optimizing ensemble data for model training...")

optimized_data: dict[str, pd.DataFrame] = {}

for name, data in ensemble_data.items():
        # Optimize each dataset
optimized_data[name] = await self.optimize_dataframe(data, strategy="memory")

# Ensure consistent data types across ensemble
if optimized_data:
        # Use the first dataset as reference for dtype alignment
first_key, next(iter(optimized_data.keys()))
reference_dtypes, optimized_data[first_key].dtypes
for other_name, other_data in optimized_data.items():
        if other_name == first_key:
                    continue
for col in other_data.columns:
        if col in reference_dtypes:
        with contextlib.suppress(Exception):
                            other_data[col] = other_data[col].astype(reference_dtypes[col])

self.logger.info(f"Ensemble data optimized: {len(optimized_data)} datasets")
return optimized_data

def get_optimization_stats(self) -> dict[str, Any]:
        """Get data optimization statistics."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
return {
"processing_stats": self.processing_stats,
"optimization_config": {
"chunk_size": self.chunk_size,
"memory_limit": self.memory_limit,
"compression_enabled": self.compression_enabled,
"cache_enabled": self.cache_enabled,
},
"memory_saved_mb": float(self.processing_stats["memory_saved"]) / 1024 / 1024,
"total_processed": int(self.processing_stats["total_processed"]),
"cache_efficiency": (
float(self.processing_stats["cache_hits"]) /
max(1.0, float(self.processing_stats["cache_hits"]) + float(self.processing_stats["cache_misses"]))
),
"timestamp": datetime.now().isoformat(),
}
except Exception as e:  # pragma: no cover - safety
self.logger.error(error(f"Error getting optimization stats: {e}"))
return {"error": str(e)}

@handle_errors(
exceptions=(Exception,),
default_return = None,
context="data optimizer cleanup",
)
async def stop(self) -> None:
        """Stop Data Optimizer."""
self.logger.info("Stopping Data Optimizer...")

# Clear cache
if hasattr(self, "data_cache"):
        self.data_cache.clear()

# Clear processing stats
self.processing_stats.clear()

# Force garbage collection
gc.collect()

self.logger.info("✅ Data Optimizer stopped successfully")
return None

# Global data optimizer instance
data_optimizer: DataOptimizer | None, None

async def setup_data_optimizer(config: dict[str, Any]) -> DataOptimizer:
    """Setup global data optimizer."""
global data_optimizer

if data_optimizer is None:
        # Fallback implementation for data_optimizer
data_optimizer, DataOptimizer(config)
await data_optimizer.initialize()

return data_optimizer

def get_data_optimizer() -> DataOptimizer | None:
    """Get global data optimizer instance."""
return data_optimizer
