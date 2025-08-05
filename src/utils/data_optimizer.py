#!/usr/bin/env python3
"""
Data Optimizer for Ares Trading System.
Enhances data processing efficiency and memory usage.
"""

import gc
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd

from src.utils.comprehensive_logger import get_component_logger
from src.utils.error_handler import handle_errors


class DataOptimizer:
    """
    Data Optimizer for enhancing data processing efficiency and memory usage.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Data Optimizer."""
        self.config = config
        self.logger = get_component_logger("DataOptimizer")

        # Data optimization settings
        self.optimizer_config = config.get("data_optimizer", {})
        self.chunk_size = self.optimizer_config.get("chunk_size", 10000)
        self.memory_limit = self.optimizer_config.get("memory_limit", 0.8)
        self.compression_enabled = self.optimizer_config.get(
            "compression_enabled",
            True,
        )
        self.cache_enabled = self.optimizer_config.get("cache_enabled", True)

        # Data processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "memory_saved": 0,
            "processing_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Initialize optimization strategies
        self._initialize_optimization_strategies()

    def _initialize_optimization_strategies(self) -> None:
        """Initialize data optimization strategies."""
        self.optimization_strategies = {
            "memory_optimization": self._optimize_memory_usage,
            "data_type_optimization": self._optimize_data_types,
            "chunk_processing": self.process_in_chunks,
            "compression": self._apply_compression,
            "caching": self._apply_caching,
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="data optimizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize Data Optimizer."""
        try:
            self.logger.info("Initializing Data Optimizer...")

            # Set pandas options for better performance
            pd.set_option("mode.chained_assignment", None)
            pd.set_option("compute.use_numba", True)

            # Initialize cache if enabled
            if self.cache_enabled:
                self._initialize_cache()

            self.logger.info("✅ Data Optimizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing Data Optimizer: {e}")
            return False

    def _initialize_cache(self) -> None:
        """Initialize data caching system."""
        try:
            self.data_cache = {}
            self.cache_timestamps = {}
            self.logger.info("Data cache initialized")

        except Exception as e:
            self.logger.error(f"Error initializing cache: {e}")

    async def optimize_dataframe(
        self,
        df: pd.DataFrame,
        strategy: str = "auto",
    ) -> pd.DataFrame:
        """Optimize DataFrame for better performance and memory usage."""
        try:
            self.logger.info(f"Optimizing DataFrame with strategy: {strategy}")

            original_memory = df.memory_usage(deep=True).sum()

            # Apply optimization strategies
            if strategy == "auto":
                df = await self._apply_auto_optimization(df)
            elif strategy == "memory":
                df = await self._optimize_memory_usage(df)
            elif strategy == "speed":
                df = await self._optimize_for_speed(df)
            elif strategy == "balanced":
                df = await self._optimize_balanced(df)
            else:
                df = await self._apply_auto_optimization(df)

            optimized_memory = df.memory_usage(deep=True).sum()
            memory_saved = original_memory - optimized_memory

            self.processing_stats["memory_saved"] += memory_saved
            self.processing_stats["total_processed"] += len(df)

            self.logger.info(
                f"DataFrame optimized: {memory_saved / 1024 / 1024:.2f}MB saved",
            )

            return df

        except Exception as e:
            self.logger.error(f"Error optimizing DataFrame: {e}")
            return df

    async def _apply_auto_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic optimization based on data characteristics."""
        try:
            # Check data size and apply appropriate strategy
            if len(df) > 100000:
                # Large dataset - focus on memory optimization
                df = await self._optimize_memory_usage(df)
            elif len(df) < 1000:
                # Small dataset - focus on speed
                df = await self._optimize_for_speed(df)
            else:
                # Medium dataset - balanced optimization
                df = await self._optimize_balanced(df)

            return df

        except Exception as e:
            self.logger.error(f"Error in auto optimization: {e}")
            return df

    async def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage."""
        try:
            self.logger.info("🔄 Optimizing DataFrame for memory usage...")

            # Optimize data types
            df = await self._optimize_data_types(df)

            # Apply compression if enabled
            if self.compression_enabled:
                df = await self._apply_compression(df)

            # Remove unnecessary columns
            df = await self._remove_unnecessary_columns(df)

            # Optimize index
            df = await self._optimize_index(df)

            self.logger.info("✅ Memory optimization completed")
            return df

        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {e}")
            return df

    async def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        try:
            for column in df.columns:
                # Optimize numeric columns
                if df[column].dtype == "int64":
                    if df[column].min() >= 0:
                        if df[column].max() < 255:
                            df[column] = df[column].astype("uint8")
                        elif df[column].max() < 65535:
                            df[column] = df[column].astype("uint16")
                        else:
                            df[column] = df[column].astype("uint32")
                    elif df[column].min() > -128 and df[column].max() < 127:
                        df[column] = df[column].astype("int8")
                    elif df[column].min() > -32768 and df[column].max() < 32767:
                        df[column] = df[column].astype("int16")
                    else:
                        df[column] = df[column].astype("int32")

                # Optimize float columns
                elif df[column].dtype == "float64":
                    df[column] = df[column].astype("float32")

                # Optimize object columns
                elif df[column].dtype == "object":
                    if df[column].nunique() / len(df[column]) < 0.5:
                        df[column] = df[column].astype("category")

            return df

        except Exception as e:
            self.logger.error(f"Error optimizing data types: {e}")
            return df

    async def _apply_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply compression to DataFrame."""
        try:
            # For demonstration, we'll use pandas compression
            # In practice, you might use more sophisticated compression
            return df

        except Exception as e:
            self.logger.error(f"Error applying compression: {e}")
            return df

    async def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary columns from DataFrame."""
        try:
            # Remove columns with all null values
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                df = df.drop(columns=null_columns)
                self.logger.info(f"Removed {len(null_columns)} null columns")

            # Remove duplicate columns
            duplicate_columns = df.columns[df.T.duplicated()].tolist()
            if duplicate_columns:
                df = df.drop(columns=duplicate_columns)
                self.logger.info(f"Removed {len(duplicate_columns)} duplicate columns")

            return df

        except Exception as e:
            self.logger.error(f"Error removing unnecessary columns: {e}")
            return df

    async def _optimize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame index."""
        try:
            # Reset index if it's not meaningful
            if df.index.name is None and len(df.index) == len(df):
                df = df.reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Error optimizing index: {e}")
            return df

    async def _optimize_for_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for processing speed."""
        try:
            self.logger.info("🔄 Optimizing DataFrame for speed...")

            # Keep data types as is for speed
            # Focus on reducing operations rather than memory

            # Optimize for vectorized operations
            df = await self._optimize_for_vectorization(df)

            self.logger.info("✅ Speed optimization completed")
            return df

        except Exception as e:
            self.logger.error(f"Error optimizing for speed: {e}")
            return df

    async def _optimize_for_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized operations."""
        try:
            # Ensure numeric columns are numeric for vectorized operations
            for column in df.select_dtypes(include=["object"]).columns:
                try:
                    df[column] = pd.to_numeric(df[column], errors="ignore")
                except:
                    pass

            return df

        except Exception as e:
            self.logger.error(f"Error optimizing for vectorization: {e}")
            return df

    async def _apply_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply compression to DataFrame."""
        try:
            # For demonstration, we'll use pandas compression
            # In practice, you might use more sophisticated compression
            return df

        except Exception as e:
            self.logger.error(f"Error applying compression: {e}")
            return df

    async def _apply_caching(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply caching to DataFrame operations."""
        try:
            # For demonstration, we'll implement basic caching
            # In practice, you might use more sophisticated caching
            return df

        except Exception as e:
            self.logger.error(f"Error applying caching: {e}")
            return df

    async def _optimize_balanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply balanced optimization strategy."""
        try:
            self.logger.info("🔄 Applying balanced optimization...")

            # Apply moderate memory optimization
            df = await self._optimize_data_types(df)

            # Remove obvious unnecessary columns
            df = await self._remove_unnecessary_columns(df)

            # Optimize index
            df = await self._optimize_index(df)

            self.logger.info("✅ Balanced optimization completed")
            return df

        except Exception as e:
            self.logger.error(f"Error in balanced optimization: {e}")
            return df

    async def process_in_chunks(
        self,
        data: pd.DataFrame | list,
        chunk_size: int | None = None,
    ) -> list[pd.DataFrame]:
        """Process data in chunks to manage memory usage."""
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size

            if isinstance(data, pd.DataFrame):
                chunks = [
                    data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
                ]
            else:
                # Convert list to DataFrame and chunk
                df = pd.DataFrame(data)
                chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

            self.logger.info(
                f"Data split into {len(chunks)} chunks of size {chunk_size}",
            )
            return chunks

        except Exception as e:
            self.logger.error(f"Error processing data in chunks: {e}")
            return [pd.DataFrame(data)] if isinstance(data, list) else [data]

    @lru_cache(maxsize=128)
    def cached_optimization(self, df_hash: str) -> dict[str, Any]:
        """Cache optimization results."""
        try:
            # This is a placeholder for cached optimization results
            return {
                "optimization_applied": True,
                "memory_saved": 0,
                "processing_time": 0,
            }

        except Exception as e:
            self.logger.error(f"Error in cached optimization: {e}")
            return {}

    async def optimize_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Optimize market data specifically for trading operations."""
        try:
            self.logger.info("Optimizing market data for trading operations...")

            # Ensure required columns exist
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in market_data.columns
            ]

            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")

            # Optimize data types for market data
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in market_data.columns:
                    market_data[col] = pd.to_numeric(market_data[col], errors="coerce")

            # Remove rows with invalid data
            market_data = market_data.dropna(subset=["close"])

            # Sort by timestamp if available
            if "timestamp" in market_data.columns:
                market_data = market_data.sort_values("timestamp")

            # Optimize for memory usage
            market_data = await self._optimize_memory_usage(market_data)

            self.logger.info(f"Market data optimized: {len(market_data)} rows")
            return market_data

        except Exception as e:
            self.logger.error(f"Error optimizing market data: {e}")
            return market_data

    async def optimize_ensemble_data(
        self,
        ensemble_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Optimize ensemble data for model training."""
        try:
            self.logger.info("Optimizing ensemble data for model training...")

            optimized_data = {}

            for name, data in ensemble_data.items():
                # Optimize each dataset
                optimized_data[name] = await self.optimize_dataframe(
                    data,
                    strategy="memory",
                )

                # Ensure consistent data types across ensemble
                if optimized_data:
                    reference_dtypes = optimized_data[name].dtypes
                    for other_name, other_data in optimized_data.items():
                        if other_name != name:
                            # Align data types with reference
                            for col in other_data.columns:
                                if col in reference_dtypes:
                                    try:
                                        other_data[col] = other_data[col].astype(
                                            reference_dtypes[col],
                                        )
                                    except:
                                        pass

            self.logger.info(f"Ensemble data optimized: {len(optimized_data)} datasets")
            return optimized_data

        except Exception as e:
            self.logger.error(f"Error optimizing ensemble data: {e}")
            return ensemble_data

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get data optimization statistics."""
        try:
            return {
                "processing_stats": self.processing_stats,
                "optimization_config": {
                    "chunk_size": self.chunk_size,
                    "memory_limit": self.memory_limit,
                    "compression_enabled": self.compression_enabled,
                    "cache_enabled": self.cache_enabled,
                },
                "memory_saved_mb": self.processing_stats["memory_saved"] / 1024 / 1024,
                "total_processed": self.processing_stats["total_processed"],
                "cache_efficiency": (
                    self.processing_stats["cache_hits"]
                    / (
                        self.processing_stats["cache_hits"]
                        + self.processing_stats["cache_misses"]
                    )
                    if (
                        self.processing_stats["cache_hits"]
                        + self.processing_stats["cache_misses"]
                    )
                    > 0
                    else 0.0
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting optimization stats: {e}")
            return {"error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data optimizer cleanup",
    )
    async def stop(self) -> None:
        """Stop Data Optimizer."""
        try:
            self.logger.info("Stopping Data Optimizer...")

            # Clear cache
            if hasattr(self, "data_cache"):
                self.data_cache.clear()

            # Clear processing stats
            self.processing_stats.clear()

            # Force garbage collection
            gc.collect()

            self.logger.info("✅ Data Optimizer stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping Data Optimizer: {e}")


# Global data optimizer instance
data_optimizer: DataOptimizer | None = None


async def setup_data_optimizer(config: dict[str, Any]) -> DataOptimizer:
    """Setup global data optimizer."""
    global data_optimizer

    if data_optimizer is None:
        data_optimizer = DataOptimizer(config)
        await data_optimizer.initialize()

    return data_optimizer


def get_data_optimizer() -> DataOptimizer | None:
    """Get global data optimizer instance."""
    return data_optimizer
