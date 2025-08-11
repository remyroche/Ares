# src/training/steps/precompute_wavelet_features.py

"""
Pre-computation script for wavelet features.
Generates and caches expensive wavelet calculations once for the entire dataset,
enabling fast loading during backtesting without recalculation.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.utils.data_optimizer import ohlcv_columns

from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
    WaveletFeatureCache,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
)


class WaveletFeaturePrecomputer:
    """
    Pre-computation system for wavelet features.
    Processes entire datasets and caches results for fast backtesting.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("WaveletFeaturePrecomputer")

        # Pre-computation configuration
        self.precompute_config = config.get("wavelet_precompute", {})
        self.enable_batch_processing = self.precompute_config.get(
            "enable_batch_processing",
            True,
        )
        self.batch_size = self.precompute_config.get("batch_size", 10000)
        self.enable_progress_tracking = self.precompute_config.get(
            "enable_progress_tracking",
            True,
        )
        self.enable_parallel_processing = self.precompute_config.get(
            "enable_parallel_processing",
            False,
        )
        self.max_workers = self.precompute_config.get("max_workers", 4)

        # Initialize components
        self.feature_engineer = None
        self.wavelet_cache = None

    async def initialize(self) -> bool:
        """Initialize the pre-computation system."""
        try:
            self.logger.info(
                "🚀 Initializing wavelet feature pre-computation system...",
            )

            # Initialize feature engineering
            self.feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
            await self.feature_engineer.initialize()

            # Initialize cache
            self.wavelet_cache = WaveletFeatureCache(self.config)

            self.logger.info(
                "✅ Wavelet feature pre-computation system initialized successfully",
            )
            return True

        except Exception:
            self.print(
                initialization_error(
                    "❌ Error initializing pre-computation system: {e}",
                ),
            )
            return False

    async def precompute_dataset(
        self,
        data_path: str,
        output_path: str | None = None,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> bool:
        """
        Pre-compute wavelet features for an entire dataset.

        Args:
            data_path: Path to the dataset file
            output_path: Path for output files (optional)
            symbol: Symbol to process (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"📊 Starting pre-computation for dataset: {data_path}")

            # Load dataset
            dataset = await self._load_dataset(data_path, symbol, start_date, end_date)
            if dataset is None or dataset.empty:
                self.print(error("No data to process"))
                return False

            # Process dataset
            success = await self._process_dataset(dataset, output_path)

            if success:
                self.logger.info("✅ Dataset pre-computation completed successfully")
            else:
                self.print(failed("❌ Dataset pre-computation failed"))

            return success

        except Exception:
            self.print(error("Error in dataset pre-computation: {e}"))
            return False

    async def _load_dataset(
        self,
        data_path: str,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Load and filter dataset."""
        try:
            # Load dataset based on file extension
            file_path = Path(data_path)

            if file_path.suffix.lower() == ".parquet":
                # Prefer dataset scan with projection if a directory is provided
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    columns = ohlcv_columns()
                    if file_path.is_dir():
                        dataset = pdm.scan_dataset(
                            str(file_path), columns=columns, to_pandas=True
                        )
                    else:
                        from src.utils.logger import log_io_operation

                        with log_io_operation(
                            self.logger,
                            "read_parquet",
                            data_path,
                            columns="ohlcv_columns",
                        ):
                            dataset = pd.read_parquet(data_path, columns=columns)
                except Exception:
                    from src.utils.logger import log_io_operation

                    with log_io_operation(self.logger, "read_parquet", data_path):
                        dataset = pd.read_parquet(data_path)
            elif file_path.suffix.lower() == ".csv":
                from src.utils.logger import log_io_operation

                with log_io_operation(self.logger, "read_csv", data_path):
                    dataset = pd.read_csv(data_path, parse_dates=True)
            elif file_path.suffix.lower() == ".h5":
                dataset = pd.read_hdf(data_path)
            else:
                self.print(error("Unsupported file format: {file_path.suffix}"))
                return None

            # Apply filters
            if symbol:
                dataset = dataset[dataset.get("symbol", "") == symbol]

            if start_date:
                dataset = dataset[dataset.index >= start_date]

            if end_date:
                dataset = dataset[dataset.index <= end_date]

            self.logger.info(
                f"📈 Loaded dataset: {len(dataset)} rows, {len(dataset.columns)} columns",
            )
            return dataset

        except Exception:
            self.print(error("Error loading dataset: {e}"))
            return None

    async def _process_dataset(
        self,
        dataset: pd.DataFrame,
        output_path: str | None = None,
    ) -> bool:
        """Process dataset in batches."""
        try:
            total_rows = len(dataset)
            total_batches = (total_rows + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"🔄 Processing {total_rows} rows in {total_batches} batches",
            )

            # Process in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_rows)

                batch_data = dataset.iloc[start_idx:end_idx]

                # Process batch
                batch_success = await self._process_batch(
                    batch_data,
                    batch_idx,
                    total_batches,
                )

                if not batch_success:
                    self.logger.error(
                        f"❌ Failed to process batch {batch_idx + 1}/{total_batches}",
                    )
                    return False

                # Progress tracking
                if self.enable_progress_tracking:
                    progress = (batch_idx + 1) / total_batches * 100
                    self.logger.info(
                        f"📊 Progress: {progress:.1f}% ({batch_idx + 1}/{total_batches} batches)",
                    )

            return True

        except Exception:
            self.print(error("Error processing dataset: {e}"))
            return False

    async def _process_batch(
        self,
        batch_data: pd.DataFrame,
        batch_idx: int,
        total_batches: int,
    ) -> bool:
        """Process a single batch of data."""
        try:
            # Extract price and volume data
            price_data = self._extract_price_data(batch_data)
            volume_data = self._extract_volume_data(batch_data)

            if price_data.empty:
                self.print(error("Empty price data in batch {batch_idx + 1}"))
                return True

            # Generate wavelet features
            wavelet_features = (
                await self.feature_engineer._get_wavelet_features_with_caching(
                    price_data,
                    volume_data,
                )
            )

            if not wavelet_features:
                self.logger.warning(
                    f"No wavelet features generated for batch {batch_idx + 1}",
                )
                return True

            # Save batch results
            return await self._save_batch_results(
                batch_data,
                wavelet_features,
                batch_idx,
                total_batches,
            )

        except Exception:
            self.print(error("Error processing batch {batch_idx + 1}: {e}"))
            return False

    def _extract_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract price data from dataset."""
        try:
            # Look for OHLCV columns
            price_columns = ["open", "high", "low", "close", "volume"]
            available_columns = [col for col in price_columns if col in data.columns]

            if len(available_columns) < 4:  # Need at least OHLC
                self.print(error("Insufficient price columns: {available_columns}"))
                return pd.DataFrame()

            price_data = data[available_columns].copy()

            # Ensure numeric data
            for col in price_data.columns:
                price_data[col] = pd.to_numeric(price_data[col], errors="coerce")

            # Remove rows with NaN values
            return price_data.dropna()

        except Exception:
            self.print(error("Error extracting price data: {e}"))
            return pd.DataFrame()

    def _extract_volume_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume data from dataset."""
        try:
            if "volume" in data.columns:
                volume_data = data[["volume"]].copy()
                volume_data["volume"] = pd.to_numeric(
                    volume_data["volume"],
                    errors="coerce",
                )
                return volume_data.dropna()
            # Create synthetic volume data if not available
            return pd.DataFrame(
                {"volume": np.random.uniform(1000, 10000, len(data))},
                index=data.index,
            )

        except Exception:
            self.print(error("Error extracting volume data: {e}"))
            return pd.DataFrame()

    async def _save_batch_results(
        self,
        batch_data: pd.DataFrame,
        wavelet_features: dict[str, Any],
        batch_idx: int,
        total_batches: int,
    ) -> bool:
        """Save batch results to cache."""
        try:
            # Generate cache key for batch
            cache_key = f"batch_{batch_idx:04d}_of_{total_batches:04d}"

            # Save to cache
            metadata = {
                "batch_idx": batch_idx,
                "total_batches": total_batches,
                "data_shape": batch_data.shape,
                "feature_count": len(wavelet_features),
                "timestamp": time.time(),
            }

            cache_success = self.wavelet_cache.save_to_cache(
                cache_key,
                wavelet_features,
                metadata,
            )

            if cache_success:
                self.logger.debug(f"💾 Cached batch {batch_idx + 1}/{total_batches}")
            else:
                self.logger.warning(
                    f"⚠️ Failed to cache batch {batch_idx + 1}/{total_batches}",
                )

            return cache_success

        except Exception:
            self.print(error("Error saving batch results: {e}"))
            return False

    async def precompute_multiple_datasets(
        self,
        dataset_configs: list[dict[str, Any]],
    ) -> bool:
        """
        Pre-compute wavelet features for multiple datasets.

        Args:
            dataset_configs: List of dataset configurations

        Returns:
            True if all successful, False otherwise
        """
        try:
            self.logger.info(
                f"🚀 Starting pre-computation for {len(dataset_configs)} datasets",
            )

            success_count = 0
            total_count = len(dataset_configs)

            for i, config in enumerate(dataset_configs):
                self.logger.info(
                    f"📊 Processing dataset {i + 1}/{total_count}: {config.get('data_path', 'Unknown')}",
                )

                success = await self.precompute_dataset(
                    data_path=config["data_path"],
                    output_path=config.get("output_path"),
                    symbol=config.get("symbol"),
                    start_date=config.get("start_date"),
                    end_date=config.get("end_date"),
                )

                if success:
                    success_count += 1
                else:
                    self.print(failed("❌ Failed to process dataset {i + 1}"))

            self.logger.info(
                f"✅ Completed pre-computation: {success_count}/{total_count} datasets successful",
            )
            return success_count == total_count

        except Exception:
            self.print(error("Error in multiple dataset pre-computation: {e}"))
            return False

    def get_precomputation_stats(self) -> dict[str, Any]:
        """Get pre-computation statistics."""
        try:
            cache_stats = (
                self.wavelet_cache.get_cache_stats() if self.wavelet_cache else {}
            )

            return {
                "precomputation_config": {
                    "batch_size": self.batch_size,
                    "enable_batch_processing": self.enable_batch_processing,
                    "enable_progress_tracking": self.enable_progress_tracking,
                    "enable_parallel_processing": self.enable_parallel_processing,
                },
                "cache_stats": cache_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.print(error("Error getting pre-computation stats: {e}"))
            return {"error": str(e)}

    def clear_all_cache(self) -> bool:
        """Clear all cached wavelet features."""
        try:
            if self.wavelet_cache:
                return self.wavelet_cache.clear_cache()
            return False

        except Exception:
            self.print(error("Error clearing cache: {e}"))
            return False


async def main():
    """Main function for pre-computation script."""
    try:
        # Configuration
        config = {
            "wavelet_cache": {
                "cache_enabled": True,
                "cache_dir": "data/wavelet_cache",
                "cache_format": "parquet",
                "compression": "snappy",
                "cache_expiry_days": 30,
            },
            "wavelet_precompute": {
                "enable_batch_processing": True,
                "batch_size": 10000,
                "enable_progress_tracking": True,
                "enable_parallel_processing": False,
                "max_workers": 4,
            },
            "vectorized_advanced_features": {
                "enable_wavelet_transforms": True,
                "enable_volatility_modeling": True,
                "enable_correlation_analysis": True,
                "enable_momentum_analysis": True,
                "enable_liquidity_analysis": True,
                "enable_candlestick_patterns": True,
                "enable_sr_distance": True,
                "enable_multi_timeframe": True,
                "enable_meta_labeling": True,
            },
        }

        # Initialize pre-computer
        precomputer = WaveletFeaturePrecomputer(config)
        await precomputer.initialize()

        # Example dataset configurations
        dataset_configs = [
            {
                "data_path": "data/price_data/ETHUSDT_1m.parquet",
                "symbol": "ETHUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
            {
                "data_path": "data/price_data/BTCUSDT_1m.parquet",
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
        ]

        # Pre-compute features
        success = await precomputer.precompute_multiple_datasets(dataset_configs)

        if success:
            print("✅ Pre-computation completed successfully!")

            # Print statistics
            stats = precomputer.get_precomputation_stats()
            print(f"📊 Cache Statistics: {stats}")
        else:
            print(failed("Pre-computation failed!"))

    except Exception:
        print(error("Error in main: {e}"))


if __name__ == "__main__":
    asyncio.run(main())
