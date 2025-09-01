# src/training/steps/backtesting_with_cached_features.py

"""Backtesting integration with cached wavelet features.
Demonstrates how to use pre-computed wavelet features for fast backtesting
without recalculating expensive wavelet transforms.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
    WaveletFeatureCache,
)
from src.utils.data_optimizer import ohlcv_columns
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class BacktestingWithCachedFeatures:
    """Backtesting system that leverages pre-computed wavelet features for fast execution."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("BacktestingWithCachedFeatures")

        # Backtesting configuration
        self.backtest_config = config.get("backtesting_with_cache", {})
        self.enable_feature_caching = self.backtest_config.get(
            "enable_feature_caching",
            True,
        )
        self.cache_lookup_timeout = self.backtest_config.get(
            "cache_lookup_timeout",
            5.0,
        )
        self.enable_performance_monitoring = self.backtest_config.get(
            "enable_performance_monitoring",
            True,
        )
        self.max_backtest_iterations = self.backtest_config.get(
            "max_backtest_iterations",
            1000,
        )

        # Initialize components
        self.feature_engineer: VectorizedAdvancedFeatureEngineering | None = None
        self.wavelet_cache: WaveletFeatureCache | None = None
        self.performance_stats: dict[str, Any] = {}

    @handle_errors(exceptions=(Exception,), default_return=False, context="backtesting.initialize")
    @handle_errors(exceptions=(Exception,), default_return={"error": "run_backtest failed"}, context="backtesting.run_backtest")
    async def run_backtest(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
        strategy_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run backtest using cached wavelet features.

        Args:
            price_data: Price data for backtesting
            volume_data: Volume data (optional)
            strategy_config: Strategy configuration (optional)

        Returns:
            Backtest results dictionary

        """
        start_time = time.time()
        self.logger.info(f"📊 Starting backtest with {len(price_data)} data points")

        # Get wavelet features with caching
        wavelet_features = await self._get_cached_wavelet_features(
            price_data,
            volume_data,
        )

        if not wavelet_features:
            self.logger.error("No wavelet features available for backtesting")
            return {"error": "No wavelet features available"}

        # Run strategy backtest
        backtest_results = await self._run_strategy_backtest(
            price_data,
            volume_data,
            wavelet_features,
            strategy_config or {},
        )

        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats["total_backtest_time"] += total_time
        self.performance_stats["iterations_completed"] += 1

        self.logger.info(f"✅ Backtest completed in {total_time:.2f}s")
        return backtest_results

    @handle_errors(exceptions=(Exception,), default_return={}, context="backtesting.get_cached_features")
    @handle_errors(exceptions=(Exception,), default_return={"error": "strategy failed"}, context="backtesting.run_strategy_backtest")
    async def _run_strategy_backtest(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None,
        wavelet_features: dict[str, Any],
        strategy_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run strategy backtest using wavelet features.

        Args:
            price_data: Price data
            volume_data: Volume data (optional)
            wavelet_features: Pre-computed wavelet features
            strategy_config: Strategy configuration (optional)

        Returns:
            Backtest results

        """
        # Combine all features
        all_features: dict[str, Any] = {
            **wavelet_features,
            "price": price_data["close"].values,
            "volume": volume_data["volume"].values
            if volume_data is not None
            else np.ones(len(price_data)),
        }

        # Simple strategy example using wavelet features
        results = await self._execute_simple_strategy(
            price_data,
            all_features,
            strategy_config or {},
        )

        return {
            "strategy_results": results,
            "feature_count": len(wavelet_features),
            "data_points": len(price_data),
            "timestamp": datetime.now().isoformat(),
        }

    @handle_errors(exceptions=(Exception,), default_return={"error": "simple strategy failed"}, context="backtesting.execute_simple_strategy")
    async def _execute_simple_strategy(
        self,
        price_data: pd.DataFrame,
        features: dict[str, Any],
        strategy_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute a simple trading strategy using wavelet features.

        Args:
            price_data: Price data
            features: Combined features including wavelet features
            strategy_config: Strategy configuration

        Returns:
            Strategy results

        """
        # Extract key wavelet features for strategy
        energy_features = {
            k: v for k, v in features.items() if isinstance(v, np.ndarray) and "energy" in k.lower()
        }
        entropy_features = {
            k: v for k, v in features.items() if isinstance(v, np.ndarray) and "entropy" in k.lower()
        }

        # Simple strategy: Buy when energy is high and entropy is low
        signals: list[int] = []
        positions: list[int] = []
        returns: list[float] = []

        for i in range(len(price_data)):
            # Calculate signal based on wavelet features
            signal = 0

            # Use energy features for trend following
            if energy_features:
                # Average last value across energy arrays if consistent
                try:
                    last_vals = [float(v[min(i, len(v) - 1)]) for v in energy_features.values() if len(v) > 0]
                    if last_vals:
                        if float(np.mean(last_vals)) > float(np.median(last_vals)):
                            signal = 1  # Buy signal
                except Exception:  # noqa: BLE001
                    pass

            # Use entropy features for mean reversion
            if entropy_features:
                try:
                    last_vals_e = [float(v[min(i, len(v) - 1)]) for v in entropy_features.values() if len(v) > 0]
                    if last_vals_e:
                        if float(np.mean(last_vals_e)) < float(np.median(last_vals_e)):
                            signal = -1  # Sell signal
                except Exception:  # noqa: BLE001
                    pass

            signals.append(signal)

            # Calculate position and returns
            if i > 0:
                price_return = (
                    float(price_data["close"].iloc[i]) - float(price_data["close"].iloc[i - 1])
                ) / max(float(price_data["close"].iloc[i - 1]), 1e-12)
                position_return = float(signal) * float(price_return)
                returns.append(position_return)
            else:
                returns.append(0.0)

            positions.append(signal)

        # Calculate performance metrics
        cumulative_returns = np.cumsum(returns)
        sharpe_ratio = (
            float(np.mean(returns)) / (float(np.std(returns)) + 1e-8) * np.sqrt(252)
        )  # Annualized
        max_drawdown = float(
            np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
        )

        return {
            "total_return": float(cumulative_returns[-1]) if len(cumulative_returns) > 0 else 0.0,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(np.sum(np.array(returns) > 0) / len(returns)) if returns else 0.0,
            "signal_count": int(len([s for s in signals if s != 0])),
            "final_position": int(positions[-1]) if positions else 0,
        }

    @handle_errors(exceptions=(Exception,), default_return=[], context="backtesting.run_multiple_backtests")
    async def run_multiple_backtests(
        self, backtest_configs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run multiple backtests with different configurations.

        Args:
            backtest_configs: List of backtest configurations

        Returns:
            List of backtest results

        """
        try:
            self.logger.info(f"🚀 Starting {len(backtest_configs)} backtests")

            results: list[dict[str, Any]] = []
            for i, config in enumerate(backtest_configs):
                self.logger.info(f"📊 Running backtest {i + 1}/{len(backtest_configs)}")

                # Load data
                price_data = await self._load_backtest_data(config.get("data_path"))
                volume_data = await self._load_volume_data(config.get("volume_path"))

                if price_data is None:
                    self.logger.error(f"Failed to load data for backtest {i + 1}")
                    continue

                # Run backtest
                result = await self.run_backtest(
                    price_data=price_data,
                    volume_data=volume_data,
                    strategy_config=config.get("strategy_config"),
                )

                result["backtest_id"] = i + 1
                result["config"] = config
                results.append(result)

            self.logger.info(f"✅ Completed {len(results)} backtests")
            return results

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"Error in multiple backtests: {e}")
            return []

    @handle_errors(exceptions=(Exception,), default_return=None, context="backtesting.load_backtest_data")
    async def _load_backtest_data(self, data_path: str) -> pd.DataFrame | None:
        """Load backtest data."""
        try:
            if not data_path:
                return None

            file_path = Path(data_path)
            if file_path.suffix.lower() == ".parquet":
                # Prefer dataset scan if a partitioned base is provided in path
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    # If data_path points to a directory, perform a dataset scan
                    if Path(data_path).is_dir():
                        return pdm.scan_dataset(
                            base_dir=data_path, columns=columns, to_pandas=True
                        )
                except Exception:
                    pass
                try:
                    from src.utils.logger import log_io_operation

                    with log_io_operation(
                        self.logger, "read_parquet", data_path, columns=ohlcv_columns()
                    ):
                        return pd.read_parquet(data_path, columns=ohlcv_columns())
                except Exception:
                    from src.utils.logger import log_io_operation

                    with log_io_operation(self.logger, "read_parquet", data_path):
                        return pd.read_parquet(data_path)
            if file_path.suffix.lower() == ".csv":
                from src.utils.logger import log_io_operation

                with log_io_operation(self.logger, "read_csv", data_path):
                    return pd.read_csv(data_path, parse_dates=True)
            self.logger.error(f"Unsupported file format: {file_path.suffix}")
            return None

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"Error loading backtest data: {e}")
            return None

    @handle_errors(exceptions=(Exception,), default_return=None, context="backtesting.load_volume_data")
    async def _load_volume_data(self, volume_path: str) -> pd.DataFrame | None:
        """Load volume data."""
        try:
            if not volume_path:
                return None

            return await self._load_backtest_data(volume_path)

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"Error loading volume data: {e}")
            return None

    def clear_cache(self) -> bool:
        """Clear wavelet cache."""
        try:
            if self.wavelet_cache:
                return bool(self.wavelet_cache.clear_cache())
            return False

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"Error clearing cache: {e}")
            return False


async def main() -> None:
    """Main function for backtesting with cached features."""
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
            "backtesting_with_cache": {
                "enable_feature_caching": True,
                "cache_lookup_timeout": 5.0,
                "enable_performance_monitoring": True,
                "max_backtest_iterations": 1000,
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
                "enable_meta_labeling": False,
            },
        }

        # Initialize backtesting system
        backtester = BacktestingWithCachedFeatures(config)
        await backtester.initialize()

        # Example backtest configurations
        backtest_configs = [
            {
                "data_path": "data/price_data/ETHUSDT_1m.parquet",
                "volume_path": "data/volume_data/ETHUSDT_1m.parquet",
                "strategy_config": {
                    "strategy_type": "wavelet_energy",
                    "parameters": {"energy_threshold": 0.5},
                },
            },
            {
                "data_path": "data/price_data/BTCUSDT_1m.parquet",
                "volume_path": "data/volume_data/BTCUSDT_1m.parquet",
                "strategy_config": {
                    "strategy_type": "wavelet_entropy",
                    "parameters": {"entropy_threshold": 0.3},
                },
            },
        ]

        # Run multiple backtests
        results = await backtester.run_multiple_backtests(backtest_configs)

        # Print results
        for i, result in enumerate(results, start=1):
            backtester.logger.info(f"Backtest {i} results summary: keys={list(result.keys())}")
        # Print performance stats
        backtester.logger.info(f"Performance stats: {backtester.get_performance_stats()}")

    except Exception as e:  # noqa: BLE001
        system_logger.exception(f"Backtesting main failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())