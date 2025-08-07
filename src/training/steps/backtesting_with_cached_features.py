# src/training/steps/backtesting_with_cached_features.py

"""
Backtesting integration with cached wavelet features.
Demonstrates how to use pre-computed wavelet features for fast backtesting
without recalculating expensive wavelet transforms.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta

from src.utils.logger import system_logger
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
    WaveletFeatureCache,
)


class BacktestingWithCachedFeatures:
    """
    Backtesting system that leverages pre-computed wavelet features for fast execution.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("BacktestingWithCachedFeatures")

        # Backtesting configuration
        self.backtest_config = config.get("backtesting_with_cache", {})
        self.enable_feature_caching = self.backtest_config.get("enable_feature_caching", True)
        self.cache_lookup_timeout = self.backtest_config.get("cache_lookup_timeout", 5.0)
        self.enable_performance_monitoring = self.backtest_config.get("enable_performance_monitoring", True)
        self.max_backtest_iterations = self.backtest_config.get("max_backtest_iterations", 1000)

        # Initialize components
        self.feature_engineer = None
        self.wavelet_cache = None
        self.performance_stats = {}

    async def initialize(self) -> bool:
        """Initialize the backtesting system."""
        try:
            self.logger.info("🚀 Initializing backtesting with cached features...")

            # Initialize feature engineering
            self.feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
            await self.feature_engineer.initialize()

            # Initialize cache
            self.wavelet_cache = WaveletFeatureCache(self.config)

            # Initialize performance monitoring
            self.performance_stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_feature_load_time": 0.0,
                "total_backtest_time": 0.0,
                "iterations_completed": 0,
            }

            self.logger.info("✅ Backtesting with cached features initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing backtesting system: {e}")
            return False

    async def run_backtest(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
        strategy_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run backtest using cached wavelet features.

        Args:
            price_data: Price data for backtesting
            volume_data: Volume data (optional)
            strategy_config: Strategy configuration (optional)

        Returns:
            Backtest results dictionary
        """
        try:
            start_time = time.time()
            self.logger.info(f"📊 Starting backtest with {len(price_data)} data points")

            # Get wavelet features with caching
            wavelet_features = await self._get_cached_wavelet_features(price_data, volume_data)

            if not wavelet_features:
                self.logger.warning("No wavelet features available for backtesting")
                return {"error": "No wavelet features available"}

            # Run strategy backtest
            backtest_results = await self._run_strategy_backtest(
                price_data, volume_data, wavelet_features, strategy_config
            )

            # Update performance stats
            total_time = time.time() - start_time
            self.performance_stats["total_backtest_time"] += total_time
            self.performance_stats["iterations_completed"] += 1

            self.logger.info(f"✅ Backtest completed in {total_time:.2f}s")
            return backtest_results

        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return {"error": str(e)}

    async def _get_cached_wavelet_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Get wavelet features with caching support.

        Args:
            price_data: Price data
            volume_data: Volume data (optional)

        Returns:
            Dictionary containing wavelet features
        """
        try:
            if not self.wavelet_cache:
                self.logger.warning("Wavelet cache not available, using direct computation")
                return await self.feature_engineer._get_wavelet_features_with_caching(
                    price_data, volume_data
                )

            # Generate cache key
            wavelet_config = self.feature_engineer.wavelet_analyzer.wavelet_config
            cache_key = self.wavelet_cache.generate_cache_key(
                price_data,
                wavelet_config,
                {"volume_data_shape": volume_data.shape if volume_data is not None else None}
            )

            # Check cache with timeout
            cache_start_time = time.time()
            if self.wavelet_cache.cache_exists(cache_key):
                self.logger.info(f"📦 Loading wavelet features from cache: {cache_key}")
                cached_features, metadata = self.wavelet_cache.load_from_cache(cache_key)
                
                cache_load_time = time.time() - cache_start_time
                self.performance_stats["cache_hits"] += 1
                self.performance_stats["total_feature_load_time"] += cache_load_time
                
                self.logger.info(f"⚡ Cache load time: {cache_load_time:.3f}s")
                return cached_features

            # Cache miss - compute features
            self.logger.info(f"🔧 Computing wavelet features (cache miss): {cache_key}")
            wavelet_features = await self.feature_engineer._get_wavelet_features_with_caching(
                price_data, volume_data
            )

            # Save to cache
            metadata = {
                "data_shape": price_data.shape,
                "volume_data_shape": volume_data.shape if volume_data is not None else None,
                "computation_time": time.time(),
                "backtest_generated": True,
            }
            
            cache_success = self.wavelet_cache.save_to_cache(cache_key, wavelet_features, metadata)
            if cache_success:
                self.logger.info(f"💾 Cached wavelet features for future backtests: {cache_key}")

            self.performance_stats["cache_misses"] += 1
            return wavelet_features

        except Exception as e:
            self.logger.error(f"Error getting cached wavelet features: {e}")
            return {}

    async def _run_strategy_backtest(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None,
        wavelet_features: dict[str, Any],
        strategy_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Run strategy backtest using wavelet features.

        Args:
            price_data: Price data
            volume_data: Volume data (optional)
            wavelet_features: Pre-computed wavelet features
            strategy_config: Strategy configuration (optional)

        Returns:
            Backtest results
        """
        try:
            # Combine all features
            all_features = {
                **wavelet_features,
                "price": price_data["close"].values,
                "volume": volume_data["volume"].values if volume_data is not None else np.ones(len(price_data)),
            }

            # Simple strategy example using wavelet features
            results = await self._execute_simple_strategy(price_data, all_features, strategy_config)

            return {
                "strategy_results": results,
                "feature_count": len(wavelet_features),
                "data_points": len(price_data),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error running strategy backtest: {e}")
            return {"error": str(e)}

    async def _execute_simple_strategy(
        self,
        price_data: pd.DataFrame,
        features: dict[str, Any],
        strategy_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Execute a simple trading strategy using wavelet features.

        Args:
            price_data: Price data
            features: Combined features including wavelet features
            strategy_config: Strategy configuration

        Returns:
            Strategy results
        """
        try:
            # Extract key wavelet features for strategy
            energy_features = {k: v for k, v in features.items() if "energy" in k.lower()}
            entropy_features = {k: v for k, v in features.items() if "entropy" in k.lower()}
            
            # Simple strategy: Buy when energy is high and entropy is low
            signals = []
            positions = []
            returns = []
            
            for i in range(len(price_data)):
                # Calculate signal based on wavelet features
                signal = 0
                
                # Use energy features for trend following
                if energy_features:
                    avg_energy = np.mean(list(energy_features.values()))
                    if avg_energy > np.median(list(energy_features.values())):
                        signal = 1  # Buy signal
                
                # Use entropy features for mean reversion
                if entropy_features:
                    avg_entropy = np.mean(list(entropy_features.values()))
                    if avg_entropy < np.median(list(entropy_features.values())):
                        signal = -1  # Sell signal
                
                signals.append(signal)
                
                # Calculate position and returns
                if i > 0:
                    price_return = (price_data["close"].iloc[i] - price_data["close"].iloc[i-1]) / price_data["close"].iloc[i-1]
                    position_return = signal * price_return
                    returns.append(position_return)
                else:
                    returns.append(0.0)
                
                positions.append(signal)
            
            # Calculate performance metrics
            cumulative_returns = np.cumsum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
            max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
            
            return {
                "total_return": cumulative_returns[-1],
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": np.sum(np.array(returns) > 0) / len(returns),
                "signal_count": len([s for s in signals if s != 0]),
                "final_position": positions[-1],
            }

        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            return {"error": str(e)}

    async def run_multiple_backtests(
        self,
        backtest_configs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run multiple backtests with different configurations.

        Args:
            backtest_configs: List of backtest configurations

        Returns:
            List of backtest results
        """
        try:
            self.logger.info(f"🚀 Starting {len(backtest_configs)} backtests")

            results = []
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

        except Exception as e:
            self.logger.error(f"Error in multiple backtests: {e}")
            return []

    async def _load_backtest_data(self, data_path: str) -> pd.DataFrame | None:
        """Load backtest data."""
        try:
            if not data_path:
                return None

            file_path = Path(data_path)
            if file_path.suffix.lower() == ".parquet":
                return pd.read_parquet(data_path)
            elif file_path.suffix.lower() == ".csv":
                return pd.read_csv(data_path, parse_dates=True)
            else:
                self.logger.error(f"Unsupported file format: {file_path.suffix}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading backtest data: {e}")
            return None

    async def _load_volume_data(self, volume_path: str) -> pd.DataFrame | None:
        """Load volume data."""
        try:
            if not volume_path:
                return None

            return await self._load_backtest_data(volume_path)

        except Exception as e:
            self.logger.error(f"Error loading volume data: {e}")
            return None

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        try:
            stats = self.performance_stats.copy()
            
            if stats["iterations_completed"] > 0:
                stats["avg_backtest_time"] = stats["total_backtest_time"] / stats["iterations_completed"]
                stats["avg_feature_load_time"] = stats["total_feature_load_time"] / stats["cache_hits"] if stats["cache_hits"] > 0 else 0
                stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"]) if (stats["cache_hits"] + stats["cache_misses"]) > 0 else 0
            
            stats["timestamp"] = datetime.now().isoformat()
            return stats

        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}

    def clear_cache(self) -> bool:
        """Clear wavelet cache."""
        try:
            if self.wavelet_cache:
                return self.wavelet_cache.clear_cache()
            return False

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False


async def main():
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
                "enable_meta_labeling": True,
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
                    "parameters": {"energy_threshold": 0.5}
                }
            },
            {
                "data_path": "data/price_data/BTCUSDT_1m.parquet",
                "volume_path": "data/volume_data/BTCUSDT_1m.parquet",
                "strategy_config": {
                    "strategy_type": "wavelet_entropy",
                    "parameters": {"entropy_threshold": 0.3}
                }
            },
        ]

        # Run multiple backtests
        results = await backtester.run_multiple_backtests(backtest_configs)

        # Print results
        print("📊 Backtest Results:")
        for i, result in enumerate(results):
            print(f"  Backtest {i + 1}:")
            print(f"    Total Return: {result.get('strategy_results', {}).get('total_return', 0):.4f}")
            print(f"    Sharpe Ratio: {result.get('strategy_results', {}).get('sharpe_ratio', 0):.4f}")
            print(f"    Max Drawdown: {result.get('strategy_results', {}).get('max_drawdown', 0):.4f}")
            print(f"    Feature Count: {result.get('feature_count', 0)}")

        # Print performance stats
        stats = backtester.get_performance_stats()
        print(f"\n📈 Performance Statistics:")
        print(f"  Cache Hit Rate: {stats.get('cache_hit_rate', 0):.2%}")
        print(f"  Avg Backtest Time: {stats.get('avg_backtest_time', 0):.3f}s")
        print(f"  Avg Feature Load Time: {stats.get('avg_feature_load_time', 0):.3f}s")
        print(f"  Iterations Completed: {stats.get('iterations_completed', 0)}")

    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())