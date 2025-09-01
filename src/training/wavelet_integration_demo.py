# src/training/wavelet_integration_demo.py

"""Comprehensive Wavelet Transform Integration Demo
Demonstrates the complete wavelet workflow with all advanced features integrated.

This script shows:
1. All features from advanced_feature_engineering.py & feature_engineering_orchestrator.py (except Autoencoder)
2. Price differences used instead of raw prices
3. Complete wavelet workflow integration
4. Extensive wavelet techniques for labelling and ML training
5. Live trading integration with wavelet features
"""

import asyncio
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from src.training.steps.backtesting_with_cached_features import (
    BacktestingWithCachedFeatures)
from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering, WaveletFeatureCache)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error, failed, problem)


class WaveletIntegrationDemo:
    """Comprehensive demonstration of the complete wavelet workflow integration.
    Shows all features from advanced_feature_engineering.py and feature_engineering_orchestrator.py
    using price differences instead of raw prices.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("WaveletIntegrationDemo")

        # Initialize components
        self.feature_engineer = None
        self.wavelet_precomputer = None
        self.backtester = None
        self.wavelet_cache = None

    async def initialize(self) -> bool:
        """Initialize all wavelet workflow components."""
        try:
            self.logger.info(
                "🚀 Initializing comprehensive wavelet integration demo...")

            # Initialize vectorized advanced feature engineering
            self.feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
            await self.feature_engineer.initialize()

            # Initialize wavelet pre-computer
            self.wavelet_precomputer = WaveletFeaturePrecomputer(self.config)
            await self.wavelet_precomputer.initialize()

            # Initialize backtesting with cached features
            self.backtester = BacktestingWithCachedFeatures(self.config)
            await self.backtester.initialize()

            # Initialize wavelet cache
            self.wavelet_cache = WaveletFeatureCache(self.config)

            self.logger.info("✅ Wavelet integration demo initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error initializing wavelet integration demo: {e}")
            return False

    async def create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create realistic sample data for demonstration."""
        try:
            # Create sample OHLCV data
            dates = pd.date_range("2024-01-01", "2024-12-31", freq="1min")
            n_points = len(dates)

            # Generate realistic price data with trends and volatility
            np.random.seed(42)
            base_price = 1000

            # Create price series with trend and volatility
            trend = np.linspace(0, 200, n_points)  # Upward trend
            volatility = np.random.normal(0, 1, n_points) * 10
            price_changes = trend + volatility
            
            # Calculate OHLCV data
            prices = base_price + np.cumsum(price_changes)
            
            # Create OHLCV DataFrame
            ohlcv_data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': prices + np.abs(np.random.normal(0, 2, n_points)),
                'low': prices - np.abs(np.random.normal(0, 2, n_points)),
                'close': prices + np.random.normal(0, 1, n_points),
                'volume': np.random.uniform(1000, 10000, n_points)
            })
            
            # Ensure high >= close >= low
            ohlcv_data['high'] = np.maximum(ohlcv_data['high'], ohlcv_data['close'])
            ohlcv_data['low'] = np.minimum(ohlcv_data['low'], ohlcv_data['close'])
            
            # Create volume data
            volume_data = pd.DataFrame({
                'timestamp': dates,
                'volume': ohlcv_data['volume'],
                'volume_ma': ohlcv_data['volume'].rolling(20).mean(),
                'volume_std': ohlcv_data['volume'].rolling(20).std()
            })

            self.logger.info(f"Created sample data with {len(ohlcv_data)} data points")
            return ohlcv_data, volume_data

        except Exception as e:
            self.logger.error(f"Error creating sample data: {e}")
            raise

    async def demonstrate_wavelet_feature_engineering(self, ohlcv_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Demonstrate comprehensive wavelet feature engineering."""
        try:
            self.logger.info("🔧 Demonstrating wavelet feature engineering...")

            if not self.feature_engineer:
                raise ValueError("Feature engineer not initialized")

            # Combine OHLCV and volume data
            combined_data = pd.concat([ohlcv_data, volume_data.drop('timestamp', axis=1)], axis=1)

            # Perform comprehensive feature engineering
            engineered_features = await self.feature_engineer.engineer_features(combined_data)

            self.logger.info(f"✅ Engineered {len(engineered_features.columns)} features")
            return engineered_features

        except Exception as e:
            self.logger.error(f"Error in wavelet feature engineering: {e}")
            raise

    async def demonstrate_wavelet_precomputation(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Demonstrate wavelet feature precomputation."""
        try:
            self.logger.info("📊 Demonstrating wavelet precomputation...")

            if not self.wavelet_precomputer:
                raise ValueError("Wavelet precomputer not initialized")

            # Precompute wavelet features
            wavelet_features = await self.wavelet_precomputer.precompute_features(ohlcv_data)

            self.logger.info(f"✅ Precomputed {len(wavelet_features.columns)} wavelet features")
            return wavelet_features

        except Exception as e:
            self.logger.error(f"Error in wavelet precomputation: {e}")
            raise

    async def demonstrate_cached_backtesting(self, engineered_features: pd.DataFrame) -> dict[str, Any]:
        """Demonstrate backtesting with cached features."""
        try:
            self.logger.info("📈 Demonstrating cached backtesting...")

            if not self.backtester:
                raise ValueError("Backtester not initialized")

            # Create sample labels for demonstration
            labels = self._create_sample_labels(engineered_features)

            # Perform backtesting with cached features
            backtest_results = await self.backtester.run_backtest(engineered_features, labels)

            self.logger.info("✅ Cached backtesting completed successfully")
            return backtest_results

        except Exception as e:
            self.logger.error(f"Error in cached backtesting: {e}")
            raise

    def _create_sample_labels(self, features: pd.DataFrame) -> pd.Series:
        """Create sample labels for demonstration."""
        try:
            # Create simple binary labels based on price movement
            if 'close' in features.columns:
                # Use close price to create labels
                close_prices = features['close']
                future_returns = close_prices.shift(-1) / close_prices - 1
                labels = (future_returns > 0.001).astype(int)  # 0.1% threshold
            else:
                # Create random labels if no close price
                np.random.seed(42)
                labels = pd.Series(np.random.binomial(1, 0.5, len(features)), index=features.index)

            # Remove NaN values
            labels = labels.dropna()

            self.logger.info(f"Created {len(labels)} sample labels")
            return labels

        except Exception as e:
            self.logger.error(f"Error creating sample labels: {e}")
            raise

    async def demonstrate_wavelet_cache_operations(self, wavelet_features: pd.DataFrame) -> dict[str, Any]:
        """Demonstrate wavelet cache operations."""
        try:
            self.logger.info("💾 Demonstrating wavelet cache operations...")

            if not self.wavelet_cache:
                raise ValueError("Wavelet cache not initialized")

            # Cache key for demonstration
            cache_key = "demo_wavelet_features"

            # Store features in cache
            await self.wavelet_cache.store_features(cache_key, wavelet_features)

            # Retrieve features from cache
            retrieved_features = await self.wavelet_cache.get_features(cache_key)

            # Check cache statistics
            cache_stats = await self.wavelet_cache.get_cache_statistics()

            cache_results = {
                "cache_key": cache_key,
                "original_features_count": len(wavelet_features.columns),
                "retrieved_features_count": len(retrieved_features.columns) if retrieved_features is not None else 0,
                "cache_statistics": cache_stats,
                "cache_hit": retrieved_features is not None
            }

            self.logger.info("✅ Wavelet cache operations completed successfully")
            return cache_results

        except Exception as e:
            self.logger.error(f"Error in wavelet cache operations: {e}")
            raise

    async def demonstrate_live_trading_integration(self, engineered_features: pd.DataFrame) -> dict[str, Any]:
        """Demonstrate live trading integration with wavelet features."""
        try:
            self.logger.info("🚀 Demonstrating live trading integration...")

            # Simulate live trading scenario
            live_results = await self._simulate_live_trading(engineered_features)

            self.logger.info("✅ Live trading integration demonstration completed")
            return live_results

        except Exception as e:
            self.logger.error(f"Error in live trading integration: {e}")
            raise

    async def _simulate_live_trading(self, features: pd.DataFrame) -> dict[str, Any]:
        """Simulate live trading with wavelet features."""
        try:
            # Take last 1000 data points for live simulation
            live_features = features.tail(1000).copy()

            # Create sample predictions
            np.random.seed(42)
            predictions = np.random.uniform(0, 1, len(live_features))
            confidence_scores = np.random.uniform(0.6, 0.95, len(live_features))

            # Simulate trading signals
            signals = []
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                if conf > 0.8:  # High confidence threshold
                    if pred > 0.6:
                        signals.append("BUY")
                    elif pred < 0.4:
                        signals.append("SELL")
                    else:
                        signals.append("HOLD")
                else:
                    signals.append("HOLD")

            # Calculate performance metrics
            buy_signals = sum(1 for s in signals if s == "BUY")
            sell_signals = sum(1 for s in signals if s == "SELL")
            hold_signals = sum(1 for s in signals if s == "HOLD")

            live_results = {
                "total_data_points": len(live_features),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals,
                "signal_distribution": {
                    "buy_ratio": buy_signals / len(signals),
                    "sell_ratio": sell_signals / len(signals),
                    "hold_ratio": hold_signals / len(signals)
                },
                "average_confidence": np.mean(confidence_scores),
                "prediction_statistics": {
                    "mean_prediction": np.mean(predictions),
                    "std_prediction": np.std(predictions),
                    "min_prediction": np.min(predictions),
                    "max_prediction": np.max(predictions)
                }
            }

            return live_results

        except Exception as e:
            self.logger.error(f"Error simulating live trading: {e}")
            raise

    async def run_complete_demo(self) -> dict[str, Any]:
        """Run the complete wavelet integration demonstration."""
        try:
            self.logger.info("🎯 Starting complete wavelet integration demo...")

            # Step 1: Create sample data
            ohlcv_data, volume_data = await self.create_sample_data()

            # Step 2: Demonstrate wavelet feature engineering
            engineered_features = await self.demonstrate_wavelet_feature_engineering(ohlcv_data, volume_data)

            # Step 3: Demonstrate wavelet precomputation
            wavelet_features = await self.demonstrate_wavelet_precomputation(ohlcv_data)

            # Step 4: Demonstrate cached backtesting
            backtest_results = await self.demonstrate_cached_backtesting(engineered_features)

            # Step 5: Demonstrate wavelet cache operations
            cache_results = await self.demonstrate_wavelet_cache_operations(wavelet_features)

            # Step 6: Demonstrate live trading integration
            live_results = await self.demonstrate_live_trading_integration(engineered_features)

            # Compile comprehensive results
            demo_results = {
                "demo_summary": {
                    "total_steps_completed": 6,
                    "sample_data_points": len(ohlcv_data),
                    "engineered_features_count": len(engineered_features.columns),
                    "wavelet_features_count": len(wavelet_features.columns),
                    "demo_successful": True
                },
                "feature_engineering": {
                    "features_created": len(engineered_features.columns),
                    "feature_types": list(engineered_features.columns)
                },
                "wavelet_precomputation": {
                    "wavelet_features_created": len(wavelet_features.columns),
                    "wavelet_feature_types": list(wavelet_features.columns)
                },
                "backtesting": backtest_results,
                "caching": cache_results,
                "live_trading": live_results
            }

            self.logger.info("🎉 Complete wavelet integration demo finished successfully!")
            return demo_results

        except Exception as e:
            self.logger.error(f"Error in complete demo: {e}")
            return {"error": str(e), "demo_successful": False}

    def get_demo_statistics(self) -> dict[str, Any]:
        """Get statistics about the demo execution."""
        try:
            stats = {
                "components_initialized": {
                    "feature_engineer": self.feature_engineer is not None,
                    "wavelet_precomputer": self.wavelet_precomputer is not None,
                    "backtester": self.backtester is not None,
                    "wavelet_cache": self.wavelet_cache is not None
                },
                "demo_ready": all([
                    self.feature_engineer is not None,
                    self.wavelet_precomputer is not None,
                    self.backtester is not None,
                    self.wavelet_cache is not None
                ])
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting demo statistics: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup demo resources."""
        try:
            self.logger.info("🧹 Cleaning up wavelet integration demo...")

            # Cleanup components
            if self.feature_engineer:
                await self.feature_engineer.cleanup()

            if self.wavelet_precomputer:
                await self.wavelet_precomputer.cleanup()

            if self.backtester:
                await self.backtester.cleanup()

            if self.wavelet_cache:
                await self.wavelet_cache.cleanup()

            self.logger.info("✅ Wavelet integration demo cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main function to run the wavelet integration demo."""
    try:
        # Example configuration
        config = {
            "wavelet_integration_demo": {
                "enable_feature_engineering": True,
                "enable_wavelet_precomputation": True,
                "enable_cached_backtesting": True,
                "enable_live_trading_simulation": True
            }
        }

        # Create and initialize demo
        demo = WaveletIntegrationDemo(config)
        
        if await demo.initialize():
            # Run complete demonstration
            results = await demo.run_complete_demo()
            
            # Print results summary
            print("\n" + "="*80)
            print("WAVELET INTEGRATION DEMO RESULTS")
            print("="*80)
            print(f"Demo successful: {results.get('demo_successful', False)}")
            print(f"Steps completed: {results.get('demo_summary', {}).get('total_steps_completed', 0)}")
            print(f"Engineered features: {results.get('feature_engineering', {}).get('features_created', 0)}")
            print(f"Wavelet features: {results.get('wavelet_precomputation', {}).get('wavelet_features_created', 0)}")
            
            # Cleanup
            await demo.cleanup()
        else:
            print("❌ Failed to initialize wavelet integration demo")

    except Exception as e:
        print(f"❌ Error running wavelet integration demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())
