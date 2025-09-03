#!/usr/bin/env python3
"""
Example: Enhanced Multi-Timeframe Optimization

This example demonstrates how to use the enhanced multi-timeframe optimizer
with dynamic lookback periods from the matrix optimization system.

Features:
- Multi-timeframe feature generation
- Regime-specific optimization
- Cross-timeframe analysis
- Quality threshold filtering
- Performance optimization
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import configuration
from src.config.enhanced_multi_timeframe_config import (
    get_enhanced_multi_timeframe_config,
    get_performance_config,
    get_quality_validation_config,
    get_regime_specific_config,
)

# Import the comprehensive feature optimizer
from src.training.comprehensive_feature_optimizer import (
    ComprehensiveFeatureConfig,
    ComprehensiveFeatureOptimizer,
)

# Import the enhanced multi-timeframe optimizer
from src.training.enhanced_multi_timeframe_optimizer import (
    EnhancedMultiTimeframeOptimizer,
    OptimizedTimeframeConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class MultiTimeframeOptimizationExample:
    """
    Example class demonstrating multi-timeframe optimization usage.
    """

    def __init__(self):
        """Initialize the example with configuration."""
        self.config = get_enhanced_multi_timeframe_config()
        self.regime_config = get_regime_specific_config()
        self.quality_config = get_quality_validation_config()
        self.performance_config = get_performance_config()

        # Initialize optimizers
        self.mtf_optimizer = None
        self.comprehensive_optimizer = None

    def setup_optimizers(self, matrix_optimization_results: dict[str, Any] | None = None):
        """Setup the multi-timeframe and comprehensive optimizers."""
        logger.info("🔧 Setting up optimizers...")

        # Initialize multi-timeframe optimizer
        mtf_config = OptimizedTimeframeConfig(
            base_timeframes=self.config["enhanced_multi_timeframe_optimization"]["base_timeframes"],
            cross_timeframe_enabled=self.config["enhanced_multi_timeframe_optimization"]["cross_timeframe_enabled"],
            regime_specific=self.config["enhanced_multi_timeframe_optimization"]["regime_specific"],
            quality_thresholds=self.config["enhanced_multi_timeframe_optimization"]["quality_thresholds"],
        )

        self.mtf_optimizer = EnhancedMultiTimeframeOptimizer(
            config=mtf_config,
            matrix_optimization_results=matrix_optimization_results,
        )

        # Initialize comprehensive feature optimizer
        comprehensive_config = ComprehensiveFeatureConfig(
            interaction_features=True,
            difference_acceleration_features=True,
            cross_timeframe_features=True,
            microstructure_features=True,
            volatility_features=True,
            momentum_features=True,
            liquidity_features=True,
            candlestick_patterns=True,
            ohlcv_price_features=True,
            max_interaction_pairs=50,
            max_difference_features=100,
            max_cross_timeframe_pairs=30,
        )

        self.comprehensive_optimizer = ComprehensiveFeatureOptimizer(
            config=comprehensive_config,
            matrix_optimization_results=matrix_optimization_results,
        )

        logger.info("✅ Optimizers setup completed")

    def generate_sample_data(self, symbol: str = "BTCUSDT", days: int = 30) -> dict[str, pd.DataFrame]:
        """Generate sample OHLCV data for multiple timeframes."""
        logger.info(f"📊 Generating sample data for {symbol} over {days} days...")

        # Generate base 1-minute data
        base_timestamps = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=days),
            end=pd.Timestamp.now(),
            freq="1T",
        )

        # Create realistic price movements
        np.random.seed(42)  # For reproducible results
        n_points = len(base_timestamps)

        # Generate price series with trend and volatility
        base_price = 50000  # Starting price
        trend = np.linspace(0, 0.1, n_points)  # 10% upward trend
        volatility = 0.02  # 2% daily volatility

        # Generate OHLCV data
        returns = np.random.normal(trend, volatility, n_points)
        prices = base_price * np.exp(np.cumsum(returns))

        # Create OHLCV structure
        data_1m = pd.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.001, n_points)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
            "close": prices,
            "volume": np.random.lognormal(10, 1, n_points) * 1000,
        }, index=base_timestamps)

        # Ensure OHLCV consistency
        data_1m["high"] = data_1m[["open", "high", "close"]].max(axis=1)
        data_1m["low"] = data_1m[["open", "low", "close"]].min(axis=1)

        # Resample to different timeframes
        data_5m = self._resample_ohlcv(data_1m, "5T")
        data_15m = self._resample_ohlcv(data_1m, "15T")
        data_30m = self._resample_ohlcv(data_1m, "30T")
        data_1h = self._resample_ohlcv(data_1m, "1H")

        sample_data = {
            "1m": data_1m,
            "5m": data_5m,
            "15m": data_15m,
            "30m": data_30m,
            "1h": data_1h,
        }

        logger.info(f"✅ Generated sample data for {len(sample_data)} timeframes")
        for timeframe, data in sample_data.items():
            logger.info(f"   {timeframe}: {data.shape[0]} records, {data.shape[1]} columns")

        return sample_data

    def _resample_ohlcv(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV data to a different frequency."""
        return data.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def generate_regime_labels(self, data: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
        """Generate sample regime labels using volatility-based clustering."""
        logger.info(f"🎯 Generating regime labels with {n_regimes} regimes...")

        # Calculate rolling volatility
        returns = data["close"].pct_change().dropna()
        volatility = returns.rolling(window=20).std()

        # Use volatility percentiles to create regimes
        vol_percentiles = volatility.quantile([0.33, 0.67])

        regime_labels = pd.Series(index=volatility.index, dtype=int)
        regime_labels[volatility <= vol_percentiles.iloc[0]] = 0  # Low volatility
        regime_labels[(volatility > vol_percentiles.iloc[0]) & (volatility <= vol_percentiles.iloc[1])] = 1  # Medium volatility
        regime_labels[volatility > vol_percentiles.iloc[1]] = 2  # High volatility

        # Forward fill any remaining NaN values
        regime_labels = regime_labels.fillna(method="ffill").fillna(0)

        regime_counts = regime_labels.value_counts().sort_index()
        logger.info(f"✅ Generated regime labels: {dict(regime_counts)}")

        return regime_labels

    async def run_multi_timeframe_optimization(self, data: dict[str, pd.DataFrame], regime_labels: pd.Series):
        """Run the multi-timeframe optimization process."""
        logger.info("🚀 Starting multi-timeframe optimization...")

        if self.mtf_optimizer is None:
            msg = "Optimizers not initialized. Call setup_optimizers() first."
            raise ValueError(msg)

        # Get base timeframe data (1m)
        base_data = data["1m"]

        # Generate features for each timeframe
        timeframe_features = {}

        for timeframe, timeframe_data in data.items():
            logger.info(f"📈 Processing {timeframe} timeframe...")

            try:
                # Generate features for this timeframe
                features = await self.mtf_optimizer.generate_timeframe_features(
                    data=timeframe_data,
                    timeframe=timeframe,
                    base_timeframe_data=base_data,
                )

                if features:
                    timeframe_features[timeframe] = features
                    logger.info(f"✅ Generated {len(features)} features for {timeframe}")
                else:
                    logger.warning(f"⚠️ No features generated for {timeframe}")

            except Exception as e:
                logger.exception(f"❌ Error processing {timeframe}: {e}")
                continue

        # Generate cross-timeframe features
        logger.info("🔗 Generating cross-timeframe features...")
        cross_timeframe_features = await self.mtf_optimizer.generate_cross_timeframe_features(
            data=data,
            regime_labels=regime_labels,
        )

        if cross_timeframe_features:
            timeframe_features["cross_timeframe"] = cross_timeframe_features
            logger.info(f"✅ Generated {len(cross_timeframe_features)} cross-timeframe features")

        return timeframe_features

    async def run_comprehensive_optimization(self, data: pd.DataFrame, target: pd.Series, regime_labels: pd.Series):
        """Run the comprehensive feature optimization process."""
        logger.info("🎯 Starting comprehensive feature optimization...")

        if self.comprehensive_optimizer is None:
            msg = "Comprehensive optimizer not initialized. Call setup_optimizers() first."
            raise ValueError(msg)

        try:
            # Generate comprehensive features
            features = await self.comprehensive_optimizer.generate_comprehensive_features(
                data=data,
                target=target,
                regime_labels=regime_labels,
            )

            logger.info(f"✅ Generated {len(features)} comprehensive features")
            return features

        except Exception as e:
            logger.exception(f"❌ Error in comprehensive optimization: {e}")
            return {}

    def analyze_results(self, timeframe_features: dict[str, Any], comprehensive_features: dict[str, Any]):
        """Analyze and summarize the optimization results."""
        logger.info("📊 Analyzing optimization results...")

        # Summary statistics
        total_features = 0
        feature_types = {}

        # Analyze timeframe features
        for timeframe, features in timeframe_features.items():
            if isinstance(features, dict):
                n_features = len(features)
                total_features += n_features
                feature_types[timeframe] = n_features

                logger.info(f"   {timeframe}: {n_features} features")

                # Show sample feature names
                if n_features > 0:
                    sample_names = list(features.keys())[:5]
                    logger.info(f"     Sample features: {', '.join(sample_names)}")

        # Analyze comprehensive features
        if comprehensive_features:
            comp_features = len(comprehensive_features)
            total_features += comp_features
            feature_types["comprehensive"] = comp_features

            logger.info(f"   Comprehensive: {comp_features} features")

            # Show feature categories
            if "metadata" in comprehensive_features:
                metadata = comprehensive_features["metadata"]
                if "feature_types" in metadata:
                    logger.info(f"     Feature types: {', '.join(metadata['feature_types'])}")

        logger.info(f"✅ Total features generated: {total_features}")
        logger.info(f"   Feature distribution: {feature_types}")

        return {
            "total_features": total_features,
            "feature_types": feature_types,
            "timeframe_features": timeframe_features,
            "comprehensive_features": comprehensive_features,
        }

    def save_results(self, results: dict[str, Any], output_dir: str = "results"):
        """Save optimization results to files."""
        logger.info(f"💾 Saving results to {output_dir}...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save timeframe features
        for timeframe, features in results["timeframe_features"].items():
            if isinstance(features, dict) and features:
                filename = f"features_{timeframe.replace(':', '_')}.parquet"
                filepath = output_path / filename

                # Convert features to DataFrame
                features_df = pd.DataFrame(features)
                features_df.to_parquet(filepath)
                logger.info(f"   Saved {timeframe} features to {filepath}")

        # Save comprehensive features
        if results["comprehensive_features"]:
            comp_filename = "comprehensive_features.parquet"
            comp_filepath = output_path / comp_filename

            # Convert to DataFrame if possible
            try:
                comp_df = pd.DataFrame(results["comprehensive_features"])
                comp_df.to_parquet(comp_filepath)
                logger.info(f"   Saved comprehensive features to {comp_filepath}")
            except Exception as e:
                logger.warning(f"   Could not save comprehensive features: {e}")

        # Save summary
        summary_filename = "optimization_summary.json"
        summary_filepath = output_path / summary_filename

        summary = {
            "total_features": results["total_features"],
            "feature_types": results["feature_types"],
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "base_timeframes": self.config["enhanced_multi_timeframe_optimization"]["base_timeframes"],
                "cross_timeframe_enabled": self.config["enhanced_multi_timeframe_optimization"]["cross_timeframe_enabled"],
                "regime_specific": self.config["enhanced_multi_timeframe_optimization"]["regime_specific"],
            },
        }

        import json
        with open(summary_filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"   Saved summary to {summary_filepath}")
        logger.info(f"✅ Results saved successfully to {output_path}")

async def main():
    """Main function demonstrating the multi-timeframe optimization."""
    logger.info("🚀 Starting Enhanced Multi-Timeframe Optimization Example")

    # Initialize the example
    example = MultiTimeframeOptimizationExample()

    try:
        # Setup optimizers (with optional matrix optimization results)
        matrix_results = {
            "diverse_lookback_periods": {
                "RSI": {"selected_periods": [7, 14, 21]},
                "MACD_fast": {"selected_periods": [8, 12, 16]},
                "Bollinger_Bands": {"selected_periods": [10, 20, 30]},
                "VWAP": {"selected_periods": [5, 10, 20]},
            },
            "regime_specific_periods": {
                "0": {  # Low volatility regime
                    "RSI": {"selected_periods": [5, 10, 15]},
                    "VWAP": {"selected_periods": [4, 8, 15]},
                },
                "1": {  # Medium volatility regime
                    "RSI": {"selected_periods": [7, 14, 21]},
                    "VWAP": {"selected_periods": [5, 10, 20]},
                },
                "2": {  # High volatility regime
                    "RSI": {"selected_periods": [10, 20, 30]},
                    "VWAP": {"selected_periods": [8, 15, 25]},
                },
            },
        }

        example.setup_optimizers(matrix_results)

        # Generate sample data
        sample_data = example.generate_sample_data(symbol="BTCUSDT", days=30)

        # Generate regime labels
        regime_labels = example.generate_regime_labels(sample_data["1m"])

        # Create target variable (next period returns)
        target = sample_data["1m"]["close"].pct_change().shift(-1).dropna()

        # Align data
        aligned_data = sample_data["1m"].loc[target.index]
        aligned_regime_labels = regime_labels.loc[target.index]

        # Run multi-timeframe optimization
        timeframe_features = await example.run_multi_timeframe_optimization(
            sample_data,
            aligned_regime_labels,
        )

        # Run comprehensive optimization
        comprehensive_features = await example.run_comprehensive_optimization(
            aligned_data,
            target,
            aligned_regime_labels,
        )

        # Analyze results
        results = example.analyze_results(timeframe_features, comprehensive_features)

        # Save results
        example.save_results(results)

        logger.info("🎉 Multi-timeframe optimization example completed successfully!")

        return results

    except Exception as e:
        logger.exception(f"❌ Error in main function: {e}")
        raise

if __name__ == "__main__":
    # Run the example
    try:
        results = asyncio.run(main())
        print("\n✅ Example completed successfully!")
        print(f"   Total features generated: {results['total_features']}")
        print(f"   Feature types: {results['feature_types']}")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
