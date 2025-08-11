#!/usr/bin/env python3
"""
Data Quality Assessment Example

This example shows how to use the VectorizedLabellingOrchestrator's
data quality assessment functionality in your own code.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the orchestrator
from src.training.steps.vectorized_labelling_orchestrator import (
    VectorizedLabellingOrchestrator,
)


def create_test_data_with_nans() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create test data with known NaN values for demonstration."""

    # Create time index
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")

    # Create price data
    np.random.seed(42)
    base_price = 100 + np.cumsum(np.random.randn(1000) * 0.1)

    price_data = pd.DataFrame(
        {
            "open": base_price + np.random.randn(1000) * 0.5,
            "high": base_price + np.random.randn(1000) * 0.8,
            "low": base_price - np.random.randn(1000) * 0.8,
            "close": base_price + np.random.randn(1000) * 0.5,
            "volume": np.random.randint(100, 1000, 1000),
        },
        index=dates,
    )

    # Introduce NaN values for testing
    price_data.loc[100:105, "close"] = np.nan  # 6 NaN values
    price_data.loc[200:210, "volume"] = np.nan  # 11 NaN values
    price_data.loc[300:305, "high"] = np.nan  # 6 NaN values
    price_data.loc[400:415, "low"] = np.nan  # 16 NaN values

    # Create volume data
    volume_data = pd.DataFrame(
        {
            "volume": price_data["volume"].copy(),
            "trade_count": np.random.randint(10, 100, 1000),
            "avg_trade_size": np.random.uniform(0.1, 10.0, 1000),
        },
        index=dates,
    )

    # Introduce NaN values in volume data
    volume_data.loc[150:155, "trade_count"] = np.nan  # 6 NaN values
    volume_data.loc[250:260, "avg_trade_size"] = np.nan  # 11 NaN values

    return price_data, volume_data


async def example_data_quality_assessment():
    """Example of how to use the data quality assessment functionality."""

    print("🔍 Data Quality Assessment Example")
    print("=" * 50)

    # 1. Initialize the orchestrator
    config = {
        "vectorized_labelling_orchestrator": {
            "enable_stationary_checks": True,
            "enable_data_normalization": True,
            "enable_lookahead_bias_handling": True,
            "enable_feature_selection": True,
            "enable_memory_efficient_types": True,
            "enable_parquet_saving": True,
        }
    }

    orchestrator = VectorizedLabellingOrchestrator(config)

    print("🚀 Initializing orchestrator...")
    success = await orchestrator.initialize()

    if not success:
        print("❌ Failed to initialize orchestrator")
        return

    print("✅ Orchestrator initialized successfully")

    # 2. Create test data
    print("\n📊 Creating test data with known NaN values...")
    price_data, volume_data = create_test_data_with_nans()

    print(f"   Price data shape: {price_data.shape}")
    print(f"   Volume data shape: {volume_data.shape}")

    # 3. Perform data quality assessment
    print("\n🔍 Performing data quality assessment...")
    quality_report = await orchestrator.assess_input_data_quality(
        price_data, volume_data
    )

    if "error" in quality_report:
        print(f"❌ Error: {quality_report['error']}")
        return

    # 4. Analyze the results
    print("\n📋 ANALYSIS RESULTS:")
    print("-" * 30)

    summary = quality_report.get("summary", {})
    print(f"Total NaN values: {summary.get('total_nan_values', 'N/A')}")
    print(f"NaN percentage: {summary.get('nan_percentage', 'N/A'):.2f}%")
    print(f"Data quality score: {summary.get('data_quality_score', 'N/A'):.1f}/100")
    print(f"Severity: {summary.get('severity', 'N/A')}")

    # 5. Check specific datasets
    print("\n📊 DATASET DETAILS:")
    print("-" * 20)

    for dataset_name, dataset_stats in quality_report.items():
        if dataset_name in ["summary", "recommendations", "assessment_timestamp"]:
            continue

        print(f"\n{dataset_name.upper()}:")
        if "error" in dataset_stats:
            print(f"  ❌ {dataset_stats['error']}")
        else:
            print(f"  Shape: {dataset_stats.get('shape', 'N/A')}")
            print(f"  Total NaNs: {dataset_stats.get('total_nans', 'N/A')}")
            print(
                f"  NaN percentage: {dataset_stats.get('nan_percentage', 'N/A'):.2f}%"
            )

            # Show columns with NaN values
            columns_with_nans = dataset_stats.get("columns_with_nans", {})
            if columns_with_nans:
                print(f"  Columns with NaN values:")
                for col, col_stats in columns_with_nans.items():
                    print(
                        f"    - {col}: {col_stats['nan_count']} NaNs ({col_stats['nan_percentage']:.2f}%)"
                    )

    # 6. Show recommendations
    recommendations = quality_report.get("recommendations", [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    # 7. Demonstrate how to use the results programmatically
    print(f"\n🔧 PROGRAMMATIC USAGE:")
    print("-" * 25)

    # Check if data quality is acceptable
    severity = summary.get("severity", "UNKNOWN")
    if severity in ["EXCELLENT", "GOOD", "ACCEPTABLE"]:
        print("✅ Data quality is acceptable for processing")
    elif severity == "POOR":
        print("⚠️ Data quality is poor - consider cleaning")
    elif severity in ["VERY_POOR", "CRITICAL"]:
        print("❌ Data quality is critical - requires immediate attention")

    # Get specific column issues
    price_issues = quality_report.get("price_data", {}).get("columns_with_nans", {})
    if price_issues:
        print(f"\nPrice data issues found in {len(price_issues)} columns")
        for col, stats in price_issues.items():
            print(f"  - {col}: {stats['nan_count']} missing values")

    print("\n✅ Example completed!")


async def example_with_real_data_loading():
    """Example showing how to integrate with real data loading."""

    print("\n" + "=" * 60)
    print("🔍 REAL DATA INTEGRATION EXAMPLE")
    print("=" * 60)

    # This is a template for how you would integrate with real data
    async def load_and_assess_data(symbol: str, exchange: str, data_path: str):
        """
        Template function showing how to load real data and assess quality.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            data_path: Path to data directory
        """

        # 1. Load your data (implement based on your data format)
        try:
            # Example for CSV files:
            # price_file = f"{data_path}/{symbol}_{exchange}_price.csv"
            # volume_file = f"{data_path}/{symbol}_{exchange}_volume.csv"

            # price_data = pd.read_csv(price_file, index_col='timestamp', parse_dates=True)
            # volume_data = pd.read_csv(volume_file, index_col='timestamp', parse_dates=True)

            print(f"📊 Loading data for {symbol} from {exchange}...")

            # For this example, use test data
            price_data, volume_data = create_test_data_with_nans()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

        # 2. Initialize orchestrator
        config = {
            "vectorized_labelling_orchestrator": {
                "enable_stationary_checks": True,
                "enable_data_normalization": True,
                "enable_lookahead_bias_handling": True,
                "enable_feature_selection": True,
                "enable_memory_efficient_types": True,
                "enable_parquet_saving": True,
            }
        }

        orchestrator = VectorizedLabellingOrchestrator(config)
        await orchestrator.initialize()

        # 3. Assess data quality
        quality_report = await orchestrator.assess_input_data_quality(
            price_data, volume_data
        )

        # 4. Make decisions based on quality
        summary = quality_report.get("summary", {})
        severity = summary.get("severity", "UNKNOWN")

        if severity in ["CRITICAL"]:
            print(f"❌ Data quality is CRITICAL for {symbol}. Skipping processing.")
            return None
        elif severity in ["VERY_POOR", "POOR"]:
            print(
                f"⚠️ Data quality is {severity} for {symbol}. Proceeding with caution."
            )
        else:
            print(f"✅ Data quality is {severity} for {symbol}. Proceeding normally.")

        # 5. Return the data and quality report for further processing
        return {
            "price_data": price_data,
            "volume_data": volume_data,
            "quality_report": quality_report,
        }

    # Example usage
    result = await load_and_assess_data("ETHUSDT", "binance", "/path/to/data")

    if result:
        print(f"\n✅ Successfully loaded and assessed data")
        print(
            f"   Quality score: {result['quality_report']['summary']['data_quality_score']:.1f}/100"
        )
    else:
        print(f"\n❌ Failed to load or assess data")


async def main():
    """Main function to run the examples."""
    print("🎯 Data Quality Assessment Examples")
    print("=" * 50)

    # Run the basic example
    await example_data_quality_assessment()

    # Run the real data integration example
    await example_with_real_data_loading()

    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
