#!/usr/bin/env python3
"""
Example usage of validation decorators for continuous file validation.

This example demonstrates how to use the validation decorators to validate
files and DataFrames at every action throughout the pipeline steps.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import validation decorators
try:
    from src.utils.validation_decorators import (
        validate_file_operation,
        validate_dataframe_operation,
        validate_step_operation,
        validate_step1_operation,
        validate_step1_5_operation,
        validate_step2_operation,
        validate_step4_operation
    )
except ImportError as e:
    print(f"Could not import validation decorators: {e}")
    # Create dummy decorators for demonstration
    def validate_file_operation(*args, **kwargs):

    def validate_dataframe_operation(*args, **kwargs):

    def validate_step_operation(*args, **kwargs):


# Example 1: File operation validation
@validate_file_operation("step1", expected_schema="klines", log_level="INFO")
async def load_klines_data(file_path: str) -> str:
    """Load klines data with automatic validation."""
    print(f"Loading klines data from: {file_path}")
    # Simulate file loading
    return f"processed_{file_path}"


@validate_file_operation("step1_5", expected_schema="features", log_level="WARNING")
def save_unified_data(data: Dict[str, Any], output_path: str) -> str:
    """Save unified data with automatic validation."""
    print(f"Saving unified data to: {output_path}")
    # Simulate file saving
    return output_path


# Example 2: DataFrame operation validation
@validate_dataframe_operation("step2", validate_before=True, validate_after=True, log_level="INFO")
def process_features(df, feature_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process features with automatic DataFrame validation."""
    print(f"Processing features for DataFrame with shape: {df.shape if hasattr(df, 'shape') else 'unknown'}")
    # Simulate feature processing
    return {
        "train": df,
        "validation": df,
        "test": df
    }


# Example 3: Step operation validation
@validate_step_operation("step4", validate_files=True, validate_dataframes=True, log_level="INFO")
async def run_labeling_step(symbol: str, exchange: str, data_dir: str) -> Dict[str, Any]:
    """Run labeling step with comprehensive validation."""
    print(f"Running labeling step for {symbol} on {exchange}")

    # Simulate step execution
    result = {
        "labeled_train": f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
        "labeled_validation": f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
        "labeled_test": f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet"
    }

    return result


# Example 4: Step-specific convenience decorators
@validate_step1_operation
async def step1_data_collection(symbol: str, exchange: str, timeframe: str) -> str:
    """Step 1 data collection with automatic validation."""
    print(f"Collecting data for {symbol} on {exchange} with {timeframe} timeframe")
    return f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"


@validate_step1_5_operation
async def step1_5_data_conversion(symbol: str, exchange: str, timeframe: str) -> str:
    """Step 1.5 data conversion with automatic validation."""
    print(f"Converting data for {symbol} on {exchange} with {timeframe} timeframe")
    return f"data_cache/unified_{exchange}_{symbol}_{timeframe}.parquet"


@validate_step2_operation
async def step2_feature_engineering(symbol: str, exchange: str, data_dir: str) -> Dict[str, str]:
    """Step 2 feature engineering with automatic validation."""
    print(f"Engineering features for {symbol} on {exchange}")
    return {
        "train": f"{data_dir}/features_{exchange}_{symbol}_train.parquet",
        "validation": f"{data_dir}/features_{exchange}_{symbol}_validation.parquet",
        "test": f"{data_dir}/features_{exchange}_{symbol}_test.parquet"
    }


@validate_step4_operation
async def step4_labeling(symbol: str, exchange: str, data_dir: str) -> Dict[str, str]:
    """Step 4 labeling with automatic validation."""
    print(f"Labeling data for {symbol} on {exchange}")
    return {
        "train": f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
        "validation": f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
        "test": f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet"
    }


# Example 5: Custom validation with specific schemas
@validate_file_operation("custom_step", expected_schema="custom", log_level="DEBUG")

# Example 6: DataFrame validation with different log levels
@validate_dataframe_operation("debug_step", validate_before=True, validate_after=True, log_level="DEBUG")

# Example usage functions
async def demonstrate_file_validation():
    """Demonstrate file validation decorators."""
    print("\n=== File Validation Examples ===")

    # Create a test file
    test_file = "test_klines.parquet"
    with open(test_file, 'w') as f:
        f.write("test data")

    try:
        # Test file operation validation
        result = await load_klines_data(test_file)
        print(f"File operation result: {result}")

        # Test file saving validation
        save_result = save_unified_data({"data": "test"}, "test_unified.parquet")
        print(f"Save operation result: {save_result}")

    finally:
        # Clean up test files
        for file in [test_file, "test_unified.parquet"]:
            if os.path.exists(file):
                os.remove(file)


async def demonstrate_dataframe_validation():
    """Demonstrate DataFrame validation decorators."""
    print("\n=== DataFrame Validation Examples ===")

    # Create a mock DataFrame-like object
    class MockDataFrame:
        def __init__(self, shape):
            self.shape = shape
            self.empty = False

    class MockSeries:
        def __init__(self, data):
            self.data = data

        def sum(self):
            return sum(self.data)

    # Test DataFrame validation
    mock_df = MockDataFrame((100, 5))
    result = process_features(mock_df, {"feature_type": "technical"})
    print(f"DataFrame operation result: {type(result)}")


async def demonstrate_step_validation():
    """Demonstrate step validation decorators."""
    print("\n=== Step Validation Examples ===")

    # Test step-specific decorators
    step1_result = await step1_data_collection("ETHUSDT", "BINANCE", "1m")
    print(f"Step 1 result: {step1_result}")

    step1_5_result = await step1_5_data_conversion("ETHUSDT", "BINANCE", "1m")
    print(f"Step 1.5 result: {step1_5_result}")

    step2_result = await step2_feature_engineering("ETHUSDT", "BINANCE", "data/training")
    print(f"Step 2 result: {step2_result}")

    step4_result = await step4_labeling("ETHUSDT", "BINANCE", "data/training")
    print(f"Step 4 result: {step4_result}")


async def main():
    """Run all validation decorator examples."""
    print("🚀 Validation Decorators Examples")
    print("=" * 50)

    try:
        await demonstrate_file_validation()
        await demonstrate_dataframe_validation()
        await demonstrate_step_validation()

        print("\n✅ All examples completed successfully!")
        print("\n📋 Key Benefits of Validation Decorators:")
        print("   - Continuous validation at every operation")
        print("   - Automatic file path and name validation")
        print("   - DataFrame quality monitoring")
        print("   - Configurable validation levels")
        print("   - Non-blocking validation (logs issues but continues)")
        print("   - Step-specific validation rules")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())