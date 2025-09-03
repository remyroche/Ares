#!/usr/bin/env python3
"""
Demonstration of Enhanced Validation in Training Pipeline

This script shows how the enhanced validation modules work in steps 1-6:
1. Cross-Step Data Consistency Validation
2. Statistical Distribution Validation
3. Feature Engineering Validation
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cross_step_validation import CrossStepValidator
from src.utils.feature_engineering_validation import FeatureEngineeringValidator
from src.utils.logger import system_logger
from src.utils.statistical_distribution_validation import StatisticalValidator

logger = system_logger.getChild("EnhancedValidationDemo")


def create_sample_data(rows: int = 1000, issue_type: str = None) -> pd.DataFrame:
    """Create sample OHLCV data with optional issues for testing."""
    np.random.seed(42)

    # Generate base timestamp
    timestamps = pd.date_range(start="2024-01-01", periods=rows, freq="1min")
    timestamps_ms = (timestamps.astype(np.int64) // 10**6).astype(np.int64)

    # Generate OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(rows) * 0.1)

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open": close_prices + np.random.randn(rows) * 0.05,
        "high": close_prices + np.abs(np.random.randn(rows) * 0.1),
        "low": close_prices - np.abs(np.random.randn(rows) * 0.1),
        "close": close_prices,
        "volume": np.abs(np.random.randn(rows) * 1000 + 5000),
    })

    # Introduce specific issues for testing
    if issue_type == "missing_data":
        # Remove 5% of rows randomly
        drop_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df = df.drop(drop_indices).reset_index(drop=True)

    elif issue_type == "outliers":
        # Add extreme outliers
        outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[outlier_indices, "volume"] *= 100  # Extreme volume spikes
        df.loc[outlier_indices[:5], "close"] *= 2  # Price spikes

    elif issue_type == "non_stationary":
        # Add strong trend and volatility changes
        df["close"] = df["close"] * (1 + np.linspace(0, 0.5, len(df)))  # 50% trend
        df["volume"] = df["volume"] * (1 + np.sin(np.linspace(0, 4*np.pi, len(df))))

    elif issue_type == "data_corruption":
        # Add NaN values and impossible OHLC relationships
        nan_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        df.loc[nan_indices, "volume"] = np.nan

        # Make high < low for some rows
        corrupt_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
        df.loc[corrupt_indices, "high"] = df.loc[corrupt_indices, "low"] - 1

    return df


def create_sample_features(df: pd.DataFrame, issue_type: str = None) -> pd.DataFrame:
    """Create engineered features with optional issues."""
    features_df = df.copy()

    # Add basic features
    features_df["returns"] = features_df["close"].pct_change()
    features_df["log_returns"] = np.log1p(features_df["returns"])
    features_df["volume_ma_20"] = features_df["volume"].rolling(20).mean()
    features_df["rsi"] = calculate_rsi(features_df["close"], 14)
    features_df["price_ma_50"] = features_df["close"].rolling(50).mean()

    # Introduce feature engineering issues
    if issue_type == "calculation_error":
        # Introduce calculation errors
        features_df["log_returns"] = np.log(features_df["returns"])  # Wrong formula (should be log1p)

    elif issue_type == "excessive_nan":
        # Create features that produce excessive NaN values
        features_df["bad_feature"] = features_df["volume"].rolling(500).std() / features_df["volume"].rolling(5).std()

    elif issue_type == "out_of_range":
        # Create features with values outside expected ranges
        features_df["rsi"] = features_df["rsi"] * 2  # RSI > 100
        features_df["returns"] = features_df["returns"] * 10  # Extreme returns

    elif issue_type == "feature_leakage":
        # Add a perfect predictor (future information)
        features_df["future_return"] = features_df["returns"].shift(-1)
        features_df["target"] = (features_df["returns"].shift(-1) > 0).astype(int)

    return features_df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


async def demo_cross_step_validation():
    """Demonstrate cross-step validation."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Cross-Step Data Consistency Validation")
    logger.info("="*80)

    validator = CrossStepValidator(logger)

    # Test 1: Normal transition (should pass)
    logger.info("\n📊 Test 1: Normal data transition")
    step1_data = create_sample_data(1000)
    step2_data = step1_data.copy()

    result = validator.validate_step_transition(
        previous_step_output=step1_data,
        current_step_input=step2_data,
        previous_step_name="step1_data_collection",
        current_step_name="step1_5_data_converter",
    )

    logger.info(f"✅ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")

    # Test 2: Data with row count changes
    logger.info("\n📊 Test 2: Data with significant row loss")
    step1_data = create_sample_data(1000)
    step2_data = create_sample_data(800, issue_type="missing_data")

    result = validator.validate_step_transition(
        previous_step_output=step1_data,
        current_step_input=step2_data,
        previous_step_name="step1_5_data_converter",
        current_step_name="step2_feature_engineering",
    )

    logger.info(f"❌ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")
    if result.issues:
        logger.info(f"🚨 Issues found: {len(result.issues)}")
        for issue in result.issues[:3]:
            logger.info(f"   - {issue.message}")


async def demo_statistical_validation():
    """Demonstrate statistical distribution validation."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Statistical Distribution Validation")
    logger.info("="*80)

    validator = StatisticalValidator(logger)

    # Test 1: Normal data
    logger.info("\n📊 Test 1: Well-behaved market data")
    data = create_sample_data(1000)

    result = validator.validate_distribution(
        df=data,
        columns=["open", "high", "low", "close", "volume"],
        check_stationarity=True,
    )

    logger.info(f"✅ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")

    # Test 2: Data with outliers
    logger.info("\n📊 Test 2: Data with extreme outliers")
    data = create_sample_data(1000, issue_type="outliers")

    result = validator.validate_distribution(
        df=data,
        columns=["open", "high", "low", "close", "volume"],
        check_stationarity=True,
    )

    logger.info(f"⚠️ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")
    if result.warnings:
        logger.info(f"⚠️ Warnings: {len(result.warnings)}")
        for warning in result.warnings[:3]:
            logger.info(f"   - {warning.message}")

    # Test 3: Non-stationary data
    logger.info("\n📊 Test 3: Non-stationary data with trends")
    data = create_sample_data(1000, issue_type="non_stationary")

    result = validator.validate_distribution(
        df=data,
        columns=["close", "volume"],
        check_stationarity=True,
    )

    logger.info(f"⚠️ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")


async def demo_feature_engineering_validation():
    """Demonstrate feature engineering validation."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Feature Engineering Validation")
    logger.info("="*80)

    validator = FeatureEngineeringValidator(logger)

    # Test 1: Correct feature engineering
    logger.info("\n📊 Test 1: Correctly engineered features")
    original_data = create_sample_data(1000)
    features_data = create_sample_features(original_data)

    result = validator.validate_engineered_features(
        original_df=original_data,
        features_df=features_data,
        feature_config={},
        validate_calculations=True,
        check_dependencies=True,
    )

    logger.info(f"✅ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")

    # Test 2: Features with calculation errors
    logger.info("\n📊 Test 2: Features with calculation errors")
    original_data = create_sample_data(1000)
    features_data = create_sample_features(original_data, issue_type="calculation_error")

    result = validator.validate_engineered_features(
        original_df=original_data,
        features_df=features_data,
        feature_config={},
        validate_calculations=True,
        check_dependencies=True,
    )

    logger.info(f"❌ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")

    # Test 3: Features with out-of-range values
    logger.info("\n📊 Test 3: Features with out-of-range values")
    original_data = create_sample_data(1000)
    features_data = create_sample_features(original_data, issue_type="out_of_range")

    result = validator.validate_engineered_features(
        original_df=original_data,
        features_df=features_data,
        feature_config={},
        validate_calculations=True,
        check_dependencies=True,
    )

    logger.info(f"❌ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")
    if result.issues:
        logger.info(f"🚨 Issues found: {len(result.issues)}")
        for issue in result.issues[:3]:
            logger.info(f"   - {issue.message}")

    # Test 4: Feature leakage detection
    logger.info("\n📊 Test 4: Feature leakage detection")
    original_data = create_sample_data(1000)
    features_data = create_sample_features(original_data, issue_type="feature_leakage")

    result = validator.validate_engineered_features(
        original_df=original_data,
        features_df=features_data,
        feature_config={},
        validate_calculations=True,
        check_dependencies=True,
    )

    logger.info(f"❌ Validation passed: {result.passed}")
    logger.info(f"📊 Quality score: {result.quality_score:.2f}")
    logger.info("🚨 Feature leakage should be detected!")


async def main():
    """Run all validation demonstrations."""
    logger.info("🚀 Starting Enhanced Validation Demonstration")

    # Run demos
    await demo_cross_step_validation()
    await demo_statistical_validation()
    await demo_feature_engineering_validation()

    logger.info("\n" + "="*80)
    logger.info("✅ Enhanced Validation Demonstration Complete!")
    logger.info("="*80)

    logger.info("\n📝 Summary:")
    logger.info("1. Cross-Step Validation: Ensures data consistency between pipeline steps")
    logger.info("2. Statistical Validation: Detects distribution anomalies and non-stationarity")
    logger.info("3. Feature Engineering Validation: Verifies feature calculations and detects leakage")
    logger.info("\nThese validations are now integrated into steps 1-6 of the enhanced training manager!")


if __name__ == "__main__":
    asyncio.run(main())
