#!/usr/bin/env python3
"""
Test file to verify that step5 decorators are working correctly.
"""

import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import the decorators
from src.utils.centralized_decorators import validate_data_quality, handle_errors

# Test the decorators on simple functions
@validate_data_quality(
    required_columns=["open", "high", "low", "close", "volume"],
    min_rows=10,
    max_null_ratio=0.1,
    check_duplicates=True,
    check_timestamps=True,
    context="S/R sample weight calculation test"
)
@handle_errors(
    error_mapping={
        ValueError: "Invalid data format for S/R analysis",
        KeyError: "Missing required OHLCV columns",
        Exception: "Unexpected error in S/R sample weight calculation"
    },
    default_return=None,
    log_level="warning"
)
async def test_calculate_sr_sample_weights(data: pd.DataFrame, timeframe: str) -> pd.Series | None:
    """Test function for S/R sample weight calculation with decorators."""
    if len(data) == 0:
        return None
    
    # Check for required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in data.columns for col in required_cols):
        raise KeyError("Missing required OHLCV columns")
    
    # Simple weight calculation
    weights = pd.Series(0.5, index=data.index)
    
    # Add some variation based on close price
    if "close" in data.columns:
        weights = weights + (data["close"] - data["close"].mean()) / data["close"].std() * 0.1
        weights = weights.clip(0.1, 0.9)
    
    return weights

@validate_data_quality(
    required_columns=None,
    min_rows=5,
    max_null_ratio=0.2,
    check_duplicates=False,
    check_timestamps=False,
    context="mutual information calculation test"
)
@handle_errors(
    error_mapping={
        ValueError: "Invalid data format for mutual information calculation",
        ImportError: "Required sklearn modules not available",
        Exception: "Unexpected error in mutual information calculation"
    },
    default_return=np.ones(1),
    log_level="warning"
)
async def test_calculate_mutual_information(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Test function for mutual information calculation with decorators."""
    if len(X) < 5:
        raise ValueError("Insufficient data for mutual information calculation")
    
    # Simple mutual information calculation (mock)
    mi_scores = np.random.random(len(X.columns))
    return mi_scores

@validate_data_quality(
    required_columns=None,
    min_rows=5,
    max_null_ratio=0.2,
    check_duplicates=False,
    check_timestamps=False,
    context="comprehensive feature scoring test"
)
@handle_errors(
    error_mapping={
        ValueError: "Invalid data format for comprehensive scoring",
        ImportError: "Required ML libraries not available",
        Exception: "Unexpected error in comprehensive scoring"
    },
    default_return={},
    log_level="warning"
)
async def test_calculate_comprehensive_scores(X: pd.DataFrame, y: pd.Series) -> dict:
    """Test function for comprehensive feature scoring with decorators."""
    if len(X) < 5:
        raise ValueError("Insufficient data for comprehensive scoring")
    
    # Simple comprehensive scoring (mock)
    feature_scores = {}
    for col in X.columns:
        feature_scores[col] = {
            "mutual_info": np.random.random(),
            "rf_importance": np.random.random(),
            "f_statistic": np.random.random(),
            "combined_score": np.random.random()
        }
    
    return feature_scores

@validate_data_quality(
    required_columns=["close", "high", "low"],
    min_rows=10,
    max_null_ratio=0.1,
    check_duplicates=False,
    check_timestamps=True,
    context="TPSL direction calculation test"
)
@handle_errors(
    error_mapping={
        ValueError: "Invalid data format for TPSL calculation",
        KeyError: "Missing required price columns",
        IndexError: "Invalid index access in TPSL calculation",
        Exception: "Unexpected error in TPSL direction calculation"
    },
    default_return=0,
    log_level="warning"
)
async def test_calculate_tpsl_direction(hmm_data: pd.DataFrame, current_idx: int, window_start: int, window_end: int) -> int:
    """Test function for TPSL direction calculation with decorators."""
    if current_idx >= len(hmm_data):
        raise IndexError("Invalid index access in TPSL calculation")
    
    required_cols = ["close", "high", "low"]
    if not all(col in hmm_data.columns for col in required_cols):
        raise KeyError("Missing required price columns")
    
    # Simple TPSL direction calculation
    current_price = hmm_data.iloc[current_idx]["close"]
    
    # Simple logic: if price is above average, go long (1), else neutral (0)
    avg_price = hmm_data["close"].mean()
    if current_price > avg_price:
        return 1  # long
    else:
        return 0  # neutral

async def run_tests():
    """Run all the decorator tests."""
    print("🧪 Testing Step5 Decorators...")
    
    # Create test data
    test_data = pd.DataFrame({
        "open": np.random.random(20) * 100,
        "high": np.random.random(20) * 100,
        "low": np.random.random(20) * 100,
        "close": np.random.random(20) * 100,
        "volume": np.random.random(20) * 1000,
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1H")
    })
    
    test_features = pd.DataFrame({
        "feature1": np.random.random(10),
        "feature2": np.random.random(10),
        "feature3": np.random.random(10)
    })
    
    test_target = pd.Series(np.random.randint(0, 3, 10))
    
    # Test 1: S/R sample weight calculation
    print("\n📊 Test 1: S/R Sample Weight Calculation")
    try:
        weights = await test_calculate_sr_sample_weights(test_data, "5m")
        print(f"✅ S/R weights calculated successfully: {len(weights)} weights")
        print(f"   Weight range: {weights.min():.3f} - {weights.max():.3f}")
    except Exception as e:
        print(f"❌ S/R weight calculation failed: {e}")
    
    # Test 2: Mutual information calculation
    print("\n📊 Test 2: Mutual Information Calculation")
    try:
        mi_scores = await test_calculate_mutual_information(test_features, test_target)
        print(f"✅ Mutual information calculated successfully: {len(mi_scores)} scores")
        print(f"   Score range: {mi_scores.min():.3f} - {mi_scores.max():.3f}")
    except Exception as e:
        print(f"❌ Mutual information calculation failed: {e}")
    
    # Test 3: Comprehensive feature scoring
    print("\n📊 Test 3: Comprehensive Feature Scoring")
    try:
        scores = await test_calculate_comprehensive_scores(test_features, test_target)
        print(f"✅ Comprehensive scoring completed successfully: {len(scores)} features")
        for feature, score_dict in scores.items():
            print(f"   {feature}: {score_dict['combined_score']:.3f}")
    except Exception as e:
        print(f"❌ Comprehensive scoring failed: {e}")
    
    # Test 4: TPSL direction calculation
    print("\n📊 Test 4: TPSL Direction Calculation")
    try:
        direction = await test_calculate_tpsl_direction(test_data, 5, 0, 10)
        print(f"✅ TPSL direction calculated successfully: {direction}")
        direction_text = "LONG" if direction == 1 else "NEUTRAL" if direction == 0 else "SHORT"
        print(f"   Direction: {direction_text}")
    except Exception as e:
        print(f"❌ TPSL direction calculation failed: {e}")
    
    # Test 5: Error handling with invalid data
    print("\n📊 Test 5: Error Handling with Invalid Data")
    try:
        invalid_data = pd.DataFrame({"wrong_column": [1, 2, 3]})
        weights = await test_calculate_sr_sample_weights(invalid_data, "5m")
        print(f"❌ Should have failed with invalid data, but got: {weights}")
    except Exception as e:
        print(f"✅ Error handling worked correctly: {type(e).__name__}: {e}")
    
    print("\n🎉 Step5 Decorator Tests Completed!")

if __name__ == "__main__":
    asyncio.run(run_tests())