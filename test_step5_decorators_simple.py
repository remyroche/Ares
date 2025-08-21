#!/usr/bin/env python3
"""
Simple test file to verify that step5 decorators are working correctly.
This version doesn't require external dependencies.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock pandas and numpy for testing
class MockDataFrame:
    def __init__(self, data=None, columns=None):
        self.data = data or {}
        self.columns = columns or []
        self.index = list(range(len(data) if data else 0))
        self.shape = (len(data) if data else 0, len(columns) if columns else 0)
    
    def __len__(self):
        return len(self.data) if self.data else 0
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data.get(key, [])
        return self.data.get(key, [])
    
    def __contains__(self, key):
        return key in self.columns

class MockSeries:
    def __init__(self, data=None, index=None):
        self.data = data or []
        self.index = index or list(range(len(data) if data else 0))
        self.dtype = "object"
    
    def __len__(self):
        return len(self.data)
    
    def unique(self):
        return list(set(self.data))
    
    def fillna(self, value):
        return self

class MockNumpy:
    @staticmethod
    def random(size):
        return [0.5] * size
    
    @staticmethod
    def ones(size):
        return [1.0] * size
    
    @staticmethod
    def mean(arr):
        return sum(arr) / len(arr) if arr else 0
    
    @staticmethod
    def std(arr):
        return 1.0  # Mock standard deviation
    
    @staticmethod
    def clip(arr, min_val, max_val):
        return [max(min_val, min(max_val, x)) for x in arr]

# Mock the imports
sys.modules['pandas'] = type('MockPandas', (), {
    'DataFrame': MockDataFrame,
    'Series': MockSeries
})
sys.modules['numpy'] = MockNumpy()

# Now try to import the decorators
try:
    from utils.centralized_decorators import validate_data_quality, handle_errors
    print("✅ Successfully imported decorators from centralized_decorators")
except ImportError as e:
    print(f"❌ Failed to import decorators: {e}")
    # Try alternative import path
    try:
        from src.utils.centralized_decorators import validate_data_quality, handle_errors
        print("✅ Successfully imported decorators from src.utils.centralized_decorators")
    except ImportError as e2:
        print(f"❌ Failed to import decorators from alternative path: {e2}")
        sys.exit(1)

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
async def test_calculate_sr_sample_weights(data, timeframe: str):
    """Test function for S/R sample weight calculation with decorators."""
    if len(data) == 0:
        return None
    
    # Check for required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in data.columns for col in required_cols):
        raise KeyError("Missing required OHLCV columns")
    
    # Simple weight calculation
    weights = [0.5] * len(data)
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
    default_return=[1.0],
    log_level="warning"
)
async def test_calculate_mutual_information(X, y):
    """Test function for mutual information calculation with decorators."""
    if len(X) < 5:
        raise ValueError("Insufficient data for mutual information calculation")
    
    # Simple mutual information calculation (mock)
    mi_scores = [0.5] * len(X.columns)
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
async def test_calculate_comprehensive_scores(X, y):
    """Test function for comprehensive feature scoring with decorators."""
    if len(X) < 5:
        raise ValueError("Insufficient data for comprehensive scoring")
    
    # Simple comprehensive scoring (mock)
    feature_scores = {}
    for col in X.columns:
        feature_scores[col] = {
            "mutual_info": 0.5,
            "rf_importance": 0.5,
            "f_statistic": 0.5,
            "combined_score": 0.5
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
async def test_calculate_tpsl_direction(hmm_data, current_idx: int, window_start: int, window_end: int):
    """Test function for TPSL direction calculation with decorators."""
    if current_idx >= len(hmm_data):
        raise IndexError("Invalid index access in TPSL calculation")
    
    required_cols = ["close", "high", "low"]
    if not all(col in hmm_data.columns for col in required_cols):
        raise KeyError("Missing required price columns")
    
    # Simple TPSL direction calculation
    current_price = hmm_data.get("close", [100])[0]
    
    # Simple logic: if price is above 100, go long (1), else neutral (0)
    if current_price > 100:
        return 1  # long
    else:
        return 0  # neutral

async def run_tests():
    """Run all the decorator tests."""
    print("🧪 Testing Step5 Decorators...")
    
    # Create test data
    test_data = MockDataFrame(
        data={
            "open": [100] * 20,
            "high": [110] * 20,
            "low": [90] * 20,
            "close": [105] * 20,
            "volume": [1000] * 20
        },
        columns=["open", "high", "low", "close", "volume"]
    )
    
    test_features = MockDataFrame(
        data={
            "feature1": [0.5] * 10,
            "feature2": [0.5] * 10,
            "feature3": [0.5] * 10
        },
        columns=["feature1", "feature2", "feature3"]
    )
    
    test_target = MockSeries([0, 1, 2] * 3 + [1])
    
    # Test 1: S/R sample weight calculation
    print("\n📊 Test 1: S/R Sample Weight Calculation")
    try:
        weights = await test_calculate_sr_sample_weights(test_data, "5m")
        print(f"✅ S/R weights calculated successfully: {len(weights)} weights")
        print(f"   Weight range: {min(weights):.3f} - {max(weights):.3f}")
    except Exception as e:
        print(f"❌ S/R weight calculation failed: {e}")
    
    # Test 2: Mutual information calculation
    print("\n📊 Test 2: Mutual Information Calculation")
    try:
        mi_scores = await test_calculate_mutual_information(test_features, test_target)
        print(f"✅ Mutual information calculated successfully: {len(mi_scores)} scores")
        print(f"   Score range: {min(mi_scores):.3f} - {max(mi_scores):.3f}")
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
        invalid_data = MockDataFrame({"wrong_column": [1, 2, 3]}, ["wrong_column"])
        weights = await test_calculate_sr_sample_weights(invalid_data, "5m")
        print(f"❌ Should have failed with invalid data, but got: {weights}")
    except Exception as e:
        print(f"✅ Error handling worked correctly: {type(e).__name__}: {e}")
    
    print("\n🎉 Step5 Decorator Tests Completed!")

if __name__ == "__main__":
    asyncio.run(run_tests())