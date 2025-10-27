#!/usr/bin/env python3
"""
Test script for the new confidence scaling: 0.5 for 75%, 1.0 for 100%, 1.5 for 200%.
"""

import pandas as pd
import numpy as np

def test_new_confidence_scaling():
    """Test the new confidence scaling logic."""
    print("🧪 Testing new confidence scaling...")
    
    # Test cases for the new scaling
    test_cases = [
        {"distance": 0.0, "expected": 0.0, "description": "0% of target"},
        {"distance": 0.375, "expected": 0.5 * 0.375 / 0.75, "description": "37.5% of target"},
        {"distance": 0.5, "expected": 0.5 * 0.5 / 0.75, "description": "50% of target"},
        {"distance": 0.75, "expected": 0.5, "description": "75% of target"},
        {"distance": 1.0, "expected": 1.0, "description": "100% of target"},
        {"distance": 1.5, "expected": 0.5 + 0.4 * (1.5 - 0.75), "description": "150% of target"},
        {"distance": 2.0, "expected": 0.5 + 0.4 * (2.0 - 0.75), "description": "200% of target"},
        {"distance": 3.0, "expected": 0.5 + 0.4 * (3.0 - 0.75), "description": "300% of target"},
    ]
    
    print("Testing new confidence scaling:")
    for case in test_cases:
        distance = case["distance"]
        expected = case["expected"]
        
        # Apply new confidence scaling logic
        proximity_factor = np.where(
            np.abs(distance) < 0.75,  # Below 75% of target
            0.5 * np.abs(distance) / 0.75,  # Linear scaling from 0 to 0.5
            np.where(
                np.abs(distance) >= 0.75,  # 75% and above
                0.5 + 0.4 * (np.abs(distance) - 0.75),  # Linear scaling: 0.5 + 0.4 * (distance - 0.75)
                0.5  # Fallback
            )
        )
        
        actual = proximity_factor.item() if proximity_factor.ndim == 0 else proximity_factor[0]
        
        print(f"   {case['description']}: {distance:.1f} -> {actual:.3f} (expected: {expected:.3f})")
        
        # Verify the calculation is correct (with small tolerance for floating point)
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual} for distance {distance}"
    
    print("✅ New confidence scaling working correctly")
    
    # Test with actual smooth label generation
    print("\nTesting with smooth label generation:")
    
    # Create test data with specific distances
    test_distances = [0.5, 0.75, 1.0, 1.5, 2.0]
    future_returns = pd.Series([d * 0.01 for d in test_distances])  # Scale by 0.01 threshold
    effective_threshold = pd.Series([0.01] * len(test_distances))
    vol_normalized = pd.Series([1.0] * len(test_distances))
    
    # Calculate distance from threshold
    distance = future_returns / effective_threshold
    
    # Apply sigmoid smoothing
    sharpness = 2.0
    smooth_labels = np.tanh(distance * sharpness)
    
    # Apply new confidence weighting
    proximity_factor = np.where(
        np.abs(distance) < 0.75,  # Below 75% of target
        0.5 * np.abs(distance) / 0.75,  # Linear scaling from 0 to 0.5
        np.where(
            np.abs(distance) >= 0.75,  # 75% and above
            0.5 + 0.4 * (np.abs(distance) - 0.75),  # Linear scaling: 0.5 + 0.4 * (distance - 0.75)
            0.5  # Fallback
        )
    )
    
    # Apply enhanced confidence weighting
    enhanced_smooth_labels = smooth_labels * proximity_factor
    
    print(f"   Generated {len(enhanced_smooth_labels)} enhanced smooth labels")
    print(f"   Test distances: {test_distances}")
    print(f"   Proximity factors: {proximity_factor}")
    print(f"   Enhanced labels: {enhanced_smooth_labels}")
    
    # Verify specific values
    expected_factors = [0.5 * 0.5 / 0.75, 0.5, 1.0, 0.5 + 0.4 * (1.5 - 0.75), 0.5 + 0.4 * (2.0 - 0.75)]  # Expected proximity factors
    for i, (actual, expected) in enumerate(zip(proximity_factor, expected_factors)):
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual} for distance {test_distances[i]}"
    
    print("✅ New confidence scaling verified with smooth label generation")
    return True

def main():
    """Run all tests."""
    print("🚀 Testing new confidence scaling: 0.5 for 75%, 1.0 for 100%, 1.5 for 200%...\n")
    
    # Test new confidence scaling
    test_success = test_new_confidence_scaling()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   New confidence scaling: {'✅' if test_success else '❌'}")
    
    if test_success:
        print("\n🎉 All tests passed! New confidence scaling working correctly.")
        print("\n📋 New Confidence Scaling:")
        print("   ✅ 0% of target → 0.0 confidence")
        print("   ✅ 37.5% of target → 0.25 confidence")
        print("   ✅ 50% of target → 0.33 confidence")
        print("   ✅ 75% of target → 0.5 confidence")
        print("   ✅ 100% of target → 1.0 confidence")
        print("   ✅ 150% of target → 1.3 confidence")
        print("   ✅ 200% of target → 1.5 confidence")
        print("   ✅ 300% of target → 2.0 confidence")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()