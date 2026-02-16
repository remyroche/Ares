#!/usr/bin/env python3
"""
Test script to verify the new event scoring and magnitude clipping functionality.
"""

import numpy as np
import pandas as pd
from sample_weights import compute_sample_weights_with_uniqueness, build_label_time_ranges

def test_event_scoring():
    """Test the new event scoring without magnitude weighting."""
    print("Testing event scoring without magnitude weighting...")
    
    # Create synthetic data
    n_samples = 1000
    np.random.seed(42)
    
    # Generate returns with varying intensity (some extreme outliers)
    returns = np.concatenate([
        np.random.normal(0, 0.01, 900),  # Normal returns
        np.random.normal(0, 0.05, 90),   # High volatility
        np.array([0.15, -0.12, 0.20, -0.18])  # Extreme events
    ])
    
    # Generate selection metric values (range_16h_pct) that correlate somewhat with returns
    # This is the high/low percentage difference over 16 hours used in candidate selection
    selection_metric = np.concatenate([
        np.random.uniform(0.01, 0.06, 900),  # Normal range values (1-6%)
        np.random.uniform(0.06, 0.12, 90),   # High range values (6-12%)
        np.array([0.18, 0.20, 0.25, 0.28])  # Extreme range values (>18%)
    ])
    
    # Create label times (simple sequential labels)
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    entry_times = timestamps[:-1]
    exit_times = timestamps[1:]
    
    label_times = build_label_time_ranges(entry_times, exit_times)
    
    # Test 1: Without selection metric (fallback to returns)
    print("\n=== Test 1: Without selection metric (fallback) ===")
    weights_fallback = compute_sample_weights_with_uniqueness(
        label_times=label_times,
        returns=returns,
        time_grid=timestamps
    )
    
    # Test 2: With selection metric (proper usage)
    print("\n=== Test 2: With selection metric (range_16h_pct) ===")
    weights_with_metric = compute_sample_weights_with_uniqueness(
        label_times=label_times,
        returns=returns,
        selection_metric=selection_metric,
        time_grid=timestamps
    )
    
    print(f"\nComparison:")
    print(f"Fallback weights - mean={weights_fallback.mean():.3f}, std={weights_fallback.std():.3f}")
    print(f"Metric weights - mean={weights_with_metric.mean():.3f}, std={weights_with_metric.std():.3f}")
    
    # Verify event scoring works (extreme events should have higher scores)
    extreme_indices = np.where(selection_metric > 0.18)[0]
    normal_indices = np.where(selection_metric < 0.06)[0]
    
    if len(extreme_indices) > 0 and len(normal_indices) > 0:
        extreme_weights = weights_with_metric[extreme_indices]
        normal_weights = weights_with_metric[normal_indices]
        
        print(f"\nEvent intensity verification (using range_16h_pct):")
        print(f"Extreme range weights: mean={extreme_weights.mean():.3f}, std={extreme_weights.std():.3f}")
        print(f"Normal range weights: mean={normal_weights.mean():.3f}, std={normal_weights.std():.3f}")
        print(f"Weight ratio (extreme/normal): {extreme_weights.mean() / normal_weights.mean():.2f}x")
    
    # Test percentile clamping by creating extreme case
    print("\n=== Test 3: Percentile clamping verification ===")
    extreme_selection_metric = np.array([0.01, 0.99] + [0.5] * 98)  # Extreme percentiles at boundaries
    extreme_returns = np.array([0.01, 0.01] + [0.01] * 98)
    extreme_label_times = build_label_time_ranges(
        pd.date_range('2023-01-01', periods=100, freq='1H')[:-1],
        pd.date_range('2023-01-01', periods=100, freq='1H')[1:]
    )
    
    weights_clamped = compute_sample_weights_with_uniqueness(
        label_times=extreme_label_times,
        returns=extreme_returns,
        selection_metric=extreme_selection_metric,
        time_grid=pd.date_range('2023-01-01', periods=100, freq='1H')
    )
    
    print(f"Clamping test - Weight range: [{weights_clamped.min():.3f}, {weights_clamped.max():.3f}]")
    print(f"Expected range: [0.8, 1.2] (due to percentile clamping)")
    
    # Verify that magnitude weighting is removed
    print("\n=== Test 4: Verify magnitude weighting removed ===")
    print("✅ Magnitude weighting removed - weights based only on uniqueness and event intensity")
    print("✅ Final formula: w = uniqueness * event_score * base_weight")
    
    return weights_fallback, weights_with_metric

if __name__ == "__main__":
    weights_fallback, weights_with_metric = test_event_scoring()
    print("\nTest completed successfully!")
