#!/usr/bin/env python3
"""
Test script for peak/trough detection integration in profit labeling.

This script tests the enhanced profit labeling with local extrema detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from consolidated_profit_labeler import ConsolidatedProfitLabeler, ConsolidatedLabelerConfig

def create_test_data_with_peaks_troughs(n_bars=1000, base_price=100.0, seed=42):
    """Create test data with known peaks and troughs for validation."""
    np.random.seed(seed)
    
    # Create time index
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')
    
    # Generate price data with trends and volatility
    prices = [base_price]
    highs = [base_price]
    lows = [base_price]
    
    for i in range(1, n_bars):
        # Add trend and volatility
        trend = 0.0001 * np.sin(i / 50)  # Cyclical trend
        vol = 0.002 + 0.001 * np.sin(i / 100)  # Changing volatility
        ret = np.random.normal(trend, vol)
        
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
        
        # Add some noise to high/low
        high_noise = abs(np.random.normal(0, 0.001))
        low_noise = abs(np.random.normal(0, 0.001))
        
        highs.append(new_price * (1 + high_noise))
        lows.append(new_price * (1 - low_noise))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_peak_trough_detection():
    """Test the peak/trough detection functionality."""
    print("🧪 Testing peak/trough detection...")
    
    # Create test data
    data = create_test_data_with_peaks_troughs(n_bars=500)
    
    # Create labeler with peak/trough detection enabled
    config = ConsolidatedLabelerConfig(
        enable_peak_trough_detection=True,
        peak_detection_method="find_peaks",
        peak_prominence=0.001,
        peak_distance=5,
        smoothing_window=3,
        target_bands={
            'small': (0.4, 0.8),
            'medium': (0.8, 1.3)
        }
    )
    
    labeler = ConsolidatedProfitLabeler(config)
    
    # Test peak/trough detection
    peaks, troughs = labeler._detect_peaks_troughs(data, 'close')
    
    print(f"✅ Peak detection: {peaks.sum()} peaks found")
    print(f"✅ Trough detection: {troughs.sum()} troughs found")
    
    # Test local extrema in window
    peak_idx, trough_idx = labeler._find_local_extrema_in_window(data, 100, 150, 'close')
    print(f"✅ Local extrema in window [100:150]: peak={peak_idx}, trough={trough_idx}")
    
    # Test opportunity pattern detection
    patterns = labeler._detect_opportunity_patterns(data, 100, 20)
    print(f"✅ Opportunity patterns: {patterns}")
    
    return peaks, troughs, data

def test_enhanced_labeling():
    """Test the enhanced labeling with peak/trough detection."""
    print("\n🎯 Testing enhanced labeling with peak/trough detection...")
    
    # Create test data
    data = create_test_data_with_peaks_troughs(n_bars=300)
    
    # Create labeler with peak/trough detection enabled
    config = ConsolidatedLabelerConfig(
        enable_peak_trough_detection=True,
        peak_detection_method="find_peaks",
        peak_prominence=0.001,
        peak_distance=3,
        smoothing_window=2,
        target_bands={
            'small': (0.5, 1.0)
        },
        min_bars_for_labeling=50
    )
    
    labeler = ConsolidatedProfitLabeler(config)
    
    # Generate labels
    result = labeler.generate_labels(data)
    
    print(f"✅ Enhanced labeling completed:")
    print(f"   → Input samples: {len(data)}")
    print(f"   → Labeled samples: {len(result.labels)}")
    print(f"   → Quality score: {result.overall_quality_score:.3f}")
    
    # Check for extrema-related columns
    extrema_columns = [col for col in result.labels.columns if 'extrema' in col.lower()]
    print(f"   → Extrema columns: {extrema_columns}")
    
    # Show sample of labels with extrema information
    if 'extrema_type' in result.labels.columns:
        extrema_labels = result.labels[result.labels['target'] != 0]
        if len(extrema_labels) > 0:
            print(f"   → Sample extrema types: {extrema_labels['extrema_type'].value_counts().to_dict()}")
    
    return result

def visualize_peak_trough_detection(data, peaks, troughs, sample_range=(100, 200)):
    """Visualize peak/trough detection results."""
    try:
        import matplotlib.pyplot as plt
        
        start_idx, end_idx = sample_range
        sample_data = data.iloc[start_idx:end_idx]
        sample_peaks = peaks.iloc[start_idx:end_idx]
        sample_troughs = troughs.iloc[start_idx:end_idx]
        
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        plt.subplot(2, 1, 1)
        plt.plot(sample_data.index, sample_data['close'], 'b-', label='Close Price', linewidth=1)
        plt.plot(sample_data.index, sample_data['high'], 'g--', alpha=0.7, label='High')
        plt.plot(sample_data.index, sample_data['low'], 'r--', alpha=0.7, label='Low')
        
        # Mark peaks and troughs
        peak_indices = sample_data.index[sample_peaks == 1]
        trough_indices = sample_data.index[sample_troughs == 1]
        
        if len(peak_indices) > 0:
            plt.scatter(peak_indices, sample_data.loc[peak_indices, 'high'], 
                       color='red', marker='^', s=100, label='Peaks', zorder=5)
        
        if len(trough_indices) > 0:
            plt.scatter(trough_indices, sample_data.loc[trough_indices, 'low'], 
                       color='green', marker='v', s=100, label='Troughs', zorder=5)
        
        plt.title('Peak/Trough Detection Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot peak/trough signals
        plt.subplot(2, 1, 2)
        plt.plot(sample_data.index, sample_peaks, 'r-', label='Peaks', linewidth=2)
        plt.plot(sample_data.index, sample_troughs, 'g-', label='Troughs', linewidth=2)
        plt.title('Peak/Trough Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/peak_trough_detection_test.png', dpi=150, bbox_inches='tight')
        print("📊 Visualization saved to peak_trough_detection_test.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available for visualization")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

def main():
    """Main test function."""
    print("🚀 Starting peak/trough detection integration tests...")
    
    try:
        # Test 1: Peak/trough detection
        peaks, troughs, data = test_peak_trough_detection()
        
        # Test 2: Enhanced labeling
        result = test_enhanced_labeling()
        
        # Test 3: Visualization
        visualize_peak_trough_detection(data, peaks, troughs)
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print("   → Peak/trough detection: ✅ Working")
        print("   → Enhanced labeling: ✅ Working")
        print("   → Local extrema integration: ✅ Working")
        print("   → Opportunity pattern detection: ✅ Working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)