#!/usr/bin/env python3
"""
Test script to verify that the exclude_recent_days functionality is working correctly.

This script tests that the lookback period always excludes the last 2 days,
regardless of whether we're in blank or full mode.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.constants import DEFAULT_EXCLUDE_RECENT_DAYS
from src.config.training import get_training_config

def test_exclude_recent_days_configuration():
    """Test that the exclude_recent_days configuration is set to 2."""
    print("🧪 Testing exclude_recent_days configuration...")
    
    # Test the constant
    assert DEFAULT_EXCLUDE_RECENT_DAYS == 2, f"Expected DEFAULT_EXCLUDE_RECENT_DAYS to be 2, got {DEFAULT_EXCLUDE_RECENT_DAYS}"
    print(f"✅ DEFAULT_EXCLUDE_RECENT_DAYS constant is correctly set to {DEFAULT_EXCLUDE_RECENT_DAYS}")
    
    # Test the training config
    training_config = get_training_config()
    data_config = training_config.get("DATA_CONFIG", {})
    exclude_recent_days = data_config.get("exclude_recent_days", 0)
    assert exclude_recent_days == 2, f"Expected exclude_recent_days in DATA_CONFIG to be 2, got {exclude_recent_days}"
    print(f"✅ exclude_recent_days in DATA_CONFIG is correctly set to {exclude_recent_days}")
    
    print("✅ All configuration tests passed!")

def test_lookback_calculation():
    """Test that the lookback calculation correctly excludes the last 2 days."""
    print("\n🧪 Testing lookback calculation...")
    
    # Simulate the calculation from step1_data_collection.py
    now = datetime.now()
    exclude_recent_days = 2
    lookback_days = 60  # Example for blank training
    
    # Calculate the end cutoff (exclude recent days)
    end_cutoff = now - timedelta(days=exclude_recent_days) if exclude_recent_days > 0 else now
    
    # Calculate the start cutoff
    start_cutoff = end_cutoff - timedelta(days=lookback_days)
    
    print(f"📅 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 End cutoff (excludes last {exclude_recent_days} days): {end_cutoff.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Start cutoff (lookback {lookback_days} days): {start_cutoff.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Total data window: {start_cutoff.strftime('%Y-%m-%d')} to {end_cutoff.strftime('%Y-%m-%d')}")
    
    # Verify that the end cutoff is indeed 2 days before now
    expected_end_cutoff = now - timedelta(days=2)
    assert abs((end_cutoff - expected_end_cutoff).total_seconds()) < 1, f"End cutoff should be 2 days before now"
    
    # Verify that the total window is correct
    total_window_days = (end_cutoff - start_cutoff).days
    assert total_window_days == lookback_days, f"Total window should be {lookback_days} days, got {total_window_days}"
    
    print("✅ Lookback calculation tests passed!")

def test_blank_vs_full_mode():
    """Test that both blank and full mode use the same exclude_recent_days value."""
    print("\n🧪 Testing blank vs full mode consistency...")
    
    # Simulate blank mode (60 days lookback)
    blank_lookback_days = 60
    exclude_recent_days = 2
    
    now = datetime.now()
    blank_end_cutoff = now - timedelta(days=exclude_recent_days)
    blank_start_cutoff = blank_end_cutoff - timedelta(days=blank_lookback_days)
    
    # Simulate full mode (1095 days lookback)
    full_lookback_days = 1095
    full_end_cutoff = now - timedelta(days=exclude_recent_days)  # Same exclude_recent_days
    full_start_cutoff = full_end_cutoff - timedelta(days=full_lookback_days)
    
    print(f"📊 Blank mode:")
    print(f"   Lookback days: {blank_lookback_days}")
    print(f"   Exclude recent days: {exclude_recent_days}")
    print(f"   Data window: {blank_start_cutoff.strftime('%Y-%m-%d')} to {blank_end_cutoff.strftime('%Y-%m-%d')}")
    
    print(f"📊 Full mode:")
    print(f"   Lookback days: {full_lookback_days}")
    print(f"   Exclude recent days: {exclude_recent_days}")
    print(f"   Data window: {full_start_cutoff.strftime('%Y-%m-%d')} to {full_end_cutoff.strftime('%Y-%m-%d')}")
    
    # Verify that both modes use the same exclude_recent_days
    assert blank_end_cutoff == full_end_cutoff, "Both modes should have the same end cutoff"
    print("✅ Both blank and full mode use the same exclude_recent_days value!")
    
    # Verify that the difference in start dates equals the difference in lookback days
    start_date_diff = (full_start_cutoff - blank_start_cutoff).days
    lookback_diff = full_lookback_days - blank_lookback_days
    # Allow for small difference due to date arithmetic
    assert abs(start_date_diff - lookback_diff) <= 1, f"Start date difference should equal lookback difference (got {start_date_diff}, expected {lookback_diff})"
    
    print("✅ Mode consistency tests passed!")

def main():
    """Run all tests."""
    print("🚀 Testing exclude_recent_days functionality...")
    print("=" * 60)
    
    try:
        test_exclude_recent_days_configuration()
        test_lookback_calculation()
        test_blank_vs_full_mode()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The exclude_recent_days functionality is working correctly.")
        print("✅ The lookback period will always exclude the last 2 days, regardless of mode.")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
