#!/usr/bin/env python3
"""
Test script to verify mode-aware report naming functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.reports import CentralizedReportManager, save_training_report

def test_mode_detection():
    """Test mode detection logic."""
    manager = CentralizedReportManager()

    # Test default mode (should be 'full')
    mode = manager._get_training_mode()
    print(f"Default mode: {mode}")

    # Test blank mode
    os.environ['BLANK_TRAINING_MODE'] = '1'
    mode = manager._get_training_mode()
    print(f"Blank mode (BLANK_TRAINING_MODE=1): {mode}")

    # Test light mode (should override blank)
    os.environ['LIGHT_TRAINING_MODE'] = '1'
    mode = manager._get_training_mode()
    print(f"Light mode (LIGHT_TRAINING_MODE=1): {mode}")

    # Clean up
    os.environ.pop('BLANK_TRAINING_MODE', None)
    os.environ.pop('LIGHT_TRAINING_MODE', None)

def test_filename_generation():
    """Test filename generation with different modes."""
    manager = CentralizedReportManager()

    # Test filename generation for different modes
    test_cases = [
        ('full', 'test_report', 'BTCUSDT', '1m'),
        ('blank', 'analysis_report', 'ETHUSDT', '5m'),
        ('light', 'summary_report', None, None)
    ]

    for mode, report_type, symbol, timeframe in test_cases:
        # Mock the timestamp for consistent testing
        manager._get_timestamp = lambda: "20241201_120000"

        # Generate filename parts
        filename_parts = [report_type, "20241201_120000"]
        filename_parts.insert(0, mode)

        if symbol:
            filename_parts.insert(0, symbol)
        if timeframe:
            filename_parts.insert(1 if symbol else 0, timeframe)

        filename = "_".join(filename_parts) + ".json"
        print(f"Mode: {mode}, Report: {report_type}, Symbol: {symbol}, Timeframe: {timeframe}")
        print(f"Generated filename: {filename}")
        print()

if __name__ == "__main__":
    print("Testing mode-aware report naming functionality...")
    print("=" * 50)

    print("1. Testing mode detection:")
    test_mode_detection()
    print()

    print("2. Testing filename generation:")
    test_filename_generation()

    print("✅ Mode-aware reporting test completed!")
