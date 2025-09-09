#!/usr/bin/env python3
"""
Test script to verify financial metrics logger functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.financial_metrics_logger import get_financial_metrics_logger

def test_financial_logger():
    """Test the financial metrics logger."""
    print("🧪 Testing Financial Metrics Logger...")

    # Get the logger
    logger = get_financial_metrics_logger()

    print(f"📁 Logger directory: {logger.log_dir}")
    print(f"📁 Logger directory exists: {logger.log_dir.exists()}")
    print(f"📁 Logger directory absolute: {logger.log_dir.absolute()}")

    # Test logging
    print("📝 Testing log_step_start...")
    logger.log_step_start("TestStep", "ETHUSDT", "BINANCE", "15m")

    print("📝 Testing log_financial_metric...")
    logger.log_financial_metric(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="15m",
        metric_name="test_accuracy",
        metric_value=0.95,
        metric_type="performance",
        step_name="TestStep"
    )

    print("📝 Testing log_step_end...")
    logger.log_step_end("TestStep", "ETHUSDT", "BINANCE", "15m", success=True)

    print("✅ Test completed!")

if __name__ == "__main__":
    test_financial_logger()
