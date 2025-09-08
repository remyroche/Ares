#!/usr/bin/env python3
"""
Test script to verify that step02_5 reporting functionality works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_step02_5_reporting():
    """Test step02_5 reporting functionality."""
    print("🧪 Testing step02_5 reporting functionality...")

    try:
        # Import step02_5 components
        from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SROptimizationStep
        from src.training.steps.data_collection.data_preparation.step02_5_financial_logging import Step02_5FinancialLogger
        from src.training.reports import save_training_report

        print("✅ Imports successful")

        # Create mock data similar to what step02_5 would generate
        mock_sr_levels = {
            'support_levels': [{'price': 25000, 'strength': 0.8}],
            'resistance_levels': [{'price': 26000, 'strength': 0.7}]
        }

        mock_ml_results = {
            'direction_accuracy': 0.85,
            'volatility_mae': 0.001
        }

        mock_execution_data = {
            'execution_time': 120.5,
            'memory_usage': 500,
            'cpu_usage': 80,
            'function_calls': 1000
        }

        # Test basic save_training_report function
        print("🧪 Testing basic save_training_report...")
        basic_result = save_training_report(
            data={'test': 'basic_report_data'},
            step_name='step02_5_sr_optimization',
            report_type='test_basic',
            symbol='ETHUSDT',
            timeframe='30m',
            file_format='json'
        )

        if basic_result:
            print(f"✅ Basic report saved: {basic_result}")
        else:
            print("❌ Basic report saving failed")
            return False

        # Test enhanced reporter
        print("🧪 Testing enhanced reporter...")
        enhanced_reporter = Step02_5FinancialLogger(
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='30m'
        )

        # Log financial metrics
        enhanced_reporter.log_step_execution(
            sr_levels=mock_sr_levels,
            ml_results=mock_ml_results,
            execution_data=mock_execution_data,
            data=None  # No data for this test
        )

        print("✅ Financial metrics logged successfully")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_step02_5_reporting())
    if success:
        print("🎉 Step02_5 reporting functionality test PASSED")
        sys.exit(0)
    else:
        print("💥 Step02_5 reporting functionality test FAILED")
        sys.exit(1)
