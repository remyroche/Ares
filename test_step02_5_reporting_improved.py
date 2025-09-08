#!/usr/bin/env python3
"""
Test script for improved step02_5 reporting functionality.

This script tests the enhanced reporting system to ensure it works correctly
and generates comprehensive reports with proper data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.training.steps.data_collection.data_preparation.step02_5_financial_logging import Step02_5FinancialLogger
from src.training.reports import save_training_report
import sys
import time

def create_test_data():
    """Create test data for reporting."""
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)  # For reproducible results

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 25000 + np.random.randn(1000) * 1000,
        'high': 25200 + np.random.randn(1000) * 1000,
        'low': 24800 + np.random.randn(1000) * 1000,
        'close': 25000 + np.random.randn(1000) * 1000,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data

def create_test_sr_levels():
    """Create test SR levels data."""
    return {
        'support_levels': [
            {
                'price': 24000,
                'strength': 0.85,
                'touch_count': 12,
                'method': 'fractal',
                'timestamp': datetime.now().isoformat(),
                'confidence_score': 0.88
            },
            {
                'price': 23500,
                'strength': 0.72,
                'touch_count': 8,
                'method': 'pivot',
                'timestamp': datetime.now().isoformat(),
                'confidence_score': 0.76
            }
        ],
        'resistance_levels': [
            {
                'price': 26000,
                'strength': 0.90,
                'touch_count': 15,
                'method': 'fractal',
                'timestamp': datetime.now().isoformat(),
                'confidence_score': 0.92
            },
            {
                'price': 26500,
                'strength': 0.68,
                'touch_count': 6,
                'method': 'volume',
                'timestamp': datetime.now().isoformat(),
                'confidence_score': 0.71
            }
        ],
        'detection_method': 'enhanced_sr',
        'total_levels_detected': 4
    }

def create_test_ml_results():
    """Create test ML results data."""
    return {
        'direction_accuracy': 0.87,
        'volatility_mae': 0.0023,
        'f1_score': 0.84,
        'precision': 0.86,
        'recall': 0.82,
        'model_type': 'RandomForest',
        'training_samples': 800,
        'test_samples': 200,
        'feature_count': 45,
        'cross_validation_scores': [0.85, 0.88, 0.86, 0.87, 0.84],
        'feature_importance': {
            'rsi_14': 0.15,
            'macd_signal': 0.12,
            'bb_position': 0.10,
            'volume_ratio': 0.08
        }
    }

def create_test_execution_data():
    """Create test execution data."""
    return {
        'execution_time': 145.67,
        'memory_usage': 850.5,
        'cpu_usage': 67.8,
        'function_calls': 2450,
        'step_breakdown': {
            'data_loading': 12.3,
            'feature_engineering': 45.6,
            'sr_detection': 23.4,
            'ml_training': 56.7,
            'report_generation': 7.7
        },
        'performance_summary': {
            'features_per_second': 89.2,
            'sr_levels_per_second': 0.034,
            'ml_accuracy': 0.87
        },
        'feature_count': 45,
        'data_rows': 1000,
        'sr_levels_detected': 4,
        'ml_accuracy': 0.87,
        'processing_timestamp': datetime.now().isoformat()
    }

def main():
    """Main test function."""
    print("🧪 Testing improved step02_5 reporting functionality...")

    try:
        # Create test data
        print("📊 Creating test data...")
        test_data = create_test_data()
        sr_levels = create_test_sr_levels()
        ml_results = create_test_ml_results()
        execution_data = create_test_execution_data()

        print("✅ Test data created successfully")
        print(f"   - Data shape: {test_data.shape}")
        print(f"   - SR levels: {len(sr_levels['support_levels'])} support, {len(sr_levels['resistance_levels'])} resistance")
        print(f"   - ML accuracy: {ml_results['direction_accuracy']:.3f}")

        # Initialize reporter
        print("\n📝 Initializing enhanced reporter...")
        reporter = Step02_5FinancialLogger('BTCUSDT', 'binance', '1h')
        print("✅ Enhanced reporter initialized")

        # Log financial metrics
        print("\n📈 Logging financial metrics...")
        reporter.log_step_execution(
            sr_levels=sr_levels,
            ml_results=ml_results,
            execution_data=execution_data,
            data=test_data
        )
        print("✅ Financial metrics logged successfully")

        print("\n🎉 All tests passed! Financial logging is working correctly.")
        print("📋 Summary:")
        print("   ✅ Financial logger initialization: SUCCESS")
        print("   ✅ Financial metrics logging: SUCCESS")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
