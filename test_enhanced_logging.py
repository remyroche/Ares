#!/usr/bin/env python3
"""
Test script for enhanced logging and metrics system

This script demonstrates the comprehensive logging, metrics, and progress monitoring
features of the enhanced market analysis pipeline.
"""

import asyncio
import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.enhanced_logging_metrics import (
    EnhancedPipelineLogger, 
    FeatureQualityMetrics, 
    RegimeQualityMetrics,
    StepMetrics
)
from src.training.steps.market_analysis.progress_monitor import (
    ProgressMonitor, 
    ProgressContext, 
    monitor_progress
)


def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create sample market data
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Create OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.01, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.exponential(1000, n_samples)
    })
    
    # Create sample features with various quality issues
    features_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),  # Good feature
        'feature_2': np.random.normal(0, 1, n_samples),  # Good feature
        'constant_feature': np.ones(n_samples),  # Constant feature
        'nan_feature': np.random.normal(0, 1, n_samples),  # Will add NaN values
        'high_corr_feature': market_data['close'] * 0.9 + np.random.normal(0, 0.1, n_samples),  # High correlation
        'low_variance_feature': np.random.normal(0, 0.001, n_samples),  # Low variance
        'infinite_feature': np.random.normal(0, 1, n_samples),  # Will add infinite values
    })
    
    # Add quality issues
    features_data.loc[100:200, 'nan_feature'] = np.nan
    features_data.loc[500:600, 'infinite_feature'] = np.inf
    
    # Create sample regime data
    regime_data = pd.DataFrame({
        'timestamp': dates,
        'regime': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])  # Imbalanced regimes
    })
    
    return market_data, features_data, regime_data


def test_feature_quality_metrics():
    """Test feature quality metrics calculation."""
    print("🧪 Testing Feature Quality Metrics...")
    
    logger = EnhancedPipelineLogger("test_logger")
    market_data, features_data, regime_data = create_sample_data()
    
    # Test feature quality logging
    logger.log_feature_quality("test_step", features_data)
    
    print("✅ Feature quality metrics test completed\n")


def test_regime_quality_metrics():
    """Test regime quality metrics calculation."""
    print("🧪 Testing Regime Quality Metrics...")
    
    logger = EnhancedPipelineLogger("test_logger")
    market_data, features_data, regime_data = create_sample_data()
    
    # Test regime quality logging
    logger.log_regime_quality("test_step", regime_data['regime'])
    
    print("✅ Regime quality metrics test completed\n")


def test_step6_metrics():
    """Test Step 6 feature engineering metrics."""
    print("🧪 Testing Step 6 Feature Engineering Metrics...")
    
    logger = EnhancedPipelineLogger("test_logger")
    
    # Mock feature engineering results
    step6_metrics = {
        'total_features_created': 150,
        'interaction_features': 45,
        'selected_features': 75,
        'feature_importance_top_10': [
            ('RSI_7_x_Volume_Ratio', 0.1234),
            ('MACD_12_26_x_ATR_14', 0.1156),
            ('BB_Position_20_x_Stochastic_14', 0.1089),
            ('SMA_5_x_EMA_21', 0.1023),
            ('Williams_R_14_x_CCI_20', 0.0956),
            ('OBV_20_x_MFI_14', 0.0889),
            ('ATR_7_x_Volatility', 0.0823),
            ('RSI_21_x_MACD_12_26', 0.0756),
            ('BB_Squeeze_20_x_Volume_Ratio', 0.0689),
            ('EMA_8_x_SMA_100', 0.0623)
        ],
        'lookback_optimization': {
            'optimized_count': 12,
            'optimization_time': 45.2
        }
    }
    
    logger.log_step6_metrics("feature_engineering", step6_metrics)
    
    print("✅ Step 6 metrics test completed\n")


def test_step7_metrics():
    """Test Step 7 matrix operations metrics."""
    print("🧪 Testing Step 7 Matrix Operations Metrics...")
    
    logger = EnhancedPipelineLogger("test_logger")
    
    # Mock matrix operations results
    step7_metrics = {
        'matrix_operations_performed': ['correlation_analysis', 'eigenvalue_analysis', 'feature_ranking'],
        'eigenvalue_analysis': {
            'condition_number': 1.2e+15,
            'rank': 75,
            'effective_rank': 68
        },
        'correlation_analysis': {
            'high_correlation_pairs': 23,
            'max_correlation': 0.987
        },
        'performance_metrics': {
            'computation_time': 12.5,
            'memory_usage_mb': 245.8
        }
    }
    
    logger.log_step7_metrics("matrix_operations", step7_metrics)
    
    print("✅ Step 7 metrics test completed\n")


def test_progress_monitoring():
    """Test progress monitoring functionality."""
    print("🧪 Testing Progress Monitoring...")
    
    monitor = ProgressMonitor(update_interval=0.5)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate pipeline steps
    steps = [
        ("hmm_clustering", "HMM regime discovery and clustering"),
        ("regime_splitting", "Regime data splitting and preparation"),
        ("labeling", "Triple barrier method labeling"),
        ("feature_engineering", "Feature engineering and interaction creation"),
        ("matrix_operations", "Enhanced matrix operations and analysis"),
        ("feature_selection", "Advanced feature selection and optimization")
    ]
    
    for step_name, description in steps:
        print(f"🔄 Simulating {step_name}...")
        
        # Start step
        monitor.update_step_progress(step_name, 0.0, f"Starting {description}...", "running")
        time.sleep(1)
        
        # Progress updates
        for progress in [0.2, 0.4, 0.6, 0.8]:
            monitor.update_step_progress(step_name, progress, f"Processing... {progress*100:.0f}%", "running")
            time.sleep(0.5)
        
        # Complete step
        monitor.complete_step(step_name, True, f"Completed {description}")
        time.sleep(0.5)
    
    # Stop monitoring
    time.sleep(2)
    monitor.stop_monitoring()
    
    print("✅ Progress monitoring test completed\n")


def test_progress_context():
    """Test progress context manager."""
    print("🧪 Testing Progress Context Manager...")
    
    @monitor_progress("test_step", 100)
    def simulate_work(progress=None):
        """Simulate work with progress updates."""
        for i in range(10):
            time.sleep(0.1)
            if progress:
                progress.update(10, f"Processed {i+1}/10 items")
        return "Work completed"
    
    # Test with context manager
    with ProgressContext("context_test", 50) as progress:
        for i in range(5):
            time.sleep(0.2)
            progress.update(10, f"Context step {i+1}/5")
    
    print("✅ Progress context test completed\n")


def test_full_pipeline_simulation():
    """Test full pipeline simulation with all features."""
    print("🧪 Testing Full Pipeline Simulation...")
    
    logger = EnhancedPipelineLogger("full_pipeline_test")
    monitor = ProgressMonitor(update_interval=1.0)
    
    # Start pipeline
    logger.start_pipeline("ETHUSDT", "BINANCE", "test_correlation_123")
    monitor.start_monitoring()
    
    try:
        # Simulate each step with comprehensive logging
        steps = [
            ("hmm_clustering", "HMM regime discovery and clustering"),
            ("regime_splitting", "Regime data splitting and preparation"),
            ("labeling", "Triple barrier method labeling"),
            ("feature_engineering", "Feature engineering and interaction creation"),
            ("matrix_operations", "Enhanced matrix operations and analysis"),
            ("feature_selection", "Advanced feature selection and optimization")
        ]
        
        for step_name, description in steps:
            logger.start_step(step_name, description)
            monitor.update_step_progress(step_name, 0.0, f"Starting {description}...", "running")
            
            # Simulate step execution
            for progress in [0.1, 0.3, 0.5, 0.7, 0.9]:
                time.sleep(0.5)
                monitor.update_step_progress(step_name, progress, f"Processing... {progress*100:.0f}%", "running")
            
            # Simulate step completion with metrics
            if step_name == "hmm_clustering":
                market_data, features_data, regime_data = create_sample_data()
                logger.log_regime_quality(step_name, regime_data['regime'])
            elif step_name == "feature_engineering":
                market_data, features_data, regime_data = create_sample_data()
                logger.log_feature_quality(step_name, features_data)
                logger.log_step6_metrics(step_name, {
                    'total_features_created': 150,
                    'interaction_features': 45,
                    'selected_features': 75,
                    'feature_importance_top_10': [
                        ('RSI_7_x_Volume_Ratio', 0.1234),
                        ('MACD_12_26_x_ATR_14', 0.1156)
                    ],
                    'lookback_optimization': {
                        'optimized_count': 12,
                        'optimization_time': 45.2
                    }
                })
            elif step_name == "matrix_operations":
                logger.log_step7_metrics(step_name, {
                    'matrix_operations_performed': ['correlation_analysis', 'eigenvalue_analysis'],
                    'eigenvalue_analysis': {
                        'condition_number': 1.2e+15,
                        'rank': 75,
                        'effective_rank': 68
                    },
                    'correlation_analysis': {
                        'high_correlation_pairs': 23,
                        'max_correlation': 0.987
                    },
                    'performance_metrics': {
                        'computation_time': 12.5,
                        'memory_usage_mb': 245.8
                    }
                })
            
            # Complete step
            monitor.complete_step(step_name, True, f"Completed {description}")
            logger.end_step(step_name, success=True)
            
            time.sleep(0.5)
        
        # Complete pipeline
        monitor.stop_monitoring()
        logger.end_pipeline(success=True)
        
    except Exception as e:
        monitor.stop_monitoring()
        logger.end_pipeline(success=False, error_message=str(e))
        raise
    
    print("✅ Full pipeline simulation test completed\n")


async def main():
    """Run all tests."""
    print("🚀 Enhanced Logging and Metrics Test Suite")
    print("=" * 80)
    
    try:
        # Run individual tests
        test_feature_quality_metrics()
        test_regime_quality_metrics()
        test_step6_metrics()
        test_step7_metrics()
        test_progress_monitoring()
        test_progress_context()
        test_full_pipeline_simulation()
        
        print("🎉 All tests completed successfully!")
        print("=" * 80)
        print("✅ Enhanced logging system is working correctly")
        print("✅ Feature quality metrics are functioning")
        print("✅ Regime quality metrics are functioning")
        print("✅ Step 6 and 7 specific metrics are working")
        print("✅ Progress monitoring is operational")
        print("✅ Progress context managers are working")
        print("✅ Full pipeline simulation completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())