#!/usr/bin/env python3
"""
Example usage of the Financial Metrics Logger

This script demonstrates how to use the new financial metrics logger
throughout training steps instead of the current reporting system.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging

from src.utils.financial_metrics_logger import (
    get_financial_metrics_logger,
    setup_financial_metrics_logging,
    financial_metrics_context,
    log_return_metric,
    log_risk_metric,
    log_sharpe_ratio,
    log_drawdown_metric
)


def simulate_training_step_data():
    """Simulate some training step data for demonstration."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Simulate price data
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'returns': returns,
        'label': np.random.choice([-1, 0, 1], len(dates), p=[0.3, 0.4, 0.3])
    })
    
    return df


def example_basic_usage():
    """Example of basic financial metrics logging."""
    print("=== Basic Financial Metrics Logging Example ===")
    
    # Get the financial metrics logger
    financial_logger = get_financial_metrics_logger()
    
    # Log individual metrics
    log_return_metric(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1d",
        step_name="Step01_DataCollection",
        return_value=0.15  # 15% return
    )
    
    log_risk_metric(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1d",
        step_name="Step02_FeatureEngineering",
        risk_value=0.25,  # 25% volatility
        risk_type="volatility"
    )
    
    log_sharpe_ratio(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1d",
        step_name="Step03_ModelTraining",
        sharpe_value=1.8
    )
    
    log_drawdown_metric(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1d",
        step_name="Step04_Validation",
        drawdown_value=0.12  # 12% max drawdown
    )
    
    print("✅ Basic metrics logged successfully")


def example_context_manager_usage():
    """Example using the context manager for a complete training step."""
    print("\n=== Context Manager Usage Example ===")
    
    # Simulate a training step with context manager
    with financial_metrics_context("Step05_Labeling", "ETHUSDT", "binance", "1h"):
        financial_logger = get_financial_metrics_logger()
        
        # Simulate some processing
        print("Processing labeling step...")
        
        # Log various metrics during the step
        financial_logger.log_financial_metric(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1h",
            metric_name="label_quality_score",
            metric_value=0.85,
            metric_type="quality",
            step_name="Step05_Labeling"
        )
        
        financial_logger.log_financial_metric(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1h",
            metric_name="labeling_efficiency",
            metric_value=0.92,
            metric_type="performance",
            step_name="Step05_Labeling"
        )
        
        print("✅ Context manager example completed")


def example_comprehensive_performance_logging():
    """Example of comprehensive trading performance logging."""
    print("\n=== Comprehensive Performance Logging Example ===")
    
    financial_logger = get_financial_metrics_logger()
    
    # Simulate comprehensive performance data
    performance_data = {
        'total_return': 0.25,  # 25% total return
        'annualized_return': 0.28,  # 28% annualized
        'volatility': 0.18,  # 18% volatility
        'sharpe_ratio': 1.56,  # Sharpe ratio
        'sortino_ratio': 2.1,  # Sortino ratio
        'calmar_ratio': 1.8,  # Calmar ratio
        'max_drawdown': 0.08,  # 8% max drawdown
        'max_drawdown_duration': 45,  # 45 days
        'var_95': 0.03,  # 3% VaR 95%
        'cvar_95': 0.045,  # 4.5% CVaR 95%
        'win_rate': 0.65,  # 65% win rate
        'profit_factor': 1.8,  # Profit factor
        'avg_win': 0.025,  # 2.5% average win
        'avg_loss': 0.015,  # 1.5% average loss
        'largest_win': 0.08,  # 8% largest win
        'largest_loss': 0.05,  # 5% largest loss
        'total_trades': 150,
        'winning_trades': 98,
        'losing_trades': 52,
        'additional_metrics': {
            'regime_performance': {
                'bull_market': 0.32,
                'bear_market': 0.15,
                'sideways_market': 0.18
            }
        }
    }
    
    # Log comprehensive performance
    financial_logger.log_trading_performance(
        symbol="ADAUSDT",
        exchange="binance",
        timeframe="4h",
        step_name="Step06_Backtesting",
        performance_data=performance_data,
        regime_id="bull_market",
        model_version="v2.1.0",
        confidence_score=0.88
    )
    
    print("✅ Comprehensive performance logged successfully")


def example_model_performance_logging():
    """Example of model-specific performance logging."""
    print("\n=== Model Performance Logging Example ===")
    
    financial_logger = get_financial_metrics_logger()
    
    # Simulate model performance metrics
    model_metrics = {
        'accuracy': 0.78,
        'precision': 0.82,
        'recall': 0.75,
        'f1_score': 0.78,
        'auc_roc': 0.85,
        'log_loss': 0.45,
        'calibration_score': 0.92
    }
    
    # Log model performance
    financial_logger.log_model_performance(
        symbol="SOLUSDT",
        exchange="binance",
        timeframe="1d",
        step_name="Step07_ModelEvaluation",
        model_name="RandomForestClassifier",
        model_version="v1.5.2",
        performance_metrics=model_metrics,
        regime_id="high_volatility"
    )
    
    print("✅ Model performance logged successfully")


def example_custom_log_directory():
    """Example of setting up custom log directory."""
    print("\n=== Custom Log Directory Example ===")
    
    # Setup with custom directory
    custom_logger = setup_financial_metrics_logging(
        log_dir="custom_logs/financial_metrics",
        enable_console=True,
        enable_file=True,
        enable_csv=True,
        enable_json=True,
        max_file_size_mb=100,
        backup_count=5
    )
    
    # Log some metrics
    custom_logger.log_financial_metric(
        symbol="DOTUSDT",
        exchange="binance",
        timeframe="1d",
        metric_name="custom_metric",
        metric_value=0.95,
        metric_type="custom",
        step_name="CustomStep"
    )
    
    print("✅ Custom logger setup and usage completed")


def example_export_summary():
    """Example of exporting metrics summary."""
    print("\n=== Export Summary Example ===")
    
    financial_logger = get_financial_metrics_logger()
    
    # Export summary
    summary = financial_logger.export_metrics_summary(
        symbol="BTCUSDT",
        step_name="Step05_Labeling"
    )
    
    print("📊 Metrics Summary:")
    print(f"  - Total CSV files: {summary['summary_statistics']['total_csv_files']}")
    print(f"  - Total JSON files: {summary['summary_statistics']['total_json_files']}")
    print(f"  - Log directory size: {summary['summary_statistics']['log_directory_size_mb']:.2f} MB")
    print(f"  - Log directory: {summary['file_locations']['log_directory']}")
    
    print("✅ Summary export completed")


def main():
    """Main function to run all examples."""
    print("🚀 Financial Metrics Logger Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        example_basic_usage()
        example_context_manager_usage()
        example_comprehensive_performance_logging()
        example_model_performance_logging()
        example_custom_log_directory()
        example_export_summary()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\n📁 Check the 'logs/financial_metrics' directory for generated log files:")
        print("   - financial_metrics_*.log (main log files)")
        print("   - financial_metrics_*.csv (CSV exports)")
        print("   - financial_metrics_*.json (JSON exports)")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()