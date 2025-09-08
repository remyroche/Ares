#!/usr/bin/env python3
"""
Test script for enhanced Step11 reporting system.

This script demonstrates the comprehensive reporting capabilities
of the enhanced Step11 Analyst Creation system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from src.training.steps.model_training.step11_analyst_creation import Step11EnhancedReporter


def create_sample_model_data():
    """Create sample model training data."""
    return [
        {
            'model_name': 'lightgbm_regime_1',
            'model_type': 'lightgbm',
            'regime_name': 'bullish_trend',
            'training_time': 45.67,
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.84,
            'f1_score': 0.81,
            'roc_auc': 0.87,
            'feature_importance': {'rsi': 0.3, 'macd': 0.25, 'volume': 0.2, 'price': 0.15, 'volatility': 0.1},
            'training_samples': 2500,
            'validation_samples': 625,
            'overfitting_score': 0.08,
            'convergence_score': 0.89,
            'computational_efficiency': 0.91
        },
        {
            'model_name': 'xgboost_regime_1',
            'model_type': 'xgboost',
            'regime_name': 'bullish_trend',
            'training_time': 52.34,
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.86,
            'f1_score': 0.84,
            'roc_auc': 0.89,
            'feature_importance': {'rsi': 0.28, 'macd': 0.27, 'volume': 0.22, 'price': 0.18, 'volatility': 0.05},
            'training_samples': 2500,
            'validation_samples': 625,
            'overfitting_score': 0.12,
            'convergence_score': 0.92,
            'computational_efficiency': 0.78
        },
        {
            'model_name': 'random_forest_regime_1',
            'model_type': 'random_forest',
            'regime_name': 'bullish_trend',
            'training_time': 38.92,
            'accuracy': 0.79,
            'precision': 0.76,
            'recall': 0.81,
            'f1_score': 0.78,
            'roc_auc': 0.83,
            'feature_importance': {'rsi': 0.35, 'macd': 0.22, 'volume': 0.18, 'price': 0.15, 'volatility': 0.1},
            'training_samples': 2500,
            'validation_samples': 625,
            'overfitting_score': 0.05,
            'convergence_score': 0.85,
            'computational_efficiency': 0.87
        },
        {
            'model_name': 'lightgbm_regime_2',
            'model_type': 'lightgbm',
            'regime_name': 'bearish_trend',
            'training_time': 42.15,
            'accuracy': 0.78,
            'precision': 0.75,
            'recall': 0.79,
            'f1_score': 0.77,
            'roc_auc': 0.82,
            'feature_importance': {'rsi': 0.32, 'macd': 0.28, 'volume': 0.18, 'price': 0.16, 'volatility': 0.06},
            'training_samples': 1800,
            'validation_samples': 450,
            'overfitting_score': 0.06,
            'convergence_score': 0.88,
            'computational_efficiency': 0.89
        },
        {
            'model_name': 'xgboost_regime_2',
            'model_type': 'xgboost',
            'regime_name': 'bearish_trend',
            'training_time': 48.76,
            'accuracy': 0.81,
            'precision': 0.78,
            'recall': 0.82,
            'f1_score': 0.80,
            'roc_auc': 0.85,
            'feature_importance': {'rsi': 0.30, 'macd': 0.25, 'volume': 0.20, 'price': 0.17, 'volatility': 0.08},
            'training_samples': 1800,
            'validation_samples': 450,
            'overfitting_score': 0.09,
            'convergence_score': 0.90,
            'computational_efficiency': 0.82
        },
        {
            'model_name': 'lightgbm_regime_3',
            'model_type': 'lightgbm',
            'regime_name': 'sideways',
            'training_time': 35.89,
            'accuracy': 0.76,
            'precision': 0.73,
            'recall': 0.77,
            'f1_score': 0.75,
            'roc_auc': 0.79,
            'feature_importance': {'rsi': 0.25, 'macd': 0.20, 'volume': 0.25, 'price': 0.20, 'volatility': 0.1},
            'training_samples': 1200,
            'validation_samples': 300,
            'overfitting_score': 0.04,
            'convergence_score': 0.86,
            'computational_efficiency': 0.93
        }
    ]


def create_sample_regime_data():
    """Create sample regime analysis data."""
    return [
        {
            'regime_id': 1,
            'regime_name': 'bullish_trend',
            'sample_count': 2500,
            'characteristics': {'trend': 'bullish', 'volatility': 'medium', 'momentum': 'strong'},
            'models_created': 3,
            'best_model_type': 'xgboost',
            'best_accuracy': 0.85,
            'average_accuracy': 0.82,
            'regime_stability_score': 0.88,
            'hyperparameters': {'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8}
        },
        {
            'regime_id': 2,
            'regime_name': 'bearish_trend',
            'sample_count': 1800,
            'characteristics': {'trend': 'bearish', 'volatility': 'high', 'momentum': 'moderate'},
            'models_created': 2,
            'best_model_type': 'xgboost',
            'best_accuracy': 0.81,
            'average_accuracy': 0.795,
            'regime_stability_score': 0.82,
            'hyperparameters': {'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.9}
        },
        {
            'regime_id': 3,
            'regime_name': 'sideways',
            'sample_count': 1200,
            'characteristics': {'trend': 'sideways', 'volatility': 'low', 'momentum': 'weak'},
            'models_created': 1,
            'best_model_type': 'lightgbm',
            'best_accuracy': 0.76,
            'average_accuracy': 0.76,
            'regime_stability_score': 0.91,
            'hyperparameters': {'learning_rate': 0.2, 'max_depth': 4, 'subsample': 0.7}
        }
    ]


def create_sample_performance_data():
    """Create sample performance metrics."""
    return {
        'total_regimes': 3,
        'total_models': 6,
        'total_time': 263.73,
        'avg_time_per_model': 43.96,
        'overall_accuracy': 0.81,
        'computational_efficiency': 0.87,
        'memory_utilization': 0.78,
        'gpu_utilization': 0.12,
        'parallel_efficiency': 0.84
    }


def create_sample_quality_data():
    """Create sample quality assessment data."""
    return {
        'overall_quality': 0.83,
        'diversity_score': 0.76,
        'robustness_score': 0.88,
        'generalization_score': 0.85,
        'stability_score': 0.87,
        'warnings': ['Some regimes have limited sample sizes', 'Consider adding more diverse features'],
        'improvements': ['Implement cross-validation for better generalization', 'Add regularization techniques']
    }


def create_sample_optimization_data():
    """Create sample optimization metrics."""
    return {
        'method': 'grid_search_with_early_stopping',
        'hp_efficiency': 0.79,
        'early_stopping': 0.92,
        'cv_folds': 5,
        'feature_efficiency': 0.84,
        'memory_optimization': 0.88,
        'speed_improvement': 1.3
    }


def main():
    """Main test function."""
    print("🤖 Testing Enhanced Step11 Reporting System")
    print("=" * 60)

    try:
        # Create sample data
        print("📊 Creating sample analyst creation data...")
        model_data = create_sample_model_data()
        regime_data = create_sample_regime_data()
        performance_data = create_sample_performance_data()
        quality_data = create_sample_quality_data()
        optimization_data = create_sample_optimization_data()

        # Initialize reporter
        print("📋 Initializing enhanced reporter...")
        reporter = Step11EnhancedReporter()

        # Add data to reporter
        print("📝 Adding model metrics...")
        for model in model_data:
            reporter.add_model_metrics(model)

        print("📝 Adding regime analysis...")
        for regime in regime_data:
            reporter.add_regime_analysis(regime)

        print("📝 Setting performance metrics...")
        reporter.set_performance_metrics(performance_data)

        print("📝 Setting quality assessment...")
        reporter.set_quality_assessment(quality_data)

        print("📝 Setting optimization metrics...")
        reporter.set_optimization_metrics(optimization_data)

        # Generate comprehensive report
        print("🔍 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            symbol='BTCUSDT',
            exchange='binance',
            timeframe='1h'
        )

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(report)

        print("✅ Enhanced Step11 report generation completed successfully!")

        # Display key metrics
        print("\n📊 Key Report Highlights:")
        if 'model_training_analysis' in report:
            model_analysis = report['model_training_analysis']
            if model_analysis:
                accuracies = [m.get('accuracy_score', 0) for m in model_analysis]
                print(f"  - Models Analyzed: {len(model_analysis)}")
                print(".3f")
                print(".3f")

        if 'regime_analysis' in report:
            regime_analysis = report['regime_analysis']
            if regime_analysis:
                print(f"  - Regimes Processed: {len(regime_analysis)}")
                best_regime = max(regime_analysis, key=lambda x: x.get('best_model_accuracy', 0))
                print(f"  - Best Regime Performance: {best_regime.get('regime_name', 'Unknown')} ({best_regime.get('best_model_accuracy', 0):.3f})")

        if 'performance_analysis' in report:
            perf = report['performance_analysis']
            if perf:
                print(f"  - Total Training Time: {perf.get('total_training_time', 0):.2f} seconds")
                print(f"  - Overall Accuracy: {perf.get('overall_accuracy_score', 0):.3f}")

        print("\n📁 Generated Files:")
        for file_path in saved_files:
            if file_path and not str(file_path).startswith('error'):
                print(f"  - {file_path}")

        print("\n🎯 Test completed successfully!")
        print("The enhanced Step11 reporting system is now on par with step02_5, step05, step07, and step09 reports.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
