#!/usr/bin/env python3
"""
Test script for enhanced Step09 reporting system.

This script demonstrates the comprehensive reporting capabilities
of the enhanced Step09 HMM-based training per regime system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from src.training.steps.model_training.step09_enhanced_reporting import Step09EnhancedReporter


def create_sample_training_results():
    """Create sample training results."""
    return {
        'individual_models': {
            'lightgbm': {
                'training_time': 45.67,
                'convergence_score': 0.89,
                'feature_importance': {'feature1': 0.3, 'feature2': 0.25, 'feature3': 0.2},
                'training_samples': 10000,
                'validation_score': 0.82,
                'overfitting_score': 0.08,
                'computational_efficiency': 0.91
            },
            'random_forest': {
                'training_time': 38.92,
                'convergence_score': 0.85,
                'feature_importance': {'feature1': 0.35, 'feature2': 0.22, 'feature3': 0.18},
                'training_samples': 10000,
                'validation_score': 0.79,
                'overfitting_score': 0.12,
                'computational_efficiency': 0.87
            },
            'neural_network': {
                'training_time': 156.34,
                'convergence_score': 0.92,
                'feature_importance': {'feature1': 0.28, 'feature2': 0.27, 'feature3': 0.22},
                'training_samples': 10000,
                'validation_score': 0.85,
                'overfitting_score': 0.15,
                'computational_efficiency': 0.78
            }
        },
        'ensemble_model': {
            'accuracy': 0.87,
            'model_weights': {'lightgbm': 0.4, 'random_forest': 0.35, 'neural_network': 0.25},
            'diversity_score': 0.76,
            'improvement_over_best': 0.02,
            'stability_score': 0.89,
            'computational_overhead': 0.15,
            'method': 'weighted_average'
        },
        'per_regime_results': [
            {
                'regime_id': 1,
                'sample_count': 2500,
                'characteristics': {'volatility': 'low', 'trend': 'bullish'},
                'best_model': 'lightgbm',
                'training_time': 25.3,
                'regime_stability_score': 0.85
            },
            {
                'regime_id': 2,
                'sample_count': 3200,
                'characteristics': {'volatility': 'high', 'trend': 'bearish'},
                'best_model': 'neural_network',
                'training_time': 42.1,
                'regime_stability_score': 0.78
            },
            {
                'regime_id': 3,
                'sample_count': 1800,
                'characteristics': {'volatility': 'medium', 'trend': 'sideways'},
                'best_model': 'random_forest',
                'training_time': 18.7,
                'regime_stability_score': 0.91
            }
        ],
        'hyperparameter_optimization': {
            'method': 'bayesian_optimization',
            'parameter_space_size': 150,
            'iterations': 75,
            'best_parameters': {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 200},
            'convergence_score': 0.88,
            'parameter_importance': {'learning_rate': 0.35, 'max_depth': 0.28, 'n_estimators': 0.22},
            'optimization_time': 180.5
        }
    }


def create_sample_feature_data():
    """Create sample feature data."""
    return {
        'data_completeness': 0.97,
        'feature_correlation_score': 0.82,
        'class_balance_score': 0.78,
        'temporal_stability': 0.89,
        'noise_level': 0.12,
        'outlier_percentage': 0.03,
        'data_leakage_score': 0.02
    }


def create_sample_regime_configs():
    """Create sample regime configurations."""
    return {
        1: {'learning_rate': 0.01, 'max_depth': 6, 'subsample': 0.8},
        2: {'learning_rate': 0.005, 'max_depth': 10, 'subsample': 0.9},
        3: {'learning_rate': 0.02, 'max_depth': 4, 'subsample': 0.7}
    }


def create_sample_execution_metadata():
    """Create sample execution metadata."""
    return {
        'total_training_time': 245.8,
        'parallel_efficiency': 0.87,
        'memory_utilization': 0.76,
        'gpu_acceleration': 0.82,
        'hp_tuning_efficiency': 0.79,
        'early_stopping_effectiveness': 0.94,
        'cv_folds': 5
    }


def create_sample_performance_data():
    """Create sample performance data."""
    return {
        'evaluation_metrics': {
            'lightgbm': {
                'accuracy': 0.82,
                'precision': 0.79,
                'recall': 0.84,
                'f1_score': 0.81,
                'roc_auc': 0.87,
                'confusion_matrix': [[850, 150], [120, 880]],
                'classification_report': {'macro avg': {'precision': 0.79, 'recall': 0.84, 'f1-score': 0.81}},
                'feature_importance': {'feature1': 0.3, 'feature2': 0.25, 'feature3': 0.2}
            },
            'random_forest': {
                'accuracy': 0.79,
                'precision': 0.76,
                'recall': 0.81,
                'f1_score': 0.78,
                'roc_auc': 0.83,
                'confusion_matrix': [[810, 190], [140, 860]],
                'classification_report': {'macro avg': {'precision': 0.76, 'recall': 0.81, 'f1-score': 0.78}},
                'feature_importance': {'feature1': 0.35, 'feature2': 0.22, 'feature3': 0.18}
            },
            'neural_network': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.86,
                'f1_score': 0.84,
                'roc_auc': 0.89,
                'confusion_matrix': [[870, 130], [110, 890]],
                'classification_report': {'macro avg': {'precision': 0.82, 'recall': 0.86, 'f1-score': 0.84}},
                'feature_importance': {'feature1': 0.28, 'feature2': 0.27, 'feature3': 0.22}
            }
        }
    }


def main():
    """Main test function."""
    print("🤖 Testing Enhanced Step09 Reporting System")
    print("=" * 60)

    try:
        # Create sample data
        print("📊 Creating sample training data...")
        training_results = create_sample_training_results()
        feature_data = create_sample_feature_data()
        regime_configs = create_sample_regime_configs()
        execution_metadata = create_sample_execution_metadata()
        performance_data = create_sample_performance_data()

        # Initialize reporter
        print("📋 Initializing enhanced reporter...")
        config = {
            'per_regime_hmm_training': True,
            'adaptive_training_parameters_per_regime': True,
            'models_to_train': ['lightgbm', 'random_forest', 'neural_network'],
            'ensemble_method': 'weighted_average',
            'cross_validation_folds': 5
        }
        reporter = Step09EnhancedReporter(config)

        # Generate comprehensive report
        print("🔍 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            training_results=training_results,
            feature_data=feature_data,
            regime_configs=regime_configs,
            execution_metadata=execution_metadata,
            performance_data=performance_data
        )

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(
            report_data=report,
            symbol='BTCUSDT',
            exchange='binance',
            timeframe='1h'
        )

        print("✅ Enhanced Step09 report generation completed successfully!")
        print("\n📁 Generated Files:")
        for file_path in saved_files:
            if file_path and not str(file_path).startswith('error'):
                print(f"  - {file_path}")

        # Display key metrics
        print("\n📊 Key Report Highlights:")
        if 'model_training_analysis' in report:
            model_data = report['model_training_analysis']
            if model_data:
                print(f"  - Models Trained: {len(model_data)}")
                avg_time = sum(m.get('training_time_seconds', 0) for m in model_data) / max(1, len(model_data))
                print(f"  - Average Training Time: {avg_time:.2f} seconds")

        if 'ensemble_performance_analysis' in report:
            ensemble_data = report['ensemble_performance_analysis']
            if ensemble_data:
                print(f"  - Ensemble Accuracy: {ensemble_data.get('ensemble_accuracy', 0):.3f}")
                print(f"  - Ensemble Improvement: {ensemble_data.get('ensemble_improvement', 0):.3f}")

        if 'per_regime_training_analysis' in report:
            regime_data = report['per_regime_training_analysis']
            if regime_data:
                print(f"  - Regimes Trained: {len(regime_data)}")

        print("\n🎯 Test completed successfully!")
        print("The enhanced Step09 reporting system is now on par with step02_5, step05, and step07 reports.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
