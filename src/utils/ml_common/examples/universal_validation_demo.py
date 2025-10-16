"""
Universal ML Validation Demonstration

Demonstrates the universal validation system for all ML models in the ml_common framework.
Shows how to integrate enhanced overfitting detection, temporal validation, and
timeframe configuration across different model types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Import universal validation components
from ..validation import (
    validate_ml_model,
    get_ml_validator,
    get_overfitting_detector,
    get_temporal_validator,
    create_time_series_split,
    UniversalMLValidationConfig,
    OverfittingConfig,
    TemporalValidationConfig
)
from ..config.universal_timeframe_config import (
    get_timeframe_config,
    get_timeframe_manager,
    validate_timeframe_consistency
)

# Import ML models for demonstration
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, n_features: int = 50,
                      time_series: bool = False) -> Dict[str, np.ndarray]:
    """Create sample data for demonstration."""
    np.random.seed(42)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create labels with some structure
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])

    # Add some structure to make it more realistic
    X[:, 0] += y * 0.5  # First feature correlates with labels
    X[:, 1] += np.random.normal(0, 0.1, n_samples)  # Add noise

    # Split data
    split_idx = int(0.7 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    result = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val
    }

    # Add timestamps for time series models
    if time_series:
        timestamps = np.arange(n_samples)
        result['timestamps'] = timestamps
        result['train_timestamps'] = timestamps[:split_idx]
        result['val_timestamps'] = timestamps[split_idx:]

    return result

def demonstrate_overfitting_detection():
    """Demonstrate overfitting detection for different model types."""
    print("🔍 Overfitting Detection Demonstration")
    print("=" * 50)

    # Create sample data
    data = create_sample_data(n_samples=1000, n_features=50)

    # Test different model types
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True)
    }

    # Get overfitting detector
    detector = get_overfitting_detector()

    results = {}

    for model_name, model in models.items():
        print(f"\n📊 Testing {model_name}...")

        # Train model
        model.fit(data['X_train'], data['y_train'])

        # Detect overfitting
        report = detector.detect_overfitting(
            train_predictions=model.predict(data['X_train']),
            val_predictions=model.predict(data['X_val']),
            train_labels=data['y_train'],
            val_labels=data['y_val'],
            train_probabilities=model.predict_proba(data['X_train']) if hasattr(model, 'predict_proba') else None,
            val_probabilities=model.predict_proba(data['X_val']) if hasattr(model, 'predict_proba') else None,
            model_name=model_name,
            model_type=model_name.lower(),
            fold_number=1
        )

        results[model_name] = report

        # Display results
        print(f"  Overfitting detected: {report.is_overfitting}")
        print(f"  Severity: {report.severity.upper()}")
        print(f"  Accuracy gap: {report.accuracy_gap:.4f}")
        print(f"  Confidence: {report.confidence_level:.2f}")

        if report.warnings:
            print(f"  Warnings: {len(report.warnings)}")
            for warning in report.warnings[:2]:  # Show first 2 warnings
                print(f"    - {warning}")

        if report.recommendations:
            print(f"  Recommendations: {len(report.recommendations)}")
            for rec in report.recommendations[:2]:  # Show first 2 recommendations
                print(f"    - {rec}")

    return results

def demonstrate_temporal_validation():
    """Demonstrate temporal validation for time series models."""
    print("\n⏰ Temporal Validation Demonstration")
    print("=" * 50)

    # Create time series data
    data = create_sample_data(n_samples=1000, n_features=30, time_series=True)

    # Get temporal validator
    validator = get_temporal_validator()

    # Test temporal validation
    temporal_report = validator.validate_temporal_split(
        X_train=data['X_train'],
        X_test=data['X_val'],
        y_train=data['y_train'],
        y_test=data['y_val'],
        timestamps=data['train_timestamps'],
        model_name="TimeSeriesModel",
        model_type="lstm"
    )

    print(f"Temporal order valid: {temporal_report.temporal_order_valid}")
    print(f"Leakage detected: {temporal_report.leakage_detected}")
    print(f"Validation score: {temporal_report.validation_score:.3f}")

    if temporal_report.warnings:
        print(f"Warnings: {len(temporal_report.warnings)}")
        for warning in temporal_report.warnings:
            print(f"  - {warning}")

    if temporal_report.recommendations:
        print(f"Recommendations: {len(temporal_report.recommendations)}")
        for rec in temporal_report.recommendations:
            print(f"  - {rec}")

    # Demonstrate temporal cross-validation
    print(f"\n📈 Temporal Cross-Validation:")
    tscv = create_time_series_split(n_splits=5, test_size=0.2, gap_size=1)

    model = RandomForestClassifier()
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data['X_train'], data['y_train'])):
        X_fold_train = data['X_train'][train_idx]
        X_fold_test = data['X_train'][test_idx]
        y_fold_train = data['y_train'][train_idx]
        y_fold_test = data['y_train'][test_idx]

        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_test, y_fold_test)
        fold_scores.append(score)

        print(f"  Fold {fold + 1}: {score:.3f}")

    print(f"  Mean CV Score: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    return temporal_report

def demonstrate_timeframe_configuration():
    """Demonstrate timeframe configuration management."""
    print("\n⏱️ Timeframe Configuration Demonstration")
    print("=" * 50)

    # Get timeframe configuration
    config = get_timeframe_config()
    manager = get_timeframe_manager()

    print(f"Primary timeframe: {config.primary_timeframe}")
    print(f"Supported timeframes: {config.supported_timeframes}")
    print(f"Cross-timeframe enabled: {config.enable_cross_timeframe_features}")
    print(f"Cross-timeframes: {config.cross_timeframe_list}")

    # Test timeframe validation
    model_types = ['random_forest', 'neural_network', 'lstm', 'hmm_model']

    for model_type in model_types:
        is_valid = validate_timeframe_consistency("15m", model_type, f"{model_type}_component")
        print(f"  {model_type}: {'✅' if is_valid else '❌'}")

    # Set model-specific timeframes
    print(f"\n🔧 Setting model-specific timeframes:")
    manager.config.set_model_timeframe("hmm_model", "15m")
    manager.config.set_model_timeframe("lstm_model", "1h")
    manager.config.set_model_timeframe("neural_network", "30m")

    for model_type in model_types:
        timeframe = manager.get_timeframe_for_model(model_type)
        print(f"  {model_type}: {timeframe}")

    # Get validation summary
    summary = manager.get_validation_summary()
    print(f"\n📊 Validation Summary:")
    print(f"  Total validations: {summary['total_validations']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")

    return summary

def demonstrate_comprehensive_validation():
    """Demonstrate comprehensive ML validation."""
    print("\n🎯 Comprehensive ML Validation Demonstration")
    print("=" * 50)

    # Create sample data
    data = create_sample_data(n_samples=1000, n_features=50, time_series=True)

    # Test different models with comprehensive validation
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300),
        'LogisticRegression': LogisticRegression(max_iter=1000)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n🔍 Comprehensive validation for {model_name}...")

        # Train model
        model.fit(data['X_train'], data['y_train'])

        # Comprehensive validation
        validation_report = validate_ml_model(
            model=model,
            X_train=data['X_train'],
            X_val=data['X_val'],
            y_train=data['y_train'],
            y_val=data['y_val'],
            timestamps=data['train_timestamps'],
            model_name=model_name,
            model_type=model_name.lower(),
            fold_number=1
        )

        results[model_name] = validation_report

        # Display comprehensive results
        print(f"  Overall validation passed: {'✅' if validation_report.overall_validation_passed else '❌'}")
        print(f"  Validation score: {validation_report.validation_score:.3f}")

        # Timeframe validation
        if validation_report.timeframe_validation:
            tf_valid = validation_report.timeframe_validation.get('valid', True)
            print(f"  Timeframe validation: {'✅' if tf_valid else '❌'}")

        # Temporal validation
        if validation_report.temporal_validation:
            temp_valid = validation_report.temporal_validation.temporal_order_valid
            leakage = validation_report.temporal_validation.leakage_detected
            print(f"  Temporal validation: {'✅' if temp_valid and not leakage else '❌'}")

        # Overfitting analysis
        if validation_report.overfitting_analysis:
            overfitting = validation_report.overfitting_analysis.is_overfitting
            severity = validation_report.overfitting_analysis.severity
            print(f"  Overfitting: {'✅' if not overfitting else f'❌ ({severity})'}")

        # Critical issues
        if validation_report.critical_issues:
            print(f"  Critical issues: {len(validation_report.critical_issues)}")
            for issue in validation_report.critical_issues:
                print(f"    - {issue}")

        # Recommendations
        if validation_report.recommendations:
            print(f"  Recommendations: {len(validation_report.recommendations)}")
            for rec in validation_report.recommendations[:3]:  # Show first 3
                print(f"    - {rec}")

    return results

def demonstrate_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n⚙️ Custom Configuration Demonstration")
    print("=" * 50)

    # Create custom configurations
    overfitting_config = OverfittingConfig(
        accuracy_gap_threshold=0.03,  # 3% gap triggers warning
        severe_accuracy_gap_threshold=0.08,  # 8% gap triggers early stopping
        enable_early_stopping=True,
        patience=3,
        save_reports=True,
        enable_visualization=True
    )

    temporal_config = TemporalValidationConfig(
        enable_temporal_checks=True,
        strict_temporal_order=True,
        min_temporal_gap=2,
        enable_walk_forward=True,
        n_splits=10,
        test_size=0.15
    )

    # Create comprehensive validation configuration
    validation_config = UniversalMLValidationConfig(
        overfitting_config=overfitting_config,
        temporal_config=temporal_config,
        enable_overfitting_detection=True,
        enable_temporal_validation=True,
        enable_timeframe_validation=True,
        save_comprehensive_reports=True,
        report_directory="demo_reports/validation",
        enable_visualization=True,
        detailed_logging=True
    )

    print("Custom configuration created:")
    print(f"  Overfitting threshold: {overfitting_config.accuracy_gap_threshold}")
    print(f"  Temporal gap: {temporal_config.min_temporal_gap}")
    print(f"  Report directory: {validation_config.report_directory}")

    # Test with custom configuration
    data = create_sample_data(n_samples=500, n_features=30)
    model = RandomForestClassifier()
    model.fit(data['X_train'], data['y_train'])

    # Get validator with custom config
    validator = get_ml_validator(validation_config)

    # Comprehensive validation with custom config
    report = validator.validate_model(
        model=model,
        X_train=data['X_train'],
        X_val=data['X_val'],
        y_train=data['y_train'],
        y_val=data['y_val'],
        model_name="CustomConfigModel",
        model_type="random_forest"
    )

    print(f"\nValidation with custom config:")
    print(f"  Overall passed: {'✅' if report.overall_validation_passed else '❌'}")
    print(f"  Validation score: {report.validation_score:.3f}")

    return report

def main():
    """Run the complete universal validation demonstration."""
    print("🚀 Universal ML Validation System Demonstration")
    print("=" * 70)

    try:
        # 1. Overfitting detection demonstration
        overfitting_results = demonstrate_overfitting_detection()

        # 2. Temporal validation demonstration
        temporal_report = demonstrate_temporal_validation()

        # 3. Timeframe configuration demonstration
        timeframe_summary = demonstrate_timeframe_configuration()

        # 4. Comprehensive validation demonstration
        comprehensive_results = demonstrate_comprehensive_validation()

        # 5. Custom configuration demonstration
        custom_report = demonstrate_custom_configuration()

        # Summary
        print("\n" + "=" * 70)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print("\n📊 Summary:")
        print(f"  Models tested: {len(overfitting_results)}")
        print(f"  Temporal validation: {'✅' if temporal_report.temporal_order_valid else '❌'}")
        print(f"  Timeframe validations: {timeframe_summary['total_validations']}")
        print(f"  Comprehensive validations: {len(comprehensive_results)}")
        print(f"  Custom configuration: {'✅' if custom_report.overall_validation_passed else '❌'}")

        print("\n🎯 Key Features Demonstrated:")
        print("✅ Universal overfitting detection for any ML model")
        print("✅ Temporal validation to prevent lookahead bias")
        print("✅ Timeframe configuration management")
        print("✅ Comprehensive validation reporting")
        print("✅ Custom configuration options")
        print("✅ Visual reporting and JSON export")

        print("\n💡 Next Steps:")
        print("1. Integrate into your existing ML pipelines")
        print("2. Configure timeframe settings for your models")
        print("3. Set up automated validation reporting")
        print("4. Monitor validation trends across models")

        return {
            'overfitting_results': overfitting_results,
            'temporal_report': temporal_report,
            'timeframe_summary': timeframe_summary,
            'comprehensive_results': comprehensive_results,
            'custom_report': custom_report
        }

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n📋 Results available in: {results}")
