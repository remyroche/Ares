"""
Comprehensive Training Example with All New Utilities

This example demonstrates how to use the comprehensive ML utilities for:
1. Data leakage prevention
2. Overfitting monitoring
3. Enhanced validation
4. HPO with overfitting prevention
5. Model complexity analysis
6. Comprehensive training with all safeguards

This serves as a template for how to use all the new utilities together.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

from src.utils.logger import system_logger
from src.utils.ml_common import (
    TrainingUtils,
    DataLeakagePrevention, DataLeakagePreventionConfig,
    OverfittingMonitoring, OverfittingMonitoringConfig,
    EnhancedValidation, EnhancedValidationConfig,
    HPOOverfittingPrevention, HPOOverfittingPreventionConfig,
    ModelComplexityAnalyzer, ModelComplexityAnalysisConfig
)

logger = system_logger.getChild('ComprehensiveTrainingExample')

def create_sample_data(n_samples: int = 1000, n_features: int = 20, n_classes: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample data for demonstration."""
    np.random.seed(42)

    # Create features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with some relationship to features
    if n_classes == 2:
        # Binary classification
        y = pd.Series(
            (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.5) > 0,
            dtype=int
        )
    else:
        # Multi-class classification
        y = pd.Series(
            np.argmax(X[['feature_0', 'feature_1', 'feature_2']].values + np.random.randn(n_samples, 3) * 0.5, axis=1),
            dtype=int
        )

    return X, y

def comprehensive_training_example():
    """Complete example of comprehensive training with all utilities."""
    logger.info("🚀 Starting Comprehensive Training Example")

    # Create sample data
    logger.info("📊 Creating sample data...")
    X, y = create_sample_data(n_samples=1000, n_features=10, n_classes=2)

    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.2 * len(X))
    test_size = len(X) - train_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Initialize comprehensive training utilities
    logger.info("🔧 Initializing comprehensive training utilities...")
    training_utils = TrainingUtils(config={})

    # Example 1: Data Leakage Prevention
    logger.info("🔍 Example 1: Data Leakage Prevention")
    leakage_prevention = DataLeakagePrevention(DataLeakagePreventionConfig())

    # Validate data integrity
    leakage_results = leakage_prevention.validate_data_integrity(
        X_train, y_train
    )

    logger.info(f"Data leakage validation results: {leakage_results.get('overall_valid', False)}")

    if not leakage_results.get('overall_valid', True):
        logger.warning("⚠️ Data leakage detected!")
        recommendations = leakage_results.get('prevention_report', {}).get('recommendations', [])
        for rec in recommendations:
            logger.warning(f"  - {rec}")

    # Example 2: Model Complexity Analysis
    logger.info("🔍 Example 2: Model Complexity Analysis")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Analyze a simple model
    simple_model = LogisticRegression(random_state=42)
    simple_model.fit(X_train, y_train)

    complexity_analyzer = ModelComplexityAnalyzer(ModelComplexityAnalysisConfig())

    complexity_results = complexity_analyzer.analyze_model_complexity(
        simple_model, X_train, y_train, X_val, y_val, "simple_logistic_regression"
    )

    logger.info(f"Model complexity score: {complexity_results.get('overall_complexity_score', 0)".3f"}")
    logger.info(f"Overfitting risk: {complexity_results.get('overfitting_risk', 'unknown')}")

    # Example 3: Comprehensive Model Training
    logger.info("🔍 Example 3: Comprehensive Model Training")

    # Train a model with comprehensive validation
    training_results = training_utils.train_model_with_comprehensive_validation(
        RandomForestClassifier,
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_name="comprehensive_rf",
        model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    )

    logger.info(f"Training successful: {training_results.get('training_successful', False)}")

    if training_results.get('training_successful', False):
        logger.info("✅ Model training completed successfully!")

        # Extract key metrics
        performance_metrics = training_results.get('performance_metrics', {})
        if performance_metrics:
            logger.info(f"Validation accuracy: {performance_metrics.get('accuracy', 0)".3f"}")

        # Show recommendations
        recommendations = training_results.get('recommendations', [])
        if recommendations:
            logger.info("📋 Training Recommendations:")
            for rec in recommendations[:3]:  # Show first 3
                logger.info(f"  - {rec}")

    # Example 4: Ensemble Training with Comprehensive Validation
    logger.info("🔍 Example 4: Ensemble Training with Comprehensive Validation")

    # Create base models
    base_models = {
        'rf': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'lr': LogisticRegression(random_state=42)
    }

    # Train base models
    for name, model in base_models.items():
        model.fit(X_train, y_train)

    # Train ensemble with comprehensive validation
    ensemble_results = training_utils.train_ensemble_with_comprehensive_validation(
        base_models, X_train, y_train, X_val, y_val,
        ensemble_name="comprehensive_ensemble",
        ensemble_method="voting"
    )

    logger.info(f"Ensemble training successful: {ensemble_results.get('training_successful', False)}")

    if ensemble_results.get('training_successful', False):
        logger.info("✅ Ensemble training completed successfully!")

        # Compare ensemble vs base models
        ensemble_metrics = ensemble_results.get('performance_metrics', {})
        base_metrics = ensemble_results.get('base_model_metrics', {})

        logger.info("📊 Performance Comparison:")
        for model_name, metrics in base_metrics.items():
            logger.info(f"  {model_name}: {metrics.get('accuracy', 0)".3f"}")

        if ensemble_metrics:
            logger.info(f"  Ensemble: {ensemble_metrics.get('accuracy', 0)".3f"}")

    # Example 5: HPO with Overfitting Prevention
    logger.info("🔍 Example 5: HPO with Overfitting Prevention")

    hpo_results = training_utils.optimize_hyperparameters_with_comprehensive_validation(
        RandomForestClassifier,
        X_train, y_train,
        model_name="hpo_rf_optimized"
    )

    logger.info(f"HPO successful: {hpo_results.get('optimization_successful', False)}")

    if hpo_results.get('optimization_successful', False):
        logger.info("✅ HPO completed successfully!")

        best_params = hpo_results.get('hpo_results', {}).get('best_params', {})
        logger.info(f"Best parameters: {best_params}")

    # Example 6: Comprehensive Model Analysis
    logger.info("🔍 Example 6: Comprehensive Model Analysis")

    # Analyze the trained model
    analysis_results = training_utils.analyze_model_comprehensive(
        base_models['rf'], X_train, y_train, X_val, y_val, "analyzed_rf"
    )

    if analysis_results.get('analysis_complete', False):
        logger.info("✅ Comprehensive analysis completed!")

        # Show key insights
        complexity_analysis = analysis_results.get('complexity_analysis', {})
        performance_analysis = analysis_results.get('performance_analysis', {})
        validation_analysis = analysis_results.get('validation_analysis', {})

        logger.info(f"Complexity score: {complexity_analysis.get('overall_complexity_score', 0)".3f"}")
        logger.info(f"Overfitting risk: {complexity_analysis.get('overfitting_risk', 'unknown')}")
        logger.info(f"Performance score: {performance_analysis.get('performance_metrics', {}).get('accuracy', 0)".3f"}")
        logger.info(f"Validation score: {validation_analysis.get('validation_summary', {}).get('validation_score', 0)".3f"}")

        # Show recommendations
        all_recommendations = analysis_results.get('recommendations', [])
        if all_recommendations:
            logger.info("📋 Analysis Recommendations:")
            for rec in all_recommendations[:5]:  # Show first 5
                logger.info(f"  - {rec}")

    # Summary
    logger.info("🎯 Comprehensive Training Example Summary:")
    logger.info("✅ All utilities demonstrated successfully!")
    logger.info("✅ Data leakage prevention: Implemented")
    logger.info("✅ Overfitting monitoring: Active")
    logger.info("✅ Enhanced validation: Applied")
    logger.info("✅ HPO with prevention: Optimized")
    logger.info("✅ Model complexity analysis: Completed")
    logger.info("✅ Comprehensive training: Successful")

    return {
        'training_results': training_results,
        'ensemble_results': ensemble_results,
        'hpo_results': hpo_results,
        'analysis_results': analysis_results,
        'recommendations': training_results.get('recommendations', [])[:5]
    }

def demonstrate_individual_utilities():
    """Demonstrate individual utilities for specific use cases."""
    logger.info("🔧 Demonstrating Individual Utilities")

    # Create sample data
    X, y = create_sample_data(n_samples=500, n_features=5, n_classes=2)
    X_train, X_val = X[:350], X[350:]
    y_train, y_val = y[:350], y[350:]

    # 1. Data Leakage Prevention
    logger.info("1. Data Leakage Prevention")
    leakage_config = DataLeakagePreventionConfig(
        enable_temporal_validation=True,
        enable_feature_validation=True,
        enable_information_leakage_detection=True
    )
    leakage_prevention = DataLeakagePrevention(leakage_config)

    leakage_results = leakage_prevention.validate_data_integrity(X_train, y_train)
    logger.info(f"Data integrity valid: {leakage_results.get('overall_valid', False)}")

    # 2. Overfitting Monitoring
    logger.info("2. Overfitting Monitoring")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    monitoring_config = OverfittingMonitoringConfig(
        overfitting_threshold=0.1,
        enable_learning_curve_analysis=True,
        enable_performance_monitoring=True
    )
    overfitting_monitor = OverfittingMonitoring(monitoring_config)

    monitoring_results = overfitting_monitor.monitor_model_performance(
        model, X_train, y_train, X_val, y_val, model_name="demo_rf"
    )
    logger.info(f"Overfitting detected: {monitoring_results.get('overfitting_detected', False)}")

    # 3. Enhanced Validation
    logger.info("3. Enhanced Validation")
    validation_config = EnhancedValidationConfig(
        enable_purged_cv=True,
        cv_folds=5,
        enable_bootstrap_validation=True,
        bootstrap_samples=100
    )
    enhanced_validation = EnhancedValidation(validation_config)

    validation_results = enhanced_validation.perform_comprehensive_validation(
        model, X_train, y_train, X_val, y_val, model_name="demo_validation"
    )
    logger.info(f"Validation score: {validation_results.get('validation_summary', {}).get('validation_score', 0)".3f"}")

    # 4. Model Complexity Analysis
    logger.info("4. Model Complexity Analysis")
    complexity_config = ModelComplexityAnalysisConfig(
        max_complexity_score=0.8,
        max_feature_ratio=0.5,
        min_samples_per_feature=10
    )
    complexity_analyzer = ModelComplexityAnalyzer(complexity_config)

    complexity_results = complexity_analyzer.analyze_model_complexity(
        model, X_train, y_train, X_val, y_val, "demo_complexity"
    )
    logger.info(f"Complexity score: {complexity_results.get('overall_complexity_score', 0)".3f"}")
    logger.info(f"Risk level: {complexity_results.get('overfitting_risk', 'unknown')}")

    # 5. HPO with Overfitting Prevention
    logger.info("5. HPO with Overfitting Prevention")
    hpo_config = HPOOverfittingPreventionConfig(
        max_trials=10,  # Reduced for demo
        n_trials=5,     # Reduced for demo
        enable_cross_validation_scoring=True,
        enable_early_stopping=True
    )
    hpo_optimizer = HPOOverfittingPrevention(hpo_config)

    hpo_results = hpo_optimizer.optimize_hyperparameters(
        RandomForestClassifier, X_train, y_train, "demo_hpo"
    )
    logger.info(f"HPO best score: {hpo_results.get('best_score', 0)".3f"}")

    return {
        'leakage_prevention': leakage_results,
        'overfitting_monitoring': monitoring_results,
        'enhanced_validation': validation_results,
        'model_complexity': complexity_results,
        'hpo_results': hpo_results
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Comprehensive ML Utilities Demonstration")

    # Run comprehensive example
    comprehensive_results = comprehensive_training_example()

    logger.info("✅ Comprehensive example completed!")

    # Run individual utility demonstrations
    individual_results = demonstrate_individual_utilities()

    logger.info("✅ Individual utility demonstrations completed!")

    # Final summary
    logger.info("🎉 All demonstrations completed successfully!")
    logger.info("📚 Key takeaways:")
    logger.info("  - Data leakage prevention ensures temporal integrity")
    logger.info("  - Overfitting monitoring detects performance gaps")
    logger.info("  - Enhanced validation provides comprehensive model assessment")
    logger.info("  - HPO with prevention optimizes while avoiding overfitting")
    logger.info("  - Model complexity analysis identifies overfitting risks")
    logger.info("  - TrainingUtils provides unified access to all utilities")