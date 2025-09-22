"""
Overfitting Reporting Demonstration

Demonstrates the enhanced overfitting detection reporting system with comprehensive
analysis, visualizations, and actionable insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Import enhanced components
from .overfitting_reporting import (
    OverfittingReporter,
    OverfittingReport,
    get_overfitting_reporter,
    create_overfitting_reporter
)
from .early_stopping import (
    get_overfitting_detector,
    EarlyStoppingConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_overfitting_data():
    """Create sample data for overfitting demonstration."""
    np.random.seed(42)
    
    # Create sample predictions with overfitting patterns
    n_samples = 1000
    
    # Training data - model performs very well
    train_predictions = np.random.choice([0, 1, 2], size=n_samples//2, p=[0.1, 0.8, 0.1])
    train_labels = np.random.choice([0, 1, 2], size=n_samples//2, p=[0.1, 0.8, 0.1])
    
    # Test data - model performs poorly (overfitting)
    test_predictions = np.random.choice([0, 1, 2], size=n_samples//2, p=[0.3, 0.4, 0.3])
    test_labels = np.random.choice([0, 1, 2], size=n_samples//2, p=[0.1, 0.8, 0.1])
    
    # Create probabilities with overconfidence
    train_probabilities = np.random.dirichlet([0.1, 0.8, 0.1], size=n_samples//2)
    test_probabilities = np.random.dirichlet([0.3, 0.4, 0.3], size=n_samples//2)
    
    # Feature importance (highly concentrated)
    feature_importance = np.random.exponential(0.1, 50)
    feature_importance = feature_importance / np.sum(feature_importance)
    
    return {
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'train_probabilities': train_probabilities,
        'test_probabilities': test_probabilities,
        'feature_importance': feature_importance
    }

def demonstrate_overfitting_reporting():
    """Demonstrate comprehensive overfitting reporting."""
    print("🚀 Enhanced Overfitting Detection Reporting Demonstration")
    print("=" * 70)
    
    # Create sample data
    data = create_sample_overfitting_data()
    
    # Initialize overfitting detector and reporter
    detector = get_overfitting_detector()
    reporter = create_overfitting_reporter(
        save_reports=True,
        report_directory="demo_reports/overfitting",
        enable_visualization=True,
        detailed_logging=True
    )
    
    print("\n📊 Running comprehensive overfitting analysis...")
    
    # Perform overfitting analysis
    overfitting_analysis = detector.comprehensive_overfitting_analysis(
        train_predictions=data['train_predictions'],
        val_predictions=data['test_predictions'],
        train_labels=data['train_labels'],
        val_labels=data['test_labels'],
        train_probabilities=data['train_probabilities'],
        val_probabilities=data['test_probabilities'],
        feature_importance=data['feature_importance']
    )
    
    print("✅ Overfitting analysis completed")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    
    report = detector.generate_comprehensive_report(
        overfitting_analysis=overfitting_analysis,
        model_name="DemoHMMModel",
        fold_number=1
    )
    
    print("✅ Comprehensive report generated")
    
    # Display report summary
    print("\n" + "=" * 70)
    print("OVERFITTING DETECTION REPORT SUMMARY")
    print("=" * 70)
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  Train Accuracy: {report.train_accuracy:.4f}")
    print(f"  Val Accuracy:   {report.val_accuracy:.4f}")
    print(f"  Accuracy Gap:   {report.accuracy_gap:.4f}")
    print(f"  Train F1:       {report.train_f1:.4f}")
    print(f"  Val F1:         {report.val_f1:.4f}")
    print(f"  F1 Gap:         {report.f1_gap:.4f}")
    
    print(f"\n🚨 OVERFITTING STATUS:")
    print(f"  Detected:       {report.is_overfitting}")
    print(f"  Severity:       {report.severity.upper()}")
    print(f"  Confidence:     {report.confidence_level:.2f}")
    
    if report.indicators:
        print(f"\n🔍 OVERFITTING INDICATORS ({len(report.indicators)}):")
        for i, indicator in enumerate(report.indicators, 1):
            print(f"  {i}. {indicator}")
    
    if report.warnings:
        print(f"\n⚠️ WARNINGS ({len(report.warnings)}):")
        for i, warning in enumerate(report.warnings, 1):
            print(f"  {i}. {warning}")
    
    if report.recommendations:
        print(f"\n💡 RECOMMENDATIONS ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Advanced metrics
    if report.train_confidence is not None:
        print(f"\n📊 ADVANCED METRICS:")
        print(f"  Train Confidence: {report.train_confidence:.4f}")
        print(f"  Val Confidence:   {report.val_confidence:.4f}")
        print(f"  Confidence Gap:   {report.confidence_gap:.4f}")
        if report.overconfident_ratio is not None:
            print(f"  Overconfident:    {report.overconfident_ratio:.4f}")
        if report.feature_concentration is not None:
            print(f"  Feature Conc:     {report.feature_concentration:.4f}")
    
    # Generate multiple reports to show trend analysis
    print(f"\n📈 Generating trend analysis...")
    
    # Simulate multiple epochs/folds
    for epoch in range(2, 6):
        # Simulate worsening overfitting
        data['test_predictions'] = np.random.choice([0, 1, 2], size=len(data['test_predictions']), 
                                                   p=[0.4, 0.3, 0.3])  # Worse performance
        
        # Re-analyze
        analysis = detector.comprehensive_overfitting_analysis(
            train_predictions=data['train_predictions'],
            val_predictions=data['test_predictions'],
            train_labels=data['train_labels'],
            val_labels=data['test_labels'],
            train_probabilities=data['train_probabilities'],
            val_probabilities=data['test_probabilities'],
            feature_importance=data['feature_importance']
        )
        
        # Generate report
        trend_report = detector.generate_comprehensive_report(
            overfitting_analysis=analysis,
            model_name="DemoHMMModel",
            fold_number=epoch
        )
        
        print(f"  Epoch {epoch}: {trend_report.severity.upper()} overfitting detected")
    
    # Get summary report
    print(f"\n📊 SUMMARY REPORT:")
    summary = detector.get_detection_summary()
    
    print(f"  Total Reports: {summary['total_reports']}")
    print(f"  Overfitting Detected: {summary['overfitting_detected']}")
    print(f"  Overfitting Rate: {summary['overfitting_rate']:.2%}")
    print(f"  Severity Distribution: {summary['severity_distribution']}")
    print(f"  Average Train Accuracy: {summary['average_metrics']['train_accuracy']:.4f}")
    print(f"  Average Val Accuracy: {summary['average_metrics']['val_accuracy']:.4f}")
    print(f"  Average Accuracy Gap: {summary['average_metrics']['accuracy_gap']:.4f}")
    
    # Check if reports were saved
    report_dir = Path("demo_reports/overfitting")
    if report_dir.exists():
        json_files = list(report_dir.glob("*.json"))
        viz_files = list((report_dir / "visualizations").glob("*.png")) if (report_dir / "visualizations").exists() else []
        
        print(f"\n💾 REPORTS SAVED:")
        print(f"  JSON Reports: {len(json_files)}")
        print(f"  Visualizations: {len(viz_files)}")
        print(f"  Report Directory: {report_dir.absolute()}")
    
    print("\n✅ Overfitting reporting demonstration completed!")
    return report, summary

def demonstrate_no_overfitting():
    """Demonstrate reporting when no overfitting is detected."""
    print("\n" + "=" * 70)
    print("NO OVERFITTING DETECTION DEMONSTRATION")
    print("=" * 70)
    
    # Create data with no overfitting
    np.random.seed(123)
    n_samples = 500
    
    # Both train and test perform similarly
    train_predictions = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])
    test_predictions = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])
    train_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])
    test_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])
    
    # Balanced probabilities
    train_probabilities = np.random.dirichlet([0.2, 0.6, 0.2], size=n_samples)
    test_probabilities = np.random.dirichlet([0.2, 0.6, 0.2], size=n_samples)
    
    # Well-distributed feature importance
    feature_importance = np.random.uniform(0.01, 0.05, 50)
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Analyze
    detector = get_overfitting_detector()
    analysis = detector.comprehensive_overfitting_analysis(
        train_predictions=train_predictions,
        val_predictions=test_predictions,
        train_labels=train_labels,
        val_labels=test_labels,
        train_probabilities=train_probabilities,
        val_probabilities=test_probabilities,
        feature_importance=feature_importance
    )
    
    # Generate report
    report = detector.generate_comprehensive_report(
        overfitting_analysis=analysis,
        model_name="WellGeneralizedModel",
        fold_number=1
    )
    
    print(f"\n📊 NO OVERFITTING DETECTED:")
    print(f"  Train Accuracy: {report.train_accuracy:.4f}")
    print(f"  Val Accuracy:   {report.val_accuracy:.4f}")
    print(f"  Accuracy Gap:   {report.accuracy_gap:.4f}")
    print(f"  Severity:       {report.severity.upper()}")
    print(f"  Confidence:     {report.confidence_level:.2f}")
    
    if report.warnings:
        print(f"\n⚠️ Warnings: {len(report.warnings)}")
        for warning in report.warnings:
            print(f"  - {warning}")
    
    return report

def main():
    """Run the complete overfitting reporting demonstration."""
    print("🎯 Enhanced Overfitting Detection Reporting System")
    print("=" * 70)
    
    try:
        # Demonstrate overfitting detection
        overfitting_report, summary = demonstrate_overfitting_reporting()
        
        # Demonstrate no overfitting case
        no_overfitting_report = demonstrate_no_overfitting()
        
        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nKey Features Demonstrated:")
        print("✅ Comprehensive overfitting analysis")
        print("✅ Detailed reporting with severity levels")
        print("✅ Actionable warnings and recommendations")
        print("✅ Trend analysis across multiple epochs")
        print("✅ Visual report generation")
        print("✅ JSON report saving")
        print("✅ Summary statistics")
        
        return {
            'overfitting_report': overfitting_report,
            'no_overfitting_report': no_overfitting_report,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n📋 Results available in: {results}")