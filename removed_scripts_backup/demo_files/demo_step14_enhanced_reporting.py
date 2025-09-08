"""
Demo Script: Step14 Enhanced Tactician Labeling Reporting

This script demonstrates the comprehensive reporting capabilities for Step 14:
Tactician Labeling, focusing on dynamic barriers, multi-precision labeling,
strategic signals, and regime-aware labeling quality assessment.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append('/Users/remyroche/Documents/Ares')

# Import enhanced reporting system
try:
    from src.training.steps.model_training.step14_enhanced_reporting import Step14EnhancedReporter
    ENHANCED_REPORTING_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced reporting not available: {e}")
    ENHANCED_REPORTING_AVAILABLE = False
    Step14EnhancedReporter = None

def setup_logging():
    """Setup basic logging for the demo."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='{"asctime": "%(asctime)s", "levelname": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Create logger
    logger = logging.getLogger("AresTradingSystem.System.Step14.Demo")
    logger.info("🚀 Starting Step14 Enhanced Tactician Labeling Reporting Demonstration")
    return logger

def create_sample_labeling_results():
    """Create sample labeling results for demonstration."""
    return {
        'duration': 145.67,
        'data_points_processed': 50000,
        'labels_generated': 12500,
        'timeframes_analyzed': ['1m', '5m'],
        'labels': [
            {
                'label_type': 'buy',
                'confidence': 0.85,
                'precision_level': 'high_precision',
                'quality_score': 0.88,
                'consistency_score': 0.82,
                'success': True
            },
            {
                'label_type': 'sell',
                'confidence': 0.72,
                'precision_level': 'standard',
                'quality_score': 0.75,
                'consistency_score': 0.78,
                'success': False
            },
            {
                'label_type': 'hold',
                'confidence': 0.65,
                'precision_level': 'conservative',
                'quality_score': 0.82,
                'consistency_score': 0.85,
                'success': True
            }
        ] * 4167,  # Multiply to reach 12500 labels
        'filter_statistics': {
            'total_points': 50000,
            'filtered_points': 8500,
            'volume_filtered': 3200,
            'spread_filtered': 2800,
            'volatility_filtered': 2500
        }
    }

def create_sample_barrier_data():
    """Create sample barrier data for demonstration."""
    return {
        'barriers': [
            {
                'regime': 'bull_trend',
                'profit_barrier': 0.025,
                'loss_barrier': 0.015,
                'effectiveness': 0.87,
                'adaptation_rate': 0.84,
                'success_rate': 0.81
            },
            {
                'regime': 'bear_trend',
                'profit_barrier': 0.018,
                'loss_barrier': 0.022,
                'effectiveness': 0.82,
                'adaptation_rate': 0.79,
                'success_rate': 0.76
            },
            {
                'regime': 'sideways',
                'profit_barrier': 0.012,
                'loss_barrier': 0.012,
                'effectiveness': 0.85,
                'adaptation_rate': 0.88,
                'success_rate': 0.83
            },
            {
                'regime': 'high_volatility',
                'profit_barrier': 0.035,
                'loss_barrier': 0.028,
                'effectiveness': 0.79,
                'adaptation_rate': 0.76,
                'success_rate': 0.74
            }
        ]
    }

def create_sample_signal_data():
    """Create sample signal data for demonstration."""
    return {
        'signals': [
            {
                'strength': 0.9,
                'regime': 'bull_trend',
                'confidence': 0.88,
                'quality_score': 0.92,
                'analyst_agreement': 0.89,
                'is_signal': True
            },
            {
                'strength': 0.7,
                'regime': 'bear_trend',
                'confidence': 0.75,
                'quality_score': 0.78,
                'analyst_agreement': 0.82,
                'is_signal': True
            },
            {
                'strength': 0.5,
                'regime': 'sideways',
                'confidence': 0.62,
                'quality_score': 0.65,
                'analyst_agreement': 0.71,
                'is_signal': False
            },
            {
                'strength': 0.3,
                'regime': 'high_volatility',
                'confidence': 0.45,
                'quality_score': 0.52,
                'analyst_agreement': 0.58,
                'is_signal': False
            }
        ] * 250  # Multiply for more signal data
    }

def create_sample_regime_data():
    """Create sample regime data for demonstration."""
    return {
        'regime_statistics': {
            'bull_trend': {
                'label_distribution': {'buy': 450, 'sell': 120, 'hold': 180},
                'performance_score': 0.87,
                'barrier_effectiveness': 0.89,
                'consistency_score': 0.85
            },
            'bear_trend': {
                'label_distribution': {'buy': 95, 'sell': 480, 'hold': 175},
                'performance_score': 0.84,
                'barrier_effectiveness': 0.82,
                'consistency_score': 0.81
            },
            'sideways': {
                'label_distribution': {'buy': 210, 'sell': 195, 'hold': 345},
                'performance_score': 0.79,
                'barrier_effectiveness': 0.88,
                'consistency_score': 0.86
            },
            'high_volatility': {
                'label_distribution': {'buy': 180, 'sell': 165, 'hold': 205},
                'performance_score': 0.76,
                'barrier_effectiveness': 0.74,
                'consistency_score': 0.78
            }
        }
    }

def create_sample_validation_results():
    """Create sample validation results for demonstration."""
    return {
        'validation_statistics': {
            'accuracy': 0.84,
            'precision': 0.81,
            'recall': 0.87,
            'f1_score': 0.84,
            'cv_scores': [0.82, 0.85, 0.81, 0.83, 0.84, 0.86, 0.83],
            'validation_time': 45.2,
            'confidence': 0.86
        }
    }

def demo_step14_enhanced_reporting():
    """Demonstrate Step14 enhanced reporting functionality."""
    logger = setup_logging()

    if not ENHANCED_REPORTING_AVAILABLE or Step14EnhancedReporter is None:
        logger.error("❌ Step14 Enhanced Reporter not available")
        return False

    try:
        # Create sample configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'reports_dir': 'src/training/reports',
            'enhanced_reporting': True
        }

        logger.info("🔧 Initializing Step14 Enhanced Reporter...")
        enhanced_reporter = Step14EnhancedReporter(config)

        # Create sample data
        logger.info("🎯 Creating sample tactician labeling data...")
        labeling_results = create_sample_labeling_results()
        barrier_data = create_sample_barrier_data()
        signal_data = create_sample_signal_data()
        regime_data = create_sample_regime_data()
        validation_results = create_sample_validation_results()

        # Generate comprehensive report
        logger.info("📈 Generating comprehensive Step14 analysis report...")
        comprehensive_report = enhanced_reporter.generate_comprehensive_report(
            labeling_results=labeling_results,
            barrier_data=barrier_data,
            signal_data=signal_data,
            regime_data=regime_data,
            validation_results=validation_results
        )

        # Display key results
        logger.info("📊 Key Step14 Analysis Results:")
        logger.info(f"   🎯 Data Points Processed: {comprehensive_report.data_points_processed:,}")
        logger.info(f"   📋 Labels Generated: {comprehensive_report.labels_generated:,}")
        logger.info(f"   ⏰ Labeling Duration: {comprehensive_report.labeling_duration:.2f}s")
        logger.info(f"   🎨 Label Quality Score: {comprehensive_report.labeling_quality.label_quality_score:.4f}")
        logger.info(f"   📊 Signal Quality Score: {comprehensive_report.strategic_signals.signal_quality_score:.4f}")
        logger.info(f"   🛡️ Barrier Effectiveness: {comprehensive_report.barrier_performance.barrier_effectiveness_score:.4f}")
        logger.info(f"   ✅ Validation Accuracy: {comprehensive_report.validation_performance.validation_accuracy:.4f}")

        # Display label distribution
        logger.info("🎯 Label Type Distribution:")
        for label_type, count in comprehensive_report.labeling_quality.label_distribution.items():
            percentage = (count / comprehensive_report.labeling_quality.total_labels_generated) * 100
            logger.info(f"   {label_type}: {count:,} ({percentage:.1f}%)")

        # Display precision levels
        logger.info("🎯 Precision Level Performance:")
        for level, perf in comprehensive_report.precision_level_performance.items():
            logger.info(f"   {level}: {perf['count']} labels, {perf['avg_accuracy']:.3f} accuracy, {perf['success_rate']:.3f} success rate")

        # Display barrier performance by regime
        logger.info("🎯 Barrier Performance by Regime:")
        for regime, effectiveness in comprehensive_report.barrier_performance.regime_barrier_distribution.items():
            logger.info(f"   {regime}: {effectiveness} barriers configured")

        # Display signal strength distribution
        logger.info("🎯 Strategic Signal Strength Distribution:")
        for strength, count in comprehensive_report.strategic_signals.signal_strength_distribution.items():
            percentage = (count / comprehensive_report.strategic_signals.total_signals_generated) * 100
            logger.info(f"   {strength}: {count} ({percentage:.1f}%)")

        # Display quality filter efficiency
        logger.info("🎯 Quality Filter Efficiency:")
        logger.info(f"   📊 Total Points: {comprehensive_report.quality_filters.total_data_points:,}")
        logger.info(f"   🎯 Filtered Points: {comprehensive_report.quality_filters.filtered_data_points:,}")
        logger.info(f"   📈 Combined Efficiency: {comprehensive_report.quality_filters.combined_filter_efficiency:.3f}")

        # Display recommendations and alerts
        if comprehensive_report.recommendations:
            logger.info("💡 Recommendations:")
            for rec in comprehensive_report.recommendations:
                logger.info(f"   • {rec}")

        if comprehensive_report.alerts:
            logger.info("🚨 Alerts:")
            for alert in comprehensive_report.alerts:
                logger.info(f"   • {alert}")

        # Save comprehensive reports
        logger.info("💾 Saving Step14 comprehensive reports...")
        saved_files = enhanced_reporter.save_comprehensive_report(
            report_data=comprehensive_report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        logger.info("✅ Step14 Demo completed successfully!")
        logger.info(f"📄 Generated {len(saved_files)} Step14 report files:")
        for file_path in saved_files:
            logger.info(f"   • {file_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Step14 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("🎯 Step14 Enhanced Tactician Labeling Reporting Demonstration")
    print("=" * 80)

    success = demo_step14_enhanced_reporting()

    if success:
        print("\n" + "=" * 80)
        print("✅ Step14 Enhanced Reporting Demo completed successfully!")
        print("\n📚 Generated comprehensive reports including:")
        print("   • JSON: Complete structured analysis data")
        print("   • Markdown: Human-readable executive summary")
        print("   • CSV: Key metrics for analysis")
        print("   • PNG: Visual performance charts and dashboards")
        print("\n📁 Reports saved to: src/training/reports/step14_tactician_labeling/")
        print("\n🎉 Step14 Tactician Labeling Enhanced Reporting System is ready!")
    else:
        print("\n❌ Step14 Enhanced Reporting Demo failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
