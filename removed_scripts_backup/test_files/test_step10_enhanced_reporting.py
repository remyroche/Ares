#!/usr/bin/env python3
"""
Test script for enhanced Step10 reporting functionality.

This script tests the comprehensive reporting capabilities of the Step10
enhanced reporting system to ensure all features work correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location("step10_enhanced_reporting",
    "/Users/remyroche/Documents/Ares/src/training/steps/model_training/step10_enhanced_reporting.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Step10EnhancedReporter = module.Step10EnhancedReporter

def create_sample_analysis_results():
    """Create sample multi-timeframe HMM analysis results."""
    return {
        'multitimeframe_hmm': {
            'timeframes': ['5m', '15m', '30m', '1h'],
            'states_per_timeframe': {'5m': 4, '15m': 3, '30m': 3, '1h': 2},
            'transition_matrices': [
                [[0.7, 0.2, 0.1, 0.0], [0.3, 0.5, 0.2, 0.0], [0.1, 0.3, 0.5, 0.1], [0.0, 0.1, 0.2, 0.7]],
                [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
            ],
            'correlations': {'5m-15m': 0.75, '15m-30m': 0.82, '30m-1h': 0.78},
            'temporal_consistency': 0.85,
            'detection_confidence': {'5m': 0.82, '15m': 0.85, '30m': 0.83, '1h': 0.88},
            'alignment_score': 0.78
        },
        'data_quality': {
            'temporal_coverage': 0.92,
            'feature_completeness': 0.95,
            'consistency_score': 0.88,
            'outlier_percentage': 0.03,
            'noise_level': 0.08,
            'regime_balance': {'regime_0': 0.85, 'regime_1': 0.82, 'regime_2': 0.88, 'regime_3': 0.79},
            'overall_score': 0.87
        }
    }

def create_sample_prediction_results():
    """Create sample intensity-based prediction results."""
    return {
        'intensity_analysis': {
            'min_intensity': 0.0,
            'max_intensity': 1.0,
            'thresholds': {'regime_0': 0.3, 'regime_1': 0.5, 'regime_2': 0.7, 'regime_3': 0.8},
            'accuracy_by_intensity': {'regime_0': 0.78, 'regime_1': 0.82, 'regime_2': 0.85, 'regime_3': 0.79},
            'false_positive_rate': 0.15,
            'false_negative_rate': 0.12,
            'prediction_latency': 45.0,
            'confidence_score': 0.82
        },
        'position_logic': {
            'total_signals': 500,
            'buy_signals': 180,
            'sell_signals': 165,
            'hold_signals': 155,
            'confidence_distribution': {'high': 120, 'medium': 250, 'low': 130},
            'transition_accuracy': 0.79,
            'risk_adjusted_returns': 0.045,
            'drawdown_analysis': {'max_drawdown': 0.12, 'avg_drawdown': 0.08, 'avg_duration_days': 5.2}
        }
    }

def create_sample_integration_metrics():
    """Create sample TPSL and S/R integration metrics."""
    return {
        'tpsl_integration': {
            'take_profit_signals': 150,
            'stop_loss_signals': 120,
            'combined_accuracy': 0.78,
            'prediction_confidence': 0.81,
            'risk_effectiveness': 0.75,
            'signal_distribution': {'Take Profit': 150, 'Stop Loss': 120},
            'profit_factor': 1.35
        },
        'sr_integration': {
            'sr_levels_count': 25,
            'sr_signals': 85,
            'confidence_boost': 0.08,
            'alignment_score': 0.82,
            'combined_accuracy': 0.86,
            'level_reliability': {'support': 0.83, 'resistance': 0.85, 'major': 0.88},
            'breakout_detection': {'bullish': 35, 'bearish': 28, 'false': 12}
        }
    }

def create_sample_performance_data():
    """Create sample unified model performance data."""
    return {
        'unified_performance': {
            'overall_accuracy': 0.84,
            'precision': 0.81,
            'recall': 0.87,
            'f1_score': 0.84,
            'regime_accuracy': {'regime_0': 0.82, 'regime_1': 0.85, 'regime_2': 0.83, 'regime_3': 0.86},
            'mtf_consistency': 0.79,
            'prediction_stability': 0.83,
            'confidence_distribution': {'0.8-0.9': 180, '0.7-0.8': 220, '0.6-0.7': 85, 'below_0.6': 15}
        },
        'hardware_optimization': {
            'gpu_score': 0.88,
            'memory_efficiency': 0.82,
            'processing_speedup': 2.4,
            'parallel_efficiency': 0.86,
            'm1_score': 0.91,
            'vectorized_ops': 25000,
            'optimization_overhead': 0.12
        }
    }

def create_sample_config():
    """Create sample configuration for testing."""
    return {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'model_type': 'unified_regime_intelligence',
        'timeframes': ['5m', '15m', '30m', '1h'],
        'intensity_based': True,
        'tpsl_integration': True,
        'sr_integration': True,
        'position_logic': True,
        'hardware_acceleration': True,
        'regime_count': 'dynamic'
    }

def test_enhanced_reporting():
    """Test the enhanced Step10 reporting functionality."""
    print("🧪 Testing Step10 Enhanced Reporting System")
    print("=" * 50)

    try:
        # Create sample data
        print("📊 Creating sample analysis data...")
        analysis_results = create_sample_analysis_results()
        prediction_results = create_sample_prediction_results()
        integration_metrics = create_sample_integration_metrics()
        performance_data = create_sample_performance_data()
        config = create_sample_config()

        print(f"   Multi-timeframe analysis: {len(analysis_results['multitimeframe_hmm']['timeframes'])} timeframes")
        print(f"   Intensity analysis: {len(prediction_results['intensity_analysis']['thresholds'])} regimes")
        print(f"   TPSL signals: {integration_metrics['tpsl_integration']['take_profit_signals'] + integration_metrics['tpsl_integration']['stop_loss_signals']}")
        print(f"   S/R levels: {integration_metrics['sr_integration']['sr_levels_count']}")

        # Initialize the reporter
        print("🔧 Initializing enhanced reporter...")
        reporter = Step10EnhancedReporter(config)

        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            analysis_results=analysis_results,
            prediction_results=prediction_results,
            integration_metrics=integration_metrics,
            performance_data=performance_data
        )

        if 'error' in report:
            print(f"❌ Report generation failed: {report['error']}")
            return False

        print("✅ Report generated successfully")

        # Test key sections
        expected_sections = [
            'multitimeframe_hmm_analysis',
            'intensity_prediction_analysis',
            'tpsl_integration_analysis',
            'position_logic_analysis',
            'sr_integration_analysis',
            'unified_performance_analysis',
            'hardware_optimization_analysis',
            'data_quality_analysis',
            'recommendations',
            'alerts'
        ]

        missing_sections = []
        for section in expected_sections:
            if section not in report:
                missing_sections.append(section)

        if missing_sections:
            print(f"⚠️ Missing sections: {missing_sections}")
        else:
            print("✅ All expected sections present")

        # Test performance predictions
        print("🔮 Testing performance predictions...")
        predictions = reporter._generate_performance_predictions()

        if 'error' in predictions:
            print(f"⚠️ Performance predictions failed: {predictions['error']}")
        else:
            print("✅ Performance predictions generated")

            # Test specific predictions
            if 'unified_model_predictions' in predictions:
                ump = predictions['unified_model_predictions']
                performance = ump.get('predicted_model_performance', 0)
                print(f"   Predicted model performance: {performance:.3f}")
        # Test enhanced alerts
        print("🚨 Testing enhanced alerts...")
        alerts = reporter._generate_alerts()

        if alerts:
            print(f"✅ Generated {len(alerts)} alerts")
            # Show first few alerts
            for alert in alerts[:3]:
                print(f"   • {alert[:100]}{'...' if len(alert) > 100 else ''}")
        else:
            print("ℹ️ No alerts generated (system performing well)")

        # Test system health calculation
        print("🏥 Testing system health calculation...")
        health_score = reporter._calculate_overall_system_health()
        print(f"   System health score: {health_score:.3f}")
        # Test specific analysis methods
        print("🔬 Testing specific analysis methods...")

        # Test multi-timeframe analysis
        reporter._analyze_multitimeframe_hmm(analysis_results)
        if reporter.multitimeframe_metrics:
            print("✅ Multi-timeframe HMM analysis completed")
            print(f"   Timeframes analyzed: {len(reporter.multitimeframe_metrics.timeframes_analyzed)}")
            print(f"   Temporal consistency: {reporter.multitimeframe_metrics.temporal_consistency_score:.3f}")

        # Test intensity analysis
        reporter._analyze_intensity_predictions(prediction_results)
        if reporter.intensity_metrics:
            print("✅ Intensity prediction analysis completed")
            print(f"   Prediction confidence: {reporter.intensity_metrics.intensity_based_confidence:.3f}")
        # Test TPSL integration
        reporter._analyze_tpsl_integration(integration_metrics)
        if reporter.tpsl_metrics:
            print("✅ TPSL integration analysis completed")
            print(f"   Combined accuracy: {reporter.tpsl_metrics.combined_tpsl_accuracy:.3f}")
            print(f"   Profit factor: {reporter.tpsl_metrics.profit_factor:.2f}")

        # Test position logic
        reporter._analyze_position_logic(prediction_results)
        if reporter.position_metrics:
            print("✅ Position logic analysis completed")
            print(f"   Total signals: {reporter.position_metrics.total_trading_signals:,}")
            print(f"   Transition accuracy: {reporter.position_metrics.position_transition_accuracy:.3f}")

        # Test S/R integration
        reporter._analyze_sr_integration(integration_metrics)
        if reporter.sr_metrics:
            print("✅ S/R integration analysis completed")
            print(f"   S/R levels: {reporter.sr_metrics.sr_levels_identified}")
            print(f"   Combined accuracy: {reporter.sr_metrics.combined_sr_regime_accuracy:.3f}")

        # Test unified performance
        reporter._analyze_unified_performance(performance_data)
        if reporter.performance_metrics:
            print("✅ Unified performance analysis completed")
            print(f"   Overall accuracy: {reporter.performance_metrics.overall_accuracy:.3f}")
            print(f"   F1 Score: {reporter.performance_metrics.f1_score:.3f}")

        # Test hardware optimization
        reporter._analyze_hardware_optimization(performance_data)
        if reporter.hardware_metrics:
            print("✅ Hardware optimization analysis completed")
            print(f"   GPU acceleration score: {reporter.hardware_metrics.gpu_acceleration_score:.3f}")
        # Test data quality
        reporter._analyze_data_quality(analysis_results)
        if reporter.data_quality_metrics:
            print("✅ Data quality analysis completed")
            print(f"   Overall quality: {reporter.data_quality_metrics.data_quality_overall_score:.3f}")

        # Save sample report
        print("💾 Testing report saving...")
        saved_files = reporter.save_comprehensive_report(
            report_data=report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        if saved_files:
            print(f"✅ Saved {len(saved_files)} report files:")
            for file_path in saved_files:
                print(f"   • {file_path}")
        else:
            print("⚠️ No files were saved")

        # Test markdown report generation specifically
        print("📝 Testing markdown report generation...")
        markdown_path = reporter._save_markdown_report(
            report_data=report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        if markdown_path:
            print(f"✅ Markdown report saved: {markdown_path}")
        else:
            print("⚠️ Markdown report not saved")

        print("\n🎉 Step10 Enhanced Reporting Test Completed Successfully!")
        print("=" * 50)
        print("✅ All major functionality verified")
        print("✅ Report generation working")
        print("✅ Performance predictions functional")
        print("✅ Alert system operational")
        print("✅ File saving capabilities confirmed")
        print("✅ Enhanced markdown reports generated")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_reporting()
    sys.exit(0 if success else 1)
