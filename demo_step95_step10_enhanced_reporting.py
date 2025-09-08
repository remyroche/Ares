#!/usr/bin/env python3
"""
Demo script for Step9_5 and Step10 Enhanced Reporting

This script demonstrates the enhanced reporting capabilities for both:
- Step9_5: HMM-LM Generalist Training
- Step10: Unified Regime Intelligence
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.steps.model_training.step09_5_enhanced_reporting import Step95EnhancedReporter
from src.training.steps.model_training.step10_enhanced_reporting import Step10EnhancedReporter
from src.utils.logger import system_logger
import logging

def create_sample_training_results():
    """Create sample training results for Step9_5 demonstration."""
    # Sample individual model results
    individual_models = {
        'lightgbm': {
            'training_time': 45.67,
            'convergence_score': 0.89,
            'feature_importance_count': 25,
            'training_samples': 15000,
            'validation_score': 0.82,
            'overfitting_score': 0.05,
            'computational_efficiency': 0.91,
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.85,
            'f1_score': 0.82,
            'roc_auc': 0.88,
            'feature_importance': {
                'close': 0.25,
                'volume': 0.18,
                'rsi': 0.15,
                'macd': 0.12,
                'bb_upper': 0.10,
                'momentum_5': 0.08,
                'volatility': 0.07,
                'spread': 0.05
            }
        },
        'neural_network': {
            'training_time': 156.89,
            'convergence_score': 0.76,
            'feature_importance_count': 25,
            'training_samples': 15000,
            'validation_score': 0.85,
            'overfitting_score': 0.12,
            'computational_efficiency': 0.65,
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85,
            'roc_auc': 0.91,
            'feature_importance': {
                'close': 0.28,
                'volume': 0.16,
                'rsi': 0.13,
                'macd': 0.10,
                'bb_upper': 0.08,
                'momentum_5': 0.07,
                'volatility': 0.10,
                'spread': 0.08
            }
        }
    }

    # Sample evaluation metrics
    evaluation_metrics = {
        'lightgbm': {
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.85,
            'f1_score': 0.82,
            'roc_auc': 0.88,
            'confusion_matrix': [[820, 180], [150, 850]],
            'classification_report': {
                '0': {'precision': 0.79, 'recall': 0.85, 'f1-score': 0.82, 'support': 1000},
                '1': {'precision': 0.85, 'recall': 0.79, 'f1-score': 0.82, 'support': 1000}
            },
            'feature_importance': {
                'close': 0.25,
                'volume': 0.18,
                'rsi': 0.15,
                'macd': 0.12,
                'bb_upper': 0.10,
                'momentum_5': 0.08,
                'volatility': 0.07,
                'spread': 0.05
            }
        },
        'ensemble': {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.89,
            'f1_score': 0.87,
            'roc_auc': 0.92,
            'confusion_matrix': [[840, 160], [110, 890]],
            'classification_report': {
                '0': {'precision': 0.84, 'recall': 0.89, 'f1-score': 0.87, 'support': 1000},
                '1': {'precision': 0.89, 'recall': 0.84, 'f1-score': 0.87, 'support': 1000}
            },
            'feature_importance': {
                'close': 0.26,
                'volume': 0.17,
                'rsi': 0.14,
                'macd': 0.11,
                'bb_upper': 0.09,
                'momentum_5': 0.08,
                'volatility': 0.09,
                'spread': 0.06
            }
        }
    }

    return {
        'individual_models': individual_models,
        'evaluation_metrics': evaluation_metrics,
        'total_training_time': 304.35,
        'accuracy': 0.87,
        'precision': 0.84,
        'recall': 0.89,
        'f1_score': 0.87,
        'roc_auc': 0.92,
        'selected_features': ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'momentum_5', 'volatility', 'spread']
    }

def create_sample_step10_data():
    """Create sample data for Step10 demonstration."""
    return {
        'analysis_results': {
            'multitimeframe_hmm': {
                'timeframes': ['5m', '15m', '30m', '1h'],
                'states_per_timeframe': {'5m': 5, '15m': 5, '30m': 5, '1h': 5},
                'transition_matrices': {},  # Would be populated with actual matrices
                'correlations': {'5m_15m': 0.75, '15m_30m': 0.78, '5m_30m': 0.72},
                'temporal_consistency': 0.85,
                'detection_confidence': {'5m': 0.82, '15m': 0.84, '30m': 0.81, '1h': 0.79},
                'alignment_score': 0.78
            },
            'data_quality': {
                'temporal_coverage': 0.92,
                'feature_completeness': 0.95,
                'consistency_score': 0.88,
                'outlier_percentage': 0.03,
                'noise_level': 0.08,
                'regime_balance': {'regime_0': 0.25, 'regime_1': 0.30, 'regime_2': 0.20, 'regime_3': 0.25},
                'overall_score': 0.87
            }
        },
        'prediction_results': {
            'intensity_analysis': {
                'min_intensity': 0.0,
                'max_intensity': 1.0,
                'thresholds': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
                'accuracy_by_intensity': {'low': 0.75, 'medium': 0.82, 'high': 0.88},
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
                'confidence_distribution': {'high': 280, 'medium': 150, 'low': 70},
                'transition_accuracy': 0.79,
                'risk_adjusted_returns': 0.045
            }
        },
        'integration_metrics': {
            'tpsl_integration': {
                'take_profit_signals': 150,
                'stop_loss_signals': 120,
                'combined_accuracy': 0.78,
                'prediction_confidence': 0.81,
                'risk_effectiveness': 0.75,
                'signal_distribution': {'take_profit': 150, 'stop_loss': 120, 'neutral': 230},
                'profit_factor': 1.35
            },
            'sr_integration': {
                'sr_levels_count': 25,
                'sr_signals': 85,
                'confidence_boost': 0.08,
                'alignment_score': 0.82,
                'combined_accuracy': 0.86,
                'level_reliability': {'strong': 15, 'medium': 7, 'weak': 3},
                'breakout_detection': {'successful': 18, 'failed': 7}
            }
        },
        'performance_data': {
            'unified_performance': {
                'overall_accuracy': 0.84,
                'precision': 0.81,
                'recall': 0.87,
                'f1_score': 0.84,
                'regime_accuracy': {'regime_0': 0.82, 'regime_1': 0.85, 'regime_2': 0.81, 'regime_3': 0.86},
                'mtf_consistency': 0.79,
                'prediction_stability': 0.83,
                'confidence_distribution': {'high': 320, 'medium': 140, 'low': 40}
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
    }

def demonstrate_step95_reporting():
    """Demonstrate the Step9_5 enhanced reporting system."""
    logger = system_logger.getChild('Step9_5.Demo')
    logger.info("🚀 Starting Step9_5 Enhanced HMM-LM Reporting Demonstration")

    try:
        # Configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache',
            'HMM_LM': {
                'generalist': {
                    'hmm_states': 5,
                    'sequence_length': 20,
                    'timeframes': ['5m', '15m', '30m', '1h'],
                    'd_model': 256,
                    'nhead': 8,
                    'num_layers': 6,
                    'dropout_rate': 0.1,
                    'learning_rate': 0.0001,
                    'batch_size': 32,
                    'epochs': 100
                }
            }
        }

        # Create sample training results
        logger.info("🤖 Creating sample HMM-LM training results...")
        training_results = create_sample_training_results()

        # Prepare model config
        model_config = config['HMM_LM']['generalist']

        # Prepare sequence data
        sequence_data = {
            'sequences': [],  # Would be populated with actual sequences
            'regime_changes': [],  # Would be populated with actual regime changes
            'tpsl_events': [],  # Would be populated with actual TPSL events
            'vocabulary': {'regime_up': 0, 'regime_down': 1, 'regime_sideways': 2, 'tpsl_take_profit': 3, 'tpsl_stop_loss': 4},
            'processing_time': 0.0,  # Would be populated with actual processing time
            'completeness': 0.95,
            'consistency': 0.9,
            'diversity': 0.85,
            'temporal_coverage': 0.92
        }

        # Prepare hardware metrics
        hardware_metrics = {
            'gpu_utilization': 0.87,
            'm1_gpu_available': True,
            'memory_usage_mb': 2048.0,
            'training_speedup': 2.3,
            'batch_processing_time': 0.12,
            'parallel_efficiency': 0.89,
            'optimization_score': 0.85
        }

        # Prepare evaluation results
        evaluation_results = training_results['evaluation_metrics']

        # Initialize enhanced reporter
        logger.info("🔧 Initializing Step9_5 Enhanced Reporter...")
        reporter = Step95EnhancedReporter(config)

        # Generate comprehensive report
        logger.info("📈 Generating comprehensive Step9_5 analysis report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            training_results=training_results,
            model_config=model_config,
            sequence_data=sequence_data,
            hardware_metrics=hardware_metrics,
            evaluation_results=evaluation_results
        )

        # Display key metrics
        logger.info("📊 Key Step9_5 Analysis Results:")
        logger.info(f"   🤖 Models Trained: {len(training_results['individual_models'])}")
        logger.info(f"   🧠 Transformer Architecture: {model_config['d_model']}d x {model_config['num_layers']} layers")
        logger.info(f"   📊 Sequence Length: {model_config['sequence_length']}")
        logger.info(f"   ⚡ Hardware Acceleration Score: {hardware_metrics['optimization_score']:.3f}")
        logger.info(f"   🎯 Test Accuracy: {training_results['accuracy']:.3f}")
        logger.info(f"   🏆 F1 Score: {training_results['f1_score']:.3f}")

        # Display recommendations and alerts
        if 'recommendations' in comprehensive_report:
            logger.info("💡 Step9_5 Recommendations:")
            for rec in comprehensive_report['recommendations'][:2]:  # Show first 2
                logger.info(f"   • {rec}")

        # Save comprehensive reports
        logger.info("💾 Saving Step9_5 comprehensive reports...")
        saved_files = reporter.save_comprehensive_report(
            report_data=comprehensive_report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        logger.info("✅ Step9_5 Demo completed successfully!")
        logger.info(f"📄 Generated {len(saved_files)} Step9_5 report files:")
        for file_path in saved_files:
            logger.info(f"   • {file_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Step9_5 Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def demonstrate_step10_reporting():
    """Demonstrate the Step10 enhanced reporting system."""
    logger = system_logger.getChild('Step10.Demo')
    logger.info("🚀 Starting Step10 Enhanced Unified Regime Intelligence Reporting Demonstration")

    try:
        # Configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframes': ['5m', '15m', '30m', '1h'],
            'hmm_states_per_tf': 5,
            'sequence_length': 20,
            'intensity_based': True,
            'tpsl_integration': True,
            'sr_integration': True,
            'position_logic': True,
            'hardware_acceleration': True,
            'regime_count': 'dynamic'
        }

        # Create sample data
        logger.info("🎯 Creating sample unified regime intelligence data...")
        step10_data = create_sample_step10_data()

        # Initialize enhanced reporter
        logger.info("🔧 Initializing Step10 Enhanced Reporter...")
        reporter = Step10EnhancedReporter(config)

        # Generate comprehensive report
        logger.info("📈 Generating comprehensive Step10 analysis report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            analysis_results=step10_data['analysis_results'],
            prediction_results=step10_data['prediction_results'],
            integration_metrics=step10_data['integration_metrics'],
            performance_data=step10_data['performance_data']
        )

        # Display key metrics
        logger.info("📊 Key Step10 Analysis Results:")
        logger.info(f"   🎯 Timeframes Analyzed: {len(step10_data['analysis_results']['multitimeframe_hmm']['timeframes'])}")
        logger.info(f"   📈 Overall Accuracy: {step10_data['performance_data']['unified_performance']['overall_accuracy']:.3f}")
        logger.info(f"   🏆 F1 Score: {step10_data['performance_data']['unified_performance']['f1_score']:.3f}")
        logger.info(f"   📊 Trading Signals Generated: {step10_data['prediction_results']['position_logic']['total_signals']}")
        logger.info(f"   💰 Risk-Adjusted Returns: {step10_data['prediction_results']['position_logic']['risk_adjusted_returns']:.3f}")
        logger.info(f"   ⚡ Processing Speedup: {step10_data['performance_data']['hardware_optimization']['processing_speedup']:.1f}x")

        # Display recommendations and alerts
        if 'recommendations' in comprehensive_report:
            logger.info("💡 Step10 Recommendations:")
            for rec in comprehensive_report['recommendations'][:2]:  # Show first 2
                logger.info(f"   • {rec}")

        # Save comprehensive reports
        logger.info("💾 Saving Step10 comprehensive reports...")
        saved_files = reporter.save_comprehensive_report(
            report_data=comprehensive_report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframes'][0]
        )

        logger.info("✅ Step10 Demo completed successfully!")
        logger.info(f"📄 Generated {len(saved_files)} Step10 report files:")
        for file_path in saved_files:
            logger.info(f"   • {file_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Step10 Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def demonstrate_combined_reporting():
    """Demonstrate both Step9_5 and Step10 enhanced reporting systems together."""
    logger = system_logger.getChild('Combined.Demo')
    logger.info("🚀 Starting Combined Step9_5 & Step10 Enhanced Reporting Demonstration")
    logger.info("="*70)

    # Demonstrate Step9_5
    logger.info("🎯 Demonstrating Step9_5: HMM-LM Generalist Training")
    logger.info("-" * 50)
    step95_success = demonstrate_step95_reporting()

    logger.info("\n" + "="*70)

    # Demonstrate Step10
    logger.info("🎯 Demonstrating Step10: Unified Regime Intelligence")
    logger.info("-" * 50)
    step10_success = demonstrate_step10_reporting()

    # Summary
    logger.info("\n" + "="*70)
    logger.info("🎉 Combined Enhanced Reporting Demo Summary")
    logger.info("="*70)
    logger.info("✅ Step9_5 HMM-LM Training Analysis:")
    logger.info("   • Transformer architecture analysis")
    logger.info("   • Sequence processing metrics")
    logger.info("   • Hardware acceleration monitoring")
    logger.info("   • Model evaluation and performance")
    logger.info("   • Multi-timeframe regime analysis")
    logger.info("")
    logger.info("✅ Step10 Unified Regime Intelligence Analysis:")
    logger.info("   • Multi-timeframe HMM state analysis")
    logger.info("   • Intensity-based predictions")
    logger.info("   • TPSL integration metrics")
    logger.info("   • Position logic analysis")
    logger.info("   • S/R integration evaluation")
    logger.info("   • Hardware optimization tracking")
    logger.info("")
    logger.info("✅ Generated comprehensive reports with multiple formats:")
    logger.info("   • JSON: Detailed structured data")
    logger.info("   • Markdown: Human-readable summaries")
    logger.info("   • CSV: Key metrics for analysis")
    logger.info("   • PNG: Visual charts and graphs")
    logger.info("")
    logger.info("✅ Both systems provide actionable recommendations and alerts")
    logger.info("✅ Robust error handling with fallback mechanisms")
    logger.info("="*70)

    if step95_success and step10_success:
        logger.info("🎉 All demonstrations completed successfully!")
        logger.info("📚 Check the generated report files in:")
        logger.info("   • src/training/reports/step09_5_hmm_lm_generalist_training/")
        logger.info("   • src/training/reports/step10_unified_regime_intelligence/")
    else:
        logger.warning("⚠️ Some demonstrations had issues - check logs for details")

    return step95_success and step10_success

if __name__ == "__main__":
    logger = system_logger.getChild('Combined.Demo.Main')
    logger.info("🎯 Starting Combined Step9_5 & Step10 Enhanced Reporting Demonstration")

    success = demonstrate_combined_reporting()

    if success:
        logger.info("🎉 Combined demonstration completed successfully!")
    else:
        logger.error("❌ Combined demonstration failed - check logs for details")
        sys.exit(1)
