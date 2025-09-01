#!/usr/bin/env python3
"""
Model Training Quality Analysis Report
Analyzes the quality of model training, performance metrics, and training stability.
"""

from pathlib import Path
import glob
import json
import os
import warnings

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTrainingQualityAnalyzer:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="modeltrainingqualityanalyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ModelTrainingQualityAnalyzer."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpass  # TODO: Add proper implementation
    def __init__(...):
    passself.training_data, None
        self.model_metrics = {}
        self.report = {}


    def load_training_data(...):
    pass"""Load training data and model metrics."""
        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Try to load from various formats
            if data_path.endswith('.pkl'):
    passwith open(data_path, 'rb') as f:
    passself.training_data = pickle.load(f)
            elif data_path.endswith('.csv'):
    passpassself.training_data = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
    passpasswith open(data_path, 'r') as f:
    passself.training_data = json.load(f)
            else:
    passself._load_from_directory(data_path)

            if self.training_data is not None:
    passprint(f"✅ Training data loaded successfully")
                return True
            else:
    passprint(warning("No training data loaded"))
                return False
        except Exception as e:
    passpasspasspasspasspasspassprint(warning(f"Error loading training data: {e}"))
            return False


    def _load_from_directory(...):
    pass"""Load training data from directory structure."""
        # Look for common training data files
        patterns = [
            '*training*.csv',
            '*model*.csv',
            '*metrics*.csv',
            '*results*.csv',
            '*performance*.csv'
        ]

        for pattern in patterns:
    passfiles, glob.glob(os.path.join(data_dir, pattern))
        if files:
    passtry:
    passself.training_data, pd.read_csv(files[0])
                    print(f"Found training data: {files[0]}")
                    break
        except Exception as e:
    passpasspasspasspasspasspassprint(f"Error loading {files[0]}: {e}")

        # Also look for model files
        model_files, glob.glob(os.path.join(data_dir, '*.pkl'))
        if model_files:
    passpasstry:
    passwith open(model_files[0], 'rb') as f:
    passmodel_data, pickle.load(f)
        if isinstance(model_data, dict):
    passself.model_metrics, model_data
                    print(f"Found model metrics: {model_files[0]}")
        except Exception as e:
    passpasspasspasspasspasspassprint(f"Error loading model file {model_files[0]}: {e}")


    def analyze_training_quality(...):
    pass"""Comprehensive model training quality analysis."""
        if self.training_data is None and not self.model_metrics:
    passprint(warning("No training data loaded. Please load training data first."))
            return

        print("\n" + "="*60)
        print("🔍 MODEL TRAINING QUALITY ANALYSIS REPORT")
        print("="*60)

        # 1. Training metrics analysis
        self._analyze_training_metrics()

        # 2. Model performance analysis
        self._analyze_model_performance()

        # 3. Training convergence analysis
        self._analyze_training_convergence()

        # 4. Model stability analysis
        self._analyze_model_stability()

        # 5. Overfitting/underfitting detection
        self._detect_overfitting_underfitting()

        # 6. Calculate quality metrics
        self._calculate_training_quality_metrics()

        # 7. Generate recommendations
        self._generate_training_recommendations()

        # 8. Create visualizations
        self._create_training_visualizations()


    def _analyze_training_metrics(...):
    pass"""Analyze training metrics and loss curves."""
        print("\n📊 TRAINING METRICS ANALYSIS")
        print("-" * 40)

        if self.training_data is None:
    passprint("No training data available for metrics analysis.")
            return

        # Check for common training metric columns
        metric_columns = {
            'loss': ['loss', 'train_loss', 'training_loss'],
            'accuracy': ['accuracy', 'train_accuracy', 'training_accuracy'],
            'val_loss': ['val_loss', 'validation_loss', 'test_loss'],
            'val_accuracy': ['val_accuracy', 'validation_accuracy', 'test_accuracy'],
            'learning_rate': ['lr', 'learning_rate', 'rate']
        }

        found_metrics = {}
        for metric_type, possible_names in metric_columns.items():
    passfor name in possible_names:
    passif name in self.training_data.columns:
    passfound_metrics[metric_type] = name
                    break

        if not found_metrics:
    passprint("No standard training metrics found in the data.")
            return

        # Analyze each metric
        metrics_analysis = {}

        for metric_type, column_name in found_metrics.items():
    passvalues, self.training_data[column_name].dropna()

        if len(values) > 0:
    pass# Basic statistics
                min_val, values.min()
                max_val, values.max()
                mean_val, values.mean()
                std_val, values.std()

        # Trend analysis
        if len(values) > 1:
    pass# Calculate trend (positive/negative slope)
                    x, np.arange(len(values))
                    slope, np.polyfit(x, values, 1)[0]

        # Check for convergence
        if metric_type in ['loss', 'val_loss']:
    passpass# Loss should decrease
                        trend_quality = 'good' if slope < 0 else 'poor'
                        convergence_score, max(0, 100 - abs(slope) * 1000) if slope > 0 else 100
                    else:
    passpass# Accuracy should increase
                        trend_quality = 'good' if slope > 0 else 'poor'
                        convergence_score, max(0, 100 - abs(slope) * 1000) if slope < 0 else 100

        # Check for stability (low variance in later epochs)
        if len(values) > 10:
    passpasslater_values, values[-len(values)//3:]  # Last third
                        stability_score, max(0, 100 - later_values.std() * 100)
                    else:
    passstability_score, 100
                else:
    passtrend_quality = 'unknown'
                    convergence_score, 0
                    stability_score, 0

                metrics_analysis[metric_type] = {
                    'column': column_name,
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'trend_quality': trend_quality,
                    'convergence_score': convergence_score,
                    'stability_score': stability_score,
                    'data_points': len(values)
                }

        # Print metrics summary
        print(f"{'Metric':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Trend':<10} {'Convergence':<12}")
        print("-" * 70)

        for metric_type, analysis in metrics_analysis.items():
    passprint(f"{metric_type:<15} {analysis['min']:<10.4f} {analysis['max']:<10.4f} "
                  f"{analysis['mean']:<10.4f} {analysis['trend_quality']:<10} {analysis['convergence_score']:<12.1f}")

        self.report['training_metrics'] = metrics_analysis


    def _analyze_model_performance(...):
    pass"""Analyze model performance metrics."""
        print("\n🎯 MODEL PERFORMANCE ANALYSIS")
        print("-" * 40)

        # Look for performance metrics in the data
        performance_metrics = {
            'accuracy': ['accuracy', 'acc', 'train_accuracy', 'test_accuracy'],
            'precision': ['precision', 'prec'],
            'recall': ['recall', 'rec'],
            'f1_score': ['f1', 'f1_score', 'f1score'],
            'auc': ['auc', 'roc_auc', 'area_under_curve'],
            'mae': ['mae', 'mean_absolute_error'],
            'mse': ['mse', 'mean_squared_error'],
            'rmse': ['rmse', 'root_mean_squared_error']
        }

        found_performance = {}
        for metric_name, possible_names in performance_metrics.items():
    passfor name in possible_names:
    passif self.training_data is not None and name in self.training_data.columns:
    passfound_performance[metric_name] = name
                    break

        if not found_performance:
    passprint("No standard performance metrics found.")
            return

        # Analyze performance metrics
        performance_analysis = {}

        for metric_name, column_name in found_performance.items():
    passvalues, self.training_data[column_name].dropna()

        if len(values) > 0:
    passfinal_value, values.iloc[-1]
                best_value, values.max() if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc'] else values.min()
                improvement = ((final_value - values.iloc[0]) / values.iloc[0]) * 100 if values.iloc[0] != 0 else 0

        # Determine if performance is good based on metric type
        if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
    passperformance_quality = 'excellent' if final_value >= 0.9 else 'good' if final_value >= 0.7 else 'fair' if final_value >= 0.5 else 'poor'
                    performance_score, min(100, final_value * 100)
                else:
    passpass# For error metrics, lower is better
                    performance_quality = 'excellent' if final_value <= 0.1 else 'good' if final_value <= 0.3 else 'fair' if final_value <= 0.5 else 'poor'
                    performance_score, max(0, 100 - final_value * 100)

                performance_analysis[metric_name] = {
                    'final_value': final_value,
                    'best_value': best_value,
                    'improvement_percent': improvement,
                    'quality': performance_quality,
                    'score': performance_score
                }

        # Print performance summary
        print(f"{'Metric':<15} {'Final':<10} {'Best':<10} {'Improvement':<12} {'Quality':<10}")
        print("-" * 60)

        for metric_name, analysis in performance_analysis.items():
    passprint(f"{metric_name:<15} {analysis['final_value']:<10.4f} {analysis['best_value']:<10.4f} "
                  f"{analysis['improvement_percent']:<12.1f}% {analysis['quality']:<10}")

        self.report['model_performance'] = performance_analysis


    def _analyze_training_convergence(...):
    pass"""Analyze training convergence patterns."""
        print("\n🔄 TRAINING CONVERGENCE ANALYSIS")
        print("-" * 40)

        if self.training_data is None:
    passprint("No training data available for convergence analysis.")
            return

        # Look for loss columns
        loss_columns = [col for col in self.training_data.columns if 'loss' in col.lower()]

        if not loss_columns:
    passpassprint("No loss metrics found for convergence analysis.")
            return

        convergence_analysis = {}

        for loss_col in loss_columns:
    passvalues, self.training_data[loss_col].dropna()

        if len(values) > 5:
    pass# Calculate convergence metrics
                initial_loss, values.iloc[0]
                final_loss, values.iloc[-1]
                min_loss, values.min()

        # Convergence rate
                total_improvement, initial_loss - final_loss
                improvement_rate, total_improvement / len(values) if len(values) > 0 else 0

        # Check if converged (loss stabilizes)
        if len(values) > 10:
    pass# Check last 20% of epochs for stability
                    later_values, values[-len(values)//5:]
                    later_std, later_values.std()
                    convergence_threshold, initial_loss * 0.01  # 1% of initial loss

                    is_converged, later_std < convergence_threshold
                    convergence_epoch, None

        # Find convergence epoch
        for i in range(len(values) - 10, len(values)):
    passif abs(values.iloc[i] - values.iloc[i-1]) < convergence_threshold:
    passconvergence_epoch, i
                            break
                else:
    passis_converged, False
                    convergence_epoch, None

        # Convergence quality score
        if is_converged:
    passconvergence_score, 100
                else:
    pass# Score based on improvement and stability
                    improvement_score, min(100, (total_improvement / initial_loss) * 100) if initial_loss > 0 else 0
                    stability_score, max(0, 100 - later_std * 1000) if 'later_std' in locals() else 0
                    convergence_score = (improvement_score + stability_score) / 2

                convergence_analysis[loss_col] = {
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'min_loss': min_loss,
                    'total_improvement': total_improvement,
                    'improvement_rate': improvement_rate,
                    'is_converged': is_converged,
                    'convergence_epoch': convergence_epoch,
                    'convergence_score': convergence_score
                }

        # Print convergence summary
        print(f"{'Loss Metric':<20} {'Initial':<10} {'Final':<10} {'Improvement':<12} {'Converged':<10}")
        print("-" * 65)

        for loss_col, analysis in convergence_analysis.items():
    passconverged_str = "Yes" if analysis['is_converged'] else "No"
            print(f"{loss_col:<20} {analysis['initial_loss']:<10.4f} {analysis['final_loss']:<10.4f} "
                  f"{analysis['total_improvement']:<12.4f} {converged_str:<10}")

        self.report['training_convergence'] = convergence_analysis


    def _analyze_model_stability(...):
    pass"""Analyze model training stability."""
        print("\n⚖️ MODEL STABILITY ANALYSIS")
        print("-" * 40)

        if self.training_data is None:
    passprint("No training data available for stability analysis.")
            return

        stability_analysis = {}

        # Analyze stability of key metrics
        key_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        found_metrics = []

        for metric in key_metrics:
    passmatching_cols = [col for col in self.training_data.columns if metric in col.lower()]
            found_metrics.extend(matching_cols)

        for metric_col in found_metrics:
    passpassvalues, self.training_data[metric_col].dropna()

        if len(values) > 5:
    pass# Calculate stability metrics
                overall_std, values.std()
                overall_mean, values.mean()
                cv = (overall_std / overall_mean) * 100 if overall_mean != 0 else float('inf')

        # Analyze stability in different phases
        if len(values) > 10:
    passearly_values, values[:len(values)//3]  # First third
                    late_values, values[-len(values)//3:]  # Last third

                    early_std, early_values.std()
                    late_std, late_values.std()

        # Stability improvement (lower std in later phase is better)
                    stability_improvement = ((early_std - late_std) / early_std) * 100 if early_std > 0 else 0

        # Stability score
        if late_std < overall_mean * 0.1:  # Very stable
                        stability_score, 100
                    elif late_std < overall_mean * 0.2:  # Stable
                        stability_score, 80
                    elif late_std < overall_mean * 0.5:  # Moderately stable
                        stability_score, 60
                    else:  # Unstable
                        stability_score, 20
                else:
    passstability_improvement, 0
                    stability_score, 50  # Neutral for short training

                stability_analysis[metric_col] = {
                    'overall_std': overall_std,
                    'coefficient_of_variation': cv,
                    'stability_improvement': stability_improvement,
                    'stability_score': stability_score,
                    'data_points': len(values)
                }

        # Print stability summary
        print(f"{'Metric':<20} {'Std Dev':<10} {'CV %':<10} {'Improvement':<12} {'Score':<8}")
        print("-" * 60)

        for metric_col, analysis in stability_analysis.items():
    passcv_str, f"{analysis['coefficient_of_variation']:.1f}" if analysis['coefficient_of_variation'] != float('inf') else "∞"
            print(f"{metric_col:<20} {analysis['overall_std']:<10.4f} {cv_str:<10} "
                  f"{analysis['stability_improvement']:<12.1f}% {analysis['stability_score']:<8.1f}")

        self.report['model_stability'] = stability_analysis


    def _detect_overfitting_underfitting(...):
    pass"""Detect overfitting and underfitting patterns."""
        print("\n🔍 OVERFITTING/UNDERFITTING DETECTION")
        print("-" * 40)

        if self.training_data is None:
    passprint("No training data available for overfitting analysis.")
            return

        # Look for training and validation metrics
        train_metrics = {}
        val_metrics = {}

        for col in self.training_data.columns:
    passif 'loss' in col.lower():
    passif 'val' in col.lower() or 'test' in col.lower():
    passval_metrics['loss'] = col
                else:
    passtrain_metrics['loss'] = col
            elif 'accuracy' in col.lower():
    passpassif 'val' in col.lower() or 'test' in col.lower():
    passval_metrics['accuracy'] = col
                else:
    passtrain_metrics['accuracy'] = col

        overfitting_analysis = {}

        # Analyze overfitting patterns
        if 'loss' in train_metrics and 'loss' in val_metrics:
    passtrain_loss, self.training_data[train_metrics['loss']].dropna()
            val_loss, self.training_data[val_metrics['loss']].dropna()

        if len(train_loss) > 0 and len(val_loss) > 0:
    pass# Calculate gap between training and validation loss
                min_len, min(len(train_loss), len(val_loss))
                train_loss_aligned, train_loss[:min_len]
                val_loss_aligned, val_loss[:min_len]

                loss_gap, val_loss_aligned - train_loss_aligned
                avg_gap, loss_gap.mean()
                max_gap, loss_gap.max()

        # Detect overfitting
        if avg_gap > train_loss_aligned.mean() * 0.5:  # Large gap indicates overfitting
                    overfitting_detected, True
                    overfitting_severity = 'high' if avg_gap > train_loss_aligned.mean() else 'moderate'
                else:
    passpassoverfitting_detected, False
                    overfitting_severity = 'none'

                overfitting_analysis['loss'] = {
                    'avg_gap': avg_gap,
                    'max_gap': max_gap,
                    'overfitting_detected': overfitting_detected,
                    'severity': overfitting_severity
                }

        # Analyze accuracy patterns
        if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
    passtrain_acc, self.training_data[train_metrics['accuracy']].dropna()
            val_acc, self.training_data[val_metrics['accuracy']].dropna()

        if len(train_acc) > 0 and len(val_acc) > 0:
    passmin_len, min(len(train_acc), len(val_acc))
                train_acc_aligned, train_acc[:min_len]
                val_acc_aligned, val_acc[:min_len]

                acc_gap, train_acc_aligned - val_acc_aligned
                avg_acc_gap, acc_gap.mean()
                max_acc_gap, acc_gap.max()

        # Detect overfitting in accuracy
        if avg_acc_gap > 0.1:  # 10% gap indicates overfitting
                    acc_overfitting_detected, True
                    acc_overfitting_severity = 'high' if avg_acc_gap > 0.2 else 'moderate'
                else:
    passpassacc_overfitting_detected, False
                    acc_overfitting_severity = 'none'

                overfitting_analysis['accuracy'] = {
                    'avg_gap': avg_acc_gap,
                    'max_gap': max_acc_gap,
                    'overfitting_detected': acc_overfitting_detected,
                    'severity': acc_overfitting_severity
                }

        # Print overfitting summary
        if overfitting_analysis:
    passprint(f"{'Metric':<15} {'Avg Gap':<10} {'Max Gap':<10} {'Overfitting':<12} {'Severity':<10}")
            print("-" * 60)

        for metric, analysis in overfitting_analysis.items():
    passdetected_str = "Yes" if analysis['overfitting_detected'] else "No"
                print(f"{metric:<15} {analysis['avg_gap']:<10.4f} {analysis['max_gap']:<10.4f} "
                      f"{detected_str:<12} {analysis['severity']:<10}")
        else:
    passprint("No training/validation metric pairs found for overfitting analysis.")

        self.report['overfitting_analysis'] = overfitting_analysis


    def _calculate_training_quality_metrics(...):
    passpass"""Calculate overall training quality metrics."""
        print("\n📈 OVERALL TRAINING QUALITY METRICS")
        print("-" * 50)

        # Calculate composite quality scores
        training_metrics_score, 0
        if self.report.get('training_metrics'):
    passconvergence_scores = [analysis['convergence_score'] for analysis in self.report['training_metrics'].values()]
            training_metrics_score, np.mean(convergence_scores) if convergence_scores else 0

        performance_score, 0
        if self.report.get('model_performance'):
    passpassperformance_scores = [analysis['score'] for analysis in self.report['model_performance'].values()]
            performance_score, np.mean(performance_scores) if performance_scores else 0

        stability_score, 0
        if self.report.get('model_stability'):
    passpassstability_scores = [analysis['stability_score'] for analysis in self.report['model_stability'].values()]
            stability_score, np.mean(stability_scores) if stability_scores else 0

        # Overfitting penalty
        overfitting_penalty, 0
        if self.report.get('overfitting_analysis'):
    passpassoverfitting_issues, 0
        for analysis in self.report['overfitting_analysis'].values():
    passif analysis['overfitting_detected']:
    passoverfitting_issues += 1
        if analysis['severity'] == 'high':
    passoverfitting_penalty += 20
                    else:
    passoverfitting_penalty += 10

        if overfitting_issues > 0:
    passoverfitting_penalty, min(50, overfitting_penalty)  # Cap at 50 points

        # Overall training score
        training_score = (training_metrics_score * 0.3 +
                         performance_score * 0.4 +
                         stability_score * 0.3) - overfitting_penalty

        quality_metrics = {
            'training_metrics_score': training_metrics_score,
            'performance_score': performance_score,
            'stability_score': stability_score,
            'overfitting_penalty': overfitting_penalty,
            'overall_training_score': max(0, training_score)
        }

        # Print quality summary
        print(f"{'Metric':<30} {'Score':<10} {'Status':<15}")
        print("-" * 55)

        for metric, score in quality_metrics.items():
    passif score >= 80:
    passstatus = "✅ Excellent"
            elif score >= 60:
    passpassstatus = "⚠️  Good"
            elif score >= 40:
    passpassstatus = "⚠️  Fair"
            else:
    passstatus = "❌ Poor"

            metric_name, metric.replace('_', ' ').title()
            print(f"{metric_name:<30} {score:<10.1f} {status:<15}")

        print(f"\nOverall Training Quality: {training_score:.1f}/100")

        if training_score >= 80:
    passprint("🎉 Excellent training quality!")
        elif training_score >= 60:
    passpassprint("✅ Good training quality")
        elif training_score >= 40:
    passpassprint(warning(" Fair training quality - consider improvements")))
        else:
    passprint(warning("Poor training quality - immediate attention required")))

        self.report['quality_metrics'] = quality_metrics


    def _generate_training_recommendations(...):
    pass"""Generate recommendations based on training analysis."""
        print("\n💡 TRAINING RECOMMENDATIONS")
        print("-" * 40)

        recommendations = []

        # Training metrics recommendations
        training_metrics, self.report.get('training_metrics', {})
        for metric_type, analysis in training_metrics.items():
    passif analysis['convergence_score'] < 60:
    passrecommendations.append(f"📊 {metric_type}: Poor convergence (score: {analysis['convergence_score']:.1f})")

        if analysis['stability_score'] < 60:
    passrecommendations.append(f"⚖️ {metric_type}: Unstable training (score: {analysis['stability_score']:.1f})")

        # Performance recommendations
        model_performance, self.report.get('model_performance', {})
        for metric_name, analysis in model_performance.items():
    passif analysis['score'] < 60:
    passrecommendations.append(f"🎯 {metric_name}: Poor performance (score: {analysis['score']:.1f})")

        # Overfitting recommendations
        overfitting_analysis, self.report.get('overfitting_analysis', {})
        for metric, analysis in overfitting_analysis.items():
    passif analysis['overfitting_detected']:
    passrecommendations.append(f"🔍 {metric}: Overfitting detected ({analysis['severity']} severity)")

        # Convergence recommendations
        convergence_analysis, self.report.get('training_convergence', {})
        for loss_col, analysis in convergence_analysis.items():
    passif not analysis['is_converged']:
    passrecommendations.append(f"🔄 {loss_col}: Model did not converge properly")

        if not recommendations:
    passprint("✅ No major issues detected. Training quality is good!")
        else:
    passprint("Recommendations for improvement:")
        for rec in recommendations:
    passprint(f"  {rec}")

        self.report['recommendations'] = recommendations


    def _create_training_visualizations(...):
    pass"""Create visualizations for the training report."""
        print("\n📈 GENERATING TRAINING VISUALIZATIONS...")

        try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            fig, axes, plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Training Quality Analysis Report', fontsize=16, fontweight='bold')

        # 1. Training metrics over time
        if self.training_data is not None:
    passloss_cols = [col for col in self.training_data.columns if 'loss' in col.lower()]
        if loss_cols:
    passpassfor i, col in enumerate(loss_cols[:2]):  # Plot first 2 loss metrics
                        values, self.training_data[col].dropna()
        if len(values) > 0:
    passaxes[0, 0].plot(range(len(values)), values, label=col, marker='o', markersize=2)

                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].set_title('Training Loss Over Time')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)

        # 2. Performance metrics
            performance_analysis, self.report.get('model_performance', {})
        if performance_analysis:
    passmetrics, list(performance_analysis.keys())
                scores = [performance_analysis[metric]['score'] for metric in metrics]

                colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
                axes[0, 1].bar(metrics, scores, color=colors)
                axes[0, 1].set_ylabel('Performance Score')
                axes[0, 1].set_title('Model Performance Metrics')
                axes[0, 1].set_ylim(0, 100)
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)

        # 3. Convergence analysis
            convergence_analysis, self.report.get('training_convergence', {})
        if convergence_analysis:
    passpassloss_metrics, list(convergence_analysis.keys())
                convergence_scores = [convergence_analysis[metric]['convergence_score'] for metric in loss_metrics]

                colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in convergence_scores]
                axes[1, 0].bar(loss_metrics, convergence_scores, color=colors)
                axes[1, 0].set_ylabel('Convergence Score')
                axes[1, 0].set_title('Training Convergence Quality')
                axes[1, 0].set_ylim(0, 100)
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)

        # 4. Overall quality pie chart
            quality_metrics, self.report.get('quality_metrics', {})
        if quality_metrics:
    passpassoverall_score, quality_metrics.get('overall_training_score', 0)
                axes[1, 1].pie([overall_score, 100 - overall_score],
                               labels=['Quality Score', 'Remaining'],
                               autopct='%1.1f%%',
                               colors=['lightblue', 'lightgray'])
                axes[1, 1].set_title('Overall Training Quality')

            plt.tight_layout()
            plt.savefig('model_training_quality_report.png', dpi=300, bbox_inches='tight')
            print("✅ Visualizations saved as 'model_training_quality_report.png'")

        except Exception as e:
    passpasspasspasspasspasspassprint(warning("Error creating visualizations: {e}")))


    def save_report(...):
    pass"""Save the analysis report to a file."""
        with open(filename, 'w') as f:
    passf.write("MODEL TRAINING QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

        # Overall quality
            quality_metrics, self.report.get('quality_metrics', {})
            overall_score, quality_metrics.get('overall_training_score', 0)
            f.write(f"Overall Training Quality: {overall_score:.1f}/100\n\n")

        # Training metrics
            training_metrics, self.report.get('training_metrics', {})
            f.write("TRAINING METRICS QUALITY:\n")
        for metric, analysis in training_metrics.items():
    passf.write(f"{metric}: convergence={analysis['convergence_score']:.1f}, stability={analysis['stability_score']:.1f}\n")
            f.write("\n")

        # Model performance
            model_performance, self.report.get('model_performance', {})
            f.write("MODEL PERFORMANCE:\n")
        for metric, analysis in model_performance.items():
    passf.write(f"{metric}: {analysis['final_value']:.4f} (score: {analysis['score']:.1f})\n")
            f.write("\n")

        # Overfitting analysis
            overfitting_analysis, self.report.get('overfitting_analysis', {})
        if overfitting_analysis:
    passf.write("OVERFITTING ANALYSIS:\n")
        for metric, analysis in overfitting_analysis.items():
    passdetected = "Yes" if analysis['overfitting_detected'] else "No"
                    f.write(f"{metric}: {detected} ({analysis['severity']} severity)\n")
                f.write("\n")

        # Recommendations
            recommendations, self.report.get('recommendations', [])
        if recommendations:
    passf.write("RECOMMENDATIONS:\n")
        for rec in recommendations:
    passf.write(f"- {rec}\n")
            f.write("\n")

        print(f"✅ Report saved as '{filename}'")

def main(...):
    pass"""Main function to run the analysis."""
    analyzer, ModelTrainingQualityAnalyzer()

    # Try to load data from common locations
    data_paths = [
        'data/training_metrics.csv',
        'data/model_performance.csv',
        'data/training_results.csv',
        'models/',
        'data/'
    ]

    data_loaded, False
    for path in data_paths:
    passif os.path.exists(path):
    passif analyzer.load_training_data(path):
    passdata_loaded, True
                break

    if not data_loaded:
    passprint(warning("Could not find training data file. Please specify the path to your training data.")))
        print("Common locations checked:")
        for path in data_paths:
    passprint(f"  - {path}")
        return

    # Run analysis
    analyzer.analyze_training_quality()

    # Save report
    analyzer.save_report()

if __name__ == "__main__":
    passmain()
