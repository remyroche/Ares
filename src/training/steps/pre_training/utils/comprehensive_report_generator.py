"""
Comprehensive Report Generator for Pre-Training Steps

This module provides utilities for generating detailed .md reports with financial,
technical, and process troubleshooting metrics for each pre-training step.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import psutil
import time

class ComprehensiveReportGenerator:
    """
    Generates comprehensive .md reports for pre-training steps with detailed
    metrics for financial, technical, and process troubleshooting.
    """
    
    def __init__(self, outcomes_dir: str = "outcomes"):
        """
        Initialize the report generator.
        
        Args:
            outcomes_dir: Directory to store reports (default: "outcomes")
        """
        self.outcomes_dir = Path(outcomes_dir)
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, 
                       step_name: str,
                       symbol: str,
                       exchange: str,
                       timeframe: str,
                       direction: str,
                       execution_mode: str,
                       general_metrics: Dict[str, Any],
                       financial_metrics: Optional[Dict[str, Any]] = None,
                       technical_metrics: Optional[Dict[str, Any]] = None,
                       process_metrics: Optional[Dict[str, Any]] = None,
                       artifacts_generated: Optional[List[str]] = None,
                       dependencies_used: Optional[Dict[str, List[str]]] = None,
                       errors: Optional[List[str]] = None,
                       warnings: Optional[List[str]] = None) -> str:
        """
        Generate a comprehensive .md report for a step.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
            execution_mode: Execution mode
            general_metrics: General step metrics
            financial_metrics: Financial performance metrics
            technical_metrics: Technical performance metrics
            process_metrics: Process execution metrics
            artifacts_generated: List of artifacts generated
            dependencies_used: Dependencies used by the step
            errors: List of errors encountered
            warnings: List of warnings encountered
            
        Returns:
            Path to the generated report file
        """
        try:
            # Generate report content
            report_content = self._generate_report_content(
                step_name, symbol, exchange, timeframe, direction, execution_mode,
                general_metrics, financial_metrics, technical_metrics, process_metrics,
                artifacts_generated, dependencies_used, errors, warnings
            )
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{step_name}_comprehensive_report_{timestamp}.md"
            report_path = self.outcomes_dir / filename
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return str(report_path)
            
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            return ""
    
    def _generate_report_content(self, 
                               step_name: str,
                               symbol: str,
                               exchange: str,
                               timeframe: str,
                               direction: str,
                               execution_mode: str,
                               general_metrics: Dict[str, Any],
                               financial_metrics: Optional[Dict[str, Any]],
                               technical_metrics: Optional[Dict[str, Any]],
                               process_metrics: Optional[Dict[str, Any]],
                               artifacts_generated: Optional[List[str]],
                               dependencies_used: Optional[Dict[str, List[str]]],
                               errors: Optional[List[str]],
                               warnings: Optional[List[str]]) -> str:
        """Generate the markdown report content."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        report = f"""# {step_name.replace('_', ' ').title()} - Comprehensive Analysis Report

## 📊 Executive Summary

**Generated:** {timestamp}  
**Symbol:** {symbol}  
**Exchange:** {exchange}  
**Timeframe:** {timeframe}  
**Direction:** {direction}  
**Execution Mode:** {execution_mode}  

---

## 💰 Financial Metrics

### Performance Indicators
"""
        
        if financial_metrics:
            report += self._format_financial_metrics(financial_metrics)
        else:
            report += """*No financial metrics available for this step.*

"""
        
        report += """## 🔧 Technical Metrics

### System Performance
"""
        
        if technical_metrics:
            report += self._format_technical_metrics(technical_metrics)
        else:
            report += """*No technical metrics available for this step.*

"""
        
        report += """## ⚙️ Process Metrics

### Execution Analysis
"""
        
        if process_metrics:
            report += self._format_process_metrics(process_metrics)
        else:
            report += """*No process metrics available for this step.*

"""
        
        report += f"""## 📈 General Metrics

### Step Performance
"""
        
        if general_metrics:
            report += self._format_general_metrics(general_metrics)
        else:
            report += """*No general metrics available.*

"""
        
        report += f"""## 🔍 Troubleshooting Guide

### Common Issues & Solutions

#### Data Quality Issues
- **Missing Data**: Check if previous steps completed successfully
- **Data Alignment**: Verify index alignment between features and targets
- **Memory Issues**: Monitor memory usage during processing

#### Performance Issues  
- **Slow Execution**: Check system resources and optimization settings
- **Memory Overflow**: Reduce batch sizes or enable memory optimization
- **Disk I/O**: Monitor disk usage and cleanup old artifacts

#### Financial Analysis Issues
- **Poor Performance**: Review feature selection and target alignment
- **Overfitting**: Check for data leakage and temporal alignment
- **Correlation Issues**: Analyze feature correlation matrices

### Diagnostic Commands

```bash
# Check system resources
htop
df -h
free -h

# Check artifact storage
ls -la artifacts/pre_training/artifact_store/

# Check step execution logs
tail -f logs/{step_name}.log
```

## 📋 Artifact Inventory

### Generated Artifacts
"""
        
        if artifacts_generated:
            for artifact in artifacts_generated:
                report += f"- **{artifact}**: Generated successfully\n"
        else:
            report += "*No artifacts found for this step.*\n"
        
        report += f"""
### Dependencies Used
"""
        
        if dependencies_used:
            for source_step, artifacts in dependencies_used.items():
                report += f"- **From {source_step}**: {', '.join(artifacts)}\n"
        else:
            report += "*No dependencies tracked.*\n"
        
        report += f"""
## 🚨 Error Analysis

### Recent Errors
"""
        
        if errors:
            for error in errors:
                report += f"- **Error**: {error}\n"
        else:
            report += "*No errors detected in this execution.*\n"
        
        report += f"""
### Warning Indicators
"""
        
        if warnings:
            for warning in warnings:
                report += f"- **Warning**: {warning}\n"
        else:
            report += "*No warnings detected in this execution.*\n"
        
        report += f"""
### System Health Indicators
- Monitor memory usage if > 80%
- Check disk space if < 10% free
- Review feature correlation if > 0.95

## 📊 Recommendations

### Performance Optimization
1. **Memory Management**: Enable memory optimization for large datasets
2. **Parallel Processing**: Use multi-threading for independent operations
3. **Caching**: Implement intelligent caching for repeated operations

### Financial Optimization
1. **Feature Selection**: Focus on features with highest information content
2. **Target Alignment**: Ensure targets are properly aligned with trading objectives
3. **Risk Management**: Implement proper risk controls and position sizing

### Process Improvement
1. **Monitoring**: Set up comprehensive monitoring and alerting
2. **Logging**: Enhance logging for better troubleshooting
3. **Validation**: Add comprehensive data validation checks

---

*Report generated by Ares Pre-Training Pipeline*  
*For technical support, check the troubleshooting guide above.*
"""
        
        return report

    def _format_financial_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format financial metrics section."""
        content = ""
        
        # Performance Indicators (traditional metrics)
        if 'sharpe_ratio' in metrics:
            content += f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.4f}\n"
        if 'max_drawdown' in metrics:
            content += f"- **Max Drawdown**: {metrics['max_drawdown']:.4f}\n"
        if 'return_metrics' in metrics:
            content += f"- **Return Metrics**: {metrics['return_metrics']}\n"
        if 'risk_metrics' in metrics:
            content += f"- **Risk Metrics**: {metrics['risk_metrics']}\n"
        if 'calmar_ratio' in metrics:
            content += f"- **Calmar Ratio**: {metrics['calmar_ratio']:.4f}\n"
        if 'sortino_ratio' in metrics:
            content += f"- **Sortino Ratio**: {metrics['sortino_ratio']:.4f}\n"
        if 'information_ratio' in metrics:
            content += f"- **Information Ratio**: {metrics['information_ratio']:.4f}\n"
        
        # Feature Performance (traditional metrics)
        if 'feature_performance' in metrics:
            content += f"- **Feature Performance**: {metrics['feature_performance']}\n"
        if 'target_correlation' in metrics:
            content += f"- **Target Correlation**: {metrics['target_correlation']:.4f}\n"
        if 'feature_importance_scores' in metrics:
            content += f"- **Feature Importance Scores**: {metrics['feature_importance_scores']}\n"
        if 'stability_scores' in metrics:
            content += f"- **Stability Scores**: {metrics['stability_scores']}\n"
        if 'redundancy_scores' in metrics:
            content += f"- **Redundancy Scores**: {metrics['redundancy_scores']}\n"
        
        # Selection Quality (traditional metrics)
        if 'selection_accuracy' in metrics:
            content += f"- **Selection Accuracy**: {metrics['selection_accuracy']:.4f}\n"
        if 'selection_precision' in metrics:
            content += f"- **Selection Precision**: {metrics['selection_precision']:.4f}\n"
        if 'selection_recall' in metrics:
            content += f"- **Selection Recall**: {metrics['selection_recall']:.4f}\n"
        if 'selection_f1_score' in metrics:
            content += f"- **Selection F1 Score**: {metrics['selection_f1_score']:.4f}\n"
        if 'selection_auc_score' in metrics:
            content += f"- **Selection AUC Score**: {metrics['selection_auc_score']:.4f}\n"
        
        # Optimization Results (traditional metrics)
        if 'best_periods' in metrics:
            content += f"- **Best Periods**: {metrics['best_periods']}\n"
        if 'best_lookbacks' in metrics:
            content += f"- **Best Lookbacks**: {metrics['best_lookbacks']}\n"
        if 'optimization_convergence' in metrics:
            content += f"- **Optimization Convergence**: {metrics['optimization_convergence']}\n"
        if 'hyperparameter_sensitivity' in metrics:
            content += f"- **Hyperparameter Sensitivity**: {metrics['hyperparameter_sensitivity']}\n"
        
        # Labeling-Specific Metrics (nested structure support)
        if 'labeling_method' in metrics:
            content += f"\n### Labeling Configuration\n"
            content += f"- **Method**: {metrics['labeling_method']}\n"
        
        if 'volatility_config' in metrics:
            vc = metrics['volatility_config']
            base_threshold = vc.get('base_threshold', 0) or 0
            volatility_threshold = vc.get('volatility_threshold', 0) or 0
            content += f"- **Base Threshold**: {base_threshold:.1%} ({volatility_threshold*100:.1f}%)\n"
            content += f"- **Lookahead Periods**: {vc.get('lookahead_periods', 'N/A')}\n"
            content += f"- **Local Maxima Detection**: {vc.get('local_maxima_detection', 'N/A')}\n"
            content += f"- **Volatility Adaptation**: {vc.get('volatility_adaptation', 'N/A')}\n"
            content += f"- **Quality Threshold**: {vc.get('quality_threshold', 0.4):.1%}\n"
            content += f"- **Predictability Threshold**: {vc.get('predictability_threshold', 0.3):.1%}\n"
        
        if 'opportunity_detection' in metrics:
            content += f"\n### Opportunity Detection\n"
            od = metrics['opportunity_detection']
            content += f"- **Total Samples Processed**: {od.get('total_samples_processed', 0):,}\n"
            content += f"- **Opportunities Detected**: {od.get('total_opportunities_detected', 0):,}\n"
            content += f"- **Detection Rate**: {od.get('opportunity_detection_rate', 0):.2f}%\n"
            content += f"- **Long Opportunities**: {od.get('long_opportunities', 0):,}\n"
            content += f"- **Short Opportunities**: {od.get('short_opportunities', 0):,}\n"
            long_short_ratio = od.get('long_short_ratio', None)
            content += f"- **Long/Short Ratio**: {long_short_ratio:.2f}" if long_short_ratio is not None else "- **Long/Short Ratio**: N/A\n"
            # Add opportunities per day if available
            avg_ops_per_day = od.get('avg_opportunities_per_day', 0)
            if avg_ops_per_day > 0:
                content += f"- **Opportunities per Day**: {avg_ops_per_day:.2f} (target: ≤8/day)\n"
        
        if 'quality_filtering' in metrics:
            content += f"\n### Quality Filtering\n"
            qf = metrics['quality_filtering']
            content += f"- **High Quality Opportunities**: {qf.get('high_quality_opportunities', 0):,}\n"
            content += f"- **Filtered Opportunities**: {qf.get('filtered_opportunities', 0):,}\n"
            quality_acceptance = qf.get('quality_acceptance_rate', 0) or 0
            content += f"- **Quality Acceptance Rate**: {quality_acceptance:.2f}%\n"
            avg_confidence = qf.get('avg_confidence_score', 0) or 0
            content += f"- **Avg Confidence Score**: {avg_confidence:.3f}\n"
            avg_volatility = qf.get('avg_volatility_adaptation', 1) or 1
            content += f"- **Avg Volatility Adaptation**: {avg_volatility:.2f}x\n"
            min_vol = qf.get('min_volatility_adaptation', 0.8) or 0.8
            max_vol = qf.get('max_volatility_adaptation', 2.1) or 2.1
            content += f"- **Volatility Range**: {min_vol:.2f}x - {max_vol:.2f}x\n"
        
        if 'expected_performance' in metrics:
            content += f"\n### Expected Performance\n"
            ep = metrics['expected_performance']
            content += f"- **Expected Profit Target**: {ep.get('expected_profit_target', 'N/A')}\n"
            volatility_targets = ep.get('volatility_adjusted_targets', 'N/A')
            content += f"- **Volatility Adjusted Targets**: {volatility_targets}\n"
            quality_signals = ep.get('quality_weighted_signals', 'N/A')
            content += f"- **Quality Weighted Signals**: {quality_signals}\n"
            filtering_eff = ep.get('filtering_efficiency', 0) or 0
            content += f"- **Filtering Efficiency**: {filtering_eff:.1f}%\n"
            trading_strength = ep.get('trading_signal_strength', 0) or 0
            content += f"- **Trading Signal Strength**: {trading_strength:.3f}\n"
            market_adaptation = ep.get('market_regime_adaptation', 'N/A')
            content += f"- **Market Regime Adaptation**: {market_adaptation}\n"

        # Target Quality Metrics - comprehensive predictability assessment
        if 'target_quality_metrics' in metrics:
            content += f"\n### Target Quality Assessment\n"
            tqm = metrics['target_quality_metrics']

            # Overall assessment
            if 'overall_assessment' in tqm:
                overall = tqm['overall_assessment']
                quality_grade = overall.get('quality_grade', 'UNKNOWN')
                quality_score = overall.get('quality_score', 0.0)

                # Use emoji/icon based on grade
                grade_icon = {
                    'EXCELLENT': '🟢',
                    'GOOD': '🟡',
                    'FAIR': '🟠',
                    'POOR': '🔴',
                    'CRITICAL': '🚨'
                }.get(quality_grade, '⚪')

                content += f"\n**Overall Quality: {grade_icon} {quality_grade}** (Score: {quality_score:.1f}/100)\n\n"

                # Issues detected
                issues = overall.get('issues_detected', [])
                if issues:
                    content += f"**Issues Detected:**\n"
                    for issue in issues:
                        content += f"- ⚠️ {issue}\n"
                    content += "\n"

                # Strengths identified
                strengths = overall.get('strengths_identified', [])
                if strengths:
                    content += f"**Strengths Identified:**\n"
                    for strength in strengths:
                        content += f"- ✅ {strength}\n"
                    content += "\n"

                # Recommendations
                recommendations = overall.get('recommendations', [])
                if recommendations:
                    content += f"**Recommendations:**\n"
                    for rec in recommendations:
                        content += f"- 💡 {rec}\n"
                    content += "\n"

            # Variance & Distribution
            if 'variance_distribution' in tqm:
                vd = tqm['variance_distribution']
                content += f"#### a. Variance & Distribution\n"
                content += f"- **Mean**: {vd.get('mean', 0):.6f}\n"
                content += f"- **Variance**: {vd.get('variance', 0):.6f}\n"
                content += f"- **Std Deviation**: {vd.get('std_deviation', 0):.6f}\n"
                cv = vd.get('coefficient_of_variation', 'N/A')
                if cv != 'inf' and cv != 'N/A':
                    content += f"- **Coefficient of Variation**: {cv:.4f}\n"
                else:
                    content += f"- **Coefficient of Variation**: {cv}\n"
                content += f"- **Range**: [{vd.get('min', 0):.4f}, {vd.get('max', 0):.4f}]\n"
                content += f"- **Has Sufficient Variation**: {'✅ Yes' if vd.get('has_sufficient_variation', False) else '❌ No'}\n"
                interp = vd.get('interpretation', '')
                if interp:
                    content += f"- *{interp}*\n"
                content += "\n"

            # Autocorrelation & Self-Consistency
            if 'autocorrelation' in tqm:
                ac = tqm['autocorrelation']
                content += f"#### b. Autocorrelation & Self-Consistency\n"
                content += f"- **Lag-1 Autocorrelation**: {ac.get('lag1_autocorrelation', 0):.4f}\n"
                content += f"- **Mean Autocorrelation**: {ac.get('mean_autocorrelation', 0):.4f}\n"
                content += f"- **Max Abs Autocorrelation**: {ac.get('max_abs_autocorrelation', 0):.4f}\n"
                content += f"- **Has Temporal Structure**: {'✅ Yes' if ac.get('has_temporal_structure', False) else '❌ No'}\n"
                content += f"- **Is Highly Noisy**: {'⚠️ Yes' if ac.get('is_highly_noisy', False) else '✅ No'}\n"
                interp = ac.get('interpretation', '')
                if interp:
                    content += f"- *{interp}*\n"
                content += "\n"

            # Distribution & Outliers
            if 'distribution_outliers' in tqm:
                do = tqm['distribution_outliers']
                content += f"#### c. Distribution & Outliers\n"
                content += f"- **Median**: {do.get('median', 0):.6f}\n"
                content += f"- **IQR (25th-75th)**: {do.get('iqr', 0):.6f}\n"
                content += f"- **Skewness**: {do.get('skewness', 0):.4f}\n"
                content += f"- **Kurtosis**: {do.get('kurtosis', 0):.4f}\n"
                content += f"- **Outliers Detected**: {do.get('n_outliers', 0)} ({do.get('outlier_percentage', 0):.2f}%)\n"
                content += f"- **Is Symmetric**: {'✅ Yes' if do.get('is_symmetric', False) else '❌ No'}\n"
                content += f"- **Is Heavy-Tailed**: {'⚠️ Yes' if do.get('is_heavy_tailed', False) else '✅ No'}\n"
                interp = do.get('interpretation', '')
                if interp:
                    content += f"- *{interp}*\n"
                content += "\n"

            # Target Entropy
            if 'entropy' in tqm:
                ent = tqm['entropy']
                content += f"#### d. Target Entropy\n"
                content += f"- **Shannon Entropy**: {ent.get('shannon_entropy', 0):.4f}\n"
                content += f"- **Normalized Entropy**: {ent.get('normalized_entropy', 0):.4f} (0=deterministic, 1=random)\n"
                content += f"- **Is Predictable**: {'✅ Yes' if ent.get('is_predictable', False) else '❌ No'}\n"
                content += f"- **Is Highly Diverse**: {'⚠️ Yes' if ent.get('is_highly_diverse', False) else '✅ No'}\n"
                interp = ent.get('interpretation', '')
                if interp:
                    content += f"- *{interp}*\n"
                content += "\n"

            # Naive Feature-Free Baselines
            if 'baseline_predictors' in tqm:
                bp = tqm['baseline_predictors']
                content += f"#### e. Naive Feature-Free Baselines\n"

                # Mean predictor
                if 'mean_predictor' in bp:
                    mp = bp['mean_predictor']
                    content += f"- **Mean Predictor**: MSE={mp.get('mse', 0):.6f}, RMSE={mp.get('rmse', 0):.6f}\n"

                # Median predictor
                if 'median_predictor' in bp:
                    mdp = bp['median_predictor']
                    content += f"- **Median Predictor**: MSE={mdp.get('mse', 0):.6f}, RMSE={mdp.get('rmse', 0):.6f}\n"

                # Persistence predictor
                if 'persistence_predictor' in bp:
                    pp = bp['persistence_predictor']
                    content += f"- **Persistence Predictor**: MSE={pp.get('mse', 0):.6f}, RMSE={pp.get('rmse', 0):.6f}\n"

                # Random sampling
                if 'random_sampling_predictor' in bp:
                    rsp = bp['random_sampling_predictor']
                    content += f"- **Random Sampling**: MSE={rsp.get('mse', 0):.6f}, RMSE={rsp.get('rmse', 0):.6f}\n"

                # Zero predictor
                if 'zero_predictor' in bp:
                    zp = bp['zero_predictor']
                    content += f"- **Zero Predictor**: MSE={zp.get('mse', 0):.6f}, RMSE={zp.get('rmse', 0):.6f}\n"

                # Best baseline
                if 'best_baseline' in bp:
                    bb = bp['best_baseline']
                    content += f"\n- **🏆 Best Baseline**: {bb.get('name', 'unknown')} (MSE={bb.get('mse', 0):.6f})\n"

                interp = bp.get('interpretation', '')
                if interp:
                    content += f"- *{interp}*\n"
                content += "\n"

        return content + "\n"

    def _format_technical_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format technical metrics section."""
        content = ""
        
        # System Performance - handle both old and new nested structure
        if 'system_performance' in metrics:
            # New nested structure
            sp = metrics['system_performance']
            memory_usage = sp.get('memory_usage_mb', 0) or 0
            exec_time = sp.get('execution_time_seconds', 0) or 0
            cpu_usage = sp.get('cpu_usage_percent', 0) or 0
            disk_io = sp.get('disk_io_mb', 0) or 0
            throughput = sp.get('throughput_rows_per_second', 0) or 0
        else:
            # Old flat structure
            memory_usage = metrics.get('memory_usage_mb', 0) or 0
            exec_time = metrics.get('execution_time_seconds', 0) or 0
            cpu_usage = metrics.get('cpu_usage_percent', 0) or 0
            disk_io = metrics.get('disk_io_mb', 0) or 0
            throughput = 0

        content += f"- **Memory Usage**: {memory_usage:.2f} MB\n"
        content += f"- **Execution Time**: {exec_time:.2f} seconds\n"
        content += f"- **CPU Usage**: {cpu_usage:.2f}%\n"
        if throughput > 0:
            content += f"- **Throughput**: {throughput:.2f} rows/sec\n"
        if disk_io > 0:
            content += f"- **Disk I/O**: {disk_io:.2f} MB\n"
        
        # Data Processing
        data_size = metrics.get('data_size_mb', 0) or 0
        if data_size > 0:
            content += f"- **Data Size**: {data_size:.2f} MB\n"
        rows_processed = metrics.get('rows_processed', 0) or 0
        if rows_processed > 0:
            content += f"- **Rows Processed**: {rows_processed:,}\n"
        columns_processed = metrics.get('columns_processed', 0) or 0
        if columns_processed > 0:
            content += f"- **Columns Processed**: {columns_processed:,}\n"
        throughput = metrics.get('throughput_rows_per_second', 0) or 0
        if throughput > 0:
            content += f"- **Throughput**: {throughput:.2f} rows/sec\n"
        compression = metrics.get('compression_ratio', 0) or 0
        if compression > 0:
            content += f"- **Compression Ratio**: {compression:.2f}\n"
        
        # Algorithm Performance
        if 'iterations_completed' in metrics:
            content += f"- **Iterations Completed**: {metrics['iterations_completed']:,}\n"
        if 'convergence_time_seconds' in metrics:
            content += f"- **Convergence Time**: {metrics['convergence_time_seconds']:.2f} seconds\n"
        if 'objective_function_evaluations' in metrics:
            content += f"- **Objective Function Evaluations**: {metrics['objective_function_evaluations']:,}\n"
        if 'gradient_computations' in metrics:
            content += f"- **Gradient Computations**: {metrics['gradient_computations']:,}\n"
        if 'hessian_computations' in metrics:
            content += f"- **Hessian Computations**: {metrics['hessian_computations']:,}\n"
        
        # Selection/Generation Performance
        if 'algorithms_tested' in metrics:
            content += f"- **Algorithms Tested**: {metrics['algorithms_tested']}\n"
        if 'features_evaluated' in metrics:
            content += f"- **Features Evaluated**: {metrics['features_evaluated']:,}\n"
        if 'combinations_tested' in metrics:
            content += f"- **Combinations Tested**: {metrics['combinations_tested']:,}\n"
        if 'cross_validation_folds' in metrics:
            content += f"- **Cross-Validation Folds**: {metrics['cross_validation_folds']}\n"
        if 'hyperparameter_combinations' in metrics:
            content += f"- **Hyperparameter Combinations**: {metrics['hyperparameter_combinations']:,}\n"
        
        # Labeling-Specific Technical Metrics (nested structure support)
        if 'labeling_engine' in metrics:
            content += f"\n### Labeling Engine\n"
            le = metrics['labeling_engine']
            content += f"- **Method**: {le.get('method', 'N/A')}\n"
            content += f"- **Algorithm Type**: {le.get('algorithm_type', 'N/A')}\n"
            content += f"- **Optimization Level**: {le.get('optimization_level', 'N/A')}\n"
            content += f"- **VectorBT Integration**: {le.get('vectorbt_integration', 'N/A')}\n"
            content += f"- **Memory Efficient Processing**: {le.get('memory_efficient_processing', 'N/A')}\n"
        
        if 'signal_processing' in metrics:
            content += f"\n### Signal Processing\n"
            sp = metrics['signal_processing']
            content += f"- **Local Maxima Detection**: {sp.get('local_maxima_detection', 'N/A')}\n"
            content += f"- **Local Minima Detection**: {sp.get('local_minima_detection', 'N/A')}\n"
            content += f"- **Volatility Adaptation**: {sp.get('volatility_adaptation', 'N/A')}\n"
            content += f"- **Quality Scoring Enabled**: {sp.get('quality_scoring_enabled', 'N/A')}\n"
            content += f"- **Confidence Calculation**: {sp.get('confidence_calculation', 'N/A')}\n"
            content += f"- **Threshold Dynamic Range**: {sp.get('threshold_dynamic_range', 'N/A')}\n"
        
        if 'performance_optimization' in metrics:
            content += f"\n### Performance Optimization\n"
            po = metrics['performance_optimization']
            content += f"- **Rolling Window Optimization**: {po.get('rolling_window_optimization', 'N/A')}\n"
            content += f"- **Batch Processing Size**: {po.get('batch_processing_size', 'N/A')}\n"
            content += f"- **Memory Management**: {po.get('memory_management', 'N/A')}\n"
            content += f"- **Cache Utilization**: {po.get('cache_utilization', 0):.1%}\n"
        
        return content + "\n"

    def _format_process_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format process metrics section."""
        content = ""
        
        # Execution Analysis
        step_duration = metrics.get('step_duration_seconds', 0) or 0
        if step_duration > 0:
            content += f"- **Step Duration**: {step_duration:.2f} seconds\n"
        artifacts_gen = metrics.get('artifacts_generated', 0) or 0
        if artifacts_gen > 0:
            content += f"- **Artifacts Generated**: {artifacts_gen}\n"
        deps_loaded = metrics.get('dependencies_loaded', 0) or 0
        if deps_loaded > 0:
            content += f"- **Dependencies Loaded**: {deps_loaded}\n"
        opt_phases = metrics.get('optimization_phases', 0) or 0
        if opt_phases > 0:
            content += f"- **Optimization Phases**: {opt_phases}\n"
        val_checks = metrics.get('validation_checks', 0) or 0
        if val_checks > 0:
            content += f"- **Validation Checks**: {val_checks}\n"
        
        # Quality Metrics
        data_quality = metrics.get('data_quality_score', 0) or 0
        if data_quality > 0:
            content += f"- **Data Quality Score**: {data_quality:.4f}\n"
        validation_passed = metrics.get('validation_passed', False)
        content += f"- **Validation Passed**: {validation_passed}\n"
        warnings_count = metrics.get('warnings_count', 0) or 0
        if warnings_count > 0:
            content += f"- **Warnings Count**: {warnings_count}\n"
        errors_count = metrics.get('errors_count', 0) or 0
        if errors_count > 0:
            content += f"- **Errors Count**: {errors_count}\n"
        retry_count = metrics.get('retry_count', 0) or 0
        if retry_count > 0:
            content += f"- **Retry Count**: {retry_count}\n"
        
        # Feature Analysis
        if 'features_analyzed' in metrics:
            content += f"- **Features Analyzed**: {metrics['features_analyzed']:,}\n"
        if 'features_selected' in metrics:
            content += f"- **Features Selected**: {metrics['features_selected']:,}\n"
        if 'selection_ratio' in metrics:
            content += f"- **Selection Ratio**: {metrics['selection_ratio']:.4f}\n"
        if 'feature_families_represented' in metrics:
            content += f"- **Feature Families Represented**: {metrics['feature_families_represented']}\n"
        if 'selection_algorithm_performance' in metrics:
            content += f"- **Selection Algorithm Performance**: {metrics['selection_algorithm_performance']}\n"
        
        # Validation Analysis
        if 'validation_tests_performed' in metrics:
            content += f"- **Validation Tests Performed**: {metrics['validation_tests_performed']}\n"
        if 'validation_tests_passed' in metrics:
            content += f"- **Validation Tests Passed**: {metrics['validation_tests_passed']}\n"
        if 'validation_tests_failed' in metrics:
            content += f"- **Validation Tests Failed**: {metrics['validation_tests_failed']}\n"
        if 'validation_coverage' in metrics:
            content += f"- **Validation Coverage**: {metrics['validation_coverage']:.4f}\n"
        if 'validation_confidence' in metrics:
            content += f"- **Validation Confidence**: {metrics['validation_confidence']:.4f}\n"
        
        # Labeling-Specific Process Metrics (nested structure support)
        if isinstance(metrics, dict):
            # Handle direct key-value pairs that might be process-related
            key_value_pairs = [
                ('data_loading', 'Data Loading'),
                ('data_quality_checks', 'Data Quality Checks'),
                ('labeling_process', 'Labeling Process'),
                ('labeling_method', 'Labeling Method'),
                ('volatility_threshold', 'Volatility Threshold'),
                ('lookahead_periods', 'Lookahead Periods'),
                ('local_maxima_detection', 'Local Maxima Detection'),
                ('optimization_applied', 'Optimization Applied'),
                ('memory_management', 'Memory Management'),
                ('error_handling', 'Error Handling'),
                ('logging_completeness', 'Logging Completeness'),
                ('artifact_management', 'Artifact Management')
            ]
            
            for key, label in key_value_pairs:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (str, int, float, bool)):
                        content += f"- **{label}**: {value}\n"
            
            # Handle volatility calibration section
            if 'volatility_calibration' in metrics:
                content += "\n### Volatility Calibration\n"
                vc = metrics['volatility_calibration']
                content += f"- **Base Threshold**: {vc.get('base_threshold_percent', 0):.2f}%\n"
                content += f"- **Effective Threshold Range**: {vc.get('effective_threshold_min', 0):.2f}% - {vc.get('effective_threshold_max', 0):.2f}%\n"
                content += f"- **Adaptation Multiplier Range**: {vc.get('adaptation_multiplier_range', 'N/A')}\n"
                content += f"- **Adaptation Active**: {'✅ Yes' if vc.get('adaptation_active', False) else '❌ No'}\n"
                content += f"- **Adaptation Spread**: {vc.get('adaptation_spread', 0):.1f}%\n"
                content += f"- **Sensitivity Parameter**: {vc.get('sensitivity_parameter', 1.0)}\n"
                content += f"- **Window Size**: {vc.get('window_size', 20)}\n"
            
            # Handle expanded analysis section
            if 'expanded_analysis' in metrics:
                content += "\n### Expanded Analysis\n"
                ea = metrics['expanded_analysis']
                
                # Signal Distribution
                if 'signal_distribution' in ea:
                    sd = ea['signal_distribution']
                    content += f"\n#### Signal Distribution\n"
                    content += f"- **Long Rate**: {sd.get('long_rate', 0):.2f}%\n"
                    content += f"- **Short Rate**: {sd.get('short_rate', 0):.2f}%\n"
                    content += f"- **Signal Balance**: {sd.get('signal_balance', 'N/A')}\n"
                
                # Performance Metrics
                if 'performance_metrics' in ea:
                    pm = ea['performance_metrics']
                    content += f"\n#### Performance Metrics\n"
                    content += f"- **Opportunities Per Week**: {pm.get('opportunities_per_week', 0):.1f}\n"
                    content += f"- **Detection Efficiency**: {pm.get('detection_efficiency', 0):.2f}%\n"
                    content += f"- **Quality Signal Ratio**: {pm.get('quality_signal_ratio', 0):.3f}\n"
                
                # Market Adaptation
                if 'market_adaptation' in ea:
                    ma = ea['market_adaptation']
                    content += f"\n#### Market Adaptation\n"
                    content += f"- **Volatility Regime**: {ma.get('volatility_regime', 'N/A')}\n"
                    content += f"- **Threshold Adjustment Active**: {'✅ Yes' if ma.get('threshold_adjustment_active', False) else '❌ No'}\n"
                    content += f"- **Adaptation Range**: {ma.get('adaptation_range_percent', 0):.1f}%\n"
        
        return content + "\n"

    def _format_general_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format general metrics section."""
        content = ""
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    content += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
                else:
                    content += f"- **{key.replace('_', ' ').title()}**: {value:,}\n"
            else:
                content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        return content + "\n"

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            return {
                'memory_usage_mb': memory_usage_mb,
                'memory_usage_percent': memory.percent,
                'cpu_usage_percent': cpu_usage,
                'available_memory_mb': memory.available / (1024 * 1024),
                'total_memory_mb': memory.total / (1024 * 1024)
            }
        except Exception as e:
            return {
                'memory_usage_mb': 0.0,
                'memory_usage_percent': 0.0,
                'cpu_usage_percent': 0.0,
                'available_memory_mb': 0.0,
                'total_memory_mb': 0.0
            }
