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
        
        # Performance Indicators
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
        
        # Feature Performance
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
        
        # Selection Quality
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
        
        # Optimization Results
        if 'best_periods' in metrics:
            content += f"- **Best Periods**: {metrics['best_periods']}\n"
        if 'best_lookbacks' in metrics:
            content += f"- **Best Lookbacks**: {metrics['best_lookbacks']}\n"
        if 'optimization_convergence' in metrics:
            content += f"- **Optimization Convergence**: {metrics['optimization_convergence']}\n"
        if 'hyperparameter_sensitivity' in metrics:
            content += f"- **Hyperparameter Sensitivity**: {metrics['hyperparameter_sensitivity']}\n"
        
        return content + "\n"

    def _format_technical_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format technical metrics section."""
        content = ""
        
        # System Performance
        if 'memory_usage_mb' in metrics:
            content += f"- **Memory Usage**: {metrics['memory_usage_mb']:.2f} MB\n"
        if 'execution_time_seconds' in metrics:
            content += f"- **Execution Time**: {metrics['execution_time_seconds']:.2f} seconds\n"
        if 'cpu_usage_percent' in metrics:
            content += f"- **CPU Usage**: {metrics['cpu_usage_percent']:.2f}%\n"
        if 'gpu_usage_percent' in metrics:
            content += f"- **GPU Usage**: {metrics['gpu_usage_percent']:.2f}%\n"
        if 'disk_io_mb' in metrics:
            content += f"- **Disk I/O**: {metrics['disk_io_mb']:.2f} MB\n"
        
        # Data Processing
        if 'data_size_mb' in metrics:
            content += f"- **Data Size**: {metrics['data_size_mb']:.2f} MB\n"
        if 'rows_processed' in metrics:
            content += f"- **Rows Processed**: {metrics['rows_processed']:,}\n"
        if 'columns_processed' in metrics:
            content += f"- **Columns Processed**: {metrics['columns_processed']:,}\n"
        if 'throughput_rows_per_second' in metrics:
            content += f"- **Throughput**: {metrics['throughput_rows_per_second']:.2f} rows/sec\n"
        if 'compression_ratio' in metrics:
            content += f"- **Compression Ratio**: {metrics['compression_ratio']:.2f}\n"
        
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
        
        return content + "\n"

    def _format_process_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format process metrics section."""
        content = ""
        
        # Execution Analysis
        if 'step_duration_seconds' in metrics:
            content += f"- **Step Duration**: {metrics['step_duration_seconds']:.2f} seconds\n"
        if 'artifacts_generated' in metrics:
            content += f"- **Artifacts Generated**: {metrics['artifacts_generated']}\n"
        if 'dependencies_loaded' in metrics:
            content += f"- **Dependencies Loaded**: {metrics['dependencies_loaded']}\n"
        if 'optimization_phases' in metrics:
            content += f"- **Optimization Phases**: {metrics['optimization_phases']}\n"
        if 'validation_checks' in metrics:
            content += f"- **Validation Checks**: {metrics['validation_checks']}\n"
        
        # Quality Metrics
        if 'data_quality_score' in metrics:
            content += f"- **Data Quality Score**: {metrics['data_quality_score']:.4f}\n"
        if 'validation_passed' in metrics:
            content += f"- **Validation Passed**: {metrics['validation_passed']}\n"
        if 'warnings_count' in metrics:
            content += f"- **Warnings Count**: {metrics['warnings_count']}\n"
        if 'errors_count' in metrics:
            content += f"- **Errors Count**: {metrics['errors_count']}\n"
        if 'retry_count' in metrics:
            content += f"- **Retry Count**: {metrics['retry_count']}\n"
        
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
