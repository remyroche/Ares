#!/usr/bin/env python3
"""Comprehensive Reporting System for Backtesting Pipeline.

This module provides detailed reporting capabilities for troubleshooting and analysis,
including quality assessment, performance metrics, and actionable recommendations.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)

class BacktestingReportGenerator:
    """Comprehensive report generator for backtesting pipeline."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, data_dir: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.data_dir = Path(data_dir)
        self.report_data = {}
        self.quality_metrics = {}
        self.performance_metrics = {}
        self.recommendations = []
        
        # Ensure data directory exists
        ensure_directory(self.data_dir)
    
    def generate_comprehensive_report(
        self, 
        pipeline_results: Dict[str, Any],
        logger_data: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive backtesting report."""
        
        report = {
            'execution_summary': self._generate_execution_summary(pipeline_results),
            'quality_assessment': self._generate_quality_assessment(pipeline_results, logger_data),
            'performance_analysis': self._generate_performance_analysis(logger_data),
            'data_quality_report': self._generate_data_quality_report(),
            'validation_results': self._generate_validation_results(pipeline_results),
            'error_analysis': self._generate_error_analysis(logger_data),
            'recommendations': self._generate_recommendations(pipeline_results, logger_data),
            'troubleshooting_guide': self._generate_troubleshooting_guide(logger_data),
            'metadata': self._generate_metadata()
        }
        
        if output_file:
            safe_json_dump(report, output_file, indent=2)
        
        return report
    
    def _generate_execution_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution summary."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'execution_date': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
            'pipeline_version': 'enhanced_v2.0_with_logging',
            'total_steps_completed': len([k for k, v in pipeline_results.items() if v is not None]),
            'success_rate': self._calculate_success_rate(pipeline_results),
            'overall_status': 'SUCCESS' if pipeline_results.get('success', False) else 'FAILED'
        }
    
    def _generate_quality_assessment(self, pipeline_results: Dict[str, Any], logger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality assessment report."""
        quality_flags = logger_data.get('quality_flags', [])
        errors = logger_data.get('errors', [])
        warnings = logger_data.get('warnings', [])
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(quality_flags, errors, warnings)
        
        # Categorize quality flags
        quality_categories = self._categorize_quality_flags(quality_flags)
        
        return {
            'overall_quality_score': quality_score,
            'quality_level': self._determine_quality_level(quality_score),
            'quality_flags_count': len(quality_flags),
            'error_count': len(errors),
            'warning_count': len(warnings),
            'quality_categories': quality_categories,
            'critical_issues': [f for f in quality_flags if f.get('severity') == 'ERROR'],
            'warnings': [f for f in quality_flags if f.get('severity') == 'WARNING'],
            'data_quality_issues': [f for f in quality_flags if f.get('type') == 'DATA_QUALITY'],
            'validation_issues': [f for f in quality_flags if f.get('type') == 'VALIDATION'],
            'performance_issues': [f for f in quality_flags if f.get('type') == 'PERFORMANCE']
        }
    
    def _generate_performance_analysis(self, logger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis report."""
        step_times = logger_data.get('step_times', {})
        performance_metrics = logger_data.get('performance_metrics', {})
        
        # Calculate performance statistics
        total_time = sum(step_times.values()) if step_times else 0
        avg_step_time = np.mean(list(step_times.values())) if step_times else 0
        max_step_time = max(step_times.values()) if step_times else 0
        min_step_time = min(step_times.values()) if step_times else 0
        
        # Memory and CPU analysis
        memory_usage = []
        cpu_usage = []
        
        for metrics in performance_metrics.values():
            if isinstance(metrics, dict):
                memory_usage.append(metrics.get('memory_mb', 0))
                cpu_usage.append(metrics.get('cpu_percent', 0))
        
        return {
            'execution_time_analysis': {
                'total_execution_time': total_time,
                'average_step_time': avg_step_time,
                'max_step_time': max_step_time,
                'min_step_time': min_step_time,
                'slowest_step': max(step_times.items(), key=lambda x: x[1])[0] if step_times else None,
                'fastest_step': min(step_times.items(), key=lambda x: x[1])[0] if step_times else None
            },
            'resource_usage_analysis': {
                'peak_memory_mb': max(memory_usage) if memory_usage else 0,
                'average_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                'peak_cpu_percent': max(cpu_usage) if cpu_usage else 0,
                'average_cpu_percent': np.mean(cpu_usage) if cpu_usage else 0
            },
            'performance_bottlenecks': self._identify_performance_bottlenecks(step_times, performance_metrics),
            'efficiency_metrics': self._calculate_efficiency_metrics(step_times, performance_metrics)
        }
    
    def _generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report."""
        data_files = [
            f"aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet",
            f"volume_{self.exchange}_{self.symbol}_consolidated.parquet"
        ]
        
        quality_report = {
            'data_files_status': {},
            'data_quality_metrics': {},
            'data_issues': []
        }
        
        for file_name in data_files:
            file_path = self.data_dir / file_name
            if safe_file_exists(file_path):
                try:
                    # Load and analyze data
                    df = pd.read_parquet(file_path)
                    
                    quality_metrics = {
                        'total_records': len(df),
                        'missing_values': df.isnull().sum().sum(),
                        'duplicate_records': df.duplicated().sum(),
                        'data_types': df.dtypes.to_dict(),
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                    
                    quality_report['data_files_status'][file_name] = 'AVAILABLE'
                    quality_report['data_quality_metrics'][file_name] = quality_metrics
                    
                    # Check for data quality issues
                    if quality_metrics['missing_values'] > 0:
                        quality_report['data_issues'].append({
                            'file': file_name,
                            'issue': 'Missing values detected',
                            'count': quality_metrics['missing_values']
                        })
                    
                    if quality_metrics['duplicate_records'] > 0:
                        quality_report['data_issues'].append({
                            'file': file_name,
                            'issue': 'Duplicate records detected',
                            'count': quality_metrics['duplicate_records']
                        })
                        
                except Exception as e:
                    quality_report['data_files_status'][file_name] = 'ERROR'
                    quality_report['data_issues'].append({
                        'file': file_name,
                        'issue': f'Error reading file: {e}',
                        'count': 0
                    })
            else:
                quality_report['data_files_status'][file_name] = 'MISSING'
                quality_report['data_issues'].append({
                    'file': file_name,
                    'issue': 'File not found',
                    'count': 0
                })
        
        return quality_report
    
    def _generate_validation_results(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation results report."""
        validation_results = {
            'walk_forward_validation': self._analyze_validation_step(
                pipeline_results.get('walk_forward_results'), 'Walk Forward'
            ),
            'monte_carlo_validation': self._analyze_validation_step(
                pipeline_results.get('monte_carlo_results'), 'Monte Carlo'
            ),
            'ab_testing': self._analyze_validation_step(
                pipeline_results.get('ab_testing_results'), 'A/B Testing'
            ),
            'model_saving': self._analyze_validation_step(
                pipeline_results.get('model_saving_results'), 'Model Saving'
            )
        }
        
        return validation_results
    
    def _generate_error_analysis(self, logger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate error analysis report."""
        errors = logger_data.get('errors', [])
        warnings = logger_data.get('warnings', [])
        
        # Categorize errors
        error_categories = {}
        for error in errors:
            error_type = error.get('type', 'Unknown')
            if error_type not in error_categories:
                error_categories[error_type] = []
            error_categories[error_type].append(error)
        
        # Categorize warnings
        warning_categories = {}
        for warning in warnings:
            context = warning.get('context', 'Unknown')
            if context not in warning_categories:
                warning_categories[context] = []
            warning_categories[context].append(warning)
        
        return {
            'error_summary': {
                'total_errors': len(errors),
                'error_categories': {k: len(v) for k, v in error_categories.items()},
                'most_common_error': max(error_categories.items(), key=lambda x: len(x[1]))[0] if error_categories else None
            },
            'warning_summary': {
                'total_warnings': len(warnings),
                'warning_categories': {k: len(v) for k, v in warning_categories.items()},
                'most_common_warning_context': max(warning_categories.items(), key=lambda x: len(x[1]))[0] if warning_categories else None
            },
            'error_details': error_categories,
            'warning_details': warning_categories,
            'error_timeline': self._generate_error_timeline(errors)
        }
    
    def _generate_recommendations(self, pipeline_results: Dict[str, Any], logger_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Quality-based recommendations
        quality_flags = logger_data.get('quality_flags', [])
        if len(quality_flags) > 5:
            recommendations.append({
                'category': 'Quality',
                'priority': 'HIGH',
                'title': 'Address Quality Flags',
                'description': f'Found {len(quality_flags)} quality flags. Review and address quality issues.',
                'action': 'Review quality flags in the report and implement fixes'
            })
        
        # Performance-based recommendations
        step_times = logger_data.get('step_times', {})
        if step_times:
            slowest_step = max(step_times.items(), key=lambda x: x[1])
            if slowest_step[1] > 300:  # More than 5 minutes
                recommendations.append({
                    'category': 'Performance',
                    'priority': 'MEDIUM',
                    'title': 'Optimize Slow Step',
                    'description': f'Step "{slowest_step[0]}" took {slowest_step[1]:.2f} seconds.',
                    'action': f'Consider optimizing the {slowest_step[0]} step for better performance'
                })
        
        # Data quality recommendations
        data_issues = self._generate_data_quality_report().get('data_issues', [])
        if data_issues:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'HIGH',
                'title': 'Fix Data Quality Issues',
                'description': f'Found {len(data_issues)} data quality issues.',
                'action': 'Review and fix data quality issues before running backtesting'
            })
        
        # Memory usage recommendations
        performance_metrics = logger_data.get('performance_metrics', {})
        if performance_metrics:
            max_memory = max(m.get('memory_mb', 0) for m in performance_metrics.values())
            if max_memory > 2000:  # More than 2GB
                recommendations.append({
                    'category': 'Performance',
                    'priority': 'MEDIUM',
                    'title': 'Optimize Memory Usage',
                    'description': f'Peak memory usage was {max_memory:.1f} MB.',
                    'action': 'Consider optimizing memory usage or increasing available memory'
                })
        
        return recommendations
    
    def _generate_troubleshooting_guide(self, logger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate troubleshooting guide."""
        return {
            'common_issues': [
                {
                    'issue': 'Missing data files',
                    'symptoms': ['File not found errors', 'Data directory validation failures'],
                    'solutions': [
                        'Run data collection: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE',
                        'Check data directory permissions',
                        'Verify file paths in configuration'
                    ]
                },
                {
                    'issue': 'High memory usage',
                    'symptoms': ['Memory errors', 'Slow performance', 'System crashes'],
                    'solutions': [
                        'Reduce data size or use smaller timeframes',
                        'Increase system memory',
                        'Optimize data processing algorithms'
                    ]
                },
                {
                    'issue': 'Validation failures',
                    'symptoms': ['Validation errors', 'Quality flags', 'Failed steps'],
                    'solutions': [
                        'Review validation criteria',
                        'Check data quality',
                        'Adjust validation thresholds'
                    ]
                }
            ],
            'debugging_steps': [
                'Check log files for detailed error messages',
                'Review quality flags and warnings',
                'Validate input data quality',
                'Check system resources (memory, CPU)',
                'Review configuration parameters'
            ],
            'support_resources': [
                'Log files in log/backtesting/ directory',
                'Quality assessment reports',
                'Performance monitoring data',
                'Configuration files'
            ]
        }
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'report_version': '2.0',
            'generated_at': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
            'generator': 'BacktestingReportGenerator',
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'data_directory': str(self.data_dir)
        }
    
    # Helper methods
    def _calculate_success_rate(self, pipeline_results: Dict[str, Any]) -> float:
        """Calculate success rate of pipeline steps."""
        total_steps = len([k for k in pipeline_results.keys() if k.endswith('_results')])
        successful_steps = len([k for k, v in pipeline_results.items() 
                               if k.endswith('_results') and v is not None])
        return (successful_steps / total_steps * 100) if total_steps > 0 else 0
    
    def _calculate_quality_score(self, quality_flags: List, errors: List, warnings: List) -> float:
        """Calculate overall quality score (0-100)."""
        base_score = 100
        
        # Deduct points for errors
        base_score -= len(errors) * 10
        
        # Deduct points for quality flags
        for flag in quality_flags:
            if flag.get('severity') == 'ERROR':
                base_score -= 15
            elif flag.get('severity') == 'WARNING':
                base_score -= 5
        
        # Deduct points for warnings
        base_score -= len(warnings) * 2
        
        return max(0, min(100, base_score))
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level based on score."""
        if quality_score >= 90:
            return 'EXCELLENT'
        elif quality_score >= 75:
            return 'GOOD'
        elif quality_score >= 60:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _categorize_quality_flags(self, quality_flags: List) -> Dict[str, List]:
        """Categorize quality flags by type."""
        categories = {}
        for flag in quality_flags:
            flag_type = flag.get('type', 'UNKNOWN')
            if flag_type not in categories:
                categories[flag_type] = []
            categories[flag_type].append(flag)
        return categories
    
    def _identify_performance_bottlenecks(self, step_times: Dict, performance_metrics: Dict) -> List[Dict]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Find slow steps
        if step_times:
            avg_time = np.mean(list(step_times.values()))
            for step, time_taken in step_times.items():
                if time_taken > avg_time * 2:  # More than 2x average
                    bottlenecks.append({
                        'type': 'SLOW_STEP',
                        'step': step,
                        'time_taken': time_taken,
                        'severity': 'HIGH' if time_taken > avg_time * 3 else 'MEDIUM'
                    })
        
        # Find memory bottlenecks
        if performance_metrics:
            memory_usage = [m.get('memory_mb', 0) for m in performance_metrics.values()]
            if memory_usage:
                max_memory = max(memory_usage)
                if max_memory > 1000:  # More than 1GB
                    bottlenecks.append({
                        'type': 'HIGH_MEMORY_USAGE',
                        'peak_memory_mb': max_memory,
                        'severity': 'HIGH' if max_memory > 2000 else 'MEDIUM'
                    })
        
        return bottlenecks
    
    def _calculate_efficiency_metrics(self, step_times: Dict, performance_metrics: Dict) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        if not step_times:
            return {}
        
        total_time = sum(step_times.values())
        step_count = len(step_times)
        
        return {
            'steps_per_minute': (step_count / total_time) * 60 if total_time > 0 else 0,
            'average_step_efficiency': total_time / step_count if step_count > 0 else 0,
            'time_distribution': {step: (time_taken / total_time) * 100 
                                for step, time_taken in step_times.items()}
        }
    
    def _analyze_validation_step(self, results: Any, step_name: str) -> Dict[str, Any]:
        """Analyze validation step results."""
        if results is None:
            return {
                'status': 'NOT_EXECUTED',
                'message': f'{step_name} validation was not executed'
            }
        
        if isinstance(results, dict):
            return {
                'status': 'COMPLETED',
                'message': f'{step_name} validation completed successfully',
                'details': results
            }
        else:
            return {
                'status': 'COMPLETED',
                'message': f'{step_name} validation completed',
                'details': str(results)
            }
    
    def _generate_error_timeline(self, errors: List) -> List[Dict]:
        """Generate error timeline."""
        timeline = []
        for error in errors:
            timeline.append({
                'timestamp': error.get('timestamp', 0),
                'error_type': error.get('type', 'Unknown'),
                'message': error.get('message', ''),
                'context': error.get('context', '')
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline

def generate_backtesting_report(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    pipeline_results: Dict[str, Any],
    logger_data: Dict[str, Any],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Generate comprehensive backtesting report."""
    
    generator = BacktestingReportGenerator(symbol, exchange, timeframe, data_dir)
    return generator.generate_comprehensive_report(pipeline_results, logger_data, output_file)