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
            'backtesting_results': self._generate_backtesting_results(pipeline_results),
            'regime_analysis': self._generate_regime_analysis(pipeline_results),
            'model_performance': self._generate_model_performance_analysis(pipeline_results),
            'risk_analysis': self._generate_risk_analysis(pipeline_results),
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

    def generate_step_report(
        self,
        step_name: str,
        step_results: Dict[str, Any],
        symbol: str,
        timeframe: str,
        data_dir: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate detailed report for a specific backtesting step."""
        timestamp = get_current_datetime()
        
        report = {
            'step_info': {
                'step_name': step_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'data_directory': data_dir
            },
            'step_results': step_results,
            'step_metrics': self._extract_step_metrics(step_results),
            'step_analysis': self._analyze_step_results(step_name, step_results),
            'quality_assessment': self._assess_step_quality(step_name, step_results),
            'recommendations': self._generate_step_recommendations(step_name, step_results),
            'troubleshooting': self._generate_step_troubleshooting(step_name, step_results)
        }
        
        # Save to file if specified
        if output_file:
            safe_json_dump(report, output_file, indent=2)
            logging.info(f"Step report saved to: {output_file}")
        
        return report

    def _extract_step_metrics(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from step results."""
        metrics = {}
        
        # Performance metrics
        if 'performance_metrics' in step_results:
            metrics['performance'] = step_results['performance_metrics']
        
        # Regime-specific metrics
        if 'regime_performance' in step_results:
            metrics['regime_performance'] = step_results['regime_performance']
        
        # Model performance
        if 'model_performance' in step_results:
            metrics['model_performance'] = step_results['model_performance']
        
        # Risk metrics
        if 'risk_metrics' in step_results:
            metrics['risk_metrics'] = step_results['risk_metrics']
        
        # Execution metrics
        if 'execution_time' in step_results:
            metrics['execution_time'] = step_results['execution_time']
        
        if 'memory_usage' in step_results:
            metrics['memory_usage'] = step_results['memory_usage']
        
        return metrics

    def _analyze_step_results(self, step_name: str, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze step results for insights."""
        analysis = {
            'step_type': step_name,
            'success': step_results.get('success', False),
            'data_quality': self._assess_data_quality(step_results),
            'performance_insights': self._extract_performance_insights(step_results),
            'regime_insights': self._extract_regime_insights(step_results),
            'model_insights': self._extract_model_insights(step_results)
        }
        
        return analysis

    def _assess_step_quality(self, step_name: str, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of step results."""
        quality_flags = []
        quality_score = 100
        
        # Check for errors
        if 'errors' in step_results and step_results['errors']:
            quality_flags.append({
                'type': 'ERROR',
                'message': f"Step {step_name} has errors",
                'details': step_results['errors']
            })
            quality_score -= 30
        
        # Check performance metrics
        if 'performance_metrics' in step_results:
            perf = step_results['performance_metrics']
            if 'sharpe_ratio' in perf and perf['sharpe_ratio'] < 1.0:
                quality_flags.append({
                    'type': 'WARNING',
                    'message': f"Low Sharpe ratio in {step_name}: {perf['sharpe_ratio']:.2f}",
                    'details': {'sharpe_ratio': perf['sharpe_ratio']}
                })
                quality_score -= 10
        
        # Check regime performance
        if 'regime_performance' in step_results:
            for regime, regime_data in step_results['regime_performance'].items():
                if 'regime_return' in regime_data and regime_data['regime_return'] < 0:
                    quality_flags.append({
                        'type': 'WARNING',
                        'message': f"Negative returns in {regime} regime",
                        'details': {'regime': regime, 'return': regime_data['regime_return']}
                    })
                    quality_score -= 5
        
        return {
            'quality_score': max(0, quality_score),
            'quality_flags': quality_flags,
            'overall_assessment': 'EXCELLENT' if quality_score >= 90 else 'GOOD' if quality_score >= 70 else 'FAIR' if quality_score >= 50 else 'POOR'
        }

    def _generate_step_recommendations(self, step_name: str, step_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations for step improvement."""
        recommendations = []
        
        # Performance-based recommendations
        if 'performance_metrics' in step_results:
            perf = step_results['performance_metrics']
            if 'sharpe_ratio' in perf and perf['sharpe_ratio'] < 1.0:
                recommendations.append({
                    'category': 'PERFORMANCE',
                    'priority': 'HIGH',
                    'recommendation': f"Improve Sharpe ratio in {step_name} - consider parameter optimization or feature engineering",
                    'action': "Review model parameters and feature selection"
                })
        
        # Regime-based recommendations
        if 'regime_performance' in step_results:
            for regime, regime_data in step_results['regime_performance'].items():
                if 'regime_return' in regime_data and regime_data['regime_return'] < 0:
                    recommendations.append({
                        'category': 'REGIME_PERFORMANCE',
                        'priority': 'MEDIUM',
                        'recommendation': f"Address negative returns in {regime} regime",
                        'action': f"Review strategy parameters for {regime} market conditions"
                    })
        
        return recommendations

    def _generate_step_troubleshooting(self, step_name: str, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate troubleshooting guide for step issues."""
        troubleshooting = {
            'common_issues': [],
            'debugging_steps': [],
            'log_locations': []
        }
        
        # Add step-specific troubleshooting
        if step_name == 'walk_forward_validation':
            troubleshooting['common_issues'].extend([
                "Data gaps in time series",
                "Insufficient training data",
                "Overfitting in validation"
            ])
            troubleshooting['debugging_steps'].extend([
                "Check data continuity",
                "Verify minimum training window",
                "Review validation metrics"
            ])
        
        elif step_name == 'monte_carlo_validation':
            troubleshooting['common_issues'].extend([
                "High variance in results",
                "Poor convergence",
                "Memory issues with large simulations"
            ])
            troubleshooting['debugging_steps'].extend([
                "Increase simulation count gradually",
                "Check random seed consistency",
                "Monitor memory usage"
            ])
        
        elif step_name == 'ab_testing':
            troubleshooting['common_issues'].extend([
                "Statistical significance issues",
                "Insufficient sample size",
                "Biased test groups"
            ])
            troubleshooting['debugging_steps'].extend([
                "Verify sample size calculations",
                "Check group randomization",
                "Review statistical tests"
            ])
        
        return troubleshooting

    def _assess_data_quality(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality for the step."""
        quality_metrics = {
            'completeness': 100,
            'consistency': 100,
            'accuracy': 100,
            'issues': []
        }
        
        # Check for missing data
        if 'missing_data_count' in step_results:
            missing_count = step_results['missing_data_count']
            if missing_count > 0:
                quality_metrics['completeness'] -= min(50, missing_count * 2)
                quality_metrics['issues'].append(f"Missing data points: {missing_count}")
        
        # Check for data inconsistencies
        if 'data_inconsistencies' in step_results:
            inconsistencies = step_results['data_inconsistencies']
            if inconsistencies:
                quality_metrics['consistency'] -= len(inconsistencies) * 10
                quality_metrics['issues'].extend(inconsistencies)
        
        return quality_metrics

    def _extract_performance_insights(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance insights from step results."""
        insights = {}
        
        if 'performance_metrics' in step_results:
            perf = step_results['performance_metrics']
            insights['total_return'] = perf.get('total_return', 0)
            insights['sharpe_ratio'] = perf.get('sharpe_ratio', 0)
            insights['max_drawdown'] = perf.get('max_drawdown', 0)
            insights['win_rate'] = perf.get('win_rate', 0)
            
            # Performance assessment
            if insights['sharpe_ratio'] > 2.0:
                insights['performance_grade'] = 'EXCELLENT'
            elif insights['sharpe_ratio'] > 1.5:
                insights['performance_grade'] = 'GOOD'
            elif insights['sharpe_ratio'] > 1.0:
                insights['performance_grade'] = 'FAIR'
            else:
                insights['performance_grade'] = 'POOR'
        
        return insights

    def _extract_regime_insights(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regime-specific insights."""
        insights = {}
        
        if 'regime_performance' in step_results:
            regime_data = step_results['regime_performance']
            insights['regime_count'] = len(regime_data)
            insights['best_regime'] = None
            insights['worst_regime'] = None
            insights['regime_stability'] = 'STABLE'
            
            best_return = float('-inf')
            worst_return = float('inf')
            
            for regime, data in regime_data.items():
                if 'regime_return' in data:
                    return_val = data['regime_return']
                    if return_val > best_return:
                        best_return = return_val
                        insights['best_regime'] = regime
                    if return_val < worst_return:
                        worst_return = return_val
                        insights['worst_regime'] = regime
            
            # Check for regime stability
            returns = [data.get('regime_return', 0) for data in regime_data.values()]
            if returns and max(returns) - min(returns) > 0.5:  # 50% difference
                insights['regime_stability'] = 'VOLATILE'
        
        return insights

    def _extract_model_insights(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model-specific insights."""
        insights = {}
        
        if 'model_performance' in step_results:
            model_data = step_results['model_performance']
            insights['model_count'] = len(model_data)
            insights['best_model'] = None
            insights['model_consistency'] = 'CONSISTENT'
            
            best_score = float('-inf')
            scores = []
            
            for model, data in model_data.items():
                if 'model_score' in data:
                    score = data['model_score']
                    scores.append(score)
                    if score > best_score:
                        best_score = score
                        insights['best_model'] = model
            
            # Check model consistency
            if scores and max(scores) - min(scores) > 0.3:  # 30% difference
                insights['model_consistency'] = 'INCONSISTENT'
        
        return insights


def generate_step_report(
    step_name: str,
    step_results: Dict[str, Any],
    symbol: str,
    timeframe: str,
    data_dir: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Generate detailed report for a specific backtesting step."""
    reporter = ComprehensiveReporter(symbol, timeframe, data_dir)
    return reporter.generate_step_report(step_name, step_results, symbol, timeframe, data_dir, output_file)


def generate_detailed_regime_metrics_report(
    pipeline_results: Dict[str, Any],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Generate detailed regime/cluster metrics report with datetime."""
    timestamp = get_current_datetime()
    
    report = {
        'report_info': {
            'type': 'regime_cluster_metrics',
            'timestamp': timestamp,
            'generated_by': 'comprehensive_reporting_system'
        },
        'regime_analysis': {},
        'cluster_analysis': {},
        'performance_by_regime': {},
        'risk_metrics_by_regime': {},
        'model_performance_by_regime': {},
        'recommendations': [],
        'quality_flags': []
    }
    
    # Extract regime data from pipeline results
    if 'walk_forward_results' in pipeline_results:
        wf_results = pipeline_results['walk_forward_results']
        if 'regime_performance' in wf_results:
            report['regime_analysis']['walk_forward'] = wf_results['regime_performance']
    
    if 'monte_carlo_results' in pipeline_results:
        mc_results = pipeline_results['monte_carlo_results']
        if 'regime_performance' in mc_results:
            report['regime_analysis']['monte_carlo'] = mc_results['regime_performance']
    
    if 'ab_testing_results' in pipeline_results:
        ab_results = pipeline_results['ab_testing_results']
        if 'regime_performance' in ab_results:
            report['regime_analysis']['ab_testing'] = ab_results['regime_performance']
    
    # Generate performance analysis by regime
    for validation_type, regime_data in report['regime_analysis'].items():
        report['performance_by_regime'][validation_type] = {}
        for regime, metrics in regime_data.items():
            report['performance_by_regime'][validation_type][regime] = {
                'return': metrics.get('regime_return', 0),
                'sharpe': metrics.get('regime_sharpe', 0),
                'volatility': metrics.get('regime_volatility', 0),
                'max_drawdown': metrics.get('regime_max_drawdown', 0),
                'win_rate': metrics.get('regime_win_rate', 0),
                'duration': metrics.get('regime_duration', 0),
                'frequency': metrics.get('regime_frequency', 0)
            }
    
    # Generate quality flags for regime performance
    for validation_type, regime_data in report['regime_analysis'].items():
        for regime, metrics in regime_data.items():
            if 'regime_return' in metrics and metrics['regime_return'] < 0:
                report['quality_flags'].append({
                    'type': 'WARNING',
                    'validation_type': validation_type,
                    'regime': regime,
                    'message': f"Negative returns in {regime} regime: {metrics['regime_return']:.2%}",
                    'severity': 'MEDIUM'
                })
            
            if 'regime_sharpe' in metrics and metrics['regime_sharpe'] < 1.0:
                report['quality_flags'].append({
                    'type': 'WARNING',
                    'validation_type': validation_type,
                    'regime': regime,
                    'message': f"Low Sharpe ratio in {regime} regime: {metrics['regime_sharpe']:.2f}",
                    'severity': 'LOW'
                })
    
    # Generate recommendations
    negative_regimes = []
    for validation_type, regime_data in report['regime_analysis'].items():
        for regime, metrics in regime_data.items():
            if 'regime_return' in metrics and metrics['regime_return'] < 0:
                negative_regimes.append((validation_type, regime))
    
    if negative_regimes:
        report['recommendations'].append({
            'category': 'REGIME_PERFORMANCE',
            'priority': 'HIGH',
            'recommendation': f"Address negative returns in {len(negative_regimes)} regime(s)",
            'action': "Review strategy parameters for underperforming market regimes",
            'affected_regimes': negative_regimes
        })
    
    # Save to file if specified
    if output_file:
        safe_json_dump(report, output_file, indent=2)
        logging.info(f"Detailed regime metrics report saved to: {output_file}")
    
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
    
    def _generate_backtesting_results(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive backtesting results analysis."""
        backtesting_results = {
            'overall_performance': {},
            'walk_forward_results': {},
            'monte_carlo_results': {},
            'ab_testing_results': {},
            'performance_summary': {}
        }
        
        # Extract overall performance metrics
        if 'walk_forward_results' in pipeline_results and pipeline_results['walk_forward_results']:
            wf_results = pipeline_results['walk_forward_results']
            if isinstance(wf_results, dict):
                backtesting_results['walk_forward_results'] = self._extract_performance_metrics(wf_results)
        
        if 'monte_carlo_results' in pipeline_results and pipeline_results['monte_carlo_results']:
            mc_results = pipeline_results['monte_carlo_results']
            if isinstance(mc_results, dict):
                backtesting_results['monte_carlo_results'] = self._extract_performance_metrics(mc_results)
        
        if 'ab_testing_results' in pipeline_results and pipeline_results['ab_testing_results']:
            ab_results = pipeline_results['ab_testing_results']
            if isinstance(ab_results, dict):
                backtesting_results['ab_testing_results'] = self._extract_performance_metrics(ab_results)
        
        # Calculate overall performance summary
        backtesting_results['performance_summary'] = self._calculate_overall_performance_summary(backtesting_results)
        
        return backtesting_results
    
    def _generate_regime_analysis(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed regime analysis."""
        regime_analysis = {
            'regime_identification': {},
            'regime_performance': {},
            'regime_transitions': {},
            'regime_stability': {},
            'regime_recommendations': []
        }
        
        # Extract regime information from results
        for result_type, results in pipeline_results.items():
            if results and isinstance(results, dict):
                if 'regimes' in results:
                    regime_analysis['regime_identification'] = results['regimes']
                if 'regime_performance' in results:
                    regime_analysis['regime_performance'] = results['regime_performance']
                if 'regime_transitions' in results:
                    regime_analysis['regime_transitions'] = results['regime_transitions']
        
        # Analyze regime stability and performance
        regime_analysis['regime_stability'] = self._analyze_regime_stability(regime_analysis)
        regime_analysis['regime_recommendations'] = self._generate_regime_recommendations(regime_analysis)
        
        return regime_analysis
    
    def _generate_model_performance_analysis(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model performance analysis."""
        model_analysis = {
            'model_accuracy': {},
            'model_confidence': {},
            'feature_importance': {},
            'model_comparison': {},
            'model_recommendations': []
        }
        
        # Extract model performance from results
        for result_type, results in pipeline_results.items():
            if results and isinstance(results, dict):
                if 'model_accuracy' in results:
                    model_analysis['model_accuracy'] = results['model_accuracy']
                if 'model_confidence' in results:
                    model_analysis['model_confidence'] = results['model_confidence']
                if 'feature_importance' in results:
                    model_analysis['feature_importance'] = results['feature_importance']
        
        # Compare model performance
        model_analysis['model_comparison'] = self._compare_model_performance(model_analysis)
        model_analysis['model_recommendations'] = self._generate_model_recommendations(model_analysis)
        
        return model_analysis
    
    def _generate_risk_analysis(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk analysis."""
        risk_analysis = {
            'portfolio_risk': {},
            'regime_risk': {},
            'model_risk': {},
            'liquidity_risk': {},
            'concentration_risk': {},
            'risk_recommendations': []
        }
        
        # Extract risk metrics from results
        for result_type, results in pipeline_results.items():
            if results and isinstance(results, dict):
                if 'risk_metrics' in results:
                    risk_analysis['portfolio_risk'] = results['risk_metrics']
                if 'regime_risk' in results:
                    risk_analysis['regime_risk'] = results['regime_risk']
                if 'model_risk' in results:
                    risk_analysis['model_risk'] = results['model_risk']
        
        # Calculate additional risk metrics
        risk_analysis['liquidity_risk'] = self._calculate_liquidity_risk(pipeline_results)
        risk_analysis['concentration_risk'] = self._calculate_concentration_risk(pipeline_results)
        risk_analysis['risk_recommendations'] = self._generate_risk_recommendations(risk_analysis)
        
        return risk_analysis
    
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
    
    # Helper methods for new reporting functionality
    def _extract_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from results."""
        metrics = {}
        
        # Common performance metrics
        performance_keys = [
            'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown',
            'total_trades', 'avg_trade_return', 'profit_factor', 'volatility',
            'var_95', 'calmar_ratio', 'sortino_ratio', 'information_ratio'
        ]
        
        for key in performance_keys:
            if key in results:
                metrics[key] = results[key]
        
        return metrics
    
    def _calculate_overall_performance_summary(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance summary."""
        summary = {
            'best_performing_method': None,
            'worst_performing_method': None,
            'average_sharpe_ratio': 0.0,
            'average_return': 0.0,
            'consistency_score': 0.0
        }
        
        # Collect metrics from all methods
        all_metrics = []
        for method, metrics in backtesting_results.items():
            if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
                all_metrics.append((method, metrics))
        
        if all_metrics:
            # Calculate averages
            sharpe_ratios = [m[1].get('sharpe_ratio', 0) for m in all_metrics]
            returns = [m[1].get('total_return', 0) for m in all_metrics]
            
            summary['average_sharpe_ratio'] = np.mean(sharpe_ratios) if sharpe_ratios else 0
            summary['average_return'] = np.mean(returns) if returns else 0
            
            # Find best and worst performing methods
            best_method = max(all_metrics, key=lambda x: x[1].get('sharpe_ratio', 0))
            worst_method = min(all_metrics, key=lambda x: x[1].get('sharpe_ratio', 0))
            
            summary['best_performing_method'] = {
                'method': best_method[0],
                'sharpe_ratio': best_method[1].get('sharpe_ratio', 0),
                'total_return': best_method[1].get('total_return', 0)
            }
            
            summary['worst_performing_method'] = {
                'method': worst_method[0],
                'sharpe_ratio': worst_method[1].get('sharpe_ratio', 0),
                'total_return': worst_method[1].get('total_return', 0)
            }
            
            # Calculate consistency score (lower std dev = higher consistency)
            if len(sharpe_ratios) > 1:
                consistency_score = 1.0 - (np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8))
                summary['consistency_score'] = max(0, min(1, consistency_score))
        
        return summary
    
    def _analyze_regime_stability(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime stability and transitions."""
        stability = {
            'regime_persistence': {},
            'transition_frequency': {},
            'regime_volatility': {},
            'stability_score': 0.0
        }
        
        # Analyze regime persistence
        if 'regime_performance' in regime_analysis:
            for regime, performance in regime_analysis['regime_performance'].items():
                if 'duration' in performance:
                    stability['regime_persistence'][regime] = performance['duration']
        
        # Analyze transition frequency
        if 'regime_transitions' in regime_analysis:
            transitions = regime_analysis['regime_transitions']
            if isinstance(transitions, dict):
                stability['transition_frequency'] = transitions
        
        # Calculate overall stability score
        if stability['regime_persistence']:
            avg_persistence = np.mean(list(stability['regime_persistence'].values()))
            stability['stability_score'] = min(1.0, avg_persistence / 30.0)  # Normalize to 30 days
        
        return stability
    
    def _generate_regime_recommendations(self, regime_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate regime-specific recommendations."""
        recommendations = []
        
        # Analyze regime performance
        if 'regime_performance' in regime_analysis:
            for regime, performance in regime_analysis['regime_performance'].items():
                if 'regime_return' in performance and performance['regime_return'] < 0:
                    recommendations.append({
                        'type': 'REGIME_PERFORMANCE',
                        'regime': regime,
                        'issue': f'Negative returns in {regime}',
                        'recommendation': f'Review strategy for {regime} regime or consider regime-specific adjustments'
                    })
        
        # Analyze regime stability
        if 'regime_stability' in regime_analysis:
            stability = regime_analysis['regime_stability']
            if stability.get('stability_score', 0) < 0.5:
                recommendations.append({
                    'type': 'REGIME_STABILITY',
                    'issue': 'Low regime stability',
                    'recommendation': 'Consider increasing regime persistence or adjusting regime detection parameters'
                })
        
        return recommendations
    
    def _compare_model_performance(self, model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different models."""
        comparison = {
            'best_model': None,
            'model_rankings': [],
            'performance_gaps': {},
            'consistency_analysis': {}
        }
        
        # Collect model performance data
        models = []
        if 'model_accuracy' in model_analysis:
            for model_name, accuracy in model_analysis['model_accuracy'].items():
                models.append({
                    'name': model_name,
                    'accuracy': accuracy,
                    'confidence': model_analysis['model_confidence'].get(model_name, 0)
                })
        
        if models:
            # Rank models by accuracy
            models.sort(key=lambda x: x['accuracy'], reverse=True)
            comparison['model_rankings'] = models
            
            # Identify best model
            if models:
                comparison['best_model'] = models[0]
            
            # Calculate performance gaps
            if len(models) > 1:
                best_accuracy = models[0]['accuracy']
                for model in models[1:]:
                    gap = best_accuracy - model['accuracy']
                    comparison['performance_gaps'][model['name']] = gap
        
        return comparison
    
    def _generate_model_recommendations(self, model_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model-specific recommendations."""
        recommendations = []
        
        # Check model accuracy
        if 'model_accuracy' in model_analysis:
            for model_name, accuracy in model_analysis['model_accuracy'].items():
                if accuracy < 0.6:
                    recommendations.append({
                        'type': 'MODEL_ACCURACY',
                        'model': model_name,
                        'issue': f'Low accuracy: {accuracy:.2%}',
                        'recommendation': f'Consider retraining {model_name} or adjusting hyperparameters'
                    })
        
        # Check model confidence
        if 'model_confidence' in model_analysis:
            for model_name, confidence in model_analysis['model_confidence'].items():
                if confidence < 0.7:
                    recommendations.append({
                        'type': 'MODEL_CONFIDENCE',
                        'model': model_name,
                        'issue': f'Low confidence: {confidence:.2%}',
                        'recommendation': f'Improve feature engineering or model architecture for {model_name}'
                    })
        
        return recommendations
    
    def _calculate_liquidity_risk(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate liquidity risk metrics."""
        liquidity_risk = {
            'volume_analysis': {},
            'spread_analysis': {},
            'execution_risk': {},
            'liquidity_score': 0.0
        }
        
        # Extract volume and spread data from results
        for result_type, results in pipeline_results.items():
            if results and isinstance(results, dict):
                if 'volume_metrics' in results:
                    liquidity_risk['volume_analysis'] = results['volume_metrics']
                if 'spread_metrics' in results:
                    liquidity_risk['spread_analysis'] = results['spread_metrics']
        
        # Calculate liquidity score (simplified)
        if liquidity_risk['volume_analysis']:
            avg_volume = liquidity_risk['volume_analysis'].get('average_volume', 0)
            liquidity_risk['liquidity_score'] = min(1.0, avg_volume / 1000000)  # Normalize to 1M
        
        return liquidity_risk
    
    def _calculate_concentration_risk(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate concentration risk metrics."""
        concentration_risk = {
            'position_concentration': {},
            'regime_concentration': {},
            'time_concentration': {},
            'concentration_score': 0.0
        }
        
        # Extract concentration data from results
        for result_type, results in pipeline_results.items():
            if results and isinstance(results, dict):
                if 'position_metrics' in results:
                    concentration_risk['position_concentration'] = results['position_metrics']
                if 'regime_distribution' in results:
                    concentration_risk['regime_concentration'] = results['regime_distribution']
        
        # Calculate concentration score
        if concentration_risk['regime_concentration']:
            regime_dist = concentration_risk['regime_concentration']
            if isinstance(regime_dist, dict):
                # Calculate Herfindahl index for concentration
                values = list(regime_dist.values())
                if values:
                    hhi = sum(v**2 for v in values)
                    concentration_risk['concentration_score'] = hhi
        
        return concentration_risk
    
    def _generate_risk_recommendations(self, risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk-specific recommendations."""
        recommendations = []
        
        # Check portfolio risk
        if 'portfolio_risk' in risk_analysis:
            portfolio_risk = risk_analysis['portfolio_risk']
            if 'var_95' in portfolio_risk and portfolio_risk['var_95'] > 0.05:
                recommendations.append({
                    'type': 'PORTFOLIO_RISK',
                    'issue': f'High VaR: {portfolio_risk["var_95"]:.2%}',
                    'recommendation': 'Consider reducing position sizes or improving risk management'
                })
        
        # Check concentration risk
        if 'concentration_risk' in risk_analysis:
            conc_risk = risk_analysis['concentration_risk']
            if conc_risk.get('concentration_score', 0) > 0.3:
                recommendations.append({
                    'type': 'CONCENTRATION_RISK',
                    'issue': f'High concentration: {conc_risk["concentration_score"]:.2f}',
                    'recommendation': 'Diversify across more regimes or time periods'
                })
        
        return recommendations

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