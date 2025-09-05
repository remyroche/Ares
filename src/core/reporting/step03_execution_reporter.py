"""
Step03 Execution Report Generator.

This module provides comprehensive execution reporting for step03 with:
1. Detailed function call tracking
2. Performance metrics analysis
3. Error analysis and patterns
4. Resource usage monitoring
5. Quality metrics assessment
6. Recommendations and insights
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

class ReportFormat(Enum):
    """Report output formats."""
    JSON = 'json'
    HTML = 'html'
    PDF = 'pdf'
    CSV = 'csv'
    MARKDOWN = 'markdown'

class ReportLevel(Enum):
    """Report detail levels."""
    SUMMARY = 'summary'
    DETAILED = 'detailed'
    COMPREHENSIVE = 'comprehensive'

@dataclass
class FunctionCallSummary:
    """Summary of function call metrics."""
    function_name: str
    module_name: str
    call_count: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    success_count: int
    failure_count: int
    success_rate: float
    total_memory_used: float
    avg_memory_per_call: float
    total_cpu_used: float
    avg_cpu_per_call: float
    error_types: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    """Performance metrics for the execution."""
    total_execution_time: float
    total_memory_peak: float
    total_cpu_usage: float
    function_calls_count: int
    nested_calls_count: int
    max_call_depth: int
    avg_call_duration: float
    slowest_function: str
    fastest_function: str
    memory_efficiency_score: float
    cpu_efficiency_score: float
    overall_performance_score: float

@dataclass
class ErrorAnalysis:
    """Error analysis results."""
    total_errors: int
    errors_by_category: Dict[str, int]
    errors_by_severity: Dict[str, int]
    most_common_errors: List[Tuple[str, int]]
    error_recovery_rate: float
    critical_errors: List[str]
    error_patterns: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class QualityMetrics:
    """Quality metrics for the execution."""
    data_quality_score: float
    algorithm_performance_score: float
    error_handling_score: float
    resource_utilization_score: float
    overall_quality_score: float
    quality_issues: List[str]
    quality_improvements: List[str]

@dataclass
class Step03ExecutionReport:
    """Comprehensive Step03 execution report."""
    report_id: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    symbol: str
    exchange: str
    timeframe: str
    data_directory: str
    function_calls: List[FunctionCallSummary]
    performance_metrics: PerformanceMetrics
    error_analysis: ErrorAnalysis
    quality_metrics: QualityMetrics
    raw_function_calls: List[Dict[str, Any]]
    raw_errors: List[Dict[str, Any]]
    raw_performance_data: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    next_steps: List[str]
    report_version: str = '1.0.0'
    generated_by: str = 'Step03ExecutionReporter'
    generated_at: datetime = field(default_factory=datetime.now)

class Step03ExecutionReporter:
    """Comprehensive execution reporter for Step03."""

    def __init__(self, output_directory: str='reports/step03', enable_html_reports: bool=True, enable_pdf_reports: bool=False, enable_csv_exports: bool=True, log_level: str='INFO') -> None:
        """
        Initialize the Step03 execution reporter.
        
        Args:
            output_directory: Directory to save reports
            enable_html_reports: Enable HTML report generation
            enable_pdf_reports: Enable PDF report generation
            enable_csv_exports: Enable CSV data exports
            log_level: Logging level
        """
        self.output_directory = Path(output_directory)
        self.enable_html_reports = enable_html_reports
        self.enable_pdf_reports = enable_pdf_reports
        self.enable_csv_exports = enable_csv_exports
        self.log_level = log_level
        self.logger = logging.getLogger(f'{__name__}.Step03ExecutionReporter')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.performance_thresholds = {'max_function_duration': 30.0, 'max_memory_usage': 1000.0, 'max_cpu_usage': 80.0, 'min_success_rate': 0.95, 'max_error_rate': 0.05}

    def _analyze_function_calls(self, raw_calls: List[Dict[str, Any]]) -> List[FunctionCallSummary]:
        """Analyze function call data and create summaries."""
        function_summaries = {}
        for call in raw_calls:
            func_name = call.get('function_name', 'unknown')
            module_name = call.get('module_name', 'unknown')
            key = f'{module_name}.{func_name}'
            if key not in function_summaries:
                function_summaries[key] = {'function_name': func_name, 'module_name': module_name, 'call_count': 0, 'durations': [], 'success_count': 0, 'failure_count': 0, 'memory_usage': [], 'cpu_usage': [], 'error_types': set(), 'performance_warnings': []}
            summary = function_summaries[key]
            summary['call_count'] += 1
            duration = call.get('duration', 0.0)
            summary['durations'].append(duration)
            if call.get('success', False):
                summary['success_count'] += 1
            else:
                summary['failure_count'] += 1
                error_type = call.get('error_type', 'unknown')
                summary['error_types'].add(error_type)
            memory_delta = call.get('memory_delta', 0.0)
            if memory_delta:
                summary['memory_usage'].append(memory_delta)
            cpu_delta = call.get('cpu_delta', 0.0)
            if cpu_delta:
                summary['cpu_usage'].append(cpu_delta)
            warnings = call.get('performance_warnings', [])
            summary['performance_warnings'].extend(warnings)
        summaries = []
        for key, data in function_summaries.items():
            durations = data['durations']
            memory_usage = data['memory_usage']
            cpu_usage = data['cpu_usage']
            summary = FunctionCallSummary(function_name=data['function_name'], module_name=data['module_name'], call_count=data['call_count'], total_duration=sum(durations), avg_duration=np.mean(durations) if durations else 0.0, min_duration=min(durations) if durations else 0.0, max_duration=max(durations) if durations else 0.0, success_count=data['success_count'], failure_count=data['failure_count'], success_rate=data['success_count'] / data['call_count'] if data['call_count'] > 0 else 0.0, total_memory_used=sum(memory_usage), avg_memory_per_call=np.mean(memory_usage) if memory_usage else 0.0, total_cpu_used=sum(cpu_usage), avg_cpu_per_call=np.mean(cpu_usage) if cpu_usage else 0.0, error_types=list(data['error_types']), performance_warnings=list(set(data['performance_warnings'])))
            summaries.append(summary)
        return summaries

    def _calculate_performance_metrics(self, function_calls: List[FunctionCallSummary], raw_calls: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        total_duration = sum((call.total_duration for call in function_calls))
        total_calls = sum((call.call_count for call in function_calls))
        slowest_function = max(function_calls, key=lambda x: x.max_duration) if function_calls else None
        fastest_function = min(function_calls, key=lambda x: x.min_duration) if function_calls else None
        nested_calls = sum((1 for call in raw_calls if call.get('nested_calls_count', 0) > 0))
        max_call_depth = max((call.get('call_depth', 0) for call in raw_calls), default=0)
        memory_efficiency = self._calculate_memory_efficiency(function_calls)
        cpu_efficiency = self._calculate_cpu_efficiency(function_calls)
        performance_score = self._calculate_overall_performance_score(function_calls)
        return PerformanceMetrics(total_execution_time=total_duration, total_memory_peak=max((call.total_memory_used for call in function_calls), default=0.0), total_cpu_usage=sum((call.total_cpu_used for call in function_calls)), function_calls_count=total_calls, nested_calls_count=nested_calls, max_call_depth=max_call_depth, avg_call_duration=total_duration / total_calls if total_calls > 0 else 0.0, slowest_function=slowest_function.function_name if slowest_function else 'N/A', fastest_function=fastest_function.function_name if fastest_function else 'N/A', memory_efficiency_score=memory_efficiency, cpu_efficiency_score=cpu_efficiency, overall_performance_score=performance_score)

    def _calculate_memory_efficiency(self, function_calls: List[FunctionCallSummary]) -> float:
        """Calculate memory efficiency score (0-100)."""
        if not function_calls:
            return 100.0
        efficiency_scores = []
        for call in function_calls:
            if call.avg_memory_per_call > self.performance_thresholds['max_memory_usage']:
                efficiency_scores.append(0.0)
            else:
                efficiency = max(0, 100 - call.avg_memory_per_call / self.performance_thresholds['max_memory_usage'] * 100)
                efficiency_scores.append(efficiency)
        return np.mean(efficiency_scores) if efficiency_scores else 100.0

    def _calculate_cpu_efficiency(self, function_calls: List[FunctionCallSummary]) -> float:
        """Calculate CPU efficiency score (0-100)."""
        if not function_calls:
            return 100.0
        efficiency_scores = []
        for call in function_calls:
            if call.avg_cpu_per_call > self.performance_thresholds['max_cpu_usage']:
                efficiency_scores.append(0.0)
            else:
                efficiency = max(0, 100 - call.avg_cpu_per_call / self.performance_thresholds['max_cpu_usage'] * 100)
                efficiency_scores.append(efficiency)
        return np.mean(efficiency_scores) if efficiency_scores else 100.0

    def _calculate_overall_performance_score(self, function_calls: List[FunctionCallSummary]) -> float:
        """Calculate overall performance score (0-100)."""
        if not function_calls:
            return 100.0
        success_rate_score = np.mean([call.success_rate for call in function_calls]) * 100
        duration_scores = []
        for call in function_calls:
            if call.avg_duration > self.performance_thresholds['max_function_duration']:
                duration_scores.append(0.0)
            else:
                duration_score = max(0, 100 - call.avg_duration / self.performance_thresholds['max_function_duration'] * 100)
                duration_scores.append(duration_score)
        duration_efficiency_score = np.mean(duration_scores) if duration_scores else 100.0
        memory_efficiency_score = self._calculate_memory_efficiency(function_calls)
        cpu_efficiency_score = self._calculate_cpu_efficiency(function_calls)
        overall_score = success_rate_score * 0.4 + duration_efficiency_score * 0.3 + memory_efficiency_score * 0.15 + cpu_efficiency_score * 0.15
        return min(100.0, max(0.0, overall_score))

    def _analyze_errors(self, raw_errors: List[Dict[str, Any]]) -> ErrorAnalysis:
        """Analyze error data and create comprehensive error analysis."""
        if not raw_errors:
            return ErrorAnalysis(total_errors=0, errors_by_category={}, errors_by_severity={}, most_common_errors=[], error_recovery_rate=100.0, critical_errors=[], error_patterns=[], recommendations=[])
        errors_by_category = {}
        errors_by_severity = {}
        error_types = {}
        recovery_attempts = 0
        successful_recoveries = 0
        for error in raw_errors:
            category = error.get('error_category', 'unknown')
            severity = error.get('severity', 'medium')
            error_type = error.get('error_type', 'unknown')
            recovery_successful = error.get('recovery_successful', False)
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
            error_types[error_type] = error_types.get(error_type, 0) + 1
            if error.get('recovery_attempted', False):
                recovery_attempts += 1
                if recovery_successful:
                    successful_recoveries += 1
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        error_recovery_rate = successful_recoveries / recovery_attempts * 100 if recovery_attempts > 0 else 100.0
        critical_errors = [error for error in raw_errors if error.get('severity') == 'critical']
        error_patterns = self._detect_error_patterns(raw_errors)
        recommendations = self._generate_error_recommendations(errors_by_category, error_patterns)
        return ErrorAnalysis(total_errors=len(raw_errors), errors_by_category=errors_by_category, errors_by_severity=errors_by_severity, most_common_errors=most_common_errors, error_recovery_rate=error_recovery_rate, critical_errors=[e.get('error_id', 'unknown') for e in critical_errors], error_patterns=error_patterns, recommendations=recommendations)

    def _detect_error_patterns(self, raw_errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in error data."""
        patterns = []
        function_errors = {}
        for error in raw_errors:
            func_name = error.get('function_name', 'unknown')
            if func_name not in function_errors:
                function_errors[func_name] = []
            function_errors[func_name].append(error)
        for func_name, errors in function_errors.items():
            if len(errors) >= 3:
                patterns.append({'pattern_type': 'frequent_errors', 'function_name': func_name, 'error_count': len(errors), 'error_categories': [e.get('error_category', 'unknown') for e in errors], 'severity': max((e.get('severity', 'medium') for e in errors)), 'recommendation': f'Function {func_name} has {len(errors)} recent errors'})
        return patterns

    def _generate_error_recommendations(self, errors_by_category: Dict[str, int], error_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        if errors_by_category.get('validation', 0) > 0:
            recommendations.append('Review input validation logic to prevent validation errors')
        if errors_by_category.get('data_quality', 0) > 0:
            recommendations.append('Implement data quality checks before processing')
        if errors_by_category.get('performance', 0) > 0:
            recommendations.append('Optimize performance-critical functions')
        if errors_by_category.get('resource', 0) > 0:
            recommendations.append('Review resource allocation and cleanup')
        for pattern in error_patterns:
            if pattern['pattern_type'] == 'frequent_errors':
                recommendations.append(f"Investigate repeated errors in {pattern['function_name']}")
        return recommendations

    def _calculate_quality_metrics(self, function_calls: List[FunctionCallSummary], error_analysis: ErrorAnalysis, performance_metrics: PerformanceMetrics) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        data_quality_score = np.mean([call.success_rate for call in function_calls]) * 100 if function_calls else 100.0
        algorithm_performance_score = performance_metrics.overall_performance_score
        error_handling_score = error_analysis.error_recovery_rate
        resource_utilization_score = (performance_metrics.memory_efficiency_score + performance_metrics.cpu_efficiency_score) / 2
        overall_quality_score = data_quality_score * 0.3 + algorithm_performance_score * 0.3 + error_handling_score * 0.2 + resource_utilization_score * 0.2
        quality_issues = []
        quality_improvements = []
        if data_quality_score < 95:
            quality_issues.append('Low data quality score - investigate validation failures')
            quality_improvements.append('Enhance input validation and data preprocessing')
        if algorithm_performance_score < 80:
            quality_issues.append('Low algorithm performance - optimize slow functions')
            quality_improvements.append('Profile and optimize performance-critical functions')
        if error_handling_score < 90:
            quality_issues.append('Poor error handling - improve recovery mechanisms')
            quality_improvements.append('Implement better error recovery strategies')
        if resource_utilization_score < 70:
            quality_issues.append('Poor resource utilization - optimize memory and CPU usage')
            quality_improvements.append('Optimize resource allocation and cleanup')
        return QualityMetrics(data_quality_score=data_quality_score, algorithm_performance_score=algorithm_performance_score, error_handling_score=error_handling_score, resource_utilization_score=resource_utilization_score, overall_quality_score=overall_quality_score, quality_issues=quality_issues, quality_improvements=quality_improvements)

    def _generate_insights(self, function_calls: List[FunctionCallSummary], performance_metrics: PerformanceMetrics, error_analysis: ErrorAnalysis, quality_metrics: QualityMetrics) -> List[str]:
        """Generate insights from the execution data."""
        insights = []
        if performance_metrics.slowest_function != 'N/A':
            insights.append(f'Slowest function: {performance_metrics.slowest_function} ({performance_metrics.avg_call_duration:.2f}s average)')
        if performance_metrics.memory_efficiency_score < 70:
            insights.append('Memory usage is high - consider optimizing memory-intensive functions')
        if performance_metrics.cpu_efficiency_score < 70:
            insights.append('CPU usage is high - consider optimizing CPU-intensive functions')
        if error_analysis.total_errors > 0:
            most_common = error_analysis.most_common_errors[0] if error_analysis.most_common_errors else None
            if most_common:
                insights.append(f'Most common error: {most_common[0]} ({most_common[1]} occurrences)')
        if error_analysis.error_recovery_rate < 80:
            insights.append('Error recovery rate is low - improve error handling mechanisms')
        if quality_metrics.overall_quality_score >= 90:
            insights.append('Overall execution quality is excellent')
        elif quality_metrics.overall_quality_score >= 80:
            insights.append('Overall execution quality is good with room for improvement')
        else:
            insights.append('Overall execution quality needs significant improvement')
        return insights

    def _generate_recommendations(self, function_calls: List[FunctionCallSummary], performance_metrics: PerformanceMetrics, error_analysis: ErrorAnalysis, quality_metrics: QualityMetrics) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        if performance_metrics.overall_performance_score < 80:
            recommendations.append('Optimize performance-critical functions')
        if performance_metrics.memory_efficiency_score < 70:
            recommendations.append('Implement memory optimization strategies')
        if performance_metrics.cpu_efficiency_score < 70:
            recommendations.append('Implement CPU optimization strategies')
        recommendations.extend(error_analysis.recommendations)
        recommendations.extend(quality_metrics.quality_improvements)
        if len(function_calls) > 50:
            recommendations.append('Consider breaking down large functions into smaller, more manageable pieces')
        if performance_metrics.max_call_depth > 10:
            recommendations.append('Reduce function call depth to improve maintainability')
        return recommendations

    def _generate_next_steps(self, quality_metrics: QualityMetrics, error_analysis: ErrorAnalysis) -> List[str]:
        """Generate next steps based on analysis."""
        next_steps = []
        if quality_metrics.overall_quality_score < 70:
            next_steps.append('URGENT: Address quality issues before proceeding to next step')
        if error_analysis.critical_errors:
            next_steps.append('CRITICAL: Fix critical errors immediately')
        if quality_metrics.quality_issues:
            next_steps.append('HIGH: Address quality issues identified in the report')
        if error_analysis.error_recovery_rate < 90:
            next_steps.append('MEDIUM: Improve error handling and recovery mechanisms')
        if quality_metrics.overall_quality_score >= 90:
            next_steps.append('LOW: Continue with current implementation - quality is excellent')
        return next_steps

    async def generate_report(self, execution_data: Dict[str, Any], symbol: str, exchange: str, timeframe: str, data_directory: str, start_time: datetime, end_time: datetime) -> Step03ExecutionReport:
        """Generate comprehensive execution report."""
        self.logger.info('📊 Generating comprehensive Step03 execution report...')
        raw_function_calls = execution_data.get('function_calls', [])
        raw_errors = execution_data.get('errors', [])
        raw_performance_data = execution_data.get('performance_data', [])
        function_calls = self._analyze_function_calls(raw_function_calls)
        performance_metrics = self._calculate_performance_metrics(function_calls, raw_function_calls)
        error_analysis = self._analyze_errors(raw_errors)
        quality_metrics = self._calculate_quality_metrics(function_calls, error_analysis, performance_metrics)
        insights = self._generate_insights(function_calls, performance_metrics, error_analysis, quality_metrics)
        recommendations = self._generate_recommendations(function_calls, performance_metrics, error_analysis, quality_metrics)
        next_steps = self._generate_next_steps(quality_metrics, error_analysis)
        report = Step03ExecutionReport(report_id=str(uuid.uuid4()), execution_id=execution_data.get('execution_id', str(uuid.uuid4())), start_time=start_time, end_time=end_time, total_duration=(end_time - start_time).total_seconds(), symbol=symbol, exchange=exchange, timeframe=timeframe, data_directory=data_directory, function_calls=function_calls, performance_metrics=performance_metrics, error_analysis=error_analysis, quality_metrics=quality_metrics, raw_function_calls=raw_function_calls, raw_errors=raw_errors, raw_performance_data=raw_performance_data, insights=insights, recommendations=recommendations, next_steps=next_steps)
        self.logger.info('✅ Step03 execution report generated successfully')
        return report

    async def save_report(self, report: Step03ExecutionReport, formats: List[ReportFormat]=None) -> Dict[str, str]:
        """Save report in multiple formats."""
        if formats is None:
            formats = [ReportFormat.JSON, ReportFormat.HTML, ReportFormat.CSV]
        saved_files = {}
        timestamp = report.start_time.strftime('%Y%m%d_%H%M%S')
        for format_type in formats:
            try:
                if format_type == ReportFormat.JSON:
                    file_path = await self._save_json_report(report, timestamp)
                    saved_files['json'] = str(file_path)
                elif format_type == ReportFormat.HTML and self.enable_html_reports:
                    file_path = await self._save_html_report(report, timestamp)
                    saved_files['html'] = str(file_path)
                elif format_type == ReportFormat.CSV and self.enable_csv_exports:
                    file_path = await self._save_csv_report(report, timestamp)
                    saved_files['csv'] = str(file_path)
                elif format_type == ReportFormat.MARKDOWN:
                    file_path = await self._save_markdown_report(report, timestamp)
                    saved_files['markdown'] = str(file_path)
            except Exception as e:
                self.logger.error(f'Failed to save {format_type.value} report: {e}')
        return saved_files

    async def _save_json_report(self, report: Step03ExecutionReport, timestamp: str) -> Path:
        """Save report as JSON."""
        file_path = self.output_directory / f'step03_execution_report_{timestamp}.json'
        report_dict = {'report_id': report.report_id, 'execution_id': report.execution_id, 'start_time': report.start_time.isoformat(), 'end_time': report.end_time.isoformat(), 'total_duration': report.total_duration, 'symbol': report.symbol, 'exchange': report.exchange, 'timeframe': report.timeframe, 'data_directory': report.data_directory, 'function_calls': [{'function_name': call.function_name, 'module_name': call.module_name, 'call_count': call.call_count, 'total_duration': call.total_duration, 'avg_duration': call.avg_duration, 'min_duration': call.min_duration, 'max_duration': call.max_duration, 'success_count': call.success_count, 'failure_count': call.failure_count, 'success_rate': call.success_rate, 'total_memory_used': call.total_memory_used, 'avg_memory_per_call': call.avg_memory_per_call, 'total_cpu_used': call.total_cpu_used, 'avg_cpu_per_call': call.avg_cpu_per_call, 'error_types': call.error_types, 'performance_warnings': call.performance_warnings} for call in report.function_calls], 'performance_metrics': {'total_execution_time': report.performance_metrics.total_execution_time, 'total_memory_peak': report.performance_metrics.total_memory_peak, 'total_cpu_usage': report.performance_metrics.total_cpu_usage, 'function_calls_count': report.performance_metrics.function_calls_count, 'nested_calls_count': report.performance_metrics.nested_calls_count, 'max_call_depth': report.performance_metrics.max_call_depth, 'avg_call_duration': report.performance_metrics.avg_call_duration, 'slowest_function': report.performance_metrics.slowest_function, 'fastest_function': report.performance_metrics.fastest_function, 'memory_efficiency_score': report.performance_metrics.memory_efficiency_score, 'cpu_efficiency_score': report.performance_metrics.cpu_efficiency_score, 'overall_performance_score': report.performance_metrics.overall_performance_score}, 'error_analysis': {'total_errors': report.error_analysis.total_errors, 'errors_by_category': report.error_analysis.errors_by_category, 'errors_by_severity': report.error_analysis.errors_by_severity, 'most_common_errors': report.error_analysis.most_common_errors, 'error_recovery_rate': report.error_analysis.error_recovery_rate, 'critical_errors': report.error_analysis.critical_errors, 'error_patterns': report.error_analysis.error_patterns, 'recommendations': report.error_analysis.recommendations}, 'quality_metrics': {'data_quality_score': report.quality_metrics.data_quality_score, 'algorithm_performance_score': report.quality_metrics.algorithm_performance_score, 'error_handling_score': report.quality_metrics.error_handling_score, 'resource_utilization_score': report.quality_metrics.resource_utilization_score, 'overall_quality_score': report.quality_metrics.overall_quality_score, 'quality_issues': report.quality_metrics.quality_issues, 'quality_improvements': report.quality_metrics.quality_improvements}, 'insights': report.insights, 'recommendations': report.recommendations, 'next_steps': report.next_steps, 'report_version': report.report_version, 'generated_by': report.generated_by, 'generated_at': report.generated_at.isoformat()}
        with open(file_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        self.logger.info(f'📄 JSON report saved to: {file_path}')
        return file_path

    async def _save_html_report(self, report: Step03ExecutionReport, timestamp: str) -> Path:
        """Save report as HTML."""
        file_path = self.output_directory / f'step03_execution_report_{timestamp}.html'
        html_content = f"""\n        <!DOCTYPE html>\n        <html>\n        <head>\n            <title>Step03 Execution Report - {report.symbol}</title>\n            <style>\n                body {{ font-family: Arial, sans-serif; margin: 20px; }}\n                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}\n                .section {{ margin: 20px 0; }}\n                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}\n                .error {{ color: red; }}\n                .warning {{ color: orange; }}\n                .success {{ color: green; }}\n                table {{ border-collapse: collapse; width: 100%; }}\n                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}\n                th {{ background-color: #f2f2f2; }}\n            </style>\n        </head>\n        <body>\n            <div class="header">\n                <h1>Step03 Execution Report</h1>\n                <p><strong>Symbol:</strong> {report.symbol} | <strong>Exchange:</strong> {report.exchange} | <strong>Timeframe:</strong> {report.timeframe}</p>\n                <p><strong>Execution Time:</strong> {report.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>\n                <p><strong>Total Duration:</strong> {report.total_duration:.2f} seconds</p>\n            </div>\n            \n            <div class="section">\n                <h2>Performance Metrics</h2>\n                <div class="metric">\n                    <strong>Overall Performance Score:</strong> {report.performance_metrics.overall_performance_score:.1f}/100\n                </div>\n                <div class="metric">\n                    <strong>Total Function Calls:</strong> {report.performance_metrics.function_calls_count}\n                </div>\n                <div class="metric">\n                    <strong>Memory Efficiency:</strong> {report.performance_metrics.memory_efficiency_score:.1f}/100\n                </div>\n                <div class="metric">\n                    <strong>CPU Efficiency:</strong> {report.performance_metrics.cpu_efficiency_score:.1f}/100\n                </div>\n            </div>\n            \n            <div class="section">\n                <h2>Quality Metrics</h2>\n                <div class="metric">\n                    <strong>Overall Quality Score:</strong> {report.quality_metrics.overall_quality_score:.1f}/100\n                </div>\n                <div class="metric">\n                    <strong>Data Quality:</strong> {report.quality_metrics.data_quality_score:.1f}/100\n                </div>\n                <div class="metric">\n                    <strong>Error Handling:</strong> {report.quality_metrics.error_handling_score:.1f}/100\n                </div>\n            </div>\n            \n            <div class="section">\n                <h2>Error Analysis</h2>\n                <p><strong>Total Errors:</strong> {report.error_analysis.total_errors}</p>\n                <p><strong>Error Recovery Rate:</strong> {report.error_analysis.error_recovery_rate:.1f}%</p>\n                {(f'<p class="error"><strong>Critical Errors:</strong> {len(report.error_analysis.critical_errors)}</p>' if report.error_analysis.critical_errors else '')}\n            </div>\n            \n            <div class="section">\n                <h2>Function Call Summary</h2>\n                <table>\n                    <tr>\n                        <th>Function</th>\n                        <th>Calls</th>\n                        <th>Avg Duration</th>\n                        <th>Success Rate</th>\n                        <th>Memory Usage</th>\n                    </tr>\n                    {''.join((f'<tr><td>{call.function_name}</td><td>{call.call_count}</td><td>{call.avg_duration:.3f}s</td><td>{call.success_rate:.1%}</td><td>{call.avg_memory_per_call:.1f}MB</td></tr>' for call in report.function_calls[:10]))}\n                </table>\n            </div>\n            \n            <div class="section">\n                <h2>Insights</h2>\n                <ul>\n                    {''.join((f'<li>{insight}</li>' for insight in report.insights))}\n                </ul>\n            </div>\n            \n            <div class="section">\n                <h2>Recommendations</h2>\n                <ul>\n                    {''.join((f'<li>{recommendation}</li>' for recommendation in report.recommendations))}\n                </ul>\n            </div>\n            \n            <div class="section">\n                <h2>Next Steps</h2>\n                <ul>\n                    {''.join((f'<li>{step}</li>' for step in report.next_steps))}\n                </ul>\n            </div>\n        </body>\n        </html>\n        """
        with open(file_path, 'w') as f:
            f.write(html_content)
        self.logger.info(f'📄 HTML report saved to: {file_path}')
        return file_path

    async def _save_csv_report(self, report: Step03ExecutionReport, timestamp: str) -> Path:
        """Save report data as CSV files."""
        csv_dir = self.output_directory / f'step03_execution_data_{timestamp}'
        csv_dir.mkdir(exist_ok=True)
        if report.function_calls:
            calls_df = pd.DataFrame([{'function_name': call.function_name, 'module_name': call.module_name, 'call_count': call.call_count, 'total_duration': call.total_duration, 'avg_duration': call.avg_duration, 'min_duration': call.min_duration, 'max_duration': call.max_duration, 'success_count': call.success_count, 'failure_count': call.failure_count, 'success_rate': call.success_rate, 'total_memory_used': call.total_memory_used, 'avg_memory_per_call': call.avg_memory_per_call, 'total_cpu_used': call.total_cpu_used, 'avg_cpu_per_call': call.avg_cpu_per_call} for call in report.function_calls])
            calls_df.to_csv(csv_dir / 'function_calls.csv', index=False)
        perf_df = pd.DataFrame([{'metric': 'total_execution_time', 'value': report.performance_metrics.total_execution_time}, {'metric': 'total_memory_peak', 'value': report.performance_metrics.total_memory_peak}, {'metric': 'total_cpu_usage', 'value': report.performance_metrics.total_cpu_usage}, {'metric': 'function_calls_count', 'value': report.performance_metrics.function_calls_count}, {'metric': 'overall_performance_score', 'value': report.performance_metrics.overall_performance_score}])
        perf_df.to_csv(csv_dir / 'performance_metrics.csv', index=False)
        quality_df = pd.DataFrame([{'metric': 'data_quality_score', 'value': report.quality_metrics.data_quality_score}, {'metric': 'algorithm_performance_score', 'value': report.quality_metrics.algorithm_performance_score}, {'metric': 'error_handling_score', 'value': report.quality_metrics.error_handling_score}, {'metric': 'resource_utilization_score', 'value': report.quality_metrics.resource_utilization_score}, {'metric': 'overall_quality_score', 'value': report.quality_metrics.overall_quality_score}])
        quality_df.to_csv(csv_dir / 'quality_metrics.csv', index=False)
        self.logger.info(f'📄 CSV data saved to: {csv_dir}')
        return csv_dir

    async def _save_markdown_report(self, report: Step03ExecutionReport, timestamp: str) -> Path:
        """Save report as Markdown."""
        file_path = self.output_directory / f'step03_execution_report_{timestamp}.md'
        markdown_content = f"# Step03 Execution Report\n\n## Overview\n- **Symbol:** {report.symbol}\n- **Exchange:** {report.exchange}\n- **Timeframe:** {report.timeframe}\n- **Execution Time:** {report.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n- **Total Duration:** {report.total_duration:.2f} seconds\n\n## Performance Metrics\n- **Overall Performance Score:** {report.performance_metrics.overall_performance_score:.1f}/100\n- **Total Function Calls:** {report.performance_metrics.function_calls_count}\n- **Memory Efficiency:** {report.performance_metrics.memory_efficiency_score:.1f}/100\n- **CPU Efficiency:** {report.performance_metrics.cpu_efficiency_score:.1f}/100\n\n## Quality Metrics\n- **Overall Quality Score:** {report.quality_metrics.overall_quality_score:.1f}/100\n- **Data Quality:** {report.quality_metrics.data_quality_score:.1f}/100\n- **Error Handling:** {report.quality_metrics.error_handling_score:.1f}/100\n\n## Error Analysis\n- **Total Errors:** {report.error_analysis.total_errors}\n- **Error Recovery Rate:** {report.error_analysis.error_recovery_rate:.1f}%\n- **Critical Errors:** {len(report.error_analysis.critical_errors)}\n\n## Insights\n{chr(10).join((f'- {insight}' for insight in report.insights))}\n\n## Recommendations\n{chr(10).join((f'- {recommendation}' for recommendation in report.recommendations))}\n\n## Next Steps\n{chr(10).join((f'- {step}' for step in report.next_steps))}\n"
        with open(file_path, 'w') as f:
            f.write(markdown_content)
        self.logger.info(f'📄 Markdown report saved to: {file_path}')
        return file_path