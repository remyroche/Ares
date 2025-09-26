"""
Execution Reporter for Step Execution Monitoring

This module provides comprehensive reporting capabilities for step execution,
performance monitoring, and quality assessment.
"""

import time
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback
import sys
from contextlib import contextmanager


class ReportLevel(Enum):
    """Report detail levels."""
    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class FunctionCallSummary:
    """Summary of function call execution."""
    function_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    input_args: Dict[str, Any] = field(default_factory=dict)
    output_data: Any = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for execution."""
    total_execution_time: float
    function_calls: List[FunctionCallSummary]
    memory_peak: float
    memory_average: float
    cpu_peak: float
    cpu_average: float
    throughput: float
    efficiency_score: float


@dataclass
class ErrorAnalysis:
    """Analysis of errors during execution."""
    total_errors: int
    error_types: Dict[str, int]
    critical_errors: List[str]
    warnings: List[str]
    error_rate: float
    recovery_attempts: int
    success_rate: float


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    data_quality_score: float
    output_consistency: float
    reliability_score: float
    maintainability_score: float
    test_coverage: float
    documentation_score: float


@dataclass
class Step03ExecutionReport:
    """Comprehensive execution report."""
    step_name: str
    execution_id: str
    start_time: float
    end_time: float
    total_duration: float
    status: str
    performance: PerformanceMetrics
    errors: ErrorAnalysis
    quality: QualityMetrics
    function_calls: List[FunctionCallSummary]
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class Step03ExecutionReporter:
    """
    Execution reporter for monitoring and reporting step execution.
    
    This class provides comprehensive monitoring, logging, and reporting
    capabilities for step execution processes.
    """
    
    def __init__(self, 
                 report_level: ReportLevel = ReportLevel.DETAILED,
                 output_format: ReportFormat = ReportFormat.JSON,
                 enable_monitoring: bool = True):
        """
        Initialize the execution reporter.
        
        Args:
            report_level: Level of detail in reports
            output_format: Format for output reports
            enable_monitoring: Whether to enable real-time monitoring
        """
        self.report_level = report_level
        self.output_format = output_format
        self.enable_monitoring = enable_monitoring
        
        # Execution tracking
        self.current_execution_id: Optional[str] = None
        self.function_calls: List[FunctionCallSummary] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Performance tracking
        self.memory_usage_history: List[float] = []
        self.cpu_usage_history: List[float] = []
        self.error_count = 0
        self.warning_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize monitoring if enabled
        if self.enable_monitoring:
            self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup performance monitoring."""
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            self.logger.warning("psutil not available, performance monitoring limited")
    
    def start_execution(self, step_name: str, execution_id: Optional[str] = None) -> str:
        """
        Start tracking execution of a step.
        
        Args:
            step_name: Name of the step being executed
            execution_id: Optional custom execution ID
            
        Returns:
            Execution ID for this run
        """
        if execution_id is None:
            execution_id = f"{step_name}_{int(time.time())}"
        
        self.current_execution_id = execution_id
        self.start_time = time.time()
        self.function_calls = []
        self.memory_usage_history = []
        self.cpu_usage_history = []
        self.error_count = 0
        self.warning_count = 0
        
        self.logger.info(f"Starting execution: {step_name} (ID: {execution_id})")
        return execution_id
    
    def end_execution(self, status: str = "completed") -> Step03ExecutionReport:
        """
        End execution tracking and generate report.
        
        Args:
            status: Final status of execution
            
        Returns:
            Complete execution report
        """
        if self.current_execution_id is None:
            raise ValueError("No active execution to end")
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Generate performance metrics
        performance = self._calculate_performance_metrics()
        
        # Generate error analysis
        errors = self._analyze_errors()
        
        # Generate quality metrics
        quality = self._calculate_quality_metrics()
        
        # Create comprehensive report
        report = Step03ExecutionReport(
            step_name=self.current_execution_id.split('_')[0],
            execution_id=self.current_execution_id,
            start_time=self.start_time,
            end_time=self.end_time,
            total_duration=total_duration,
            status=status,
            performance=performance,
            errors=errors,
            quality=quality,
            function_calls=self.function_calls,
            metadata=self._get_execution_metadata(),
            recommendations=self._generate_recommendations()
        )
        
        self.logger.info(f"Execution completed: {self.current_execution_id} (Status: {status})")
        
        # Reset for next execution
        self.current_execution_id = None
        
        return report
    
    def track_function_call(self, 
                           function_name: str,
                           start_time: float,
                           end_time: float,
                           success: bool,
                           error_message: Optional[str] = None,
                           input_args: Optional[Dict[str, Any]] = None,
                           output_data: Any = None) -> FunctionCallSummary:
        """
        Track a function call during execution.
        
        Args:
            function_name: Name of the function called
            start_time: Start time of function call
            end_time: End time of function call
            success: Whether the function call succeeded
            error_message: Error message if failed
            input_args: Input arguments to the function
            output_data: Output data from the function
            
        Returns:
            Function call summary
        """
        duration = end_time - start_time
        
        # Get resource usage if available
        memory_usage = 0.0
        cpu_usage = 0.0
        
        if self.psutil_available:
            try:
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                cpu_usage = process.cpu_percent()
            except Exception:
                pass
        
        summary = FunctionCallSummary(
            function_name=function_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            error_message=error_message,
            input_args=input_args or {},
            output_data=output_data,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
        self.function_calls.append(summary)
        
        # Track errors
        if not success:
            self.error_count += 1
            self.logger.error(f"Function call failed: {function_name} - {error_message}")
        else:
            self.logger.debug(f"Function call completed: {function_name} ({duration:.3f}s)")
        
        return summary
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics from execution data."""
        if not self.function_calls:
            return PerformanceMetrics(
                total_execution_time=0.0,
                function_calls=[],
                memory_peak=0.0,
                memory_average=0.0,
                cpu_peak=0.0,
                cpu_average=0.0,
                throughput=0.0,
                efficiency_score=0.0
            )
        
        total_time = sum(fc.duration for fc in self.function_calls)
        memory_usage = [fc.memory_usage for fc in self.function_calls if fc.memory_usage > 0]
        cpu_usage = [fc.cpu_usage for fc in self.function_calls if fc.cpu_usage > 0]
        
        successful_calls = [fc for fc in self.function_calls if fc.success]
        success_rate = len(successful_calls) / len(self.function_calls) if self.function_calls else 0
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            function_calls=self.function_calls,
            memory_peak=max(memory_usage) if memory_usage else 0.0,
            memory_average=sum(memory_usage) / len(memory_usage) if memory_usage else 0.0,
            cpu_peak=max(cpu_usage) if cpu_usage else 0.0,
            cpu_average=sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0,
            throughput=len(self.function_calls) / total_time if total_time > 0 else 0.0,
            efficiency_score=success_rate * 100
        )
    
    def _analyze_errors(self) -> ErrorAnalysis:
        """Analyze errors during execution."""
        failed_calls = [fc for fc in self.function_calls if not fc.success]
        
        error_types = {}
        critical_errors = []
        warnings = []
        
        for call in failed_calls:
            if call.error_message:
                error_type = type(Exception()).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                if "critical" in call.error_message.lower():
                    critical_errors.append(call.error_message)
                elif "warning" in call.error_message.lower():
                    warnings.append(call.error_message)
        
        total_calls = len(self.function_calls)
        error_rate = len(failed_calls) / total_calls if total_calls > 0 else 0.0
        success_rate = 1.0 - error_rate
        
        return ErrorAnalysis(
            total_errors=len(failed_calls),
            error_types=error_types,
            critical_errors=critical_errors,
            warnings=warnings,
            error_rate=error_rate,
            recovery_attempts=0,  # Could be enhanced to track actual recovery attempts
            success_rate=success_rate
        )
    
    def _calculate_quality_metrics(self) -> QualityMetrics:
        """Calculate quality metrics for the execution."""
        # Simplified quality metrics - could be enhanced with more sophisticated analysis
        successful_calls = [fc for fc in self.function_calls if fc.success]
        success_rate = len(successful_calls) / len(self.function_calls) if self.function_calls else 0
        
        # Estimate data quality based on successful function calls
        data_quality_score = success_rate * 100
        
        # Estimate output consistency based on similar function call patterns
        output_consistency = 100.0 if len(set(fc.function_name for fc in successful_calls)) > 0 else 0.0
        
        # Reliability based on success rate
        reliability_score = success_rate * 100
        
        # Maintainability based on function call complexity
        maintainability_score = max(0, 100 - len(self.function_calls) * 2)
        
        return QualityMetrics(
            data_quality_score=data_quality_score,
            output_consistency=output_consistency,
            reliability_score=reliability_score,
            maintainability_score=maintainability_score,
            test_coverage=0.0,  # Would need test execution data
            documentation_score=0.0  # Would need documentation analysis
        )
    
    def _get_execution_metadata(self) -> Dict[str, Any]:
        """Get metadata about the execution environment."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'execution_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'monitoring_enabled': self.enable_monitoring,
            'report_level': self.report_level.value,
            'output_format': self.output_format.value
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on execution analysis."""
        recommendations = []
        
        if self.error_count > 0:
            recommendations.append(f"Consider error handling improvements - {self.error_count} errors occurred")
        
        if len(self.function_calls) > 50:
            recommendations.append("Consider breaking down into smaller functions - high function call count")
        
        # Performance recommendations
        if self.function_calls:
            avg_duration = sum(fc.duration for fc in self.function_calls) / len(self.function_calls)
            if avg_duration > 1.0:
                recommendations.append("Consider performance optimization - high average function duration")
        
        return recommendations
    
    def generate_report(self, report: Step03ExecutionReport, output_path: Optional[str] = None) -> str:
        """
        Generate a formatted report.
        
        Args:
            report: Execution report to format
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        if self.output_format == ReportFormat.JSON:
            report_str = self._generate_json_report(report)
        elif self.output_format == ReportFormat.TEXT:
            report_str = self._generate_text_report(report)
        elif self.output_format == ReportFormat.HTML:
            report_str = self._generate_html_report(report)
        elif self.output_format == ReportFormat.MARKDOWN:
            report_str = self._generate_markdown_report(report)
        else:
            report_str = self._generate_text_report(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            self.logger.info(f"Report saved to: {output_path}")
        
        return report_str
    
    def _generate_json_report(self, report: Step03ExecutionReport) -> str:
        """Generate JSON formatted report."""
        report_dict = {
            'step_name': report.step_name,
            'execution_id': report.execution_id,
            'start_time': report.start_time,
            'end_time': report.end_time,
            'total_duration': report.total_duration,
            'status': report.status,
            'performance': {
                'total_execution_time': report.performance.total_execution_time,
                'memory_peak': report.performance.memory_peak,
                'memory_average': report.performance.memory_average,
                'cpu_peak': report.performance.cpu_peak,
                'cpu_average': report.performance.cpu_average,
                'throughput': report.performance.throughput,
                'efficiency_score': report.performance.efficiency_score
            },
            'errors': {
                'total_errors': report.errors.total_errors,
                'error_types': report.errors.error_types,
                'critical_errors': report.errors.critical_errors,
                'warnings': report.errors.warnings,
                'error_rate': report.errors.error_rate,
                'success_rate': report.errors.success_rate
            },
            'quality': {
                'data_quality_score': report.quality.data_quality_score,
                'output_consistency': report.quality.output_consistency,
                'reliability_score': report.quality.reliability_score,
                'maintainability_score': report.quality.maintainability_score
            },
            'function_calls': [
                {
                    'function_name': fc.function_name,
                    'duration': fc.duration,
                    'success': fc.success,
                    'memory_usage': fc.memory_usage,
                    'cpu_usage': fc.cpu_usage
                }
                for fc in report.function_calls
            ],
            'metadata': report.metadata,
            'recommendations': report.recommendations
        }
        
        return json.dumps(report_dict, indent=2, default=str)
    
    def _generate_text_report(self, report: Step03ExecutionReport) -> str:
        """Generate text formatted report."""
        lines = [
            f"Execution Report: {report.step_name}",
            f"Execution ID: {report.execution_id}",
            f"Status: {report.status}",
            f"Duration: {report.total_duration:.3f}s",
            "",
            "Performance Metrics:",
            f"  Total Execution Time: {report.performance.total_execution_time:.3f}s",
            f"  Memory Peak: {report.performance.memory_peak:.1f} MB",
            f"  Memory Average: {report.performance.memory_average:.1f} MB",
            f"  CPU Peak: {report.performance.cpu_peak:.1f}%",
            f"  CPU Average: {report.performance.cpu_average:.1f}%",
            f"  Throughput: {report.performance.throughput:.2f} calls/s",
            f"  Efficiency Score: {report.performance.efficiency_score:.1f}%",
            "",
            "Error Analysis:",
            f"  Total Errors: {report.errors.total_errors}",
            f"  Error Rate: {report.errors.error_rate:.2%}",
            f"  Success Rate: {report.errors.success_rate:.2%}",
            f"  Critical Errors: {len(report.errors.critical_errors)}",
            f"  Warnings: {len(report.errors.warnings)}",
            "",
            "Quality Metrics:",
            f"  Data Quality Score: {report.quality.data_quality_score:.1f}%",
            f"  Output Consistency: {report.quality.output_consistency:.1f}%",
            f"  Reliability Score: {report.quality.reliability_score:.1f}%",
            f"  Maintainability Score: {report.quality.maintainability_score:.1f}%",
            "",
            "Function Calls:",
        ]
        
        for fc in report.function_calls:
            status = "✓" if fc.success else "✗"
            lines.append(f"  {status} {fc.function_name} ({fc.duration:.3f}s)")
        
        if report.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in report.recommendations:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report: Step03ExecutionReport) -> str:
        """Generate HTML formatted report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Execution Report: {report.step_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Execution Report: {report.step_name}</h1>
                <p><strong>Execution ID:</strong> {report.execution_id}</p>
                <p><strong>Status:</strong> <span class="{'success' if report.status == 'completed' else 'error'}">{report.status}</span></p>
                <p><strong>Duration:</strong> {report.total_duration:.3f}s</p>
            </div>
            
            <h2>Performance Metrics</h2>
            <div class="metric">Total Execution Time: {report.performance.total_execution_time:.3f}s</div>
            <div class="metric">Memory Peak: {report.performance.memory_peak:.1f} MB</div>
            <div class="metric">Memory Average: {report.performance.memory_average:.1f} MB</div>
            <div class="metric">CPU Peak: {report.performance.cpu_peak:.1f}%</div>
            <div class="metric">CPU Average: {report.performance.cpu_average:.1f}%</div>
            <div class="metric">Throughput: {report.performance.throughput:.2f} calls/s</div>
            <div class="metric">Efficiency Score: {report.performance.efficiency_score:.1f}%</div>
            
            <h2>Error Analysis</h2>
            <div class="metric">Total Errors: {report.errors.total_errors}</div>
            <div class="metric">Error Rate: {report.errors.error_rate:.2%}</div>
            <div class="metric">Success Rate: {report.errors.success_rate:.2%}</div>
            
            <h2>Function Calls</h2>
            <table>
                <tr><th>Function</th><th>Duration</th><th>Status</th><th>Memory</th><th>CPU</th></tr>
        """
        
        for fc in report.function_calls:
            status_class = "success" if fc.success else "error"
            status_text = "✓ Success" if fc.success else "✗ Failed"
            html += f"""
                <tr>
                    <td>{fc.function_name}</td>
                    <td>{fc.duration:.3f}s</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{fc.memory_usage:.1f} MB</td>
                    <td>{fc.cpu_usage:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        if report.recommendations:
            html += "<h2>Recommendations</h2><ul>"
            for rec in report.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, report: Step03ExecutionReport) -> str:
        """Generate Markdown formatted report."""
        lines = [
            f"# Execution Report: {report.step_name}",
            "",
            f"**Execution ID:** {report.execution_id}",
            f"**Status:** {report.status}",
            f"**Duration:** {report.total_duration:.3f}s",
            "",
            "## Performance Metrics",
            "",
            f"- **Total Execution Time:** {report.performance.total_execution_time:.3f}s",
            f"- **Memory Peak:** {report.performance.memory_peak:.1f} MB",
            f"- **Memory Average:** {report.performance.memory_average:.1f} MB",
            f"- **CPU Peak:** {report.performance.cpu_peak:.1f}%",
            f"- **CPU Average:** {report.performance.cpu_average:.1f}%",
            f"- **Throughput:** {report.performance.throughput:.2f} calls/s",
            f"- **Efficiency Score:** {report.performance.efficiency_score:.1f}%",
            "",
            "## Error Analysis",
            "",
            f"- **Total Errors:** {report.errors.total_errors}",
            f"- **Error Rate:** {report.errors.error_rate:.2%}",
            f"- **Success Rate:** {report.errors.success_rate:.2%}",
            f"- **Critical Errors:** {len(report.errors.critical_errors)}",
            f"- **Warnings:** {len(report.errors.warnings)}",
            "",
            "## Quality Metrics",
            "",
            f"- **Data Quality Score:** {report.quality.data_quality_score:.1f}%",
            f"- **Output Consistency:** {report.quality.output_consistency:.1f}%",
            f"- **Reliability Score:** {report.quality.reliability_score:.1f}%",
            f"- **Maintainability Score:** {report.quality.maintainability_score:.1f}%",
            "",
            "## Function Calls",
            "",
            "| Function | Duration | Status | Memory | CPU |",
            "|----------|----------|--------|--------|-----|"
        ]
        
        for fc in report.function_calls:
            status = "✓ Success" if fc.success else "✗ Failed"
            lines.append(f"| {fc.function_name} | {fc.duration:.3f}s | {status} | {fc.memory_usage:.1f} MB | {fc.cpu_usage:.1f}% |")
        
        if report.recommendations:
            lines.extend(["", "## Recommendations", ""])
            for rec in report.recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    @contextmanager
    def monitor_function_call(self, function_name: str, *args, **kwargs):
        """
        Context manager for monitoring function calls.
        
        Args:
            function_name: Name of the function being called
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        start_time = time.time()
        success = True
        error_message = None
        output_data = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"Function call failed: {function_name} - {error_message}")
            raise
        finally:
            end_time = time.time()
            self.track_function_call(
                function_name=function_name,
                start_time=start_time,
                end_time=end_time,
                success=success,
                error_message=error_message,
                input_args={'args': args, 'kwargs': kwargs},
                output_data=output_data
            )


# Convenience functions
def create_execution_reporter(report_level: ReportLevel = ReportLevel.DETAILED,
                            output_format: ReportFormat = ReportFormat.JSON,
                            enable_monitoring: bool = True) -> Step03ExecutionReporter:
    """Create an execution reporter with default configuration."""
    return Step03ExecutionReporter(
        report_level=report_level,
        output_format=output_format,
        enable_monitoring=enable_monitoring
    )


def quick_execution_report(step_name: str, 
                          function_calls: List[Callable],
                          *args, **kwargs) -> Step03ExecutionReport:
    """
    Generate a quick execution report for a list of function calls.
    
    Args:
        step_name: Name of the step
        function_calls: List of functions to execute and monitor
        *args: Arguments to pass to functions
        **kwargs: Keyword arguments to pass to functions
        
    Returns:
        Execution report
    """
    reporter = create_execution_reporter()
    execution_id = reporter.start_execution(step_name)
    
    try:
        for func in function_calls:
            with reporter.monitor_function_call(func.__name__, *args, **kwargs):
                func(*args, **kwargs)
        
        return reporter.end_execution("completed")
    except Exception as e:
        return reporter.end_execution(f"failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example usage of the execution reporter
    reporter = create_execution_reporter()
    
    # Start execution
    execution_id = reporter.start_execution("example_step")
    
    # Monitor some function calls
    def example_function_1():
        time.sleep(0.1)
        return "result1"
    
    def example_function_2():
        time.sleep(0.2)
        return "result2"
    
    with reporter.monitor_function_call("example_function_1"):
        result1 = example_function_1()
    
    with reporter.monitor_function_call("example_function_2"):
        result2 = example_function_2()
    
    # End execution and generate report
    report = reporter.end_execution("completed")
    
    # Generate and print report
    report_str = reporter.generate_report(report)
    print(report_str)