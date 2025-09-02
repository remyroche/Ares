"""
Error Reporter

Provides comprehensive error summaries, statistics, and detailed error analysis.
Aggregates results from multiple analysis tools to give a complete picture of code quality issues.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
from collections import defaultdict, Counter

from ..core.config import ReportingConfig
from ..utils.file_utils import find_python_files


@dataclass
class ErrorSummary:
    """Container for error summary statistics."""
    total_errors: int
    total_warnings: int
    total_files: int
    files_with_errors: int
    files_with_warnings: int
    error_rate: float  # Errors per file
    warning_rate: float  # Warnings per file
    critical_errors: int
    high_priority_errors: int
    medium_priority_errors: int
    low_priority_errors: int


@dataclass
class ErrorCategory:
    """Container for error category information."""
    name: str
    count: int
    severity_distribution: Dict[str, int]
    files_affected: int
    examples: List[str]
    description: str


@dataclass
class FileErrorSummary:
    """Container for file-level error summary."""
    file_path: str
    total_errors: int
    total_warnings: int
    error_categories: Dict[str, int]
    worst_errors: List[str]
    error_density: float  # Errors per line of code


@dataclass
class ErrorReport:
    """Container for comprehensive error report."""
    timestamp: str
    summary: ErrorSummary
    categories: List[ErrorCategory]
    files: List[FileErrorSummary]
    trends: Dict[str, Any]
    recommendations: List[str]


class ErrorReporter:
    """
    Comprehensive error reporting and analysis.
    
    Aggregates errors from multiple sources:
    - Linting tools (flake8, pylint, mypy)
    - Syntax analysis
    - Complexity analysis
    - Dead code analysis
    - Security analysis
    """
    
    def __init__(self, config: Optional[ReportingConfig] = None):
        """
        Initialize the error reporter.
        
        Args:
            config: Reporting configuration
        """
        self.config = config or ReportingConfig()
        self.severity_levels = ['critical', 'high', 'medium', 'low']
        self.error_sources = []
        self.error_data = {}
        
    def add_error_source(self, source_name: str, errors: List[Dict]):
        """
        Add errors from a specific analysis tool.
        
        Args:
            source_name: Name of the analysis tool
            errors: List of error dictionaries
        """
        self.error_sources.append(source_name)
        self.error_data[source_name] = errors
    
    def add_linter_errors(self, linter_name: str, errors: List[Dict]):
        """Add errors from a linter tool."""
        self.add_error_source(f"linter_{linter_name}", errors)
    
    def add_syntax_errors(self, errors: List[Dict]):
        """Add syntax analysis errors."""
        self.add_error_source("syntax_analysis", errors)
    
    def add_complexity_issues(self, issues: List[Dict]):
        """Add complexity analysis issues."""
        self.add_error_source("complexity_analysis", issues)
    
    def add_dead_code_issues(self, issues: List[Dict]):
        """Add dead code analysis issues."""
        self.add_error_source("dead_code_analysis", issues)
    
    def add_security_issues(self, issues: List[Dict]):
        """Add security analysis issues."""
        self.add_error_source("security_analysis", issues)
    
    def generate_report(self, target_path: Optional[Union[str, Path]] = None) -> ErrorReport:
        """
        Generate comprehensive error report.
        
        Args:
            target_path: Optional path to analyze for additional context
            
        Returns:
            ErrorReport object
        """
        # Aggregate all errors
        all_errors = self._aggregate_errors()
        
        # Generate summary
        summary = self._generate_summary(all_errors)
        
        # Categorize errors
        categories = self._categorize_errors(all_errors)
        
        # Analyze files
        files = self._analyze_files(all_errors, target_path)
        
        # Generate trends
        trends = self._generate_trends(all_errors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary, categories, files)
        
        return ErrorReport(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            categories=categories,
            files=files,
            trends=trends,
            recommendations=recommendations
        )
    
    def _aggregate_errors(self) -> List[Dict]:
        """Aggregate all errors from different sources."""
        all_errors = []
        
        for source, errors in self.error_data.items():
            for error in errors:
                # Add source information
                error_with_source = error.copy()
                error_with_source['source'] = source
                all_errors.append(error_with_source)
        
        return all_errors
    
    def _generate_summary(self, errors: List[Dict]) -> ErrorSummary:
        """Generate error summary statistics."""
        if not errors:
            return ErrorSummary(
                total_errors=0, total_warnings=0, total_files=0,
                files_with_errors=0, files_with_warnings=0,
                error_rate=0.0, warning_rate=0.0,
                critical_errors=0, high_priority_errors=0,
                medium_priority_errors=0, low_priority_errors=0
            )
        
        # Count errors and warnings
        total_errors = len([e for e in errors if e.get('type') == 'error'])
        total_warnings = len([e for e in errors if e.get('type') == 'warning'])
        
        # Count files
        files_with_errors = len(set(e.get('file_path', '') for e in errors if e.get('type') == 'error'))
        files_with_warnings = len(set(e.get('file_path', '') for e in errors if e.get('type') == 'warning'))
        total_files = len(set(e.get('file_path', '') for e in errors))
        
        # Calculate rates
        error_rate = total_errors / total_files if total_files > 0 else 0.0
        warning_rate = total_warnings / total_files if total_files > 0 else 0.0
        
        # Count by priority
        critical_errors = len([e for e in errors if e.get('severity') == 'critical'])
        high_priority_errors = len([e for e in errors if e.get('severity') == 'high'])
        medium_priority_errors = len([e for e in errors if e.get('severity') == 'medium'])
        low_priority_errors = len([e for e in errors if e.get('severity') == 'low'])
        
        return ErrorSummary(
            total_errors=total_errors,
            total_warnings=total_warnings,
            total_files=total_files,
            files_with_errors=files_with_errors,
            files_with_warnings=files_with_warnings,
            error_rate=error_rate,
            warning_rate=warning_rate,
            critical_errors=critical_errors,
            high_priority_errors=high_priority_errors,
            medium_priority_errors=medium_priority_errors,
            low_priority_errors=low_priority_errors
        )
    
    def _categorize_errors(self, errors: List[Dict]) -> List[ErrorCategory]:
        """Categorize errors by type and severity."""
        if not errors:
            return []
        
        # Group errors by category
        category_groups = defaultdict(list)
        for error in errors:
            category = error.get('category', error.get('issue_type', 'unknown'))
            category_groups[category].append(error)
        
        categories = []
        for category_name, category_errors in category_groups.items():
            # Count by severity
            severity_distribution = Counter(e.get('severity', 'unknown') for e in category_errors)
            
            # Count affected files
            files_affected = len(set(e.get('file_path', '') for e in category_errors))
            
            # Get examples
            examples = [e.get('description', '')[:100] for e in category_errors[:5]]
            
            # Get description
            description = self._get_category_description(category_name)
            
            categories.append(ErrorCategory(
                name=category_name,
                count=len(category_errors),
                severity_distribution=dict(severity_distribution),
                files_affected=files_affected,
                examples=examples,
                description=description
            ))
        
        # Sort by count (descending)
        categories.sort(key=lambda x: x.count, reverse=True)
        
        return categories
    
    def _get_category_description(self, category: str) -> str:
        """Get human-readable description for error category."""
        descriptions = {
            'syntax_error': 'Python syntax errors that prevent code execution',
            'unused_import': 'Imports that are not used in the code',
            'unused_variable': 'Variables that are defined but never used',
            'unused_function': 'Functions that are defined but never called',
            'unused_class': 'Classes that are defined but never instantiated',
            'complexity': 'Code that is too complex to maintain easily',
            'style': 'Code style violations (PEP 8, etc.)',
            'security': 'Potential security vulnerabilities',
            'performance': 'Code that may cause performance issues',
            'maintainability': 'Code that is difficult to maintain',
            'documentation': 'Missing or inadequate documentation',
            'test_coverage': 'Insufficient test coverage',
            'import_error': 'Issues with import statements',
            'naming': 'Naming convention violations',
            'duplicate_code': 'Code that is duplicated across files',
            'dead_code': 'Code that can never be executed',
            'type_error': 'Type-related issues (mypy, etc.)',
            'linting': 'General linting rule violations'
        }
        
        return descriptions.get(category, f'Issues related to {category}')
    
    def _analyze_files(self, errors: List[Dict], target_path: Optional[Union[str, Path]] = None) -> List[FileErrorSummary]:
        """Analyze errors by file."""
        if not errors:
            return []
        
        # Group errors by file
        file_groups = defaultdict(list)
        for error in errors:
            file_path = error.get('file_path', '')
            if file_path:
                file_groups[file_path].append(error)
        
        file_summaries = []
        for file_path, file_errors in file_groups.items():
            # Count errors and warnings
            total_errors = len([e for e in file_errors if e.get('type') == 'error'])
            total_warnings = len([e for e in file_errors if e.get('type') == 'warning'])
            
            # Count by category
            error_categories = Counter(e.get('category', e.get('issue_type', 'unknown')) for e in file_errors)
            
            # Get worst errors (by severity)
            worst_errors = [e.get('description', '') for e in sorted(
                file_errors, 
                key=lambda x: self._severity_score(x.get('severity', 'low'))
            )[:5]]
            
            # Calculate error density
            error_density = self._calculate_error_density(file_path, total_errors, target_path)
            
            file_summaries.append(FileErrorSummary(
                file_path=file_path,
                total_errors=total_errors,
                total_warnings=total_warnings,
                error_categories=dict(error_categories),
                worst_errors=worst_errors,
                error_density=error_density
            ))
        
        # Sort by total errors (descending)
        file_summaries.sort(key=lambda x: x.total_errors, reverse=True)
        
        return file_summaries
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting."""
        scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
        return scores.get(severity.lower(), 0)
    
    def _calculate_error_density(self, file_path: str, error_count: int, target_path: Optional[Union[str, Path]] = None) -> float:
        """Calculate error density (errors per line of code)."""
        try:
            if target_path:
                full_path = Path(target_path) / file_path
            else:
                full_path = Path(file_path)
            
            if full_path.exists() and full_path.is_file():
                with open(full_path, 'r', encoding='utf-8') as f:
                    line_count = len(f.readlines())
                    return error_count / line_count if line_count > 0 else 0.0
        except Exception:
            pass
        
        return 0.0
    
    def _generate_trends(self, errors: List[Dict]) -> Dict[str, Any]:
        """Generate trend analysis."""
        if not errors:
            return {}
        
        trends = {}
        
        # Error distribution by source
        source_distribution = Counter(e.get('source', 'unknown') for e in errors)
        trends['by_source'] = dict(source_distribution)
        
        # Error distribution by severity
        severity_distribution = Counter(e.get('severity', 'unknown') for e in errors)
        trends['by_severity'] = dict(severity_distribution)
        
        # Error distribution by category
        category_distribution = Counter(e.get('category', e.get('issue_type', 'unknown')) for e in errors)
        trends['by_category'] = dict(category_distribution)
        
        # Most problematic files
        file_error_counts = Counter(e.get('file_path', '') for e in errors)
        trends['problematic_files'] = dict(file_error_counts.most_common(10))
        
        return trends
    
    def _generate_recommendations(self, summary: ErrorSummary, categories: List[ErrorCategory], files: List[FileErrorSummary]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if summary.total_errors == 0 and summary.total_warnings == 0:
            recommendations.append("✅ Excellent! No errors or warnings found.")
            return recommendations
        
        # Critical issues
        if summary.critical_errors > 0:
            recommendations.append(f"🚨 {summary.critical_errors} critical errors found. These must be fixed immediately.")
        
        # High priority issues
        if summary.high_priority_errors > 0:
            recommendations.append(f"🔴 {summary.high_priority_errors} high priority errors found. Address these next.")
        
        # Files with many errors
        files_with_many_errors = [f for f in files if f.total_errors > 10]
        if files_with_many_errors:
            worst_file = files_with_many_errors[0]
            recommendations.append(f"📁 {worst_file.file_path} has {worst_file.total_errors} errors. Consider refactoring this file.")
        
        # Error categories
        if categories:
            worst_category = categories[0]
            if worst_category.count > 20:
                recommendations.append(f"📊 {worst_category.name} is the most common issue ({worst_category.count} occurrences). Focus on this category first.")
        
        # Error density
        high_density_files = [f for f in files if f.error_density > 0.1]
        if high_density_files:
            recommendations.append(f"⚠️ {len(high_density_files)} files have high error density. Consider breaking them into smaller modules.")
        
        # General recommendations
        if summary.total_errors > 100:
            recommendations.append("📈 Large number of errors found. Consider addressing them incrementally, starting with critical and high priority issues.")
        
        if summary.files_with_errors > 50:
            recommendations.append("📁 Errors are spread across many files. Consider systematic cleanup by category or file type.")
        
        return recommendations
    
    def export_report(self, report: ErrorReport, format: str = 'json', output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export error report in various formats.
        
        Args:
            report: ErrorReport object to export
            format: Export format ('json', 'csv', 'text', 'html')
            output_path: Optional path to save the report
            
        Returns:
            Exported report content
        """
        if format.lower() == 'json':
            content = json.dumps(asdict(report), indent=2)
        elif format.lower() == 'csv':
            content = self._report_to_csv(report)
        elif format.lower() == 'text':
            content = self._report_to_text(report)
        elif format.lower() == 'html':
            content = self._report_to_html(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return content
    
    def _report_to_csv(self, report: ErrorReport) -> str:
        """Convert report to CSV format."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Summary
        writer.writerow(['Category', 'Value'])
        writer.writerow(['Total Errors', report.summary.total_errors])
        writer.writerow(['Total Warnings', report.summary.total_warnings])
        writer.writerow(['Files with Errors', report.summary.files_with_errors])
        writer.writerow(['Critical Errors', report.summary.critical_errors])
        writer.writerow(['High Priority Errors', report.summary.high_priority_errors])
        writer.writerow(['Medium Priority Errors', report.summary.medium_priority_errors])
        writer.writerow(['Low Priority Errors', report.summary.low_priority_errors])
        writer.writerow([])
        
        # Categories
        writer.writerow(['Error Category', 'Count', 'Files Affected'])
        for category in report.categories:
            writer.writerow([category.name, category.count, category.files_affected])
        writer.writerow([])
        
        # Files
        writer.writerow(['File', 'Errors', 'Warnings', 'Error Density'])
        for file_summary in report.files:
            writer.writerow([
                file_summary.file_path,
                file_summary.total_errors,
                file_summary.total_warnings,
                f"{file_summary.error_density:.3f}"
            ])
        
        return output.getvalue()
    
    def _report_to_text(self, report: ErrorReport) -> str:
        """Convert report to human-readable text format."""
        lines = []
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {report.timestamp}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Total Errors: {report.summary.total_errors}")
        lines.append(f"Total Warnings: {report.summary.total_warnings}")
        lines.append(f"Files with Errors: {report.summary.files_with_errors}")
        lines.append(f"Critical Errors: {report.summary.critical_errors}")
        lines.append(f"High Priority Errors: {report.summary.high_priority_errors}")
        lines.append("")
        
        # Categories
        lines.append("ERROR CATEGORIES")
        lines.append("-" * 20)
        for category in report.categories:
            lines.append(f"{category.name}: {category.count} issues affecting {category.files_affected} files")
            lines.append(f"  Description: {category.description}")
            lines.append("")
        
        # Files
        lines.append("FILES WITH ERRORS")
        lines.append("-" * 20)
        for file_summary in report.files[:10]:  # Top 10
            lines.append(f"{file_summary.file_path}: {file_summary.total_errors} errors, {file_summary.total_warnings} warnings")
        
        if len(report.files) > 10:
            lines.append(f"... and {len(report.files) - 10} more files")
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 20)
        for recommendation in report.recommendations:
            lines.append(f"• {recommendation}")
        
        return '\n'.join(lines)
    
    def _report_to_html(self, report: ErrorReport) -> str:
        """Convert report to HTML format."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .category {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007acc; }}
        .file {{ background-color: #fff; padding: 10px; margin: 5px 0; border: 1px solid #ddd; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
        .critical {{ color: #dc3545; font-weight: bold; }}
        .high {{ color: #fd7e14; font-weight: bold; }}
        .medium {{ color: #ffc107; font-weight: bold; }}
        .low {{ color: #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Error Analysis Report</h1>
        <p>Generated: {report.timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Errors:</strong> <span class="critical">{report.summary.total_errors}</span></p>
        <p><strong>Total Warnings:</strong> <span class="medium">{report.summary.total_warnings}</span></p>
        <p><strong>Files with Errors:</strong> {report.summary.files_with_errors}</p>
        <p><strong>Critical Errors:</strong> <span class="critical">{report.summary.critical_errors}</span></p>
        <p><strong>High Priority Errors:</strong> <span class="high">{report.summary.high_priority_errors}</span></p>
    </div>
    
    <h2>Error Categories</h2>
"""
        
        for category in report.categories:
            html += f"""
    <div class="category">
        <h3>{category.name} ({category.count} issues)</h3>
        <p><strong>Files Affected:</strong> {category.files_affected}</p>
        <p><strong>Description:</strong> {category.description}</p>
    </div>
"""
        
        html += """
    <h2>Files with Errors</h2>
"""
        
        for file_summary in report.files[:20]:  # Top 20
            html += f"""
    <div class="file">
        <h4>{file_summary.file_path}</h4>
        <p>Errors: {file_summary.total_errors}, Warnings: {file_summary.total_warnings}</p>
        <p>Error Density: {file_summary.error_density:.3f}</p>
    </div>
"""
        
        html += """
    <h2>Recommendations</h2>
"""
        
        for recommendation in report.recommendations:
            html += f"""
    <div class="recommendation">
        <p>{recommendation}</p>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        return html
    
    def get_error_statistics(self, report: ErrorReport) -> Dict[str, Any]:
        """Get statistical analysis of errors."""
        if not report.files:
            return {}
        
        # File-level statistics
        error_counts = [f.total_errors for f in report.files]
        warning_counts = [f.total_warnings for f in report.files]
        error_densities = [f.error_density for f in report.files if f.error_density > 0]
        
        stats = {
            'file_count': len(report.files),
            'error_statistics': {
                'mean': statistics.mean(error_counts) if error_counts else 0,
                'median': statistics.median(error_counts) if error_counts else 0,
                'std_dev': statistics.stdev(error_counts) if len(error_counts) > 1 else 0,
                'min': min(error_counts) if error_counts else 0,
                'max': max(error_counts) if error_counts else 0
            },
            'warning_statistics': {
                'mean': statistics.mean(warning_counts) if warning_counts else 0,
                'median': statistics.median(warning_counts) if warning_counts else 0,
                'std_dev': statistics.stdev(warning_counts) if len(warning_counts) > 1 else 0,
                'min': min(warning_counts) if warning_counts else 0,
                'max': max(warning_counts) if warning_counts else 0
            }
        }
        
        if error_densities:
            stats['density_statistics'] = {
                'mean': statistics.mean(error_densities),
                'median': statistics.median(error_densities),
                'std_dev': statistics.stdev(error_densities) if len(error_densities) > 1 else 0,
                'min': min(error_densities),
                'max': max(error_densities)
            }
        
        return stats