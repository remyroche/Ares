"""
Dead Code Analyzer

Detects unused code, dead imports, and unreachable code using Vulture library.
Helps identify code that can be safely removed to improve codebase cleanliness.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass
import vulture
from vulture.core import Vulture

from ..core.config import AnalysisConfig
from minimal_file_utils import find_python_files


@dataclass
class DeadCodeIssue:
    """Container for dead code analysis results."""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    confidence: float
    code_snippet: str
    severity: str


@dataclass
class DeadCodeReport:
    """Container for dead code analysis report."""
    total_issues: int
    issues_by_type: Dict[str, int]
    issues_by_file: Dict[str, List[DeadCodeIssue]]
    issues_by_severity: Dict[str, List[DeadCodeIssue]]
    confidence_distribution: Dict[str, int]
    potential_savings: Dict[str, int]  # Lines of code that could be removed


class DeadCodeAnalyzer:
    """
    Analyzes Python code for dead/unused code using Vulture.
    
    Detects:
    - Unused imports
    - Unused variables
    - Unused functions and classes
    - Dead code blocks
    - Unreachable code
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the dead code analyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()
        self.confidence_threshold = getattr(self.config, 'confidence_threshold', 80.0)
        self.ignore_patterns = getattr(self.config, 'ignore_patterns', [])
        self.whitelist = getattr(self.config, 'whitelist', [])
        
        # Initialize Vulture with custom configuration
        self.vulture = Vulture()
        self._configure_vulture()
    
    def _configure_vulture(self):
        """Configure Vulture with custom settings."""
        # Set confidence threshold
        self.vulture.min_confidence = self.confidence_threshold
        
        # Add common whitelist patterns
        default_whitelist = [
            # Common patterns that might be false positives
            'unused_import',
            'unused_variable',
            'unused_function',
            'unused_class',
            'unused_method',
            'unused_attribute',
            'unused_argument',
            'unused_parameter',
            'unused_return_value',
            'unused_assignment',
            'unused_expression',
            'unused_statement',
            'unused_import_statement',
            'unused_from_import',
            'unused_import_alias',
            'unused_import_from',
            'unused_import_as',
            'unused_import_from_as',
            'unused_import_from_star',
            'unused_import_star',
            'unused_import_relative',
            'unused_import_absolute',
            'unused_import_relative_from',
            'unused_import_absolute_from',
            'unused_import_relative_as',
            'unused_import_absolute_as',
            'unused_import_relative_star',
            'unused_import_absolute_star',
            'unused_import_relative_from_star',
            'unused_import_absolute_from_star',
            'unused_import_relative_from_as',
            'unused_import_absolute_from_as',
            'unused_import_relative_from_star_as',
            'unused_import_absolute_from_star_as',
        ]
        
        # Add user-defined whitelist
        if self.whitelist:
            default_whitelist.extend(self.whitelist)
            
        self.vulture.whitelist = default_whitelist
    
    def analyze_file(self, file_path: Union[str, Path]) -> List[DeadCodeIssue]:
        """
        Analyze a single Python file for dead code.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of DeadCodeIssue objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.suffix == '.py':
            raise ValueError(f"File must be a Python file: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Analyze with Vulture
            issues = self._analyze_source(source, str(file_path))
            
            # Filter issues based on configuration
            filtered_issues = self._filter_issues(issues)
            
            return filtered_issues
            
        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return []
    
    def analyze_directory(self, directory: Union[str, Path]) -> DeadCodeReport:
        """
        Analyze all Python files in a directory for dead code.
        
        Args:
            directory: Path to directory
            
        Returns:
            DeadCodeReport object with analysis results
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
            
        python_files = find_python_files(directory)
        all_issues = []
        
        for file_path in python_files:
            try:
                file_issues = self.analyze_file(file_path)
                all_issues.extend(file_issues)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
        
        return self._generate_report(all_issues)
    
    def analyze_files(self, file_paths: List[Union[str, Path]]) -> DeadCodeReport:
        """
        Analyze multiple Python files for dead code.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            DeadCodeReport object with analysis results
        """
        all_issues = []
        
        for file_path in file_paths:
            try:
                file_issues = self.analyze_file(file_path)
                all_issues.extend(file_issues)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
        
        return self._generate_report(all_issues)
    
    def _analyze_source(self, source: str, file_path: str) -> List[DeadCodeIssue]:
        """Analyze source code for dead code issues."""
        issues = []
        
        try:
            # Parse AST to get line information
            tree = ast.parse(source)
            lines = source.split('\n')
            
            # Use Vulture to find dead code
            vulture_issues = self.vulture.scan(source, filename=file_path)
            
            for issue in vulture_issues:
                # Extract line number and description
                line_number = getattr(issue, 'lineno', 0)
                description = getattr(issue, 'description', str(issue))
                confidence = getattr(issue, 'confidence', 100.0)
                
                # Get code snippet
                code_snippet = self._extract_code_snippet(lines, line_number)
                
                # Determine issue type
                issue_type = self._classify_issue(description)
                
                # Determine severity
                severity = self._determine_severity(confidence, issue_type)
                
                issues.append(DeadCodeIssue(
                    file_path=file_path,
                    line_number=line_number,
                    issue_type=issue_type,
                    description=description,
                    confidence=confidence,
                    code_snippet=code_snippet,
                    severity=severity
                ))
                
        except SyntaxError as e:
            # Handle syntax errors gracefully
            issues.append(DeadCodeIssue(
                file_path=file_path,
                line_number=getattr(e, 'lineno', 0),
                issue_type='syntax_error',
                description=f"Syntax error: {str(e)}",
                confidence=100.0,
                code_snippet="",
                severity='high'
            ))
        except Exception as e:
            print(f"Warning: Error analyzing {file_path}: {e}")
        
        return issues
    
    def _extract_code_snippet(self, lines: List[str], line_number: int) -> str:
        """Extract code snippet around the specified line."""
        if line_number <= 0 or line_number > len(lines):
            return ""
        
        # Get context (2 lines before and after)
        start_line = max(0, line_number - 3)
        end_line = min(len(lines), line_number + 2)
        
        snippet_lines = []
        for i in range(start_line, end_line):
            if i == line_number - 1:  # Target line (0-indexed)
                snippet_lines.append(f"  {i+1:4d}: >>> {lines[i]}")
            else:
                snippet_lines.append(f"  {i+1:4d}:     {lines[i]}")
        
        return '\n'.join(snippet_lines)
    
    def _classify_issue(self, description: str) -> str:
        """Classify the type of dead code issue."""
        description_lower = description.lower()
        
        if 'import' in description_lower:
            return 'unused_import'
        elif 'variable' in description_lower:
            return 'unused_variable'
        elif 'function' in description_lower:
            return 'unused_function'
        elif 'class' in description_lower:
            return 'unused_class'
        elif 'method' in description_lower:
            return 'unused_method'
        elif 'attribute' in description_lower:
            return 'unused_attribute'
        elif 'argument' in description_lower or 'parameter' in description_lower:
            return 'unused_parameter'
        elif 'assignment' in description_lower:
            return 'unused_assignment'
        elif 'expression' in description_lower:
            return 'unused_expression'
        elif 'statement' in description_lower:
            return 'unused_statement'
        else:
            return 'unknown'
    
    def _determine_severity(self, confidence: float, issue_type: str) -> str:
        """Determine the severity of an issue."""
        if confidence >= 95:
            return 'high'
        elif confidence >= 80:
            return 'medium'
        else:
            return 'low'
    
    def _filter_issues(self, issues: List[DeadCodeIssue]) -> List[DeadCodeIssue]:
        """Filter issues based on configuration."""
        filtered = []
        
        for issue in issues:
            # Check confidence threshold
            if issue.confidence < self.confidence_threshold:
                continue
            
            # Check ignore patterns
            if self._should_ignore_issue(issue):
                continue
            
            filtered.append(issue)
        
        return filtered
    
    def _should_ignore_issue(self, issue: DeadCodeIssue) -> bool:
        """Check if an issue should be ignored based on patterns."""
        for pattern in self.ignore_patterns:
            if pattern in issue.description.lower():
                return True
            if pattern in issue.file_path.lower():
                return True
        return False
    
    def _generate_report(self, issues: List[DeadCodeIssue]) -> DeadCodeReport:
        """Generate a comprehensive report from all issues."""
        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = 0
            issues_by_type[issue.issue_type] += 1
        
        # Group issues by file
        issues_by_file = {}
        for issue in issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)
        
        # Calculate confidence distribution
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}
        for issue in issues:
            confidence_distribution[issue.severity] += 1
        
        # Calculate potential savings
        potential_savings = self._calculate_potential_savings(issues)
        
        return DeadCodeReport(
            total_issues=len(issues),
            issues_by_type=issues_by_type,
            issues_by_file=issues_by_file,
            issues_by_severity=issues_by_severity,
            confidence_distribution=confidence_distribution,
            potential_savings=potential_savings
        )
    
    def _calculate_potential_savings(self, issues: List[DeadCodeIssue]) -> Dict[str, int]:
        """Calculate potential lines of code that could be removed."""
        savings = {
            'total_lines': 0,
            'import_lines': 0,
            'function_lines': 0,
            'class_lines': 0,
            'variable_lines': 0
        }
        
        for issue in issues:
            if issue.confidence >= 90:  # Only count high-confidence issues
                if issue.issue_type == 'unused_import':
                    savings['import_lines'] += 1
                elif issue.issue_type in ['unused_function', 'unused_method']:
                    savings['function_lines'] += 1
                elif issue.issue_type == 'unused_class':
                    savings['class_lines'] += 1
                elif issue.issue_type == 'unused_variable':
                    savings['variable_lines'] += 1
                
                savings['total_lines'] += 1
        
        return savings
    
    def get_dead_code_summary(self, report: DeadCodeReport) -> Dict:
        """Generate a summary of dead code analysis."""
        summary = {
            'total_issues': report.total_issues,
            'issues_by_type': report.issues_by_type,
            'issues_by_severity': report.issues_by_severity,
            'potential_savings': report.potential_savings,
            'files_affected': len(report.issues_by_file),
            'high_confidence_issues': len([i for i in report.issues_by_severity.get('high', []) if i.confidence >= 95]),
            'medium_confidence_issues': len([i for i in report.issues_by_severity.get('medium', []) if i.confidence >= 80]),
            'low_confidence_issues': len([i for i in report.issues_by_severity.get('low', []) if i.confidence < 80])
        }
        
        return summary
    
    def find_critical_issues(self, report: DeadCodeReport) -> List[DeadCodeIssue]:
        """Find critical dead code issues that should be addressed first."""
        critical_issues = []
        
        for issue in report.issues_by_severity.get('high', []):
            if issue.confidence >= 95:
                critical_issues.append(issue)
        
        # Sort by confidence and line number
        critical_issues.sort(key=lambda x: (-x.confidence, x.line_number))
        
        return critical_issues
    
    def generate_cleanup_recommendations(self, report: DeadCodeReport) -> List[str]:
        """Generate cleanup recommendations based on analysis."""
        recommendations = []
        
        if report.total_issues == 0:
            recommendations.append("✅ No dead code issues found. Your codebase is clean!")
            return recommendations
        
        # High confidence issues
        high_confidence = len([i for i in report.issues_by_severity.get('high', []) if i.confidence >= 95])
        if high_confidence > 0:
            recommendations.append(f"🔴 {high_confidence} high-confidence issues found. These should be addressed immediately.")
        
        # Import issues
        import_issues = report.issues_by_type.get('unused_import', 0)
        if import_issues > 0:
            recommendations.append(f"📦 {import_issues} unused imports found. Consider removing them to improve startup time.")
        
        # Function issues
        function_issues = report.issues_by_type.get('unused_function', 0)
        if function_issues > 0:
            recommendations.append(f"⚙️ {function_issues} unused functions found. Consider removing or documenting them.")
        
        # Class issues
        class_issues = report.issues_by_type.get('unused_class', 0)
        if class_issues > 0:
            recommendations.append(f"🏗️ {class_issues} unused classes found. Consider removing or documenting them.")
        
        # Potential savings
        total_savings = report.potential_savings['total_lines']
        if total_savings > 0:
            recommendations.append(f"💾 Potential to remove {total_savings} lines of dead code.")
        
        # General recommendations
        if report.total_issues > 50:
            recommendations.append("⚠️ Large number of issues found. Consider addressing them incrementally.")
        
        if len(report.issues_by_file) > 20:
            recommendations.append("📁 Issues spread across many files. Consider systematic cleanup.")
        
        return recommendations
    
    def export_issues(self, report: DeadCodeReport, format: str = 'json') -> str:
        """Export issues in various formats."""
        if format.lower() == 'json':
            import json
            return json.dumps(self._report_to_dict(report), indent=2)
        elif format.lower() == 'csv':
            return self._report_to_csv(report)
        elif format.lower() == 'text':
            return self._report_to_text(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _report_to_dict(self, report: DeadCodeReport) -> Dict:
        """Convert report to dictionary for JSON export."""
        return {
            'total_issues': report.total_issues,
            'issues_by_type': report.issues_by_type,
            'issues_by_file': {
                file_path: [
                    {
                        'line_number': issue.line_number,
                        'issue_type': issue.issue_type,
                        'description': issue.description,
                        'confidence': issue.confidence,
                        'severity': issue.severity,
                        'code_snippet': issue.code_snippet
                    }
                    for issue in issues
                ]
                for file_path, issues in report.issues_by_file.items()
            },
            'issues_by_severity': {
                severity: [
                    {
                        'file_path': issue.file_path,
                        'line_number': issue.line_number,
                        'issue_type': issue.issue_type,
                        'description': issue.description,
                        'confidence': issue.confidence,
                        'code_snippet': issue.code_snippet
                    }
                    for issue in issues
                ]
                for severity, issues in report.issues_by_severity.items()
            },
            'potential_savings': report.potential_savings
        }
    
    def _report_to_csv(self, report: DeadCodeReport) -> str:
        """Convert report to CSV format."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['File', 'Line', 'Type', 'Description', 'Confidence', 'Severity'])
        
        # Data
        for issue in [i for issues in report.issues_by_severity.values() for i in issues]:
            writer.writerow([
                issue.file_path,
                issue.line_number,
                issue.issue_type,
                issue.description,
                issue.confidence,
                issue.severity
            ])
        
        return output.getvalue()
    
    def _report_to_text(self, report: DeadCodeReport) -> str:
        """Convert report to human-readable text format."""
        lines = []
        lines.append("DEAD CODE ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Issues: {report.total_issues}")
        lines.append(f"Files Affected: {len(report.issues_by_file)}")
        lines.append("")
        
        # Issues by type
        lines.append("Issues by Type:")
        for issue_type, count in report.issues_by_type.items():
            lines.append(f"  {issue_type}: {count}")
        lines.append("")
        
        # Issues by severity
        lines.append("Issues by Severity:")
        for severity, count in report.issues_by_severity.items():
            lines.append(f"  {severity}: {len(count)}")
        lines.append("")
        
        # Potential savings
        lines.append("Potential Savings:")
        for category, count in report.potential_savings.items():
            lines.append(f"  {category}: {count} lines")
        lines.append("")
        
        # Detailed issues
        lines.append("Detailed Issues:")
        for file_path, issues in report.issues_by_file.items():
            lines.append(f"\n{file_path}:")
            for issue in issues:
                lines.append(f"  Line {issue.line_number}: {issue.description} (Confidence: {issue.confidence}%)")
        
        return '\n'.join(lines)