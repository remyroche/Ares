"""
Placeholder Detector - Comprehensive detection of placeholders, stubs, and incomplete code.
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json

# Try to import from the code_quality package, fall back to direct imports if running standalone
try:
    from ..core.config import CodeQualityConfig, get_default_config
    from ..utils.file_utils import find_python_files
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from core.config import CodeQualityConfig, get_default_config
        from utils.file_utils import find_python_files
    except ImportError:
        # Create minimal fallback classes
        class CodeQualityConfig:
            def __init__(self):
                class AnalysisConfig:
                    exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env"]
                self.analysis = AnalysisConfig()
        
        def get_default_config():
            return CodeQualityConfig()
        
        def find_python_files(directory, exclude_patterns=None):
            """Fallback implementation of find_python_files."""
            if exclude_patterns is None:
                exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env"]
            
            python_files = []
            for root, dirs, files in os.walk(directory):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(d.startswith(pattern.replace('*', '')) for pattern in exclude_patterns)]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if not any(file_path.startswith(pattern.replace('*', '')) for pattern in exclude_patterns):
                            python_files.append(file_path)
            
            return python_files


class Placeholder:
    """Container for placeholder information."""
    
    def __init__(self, file_path: str, line: int, column: int, placeholder_type: str, 
                 content: str, severity: str = "medium", context: str = ""):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.placeholder_type = placeholder_type
        self.content = content
        self.severity = severity
        self.context = context
    
    def __repr__(self):
        return f"Placeholder({self.file_path}:{self.line}, {self.placeholder_type}: {self.content})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "placeholder_type": self.placeholder_type,
            "content": self.content,
            "severity": self.severity,
            "context": self.context
        }


class PlaceholderDetector:
    """
    Comprehensive detector for placeholders, stubs, and incomplete code.
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.placeholders: List[Placeholder] = []
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        self.summary_stats: Dict[str, Any] = {}
        
        # Define placeholder patterns
        self.comment_patterns = {
            "TODO": r"#\s*TODO[:\s]*(.+)",
            "FIXME": r"#\s*FIXME[:\s]*(.+)",
            "HACK": r"#\s*HACK[:\s]*(.+)",
            "XXX": r"#\s*XXX[:\s]*(.+)",
            "NOTE": r"#\s*NOTE[:\s]*(.+)",
            "PLACEHOLDER": r"#\s*PLACEHOLDER[:\s]*(.+)",
            "STUB": r"#\s*STUB[:\s]*(.+)",
            "IMPLEMENT": r"#\s*IMPLEMENT[:\s]*(.+)",
            "FUTURE": r"#\s*FUTURE[:\s]*(.+)",
            "LATER": r"#\s*LATER[:\s]*(.+)",
            "SOON": r"#\s*SOON[:\s]*(.+)",
            "TEMP": r"#\s*TEMP[:\s]*(.+)",
            "REMOVE": r"#\s*REMOVE[:\s]*(.+)",
            "REVIEW": r"#\s*REVIEW[:\s]*(.+)",
            "OPTIMIZE": r"#\s*OPTIMIZE[:\s]*(.+)",
            "REFACTOR": r"#\s*REFACTOR[:\s]*(.+)",
            "CLEANUP": r"#\s*CLEANUP[:\s]*(.+)",
            "BUG": r"#\s*BUG[:\s]*(.+)",
            "WARNING": r"#\s*WARNING[:\s]*(.+)",
            "DEPRECATED": r"#\s*DEPRECATED[:\s]*(.+)",
        }
        
        # Define placeholder values
        self.placeholder_values = {
            "None": "None placeholder",
            "0": "Zero placeholder",
            "0.0": "Zero float placeholder",
            "0.05": "Small value placeholder",
            "0.1": "Small value placeholder",
            "1": "One placeholder",
            "100.0": "Large value placeholder",
            "100": "Large value placeholder",
            "1000": "Large value placeholder",
            "10000": "Large value placeholder",
            "999999": "Large value placeholder",
            "999999999": "Large value placeholder",
            "True": "Boolean placeholder",
            "False": "Boolean placeholder",
            "''": "Empty string placeholder",
            '""': "Empty string placeholder",
            "[]": "Empty list placeholder",
            "{}": "Empty dict placeholder",
            "()": "Empty tuple placeholder",
            "set()": "Empty set placeholder",
            "None": "None placeholder",
            "pass": "Pass statement placeholder",
            "return": "Return statement placeholder",
            "return None": "Return None placeholder",
            "return 0": "Return zero placeholder",
            "return True": "Return True placeholder",
            "return False": "Return False placeholder",
            "return []": "Return empty list placeholder",
            "return {}": "Return empty dict placeholder",
            "return ''": "Return empty string placeholder",
        }
        
        # Define stub function patterns
        self.stub_patterns = [
            r"def\s+\w+\s*\([^)]*\):\s*pass",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+None",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+0",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+True",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+False",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+\[\]",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+\{\}",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+''",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+\"\"",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+\(\)",
            r"def\s+\w+\s*\([^)]*\):\s*return\s+set\(\)",
        ]
        
        # Define incomplete implementation patterns
        self.incomplete_patterns = [
            r"raise\s+NotImplementedError",
            r"raise\s+NotImplemented",
            r"raise\s+Exception\s*\(\s*['\"]Not implemented['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]TODO['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]FIXME['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]PLACEHOLDER['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]STUB['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]IMPLEMENT['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]FUTURE['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]LATER['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]SOON['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]TEMP['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]REMOVE['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]REVIEW['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]OPTIMIZE['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]REFACTOR['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]CLEANUP['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]BUG['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]WARNING['\"]\s*\)",
            r"raise\s+Exception\s*\(\s*['\"]DEPRECATED['\"]\s*\)",
        ]
        
    def detect_comment_placeholders(self, file_path: str, content: str) -> List[Placeholder]:
        """Detect placeholder comments in file content."""
        placeholders = []
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern in self.comment_patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    content_text = match.group(1).strip()
                    severity = self._get_comment_severity(pattern_name)
                    placeholders.append(Placeholder(
                        file_path=file_path,
                        line=line_num,
                        column=line.find('#') + 1,
                        placeholder_type=f"comment_{pattern_name.lower()}",
                        content=content_text,
                        severity=severity,
                        context=line.strip()
                    ))
        
        return placeholders
    
    def detect_value_placeholders(self, file_path: str, content: str) -> List[Placeholder]:
        """Detect placeholder values in file content."""
        placeholders = []
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            for value, description in self.placeholder_values.items():
                # Look for standalone values or in assignments
                patterns = [
                    rf"\b{re.escape(value)}\b",  # Standalone value
                    rf"=\s*{re.escape(value)}\b",  # Assignment
                    rf"return\s+{re.escape(value)}\b",  # Return statement
                    rf"yield\s+{re.escape(value)}\b",  # Yield statement
                ]
                
                for pattern in patterns:
                    if re.search(pattern, line):
                        # Skip if it's in a comment or string
                        if not self._is_in_comment_or_string(line, line.find(value)):
                            placeholders.append(Placeholder(
                                file_path=file_path,
                                line=line_num,
                                column=line.find(value) + 1,
                                placeholder_type="value_placeholder",
                                content=description,
                                severity="low",
                                context=line.strip()
                            ))
                            break
        
        return placeholders
    
    def detect_stub_functions(self, file_path: str, content: str) -> List[Placeholder]:
        """Detect stub function patterns in file content."""
        placeholders = []
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern in self.stub_patterns:
                if re.search(pattern, line):
                    # Extract function name
                    func_match = re.search(r"def\s+(\w+)", line)
                    func_name = func_match.group(1) if func_match else "unknown"
                    
                    placeholders.append(Placeholder(
                        file_path=file_path,
                        line=line_num,
                        column=line.find('def') + 1,
                        placeholder_type="stub_function",
                        content=f"Stub function: {func_name}",
                        severity="medium",
                        context=line.strip()
                    ))
                    break
        
        return placeholders
    
    def detect_incomplete_implementations(self, file_path: str, content: str) -> List[Placeholder]:
        """Detect incomplete implementation patterns."""
        placeholders = []
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern in self.incomplete_patterns:
                if re.search(pattern, line):
                    placeholders.append(Placeholder(
                        file_path=file_path,
                        line=line_num,
                        column=line.find('raise') + 1,
                        placeholder_type="incomplete_implementation",
                        content="Incomplete implementation",
                        severity="high",
                        context=line.strip()
                    ))
                    break
        
        return placeholders
    
    def detect_ast_placeholders(self, file_path: str, content: str) -> List[Placeholder]:
        """Detect placeholders using AST analysis."""
        placeholders = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for empty function bodies
                    if len(node.body) == 0 or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                        placeholders.append(Placeholder(
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset + 1,
                            placeholder_type="empty_function",
                            content=f"Empty function: {node.name}",
                            severity="medium",
                            context=f"def {node.name}(...):"
                        ))
                    
                    # Check for functions that only return placeholder values
                    elif len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                        return_value = node.body[0].value
                        if isinstance(return_value, ast.Constant):
                            if return_value.value in [None, 0, 0.0, True, False, "", [], {}, ()]:
                                placeholders.append(Placeholder(
                                    file_path=file_path,
                                    line=node.lineno,
                                    column=node.col_offset + 1,
                                    placeholder_type="placeholder_return",
                                    content=f"Function returns placeholder value: {return_value.value}",
                                    severity="low",
                                    context=f"def {node.name}(...):"
                                ))
                
                elif isinstance(node, ast.ClassDef):
                    # Check for empty class bodies
                    if len(node.body) == 0 or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                        placeholders.append(Placeholder(
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset + 1,
                            placeholder_type="empty_class",
                            content=f"Empty class: {node.name}",
                            severity="medium",
                            context=f"class {node.name}:"
                        ))
                
                elif isinstance(node, ast.Assign):
                    # Check for placeholder assignments
                    if isinstance(node.value, ast.Constant):
                        if node.value.value in [None, 0, 0.0, True, False, "", [], {}, ()]:
                            # Get variable name
                            if node.targets and isinstance(node.targets[0], ast.Name):
                                var_name = node.targets[0].id
                                placeholders.append(Placeholder(
                                    file_path=file_path,
                                    line=node.lineno,
                                    column=node.col_offset + 1,
                                    placeholder_type="placeholder_assignment",
                                    content=f"Placeholder assignment: {var_name} = {node.value.value}",
                                    severity="low",
                                    context=f"{var_name} = {node.value.value}"
                                ))
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
        
        return placeholders
    
    def _get_comment_severity(self, pattern_name: str) -> str:
        """Get severity level for comment patterns."""
        high_severity = ["FIXME", "BUG", "HACK", "XXX"]
        medium_severity = ["TODO", "IMPLEMENT", "STUB", "OPTIMIZE", "REFACTOR"]
        low_severity = ["NOTE", "REVIEW", "CLEANUP", "FUTURE", "LATER", "SOON", "TEMP", "REMOVE", "WARNING", "DEPRECATED"]
        
        if pattern_name in high_severity:
            return "high"
        elif pattern_name in medium_severity:
            return "medium"
        else:
            return "low"
    
    def _is_in_comment_or_string(self, line: str, position: int) -> bool:
        """Check if a position in a line is inside a comment or string."""
        # Simple heuristic - if there's a # before the position, it's in a comment
        comment_pos = line.find('#')
        if comment_pos != -1 and comment_pos < position:
            return True
        
        # Check for string literals (simplified)
        quote_chars = ['"', "'"]
        in_string = False
        string_char = None
        
        for i, char in enumerate(line[:position]):
            if char in quote_chars:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
        
        return in_string
    
    def analyze_file(self, file_path: str) -> List[Placeholder]:
        """
        Analyze a single file for placeholders.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of Placeholder objects found in the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        file_placeholders = []
        
        # Detect different types of placeholders
        file_placeholders.extend(self.detect_comment_placeholders(file_path, content))
        file_placeholders.extend(self.detect_value_placeholders(file_path, content))
        file_placeholders.extend(self.detect_stub_functions(file_path, content))
        file_placeholders.extend(self.detect_incomplete_implementations(file_path, content))
        file_placeholders.extend(self.detect_ast_placeholders(file_path, content))
        
        # Store file statistics
        self.file_stats[file_path] = {
            'total_placeholders': len(file_placeholders),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        for placeholder in file_placeholders:
            self.file_stats[file_path]['by_type'][placeholder.placeholder_type] += 1
            self.file_stats[file_path]['by_severity'][placeholder.severity] += 1
        
        return file_placeholders
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze all Python files in a directory for placeholders.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing {len(python_files)} Python files for placeholders...")
        
        # Clear previous results
        self.placeholders.clear()
        self.file_stats.clear()
        self.summary_stats.clear()
        
        total_placeholders = 0
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        files_with_placeholders = 0
        
        for file_path in python_files:
            file_placeholders = self.analyze_file(file_path)
            self.placeholders.extend(file_placeholders)
            
            if file_placeholders:
                files_with_placeholders += 1
                total_placeholders += len(file_placeholders)
                
                for placeholder in file_placeholders:
                    by_type[placeholder.placeholder_type] += 1
                    by_severity[placeholder.severity] += 1
        
        # Generate summary statistics
        self.summary_stats = {
            'total_files_analyzed': len(python_files),
            'files_with_placeholders': files_with_placeholders,
            'total_placeholders': total_placeholders,
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'average_placeholders_per_file': total_placeholders / len(python_files) if python_files else 0,
            'files_with_placeholders_percentage': (files_with_placeholders / len(python_files) * 100) if python_files else 0
        }
        
        return {
            'placeholders': self.placeholders,
            'file_stats': self.file_stats,
            'summary_stats': self.summary_stats
        }
    
    def generate_report(self, output_format: str = "terminal") -> str:
        """
        Generate a report of placeholder analysis results.
        
        Args:
            output_format: Format of the report ("terminal", "json", "html")
            
        Returns:
            Report string in the specified format
        """
        if output_format == "json":
            return self._generate_json_report()
        elif output_format == "html":
            return self._generate_html_report()
        else:
            return self._generate_terminal_report()
    
    def _generate_terminal_report(self) -> str:
        """Generate a terminal-friendly report."""
        if not self.placeholders:
            return "No placeholders found in the analyzed files."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PLACEHOLDER DETECTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total files analyzed: {self.summary_stats['total_files_analyzed']}")
        report_lines.append(f"Files with placeholders: {self.summary_stats['files_with_placeholders']}")
        report_lines.append(f"Total placeholders found: {self.summary_stats['total_placeholders']}")
        report_lines.append(f"Average placeholders per file: {self.summary_stats['average_placeholders_per_file']:.2f}")
        report_lines.append(f"Files with placeholders: {self.summary_stats['files_with_placeholders_percentage']:.1f}%")
        report_lines.append("")
        
        # Breakdown by type
        report_lines.append("PLACEHOLDERS BY TYPE:")
        report_lines.append("-" * 40)
        for placeholder_type, count in sorted(self.summary_stats['by_type'].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"{placeholder_type}: {count}")
        report_lines.append("")
        
        # Breakdown by severity
        report_lines.append("PLACEHOLDERS BY SEVERITY:")
        report_lines.append("-" * 40)
        severity_order = ['high', 'medium', 'low']
        for severity in severity_order:
            if severity in self.summary_stats['by_severity']:
                count = self.summary_stats['by_severity'][severity]
                report_lines.append(f"{severity.upper()}: {count}")
        report_lines.append("")
        
        # Detailed findings by file
        report_lines.append("DETAILED FINDINGS BY FILE:")
        report_lines.append("-" * 40)
        
        # Group placeholders by file
        by_file = defaultdict(list)
        for placeholder in self.placeholders:
            by_file[placeholder.file_path].append(placeholder)
        
        for file_path, file_placeholders in sorted(by_file.items()):
            report_lines.append(f"\n{file_path}:")
            for placeholder in sorted(file_placeholders, key=lambda x: x.line):
                severity_indicator = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(placeholder.severity, '⚪')
                
                report_lines.append(f"  {severity_indicator} Line {placeholder.line}: {placeholder.placeholder_type}")
                report_lines.append(f"     Content: {placeholder.content}")
                report_lines.append(f"     Context: {placeholder.context}")
        
        return "\n".join(report_lines)
    
    def _generate_json_report(self) -> str:
        """Generate a JSON report."""
        return json.dumps({
            'summary_stats': self.summary_stats,
            'file_stats': self.file_stats,
            'placeholders': [p.to_dict() for p in self.placeholders]
        }, indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate an HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Placeholder Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }
                .file-section { margin: 20px 0; }
                .placeholder { margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }
                .high { border-left-color: #ff4444; }
                .medium { border-left-color: #ffaa00; }
                .low { border-left-color: #44aa44; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
                .stat-box { background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Placeholder Detection Report</h1>
                <p>Comprehensive analysis of placeholders, stubs, and incomplete code</p>
            </div>
        """
        
        # Add summary statistics
        html += f"""
            <div class="summary">
                <h2>📊 Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <h3>Files Analyzed</h3>
                        <p>{self.summary_stats['total_files_analyzed']}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Files with Placeholders</h3>
                        <p>{self.summary_stats['files_with_placeholders']}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Total Placeholders</h3>
                        <p>{self.summary_stats['total_placeholders']}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Average per File</h3>
                        <p>{self.summary_stats['average_placeholders_per_file']:.2f}</p>
                    </div>
                </div>
            </div>
        """
        
        # Add detailed findings
        html += "<h2>📁 Detailed Findings by File</h2>"
        
        # Group placeholders by file
        by_file = defaultdict(list)
        for placeholder in self.placeholders:
            by_file[placeholder.file_path].append(placeholder)
        
        for file_path, file_placeholders in sorted(by_file.items()):
            html += f"""
                <div class="file-section">
                    <h3>📄 {file_path}</h3>
                    <p>Found {len(file_placeholders)} placeholders</p>
            """
            
            for placeholder in sorted(file_placeholders, key=lambda x: x.line):
                html += f"""
                    <div class="placeholder {placeholder.severity}">
                        <strong>Line {placeholder.line}:</strong> {placeholder.placeholder_type}<br>
                        <strong>Content:</strong> {placeholder.content}<br>
                        <strong>Context:</strong> <code>{placeholder.context}</code><br>
                        <strong>Severity:</strong> {placeholder.severity.upper()}
                    </div>
                """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, output_path: str, output_format: str = "terminal"):
        """Save the report to a file."""
        report_content = self.generate_report(output_format)
        
        # Determine file extension
        if output_format == "json":
            extension = ".json"
        elif output_format == "html":
            extension = ".html"
        else:
            extension = ".txt"
        
        # Create output path
        if not output_path.endswith(extension):
            output_path += extension
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Report saved to: {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect placeholders in Python code")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["terminal", "json", "html"], 
                       default="terminal", help="Output format")
    parser.add_argument("--config", "-c", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = CodeQualityConfig()
        # Load config from file (simplified)
    
    # Create detector and analyze
    detector = PlaceholderDetector(config)
    results = detector.analyze_directory(args.directory)
    
    # Generate and save report
    if args.output:
        detector.save_report(args.output, args.format)
    else:
        print(detector.generate_report(args.format))


if __name__ == "__main__":
    main()