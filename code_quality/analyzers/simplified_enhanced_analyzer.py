"""
Simplified Enhanced Dead Code Analyzer

A simplified version that demonstrates the enhanced capabilities without
requiring additional dependencies. This version focuses on the core
improvements while using only standard library modules.
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from core.config import AnalysisConfig
from utils.file_utils import find_python_files


@dataclass
class EnhancedDeadCodeIssue:
    """Enhanced container for dead code analysis results."""
    file_path: str
    line_number: int
    issue_type: str  # "dead_code", "unreachable_code", "unused_import", "unused_dependency"
    description: str
    confidence: float
    code_snippet: str
    severity: str
    removal_impact: str = "low"
    dependencies: List[str] = field(default_factory=list)
    is_bug: bool = False
    tool_source: str = ""  # Which tool detected this issue
    call_graph_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallGraphNode:
    """Represents a node in the call graph."""
    name: str
    file_path: str
    line_number: int
    node_type: str  # "function", "class", "method", "module"
    is_defined: bool = True
    is_called: bool = False
    callers: Set[str] = field(default_factory=set)
    callees: Set[str] = field(default_factory=set)


@dataclass
class EnhancedDeadCodeReport:
    """Enhanced container for dead code analysis report."""
    total_issues: int
    issues_by_type: Dict[str, int]
    issues_by_file: Dict[str, List[EnhancedDeadCodeIssue]]
    issues_by_severity: Dict[str, List[EnhancedDeadCodeIssue]]
    issues_by_tool: Dict[str, List[EnhancedDeadCodeIssue]]
    confidence_distribution: Dict[str, int]
    potential_savings: Dict[str, int]
    call_graph_nodes: Dict[str, CallGraphNode] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    false_positives_filtered: int = 0
    impact_analysis: Dict[str, Any] = field(default_factory=dict)


class SimplifiedEnhancedDeadCodeAnalyzer:
    """
    Simplified enhanced dead code analyzer using only standard library modules.
    
    Demonstrates enhanced capabilities:
    - Improved AST analysis with better accuracy
    - Call graph building for dependency analysis
    - Cross-validation to reduce false positives
    - Enhanced reporting with tool attribution
    """

    def __init__(self, config: AnalysisConfig | None = None):
        """Initialize the simplified enhanced dead code analyzer."""
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis results storage
        self.call_graph_nodes = {}
        self.dependency_graph = defaultdict(set)
        self.all_issues = []
        
        # Configuration
        self.confidence_threshold = getattr(self.config, "confidence_threshold", 80.0)
        self.ignore_patterns = getattr(self.config, "ignore_patterns", [])
        self.whitelist = getattr(self.config, "whitelist", [])

    def analyze_directory(self, directory: str | Path) -> EnhancedDeadCodeReport:
        """
        Analyze all Python files in a directory for dead code.
        
        Args:
            directory: Path to directory
            
        Returns:
            EnhancedDeadCodeReport with comprehensive analysis
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        self.logger.info(f"Starting enhanced dead code analysis of {directory}")
        
        # Find all Python files
        python_files = find_python_files(directory)
        self.logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Phase 1: Build comprehensive call graph
        self.logger.info("Phase 1: Building call graph...")
        self._build_comprehensive_call_graph(python_files)
        
        # Phase 2: Run enhanced AST analysis
        self.logger.info("Phase 2: Running enhanced AST analysis...")
        ast_issues = self._run_enhanced_ast_analysis(python_files)
        self.logger.info(f"Enhanced AST analysis found {len(ast_issues)} issues")
        
        # Phase 3: Run import analysis
        self.logger.info("Phase 3: Running import analysis...")
        import_issues = self._run_import_analysis(python_files)
        self.logger.info(f"Import analysis found {len(import_issues)} issues")
        
        # Phase 4: Cross-validate and filter false positives
        self.logger.info("Phase 4: Cross-validating results...")
        all_issues = ast_issues + import_issues
        validated_issues = self._cross_validate_issues(all_issues)
        
        # Phase 5: Generate comprehensive report
        self.logger.info("Phase 5: Generating report...")
        report = self._generate_enhanced_report(validated_issues)
        
        self.logger.info(f"Analysis complete. Found {report.total_issues} total issues")
        return report

    def _build_comprehensive_call_graph(self, python_files: List[Path]) -> None:
        """Build comprehensive call graph using enhanced AST analysis."""
        self.logger.info("Building comprehensive call graph...")
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                self._analyze_ast_for_call_graph(tree, file_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path} for call graph: {e}")
        
        self.logger.info(f"Call graph built: {len(self.call_graph_nodes)} nodes")
        self.logger.info(f"Dependency graph built: {len(self.dependency_graph)} modules")

    def _analyze_ast_for_call_graph(self, tree: ast.AST, file_path: Path) -> None:
        """Analyze AST to build call graph."""
        module_name = file_path.stem
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = f"{module_name}::{node.name}"
                node_obj = CallGraphNode(
                    name=func_name,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    node_type="function"
                )
                self.call_graph_nodes[func_name] = node_obj
                
                # Find function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            callee_name = f"{module_name}::{child.func.id}"
                            node_obj.callees.add(callee_name)
                            # Mark callee as called
                            if callee_name in self.call_graph_nodes:
                                self.call_graph_nodes[callee_name].is_called = True
                                self.call_graph_nodes[callee_name].callers.add(func_name)
                        elif isinstance(child.func, ast.Attribute):
                            callee_name = f"{module_name}::{child.func.attr}"
                            node_obj.callees.add(callee_name)
                            
            elif isinstance(node, ast.ClassDef):
                class_name = f"{module_name}::{node.name}"
                node_obj = CallGraphNode(
                    name=class_name,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    node_type="class"
                )
                self.call_graph_nodes[class_name] = node_obj
                
                # Find method calls within this class
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            callee_name = f"{module_name}::{child.func.id}"
                            node_obj.callees.add(callee_name)
                        elif isinstance(child.func, ast.Attribute):
                            callee_name = f"{module_name}::{child.func.attr}"
                            node_obj.callees.add(callee_name)

    def _run_enhanced_ast_analysis(self, python_files: List[Path]) -> List[EnhancedDeadCodeIssue]:
        """Run enhanced AST-based dead code analysis."""
        issues = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                file_issues = self._analyze_file_ast(tree, file_path)
                issues.extend(file_issues)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
                
        return issues

    def _analyze_file_ast(self, tree: ast.AST, file_path: Path) -> List[EnhancedDeadCodeIssue]:
        """Analyze single file AST for dead code."""
        issues = []
        lines = source.split('\n') if hasattr(tree, 'source') else []
        
        # Track function definitions and calls
        defined_functions = {}
        called_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_functions[node.name] = node
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_functions.add(node.func.attr)
        
        # Find unused functions
        for func_name, func_node in defined_functions.items():
            if func_name not in called_functions:
                # Check if it's a special method or likely to be used
                if not self._is_likely_used_function(func_name, func_node, lines, str(file_path)):
                    issue = EnhancedDeadCodeIssue(
                        file_path=str(file_path),
                        line_number=func_node.lineno,
                        issue_type="dead_code",
                        description=f"Function '{func_name}' is defined but never called",
                        confidence=80.0,
                        code_snippet=self._extract_code_snippet(lines, func_node.lineno),
                        severity="medium",
                        tool_source="Enhanced AST"
                    )
                    issues.append(issue)
        
        return issues

    def _run_import_analysis(self, python_files: List[Path]) -> List[EnhancedDeadCodeIssue]:
        """Run import analysis to find unused imports."""
        issues = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                file_issues = self._analyze_imports(tree, file_path)
                issues.extend(file_issues)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze imports in {file_path}: {e}")
                
        return issues

    def _analyze_imports(self, tree: ast.AST, file_path: Path) -> List[EnhancedDeadCodeIssue]:
        """Analyze imports for unused imports."""
        issues = []
        lines = source.split('\n') if hasattr(tree, 'source') else []
        
        # Track imports and their usage
        imports = {}
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = node
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = node
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Find unused imports
        for name, import_node in imports.items():
            if name not in used_names and not self._is_likely_used_import(name, import_node):
                issue = EnhancedDeadCodeIssue(
                    file_path=str(file_path),
                    line_number=import_node.lineno,
                    issue_type="unused_import",
                    description=f"Import '{name}' is unused",
                    confidence=90.0,
                    code_snippet=self._extract_code_snippet(lines, import_node.lineno),
                    severity="low",
                    tool_source="Import Analysis"
                )
                issues.append(issue)
        
        return issues

    def _is_likely_used_function(self, func_name: str, func_node: ast.AST, lines: List[str], file_path: str) -> bool:
        """Check if a function is likely to be used based on various heuristics."""
        # Skip private functions (except __init__, __call__, etc.)
        if func_name.startswith('_') and not func_name.startswith('__'):
            return True
            
        # Skip special methods
        if func_name.startswith('__') and func_name.endswith('__'):
            return True
            
        # Skip functions in test files
        if 'test' in file_path.lower() or 'tests' in file_path.lower():
            return True
            
        # Skip functions in __init__.py files (likely exports)
        if file_path.endswith('__init__.py'):
            return True
            
        # Check for decorators that indicate usage
        if hasattr(func_node, 'decorator_list') and func_node.decorator_list:
            for decorator in func_node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorator_name = decorator.id.lower()
                    if any(keyword in decorator_name for keyword in ['app', 'route', 'handler', 'callback', 'listener']):
                        return True
        
        # Check if function is in call graph
        module_name = Path(file_path).stem
        func_key = f"{module_name}::{func_name}"
        if func_key in self.call_graph_nodes:
            return True
            
        return False

    def _is_likely_used_import(self, name: str, import_node: ast.AST) -> bool:
        """Check if an import is likely to be used."""
        # Skip common imports that might be used dynamically
        common_imports = ['os', 'sys', 'json', 'logging', 'datetime', 'pathlib']
        if name in common_imports:
            return True
            
        # Skip imports that might be used in string contexts
        if isinstance(import_node, ast.ImportFrom):
            if import_node.module and any(keyword in import_node.module.lower() 
                                        for keyword in ['typing', 'abc', 'enum']):
                return True
                
        return False

    def _cross_validate_issues(self, all_issues: List[EnhancedDeadCodeIssue]) -> List[EnhancedDeadCodeIssue]:
        """Cross-validate issues to reduce false positives."""
        validated_issues = []
        false_positives_filtered = 0
        
        # Group issues by file and line
        issues_by_location = defaultdict(list)
        for issue in all_issues:
            key = (issue.file_path, issue.line_number)
            issues_by_location[key].append(issue)
        
        # Validate each group
        for location, issues in issues_by_location.items():
            if len(issues) >= 2:  # Multiple tools agree
                # Take the issue with highest confidence
                best_issue = max(issues, key=lambda x: x.confidence)
                validated_issues.append(best_issue)
            elif len(issues) == 1:
                issue = issues[0]
                # Additional validation for single-tool issues
                if self._validate_single_tool_issue(issue):
                    validated_issues.append(issue)
                else:
                    false_positives_filtered += 1
        
        self.logger.info(f"Cross-validation complete. Filtered {false_positives_filtered} false positives")
        return validated_issues

    def _validate_single_tool_issue(self, issue: EnhancedDeadCodeIssue) -> bool:
        """Validate a single-tool issue to reduce false positives."""
        # Check if function is in call graph
        if issue.issue_type == "dead_code":
            func_name = issue.description.split("'")[1] if "'" in issue.description else ""
            if func_name:
                module_name = Path(issue.file_path).stem
                func_key = f"{module_name}::{func_name}"
                if func_key in self.call_graph_nodes:
                    return False  # Function is in call graph, likely used
        
        # Check for dynamic usage patterns
        if self._check_dynamic_usage(issue):
            return False  # Function is used dynamically
        
        return True

    def _check_dynamic_usage(self, issue: EnhancedDeadCodeIssue) -> bool:
        """Check if a function is used dynamically."""
        if issue.issue_type != "dead_code":
            return False
            
        func_name = issue.description.split("'")[1] if "'" in issue.description else ""
        if not func_name:
            return False
        
        # Search for string references
        file_path = Path(issue.file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for string references
            string_patterns = [f'"{func_name}"', f"'{func_name}'"]
            for pattern in string_patterns:
                if pattern in content:
                    return True
                    
        except Exception:
            pass
            
        return False

    def _extract_code_snippet(self, lines: List[str], line_number: int, context: int = 3) -> str:
        """Extract code snippet around a line number."""
        if not lines or line_number <= 0:
            return ""
        
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        
        snippet_lines = []
        for i in range(start, end):
            marker = ">>> " if i == line_number - 1 else "    "
            snippet_lines.append(f"{marker}{i+1:4d}: {lines[i]}")
        
        return "\n".join(snippet_lines)

    def _generate_enhanced_report(self, issues: List[EnhancedDeadCodeIssue]) -> EnhancedDeadCodeReport:
        """Generate comprehensive dead code analysis report."""
        # Group issues by type
        issues_by_type = defaultdict(int)
        for issue in issues:
            issues_by_type[issue.issue_type] += 1
        
        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.file_path].append(issue)
        
        # Group issues by severity
        issues_by_severity = defaultdict(list)
        for issue in issues:
            issues_by_severity[issue.severity].append(issue)
        
        # Group issues by tool
        issues_by_tool = defaultdict(list)
        for issue in issues:
            issues_by_tool[issue.tool_source].append(issue)
        
        # Calculate confidence distribution
        confidence_distribution = defaultdict(int)
        for issue in issues:
            if issue.confidence >= 95:
                confidence_distribution["high"] += 1
            elif issue.confidence >= 80:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
        
        # Calculate potential savings
        potential_savings = {
            "total_lines": len(issues),
            "high_confidence": len([i for i in issues if i.confidence >= 95]),
            "medium_confidence": len([i for i in issues if 80 <= i.confidence < 95]),
            "low_confidence": len([i for i in issues if i.confidence < 80])
        }
        
        return EnhancedDeadCodeReport(
            total_issues=len(issues),
            issues_by_type=dict(issues_by_type),
            issues_by_file=dict(issues_by_file),
            issues_by_severity=dict(issues_by_severity),
            issues_by_tool=dict(issues_by_tool),
            confidence_distribution=dict(confidence_distribution),
            potential_savings=potential_savings,
            call_graph_nodes=self.call_graph_nodes,
            dependency_graph=dict(self.dependency_graph),
            false_positives_filtered=0,  # Will be set during cross-validation
            impact_analysis=self._analyze_removal_impact(issues)
        )

    def _analyze_removal_impact(self, issues: List[EnhancedDeadCodeIssue]) -> Dict[str, Any]:
        """Analyze the impact of removing dead code."""
        impact_analysis = {
            "high_impact": [],
            "medium_impact": [],
            "low_impact": [],
            "estimated_time_savings": {
                "estimated_hours_saved": len(issues) * 0.1,  # Rough estimate
                "estimated_days_saved": len(issues) * 0.1 / 8,
            },
            "complexity_reduction": len([i for i in issues if i.issue_type in ["dead_code", "unused_function"]]),
        }
        
        # Categorize issues by impact
        for issue in issues:
            if issue.confidence >= 95 and issue.severity == "high":
                impact_analysis["high_impact"].append(issue)
            elif issue.confidence >= 80:
                impact_analysis["medium_impact"].append(issue)
            else:
                impact_analysis["low_impact"].append(issue)
        
        return impact_analysis

    def export_results(self, report: EnhancedDeadCodeReport, output_path: Path) -> None:
        """Export analysis results to JSON."""
        try:
            # Convert call graph nodes to serializable format
            call_graph_data = {}
            for name, node in self.call_graph_nodes.items():
                call_graph_data[name] = {
                    "name": node.name,
                    "file_path": node.file_path,
                    "line_number": node.line_number,
                    "node_type": node.node_type,
                    "is_defined": node.is_defined,
                    "is_called": node.is_called,
                    "callers": list(node.callers),
                    "callees": list(node.callees)
                }
            
            export_data = {
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_severity": report.issues_by_severity,
                "issues_by_tool": report.issues_by_tool,
                "confidence_distribution": report.confidence_distribution,
                "potential_savings": report.potential_savings,
                "false_positives_filtered": report.false_positives_filtered,
                "impact_analysis": report.impact_analysis,
                "call_graph_nodes": call_graph_data,
                "dependency_graph": report.dependency_graph,
                "issues": [
                    {
                        "file_path": issue.file_path,
                        "line_number": issue.line_number,
                        "issue_type": issue.issue_type,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "severity": issue.severity,
                        "tool_source": issue.tool_source,
                        "code_snippet": issue.code_snippet
                    }
                    for file_path in report.issues_by_file
                    for issue in report.issues_by_file[file_path]
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")