from src.utils.tprint import tprint

from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

"""
Enhanced Dead Code Analyzer

Integrates multiple dead code detection tools for comprehensive analysis:
- DeadCodeRemover for advanced dead code detection
- PyCG for accurate call graph generation
- NetworkX for dependency graph analysis
- Enhanced AST analysis for better accuracy

This analyzer provides superior dead code detection with reduced false positives
and better understanding of code interactions.
"""

import ast
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging
import subprocess
import tempfile
import shutil

# Try to import optional dependencies
try:
    import pycg
    PYCG_AVAILABLE = True
except ImportError:
    PYCG_AVAILABLE = False
    tprint("Warning: PyCG not available. Install with: pip install pycg")
    
    # Create a mock pycg module to prevent import errors
    class MockPyCG:
        def __init__(self, *args, **kwargs):
            pass
        
        def analyze(self, *args, **kwargs):
            return {}
    
    pycg = MockPyCG()

try:
    import deadcode
    DEADCODE_AVAILABLE = True
except ImportError:
    DEADCODE_AVAILABLE = False
    tprint("Warning: DeadCodeRemover not available. Install with: pip install deadcode")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    tprint("Warning: NetworkX not available. Install with: pip install networkx")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    tprint("Warning: Matplotlib not available. Install with: pip install matplotlib")

from core.config import AnalysisConfig, CodeQualityConfig
from utils.file_utils import find_python_files
from typing import Set
from typing import Any

from typing import Dict
from typing import List


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
    call_graph: Any = field(default_factory=lambda: nx.DiGraph() if NETWORKX_AVAILABLE else None)
    dependency_graph: Any = field(default_factory=lambda: nx.DiGraph() if NETWORKX_AVAILABLE else None)
    false_positives_filtered: int = 0
    impact_analysis: Dict[str, Any] = field(default_factory=dict)


class EnhancedDeadCodeAnalyzer:
    """
    Enhanced dead code analyzer using multiple tools and techniques.
    
    Integrates:
    - DeadCodeRemover for advanced dead code detection
    - PyCG for accurate call graph generation
    - NetworkX for dependency analysis
    - Enhanced AST analysis for better accuracy
    """

    def __init__(self, config: AnalysisConfig | None = None):
        """Initialize the enhanced dead code analyzer."""
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Tool availability flags
        self.pycg_available = PYCG_AVAILABLE
        self.deadcode_available = DEADCODE_AVAILABLE
        
        # Analysis results storage
        if NETWORKX_AVAILABLE:
            self.call_graph = nx.DiGraph()
            self.dependency_graph = nx.DiGraph()
        else:
            self.call_graph = None
            self.dependency_graph = None
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
        
        # Phase 2: Run multiple dead code detection tools
        self.logger.info("Phase 2: Running dead code detection tools...")
        all_issues = []
        
        # Run DeadCodeRemover if available
        if self.deadcode_available:
            deadcode_issues = self._run_deadcode_analysis(python_files)
            all_issues.extend(deadcode_issues)
            self.logger.info(f"DeadCodeRemover found {len(deadcode_issues)} issues")
        
        # Run PyCG-based analysis if available
        if self.pycg_available:
            pycg_issues = self._run_pycg_analysis(python_files)
            all_issues.extend(pycg_issues)
            self.logger.info(f"PyCG analysis found {len(pycg_issues)} issues")
        
        # Run enhanced AST analysis
        ast_issues = self._run_enhanced_ast_analysis(python_files)
        all_issues.extend(ast_issues)
        self.logger.info(f"Enhanced AST analysis found {len(ast_issues)} issues")
        
        # Phase 3: Cross-validate and filter false positives
        self.logger.info("Phase 3: Cross-validating results...")
        validated_issues = self._cross_validate_issues(all_issues)
        
        # Phase 4: Generate comprehensive report
        self.logger.info("Phase 4: Generating report...")
        report = self._generate_enhanced_report(validated_issues)
        
        self.logger.info(f"Analysis complete. Found {report.total_issues} total issues")
        return report

    def _build_comprehensive_call_graph(self, python_files: List[Path]) -> None:
        """Build comprehensive call graph using multiple techniques."""
        self.logger.info("Building comprehensive call graph...")
        
        # Method 1: PyCG-based call graph (if available)
        if self.pycg_available:
            self._build_pycg_call_graph(python_files)
        
        # Method 2: Enhanced AST-based call graph
        self._build_ast_call_graph(python_files)
        
        # Method 3: Import-based dependency graph
        self._build_import_dependency_graph(python_files)
        
        self.logger.info(f"Call graph built: {self.call_graph.number_of_nodes()} nodes, {self.call_graph.number_of_edges()} edges")
        self.logger.info(f"Dependency graph built: {self.dependency_graph.number_of_nodes()} nodes, {self.dependency_graph.number_of_edges()} edges")

    def _build_pycg_call_graph(self, python_files: List[Path]) -> None:
        """Build call graph using PyCG."""
        if not self.pycg_available:
            return
            
        try:
            # Create temporary directory for PyCG analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Copy Python files to temp directory
                for py_file in python_files:
                    rel_path = py_file.relative_to(py_file.parents[len(py_file.parts) - len(Path.cwd().parts)])
                    temp_file = temp_dir_path / rel_path
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(py_file, temp_file)
                
                # Run PyCG analysis
                result = subprocess.run([
                    'python', '-m', 'pycg', '--package', str(temp_dir_path), 
                    '--output', str(temp_dir_path / 'call_graph.json')
                ], capture_output=True, text=True, cwd=temp_dir_path)
                
                if result.returncode == 0:
                    # Parse PyCG results
                    call_graph_file = temp_dir_path / 'call_graph.json'
                    if call_graph_file.exists():
                        with open(call_graph_file, 'r') as f:
                            pycg_data = json.load(f)
                        self._parse_pycg_results(pycg_data, python_files)
                        
        except Exception as e:
            self.logger.warning(f"PyCG analysis failed: {e}")

    def _parse_pycg_results(self, pycg_data: Dict, python_files: List[Path]) -> None:
        """Parse PyCG results and add to call graph."""
        for caller, callees in pycg_data.items():
            # Add caller node
            if caller not in self.call_graph:
                self.call_graph.add_node(caller, node_type="function")
            
            # Add callee nodes and edges
            for callee in callees:
                if callee not in self.call_graph:
                    self.call_graph.add_node(callee, node_type="function")
                self.call_graph.add_edge(caller, callee)

    def _build_ast_call_graph(self, python_files: List[Path]) -> None:
        """Build call graph using enhanced AST analysis."""
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                self._analyze_ast_for_call_graph(tree, file_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path} for call graph: {e}")

    def _analyze_ast_for_call_graph(self, tree: ast.AST, file_path: Path) -> None:
        """Analyze AST to build call graph."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = f"{file_path}::{node.name}"
                self.call_graph.add_node(func_name, 
                                       node_type="function",
                                       file_path=str(file_path),
                                       line_number=node.lineno)
                
                # Find function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            callee_name = f"{file_path}::{child.func.id}"
                            self.call_graph.add_node(callee_name, node_type="function")
                            self.call_graph.add_edge(func_name, callee_name)
                        elif isinstance(child.func, ast.Attribute):
                            callee_name = f"{file_path}::{child.func.attr}"
                            self.call_graph.add_node(callee_name, node_type="method")
                            self.call_graph.add_edge(func_name, callee_name)
                            
            elif isinstance(node, ast.ClassDef):
                class_name = f"{file_path}::{node.name}"
                self.call_graph.add_node(class_name,
                                       node_type="class",
                                       file_path=str(file_path),
                                       line_number=node.lineno)

    def _build_import_dependency_graph(self, python_files: List[Path]) -> None:
        """Build dependency graph based on imports."""
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                self._analyze_imports_for_dependency_graph(tree, file_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze imports in {file_path}: {e}")

    def _analyze_imports_for_dependency_graph(self, tree: ast.AST, file_path: Path) -> None:
        """Analyze imports to build dependency graph."""
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        module_name = file_path.stem
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep_name = alias.name.split('.')[0]
                        self.dependency_graph.add_edge(module_name, dep_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dep_name = node.module.split('.')[0]
                        self.dependency_graph.add_edge(module_name, dep_name)

    def _run_deadcode_analysis(self, python_files: List[Path]) -> List[EnhancedDeadCodeIssue]:
        """Run DeadCodeRemover analysis."""
        if not self.deadcode_available:
            return []
            
        issues = []
        try:
            # Run deadcode on each file
            for file_path in python_files:
                result = subprocess.run([
                    'python', '-m', 'deadcode', str(file_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Parse deadcode output
                    deadcode_issues = self._parse_deadcode_output(result.stdout, file_path)
                    issues.extend(deadcode_issues)
                    
        except Exception as e:
            self.logger.warning(f"DeadCodeRemover analysis failed: {e}")
            
        return issues

    def _parse_deadcode_output(self, output: str, file_path: Path) -> List[EnhancedDeadCodeIssue]:
        """Parse DeadCodeRemover output."""
        issues = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if ':' in line and 'unused' in line.lower():
                try:
                    # Parse line format: "file:line: message"
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        line_num = int(parts[1])
                        message = parts[2].strip()
                        
                        issue = EnhancedDeadCodeIssue(
                            file_path=str(file_path),
                            line_number=line_num,
                            issue_type="dead_code",
                            description=message,
                            confidence=90.0,  # DeadCodeRemover is quite accurate
                            code_snippet="",
                            severity="medium",
                            tool_source="DeadCodeRemover"
                        )
                        issues.append(issue)
                        
                except (ValueError, IndexError):
                    continue
                    
        return issues

    def _run_pycg_analysis(self, python_files: List[Path]) -> List[EnhancedDeadCodeIssue]:
        """Run PyCG-based dead code analysis."""
        if not self.pycg_available:
            return []
            
        issues = []
        
        # Analyze call graph for unused functions
        defined_functions = set()
        called_functions = set()
        
        for node in self.call_graph.nodes():
            if self.call_graph.nodes[node].get('node_type') == 'function':
                defined_functions.add(node)
                
        for edge in self.call_graph.edges():
            called_functions.add(edge[1])
        
        # Find unused functions
        unused_functions = defined_functions - called_functions
        
        for func_name in unused_functions:
            node_data = self.call_graph.nodes[func_name]
            if 'file_path' in node_data and 'line_number' in node_data:
                issue = EnhancedDeadCodeIssue(
                    file_path=node_data['file_path'],
                    line_number=node_data['line_number'],
                    issue_type="dead_code",
                    description=f"Function '{func_name.split('::')[-1]}' is defined but never called",
                    confidence=85.0,
                    code_snippet="",
                    severity="medium",
                    tool_source="PyCG",
                    call_graph_context={"defined": True, "called": False}
                )
                issues.append(issue)
                
        return issues

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
        # Read the file content to get lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
        except Exception:
            lines = []
        
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
        
        # Find unreachable code after return statements
        unreachable_issues = self._find_unreachable_code(tree, file_path, lines)
        issues.extend(unreachable_issues)
        
        return issues

    def _find_unreachable_code(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[EnhancedDeadCodeIssue]:
        """Find unreachable code after return statements."""
        issues = []
        
        class UnreachableCodeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.lines = lines
                self.file_path = file_path
            
            def _extract_code_snippet_simple(self, lines: List[str], line_number: int, context: int = 2) -> str:
                """Extract code snippet around a line number."""
                try:
                    start = max(0, line_number - context - 1)
                    end = min(len(lines), line_number + context)
                    snippet_lines = lines[start:end]
                    return '\\n'.join(snippet_lines)
                except:
                    return f"Line {line_number}"
            
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._check_unreachable_in_function(node)
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._check_unreachable_in_function(node)
                self.generic_visit(node)
            
            def _check_unreachable_in_function(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                """Check for unreachable code in a function."""
                for i, stmt in enumerate(func_node.body):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        # Check if there are statements after this return/raise
                        remaining_statements = func_node.body[i + 1:]
                        for remaining_stmt in remaining_statements:
                            # Skip docstrings and comments
                            if isinstance(remaining_stmt, ast.Expr) and isinstance(remaining_stmt.value, ast.Constant):
                                if isinstance(remaining_stmt.value.value, str):
                                    continue  # Skip docstrings
                            
                            # Found unreachable code
                            issue = EnhancedDeadCodeIssue(
                                file_path=str(self.file_path),
                                line_number=remaining_stmt.lineno,
                                issue_type="unreachable_code",
                                description=f"Unreachable code after {type(stmt).__name__.lower()} statement",
                                confidence=95.0,
                                code_snippet=self._extract_code_snippet_simple(self.lines, remaining_stmt.lineno),
                                severity="high",
                                tool_source="Enhanced AST"
                            )
                            self.issues.append(issue)
        
        visitor = UnreachableCodeVisitor()
        visitor.visit(tree)
        return visitor.issues

    def _is_likely_used_function(self, func_name: str, func_node: ast.AST, lines: List[str], file_path: str) -> bool:
        """Check if a function is likely to be used based on various heuristics."""
        # Skip private functions (except __init__, __call__, etc.)
        if func_name.startswith('_') and not func_name.startswith('__'):
            return True
            
        # Skip special methods
        if func_name.startswith('__') and func_name.endswith('__'):
            return True
            
        # Skip functions in test files (but not files that just happen to contain 'test' in the name)
        if (file_path.lower().endswith('_test.py') or file_path.lower().endswith('test_.py') or
            file_path.lower().endswith('_tests.py') or
            '/tests/' in file_path.lower() or
            file_path.lower().endswith('/test.py') or
            file_path.lower().endswith('/test_runner.py')):
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
        
        # Check if function is in call graph and actually called (not just defined)
        func_key = f"{file_path}::{func_name}"
        if func_key in self.call_graph:
            # Check if the function has any callers (is actually called)
            if self.call_graph.has_node(func_key):
                # Get the predecessors (callers) of this function
                callers = list(self.call_graph.predecessors(func_key))
                if callers:  # Only return True if the function is actually called
                    return True
            
        return False

    def _cross_validate_issues(self, all_issues: List[EnhancedDeadCodeIssue]) -> List[EnhancedDeadCodeIssue]:
        """Cross-validate issues from different tools to reduce false positives."""
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
        # Check if function is in call graph and actually called
        if issue.issue_type == "dead_code":
            func_name = issue.description.split("'")[1] if "'" in issue.description else ""
            if func_name:
                func_key = f"{issue.file_path}::{func_name}"
                if func_key in self.call_graph:
                    # Check if the function has any callers (is actually called)
                    callers = list(self.call_graph.predecessors(func_key))
                    if callers:  # Function is actually called
                        return False  # Function is in call graph and called, likely used
        
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
            call_graph=self.call_graph,
            dependency_graph=self.dependency_graph,
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

    def generate_visualization(self, report: EnhancedDeadCodeReport, output_dir: Path) -> None:
        """Generate visualizations of the analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate call graph visualization
        if self.call_graph.number_of_nodes() > 0:
            self._visualize_call_graph(output_dir / "call_graph.png")
        
        # Generate dependency graph visualization
        if self.dependency_graph.number_of_nodes() > 0:
            self._visualize_dependency_graph(output_dir / "dependency_graph.png")
        
        # Generate issue distribution charts
        self._visualize_issue_distribution(report, output_dir)

    def _visualize_call_graph(self, output_path: Path) -> None:
        """Visualize the call graph."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping call graph visualization")
            return
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.call_graph, k=1, iterations=50)
            nx.draw(self.call_graph, pos, with_labels=True, node_size=100, font_size=8)
            plt.title("Call Graph")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to generate call graph visualization: {e}")

    def _visualize_dependency_graph(self, output_path: Path) -> None:
        """Visualize the dependency graph."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping dependency graph visualization")
            return
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.dependency_graph, k=1, iterations=50)
            nx.draw(self.dependency_graph, pos, with_labels=True, node_size=100, font_size=8)
            plt.title("Dependency Graph")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to generate dependency graph visualization: {e}")

    def _visualize_issue_distribution(self, report: EnhancedDeadCodeReport, output_dir: Path) -> None:
        """Generate issue distribution visualizations."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping issue distribution visualization")
            return
        try:
            # Issues by type
            plt.figure(figsize=(10, 6))
            types = list(report.issues_by_type.keys())
            counts = list(report.issues_by_type.values())
            plt.bar(types, counts)
            plt.title("Issues by Type")
            plt.xlabel("Issue Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "issues_by_type.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Issues by tool
            plt.figure(figsize=(10, 6))
            tools = list(report.issues_by_tool.keys())
            counts = list(report.issues_by_tool.values())
            plt.bar(tools, [len(c) for c in counts])
            plt.title("Issues by Tool")
            plt.xlabel("Tool")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "issues_by_tool.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate issue distribution visualizations: {e}")

    def export_results(self, report: EnhancedDeadCodeReport, output_path: Path) -> None:
        """Export analysis results to JSON."""
        try:
            # Convert NetworkX graphs to serializable format
            export_data = {
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_severity": report.issues_by_severity,
                "issues_by_tool": report.issues_by_tool,
                "confidence_distribution": report.confidence_distribution,
                "potential_savings": report.potential_savings,
                "false_positives_filtered": report.false_positives_filtered,
                "impact_analysis": report.impact_analysis,
                "call_graph_nodes": list(self.call_graph.nodes(data=True)),
                "call_graph_edges": list(self.call_graph.edges()),
                "dependency_graph_nodes": list(self.dependency_graph.nodes(data=True)),
                "dependency_graph_edges": list(self.dependency_graph.edges()),
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

    def analyze_dead_code(self, directory: str | Path) -> Dict[str, Any]:
        """
        Analyze dead code in a directory (compatibility method for pipeline).
        
        Args:
            directory: Path to directory to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            report = self.analyze_directory(directory)
            
            # Convert report to dictionary format expected by pipeline
            issues = []
            for file_path, file_issues in report.issues_by_file.items():
                for issue in file_issues:
                    issues.append({
                        "file": file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "severity": issue.severity,
                        "code_snippet": issue.code_snippet,
                        "tool_source": issue.tool_source
                    })
            
            return {
                "issues": issues,
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_severity": {k: len(v) for k, v in report.issues_by_severity.items()},
                "issues_by_tool": {k: len(v) for k, v in report.issues_by_tool.items()},
                "confidence_distribution": report.confidence_distribution,
                "potential_savings": report.potential_savings,
                "false_positives_filtered": report.false_positives_filtered
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze dead code: {e}")
            return {"issues": [], "error": str(e)}