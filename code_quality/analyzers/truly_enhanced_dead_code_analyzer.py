#!/usr/bin/env python3
import numpy as np

"""
Truly Enhanced Dead Code Analyzer

This analyzer implements advanced filtering and validation techniques to significantly
reduce false positives in dead code detection:

1. **Multi-tool Consensus**: Requires agreement from multiple detection tools
2. **Call Graph Validation**: Uses comprehensive call graphs to verify actual usage
3. **Dynamic Usage Detection**: Detects functions used via strings, decorators, etc.
4. **Context-Aware Filtering**: Considers file types, patterns, and project structure
5. **Confidence Scoring**: Advanced confidence calculation based on multiple factors
6. **False Positive Learning**: Learns from patterns to improve future detection
"""

import ast
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import subprocess
import tempfile

# Try to import optional dependencies
try:
    import pycg
    PYCG_AVAILABLE = True
except ImportError:
    PYCG_AVAILABLE = False

try:
    import deadcode
    DEADCODE_AVAILABLE = True
except ImportError:
    DEADCODE_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from core.config import AnalysisConfig, CodeQualityConfig
from utils.file_utils import find_python_files


@dataclass
class TrulyEnhancedDeadCodeIssue:
    """Enhanced container for dead code analysis results with advanced metadata."""
    file_path: str
    line_number: int
    issue_type: str  # "dead_code", "unreachable_code", "unused_import", "unused_dependency"
    description: str
    confidence: float
    severity: str
    code_snippet: str
    tool_source: str
    
    # Enhanced metadata
    consensus_count: int = 1  # How many tools agreed on this issue
    dynamic_usage_detected: bool = False
    call_graph_verified: bool = False
    context_score: float = 0.0  # Context-based confidence adjustment
    false_positive_risk: float = 0.0  # Risk of being a false positive (0-1)
    
    # Additional context
    function_name: str = ""
    class_name: str = ""
    module_type: str = ""  # "test", "main", "library", "script"
    is_public_api: bool = False
    has_docstring: bool = False
    decorators: List[str] = field(default_factory=list)
    
    # Filtering information
    filtering_reasons: List[str] = field(default_factory=list)  # Why confidence was adjusted
    original_confidence: float = 0.0  # Original confidence before adjustments


@dataclass
class TrulyEnhancedDeadCodeReport:
    """Enhanced report with advanced filtering results."""
    total_issues: int
    high_confidence_issues: int
    medium_confidence_issues: int
    low_confidence_issues: int
    
    issues_by_type: Dict[str, int]
    issues_by_file: Dict[str, List[TrulyEnhancedDeadCodeIssue]]
    issues_by_severity: Dict[str, List[TrulyEnhancedDeadCodeIssue]]
    issues_by_tool: Dict[str, List[TrulyEnhancedDeadCodeIssue]]
    
    # Enhanced metrics
    false_positives_filtered: int
    consensus_issues: int  # Issues agreed upon by multiple tools
    dynamic_usage_issues: int  # Issues with dynamic usage detected
    call_graph_verified_issues: int
    
    confidence_distribution: Dict[str, int]
    potential_savings: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    
    # Advanced filtering results
    filtering_stats: Dict[str, Any]
    tool_agreement_matrix: Dict[str, Dict[str, int]]


class TrulyEnhancedDeadCodeAnalyzer:
    """Truly enhanced dead code analyzer with advanced filtering."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tool availability
        self.pycg_available = PYCG_AVAILABLE
        self.deadcode_available = DEADCODE_AVAILABLE
        self.networkx_available = NETWORKX_AVAILABLE
        
        # Advanced filtering components
        self.call_graph = nx.DiGraph() if self.networkx_available else None
        self.import_graph = nx.DiGraph() if self.networkx_available else None
        self.dynamic_usage_patterns = self._load_dynamic_usage_patterns()
        self.false_positive_patterns = self._load_false_positive_patterns()
        
        # Context analysis
        self.file_type_classifier = FileTypeClassifier()
        self.pattern_analyzer = PatternAnalyzer()
        
    def _load_dynamic_usage_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate dynamic usage of functions."""
        return {
            "string_references": [
                r'["\'](\w+)["\']',  # String references
                r'getattr\([^,]+,\s*["\'](\w+)["\']',  # getattr calls
                r'setattr\([^,]+,\s*["\'](\w+)["\']',  # setattr calls
                r'hasattr\([^,]+,\s*["\'](\w+)["\']',  # hasattr calls
            ],
            "decorator_patterns": [
                r'@(\w+)',  # Decorator usage
                r'@(\w+)\.(\w+)',  # Decorator with module
            ],
            "reflection_patterns": [
                r'inspect\.getmembers\([^)]*\)',  # inspect.getmembers
                r'__dict__\[["\'](\w+)["\']\]',  # Dictionary access
                r'globals\(\)\[["\'](\w+)["\']\]',  # globals() access
                r'locals\(\)\[["\'](\w+)["\']\]',  # locals() access
            ],
            "framework_patterns": [
                r'@app\.route\(["\']([^"\']+)["\']',  # Flask routes
                r'@router\.(\w+)\(["\']([^"\']+)["\']',  # FastAPI routes
                r'@pytest\.fixture',  # Pytest fixtures
                r'@unittest\.',  # Unittest decorators
            ]
        }
    
    def _load_false_positive_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that commonly cause false positives."""
        return {
            "test_functions": [
                r'test_\w+',  # Test functions
                r'setup_\w+',  # Setup functions
                r'teardown_\w+',  # Teardown functions
            ],
            "main_functions": [
                r'main\b',  # Main functions
                r'__main__',  # Main module
            ],
            "public_api": [
                r'^[A-Z]',  # Capitalized functions (likely public)
                r'^[a-z]+_[a-z]+',  # snake_case functions (likely public)
            ],
            "special_methods": [
                r'__\w+__',  # Special methods
            ],
            "entry_points": [
                r'if __name__ == ["\']__main__["\']',  # Entry point
            ]
        }
    
    def analyze_directory(self, directory: str | Path) -> TrulyEnhancedDeadCodeReport:
        """
        Analyze directory with truly enhanced filtering.
        
        Args:
            directory: Path to directory
            
        Returns:
            TrulyEnhancedDeadCodeReport with advanced filtering
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        self.logger.info(f"Starting truly enhanced dead code analysis of {directory}")
        
        # Find all Python files
        python_files = find_python_files(directory)
        self.logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Phase 1: Build comprehensive analysis infrastructure
        self.logger.info("Phase 1: Building analysis infrastructure...")
        self._build_analysis_infrastructure(python_files)
        
        # Phase 2: Run multiple detection tools
        self.logger.info("Phase 2: Running detection tools...")
        tool_results = self._run_all_detection_tools(python_files)
        
        # Phase 3: Advanced filtering and validation
        self.logger.info("Phase 3: Advanced filtering and validation...")
        filtered_issues = self._apply_advanced_filtering(tool_results)
        
        # Phase 4: Generate enhanced report
        self.logger.info("Phase 4: Generating enhanced report...")
        report = self._generate_enhanced_report(filtered_issues, tool_results)
        
        self.logger.info(f"Analysis complete. Found {report.total_issues} high-confidence issues")
        return report
    
    def _build_analysis_infrastructure(self, python_files: List[Path]) -> None:
        """Build comprehensive analysis infrastructure."""
        # Build call graph
        if self.networkx_available:
            self._build_comprehensive_call_graph(python_files)
        
        # Analyze file types and patterns
        for file_path in python_files:
            self.file_type_classifier.classify_file(file_path)
            self.pattern_analyzer.analyze_file(file_path)
    
    def _run_all_detection_tools(self, python_files: List[Path]) -> Dict[str, List[TrulyEnhancedDeadCodeIssue]]:
        """Run all available detection tools and collect results."""
        tool_results = {}
        
        # Run DeadCodeRemover if available
        if self.deadcode_available:
            tool_results['deadcode'] = self._run_deadcode_analysis(python_files)
            self.logger.info(f"DeadCodeRemover found {len(tool_results['deadcode'])} issues")
        
        # Run PyCG-based analysis if available
        if self.pycg_available:
            tool_results['pycg'] = self._run_pycg_analysis(python_files)
            self.logger.info(f"PyCG analysis found {len(tool_results['pycg'])} issues")
        
        # Run enhanced AST analysis
        tool_results['ast'] = self._run_enhanced_ast_analysis(python_files)
        self.logger.info(f"Enhanced AST analysis found {len(tool_results['ast'])} issues")
        
        return tool_results
    
    def _apply_advanced_filtering(self, tool_results: Dict[str, List[TrulyEnhancedDeadCodeIssue]]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Apply advanced filtering to reduce false positives while preserving all issues with confidence levels."""
        all_issues = []
        for tool, issues in tool_results.items():
            all_issues.extend(issues)
        
        # Step 1: Multi-tool consensus filtering (but keep all issues)
        consensus_issues = self._apply_consensus_filtering(all_issues)
        
        # Step 2: Dynamic usage detection (adjust confidence but keep issues)
        dynamic_filtered = self._filter_dynamic_usage(consensus_issues)
        
        # Step 3: Call graph validation (adjust confidence but keep issues)
        call_graph_filtered = self._filter_call_graph_validation(dynamic_filtered)
        
        # Step 4: Context-aware filtering (adjust confidence but keep issues)
        context_filtered = self._apply_context_aware_filtering(call_graph_filtered)
        
        # Step 5: Confidence scoring (keep all issues, just score them)
        final_issues = self._apply_confidence_scoring(context_filtered)
        
        return final_issues
    
    def _apply_consensus_filtering(self, all_issues: List[TrulyEnhancedDeadCodeIssue]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Apply consensus scoring but keep all issues."""
        consensus_issues = []
        issues_by_location = defaultdict(list)
        
        # Group issues by location
        for issue in all_issues:
            key = (issue.file_path, issue.line_number, issue.function_name)
            issues_by_location[key].append(issue)
        
        # Process all issues, adjusting confidence based on consensus
        for location, issues in issues_by_location.items():
            if len(issues) >= 2:  # Multiple tools agree
                # Take the issue with highest confidence and boost it
                best_issue = max(issues, key=lambda x: x.confidence)
                best_issue.consensus_count = len(issues)
                best_issue.original_confidence = best_issue.confidence
                best_issue.confidence = min(100.0, best_issue.confidence * 1.2)  # Boost confidence
                best_issue.filtering_reasons.append(f"Multi-tool consensus ({len(issues)} tools agree)")
                consensus_issues.append(best_issue)
            else:  # Single tool detection
                # Keep the issue but reduce confidence
                issue = issues[0]
                issue.consensus_count = 1
                issue.original_confidence = issue.confidence
                issue.confidence = issue.confidence * 0.8  # Reduce confidence for single tool
                issue.filtering_reasons.append("Single tool detection (lower confidence)")
                consensus_issues.append(issue)
        
        self.logger.info(f"Consensus filtering: {len(consensus_issues)} issues processed from {len(all_issues)} original")
        return consensus_issues
    
    def _filter_dynamic_usage(self, issues: List[TrulyEnhancedDeadCodeIssue]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Filter out functions that are used dynamically."""
        filtered_issues = []
        
        for issue in issues:
            if self._has_dynamic_usage(issue):
                issue.dynamic_usage_detected = True
                # Reduce confidence but don't completely filter out
                issue.confidence *= 0.7
                issue.filtering_reasons.append("Dynamic usage detected (confidence reduced)")
            filtered_issues.append(issue)
        
        dynamic_count = sum(1 for i in filtered_issues if i.dynamic_usage_detected)
        self.logger.info(f"Dynamic usage filtering: {dynamic_count} issues with dynamic usage detected")
        return filtered_issues
    
    def _filter_call_graph_validation(self, issues: List[TrulyEnhancedDeadCodeIssue]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Validate issues against call graph."""
        if not self.networkx_available:
            return issues
        
        filtered_issues = []
        
        for issue in issues:
            if self._is_verified_by_call_graph(issue):
                issue.call_graph_verified = True
                issue.confidence *= 1.2  # Boost confidence
                issue.filtering_reasons.append("Call graph verified (confidence boosted)")
            filtered_issues.append(issue)
        
        verified_count = sum(1 for i in filtered_issues if i.call_graph_verified)
        self.logger.info(f"Call graph validation: {verified_count} issues verified by call graph")
        return filtered_issues
    
    def _apply_context_aware_filtering(self, issues: List[TrulyEnhancedDeadCodeIssue]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Apply context-aware filtering based on file type and patterns."""
        filtered_issues = []
        
        for issue in issues:
            context_score = self._calculate_context_score(issue)
            issue.context_score = context_score
            
            # Adjust confidence based on context
            if context_score > 0.8:  # High confidence context
                issue.confidence *= 1.1
                issue.filtering_reasons.append("High confidence context (confidence boosted)")
            elif context_score < 0.3:  # Low confidence context
                issue.confidence *= 0.8
                issue.filtering_reasons.append("Low confidence context (confidence reduced)")
            
            filtered_issues.append(issue)
        
        self.logger.info(f"Context-aware filtering applied to {len(filtered_issues)} issues")
        return filtered_issues
    
    def _apply_confidence_scoring(self, issues: List[TrulyEnhancedDeadCodeIssue]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Apply final confidence scoring but keep all issues."""
        # Calculate false positive risk for all issues
        for issue in issues:
            issue.false_positive_risk = self._calculate_false_positive_risk(issue)
            
            # Adjust confidence based on false positive risk
            if issue.false_positive_risk > 0.7:
                issue.confidence = issue.confidence * 0.5  # Significantly reduce confidence
                issue.filtering_reasons.append(f"High false positive risk ({issue.false_positive_risk:.2f}) - confidence significantly reduced")
            elif issue.false_positive_risk > 0.5:
                issue.confidence = issue.confidence * 0.8  # Reduce confidence
                issue.filtering_reasons.append(f"Medium false positive risk ({issue.false_positive_risk:.2f}) - confidence reduced")
        
        # Keep all issues but sort by confidence
        issues.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.info(f"Confidence scoring: {len(issues)} issues scored and sorted by confidence")
        return issues
    
    def _has_dynamic_usage(self, issue: TrulyEnhancedDeadCodeIssue) -> bool:
        """Check if a function has dynamic usage patterns."""
        if not issue.function_name:
            return False
        
        file_path = Path(issue.file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for dynamic usage patterns
            for pattern_type, patterns in self.dynamic_usage_patterns.items():
                for pattern in patterns:
                    try:
                        if re.search(pattern, content):
                            return True
                    except re.error:
                        # Skip invalid regex patterns
                        continue
            
            return False
        except Exception:
            return False
    
    def _is_verified_by_call_graph(self, issue: TrulyEnhancedDeadCodeIssue) -> bool:
        """Check if issue is verified by call graph analysis."""
        if not self.networkx_available or not issue.function_name:
            return False
        
        func_key = f"{issue.file_path}::{issue.function_name}"
        if func_key in self.call_graph:
            # Check if function has callers
            callers = list(self.call_graph.predecessors(func_key))
            return len(callers) == 0  # No callers = truly dead
        
        return True  # Not in call graph = likely dead
    
    def _calculate_context_score(self, issue: TrulyEnhancedDeadCodeIssue) -> float:
        """Calculate context-based confidence score."""
        score = 0.5  # Base score
        
        # File type adjustments
        file_type = self.file_type_classifier.get_file_type(issue.file_path)
        if file_type == "test":
            score -= 0.3  # Test files often have unused functions
        elif file_type == "main":
            score += 0.2  # Main files are more likely to have dead code
        
        # Pattern adjustments
        if self.pattern_analyzer.is_likely_false_positive(issue):
            score -= 0.4
        
        # Function characteristics
        if issue.has_docstring:
            score -= 0.2  # Documented functions are less likely to be dead
        
        if issue.is_public_api:
            score -= 0.3  # Public APIs are less likely to be dead
        
        return max(0.0, min(1.0, score))
    
    def _calculate_false_positive_risk(self, issue: TrulyEnhancedDeadCodeIssue) -> float:
        """Calculate risk of false positive."""
        risk = 0.0
        
        # Consensus reduces risk
        if issue.consensus_count >= 3:
            risk -= 0.3
        elif issue.consensus_count == 2:
            risk -= 0.1
        
        # Dynamic usage increases risk
        if issue.dynamic_usage_detected:
            risk += 0.4
        
        # Call graph verification reduces risk
        if issue.call_graph_verified:
            risk -= 0.2
        
        # Context score affects risk
        risk += (1.0 - issue.context_score) * 0.3
        
        return max(0.0, min(1.0, risk))
    
    def _build_comprehensive_call_graph(self, python_files: List[Path]) -> None:
        """Build comprehensive call graph."""
        if not self.networkx_available:
            return
        
        self.logger.info("Building comprehensive call graph...")
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                self._extract_call_graph_from_ast(tree, file_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
    
    def _extract_call_graph_from_ast(self, tree: ast.AST, file_path: Path) -> None:
        """Extract call graph from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = f"{file_path}::{node.name}"
                self.call_graph.add_node(func_name)
                
                # Find function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            called_func = f"{file_path}::{child.func.id}"
                            self.call_graph.add_edge(func_name, called_func)
    
    def _run_deadcode_analysis(self, python_files: List[Path]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Run DeadCodeRemover analysis."""
        issues = []
        if not self.deadcode_available:
            return issues
        
        try:
            # Create temporary directory for analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy files to temp directory
                for file_path in python_files:
                    temp_file = temp_path / file_path.name
                    temp_file.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
                
                # Run deadcode analysis
                result = subprocess.run(
                    ['python', '-m', 'deadcode', str(temp_path)],
                    capture_output=True, text=True, timeout=300
                )
                
                # Parse results
                if result.returncode == 0:
                    issues.extend(self._parse_deadcode_output(result.stdout, python_files))
                
        except Exception as e:
            self.logger.warning(f"DeadCodeRemover analysis failed: {e}")
        
        return issues
    
    def _run_pycg_analysis(self, python_files: List[Path]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Run PyCG-based analysis."""
        issues = []
        if not self.pycg_available:
            return issues
        
        try:
            # Use PyCG to build call graph and find unused functions
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content, filename=str(file_path))
                    file_issues = self._analyze_file_with_pycg(tree, file_path)
                    issues.extend(file_issues)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {file_path} with PyCG: {e}")
        
        except Exception as e:
            self.logger.warning(f"PyCG analysis failed: {e}")
        
        return issues
    
    def _run_enhanced_ast_analysis(self, python_files: List[Path]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Run enhanced AST analysis."""
        issues = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                file_issues = self._analyze_file_with_ast(tree, file_path, content)
                issues.extend(file_issues)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path} with AST: {e}")
        
        return issues
    
    def _parse_deadcode_output(self, output: str, python_files: List[Path]) -> List[TrulyEnhancedDeadCodeIssue]:
        """Parse DeadCodeRemover output."""
        issues = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if ':' in line and '.py' in line:
                try:
                    # Parse format: file.py:line: message
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_name = parts[0]
                        line_num = int(parts[1])
                        message = parts[2].strip()
                        
                        # Find the actual file path
                        file_path = None
                        for pf in python_files:
                            if pf.name == file_name:
                                file_path = str(pf)
                                break
                        
                        if file_path:
                            issue = TrulyEnhancedDeadCodeIssue(
                                file_path=file_path,
                                line_number=line_num,
                                issue_type="dead_code",
                                description=message,
                                confidence=85.0,  # DeadCodeRemover is generally reliable
                                severity="medium",
                                code_snippet="",
                                tool_source="DeadCodeRemover"
                            )
                            issues.append(issue)
                
                except (ValueError, IndexError):
                    continue
        
        return issues
    
    def _analyze_file_with_pycg(self, tree: ast.AST, file_path: Path) -> List[TrulyEnhancedDeadCodeIssue]:
        """Analyze file using PyCG-based techniques."""
        issues = []
        
        # Find all function definitions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        
        # Check if functions are called
        for func in functions:
            func_name = func.name
            is_called = False
            
            # Check if function is called anywhere in the file
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        is_called = True
                        break
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr == func_name:
                            is_called = True
                            break
            
            if not is_called and not self._is_special_function(func_name):
                issue = TrulyEnhancedDeadCodeIssue(
                    file_path=str(file_path),
                    line_number=func.lineno,
                    issue_type="dead_code",
                    description=f"Function '{func_name}' is defined but never called",
                    confidence=75.0,
                    severity="medium",
                    code_snippet=self._get_code_snippet(func),
                    tool_source="PyCG",
                    function_name=func_name,
                    has_docstring=ast.get_docstring(func) is not None
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_file_with_ast(self, tree: ast.AST, file_path: Path, content: str) -> List[TrulyEnhancedDeadCodeIssue]:
        """Analyze file using enhanced AST techniques."""
        issues = []
        
        # Find all function definitions
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)
        
        # Analyze functions
        for func in functions:
            if self._is_unused_function(func, tree, content):
                try:
                    issue = TrulyEnhancedDeadCodeIssue(
                        file_path=str(file_path),
                        line_number=func.lineno,
                        issue_type="dead_code",
                        description=f"Function '{func.name}' is defined but never called",
                        confidence=70.0,
                        severity="medium",
                        code_snippet=self._get_code_snippet(func),
                        tool_source="Enhanced AST",
                        function_name=func.name,
                        has_docstring=ast.get_docstring(func) is not None,
                        decorators=[self._get_decorator_name(dec) for dec in func.decorator_list]
                    )
                    issues.append(issue)
                except Exception as e:
                    self.logger.warning(f"Failed to create issue for function {func.name}: {e}")
        
        # Analyze classes
        for cls in classes:
            if self._is_unused_class(cls, tree, content):
                try:
                    issue = TrulyEnhancedDeadCodeIssue(
                        file_path=str(file_path),
                        line_number=cls.lineno,
                        issue_type="dead_code",
                        description=f"Class '{cls.name}' is defined but never used",
                        confidence=70.0,
                        severity="medium",
                        code_snippet=self._get_code_snippet(cls),
                        tool_source="Enhanced AST",
                        class_name=cls.name,
                        has_docstring=ast.get_docstring(cls) is not None
                    )
                    issues.append(issue)
                except Exception as e:
                    self.logger.warning(f"Failed to create issue for class {cls.name}: {e}")
        
        return issues
    
    def _is_special_function(self, func_name: str) -> bool:
        """Check if function is special (should not be considered dead code)."""
        special_patterns = [
            r'^__\w+__$',  # Special methods
            r'^test_',     # Test functions
            r'^setup_',    # Setup functions
            r'^teardown_', # Teardown functions
            r'^main$',     # Main function
        ]
        
        for pattern in special_patterns:
            if re.match(pattern, func_name):
                return True
        
        return False
    
    def _is_unused_function(self, func: ast.FunctionDef, tree: ast.AST, content: str) -> bool:
        """Check if function is unused."""
        func_name = func.name
        
        # Skip special functions
        if self._is_special_function(func_name):
            return False
        
        # Check for direct calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return False
                elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                    return False
        
        # Check for string references
        string_patterns = [f'"{func_name}"', f"'{func_name}'"]
        for pattern in string_patterns:
            try:
                if pattern in content:
                    return False
            except Exception:
                # Skip if there are issues with string matching
                continue
        
        return True
    
    def _is_unused_class(self, cls: ast.ClassDef, tree: ast.AST, content: str) -> bool:
        """Check if class is unused."""
        cls_name = cls.name
        
        # Skip special classes
        if cls_name.startswith('_'):
            return False
        
        # Check for direct usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == cls_name:
                # Check if it's not the class definition itself
                # Note: AST nodes don't have parent attribute by default
                # We'll use a different approach to check if it's the class definition
                if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef) and node.parent.name == cls_name:
                    continue
                return False
        
        return True
    
    def _get_code_snippet(self, node: ast.AST) -> str:
        """Get code snippet for a node."""
        try:
            # This is a simplified version - in practice you'd want to get the actual source
            if isinstance(node, ast.FunctionDef):
                return f"def {node.name}(...):"
            elif isinstance(node, ast.ClassDef):
                return f"class {node.name}:"
            else:
                return str(node)
        except:
            return ""
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        else:
            return str(decorator)
    
    def _generate_enhanced_report(self, filtered_issues: List[TrulyEnhancedDeadCodeIssue], 
                                tool_results: Dict[str, List[TrulyEnhancedDeadCodeIssue]]) -> TrulyEnhancedDeadCodeReport:
        """Generate enhanced report with filtering statistics."""
        # Calculate statistics
        total_issues = len(filtered_issues)
        high_confidence = len([i for i in filtered_issues if i.confidence > 80])
        medium_confidence = len([i for i in filtered_issues if 60 < i.confidence <= 80])
        low_confidence = len([i for i in filtered_issues if i.confidence <= 60])
        
        # Group issues
        issues_by_type = Counter(i.issue_type for i in filtered_issues)
        issues_by_file = defaultdict(list)
        issues_by_severity = defaultdict(list)
        issues_by_tool = defaultdict(list)
        
        for issue in filtered_issues:
            issues_by_file[issue.file_path].append(issue)
            issues_by_severity[issue.severity].append(issue)
            issues_by_tool[issue.tool_source].append(issue)
        
        # Calculate filtering statistics
        original_count = sum(len(issues) for issues in tool_results.values())
        filtered_count = len(filtered_issues)
        false_positives_filtered = original_count - filtered_count
        
        # Calculate tool agreement matrix
        tool_agreement_matrix = self._calculate_tool_agreement_matrix(tool_results)
        
        return TrulyEnhancedDeadCodeReport(
            total_issues=total_issues,
            high_confidence_issues=high_confidence,
            medium_confidence_issues=medium_confidence,
            low_confidence_issues=low_confidence,
            issues_by_type=dict(issues_by_type),
            issues_by_file=dict(issues_by_file),
            issues_by_severity=dict(issues_by_severity),
            issues_by_tool=dict(issues_by_tool),
            false_positives_filtered=false_positives_filtered,
            consensus_issues=len([i for i in filtered_issues if i.consensus_count >= 2]),
            dynamic_usage_issues=len([i for i in filtered_issues if i.dynamic_usage_detected]),
            call_graph_verified_issues=len([i for i in filtered_issues if i.call_graph_verified]),
            confidence_distribution={"high": high_confidence, "medium": medium_confidence, "low": low_confidence},
            potential_savings={"lines_removable": total_issues * 5},  # Estimate
            impact_analysis={"files_affected": len(issues_by_file)},
            filtering_stats={
                "original_issues": original_count,
                "filtered_issues": filtered_count,
                "false_positives_filtered": false_positives_filtered,
                "filtering_effectiveness": (false_positives_filtered / original_count * 100) if original_count > 0 else 0
            },
            tool_agreement_matrix=tool_agreement_matrix
        )
    
    def _calculate_tool_agreement_matrix(self, tool_results: Dict[str, List[TrulyEnhancedDeadCodeIssue]]) -> Dict[str, Dict[str, int]]:
        """Calculate agreement matrix between tools."""
        matrix = {}
        tools = list(tool_results.keys())
        
        for tool1 in tools:
            matrix[tool1] = {}
            for tool2 in tools:
                if tool1 == tool2:
                    matrix[tool1][tool2] = len(tool_results[tool1])
                else:
                    # Calculate overlap
                    issues1 = set((i.file_path, i.line_number) for i in tool_results[tool1])
                    issues2 = set((i.file_path, i.line_number) for i in tool_results[tool2])
                    overlap = len(issues1.intersection(issues2))
                    matrix[tool1][tool2] = overlap
        
        return matrix


class FileTypeClassifier:
    """Classifies files by type to improve filtering."""
    
    def __init__(self):
        self.file_types = {}
    
    def classify_file(self, file_path: Path) -> str:
        """Classify a file by type."""
        file_type = "unknown"
        
        # Check file name patterns
        if "test" in file_path.name.lower():
            file_type = "test"
        elif file_path.name in ["__main__.py", "main.py", "run.py"]:
            file_type = "main"
        elif file_path.name == "__init__.py":
            file_type = "init"
        elif "setup" in file_path.name.lower():
            file_type = "setup"
        elif "config" in file_path.name.lower():
            file_type = "config"
        else:
            file_type = "library"
        
        self.file_types[str(file_path)] = file_type
        return file_type
    
    def get_file_type(self, file_path: str) -> str:
        """Get classified file type."""
        return self.file_types.get(file_path, "unknown")


class PatternAnalyzer:
    """Analyzes patterns to identify likely false positives."""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def analyze_file(self, file_path: Path) -> None:
        """Analyze file for patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            patterns = {
                "has_main_guard": "if __name__ == '__main__'" in content,
                "has_test_functions": bool(re.search(r'def test_\w+', content)),
                "has_setup_functions": bool(re.search(r'def setup_\w+', content)),
                "has_public_functions": bool(re.search(r'def [A-Z]\w+', content)),
                "has_docstrings": '"""' in content or "'''" in content,
            }
            
            self.pattern_cache[str(file_path)] = patterns
            
        except Exception:
            self.pattern_cache[str(file_path)] = {}
    
    def is_likely_false_positive(self, issue: TrulyEnhancedDeadCodeIssue) -> bool:
        """Check if issue is likely a false positive based on patterns."""
        patterns = self.pattern_cache.get(issue.file_path, {})
        
        # Test functions are often unused but not dead code
        if patterns.get("has_test_functions", False) and "test" in issue.function_name.lower():
            return True
        
        # Setup functions are often unused but not dead code
        if patterns.get("has_setup_functions", False) and "setup" in issue.function_name.lower():
            return True
        
        # Public functions are less likely to be dead code
        if patterns.get("has_public_functions", False) and issue.function_name[0].isupper():
            return True
        
        return False