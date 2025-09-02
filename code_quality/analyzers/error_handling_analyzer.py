"""
Error Handling Analyzer - Analyzes exception handling patterns, error recovery, and error handling quality.
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import json
import logging

from ..core.config import CodeQualityConfig, get_default_config
from ..utils.file_utils import find_python_files


class ErrorHandlingIssue:
    """Container for error handling issue information."""
    
    def __init__(self, issue_type: str, description: str, line: int, 
                 severity: str = "warning", details: Dict[str, Any] = None):
        self.issue_type = issue_type
        self.description = description
        self.line = line
        self.severity = severity
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "line": self.line,
            "severity": self.severity,
            "details": self.details
        }


class ErrorHandlingPattern:
    """Container for error handling pattern information."""
    
    def __init__(self, pattern_type: str, description: str, line: int, 
                 quality_score: float, details: Dict[str, Any] = None):
        self.pattern_type = pattern_type
        self.description = description
        self.line = line
        self.quality_score = quality_score
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "line": self.line,
            "quality_score": self.quality_score,
            "details": self.details
        }


class ErrorHandlingAnalyzer:
    """
    Comprehensive error handling analysis and quality assessment.
    
    Features:
    - Exception handling pattern detection
    - Error recovery mechanism analysis
    - Error propagation analysis
    - Error handling quality scoring
    - Best practice validation
    - Error handling coverage analysis
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.error_issues: List[ErrorHandlingIssue] = []
        self.error_patterns: List[ErrorHandlingPattern] = []
        self.error_metrics: Dict[str, Dict[str, Any]] = {}
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze error handling for a single Python file.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Clear previous results for this file
            self.error_issues = [issue for issue in self.error_issues if issue.description != str(file_path)]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform error handling analysis
            exception_analysis = self._analyze_exception_handling(tree)
            error_recovery_analysis = self._analyze_error_recovery(tree)
            error_propagation_analysis = self._analyze_error_propagation(tree)
            quality_analysis = self._assess_error_handling_quality(tree)
            
            # Combine results
            combined_results = {
                "exceptions": exception_analysis,
                "error_recovery": error_recovery_analysis,
                "error_propagation": error_propagation_analysis,
                "quality": quality_analysis
            }
            
            # Calculate overall error handling score
            error_handling_score = self._calculate_error_handling_score(combined_results)
            
            # Store results
            self.file_stats[str(file_path)] = combined_results
            
            return {
                "status": "success",
                "issues_found": len(self.error_issues),
                "issues_fixed": 0,
                "details": combined_results,
                "error_handling_score": error_handling_score
            }
            
        except Exception as e:
            logging.error(f"Error in error handling analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "error_handling_score": 0
            }
    
    def _analyze_exception_handling(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze exception handling patterns."""
        try_blocks = []
        except_blocks = []
        finally_blocks = []
        bare_excepts = []
        specific_excepts = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_blocks.append({
                    "line": node.lineno,
                    "body_length": len(node.body),
                    "handlers": len(node.handlers),
                    "has_finally": node.finalbody is not None
                })
                
                # Analyze except handlers
                for handler in node.handlers:
                    if handler.type is None:
                        bare_excepts.append({
                            "line": handler.lineno,
                            "description": "Bare except clause"
                        })
                    else:
                        specific_excepts.append({
                            "line": handler.lineno,
                            "exception_type": self._get_exception_type(handler.type),
                            "description": f"Specific exception: {self._get_exception_type(handler.type)}"
                        })
                
                # Analyze finally blocks
                if node.finalbody:
                    finally_blocks.append({
                        "line": node.lineno,
                        "body_length": len(node.finalbody)
                    })
        
        # Detect anti-patterns
        anti_patterns = []
        for bare_except in bare_excepts:
            anti_patterns.append(ErrorHandlingIssue(
                "anti_pattern",
                "Bare except clause - catches all exceptions",
                bare_except["line"],
                "error",
                {"type": "bare_except", "description": "Consider specifying exception types"}
            ))
        
        return {
            "try_blocks": try_blocks,
            "except_blocks": len(bare_excepts) + len(specific_excepts),
            "finally_blocks": finally_blocks,
            "bare_excepts": bare_excepts,
            "specific_excepts": specific_excepts,
            "anti_patterns": [p.to_dict() for p in anti_patterns],
            "total_exception_handlers": len(bare_excepts) + len(specific_excepts)
        }
    
    def _analyze_error_recovery(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze error recovery mechanisms."""
        recovery_patterns = []
        error_logging = []
        error_returns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    # Analyze recovery patterns in except blocks
                    recovery_analysis = self._analyze_handler_recovery(handler)
                    if recovery_analysis:
                        recovery_patterns.append(recovery_analysis)
                    
                    # Check for error logging
                    if self._has_error_logging(handler):
                        error_logging.append({
                            "line": handler.lineno,
                            "type": "error_logging"
                        })
                    
                    # Check for error returns
                    if self._has_error_return(handler):
                        error_returns.append({
                            "line": handler.lineno,
                            "type": "error_return"
                        })
        
        return {
            "recovery_patterns": recovery_patterns,
            "error_logging": error_logging,
            "error_returns": error_returns,
            "total_recovery_mechanisms": len(recovery_patterns) + len(error_logging) + len(error_returns)
        }
    
    def _analyze_error_propagation(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze error propagation patterns."""
        error_propagation = []
        error_suppression = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    # Check if exceptions are re-raised
                    if self._re_raises_exception(handler):
                        error_propagation.append({
                            "line": handler.lineno,
                            "type": "re_raise",
                            "description": "Exception is re-raised"
                        })
                    
                    # Check if exceptions are suppressed
                    if self._suppresses_exception(handler):
                        error_suppression.append({
                            "line": handler.lineno,
                            "type": "suppression",
                            "description": "Exception is suppressed"
                        })
        
        return {
            "error_propagation": error_propagation,
            "error_suppression": error_suppression,
            "total_propagation_patterns": len(error_propagation) + len(error_suppression)
        }
    
    def _assess_error_handling_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Assess overall error handling quality."""
        quality_score = 100.0
        issues = []
        
        # Analyze exception handling coverage
        functions_with_exceptions = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if self._function_has_exception_handling(node):
                    functions_with_exceptions += 1
        
        # Calculate coverage
        exception_coverage = (functions_with_exceptions / total_functions * 100) if total_functions > 0 else 0
        
        # Penalize based on issues
        bare_except_count = len([issue for issue in self.error_issues if issue.issue_type == "anti_pattern"])
        quality_score -= bare_except_count * 20
        
        # Penalize for low coverage
        if exception_coverage < 50:
            quality_score -= 30
            issues.append("Low exception handling coverage")
        elif exception_coverage < 80:
            quality_score -= 15
            issues.append("Moderate exception handling coverage")
        
        return {
            "quality_score": max(0.0, quality_score),
            "exception_coverage": exception_coverage,
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def _get_exception_type(self, node: ast.expr) -> str:
        """Extract exception type from AST node."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_exception_type(node.value)}.{node.attr}"
            elif isinstance(node, ast.Tuple):
                return f"({', '.join(self._get_exception_type(el) for el in node.elts)})"
            else:
                return str(node)
        except Exception:
            return "unknown"
    
    def _analyze_handler_recovery(self, handler: ast.ExceptHandler) -> Optional[Dict[str, Any]]:
        """Analyze recovery patterns in an except handler."""
        recovery_patterns = []
        
        for stmt in handler.body:
            if isinstance(stmt, ast.Return):
                recovery_patterns.append("return_value")
            elif isinstance(stmt, ast.Assign):
                recovery_patterns.append("assign_default")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._is_logging_call(stmt.value):
                    recovery_patterns.append("log_error")
                elif self._is_recovery_call(stmt.value):
                    recovery_patterns.append("recovery_action")
        
        if recovery_patterns:
            return {
                "line": handler.lineno,
                "patterns": recovery_patterns,
                "recovery_score": len(recovery_patterns) * 10
            }
        return None
    
    def _has_error_logging(self, handler: ast.ExceptHandler) -> bool:
        """Check if handler has error logging."""
        for stmt in handler.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._is_logging_call(stmt.value):
                    return True
        return False
    
    def _has_error_return(self, handler: ast.ExceptHandler) -> bool:
        """Check if handler has error return."""
        for stmt in handler.body:
            if isinstance(stmt, ast.Return):
                return True
        return False
    
    def _re_raises_exception(self, handler: ast.ExceptHandler) -> bool:
        """Check if handler re-raises the exception."""
        for stmt in handler.body:
            if isinstance(stmt, ast.Raise):
                return True
        return False
    
    def _suppresses_exception(self, handler: ast.ExceptHandler) -> bool:
        """Check if handler suppresses the exception."""
        # If handler has no body or only contains pass, it suppresses
        if not handler.body:
            return True
        
        for stmt in handler.body:
            if isinstance(stmt, ast.Pass):
                return True
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value == "pass":
                    return True
        
        return False
    
    def _function_has_exception_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function has exception handling."""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False
    
    def _is_logging_call(self, node: ast.Call) -> bool:
        """Check if function call is for logging."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['log', 'error', 'warning', 'info', 'debug']:
                return True
        elif isinstance(node.func, ast.Name):
            if node.func.id in ['print', 'log', 'error']:
                return True
        return False
    
    def _is_recovery_call(self, node: ast.Call) -> bool:
        """Check if function call is for recovery."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['retry', 'fallback', 'default', 'reset']:
                return True
        elif isinstance(node.func, ast.Name):
            if node.func.id in ['retry', 'fallback', 'default', 'reset']:
                return True
        return False
    
    def _calculate_error_handling_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall error handling quality score."""
        score = 100.0
        
        # Quality score from analysis
        quality_score = analysis_results.get("quality", {}).get("quality_score", 100)
        score = (score + quality_score) / 2
        
        # Exception coverage bonus
        exception_coverage = analysis_results.get("quality", {}).get("exception_coverage", 0)
        if exception_coverage > 80:
            score += 10
        elif exception_coverage > 60:
            score += 5
        
        # Anti-pattern penalties
        anti_patterns = analysis_results.get("exceptions", {}).get("anti_patterns", [])
        score -= len(anti_patterns) * 15
        
        # Recovery mechanism bonus
        recovery_mechanisms = analysis_results.get("error_recovery", {}).get("total_recovery_mechanisms", 0)
        if recovery_mechanisms > 0:
            score += min(20, recovery_mechanisms * 5)
        
        return max(0.0, min(100.0, score))
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze error handling for all Python files in a directory.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing error handling for {len(python_files)} Python files...")
        
        # Clear previous results
        self.error_issues.clear()
        self.error_patterns.clear()
        self.error_metrics.clear()
        self.file_stats.clear()
        
        total_issues = 0
        total_error_handling_score = 0.0
        successful_files = 0
        
        for file_path in python_files:
            try:
                result = self.analyze_file(str(file_path))
                if result["status"] == "success":
                    total_issues += result["issues_found"]
                    total_error_handling_score += result["error_handling_score"]
                    successful_files += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        avg_error_handling_score = total_error_handling_score / successful_files if successful_files > 0 else 0.0
        
        return {
            "status": "success",
            "total_files": len(python_files),
            "successful_files": successful_files,
            "total_issues": total_issues,
            "average_error_handling_score": avg_error_handling_score,
            "file_stats": self.file_stats
        }
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze the given file (alias for analyze_file)."""
        return self.analyze_file(file_path)