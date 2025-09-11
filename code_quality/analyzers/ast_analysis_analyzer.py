from src.utils.tprint import tprint

from typing import Dict, List, Any, Optional
import numpy as np

"""
AST Analysis Analyzer - Integrates Astroid, Rope, and Jedi for advanced AST-based code analysis.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.config import AnalysisConfig, CodeQualityConfig

try:
    import astroid
    ASTROID_AVAILABLE = True
except ImportError:
    astroid = None
    ASTROID_AVAILABLE = False
    tprint("Warning: astroid not available")

# Rope integration removed as requested
ROPE_AVAILABLE = False

try:
    import jedi

    JEDI_AVAILABLE = True
except ImportError:
    JEDI_AVAILABLE = False


class ASTAnalysisAnalyzer:
    """
    Advanced AST-based analysis using multiple tools:
    - Astroid: Advanced AST parsing and analysis
    - Jedi: Code completion and static analysis
    - Custom AST Analysis: Cyclomatic complexity, nesting levels, unused variables
    """

    def __init__(self, config: CodeQualityConfig):
        self.config = config
        self.results = {}

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file with all AST analysis tools."""
        results = {
            "file": file_path,
            "tools": {},
            "summary": {
                "total_issues": 0,
                "complexity_issues": 0,
                "refactoring_opportunities": 0,
                "code_completion_issues": 0,
                "ast_analysis_issues": 0,
            }
        }

        # Run Astroid analysis
        if ASTROID_AVAILABLE:
            try:
                astroid_result = self._run_astroid_analysis(file_path)
                results["tools"]["astroid"] = astroid_result
                results["summary"]["ast_analysis_issues"] += len(astroid_result.get("issues", []))
                results["summary"]["total_issues"] += len(astroid_result.get("issues", []))
            except Exception as e:
                results["tools"]["astroid"] = {"status": "error", "error": str(e)}

        # Rope analysis removed as requested

        # Run Jedi analysis
        if JEDI_AVAILABLE:
            try:
                jedi_result = self._run_jedi_analysis(file_path)
                results["tools"]["jedi"] = jedi_result
                results["summary"]["code_completion_issues"] += len(jedi_result.get("issues", []))
                results["summary"]["total_issues"] += len(jedi_result.get("issues", []))
            except Exception as e:
                results["tools"]["jedi"] = {"status": "error", "error": str(e)}

        # Run custom AST analysis
        try:
            custom_ast_result = self._run_custom_ast_analysis(file_path)
            results["tools"]["custom_ast"] = custom_ast_result
            results["summary"]["complexity_issues"] += len(custom_ast_result.get("complexity_issues", []))
            results["summary"]["total_issues"] += len(custom_ast_result.get("complexity_issues", []))
        except Exception as e:
            results["tools"]["custom_ast"] = {"status": "error", "error": str(e)}

        return results

    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        directory = Path(directory_path)
        python_files = list(directory.rglob("*.py"))
        
        results = {
            "directory": directory_path,
            "files": {},
            "summary": {
                "total_files": len(python_files),
                "total_issues": 0,
                "complexity_issues": 0,
                "refactoring_opportunities": 0,
                "code_completion_issues": 0,
                "ast_analysis_issues": 0,
                "tools_availability": {
                    "astroid": ASTROID_AVAILABLE,
                    "jedi": JEDI_AVAILABLE
                }
            }
        }

        for file_path in python_files:
            file_result = self.analyze_file(str(file_path))
            results["files"][str(file_path)] = file_result
            
            # Update summary
            file_summary = file_result["summary"]
            results["summary"]["total_issues"] += file_summary["total_issues"]
            results["summary"]["complexity_issues"] += file_summary["complexity_issues"]
            results["summary"]["refactoring_opportunities"] += file_summary["refactoring_opportunities"]
            results["summary"]["code_completion_issues"] += file_summary["code_completion_issues"]
            results["summary"]["ast_analysis_issues"] += file_summary["ast_analysis_issues"]

        return results

    def _run_astroid_analysis(self, file_path: str) -> Dict[str, Any]:
        """Run Astroid analysis on a file."""
        if not ASTROID_AVAILABLE:
            return {"status": "skipped", "reason": "astroid not available"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse with Astroid
            tree = astroid.parse(content, file_path)
            
            issues = []
            
            # Analyze for common issues
            for node in astroid.walk(tree):
                # Check for long functions
                if isinstance(node, astroid.FunctionDef):
                    if len(node.body) > 20:  # More than 20 statements
                        issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' is too long ({len(node.body)} statements)",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "long_function"
                        })
                
                # Check for deep nesting
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    nesting_level = self._get_nesting_level(node)
                    if nesting_level > 4:
                        issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Code nesting level too deep ({nesting_level} levels)",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "deep_nesting"
                        })
                
                # Check for unused variables
                if isinstance(node, astroid.Assign):
                    for target in node.targets:
                        if isinstance(target, astroid.Name):
                            if not self._is_variable_used(target.name, node):
                                issues.append({
                                    "line": node.lineno,
                                    "column": node.col_offset,
                                    "message": f"Unused variable '{target.name}'",
                                    "severity": "info",
                                    "category": "unused_code",
                                    "code": "unused_variable"
                                })
            
            return {
                "status": "success",
                "issues": issues,
                "ast_info": {
                    "total_nodes": len(list(astroid.walk(tree))),
                    "functions": len([n for n in astroid.walk(tree) if isinstance(n, astroid.FunctionDef)]),
                    "classes": len([n for n in astroid.walk(tree) if isinstance(n, astroid.ClassDef)]),
                    "imports": len([n for n in astroid.walk(tree) if isinstance(n, (astroid.Import, astroid.ImportFrom))])
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


    def _run_jedi_analysis(self, file_path: str) -> Dict[str, Any]:
        """Run Jedi analysis on a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create Jedi script
            script = jedi.Script(content, path=file_path)
            
            issues = []
            
            # Analyze for completion issues and undefined names
            try:
                # Get all names in the file
                names = script.get_names()
                
                for name in names:
                    # Check if the name is defined
                    if name.type == 'name' and not name.is_definition():
                        # Try to get completions for this name
                        completions = script.complete(name.line, name.column)
                        if not completions:
                            issues.append({
                                "line": name.line,
                                "column": name.column,
                                "message": f"Undefined name '{name.name}'",
                                "severity": "warning",
                                "category": "undefined_name",
                                "code": "undefined_name"
                            })
            except Exception:
                pass  # Jedi analysis might fail for some files
            
            # Analyze for import issues
            try:
                imports = [name for name in script.get_names() if name.type == 'import']
                for imp in imports:
                    # Check if the import can be resolved
                    completions = script.complete(imp.line, imp.column)
                    if not completions:
                        issues.append({
                            "line": imp.line,
                            "column": imp.column,
                            "message": f"Import cannot be resolved: {imp.name}",
                            "severity": "warning",
                            "category": "import_issue",
                            "code": "unresolved_import"
                        })
            except Exception:
                pass
            
            return {
                "status": "success",
                "issues": issues,
                "jedi_info": {
                    "total_names": len(script.get_names()),
                    "imports": len([n for n in script.get_names() if n.type == 'import']),
                    "functions": len([n for n in script.get_names() if n.type == 'function']),
                    "classes": len([n for n in script.get_names() if n.type == 'class'])
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_custom_ast_analysis(self, file_path: str) -> Dict[str, Any]:
        """Run custom AST analysis for complexity and code quality."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, file_path)
            
            complexity_issues = []
            function_metrics = []
            class_metrics = []
            
            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Calculate comprehensive metrics
                    metrics = self._calculate_function_metrics(node, content)
                    function_metrics.append(metrics)
                    
                    # Check cyclomatic complexity
                    if metrics["cyclomatic_complexity"] > 10:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' has high cyclomatic complexity ({metrics['cyclomatic_complexity']})",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "high_complexity",
                            "complexity_score": metrics["cyclomatic_complexity"]
                        })
                    
                    # Check parameter count
                    if metrics["parameter_count"] > 5:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' has too many parameters ({metrics['parameter_count']})",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "too_many_parameters"
                        })
                    
                    # Check nesting depth
                    if metrics["nesting_depth"] > 4:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' has deep nesting ({metrics['nesting_depth']} levels)",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "deep_nesting"
                        })
                
                # Analyze classes
                elif isinstance(node, ast.ClassDef):
                    class_metric = self._calculate_class_metrics(node, content)
                    class_metrics.append(class_metric)
                    
                    # Check class complexity
                    if class_metric["complexity"] > 10:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Class '{node.name}' has high complexity ({class_metric['complexity']})",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "high_class_complexity"
                        })
                
                # Check for long lines (approximate)
                if hasattr(node, 'lineno'):
                    line_content = content.split('\n')[node.lineno - 1] if node.lineno <= len(content.split('\n')) else ""
                    if len(line_content) > 120:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Line too long ({len(line_content)} characters)",
                            "severity": "info",
                            "category": "style",
                            "code": "line_too_long"
                        })
            
            # Calculate file-level metrics
            file_metrics = self._calculate_file_metrics(content, function_metrics, class_metrics)
            
            return {
                "status": "success",
                "complexity_issues": complexity_issues,
                "function_metrics": function_metrics,
                "class_metrics": class_metrics,
                "file_metrics": file_metrics,
                "ast_info": {
                    "total_nodes": len(list(ast.walk(tree))),
                    "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                    "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                    "imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _get_nesting_level(self, node) -> int:
        """Calculate the nesting level of a node."""
        if not ASTROID_AVAILABLE:
            return 0
        
        level = 0
        current = node
        while current.parent:
            if isinstance(current, (astroid.If, astroid.For, astroid.While, astroid.TryExcept, astroid.With)):
                level += 1
            current = current.parent
        return level

    def _is_variable_used(self, var_name: str, assign_node) -> bool:
        """Check if a variable is used after assignment."""
        if not ASTROID_AVAILABLE:
            return True  # Assume it's used if astroid is not available
        
        # This is a simplified check - in practice, you'd need more sophisticated analysis
        try:
            # Look for uses of the variable in the same scope
            scope = assign_node.scope()
            for node in astroid.walk(scope):
                if isinstance(node, astroid.Name) and node.name == var_name and node != assign_node:
                    return True
            return False
        except Exception:
            return True  # Assume it's used if we can't determine

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity

    def _calculate_function_metrics(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a function."""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line
        
        # Extract function source
        function_source = '\n'.join(lines[start_line:end_line + 1])
        
        # Calculate metrics
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(node)
        parameter_count = len(node.args.args)
        nesting_depth = self._calculate_nesting_depth(node)
        lines_of_code = end_line - start_line + 1
        
        # Calculate Halstead metrics (simplified)
        halstead_metrics = self._calculate_halstead_metrics(function_source)
        
        # Calculate maintainability index (simplified)
        maintainability_index = self._calculate_maintainability_index(
            cyclomatic_complexity, lines_of_code, parameter_count
        )
        
        return {
            "name": node.name,
            "line_number": node.lineno,
            "cyclomatic_complexity": cyclomatic_complexity,
            "parameter_count": parameter_count,
            "nesting_depth": nesting_depth,
            "lines_of_code": lines_of_code,
            "halstead_metrics": halstead_metrics,
            "maintainability_index": maintainability_index,
            "return_points": self._count_return_points(node)
        }

    def _calculate_class_metrics(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a class."""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line
        
        # Count methods
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        method_count = len(methods)
        
        # Calculate class complexity (sum of method complexities)
        total_complexity = sum(self._calculate_cyclomatic_complexity(method) for method in methods)
        
        # Calculate inheritance depth
        inheritance_depth = len(node.bases)
        
        return {
            "name": node.name,
            "line_number": node.lineno,
            "method_count": method_count,
            "complexity": total_complexity,
            "inheritance_depth": inheritance_depth,
            "lines_of_code": end_line - start_line + 1
        }

    def _calculate_file_metrics(self, content: str, function_metrics: List[Dict], class_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate file-level metrics."""
        lines = content.split('\n')
        
        # Basic line metrics
        total_lines = len(lines)
        source_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        blank_lines = len([l for l in lines if not l.strip()])
        
        # Complexity metrics
        all_complexities = [f["cyclomatic_complexity"] for f in function_metrics]
        avg_complexity = sum(all_complexities) / len(all_complexities) if all_complexities else 0
        max_complexity = max(all_complexities) if all_complexities else 0
        
        # Calculate file maintainability index
        file_mi = self._calculate_file_maintainability_index(function_metrics, source_lines)
        
        return {
            "total_lines": total_lines,
            "source_lines": source_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "function_count": len(function_metrics),
            "class_count": len(class_metrics),
            "average_complexity": avg_complexity,
            "max_complexity": max_complexity,
            "maintainability_index": file_mi
        }

    def _calculate_nesting_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth in a function."""
        max_depth = 0
        current_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.With)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.With)):
                current_depth -= 1
        
        return max_depth

    def _calculate_halstead_metrics(self, source: str) -> Dict[str, float]:
        """Calculate simplified Halstead metrics."""
        # This is a simplified implementation
        # In a full implementation, you'd use a proper Halstead metrics library
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not']
        operands = []
        
        # Count operators and operands (simplified)
        operator_count = sum(source.count(op) for op in operators)
        operand_count = len([word for word in source.split() if word.isalnum() and not word in operators])
        
        # Calculate metrics
        n1 = len(set(op for op in operators if op in source))  # Unique operators
        n2 = len(set(word for word in source.split() if word.isalnum() and not word in operators))  # Unique operands
        N1 = operator_count  # Total operators
        N2 = operand_count  # Total operands
        
        if n1 == 0 or n2 == 0:
            return {
                "volume": 0.0,
                "difficulty": 0.0,
                "effort": 0.0,
                "time": 0.0,
                "bugs": 0.0
            }
        
        volume = (N1 + N2) * (n1 + n2).bit_length()  # Simplified volume calculation
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = volume * difficulty
        time = effort / 18  # Simplified time calculation
        bugs = volume / 3000  # Simplified bugs calculation
        
        return {
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
            "time": time,
            "bugs": bugs
        }

    def _calculate_maintainability_index(self, complexity: int, lines_of_code: int, parameter_count: int) -> float:
        """Calculate maintainability index for a function."""
        # Simplified maintainability index calculation
        # Based on cyclomatic complexity, lines of code, and parameter count
        if lines_of_code == 0:
            return 100.0
        
        # Penalty factors
        complexity_penalty = min(50, complexity * 2)
        loc_penalty = min(30, lines_of_code / 10)
        param_penalty = min(20, parameter_count * 2)
        
        mi = max(0, 100 - complexity_penalty - loc_penalty - param_penalty)
        return mi

    def _calculate_file_maintainability_index(self, function_metrics: List[Dict], source_lines: int) -> float:
        """Calculate maintainability index for a file."""
        if not function_metrics or source_lines == 0:
            return 100.0
        
        total_complexity = sum(f["cyclomatic_complexity"] for f in function_metrics)
        avg_complexity = total_complexity / len(function_metrics)
        
        # Simplified file-level maintainability index
        complexity_penalty = min(40, avg_complexity * 3)
        loc_penalty = min(30, source_lines / 20)
        
        mi = max(0, 100 - complexity_penalty - loc_penalty)
        return mi

    def _count_return_points(self, node: ast.FunctionDef) -> int:
        """Count the number of return statements in a function."""
        return len([n for n in ast.walk(node) if isinstance(n, ast.Return)])

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive AST analysis report."""
        return {
            "analyzer": "ASTAnalysisAnalyzer",
            "tools_used": ["astroid", "jedi", "custom_ast"],
            "tools_available": {
                "astroid": ASTROID_AVAILABLE,
                "jedi": JEDI_AVAILABLE
            },
            "results": self.results,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"status": "no_analysis_performed"}
        
        total_files = 0
        total_issues = 0
        complexity_issues = 0
        refactoring_opportunities = 0
        
        for file_result in self.results.get("files", {}).values():
            total_files += 1
            summary = file_result.get("summary", {})
            total_issues += summary.get("total_issues", 0)
            complexity_issues += summary.get("complexity_issues", 0)
            refactoring_opportunities += summary.get("refactoring_opportunities", 0)
        
        return {
            "total_files_analyzed": total_files,
            "total_issues_found": total_issues,
            "complexity_issues": complexity_issues,
            "refactoring_opportunities": refactoring_opportunities,
            "average_issues_per_file": total_issues / total_files if total_files > 0 else 0
        }