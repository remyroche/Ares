"""
AST Analysis Analyzer - Integrates Astroid, Rope, and Jedi for advanced AST-based code analysis.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.config import CodeQualityConfig

try:
    import astroid
    ASTROID_AVAILABLE = True
except ImportError:
    ASTROID_AVAILABLE = False

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
            
            # Analyze cyclomatic complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    if complexity > 10:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' has high cyclomatic complexity ({complexity})",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "high_complexity",
                            "complexity_score": complexity
                        })
                
                # Check for long parameter lists
                if isinstance(node, ast.FunctionDef):
                    if len(node.args.args) > 5:
                        complexity_issues.append({
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                            "severity": "warning",
                            "category": "complexity",
                            "code": "too_many_parameters"
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
            
            return {
                "status": "success",
                "complexity_issues": complexity_issues,
                "ast_info": {
                    "total_nodes": len(list(ast.walk(tree))),
                    "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                    "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                    "imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _get_nesting_level(self, node: astroid.NodeNG) -> int:
        """Calculate the nesting level of a node."""
        level = 0
        current = node
        while current.parent:
            if isinstance(current, (astroid.If, astroid.For, astroid.While, astroid.TryExcept, astroid.With)):
                level += 1
            current = current.parent
        return level

    def _is_variable_used(self, var_name: str, assign_node: astroid.Assign) -> bool:
        """Check if a variable is used after assignment."""
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