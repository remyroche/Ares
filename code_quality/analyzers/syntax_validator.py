"""
Syntax Validator - Comprehensive Python syntax validation, AST parsing, and compilation checks.
"""

import os
import ast
import sys
import tokenize
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from minimal_config import CodeQualityConfig, get_default_config
from minimal_file_utils import find_python_files


class SyntaxError:
    """Container for syntax error information."""
    
    def __init__(self, file_path: str, line: int, column: int, message: str, 
                 error_type: str, severity: str = "error"):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.message = message
        self.error_type = error_type
        self.severity = severity
    
    def __repr__(self):
        return f"SyntaxError({self.file_path}:{self.line}:{self.column}, {self.message})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "error_type": self.error_type,
            "severity": self.severity
        }


class ASTNode:
    """Container for AST node information."""
    
    def __init__(self, node_type: str, name: str = "", line: int = 0, 
                 column: int = 0, parent: Optional[str] = None):
        self.node_type = node_type
        self.name = name
        self.line = line
        self.column = column
        self.parent = parent
        self.children: List[str] = []
    
    def __repr__(self):
        return f"ASTNode({self.node_type}:{self.name}@{self.line}:{self.column})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_type": self.node_type,
            "name": self.name,
            "line": self.line,
            "column": self.column,
            "parent": self.parent,
            "children": self.children
        }


class SyntaxValidator:
    """
    Comprehensive Python syntax validation, AST parsing, and compilation checker.
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.syntax_errors: List[SyntaxError] = []
        self.ast_nodes: Dict[str, ASTNode] = {}
        self.compilation_results: Dict[str, Dict[str, Any]] = {}
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        
    def validate_directory(self, directory: str) -> Dict[str, Any]:
        """
        Validate syntax for all Python files in a directory.
        
        Args:
            directory: Directory containing Python files to validate
            
        Returns:
            Dictionary containing validation results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Validating syntax for {len(python_files)} Python files...")
        
        # Clear previous results
        self.syntax_errors.clear()
        self.ast_nodes.clear()
        self.compilation_results.clear()
        self.file_stats.clear()
        
        for file_path in python_files:
            self._validate_file(file_path)
        
        # Generate comprehensive report
        validation_results = self._generate_validation_report()
        
        return validation_results
    
    def _validate_file(self, file_path: str) -> None:
        """Validate a single Python file."""
        file_stats = {
            "file_path": file_path,
            "file_size": 0,
            "line_count": 0,
            "syntax_valid": False,
            "ast_parseable": False,
            "compilable": False,
            "encoding": "unknown",
            "syntax_errors": [],
            "ast_nodes": [],
            "compilation_issues": []
        }
        
        try:
            # Get file info
            file_size = os.path.getsize(file_path)
            file_stats["file_size"] = file_size
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                file_stats["line_count"] = len(lines)
                file_stats["encoding"] = "utf-8"
            
            # Check if file is empty
            if not content.strip():
                file_stats["syntax_valid"] = True
                file_stats["ast_parseable"] = True
                file_stats["compilable"] = True
                self.file_stats[file_path] = file_stats
                return
            
            # 1. Basic syntax validation with tokenize
            tokenize_errors = self._validate_with_tokenize(file_path, content)
            file_stats["syntax_errors"].extend(tokenize_errors)
            
            # 2. AST parsing validation
            ast_errors, ast_nodes = self._validate_with_ast(file_path, content)
            file_stats["syntax_errors"].extend(ast_errors)
            file_stats["ast_nodes"] = ast_nodes
            
            # 3. Compilation validation
            compilation_issues = self._validate_compilation(file_path, content)
            file_stats["compilation_issues"] = compilation_issues
            
            # 4. Advanced syntax checks
            advanced_errors = self._advanced_syntax_checks(file_path, content)
            file_stats["syntax_errors"].extend(advanced_errors)
            
            # Determine overall validity
            file_stats["syntax_valid"] = len(file_stats["syntax_errors"]) == 0
            file_stats["ast_parseable"] = len(ast_errors) == 0
            file_stats["compilable"] = len(compilation_issues) == 0
            
            # Add to global results
            self.syntax_errors.extend(file_stats["syntax_errors"])
            for node_id, node in ast_nodes.items():
                self.ast_nodes[node_id] = node
            
            self.file_stats[file_path] = file_stats
            
        except Exception as e:
            # File couldn't be read or processed
            error = SyntaxError(
                file_path=file_path,
                line=0,
                column=0,
                message=f"File processing error: {str(e)}",
                error_type="file_error",
                severity="error"
            )
            file_stats["syntax_errors"].append(error.to_dict())
            file_stats["syntax_valid"] = False
            file_stats["ast_parseable"] = False
            file_stats["compilable"] = False
            
            self.syntax_errors.append(error)
            self.file_stats[file_path] = file_stats
    
    def _validate_with_tokenize(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Validate Python syntax using the tokenize module."""
        errors = []
        
        try:
            # Try to tokenize the content
            tokens = list(tokenize.tokenize(iter(content.splitlines(keepends=True))))
            
            # Check for common tokenization issues
            for i, token in enumerate(tokens):
                if token.type == tokenize.ERRORTOKEN:
                    errors.append({
                        "file_path": file_path,
                        "line": token.start[0],
                        "column": token.start[1],
                        "message": f"Invalid token: {token.string}",
                        "error_type": "tokenization_error",
                        "severity": "error"
                    })
                
                # Check for unmatched parentheses/brackets
                elif token.string in '([{':
                    # Look for matching closing token
                    if not self._find_matching_token(tokens, i, token.string):
                        errors.append({
                            "file_path": file_path,
                            "line": token.start[0],
                            "column": token.start[1],
                            "message": f"Unmatched opening {token.string}",
                            "error_type": "unmatched_bracket",
                            "severity": "error"
                        })
                        
        except tokenize.TokenError as e:
            errors.append({
                "file_path": file_path,
                "line": 0,
                "column": 0,
                "message": f"Tokenization error: {str(e)}",
                "error_type": "tokenization_error",
                "severity": "error"
            })
        except Exception as e:
            errors.append({
                "file_path": file_path,
                "line": 0,
                "column": 0,
                "message": f"Unexpected tokenization error: {str(e)}",
                "error_type": "tokenization_error",
                "severity": "error"
            })
        
        return errors
    
    def _find_matching_token(self, tokens: List[tokenize.TokenInfo], start_idx: int, 
                           opening: str) -> bool:
        """Find matching closing token for opening bracket/parenthesis."""
        closing_map = {'(': ')', '[': ']', '{': '}'}
        closing = closing_map.get(opening)
        if not closing:
            return False
        
        stack = 1
        for i in range(start_idx + 1, len(tokens)):
            if tokens[i].string == opening:
                stack += 1
            elif tokens[i].string == closing:
                stack -= 1
                if stack == 0:
                    return True
        
        return False
    
    def _validate_with_ast(self, file_path: str, content: str) -> Tuple[List[Dict[str, Any]], Dict[str, ASTNode]]:
        """Validate Python syntax using AST parsing."""
        errors = []
        ast_nodes = {}
        node_counter = 0
        
        try:
            # Parse the AST
            tree = ast.parse(content, filename=file_path)
            
            # Walk through the AST and collect node information
            for node in ast.walk(tree):
                node_id = f"{file_path}_{node_counter}"
                node_counter += 1
                
                # Get node type and position
                node_type = type(node).__name__
                line = getattr(node, 'lineno', 0)
                col = getattr(node, 'col_offset', 0)
                
                # Get node name if available
                name = ""
                if hasattr(node, 'name'):
                    name = node.name
                elif hasattr(node, 'id'):
                    name = node.id
                elif hasattr(node, 'attr'):
                    name = node.attr
                
                # Create AST node
                ast_node = ASTNode(
                    node_type=node_type,
                    name=name,
                    line=line,
                    column=col
                )
                
                ast_nodes[node_id] = ast_node
                
                # Check for specific AST issues
                if isinstance(node, ast.FunctionDef):
                    # Check function definition issues
                    if not name or name.startswith('_'):
                        continue  # Skip private functions
                    
                    # Check for empty function bodies
                    if not node.body:
                        errors.append({
                            "file_path": file_path,
                            "line": line,
                            "column": col,
                            "message": f"Empty function definition: {name}",
                            "error_type": "empty_function",
                            "severity": "warning"
                        })
                
                elif isinstance(node, ast.ClassDef):
                    # Check class definition issues
                    if not name or name.startswith('_'):
                        continue  # Skip private classes
                    
                    # Check for empty class bodies
                    if not node.body:
                        errors.append({
                            "file_path": file_path,
                            "line": line,
                            "column": col,
                            "message": f"Empty class definition: {name}",
                            "error_type": "empty_class",
                            "severity": "warning"
                        })
                
                elif isinstance(node, ast.Import):
                    # Check import issues
                    for alias in node.names:
                        if not alias.name:
                            errors.append({
                                "file_path": file_path,
                                "line": line,
                                "column": col,
                                "message": "Empty import name",
                                "error_type": "empty_import",
                                "severity": "error"
                            })
                
                elif isinstance(node, ast.ImportFrom):
                    # Check from import issues
                    if not node.module:
                        errors.append({
                            "file_path": file_path,
                            "line": line,
                            "column": col,
                            "message": "Empty module name in from import",
                            "error_type": "empty_from_import",
                            "severity": "error"
                        })
            
        except SyntaxError as e:
            errors.append({
                "file_path": file_path,
                "line": e.lineno or 0,
                "column": e.offset or 0,
                "message": f"Syntax error: {str(e)}",
                "error_type": "syntax_error",
                "severity": "error"
            })
        except Exception as e:
            errors.append({
                "file_path": file_path,
                "line": 0,
                "column": 0,
                "message": f"AST parsing error: {str(e)}",
                "error_type": "ast_error",
                "severity": "error"
            })
        
        return errors, ast_nodes
    
    def _validate_compilation(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Validate that Python code can be compiled."""
        issues = []
        
        try:
            # Try to compile the code
            compile_result = compile(content, file_path, 'exec')
            
            # If we get here, compilation succeeded
            # Check for some runtime issues that might not be caught by AST
            if 'exec(' in content or 'eval(' in content:
                issues.append({
                    "file_path": file_path,
                    "line": 0,
                    "column": 0,
                    "message": "Code contains exec() or eval() calls (security risk)",
                    "error_type": "security_warning",
                    "severity": "warning"
                })
            
        except SyntaxError as e:
            issues.append({
                "file_path": file_path,
                "line": e.lineno or 0,
                "column": e.offset or 0,
                "message": f"Compilation error: {str(e)}",
                "error_type": "compilation_error",
                "severity": "error"
            })
        except Exception as e:
            issues.append({
                "file_path": file_path,
                "line": 0,
                "column": 0,
                "message": f"Compilation error: {str(e)}",
                "error_type": "compilation_error",
                "severity": "error"
            })
        
        return issues
    
    def _advanced_syntax_checks(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Perform advanced syntax and style checks."""
        issues = []
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            # Check for common issues
            
            # Trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    "file_path": file_path,
                    "line": line_num,
                    "column": len(line.rstrip()),
                    "message": "Trailing whitespace",
                    "error_type": "style_issue",
                    "severity": "warning"
                })
            
            # Mixed tabs and spaces
            if '\t' in line and ' ' in line:
                issues.append({
                    "file_path": file_path,
                    "line": line_num,
                    "column": 0,
                    "message": "Mixed tabs and spaces",
                    "error_type": "style_issue",
                    "severity": "warning"
                })
            
            # Line too long (configurable)
            max_line_length = self.config.auto_fix.max_line_length
            if len(line) > max_line_length:
                issues.append({
                    "file_path": file_path,
                    "line": line_num,
                    "column": max_line_length,
                    "message": f"Line too long ({len(line)} > {max_line_length})",
                    "error_type": "style_issue",
                    "severity": "warning"
                })
            
            # Check for common Python anti-patterns
            if 'import *' in line:
                issues.append({
                    "file_path": file_path,
                    "line": line_num,
                    "column": line.find('import *'),
                    "message": "Wildcard import (import *) is discouraged",
                    "error_type": "style_issue",
                    "severity": "warning"
                })
            
            # Check for unused imports (basic check)
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_name = line.split()[1] if line.strip().startswith('import ') else line.split()[3]
                if not self._is_import_used(content, import_name):
                    issues.append({
                        "file_path": file_path,
                        "line": line_num,
                        "column": 0,
                        "message": f"Potentially unused import: {import_name}",
                        "error_type": "unused_import",
                        "severity": "warning"
                    })
        
        return issues
    
    def _is_import_used(self, content: str, import_name: str) -> bool:
        """Check if an import is used in the content."""
        # Simple check - look for the import name in the content
        # This is a basic implementation and might have false positives
        lines = content.splitlines()
        
        # Skip import lines
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                continue
            
            # Check if the import name appears in the line
            if import_name in line and not line.strip().startswith('#'):
                return True
        
        return False
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        # Count errors by type
        error_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in self.syntax_errors:
            error_counts[error.error_type] += 1
            severity_counts[error.severity] += 1
        
        # Count files by status
        total_files = len(self.file_stats)
        valid_files = len([f for f in self.file_stats.values() if f["syntax_valid"]])
        ast_parseable_files = len([f for f in self.file_stats.values() if f["ast_parseable"]])
        compilable_files = len([f for f in self.file_stats.values() if f["compilable"]])
        
        # Group errors by file
        errors_by_file = defaultdict(list)
        for error in self.syntax_errors:
            errors_by_file[error.file_path].append(error.to_dict())
        
        # Group errors by directory
        errors_by_directory = defaultdict(lambda: {"files": 0, "errors": 0, "warnings": 0})
        for error in self.syntax_errors:
            dir_path = str(Path(error.file_path).parent)
            errors_by_directory[dir_path]["files"] += 1
            if error.severity == "error":
                errors_by_directory[dir_path]["errors"] += 1
            else:
                errors_by_directory[dir_path]["warnings"] += 1
        
        report = {
            "summary": {
                "total_files": total_files,
                "valid_files": valid_files,
                "invalid_files": total_files - valid_files,
                "ast_parseable_files": ast_parseable_files,
                "compilable_files": compilable_files,
                "total_errors": len(self.syntax_errors),
                "total_ast_nodes": len(self.ast_nodes)
            },
            "error_counts": dict(error_counts),
            "severity_counts": dict(severity_counts),
            "errors_by_file": dict(errors_by_file),
            "errors_by_directory": dict(errors_by_directory),
            "file_details": self.file_stats,
            "ast_nodes": {node_id: node.to_dict() for node_id, node in self.ast_nodes.items()},
            "compilation_results": self.compilation_results
        }
        
        return report
    
    def get_file_errors(self, file_path: str) -> List[SyntaxError]:
        """Get all syntax errors for a specific file."""
        return [error for error in self.syntax_errors if error.file_path == file_path]
    
    def get_directory_errors(self, directory: str) -> List[SyntaxError]:
        """Get all syntax errors for a specific directory."""
        return [error for error in self.syntax_errors if error.file_path.startswith(directory)]
    
    def get_errors_by_type(self, error_type: str) -> List[SyntaxError]:
        """Get all errors of a specific type."""
        return [error for error in self.syntax_errors if error.error_type == error_type]
    
    def get_errors_by_severity(self, severity: str) -> List[SyntaxError]:
        """Get all errors of a specific severity."""
        return [error for error in self.syntax_errors if error.severity == severity]
    
    def export_report(self, output_path: str) -> None:
        """Export the validation report to JSON."""
        try:
            report = self._generate_validation_report()
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"Validation report exported to {output_path}")
            
        except Exception as e:
            print(f"Error exporting report: {e}")
    
    def print_summary(self) -> None:
        """Print a summary of validation results."""
        report = self._generate_validation_report()
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("SYNTAX VALIDATION SUMMARY")
        print("="*60)
        print(f"Total files analyzed: {summary['total_files']}")
        print(f"Valid files: {summary['valid_files']}")
        print(f"Invalid files: {summary['invalid_files']}")
        print(f"AST parseable files: {summary['ast_parseable_files']}")
        print(f"Compilable files: {summary['compilable_files']}")
        print(f"Total syntax errors: {summary['total_errors']}")
        print(f"Total AST nodes: {summary['total_ast_nodes']}")
        
        if report["error_counts"]:
            print(f"\nErrors by type:")
            for error_type, count in sorted(report["error_counts"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        if report["severity_counts"]:
            print(f"\nErrors by severity:")
            for severity, count in sorted(report["severity_counts"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {severity}: {count}")
        
        # Show top problematic files
        if report["errors_by_file"]:
            print(f"\nTop problematic files:")
            sorted_files = sorted(report["errors_by_file"].items(), 
                               key=lambda x: len(x[1]), reverse=True)
            for file_path, errors in sorted_files[:5]:
                print(f"  {file_path}: {len(errors)} issues")


def main():
    """Command-line interface for the syntax validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Python syntax and AST parsing")
    parser.add_argument("--path", required=True, help="Path to directory containing Python files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file for validation report (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        from ..core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Run syntax validation
    validator = SyntaxValidator(config)
    results = validator.validate_directory(args.path)
    
    # Print summary
    validator.print_summary()
    
    # Export results if requested
    if args.output:
        validator.export_report(args.output)
    
    # Exit with error code if there are syntax errors
    if results["summary"]["total_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()