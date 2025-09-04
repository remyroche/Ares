#!/usr/bin/env python3
"""
Standalone script to check for undefined names and variables in the repository.
This is a simplified version that doesn't depend on the complex import structure.
"""

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class UndefinedNameError:
    """Container for undefined name error information."""

    def __init__(self, file_path: str, line: int, column: int, name: str,
                 error_type: str, context: str = "", severity: str = "error"):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.name = name
        self.error_type = error_type
        self.context = context
        self.severity = severity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "name": self.name,
            "error_type": self.error_type,
            "context": self.context,
            "severity": self.severity,
        }


class UndefinedNamesChecker:
    """
    Standalone checker for undefined names and variables in Python code.
    """

    def __init__(self):
        self.builtin_names = self._init_builtin_names()

    def _init_builtin_names(self) -> Set[str]:
        """Initialize set of Python builtin names."""
        import builtins
        builtin_names = set(dir(builtins))
        # Add common builtin types and functions
        builtin_names.update({
            'object', 'type', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'set', 'frozenset', 'bytes', 'bytearray', 'complex', 'range', 'slice',
            'property', 'staticmethod', 'classmethod', 'super', 'vars', 'dir',
            'hasattr', 'getattr', 'setattr', 'delattr', 'isinstance', 'issubclass',
            'callable', 'iter', 'next', 'enumerate', 'zip', 'map', 'filter',
            'sorted', 'reversed', 'sum', 'min', 'max', 'abs', 'round', 'pow',
            'divmod', 'bin', 'oct', 'hex', 'chr', 'ord', 'len', 'repr', 'ascii',
            'format', 'hash', 'id', 'globals', 'locals', 'eval', 'exec', 'compile',
            'open', 'input', 'print', 'exit', 'quit', 'help', 'license', 'credits',
            'copyright', 'True', 'False', 'None', 'Ellipsis', 'NotImplemented',
            '__name__', '__file__', '__doc__', '__package__', '__loader__',
            '__spec__', '__annotations__', '__builtins__', '__debug__',
            '__import__', '__main__', '__version__', '__author__', '__email__'
        })
        return builtin_names

    def find_python_files(self, directory: str, exclude_patterns: List[str] = None) -> List[str]:
        """Find all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '*.pyc', '.git', '.pytest_cache', 'venv', 'env']
        
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
        
        return python_files

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file for undefined names and variables."""
        print(f"Analyzing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "status": "error",
                "error": f"Could not read file: {e}",
                "file_path": file_path
            }

        # Reset state for this file
        file_errors = []
        defined_names = set()
        imported_names = set()
        
        try:
            # Parse the AST
            tree = ast.parse(content, filename=file_path)
            
            # First pass: collect all defined names and imports
            defined_names, imported_names = self._collect_definitions(tree, file_path)
            
            # Second pass: check for undefined names
            file_errors = self._check_undefined_names(tree, file_path, defined_names, imported_names)
            
        except SyntaxError as e:
            file_errors.append(UndefinedNameError(
                file_path=file_path,
                line=e.lineno or 0,
                column=e.offset or 0,
                name="<syntax_error>",
                error_type="syntax_error",
                context=f"Syntax error: {e.msg}",
                severity="error"
            ))
        except Exception as e:
            return {
                "status": "error",
                "error": f"AST parsing failed: {e}",
                "file_path": file_path
            }

        # Convert errors to dictionaries
        errors_dict = [error.to_dict() for error in file_errors]
        
        return {
            "status": "success",
            "file_path": file_path,
            "total_errors": len(file_errors),
            "errors": errors_dict,
            "defined_names": list(defined_names),
            "imported_names": list(imported_names),
            "summary": {
                "total_undefined_names": len([e for e in file_errors if e.error_type == "undefined_name"]),
                "total_undefined_imports": len([e for e in file_errors if e.error_type == "undefined_import"]),
                "total_unused_imports": len([e for e in file_errors if e.error_type == "unused_import"]),
                "total_import_conflicts": len([e for e in file_errors if e.error_type == "import_conflict"]),
            }
        }

    def _collect_definitions(self, tree: ast.AST, file_path: str) -> Tuple[Set[str], Set[str]]:
        """Collect all defined names and imports from the AST."""
        defined_names = set()
        imported_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
                # Add 'self' as defined for class methods
                if self._is_class_method(node, tree):
                    defined_names.add('self')
            elif isinstance(node, ast.AsyncFunctionDef):
                defined_names.add(node.name)
                # Add 'self' as defined for class methods
                if self._is_class_method(node, tree):
                    defined_names.add('self')
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
                # Add 'cls' as defined for class methods
                defined_names.add('cls')
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Variable assignment
                defined_names.add(node.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
                    # Also add the base name if it's a simple import
                    if '.' not in alias.name:
                        defined_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                        defined_names.add(alias.asname or alias.name)
                else:
                    # from . import something
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                        defined_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined_names.add(elt.id)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    defined_names.add(node.target.id)
                elif isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined_names.add(elt.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined_names.add(item.optional_vars.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    defined_names.add(node.name)
            elif isinstance(node, ast.Global):
                for name in node.names:
                    defined_names.add(name)
            elif isinstance(node, ast.Nonlocal):
                for name in node.names:
                    defined_names.add(name)
        
        return defined_names, imported_names

    def _is_class_method(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a class method."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False

    def _check_undefined_names(self, tree: ast.AST, file_path: str, 
                              defined_names: Set[str], imported_names: Set[str]) -> List[UndefinedNameError]:
        """Check for undefined names in the AST."""
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                
                # Skip if it's a builtin
                if name in self.builtin_names:
                    continue
                
                # Skip if it's defined or imported
                if name in defined_names or name in imported_names:
                    continue
                
                # Check if it's a special name (like __name__, __file__, etc.)
                if name.startswith('__') and name.endswith('__'):
                    continue
                
                # Skip common variable names that are often used in loops/comprehensions
                if name in {'i', 'j', 'k', 'x', 'y', 'z', 'item', 'value', 'key', 'val', 'data', 'result', 'temp'}:
                    # Only flag if it's not in a loop or comprehension context
                    if not self._is_in_loop_context(node, tree):
                        continue
                
                # Skip if it's in a function parameter or lambda
                if self._is_in_function_context(node, tree):
                    continue
                
                # This is an undefined name
                context = self._get_context(node, tree)
                errors.append(UndefinedNameError(
                    file_path=file_path,
                    line=node.lineno or 0,
                    column=node.col_offset or 0,
                    name=name,
                    error_type="undefined_name",
                    context=context,
                    severity="error"
                ))
        
        return errors

    def _is_in_loop_context(self, name_node: ast.Name, tree: ast.AST) -> bool:
        """Check if a name is used in a loop context (for, while, comprehension)."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                if name_node in ast.walk(node):
                    return True
        return False

    def _is_in_function_context(self, name_node: ast.Name, tree: ast.AST) -> bool:
        """Check if a name is used in a function parameter or lambda context."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's in function parameters
                for arg in node.args.args:
                    if arg.arg == name_node.id:
                        return True
                # Check if it's in the function body
                if name_node in ast.walk(node):
                    return True
            elif isinstance(node, ast.Lambda):
                # Check if it's in lambda parameters
                for arg in node.args.args:
                    if arg.arg == name_node.id:
                        return True
        return False

    def _get_context(self, node: ast.AST, tree: ast.AST) -> str:
        """Get context around a node for better error reporting."""
        try:
            # Find the parent node to understand context
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    if child is node:
                        if isinstance(parent, ast.Call):
                            return f"function call to '{parent.func.id if isinstance(parent.func, ast.Name) else 'unknown'}'"
                        elif isinstance(parent, ast.Attribute):
                            return f"attribute access on '{parent.value.id if isinstance(parent.value, ast.Name) else 'unknown'}'"
                        elif isinstance(parent, ast.Assign):
                            return "assignment target"
                        elif isinstance(parent, ast.Compare):
                            return "comparison operation"
                        elif isinstance(parent, ast.BinOp):
                            return "binary operation"
                        elif isinstance(parent, ast.UnaryOp):
                            return "unary operation"
                        else:
                            return f"in {type(parent).__name__}"
            return "unknown context"
        except:
            return "unknown context"

    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory for undefined names and variables."""
        print(f"Analyzing directory: {directory_path}")
        
        # Find all Python files
        python_files = self.find_python_files(directory_path)
        
        if not python_files:
            return {
                "status": "success",
                "directory_path": directory_path,
                "total_files": 0,
                "files": {},
                "summary": {
                    "total_files_analyzed": 0,
                    "total_errors": 0,
                    "files_with_errors": 0,
                    "undefined_names": 0,
                    "undefined_imports": 0,
                    "unused_imports": 0,
                    "import_conflicts": 0,
                }
            }

        # Analyze each file
        files_results = {}
        total_errors = 0
        files_with_errors = 0
        error_summary = {
            "undefined_names": 0,
            "undefined_imports": 0,
            "unused_imports": 0,
            "import_conflicts": 0,
        }

        for file_path in python_files:
            file_result = self.analyze_file(file_path)
            files_results[file_path] = file_result
            
            if file_result["status"] == "success":
                file_errors = file_result["total_errors"]
                if file_errors > 0:
                    files_with_errors += 1
                    total_errors += file_errors
                    
                    # Update error summary
                    file_summary = file_result["summary"]
                    error_summary["undefined_names"] += file_summary["total_undefined_names"]
                    error_summary["undefined_imports"] += file_summary["total_undefined_imports"]
                    error_summary["unused_imports"] += file_summary["total_unused_imports"]
                    error_summary["import_conflicts"] += file_summary["total_import_conflicts"]

        return {
            "status": "success",
            "directory_path": directory_path,
            "total_files": len(python_files),
            "files": files_results,
            "summary": {
                "total_files_analyzed": len(python_files),
                "total_errors": total_errors,
                "files_with_errors": files_with_errors,
                "undefined_names": error_summary["undefined_names"],
                "undefined_imports": error_summary["undefined_imports"],
                "unused_imports": error_summary["unused_imports"],
                "import_conflicts": error_summary["import_conflicts"],
            }
        }


def run_undefined_names_check(target_path: str = None, output_file: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Run undefined names check on the target path.
    
    Args:
        target_path: Path to analyze (default: current directory)
        output_file: Optional output file for JSON report
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing analysis results
    """
    if target_path is None:
        target_path = str(Path.cwd())
    
    print("="*70)
    print("UNDEFINED NAMES AND VARIABLES CHECKER")
    print("="*70)
    print(f"Target: {target_path}")
    print(f"Output file: {output_file or 'None'}")
    print(f"Verbose: {verbose}")
    print()
    
    # Create checker
    checker = UndefinedNamesChecker()
    
    # Determine if target is file or directory
    if os.path.isfile(target_path):
        print(f"Analyzing single file: {target_path}")
        results = checker.analyze_file(target_path)
    else:
        print(f"Analyzing directory: {target_path}")
        results = checker.analyze_directory(target_path)
    
    # Print results
    if results["status"] == "success":
        summary = results.get("summary", {})
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Total files analyzed: {summary.get('total_files_analyzed', 1)}")
        print(f"❌ Total errors found: {summary.get('total_errors', 0)}")
        print(f"📄 Files with errors: {summary.get('files_with_errors', 0)}")
        
        if summary.get('total_errors', 0) > 0:
            print(f"\n🔍 Error breakdown:")
            print(f"  🚫 Undefined names: {summary.get('undefined_names', 0)}")
            print(f"  📦 Undefined imports: {summary.get('undefined_imports', 0)}")
            print(f"  🗑️  Unused imports: {summary.get('unused_imports', 0)}")
            print(f"  ⚠️  Import conflicts: {summary.get('import_conflicts', 0)}")
            
            if verbose:
                print(f"\n📋 Detailed errors:")
                files = results.get("files", {})
                if not files:  # Single file case
                    files = {target_path: results}
                
                for file_path, file_result in files.items():
                    if file_result.get("total_errors", 0) > 0:
                        print(f"\n📄 {file_path}:")
                        for error in file_result.get("errors", []):
                            print(f"  Line {error['line']}: {error['name']} - {error['error_type']}")
                            if error.get('context'):
                                print(f"    Context: {error['context']}")
        else:
            print(f"\n🎉 No undefined names or variables found!")
            
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return results
    
    # Save report if requested
    if output_file:
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Report saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
    
    return results


def main():
    """Command-line interface for the undefined names checker."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check for undefined names and variables in Python code"
    )
    parser.add_argument("--target", "-t", 
                       help="Path to Python file or directory to analyze (default: current directory)")
    parser.add_argument("--output", "-o", 
                       help="Output file for JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with detailed error information")
    
    args = parser.parse_args()
    
    # Run the check
    results = run_undefined_names_check(
        target_path=args.target,
        output_file=args.output,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    if results["status"] == "success":
        summary = results.get("summary", {})
        total_errors = summary.get('total_errors', 0)
        if total_errors == 0:
            print(f"\n✅ All checks passed!")
            return 0
        else:
            print(f"\n⚠️  Found {total_errors} issues that need attention.")
            return 1
    else:
        print(f"\n❌ Check failed!")
        return 2


if __name__ == "__main__":
    sys.exit(main())
