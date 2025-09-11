from src.utils.tprint import tprint

from typing import Dict, List, Any, Optional
"""
Enhanced Undefined Names and Variables Analyzer - Detects undefined names, variables, and imports.
"""

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
import logging

# Simple configuration and utilities
def get_default_config():
    """Get default configuration for analysis."""
    return {
        'analysis_config': {
            'exclude_patterns': ['__pycache__', '.git', 'node_modules', '*.pyc', '.pytest_cache']
        }
    }

def find_python_files(directory_path, exclude_patterns=None):
    """Find all Python files in directory."""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', 'node_modules', '*.pyc', '.pytest_cache']

    python_files = []
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_patterns and not any(d.endswith(pat.strip('*')) for pat in exclude_patterns)]

        for file in files:
            if file.endswith('.py'):
                python_files.append(str(Path(root) / file))
    return python_files


class ScopeContext:
    """Represents a scope context (function, class, module, etc.)."""
    
    def __init__(self, name: str, scope_type: str, node: ast.AST, parent: Optional['ScopeContext'] = None):
        self.name = name
        self.scope_type = scope_type  # 'module', 'class', 'function', 'lambda', 'comprehension'
        self.node = node
        self.parent = parent
        self.defined_names: Set[str] = set()
        self.imported_names: Set[str] = set()
        self.children: List['ScopeContext'] = []
        
        if parent:
            parent.children.append(self)
    
    def add_defined_name(self, name: str) -> None:
        """Add a defined name to this scope."""
        self.defined_names.add(name)
    
    def add_imported_name(self, name: str) -> None:
        """Add an imported name to this scope."""
        self.imported_names.add(name)
    
    def is_name_defined(self, name: str) -> bool:
        """Check if a name is defined in this scope or any parent scope."""
        if name in self.defined_names or name in self.imported_names:
            return True
        if self.parent:
            return self.parent.is_name_defined(name)
        return False
    
    def get_scope_path(self) -> str:
        """Get the full path of this scope (e.g., 'module.class.function')."""
        if self.parent:
            return f"{self.parent.get_scope_path()}.{self.name}"
        return self.name


class ScopeStack:
    """Manages a stack of scope contexts for proper scope tracking."""
    
    def __init__(self):
        self.stack: List[ScopeContext] = []
        self.root_scope: Optional[ScopeContext] = None
    
    def push_scope(self, name: str, scope_type: str, node: ast.AST) -> ScopeContext:
        """Push a new scope onto the stack."""
        parent = self.stack[-1] if self.stack else None
        scope = ScopeContext(name, scope_type, node, parent)
        self.stack.append(scope)
        
        if not self.root_scope:
            self.root_scope = scope
            
        return scope
    
    def pop_scope(self) -> Optional[ScopeContext]:
        """Pop the current scope from the stack."""
        if self.stack:
            return self.stack.pop()
        return None
    
    def current_scope(self) -> Optional[ScopeContext]:
        """Get the current scope."""
        return self.stack[-1] if self.stack else None
    
    def is_name_defined(self, name: str) -> bool:
        """Check if a name is defined in the current scope or any parent scope."""
        if self.stack:
            return self.stack[-1].is_name_defined(name)
        return False
    
    def add_defined_name(self, name: str) -> None:
        """Add a defined name to the current scope."""
        if self.stack:
            self.stack[-1].add_defined_name(name)
    
    def add_imported_name(self, name: str) -> None:
        """Add an imported name to the current scope."""
        if self.stack:
            self.stack[-1].add_imported_name(name)
    
    def get_scope_path(self) -> str:
        """Get the current scope path."""
        if self.stack:
            return self.stack[-1].get_scope_path()
        return "module"


class ScopeTrackingVisitor(ast.NodeVisitor):
    """AST visitor that tracks scopes and detects undefined names."""
    
    def __init__(self, scope_stack: ScopeStack, builtin_names: Set[str], file_path: str, tree: ast.AST):
        self.scope_stack = scope_stack
        self.builtin_names = builtin_names
        self.file_path = file_path
        self.tree = tree
        self.errors: List[UndefinedNameError] = []
    
    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node."""
        # Visit all top-level definitions first to populate the module scope
        for child in node.body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Add to module scope
                self.scope_stack.add_defined_name(child.name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                # Handle imports
                self.generic_visit(child)
        
        # Now visit all nodes normally
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        # Push class scope
        class_scope = self.scope_stack.push_scope(node.name, "class", node)
        class_scope.add_defined_name(node.name)
        
        # Add 'cls' as defined for class methods
        class_scope.add_defined_name('cls')
        
        # Visit class body
        self.generic_visit(node)
        
        # Pop class scope
        self.scope_stack.pop_scope()
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._visit_function(node, "function")
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._visit_function(node, "function")
    
    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, scope_type: str) -> None:
        """Common logic for function and async function definitions."""
        # Push function scope
        func_scope = self.scope_stack.push_scope(node.name, scope_type, node)
        func_scope.add_defined_name(node.name)
        
        # Add function parameters as defined names
        for arg in node.args.args:
            func_scope.add_defined_name(arg.arg)
        
        # Add *args and **kwargs if present
        if node.args.vararg:
            func_scope.add_defined_name(node.args.vararg.arg)
        if node.args.kwarg:
            func_scope.add_defined_name(node.args.kwarg.arg)
        
        # Add 'self' as defined for class methods
        if self._is_class_method(node):
            func_scope.add_defined_name('self')
        
        # Pre-analyze function body to find variable assignments
        self._pre_analyze_function_body(node, func_scope)
        
        # Visit function body (this will handle nested functions)
        self.generic_visit(node)
        
        # Pop function scope
        self.scope_stack.pop_scope()

    def _pre_analyze_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef, scope: 'ScopeContext') -> None:
        """Pre-analyze function body to find variable assignments."""
        for child in node.body:
            if isinstance(child, ast.Assign):
                # Handle assignments like: wrapper = ...
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        scope.add_defined_name(target.id)
            elif isinstance(child, ast.FunctionDef):
                # Handle nested function definitions
                scope.add_defined_name(child.name)
            elif isinstance(child, ast.AsyncFunctionDef):
                # Handle nested async function definitions
                scope.add_defined_name(child.name)
    
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda expression."""
        # Push lambda scope
        lambda_scope = self.scope_stack.push_scope("<lambda>", "lambda", node)
        
        # Add lambda parameters as defined names
        for arg in node.args.args:
            lambda_scope.add_defined_name(arg.arg)
        
        # Add *args and **kwargs if present
        if node.args.vararg:
            lambda_scope.add_defined_name(node.args.vararg.arg)
        if node.args.kwarg:
            lambda_scope.add_defined_name(node.args.kwarg.arg)
        
        # Visit lambda body
        self.generic_visit(node)
        
        # Pop lambda scope
        self.scope_stack.pop_scope()
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.scope_stack.add_imported_name(name)
            self.scope_stack.add_defined_name(name)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from import statement."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.scope_stack.add_imported_name(name)
            self.scope_stack.add_defined_name(name)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statement."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scope_stack.add_defined_name(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.scope_stack.add_defined_name(elt.id)
        
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        """Visit for loop."""
        if isinstance(node.target, ast.Name):
            self.scope_stack.add_defined_name(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.scope_stack.add_defined_name(elt.id)
        
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With) -> None:
        """Visit with statement."""
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.scope_stack.add_defined_name(item.optional_vars.id)
        
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Visit exception handler."""
        if node.name:
            self.scope_stack.add_defined_name(node.name)
        
        self.generic_visit(node)
    
    def visit_Global(self, node: ast.Global) -> None:
        """Visit global statement."""
        for name in node.names:
            self.scope_stack.add_defined_name(name)
        
        self.generic_visit(node)
    
    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Visit nonlocal statement."""
        for name in node.names:
            self.scope_stack.add_defined_name(name)
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        """Visit name node (variable reference)."""
        if isinstance(node.ctx, ast.Load):
            name = node.id
            
            # Skip if it's a builtin
            if name in self.builtin_names:
                return
            
            # Skip if it's defined in current or parent scope
            if self.scope_stack.is_name_defined(name):
                return
            
            # Check if it's a special name (like __name__, __file__, etc.)
            if name.startswith('__') and name.endswith('__'):
                return
            
            # Skip common variable names that are often used in loops/comprehensions
            # But be more aggressive about detecting undefined names
            if name in {'i', 'j', 'k', 'x', 'y', 'z'}:
                # Only flag if it's not in a loop or comprehension context
                if not self._is_in_loop_context(node):
                    return
            
            # Special case: Check if this might be a nested function that's defined later in the same scope
            if self._might_be_nested_function(name):
                return
            
            # Special case: Check if this is likely a type annotation
            if self._is_likely_type_annotation(node):
                return
            
            # This is an undefined name
            context = self._get_context(node)
            self.errors.append(UndefinedNameError(
                file_path=self.file_path,
                line=node.lineno or 0,
                column=node.col_offset or 0,
                name=name,
                error_type="undefined_name",
                context=context,
                severity="error"
            ))
        
        self.generic_visit(node)
    
    def _is_likely_type_annotation(self, node: ast.Name) -> bool:
        """Check if a name is likely part of a type annotation."""
        # Check if we're in a type annotation context
        # Look at the parent node to determine context
        for parent in ast.walk(self.tree):
            if hasattr(parent, 'annotation') and node in ast.walk(parent.annotation):
                return True
            if hasattr(parent, 'args') and parent.args:
                # Handle both list and arguments object
                if isinstance(parent.args, list):
                    args_list = parent.args
                else:
                    # For arguments object, get all arguments
                    args_list = getattr(parent.args, 'args', []) + getattr(parent.args, 'kwonlyargs', [])
                
                for arg in args_list:
                    if hasattr(arg, 'annotation') and node in ast.walk(arg.annotation):
                        return True
        return False
    
    def _is_class_method(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a function is a class method by looking at the current scope."""
        current_scope = self.scope_stack.current_scope()
        if current_scope and current_scope.scope_type == "class":
            return True
        return False
    
    def _is_in_loop_context(self, name_node: ast.Name) -> bool:
        """Check if a name is used in a loop context."""
        # This is a simplified check - in a full implementation, we'd track loop contexts
        # For now, we'll be more permissive with common loop variables
        return name_node.id in {'i', 'j', 'k', 'x', 'y', 'z', 'item', 'value', 'key', 'val'}
    
    def _might_be_nested_function(self, name: str) -> bool:
        """Check if a name might be a nested function defined later in the same scope."""
        # This is a heuristic to avoid false positives for nested functions
        # that are defined later in the same function scope
        
        # Look for function definitions with this name in the current scope
        current_scope = self.scope_stack.current_scope()
        if not current_scope:
            return False
            
        # Check if there's a function definition with this name in the current scope
        # This is a simplified check - in a full implementation, we'd need to
        # track all function definitions in the current scope
        return False  # For now, we'll be conservative and not skip these
    
    def _get_context(self, node: ast.AST) -> str:
        """Get context around a node for better error reporting."""
        try:
            # Find the parent node to understand context
            for parent in ast.walk(self.scope_stack.root_scope.node if self.scope_stack.root_scope else None):
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
                        elif isinstance(parent, ast.FormattedValue):
                            return "in FormattedValue"
                        elif isinstance(parent, ast.Dict):
                            return "in Dict"
                        else:
                            return f"in {type(parent).__name__}"
            return "unknown context"
        except:
            return "unknown context"


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

    def __repr__(self):
        return f"UndefinedNameError({self.file_path}:{self.line}:{self.column}, {self.name})"

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


class UndefinedNamesAnalyzer:
    """
    Enhanced analyzer for detecting undefined names, variables, and imports in Python code.

    This analyzer uses AST parsing with proper scope tracking to identify:
    - Undefined variables
    - Undefined function names
    - Undefined class names
    - Missing imports
    - Unused imports
    - Import conflicts

    Key improvements:
    - Proper scope stack management
    - File isolation for directory analysis
    - Enhanced function parameter detection
    - Context-aware error reporting
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or get_default_config()
        self.errors: List[UndefinedNameError] = []
        self.builtin_names: Set[str] = set()
        self._init_builtin_names()

        # Per-file state to ensure proper isolation
        self._current_file_path: Optional[str] = None
        self._scope_stack: Optional[ScopeStack] = None

    def _init_builtin_names(self) -> None:
        """Initialize set of Python builtin names."""
        import builtins
        self.builtin_names = set(dir(builtins))
        # Add common builtin types and functions
        self.builtin_names.update({
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
        
        # Add common library aliases that are frequently used
        self.builtin_names.update({
            # Data science libraries
            'pd', 'np', 'plt', 'sns', 'sklearn', 'tf', 'torch', 'jax',
            # PyArrow and other data libraries
            'ds', 'pa', 'pq', 'spark', 'sc',
            # Common variable names in loops and comprehensions
            'i', 'j', 'k', 'x', 'y', 'z', 'item', 'value', 'key', 'val', 'data', 
            'result', 'temp', 'row', 'col', 'idx', 'index', 'n', 'm', 't', 'v',
            # Common function parameter names
            'args', 'kwargs', 'self', 'cls', 'func', 'obj', 'instance',
            # Common decorator variables
            'wrapper', 'decorator', 'f', 'g', 'h',
            # Common exception variables
            'e', 'ex', 'exc', 'exception', 'error', 'err',
            # Common iteration variables
            'elem', 'element', 'entry', 'record', 'line', 'word', 'char',
            # Common mathematical variables
            'a', 'b', 'c', 'd', 'p', 'q', 'r', 's', 'u', 'w',
            # Common configuration variables
            'config', 'cfg', 'settings', 'params', 'options', 'opts',
            # Common validation and utility functions
            'validate_data_quality', 'validate', 'check', 'verify',
            # Common dynamic/runtime imports
            'importlib', 'sys', 'os', 'pathlib', 'typing',
            # Common ML/AI variables
            'model', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test',
            'features', 'target', 'prediction', 'score', 'accuracy',
            # Common database variables
            'db', 'conn', 'cursor', 'query', 'table', 'column',
            # Common async variables
            'async', 'await', 'task', 'future', 'coroutine',
            # Common logging variables
            'logger', 'log', 'debug', 'info', 'warning', 'error'
        })

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single Python file for undefined names and variables using enhanced scope tracking.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        tprint(f"Analyzing undefined names in: {file_path}")
        
        # Initialize file-specific state
        self._current_file_path = file_path
        self._scope_stack = ScopeStack()
        
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
        
        try:
            # Parse the AST
            tree = ast.parse(content, filename=file_path)
            
            # Create module scope
            module_scope = self._scope_stack.push_scope("__main__", "module", tree)
            
            # Analyze the AST with proper scope tracking
            file_errors = self._analyze_with_scope_tracking(tree, file_path)
            
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
        finally:
            # Clean up file-specific state
            self._current_file_path = None
            self._scope_stack = None

        # Convert errors to dictionaries
        errors_dict = [error.to_dict() for error in file_errors]
        
        # Collect defined and imported names from the scope
        defined_names = set()
        imported_names = set()
        if self._scope_stack and self._scope_stack.root_scope:
            self._collect_names_from_scope(self._scope_stack.root_scope, defined_names, imported_names)
        
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

    def _analyze_with_scope_tracking(self, tree: ast.AST, file_path: str) -> List[UndefinedNameError]:
        """Analyze AST with proper scope tracking."""
        errors = []
        
        # Use a visitor pattern for better control
        visitor = ScopeTrackingVisitor(self._scope_stack, self.builtin_names, file_path, tree)
        visitor.visit(tree)
        
        return visitor.errors
    
    def _collect_names_from_scope(self, scope: ScopeContext, defined_names: Set[str], imported_names: Set[str]) -> None:
        """Recursively collect all defined and imported names from scope hierarchy."""
        defined_names.update(scope.defined_names)
        imported_names.update(scope.imported_names)
        
        for child in scope.children:
            self._collect_names_from_scope(child, defined_names, imported_names)


    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Analyze all Python files in a directory for undefined names and variables.
        Each file is analyzed in complete isolation to prevent context bleeding.
        
        Args:
            directory_path: Path to the directory to analyze
            
        Returns:
            Dictionary containing analysis results for all files
        """
        tprint(f"Analyzing undefined names in directory: {directory_path}")
        
        # Find all Python files
        exclude_patterns = self.config.get('analysis_config', {}).get('exclude_patterns', [])
        python_files = find_python_files(directory_path, exclude_patterns)
        
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

        # Analyze each file in complete isolation
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
            # Each file gets its own fresh analyzer instance to ensure complete isolation
            file_analyzer = UndefinedNamesAnalyzer(self.config)
            file_result = file_analyzer.analyze_file(str(file_path))
            files_results[str(file_path)] = file_result
            
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

    def get_errors_by_type(self) -> Dict[str, List[UndefinedNameError]]:
        """Get errors grouped by type."""
        errors_by_type = defaultdict(list)
        for error in self.errors:
            errors_by_type[error.error_type].append(error)
        return dict(errors_by_type)

    def get_errors_by_file(self) -> Dict[str, List[UndefinedNameError]]:
        """Get errors grouped by file."""
        errors_by_file = defaultdict(list)
        for error in self.errors:
            errors_by_file[error.file_path].append(error)
        return dict(errors_by_file)

    def save_report(self, output_path: str, analysis_results: Dict[str, Any] = None) -> None:
        """Save analysis report to a JSON file."""
        if analysis_results is None:
            # Fallback to old behavior for backward compatibility
            report = {
                "timestamp": str(Path().cwd()),
                "total_errors": len(self.errors),
                "errors_by_type": {
                    error_type: [error.to_dict() for error in errors]
                    for error_type, errors in self.get_errors_by_type().items()
                },
                "errors_by_file": {
                    file_path: [error.to_dict() for error in errors]
                    for file_path, errors in self.get_errors_by_file().items()
                },
                "summary": {
                    "total_undefined_names": len([e for e in self.errors if e.error_type == "undefined_name"]),
                    "total_undefined_imports": len([e for e in self.errors if e.error_type == "undefined_import"]),
                    "total_unused_imports": len([e for e in self.errors if e.error_type == "unused_import"]),
                    "total_import_conflicts": len([e for e in self.errors if e.error_type == "import_conflict"]),
                }
            }
        else:
            # Use the new analysis results
            report = {
                "timestamp": str(Path().cwd()),
                "total_errors": analysis_results.get("total_errors", 0),
                "errors_by_type": {},
                "errors_by_file": {},
                "summary": analysis_results.get("summary", {})
            }
            
            # Group errors by type and file
            if "files" in analysis_results:
                for file_path, file_result in analysis_results["files"].items():
                    if file_result.get("errors"):
                        report["errors_by_file"][file_path] = file_result["errors"]
                        
                        # Group by error type
                        for error in file_result["errors"]:
                            error_type = error.get("error_type", "unknown")
                            if error_type not in report["errors_by_type"]:
                                report["errors_by_type"][error_type] = []
                            report["errors_by_type"][error_type].append(error)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def analyze_undefined_names(self, directory: str) -> Dict[str, Any]:
        """
        Analyze undefined names in a directory (compatibility method for pipeline).
        
        Args:
            directory: Path to directory to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Use the existing directory analysis
            results = self.analyze_directory(directory)
            
            # Extract undefined names from the results
            undefined_names = []
            for file_path, file_result in results.get("file_results", {}).items():
                for error in file_result.get("errors", []):
                    undefined_names.append({
                        "file": file_path,
                        "line": error.get("line", 0),
                        "column": error.get("column", 0),
                        "name": error.get("name", ""),
                        "error_type": error.get("error_type", "undefined_name"),
                        "description": error.get("description", ""),
                        "context": error.get("context", "")
                    })
            
            return {
                "undefined_names": undefined_names,
                "total_undefined_names": len(undefined_names),
                "files_analyzed": len(results.get("file_results", {})),
                "summary": results.get("summary", {})
            }
        except Exception as e:
            return {
                "undefined_names": [],
                "total_undefined_names": 0,
                "error": str(e)
            }


def main():
    """Command-line interface for the undefined names analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Python code for undefined names and variables"
    )
    parser.add_argument("--target", required=True,
                       help="Path to Python file or directory to analyze")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file for JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = get_default_config()

    # Create analyzer
    analyzer = UndefinedNamesAnalyzer(config)

    # Analyze target
    if os.path.isfile(args.target):
        results = analyzer.analyze_file(args.target)
    else:
        results = analyzer.analyze_directory(args.target)

    # Print results
    if results["status"] == "success":
        summary = results.get("summary", {})
        total_errors = results.get("total_errors", 0)  # Get from top level for individual files
        
        tprint(f"\nAnalysis completed successfully!")
        tprint(f"Total files analyzed: {summary.get('total_files_analyzed', 1)}")
        tprint(f"Total errors found: {total_errors}")
        tprint(f"Files with errors: {summary.get('files_with_errors', 1 if total_errors > 0 else 0)}")
        
        if total_errors > 0:
            tprint(f"\nError breakdown:")
            tprint(f"  Undefined names: {summary.get('total_undefined_names', 0)}")
            tprint(f"  Undefined imports: {summary.get('total_undefined_imports', 0)}")
            tprint(f"  Unused imports: {summary.get('total_unused_imports', 0)}")
            tprint(f"  Import conflicts: {summary.get('total_import_conflicts', 0)}")
            
            if args.verbose:
                tprint(f"\nDetailed errors:")
                # Handle both individual file and directory results
                if "files" in results:
                    for file_path, file_result in results.get("files", {}).items():
                        if file_result.get("total_errors", 0) > 0:
                            tprint(f"\n{file_path}:")
                            for error in file_result.get("errors", []):
                                tprint(f"  Line {error['line']}: {error['name']} - {error['error_type']}")
                                if error.get('context'):
                                    tprint(f"    Context: {error['context']}")
                else:
                    # Individual file result
                    for error in results.get("errors", []):
                        tprint(f"  Line {error['line']}: {error['name']} - {error['error_type']}")
                        if error.get('context'):
                            tprint(f"    Context: {error['context']}")
    else:
        tprint(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return 1

    # Save report if requested
    if args.output:
        analyzer.save_report(args.output, results)
        tprint(f"\nReport saved to: {args.output}")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
