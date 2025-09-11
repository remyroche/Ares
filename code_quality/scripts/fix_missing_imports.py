#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Script to analyze and fix missing imports for common operations.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import logging
import os
import pandas as pd
import time
import typing
import re

# Enhanced function to module mappings with comprehensive coverage
COMMON_IMPORTS = {
    # DateTime operations
    "now": ("datetime", "datetime"),
    "today": ("datetime", "date"),
    "timedelta": ("datetime", "timedelta"),
    "datetime": ("datetime", "datetime"),
    "date": ("datetime", "date"),
    "time": ("datetime", "time"),
    "isoformat": None,  # This is a method, not an import
    "strftime": None,   # This is a method, not an import
    "total_seconds": None,  # This is a method, not an import

    # Pandas operations - Enhanced coverage
    "DataFrame": ("pandas", "pd"),
    "Series": ("pandas", "pd"),
    "read_csv": ("pandas", "pd"),
    "read_parquet": ("pandas", "pd"),
    "read_excel": ("pandas", "pd"),
    "read_json": ("pandas", "pd"),
    "concat": ("pandas", "pd"),
    "merge": ("pandas", "pd"),
    "to_datetime": ("pandas", "pd"),
    "fillna": ("pandas", "pd"),  # Common missing import
    "dropna": ("pandas", "pd"),  # Common missing import
    "groupby": ("pandas", "pd"),  # Common missing import
    "rolling": ("pandas", "pd"),
    "shift": ("pandas", "pd"),
    "diff": ("pandas", "pd"),
    "cumsum": ("pandas", "pd"),
    "cumprod": ("pandas", "pd"),
    "pivot_table": ("pandas", "pd"),
    "melt": ("pandas", "pd"),
    "get_dummies": ("pandas", "pd"),
    "cut": ("pandas", "pd"),
    "qcut": ("pandas", "pd"),
    "crosstab": ("pandas", "pd"),
    "value_counts": ("pandas", "pd"),

    # NumPy operations - Enhanced coverage
    "array": ("numpy", "np"),
    "zeros": ("numpy", "np"),
    "ones": ("numpy", "np"),
    "empty": ("numpy", "np"),
    "full": ("numpy", "np"),
    "arange": ("numpy", "np"),
    "linspace": ("numpy", "np"),
    "logspace": ("numpy", "np"),
    "mean": ("numpy", "np"),
    "std": ("numpy", "np"),
    "var": ("numpy", "np"),
    "sum": ("numpy", "np"),  # Common missing import
    "min": ("numpy", "np"),
    "max": ("numpy", "np"),
    "argmin": ("numpy", "np"),
    "argmax": ("numpy", "np"),
    "nan": ("numpy", "np"),
    "inf": ("numpy", "np"),
    "isnan": ("numpy", "np"),
    "isinf": ("numpy", "np"),
    "isfinite": ("numpy", "np"),
    "abs": ("numpy", "np"),
    "sqrt": ("numpy", "np"),
    "exp": ("numpy", "np"),
    "log": ("numpy", "np"),
    "sin": ("numpy", "np"),
    "cos": ("numpy", "np"),
    "tan": ("numpy", "np"),
    "dot": ("numpy", "np"),
    "transpose": ("numpy", "np"),
    "reshape": ("numpy", "np"),
    "flatten": ("numpy", "np"),
    "concatenate": ("numpy", "np"),
    "stack": ("numpy", "np"),
    "hstack": ("numpy", "np"),
    "vstack": ("numpy", "np"),
    "where": ("numpy", "np"),
    "clip": ("numpy", "np"),
    "round": ("numpy", "np"),

    # Path operations
    "Path": ("pathlib", "Path"),
    "exists": None,  # This is usually a method on Path
    "mkdir": None,   # This is usually a method on Path
    "join": ("os.path", None),

    # Asyncio operations
    "create_task": ("asyncio", None),
    "gather": ("asyncio", None),
    "sleep": ("asyncio", None),
    "run": ("asyncio", None),
    "get_event_loop": ("asyncio", None),

    # Typing operations
    "List": ("typing", None),
    "Dict": ("typing", None),
    "Set": ("typing", None),
    "Tuple": ("typing", None),
    "Optional": ("typing", None),
    "Union": ("typing", None),
    "Any": ("typing", None),

    # Logging operations - Enhanced coverage
    "getLogger": ("logging", None),
    "info": ("logging", None),
    "debug": ("logging", None),
    "warning": ("logging", None),
    "error": ("logging", None),
    "critical": ("logging", None),
    "basicConfig": ("logging", None),
    "setLevel": ("logging", None),
    "Logger": ("logging", None),
    "StreamHandler": ("logging", None),
    "FileHandler": ("logging", None),
    "Formatter": ("logging", None),

    # JSON operations - Enhanced coverage
    "dumps": ("json", None),
    "loads": ("json", None),
    "dump": ("json", None),
    "load": ("json", None),

    # Other common operations - Enhanced coverage
    "ArgumentParser": ("argparse", None),
    "defaultdict": ("collections", None),
    "Counter": ("collections", None),
    "deque": ("collections", None),
    "OrderedDict": ("collections", None),
    "namedtuple": ("collections", None),
    "deepcopy": ("copy", None),
    "copy": ("copy", None),
    
    # OS operations - Enhanced coverage
    "listdir": ("os", None),
    "makedirs": ("os", None),
    "remove": ("os", None),
    "rmdir": ("os", None),
    "getcwd": ("os", None),
    "chdir": ("os", None),
    "path": ("os", None),
    "environ": ("os", None),
    "system": ("os", None),
    "popen": ("os", None),
    
    # Time operations - Enhanced coverage
    "sleep": ("time", None),
    "time": ("time", None),
    "ctime": ("time", None),
    "gmtime": ("time", None),
    "localtime": ("time", None),
    "strftime": ("time", None),
    "strptime": ("time", None),
    "mktime": ("time", None),
    
    # Additional common functions that are often missing
    "filterwarnings": ("warnings", None),  # Common missing import
    "warn": ("warnings", None),
    "all": None,  # Built-in function
    "any": None,  # Built-in function
    "type": None,  # Built-in function
    "isinstance": None,  # Built-in function
    "hasattr": None,  # Built-in function
    "getattr": None,  # Built-in function
    "setattr": None,  # Built-in function
    "delattr": None,  # Built-in function
    "callable": None,  # Built-in function
    "enumerate": None,  # Built-in function
    "zip": None,  # Built-in function
    "map": None,  # Built-in function
    "filter": None,  # Built-in function
    "sorted": None,  # Built-in function
    "reversed": None,  # Built-in function
    "range": None,  # Built-in function
    "len": None,  # Built-in function
    "str": None,  # Built-in function
    "int": None,  # Built-in function
    "float": None,  # Built-in function
    "bool": None,  # Built-in function
    "list": None,  # Built-in function
    "dict": None,  # Built-in function
    "set": None,  # Built-in function
    "tuple": None,  # Built-in function
    "print": None,  # Built-in function
    "input": None,  # Built-in function
    "open": None,  # Built-in function
    "abs": None,  # Built-in function
    "round": None,  # Built-in function
    "min": None,  # Built-in function
    "max": None,  # Built-in function
    "sum": None,  # Built-in function
}

# Methods that don't need imports (they're attributes of objects)
OBJECT_METHODS = {
    "append", "extend", "insert", "remove", "pop",  # List methods
    "keys", "values", "items", "get", "update",     # Dict methods
    "lower", "upper", "strip", "split", "replace",  # String methods
    "fillna", "rolling", "shift", "diff", "cumsum", # DataFrame methods
    "isoformat", "strftime", "total_seconds",        # DateTime methods
    "exists", "mkdir", "unlink", "rmdir",            # Path methods
}


class ImportFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_by_file = defaultdict(list)
        self.imports_to_add = defaultdict(set)
        self.fixed_files = []
        self.failed_files = []
        
        # Enhanced detection patterns
        self.numpy_patterns = {
            'array', 'zeros', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace',
            'mean', 'std', 'var', 'sum', 'min', 'max', 'argmin', 'argmax', 'nan', 'inf',
            'isnan', 'isinf', 'isfinite', 'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
            'dot', 'transpose', 'reshape', 'flatten', 'concatenate', 'stack', 'hstack', 'vstack',
            'where', 'clip', 'round'
        }
        
        self.pandas_patterns = {
            'DataFrame', 'Series', 'read_csv', 'read_parquet', 'read_excel', 'read_json',
            'concat', 'merge', 'to_datetime', 'fillna', 'dropna', 'groupby', 'rolling',
            'shift', 'diff', 'cumsum', 'cumprod', 'pivot_table', 'melt', 'get_dummies',
            'cut', 'qcut', 'crosstab', 'value_counts'
        }
        
        self.warnings_patterns = {
            'filterwarnings', 'warn', 'warnings'
        }
        
        # Additional pattern sets for comprehensive coverage
        self.datetime_patterns = {
            'now', 'today', 'timedelta', 'datetime', 'date', 'time', 'utcnow', 'fromtimestamp'
        }
        
        self.time_patterns = {
            'time', 'sleep', 'monotonic', 'perf_counter', 'process_time'
        }
        
        self.uuid_patterns = {
            'uuid4', 'uuid1', 'uuid3', 'uuid5', 'uuid'
        }
        
        self.csv_patterns = {
            'QUOTE_NONNUMERIC', 'QUOTE_ALL', 'QUOTE_MINIMAL', 'QUOTE_NONE',
            'writer', 'reader', 'DictWriter', 'DictReader'
        }
        
        self.json_patterns = {
            'dumps', 'loads', 'dump', 'load'
        }
        
        self.logging_patterns = {
            'getLogger', 'basicConfig', 'info', 'debug', 'warning', 'error', 'critical'
        }
        
        self.typing_patterns = {
            'List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union', 'Any', 'Callable',
            'Type', 'Generic', 'TypeVar', 'Protocol', 'Literal', 'Final'
        }
        
        self.collections_patterns = {
            'defaultdict', 'Counter', 'deque', 'OrderedDict', 'ChainMap', 'namedtuple'
        }
        
        self.copy_patterns = {
            'deepcopy', 'copy'
        }
        
        self.os_patterns = {
            'path', 'environ', 'getenv', 'listdir', 'makedirs', 'remove', 'rename'
        }
        
        self.sys_patterns = {
            'argv', 'exit', 'path', 'version', 'platform', 'stdout', 'stderr'
        }
        
        self.re_patterns = {
            'match', 'search', 'findall', 'sub', 'compile', 'split'
        }
        
        self.math_patterns = {
            'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'pi', 'e', 'ceil', 'floor'
        }
        
        self.random_patterns = {
            'random', 'randint', 'choice', 'shuffle', 'uniform', 'gauss'
        }

    def load_issues(self, json_file: str):
        """Load issues from the validation report."""
        with open(json_file) as f:
            data = json.load(f)

        # Group undefined functions by file
        for issue in data.get("issues", []):
            if issue["issue_type"] == "undefined_function":
                msg = issue["message"]
                if "Function '" in msg:
                    func_name = msg.split("'")[1]
                    if func_name not in OBJECT_METHODS:
                        self.issues_by_file[issue["file_path"]].append({
                            "function": func_name,
                            "line": issue["line_number"],
                        })

    def analyze_imports_needed(self):
        """Analyze which imports are needed for each file."""
        for file_path, issues in self.issues_by_file.items():
            imports_needed = set()

            for issue in issues:
                func = issue["function"]
                if func in COMMON_IMPORTS and COMMON_IMPORTS[func]:
                    module, alias = COMMON_IMPORTS[func]
                    imports_needed.add((module, alias))

            if imports_needed:
                self.imports_to_add[file_path] = imports_needed

    def auto_detect_missing_imports(self, file_path: str) -> set:
        """Automatically detect missing imports by analyzing the file content."""
        missing_imports = set()  # Initialize outside try block
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Get existing imports to avoid duplicates
            existing_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module.split('.')[0])
            
            # Find all function calls and attribute access
            for node in ast.walk(tree):
                # Handle direct function calls like np.array()
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    # Check for common imports first
                    if func_name in COMMON_IMPORTS and COMMON_IMPORTS[func_name]:
                        module, alias = COMMON_IMPORTS[func_name]
                        if module not in existing_imports:
                            missing_imports.add((module, alias))
                    
                    # Check for numpy functions
                    elif func_name in self.numpy_patterns:
                        missing_imports.add(('numpy', 'np'))
                    
                    # Check for pandas functions
                    elif func_name in self.pandas_patterns:
                        missing_imports.add(('pandas', 'pd'))
                    
                    # Check for warnings functions
                    elif func_name in self.warnings_patterns:
                        missing_imports.add(('warnings', None))
                    
                    # Check for datetime functions
                    elif func_name in self.datetime_patterns:
                        missing_imports.add(('datetime', None))
                    
                    # Check for time functions
                    elif func_name in self.time_patterns:
                        missing_imports.add(('time', None))
                    
                    # Check for uuid functions
                    elif func_name in self.uuid_patterns:
                        missing_imports.add(('uuid', None))
                    
                    # Check for json functions
                    elif func_name in self.json_patterns:
                        missing_imports.add(('json', None))
                    
                    # Check for logging functions
                    elif func_name in self.logging_patterns:
                        missing_imports.add(('logging', None))
                    
                    # Check for typing functions
                    elif func_name in self.typing_patterns:
                        missing_imports.add(('typing', None))
                    
                    # Check for collections functions
                    elif func_name in self.collections_patterns:
                        missing_imports.add(('collections', None))
                    
                    # Check for copy functions
                    elif func_name in self.copy_patterns:
                        missing_imports.add(('copy', None))
                    
                    # Check for os functions
                    elif func_name in self.os_patterns:
                        missing_imports.add(('os', None))
                    
                    # Check for sys functions
                    elif func_name in self.sys_patterns:
                        missing_imports.add(('sys', None))
                    
                    # Check for re functions
                    elif func_name in self.re_patterns:
                        missing_imports.add(('re', None))
                    
                    # Check for math functions
                    elif func_name in self.math_patterns:
                        missing_imports.add(('math', None))
                    
                    # Check for random functions
                    elif func_name in self.random_patterns:
                        missing_imports.add(('random', None))
                
                # Handle attribute access like np.array, pd.DataFrame, datetime.now
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        func_name = node.func.attr
                        
                        # Check for numpy patterns
                        if module_name == 'np' and func_name in self.numpy_patterns:
                            missing_imports.add(('numpy', 'np'))
                        
                        # Check for pandas patterns
                        elif module_name == 'pd' and func_name in self.pandas_patterns:
                            missing_imports.add(('pandas', 'pd'))
                        
                        # Check for datetime patterns
                        elif module_name == 'datetime' and func_name in self.datetime_patterns:
                            missing_imports.add(('datetime', None))
                        
                        # Check for time patterns
                        elif module_name == 'time' and func_name in self.time_patterns:
                            missing_imports.add(('time', None))
                        
                        # Check for uuid patterns
                        elif module_name == 'uuid' and func_name in self.uuid_patterns:
                            missing_imports.add(('uuid', None))
                        
                        # Check for csv patterns
                        elif module_name == 'csv' and func_name in self.csv_patterns:
                            missing_imports.add(('csv', None))
                        
                        # Check for json patterns
                        elif module_name == 'json' and func_name in self.json_patterns:
                            missing_imports.add(('json', None))
                        
                        # Check for logging patterns
                        elif module_name == 'logging' and func_name in self.logging_patterns:
                            missing_imports.add(('logging', None))
                        
                        # Check for os patterns
                        elif module_name == 'os' and func_name in self.os_patterns:
                            missing_imports.add(('os', None))
                        
                        # Check for sys patterns
                        elif module_name == 'sys' and func_name in self.sys_patterns:
                            missing_imports.add(('sys', None))
                        
                        # Check for re patterns
                        elif module_name == 're' and func_name in self.re_patterns:
                            missing_imports.add(('re', None))
                        
                        # Check for math patterns
                        elif module_name == 'math' and func_name in self.math_patterns:
                            missing_imports.add(('math', None))
                        
                        # Check for random patterns
                        elif module_name == 'random' and func_name in self.random_patterns:
                            missing_imports.add(('random', None))
                
                # Handle attribute access without calls like np.inf, csv.QUOTE_NONNUMERIC
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    module_name = node.value.id
                    attr_name = node.attr
                    
                    # Check for numpy constants
                    if module_name == 'np' and attr_name in {'inf', 'nan', 'pi', 'e'}:
                        missing_imports.add(('numpy', 'np'))
                    
                    # Check for csv constants
                    elif module_name == 'csv' and attr_name in {'QUOTE_NONNUMERIC', 'QUOTE_ALL', 'QUOTE_MINIMAL', 'QUOTE_NONE'}:
                        missing_imports.add(('csv', None))
                    
                    # Check for math constants
                    elif module_name == 'math' and attr_name in {'pi', 'e', 'inf', 'nan'}:
                        missing_imports.add(('math', None))
                    
                    # Check for sys attributes
                    elif module_name == 'sys' and attr_name in {'stdout', 'stderr', 'stdin', 'argv', 'version', 'platform'}:
                        missing_imports.add(('sys', None))
                    
                    # Check for os attributes
                    elif module_name == 'os' and attr_name in {'path', 'environ', 'sep', 'linesep'}:
                        missing_imports.add(('os', None))
            
            # Check existing imports to avoid duplicates
            existing_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module)
            
            # Filter out already imported modules
            filtered_imports = set()
            for module, alias in missing_imports:
                if module not in existing_imports:
                    filtered_imports.add((module, alias))
            
            return filtered_imports
            
        except SyntaxError as e:
            tprint(f"Syntax error in {file_path}: {e}")
            return set()
        except IndentationError as e:
            tprint(f"Indentation error in {file_path}: {e}")
            return set()
        except ImportError as e:
            tprint(f"Import error in {file_path}: {e}")
            return set()
        except Exception as e:
            tprint(f"Unknown error analyzing {file_path}: {e}")
            return set()

    def add_common_missing_imports(self, file_path: str) -> set:
        """Add the most common missing imports based on usage patterns."""
        missing_imports = set()  # Initialize outside try block
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Get existing imports
            existing_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module.split('.')[0])
            
            # Check for common patterns that indicate missing imports
            content_lower = content.lower()
            
            # Check for numpy usage patterns
            if any(pattern in content_lower for pattern in ['np.', 'numpy.', 'array(', 'zeros(', 'ones(', 'mean(', 'std(', 'sum(']):
                if 'numpy' not in existing_imports:
                    missing_imports.add(('numpy', 'np'))
            
            # Check for pandas usage patterns
            if any(pattern in content_lower for pattern in ['pd.', 'pandas.', 'dataframe', 'series', 'read_csv', 'read_parquet']):
                if 'pandas' not in existing_imports:
                    missing_imports.add(('pandas', 'pd'))
            
            # Check for logging usage patterns
            if any(pattern in content_lower for pattern in ['logger.', 'logging.', 'info(', 'debug(', 'warning(', 'error(', 'critical(']):
                if 'logging' not in existing_imports:
                    missing_imports.add(('logging', None))
            
            # Check for datetime usage patterns
            if any(pattern in content_lower for pattern in ['datetime.', 'now()', 'today()', 'timedelta']):
                if 'datetime' not in existing_imports:
                    missing_imports.add(('datetime', None))
            
            # Check for os usage patterns
            if any(pattern in content_lower for pattern in ['os.', 'listdir', 'makedirs', 'getcwd', 'chdir']):
                if 'os' not in existing_imports:
                    missing_imports.add(('os', None))
            
            # Check for time usage patterns
            if any(pattern in content_lower for pattern in ['time.', 'sleep(', 'time()']):
                if 'time' not in existing_imports:
                    missing_imports.add(('time', None))
            
            # Check for collections usage patterns
            if any(pattern in content_lower for pattern in ['defaultdict', 'counter', 'deque', 'ordereddict']):
                if 'collections' not in existing_imports:
                    missing_imports.add(('collections', None))
            
            # Check for json usage patterns
            if any(pattern in content_lower for pattern in ['json.', 'dumps(', 'loads(', 'dump(', 'load(']):
                if 'json' not in existing_imports:
                    missing_imports.add(('json', None))
            
            # Check for typing usage patterns
            if any(pattern in content_lower for pattern in ['list[', 'dict[', 'set[', 'tuple[', 'optional[', 'union[']):
                if 'typing' not in existing_imports:
                    missing_imports.add(('typing', None))
            
        except Exception as e:
            tprint(f"Error analyzing {file_path}: {e}")
            
        return missing_imports

    def analyze_file_with_categorization(self, file_path: str) -> dict:
        """Analyze a file and return categorized results."""
        result = {
            "file_path": file_path,
            "status": "success",
            "error_type": None,
            "error_message": None,
            "missing_imports": set(),
            "syntax_valid": False,
            "can_parse_ast": False
        }
        
        try:
            # First check if file has valid Python syntax
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test basic Python syntax
            try:
                compile(content, file_path, 'exec')
                result["syntax_valid"] = True
            except SyntaxError as e:
                result["status"] = "syntax_error"
                result["error_type"] = "syntax_error"
                result["error_message"] = f"Syntax error: {e.msg} (line {e.lineno})"
                return result
            except IndentationError as e:
                result["status"] = "indentation_error"
                result["error_type"] = "indentation_error"
                result["error_message"] = f"Indentation error: {e.msg} (line {e.lineno})"
                return result
            
            # Test AST parsing
            try:
                tree = ast.parse(content)
                result["can_parse_ast"] = True
                
                # Now analyze for missing imports using enhanced detection
                missing_imports = self.auto_detect_missing_imports(file_path)
                common_missing = self.add_common_missing_imports(file_path)
                
                # Combine both sets
                all_missing_imports = missing_imports.union(common_missing)
                result["missing_imports"] = all_missing_imports
                
            except SyntaxError as e:
                result["status"] = "ast_parse_error"
                result["error_type"] = "ast_parse_error"
                result["error_message"] = f"AST parse error: {e.msg} (line {e.lineno})"
                return result
            except Exception as e:
                result["status"] = "ast_parse_error"
                result["error_type"] = "ast_parse_error"
                result["error_message"] = f"AST parse error: {str(e)}"
                return result
                
        except FileNotFoundError:
            result["status"] = "file_not_found"
            result["error_type"] = "file_not_found"
            result["error_message"] = "File not found"
            return result
        except PermissionError:
            result["status"] = "permission_error"
            result["error_type"] = "permission_error"
            result["error_message"] = "Permission denied"
            return result
        except Exception as e:
            result["status"] = "unknown_error"
            result["error_type"] = "unknown_error"
            result["error_message"] = f"Unknown error: {str(e)}"
            return result
        
        return result

    def auto_fix_file_imports(self, file_path: str) -> bool:
        """Automatically detect and fix missing imports in a file."""
        try:
            # Auto-detect missing imports using both methods
            missing_imports = self.auto_detect_missing_imports(file_path)
            common_missing = self.add_common_missing_imports(file_path)
            
            # Combine both sets
            all_missing_imports = missing_imports.union(common_missing)
            
            if not all_missing_imports:
                return False
            
            # Fix the imports
            return self.fix_file_imports(file_path, all_missing_imports)
            
        except Exception as e:
            tprint(f"Error auto-fixing {file_path}: {e}")
            return False

    def fix_file_imports(self, file_path: str, imports_needed: set[tuple[str, str]]) -> bool:
        """Add missing imports to a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file to find where to insert imports
            tree = ast.parse(content)

            # Find existing imports at the top of the file only
            existing_imports = set()
            last_import_line = 0
            
            # Only consider imports at the top of the file (before any non-import statements)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name)
                    # Only update last_import_line if this import is at the top
                    if node.lineno <= 50:  # Only consider imports in first 50 lines
                        last_import_line = max(last_import_line, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module)
                    # Only update last_import_line if this import is at the top
                    if node.lineno <= 50:  # Only consider imports in first 50 lines
                        last_import_line = max(last_import_line, node.lineno)

            # Prepare new imports
            new_imports = []
            for module, alias in imports_needed:
                if module not in existing_imports:
                    if alias:
                        new_imports.append(f"import {module} as {alias}")
                    else:
                        new_imports.append(f"import {module}")

            if not new_imports:
                return False

            # Insert imports after existing imports or at the beginning
            lines = content.split("\n")

            # Find the right place to insert - be more careful about placement
            if last_import_line > 0:
                # Insert after the last import at the top
                insert_line = last_import_line
            else:
                # No imports found at the top, insert at the beginning
                insert_line = 0
                
                # Handle module docstrings
                if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
                    # Find end of docstring
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip().endswith('"""') or line.strip().endswith("'''"):
                            insert_line = i + 1
                            break
                
                # Handle shebang lines
                if lines and lines[0].startswith('#!'):
                    insert_line = 1

            # Insert the imports
            for imp in sorted(new_imports):
                lines.insert(insert_line, imp)
                insert_line += 1

            # Add blank line after imports if needed
            if insert_line < len(lines) and lines[insert_line].strip():
                lines.insert(insert_line, "")

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True

        except Exception as e:
            tprint(f"Error fixing {file_path}: {e}")
            return False

    def generate_report(self) -> dict:
        """Generate a report of fixes to be made."""
        report = {
            "total_files": len(self.imports_to_add),
            "imports_by_file": {},
            "summary": defaultdict(int),
        }

        for file_path, imports in self.imports_to_add.items():
            report["imports_by_file"][file_path] = [
                f"import {module} as {alias}" if alias else f"import {module}"
                for module, alias in imports
            ]

            for module, _ in imports:
                report["summary"][module] += 1

        return report

    def fix_all_imports(self, dry_run: bool = True):
        """Fix imports in all files."""
        self.analyze_imports_needed()

        if dry_run:
            report = self.generate_report()
            tprint("\nDRY RUN - Imports that would be added:")
            tprint("=" * 60)

            # Show summary
            tprint("\nSummary by module:")
            for module, count in sorted(report["summary"].items(), key=lambda x: x[1], reverse=True):
                tprint(f"  {module}: {count} files")

            # Show sample files
            tprint("\nSample files to be fixed (showing first 5):")
            for file_path, imports in list(report["imports_by_file"].items())[:5]:
                tprint(f"\n{file_path}:")
                for imp in imports:
                    tprint(f"  + {imp}")

            if len(report["imports_by_file"]) > 5:
                tprint(f"\n... and {len(report['imports_by_file']) - 5} more files")

            return report
        # Actually fix the files
        fixed = 0
        failed = 0

        for file_path, imports in self.imports_to_add.items():
            if self.fix_file_imports(file_path, imports):
                fixed += 1
                tprint(f"✓ Fixed {file_path}")
            else:
                failed += 1
                tprint(f"✗ Failed to fix {file_path}")

        tprint(f"\nFixed {fixed} files, {failed} failures")
        return {"fixed": fixed, "failed": failed}

    def auto_fix_all_files(self, file_paths: list, dry_run: bool = True):
        """Automatically detect and fix missing imports in all files with categorization."""
        # Categorize all files first
        categorized_results = self.categorize_all_files(file_paths)
        
        if dry_run:
            tprint("\nAUTO-DETECTION DRY RUN - Categorized Analysis:")
            tprint("=" * 70)
            
            # Print categorized results
            self._print_categorized_results(categorized_results)
            
            return categorized_results
        
        # Actually fix the files
        fixed = 0
        failed = 0
        
        for file_path in file_paths:
            if self.auto_fix_file_imports(file_path):
                fixed += 1
                self.fixed_files.append(file_path)
                tprint(f"✓ Auto-fixed {file_path}")
            else:
                failed += 1
                self.failed_files.append(file_path)
        
        tprint(f"\nAuto-fixed {fixed} files, {failed} failures")
        return {"fixed": fixed, "failed": failed, "fixed_files": self.fixed_files, "failed_files": self.failed_files}

    def categorize_all_files(self, file_paths: list) -> dict:
        """Categorize all files by their analysis results."""
        results = {
            "total_files": len(file_paths),
            "syntax_errors": [],
            "indentation_errors": [],
            "ast_parse_errors": [],
            "file_not_found": [],
            "permission_errors": [],
            "unknown_errors": [],
            "successful_analysis": [],
            "files_with_missing_imports": [],
            "error_counts": {
                "syntax_errors": 0,
                "indentation_errors": 0,
                "ast_parse_errors": 0,
                "file_not_found": 0,
                "permission_errors": 0,
                "unknown_errors": 0,
                "successful_analysis": 0,
                "files_with_missing_imports": 0
            }
        }
        
        for file_path in file_paths:
            analysis = self.analyze_file_with_categorization(file_path)
            
            if analysis["status"] == "success":
                results["successful_analysis"].append(analysis)
                results["error_counts"]["successful_analysis"] += 1
                
                if analysis["missing_imports"]:
                    results["files_with_missing_imports"].append(analysis)
                    results["error_counts"]["files_with_missing_imports"] += 1
                    
            elif analysis["error_type"] == "syntax_error":
                results["syntax_errors"].append(analysis)
                results["error_counts"]["syntax_errors"] += 1
                
            elif analysis["error_type"] == "indentation_error":
                results["indentation_errors"].append(analysis)
                results["error_counts"]["indentation_errors"] += 1
                
            elif analysis["error_type"] == "ast_parse_error":
                results["ast_parse_errors"].append(analysis)
                results["error_counts"]["ast_parse_errors"] += 1
                
            elif analysis["error_type"] == "file_not_found":
                results["file_not_found"].append(analysis)
                results["error_counts"]["file_not_found"] += 1
                
            elif analysis["error_type"] == "permission_error":
                results["permission_errors"].append(analysis)
                results["error_counts"]["permission_errors"] += 1
                
            else:
                results["unknown_errors"].append(analysis)
                results["error_counts"]["unknown_errors"] += 1
        
        return results

    def _print_categorized_results(self, results: dict):
        """Print categorized analysis results."""
        tprint(f"\n📊 CATEGORIZED ANALYSIS RESULTS:")
        tprint(f"Total files analyzed: {results['total_files']}")
        tprint()
        
        # Print each category
        categories = [
            ("syntax_errors", "🔴 Syntax Errors", "Files with invalid Python syntax"),
            ("indentation_errors", "🟠 Indentation Errors", "Files with indentation issues"),
            ("ast_parse_errors", "🟡 AST Parse Errors", "Files that can't be parsed by AST (semantic issues)"),
            ("file_not_found", "🔵 File Not Found", "Files that don't exist"),
            ("permission_errors", "🟣 Permission Errors", "Files with permission issues"),
            ("unknown_errors", "⚫ Unknown Errors", "Files with unexpected errors"),
            ("successful_analysis", "✅ Successful Analysis", "Files that parsed successfully"),
            ("files_with_missing_imports", "📦 Missing Imports", "Files needing import fixes")
        ]
        
        for category, title, description in categories:
            count = results["error_counts"][category]
            if count > 0:
                tprint(f"{title}: {count} files")
                tprint(f"  {description}")
                
                # Show first few examples
                if category in results and results[category]:
                    examples = results[category][:3]  # Show first 3 examples
                    for example in examples:
                        if isinstance(example, dict) and "file_path" in example:
                            file_path = example["file_path"]
                            if "error_message" in example and example["error_message"]:
                                tprint(f"    - {file_path}: {example['error_message']}")
                            else:
                                tprint(f"    - {file_path}")
                tprint()
        
        # Summary
        total_errors = sum(results["error_counts"][cat] for cat in ["syntax_errors", "indentation_errors", "ast_parse_errors", "file_not_found", "permission_errors", "unknown_errors"])
        tprint(f"📈 SUMMARY:")
        tprint(f"  ✅ Files with valid syntax: {results['error_counts']['successful_analysis']}")
        tprint(f"  ❌ Files with issues: {total_errors}")
        tprint(f"  📦 Files needing import fixes: {results['error_counts']['files_with_missing_imports']}")
        tprint(f"  🔴 Real syntax errors: {results['error_counts']['syntax_errors'] + results['error_counts']['indentation_errors']}")
        tprint(f"  🟡 Semantic/AST issues: {results['error_counts']['ast_parse_errors']}")


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Fix missing imports in Python files")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--issues-file", default="/workspace/code_quality/interaction_analysis.json",
                       help="JSON file with validation issues")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the files (default is dry run)")
    parser.add_argument("--auto-detect", action="store_true",
                       help="Use auto-detection instead of issues file")
    parser.add_argument("--file-pattern", default="**/*.py",
                       help="File pattern for auto-detection (default: **/*.py)")

    args = parser.parse_args()

    fixer = ImportFixer(args.project_root)

    if args.auto_detect:
        # Auto-detect mode: scan all Python files
        tprint("🔍 Auto-detecting missing imports...")
        file_paths = list(Path(args.project_root).glob(args.file_pattern))
        tprint(f"Found {len(file_paths)} Python files to analyze")
        
        result = fixer.auto_fix_all_files([str(f) for f in file_paths], dry_run=not args.fix)
        
        # Save report
        if not args.fix:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"/workspace/code_quality/reports/auto_import_fixes_report_{timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2)
            tprint(f"\nReport saved to: {report_file}")
    else:
        # Issues file mode: use existing issues
        fixer.load_issues(args.issues_file)
        result = fixer.fix_all_imports(dry_run=not args.fix)

        # Save report
        if not args.fix:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"/workspace/code_quality/reports/import_fixes_report_{timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2)
            tprint(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
