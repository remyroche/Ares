"""
Function Signature Analysis - Detects function signature changes and ensures calling code compatibility.
"""

import ast
import os
from collections import defaultdict
from typing import Any, NamedTuple

from ..core.config import CodeQualityConfig

# Built-in functions that should be excluded from analysis
BUILTIN_FUNCTIONS = {
    'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
    'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed', 'sum',
    'max', 'min', 'abs', 'round', 'pow', 'divmod', 'bin', 'hex', 'oct', 'chr',
    'ord', 'hash', 'id', 'type', 'isinstance', 'issubclass', 'hasattr', 'getattr',
    'setattr', 'delattr', 'dir', 'vars', 'globals', 'locals', 'eval', 'exec',
    'compile', 'open', 'input', 'exit', 'quit', 'help', 'repr', 'ascii', 'format',
    'super', 'property', 'staticmethod', 'classmethod', 'all', 'any', 'next',
    'iter', 'callable', 'memoryview', 'slice', 'object', 'Exception', 'ValueError',
    'TypeError', 'AttributeError', 'KeyError', 'IndexError', 'ImportError',
    'ModuleNotFoundError', 'FileNotFoundError', 'OSError', 'RuntimeError',
    'NotImplementedError', 'StopIteration', 'GeneratorExit', 'SystemExit',
    'KeyboardInterrupt', 'BaseException', 'Warning', 'UserWarning', 'DeprecationWarning',
    'PendingDeprecationWarning', 'SyntaxWarning', 'RuntimeWarning', 'FutureWarning',
    'ImportWarning', 'UnicodeWarning', 'BytesWarning', 'ResourceWarning',
    # Common external library functions that are often imported
    'np', 'pd', 'plt', 'sns', 'sklearn', 'tensorflow', 'torch', 'requests',
    'json', 'os', 'sys', 'time', 'datetime', 'pathlib', 'logging', 'asyncio',
    'threading', 'multiprocessing', 'subprocess', 'shutil', 'tempfile', 'uuid',
    'random', 'math', 'statistics', 'collections', 'itertools', 'functools',
    'operator', 'copy', 'pickle', 'csv', 'xml', 'html', 'urllib', 'http',
    'socket', 'ssl', 'hashlib', 'hmac', 'base64', 'zlib', 'gzip', 'bz2',
    'lzma', 'tarfile', 'zipfile', 'sqlite3', 're', 'string', 'unicodedata',
    'codecs', 'io', 'contextlib', 'weakref', 'gc', 'inspect', 'traceback',
    'warnings', 'dis', 'pickletools', 'profile', 'pstats', 'timeit', 'doctest',
    'unittest', 'test', 'pdb', 'cProfile', 'pstats', 'trace', 'faulthandler',
    'signal', 'atexit', 'argparse', 'getopt', 'optparse', 'configparser',
    'fileinput', 'linecache', 'filecmp', 'tempfile', 'glob', 'fnmatch',
    'linecache', 'shlex', 'struct', 'array', 'mmap', 'select', 'selectors',
    'asyncio', 'concurrent', 'queue', 'sched', 'threading', 'multiprocessing',
    'subprocess', 'sched', 'queue', 'dummy_threading', 'dummy_thread',
    'ctypes', 'ctypes.util', 'ctypes.wintypes', 'msvcrt', 'winsound',
    'winreg', 'winsound', 'msvcrt', 'nt', 'posix', 'pwd', 'grp', 'crypt',
    'termios', 'tty', 'pty', 'fcntl', 'pipes', 'resource', 'syslog',
    'getpass', 'curses', 'readline', 'rlcompleter', 'cmd', 'shlex',
    'tkinter', 'turtle', 'turtledemo', 'bdb', 'pdb', 'profile', 'pstats',
    'hotshot', 'timeit', 'trace', 'faulthandler', 'tracemalloc', 'gc',
    'sys', 'builtins', '__builtin__', '__builtins__', 'main', 'if __name__',
    # Common pandas/numpy methods that are often called directly
    'DataFrame', 'Series', 'array', 'zeros', 'ones', 'empty', 'full',
    'arange', 'linspace', 'logspace', 'meshgrid', 'mgrid', 'ogrid',
    'eye', 'identity', 'diag', 'tri', 'tril', 'triu', 'vander',
    'histogram', 'histogram2d', 'histogramdd', 'bincount', 'digitize',
    'searchsorted', 'corrcoef', 'cov', 'polyfit', 'polyval', 'roots',
    'poly', 'polyder', 'polyint', 'polyadd', 'polysub', 'polymul',
    'polydiv', 'polyval', 'polyfit', 'roots', 'poly', 'polyder',
    'polyint', 'polyadd', 'polysub', 'polymul', 'polydiv', 'polyval',
    'polyfit', 'roots', 'poly', 'polyder', 'polyint', 'polyadd',
    'polysub', 'polymul', 'polydiv', 'polyval', 'polyfit', 'roots'
}


class FunctionSignature(NamedTuple):
    """Represents a function signature."""
    name: str
    args: list[str]
    defaults: list[Any]
    vararg: str | None
    kwarg: str | None
    returns: str | None
    decorators: list[str]
    line_number: int
    file_path: str


class FunctionCall(NamedTuple):
    """Represents a function call."""
    function_name: str
    args: list[str]
    keywords: list[tuple[str, str]]
    line_number: int
    file_path: str


class SignatureIssue:
    """Represents a function signature-related issue."""

    def __init__(self, file_path: str, line_number: int, issue_type: str,
                 message: str, severity: str = "warning", details: dict | None = None):
        self.file_path = file_path
        self.line_number = line_number
        self.issue_type = issue_type
        self.message = message
        self.severity = severity
        self.details = details or {}


class SignatureAnalyzer:
    """Analyzes function signatures for changes and compatibility issues."""

    def __init__(self, config: CodeQualityConfig):
        self.config = config
        self.functions_by_file = defaultdict(list)
        self.function_calls_by_file = defaultdict(list)
        self.imports_by_file = defaultdict(list)  # Track imports to avoid false positives
        self.signature_changes = []
        self.compatibility_issues = []
        self.missing_functions = []
        self.unused_functions = []

    def analyze_directory(self, directory_path: str) -> dict[str, Any]:
        """Analyze function signatures in all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.analysis.exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        return self.analyze_files(python_files)

    def analyze_files(self, file_paths: list[str]) -> dict[str, Any]:
        """Analyze function signatures in specific Python files."""
        print(f"Analyzing function signatures in {len(file_paths)} files...")

        # First pass: collect all function definitions and calls
        for file_path in file_paths:
            try:
                self._analyze_file_signatures(file_path)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        # Second pass: detect issues
        self._detect_signature_changes()
        self._detect_compatibility_issues()
        self._detect_missing_functions()
        self._detect_unused_functions()

        return self._generate_report()

    def _analyze_file_signatures(self, file_path: str) -> None:
        """Analyze function signatures in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Extract imports first
            self._extract_imports(tree, file_path)

            # Collect function definitions and calls
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    signature = self._extract_function_signature(node, file_path)
                    self.functions_by_file[file_path].append(signature)

                elif isinstance(node, ast.Call):
                    # Skip decorator calls and other non-function calls
                    if self._is_decorator_call(node, tree):
                        continue
                    if self._is_import_call(node, tree):
                        continue
                    
                    call = self._extract_function_call(node, file_path)
                    if call:
                        self.function_calls_by_file[file_path].append(call)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    def _is_decorator_call(self, call_node: ast.Call, tree: ast.AST) -> bool:
        """Check if a call node is part of a decorator."""
        # Walk up the tree to see if this call is in a decorator list
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    # Check if the call node is the same as the decorator
                    if self._is_same_node(call_node, decorator):
                        return True
                    # Also check if the call node is inside the decorator (for complex decorators)
                    if isinstance(decorator, ast.Call):
                        if self._is_same_node(call_node, decorator):
                            return True
        return False

    def _is_import_call(self, call_node: ast.Call, tree: ast.AST) -> bool:
        """Check if a call node is part of an import statement."""
        # Walk up the tree to see if this call is in an import statement
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Check if the call is part of an import statement
                if hasattr(node, 'names'):
                    for alias in node.names:
                        if hasattr(alias, 'name') and call_node.func and hasattr(call_node.func, 'id'):
                            if alias.name == call_node.func.id:
                                return True
        return False

    def _is_same_node(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two AST nodes are the same (same line and column)."""
        return (hasattr(node1, 'lineno') and hasattr(node2, 'lineno') and 
                hasattr(node1, 'col_offset') and hasattr(node2, 'col_offset') and
                node1.lineno == node2.lineno and node1.col_offset == node2.col_offset)

    def _extract_imports(self, tree: ast.AST, file_path: str) -> None:
        """Extract imports from a file to track available functions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Store both the full name and the alias
                    self.imports_by_file[file_path].append(alias.name)
                    if alias.asname:
                        self.imports_by_file[file_path].append(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        # Store the imported name, the full module path, and any alias
                        imported_name = alias.name
                        full_name = f"{node.module}.{imported_name}"
                        self.imports_by_file[file_path].append(imported_name)
                        self.imports_by_file[file_path].append(full_name)
                        if alias.asname:
                            self.imports_by_file[file_path].append(alias.asname)
                else:
                    # Handle relative imports
                    for alias in node.names:
                        self.imports_by_file[file_path].append(alias.name)
                        if alias.asname:
                            self.imports_by_file[file_path].append(alias.asname)

    def _extract_function_signature(self, node: ast.FunctionDef, file_path: str) -> FunctionSignature:
        """Extract function signature from an AST node."""
        # Extract arguments
        args = []
        defaults = []

        # Positional arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # Default values
        if node.args.defaults:
            # Calculate how many args have defaults
            num_defaults = len(node.args.defaults)
            num_args = len(node.args.args)

            # Add None for args without defaults
            for _i in range(num_args - num_defaults):
                defaults.append(None)

            # Add actual default values
            defaults.extend([self._get_default_value(d) for d in node.args.defaults])
        else:
            defaults = [None] * len(node.args.args)

        # Vararg (*args)
        vararg = node.args.vararg.arg if node.args.vararg else None

        # Kwarg (**kwargs)
        kwarg = node.args.kwarg.arg if node.args.kwarg else None

        # Return annotation
        returns = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                returns = str(node.returns.value)
            else:
                returns = ast.unparse(node.returns) if hasattr(ast, "unparse") else str(node.returns)

        # Decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
            else:
                decorators.append(str(decorator))

        return FunctionSignature(
            name=node.name,
            args=args,
            defaults=defaults,
            vararg=vararg,
            kwarg=kwarg,
            returns=returns,
            decorators=decorators,
            line_number=node.lineno,
            file_path=file_path,
        )

    def _get_default_value(self, node: ast.AST) -> Any:
        """Extract the default value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return "[]"
        if isinstance(node, ast.Dict):
            return "{}"
        if isinstance(node, ast.Tuple):
            return "()"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return f"{node.func.id}()"
        return str(node)

    def _is_likely_class_instantiation(self, node: ast.Call, function_name: str) -> bool:
        """Check if a call is likely a class instantiation rather than a function call."""
        # Class names typically start with uppercase letters
        if function_name[0].isupper():
            return True
        
        # Check if it's assigned to a variable (common pattern for class instantiation)
        # This is a heuristic - we'd need to analyze the AST more deeply for certainty
        return False

    def _extract_function_call(self, node: ast.Call, file_path: str) -> FunctionCall | None:
        """Extract function call information from an AST node."""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
            is_method_call = False
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
            is_method_call = True
        else:
            return None

        # Skip built-in functions
        if function_name in BUILTIN_FUNCTIONS:
            return None

        # Skip method calls - these are handled by the object's class definition
        if is_method_call:
            return None

        # Skip very short function names (likely variables or parameters)
        if len(function_name) <= 2:
            return None

        # Skip common variable names that might be called as functions
        common_variables = {'x', 'y', 'z', 'i', 'j', 'k', 'n', 'm', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w'}
        if function_name.lower() in common_variables:
            return None

        # Skip common imported functions that are typically available
        common_imports = {
            'Path', 'datetime', 'timedelta', 'timezone', 'date', 'time',
            'DataFrame', 'Series', 'array', 'zeros', 'ones', 'empty', 'full',
            'arange', 'linspace', 'logspace', 'meshgrid', 'mgrid', 'ogrid',
            'eye', 'identity', 'diag', 'tri', 'tril', 'triu', 'vander',
            'histogram', 'histogram2d', 'histogramdd', 'bincount', 'digitize',
            'searchsorted', 'corrcoef', 'cov', 'polyfit', 'polyval', 'roots',
            'poly', 'polyder', 'polyint', 'polyadd', 'polysub', 'polymul',
            'polydiv', 'polyval', 'polyfit', 'roots', 'poly', 'polyder',
            'polyint', 'polyadd', 'polysub', 'polymul', 'polydiv', 'polyval',
            'polyfit', 'roots', 'poly', 'polyder', 'polyint', 'polyadd',
            'polysub', 'polymul', 'polydiv', 'polyval', 'polyfit', 'roots',
            'setup_signal_handlers', 'test_func', 'main', 'run', 'start',
            'stop', 'pause', 'resume', 'init', 'cleanup', 'configure',
            'validate', 'process', 'handle', 'execute', 'perform', 'create',
            'destroy', 'update', 'refresh', 'reload', 'reset', 'clear',
            'load', 'save', 'export', 'import', 'parse', 'format', 'encode',
            'decode', 'serialize', 'deserialize', 'compress', 'decompress',
            'encrypt', 'decrypt', 'hash', 'sign', 'verify', 'authenticate',
            'authorize', 'login', 'logout', 'register', 'unregister', 'subscribe',
            'unsubscribe', 'publish', 'notify', 'alert', 'warn', 'error',
            'debug', 'info', 'trace', 'log', 'monitor', 'track', 'measure',
            'calculate', 'compute', 'estimate', 'predict', 'forecast', 'analyze',
            'evaluate', 'assess', 'score', 'rank', 'sort', 'filter', 'search',
            'find', 'locate', 'detect', 'identify', 'recognize', 'classify',
            'categorize', 'group', 'cluster', 'aggregate', 'summarize', 'report',
            'generate', 'produce', 'create', 'build', 'construct', 'assemble',
            'compose', 'combine', 'merge', 'join', 'split', 'divide', 'separate',
            'extract', 'isolate', 'remove', 'delete', 'clear', 'clean', 'purge',
            'truncate', 'cut', 'slice', 'chop', 'trim', 'strip', 'pad', 'fill',
            'expand', 'contract', 'shrink', 'grow', 'increase', 'decrease',
            'add', 'subtract', 'multiply', 'divide', 'power', 'root', 'log',
            'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh',
            'tanh', 'asinh', 'acosh', 'atanh', 'ceil', 'floor', 'round', 'trunc',
            'abs', 'sign', 'sqrt', 'cbrt', 'factorial', 'gcd', 'lcm', 'mod',
            'divmod', 'pow', 'bin', 'hex', 'oct', 'chr', 'ord', 'ascii', 'repr',
            'str', 'int', 'float', 'bool', 'complex', 'bytes', 'bytearray',
            'memoryview', 'list', 'tuple', 'set', 'frozenset', 'dict', 'range',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed', 'sum',
            'max', 'min', 'all', 'any', 'next', 'iter', 'len', 'type', 'isinstance',
            'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'dir',
            'vars', 'globals', 'locals', 'eval', 'exec', 'compile', 'open',
            'input', 'print', 'help', 'quit', 'exit', 'copyright', 'credits',
            'license', 'reload', 'super', 'property', 'staticmethod', 'classmethod',
            'callable', 'hash', 'id', 'object', 'Exception', 'ValueError',
            'TypeError', 'AttributeError', 'KeyError', 'IndexError', 'ImportError',
            'ModuleNotFoundError', 'FileNotFoundError', 'OSError', 'RuntimeError',
            'NotImplementedError', 'StopIteration', 'GeneratorExit', 'SystemExit',
            'KeyboardInterrupt', 'BaseException', 'Warning', 'UserWarning',
            'DeprecationWarning', 'PendingDeprecationWarning', 'SyntaxWarning',
            'RuntimeWarning', 'FutureWarning', 'ImportWarning', 'UnicodeWarning',
            'BytesWarning', 'ResourceWarning'
        }
        
        if function_name in common_imports:
            return None

        # Skip class instantiations (they look like function calls but are actually constructors)
        if self._is_likely_class_instantiation(node, function_name):
            return None

        # Extract arguments
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args.append(arg.id)
            elif isinstance(arg, ast.Constant):
                args.append(str(arg.value))
            else:
                args.append(str(arg))

        # Extract keyword arguments
        keywords = []
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name):
                keywords.append((keyword.arg, keyword.value.id))
            elif isinstance(keyword.value, ast.Constant):
                keywords.append((keyword.arg, str(keyword.value.value)))
            else:
                keywords.append((keyword.arg, str(keyword.value)))

        return FunctionCall(
            function_name=function_name,
            args=args,
            keywords=keywords,
            line_number=node.lineno,
            file_path=file_path,
        )

    def _detect_signature_changes(self) -> None:
        """Detect function signature changes across files."""
        # Group functions by name across all files
        functions_by_name = defaultdict(list)
        for functions in self.functions_by_file.values():
            for func in functions:
                functions_by_name[func.name].append(func)

        # Check for signature changes
        for func_name, func_list in functions_by_name.items():
            if len(func_list) > 1:
                # Compare signatures
                base_signature = func_list[0]
                for other_func in func_list[1:]:
                    if self._signatures_differ(base_signature, other_func):
                        self.signature_changes.append(SignatureIssue(
                            file_path=other_func.file_path,
                            line_number=other_func.line_number,
                            issue_type="signature_change",
                            message=f"Function '{func_name}' has different signature than in {base_signature.file_path}",
                            severity="warning",
                            details={
                                "function_name": func_name,
                                "base_signature": self._signature_to_dict(base_signature),
                                "changed_signature": self._signature_to_dict(other_func),
                                "differences": self._get_signature_differences(base_signature, other_func),
                            },
                        ))

    def _signatures_differ(self, sig1: FunctionSignature, sig2: FunctionSignature) -> bool:
        """Check if two function signatures are different."""
        return (sig1.args != sig2.args or
                sig1.defaults != sig2.defaults or
                sig1.vararg != sig2.vararg or
                sig1.kwarg != sig2.kwarg or
                sig1.returns != sig2.returns)

    def _signature_to_dict(self, sig: FunctionSignature) -> dict[str, Any]:
        """Convert function signature to dictionary for comparison."""
        return {
            "args": sig.args,
            "defaults": sig.defaults,
            "vararg": sig.vararg,
            "kwarg": sig.kwarg,
            "returns": sig.returns,
            "decorators": sig.decorators,
        }

    def _get_signature_differences(self, sig1: FunctionSignature, sig2: FunctionSignature) -> list[str]:
        """Get a list of differences between two signatures."""
        differences = []

        if sig1.args != sig2.args:
            differences.append(f"Arguments: {sig1.args} vs {sig2.args}")

        if sig1.defaults != sig2.defaults:
            differences.append(f"Defaults: {sig1.defaults} vs {sig2.defaults}")

        if sig1.vararg != sig2.vararg:
            differences.append(f"Vararg: {sig1.vararg} vs {sig2.vararg}")

        if sig1.kwarg != sig2.kwarg:
            differences.append(f"Kwarg: {sig1.kwarg} vs {sig2.kwarg}")

        if sig1.returns != sig2.returns:
            differences.append(f"Returns: {sig1.returns} vs {sig2.returns}")

        return differences

    def _detect_compatibility_issues(self) -> None:
        """Detect compatibility issues between function calls and definitions."""
        # Create a map of function definitions
        function_definitions = {}
        for functions in self.functions_by_file.values():
            for func in functions:
                function_definitions[func.name] = func

        # Check each function call against definitions
        for calls in self.function_calls_by_file.values():
            for call in calls:
                # Skip built-in functions even if they're defined in the codebase
                if call.function_name in BUILTIN_FUNCTIONS:
                    continue
                    
                if call.function_name in function_definitions:
                    func_def = function_definitions[call.function_name]
                    issues = self._check_call_compatibility(call, func_def)

                    for issue in issues:
                        self.compatibility_issues.append(SignatureIssue(
                            file_path=call.file_path,
                            line_number=call.line_number,
                            issue_type="compatibility_issue",
                            message=issue,
                            severity="error",
                            details={
                                "function_name": call.function_name,
                                "call": {
                                    "args": call.args,
                                    "keywords": call.keywords,
                                },
                                "definition": self._signature_to_dict(func_def),
                            },
                        ))
                else:
                    # Function not defined anywhere - check if it's imported or should be skipped
                    if self._should_report_missing_function(call):
                        self.missing_functions.append(SignatureIssue(
                            file_path=call.file_path,
                            line_number=call.line_number,
                            issue_type="missing_function",
                            message=f"Function '{call.function_name}' is called but not defined",
                            severity="error",
                            details={
                                "function_name": call.function_name,
                                "call": {
                                    "args": call.args,
                                    "keywords": call.keywords,
                                },
                            },
                        ))

    def _should_report_missing_function(self, call: FunctionCall) -> bool:
        """Determine if a missing function should be reported as an issue."""
        function_name = call.function_name
        
        # Skip built-in functions
        if function_name in BUILTIN_FUNCTIONS:
            return False
        
        # Check if the function is imported in the same file
        file_imports = self.imports_by_file.get(call.file_path, [])
        
        # Check for direct imports (e.g., "defaultdict" from "collections")
        for import_name in file_imports:
            if import_name.endswith(f".{function_name}") or import_name == function_name:
                return False
        
        # Check for common patterns that indicate the function is likely imported
        common_import_patterns = {
            'defaultdict': ['collections'],
            'asdict': ['dataclasses'],
            'field': ['dataclasses'],
            'dataclass': ['dataclasses'],
            'cosine_similarity': ['sklearn.metrics.pairwise'],
            'DataFrame': ['pandas'],
            'Series': ['pandas'],
            'array': ['numpy'],
            'zeros': ['numpy'],
            'ones': ['numpy'],
            'Path': ['pathlib'],
            'datetime': ['datetime'],
            'timedelta': ['datetime'],
            'get_logger': ['centralized_logging', 'logging'],
            'setup_logging': ['src.utils.logger', 'logging'],
            'register_decorator': ['src.utils.decorator_registry'],
            'safe_dict_get': ['src.utils.common_operations'],
            'run_command': ['subprocess'],
            'run_step': ['src.training.steps'],
            'handle_errors': ['ares_launcher'],
            'get_backtesting_logger': ['src.training.steps.backtesting'],
        }
        
        if function_name in common_import_patterns:
            expected_modules = common_import_patterns[function_name]
            for import_name in file_imports:
                for module in expected_modules:
                    if module in import_name:
                        return False
        
        # Skip very generic function names that are likely parameters or variables
        generic_names = {'func', 'callback', 'handler', 'processor', 'method', 'function', 'processor_func'}
        if function_name.lower() in generic_names:
            return False
        
        # Skip single letter function names (likely variables)
        if len(function_name) == 1:
            return False
        
        # Skip very short names (likely variables)
        if len(function_name) <= 2:
            return False
        
        # Skip common variable names that might be called as functions
        common_variables = {'x', 'y', 'z', 'i', 'j', 'k', 'n', 'm', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w'}
        if function_name.lower() in common_variables:
            return False
        
        return True

    def _check_call_compatibility(self, call: FunctionCall, func_def: FunctionSignature) -> list[str]:
        """Check if a function call is compatible with its definition."""
        issues = []

        # Skip issues for very generic function names (likely parameters)
        generic_names = {'func', 'callback', 'handler', 'processor', 'method', 'function', 'processor_func'}
        if call.function_name.lower() in generic_names:
            return []

        # Skip decorator calls - these are handled differently
        if call.function_name in ['handle_errors', 'register_decorator', 'monitor_data_collection', 'handles_errors']:
            return []

        # Check positional arguments
        if len(call.args) > len(func_def.args):
            if not func_def.vararg:
                issues.append(f"Too many positional arguments: {len(call.args)} provided, {len(func_def.args)} expected")

        # Check keyword arguments
        defined_args = set(func_def.args)
        for keyword_name, _ in call.keywords:
            if keyword_name not in defined_args and not func_def.kwarg:
                issues.append(f"Unknown keyword argument: '{keyword_name}'")

        # Check required arguments - improved logic
        # Count how many arguments have default values
        num_defaults = len(func_def.defaults) if func_def.defaults else 0
        num_required = len(func_def.args) - num_defaults
        
        # Only check if we have fewer positional arguments than required
        if len(call.args) < num_required:
            missing = func_def.args[len(call.args):num_required]
            issues.append(f"Missing required arguments: {', '.join(missing)}")

        return issues

    def _detect_missing_functions(self) -> None:
        """Detect functions that are called but not defined."""
        # This is already handled in _detect_compatibility_issues

    def _detect_unused_functions(self) -> None:
        """Detect functions that are defined but never called."""
        # Create a set of all called function names
        called_functions = set()
        for calls in self.function_calls_by_file.values():
            for call in calls:
                called_functions.add(call.function_name)

        # Check for unused functions
        for functions in self.functions_by_file.values():
            for func in functions:
                if func.name not in called_functions:
                    # Check if it's a main function or has special decorators
                    if not self._is_special_function(func):
                        self.unused_functions.append(SignatureIssue(
                            file_path=func.file_path,
                            line_number=func.line_number,
                            issue_type="unused_function",
                            message=f"Function '{func.name}' is defined but never called",
                            severity="warning",
                            details={
                                "function_name": func.name,
                                "signature": self._signature_to_dict(func),
                            },
                        ))

    def _is_special_function(self, func: FunctionSignature) -> bool:
        """Check if a function is special (main, test, etc.)."""
        special_names = {"main", "__main__", "test_", "setup", "teardown"}
        special_decorators = {"pytest", "test", "main", "cli"}

        # Check function name
        if any(name in func.name for name in special_names):
            return True

        # Check decorators
        return bool(any(decorator in str(func.decorators) for decorator in special_decorators))

    def _generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive signature analysis report."""
        total_issues = (len(self.signature_changes) +
                       len(self.compatibility_issues) +
                       len(self.missing_functions) +
                       len(self.unused_functions))

        return {
            "summary": {
                "total_files_analyzed": len(self.functions_by_file),
                "total_functions": sum(len(funcs) for funcs in self.functions_by_file.values()),
                "total_function_calls": sum(len(calls) for calls in self.function_calls_by_file.values()),
                "total_issues": total_issues,
                "signature_changes": len(self.signature_changes),
                "compatibility_issues": len(self.compatibility_issues),
                "missing_functions": len(self.missing_functions),
                "unused_functions": len(self.unused_functions),
            },
            "issues": {
                "signature_changes": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.signature_changes
                ],
                "compatibility_issues": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.compatibility_issues
                ],
                "missing_functions": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.missing_functions
                ],
                "unused_functions": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.unused_functions
                ],
            },
            "functions": {
                file_path: [
                    {
                        "name": func.name,
                        "args": func.args,
                        "defaults": func.defaults,
                        "vararg": func.vararg,
                        "kwarg": func.kwarg,
                        "returns": func.returns,
                        "decorators": func.decorators,
                        "line": func.line_number,
                    }
                    for func in functions
                ]
                for file_path, functions in self.functions_by_file.items()
            },
            "calls": {
                file_path: [
                    {
                        "function_name": call.function_name,
                        "args": call.args,
                        "keywords": call.keywords,
                        "line": call.line_number,
                    }
                    for call in calls
                ]
                for file_path, calls in self.function_calls_by_file.items()
            },
        }

    def get_function_signatures(self) -> dict[str, list[FunctionSignature]]:
        """Get all function signatures by file."""
        return dict(self.functions_by_file)

    def get_function_calls(self) -> dict[str, list[FunctionCall]]:
        """Get all function calls by file."""
        return dict(self.function_calls_by_file)


def analyze_signatures(directory_path: str, config: CodeQualityConfig) -> dict[str, Any]:
    """Convenience function to analyze function signatures in a directory."""
    analyzer = SignatureAnalyzer(config)
    return analyzer.analyze_directory(directory_path)
