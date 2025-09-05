"""
Code Quality Tools - Comprehensive Python code analysis and fixing suite.
"""

from .core.config import (
    AnalysisConfig,
)

# Delayed imports for optional components to avoid heavy dependencies at import time

def _lazy_import_fixers():
    from .fixers.auto_fixer import AutoFixer  # noqa: F401
    from .fixers.sequential_fixer import SequentialFixer  # noqa: F401
    return AutoFixer, SequentialFixer


def _lazy_import_analyzers():
    from .analyzers.complexity_analyzer import (
        ComplexityAnalyzer,  # noqa: F401
        ClassComplexity,
        ComplexityMetrics,
        FunctionComplexity,
        ModuleComplexity,
    )
    from .analyzers.dead_code_analyzer import (
        DeadCodeAnalyzer,  # noqa: F401
        DeadCodeIssue,
        DeadCodeReport,
    )
    from .analyzers.signature_analyzer import (
        SignatureAnalyzer,  # noqa: F401
        FunctionCall,
        FunctionSignature,
        SignatureIssue,
    )
    return locals()


def _lazy_import_reporters():
    from .reporters.error_reporter import (
        ErrorReporter,  # noqa: F401
        ErrorCategory,
        ErrorReport,
        ErrorSummary,
        FileErrorSummary,
    )
    from .reporters.trend_reporter import (
        TrendAnalysis,
        TrendPoint,
        TrendReport,
        TrendReporter,
    )
    return locals()

from .utils.file_utils import (
    find_python_files,
    read_file_safely,
    parse_ast_safely,
    extract_function_name_from_issue,
    get_module_from_file_path,
    is_documentation_file,
)

__version__ = "1.0.0"
__author__ = "Code Quality Tools Team"

__all__ = [
    # Core configuration
    "AnalysisConfig",
    "get_default_config",
    "find_python_files",
    "read_file_safely",
    "parse_ast_safely",
    "extract_function_name_from_issue",
    "get_module_from_file_path",
    "is_documentation_file",
]

