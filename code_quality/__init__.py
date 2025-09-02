"""
Code Quality Tools - Comprehensive Python code analysis and fixing suite.
"""

from .core.config import (
    CodeQualityConfig,
    AutoFixConfig,
    AnalysisConfig,
    ReportingConfig,
    get_default_config,
    load_config
)

# Delayed imports for optional components to avoid heavy dependencies at import time

def _lazy_import_fixers():
    from .fixers.auto_fixer import AutoFixer  # noqa: F401
    from .fixers.sequential_fixer import SequentialFixer  # noqa: F401
    return AutoFixer, SequentialFixer


def _lazy_import_analyzers():
    from .analyzers.linter_analyzer import LinterAnalyzer  # noqa: F401
    from .analyzers.syntax_validator import SyntaxValidator  # noqa: F401
    from .analyzers.call_graph_analyzer import CallGraphAnalyzer  # noqa: F401
    from .analyzers.dependency_analyzer import DependencyAnalyzer  # noqa: F401
    from .analyzers.import_analyzer import ImportAnalyzer, ImportIssue  # noqa: F401
    from .analyzers.signature_analyzer import (
        SignatureAnalyzer, SignatureIssue, FunctionSignature, FunctionCall  # noqa: F401
    )
    from .analyzers.complexity_analyzer import (
        ComplexityAnalyzer, ModuleComplexity, FunctionComplexity, ClassComplexity, ComplexityMetrics  # noqa: F401
    )
    from .analyzers.dead_code_analyzer import (
        DeadCodeAnalyzer, DeadCodeIssue, DeadCodeReport  # noqa: F401
    )
    return locals()


def _lazy_import_reporters():
    from .reporters.quality_reporter import QualityReporter  # noqa: F401
    from .reporters.error_reporter import (
        ErrorReporter, ErrorReport, ErrorSummary, ErrorCategory, FileErrorSummary  # noqa: F401
    )
    from .reporters.html_reporter import HTMLReporter, HTMLReportConfig  # noqa: F401
    from .reporters.trend_reporter import TrendReporter, TrendReport, TrendPoint, TrendAnalysis  # noqa: F401
    return locals()

from .utils.file_utils import (
    find_python_files,
    is_valid_python_file,
    get_file_info,
    get_directory_stats,
    backup_file,
    restore_file,
    get_file_dependencies,
    find_unused_imports
)


__version__ = "1.0.0"
__author__ = "Code Quality Tools Team"

__all__ = [
    # Core configuration
    "CodeQualityConfig",
    "AutoFixConfig", 
    "AnalysisConfig",
    "ReportingConfig",
    "get_default_config",
    "load_config",
    
    # Quick access functions
    "auto_fix",
    "sequential_fix",
    "analyze_imports",
    "analyze_signatures",
    "analyze_complexity",
    "analyze_dead_code",
    "generate_error_report",
    "generate_html_report",
    "track_quality_trends"
]


# Quick access functions
import os

def auto_fix(target: str, config: CodeQualityConfig = None) -> dict:
    """
    Quick auto-fix for Python code.
    
    Args:
        target: File or directory path
        config: Optional configuration
        
    Returns:
        Fix results
    """
    config = config or get_default_config()
    AutoFixer, _SequentialFixer = _lazy_import_fixers()
    fixer = AutoFixer(config)
    
    if os.path.isfile(target):
        return fixer.fix_file(target)
    else:
        return fixer.fix_all(target)


def sequential_fix(target: str, output_dir: str = None) -> dict:
    """
    Run the sequential auto-fix pipeline on a target.
    
    Args:
        target: File, directory, or comma-separated list of files
        output_dir: Optional output directory for reports
        
    Returns:
        Pipeline results
    """
    _AutoFixer, SequentialFixer = _lazy_import_fixers()
    fixer = SequentialFixer()
    return fixer.run_pipeline(target=target, output_dir=output_dir)


def analyze_imports(target: str, config: CodeQualityConfig = None) -> dict:
    """
    Quick import analysis for Python code.
    
    Args:
        target: File or directory path
        config: Optional configuration
        
    Returns:
        Import analysis results
    """
    config = config or get_default_config()
    ImportAnalyzer = _lazy_import_analyzers()["ImportAnalyzer"]
    
    if os.path.isfile(target):
        return ImportAnalyzer(config).analyze_files([target])
    else:
        return ImportAnalyzer(config).analyze_directory(target)


def analyze_signatures(target: str, config: CodeQualityConfig = None) -> dict:
    """
    Quick function signature analysis for Python code.
    
    Args:
        target: File or directory path
        config: Optional configuration
        
    Returns:
        Signature analysis results
    """
    config = config or get_default_config()
    SignatureAnalyzer = _lazy_import_analyzers()["SignatureAnalyzer"]
    
    if os.path.isfile(target):
        return SignatureAnalyzer(config).analyze_files([target])
    else:
        return SignatureAnalyzer(config).analyze_directory(target)


def analyze_complexity(target: str, config: CodeQualityConfig = None) -> dict:
    """
    Quick complexity analysis for Python code.
    
    Args:
        target: File or directory path
        config: Optional configuration
        
    Returns:
        Complexity analysis results
    """
    config = config or get_default_config()
    ComplexityAnalyzer = _lazy_import_analyzers()["ComplexityAnalyzer"]
    
    if os.path.isfile(target):
        return ComplexityAnalyzer(config).analyze_file(target)
    else:
        return ComplexityAnalyzer(config).analyze_directory(target)


def analyze_dead_code(target: str, config: CodeQualityConfig = None) -> dict:
    """
    Quick dead code analysis for Python code.
    
    Args:
        target: File or directory path
        config: Optional configuration
        
    Returns:
        Dead code analysis results
    """
    config = config or get_default_config()
    DeadCodeAnalyzer = _lazy_import_analyzers()["DeadCodeAnalyzer"]
    
    if os.path.isfile(target):
        return DeadCodeAnalyzer(config).analyze_file(target)
    else:
        return DeadCodeAnalyzer(config).analyze_directory(target)


def generate_error_report(analyzers_results: dict, config: CodeQualityConfig = None):
    """
    Generate comprehensive error report from analyzer results.
    
    Args:
        analyzers_results: Results from various analyzers
        config: Optional configuration
        
    Returns:
        ErrorReport object
    """
    config = config or get_default_config()
    ErrorReporter = _lazy_import_reporters()["ErrorReporter"]
    
    reporter = ErrorReporter(config)
    
    # Add results from different analyzers
    if 'complexity' in analyzers_results:
        complexity_issues = analyzers_results['complexity'].get('issues', [])
        reporter.add_complexity_issues(complexity_issues)
    
    if 'dead_code' in analyzers_results:
        dead_code_issues = analyzers_results['dead_code'].get('issues', [])
        reporter.add_dead_code_issues(dead_code_issues)
    
    return reporter.generate_report()


def generate_html_report(analyzers_results: dict, title: str = "Code Quality Report") -> str:
    """
    Generate HTML report from analyzer results.
    
    Args:
        analyzers_results: Results from various analyzers
        title: Report title
        
    Returns:
        HTML string
    """
    HTMLReporter = _lazy_import_reporters()["HTMLReporter"]
    reporter = HTMLReporter()
    return reporter.generate_from_analyzer_results(analyzers_results, title)


def track_quality_trends(metrics: dict, project_name: str = "default") -> None:
    """
    Track code quality metrics for trend analysis.
    
    Args:
        metrics: Current quality metrics
        project_name: Name of the project
    """
    TrendReporter = _lazy_import_reporters()["TrendReporter"]
    reporter = TrendReporter()
    reporter.add_data_point(metrics, project_name)