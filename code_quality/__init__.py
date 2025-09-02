"""
Code Quality Tools - A comprehensive suite for Python code analysis, auto-fixing, and reporting.

This package provides tools for:
- Auto-fixing Python syntax and style issues
- Comprehensive code quality analysis
- Call graph mapping and dead code detection
- Dependency analysis and management
- Syntax validation and AST parsing
- Detailed reporting and visualization
"""

from .core import (
    CodeQualityConfig,
    AutoFixConfig,
    AnalysisConfig,
    ReportingConfig,
    ConfigManager,
    get_default_config,
    load_config
)

from .fixers.auto_fixer import AutoFixer
from .analyzers.linter_analyzer import LinterAnalyzer, LinterResult
from .analyzers.call_graph_analyzer import CallGraphAnalyzer, CallNode
from .analyzers.dependency_analyzer import DependencyAnalyzer, DependencyInfo
from .analyzers.syntax_validator import SyntaxValidator, SyntaxError, ASTNode
from .reporters.quality_reporter import QualityReporter
from .utils import (
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
    "ConfigManager",
    "get_default_config",
    "load_config",
    
    # Main tools
    "AutoFixer",
    "LinterAnalyzer",
    "CallGraphAnalyzer", 
    "DependencyAnalyzer",
    "SyntaxValidator",
    "QualityReporter",
    
    # Data classes
    "LinterResult",
    "CallNode",
    "DependencyInfo",
    "SyntaxError",
    "ASTNode",
    
    # Utility functions
    "find_python_files",
    "is_valid_python_file",
    "get_file_info",
    "get_directory_stats",
    "backup_file",
    "restore_file",
    "get_file_dependencies",
    "find_unused_imports"
]

# Quick access functions for common operations
def quick_analysis(directory: str, output_dir: str = None) -> dict:
    """
    Run a quick comprehensive analysis of a directory.
    
    Args:
        directory: Directory to analyze
        output_dir: Optional output directory for reports
        
    Returns:
        Analysis results
    """
    reporter = QualityReporter()
    return reporter.generate_comprehensive_report(directory, output_dir=output_dir)

def auto_fix(directory: str) -> dict:
    """
    Auto-fix common issues in a directory.
    
    Args:
        directory: Directory to fix
        
    Returns:
        Fix results
    """
    fixer = AutoFixer()
    return fixer.fix_all(directory)

def validate_syntax(directory: str) -> dict:
    """
    Validate syntax for all Python files in a directory.
    
    Args:
        directory: Directory to validate
        
    Returns:
        Validation results
    """
    validator = SyntaxValidator()
    return validator.validate_directory(directory)

def analyze_dependencies(directory: str) -> dict:
    """
    Analyze dependencies for all Python files in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dependency analysis results
    """
    analyzer = DependencyAnalyzer()
    return analyzer.analyze_directory(directory)

def map_call_graph(directory: str) -> dict:
    """
    Map the call graph for all Python files in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Call graph analysis results
    """
    analyzer = CallGraphAnalyzer()
    return analyzer.analyze_directory(directory)