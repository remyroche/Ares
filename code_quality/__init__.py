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

from .fixers.auto_fixer import AutoFixer
from .fixers.sequential_fixer import SequentialFixer

from .analyzers.linter_analyzer import LinterAnalyzer
from .analyzers.syntax_validator import SyntaxValidator
from .analyzers.call_graph_analyzer import CallGraphAnalyzer
from .analyzers.dependency_analyzer import DependencyAnalyzer
from .analyzers.import_analyzer import ImportAnalyzer, ImportIssue
from .analyzers.signature_analyzer import SignatureAnalyzer, SignatureIssue, FunctionSignature, FunctionCall

from .reporters.quality_reporter import QualityReporter

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
    
    # Fixers
    "AutoFixer",
    "SequentialFixer",
    
    # Analyzers
    "LinterAnalyzer",
    "SyntaxValidator",
    "CallGraphAnalyzer",
    "DependencyAnalyzer",
    "ImportAnalyzer",
    "SignatureAnalyzer",
    
    # Issue classes
    "ImportIssue",
    "SignatureIssue",
    "FunctionSignature",
    "FunctionCall",
    
    # Reporters
    "QualityReporter",
    
    # Utilities
    "find_python_files",
    "is_valid_python_file",
    "get_file_info",
    "get_directory_stats",
    "backup_file",
    "restore_file",
    "get_file_dependencies",
    "find_unused_imports"
]


# Quick access functions
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
    analyzer = ImportAnalyzer(config)
    
    if os.path.isfile(target):
        return analyzer.analyze_files([target])
    else:
        return analyzer.analyze_directory(target)


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
    analyzer = SignatureAnalyzer(config)
    
    if os.path.isfile(target):
        return analyzer.analyze_files([target])
    else:
        return analyzer.analyze_directory(target)


# Add missing import
import os