# Pipeline framework for Code Quality Analysis
#
# This module provides comprehensive pipeline classes for different aspects of code quality analysis:
# - Syntax validation and code quality
# - Import analysis and dependency mapping
# - Import-free analysis for troubleshooting
# - Dead code detection and elimination
# - Code graph construction and mapping
# - Complexity analysis and metrics
# - Auto-fixer for automatic code improvements

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, PipelineStatus, PipelineResult, StageResult
from .syntax_validation_pipeline import SyntaxValidationPipeline, run_syntax_validation
from .import_analysis_pipeline import ImportAnalysisPipeline, run_import_analysis
from .import_free_analysis_pipeline import ImportFreeAnalysisPipeline, run_import_free_analysis
from .dead_code_analysis_pipeline import DeadCodeAnalysisPipeline, run_dead_code_analysis
from .code_graph_pipeline import CodeGraphPipeline, run_code_graph_analysis
from .complexity_analysis_pipeline import ComplexityAnalysisPipeline, run_complexity_analysis
from .auto_fixer_pipeline import AutoFixerPipeline, run_auto_fixer
from .function_import_analysis_pipeline import FunctionImportAnalysisPipeline, run_function_import_analysis
from .simple_plugin_pipeline import SimplePluginPipeline, run_simple_plugin_discovery

__all__ = [
    # Base classes
    "BasePipeline",
    "PipelineConfig", 
    "PipelineStage",
    "PipelineStatus",
    "PipelineResult",
    "StageResult",
    
    # Pipeline classes
    "SyntaxValidationPipeline",
    "ImportAnalysisPipeline", 
    "ImportFreeAnalysisPipeline",
    "DeadCodeAnalysisPipeline",
    "CodeGraphPipeline",
    "ComplexityAnalysisPipeline",
    "AutoFixerPipeline",
    "FunctionImportAnalysisPipeline",
    "SimplePluginPipeline",
    
    # Convenience functions
    "run_syntax_validation",
    "run_import_analysis",
    "run_import_free_analysis", 
    "run_dead_code_analysis",
    "run_code_graph_analysis",
    "run_complexity_analysis",
    "run_auto_fixer",
    "run_function_import_analysis",
    "run_simple_plugin_discovery"
]