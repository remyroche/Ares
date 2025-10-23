#!/usr/bin/env python3
"""Configuration management for code analysis."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from src.common.config.loader import save_to_file as _unified_save_to_file, load_from_file as _unified_load_from_file


@dataclass
class CodeQualityConfig:
    """Main configuration class for code quality analysis."""
    
    # Project settings
    project_root: str = "."
    output_dir: str = "analysis_results"
    
    # Analysis configuration
    analysis_config: Optional['AnalysisConfig'] = None
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.analysis_config is None:
            self.analysis_config = AnalysisConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_root": self.project_root,
            "output_dir": self.output_dir,
            "analysis_config": self.analysis_config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CodeQualityConfig':
        """Create configuration from dictionary."""
        analysis_config = AnalysisConfig.from_dict(config_dict.get("analysis_config", {}))
        return cls(
            project_root=config_dict.get("project_root", "."),
            output_dir=config_dict.get("output_dir", "analysis_results"),
            analysis_config=analysis_config
        )

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        _unified_save_to_file(self, filepath)


@dataclass
class ReportingConfig:
    """Configuration for reporting options."""
    
    # Report formats
    generate_html: bool = True
    generate_json: bool = True
    generate_text: bool = True
    generate_csv: bool = False
    
    # Report content
    include_metrics: bool = True
    include_visualizations: bool = True
    include_recommendations: bool = True
    
    # Output options
    output_directory: str = "reports"
    report_filename: str = "code_quality_report"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "generate_html": self.generate_html,
            "generate_json": self.generate_json,
            "generate_text": self.generate_text,
            "generate_csv": self.generate_csv,
            "include_metrics": self.include_metrics,
            "include_visualizations": self.include_visualizations,
            "include_recommendations": self.include_recommendations,
            "output_directory": self.output_directory,
            "report_filename": self.report_filename
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ReportingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class AnalysisConfig:
    """Configuration for code analysis operations."""
    
    # File patterns to exclude
    exclude_patterns: List[str] = None
    
    # Directories to exclude
    exclude_directories: List[str] = None
    
    # Analysis options
    enable_dead_code_analysis: bool = True
    enable_dependency_analysis: bool = True
    enable_call_graph_analysis: bool = True
    enable_complexity_analysis: bool = True
    
    # Output options
    generate_html_reports: bool = True
    generate_text_reports: bool = True
    generate_json_reports: bool = True
    generate_visualizations: bool = True
    
    # Analysis thresholds
    complexity_threshold: int = 10
    confidence_threshold: float = 0.8
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.exclude_patterns is None:
            self.exclude_patterns = ["*.pyc", "*.pyo", "__pycache__"]
        
        if self.exclude_directories is None:
            self.exclude_directories = ["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "exclude_patterns": self.exclude_patterns,
            "exclude_directories": self.exclude_directories,
            "enable_dead_code_analysis": self.enable_dead_code_analysis,
            "enable_dependency_analysis": self.enable_dependency_analysis,
            "enable_call_graph_analysis": self.enable_call_graph_analysis,
            "enable_complexity_analysis": self.enable_complexity_analysis,
            "generate_html_reports": self.generate_html_reports,
            "generate_text_reports": self.generate_text_reports,
            "generate_json_reports": self.generate_json_reports,
            "generate_visualizations": self.generate_visualizations,
            "complexity_threshold": self.complexity_threshold,
            "confidence_threshold": self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def get_default_config() -> AnalysisConfig:
    """Get default analysis configuration."""
    return AnalysisConfig()


def load_config(config_file: str) -> CodeQualityConfig:
    """Load configuration from a YAML or JSON file using the unified loader."""
    return _unified_load_from_file(config_file, CodeQualityConfig)  # type: ignore[return-value]
