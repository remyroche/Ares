"""
Configuration management for code quality tools.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AutoFixConfig:
    """Configuration for auto-fixing tools."""
    enabled: bool = True
    tools: List[str] = field(default_factory=lambda: ["black", "isort", "autopep8"])
    max_line_length: int = 88
    aggressive: bool = False
    skip_errors: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Configuration for analysis tools."""
    linters: List[str] = field(default_factory=lambda: ["flake8", "pylint", "mypy"])
    complexity_threshold: int = 10
    security_checks: bool = True
    dead_code_detection: bool = True
    dependency_analysis: bool = True
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])


@dataclass
class ReportingConfig:
    """Configuration for reporting tools."""
    output_format: List[str] = field(default_factory=lambda: ["terminal", "html"])
    include_metrics: bool = True
    save_reports: bool = True
    report_dir: str = "code_quality_reports"
    verbose: bool = False


@dataclass
class CodeQualityConfig:
    """Main configuration class for code quality tools."""
    auto_fix: AutoFixConfig = field(default_factory=AutoFixConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    project_root: Optional[str] = None
    
    def __post_init__(self):
        if self.project_root is None:
            self.project_root = os.getcwd()


class ConfigManager:
    """Manages configuration loading and validation."""
    
    DEFAULT_CONFIG = {
        "auto_fix": {
            "enabled": True,
            "tools": ["black", "isort", "autopep8"],
            "max_line_length": 88,
            "aggressive": False,
            "skip_errors": []
        },
        "analysis": {
            "linters": ["flake8", "pylint", "mypy"],
            "complexity_threshold": 10,
            "security_checks": True,
            "dead_code_detection": True,
            "dependency_analysis": True,
            "exclude_patterns": ["__pycache__", "*.pyc", ".git"]
        },
        "reporting": {
            "output_format": ["terminal", "html"],
            "include_metrics": True,
            "save_reports": True,
            "report_dir": "code_quality_reports",
            "verbose": False
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> CodeQualityConfig:
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    return self._merge_configs(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration.")
        
        return self._create_config_from_dict(self.DEFAULT_CONFIG)
    
    def _merge_configs(self, file_config: Dict[str, Any]) -> CodeQualityConfig:
        """Merge file configuration with defaults."""
        merged = self.DEFAULT_CONFIG.copy()
        
        if "code_quality" in file_config:
            file_config = file_config["code_quality"]
        
        for section in ["auto_fix", "analysis", "reporting"]:
            if section in file_config:
                merged[section].update(file_config[section])
        
        return self._create_config_from_dict(merged)
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> CodeQualityConfig:
        """Create configuration object from dictionary."""
        return CodeQualityConfig(
            auto_fix=AutoFixConfig(**config_dict.get("auto_fix", {})),
            analysis=AnalysisConfig(**config_dict.get("analysis", {})),
            reporting=ReportingConfig(**config_dict.get("reporting", {})),
            project_root=os.getcwd()
        )
    
    def get_config(self) -> CodeQualityConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        # This is a simplified update - in practice, you'd want more sophisticated merging
        if "auto_fix" in updates:
            for key, value in updates["auto_fix"].items():
                setattr(self.config.auto_fix, key, value)
        
        if "analysis" in updates:
            for key, value in updates["analysis"].items():
                setattr(self.config.analysis, key, value)
        
        if "reporting" in updates:
            for key, value in updates["reporting"].items():
                setattr(self.config.reporting, key, value)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path or "code_quality_config.yaml"
        
        config_dict = {
            "code_quality": {
                "auto_fix": {
                    "enabled": self.config.auto_fix.enabled,
                    "tools": self.config.auto_fix.tools,
                    "max_line_length": self.config.auto_fix.max_line_length,
                    "aggressive": self.config.auto_fix.aggressive,
                    "skip_errors": self.config.auto_fix.skip_errors
                },
                "analysis": {
                    "linters": self.config.analysis.linters,
                    "complexity_threshold": self.config.analysis.complexity_threshold,
                    "security_checks": self.config.analysis.security_checks,
                    "dead_code_detection": self.config.analysis.dead_code_detection,
                    "dependency_analysis": self.config.analysis.dependency_analysis,
                    "exclude_patterns": self.config.analysis.exclude_patterns
                },
                "reporting": {
                    "output_format": self.config.reporting.output_format,
                    "include_metrics": self.config.reporting.include_metrics,
                    "save_reports": self.config.reporting.save_reports,
                    "report_dir": self.config.reporting.report_dir,
                    "verbose": self.config.reporting.verbose
                }
            }
        }
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")


def get_default_config() -> CodeQualityConfig:
    """Get default configuration."""
    return ConfigManager().get_config()


def load_config(config_path: str) -> CodeQualityConfig:
    """Load configuration from file."""
    return ConfigManager(config_path).get_config()


def save_config(config: CodeQualityConfig, path: str) -> None:
    """Save configuration to file."""
    config_manager = ConfigManager()
    config_manager.config = config
    config_manager.save_config(path)