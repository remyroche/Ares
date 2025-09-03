"""
Minimal configuration module for code quality analysis.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""
    exclude_patterns: list[str]
    include_patterns: list[str]
    max_file_size: int
    timeout: int


@dataclass
class CodeQualityConfig:
    """Main configuration for code quality analysis."""
    analysis: AnalysisConfig
    output: dict[str, Any]
    logging: dict[str, Any]


def get_default_config() -> CodeQualityConfig:
    """Get default configuration."""
    return CodeQualityConfig(
        analysis=AnalysisConfig(
            exclude_patterns=[
                "__pycache__",
                "*.pyc",
                "*.pyo",
                "*.pyd",
                ".git",
                ".svn",
                ".hg",
                "venv",
                "env",
                "node_modules",
                ".tox",
                ".pytest_cache",
            ],
            include_patterns=["*.py"],
            max_file_size=10 * 1024 * 1024,  # 10MB
            timeout=30,
        ),
        output={
            "format": "json",
            "include_details": True,
            "include_metrics": True,
        },
        logging={
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
    )
