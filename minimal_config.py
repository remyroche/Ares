"""
Minimal configuration for code quality tools.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class CodeQualityConfig:
    """Configuration for code quality tools."""
    auto_fix: bool = True
    linters: List[str] = field(default_factory=lambda: ["flake8", "pylint"])
    complexity_threshold: int = 10
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git", "venv", "env"])
    output_format: str = "terminal"
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        pass


def get_default_config() -> CodeQualityConfig:
    """Get the default configuration."""
    return CodeQualityConfig()