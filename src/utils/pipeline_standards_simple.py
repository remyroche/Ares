from __future__ import annotations
"""
Simplified Pipeline Standards and Utilities

This module provides standardized utilities for the data pipeline without external dependencies.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DataQualityLevel(Enum):
    """Data quality levels for validation."""
    CRITICAL = 'critical'
    WARNING = 'warning'
    INFO = 'info'


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    severity: DataQualityLevel
    message: str
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    quality_score: float = 1.0


class PipelineStandards:
    """Pipeline standards and utilities without external dependencies."""
    
    # Directory structure templates
    DIRECTORY_STRUCTURE = {
        'data_cache': 'data_cache',
        'models': 'models',
        'logs': 'logs',
        'config': 'config',
        'results': 'results'
    }
    
    def __init__(self, logger: logging.Logger | None=None):
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def safe_import(module_name: str, fallback_value: Any=None, logger: logging.Logger | None=None) -> Any:
        """
        Safely import a module with consistent fallback pattern.

        Args:
            module_name: Name of module to import
            fallback_value: Value to return if import fails
            logger: Logger instance

        Returns:
            Imported module or fallback value
        """
        logger = logger or logging.getLogger(__name__)
        try:
            return __import__(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}. Using fallback value.")
            return fallback_value

    @staticmethod
    def validate_environment_dependencies(required_modules: list[str], logger: logging.Logger | None=None) -> dict[str, bool]:
        """
        Validate that required dependencies are available.

        Args:
            required_modules: List of module names to check
            logger: Logger instance

        Returns:
            Dictionary mapping module names to availability status
        """
        logger = logger or logging.getLogger(__name__)
        availability = {}
        for module in required_modules:
            try:
                __import__(module)
                availability[module] = True
                logger.debug(f"Module {module} is available")
            except ImportError:
                availability[module] = False
                logger.warning(f"Module {module} is not available")
        return availability

    @staticmethod
    def build_path(path_type: str, exchange: str, asset: str, **kwargs) -> str:
        """
        Build standardized path using the directory structure.

        Args:
            path_type: Type of path to build
            exchange: Exchange name
            asset: Asset symbol
            **kwargs: Additional path parameters

        Returns:
            Built path string
        """
        if path_type not in PipelineStandards.DIRECTORY_STRUCTURE:
            msg = f'Unknown path type: {path_type}'
            raise ValueError(msg)
        path_template = PipelineStandards.DIRECTORY_STRUCTURE[path_type]
        return path_template.format(exchange=exchange.lower(), asset=asset.lower(), **kwargs)

    @staticmethod
    def generate_file_name(file_type: str, exchange: str, asset: str, timeframe: str=None, **kwargs) -> str:
        """
        Generate standardized file name.

        Args:
            file_type: Type of file
            exchange: Exchange name
            asset: Asset symbol
            timeframe: Timeframe (optional)
            **kwargs: Additional parameters

        Returns:
            Generated file name
        """
        template = f"{file_type}_{exchange}_{asset}"
        if timeframe:
            template += f"_{timeframe}"
        
        params = {
            'file_type': file_type,
            'exchange': exchange.upper(),
            'asset': asset.upper(),
            'timeframe': timeframe or '',
            **kwargs
        }
        
        return template.format(**params)

    @staticmethod
    def create_metadata(schema_name: str, exchange: str, asset: str, timeframe: str, **kwargs) -> dict[str, Any]:
        """
        Create standardized metadata for files.

        Args:
            schema_name: Schema name
            exchange: Exchange name
            asset: Asset symbol
            timeframe: Timeframe
            **kwargs: Additional metadata

        Returns:
            Metadata dictionary
        """
        return {
            'schema_name': schema_name,
            'exchange': exchange.upper(),
            'asset': asset.upper(),
            'timeframe': timeframe,
            'created_at': datetime.now(UTC).isoformat(),
            'pipeline_version': '1.0.0',
            'data_format': 'parquet',
            'compression': 'snappy',
            **kwargs
        }

    @staticmethod
    def validate_basic_data_quality(file_path: str | Path, min_size: int = 0) -> ValidationResult:
        """
        Basic data quality validation for files.

        Args:
            file_path: Path to file to validate
            min_size: Minimum file size in bytes

        Returns:
            Validation result
        """
        result = ValidationResult(passed=True)
        
        try:
            path = Path(file_path)
            if not path.exists():
                result.passed = False
                result.issues.append(ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"File does not exist: {file_path}"
                ))
                return result
            
            file_size = path.stat().st_size
            if file_size < min_size:
                result.passed = False
                result.issues.append(ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"File too small: {file_size} bytes < {min_size} bytes"
                ))
            
            if file_size == 0:
                result.passed = False
                result.issues.append(ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message="File is empty"
                ))
            
            result.quality_score = 1.0 if result.passed else 0.0
            
        except Exception as e:
            result.passed = False
            result.issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message=f"Error validating file: {e}"
            ))
            result.quality_score = 0.0
        
        return result


# Create global instance
pipeline_standards = PipelineStandards()