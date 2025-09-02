"""
Comprehensive File Validation Module

This module provides comprehensive file validation capabilities for the Ares trading system,
including validation of various file formats, data quality checks, and validation result reporting.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger

class ValidationSeverity(Enum):
    """Enumeration for validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    issue_type: str
    severity: ValidationSeverity
    description: str
    details: Optional[Dict[str, Any]] = None
    affected_columns: Optional[List[str]] = None
    affected_rows: Optional[List[int]] = None

@dataclass
class FileValidationResult:
    """Represents the result of a file validation operation."""
    is_valid: bool
    issues: List[ValidationIssue]
    file_path: str
    file_size: Optional[int] = None
    validation_timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ComprehensiveFileValidator:
    """Comprehensive file validator for various file formats and data quality checks."""
    
    def __init__(self):
        """Initialize the ComprehensiveFileValidator."""
        self.logger = system_logger.getChild("ComprehensiveFileValidator")
        self.is_initialized = False
        self.validation_rules = {}
        self.supported_formats = ['.csv', '.parquet', '.json', '.pkl', '.pickle', '.h5', '.hdf5']
        
    async def initialize(self) -> bool:
        """Initialize the validator."""
        try:
            self.logger.info("🚀 Initializing ComprehensiveFileValidator...")
            
            # Load validation rules
            await self._load_validation_rules()
            
            self.is_initialized = True
            self.logger.info("✅ ComprehensiveFileValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing ComprehensiveFileValidator: {e}")
            return False
    
    async def _load_validation_rules(self) -> None:
        """Load validation rules from configuration."""
        try:
            # Default validation rules
            self.validation_rules = {
                'csv': {
                    'max_file_size_mb': 1000,
                    'required_columns': [],
                    'max_rows': 10000000,
                    'encoding': ['utf-8', 'latin-1']
                },
                'parquet': {
                    'max_file_size_mb': 2000,
                    'required_columns': [],
                    'max_rows': 50000000
                },
                'json': {
                    'max_file_size_mb': 500,
                    'max_depth': 10,
                    'max_items': 1000000
                },
                'pkl': {
                    'max_file_size_mb': 1000,
                    'allowed_types': ['DataFrame', 'Series', 'dict', 'list']
                }
            }
            
            self.logger.info(f"Loaded validation rules for {len(self.validation_rules)} formats")
        except Exception as e:
            self.logger.error(f"Error loading validation rules: {e}")
    
    async def validate_file(self, file_path: str, expected_schema: Optional[Dict] = None) -> FileValidationResult:
        """
        Validate a file asynchronously.
        
        Args:
            file_path: Path to the file to validate
            expected_schema: Expected schema for validation
            
        Returns:
            FileValidationResult containing validation results
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            self.logger.info(f"🔍 Validating file: {file_path}")
            
            # Basic file existence and accessibility checks
            if not os.path.exists(file_path):
                return FileValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        issue_type="file_not_found",
                        severity=ValidationSeverity.ERROR,
                        description=f"File not found: {file_path}"
                    )],
                    file_path=file_path
                )
            
            # Get file information
            file_size = os.path.getsize(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            # Validate file format
            if file_extension not in self.supported_formats:
                return FileValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        issue_type="unsupported_format",
                        severity=ValidationSeverity.ERROR,
                        description=f"Unsupported file format: {file_extension}"
                    )],
                    file_path=file_path,
                    file_size=file_size
                )
            
            # Apply format-specific validation
            validation_result = await self._validate_by_format(file_path, file_extension, expected_schema)
            
            # Add file metadata
            validation_result.file_size = file_size
            validation_result.validation_timestamp = asyncio.get_event_loop().time()
            
            self.logger.info(f"✅ File validation completed: {file_path}")
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating file {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    def validate_file_sync(self, file_path: str, expected_schema: Optional[Dict] = None) -> FileValidationResult:
        """
        Validate a file synchronously.
        
        Args:
            file_path: Path to the file to validate
            expected_schema: Expected schema for validation
            
        Returns:
            FileValidationResult containing validation results
        """
        try:
            # Run async validation in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.validate_file(file_path, expected_schema))
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.exception(f"❌ Error in sync validation for {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="sync_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Sync validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def _validate_by_format(self, file_path: str, file_extension: str, expected_schema: Optional[Dict]) -> FileValidationResult:
        """Validate file based on its format."""
        try:
            if file_extension == '.csv':
                return await self._validate_csv(file_path, expected_schema)
            elif file_extension == '.parquet':
                return await self._validate_parquet(file_path, expected_schema)
            elif file_extension == '.json':
                return await self._validate_json(file_path, expected_schema)
            elif file_extension in ['.pkl', '.pickle']:
                return await self._validate_pickle(file_path, expected_schema)
            elif file_extension in ['.h5', '.hdf5']:
                return await self._validate_hdf5(file_path, expected_schema)
            else:
                return FileValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        issue_type="unknown_format",
                        severity=ValidationSeverity.ERROR,
                        description=f"Unknown file format: {file_extension}"
                    )],
                    file_path=file_path
                )
        except Exception as e:
            self.logger.error(f"Error in format-specific validation for {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="format_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Format validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def _validate_csv(self, file_path: str, expected_schema: Optional[Dict]) -> FileValidationResult:
        """Validate CSV file."""
        try:
            import pandas as pd
            
            # Read CSV file
            df = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows for validation
            
            issues = []
            
            # Check for required columns if schema is provided
            if expected_schema and 'required_columns' in expected_schema:
                required_cols = expected_schema['required_columns']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    issues.append(ValidationIssue(
                        issue_type="missing_required_columns",
                        severity=ValidationSeverity.ERROR,
                        description=f"Missing required columns: {missing_cols}",
                        affected_columns=missing_cols
                    ))
            
            # Check for empty DataFrame
            if df.empty:
                issues.append(ValidationIssue(
                    issue_type="empty_dataframe",
                    severity=ValidationSeverity.WARNING,
                    description="CSV file contains no data"
                ))
            
            # Check for excessive missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if missing_ratio > 0.5:
                issues.append(ValidationIssue(
                    issue_type="high_missing_values",
                    severity=ValidationSeverity.WARNING,
                    description=f"High ratio of missing values: {missing_ratio:.2%}"
                ))
            
            is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0
            
            return FileValidationResult(
                is_valid=is_valid,
                issues=issues,
                file_path=file_path
            )
            
        except Exception as e:
            self.logger.error(f"Error validating CSV file {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="csv_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"CSV validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def _validate_parquet(self, file_path: str, expected_schema: Optional[Dict]) -> FileValidationResult:
        """Validate Parquet file."""
        try:
            import pandas as pd
            
            # Read Parquet file metadata
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            issues = []
            
            # Check for required columns if schema is provided
            if expected_schema and 'required_columns' in expected_schema:
                required_cols = expected_schema['required_columns']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    issues.append(ValidationIssue(
                        issue_type="missing_required_columns",
                        severity=ValidationSeverity.ERROR,
                        description=f"Missing required columns: {missing_cols}",
                        affected_columns=missing_cols
                    ))
            
            # Check for empty DataFrame
            if df.empty:
                issues.append(ValidationIssue(
                    issue_type="empty_dataframe",
                    severity=ValidationSeverity.WARNING,
                    description="Parquet file contains no data"
                ))
            
            is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0
            
            return FileValidationResult(
                is_valid=is_valid,
                issues=issues,
                file_path=file_path
            )
            
        except Exception as e:
            self.logger.error(f"Error validating Parquet file {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="parquet_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Parquet validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def _validate_json(self, file_path: str, expected_schema: Optional[Dict]) -> FileValidationResult:
        """Validate JSON file."""
        try:
            import json
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            issues = []
            
            # Check for empty data
            if not data:
                issues.append(ValidationIssue(
                    issue_type="empty_data",
                    severity=ValidationSeverity.WARNING,
                    description="JSON file contains no data"
                ))
            
            # Check data structure if schema is provided
            if expected_schema and 'required_keys' in expected_schema:
                required_keys = expected_schema['required_keys']
                if isinstance(data, dict):
                    missing_keys = [key for key in required_keys if key not in data]
                    if missing_keys:
                        issues.append(ValidationIssue(
                            issue_type="missing_required_keys",
                            severity=ValidationSeverity.ERROR,
                            description=f"Missing required keys: {missing_keys}"
                        ))
            
            is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0
            
            return FileValidationResult(
                is_valid=is_valid,
                issues=issues,
                file_path=file_path
            )
            
        except Exception as e:
            self.logger.error(f"Error validating JSON file {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="json_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"JSON validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def _validate_pickle(self, file_path: str, expected_schema: Optional[Dict]) -> FileValidationResult:
        """Validate Pickle file."""
        try:
            import pickle
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            issues = []
            
            # Check for empty data
            if not data:
                issues.append(ValidationIssue(
                    issue_type="empty_data",
                    severity=ValidationSeverity.WARNING,
                    description="Pickle file contains no data"
                ))
            
            # Check data type if schema is provided
            if expected_schema and 'allowed_types' in expected_schema:
                allowed_types = expected_schema['allowed_types']
                data_type = type(data).__name__
                if data_type not in allowed_types:
                    issues.append(ValidationIssue(
                        issue_type="invalid_data_type",
                        severity=ValidationSeverity.WARNING,
                        description=f"Data type {data_type} not in allowed types: {allowed_types}"
                    ))
            
            is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0
            
            return FileValidationResult(
                is_valid=is_valid,
                issues=issues,
                file_path=file_path
            )
            
        except Exception as e:
            self.logger.error(f"Error validating Pickle file {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="pickle_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Pickle validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def _validate_hdf5(self, file_path: str, expected_schema: Optional[Dict]) -> FileValidationResult:
        """Validate HDF5 file."""
        try:
            import h5py
            
            with h5py.File(file_path, 'r') as f:
                # Check for empty file
                if not f.keys():
                    return FileValidationResult(
                        is_valid=False,
                        issues=[ValidationIssue(
                            issue_type="empty_hdf5",
                            severity=ValidationSeverity.WARNING,
                            description="HDF5 file contains no datasets"
                        )],
                        file_path=file_path
                    )
                
                # Check for required datasets if schema is provided
                issues = []
                if expected_schema and 'required_datasets' in expected_schema:
                    required_datasets = expected_schema['required_datasets']
                    missing_datasets = [ds for ds in required_datasets if ds not in f.keys()]
                    if missing_datasets:
                        issues.append(ValidationIssue(
                            issue_type="missing_required_datasets",
                            severity=ValidationSeverity.ERROR,
                            description=f"Missing required datasets: {missing_datasets}"
                        ))
                
                is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0
                
                return FileValidationResult(
                    is_valid=is_valid,
                    issues=issues,
                    file_path=file_path
                )
                
        except Exception as e:
            self.logger.error(f"Error validating HDF5 file {file_path}: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="hdf5_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"HDF5 validation error: {str(e)}"
                )],
                file_path=file_path
            )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.logger.info("🧹 Cleaning up ComprehensiveFileValidator...")
            self.validation_rules.clear()
            self.is_initialized = False
            self.logger.info("✅ ComprehensiveFileValidator cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Convenience function for creating validator instance
async def create_file_validator() -> ComprehensiveFileValidator:
    """Create and initialize a ComprehensiveFileValidator instance."""
    validator = ComprehensiveFileValidator()
    success = await validator.initialize()
    if not success:
        raise RuntimeError("Failed to initialize ComprehensiveFileValidator")
    return validator