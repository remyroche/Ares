#!/usr/bin/env python3
"""
Pipeline Utilities for Data Operations

This module provides comprehensive utilities for data access, analysis,
and manipulation with built-in protection mechanisms.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, AsyncIterator
import pandas as pd
import numpy as np
import pickle
import hashlib

from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.validate import validates
from src.utils.pipeline_decorators import (
    data_reader, data_writer, data_transformer, data_analyzer,
    data_access_control, data_integrity_check, performance_monitor
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory
)


class DataFormat(Enum):
    """Supported data formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    FEATHER = "feather"


class DataAccessMode(Enum):
    """Data access modes."""
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class DataMetadata:
    """Metadata for data operations."""
    source: str
    format: DataFormat
    size_bytes: int
    row_count: int
    column_count: int
    created_at: str
    modified_at: str
    checksum: str
    schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source": self.source,
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "checksum": self.checksum,
            "schema": self.schema,
            "tags": self.tags
        }


class DataProtectionManager:
    """Manages data protection and access control."""
    
    def __init__(self):
        self.logger = logging.getLogger("data_protection_manager")
        self.access_log = []
        self.backup_registry = {}
    
    def log_access(self, operation: str, file_path: str, user: str = "system") -> None:
        """Log data access for audit purposes."""
        access_record = {
            "timestamp": format_datetime(get_current_datetime()),
            "operation": operation,
            "file_path": file_path,
            "user": user
        }
        self.access_log.append(access_record)
        self.logger.info(f"Data access logged: {access_record}")
    
    def create_backup(self, file_path: str, backup_dir: str = "backups") -> str:
        """Create a backup of a file."""
        try:
            ensure_directory(backup_dir)
            backup_path = Path(backup_dir) / f"{Path(file_path).stem}_{int(time.time())}{Path(file_path).suffix}"
            shutil.copy2(file_path, backup_path)
            
            self.backup_registry[str(backup_path)] = {
                "original": file_path,
                "created_at": format_datetime(get_current_datetime()),
                "size": os.path.getsize(backup_path)
            }
            
            self.logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            raise
    
    def restore_from_backup(self, backup_path: str, target_path: str) -> None:
        """Restore a file from backup."""
        try:
            if backup_path not in self.backup_registry:
                raise ValueError(f"Backup not found in registry: {backup_path}")
            
            shutil.copy2(backup_path, target_path)
            self.logger.info(f"Restored from backup: {backup_path} -> {target_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_path}: {e}")
            raise


class DataFormatManager:
    """Manages data format operations with validation."""
    
    def __init__(self):
        self.logger = logging.getLogger("data_format_manager")
        self.protection_manager = DataProtectionManager()
    
    @data_reader(validate_schema=True)
    @data_integrity_check(checksum_validation=True, schema_validation=True)
    def read_data(
        self,
        file_path: str,
        format: DataFormat = DataFormat.PARQUET,
        **kwargs
    ) -> pd.DataFrame:
        """Read data from file with format validation."""
        self.protection_manager.log_access("read", file_path)
        
        if not safe_file_exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            if format == DataFormat.PARQUET:
                data = pd.read_parquet(file_path, **kwargs)
            elif format == DataFormat.CSV:
                data = pd.read_csv(file_path, **kwargs)
            elif format == DataFormat.JSON:
                data = pd.read_json(file_path, **kwargs)
            elif format == DataFormat.FEATHER:
                data = pd.read_feather(file_path, **kwargs)
            elif format == DataFormat.HDF5:
                data = pd.read_hdf(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Successfully read data from {file_path}: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to read data from {file_path}: {e}")
            raise
    
    @data_writer(validate_schema=True)
    @data_integrity_check(checksum_validation=True, schema_validation=True)
    def write_data(
        self,
        data: pd.DataFrame,
        file_path: str,
        format: DataFormat = DataFormat.PARQUET,
        create_backup: bool = True,
        **kwargs
    ) -> None:
        """Write data to file with format validation."""
        self.protection_manager.log_access("write", file_path)
        
        # Create backup if file exists and backup is requested
        if create_backup and safe_file_exists(file_path):
            self.protection_manager.create_backup(file_path)
        
        try:
            # Ensure directory exists
            ensure_directory(Path(file_path).parent)
            
            if format == DataFormat.PARQUET:
                data.to_parquet(file_path, **kwargs)
            elif format == DataFormat.CSV:
                data.to_csv(file_path, **kwargs)
            elif format == DataFormat.JSON:
                data.to_json(file_path, **kwargs)
            elif format == DataFormat.FEATHER:
                data.to_feather(file_path, **kwargs)
            elif format == DataFormat.HDF5:
                data.to_hdf(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Successfully wrote data to {file_path}: {data.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to write data to {file_path}: {e}")
            raise
    
    def get_data_metadata(self, file_path: str) -> DataMetadata:
        """Get metadata for a data file."""
        if not safe_file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            stat = os.stat(file_path)
            file_size = stat.st_size
            created_at = format_datetime(get_current_datetime())
            modified_at = format_datetime(get_current_datetime())
            
            # Determine format from extension
            ext = Path(file_path).suffix.lower()
            format_mapping = {
                '.parquet': DataFormat.PARQUET,
                '.csv': DataFormat.CSV,
                '.json': DataFormat.JSON,
                '.pkl': DataFormat.PICKLE,
                '.h5': DataFormat.HDF5,
                '.feather': DataFormat.FEATHER
            }
            data_format = format_mapping.get(ext, DataFormat.PARQUET)
            
            # Read a sample to get schema info
            try:
                sample_data = self.read_data(file_path, data_format, nrows=1)
                row_count = len(sample_data)
                column_count = len(sample_data.columns)
                schema = {
                    "columns": list(sample_data.columns),
                    "dtypes": sample_data.dtypes.to_dict()
                }
            except Exception:
                row_count = 0
                column_count = 0
                schema = {}
            
            # Calculate checksum
            checksum = self._calculate_checksum(file_path)
            
            return DataMetadata(
                source=file_path,
                format=data_format,
                size_bytes=file_size,
                row_count=row_count,
                column_count=column_count,
                created_at=created_at,
                modified_at=modified_at,
                checksum=checksum,
                schema=schema
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {file_path}: {e}")
            raise
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class DataAnalysisManager:
    """Manages data analysis operations with validation."""
    
    def __init__(self):
        self.logger = logging.getLogger("data_analysis_manager")
        self.format_manager = DataFormatManager()
    
    @data_analyzer(validate_schema=True)
    @performance_monitor(log_performance=True, memory_monitoring=True)
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        try:
            analysis = {
                "basic_info": {
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict()
                },
                "missing_values": {
                    "total_missing": data.isnull().sum().sum(),
                    "missing_by_column": data.isnull().sum().to_dict(),
                    "missing_percentage": (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                },
                "duplicates": {
                    "duplicate_rows": data.duplicated().sum(),
                    "duplicate_percentage": (data.duplicated().sum() / len(data)) * 100
                },
                "numeric_analysis": {},
                "categorical_analysis": {}
            }
            
            # Numeric column analysis
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if len(data[col].dropna()) > 0:
                    analysis["numeric_analysis"][col] = {
                        "mean": data[col].mean(),
                        "std": data[col].std(),
                        "min": data[col].min(),
                        "max": data[col].max(),
                        "median": data[col].median(),
                        "skewness": data[col].skew(),
                        "kurtosis": data[col].kurtosis()
                    }
            
            # Categorical column analysis
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                if len(data[col].dropna()) > 0:
                    value_counts = data[col].value_counts()
                    analysis["categorical_analysis"][col] = {
                        "unique_values": data[col].nunique(),
                        "most_common": value_counts.head(5).to_dict(),
                        "value_distribution": (value_counts / len(data[col].dropna())).to_dict()
                    }
            
            self.logger.info(f"Data quality analysis completed for {data.shape[0]} rows, {data.shape[1]} columns")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Data quality analysis failed: {e}")
            raise
    
    @data_analyzer(validate_schema=True)
    @performance_monitor(log_performance=True)
    def detect_outliers(self, data: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            outlier_results = {}
            
            for col in numeric_columns:
                if len(data[col].dropna()) > 0:
                    if method == "iqr":
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
                        outlier_count = outliers.sum()
                        outlier_percentage = (outlier_count / len(data[col].dropna())) * 100
                        
                        outlier_results[col] = {
                            "outlier_count": outlier_count,
                            "outlier_percentage": outlier_percentage,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "outlier_indices": data[outliers].index.tolist()
                        }
            
            self.logger.info(f"Outlier detection completed for {len(numeric_columns)} numeric columns")
            return outlier_results
            
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {e}")
            raise
    
    @data_analyzer(validate_schema=True)
    @performance_monitor(log_performance=True)
    def correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis on numeric columns."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {"message": "No numeric columns found for correlation analysis"}
            
            correlation_matrix = numeric_data.corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            "column1": correlation_matrix.columns[i],
                            "column2": correlation_matrix.columns[j],
                            "correlation": corr_value
                        })
            
            analysis = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlations": high_correlations,
                "summary": {
                    "total_pairs": len(correlation_matrix.columns) * (len(correlation_matrix.columns) - 1) // 2,
                    "high_correlation_pairs": len(high_correlations)
                }
            }
            
            self.logger.info(f"Correlation analysis completed for {len(numeric_data.columns)} numeric columns")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            raise


class DataManipulationManager:
    """Manages data manipulation operations with protection."""
    
    def __init__(self):
        self.logger = logging.getLogger("data_manipulation_manager")
        self.format_manager = DataFormatManager()
        self.protection_manager = DataProtectionManager()
    
    @data_transformer(validate_schema=True)
    @data_integrity_check(checksum_validation=True, schema_validation=True)
    def clean_data(
        self,
        data: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: str = "drop",  # drop, fill, interpolate
        fill_value: Any = None,
        remove_outliers: bool = False,
        outlier_method: str = "iqr"
    ) -> pd.DataFrame:
        """Clean data with various strategies."""
        try:
            original_shape = data.shape
            cleaned_data = data.copy()
            
            # Remove duplicates
            if remove_duplicates:
                before_duplicates = len(cleaned_data)
                cleaned_data = cleaned_data.drop_duplicates()
                removed_duplicates = before_duplicates - len(cleaned_data)
                if removed_duplicates > 0:
                    self.logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            if handle_missing == "drop":
                before_missing = len(cleaned_data)
                cleaned_data = cleaned_data.dropna()
                removed_missing = before_missing - len(cleaned_data)
                if removed_missing > 0:
                    self.logger.info(f"Removed {removed_missing} rows with missing values")
            elif handle_missing == "fill":
                if fill_value is not None:
                    cleaned_data = cleaned_data.fillna(fill_value)
                    self.logger.info(f"Filled missing values with {fill_value}")
            elif handle_missing == "interpolate":
                numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if cleaned_data[col].isnull().any():
                        cleaned_data[col] = cleaned_data[col].interpolate()
                self.logger.info("Interpolated missing values in numeric columns")
            
            # Remove outliers
            if remove_outliers:
                numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if outlier_method == "iqr":
                        Q1 = cleaned_data[col].quantile(0.25)
                        Q3 = cleaned_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        before_outliers = len(cleaned_data)
                        cleaned_data = cleaned_data[
                            (cleaned_data[col] >= lower_bound) & 
                            (cleaned_data[col] <= upper_bound)
                        ]
                        removed_outliers = before_outliers - len(cleaned_data)
                        if removed_outliers > 0:
                            self.logger.info(f"Removed {removed_outliers} outliers from column {col}")
            
            final_shape = cleaned_data.shape
            self.logger.info(f"Data cleaning completed: {original_shape} -> {final_shape}")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise
    
    @data_transformer(validate_schema=True)
    @data_integrity_check(schema_validation=True)
    def transform_data(
        self,
        data: pd.DataFrame,
        transformations: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply transformations to data columns."""
        try:
            transformed_data = data.copy()
            
            for column, transform_config in transformations.items():
                if column not in transformed_data.columns:
                    self.logger.warning(f"Column {column} not found in data")
                    continue
                
                transform_type = transform_config.get("type")
                transform_params = transform_config.get("params", {})
                
                if transform_type == "log":
                    # Log transformation
                    if (transformed_data[column] > 0).all():
                        transformed_data[column] = np.log(transformed_data[column])
                        self.logger.info(f"Applied log transformation to {column}")
                    else:
                        self.logger.warning(f"Cannot apply log transformation to {column}: contains non-positive values")
                
                elif transform_type == "sqrt":
                    # Square root transformation
                    if (transformed_data[column] >= 0).all():
                        transformed_data[column] = np.sqrt(transformed_data[column])
                        self.logger.info(f"Applied square root transformation to {column}")
                    else:
                        self.logger.warning(f"Cannot apply sqrt transformation to {column}: contains negative values")
                
                elif transform_type == "standardize":
                    # Standardization
                    mean_val = transformed_data[column].mean()
                    std_val = transformed_data[column].std()
                    if std_val > 0:
                        transformed_data[column] = (transformed_data[column] - mean_val) / std_val
                        self.logger.info(f"Applied standardization to {column}")
                    else:
                        self.logger.warning(f"Cannot standardize {column}: standard deviation is zero")
                
                elif transform_type == "normalize":
                    # Min-max normalization
                    min_val = transformed_data[column].min()
                    max_val = transformed_data[column].max()
                    if max_val > min_val:
                        transformed_data[column] = (transformed_data[column] - min_val) / (max_val - min_val)
                        self.logger.info(f"Applied normalization to {column}")
                    else:
                        self.logger.warning(f"Cannot normalize {column}: min and max values are equal")
                
                elif transform_type == "bin":
                    # Binning
                    bins = transform_params.get("bins", 10)
                    labels = transform_params.get("labels")
                    transformed_data[column] = pd.cut(transformed_data[column], bins=bins, labels=labels)
                    self.logger.info(f"Applied binning to {column} with {bins} bins")
                
                elif transform_type == "encode":
                    # One-hot encoding
                    if transform_params.get("method") == "onehot":
                        encoded = pd.get_dummies(transformed_data[column], prefix=column)
                        transformed_data = pd.concat([transformed_data.drop(columns=[column]), encoded], axis=1)
                        self.logger.info(f"Applied one-hot encoding to {column}")
                    elif transform_params.get("method") == "label":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        transformed_data[column] = le.fit_transform(transformed_data[column].astype(str))
                        self.logger.info(f"Applied label encoding to {column}")
            
            self.logger.info(f"Data transformation completed for {len(transformations)} columns")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {e}")
            raise


class PipelineUtilities:
    """Main utilities class that combines all pipeline utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline_utilities")
        self.format_manager = DataFormatManager()
        self.analysis_manager = DataAnalysisManager()
        self.manipulation_manager = DataManipulationManager()
        self.protection_manager = DataProtectionManager()
    
    @contextmanager
    def safe_data_operation(self, operation_name: str, file_path: str = None):
        """Context manager for safe data operations with automatic backup and recovery."""
        backup_path = None
        try:
            # Create backup if file exists
            if file_path and safe_file_exists(file_path):
                backup_path = self.protection_manager.create_backup(file_path)
                self.logger.info(f"Created backup for {operation_name}: {backup_path}")
            
            yield
            
            self.logger.info(f"Safe data operation '{operation_name}' completed successfully")
            
        except Exception as e:
            self.logger.error(f"Safe data operation '{operation_name}' failed: {e}")
            
            # Restore from backup if available
            if backup_path and file_path:
                try:
                    self.protection_manager.restore_from_backup(backup_path, file_path)
                    self.logger.info(f"Restored from backup after failure: {backup_path}")
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore from backup: {restore_error}")
            
            raise
    
    @asynccontextmanager
    async def async_safe_data_operation(self, operation_name: str, file_path: str = None):
        """Async context manager for safe data operations."""
        backup_path = None
        try:
            # Create backup if file exists
            if file_path and safe_file_exists(file_path):
                backup_path = self.protection_manager.create_backup(file_path)
                self.logger.info(f"Created backup for {operation_name}: {backup_path}")
            
            yield
            
            self.logger.info(f"Async safe data operation '{operation_name}' completed successfully")
            
        except Exception as e:
            self.logger.error(f"Async safe data operation '{operation_name}' failed: {e}")
            
            # Restore from backup if available
            if backup_path and file_path:
                try:
                    self.protection_manager.restore_from_backup(backup_path, file_path)
                    self.logger.info(f"Restored from backup after failure: {backup_path}")
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore from backup: {restore_error}")
            
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of pipeline utilities."""
        return {
            "format_manager": "active",
            "analysis_manager": "active",
            "manipulation_manager": "active",
            "protection_manager": "active",
            "access_log_entries": len(self.protection_manager.access_log),
            "backup_registry_entries": len(self.protection_manager.backup_registry),
            "timestamp": format_datetime(get_current_datetime())
        }


# Global utilities instance
pipeline_utilities = PipelineUtilities()