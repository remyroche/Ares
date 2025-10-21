"""
Metadata Consistency Utilities

This module provides utilities to ensure metadata consistency across
KlinesParquetManager, serialization_utils, and artifact_manager components.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

from .logger import system_logger
from .tprint import tprint, tprint_success, tprint_info, tprint_warning, tprint_error


@dataclass
class StandardMetadata:
    """
    Standard metadata format that works across all components.
    
    This is the canonical metadata format that all components should use
    to ensure consistency and interoperability.
    """
    # Core identification
    artifact_id: str
    artifact_type: str  # klines, data, model, metadata, etc.
    step_name: Optional[str] = None
    
    # Data characteristics
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    interval: Optional[str] = None
    direction: Optional[str] = None
    model: Optional[str] = None
    
    # Storage information
    storage_location: str = "memory"
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    compressed_size_bytes: Optional[int] = None
    compression_ratio: float = 1.0
    compression_type: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # Data quality
    data_quality_score: float = 1.0
    validation_status: str = "unknown"
    validation_errors: List[str] = field(default_factory=list)
    
    # Relationships
    parent_artifacts: List[str] = field(default_factory=list)
    child_artifacts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Additional metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0"
    schema_version: str = "1.0"
    
    # Component-specific metadata
    klines_metadata: Optional[Dict[str, Any]] = None
    artifact_metadata: Optional[Dict[str, Any]] = None
    serialization_metadata: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    creation_time_seconds: float = 0.0
    load_time_seconds: float = 0.0
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization."""
        data = asdict(self)
        
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardMetadata':
        """Create from dictionary with proper datetime deserialization."""
        # Convert ISO format strings back to datetime objects
        datetime_fields = ['created_at', 'modified_at', 'last_accessed']
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except ValueError:
                    data[field] = datetime.utcnow()
        
        return cls(**data)
    
    def update_access(self) -> None:
        """Update last accessed timestamp and access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def add_validation_error(self, error: str) -> None:
        """Add a validation error."""
        if error not in self.validation_errors:
            self.validation_errors.append(error)
            self.validation_status = "failed"
    
    def clear_validation_errors(self) -> None:
        """Clear all validation errors."""
        self.validation_errors.clear()
        self.validation_status = "passed"
    
    def add_parent_artifact(self, parent_id: str) -> None:
        """Add a parent artifact."""
        if parent_id not in self.parent_artifacts:
            self.parent_artifacts.append(parent_id)
    
    def add_child_artifact(self, child_id: str) -> None:
        """Add a child artifact."""
        if child_id not in self.child_artifacts:
            self.child_artifacts.append(child_id)
    
    def add_dependency(self, dependency_id: str) -> None:
        """Add a dependency."""
        if dependency_id not in self.dependencies:
            self.dependencies.append(dependency_id)


class MetadataConverter:
    """
    Converter for translating between different metadata formats.
    
    This class provides methods to convert metadata between:
    - KlinesMetadata (from KlinesParquetManager)
    - ArtifactMetadata (from artifact_manager)
    - StandardMetadata (unified format)
    """
    
    def __init__(self):
        self.logger = system_logger.getChild("MetadataConverter")
    
    def klines_to_standard(self, klines_metadata: Any) -> StandardMetadata:
        """Convert KlinesMetadata to StandardMetadata."""
        try:
            # Extract data from klines metadata
            return StandardMetadata(
                artifact_id=f"klines_{klines_metadata.symbol}_{klines_metadata.exchange}_{klines_metadata.interval}_{klines_metadata.batch_id}",
                artifact_type="klines",
                symbol=klines_metadata.symbol,
                exchange=klines_metadata.exchange,
                interval=klines_metadata.interval,
                file_size_bytes=klines_metadata.file_size_bytes,
                compression_ratio=klines_metadata.compression_ratio,
                created_at=klines_metadata.created_at,
                data_quality_score=klines_metadata.data_quality_score,
                klines_metadata={
                    'batch_id': klines_metadata.batch_id,
                    'start_time': klines_metadata.start_time.isoformat() if hasattr(klines_metadata, 'start_time') else None,
                    'end_time': klines_metadata.end_time.isoformat() if hasattr(klines_metadata, 'end_time') else None,
                    'record_count': klines_metadata.record_count,
                    'gaps_detected': getattr(klines_metadata, 'gaps_detected', 0),
                    'gaps_filled': getattr(klines_metadata, 'gaps_filled', 0),
                    'resampled_intervals': getattr(klines_metadata, 'resampled_intervals', []),
                    'additional_metadata': getattr(klines_metadata, 'additional_metadata', {})
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to convert klines metadata: {e}")
            raise
    
    def artifact_to_standard(self, artifact_metadata: Any) -> StandardMetadata:
        """Convert ArtifactMetadata to StandardMetadata."""
        try:
            return StandardMetadata(
                artifact_id=artifact_metadata.artifact_key,
                artifact_type=artifact_metadata.artifact_type,
                step_name=artifact_metadata.step_name,
                file_size_bytes=artifact_metadata.size_bytes,
                compressed_size_bytes=artifact_metadata.compressed_size_bytes,
                checksum=artifact_metadata.checksum,
                created_at=artifact_metadata.created_at,
                modified_at=artifact_metadata.modified_at,
                compression_used=artifact_metadata.compression_used.value if hasattr(artifact_metadata.compression_used, 'value') else str(artifact_metadata.compression_used),
                storage_location=artifact_metadata.storage_location,
                parent_artifacts=artifact_metadata.parent_artifacts,
                tags=artifact_metadata.tags,
                description=artifact_metadata.description,
                version=artifact_metadata.version,
                artifact_metadata={
                    'compression_used': str(artifact_metadata.compression_used),
                    'storage_location': artifact_metadata.storage_location,
                    'parent_artifacts': artifact_metadata.parent_artifacts,
                    'tags': artifact_metadata.tags,
                    'description': artifact_metadata.description,
                    'version': artifact_metadata.version
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to convert artifact metadata: {e}")
            raise
    
    def standard_to_klines(self, standard_metadata: StandardMetadata) -> Dict[str, Any]:
        """Convert StandardMetadata to klines metadata format."""
        try:
            return {
                'symbol': standard_metadata.symbol,
                'exchange': standard_metadata.exchange,
                'interval': standard_metadata.interval,
                'batch_id': standard_metadata.klines_metadata.get('batch_id', 'unknown') if standard_metadata.klines_metadata else 'unknown',
                'start_time': standard_metadata.klines_metadata.get('start_time') if standard_metadata.klines_metadata else None,
                'end_time': standard_metadata.klines_metadata.get('end_time') if standard_metadata.klines_metadata else None,
                'record_count': standard_metadata.klines_metadata.get('record_count', 0) if standard_metadata.klines_metadata else 0,
                'file_size_bytes': standard_metadata.file_size_bytes,
                'compression_ratio': standard_metadata.compression_ratio,
                'created_at': standard_metadata.created_at.isoformat(),
                'data_quality_score': standard_metadata.data_quality_score,
                'gaps_detected': standard_metadata.klines_metadata.get('gaps_detected', 0) if standard_metadata.klines_metadata else 0,
                'gaps_filled': standard_metadata.klines_metadata.get('gaps_filled', 0) if standard_metadata.klines_metadata else 0,
                'resampled_intervals': standard_metadata.klines_metadata.get('resampled_intervals', []) if standard_metadata.klines_metadata else [],
                'additional_metadata': standard_metadata.klines_metadata.get('additional_metadata', {}) if standard_metadata.klines_metadata else {}
            }
        except Exception as e:
            self.logger.error(f"Failed to convert to klines metadata: {e}")
            raise
    
    def standard_to_artifact(self, standard_metadata: StandardMetadata) -> Dict[str, Any]:
        """Convert StandardMetadata to artifact metadata format."""
        try:
            return {
                'artifact_key': standard_metadata.artifact_id,
                'step_name': standard_metadata.step_name,
                'artifact_type': standard_metadata.artifact_type,
                'size_bytes': standard_metadata.file_size_bytes,
                'compressed_size_bytes': standard_metadata.compressed_size_bytes,
                'checksum': standard_metadata.checksum if hasattr(standard_metadata, 'checksum') else '',
                'created_at': standard_metadata.created_at,
                'modified_at': standard_metadata.modified_at,
                'compression_used': standard_metadata.compression_type or 'none',
                'storage_location': standard_metadata.storage_location,
                'parent_artifacts': standard_metadata.parent_artifacts,
                'tags': standard_metadata.tags,
                'description': standard_metadata.description,
                'version': standard_metadata.version
            }
        except Exception as e:
            self.logger.error(f"Failed to convert to artifact metadata: {e}")
            raise


class MetadataValidator:
    """
    Validator for ensuring metadata consistency and correctness.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild("MetadataValidator")
    
    def validate_standard_metadata(self, metadata: StandardMetadata) -> List[str]:
        """Validate standard metadata and return list of errors."""
        errors = []
        
        # Required fields
        if not metadata.artifact_id:
            errors.append("artifact_id is required")
        
        if not metadata.artifact_type:
            errors.append("artifact_type is required")
        
        # Data type validation
        if metadata.file_size_bytes < 0:
            errors.append("file_size_bytes must be non-negative")
        
        if not 0 <= metadata.data_quality_score <= 1:
            errors.append("data_quality_score must be between 0 and 1")
        
        if metadata.compression_ratio < 0:
            errors.append("compression_ratio must be non-negative")
        
        # Timestamp validation
        if metadata.modified_at < metadata.created_at:
            errors.append("modified_at cannot be before created_at")
        
        if metadata.last_accessed < metadata.created_at:
            errors.append("last_accessed cannot be before created_at")
        
        # Klines-specific validation
        if metadata.artifact_type == "klines":
            if not metadata.symbol:
                errors.append("symbol is required for klines data")
            if not metadata.exchange:
                errors.append("exchange is required for klines data")
            if not metadata.interval:
                errors.append("interval is required for klines data")
        
        # Step-specific validation
        if metadata.step_name and not metadata.step_name.strip():
            errors.append("step_name cannot be empty")
        
        return errors
    
    def validate_consistency(self, metadata_list: List[StandardMetadata]) -> List[str]:
        """Validate consistency across multiple metadata entries."""
        errors = []
        
        # Check for duplicate artifact IDs
        artifact_ids = [m.artifact_id for m in metadata_list]
        if len(artifact_ids) != len(set(artifact_ids)):
            errors.append("Duplicate artifact IDs found")
        
        # Check parent-child relationships
        for metadata in metadata_list:
            for parent_id in metadata.parent_artifacts:
                parent_exists = any(m.artifact_id == parent_id for m in metadata_list)
                if not parent_exists:
                    errors.append(f"Parent artifact {parent_id} not found for {metadata.artifact_id}")
            
            for child_id in metadata.child_artifacts:
                child_exists = any(m.artifact_id == child_id for m in metadata_list)
                if not child_exists:
                    errors.append(f"Child artifact {child_id} not found for {metadata.artifact_id}")
        
        return errors


class MetadataRegistry:
    """
    Central registry for managing metadata across all components.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild("MetadataRegistry")
        self._metadata: Dict[str, StandardMetadata] = {}
        self._converter = MetadataConverter()
        self._validator = MetadataValidator()
    
    def register_metadata(self, metadata: StandardMetadata) -> bool:
        """Register metadata in the central registry."""
        try:
            # Validate metadata
            errors = self._validator.validate_standard_metadata(metadata)
            if errors:
                self.logger.warning(f"Metadata validation errors for {metadata.artifact_id}: {errors}")
                for error in errors:
                    metadata.add_validation_error(error)
            
            # Register metadata
            self._metadata[metadata.artifact_id] = metadata
            
            tprint_success(f"✅ Registered metadata: {metadata.artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register metadata: {e}")
            return False
    
    def get_metadata(self, artifact_id: str) -> Optional[StandardMetadata]:
        """Get metadata by artifact ID."""
        metadata = self._metadata.get(artifact_id)
        if metadata:
            metadata.update_access()
        return metadata
    
    def list_metadata(self, artifact_type: Optional[str] = None) -> List[StandardMetadata]:
        """List all metadata, optionally filtered by type."""
        metadata_list = list(self._metadata.values())
        
        if artifact_type:
            metadata_list = [m for m in metadata_list if m.artifact_type == artifact_type]
        
        return metadata_list
    
    def update_metadata(self, artifact_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata with new values."""
        try:
            metadata = self._metadata.get(artifact_id)
            if not metadata:
                self.logger.warning(f"Metadata not found for {artifact_id}")
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
                else:
                    self.logger.warning(f"Unknown field {key} for metadata update")
            
            # Update modified timestamp
            metadata.modified_at = datetime.utcnow()
            
            tprint_success(f"✅ Updated metadata: {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")
            return False
    
    def remove_metadata(self, artifact_id: str) -> bool:
        """Remove metadata from registry."""
        try:
            if artifact_id in self._metadata:
                del self._metadata[artifact_id]
                tprint_success(f"✅ Removed metadata: {artifact_id}")
                return True
            else:
                self.logger.warning(f"Metadata not found for removal: {artifact_id}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to remove metadata: {e}")
            return False
    
    def validate_all_metadata(self) -> Dict[str, List[str]]:
        """Validate all metadata in the registry."""
        validation_results = {}
        
        for artifact_id, metadata in self._metadata.items():
            errors = self._validator.validate_standard_metadata(metadata)
            if errors:
                validation_results[artifact_id] = errors
        
        return validation_results
    
    def get_consistency_report(self) -> Dict[str, Any]:
        """Get a consistency report for all metadata."""
        metadata_list = list(self._metadata.values())
        
        # Individual validation
        individual_errors = self._validator.validate_standard_metadata(metadata_list[0]) if metadata_list else []
        
        # Consistency validation
        consistency_errors = self._validator.validate_consistency(metadata_list)
        
        # Statistics
        total_artifacts = len(metadata_list)
        by_type = {}
        for metadata in metadata_list:
            artifact_type = metadata.artifact_type
            by_type[artifact_type] = by_type.get(artifact_type, 0) + 1
        
        return {
            'total_artifacts': total_artifacts,
            'by_type': by_type,
            'individual_errors': individual_errors,
            'consistency_errors': consistency_errors,
            'validation_passed': len(individual_errors) == 0 and len(consistency_errors) == 0
        }
    
    def export_metadata(self, filepath: str, format: str = "json") -> bool:
        """Export all metadata to file."""
        try:
            metadata_data = [metadata.to_dict() for metadata in self._metadata.values()]
            
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(metadata_data, f, indent=2, default=str)
            elif format.lower() == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(metadata_data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            tprint_success(f"✅ Exported metadata to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metadata: {e}")
            return False
    
    def import_metadata(self, filepath: str, format: str = "json") -> bool:
        """Import metadata from file."""
        try:
            if format.lower() == "json":
                with open(filepath, 'r') as f:
                    metadata_data = json.load(f)
            elif format.lower() == "pickle":
                with open(filepath, 'rb') as f:
                    metadata_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Convert to StandardMetadata objects
            for data in metadata_data:
                metadata = StandardMetadata.from_dict(data)
                self.register_metadata(metadata)
            
            tprint_success(f"✅ Imported metadata from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import metadata: {e}")
            return False


# Global metadata registry instance
_global_registry = None

def get_metadata_registry() -> MetadataRegistry:
    """Get the global metadata registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MetadataRegistry()
    return _global_registry

def get_metadata_converter() -> MetadataConverter:
    """Get a metadata converter instance."""
    return MetadataConverter()

def get_metadata_validator() -> MetadataValidator:
    """Get a metadata validator instance."""
    return MetadataValidator()