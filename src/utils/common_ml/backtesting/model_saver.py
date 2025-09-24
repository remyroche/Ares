"""
Model Saver with Comprehensive Persistence

This module provides comprehensive model saving and persistence utilities with
M1 optimizations for memory efficiency and performance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
import pickle
import joblib
import json
import yaml
from pathlib import Path
import hashlib
import shutil
import tempfile

# M1 Optimization imports
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models that can be saved."""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM = "custom"
    ENSEMBLE = "ensemble"


class SaveFormat(Enum):
    """Supported save formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    JSON = "json"
    YAML = "yaml"
    PARQUET = "parquet"
    NPZ = "npz"
    HDF5 = "hdf5"
    ONNX = "onnx"


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    created_at: datetime
    model_type: ModelType
    model_size_mb: float
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    # Basic information
    model_name: str
    model_type: ModelType
    version: str
    created_at: datetime
    created_by: str = "system"
    
    # Model details
    model_class: str = ""
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    
    # Training information
    training_data_size: int = 0
    training_duration: float = 0.0
    training_algorithm: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Data information
    data_preprocessing: Dict[str, Any] = field(default_factory=dict)
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    data_schema: Dict[str, Any] = field(default_factory=dict)
    
    # System information
    python_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # File information
    file_path: str = ""
    file_size_mb: float = 0.0
    checksum: str = ""
    save_format: SaveFormat = SaveFormat.PICKLE
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    notes: str = ""


@dataclass
class ModelSaveConfig:
    """Configuration for model saving."""
    # Basic configuration
    model_name: str
    output_dir: str
    
    # Save options
    save_formats: List[SaveFormat] = field(default_factory=lambda: [SaveFormat.PICKLE, SaveFormat.JOBLIB])
    save_metadata: bool = True
    save_preprocessing: bool = True
    save_feature_engineering: bool = True
    save_performance_metrics: bool = True
    
    # Versioning
    enable_versioning: bool = True
    version_format: str = "v{version}"  # v1.0.0, v1.0.1, etc.
    max_versions: int = 10
    
    # Compression
    enable_compression: bool = True
    compression_level: int = 6
    
    # M1 optimization settings
    enable_memory_optimization: bool = True
    memory_limit_gb: float = 8.0
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Validation
    validate_model: bool = True
    validate_data_integrity: bool = True
    create_checksum: bool = True
    
    # Backup
    create_backup: bool = True
    backup_dir: Optional[str] = None
    
    # Cleanup
    cleanup_temp_files: bool = True
    remove_old_versions: bool = True


class ModelSaver:
    """Comprehensive model saver with M1 optimizations."""
    
    def __init__(self, config: ModelSaveConfig):
        """Initialize model saver."""
        self.config = config
        self.logger = logger.getChild('ModelSaver')
        
        # Initialize M1 optimizers
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        self.logger.info(f"🚀 ModelSaver initialized for {config.model_name}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"💾 Save formats: {[f.value for f in config.save_formats]}")
    
    @traced(span_name='save_model')
    async def save_model(
        self, 
        model: Any,
        metadata: Optional[ModelMetadata] = None,
        preprocessing_pipeline: Optional[Any] = None,
        feature_engineering_pipeline: Optional[Any] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ModelMetadata:
        """Save model with comprehensive metadata and M1 optimizations."""
        
        self.logger.info("🚀 Starting model saving...")
        start_time = time.time()
        
        # Validate inputs
        self._validate_model(model)
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                result_metadata = await self._save_model_comprehensive(
                    model, metadata, preprocessing_pipeline, 
                    feature_engineering_pipeline, performance_metrics, **kwargs
                )
        else:
            result_metadata = await self._save_model_comprehensive(
                model, metadata, preprocessing_pipeline, 
                feature_engineering_pipeline, performance_metrics, **kwargs
            )
        
        execution_time = time.time() - start_time
        self.logger.info(f"✅ Model saving completed in {execution_time:.2f}s")
        self.logger.info(f"💾 Model saved to: {result_metadata.file_path}")
        self.logger.info(f"📊 File size: {result_metadata.file_size_mb:.2f} MB")
        
        return result_metadata
    
    def _validate_model(self, model: Any) -> None:
        """Validate model before saving."""
        if model is None:
            raise ValidationError("Model cannot be None")
        
        # Check if model has required methods
        if hasattr(model, 'predict'):
            if not callable(getattr(model, 'predict')):
                raise ValidationError("Model predict method is not callable")
        else:
            self.logger.warning("Model does not have predict method")
        
        # Check model size
        try:
            model_size = self._estimate_model_size(model)
            if model_size > 1000:  # 1GB
                self.logger.warning(f"Large model detected: {model_size:.2f} MB")
        except Exception as e:
            self.logger.warning(f"Could not estimate model size: {e}")
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            # Try to serialize to get size estimate
            with tempfile.NamedTemporaryFile() as tmp:
                pickle.dump(model, tmp)
                tmp.flush()
                size_bytes = tmp.tell()
                return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            # Fallback estimation
            return 10.0  # Default estimate
    
    async def _save_model_comprehensive(
        self, 
        model: Any,
        metadata: Optional[ModelMetadata],
        preprocessing_pipeline: Optional[Any],
        feature_engineering_pipeline: Optional[Any],
        performance_metrics: Optional[Dict[str, float]],
        **kwargs
    ) -> ModelMetadata:
        """Save model with comprehensive metadata."""
        
        # Create or enhance metadata
        if metadata is None:
            metadata = self._create_default_metadata(model)
        else:
            metadata = self._enhance_metadata(metadata, model, performance_metrics)
        
        # Determine model type
        model_type = self._detect_model_type(model)
        metadata.model_type = model_type
        
        # Create version
        if self.config.enable_versioning:
            version = self._get_next_version()
            metadata.version = version
        
        # Save model in multiple formats
        saved_files = []
        for save_format in self.config.save_formats:
            try:
                file_path = await self._save_model_format(
                    model, metadata, save_format, **kwargs
                )
                saved_files.append(file_path)
            except Exception as e:
                self.logger.error(f"Failed to save in {save_format.value} format: {e}")
        
        if not saved_files:
            raise RuntimeError("Failed to save model in any format")
        
        # Use the first successful save as primary
        metadata.file_path = saved_files[0]
        metadata.save_format = self.config.save_formats[0]
        
        # Calculate file size and checksum
        metadata.file_size_mb = get_file_size(metadata.file_path) / (1024 * 1024)
        if self.config.create_checksum:
            metadata.checksum = self._calculate_checksum(metadata.file_path)
        
        # Save additional components
        if self.config.save_preprocessing and preprocessing_pipeline is not None:
            await self._save_preprocessing_pipeline(preprocessing_pipeline, metadata)
        
        if self.config.save_feature_engineering and feature_engineering_pipeline is not None:
            await self._save_feature_engineering_pipeline(feature_engineering_pipeline, metadata)
        
        # Save metadata
        if self.config.save_metadata:
            await self._save_metadata(metadata)
        
        # Create backup if requested
        if self.config.create_backup:
            await self._create_backup(metadata)
        
        # Cleanup old versions if requested
        if self.config.remove_old_versions and self.config.enable_versioning:
            await self._cleanup_old_versions()
        
        return metadata
    
    def _create_default_metadata(self, model: Any) -> ModelMetadata:
        """Create default metadata for model."""
        return ModelMetadata(
            model_name=self.config.model_name,
            model_type=ModelType.CUSTOM,
            version="1.0.0",
            created_at=datetime.now(),
            model_class=model.__class__.__name__,
            python_version=f"{psutil.Process().environ.get('PYTHON_VERSION', 'unknown')}",
            system_info=self._get_system_info()
        )
    
    def _enhance_metadata(
        self, 
        metadata: ModelMetadata, 
        model: Any, 
        performance_metrics: Optional[Dict[str, float]]
    ) -> ModelMetadata:
        """Enhance existing metadata with additional information."""
        
        # Update performance metrics
        if performance_metrics:
            metadata.performance_metrics.update(performance_metrics)
        
        # Update model parameters
        if hasattr(model, 'get_params'):
            try:
                metadata.model_parameters = model.get_params()
            except Exception as e:
                self.logger.warning(f"Could not extract model parameters: {e}")
        
        # Update feature names if available
        if hasattr(model, 'feature_names_in_'):
            try:
                metadata.feature_names = list(model.feature_names_in_)
            except Exception as e:
                self.logger.warning(f"Could not extract feature names: {e}")
        
        return metadata
    
    def _detect_model_type(self, model: Any) -> ModelType:
        """Detect the type of model."""
        model_class_name = model.__class__.__name__.lower()
        
        if 'sklearn' in str(type(model)) or any(x in model_class_name for x in ['sklearn', 'linear', 'tree', 'forest']):
            return ModelType.SKLEARN
        elif 'xgboost' in model_class_name or 'xgb' in model_class_name:
            return ModelType.XGBOOST
        elif 'lightgbm' in model_class_name or 'lgb' in model_class_name:
            return ModelType.LIGHTGBM
        elif 'catboost' in model_class_name or 'cat' in model_class_name:
            return ModelType.CATBOOST
        elif 'torch' in str(type(model)) or 'pytorch' in model_class_name:
            return ModelType.PYTORCH
        elif 'tensorflow' in str(type(model)) or 'keras' in model_class_name:
            return ModelType.TENSORFLOW
        elif 'ensemble' in model_class_name:
            return ModelType.ENSEMBLE
        else:
            return ModelType.CUSTOM
    
    def _get_next_version(self) -> str:
        """Get the next version number."""
        version_file = f"{self.config.output_dir}/versions.json"
        
        if safe_file_exists(version_file):
            try:
                versions_data = safe_json_load(version_file)
                current_version = versions_data.get('current_version', '0.0.0')
            except Exception:
                current_version = '0.0.0'
        else:
            current_version = '0.0.0'
        
        # Increment version
        version_parts = current_version.split('.')
        if len(version_parts) == 3:
            major, minor, patch = map(int, version_parts)
            patch += 1
            if patch >= 100:
                patch = 0
                minor += 1
                if minor >= 100:
                    minor = 0
                    major += 1
            new_version = f"{major}.{minor}.{patch}"
        else:
            new_version = "1.0.0"
        
        # Save new version
        versions_data = {'current_version': new_version}
        safe_json_dump(version_file, versions_data)
        
        return new_version
    
    async def _save_model_format(
        self, 
        model: Any, 
        metadata: ModelMetadata, 
        save_format: SaveFormat,
        **kwargs
    ) -> str:
        """Save model in specific format."""
        
        # Create filename
        version_str = f"_{metadata.version}" if self.config.enable_versioning else ""
        filename = f"{self.config.model_name}{version_str}.{save_format.value}"
        file_path = f"{self.config.output_dir}/{filename}"
        
        # Save based on format
        if save_format == SaveFormat.PICKLE:
            await self._save_pickle(model, file_path)
        elif save_format == SaveFormat.JOBLIB:
            await self._save_joblib(model, file_path)
        elif save_format == SaveFormat.JSON:
            await self._save_json(model, file_path)
        elif save_format == SaveFormat.YAML:
            await self._save_yaml(model, file_path)
        elif save_format == SaveFormat.NPZ:
            await self._save_npz(model, file_path)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        return file_path
    
    async def _save_pickle(self, model: Any, file_path: str) -> None:
        """Save model using pickle."""
        with open(file_path, 'wb') as f:
            if self.config.enable_compression:
                import gzip
                with gzip.GzipFile(fileobj=f, compresslevel=self.config.compression_level) as gz:
                    pickle.dump(model, gz)
            else:
                pickle.dump(model, f)
    
    async def _save_joblib(self, model: Any, file_path: str) -> None:
        """Save model using joblib."""
        if self.config.enable_compression:
            joblib.dump(model, file_path, compress=('gzip', self.config.compression_level))
        else:
            joblib.dump(model, file_path)
    
    async def _save_json(self, model: Any, file_path: str) -> None:
        """Save model using JSON (for simple models)."""
        # This is a simplified JSON save - in practice, you'd need custom serialization
        model_data = {
            'model_class': model.__class__.__name__,
            'model_type': str(type(model)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to extract parameters
        if hasattr(model, 'get_params'):
            try:
                model_data['parameters'] = model.get_params()
            except Exception:
                pass
        
        safe_json_dump(file_path, model_data)
    
    async def _save_yaml(self, model: Any, file_path: str) -> None:
        """Save model using YAML (for simple models)."""
        # This is a simplified YAML save - in practice, you'd need custom serialization
        model_data = {
            'model_class': model.__class__.__name__,
            'model_type': str(type(model)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to extract parameters
        if hasattr(model, 'get_params'):
            try:
                model_data['parameters'] = model.get_params()
            except Exception:
                pass
        
        with open(file_path, 'w') as f:
            yaml.dump(model_data, f, default_flow_style=False)
    
    async def _save_npz(self, model: Any, file_path: str) -> None:
        """Save model using NPZ format (for numpy-based models)."""
        # This is a simplified NPZ save - in practice, you'd need custom serialization
        model_data = {}
        
        # Try to extract numpy arrays from model
        if hasattr(model, 'coef_'):
            model_data['coef'] = model.coef_
        if hasattr(model, 'intercept_'):
            model_data['intercept'] = model.intercept_
        if hasattr(model, 'feature_importances_'):
            model_data['feature_importances'] = model.feature_importances_
        
        if model_data:
            np.savez_compressed(file_path, **model_data)
        else:
            raise ValueError("Model does not contain numpy arrays suitable for NPZ format")
    
    async def _save_preprocessing_pipeline(self, pipeline: Any, metadata: ModelMetadata) -> None:
        """Save preprocessing pipeline."""
        pipeline_file = f"{self.config.output_dir}/{self.config.model_name}_{metadata.version}_preprocessing.pkl"
        
        with open(pipeline_file, 'wb') as f:
            if self.config.enable_compression:
                with gzip.GzipFile(fileobj=f, compresslevel=self.config.compression_level) as gz:
                    pickle.dump(pipeline, gz)
            else:
                pickle.dump(pipeline, f)
        
        self.logger.info(f"💾 Preprocessing pipeline saved to {pipeline_file}")
    
    async def _save_feature_engineering_pipeline(self, pipeline: Any, metadata: ModelMetadata) -> None:
        """Save feature engineering pipeline."""
        pipeline_file = f"{self.config.output_dir}/{self.config.model_name}_{metadata.version}_feature_generation.utils.pkl"
        
        with open(pipeline_file, 'wb') as f:
            if self.config.enable_compression:
                with gzip.GzipFile(fileobj=f, compresslevel=self.config.compression_level) as gz:
                    pickle.dump(pipeline, gz)
            else:
                pickle.dump(pipeline, f)
        
        self.logger.info(f"💾 Feature engineering pipeline saved to {pipeline_file}")
    
    async def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata."""
        metadata_file = f"{self.config.output_dir}/{self.config.model_name}_{metadata.version}_metadata.json"
        
        # Convert metadata to dict and handle datetime serialization
        metadata_dict = metadata.__dict__.copy()
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        metadata_dict['model_type'] = metadata.model_type.value
        metadata_dict['save_format'] = metadata.save_format.value
        
        safe_json_dump(metadata_file, metadata_dict)
        self.logger.info(f"💾 Metadata saved to {metadata_file}")
    
    async def _create_backup(self, metadata: ModelMetadata) -> None:
        """Create backup of saved model."""
        backup_dir = self.config.backup_dir or f"{self.config.output_dir}/backups"
        ensure_directory(backup_dir)
        
        backup_file = f"{backup_dir}/{self.config.model_name}_{metadata.version}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Copy main model file
        shutil.copy2(metadata.file_path, f"{backup_file}.{metadata.save_format.value}")
        
        # Copy metadata if exists
        metadata_file = f"{self.config.output_dir}/{self.config.model_name}_{metadata.version}_metadata.json"
        if safe_file_exists(metadata_file):
            shutil.copy2(metadata_file, f"{backup_file}_metadata.json")
        
        self.logger.info(f"💾 Backup created at {backup_file}")
    
    async def _cleanup_old_versions(self) -> None:
        """Clean up old model versions."""
        if self.config.max_versions <= 0:
            return
        
        # Get all version files
        version_files = []
        for file in Path(self.config.output_dir).glob(f"{self.config.model_name}_v*"):
            if file.is_file():
                version_files.append(file)
        
        # Sort by modification time (newest first)
        version_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old versions
        if len(version_files) > self.config.max_versions:
            for old_file in version_files[self.config.max_versions:]:
                try:
                    old_file.unlink()
                    self.logger.info(f"🗑️ Removed old version: {old_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to remove old version {old_file.name}: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'platform': psutil.Process().environ.get('PLATFORM', 'unknown'),
            'python_version': psutil.Process().environ.get('PYTHON_VERSION', 'unknown'),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> Tuple[Any, Optional[ModelMetadata]]:
        """Load model and metadata."""
        self.logger.info(f"📂 Loading model from {model_path}")
        
        # Load model
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        # Load metadata if provided
        metadata = None
        if metadata_path and safe_file_exists(metadata_path):
            metadata_dict = safe_json_load(metadata_path)
            metadata = ModelMetadata(**metadata_dict)
        
        self.logger.info(f"✅ Model loaded successfully")
        return model, metadata