"""
Enhanced Pre-Training Artifact Manager

This module provides comprehensive artifact management for the pre-training pipeline
with enhanced file naming, joint Parquet support, JSON metadata generation, and
data alignment verification.

Key Features:
- Enhanced file naming with symbol, exchange, datetime, and information context
- Joint Parquet file creation for unified OHLCV + labels + features
- Automatic JSON metadata generation
- Data alignment verification across steps
- Comprehensive logging and monitoring
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

@dataclass
class ArtifactConfig:
    """Configuration for artifact management."""
    # File naming options
    include_symbol_in_filename: bool = True
    include_exchange_in_filename: bool = True
    include_datetime_in_filename: bool = True
    include_information_in_filename: bool = True
    include_direction_in_filename: bool = True
    include_model_in_filename: bool = True
    include_timeframe_in_filename: bool = True

    # Data format options
    use_joint_parquet_format: bool = True
    generate_json_metadata: bool = True

    # Directory structure
    base_dir: str = "artifacts/pre_training/artifact_store"

    # Performance and monitoring
    enable_comprehensive_logging: bool = True
    enable_data_verification: bool = True

@dataclass
class ArtifactKeys:
    """Standardized artifact keys for consistency."""
    # Data artifacts
    RAW_DATAFRAME = 'raw_dataframe'
    CLEANED_DATAFRAME = 'cleaned_dataframe'
    FEATURE_DATAFRAME = 'feature_dataframe'
    LABEL_DATAFRAME = 'label_dataframe'
    JOINT_DATAFRAME = 'joint_dataframe'

    # Metadata artifacts
    FEATURE_NAMES = 'feature_names'
    FEATURE_CATEGORIES = 'feature_categories'
    VALIDATION_METRICS = 'validation_metrics'
    GENERATION_METRICS = 'generation_metrics'
    PROCESSING_METADATA = 'processing_metadata'

    # Model artifacts
    MODEL_OBJECT = 'model_object'
    MODEL_CONFIG = 'model_config'
    MODEL_METRICS = 'model_metrics'

class PreTrainingArtifactManager:
    """Enhanced artifact manager for pre-training pipeline."""

    def __init__(self, config: Optional[ArtifactConfig] = None):
        """Initialize artifact manager with configuration."""
        self.config = config or ArtifactConfig()
        self._context: Dict[str, Any] = {}
        self._artifact_registry: Dict[str, Any] = {}

    def set_context(self,
                   symbol: Optional[str] = None,
                   exchange: Optional[str] = None,
                   datetime_obj: Optional[datetime] = None,
                   information: str = "pre_training",
                   direction: str = "long",
                   model: str = "Analyst",
                   timeframe: Optional[str] = None) -> None:
        """Set context for enhanced file naming and organization."""
        self._context = {
            'symbol': symbol,
            'exchange': exchange,
            'datetime': datetime_obj or datetime.now(),
            'information': information,
            'direction': direction,
            'model': model,
            'timeframe': timeframe
        }

        if self.config.enable_comprehensive_logging:
            logger.info(f"🔧 Artifact manager context set: {self._context}")

    def _generate_filename(self,
                          step_name: str,
                          key: str,
                          extension: str = 'parquet') -> str:
        """Generate enhanced filename with context information."""
        parts = []

        # Add information prefix
        if self.config.include_information_in_filename and self._context.get('information'):
            parts.append(self._context['information'])

        # Add step name
        parts.append(step_name)

        # Add key
        parts.append(key)

        # Add symbol
        if self.config.include_symbol_in_filename and self._context.get('symbol'):
            parts.append(self._context['symbol'])

        # Add exchange
        if self.config.include_exchange_in_filename and self._context.get('exchange'):
            parts.append(self._context['exchange'])

        # Add direction and model if specified
        if self.config.include_direction_in_filename and self._context.get('direction'):
            parts.append(self._context['direction'])

        if self.config.include_model_in_filename and self._context.get('model'):
            parts.append(self._context['model'])

        # Add timeframe
        if self.config.include_timeframe_in_filename and self._context.get('timeframe'):
            parts.append(self._context['timeframe'])

        # Add datetime
        if self.config.include_datetime_in_filename and self._context.get('datetime'):
            dt_str = self._context['datetime'].strftime('%Y%m%d_%H%M%S')
            parts.append(dt_str)

        filename = '_'.join(parts) + f'.{extension}'
        return filename

    def _generate_directory_path(self, step_name: str) -> Path:
        """Generate directory path based on context."""
        base_path = Path(self.config.base_dir)

        # Create symbol-specific subdirectory
        if self._context.get('symbol'):
            base_path = base_path / self._context['symbol']

        # Create exchange-specific subdirectory
        if self._context.get('exchange'):
            base_path = base_path / self._context['exchange']

        # Create step-specific subdirectory
        base_path = base_path / step_name

        return base_path

    def _verify_data_alignment(self,
                             ohlcv_data: Optional[pd.DataFrame] = None,
                             labels_data: Optional[pd.DataFrame] = None,
                             features_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Verify data alignment across datasets."""
        verification_result = {
            'aligned': True,
            'issues': [],
            'metrics': {}
        }

        datasets = [
            ('ohlcv', ohlcv_data),
            ('labels', labels_data),
            ('features', features_data)
        ]

        indices = []
        for name, data in datasets:
            if data is not None and hasattr(data, 'index'):
                indices.append((name, data.index))

        if len(indices) > 1:
            # Check if all indices are identical
            base_index = indices[0][1]
            for name, index in indices[1:]:
                if not base_index.equals(index):
                    verification_result['aligned'] = False
                    verification_result['issues'].append(
                        f"Index mismatch between datasets: {indices[0][0]} vs {name}"
                    )

        # Calculate metrics
        for name, data in datasets:
            if data is not None:
                verification_result['metrics'][name] = {
                    'shape': data.shape,
                    'index_type': str(type(data.index).__name__),
                    'start_date': str(data.index.min()) if hasattr(data.index, 'min') else None,
                    'end_date': str(data.index.max()) if hasattr(data.index, 'max') else None,
                    'has_duplicates': data.index.duplicated().any() if hasattr(data.index, 'duplicated') else False
                }

        return verification_result

    def create_joint_parquet_file(self,
                                 step_name: str,
                                 ohlcv_data: Optional[pd.DataFrame] = None,
                                 labels_data: Optional[pd.DataFrame] = None,
                                 features_data: Optional[pd.DataFrame] = None,
                                 key: str = 'joint_dataset') -> Optional[str]:
        """Create joint Parquet file with aligned OHLCV + labels + features."""
        try:
            # Verify data alignment
            if self.config.enable_data_verification:
                alignment_result = self._verify_data_alignment(
                    ohlcv_data, labels_data, features_data
                )

                if not alignment_result['aligned']:
                    logger.warning(f"⚠️ Data alignment issues detected: {alignment_result['issues']}")
                else:
                    logger.info("✅ Data alignment verified successfully")

            # Create joint dataframe
            joint_data = {}

            if ohlcv_data is not None:
                joint_data.update(ohlcv_data.to_dict('series'))

            if labels_data is not None:
                # Add label prefix to avoid column conflicts
                for col in labels_data.columns:
                    joint_data[f'label_{col}'] = labels_data[col]

            if features_data is not None:
                # Add feature prefix to avoid column conflicts
                for col in features_data.columns:
                    joint_data[f'feature_{col}'] = features_data[col]

            if not joint_data:
                logger.error("❌ No data provided for joint Parquet file creation")
                return None

            joint_df = pd.DataFrame(joint_data)

            # Generate filename and path
            filename = self._generate_filename(step_name, key, 'parquet')
            directory = self._generate_directory_path(step_name)
            directory.mkdir(parents=True, exist_ok=True)

            filepath = directory / filename

            # Save joint Parquet file
            joint_df.to_parquet(filepath)

            # Generate JSON metadata
            if self.config.generate_json_metadata:
                metadata = {
                    'step_name': step_name,
                    'key': key,
                    'timestamp': datetime.now().isoformat(),
                    'context': self._context,
                    'data_alignment': alignment_result if self.config.enable_data_verification else {},
                    'joint_file_info': {
                        'filepath': str(filepath),
                        'shape': joint_df.shape,
                        'columns': list(joint_df.columns),
                        'dtypes': {col: str(dtype) for col, dtype in joint_df.dtypes.items()}
                    }
                }

                metadata_filename = self._generate_filename(step_name, f'{key}_metadata', 'json')
                metadata_filepath = directory / metadata_filename

                with open(metadata_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

                logger.info(f"✅ JSON metadata saved to {metadata_filepath}")

            logger.info(f"✅ Joint Parquet file saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Failed to create joint Parquet file: {e}")
            return None

    def save(self,
            step_name: str,
            artifacts: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Save artifacts with enhanced naming and metadata generation."""
        saved_files = {}
        start_time = time.time()

        try:
            directory = self._generate_directory_path(step_name)
            directory.mkdir(parents=True, exist_ok=True)

            # Process each artifact
            for key, artifact in artifacts.items():
                filename = self._generate_filename(step_name, key, 'parquet')
                filepath = directory / filename

                # Save based on artifact type
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_parquet(filepath)
                elif isinstance(artifact, (dict, list)):
                    # Save as JSON
                    json_filename = self._generate_filename(step_name, key, 'json')
                    json_filepath = directory / json_filename
                    with open(json_filepath, 'w') as f:
                        json.dump(artifact, f, indent=2, default=str)
                    saved_files[key] = str(json_filepath)
                    continue
                else:
                    # Try to save as pickle
                    import pickle
                    pickle_filename = self._generate_filename(step_name, key, 'pkl')
                    pickle_filepath = directory / pickle_filename
                    with open(pickle_filepath, 'wb') as f:
                        pickle.dump(artifact, f)
                    saved_files[key] = str(pickle_filepath)
                    continue

                saved_files[key] = str(filepath)
                logger.info(f"✅ Artifact '{key}' saved to {filepath}")

            # Generate JSON metadata if enabled
            if self.config.generate_json_metadata and artifacts:
                metadata_content = {
                    'step_name': step_name,
                    'timestamp': datetime.now().isoformat(),
                    'context': self._context,
                    'saved_files': saved_files,
                    'processing_time_seconds': time.time() - start_time,
                    'artifacts_metadata': metadata or {}
                }

                # Add feature-specific metadata for feature-related artifacts
                for key, filepath in saved_files.items():
                    if 'feature' in key.lower() and Path(filepath).suffix == '.parquet':
                        try:
                            df = pd.read_parquet(filepath)
                            metadata_content[key] = {
                                'shape': df.shape,
                                'columns': list(df.columns),
                                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                                'memory_usage': df.memory_usage(deep=True).sum()
                            }
                        except Exception as e:
                            logger.warning(f"⚠️ Could not extract metadata for {key}: {e}")

                metadata_filename = self._generate_filename(step_name, 'metadata', 'json')
                metadata_filepath = directory / metadata_filename

                with open(metadata_filepath, 'w') as f:
                    json.dump(metadata_content, f, indent=2, default=str)

                logger.info(f"✅ JSON metadata saved to {metadata_filepath}")

            processing_time = time.time() - start_time
            logger.info(f"✅ All artifacts saved in {processing_time:.2f}s")

            return saved_files

        except Exception as e:
            logger.error(f"❌ Failed to save artifacts: {e}")
            return {}

    def load(self, step_name: str, key: str) -> Optional[Any]:
        """Load artifact by step name and key."""
        try:
            directory = self._generate_directory_path(step_name)

            # Try Parquet first
            parquet_path = directory / self._generate_filename(step_name, key, 'parquet')
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)

            # Try JSON
            json_path = directory / self._generate_filename(step_name, key, 'json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    return json.load(f)

            # Try pickle
            pickle_path = directory / self._generate_filename(step_name, key, 'pkl')
            if pickle_path.exists():
                import pickle
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)

            logger.warning(f"⚠️ Artifact '{key}' not found in step '{step_name}'")
            return None

        except Exception as e:
            logger.error(f"❌ Failed to load artifact '{key}' from step '{step_name}': {e}")
            return None

    def list_artifacts(self, step_name: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available artifacts."""
        try:
            if step_name:
                directory = self._generate_directory_path(step_name)
                if directory.exists():
                    return {step_name: [f.name for f in directory.glob('*') if f.is_file()]}
                return {step_name: []}

            # List all steps
            base_path = Path(self.config.base_dir)
            if not base_path.exists():
                return {}

            result = {}
            for symbol_dir in base_path.iterdir():
                if symbol_dir.is_dir():
                    for exchange_dir in symbol_dir.iterdir():
                        if exchange_dir.is_dir():
                            for step_dir in exchange_dir.iterdir():
                                if step_dir.is_dir():
                                    step_name = step_dir.name
                                    if step_name not in result:
                                        result[step_name] = []
                                    result[step_name].extend([f.name for f in step_dir.glob('*') if f.is_file()])

            return result

        except Exception as e:
            logger.error(f"❌ Failed to list artifacts: {e}")
            return {}

# Global artifact manager instance
_artifact_manager_instance: Optional[PreTrainingArtifactManager] = None

def get_pretraining_artifact_manager() -> PreTrainingArtifactManager:
    """Get global pre-training artifact manager instance."""
    global _artifact_manager_instance

    if _artifact_manager_instance is None:
        _artifact_manager_instance = PreTrainingArtifactManager()

    return _artifact_manager_instance

@contextmanager
def artifact_context(symbol: str,
                    exchange: str,
                    information: str = "pre_training",
                    timeframe: Optional[str] = None,
                    **kwargs):
    """Context manager for artifact manager with automatic context setting."""
    am = get_pretraining_artifact_manager()

    # Save original context
    original_context = am._context.copy()

    try:
        # Set new context
        am.set_context(
            symbol=symbol,
            exchange=exchange,
            information=information,
            timeframe=timeframe,
            **kwargs
        )

        yield am

    finally:
        # Restore original context
        am._context = original_context

__all__ = [
    'PreTrainingArtifactManager',
    'get_pretraining_artifact_manager',
    'ArtifactConfig',
    'ArtifactKeys',
    'artifact_context'
]