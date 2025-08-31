"""
Standardized Model Manager

This module provides centralized model management functionality including:
- Model saving/loading with standardized paths
- Model versioning and metadata tracking
- Model validation and testing
- Model lifecycle management
"""

import os
import json
import pickle
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors


class ModelMetadata:
    """Model metadata container."""
    
    def __init__(self, model_id: str, step_name: str, model_type: str, **kwargs):
        self.model_id = model_id
        self.step_name = step_name
        self.model_type = model_type
        self.created_at = datetime.now().isoformat()
        self.version = kwargs.get('version', '1.0.0')
        self.description = kwargs.get('description', '')
        self.parameters = kwargs.get('parameters', {})
        self.metrics = kwargs.get('metrics', {})
        self.features = kwargs.get('features', [])
        self.tags = kwargs.get('tags', [])
        self.file_path = kwargs.get('file_path', '')
        self.file_size = kwargs.get('file_size', 0)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'model_id': self.model_id,
            'step_name': self.step_name,
            'model_type': self.model_type,
            'created_at': self.created_at,
            'version': self.version,
            'description': self.description,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'features': self.features,
            'tags': self.tags,
            'file_path': self.file_path,
            'file_size': self.file_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class StandardizedModelManager:
    """Centralized model management system."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize the model manager.
        
        Args:
            base_path: Base path for model storage. Defaults to data_cache/models/
        """
        self.standards = pipeline_standards
        self.logger = system_logger
        
        if base_path is None:
            self.base_path = Path("data_cache/models")
        else:
            self.base_path = Path(base_path)
            
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "model_registry.json"
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the model registry from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.registry = json.load(f)
            else:
                self.registry = {}
        except Exception as e:
            self.logger.warning(f"Could not load model registry: {e}")
            self.registry = {}
    
    def _save_registry(self) -> None:
        """Save the model registry to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save model registry: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model saving"
    )
    def save_model(
        self, 
        model: Any, 
        metadata: Union[ModelMetadata, Dict[str, Any]], 
        step_name: str,
        model_id: Optional[str] = None
    ) -> bool:
        """Save a model with metadata.
        
        Args:
            model: The model object to save
            metadata: Model metadata or dictionary
            step_name: Name of the step that created the model
            model_id: Optional model ID. If not provided, will be generated
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert metadata to ModelMetadata if needed
            if isinstance(metadata, dict):
                metadata = ModelMetadata(**metadata)
            
            # Generate model ID if not provided
            if model_id is None:
                model_id = f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            metadata.model_id = model_id
            metadata.step_name = step_name
            
            # Create step directory
            step_dir = self.base_path / step_name
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file extension based on model type
            if hasattr(model, 'save') and callable(getattr(model, 'save', None)):
                # PyTorch model
                file_path = step_dir / f"{model_id}.pth"
                import torch
                torch.save(model.state_dict(), file_path)
            elif hasattr(model, 'save_model'):
                # LightGBM model
                file_path = step_dir / f"{model_id}.txt"
                model.save_model(str(file_path))
            elif hasattr(model, 'save'):
                # XGBoost model
                file_path = step_dir / f"{model_id}.json"
                model.save_model(str(file_path))
            else:
                # Generic model (pickle/joblib)
                file_path = step_dir / f"{model_id}.joblib"
                joblib.dump(model, file_path)
            
            # Update metadata
            metadata.file_path = str(file_path)
            metadata.file_size = file_path.stat().st_size if file_path.exists() else 0
            
            # Save metadata
            metadata_path = step_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update registry
            self.registry[model_id] = metadata.to_dict()
            self._save_registry()
            
            self.logger.info(f"Model saved successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model loading"
    )
    def load_model(self, model_id: str, step_name: Optional[str] = None) -> Optional[Tuple[Any, ModelMetadata]]:
        """Load a model and its metadata.
        
        Args:
            model_id: ID of the model to load
            step_name: Optional step name for faster lookup
            
        Returns:
            Tuple of (model, metadata) or None if failed
        """
        try:
            # Get metadata from registry
            if model_id not in self.registry:
                self.logger.error(f"Model not found in registry: {model_id}")
                return None
            
            metadata_dict = self.registry[model_id]
            metadata = ModelMetadata.from_dict(metadata_dict)
            
            # Determine step name
            if step_name is None:
                step_name = metadata.step_name
            
            # Load model
            file_path = Path(metadata.file_path)
            if not file_path.exists():
                self.logger.error(f"Model file not found: {file_path}")
                return None
            
            # Load based on file extension
            if file_path.suffix == '.pth':
                # PyTorch model - requires model class to be provided
                self.logger.warning("PyTorch models require model class for loading")
                return None, metadata
            elif file_path.suffix == '.txt':
                # LightGBM model
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(file_path))
            elif file_path.suffix == '.json':
                # XGBoost model
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(file_path))
            else:
                # Generic model (joblib)
                model = joblib.load(file_path)
            
            self.logger.info(f"Model loaded successfully: {model_id}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model validation"
    )
    def validate_model(self, model: Any, test_data: Union[pd.DataFrame, np.ndarray], 
                      expected_output_shape: Optional[Tuple] = None) -> bool:
        """Validate a model with test data.
        
        Args:
            model: The model to validate
            test_data: Test data for validation
            expected_output_shape: Expected output shape (optional)
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Basic model validation
            if model is None:
                self.logger.error("Model is None")
                return False
            
            # Test prediction
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data)
                
                # Check output shape if specified
                if expected_output_shape is not None:
                    if predictions.shape != expected_output_shape:
                        self.logger.error(f"Output shape mismatch: {predictions.shape} != {expected_output_shape}")
                        return False
                
                # Check for NaN/Inf values
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    self.logger.error("Predictions contain NaN or Inf values")
                    return False
                
                self.logger.info("Model validation passed")
                return True
            else:
                self.logger.error("Model does not have predict method")
                return False
                
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            ModelMetadata or None if not found
        """
        if model_id in self.registry:
            return ModelMetadata.from_dict(self.registry[model_id])
        return None
    
    def list_models(self, step_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models or models for a specific step.
        
        Args:
            step_name: Optional step name to filter by
            
        Returns:
            List of model metadata dictionaries
        """
        if step_name is None:
            return list(self.registry.values())
        else:
            return [
                metadata for metadata in self.registry.values()
                if metadata.get('step_name') == step_name
            ]
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            bool: True if successful
        """
        try:
            if model_id not in self.registry:
                self.logger.error(f"Model not found: {model_id}")
                return False
            
            metadata = self.registry[model_id]
            file_path = Path(metadata['file_path'])
            
            # Delete model file
            if file_path.exists():
                file_path.unlink()
            
            # Delete metadata file
            metadata_path = file_path.parent / f"{model_id}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from registry
            del self.registry[model_id]
            self._save_registry()
            
            self.logger.info(f"Model deleted successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about stored models.
        
        Returns:
            Dictionary with model statistics
        """
        stats = {
            'total_models': len(self.registry),
            'models_by_step': {},
            'models_by_type': {},
            'total_size': 0
        }
        
        for model_id, metadata in self.registry.items():
            step_name = metadata.get('step_name', 'unknown')
            model_type = metadata.get('model_type', 'unknown')
            file_size = metadata.get('file_size', 0)
            
            # Count by step
            stats['models_by_step'][step_name] = stats['models_by_step'].get(step_name, 0) + 1
            
            # Count by type
            stats['models_by_type'][model_type] = stats['models_by_type'].get(model_type, 0) + 1
            
            # Total size
            stats['total_size'] += file_size
        
        return stats


# Global instance
standardized_model_manager = StandardizedModelManager()