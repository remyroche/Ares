"""
Model Management Utilities

Common model management patterns shared across all training modules.
"""

import joblib
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """Common model management utilities."""
    
    def __init__(self, save_path: str, save_format: str = "joblib"):
        """
        Initialize model manager.
        
        Args:
            save_path: Base path for saving models
            save_format: Format for saving models (joblib, pickle, h5)
        """
        self.save_path = Path(save_path)
        self.save_format = save_format
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def save_models(
        self, 
        models: Dict[str, Any], 
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> List[str]:
        """
        Save models with common logic.
        
        Args:
            models: Dictionary of models to save
            model_type: Type of models (e.g., 'hmm_base', 'analyst_ensemble')
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier
            
        Returns:
            List of saved model file paths
        """
        model_paths = []
        
        # Create model-specific directory
        if regime is not None:
            model_dir = self.save_path / model_type / f"regime_{regime}"
        else:
            model_dir = self.save_path / model_type
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            # Create filename
            filename_parts = [model_type, model_name]
            if symbol:
                filename_parts.append(symbol)
            if exchange:
                filename_parts.append(exchange)
            if timeframe:
                filename_parts.append(timeframe)
            
            filename = "_".join(filename_parts) + f".{self.save_format}"
            model_file = model_dir / filename
            
            try:
                # Save model based on format
                if self.save_format == "joblib":
                    joblib.dump(model, model_file)
                elif self.save_format == "pickle":
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                elif self.save_format == "h5":
                    # For Keras models
                    if hasattr(model, 'save'):
                        model.save(str(model_file))
                    else:
                        logger.warning(f"⚠️ Model {model_name} doesn't support H5 format, using joblib")
                        joblib.dump(model, model_file.with_suffix('.joblib'))
                else:
                    raise ValueError(f"Unsupported save format: {self.save_format}")
                
                model_paths.append(str(model_file))
                logger.debug(f"💾 Saved {model_name} to {model_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save {model_name}: {e}")
                continue
        
        logger.info(f"💾 Saved {len(model_paths)} models to {model_dir}")
        return model_paths
    
    def load_models(
        self, 
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load models with common logic.
        
        Args:
            model_type: Type of models to load
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier
            
        Returns:
            Dictionary of loaded models
        """
        models = {}
        
        # Determine model directory
        if regime is not None:
            model_dir = self.save_path / model_type / f"regime_{regime}"
        else:
            model_dir = self.save_path / model_type
        
        if not model_dir.exists():
            logger.warning(f"⚠️ Model directory not found: {model_dir}")
            return models
        
        # Load all model files in directory
        for model_file in model_dir.glob(f"*.{self.save_format}"):
            model_name = model_file.stem
            
            try:
                # Load model based on format
                if self.save_format == "joblib":
                    model = joblib.load(model_file)
                elif self.save_format == "pickle":
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                elif self.save_format == "h5":
                    # For Keras models
                    from tensorflow.keras.models import load_model
                    model = load_model(str(model_file))
                else:
                    raise ValueError(f"Unsupported save format: {self.save_format}")
                
                models[model_name] = model
                logger.debug(f"📂 Loaded {model_name} from {model_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
                continue
        
        logger.info(f"📂 Loaded {len(models)} models from {model_dir}")
        return models
    
    def save_metadata(
        self, 
        metadata: Dict[str, Any], 
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> str:
        """
        Save model metadata.
        
        Args:
            metadata: Model metadata to save
            model_type: Type of models
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier
            
        Returns:
            Path to saved metadata file
        """
        # Create model-specific directory
        if regime is not None:
            model_dir = self.save_path / model_type / f"regime_{regime}"
        else:
            model_dir = self.save_path / model_type
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata filename
        filename_parts = [model_type, "metadata"]
        if symbol:
            filename_parts.append(symbol)
        if exchange:
            filename_parts.append(exchange)
        if timeframe:
            filename_parts.append(timeframe)
        
        metadata_file = model_dir / "_".join(filename_parts) + ".json"
        
        # Add timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Saved metadata to {metadata_file}")
        return str(metadata_file)
    
    def load_metadata(
        self, 
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load model metadata.
        
        Args:
            model_type: Type of models
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier
            
        Returns:
            Loaded metadata or None if not found
        """
        # Determine model directory
        if regime is not None:
            model_dir = self.save_path / model_type / f"regime_{regime}"
        else:
            model_dir = self.save_path / model_type
        
        # Create metadata filename
        filename_parts = [model_type, "metadata"]
        if symbol:
            filename_parts.append(symbol)
        if exchange:
            filename_parts.append(exchange)
        if timeframe:
            filename_parts.append(timeframe)
        
        metadata_file = model_dir / "_".join(filename_parts) + ".json"
        
        if not metadata_file.exists():
            logger.warning(f"⚠️ Metadata file not found: {metadata_file}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📂 Loaded metadata from {metadata_file}")
            return metadata
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            return None
    
    def get_model_metadata(
        self, 
        model: Any, 
        model_name: str,
        training_time: float = 0.0,
        optimization_time: float = 0.0,
        samples: int = 0,
        features: int = 0
    ) -> Dict[str, Any]:
        """
        Extract common model metadata.
        
        Args:
            model: Trained model
            model_name: Name of the model
            training_time: Training time in seconds
            optimization_time: Optimization time in seconds
            samples: Number of training samples
            features: Number of features
            
        Returns:
            Dictionary containing model metadata
        """
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_time': training_time,
            'optimization_time': optimization_time,
            'samples': samples,
            'features': features,
            'created_at': datetime.now().isoformat()
        }
        
        # Add model-specific metadata
        if hasattr(model, 'get_params'):
            try:
                metadata['model_params'] = model.get_params()
            except:
                pass
        
        if hasattr(model, 'feature_importances_'):
            try:
                metadata['feature_importances'] = model.feature_importances_.tolist()
            except:
                pass
        
        if hasattr(model, 'n_features_in_'):
            try:
                metadata['n_features_in'] = model.n_features_in_
            except:
                pass
        
        return metadata
    
    def cleanup_old_models(
        self, 
        model_type: str,
        keep_latest: int = 5,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> int:
        """
        Clean up old model files, keeping only the latest ones.
        
        Args:
            model_type: Type of models to clean up
            keep_latest: Number of latest models to keep
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            
        Returns:
            Number of files deleted
        """
        # Determine model directory
        model_dir = self.save_path / model_type
        
        if not model_dir.exists():
            return 0
        
        # Find all model files
        pattern = f"*.{self.save_format}"
        if symbol:
            pattern = f"*{symbol}*{pattern}"
        if exchange:
            pattern = f"*{exchange}*{pattern}"
        if timeframe:
            pattern = f"*{timeframe}*{pattern}"
        
        model_files = list(model_dir.glob(pattern))
        
        if len(model_files) <= keep_latest:
            return 0
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Delete old files
        files_to_delete = model_files[keep_latest:]
        deleted_count = 0
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"🗑️ Deleted old model: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {file_path}: {e}")
        
        logger.info(f"🗑️ Cleaned up {deleted_count} old model files")
        return deleted_count