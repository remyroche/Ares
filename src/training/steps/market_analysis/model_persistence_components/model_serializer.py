"""Model serializer component for model persistence."""
import asyncio
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
try:
    import onnx
    import skl2onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class ModelSerializer:
    """Handles model serialization in multiple formats."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model serializer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('serialization', {})
        self.logger = system_logger.getChild('model_serializer')
        self.compression = self.config.get('compression', True)
        self.protocol_version = self.config.get('pickle_protocol', pickle.HIGHEST_PROTOCOL)
        self.base_dir = Path(self.config.get('base_dir', 'models'))
        self.format_handlers = {'pickle': self._save_pickle, 'joblib': self._save_joblib, 'onnx': self._save_onnx, 'json': self._save_json}

    @handles_errors(exceptions=(Exception,), default_return=None, context='model serialization')
    async def save_model(self, model: Any, model_id: str, format_name: str, version_info: Dict[str, Any], metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Save a model in specified format.
        
        Args:
            model: Model to save
            model_id: Unique model identifier
            format_name: Serialization format
            version_info: Version information
            metadata: Optional model metadata
            
        Returns:
            Path to saved model file
        """
        if format_name not in self.format_handlers:
            self.logger.warning(f'Unknown format: {format_name}')
            return None
        save_dir = self.base_dir / version_info['version'] / 'models' / format_name
        save_dir.mkdir(parents=True, exist_ok=True)
        handler = self.format_handlers[format_name]
        file_path = await handler(model, model_id, save_dir, metadata)
        return file_path

    async def _save_pickle(self, model: Any, model_id: str, save_dir: Path, metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Save model using pickle.
        
        Args:
            model: Model to save
            model_id: Model identifier
            save_dir: Directory to save in
            metadata: Optional metadata
            
        Returns:
            Path to saved file
        """
        try:
            file_path = save_dir / f'{model_id}.pkl'
            model_wrapper = {'model': model, 'metadata': metadata or {}, 'format': 'pickle', 'protocol': self.protocol_version}
            with open(file_path, 'wb') as f:
                pickle.dump(model_wrapper, f, protocol=self.protocol_version)
            return str(file_path)
        except Exception as e:
            self.logger.error(f'Failed to save {model_id} as pickle: {str(e)}')
            return None

    async def _save_joblib(self, model: Any, model_id: str, save_dir: Path, metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Save model using joblib.
        
        Args:
            model: Model to save
            model_id: Model identifier
            save_dir: Directory to save in
            metadata: Optional metadata
            
        Returns:
            Path to saved file
        """
        if not JOBLIB_AVAILABLE:
            self.logger.warning('Joblib not available')
            return None
        try:
            file_path = save_dir / f'{model_id}.joblib'
            model_wrapper = {'model': model, 'metadata': metadata or {}, 'format': 'joblib'}
            compress = 3 if self.compression else 0
            joblib.dump(model_wrapper, file_path, compress=compress)
            return str(file_path)
        except Exception as e:
            self.logger.error(f'Failed to save {model_id} as joblib: {str(e)}')
            return None

    async def _save_onnx(self, model: Any, model_id: str, save_dir: Path, metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Save model in ONNX format.
        
        Args:
            model: Model to save
            model_id: Model identifier
            save_dir: Directory to save in
            metadata: Optional metadata
            
        Returns:
            Path to saved file
        """
        if not ONNX_AVAILABLE:
            self.logger.warning('ONNX not available')
            return None
        try:
            if not hasattr(model, 'predict'):
                self.logger.warning(f'Model {model_id} cannot be converted to ONNX')
                return None
            file_path = save_dir / f'{model_id}.onnx'
            n_features = metadata.get('n_features', 10) if metadata else 10
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            try:
                onx = convert_sklearn(model, initial_types=initial_type)
                with open(file_path, 'wb') as f:
                    f.write(onx.SerializeToString())
                if metadata:
                    meta_path = save_dir / f'{model_id}_metadata.json'
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                return str(file_path)
            except Exception as e:
                self.logger.warning(f'ONNX conversion failed for {model_id}: {str(e)}')
                return None
        except Exception as e:
            self.logger.error(f'Failed to save {model_id} as ONNX: {str(e)}')
            return None

    async def _save_json(self, model: Any, model_id: str, save_dir: Path, metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Save model metadata as JSON.
        
        Args:
            model: Model (not saved, only metadata)
            model_id: Model identifier
            save_dir: Directory to save in
            metadata: Model metadata
            
        Returns:
            Path to saved file
        """
        try:
            file_path = save_dir / f'{model_id}_info.json'
            model_info = {'model_id': model_id, 'model_type': model.__class__.__name__, 'model_module': model.__class__.__module__, 'metadata': metadata or {}, 'format': 'json_metadata'}
            if hasattr(model, 'get_params'):
                try:
                    model_info['parameters'] = model.get_params()
                except:
                    pass
            if hasattr(model, 'feature_importances_'):
                model_info['feature_importances'] = model.feature_importances_.tolist()
            with open(file_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            return str(file_path)
        except Exception as e:
            self.logger.error(f'Failed to save {model_id} metadata as JSON: {str(e)}')
            return None

    @handles_errors(exceptions=(Exception,), default_return=None, context='model loading')
    async def load_model(self, file_path: str, format_name: Optional[str]=None) -> Optional[Any]:
        """Load a model from file.
        
        Args:
            file_path: Path to model file
            format_name: Format to use (auto-detect if None)
            
        Returns:
            Loaded model or None
        """
        path = Path(file_path)
        if not path.exists():
            self.logger.error(f'Model file not found: {file_path}')
            return None
        if format_name is None:
            if path.suffix == '.pkl':
                format_name = 'pickle'
            elif path.suffix == '.joblib':
                format_name = 'joblib'
            elif path.suffix == '.onnx':
                format_name = 'onnx'
            else:
                self.logger.error(f'Unknown file format: {path.suffix}')
                return None
        if format_name == 'pickle':
            return await self._load_pickle(path)
        elif format_name == 'joblib':
            return await self._load_joblib(path)
        elif format_name == 'onnx':
            return await self._load_onnx(path)
        else:
            self.logger.error(f'Unsupported format: {format_name}')
            return None

    async def _load_pickle(self, file_path: Path) -> Optional[Any]:
        """Load model from pickle file."""
        try:
            with open(file_path, 'rb') as f:
                model_wrapper = pickle.load(f)
            if isinstance(model_wrapper, dict) and 'model' in model_wrapper:
                return model_wrapper['model']
            else:
                return model_wrapper
        except Exception as e:
            self.logger.error(f'Failed to load pickle model: {str(e)}')
            return None

    async def _load_joblib(self, file_path: Path) -> Optional[Any]:
        """Load model from joblib file."""
        if not JOBLIB_AVAILABLE:
            self.logger.error('Joblib not available')
            return None
        try:
            model_wrapper = joblib.load(file_path)
            if isinstance(model_wrapper, dict) and 'model' in model_wrapper:
                return model_wrapper['model']
            else:
                return model_wrapper
        except Exception as e:
            self.logger.error(f'Failed to load joblib model: {str(e)}')
            return None

    async def _load_onnx(self, file_path: Path) -> Optional[Any]:
        """Load model from ONNX file."""
        if not ONNX_AVAILABLE:
            self.logger.error('ONNX not available')
            return None
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(file_path))
            return session
        except Exception as e:
            self.logger.error(f'Failed to load ONNX model: {str(e)}')
            return None