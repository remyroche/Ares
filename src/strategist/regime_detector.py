"""
Regime Detector

Loads models from market_analysis/regime_models_training and regime_ensemble_training
and provides regime predictions for the trading system.
"""

import asyncio
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success
)
from src.utils.model_manager import ModelManager
from src.core.decorators import handles_errors


logger = system_logger.getChild('RegimeDetector')


class RegimeDetector:
    """
    Regime Detector
    
    Loads and uses models from:
    - market_analysis/regime_models_training (base models - flexible, works with any models found)
    - market_analysis/regime_ensemble_training (ensemble model - flexible, works with any ensemble found)
    
    Automatically discovers and loads whatever models are present in the artifacts,
    regardless of model type (CatBoost, GreedyRuleLists, ExtraTrees, XGBoost, LightGBM, etc.).
    
    Provides regime predictions for Analyst and Tactician.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Regime Detector.
        
        Args:
            config: Configuration dictionary with model paths and settings
        """
        self.config = config
        self.logger = logger.getChild('RegimeDetector')
        
        # Model paths
        self.models_directory = config.get('models_directory', 'artifacts/regime_models')
        self.regime_models_path = config.get(
            'regime_models_path',
            f'{self.models_directory}/regime_models_training_result.pkl'
        )
        self.ensemble_model_path = config.get(
            'ensemble_model_path',
            f'{self.models_directory}/regime_ensemble_training_result.pkl'
        )
        
        # Model references
        self.base_models: Dict[str, Any] = {}  # CatBoost, GreedyRuleLists, ExtraTrees
        self.ensemble_model: Any = None  # stacker_lgbm_calibrated
        self.feature_names: List[str] = []
        self.regime_count: int = 0
        
        # Model Manager for loading
        self.model_manager: Optional[ModelManager] = None
        
        # State
        self.is_initialized = False
        
        tprint_info("🚀 Initializing Regime Detector...")

    @handles_errors
    async def initialize(self) -> bool:
        """
        Initialize regime detector by loading models.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Loading regime detection models...")
            
            # Initialize Model Manager if needed
            if self.model_manager is None:
                self.model_manager = ModelManager()
            
            # Load base models from regime_models_training
            await self._load_base_models()
            
            # Load ensemble model from regime_ensemble_training
            await self._load_ensemble_model()
            
            # Load feature names and metadata
            await self._load_metadata()
            
            if not self.base_models and self.ensemble_model is None:
                error_msg = "Failed to load any regime detection models"
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            self.is_initialized = True
            self.logger.info(
                f"✅ Regime Detector initialized: {len(self.base_models)} base models, "
                f"ensemble: {self.ensemble_model is not None}, regimes: {self.regime_count}"
            )
            tprint_success("✅ Regime Detector initialized successfully")
            
            return True

        except Exception as e:
            error_msg = f"Failed to initialize Regime Detector: {e}"
            self.logger.error(f"❌ {error_msg}")
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _load_base_models(self) -> None:
        """Load base models from regime_models_training artifacts (flexible, works with any models found)."""
        try:
            models_path = Path(self.regime_models_path)
            if not models_path.exists():
                self.logger.warning(
                    f"⚠️ Regime models file not found: {models_path}. "
                    "Attempting to load from artifacts directory."
                )
                # Try alternative paths
                artifacts_dir = Path(self.models_directory)
                if artifacts_dir.exists():
                    # Look for regime_models_training_result in artifacts
                    for artifact_file in artifacts_dir.glob("**/regime_models_training_result.pkl"):
                        models_path = artifact_file
                        break
                    else:
                        raise FileNotFoundError(
                            f"Regime models not found in {artifacts_dir}"
                        )
                else:
                    raise FileNotFoundError(f"Models directory not found: {artifacts_dir}")
            
            self.logger.info(f"Loading regime models from: {models_path}")
            
            with open(models_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            # Extract base models from artifacts - flexible approach
            # Try multiple artifact structures and extract any models found
            self.base_models = {}
            
            if isinstance(artifacts, dict):
                # Strategy 1: Check for 'models' key (dict of models)
                if 'models' in artifacts and isinstance(artifacts['models'], dict):
                    self.base_models.update(artifacts['models'])
                    self.logger.debug(f"Found models in 'models' key: {list(artifacts['models'].keys())}")
                
                # Strategy 2: Check for component_result structure
                if 'component_result' in artifacts:
                    result = artifacts['component_result']
                    if isinstance(result, dict):
                        if 'models' in result and isinstance(result['models'], dict):
                            self.base_models.update(result['models'])
                            self.logger.debug(f"Found models in component_result: {list(result['models'].keys())}")
                        # Also check if component_result itself contains model-like objects
                        self._extract_models_from_dict(result, 'component_result')
                
                # Strategy 3: Check for training_result structure
                if 'training_result' in artifacts:
                    result = artifacts['training_result']
                    if isinstance(result, dict):
                        if 'models' in result and isinstance(result['models'], dict):
                            self.base_models.update(result['models'])
                            self.logger.debug(f"Found models in training_result: {list(result['models'].keys())}")
                        self._extract_models_from_dict(result, 'training_result')
                
                # Strategy 4: Search entire artifact dict for model-like objects
                self._extract_models_from_dict(artifacts, 'root')
                
                # Strategy 5: Check for list of models
                if 'models' in artifacts and isinstance(artifacts['models'], list):
                    for i, model in enumerate(artifacts['models']):
                        if self._is_model_object(model):
                            model_name = f"model_{i}"
                            self.base_models[model_name] = model
                            self.logger.debug(f"Found model in models list: {model_name}")
            
            if not self.base_models:
                self.logger.warning("⚠️ No base models found in artifacts")
            else:
                model_names = list(self.base_models.keys())
                self.logger.info(
                    f"✅ Loaded {len(self.base_models)} base models: {model_names}"
                )
                tprint_info(f"✅ Loaded {len(self.base_models)} base models: {', '.join(model_names)}")
                
        except FileNotFoundError as e:
            error_msg = f"Regime models file not found: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load base models: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _extract_models_from_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
        """
        Recursively extract model-like objects from a dictionary and add to base_models.
        
        Args:
            data: Dictionary to search
            prefix: Prefix for model names (for nested structures)
        """
        if not isinstance(data, dict):
            return
        
        for key, value in data.items():
            # Skip if already in base_models
            if key in self.base_models:
                continue
            
            # Check if value is a model object
            if self._is_model_object(value):
                model_name = key if not prefix else f"{prefix}_{key}"
                self.base_models[model_name] = value
                self.logger.debug(f"Found model: {model_name} (type: {type(value).__name__})")
            # Recursively search nested dicts (but limit depth to avoid recursion issues)
            elif isinstance(value, dict) and len(str(prefix)) < 50:  # Depth limit
                self._extract_models_from_dict(value, f"{prefix}_{key}" if prefix else key)

    def _search_for_models_in_dict(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Search for model-like objects in a dictionary and return them without modifying base_models.
        
        Args:
            data: Dictionary to search
            prefix: Prefix for model names (for nested structures)
            
        Returns:
            Dict of model_name -> model_object
        """
        found_models = {}
        
        if not isinstance(data, dict):
            return found_models
        
        for key, value in data.items():
            # Check if value is a model object
            if self._is_model_object(value):
                model_name = key if not prefix else f"{prefix}_{key}"
                found_models[model_name] = value
            # Recursively search nested dicts (but limit depth to avoid recursion issues)
            elif isinstance(value, dict) and len(str(prefix)) < 50:  # Depth limit
                nested_models = self._search_for_models_in_dict(value, f"{prefix}_{key}" if prefix else key)
                found_models.update(nested_models)
        
        return found_models

    def _is_model_object(self, obj: Any) -> bool:
        """
        Check if an object is likely a trained ML model.
        
        Args:
            obj: Object to check
            
        Returns:
            bool: True if object appears to be a model
        """
        if obj is None:
            return False
        
        # Check for common model attributes/methods
        model_indicators = [
            'predict',  # Most models have predict method
            'predict_proba',  # Many classification models
            'fit',  # Training method (present even after training)
            'transform',  # Some models have transform
        ]
        
        # Check if object has at least one model indicator
        has_predict = hasattr(obj, 'predict') and callable(getattr(obj, 'predict', None))
        has_predict_proba = hasattr(obj, 'predict_proba') and callable(getattr(obj, 'predict_proba', None))
        
        if has_predict or has_predict_proba:
            # Additional check: exclude some common non-model types
            obj_type = type(obj).__name__.lower()
            excluded_types = [
                'dict', 'list', 'tuple', 'str', 'int', 'float', 'bool',
                'dataframe', 'series', 'array', 'ndarray', 'series',
                'scaler', 'encoder', 'transformer'  # Preprocessing objects
            ]
            
            if obj_type not in excluded_types:
                return True
        
        return False

    async def _load_ensemble_model(self) -> None:
        """Load ensemble model from regime_ensemble_training artifacts (flexible, works with any ensemble model found)."""
        try:
            ensemble_path = Path(self.ensemble_model_path)
            if not ensemble_path.exists():
                self.logger.warning(
                    f"⚠️ Ensemble model file not found: {ensemble_path}. "
                    "Attempting to load from artifacts directory."
                )
                # Try alternative paths
                artifacts_dir = Path(self.models_directory)
                if artifacts_dir.exists():
                    # Look for regime_ensemble_training_result in artifacts
                    for artifact_file in artifacts_dir.glob("**/regime_ensemble_training_result.pkl"):
                        ensemble_path = artifact_file
                        break
                    else:
                        self.logger.warning("⚠️ Ensemble model not found, will use base models only")
                        return
                else:
                    self.logger.warning(f"⚠️ Models directory not found: {artifacts_dir}")
                    return
            
            self.logger.info(f"Loading ensemble model from: {ensemble_path}")
            
            with open(ensemble_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            # Extract ensemble model - flexible approach
            # Strategy 1: Check for known ensemble model keys
            if isinstance(artifacts, dict):
                for key in ['ensemble_model', 'stacker_lgbm_calibrated', 'meta_model', 'stacker_model', 'meta_learner']:
                    if key in artifacts:
                        candidate = artifacts[key]
                        if self._is_model_object(candidate):
                            self.ensemble_model = candidate
                            self.logger.debug(f"Found ensemble model in key '{key}': {type(candidate).__name__}")
                            break
                
                # Strategy 2: Check component_result structure
                if self.ensemble_model is None and 'component_result' in artifacts:
                    result = artifacts['component_result']
                    if isinstance(result, dict):
                        for key in ['ensemble_model', 'stacker_lgbm_calibrated', 'meta_model', 'meta_learner']:
                            if key in result:
                                candidate = result[key]
                                if self._is_model_object(candidate):
                                    self.ensemble_model = candidate
                                    self.logger.debug(f"Found ensemble model in component_result['{key}']: {type(candidate).__name__}")
                                    break
                        # Search recursively in component_result for ensemble (only)
                        if self.ensemble_model is None:
                            ensemble_candidates = self._search_for_models_in_dict(result, prefix='component_result')
                            # If exactly one model found, it's likely the ensemble
                            if len(ensemble_candidates) == 1:
                                self.ensemble_model = list(ensemble_candidates.values())[0]
                                self.logger.debug(f"Found ensemble model via recursive search: {type(self.ensemble_model).__name__}")
                
                # Strategy 3: Search entire artifact for model-like objects (ensemble only)
                if self.ensemble_model is None:
                    ensemble_candidates = self._search_for_models_in_dict(artifacts, prefix='root')
                    # If exactly one model found and it's not in base_models, it might be the ensemble
                    if len(ensemble_candidates) == 1:
                        candidate_name, candidate_model = list(ensemble_candidates.items())[0]
                        if candidate_name not in self.base_models:
                            self.ensemble_model = candidate_model
                            self.logger.debug(f"Found ensemble model via artifact search: {type(candidate_model).__name__}")
            
            if self.ensemble_model is None:
                self.logger.warning("⚠️ Ensemble model not found in artifacts")
            else:
                model_type = type(self.ensemble_model).__name__
                self.logger.info(f"✅ Loaded ensemble model: {model_type}")
                tprint_info(f"✅ Loaded ensemble model: {model_type}")
                
        except Exception as e:
            # Non-critical: we can still use base models
            self.logger.warning(f"⚠️ Failed to load ensemble model: {e}. Will use base models only.")

    async def _load_metadata(self) -> None:
        """Load feature names and metadata."""
        try:
            # Try to load metadata from artifacts
            metadata_path = Path(self.models_directory) / "regime_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    self.regime_count = metadata.get('regime_count', 0)
            else:
                # Try to infer from models
                if self.base_models:
                    # Try to get feature names from first model
                    for model_name, model in self.base_models.items():
                        if hasattr(model, 'feature_names_'):
                            self.feature_names = list(model.feature_names_)
                            break
                        elif hasattr(model, 'feature_names_in_'):
                            self.feature_names = list(model.feature_names_in_)
                            break
                
                # Try to get regime count from models
                if self.ensemble_model is not None:
                    if hasattr(self.ensemble_model, 'n_classes_'):
                        self.regime_count = self.ensemble_model.n_classes_
                    elif hasattr(self.ensemble_model, 'classes_'):
                        self.regime_count = len(self.ensemble_model.classes_)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load metadata: {e}")

    async def predict_regime(
        self,
        market_data: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict regime for given market data.
        
        Works with any models found in artifacts - flexible approach.
        
        Args:
            market_data: Market data DataFrame (OHLCV format)
            return_probabilities: Whether to return regime probabilities
            
        Returns:
            Dictionary with:
            - primary_regime: Predicted regime ID
            - confidence: Confidence score
            - regime_probabilities: Dict of regime_id -> probability
            - regime_strength: Regime strength metric
            - transition_probability: Transition probability
            - features_used: Features used for prediction
        """
        if not self.is_initialized:
            raise RuntimeError("Regime Detector not initialized. Call initialize() first.")
        
        try:
            # Prepare features from market data
            features = self._prepare_features(market_data)
            
            if len(features) == 0:
                raise ValueError("No features extracted from market data")
            
            # Get predictions from base models (flexible - works with any model type)
            base_predictions = {}
            base_probabilities = {}
            
            for model_name, model in self.base_models.items():
                try:
                    # Try predict method (works for most models)
                    if hasattr(model, 'predict') and callable(getattr(model, 'predict', None)):
                        pred = model.predict(features)
                        # Handle different return shapes
                        if isinstance(pred, np.ndarray):
                            base_predictions[model_name] = int(pred[0]) if len(pred) > 0 else None
                        elif isinstance(pred, (list, tuple)):
                            base_predictions[model_name] = int(pred[0]) if len(pred) > 0 else None
                        else:
                            base_predictions[model_name] = int(pred) if pred is not None else None
                    else:
                        self.logger.warning(f"⚠️ Model {model_name} does not have predict method, skipping")
                        continue
                    
                    # Try to get probabilities if available and requested
                    if return_probabilities and hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba', None)):
                        try:
                            proba = model.predict_proba(features)
                            # Handle different probability formats
                            if isinstance(proba, np.ndarray):
                                if len(proba.shape) == 2 and proba.shape[0] > 0:
                                    base_probabilities[model_name] = proba[0]
                                else:
                                    base_probabilities[model_name] = proba.flatten()
                            else:
                                base_probabilities[model_name] = proba
                        except Exception as e:
                            self.logger.debug(f"Model {model_name} predict_proba failed (non-critical): {e}")
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ Model {model_name} prediction failed: {e}")
                    continue
            
            if not base_predictions:
                raise RuntimeError("No base model predictions available")
            
            # Use ensemble model if available, otherwise use voting from base models
            if self.ensemble_model is not None:
                try:
                    # Prepare features for ensemble (base model predictions)
                    # Get predictions in consistent order
                    model_names = list(self.base_models.keys())
                    ensemble_features = np.array([
                        base_predictions.get(name, 0) for name in model_names
                    ])
                    ensemble_features = ensemble_features.reshape(1, -1)
                    
                    # Predict with ensemble
                    if hasattr(self.ensemble_model, 'predict') and callable(getattr(self.ensemble_model, 'predict', None)):
                        ensemble_pred = self.ensemble_model.predict(ensemble_features)
                        primary_regime = int(ensemble_pred[0]) if len(ensemble_pred) > 0 else int(ensemble_pred)
                    else:
                        raise AttributeError("Ensemble model does not have predict method")
                    
                    # Get probabilities if available
                    if return_probabilities and hasattr(self.ensemble_model, 'predict_proba') and callable(getattr(self.ensemble_model, 'predict_proba', None)):
                        try:
                            ensemble_proba = self.ensemble_model.predict_proba(ensemble_features)
                            if isinstance(ensemble_proba, np.ndarray) and len(ensemble_proba.shape) == 2:
                                ensemble_proba_flat = ensemble_proba[0]
                            else:
                                ensemble_proba_flat = ensemble_proba.flatten()
                            regime_probabilities = {
                                f"regime_{i}": float(prob) for i, prob in enumerate(ensemble_proba_flat)
                            }
                        except Exception as e:
                            self.logger.debug(f"Ensemble predict_proba failed: {e}")
                            regime_probabilities = {f"regime_{primary_regime}": 1.0}
                    else:
                        regime_probabilities = {f"regime_{primary_regime}": 1.0}
                    
                    confidence = float(max(regime_probabilities.values())) if regime_probabilities else 0.5
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Ensemble model prediction failed: {e}, using base models")
                    # Fallback to voting
                    primary_regime, regime_probabilities, confidence = self._vote_from_base_models(
                        base_predictions, base_probabilities
                    )
            else:
                # Use voting from base models
                primary_regime, regime_probabilities, confidence = self._vote_from_base_models(
                    base_predictions, base_probabilities
                )
            
            # Calculate regime strength (average probability)
            regime_strength = sum(regime_probabilities.values()) / len(regime_probabilities) if regime_probabilities else 0.5
            
            # Calculate transition probability (placeholder - would need historical data)
            transition_probability = 0.5
            
            return {
                'primary_regime': int(primary_regime),
                'confidence': float(confidence),
                'regime_probabilities': regime_probabilities,
                'regime_strength': float(regime_strength),
                'transition_probability': float(transition_probability),
                'features_used': {
                    'feature_count': len(features),
                    'feature_names': self.feature_names[:10] if self.feature_names else [],  # First 10
                    'models_used': list(self.base_models.keys())  # Include which models were used
                }
            }
            
        except Exception as e:
            error_msg = f"Regime prediction failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from market data.
        
        This is a simplified version. In production, this should use the same
        feature engineering pipeline as used during training.
        """
        try:
            # Basic feature extraction (should match training pipeline)
            if 'close' not in market_data.columns:
                raise ValueError("Market data must contain 'close' column")
            
            # Calculate basic features
            features = []
            close = market_data['close'].values
            
            # Price-based features
            if len(close) >= 2:
                features.append((close[-1] - close[-2]) / close[-2])  # Price change
            if len(close) >= 10:
                features.append(np.mean(close[-10:]))  # Moving average
                features.append(np.std(close[-10:]))  # Volatility
            
            # Volume-based features
            if 'volume' in market_data.columns:
                volume = market_data['volume'].values
                if len(volume) >= 10:
                    features.append(np.mean(volume[-10:]))  # Average volume
                    features.append(np.std(volume[-10:]))  # Volume volatility
            
            # OHLC features
            for col in ['open', 'high', 'low']:
                if col in market_data.columns:
                    col_values = market_data[col].values
                    if len(col_values) > 0:
                        features.append(col_values[-1])
            
            features_array = np.array(features).reshape(1, -1)
            
            return features_array
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            return np.array([])

    def _vote_from_base_models(
        self,
        base_predictions: Dict[str, Any],
        base_probabilities: Dict[str, Any]
    ) -> tuple:
        """Vote from base model predictions."""
        if not base_predictions:
            return 0, {"regime_0": 1.0}, 0.5
        
        # Simple voting: most common prediction
        predictions = [p for p in base_predictions.values() if p is not None]
        if not predictions:
            return 0, {"regime_0": 1.0}, 0.5
        
        # Get most common regime
        primary_regime = int(max(set(predictions), key=predictions.count))
        
        # Average probabilities if available
        if base_probabilities:
            all_proba = [p for p in base_probabilities.values() if p is not None]
            if all_proba:
                avg_proba = np.mean(all_proba, axis=0)
                regime_probabilities = {
                    f"regime_{i}": float(prob) for i, prob in enumerate(avg_proba)
                }
                confidence = float(max(regime_probabilities.values()))
            else:
                regime_probabilities = {f"regime_{primary_regime}": 1.0}
                confidence = 1.0
        else:
            regime_probabilities = {f"regime_{primary_regime}": 1.0}
            confidence = 1.0
        
        return primary_regime, regime_probabilities, confidence

    async def stop(self) -> None:
        """Stop the regime detector."""
        try:
            self.is_initialized = False
            self.base_models.clear()
            self.ensemble_model = None
            self.logger.info("Regime Detector stopped")
        except Exception as e:
            self.logger.warning(f"Error stopping Regime Detector: {e}")
