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
    - market_analysis/regime_models_training (base models: CatBoost, GreedyRuleLists, ExtraTrees)
    - market_analysis/regime_ensemble_training (meta-learner: stacker_lgbm_calibrated)
    
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
        """Load base models from regime_models_training artifacts."""
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
            
            # Extract base models from artifacts
            # Structure may vary, try common patterns
            if isinstance(artifacts, dict):
                # Check for 'models' key
                if 'models' in artifacts:
                    self.base_models = artifacts['models']
                # Check for component_result structure
                elif 'component_result' in artifacts:
                    result = artifacts['component_result']
                    if isinstance(result, dict) and 'models' in result:
                        self.base_models = result['models']
                # Check for training_result structure
                elif 'training_result' in artifacts:
                    result = artifacts['training_result']
                    if isinstance(result, dict) and 'models' in result:
                        self.base_models = result['models']
                # Try direct model keys
                else:
                    for key in ['catboost_model', 'greedy_rule_lists_model', 'extratrees_model']:
                        if key in artifacts:
                            model_name = key.replace('_model', '')
                            self.base_models[model_name] = artifacts[key]
            
            if not self.base_models:
                self.logger.warning("⚠️ No base models found in artifacts")
            else:
                self.logger.info(
                    f"✅ Loaded {len(self.base_models)} base models: {list(self.base_models.keys())}"
                )
                
        except FileNotFoundError as e:
            error_msg = f"Regime models file not found: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load base models: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _load_ensemble_model(self) -> None:
        """Load ensemble model (stacker_lgbm_calibrated) from regime_ensemble_training artifacts."""
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
            
            # Extract ensemble model
            if isinstance(artifacts, dict):
                # Try common keys
                for key in ['ensemble_model', 'stacker_lgbm_calibrated', 'meta_model', 'stacker_model']:
                    if key in artifacts:
                        self.ensemble_model = artifacts[key]
                        break
                # Check for component_result structure
                if self.ensemble_model is None and 'component_result' in artifacts:
                    result = artifacts['component_result']
                    if isinstance(result, dict):
                        for key in ['ensemble_model', 'stacker_lgbm_calibrated', 'meta_model']:
                            if key in result:
                                self.ensemble_model = result[key]
                                break
            
            if self.ensemble_model is None:
                self.logger.warning("⚠️ Ensemble model not found in artifacts")
            else:
                self.logger.info("✅ Loaded ensemble model (stacker_lgbm_calibrated)")
                
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
            
            # Get predictions from base models
            base_predictions = {}
            base_probabilities = {}
            
            for model_name, model in self.base_models.items():
                try:
                    pred = model.predict(features)
                    if return_probabilities and hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)
                        base_probabilities[model_name] = proba[0] if len(proba) > 0 else None
                    base_predictions[model_name] = pred[0] if len(pred) > 0 else None
                except Exception as e:
                    self.logger.warning(f"⚠️ Model {model_name} prediction failed: {e}")
                    continue
            
            # Use ensemble model if available, otherwise use voting from base models
            if self.ensemble_model is not None:
                try:
                    # Prepare features for ensemble (should be base model predictions)
                    ensemble_features = np.array([base_predictions.get(name, 0) for name in self.base_models.keys()])
                    ensemble_features = ensemble_features.reshape(1, -1)
                    
                    primary_regime = int(self.ensemble_model.predict(ensemble_features)[0])
                    
                    if return_probabilities and hasattr(self.ensemble_model, 'predict_proba'):
                        ensemble_proba = self.ensemble_model.predict_proba(ensemble_features)[0]
                        regime_probabilities = {
                            f"regime_{i}": float(prob) for i, prob in enumerate(ensemble_proba)
                        }
                    else:
                        # Create probabilities from prediction
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
                    'feature_names': self.feature_names[:10] if self.feature_names else []  # First 10
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
