"""
Model Integration

Integration utilities for loading and using models trained
in the training pipeline within trading operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import pickle

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.standardized_model_manager import standardized_model_manager, ModelMetadata
from .unified_model_loader import get_unified_model_loader, UnifiedModelLoader
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.validation import validate_trading_config

logger = system_logger.getChild('ModelIntegration')

class TrainingModelLoader:
    """
    Loads and manages models trained in the training pipeline
    for use in trading operations.
    """

    def __init__(self):
        self.logger = logger.getChild('TrainingModelLoader')
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        # Use unified model loader for accessing models from artifact_manager
        self.unified_loader = get_unified_model_loader()

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_analyst_models(
        self,
        model_ids: Optional[List[str]] = None,
        step_name: str = "analyst_base_training",
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load trained analyst models from the training pipeline.

        Args:
            model_ids: Specific model IDs to load (if None, loads all available)
            step_name: Training step name where models were saved
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading analyst models from step: {step_name}")

        loaded_models = {}

        try:
            # First try unified model loader (accesses artifact_manager and standardized_model_manager)
            base_models = await self.unified_loader.load_analyst_base_models(symbol, exchange, timeframe, direction)
            ensemble_model = await self.unified_loader.load_analyst_ensemble_model(symbol, exchange, timeframe, direction)
            
            loaded_models.update(base_models)
            if ensemble_model:
                loaded_models['analyst_ensemble'] = ensemble_model

            # Also try standardized model manager for specific model IDs
            if model_ids:
                for model_id in model_ids:
                    if model_id not in loaded_models:
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)

                            if model_result:
                                model, metadata = model_result
                                loaded_models[model_id] = model
                                self.model_metadata[model_id] = metadata

                                tprint_success(f"✅ Loaded analyst model: {model_id}")

                                # Validate model compatibility
                                await self._validate_model_compatibility(model, metadata, "analyst")

                        except Exception as e:
                            tprint_warning(f"⚠️ Could not load analyst model {model_id}: {e}")

            self.loaded_models.update(loaded_models)

            tprint_success(f"✅ Successfully loaded {len(loaded_models)} analyst models")
            return loaded_models

        except Exception as e:
            raise TradingError(
                f"Failed to load analyst models: {e}",
                error_code="ANALYST_MODEL_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={
                    'step_name': step_name,
                    'model_ids': model_ids,
                    'loaded_count': len(loaded_models)
                }
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_tactician_models(
        self,
        model_ids: Optional[List[str]] = None,
        step_name: str = "tactician_base_training",
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "5m",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load trained tactician models from the training pipeline.

        Args:
            model_ids: Specific model IDs to load (if None, loads all available)
            step_name: Training step name where models were saved
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading tactician models from step: {step_name}")

        loaded_models = {}

        try:
            # First try unified model loader (accesses artifact_manager and standardized_model_manager)
            base_models = await self.unified_loader.load_tactician_base_models(symbol, exchange, timeframe, direction)
            ensemble_model = await self.unified_loader.load_tactician_ensemble_model(symbol, exchange, timeframe, direction)
            
            loaded_models.update(base_models)
            if ensemble_model:
                loaded_models['tactician_ensemble'] = ensemble_model

            # Also try standardized model manager for specific model IDs
            if model_ids:
                for model_id in model_ids:
                    if model_id not in loaded_models:
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)

                            if model_result:
                                model, metadata = model_result
                                loaded_models[model_id] = model
                                self.model_metadata[model_id] = metadata

                                tprint_success(f"✅ Loaded tactician model: {model_id}")

                                # Validate model compatibility
                                await self._validate_model_compatibility(model, metadata, "tactician")

                        except Exception as e:
                            tprint_warning(f"⚠️ Could not load tactician model {model_id}: {e}")

            self.loaded_models.update(loaded_models)

            tprint_success(f"✅ Successfully loaded {len(loaded_models)} tactician models")
            return loaded_models

        except Exception as e:
            raise TradingError(
                f"Failed to load tactician models: {e}",
                error_code="TACTICIAN_MODEL_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={
                    'step_name': step_name,
                    'model_ids': model_ids,
                    'loaded_count': len(loaded_models)
                }
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_hmm_models(
        self,
        model_ids: Optional[List[str]] = None,
        step_name: str = "enhanced_hmm_training"
    ) -> Dict[str, Any]:
        """
        Load trained HMM models from the training pipeline.

        Args:
            model_ids: Specific model IDs to load (if None, loads all available)
            step_name: Training step name where models were saved

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading HMM models from step: {step_name}")

        loaded_models = {}

        try:
            # Get available models from standardized model manager
            if model_ids is None:
                # Load all HMM models from the step
                model_ids = await self._discover_models(step_name, "hmm")

            for model_id in model_ids:
                try:
                    model_result = standardized_model_manager.load_model(model_id, step_name)

                    if model_result:
                        model, metadata = model_result
                        loaded_models[model_id] = model
                        self.model_metadata[model_id] = metadata

                        tprint_success(f"✅ Loaded HMM model: {model_id}")

                        # Validate model compatibility
                        await self._validate_model_compatibility(model, metadata, "hmm")

                    else:
                        tprint_warning(f"⚠️ Could not load HMM model: {model_id}")

                except Exception as e:
                    tprint_error(f"❌ Failed to load HMM model {model_id}: {e}")
                    continue

            self.loaded_models.update(loaded_models)

            tprint_success(f"✅ Successfully loaded {len(loaded_models)} HMM models")
            return loaded_models

        except Exception as e:
            raise TradingError(
                f"Failed to load HMM models: {e}",
                error_code="HMM_MODEL_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={
                    'step_name': step_name,
                    'model_ids': model_ids,
                    'loaded_count': len(loaded_models)
                }
            )

    async def _discover_models(self, step_name: str, model_type: str) -> List[str]:
        """Discover available models for a given step and type."""
        try:
            # This would integrate with the standardized model manager
            # to discover available models

            # For now, return common model IDs based on training pipeline
            if model_type == "analyst":
                return [
                    f"analyst_model_1_{step_name}",
                    f"analyst_model_2_{step_name}",
                    f"analyst_model_3_{step_name}",
                    f"analyst_ensemble_{step_name}"
                ]
            elif model_type == "tactician":
                return [
                    f"tactician_model_1_{step_name}",
                    f"tactician_model_2_{step_name}",
                    f"tactician_model_3_{step_name}",
                    f"tactician_ensemble_{step_name}"
                ]
            elif model_type == "hmm":
                return [
                    f"hmm_model_main_{step_name}",
                    f"hmm_model_regime_{step_name}",
                    f"hmm_model_enhanced_{step_name}"
                ]
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Model discovery failed for {step_name}/{model_type}: {e}")
            return []

    async def _validate_model_compatibility(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_type: str
    ):
        """Validate that a loaded model is compatible with trading operations."""
        try:
            # Check if model has required methods
            required_methods = {
                "analyst": ["predict", "predict_proba"],
                "tactician": ["predict", "predict_proba"],
                "hmm": ["predict", "score"]
            }

            methods_to_check = required_methods.get(model_type, ["predict"])

            for method in methods_to_check:
                if not hasattr(model, method):
                    tprint_warning(f"⚠️ Model {metadata.model_id} missing method: {method}")

            # Check model features compatibility
            if metadata.features:
                tprint_info(f"📊 Model {metadata.model_id} trained on {len(metadata.features)} features")

            # Check model metrics
            if metadata.metrics:
                tprint_info(f"📈 Model {metadata.model_id} performance: {metadata.metrics}")

        except Exception as e:
            tprint_warning(f"⚠️ Model compatibility validation failed: {e}")

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a loaded model by ID."""
        return self.loaded_models.get(model_id)

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a loaded model."""
        return self.model_metadata.get(model_id)

    def list_loaded_models(self) -> List[str]:
        """List all loaded model IDs."""
        return list(self.loaded_models.keys())

    def get_models_by_type(self, model_type: str) -> Dict[str, Any]:
        """Get all models of a specific type."""
        return {
            model_id: model
            for model_id, model in self.loaded_models.items()
            if model_type.lower() in model_id.lower()
        }

# Global model loader instance
training_model_loader = TrainingModelLoader()

# Convenience functions
async def load_trained_models(
    analyst_models: bool = True,
    tactician_models: bool = True,
    hmm_models: bool = True,
    model_ids: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load all trained models for trading operations.

    Args:
        analyst_models: Whether to load analyst models
        tactician_models: Whether to load tactician models
        hmm_models: Whether to load HMM models
        model_ids: Specific model IDs to load for each type

    Returns:
        Dictionary with loaded models by type
    """
    tprint_info("🚀 Loading trained models for trading operations...")

    loaded_models = {}

    try:
        if analyst_models:
            analyst_ids = model_ids.get('analyst') if model_ids else None
            loaded_models['analyst'] = await training_model_loader.load_analyst_models(analyst_ids)

        if tactician_models:
            tactician_ids = model_ids.get('tactician') if model_ids else None
            loaded_models['tactician'] = await training_model_loader.load_tactician_models(tactician_ids)

        if hmm_models:
            hmm_ids = model_ids.get('hmm') if model_ids else None
            loaded_models['hmm'] = await training_model_loader.load_hmm_models(hmm_ids)

        total_models = sum(len(models) for models in loaded_models.values())
        tprint_success(f"✅ Successfully loaded {total_models} trained models")

        return loaded_models

    except Exception as e:
        tprint_error(f"❌ Failed to load trained models: {e}")
        raise

async def validate_model_compatibility(
    models: Dict[str, Any],
    trading_config: Dict[str, Any]
) -> bool:
    """
    Validate that loaded models are compatible with trading configuration.

    Args:
        models: Dictionary of loaded models
        trading_config: Trading configuration

    Returns:
        True if all models are compatible
    """
    tprint_info("🔍 Validating model compatibility with trading configuration...")

    try:
        # Validate trading config first
        validate_trading_config(trading_config)

        # Check model compatibility
        for model_type, model_dict in models.items():
            for model_id, model in model_dict.items():
                # Get model metadata
                metadata = training_model_loader.get_model_metadata(model_id)

                if metadata:
                    # Validate model with trading config
                    await training_model_loader._validate_model_compatibility(
                        model, metadata, model_type
                    )

        tprint_success("✅ All models are compatible with trading configuration")
        return True

    except Exception as e:
        tprint_error(f"❌ Model compatibility validation failed: {e}")
        return False
