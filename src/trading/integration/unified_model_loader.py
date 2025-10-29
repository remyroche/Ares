"""
Unified Model Loader

This module provides a comprehensive model loading system that can access models
trained in src/steps/training/ and stored through BaseStep's artifact_management
or standardized_model_manager.

Supports loading:
1. Regime base ML models
2. Regime ensemble ML model
3. Analyst base ML models
4. Analyst ensemble ML model
5. Tactician base ML models
6. Tactician ensemble ML model

Also loads optimized parameters from final_parameters_optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pickle

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.artifact_manager import ArtifactManager
from src.utils.standardized_model_manager import standardized_model_manager, ModelMetadata
from src.training.steps.base_step import BaseStep
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler

logger = system_logger.getChild('UnifiedModelLoader')


class UnifiedModelLoader:
    """
    Unified model loader that can access models from both artifact_manager
    and standardized_model_manager.
    """

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        """
        Initialize unified model loader.

        Args:
            artifact_manager: Optional artifact manager instance.
                             If None, will create one using BaseStep pattern.
        """
        self.logger = logger.getChild('UnifiedModelLoader')
        
        # Initialize artifact manager
        if artifact_manager is None:
            # Create artifact manager following BaseStep pattern
            self.artifact_manager = ArtifactManager(config={})
            # Set context for training artifacts
            self.artifact_manager.set_context(
                step_name='trading',
                category='models_training',
                symbol='all',
                exchange='all'
            )
        else:
            self.artifact_manager = artifact_manager

        # Model cache
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}

        # Optimized parameters cache
        self.optimized_parameters: Dict[str, Any] = {}

        # Step name mappings
        self.step_mappings = {
            'regime_base': 'regime_models_training',
            'regime_ensemble': 'regime_ensemble_training',
            'analyst_base': 'analyst_base_training',
            'analyst_ensemble': 'analyst_ensemble_training',
            'tactician_base': 'tactician_base_training',
            'tactician_ensemble': 'tactician_ensemble_training'
        }

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_regime_base_models(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        Load regime base ML models.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (defaults to 1h for regime models)

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading regime base ML models for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['regime_base']
        models = {}

        try:
            # Try loading from artifact manager first
            artifact_key = f"regime_models_training_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'model')
            
            if artifact_data:
                training_result = artifact_data.get('regime_models_training_result', {})
                model_dict = training_result.get('models', {})
                
                for model_name, model_data in model_dict.items():
                    if isinstance(model_data, dict) and 'model' in model_data:
                        models[model_name] = model_data['model']
                    elif hasattr(model_data, 'predict'):
                        models[model_name] = model_data

            # Also try standardized_model_manager
            available_model_ids = standardized_model_manager.list_available_models(step_name)
            
            for model_id in available_model_ids:
                if model_id not in models:
                    try:
                        model_result = standardized_model_manager.load_model(model_id, step_name)
                        if model_result:
                            model, metadata = model_result
                            models[model_id] = model
                            self.model_metadata[model_id] = metadata
                    except Exception as e:
                        self.logger.debug(f"Could not load regime model {model_id}: {e}")

            # Also try common artifact names
            common_artifact_keys = [
                'regime_catboost_model',
                'regime_extratrees_model',
                'regime_greedy_rule_list_model'
            ]
            
            for artifact_key in common_artifact_keys:
                if artifact_key not in models:
                    artifact_data = self._get_artifact_safe(artifact_key, 'model')
                    if artifact_data:
                        models[artifact_key] = artifact_data

            self.loaded_models.update({f'regime_base_{k}': v for k, v in models.items()})
            
            if models:
                tprint_success(f"✅ Loaded {len(models)} regime base ML models")
            else:
                tprint_warning("⚠️ No regime base models found")

            return models

        except Exception as e:
            raise TradingError(
                f"Failed to load regime base models: {e}",
                error_code="REGIME_BASE_MODELS_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={'symbol': symbol, 'timeframe': timeframe}
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_regime_ensemble_model(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "1h"
    ) -> Optional[Any]:
        """
        Load regime ensemble ML model.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (defaults to 1h for regime models)

        Returns:
            Ensemble model or None
        """
        tprint_info(f"🔄 Loading regime ensemble ML model for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['regime_ensemble']
        model = None

        try:
            # Try loading from artifact manager first
            artifact_key = f"regime_ensemble_training_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'model')
            
            if artifact_data:
                training_result = artifact_data.get('regime_ensemble_training_result', {})
                model = training_result.get('ensemble_model') or training_result.get('stacker_lgbm_calibrated')

            # Also try standardized_model_manager
            if model is None:
                available_model_ids = standardized_model_manager.list_available_models(step_name)
                
                for model_id in available_model_ids:
                    if 'ensemble' in model_id.lower() or 'stacker' in model_id.lower():
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)
                            if model_result:
                                model, metadata = model_result
                                self.model_metadata[model_id] = metadata
                                break
                        except Exception as e:
                            self.logger.debug(f"Could not load regime ensemble model {model_id}: {e}")

            # Also try common artifact names
            if model is None:
                common_artifact_keys = [
                    'regime_ensemble_model',
                    'regime_stacker_model',
                    'regime_stacker_lgbm_calibrated'
                ]
                
                for artifact_key in common_artifact_keys:
                    artifact_data = self._get_artifact_safe(artifact_key, 'model')
                    if artifact_data:
                        model = artifact_data
                        break

            if model:
                self.loaded_models['regime_ensemble'] = model
                tprint_success("✅ Loaded regime ensemble ML model")
            else:
                tprint_warning("⚠️ No regime ensemble model found")

            return model

        except Exception as e:
            raise TradingError(
                f"Failed to load regime ensemble model: {e}",
                error_code="REGIME_ENSEMBLE_MODEL_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={'symbol': symbol, 'timeframe': timeframe}
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_analyst_base_models(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "15m"
    ) -> Dict[str, Any]:
        """
        Load analyst base ML models.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading analyst base ML models for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['analyst_base']
        models = {}

        try:
            # Try loading from artifact manager first
            artifact_key = f"analyst_base_training_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'model')
            
            if artifact_data:
                training_result = artifact_data.get('analyst_base_training_result', {})
                model_dict = training_result.get('models', {})
                
                for model_name, model_data in model_dict.items():
                    if isinstance(model_data, dict) and 'model' in model_data:
                        models[model_name] = model_data['model']
                    elif hasattr(model_data, 'predict'):
                        models[model_name] = model_data

            # Also try standardized_model_manager
            available_model_ids = standardized_model_manager.list_available_models(step_name)
            
            for model_id in available_model_ids:
                if model_id not in models:
                    try:
                        model_result = standardized_model_manager.load_model(model_id, step_name)
                        if model_result:
                            model, metadata = model_result
                            models[model_id] = model
                            self.model_metadata[model_id] = metadata
                    except Exception as e:
                        self.logger.debug(f"Could not load analyst model {model_id}: {e}")

            # Also try common artifact names
            common_artifact_keys = [
                'analyst_base_models',
                'analyst_models',
                'analyst_training_result'
            ]
            
            for artifact_key in common_artifact_keys:
                if artifact_key not in models:
                    artifact_data = self._get_artifact_safe(artifact_key, 'model')
                    if artifact_data:
                        if isinstance(artifact_data, dict):
                            models.update(artifact_data)
                        else:
                            models[artifact_key] = artifact_data

            self.loaded_models.update({f'analyst_base_{k}': v for k, v in models.items()})
            
            if models:
                tprint_success(f"✅ Loaded {len(models)} analyst base ML models")
            else:
                tprint_warning("⚠️ No analyst base models found")

            return models

        except Exception as e:
            raise TradingError(
                f"Failed to load analyst base models: {e}",
                error_code="ANALYST_BASE_MODELS_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={'symbol': symbol, 'timeframe': timeframe}
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_analyst_ensemble_model(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "15m"
    ) -> Optional[Any]:
        """
        Load analyst ensemble ML model.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Ensemble model or None
        """
        tprint_info(f"🔄 Loading analyst ensemble ML model for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['analyst_ensemble']
        model = None

        try:
            # Try loading from artifact manager first
            artifact_key = f"analyst_ensemble_training_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'model')
            
            if artifact_data:
                training_result = artifact_data.get('analyst_ensemble_training_result', {})
                model = training_result.get('ensemble_model') or training_result.get('meta_model')

            # Also try standardized_model_manager
            if model is None:
                available_model_ids = standardized_model_manager.list_available_models(step_name)
                
                for model_id in available_model_ids:
                    if 'ensemble' in model_id.lower() or 'meta' in model_id.lower():
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)
                            if model_result:
                                model, metadata = model_result
                                self.model_metadata[model_id] = metadata
                                break
                        except Exception as e:
                            self.logger.debug(f"Could not load analyst ensemble model {model_id}: {e}")

            # Also try common artifact names
            if model is None:
                common_artifact_keys = [
                    'analyst_ensemble_model',
                    'analyst_meta_model',
                    'analyst_ensemble'
                ]
                
                for artifact_key in common_artifact_keys:
                    artifact_data = self._get_artifact_safe(artifact_key, 'model')
                    if artifact_data:
                        model = artifact_data
                        break

            if model:
                self.loaded_models['analyst_ensemble'] = model
                tprint_success("✅ Loaded analyst ensemble ML model")
            else:
                tprint_warning("⚠️ No analyst ensemble model found")

            return model

        except Exception as e:
            raise TradingError(
                f"Failed to load analyst ensemble model: {e}",
                error_code="ANALYST_ENSEMBLE_MODEL_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={'symbol': symbol, 'timeframe': timeframe}
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_tactician_base_models(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "5m"
    ) -> Dict[str, Any]:
        """
        Load tactician base ML models.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading tactician base ML models for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['tactician_base']
        models = {}

        try:
            # Try loading from artifact manager first
            artifact_key = f"tactician_base_training_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'model')
            
            if artifact_data:
                training_result = artifact_data.get('tactician_base_training_result', {})
                model_dict = training_result.get('models', {})
                
                for model_name, model_data in model_dict.items():
                    if isinstance(model_data, dict) and 'model' in model_data:
                        models[model_name] = model_data['model']
                    elif hasattr(model_data, 'predict'):
                        models[model_name] = model_data

            # Also try standardized_model_manager
            available_model_ids = standardized_model_manager.list_available_models(step_name)
            
            for model_id in available_model_ids:
                if model_id not in models:
                    try:
                        model_result = standardized_model_manager.load_model(model_id, step_name)
                        if model_result:
                            model, metadata = model_result
                            models[model_id] = model
                            self.model_metadata[model_id] = metadata
                    except Exception as e:
                        self.logger.debug(f"Could not load tactician model {model_id}: {e}")

            # Also try common artifact names
            common_artifact_keys = [
                'tactician_base_models',
                'tactician_models',
                'tactician_training_result'
            ]
            
            for artifact_key in common_artifact_keys:
                if artifact_key not in models:
                    artifact_data = self._get_artifact_safe(artifact_key, 'model')
                    if artifact_data:
                        if isinstance(artifact_data, dict):
                            models.update(artifact_data)
                        else:
                            models[artifact_key] = artifact_data

            self.loaded_models.update({f'tactician_base_{k}': v for k, v in models.items()})
            
            if models:
                tprint_success(f"✅ Loaded {len(models)} tactician base ML models")
            else:
                tprint_warning("⚠️ No tactician base models found")

            return models

        except Exception as e:
            raise TradingError(
                f"Failed to load tactician base models: {e}",
                error_code="TACTICIAN_BASE_MODELS_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={'symbol': symbol, 'timeframe': timeframe}
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_tactician_ensemble_model(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "5m"
    ) -> Optional[Any]:
        """
        Load tactician ensemble ML model.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Ensemble model or None
        """
        tprint_info(f"🔄 Loading tactician ensemble ML model for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['tactician_ensemble']
        model = None

        try:
            # Try loading from artifact manager first
            artifact_key = f"tactician_ensemble_training_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'model')
            
            if artifact_data:
                training_result = artifact_data.get('tactician_ensemble_training_result', {})
                model = training_result.get('ensemble_model') or training_result.get('meta_model')

            # Also try standardized_model_manager
            if model is None:
                available_model_ids = standardized_model_manager.list_available_models(step_name)
                
                for model_id in available_model_ids:
                    if 'ensemble' in model_id.lower() or 'meta' in model_id.lower():
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)
                            if model_result:
                                model, metadata = model_result
                                self.model_metadata[model_id] = metadata
                                break
                        except Exception as e:
                            self.logger.debug(f"Could not load tactician ensemble model {model_id}: {e}")

            # Also try common artifact names
            if model is None:
                common_artifact_keys = [
                    'tactician_ensemble_model',
                    'tactician_meta_model',
                    'tactician_ensemble'
                ]
                
                for artifact_key in common_artifact_keys:
                    artifact_data = self._get_artifact_safe(artifact_key, 'model')
                    if artifact_data:
                        model = artifact_data
                        break

            if model:
                self.loaded_models['tactician_ensemble'] = model
                tprint_success("✅ Loaded tactician ensemble ML model")
            else:
                tprint_warning("⚠️ No tactician ensemble model found")

            return model

        except Exception as e:
            raise TradingError(
                f"Failed to load tactician ensemble model: {e}",
                error_code="TACTICIAN_ENSEMBLE_MODEL_LOAD_ERROR",
                severity=TradingErrorSeverity.HIGH,
                context={'symbol': symbol, 'timeframe': timeframe}
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def load_optimized_parameters(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "15m"
    ) -> Dict[str, Any]:
        """
        Load optimized parameters from final_parameters_optimization.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary of optimized parameters
        """
        tprint_info(f"🔄 Loading optimized parameters for {symbol} ({timeframe})...")
        
        parameters = {}

        try:
            # Try loading from artifact manager first
            artifact_key = "final_parameters_optimization_result"
            artifact_data = self._get_artifact_safe(artifact_key, 'data')
            
            if artifact_data:
                optimization_result = artifact_data.get('final_parameters_optimization_result', {})
                parameters = optimization_result.get('optimized_parameters', {})
                
                if parameters:
                    self.optimized_parameters = parameters
                    tprint_success(f"✅ Loaded optimized parameters from artifacts")
                    return parameters

            # Try loading from file path
            optimization_dir = Path("data_cache/optimization")
            results_file = optimization_dir / f"binance_{symbol}_final_parameters.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    parameters = json.load(f)
                    self.optimized_parameters = parameters
                    tprint_success(f"✅ Loaded optimized parameters from {results_file}")
                    return parameters

            # Try loading from pickle file
            pickle_file = optimization_dir / f"binance_{symbol}_final_parameters.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    parameters = pickle.load(f)
                    if isinstance(parameters, dict):
                        self.optimized_parameters = parameters
                        tprint_success(f"✅ Loaded optimized parameters from {pickle_file}")
                        return parameters

            # Try loading from final_parameters_optimization step artifacts
            step_name = "final_parameters_optimization"
            artifact_data = self._get_artifact_safe(f"{step_name}_result", 'data')
            
            if artifact_data:
                optimization_result = artifact_data.get('optimization_result', {})
                parameters = optimization_result.get('optimized_parameters', {})
                
                if parameters:
                    self.optimized_parameters = parameters
                    tprint_success("✅ Loaded optimized parameters from step artifacts")
                    return parameters

            tprint_warning("⚠️ No optimized parameters found, using defaults")
            return self._get_default_parameters()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load optimized parameters: {e}")
            return self._get_default_parameters()

    def _get_artifact_safe(self, artifact_key: str, artifact_type: str = 'data') -> Optional[Any]:
        """
        Safely get artifact from artifact manager with fallback logic.

        Args:
            artifact_key: Artifact key
            artifact_type: Artifact type

        Returns:
            Artifact data or None
        """
        try:
            # Try different step contexts
            step_contexts = [
                ('models_training', 'model'),
                ('backtesting', 'data'),
                ('pre_training', 'data'),
                ('market_analysis', 'model')
            ]
            
            for category, default_type in step_contexts:
                try:
                    self.artifact_manager.set_context(
                        step_name=category,
                        category=category,
                        symbol='all',
                        exchange='all'
                    )
                    artifact = self.artifact_manager.get_artifact(
                        artifact_key,
                        artifact_type or default_type
                    )
                    if artifact is not None:
                        return artifact
                except Exception:
                    continue
            
            # Try BaseStep pattern
            try:
                # Reset to default context
                self.artifact_manager.set_context(
                    step_name='trading',
                    category='models_training',
                    symbol='all',
                    exchange='all'
                )
                artifact = self.artifact_manager.get_artifact(artifact_key, artifact_type)
                if artifact is not None:
                    return artifact
            except Exception:
                pass

            return None

        except Exception as e:
            self.logger.debug(f"Could not get artifact {artifact_key}: {e}")
            return None

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters if optimized ones are not available."""
        return {
            'confidence_threshold': 0.75,
            'position_sizing_factor': 0.02,
            'leverage_multiplier': 1.5,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06,
            'ensemble_weight_analyst': 0.6,
            'ensemble_weight_tactician': 0.4,
            'analyst_confidence_weight': 0.6,
            'tactician_confidence_weight': 0.4,
            'regime_confidence_threshold': 0.7,
            'signal_confidence_threshold': 0.6,
            'exit_confidence_threshold': 0.5,
            'tactician_exit_confidence_weight': 0.6,
            'analyst_exit_confidence_weight': 0.4,
            'exit_confidence_combination_method': 'multiplicative'
        }

    async def load_all_models(
        self,
        symbol: str = "ETHUSDT",
        analyst_timeframe: str = "15m",
        tactician_timeframe: str = "5m",
        regime_timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        Load all models and optimized parameters.

        Returns:
            Dictionary containing all loaded models and parameters
        """
        tprint_info("🚀 Loading all trained models and optimized parameters...")

        result = {
            'regime_base_models': await self.load_regime_base_models(symbol, regime_timeframe),
            'regime_ensemble_model': await self.load_regime_ensemble_model(symbol, regime_timeframe),
            'analyst_base_models': await self.load_analyst_base_models(symbol, analyst_timeframe),
            'analyst_ensemble_model': await self.load_analyst_ensemble_model(symbol, analyst_timeframe),
            'tactician_base_models': await self.load_tactician_base_models(symbol, tactician_timeframe),
            'tactician_ensemble_model': await self.load_tactician_ensemble_model(symbol, tactician_timeframe),
            'optimized_parameters': await self.load_optimized_parameters(symbol, analyst_timeframe)
        }

        tprint_success("✅ All models and parameters loaded successfully")
        return result

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a loaded model by ID."""
        return self.loaded_models.get(model_id)

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a loaded model."""
        return self.model_metadata.get(model_id)

    def get_optimized_parameters(self) -> Dict[str, Any]:
        """Get loaded optimized parameters."""
        return self.optimized_parameters.copy() if self.optimized_parameters else self._get_default_parameters()


# Global instance
_unified_model_loader: Optional[UnifiedModelLoader] = None


def get_unified_model_loader(artifact_manager: Optional[ArtifactManager] = None) -> UnifiedModelLoader:
    """Get or create global unified model loader instance."""
    global _unified_model_loader
    
    if _unified_model_loader is None:
        _unified_model_loader = UnifiedModelLoader(artifact_manager)
    
    return _unified_model_loader
