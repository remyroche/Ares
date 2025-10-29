"""
Unified Model Loader

This module provides a comprehensive model loading system that can access models
trained in src/steps/training/ and stored through BaseStep's artifact_management
or standardized_model_manager.

Supports loading:
1. Regime base ML models
2. Regime ensemble ML model
3. Analyst base ML models (dispatched to Analyst)
4. Analyst ensemble ML model (dispatched to Analyst)
5. Tactician base ML models (dispatched to Tactician)
6. Tactician ensemble ML model (dispatched to Tactician)

Also loads optimized parameters from final_parameters_optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pickle
import os
from datetime import datetime

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
    and standardized_model_manager with proper context filtering and timestamp selection.
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

    def _find_most_recent_artifact_file(
        self,
        artifact_key: str,
        step_name: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        direction: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Optional[Path]:
        """
        Find the most recent artifact file matching the context criteria.
        Uses file modification time to determine the most recent.

        Args:
            artifact_key: Artifact key to search for
            step_name: Step name
            symbol: Symbol filter
            exchange: Exchange filter
            timeframe: Timeframe filter
            direction: Direction filter
            model_type: Model type filter ('Analyst' or 'Tactician')

        Returns:
            Path to most recent artifact file or None
        """
        try:
            step_category = self._get_step_category(step_name)
            
            # Build search path with context
            search_parts = ['artifacts', step_category]
            if symbol:
                search_parts.append(symbol)
            if exchange:
                search_parts.append(exchange)
            if direction:
                search_parts.append(direction)
            if model_type:
                search_parts.append(model_type)
            search_parts.append(step_name)
            
            search_base = Path(*search_parts)
            
            if not search_base.exists():
                return None
            
            # Find all files matching artifact_key pattern
            matching_files = []
            for file_path in search_base.rglob(f"*{artifact_key}*"):
                if file_path.is_file():
                    # Check file extension (prefer .pkl, .joblib, .parquet)
                    if file_path.suffix.lower() in ['.pkl', '.joblib', '.parquet', '.json']:
                        matching_files.append(file_path)
            
            if not matching_files:
                return None
            
            # Sort by modification time (most recent first)
            matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            most_recent = matching_files[0]
            self.logger.info(
                f"✅ Found {len(matching_files)} artifacts for {artifact_key}, "
                f"using most recent: {most_recent.name} "
                f"(timestamp: {datetime.fromtimestamp(most_recent.stat().st_mtime)})"
            )
            
            return most_recent
            
        except Exception as e:
            self.logger.debug(f"Could not find artifact file {artifact_key}: {e}")
            return None

    def _get_artifact_with_context(
        self,
        artifact_key: str,
        step_name: str,
        artifact_type: str = 'model',
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        direction: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get artifact with proper context filtering.
        When multiple artifacts exist, selects the most recent one by timestamp.

        Args:
            artifact_key: Artifact key
            step_name: Step name where artifact was saved
            artifact_type: Artifact type ('model', 'data', etc.)
            symbol: Trading symbol filter
            exchange: Exchange filter
            timeframe: Timeframe filter
            direction: Direction filter ('long', 'short', 'both')
            model_type: Model type filter ('Analyst', 'Tactician', etc.)

        Returns:
            Artifact data or None
        """
        try:
            # Set context before calling get_artifact
            self.artifact_manager.set_context(
                step_name=step_name,
                symbol=symbol or 'all',
                exchange=exchange or 'all',
                direction=direction or 'long',
                model=model_type or 'Analyst'
            )
            
            # Try to find most recent artifact file first
            artifact_file = self._find_most_recent_artifact_file(
                artifact_key, step_name, symbol, exchange, timeframe, direction, model_type
            )
            
            if artifact_file:
                # Load from file directly
                return self._load_artifact_from_path(artifact_file)
            
            # Fallback: try artifact_manager.get_artifact()
            try:
                artifact_data = self.artifact_manager.get_artifact(artifact_key, artifact_type)
                if artifact_data is not None:
                    return artifact_data
            except Exception as e:
                self.logger.debug(f"Could not load via artifact_manager: {e}")
            
            # Fallback: try with relaxed context
            for relax_symbol in ([symbol] if symbol else [None, 'all']):
                for relax_exchange in ([exchange] if exchange else [None, 'all']):
                    for relax_direction in ([direction] if direction else ['long', 'short', 'both']):
                        for relax_model in ([model_type] if model_type else ['Analyst', 'Tactician']):
                            try:
                                self.artifact_manager.set_context(
                                    step_name=step_name,
                                    symbol=relax_symbol or 'all',
                                    exchange=relax_exchange or 'all',
                                    direction=relax_direction or 'long',
                                    model=relax_model or 'Analyst'
                                )
                                
                                artifact = self.artifact_manager.get_artifact(artifact_key, artifact_type)
                                if artifact is not None:
                                    self.logger.info(f"✅ Found artifact {artifact_key} with relaxed context")
                                    return artifact
                            except Exception:
                                continue
            
            return None

        except Exception as e:
            self.logger.debug(f"Could not get artifact {artifact_key}: {e}")
            return None

    def _load_artifact_from_path(self, path: Path) -> Optional[Any]:
        """Load artifact from file path."""
        try:
            import pandas as pd
            
            if path.suffix == '.parquet':
                return pd.read_parquet(path)
            elif path.suffix == '.csv':
                return pd.read_csv(path, index_col=0)
            elif path.suffix == '.pkl':
                with open(path, 'rb') as f:
                    return pickle.load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.suffix == '.joblib':
                import joblib
                return joblib.load(path)
            else:
                self.logger.warning(f"Unknown file extension: {path.suffix}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load artifact from {path}: {e}")
            return None

    def _get_step_category(self, step_name: str) -> str:
        """Determine step category from step name."""
        step_name_lower = step_name.lower()
        
        if any(x in step_name_lower for x in ['regime', 'hmm', 'clustering']):
            return 'market_analysis'
        elif any(x in step_name_lower for x in ['analyst', 'tactician', 'model']):
            return 'models_training'
        elif any(x in step_name_lower for x in ['backtest', 'parameter', 'optimization']):
            return 'backtesting'
        elif any(x in step_name_lower for x in ['feature', 'pre_training']):
            return 'pre_training'
        else:
            return 'models_training'  # Default

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def load_regime_base_models(
        self,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "1h",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load regime base ML models.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (defaults to 1h for regime models)
            direction: Trading direction

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading regime base ML models for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['regime_base']
        models = {}

        try:
            # Try loading from artifact manager with proper context
            artifact_key = "regime_models_training_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='model',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model_type=None  # Regime models don't have Analyst/Tactician distinction
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    training_result = artifact_data.get('regime_models_training_result', artifact_data)
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

            # Try common artifact names
            common_artifact_keys = [
                'regime_catboost_model',
                'regime_extratrees_model',
                'regime_greedy_rule_list_model'
            ]
            
            for artifact_key in common_artifact_keys:
                if artifact_key not in models:
                    artifact_data = self._get_artifact_with_context(
                        artifact_key=artifact_key,
                        step_name=step_name,
                        artifact_type='model',
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction
                    )
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
        exchange: str = "binance",
        timeframe: str = "1h",
        direction: str = "long"
    ) -> Optional[Any]:
        """
        Load regime ensemble ML model.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (defaults to 1h for regime models)
            direction: Trading direction

        Returns:
            Ensemble model or None
        """
        tprint_info(f"🔄 Loading regime ensemble ML model for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['regime_ensemble']
        model = None

        try:
            # Try loading from artifact manager with proper context
            artifact_key = "regime_ensemble_training_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='model',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    training_result = artifact_data.get('regime_ensemble_training_result', artifact_data)
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

            # Try common artifact names
            if model is None:
                common_artifact_keys = [
                    'regime_ensemble_model',
                    'regime_stacker_model',
                    'regime_stacker_lgbm_calibrated'
                ]
                
                for artifact_key in common_artifact_keys:
                    artifact_data = self._get_artifact_with_context(
                        artifact_key=artifact_key,
                        step_name=step_name,
                        artifact_type='model',
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction
                    )
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
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load analyst base ML models (dispatched to Analyst components).

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading analyst base ML models for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['analyst_base']
        models = {}

        try:
            # Try loading from artifact manager with proper context (Analyst models)
            artifact_key = "analyst_base_training_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='model',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model_type='Analyst'  # Explicitly filter for Analyst models
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    training_result = artifact_data.get('analyst_base_training_result', artifact_data)
                    model_dict = training_result.get('models', {})
                    
                    for model_name, model_data in model_dict.items():
                        if isinstance(model_data, dict) and 'model' in model_data:
                            models[model_name] = model_data['model']
                        elif hasattr(model_data, 'predict'):
                            models[model_name] = model_data

            # Also try standardized_model_manager
            available_model_ids = standardized_model_manager.list_available_models(step_name)
            
            for model_id in available_model_ids:
                if model_id not in models and 'analyst' in model_id.lower():
                    try:
                        model_result = standardized_model_manager.load_model(model_id, step_name)
                        if model_result:
                            model, metadata = model_result
                            models[model_id] = model
                            self.model_metadata[model_id] = metadata
                    except Exception as e:
                        self.logger.debug(f"Could not load analyst model {model_id}: {e}")

            # Try common artifact names
            common_artifact_keys = [
                'analyst_base_models',
                'analyst_models',
                'analyst_training_result'
            ]
            
            for artifact_key in common_artifact_keys:
                if artifact_key not in models:
                    artifact_data = self._get_artifact_with_context(
                        artifact_key=artifact_key,
                        step_name=step_name,
                        artifact_type='model',
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        model_type='Analyst'  # Explicitly filter for Analyst
                    )
                    if artifact_data:
                        if isinstance(artifact_data, dict):
                            models.update(artifact_data)
                        else:
                            models[artifact_key] = artifact_data

            self.loaded_models.update({f'analyst_base_{k}': v for k, v in models.items()})
            
            if models:
                tprint_success(f"✅ Loaded {len(models)} analyst base ML models (dispatched to Analyst)")
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
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ) -> Optional[Any]:
        """
        Load analyst ensemble ML model (dispatched to Analyst components).

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction

        Returns:
            Ensemble model or None
        """
        tprint_info(f"🔄 Loading analyst ensemble ML model for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['analyst_ensemble']
        model = None

        try:
            # Try loading from artifact manager with proper context (Analyst models)
            artifact_key = "analyst_ensemble_training_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='model',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model_type='Analyst'  # Explicitly filter for Analyst models
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    training_result = artifact_data.get('analyst_ensemble_training_result', artifact_data)
                    model = training_result.get('ensemble_model') or training_result.get('meta_model')

            # Also try standardized_model_manager
            if model is None:
                available_model_ids = standardized_model_manager.list_available_models(step_name)
                
                for model_id in available_model_ids:
                    if ('ensemble' in model_id.lower() or 'meta' in model_id.lower()) and 'analyst' in model_id.lower():
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)
                            if model_result:
                                model, metadata = model_result
                                self.model_metadata[model_id] = metadata
                                break
                        except Exception as e:
                            self.logger.debug(f"Could not load analyst ensemble model {model_id}: {e}")

            # Try common artifact names
            if model is None:
                common_artifact_keys = [
                    'analyst_ensemble_model',
                    'analyst_meta_model',
                    'analyst_ensemble'
                ]
                
                for artifact_key in common_artifact_keys:
                    artifact_data = self._get_artifact_with_context(
                        artifact_key=artifact_key,
                        step_name=step_name,
                        artifact_type='model',
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        model_type='Analyst'  # Explicitly filter for Analyst
                    )
                    if artifact_data:
                        model = artifact_data
                        break

            if model:
                self.loaded_models['analyst_ensemble'] = model
                tprint_success("✅ Loaded analyst ensemble ML model (dispatched to Analyst)")
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
        exchange: str = "binance",
        timeframe: str = "5m",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load tactician base ML models (dispatched to Tactician components).

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction

        Returns:
            Dictionary of loaded models
        """
        tprint_info(f"🔄 Loading tactician base ML models for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['tactician_base']
        models = {}

        try:
            # Try loading from artifact manager with proper context (Tactician models)
            artifact_key = "tactician_base_training_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='model',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model_type='Tactician'  # Explicitly filter for Tactician models
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    training_result = artifact_data.get('tactician_base_training_result', artifact_data)
                    model_dict = training_result.get('models', {})
                    
                    for model_name, model_data in model_dict.items():
                        if isinstance(model_data, dict) and 'model' in model_data:
                            models[model_name] = model_data['model']
                        elif hasattr(model_data, 'predict'):
                            models[model_name] = model_data

            # Also try standardized_model_manager
            available_model_ids = standardized_model_manager.list_available_models(step_name)
            
            for model_id in available_model_ids:
                if model_id not in models and 'tactician' in model_id.lower():
                    try:
                        model_result = standardized_model_manager.load_model(model_id, step_name)
                        if model_result:
                            model, metadata = model_result
                            models[model_id] = model
                            self.model_metadata[model_id] = metadata
                    except Exception as e:
                        self.logger.debug(f"Could not load tactician model {model_id}: {e}")

            # Try common artifact names
            common_artifact_keys = [
                'tactician_base_models',
                'tactician_models',
                'tactician_training_result'
            ]
            
            for artifact_key in common_artifact_keys:
                if artifact_key not in models:
                    artifact_data = self._get_artifact_with_context(
                        artifact_key=artifact_key,
                        step_name=step_name,
                        artifact_type='model',
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        model_type='Tactician'  # Explicitly filter for Tactician
                    )
                    if artifact_data:
                        if isinstance(artifact_data, dict):
                            models.update(artifact_data)
                        else:
                            models[artifact_key] = artifact_data

            self.loaded_models.update({f'tactician_base_{k}': v for k, v in models.items()})
            
            if models:
                tprint_success(f"✅ Loaded {len(models)} tactician base ML models (dispatched to Tactician)")
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
        exchange: str = "binance",
        timeframe: str = "5m",
        direction: str = "long"
    ) -> Optional[Any]:
        """
        Load tactician ensemble ML model (dispatched to Tactician components).

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction

        Returns:
            Ensemble model or None
        """
        tprint_info(f"🔄 Loading tactician ensemble ML model for {symbol} ({timeframe})...")
        
        step_name = self.step_mappings['tactician_ensemble']
        model = None

        try:
            # Try loading from artifact manager with proper context (Tactician models)
            artifact_key = "tactician_ensemble_training_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='model',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model_type='Tactician'  # Explicitly filter for Tactician models
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    training_result = artifact_data.get('tactician_ensemble_training_result', artifact_data)
                    model = training_result.get('ensemble_model') or training_result.get('meta_model')

            # Also try standardized_model_manager
            if model is None:
                available_model_ids = standardized_model_manager.list_available_models(step_name)
                
                for model_id in available_model_ids:
                    if ('ensemble' in model_id.lower() or 'meta' in model_id.lower()) and 'tactician' in model_id.lower():
                        try:
                            model_result = standardized_model_manager.load_model(model_id, step_name)
                            if model_result:
                                model, metadata = model_result
                                self.model_metadata[model_id] = metadata
                                break
                        except Exception as e:
                            self.logger.debug(f"Could not load tactician ensemble model {model_id}: {e}")

            # Try common artifact names
            if model is None:
                common_artifact_keys = [
                    'tactician_ensemble_model',
                    'tactician_meta_model',
                    'tactician_ensemble'
                ]
                
                for artifact_key in common_artifact_keys:
                    artifact_data = self._get_artifact_with_context(
                        artifact_key=artifact_key,
                        step_name=step_name,
                        artifact_type='model',
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        model_type='Tactician'  # Explicitly filter for Tactician
                    )
                    if artifact_data:
                        model = artifact_data
                        break

            if model:
                self.loaded_models['tactician_ensemble'] = model
                tprint_success("✅ Loaded tactician ensemble ML model (dispatched to Tactician)")
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
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load optimized parameters from final_parameters_optimization with proper context.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction

        Returns:
            Dictionary of optimized parameters
        """
        tprint_info(f"🔄 Loading optimized parameters for {symbol} ({timeframe})...")
        
        parameters = {}

        try:
            step_name = "final_parameters_optimization"
            
            # Try loading from artifact manager with proper context
            artifact_key = "final_parameters_optimization_result"
            artifact_data = self._get_artifact_with_context(
                artifact_key=artifact_key,
                step_name=step_name,
                artifact_type='data',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction
            )
            
            if artifact_data:
                if isinstance(artifact_data, dict):
                    optimization_result = artifact_data.get('final_parameters_optimization_result', artifact_data)
                    parameters = optimization_result.get('optimized_parameters', {})
                    
                    if parameters:
                        self.optimized_parameters = parameters
                        tprint_success(f"✅ Loaded optimized parameters from artifacts")
                        return parameters

            # Try loading from file path
            optimization_dir = Path("data_cache/optimization")
            results_file = optimization_dir / f"{exchange}_{symbol}_final_parameters.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    parameters = json.load(f)
                    if isinstance(parameters, dict):
                        self.optimized_parameters = parameters
                        tprint_success(f"✅ Loaded optimized parameters from {results_file}")
                        return parameters

            # Try loading from pickle file
            pickle_file = optimization_dir / f"{exchange}_{symbol}_final_parameters.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    parameters = pickle.load(f)
                    if isinstance(parameters, dict):
                        self.optimized_parameters = parameters
                        tprint_success(f"✅ Loaded optimized parameters from {pickle_file}")
                        return parameters

            tprint_warning("⚠️ No optimized parameters found, using defaults")
            return self._get_default_parameters()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load optimized parameters: {e}")
            return self._get_default_parameters()

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
        exchange: str = "binance",
        analyst_timeframe: str = "15m",
        tactician_timeframe: str = "5m",
        regime_timeframe: str = "1h",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Load all models and optimized parameters with proper context.

        Returns:
            Dictionary containing all loaded models and parameters
        """
        tprint_info("🚀 Loading all trained models and optimized parameters...")

        result = {
            'regime_base_models': await self.load_regime_base_models(symbol, exchange, regime_timeframe, direction),
            'regime_ensemble_model': await self.load_regime_ensemble_model(symbol, exchange, regime_timeframe, direction),
            'analyst_base_models': await self.load_analyst_base_models(symbol, exchange, analyst_timeframe, direction),
            'analyst_ensemble_model': await self.load_analyst_ensemble_model(symbol, exchange, analyst_timeframe, direction),
            'tactician_base_models': await self.load_tactician_base_models(symbol, exchange, tactician_timeframe, direction),
            'tactician_ensemble_model': await self.load_tactician_ensemble_model(symbol, exchange, tactician_timeframe, direction),
            'optimized_parameters': await self.load_optimized_parameters(symbol, exchange, analyst_timeframe, direction)
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
