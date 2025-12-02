"""
Unified Model Loader

This module provides a comprehensive model loading system that can access models
trained in src/steps/training/ and stored through BaseStep's artifact_management
or standardized_model_manager.

Supports loading:
1. Specialist models (Risk, SMC, Liquidity, Breakout, Meso, Macro, Mean Reversion)
2. Analyst Base models (Generic loading of all base models)
3. Legacy Regime/Tactician models (for backward compatibility where needed)

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
from src.utils.tprint import tprint
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

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None) -> None:
        """
        Initialize unified model loader.

        Args:
            artifact_manager: Optional artifact manager instance.
                             If None, will create one using BaseStep pattern.
        """
        self.logger = logger.getChild('UnifiedModelLoader')
        tprint("🔄 Initializing unified model loader")
        
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
        self.step_mappings: Dict[str, str] = {
            'regime_base': 'regime_models_training',
            'regime_ensemble': 'regime_ensemble_training',
            'analyst_base': 'analyst_base_training',
            'analyst_ensemble': 'analyst_ensemble_training',
            'tactician_base': 'tactician_base_training',
            'tactician_ensemble': 'tactician_ensemble_training',
            # Specialists
            'risk': 'ml_risk_regime_step',
            'smc': 'ml_smc_regime_step',
            'liquidity': 'ml_liquidity_regime_step',
            'breakout': 'ml_breakout_bounce_regime_step',
            'meso': 'xgb_meso_regime_step',
            'macro': 'hmm_macro_regime',
            'reversion': 'ml_mean_reversion_step'
        }
        tprint("✅ Unified model loader initialized")

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
                tprint(f"⚠️ Unknown file extension: {path.suffix}")
                self.logger.warning(f"Unknown file extension: {path.suffix}")
                return None
        except Exception as e:
            tprint(f"❌ Failed to load artifact from {path}: {e}")
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
    async def load_specialist_models(
        self,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long",
        regime_timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        Load all specialist models required for feature generation.
        """
        tprint(f"🔄 Loading specialist models for {symbol} ({timeframe})...")
        models = {}

        # Helper to load artifact safely
        def _load_spec(key, step, type_='model', tf=timeframe):
            return self._get_artifact_with_context(
                artifact_key=key,
                step_name=self.step_mappings.get(step, step),
                artifact_type=type_,
                symbol=symbol,
                exchange=exchange,
                timeframe=tf,
                direction=direction
            )

        # 1. Risk Model
        risk_step = 'risk'
        models['risk_model'] = _load_spec('ml_risk_model_1h', risk_step, tf='1h')
        models['risk_hmm'] = _load_spec('risk_live_hmm', risk_step, type_='model', tf='1h')
        models['risk_pipeline'] = _load_spec('ml_risk_feature_pipeline_1h', risk_step, tf='1h')

        # 2. SMC Model
        models['smc_model'] = _load_spec('smc_xgb_model', 'smc')
        # Assuming SMC might have a pipeline too, though not explicitly seen.
        # If needed, can add here.

        # 3. Liquidity
        models['liquidity_tree'] = _load_spec('ml_liquidity_regime_tree_15m', 'liquidity', tf='15m')

        # 4. Breakout Model
        models['breakout_model'] = _load_spec(f'ml_breakout_bounce_model_{timeframe}', 'breakout', tf=timeframe)

        # 5. Meso Trend
        models['meso_model'] = _load_spec('xgb_meso_trend_model_15m', 'meso', tf='15m')
        models['meso_pipeline'] = _load_spec('xgb_meso_trend_feature_pipeline_15m', 'meso', tf='15m')

        # 6. Macro Trend
        models['macro_model'] = _load_spec('hmm_macro_trend_model_15m', 'macro', tf='15m')
        models['macro_pipeline'] = _load_spec('hmm_macro_trend_feature_pipeline_15m', 'macro', tf='15m')
        models['macro_thresholds'] = _load_spec('hmm_macro_trend_regime_thresholds_15m', 'macro', type_='data', tf='15m')

        # 7. Mean Reversion
        models['reversion_model'] = _load_spec(f'ml_mean_reversion_model_base_{timeframe}_{direction}', 'reversion', tf=timeframe)
        models['reversion_calibrated'] = _load_spec(f'ml_mean_reversion_model_calibrated_{timeframe}_{direction}', 'reversion', tf=timeframe)

        # Filter out Nones
        loaded = {k: v for k, v in models.items() if v is not None}
        
        if loaded:
            tprint(f"✅ Loaded {len(loaded)} specialist artifacts")
            self.loaded_models.update(loaded)
        else:
            tprint("⚠️ No specialist models found")

        return loaded

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
        Load all analyst base ML models generically.
        This loads the dictionary of models produced by UnifiedModelsTrainingStep.
        """
        tprint(f"🔄 Loading Analyst Base models for {symbol}...")
        step_name = self.step_mappings['analyst_base']
        models = {}

        # Try loading from artifact manager with proper context
        artifact_key = "analyst_base_training_result"
        artifact_data = self._get_artifact_with_context(
            artifact_key=artifact_key,
            step_name=step_name,
            artifact_type='model',
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model_type='Analyst'
        )
        
        if artifact_data:
            if isinstance(artifact_data, dict):
                # Extract models from result dict
                training_result = artifact_data.get('analyst_base_training_result', artifact_data)
                model_dict = training_result.get('models', {})
                
                for model_name, model_data in model_dict.items():
                    if isinstance(model_data, dict) and 'model' in model_data:
                        models[model_name] = model_data['model']
                    elif hasattr(model_data, 'predict'):
                        models[model_name] = model_data
            else:
                 tprint("⚠️ Unexpected artifact format for analyst_base_training_result")

        # Fallback: try common individual artifact names if dict not found
        if not models:
             tprint("⚠️ No model dict found, trying individual artifacts...")
             # Common model names (LGBM, NGBoost, CatBoost, XGBoost)
             for name in ['lgbm', 'ngboost', 'catboost', 'xgboost', 'tcn']:
                 model = self._get_artifact_with_context(
                    artifact_key=f'analyst_base_{name}',
                    step_name=step_name,
                    artifact_type='model',
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction,
                    model_type='Analyst'
                 )
                 if model:
                     models[name] = model

        self.loaded_models.update({f'analyst_base_{k}': v for k, v in models.items()})
        
        if models:
            tprint(f"✅ Loaded {len(models)} Analyst Base models: {list(models.keys())}")
        else:
            tprint("⚠️ No Analyst Base models found")
            
        return models

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
        """
        tprint(f"🔄 Loading optimized parameters for {symbol} ({timeframe})...")
        
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
                        tprint(f"✅ Loaded optimized parameters from artifacts")
                        return parameters

            # Try loading from file path (fallback)
            optimization_dir = Path("data_cache/optimization")
            results_file = optimization_dir / f"{exchange}_{symbol}_final_parameters.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    parameters = json.load(f)
                    if isinstance(parameters, dict):
                        self.optimized_parameters = parameters
                        tprint(f"✅ Loaded optimized parameters from {results_file}")
                        return parameters

            tprint("⚠️ No optimized parameters found, using defaults")
            return self._get_default_parameters()

        except Exception as e:
            tprint(f"⚠️ Failed to load optimized parameters: {e}")
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
            # Defaults for new architecture
            'signal_confidence_threshold': 0.6,
            'exit_confidence_threshold': 0.5,
            'uncertainty_threshold': 0.5,
            'uncertainty_penalty_factor': 0.5
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
        Updated to load Specialist and Analyst Base models (Generically).
        """
        tprint("🚀 Loading all trained models and optimized parameters...")

        # Load specialists
        specialists = await self.load_specialist_models(symbol, exchange, analyst_timeframe, direction, regime_timeframe)

        # Load Analyst Base models (Generic load, returns dict)
        analyst_base_models = await self.load_analyst_base_models(symbol, exchange, analyst_timeframe, direction)

        # Load parameters
        params = await self.load_optimized_parameters(symbol, exchange, analyst_timeframe, direction)

        result = {
            'specialists': specialists,
            'analyst_base_models': analyst_base_models,
            'optimized_parameters': params,
            # Legacy keys kept for backward compatibility
            'regime_base_models': {},
            'regime_ensemble_model': None,
            'analyst_ensemble_model': None,
            'tactician_base_models': {},
            'tactician_ensemble_model': None,
        }

        tprint("✅ All models and parameters loaded successfully")
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
