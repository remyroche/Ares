from ..utils.logger import system_logger
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
'Integration decorators for adding explainability to existing model pipelines.\n\nThis module provides decorators that can be applied to existing model methods\nto automatically generate explanations and trace trade decisions.\n'
import functools
from datetime import datetime
from .explainability.explainability_orchestrator import ExplainabilityOrchestrator

import pandas as pd
import logging
import time

class ExplainabilityIntegration:
    """Integration class for adding explainability to existing models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize explainability integration."""
        self.config = config
        self.logger = system_logger.getChild('ExplainabilityIntegration')
        self.orchestrator = ExplainabilityOrchestrator(config)
        self.integration_config = config.get('explainability', {}).get('integration', {})
        self.auto_register_models = self.integration_config.get('auto_register_models', True)
        self.auto_trace_decisions = self.integration_config.get('auto_trace_decisions', True)
        self.explanation_timeout = self.integration_config.get('explanation_timeout', 30)
        self.model_registry = {}

    def explainable_prediction(self, model_type: str, model_name: str='main', feature_extractor: Optional[Callable]=None, decision_tracer: bool = True) -> None:
        """Decorator for making model predictions explainable."""

        def decorator(func: Callable) -> None:

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> None:
                try:
                    features = None
                    feature_names = None
                    if feature_extractor:
                        features, feature_names = await self._extract_features(feature_extractor, *args, **kwargs)
                    prediction = await func(*args, **kwargs)
                    explanation = None
                    if features is not None and feature_names is not None:
                        explanation = await self.orchestrator.explain_model_prediction(model_type, model_name, features, feature_names, prediction)
                    if explanation:
                        if isinstance(prediction, dict):
                            prediction['explanation'] = explanation
                        else:
                            prediction = {'prediction': prediction, 'explanation': explanation}
                    return prediction
                except Exception as e:
                    self.logger.error(f'❌ Error in explainable prediction: {e}')
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def explainable_decision(self, decision_type: str, model_types: List[str]=None, feature_extractors: Optional[Dict[str, Callable]]=None) -> None:
        """Decorator for making trading decisions explainable."""

        def decorator(func: Callable) -> None:

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> None:
                try:
                    decision_id = f"{decision_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    trace = None
                    if self.auto_trace_decisions:
                        trace = await self.orchestrator.start_trade_decision_trace(decision_id, decision_type)
                    model_features = {}
                    if feature_extractors:
                        for model_type, extractor in feature_extractors.items():
                            try:
                                features, feature_names = await self._extract_features(extractor, *args, **kwargs)
                                if features is not None and feature_names is not None:
                                    model_features[model_type] = (features, feature_names)
                            except Exception as e:
                                self.logger.warning(f'⚠️ Failed to extract features for {model_type}: {e}')
                    decision = await func(*args, **kwargs)
                    if trace and model_features:
                        for model_type, (features, feature_names) in model_features.items():
                            explanation = await self.orchestrator.explain_model_prediction(model_type, 'main', features, feature_names)
                            if explanation:
                                await self.orchestrator.add_explanation_to_trace(decision_id, model_type, explanation)
                    if trace:
                        confidence = self._extract_confidence(decision)
                        final_trace = await self.orchestrator.finalize_trade_decision_trace(decision_id, decision, confidence)
                        if isinstance(decision, dict):
                            decision['decision_trace'] = final_trace
                        else:
                            decision = {'decision': decision, 'decision_trace': final_trace}
                    return decision
                except Exception as e:
                    self.logger.error(f'❌ Error in explainable decision: {e}')
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def auto_register_model(self, model_type: str, model_name: str='main') -> None:
        """Decorator for automatically registering models."""

        def decorator(func: Callable) -> None:

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> None:
                try:
                    model = None
                    if args and hasattr(args[0], '__class__'):
                        model = args[0]
                    elif 'self' in kwargs:
                        model = kwargs['self']
                    if model and self.auto_register_models:
                        model_key = f'{model_type}_{model_name}'
                        if model_key not in self.model_registry:
                            await self.orchestrator.register_model(model_type, model_name, model)
                            self.model_registry[model_key] = True
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f'❌ Error in auto register model: {e}')
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    async def _extract_features(self, feature_extractor: Callable, *args, **kwargs) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Extract features using the provided extractor function."""
        try:
            result = await feature_extractor(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                features, feature_names = result
                return (features, feature_names)
            elif isinstance(result, dict):
                features = result.get('features')
                feature_names = result.get('feature_names')
                return (features, feature_names)
            else:
                self.logger.warning('⚠️ Feature extractor returned unexpected format')
                return (None, None)
        except Exception as e:
            self.logger.error(f'❌ Failed to extract features: {e}')
            return (None, None)

    def _extract_confidence(self, decision: Any) -> float:
        """Extract confidence from decision result."""
        try:
            if isinstance(decision, dict):
                for key in ['confidence', 'probability', 'score', 'certainty']:
                    if key in decision and isinstance(decision[key], (int, float)):
                        return float(decision[key])
            return 0.5
        except Exception as e:
            self.logger.error(f'❌ Failed to extract confidence: {e}')
            return 0.5
_integration_instance = None

def get_explainability_integration(config: Dict[str, Any]) -> ExplainabilityIntegration:
    """Get or create global explainability integration instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = ExplainabilityIntegration(config)
    return _integration_instance

def explainable_tactician_prediction(model_name: str='main', feature_extractor: Optional[Callable]=None) -> None:
    """Decorator for Tactician predictions."""

    def decorator(func: Callable) -> None:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            if args and hasattr(args[0], 'config'):
                config = args[0].config
            else:
                config = {}
            integration = get_explainability_integration(config)
            explainable_func = integration.explainable_prediction('tactician', model_name, feature_extractor)(func)
            return await explainable_func(*args, **kwargs)
        return wrapper
    return decorator

def explainable_hmm_prediction(model_name: str='main', feature_extractor: Optional[Callable]=None) -> None:
    """Decorator for HMM predictions."""

    def decorator(func: Callable) -> None:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            if args and hasattr(args[0], 'config'):
                config = args[0].config
            else:
                config = {}
            integration = get_explainability_integration(config)
            explainable_func = integration.explainable_prediction('hmm', model_name, feature_extractor)(func)
            return await explainable_func(*args, **kwargs)
        return wrapper
    return decorator

def explainable_sr_prediction(model_name: str='main', feature_extractor: Optional[Callable]=None) -> None:
    """Decorator for SR predictions."""

    def decorator(func: Callable) -> None:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            if args and hasattr(args[0], 'config'):
                config = args[0].config
            else:
                config = {}
            integration = get_explainability_integration(config)
            explainable_func = integration.explainable_prediction('sr', model_name, feature_extractor)(func)
            return await explainable_func(*args, **kwargs)
        return wrapper
    return decorator

def explainable_analyst_prediction(model_name: str='main', feature_extractor: Optional[Callable]=None) -> None:
    """Decorator for Analyst predictions."""

    def decorator(func: Callable) -> None:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            if args and hasattr(args[0], 'config'):
                config = args[0].config
            else:
                config = {}
            integration = get_explainability_integration(config)
            explainable_func = integration.explainable_prediction('analyst', model_name, feature_extractor)(func)
            return await explainable_func(*args, **kwargs)
        return wrapper
    return decorator

def explainable_trading_decision(decision_type: str, model_types: List[str]=None, feature_extractors: Optional[Dict[str, Callable]]=None) -> None:
    """Decorator for trading decisions."""

    def decorator(func: Callable) -> None:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            if args and hasattr(args[0], 'config'):
                config = args[0].config
            else:
                config = {}
            integration = get_explainability_integration(config)
            explainable_func = integration.explainable_decision(decision_type, model_types, feature_extractors)(func)
            return await explainable_func(*args, **kwargs)
        return wrapper
    return decorator

class FeatureExtractor:
    """Utility class for creating feature extractors."""

    @staticmethod
    def from_dataframe(columns: List[str], preprocessing: Optional[Callable]=None) -> None:
        """Create feature extractor from DataFrame columns."""

        async def extractor(*args, **kwargs) -> None:
            try:
                df = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        df = arg
                        break
                if df is None:
                    return (None, None)
                if columns:
                    features_df = df[columns].fillna(0)
                else:
                    features_df = df.fillna(0)
                if preprocessing:
                    features_df = preprocessing(features_df)
                features = features_df.values
                feature_names = features_df.columns.tolist()
                return (features, feature_names)
            except Exception as e:
                system_logger.error(f'❌ Feature extraction failed: {e}')
                return (None, None)
        return extractor

    @staticmethod
    def from_dict(key_mapping: Dict[str, str]) -> None:
        """Create feature extractor from dictionary."""

        async def extractor(*args, **kwargs) -> None:
            try:
                data_dict = None
                for arg in args:
                    if isinstance(arg, dict):
                        data_dict = arg
                        break
                if data_dict is None:
                    return (None, None)
                features = []
                feature_names = []
                for feature_name, key in key_mapping.items():
                    if key in data_dict:
                        features.append(data_dict[key])
                        feature_names.append(feature_name)
                if not features:
                    return (None, None)
                return (np.array(features), feature_names)
            except Exception as e:
                system_logger.error(f'❌ Feature extraction failed: {e}')
                return (None, None)
        return extractor

    @staticmethod
    def custom(extractor_func: Callable) -> None:
        """Create custom feature extractor."""

        async def extractor(*args, **kwargs) -> None:
            try:
                result = await extractor_func(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                elif isinstance(result, dict):
                    return (result.get('features'), result.get('feature_names'))
                else:
                    return (None, None)
            except Exception as e:
                system_logger.error(f'❌ Custom feature extraction failed: {e}')
                return (None, None)
        return extractor
