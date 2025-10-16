from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import numpy as np
import pandas as pd
'Analyst model explainer for SHAP and LIME integration.\n\nThis module provides explainability for Analyst ensemble models, enabling\ntraceability of market analysis decisions back to individual factors.\n'
from .explainability.base_explainer import BaseExplainer, ExplanationResult
import logging

try:
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

class AnalystExplainer(BaseExplainer):
    """Explainer for Analyst ensemble models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Analyst explainer."""
        super().__init__(config, 'Analyst')
        self.analyst_config = config.get('analyst', {})
        self.explain_config = config.get('explainability', {}).get('analyst', {})
        self.explain_regime_classifier = self.explain_config.get('explain_regime_classifier', True)
        self.explain_location_classifier = self.explain_config.get('explain_location_classifier', True)
        self.explain_ensemble_prediction = self.explain_config.get('explain_ensemble_prediction', True)
        self.explain_confidence_prediction = self.explain_config.get('explain_confidence_prediction', True)
        self.feature_groups = {'price_features': ['close', 'open', 'high', 'low', 'log_returns', 'price_momentum'], 'volume_features': ['volume', 'volume_ratio', 'volume_momentum', 'volume_volatility'], 'volatility_features': ['volatility_20', 'volatility_5', 'atr', 'bb_width', 'volatility_regime'], 'technical_indicators': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'adx'], 'regime_features': ['regime_probability', 'regime_confidence', 'regime_persistence'], 'location_features': ['location_probability', 'location_confidence', 'location_stability'], 'ensemble_features': ['ensemble_agreement', 'ensemble_confidence', 'ensemble_diversity'], 'market_health': ['market_health_score', 'liquidation_risk', 'correlation_risk']}
        self.regime_types = ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE', 'TRANSITION']
        self.location_types = ['TOP', 'BOTTOM', 'MIDDLE', 'TRANSITION']

    async def explain_regime_classification(self, analyst_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], regime_prediction: Optional[Union[str, int]]=None) -> ExplanationResult:
        """Explain regime classification decision."""
        try:
            self.logger.info('🔍 Explaining Analyst regime classification...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(analyst_model, market_data)
            if regime_prediction is None:
                if hasattr(analyst_model, 'predict_regime'):
                    regime_prediction = await analyst_model.predict_regime(features)
                elif hasattr(analyst_model, 'predict'):
                    regime_prediction = analyst_model.predict(features.reshape(1, -1))[0]
                else:
                    regime_prediction = None
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, analyst_model, features, feature_names)
            feature_importance = self._extract_feature_importance(analyst_model)
            confidence = self._calculate_regime_confidence(analyst_model, features, regime_prediction)
            explanation = ExplanationResult(model_name='Analyst_RegimeClassification', prediction = regime_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'regime_classifier', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'explanation_type': 'regime_classification'})
            self.save_explanation(explanation)
            self.logger.info('✅ Analyst regime classification explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain Analyst regime classification: {e}')
            return None

    async def explain_location_classification(self, analyst_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], location_prediction: Optional[Union[str, int]]=None) -> ExplanationResult:
        """Explain location classification decision."""
        try:
            self.logger.info('🔍 Explaining Analyst location classification...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(analyst_model, market_data)
            if location_prediction is None:
                if hasattr(analyst_model, 'predict_location'):
                    location_prediction = await analyst_model.predict_location(features)
                elif hasattr(analyst_model, 'predict'):
                    location_prediction = analyst_model.predict(features.reshape(1, -1))[0]
                else:
                    location_prediction = None
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, analyst_model, features, feature_names)
            feature_importance = self._extract_feature_importance(analyst_model)
            confidence = self._calculate_location_confidence(analyst_model, features, location_prediction)
            explanation = ExplanationResult(model_name='Analyst_LocationClassification', prediction = location_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'location_classifier', 'feature_groups': self.feature_groups, 'location_types': self.location_types, 'explanation_type': 'location_classification'})
            self.save_explanation(explanation)
            self.logger.info('✅ Analyst location classification explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain Analyst location classification: {e}')
            return None

    async def explain_ensemble_prediction(self, analyst_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], ensemble_prediction: Optional[Dict[str, Any]]=None) -> ExplanationResult:
        """Explain ensemble prediction decision."""
        try:
            self.logger.info('🔍 Explaining Analyst ensemble prediction...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(analyst_model, market_data)
            if ensemble_prediction is None:
                if hasattr(analyst_model, 'predict_ensemble'):
                    ensemble_prediction = await analyst_model.predict_ensemble(features)
                elif hasattr(analyst_model, 'predict_regime_and_location'):
                    ensemble_prediction = await analyst_model.predict_regime_and_location(features)
                elif hasattr(analyst_model, 'predict'):
                    ensemble_prediction = analyst_model.predict(features.reshape(1, -1))
                else:
                    ensemble_prediction = {}
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_ensemble_shap_values(analyst_model, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, analyst_model, features, feature_names)
            feature_importance = self._extract_ensemble_feature_importance(analyst_model)
            confidence = self._calculate_ensemble_confidence(ensemble_prediction, shap_values)
            explanation = ExplanationResult(model_name='Analyst_EnsemblePrediction', prediction = ensemble_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'ensemble_prediction', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'location_types': self.location_types, 'explanation_type': 'ensemble_prediction'})
            self.save_explanation(explanation)
            self.logger.info('✅ Analyst ensemble prediction explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain Analyst ensemble prediction: {e}')
            return None

    async def explain_confidence_prediction(self, analyst_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], confidence_prediction: Optional[float]=None) -> ExplanationResult:
        """Explain confidence prediction decision."""
        try:
            self.logger.info('🔍 Explaining Analyst confidence prediction...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(analyst_model, market_data)
            if confidence_prediction is None:
                if hasattr(analyst_model, 'predict_confidence'):
                    confidence_prediction = await analyst_model.predict_confidence(features)
                elif hasattr(analyst_model, 'get_confidence'):
                    confidence_prediction = await analyst_model.get_confidence(features)
                elif hasattr(analyst_model, 'predict'):
                    confidence_prediction = analyst_model.predict(features.reshape(1, -1))[0]
                else:
                    confidence_prediction = 0.5
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, analyst_model, features, feature_names)
            feature_importance = self._extract_feature_importance(analyst_model)
            confidence = self._calculate_confidence_confidence(confidence_prediction, shap_values)
            explanation = ExplanationResult(model_name='Analyst_ConfidencePrediction', prediction = confidence_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'confidence_prediction', 'feature_groups': self.feature_groups, 'explanation_type': 'confidence_prediction'})
            self.save_explanation(explanation)
            self.logger.info('✅ Analyst confidence prediction explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain Analyst confidence prediction: {e}')
            return None

    async def explain_prediction(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a general Analyst model prediction."""
        try:
            self.logger.info('🔍 Explaining Analyst prediction...')
            model_type = self._identify_analyst_model_type(model)
            if model_type == 'regime_classifier':
                return await self.explain_regime_classification(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'location_classifier':
                return await self.explain_location_classification(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'ensemble_prediction':
                return await self.explain_ensemble_prediction(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'confidence_prediction':
                return await self.explain_confidence_prediction(model, pd.DataFrame(), features, feature_names, prediction)
            else:
                return await self._explain_generic_analyst_model(model, features, feature_names, prediction)
        except Exception as e:
            self.logger.error(f'❌ Failed to explain Analyst prediction: {e}')
            return None

    async def initialize_explainers(self, model: Any, training_data: pd.DataFrame) -> bool:
        """Initialize SHAP and LIME explainers for Analyst models."""
        try:
            self.logger.info('🔧 Initializing Analyst explainers...')
            self.shap_explainer = self._create_shap_explainer(model, training_data)
            if not training_data.empty:
                feature_names = training_data.columns.tolist()
                self.lime_explainer = self._create_lime_explainer(training_data, feature_names)
                self.feature_names = feature_names
            else:
                self.lime_explainer = None
            self.logger.info('✅ Analyst explainers initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize Analyst explainers: {e}')
            return False

    def _identify_analyst_model_type(self, model: Any) -> str:
        """Identify the type of Analyst model."""
        model_name = str(type(model).__name__).lower()
        if 'regime' in model_name and 'classifier' in model_name:
            return 'regime_classifier'
        elif 'location' in model_name and 'classifier' in model_name:
            return 'location_classifier'
        elif 'ensemble' in model_name or 'unified' in model_name:
            return 'ensemble_prediction'
        elif 'confidence' in model_name or 'predictor' in model_name:
            return 'confidence_prediction'
        else:
            return 'generic'

    async def _explain_generic_analyst_model(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a generic Analyst model."""
        try:
            if prediction is None:
                if hasattr(model, 'predict'):
                    prediction = model.predict(features.reshape(1, -1))
                else:
                    prediction = None
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, model, features, feature_names)
            feature_importance = self._extract_feature_importance(model)
            confidence = self._calculate_prediction_confidence(prediction, shap_values)
            explanation = ExplanationResult(model_name='Analyst_Generic', prediction = prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'generic', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'location_types': self.location_types, 'explanation_type': 'generic_prediction'})
            self.save_explanation(explanation)
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain generic Analyst model: {e}')
            return None

    def _calculate_ensemble_shap_values(self, analyst_model: Any, features: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Calculate SHAP values for ensemble models."""
        try:
            if self.shap_explainer is None:
                return None
            ensemble_shap = {}
            if hasattr(analyst_model, 'regime_classifier') and analyst_model.regime_classifier is not None:
                regime_shap = self._calculate_shap_values(self.shap_explainer, features)
                if regime_shap is not None:
                    ensemble_shap['regime_classifier'] = regime_shap
            if hasattr(analyst_model, 'location_classifier') and analyst_model.location_classifier is not None:
                location_shap = self._calculate_shap_values(self.shap_explainer, features)
                if location_shap is not None:
                    ensemble_shap['location_classifier'] = location_shap
            if not ensemble_shap:
                main_shap = self._calculate_shap_values(self.shap_explainer, features)
                if main_shap is not None:
                    ensemble_shap['main_model'] = main_shap
            return ensemble_shap if ensemble_shap else None
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate ensemble SHAP values: {e}')
            return None

    def _extract_ensemble_feature_importance(self, analyst_model: Any) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract feature importance from ensemble models."""
        try:
            ensemble_importance = {}
            if hasattr(analyst_model, 'regime_classifier') and analyst_model.regime_classifier is not None:
                regime_importance = self._extract_feature_importance(analyst_model.regime_classifier)
                if regime_importance:
                    ensemble_importance['regime_classifier'] = regime_importance
            if hasattr(analyst_model, 'location_classifier') and analyst_model.location_classifier is not None:
                location_importance = self._extract_feature_importance(analyst_model.location_classifier)
                if location_importance:
                    ensemble_importance['location_classifier'] = location_importance
            main_importance = self._extract_feature_importance(analyst_model)
            if main_importance:
                ensemble_importance['main_model'] = main_importance
            return ensemble_importance if ensemble_importance else None
        except Exception as e:
            self.logger.error(f'❌ Failed to extract ensemble feature importance: {e}')
            return None

    def _calculate_regime_confidence(self, analyst_model: Any, features: np.ndarray, regime_prediction: Union[str, int]) -> float:
        """Calculate confidence for regime classification."""
        try:
            confidence = 0.5
            if hasattr(analyst_model, 'predict_proba'):
                probas = analyst_model.predict_proba(features.reshape(1, -1))[0]
                if isinstance(regime_prediction, str):
                    regime_index = self.regime_types.index(regime_prediction) if regime_prediction in self.regime_types else 0
                else:
                    regime_index = regime_prediction
                if regime_index < len(probas):
                    confidence = probas[regime_index]
            if hasattr(analyst_model, 'get_ensemble_agreement'):
                agreement = analyst_model.get_ensemble_agreement(features)
                if agreement is not None:
                    confidence = (confidence + agreement) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate regime confidence: {e}')
            return 0.5

    def _calculate_location_confidence(self, analyst_model: Any, features: np.ndarray, location_prediction: Union[str, int]) -> float:
        """Calculate confidence for location classification."""
        try:
            confidence = 0.5
            if hasattr(analyst_model, 'predict_proba'):
                probas = analyst_model.predict_proba(features.reshape(1, -1))[0]
                if isinstance(location_prediction, str):
                    location_index = self.location_types.index(location_prediction) if location_prediction in self.location_types else 0
                else:
                    location_index = location_prediction
                if location_index < len(probas):
                    confidence = probas[location_index]
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate location confidence: {e}')
            return 0.5

    def _calculate_ensemble_confidence(self, ensemble_prediction: Dict[str, Any], shap_values: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate confidence for ensemble prediction."""
        try:
            confidence = 0.5
            if isinstance(ensemble_prediction, dict):
                if 'confidence' in ensemble_prediction:
                    confidence = ensemble_prediction['confidence']
                elif 'ensemble_confidence' in ensemble_prediction:
                    confidence = ensemble_prediction['ensemble_confidence']
                elif 'agreement' in ensemble_prediction:
                    confidence = ensemble_prediction['agreement']
            if isinstance(ensemble_prediction, dict) and 'regime' in ensemble_prediction and ('location' in ensemble_prediction):
                confidence += 0.1
            if isinstance(shap_values, dict) and len(shap_values) > 1:
                shap_consistency = self._calculate_ensemble_shap_consistency(shap_values)
                confidence = (confidence + shap_consistency) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate ensemble confidence: {e}')
            return 0.5

    def _calculate_confidence_confidence(self, confidence_prediction: float, shap_values: Optional[np.ndarray]) -> float:
        """Calculate confidence for confidence prediction."""
        try:
            confidence = 0.5
            if isinstance(confidence_prediction, (int, float)):
                if confidence_prediction > 0.8:
                    confidence += 0.2
                elif confidence_prediction > 0.6:
                    confidence += 0.1
                elif confidence_prediction < 0.4:
                    confidence -= 0.1
            if shap_values is not None:
                shap_consistency = 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-08)
                confidence = (confidence + shap_consistency) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate confidence confidence: {e}')
            return 0.5

    def _calculate_ensemble_shap_consistency(self, shap_values: Dict[str, np.ndarray]) -> float:
        """Calculate consistency of SHAP values across ensemble components."""
        try:
            if len(shap_values) < 2:
                return 1.0
            shap_arrays = list(shap_values.values())
            correlations = []
            for i in range(len(shap_arrays)):
                for j in range(i + 1, len(shap_arrays)):
                    if len(shap_arrays[i]) == len(shap_arrays[j]):
                        corr = np.corrcoef(shap_arrays[i], shap_arrays[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            if correlations:
                return np.mean(correlations)
            else:
                return 0.5
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate ensemble SHAP consistency: {e}')
            return 0.5

    def get_ensemble_component_importance(self, explanation: ExplanationResult) -> Dict[str, Dict[str, float]]:
        """Get feature importance by ensemble component."""
        try:
            component_importance = {}
            if isinstance(explanation.shap_values, dict):
                for component, shap_values in explanation.shap_values.items():
                    if shap_values is not None:
                        component_importance[component] = dict(zip(explanation.feature_names, shap_values))
            elif explanation.shap_values is not None:
                component_importance['main'] = dict(zip(explanation.feature_names, explanation.shap_values))
            return component_importance
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate ensemble component importance: {e}')
            return {}

    def generate_analyst_explanation_summary(self, explanation: ExplanationResult) -> str:
        """Generate human-readable Analyst explanation summary."""
        try:
            summary_parts = []
            summary_parts.append(f'Model: {explanation.model_name}')
            summary_parts.append(f'Prediction: {explanation.prediction}')
            summary_parts.append(f'Confidence: {explanation.confidence:.2f}')
            if isinstance(explanation.prediction, dict):
                if 'regime' in explanation.prediction:
                    summary_parts.append(f"Regime: {explanation.prediction['regime']}")
                if 'location' in explanation.prediction:
                    summary_parts.append(f"Location: {explanation.prediction['location']}")
                if 'ensemble_confidence' in explanation.prediction:
                    summary_parts.append(f"Ensemble Confidence: {explanation.prediction['ensemble_confidence']:.3f}")
                if 'agreement' in explanation.prediction:
                    summary_parts.append(f"Ensemble Agreement: {explanation.prediction['agreement']:.3f}")
            if explanation.shap_values is not None:
                if isinstance(explanation.shap_values, dict):
                    for component, shap_values in explanation.shap_values.items():
                        if shap_values is not None:
                            summary_parts.append(f'\nTop Factors for {component}:')
                            feature_importance = list(zip(explanation.feature_names, shap_values))
                            feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
                            for i, (feature, importance) in enumerate(feature_importance[:3]):
                                direction = 'increases' if importance > 0 else 'decreases'
                                summary_parts.append(f'  {i + 1}. {feature}: {direction} by {abs(importance):.3f}')
                else:
                    summary_parts.append('\nTop Contributing Factors:')
                    feature_importance = list(zip(explanation.feature_names, explanation.shap_values))
                    feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
                    for i, (feature, importance) in enumerate(feature_importance[:5]):
                        direction = 'increases' if importance > 0 else 'decreases'
                        summary_parts.append(f'  {i + 1}. {feature}: {direction} prediction by {abs(importance):.3f}')
            group_importance = self.get_feature_group_importance(explanation)
            if group_importance:
                summary_parts.append('\nFeature Group Importance:')
                for group, importance in sorted(group_importance.items(), key = lambda x: x[1], reverse = True):
                    summary_parts.append(f'  {group}: {importance:.3f}')
            return '\n'.join(summary_parts)
        except Exception as e:
            self.logger.error(f'❌ Failed to generate Analyst explanation summary: {e}')
            return f'Analyst explanation summary generation failed: {e}'
