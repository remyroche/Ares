from typing import List
from typing import Any
from typing import Dict
import pandas as pd
from typing import Optional
from typing import Union
import numpy as np
'HMM model explainer for SHAP and LIME integration.\n\nThis module provides explainability for HMM regime discovery models, enabling\ntraceability of regime classification decisions back to individual factors.\n'
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

class HMMExplainer(BaseExplainer):
    """Explainer for HMM regime discovery models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize HMM explainer."""
        super().__init__(config, 'HMM')
        self.hmm_config = config.get('hmm', {})
        self.explain_config = config.get('explainability', {}).get('hmm', {})
        self.explain_regime_classifier = self.explain_config.get('explain_regime_classifier', True)
        self.explain_transition_predictor = self.explain_config.get('explain_transition_predictor', True)
        self.explain_regime_probability = self.explain_config.get('explain_regime_probability', True)
        self.feature_groups = {'price_features': ['close', 'open', 'high', 'low', 'log_returns', 'price_momentum'], 'volatility_features': ['volatility_20', 'volatility_5', 'atr', 'bb_width', 'volatility_regime'], 'volume_features': ['volume', 'volume_ratio', 'volume_momentum', 'volume_volatility'], 'technical_indicators': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'adx'], 'regime_features': ['regime_probability', 'regime_confidence', 'regime_persistence'], 'transition_features': ['transition_probability', 'transition_confidence', 'regime_stability']}
        self.regime_types = ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE', 'TRANSITION']

    async def explain_regime_classification(self, hmm_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], regime_prediction: Optional[Union[str, int]]=None) -> ExplanationResult:
        """Explain regime classification decision."""
        try:
            self.logger.info('🔍 Explaining HMM regime classification...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(hmm_model, market_data)
            if regime_prediction is None:
                if hasattr(hmm_model, 'predict_regime'):
                    regime_prediction = await hmm_model.predict_regime(features)
                elif hasattr(hmm_model, 'predict'):
                    regime_prediction = hmm_model.predict(features.reshape(1, -1))[0]
                else:
                    regime_prediction = None
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, hmm_model, features, feature_names)
            feature_importance = self._extract_feature_importance(hmm_model)
            confidence = self._calculate_regime_confidence(hmm_model, features, regime_prediction)
            explanation = ExplanationResult(model_name='HMM_RegimeClassification', prediction = regime_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'regime_classifier', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'explanation_type': 'regime_classification'})
            self.save_explanation(explanation)
            self.logger.info('✅ HMM regime classification explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain HMM regime classification: {e}')
            return None

    async def explain_regime_probabilities(self, hmm_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], regime_probabilities: Optional[Dict[str, float]]=None) -> ExplanationResult:
        """Explain regime probability distribution."""
        try:
            self.logger.info('🔍 Explaining HMM regime probabilities...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(hmm_model, market_data)
            if regime_probabilities is None:
                if hasattr(hmm_model, 'predict_proba'):
                    proba_array = hmm_model.predict_proba(features.reshape(1, -1))[0]
                    regime_probabilities = dict(zip(self.regime_types[:len(proba_array)], proba_array))
                elif hasattr(hmm_model, 'get_regime_probabilities'):
                    regime_probabilities = await hmm_model.get_regime_probabilities(features)
                else:
                    regime_probabilities = {}
            shap_values_by_regime = {}
            if self.shap_explainer is not None:
                for regime in self.regime_types:
                    if regime in regime_probabilities:
                        regime_shap = self._calculate_regime_specific_shap(hmm_model, features, regime)
                        shap_values_by_regime[regime] = regime_shap
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, hmm_model, features, feature_names)
            feature_importance = self._extract_feature_importance(hmm_model)
            confidence = self._calculate_probability_confidence(regime_probabilities)
            explanation = ExplanationResult(model_name='HMM_RegimeProbabilities', prediction = regime_probabilities, feature_names = feature_names, feature_values = features, shap_values = shap_values_by_regime, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'regime_probability', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'explanation_type': 'regime_probabilities'})
            self.save_explanation(explanation)
            self.logger.info('✅ HMM regime probabilities explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain HMM regime probabilities: {e}')
            return None

    async def explain_transition_prediction(self, hmm_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], transition_prediction: Optional[Dict[str, Any]]=None) -> ExplanationResult:
        """Explain regime transition prediction."""
        try:
            self.logger.info('🔍 Explaining HMM transition prediction...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(hmm_model, market_data)
            if transition_prediction is None:
                if hasattr(hmm_model, 'predict_transition'):
                    transition_prediction = await hmm_model.predict_transition(features)
                elif hasattr(hmm_model, 'get_transition_probabilities'):
                    transition_prediction = await hmm_model.get_transition_probabilities(features)
                else:
                    transition_prediction = {}
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, hmm_model, features, feature_names)
            feature_importance = self._extract_feature_importance(hmm_model)
            confidence = self._calculate_transition_confidence(transition_prediction)
            explanation = ExplanationResult(model_name='HMM_TransitionPrediction', prediction = transition_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'transition_predictor', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'explanation_type': 'transition_prediction'})
            self.save_explanation(explanation)
            self.logger.info('✅ HMM transition prediction explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain HMM transition prediction: {e}')
            return None

    async def explain_prediction(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a general HMM model prediction."""
        try:
            self.logger.info('🔍 Explaining HMM prediction...')
            model_type = self._identify_hmm_model_type(model)
            if model_type == 'regime_classifier':
                return await self.explain_regime_classification(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'regime_probability':
                return await self.explain_regime_probabilities(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'transition_predictor':
                return await self.explain_transition_prediction(model, pd.DataFrame(), features, feature_names, prediction)
            else:
                return await self._explain_generic_hmm_model(model, features, feature_names, prediction)
        except Exception as e:
            self.logger.error(f'❌ Failed to explain HMM prediction: {e}')
            return None

    async def initialize_explainers(self, model: Any, training_data: pd.DataFrame) -> bool:
        """Initialize SHAP and LIME explainers for HMM models."""
        try:
            self.logger.info('🔧 Initializing HMM explainers...')
            self.shap_explainer = self._create_shap_explainer(model, training_data)
            if not training_data.empty:
                feature_names = training_data.columns.tolist()
                self.lime_explainer = self._create_lime_explainer(training_data, feature_names)
                self.feature_names = feature_names
            else:
                self.lime_explainer = None
            self.logger.info('✅ HMM explainers initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize HMM explainers: {e}')
            return False

    def _identify_hmm_model_type(self, model: Any) -> str:
        """Identify the type of HMM model."""
        model_name = str(type(model).__name__).lower()
        if 'regime' in model_name and 'classifier' in model_name:
            return 'regime_classifier'
        elif 'regime' in model_name and 'probability' in model_name:
            return 'regime_probability'
        elif 'transition' in model_name or 'predictor' in model_name:
            return 'transition_predictor'
        else:
            return 'generic'

    async def _explain_generic_hmm_model(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a generic HMM model."""
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
            explanation = ExplanationResult(model_name='HMM_Generic', prediction = prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'generic', 'feature_groups': self.feature_groups, 'regime_types': self.regime_types, 'explanation_type': 'generic_prediction'})
            self.save_explanation(explanation)
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain generic HMM model: {e}')
            return None

    def _calculate_regime_specific_shap(self, model: Any, features: np.ndarray, regime: str) -> Optional[np.ndarray]:
        """Calculate SHAP values for a specific regime."""
        try:
            if self.shap_explainer is None:
                return None
            if hasattr(model, 'predict_proba'):
                all_probas = model.predict_proba(features.reshape(1, -1))[0]
                regime_index = self.regime_types.index(regime) if regime in self.regime_types else 0
                if regime_index < len(all_probas):
                    regime_prob = all_probas[regime_index]
                    shap_values = self._calculate_shap_values(self.shap_explainer, features)
                    if shap_values is not None:
                        shap_values = shap_values * regime_prob
                    return shap_values
            return None
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate regime-specific SHAP values: {e}')
            return None

    def _calculate_regime_confidence(self, model: Any, features: np.ndarray, regime_prediction: Union[str, int]) -> float:
        """Calculate confidence for regime classification."""
        try:
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(features.reshape(1, -1))[0]
                if isinstance(regime_prediction, str):
                    regime_index = self.regime_types.index(regime_prediction) if regime_prediction in self.regime_types else 0
                else:
                    regime_index = regime_prediction
                if regime_index < len(probas):
                    confidence = probas[regime_index]
            if isinstance(regime_prediction, str):
                if 'confidence' in str(regime_prediction).lower():
                    confidence += 0.1
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate regime confidence: {e}')
            return 0.5

    def _calculate_probability_confidence(self, regime_probabilities: Dict[str, float]) -> float:
        """Calculate confidence for regime probability distribution."""
        try:
            if not regime_probabilities:
                return 0.0
            probabilities = list(regime_probabilities.values())
            max_prob = max(probabilities)
            entropy = -sum((p * np.log(p + 1e-08) for p in probabilities if p > 0))
            max_entropy = np.log(len(probabilities))
            confidence = max_prob * (1 - entropy / max_entropy)
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate probability confidence: {e}')
            return 0.5

    def _calculate_transition_confidence(self, transition_prediction: Dict[str, Any]) -> float:
        """Calculate confidence for transition prediction."""
        try:
            if not transition_prediction:
                return 0.0
            if 'confidence' in transition_prediction:
                return transition_prediction['confidence']
            elif 'probability' in transition_prediction:
                return transition_prediction['probability']
            else:
                if isinstance(transition_prediction, dict):
                    values = [v for v in transition_prediction.values() if isinstance(v, (int, float))]
                    if values:
                        return max(values)
                return 0.5
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate transition confidence: {e}')
            return 0.5

    def get_regime_feature_importance(self, explanation: ExplanationResult) -> Dict[str, Dict[str, float]]:
        """Get feature importance by regime type."""
        try:
            regime_importance = {}
            if isinstance(explanation.shap_values, dict):
                for regime, shap_values in explanation.shap_values.items():
                    if shap_values is not None:
                        regime_importance[regime] = dict(zip(explanation.feature_names, shap_values))
            elif explanation.shap_values is not None:
                regime_importance['current'] = dict(zip(explanation.feature_names, explanation.shap_values))
            return regime_importance
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate regime feature importance: {e}')
            return {}

    def generate_regime_explanation_summary(self, explanation: ExplanationResult) -> str:
        """Generate human-readable regime explanation summary."""
        try:
            summary_parts = []
            summary_parts.append(f'Model: {explanation.model_name}')
            summary_parts.append(f'Prediction: {explanation.prediction}')
            summary_parts.append(f'Confidence: {explanation.confidence:.2f}')
            if isinstance(explanation.prediction, dict):
                summary_parts.append('\nRegime Probabilities:')
                for regime, prob in explanation.prediction.items():
                    summary_parts.append(f'  {regime}: {prob:.3f}')
            if explanation.shap_values is not None:
                if isinstance(explanation.shap_values, dict):
                    for regime, shap_values in explanation.shap_values.items():
                        if shap_values is not None:
                            summary_parts.append(f'\nTop Factors for {regime}:')
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
            self.logger.error(f'❌ Failed to generate regime explanation summary: {e}')
            return f'Regime explanation summary generation failed: {e}'