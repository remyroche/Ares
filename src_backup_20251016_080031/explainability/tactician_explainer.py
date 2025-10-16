from typing import List
from typing import Any
from typing import Dict
import pandas as pd
from typing import Optional
import numpy as np
'Tactician model explainer for SHAP and LIME integration.\n\nThis module provides explainability for Tactician ML models, enabling\ntraceability of trading decisions back to individual factors.\n'
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

class TacticianExplainer(BaseExplainer):
    """Explainer for Tactician ML models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Tactician explainer."""
        super().__init__(config, 'Tactician')
        self.tactician_config = config.get('tactician', {})
        self.explain_config = config.get('explainability', {}).get('tactician', {})
        self.explain_scenario_predictor = self.explain_config.get('explain_scenario_predictor', True)
        self.explain_position_sizer = self.explain_config.get('explain_position_sizer', True)
        self.explain_leverage_sizer = self.explain_config.get('explain_leverage_sizer', True)
        self.feature_groups = {'market_conditions': ['volatility', 'volume', 'price_momentum', 'trend_strength'], 'technical_indicators': ['rsi', 'macd', 'bb_position', 'atr', 'adx'], 'regime_factors': ['regime_probability', 'regime_confidence', 'regime_transition_prob'], 'sr_factors': ['sr_proximity', 'sr_strength', 'sr_breakout_probability'], 'risk_factors': ['liquidation_risk', 'correlation_risk', 'volatility_risk'], 'scenario_factors': ['scenario_probability', 'scenario_confidence', 'scenario_dominance']}

    async def explain_scenario_prediction(self, scenario_predictor: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], prediction: Optional[Dict[str, Any]]=None) -> ExplanationResult:
        """Explain scenario-based prediction."""
        try:
            self.logger.info('🔍 Explaining scenario prediction...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(scenario_predictor, market_data)
            if prediction is None:
                if hasattr(scenario_predictor, 'predict_scenarios'):
                    prediction = await scenario_predictor.predict_scenarios(features)
                else:
                    prediction = scenario_predictor.predict(features.reshape(1, -1))
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, scenario_predictor, features, feature_names)
            feature_importance = self._extract_feature_importance(scenario_predictor)
            confidence = self._calculate_prediction_confidence(prediction, shap_values)
            explanation = ExplanationResult(model_name='Tactician_Scenario', prediction = prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'scenario_predictor', 'feature_groups': self.feature_groups, 'explanation_type': 'scenario_prediction'})
            self.save_explanation(explanation)
            self.logger.info('✅ Scenario prediction explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain scenario prediction: {e}')
            return None

    async def explain_position_sizing(self, position_sizer: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], position_size: Optional[float]=None) -> ExplanationResult:
        """Explain position sizing decision."""
        try:
            self.logger.info('🔍 Explaining position sizing decision...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(position_sizer, market_data)
            if position_size is None:
                if hasattr(position_sizer, 'calculate_position_size'):
                    position_size = await position_sizer.calculate_position_size(features)
                else:
                    position_size = position_sizer.predict(features.reshape(1, -1))[0]
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, position_sizer, features, feature_names)
            feature_importance = self._extract_feature_importance(position_sizer)
            confidence = self._calculate_prediction_confidence(position_size, shap_values)
            explanation = ExplanationResult(model_name='Tactician_PositionSizing', prediction = position_size, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'position_sizer', 'feature_groups': self.feature_groups, 'explanation_type': 'position_sizing'})
            self.save_explanation(explanation)
            self.logger.info('✅ Position sizing explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain position sizing: {e}')
            return None

    async def explain_leverage_decision(self, leverage_sizer: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], leverage: Optional[float]=None) -> ExplanationResult:
        """Explain leverage decision."""
        try:
            self.logger.info('🔍 Explaining leverage decision...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(leverage_sizer, market_data)
            if leverage is None:
                if hasattr(leverage_sizer, 'calculate_leverage'):
                    leverage = await leverage_sizer.calculate_leverage(features)
                else:
                    leverage = leverage_sizer.predict(features.reshape(1, -1))[0]
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, leverage_sizer, features, feature_names)
            feature_importance = self._extract_feature_importance(leverage_sizer)
            confidence = self._calculate_prediction_confidence(leverage, shap_values)
            explanation = ExplanationResult(model_name='Tactician_Leverage', prediction = leverage, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'leverage_sizer', 'feature_groups': self.feature_groups, 'explanation_type': 'leverage_decision'})
            self.save_explanation(explanation)
            self.logger.info('✅ Leverage decision explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain leverage decision: {e}')
            return None

    async def explain_prediction(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a general Tactician model prediction."""
        try:
            self.logger.info('🔍 Explaining Tactician prediction...')
            model_type = self._identify_model_type(model)
            if model_type == 'scenario_predictor':
                return await self.explain_scenario_prediction(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'position_sizer':
                return await self.explain_position_sizing(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'leverage_sizer':
                return await self.explain_leverage_decision(model, pd.DataFrame(), features, feature_names, prediction)
            else:
                return await self._explain_generic_model(model, features, feature_names, prediction)
        except Exception as e:
            self.logger.error(f'❌ Failed to explain Tactician prediction: {e}')
            return None

    async def initialize_explainers(self, model: Any, training_data: pd.DataFrame) -> bool:
        """Initialize SHAP and LIME explainers for Tactician models."""
        try:
            self.logger.info('🔧 Initializing Tactician explainers...')
            self.shap_explainer = self._create_shap_explainer(model, training_data)
            if not training_data.empty:
                feature_names = training_data.columns.tolist()
                self.lime_explainer = self._create_lime_explainer(training_data, feature_names)
                self.feature_names = feature_names
            else:
                self.lime_explainer = None
            self.logger.info('✅ Tactician explainers initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize Tactician explainers: {e}')
            return False

    def _identify_model_type(self, model: Any) -> str:
        """Identify the type of Tactician model."""
        model_name = str(type(model).__name__).lower()
        if 'scenario' in model_name or 'predictor' in model_name:
            return 'scenario_predictor'
        elif 'position' in model_name or 'sizer' in model_name:
            return 'position_sizer'
        elif 'leverage' in model_name:
            return 'leverage_sizer'
        else:
            return 'generic'

    async def _explain_generic_model(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a generic Tactician model."""
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
            explanation = ExplanationResult(model_name='Tactician_Generic', prediction = prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'generic', 'feature_groups': self.feature_groups, 'explanation_type': 'generic_prediction'})
            self.save_explanation(explanation)
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain generic model: {e}')
            return None

    def _calculate_prediction_confidence(self, prediction: Any, shap_values: Optional[np.ndarray]) -> float:
        """Calculate confidence score for prediction."""
        try:
            confidence = 0.5
            if shap_values is not None:
                shap_consistency = 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-08)
                confidence = (confidence + shap_consistency) / 2
            if isinstance(prediction, (int, float)):
                if abs(prediction) > 0.1:
                    confidence += 0.1
            elif isinstance(prediction, dict):
                if 'confidence' in prediction:
                    confidence = prediction['confidence']
                elif 'probability' in prediction:
                    confidence = prediction['probability']
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate prediction confidence: {e}')
            return 0.5

    def get_feature_group_importance(self, explanation: ExplanationResult) -> Dict[str, float]:
        """Get importance by feature group."""
        try:
            group_importance = {}
            if explanation.shap_values is not None:
                for group_name, group_features in self.feature_groups.items():
                    group_shap_sum = 0.0
                    group_count = 0
                    for i, feature_name in enumerate(explanation.feature_names):
                        if feature_name in group_features and i < len(explanation.shap_values):
                            group_shap_sum += abs(explanation.shap_values[i])
                            group_count += 1
                    if group_count > 0:
                        group_importance[group_name] = group_shap_sum / group_count
            return group_importance
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate feature group importance: {e}')
            return {}

    def generate_explanation_summary(self, explanation: ExplanationResult) -> str:
        """Generate human-readable explanation summary."""
        try:
            summary_parts = []
            summary_parts.append(f'Model: {explanation.model_name}')
            summary_parts.append(f'Prediction: {explanation.prediction}')
            summary_parts.append(f'Confidence: {explanation.confidence:.2f}')
            if explanation.shap_values is not None:
                feature_importance = list(zip(explanation.feature_names, explanation.shap_values))
                feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
                summary_parts.append('\nTop Contributing Factors:')
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
            self.logger.error(f'❌ Failed to generate explanation summary: {e}')
            return f'Explanation summary generation failed: {e}'