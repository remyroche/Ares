from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import numpy as np
import pandas as pd
'SR (Support/Resistance) model explainer for SHAP and LIME integration.\n\nThis module provides explainability for SR level detection models, enabling\ntraceability of support/resistance decisions back to individual factors.\n'
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

class SRExplainer(BaseExplainer):
    """Explainer for SR level detection models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR explainer."""
        super().__init__(config, 'SR')
        self.sr_config = config.get('sr_levels', {})
        self.explain_config = config.get('explainability', {}).get('sr', {})
        self.explain_level_detection = self.explain_config.get('explain_level_detection', True)
        self.explain_breakout_prediction = self.explain_config.get('explain_breakout_prediction', True)
        self.explain_level_quality = self.explain_config.get('explain_level_quality', True)
        self.explain_strength_calculation = self.explain_config.get('explain_strength_calculation', True)
        self.feature_groups = {'price_features': ['close', 'open', 'high', 'low', 'price_momentum', 'price_volatility'], 'volume_features': ['volume', 'volume_ratio', 'volume_momentum', 'volume_at_level'], 'level_features': ['level_price', 'level_age', 'touch_count', 'bounce_rate', 'isolation_score'], 'technical_indicators': ['rsi', 'macd', 'bb_position', 'atr', 'adx', 'stochastic'], 'market_structure': ['support_levels_count', 'resistance_levels_count', 'level_density'], 'breakout_features': ['breakout_probability', 'breakout_confidence', 'time_to_breakout'], 'quality_features': ['level_strength', 'level_confidence', 'level_reliability']}
        self.level_types = ['support', 'resistance', 'dynamic_support', 'dynamic_resistance']

    async def explain_level_detection(self, sr_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], detected_levels: Optional[List[Dict[str, Any]]]=None) -> ExplanationResult:
        """Explain SR level detection decision."""
        try:
            self.logger.info('🔍 Explaining SR level detection...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(sr_model, market_data)
            if detected_levels is None:
                if hasattr(sr_model, 'detect_levels'):
                    detected_levels = await sr_model.detect_levels(features)
                elif hasattr(sr_model, 'predict'):
                    detected_levels = sr_model.predict(features.reshape(1, -1))
                else:
                    detected_levels = []
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, sr_model, features, feature_names)
            feature_importance = self._extract_feature_importance(sr_model)
            confidence = self._calculate_detection_confidence(detected_levels, shap_values)
            explanation = ExplanationResult(model_name='SR_LevelDetection', prediction = detected_levels, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'level_detection', 'feature_groups': self.feature_groups, 'level_types': self.level_types, 'explanation_type': 'level_detection'})
            self.save_explanation(explanation)
            self.logger.info('✅ SR level detection explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain SR level detection: {e}')
            return None

    async def explain_breakout_prediction(self, sr_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], breakout_prediction: Optional[Dict[str, Any]]=None) -> ExplanationResult:
        """Explain breakout prediction decision."""
        try:
            self.logger.info('🔍 Explaining SR breakout prediction...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(sr_model, market_data)
            if breakout_prediction is None:
                if hasattr(sr_model, 'predict_breakout'):
                    breakout_prediction = await sr_model.predict_breakout(features)
                elif hasattr(sr_model, 'predict_breakouts'):
                    breakout_prediction = await sr_model.predict_breakouts(features)
                elif hasattr(sr_model, 'predict'):
                    breakout_prediction = sr_model.predict(features.reshape(1, -1))
                else:
                    breakout_prediction = {}
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, sr_model, features, feature_names)
            feature_importance = self._extract_feature_importance(sr_model)
            confidence = self._calculate_breakout_confidence(breakout_prediction, shap_values)
            explanation = ExplanationResult(model_name='SR_BreakoutPrediction', prediction = breakout_prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'breakout_prediction', 'feature_groups': self.feature_groups, 'level_types': self.level_types, 'explanation_type': 'breakout_prediction'})
            self.save_explanation(explanation)
            self.logger.info('✅ SR breakout prediction explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain SR breakout prediction: {e}')
            return None

    async def explain_level_quality(self, sr_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], quality_score: Optional[float]=None) -> ExplanationResult:
        """Explain SR level quality assessment."""
        try:
            self.logger.info('🔍 Explaining SR level quality...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(sr_model, market_data)
            if quality_score is None:
                if hasattr(sr_model, 'calculate_quality'):
                    quality_score = await sr_model.calculate_quality(features)
                elif hasattr(sr_model, 'predict_quality'):
                    quality_score = await sr_model.predict_quality(features)
                elif hasattr(sr_model, 'predict'):
                    quality_score = sr_model.predict(features.reshape(1, -1))[0]
                else:
                    quality_score = 0.5
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, sr_model, features, feature_names)
            feature_importance = self._extract_feature_importance(sr_model)
            confidence = self._calculate_quality_confidence(quality_score, shap_values)
            explanation = ExplanationResult(model_name='SR_LevelQuality', prediction = quality_score, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'level_quality', 'feature_groups': self.feature_groups, 'level_types': self.level_types, 'explanation_type': 'level_quality'})
            self.save_explanation(explanation)
            self.logger.info('✅ SR level quality explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain SR level quality: {e}')
            return None

    async def explain_strength_calculation(self, sr_model: Any, market_data: pd.DataFrame, features: np.ndarray, feature_names: List[str], strength_score: Optional[float]=None) -> ExplanationResult:
        """Explain SR level strength calculation."""
        try:
            self.logger.info('🔍 Explaining SR level strength calculation...')
            if self.shap_explainer is None or self.lime_explainer is None:
                await self.initialize_explainers(sr_model, market_data)
            if strength_score is None:
                if hasattr(sr_model, 'calculate_strength'):
                    strength_score = await sr_model.calculate_strength(features)
                elif hasattr(sr_model, 'predict_strength'):
                    strength_score = await sr_model.predict_strength(features)
                elif hasattr(sr_model, 'predict'):
                    strength_score = sr_model.predict(features.reshape(1, -1))[0]
                else:
                    strength_score = 0.5
            shap_values = None
            if self.shap_explainer is not None:
                shap_values = self._calculate_shap_values(self.shap_explainer, features)
            lime_explanation = None
            if self.lime_explainer is not None:
                lime_explanation = self._calculate_lime_explanation(self.lime_explainer, sr_model, features, feature_names)
            feature_importance = self._extract_feature_importance(sr_model)
            confidence = self._calculate_strength_confidence(strength_score, shap_values)
            explanation = ExplanationResult(model_name='SR_LevelStrength', prediction = strength_score, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'level_strength', 'feature_groups': self.feature_groups, 'level_types': self.level_types, 'explanation_type': 'level_strength'})
            self.save_explanation(explanation)
            self.logger.info('✅ SR level strength explained successfully')
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain SR level strength: {e}')
            return None

    async def explain_prediction(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a general SR model prediction."""
        try:
            self.logger.info('🔍 Explaining SR prediction...')
            model_type = self._identify_sr_model_type(model)
            if model_type == 'level_detection':
                return await self.explain_level_detection(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'breakout_prediction':
                return await self.explain_breakout_prediction(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'level_quality':
                return await self.explain_level_quality(model, pd.DataFrame(), features, feature_names, prediction)
            elif model_type == 'level_strength':
                return await self.explain_strength_calculation(model, pd.DataFrame(), features, feature_names, prediction)
            else:
                return await self._explain_generic_sr_model(model, features, feature_names, prediction)
        except Exception as e:
            self.logger.error(f'❌ Failed to explain SR prediction: {e}')
            return None

    async def initialize_explainers(self, model: Any, training_data: pd.DataFrame) -> bool:
        """Initialize SHAP and LIME explainers for SR models."""
        try:
            self.logger.info('🔧 Initializing SR explainers...')
            self.shap_explainer = self._create_shap_explainer(model, training_data)
            if not training_data.empty:
                feature_names = training_data.columns.tolist()
                self.lime_explainer = self._create_lime_explainer(training_data, feature_names)
                self.feature_names = feature_names
            else:
                self.lime_explainer = None
            self.logger.info('✅ SR explainers initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize SR explainers: {e}')
            return False

    def _identify_sr_model_type(self, model: Any) -> str:
        """Identify the type of SR model."""
        model_name = str(type(model).__name__).lower()
        if 'detect' in model_name or 'level' in model_name:
            return 'level_detection'
        elif 'breakout' in model_name or 'predict' in model_name:
            return 'breakout_prediction'
        elif 'quality' in model_name:
            return 'level_quality'
        elif 'strength' in model_name:
            return 'level_strength'
        else:
            return 'generic'

    async def _explain_generic_sr_model(self, model: Any, features: np.ndarray, feature_names: List[str], prediction: Any = None) -> ExplanationResult:
        """Explain a generic SR model."""
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
            explanation = ExplanationResult(model_name='SR_Generic', prediction = prediction, feature_names = feature_names, feature_values = features, shap_values = shap_values, lime_explanation = lime_explanation, feature_importance = feature_importance, confidence = confidence, metadata={'model_type': 'generic', 'feature_groups': self.feature_groups, 'level_types': self.level_types, 'explanation_type': 'generic_prediction'})
            self.save_explanation(explanation)
            return explanation
        except Exception as e:
            self.logger.error(f'❌ Failed to explain generic SR model: {e}')
            return None

    def _calculate_detection_confidence(self, detected_levels: List[Dict[str, Any]], shap_values: Optional[np.ndarray]) -> float:
        """Calculate confidence for level detection."""
        try:
            confidence = 0.5
            if detected_levels:
                level_count = len(detected_levels)
                if level_count > 0:
                    confidence += min(0.3, level_count * 0.05)
                for level in detected_levels:
                    if isinstance(level, dict):
                        if 'confidence' in level:
                            confidence = max(confidence, level['confidence'])
                        elif 'strength' in level:
                            confidence = max(confidence, level['strength'])
            if shap_values is not None:
                shap_consistency = 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-08)
                confidence = (confidence + shap_consistency) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate detection confidence: {e}')
            return 0.5

    def _calculate_breakout_confidence(self, breakout_prediction: Dict[str, Any], shap_values: Optional[np.ndarray]) -> float:
        """Calculate confidence for breakout prediction."""
        try:
            confidence = 0.5
            if isinstance(breakout_prediction, dict):
                if 'confidence' in breakout_prediction:
                    confidence = breakout_prediction['confidence']
                elif 'probability' in breakout_prediction:
                    confidence = breakout_prediction['probability']
                elif 'breakout_probability' in breakout_prediction:
                    confidence = breakout_prediction['breakout_probability']
            if shap_values is not None:
                shap_consistency = 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-08)
                confidence = (confidence + shap_consistency) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate breakout confidence: {e}')
            return 0.5

    def _calculate_quality_confidence(self, quality_score: float, shap_values: Optional[np.ndarray]) -> float:
        """Calculate confidence for quality assessment."""
        try:
            confidence = 0.5
            if isinstance(quality_score, (int, float)):
                if quality_score > 0.7:
                    confidence += 0.2
                elif quality_score > 0.5:
                    confidence += 0.1
                elif quality_score < 0.3:
                    confidence -= 0.1
            if shap_values is not None:
                shap_consistency = 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-08)
                confidence = (confidence + shap_consistency) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate quality confidence: {e}')
            return 0.5

    def _calculate_strength_confidence(self, strength_score: float, shap_values: Optional[np.ndarray]) -> float:
        """Calculate confidence for strength calculation."""
        try:
            confidence = 0.5
            if isinstance(strength_score, (int, float)):
                if strength_score > 0.8:
                    confidence += 0.2
                elif strength_score > 0.6:
                    confidence += 0.1
                elif strength_score < 0.4:
                    confidence -= 0.1
            if shap_values is not None:
                shap_consistency = 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-08)
                confidence = (confidence + shap_consistency) / 2
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate strength confidence: {e}')
            return 0.5

    def get_level_type_importance(self, explanation: ExplanationResult) -> Dict[str, float]:
        """Get feature importance by level type."""
        try:
            level_importance = {}
            return level_importance
        except Exception as e:
            self.logger.error(f'❌ Failed to calculate level type importance: {e}')
            return {}

    def generate_sr_explanation_summary(self, explanation: ExplanationResult) -> str:
        """Generate human-readable SR explanation summary."""
        try:
            summary_parts = []
            summary_parts.append(f'Model: {explanation.model_name}')
            summary_parts.append(f'Prediction: {explanation.prediction}')
            summary_parts.append(f'Confidence: {explanation.confidence:.2f}')
            if isinstance(explanation.prediction, dict):
                if 'breakout_probability' in explanation.prediction:
                    summary_parts.append(f"Breakout Probability: {explanation.prediction['breakout_probability']:.3f}")
                if 'expected_direction' in explanation.prediction:
                    summary_parts.append(f"Expected Direction: {explanation.prediction['expected_direction']}")
                if 'time_to_breakout' in explanation.prediction:
                    summary_parts.append(f"Time to Breakout: {explanation.prediction['time_to_breakout']} bars")
            if explanation.shap_values is not None:
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
            self.logger.error(f'❌ Failed to generate SR explanation summary: {e}')
            return f'SR explanation summary generation failed: {e}'
