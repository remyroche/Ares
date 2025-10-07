#!/usr/bin/env python3
from ...utils.logger import system_logger
from src.core.decorators import handles_errors
"""
Explainability Integration for Enhanced ML Monitoring

Integrates SHAP and LIME explanations with the monitoring system for detailed
model decision explanations.
"""

import time

from .utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
)

from .training.model_interpretability.shap_analyzer import SHAPAnalyzer
from .training.model_interpretability.lime_analyzer import LIMEAnalyzer
import numpy as np
import logging
import typing

@dataclass
class FeatureExplanation:
    """Feature-level explanation from SHAP or LIME."""
    feature_name: str
    feature_value: float
    importance_score: float
    contribution: float
    explanation_type: str  # "shap" or "lime"
    confidence: float = 0.0

@dataclass
class ModelExplanation:
    """Complete model explanation with SHAP and LIME."""
    model_id: str
    prediction: float
    base_value: float
    feature_explanations: List[FeatureExplanation]
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    explanation_confidence: float = 0.0
    processing_time_ms: float = 0.0

@dataclass
class EnsembleExplanation:
    """Ensemble-level explanation aggregating individual model explanations."""
    ensemble_id: str
    final_prediction: float
    model_explanations: List[ModelExplanation]
    consensus_features: List[str]
    disagreement_features: List[str]
    explanation_quality_score: float = 0.0

class ExplainabilityIntegrator:
    """
    Integrates SHAP and LIME explanations with the monitoring system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the explainability integrator."""
        self.config = config
        self.logger = system_logger.getChild("ExplainabilityIntegrator")
        
        # Configuration
        self.explain_config = config.get("explainability_integration", {})
        self.enable_shap = self.explain_config.get("enable_shap", True)
        self.enable_lime = self.explain_config.get("enable_lime", True)
        self.max_features_explained = self.explain_config.get("max_features_explained", 20)
        self.explanation_cache_size = self.explain_config.get("explanation_cache_size", 1000)
        
        # Initialize analyzers
        self.shap_analyzer = None
        self.lime_analyzer = None
        self._initialize_analyzers()
        
        # Explanation cache
        self.explanation_cache: Dict[str, ModelExplanation] = {}
        
        self.logger.info("Explainability Integrator initialized")
    
    def _initialize_analyzers(self):
        """Initialize SHAP and LIME analyzers."""
        try:
            if self.enable_shap:
                self.shap_analyzer = SHAPAnalyzer(self.config)
                if not self.shap_analyzer.shap_available:
                    self.logger.warning("SHAP analyzer initialized but SHAP library not available")
                    self.shap_analyzer = None
            
            if self.enable_lime:
                self.lime_analyzer = LIMEAnalyzer(self.config)
                if not self.lime_analyzer.lime_available:
                    self.logger.warning("LIME analyzer initialized but LIME library not available")
                    self.lime_analyzer = None
            
            self.logger.info("Explainability analyzers initialized")
            
        except ImportError as e:
            self.logger.error(f"Could not initialize explainability analyzers: {e}")
            self.shap_analyzer = None
            self.lime_analyzer = None
    
    @handles_errors(default_return = None, context="explainability_integrator.explain_model_prediction")
    async def explain_model_prediction(self, model_id: str, model: Any, 
                                    features: np.ndarray, feature_names: List[str],
                                    prediction: float) -> ModelExplanation:
        """Generate comprehensive explanation for a model prediction."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(model_id, features, prediction)
            if cache_key in self.explanation_cache:
                self.logger.debug(f"Using cached explanation for {model_id}")
                return self.explanation_cache[cache_key]
            
            # Generate explanations
            feature_explanations = []
            shap_values = None
            lime_explanation = None
            
            # SHAP explanation
            if self.shap_analyzer and self.shap_analyzer.shap_available:
                try:
                    shap_result = await self._generate_shap_explanation(
                        model, features, feature_names, model_id
                    )
                    if shap_result:
                        feature_explanations.extend(shap_result['feature_explanations'])
                        shap_values = shap_result['shap_values']
                except Exception as e:
                    self.logger.warning(f"SHAP explanation failed for {model_id}: {e}")
            
            # LIME explanation
            if self.lime_analyzer and self.lime_analyzer.lime_available:
                try:
                    lime_result = await self._generate_lime_explanation(
                        model, features, feature_names, model_id
                    )
                    if lime_result:
                        # Merge LIME explanations with SHAP
                        lime_explanations = lime_result['feature_explanations']
                        feature_explanations = self._merge_explanations(
                            feature_explanations, lime_explanations
                        )
                        lime_explanation = lime_result['lime_explanation']
                except Exception as e:
                    self.logger.warning(f"LIME explanation failed for {model_id}: {e}")
            
            # Calculate explanation confidence
            explanation_confidence = self._calculate_explanation_confidence(
                feature_explanations, shap_values, lime_explanation
            )
            
            # Create model explanation
            model_explanation = ModelExplanation(
                model_id = model_id,
                prediction = prediction,
                base_value = 0.0,  # Will be updated if SHAP provides it
                feature_explanations = feature_explanations[:self.max_features_explained],
                shap_values = shap_values,
                lime_explanation = lime_explanation,
                explanation_confidence = explanation_confidence,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Cache the explanation
            self._cache_explanation(cache_key, model_explanation)
            
            self.logger.debug(
                f"Generated explanation for {model_id}: "
                f"{len(feature_explanations)} features, "
                f"confidence={explanation_confidence:.3f}, "
                f"time={model_explanation.processing_time_ms:.1f}ms"
            )
            
            return model_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation for {model_id}: {e}")
            # Return minimal explanation
            return ModelExplanation(
                model_id = model_id,
                prediction = prediction,
                base_value = 0.0,
                feature_explanations=[],
                explanation_confidence = 0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _generate_shap_explanation(self, model: Any, features: np.ndarray,
                                    feature_names: List[str], model_id: str) -> Optional[Dict[str, Any]]:
        """Generate SHAP explanation for a prediction."""
        try:
            if not self.shap_analyzer or not self.shap_analyzer.shap_available:
                return None
            
            # Prepare features for SHAP
            if len(features.shape) == 1:
                features_reshaped = features.reshape(1, -1)
            else:
                features_reshaped = features
            
            # Generate SHAP values
            shap_values = await self.shap_analyzer.explain_prediction(
                model, features_reshaped, model_id
            )
            
            if not shap_values:
                return None
            
            # Extract feature explanations
            feature_explanations = []
            if 'values' in shap_values and 'feature_names' in shap_values:
                values = shap_values['values']
                names = shap_values['feature_names']
                
                for i, (name, value) in enumerate(zip(names, values)):
                    if i < len(features_reshaped[0]):
                        feature_explanations.append(FeatureExplanation(
                            feature_name = name,
                            feature_value = float(features_reshaped[0][i]),
                            importance_score = abs(float(value)),
                            contribution = float(value),
                            explanation_type="shap",
                            confidence = 0.8  # SHAP confidence
                        ))
            
            return {
                'feature_explanations': feature_explanations,
                'shap_values': shap_values
            }
            
        except Exception as e:
            self.logger.error(f"SHAP explanation generation failed: {e}")
            return None
    
    async def _generate_lime_explanation(self, model: Any, features: np.ndarray,
                                    feature_names: List[str], model_id: str) -> Optional[Dict[str, Any]]:
        """Generate LIME explanation for a prediction."""
        try:
            if not self.lime_analyzer or not self.lime_analyzer.lime_available:
                return None
            
            # Prepare features for LIME
            if len(features.shape) == 1:
                features_reshaped = features.reshape(1, -1)
            else:
                features_reshaped = features
            
            # Generate LIME explanation
            lime_explanation = await self.lime_analyzer.explain_prediction(
                model, features_reshaped, model_id
            )
            
            if not lime_explanation:
                return None
            
            # Extract feature explanations
            feature_explanations = []
            if 'explanation' in lime_explanation:
                explanation = lime_explanation['explanation']
                
                for item in explanation:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        feature_name = str(item[0])
                        importance = float(item[1])
                        
                        # Find corresponding feature value
                        feature_value = 0.0
                        if feature_name in feature_names:
                            idx = feature_names.index(feature_name)
                            if idx < len(features_reshaped[0]):
                                feature_value = float(features_reshaped[0][idx])
                        
                        feature_explanations.append(FeatureExplanation(
                            feature_name = feature_name,
                            feature_value = feature_value,
                            importance_score = abs(importance),
                            contribution = importance,
                            explanation_type="lime",
                            confidence = 0.7  # LIME confidence
                        ))
            
            return {
                'feature_explanations': feature_explanations,
                'lime_explanation': lime_explanation
            }
            
        except Exception as e:
            self.logger.error(f"LIME explanation generation failed: {e}")
            return None
    
    def _merge_explanations(self, shap_explanations: List[FeatureExplanation],
                        lime_explanations: List[FeatureExplanation]) -> List[FeatureExplanation]:
        """Merge SHAP and LIME explanations for comprehensive feature importance."""
        merged = {}
        
        # Add SHAP explanations
        for exp in shap_explanations:
            merged[exp.feature_name] = exp
        
        # Merge with LIME explanations
        for exp in lime_explanations:
            if exp.feature_name in merged:
                # Average the importance scores
                existing = merged[exp.feature_name]
                merged[exp.feature_name] = FeatureExplanation(
                    feature_name = exp.feature_name,
                    feature_value = exp.feature_value,
                    importance_score=(existing.importance_score + exp.importance_score) / 2,
                    contribution=(existing.contribution + exp.contribution) / 2,
                    explanation_type="combined",
                    confidence=(existing.confidence + exp.confidence) / 2
                )
            else:
                merged[exp.feature_name] = exp
        
        # Sort by importance and return
        sorted_explanations = sorted(
            merged.values(), 
            key = lambda x: x.importance_score, 
            reverse = True
        )
        
        return sorted_explanations
    
    def _calculate_explanation_confidence(self, feature_explanations: List[FeatureExplanation],
                                        shap_values: Optional[Dict[str, Any]],
                                        lime_explanation: Optional[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the explanation."""
        confidence_factors = []
        
        # Factor 1: Number of features explained
        if feature_explanations:
            feature_coverage = min(1.0, len(feature_explanations) / 10.0)
            confidence_factors.append(feature_coverage)
        
        # Factor 2: SHAP availability
        if shap_values:
            confidence_factors.append(0.8)
        
        # Factor 3: LIME availability
        if lime_explanation:
            confidence_factors.append(0.7)
        
        # Factor 4: Explanation consistency
        if len(feature_explanations) > 1:
            importance_scores = [exp.importance_score for exp in feature_explanations]
            if importance_scores:
                score_variance = np.var(importance_scores)
                consistency = max(0.0, 1.0 - score_variance)
                confidence_factors.append(consistency)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _generate_cache_key(self, model_id: str, features: np.ndarray, prediction: float) -> str:
        """Generate cache key for explanation."""
        # Create a hash of the model_id, features, and prediction
        feature_hash = hash(tuple(features.flatten()))
        return f"{model_id}_{feature_hash}_{prediction:.6f}"
    
    def _cache_explanation(self, cache_key: str, explanation: ModelExplanation):
        """Cache explanation with size limit."""
        self.explanation_cache[cache_key] = explanation
        
        # Maintain cache size
        if len(self.explanation_cache) > self.explanation_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.explanation_cache.keys())[:len(self.explanation_cache) - self.explanation_cache_size]
            for key in oldest_keys:
                del self.explanation_cache[key]
    
    @handles_errors(default_return = None, context="explainability_integrator.explain_ensemble_prediction")
    async def explain_ensemble_prediction(self, ensemble_id: str, 
                                        model_explanations: List[ModelExplanation],
                                        final_prediction: float) -> EnsembleExplanation:
        """Generate ensemble-level explanation from individual model explanations."""
        try:
            # Find consensus features (features that most models agree on)
            feature_agreement = {}
            all_features = set()
            
            for model_exp in model_explanations:
                for feature_exp in model_exp.feature_explanations:
                    feature_name = feature_exp.feature_name
                    all_features.add(feature_name)
                    
                    if feature_name not in feature_agreement:
                        feature_agreement[feature_name] = {
                            'count': 0,
                            'total_importance': 0.0,
                            'contributions': []
                        }
                    
                    feature_agreement[feature_name]['count'] += 1
                    feature_agreement[feature_name]['total_importance'] += feature_exp.importance_score
                    feature_agreement[feature_name]['contributions'].append(feature_exp.contribution)
            
            # Calculate consensus and disagreement features
            consensus_features = []
            disagreement_features = []
            
            for feature_name, agreement in feature_agreement.items():
                agreement_ratio = agreement['count'] / len(model_explanations)
                avg_importance = agreement['total_importance'] / agreement['count']
                
                if agreement_ratio >= 0.6:  # 60% of models agree
                    consensus_features.append(feature_name)
                elif agreement_ratio <= 0.3:  # Less than 30% agree
                    disagreement_features.append(feature_name)
            
            # Calculate explanation quality score
            explanation_quality = self._calculate_ensemble_explanation_quality(
                model_explanations, consensus_features, disagreement_features
            )
            
            ensemble_explanation = EnsembleExplanation(
                ensemble_id = ensemble_id,
                final_prediction = final_prediction,
                model_explanations = model_explanations,
                consensus_features = consensus_features,
                disagreement_features = disagreement_features,
                explanation_quality_score = explanation_quality
            )
            
            self.logger.debug(
                f"Generated ensemble explanation for {ensemble_id}: "
                f"{len(consensus_features)} consensus, "
                f"{len(disagreement_features)} disagreement features, "
                f"quality={explanation_quality:.3f}"
            )
            
            return ensemble_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble explanation: {e}")
            return EnsembleExplanation(
                ensemble_id = ensemble_id,
                final_prediction = final_prediction,
                model_explanations = model_explanations,
                consensus_features=[],
                disagreement_features=[],
                explanation_quality_score = 0.0
            )
    
    def _calculate_ensemble_explanation_quality(self, model_explanations: List[ModelExplanation],
                                            consensus_features: List[str],
                                            disagreement_features: List[str]) -> float:
        """Calculate quality score for ensemble explanation."""
        quality_factors = []
        
        # Factor 1: Model explanation confidence
        if model_explanations:
            avg_confidence = np.mean([exp.explanation_confidence for exp in model_explanations])
            quality_factors.append(avg_confidence)
        
        # Factor 2: Consensus ratio
        total_features = len(consensus_features) + len(disagreement_features)
        if total_features > 0:
            consensus_ratio = len(consensus_features) / total_features
            quality_factors.append(consensus_ratio)
        
        # Factor 3: Number of models with explanations
        models_with_explanations = sum(1 for exp in model_explanations if exp.feature_explanations)
        explanation_coverage = models_with_explanations / len(model_explanations)
        quality_factors.append(explanation_coverage)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def get_explanation_stats(self) -> Dict[str, Any]:
        """Get statistics about explanation generation."""
        return {
            'cache_size': len(self.explanation_cache),
            'cache_limit': self.explanation_cache_size,
            'shap_available': self.shap_analyzer is not None and self.shap_analyzer.shap_available,
            'lime_available': self.lime_analyzer is not None and self.lime_analyzer.lime_available,
            'max_features_explained': self.max_features_explained,
        }