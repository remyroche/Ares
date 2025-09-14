"""
Model Interpretability Utilities

This module provides comprehensive model interpretability and explainability
capabilities for trading models, including SHAP, LIME, and custom explanations.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import joblib
from pathlib import Path

# Core utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.logger import system_logger

# Optional imports for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance, partial_dependence
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False

class InterpretabilityMethod(Enum):
    """Interpretability method types."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    FEATURE_IMPORTANCE = "feature_importance"
    CUSTOM = "custom"

class ExplanationType(Enum):
    """Explanation type categories."""
    GLOBAL = "global"
    LOCAL = "local"
    INTERACTION = "interaction"
    CAUSAL = "causal"

@dataclass
class ExplanationResult:
    """Model explanation result."""
    method: InterpretabilityMethod
    explanation_type: ExplanationType
    feature_names: List[str]
    feature_importance: Dict[str, float]
    explanations: Dict[str, Any]
    confidence_score: float
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterpretabilityReport:
    """Comprehensive interpretability report."""
    model_id: str
    model_name: str
    explanation_results: List[ExplanationResult]
    global_insights: List[str]
    local_insights: List[str]
    recommendations: List[str]
    overall_interpretability_score: float
    timestamp: datetime

class ModelInterpretabilityEngine:
    """
    Comprehensive model interpretability engine.
    
    This engine provides multiple interpretability methods including:
    - SHAP (SHapley Additive exPlanations)
    - LIME (Local Interpretable Model-agnostic Explanations)
    - Permutation importance
    - Partial dependence plots
    - Feature importance analysis
    - Custom explanation methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model interpretability engine."""
        self.config = config
        self.logger = system_logger.getChild('ModelInterpretabilityEngine')
        
        # Interpretability configuration
        self.interpretability_config = config.get('interpretability', {})
        self.enabled_methods = self.interpretability_config.get('enabled_methods', [
            InterpretabilityMethod.FEATURE_IMPORTANCE.value,
            InterpretabilityMethod.PERMUTATION_IMPORTANCE.value
        ])
        self.sample_size = self.interpretability_config.get('sample_size', 1000)
        self.max_features = self.interpretability_config.get('max_features', 50)
        self.confidence_threshold = self.interpretability_config.get('confidence_threshold', 0.7)
        
        # Method availability
        self.method_availability = {
            InterpretabilityMethod.SHAP: SHAP_AVAILABLE,
            InterpretabilityMethod.LIME: LIME_AVAILABLE,
            InterpretabilityMethod.PERMUTATION_IMPORTANCE: SKLEARN_INSPECTION_AVAILABLE,
            InterpretabilityMethod.PARTIAL_DEPENDENCE: SKLEARN_INSPECTION_AVAILABLE,
            InterpretabilityMethod.FEATURE_IMPORTANCE: True,
            InterpretabilityMethod.CUSTOM: True
        }
        
        # Results storage
        self.interpretability_results: Dict[str, InterpretabilityReport] = {}

    # Temporarily disabled problematic decorators
    # @handles_errors(Exception, fallback=False, log_level='ERROR')
    # @validates(strict=True)
    # @traced(span_name="explain_model")
    # @log_execution_time  # Temporarily disabled due to import conflicts
    async def explain_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        model_name: str = "model",
        methods: Optional[List[InterpretabilityMethod]] = None
    ) -> InterpretabilityReport:
        """
        Generate comprehensive model explanations.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            feature_names: Feature names
            model_name: Model name
            methods: List of interpretability methods to use
            
        Returns:
            InterpretabilityReport: Comprehensive interpretability report
        """
        try:
            self.logger.info(f"🔍 Starting model interpretability analysis for {model_name}")
            
            if methods is None:
                methods = [InterpretabilityMethod(method) for method in self.enabled_methods]
            
            # Filter available methods
            available_methods = [method for method in methods if self.method_availability.get(method, False)]
            
            if not available_methods:
                self.logger.warning("⚠️ No interpretability methods available")
                return self._create_empty_report(model_name)
            
            self.logger.info(f"📊 Using interpretability methods: {[m.value for m in available_methods]}")
            
            # Sample data if too large
            X_train_sample, X_test_sample = self._sample_data(X_train, X_test)
            
            # Generate explanations
            explanation_results = []
            for method in available_methods:
                try:
                    result = await self._generate_explanation(
                        method, model, X_train_sample, X_test_sample, 
                        y_train, y_test, feature_names
                    )
                    if result:
                        explanation_results.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate {method.value} explanation: {e}")
                    continue
            
            # Generate insights and recommendations
            global_insights = self._generate_global_insights(explanation_results)
            local_insights = self._generate_local_insights(explanation_results)
            recommendations = self._generate_recommendations(explanation_results)
            
            # Calculate overall interpretability score
            interpretability_score = self._calculate_interpretability_score(explanation_results)
            
            # Create report
            report = InterpretabilityReport(
                model_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_name=model_name,
                explanation_results=explanation_results,
                global_insights=global_insights,
                local_insights=local_insights,
                recommendations=recommendations,
                overall_interpretability_score=interpretability_score,
                timestamp=datetime.now()
            )
            
            self.interpretability_results[report.model_id] = report
            
            self.logger.info(f"✅ Model interpretability analysis completed for {model_name}")
            self.logger.info(f"📊 Overall interpretability score: {interpretability_score:.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Model interpretability analysis failed: {e}")
            raise

    def _sample_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample data if too large for interpretability analysis."""
        if len(X_train) > self.sample_size:
            X_train_sample = X_train.sample(n=self.sample_size, random_state=42)
        else:
            X_train_sample = X_train
        
        if len(X_test) > self.sample_size:
            X_test_sample = X_test.sample(n=self.sample_size, random_state=42)
        else:
            X_test_sample = X_test
        
        return X_train_sample, X_test_sample

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_explanation")
    async def _generate_explanation(
        self,
        method: InterpretabilityMethod,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str]
    ) -> Optional[ExplanationResult]:
        """Generate explanation using specific method."""
        start_time = time.time()
        
        try:
            if method == InterpretabilityMethod.SHAP:
                return await self._generate_shap_explanation(model, X_train, X_test, feature_names, start_time)
            elif method == InterpretabilityMethod.LIME:
                return await self._generate_lime_explanation(model, X_train, X_test, feature_names, start_time)
            elif method == InterpretabilityMethod.PERMUTATION_IMPORTANCE:
                return await self._generate_permutation_importance_explanation(model, X_test, y_test, feature_names, start_time)
            elif method == InterpretabilityMethod.PARTIAL_DEPENDENCE:
                return await self._generate_partial_dependence_explanation(model, X_train, feature_names, start_time)
            elif method == InterpretabilityMethod.FEATURE_IMPORTANCE:
                return await self._generate_feature_importance_explanation(model, feature_names, start_time)
            elif method == InterpretabilityMethod.CUSTOM:
                return await self._generate_custom_explanation(model, X_train, X_test, feature_names, start_time)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating {method.value} explanation: {e}")
            return None

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_shap_explanation")
    async def _generate_shap_explanation(
        self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
        feature_names: List[str], start_time: float
    ) -> Optional[ExplanationResult]:
        """Generate SHAP explanation."""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            else:
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)
            
            # Calculate feature importance
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values_mean = np.abs(shap_values.values)
            
            feature_importance = dict(zip(feature_names, np.mean(shap_values_mean, axis=0)))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:self.max_features])
            
            computation_time = time.time() - start_time
            
            return ExplanationResult(
                method=InterpretabilityMethod.SHAP,
                explanation_type=ExplanationType.GLOBAL,
                feature_names=feature_names,
                feature_importance=top_features,
                explanations={
                    'shap_values': shap_values,
                    'explainer': explainer,
                    'feature_importance_raw': feature_importance
                },
                confidence_score=0.9,  # SHAP is generally reliable
                computation_time=computation_time,
                metadata={'sample_size': len(X_test)}
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ SHAP explanation failed: {e}")
            return None

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_lime_explanation")
    async def _generate_lime_explanation(
        self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
        feature_names: List[str], start_time: float
    ) -> Optional[ExplanationResult]:
        """Generate LIME explanation."""
        if not LIME_AVAILABLE:
            return None
        
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                mode='regression' if not hasattr(model, 'predict_proba') else 'classification',
                random_state=42
            )
            
            # Generate explanations for sample of test data
            sample_size = min(10, len(X_test))
            explanations = {}
            feature_importance_scores = {}
            
            for i in range(sample_size):
                explanation = explainer.explain_instance(
                    X_test.iloc[i].values,
                    model.predict,
                    num_features=min(10, len(feature_names))
                )
                explanations[f'sample_{i}'] = explanation
                
                # Aggregate feature importance
                for feature, importance in explanation.as_list():
                    if feature in feature_importance_scores:
                        feature_importance_scores[feature] += abs(importance)
                    else:
                        feature_importance_scores[feature] = abs(importance)
            
            # Normalize feature importance
            total_importance = sum(feature_importance_scores.values())
            if total_importance > 0:
                feature_importance = {k: v/total_importance for k, v in feature_importance_scores.items()}
            else:
                feature_importance = feature_importance_scores
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:self.max_features])
            
            computation_time = time.time() - start_time
            
            return ExplanationResult(
                method=InterpretabilityMethod.LIME,
                explanation_type=ExplanationType.LOCAL,
                feature_names=feature_names,
                feature_importance=top_features,
                explanations=explanations,
                confidence_score=0.8,  # LIME is generally reliable for local explanations
                computation_time=computation_time,
                metadata={'sample_size': sample_size}
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ LIME explanation failed: {e}")
            return None

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_permutation_importance_explanation")
    async def _generate_permutation_importance_explanation(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
        feature_names: List[str], start_time: float
    ) -> Optional[ExplanationResult]:
        """Generate permutation importance explanation."""
        if not SKLEARN_INSPECTION_AVAILABLE:
            return None
        
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, perm_importance.importances_mean))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:self.max_features])
            
            computation_time = time.time() - start_time
            
            return ExplanationResult(
                method=InterpretabilityMethod.PERMUTATION_IMPORTANCE,
                explanation_type=ExplanationType.GLOBAL,
                feature_names=feature_names,
                feature_importance=top_features,
                explanations={
                    'permutation_importance': perm_importance,
                    'feature_importance_raw': feature_importance
                },
                confidence_score=0.85,  # Permutation importance is reliable
                computation_time=computation_time,
                metadata={'n_repeats': 10}
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Permutation importance explanation failed: {e}")
            return None

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_partial_dependence_explanation")
    async def _generate_partial_dependence_explanation(
        self, model: Any, X_train: pd.DataFrame, feature_names: List[str], start_time: float
    ) -> Optional[ExplanationResult]:
        """Generate partial dependence explanation."""
        if not SKLEARN_INSPECTION_AVAILABLE:
            return None
        
        try:
            # Select top features for partial dependence
            top_features = feature_names[:min(5, len(feature_names))]
            
            # Calculate partial dependence
            pd_results = {}
            for feature in top_features:
                try:
                    pd_values = partial_dependence(model, X_train, [feature])
                    pd_results[feature] = pd_values
                except Exception as e:
                    self.logger.warning(f"⚠️ Partial dependence failed for {feature}: {e}")
                    continue
            
            # Create feature importance based on partial dependence variance
            feature_importance = {}
            for feature, pd_result in pd_results.items():
                if hasattr(pd_result, 'values'):
                    variance = np.var(pd_result.values)
                    feature_importance[feature] = variance
            
            computation_time = time.time() - start_time
            
            return ExplanationResult(
                method=InterpretabilityMethod.PARTIAL_DEPENDENCE,
                explanation_type=ExplanationType.GLOBAL,
                feature_names=feature_names,
                feature_importance=feature_importance,
                explanations=pd_results,
                confidence_score=0.8,
                computation_time=computation_time,
                metadata={'features_analyzed': len(pd_results)}
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Partial dependence explanation failed: {e}")
            return None

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_feature_importance_explanation")
    async def _generate_feature_importance_explanation(
        self, model: Any, feature_names: List[str], start_time: float
    ) -> Optional[ExplanationResult]:
        """Generate feature importance explanation."""
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                feature_importance = dict(zip(feature_names, coef))
            else:
                # Fallback: equal importance
                feature_importance = {name: 1.0/len(feature_names) for name in feature_names}
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:self.max_features])
            
            computation_time = time.time() - start_time
            
            return ExplanationResult(
                method=InterpretabilityMethod.FEATURE_IMPORTANCE,
                explanation_type=ExplanationType.GLOBAL,
                feature_names=feature_names,
                feature_importance=top_features,
                explanations={'feature_importance_raw': feature_importance},
                confidence_score=0.7,  # Feature importance can be model-dependent
                computation_time=computation_time,
                metadata={'method': 'built_in' if hasattr(model, 'feature_importances_') else 'coefficients'}
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance explanation failed: {e}")
            return None

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced(span_name="generate_custom_explanation")
    async def _generate_custom_explanation(
        self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
        feature_names: List[str], start_time: float
    ) -> Optional[ExplanationResult]:
        """Generate custom explanation."""
        try:
            # Custom explanation: variance-based feature importance
            feature_importance = {}
            
            for feature in feature_names:
                if feature in X_train.columns:
                    # Calculate variance of the feature
                    variance = X_train[feature].var()
                    feature_importance[feature] = variance
            
            # Normalize importance scores
            total_variance = sum(feature_importance.values())
            if total_variance > 0:
                feature_importance = {k: v/total_variance for k, v in feature_importance.items()}
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:self.max_features])
            
            computation_time = time.time() - start_time
            
            return ExplanationResult(
                method=InterpretabilityMethod.CUSTOM,
                explanation_type=ExplanationType.GLOBAL,
                feature_names=feature_names,
                feature_importance=top_features,
                explanations={'feature_importance_raw': feature_importance},
                confidence_score=0.6,  # Custom method is less reliable
                computation_time=computation_time,
                metadata={'method': 'variance_based'}
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Custom explanation failed: {e}")
            return None

    def _generate_global_insights(self, explanation_results: List[ExplanationResult]) -> List[str]:
        """Generate global insights from explanation results."""
        insights = []
        
        if not explanation_results:
            return insights
        
        # Aggregate feature importance across methods
        all_importances = {}
        for result in explanation_results:
            for feature, importance in result.feature_importance.items():
                if feature in all_importances:
                    all_importances[feature].append(importance)
                else:
                    all_importances[feature] = [importance]
        
        # Calculate average importance
        avg_importances = {feature: safe_mean(importances) for feature, importances in all_importances.items()}
        
        # Sort by average importance
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
        
        # Generate insights
        if sorted_features:
            top_feature, top_importance = sorted_features[0]
            insights.append(f"Most important feature: {top_feature} (importance: {top_importance:.3f})")
        
        if len(sorted_features) >= 3:
            top_3_features = [f[0] for f in sorted_features[:3]]
            insights.append(f"Top 3 features: {', '.join(top_3_features)}")
        
        # Feature importance distribution
        importance_values = list(avg_importances.values())
        if importance_values:
            importance_std = safe_std(importance_values)
            if importance_std < 0.1:
                insights.append("Feature importance is relatively uniform across features")
            elif importance_std > 0.3:
                insights.append("Feature importance shows high variance - some features are much more important")
        
        return insights

    def _generate_local_insights(self, explanation_results: List[ExplanationResult]) -> List[str]:
        """Generate local insights from explanation results."""
        insights = []
        
        # Look for LIME results (local explanations)
        lime_results = [r for r in explanation_results if r.method == InterpretabilityMethod.LIME]
        
        if lime_results:
            insights.append("Local explanations available via LIME method")
            
            # Analyze local explanation consistency
            for result in lime_results:
                if 'explanations' in result.explanations:
                    num_explanations = len(result.explanations['explanations'])
                    insights.append(f"Generated {num_explanations} local explanations")
        
        return insights

    def _generate_recommendations(self, explanation_results: List[ExplanationResult]) -> List[str]:
        """Generate recommendations based on explanation results."""
        recommendations = []
        
        if not explanation_results:
            recommendations.append("No interpretability results available - consider enabling more methods")
            return recommendations
        
        # Check method coverage
        methods_used = [r.method.value for r in explanation_results]
        recommendations.append(f"Used interpretability methods: {', '.join(methods_used)}")
        
        # Check confidence scores
        avg_confidence = safe_mean([r.confidence_score for r in explanation_results])
        if avg_confidence < 0.7:
            recommendations.append("Low confidence in explanations - consider using more reliable methods")
        else:
            recommendations.append("High confidence in explanations - results are reliable")
        
        # Feature importance recommendations
        all_features = set()
        for result in explanation_results:
            all_features.update(result.feature_importance.keys())
        
        if len(all_features) > 50:
            recommendations.append("Large number of features - consider feature selection")
        
        # Method-specific recommendations
        if InterpretabilityMethod.SHAP in [r.method for r in explanation_results]:
            recommendations.append("SHAP explanations available - good for global and local interpretability")
        
        if InterpretabilityMethod.LIME in [r.method for r in explanation_results]:
            recommendations.append("LIME explanations available - good for local interpretability")
        
        return recommendations

    def _calculate_interpretability_score(self, explanation_results: List[ExplanationResult]) -> float:
        """Calculate overall interpretability score."""
        if not explanation_results:
            return 0.0
        
        # Factors for interpretability score
        method_diversity = len(set(r.method for r in explanation_results)) / len(InterpretabilityMethod)
        avg_confidence = safe_mean([r.confidence_score for r in explanation_results])
        coverage = len(explanation_results) / len(self.enabled_methods) if self.enabled_methods else 0
        
        # Weighted score
        interpretability_score = (
            method_diversity * 0.3 +
            avg_confidence * 0.4 +
            coverage * 0.3
        )
        
        return min(1.0, max(0.0, interpretability_score))

    def _create_empty_report(self, model_name: str) -> InterpretabilityReport:
        """Create empty interpretability report."""
        return InterpretabilityReport(
            model_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            explanation_results=[],
            global_insights=["No interpretability methods available"],
            local_insights=[],
            recommendations=["Install required packages (shap, lime) for interpretability"],
            overall_interpretability_score=0.0,
            timestamp=datetime.now()
        )

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced(span_name="save_interpretability_report")
    async def save_interpretability_report(self, report: InterpretabilityReport, output_dir: str) -> str:
        """Save interpretability report to file."""
        ensure_directory(output_dir)
        
        report_data = {
            'model_id': report.model_id,
            'model_name': report.model_name,
            'explanation_results': [
                {
                    'method': result.method.value,
                    'explanation_type': result.explanation_type.value,
                    'feature_names': result.feature_names,
                    'feature_importance': result.feature_importance,
                    'confidence_score': result.confidence_score,
                    'computation_time': result.computation_time,
                    'metadata': result.metadata
                }
                for result in report.explanation_results
            ],
            'global_insights': report.global_insights,
            'local_insights': report.local_insights,
            'recommendations': report.recommendations,
            'overall_interpretability_score': report.overall_interpretability_score,
            'timestamp': report.timestamp.isoformat()
        }
        
        report_file = f"{output_dir}/interpretability_report_{report.model_name}_{report.model_id}.json"
        safe_json_dump(report_data, report_file, indent=2)
        
        self.logger.info(f"💾 Interpretability report saved to: {report_file}")
        return report_file