
from typing import List
from typing import Any
from typing import Dict
import pandas as pd
from typing import Optional
import numpy as np
#!/usr/bin/env python3
"""Base explainer classes for SHAP and LIME integration.

This module provides the foundation for model explainability across all ML models
in the trading system, enabling traceability of trade decisions back to individual factors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from .utils.logger import system_logger

# SHAP imports with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, SHAP explanations disabled")

# LIME imports with fallback
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available, LIME explanations disabled")


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    model_name: str
    prediction: Any
    feature_names: List[str]
    feature_values: np.ndarray
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TradeDecisionTrace:
    """Trace of a trade decision back to individual factors."""
    decision_id: str
    timestamp: datetime
    decision_type: str  # 'entry', 'exit', 'position_size', 'leverage'
    final_decision: Any
    confidence: float
    
    # Model explanations
    tactician_explanation: Optional[ExplanationResult] = None
    hmm_explanation: Optional[ExplanationResult] = None
    sr_explanation: Optional[ExplanationResult] = None
    analyst_explanation: Optional[ExplanationResult] = None
    
    # Aggregated factors
    top_contributing_factors: List[Dict[str, Any]] = None
    risk_factors: List[Dict[str, Any]] = None
    opportunity_factors: List[Dict[str, Any]] = None
    
    # Metadata
    market_conditions: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.top_contributing_factors is None:
            self.top_contributing_factors = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.opportunity_factors is None:
            self.opportunity_factors = []
        if self.market_conditions is None:
            self.market_conditions = {}
        if self.metadata is None:
            self.metadata = {}


class BaseExplainer(ABC):
    """Base class for model explainers."""
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """Initialize base explainer."""
        self.config = config
        self.model_name = model_name
        self.logger = system_logger.getChild(f"Explainer_{model_name}")
        
        # Explanation settings
        self.explain_config = config.get("explainability", {})
        self.enable_shap = self.explain_config.get("enable_shap", True) and SHAP_AVAILABLE
        self.enable_lime = self.explain_config.get("enable_lime", True) and LIME_AVAILABLE
        self.max_features = self.explain_config.get("max_features", 20)
        
        # Storage
        self.explanations_storage = Path(self.explain_config.get("storage_path", "data/explanations"))
        self.explanations_storage.mkdir(parents=True, exist_ok=True)
        
        # SHAP explainer (will be initialized per model)
        self.shap_explainer = None
        self.lime_explainer = None
        
    @abstractmethod
    async def explain_prediction(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction: Any = None
    ) -> ExplanationResult:
        """Explain a model prediction."""
        pass
    
    @abstractmethod
    async def initialize_explainers(self, model: Any, training_data: pd.DataFrame) -> bool:
        """Initialize SHAP and LIME explainers for the model."""
        pass
    
    def _create_shap_explainer(self, model: Any, training_data: pd.DataFrame) -> Optional[Any]:
        """Create SHAP explainer for the model."""
        if not self.enable_shap or not SHAP_AVAILABLE:
            return None
            
        try:
            # Determine explainer type based on model
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    # For other models, use KernelExplainer with sample
                    sample_size = min(100, len(training_data))
                    sample_data = training_data.sample(n=sample_size, random_state=42)
                    explainer = shap.KernelExplainer(model.predict_proba, sample_data)
            else:
                # Regression models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    sample_size = min(100, len(training_data))
                    sample_data = training_data.sample(n=sample_size, random_state=42)
                    explainer = shap.KernelExplainer(model.predict, sample_data)
            
            self.logger.info(f"✅ SHAP explainer created for {self.model_name}")
            return explainer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create SHAP explainer: {e}")
            return None
    
    def _create_lime_explainer(self, training_data: pd.DataFrame, feature_names: List[str]) -> Optional[Any]:
        """Create LIME explainer for the model."""
        if not self.enable_lime or not LIME_AVAILABLE:
            return None
            
        try:
            explainer = LimeTabularExplainer(
                training_data.values,
                feature_names=feature_names,
                class_names=['class_0', 'class_1'] if len(training_data.columns) > 1 else None,
                mode='classification' if len(training_data.columns) > 1 else 'regression',
                discretize_continuous=True
            )
            
            self.logger.info(f"✅ LIME explainer created for {self.model_name}")
            return explainer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create LIME explainer: {e}")
            return None
    
    def _calculate_shap_values(self, explainer: Any, features: np.ndarray) -> Optional[np.ndarray]:
        """Calculate SHAP values for features."""
        if explainer is None:
            return None
            
        try:
            if hasattr(explainer, 'shap_values'):
                # TreeExplainer
                shap_values = explainer.shap_values(features)
                if isinstance(shap_values, list):
                    # Multi-class case, take the first class
                    shap_values = shap_values[0]
                return shap_values
            else:
                # KernelExplainer
                shap_values = explainer(features)
                return shap_values.values
                
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate SHAP values: {e}")
            return None
    
    def _calculate_lime_explanation(self, explainer: Any, model: Any, features: np.ndarray, feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """Calculate LIME explanation for features."""
        if explainer is None:
            return None
            
        try:
            # Create prediction function
            def predict_fn(x):
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(x)
                else:
                    return model.predict(x).reshape(-1, 1)
            
            # Get explanation
            explanation = explainer.explain_instance(
                features.flatten(),
                predict_fn,
                num_features=min(self.max_features, len(feature_names))
            )
            
            # Extract explanation data
            explanation_data = {
                'feature_importance': dict(explanation.as_list()),
                'explanation_text': explanation.as_html(),
                'prediction': explanation.predicted_value,
                'confidence': explanation.score
            }
            
            return explanation_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate LIME explanation: {e}")
            return None
    
    def _extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(self.feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) > 1:
                    coef = model.coef_[0]  # Take first class for multi-class
                else:
                    coef = model.coef_
                return dict(zip(self.feature_names, coef))
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to extract feature importance: {e}")
            return None
    
    def save_explanation(self, explanation: ExplanationResult) -> bool:
        """Save explanation to storage."""
        try:
            timestamp_str = explanation.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_explanation_{timestamp_str}.json"
            filepath = self.explanations_storage / filename
            
            # Convert to serializable format
            explanation_dict = {
                'model_name': explanation.model_name,
                'prediction': explanation.prediction,
                'feature_names': explanation.feature_names,
                'feature_values': explanation.feature_values.tolist(),
                'shap_values': explanation.shap_values.tolist() if explanation.shap_values is not None else None,
                'lime_explanation': explanation.lime_explanation,
                'feature_importance': explanation.feature_importance,
                'confidence': explanation.confidence,
                'timestamp': explanation.timestamp.isoformat(),
                'metadata': explanation.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(explanation_dict, f, indent=2)
            
            self.logger.info(f"💾 Explanation saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save explanation: {e}")
            return False


class TradeDecisionTracer:
    """Tracer for trade decisions back to individual factors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trade decision tracer."""
        self.config = config
        self.logger = system_logger.getChild("TradeDecisionTracer")
        
        # Storage
        self.traces_storage = Path(config.get("explainability", {}).get("traces_storage_path", "data/decision_traces"))
        self.traces_storage.mkdir(parents=True, exist_ok=True)
        
        # Active traces
        self.active_traces: Dict[str, TradeDecisionTrace] = {}
        
    async def start_decision_trace(
        self,
        decision_id: str,
        decision_type: str,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> TradeDecisionTrace:
        """Start tracing a trade decision."""
        trace = TradeDecisionTrace(
            decision_id=decision_id,
            timestamp=datetime.now(),
            decision_type=decision_type,
            final_decision=None,
            confidence=0.0,
            market_conditions=market_conditions or {}
        )
        
        self.active_traces[decision_id] = trace
        self.logger.info(f"🔍 Started decision trace for {decision_id}")
        return trace
    
    async def add_model_explanation(
        self,
        decision_id: str,
        model_name: str,
        explanation: ExplanationResult
    ) -> bool:
        """Add model explanation to decision trace."""
        if decision_id not in self.active_traces:
            self.logger.error(f"❌ No active trace found for decision {decision_id}")
            return False
        
        trace = self.active_traces[decision_id]
        
        # Add explanation based on model type
        if model_name.lower() == 'tactician':
            trace.tactician_explanation = explanation
        elif model_name.lower() == 'hmm':
            trace.hmm_explanation = explanation
        elif model_name.lower() == 'sr':
            trace.sr_explanation = explanation
        elif model_name.lower() == 'analyst':
            trace.analyst_explanation = explanation
        
        self.logger.info(f"📊 Added {model_name} explanation to trace {decision_id}")
        return True
    
    async def finalize_decision_trace(
        self,
        decision_id: str,
        final_decision: Any,
        confidence: float
    ) -> TradeDecisionTrace:
        """Finalize decision trace with final decision."""
        if decision_id not in self.active_traces:
            self.logger.error(f"❌ No active trace found for decision {decision_id}")
            return None
        
        trace = self.active_traces[decision_id]
        trace.final_decision = final_decision
        trace.confidence = confidence
        
        # Analyze contributing factors
        await self._analyze_contributing_factors(trace)
        
        # Save trace
        await self._save_decision_trace(trace)
        
        # Remove from active traces
        del self.active_traces[decision_id]
        
        self.logger.info(f"✅ Finalized decision trace for {decision_id}")
        return trace
    
    async def _analyze_contributing_factors(self, trace: TradeDecisionTrace) -> None:
        """Analyze contributing factors from all model explanations."""
        all_factors = []
        
        # Collect factors from all explanations
        for explanation in [trace.tactician_explanation, trace.hmm_explanation, 
                          trace.sr_explanation, trace.analyst_explanation]:
            if explanation is None:
                continue
                
            # Extract factors from SHAP values
            if explanation.shap_values is not None:
                for i, (feature, shap_val) in enumerate(zip(explanation.feature_names, explanation.shap_values)):
                    all_factors.append({
                        'feature': feature,
                        'importance': abs(shap_val),
                        'value': explanation.feature_values[i] if i < len(explanation.feature_values) else 0,
                        'model': explanation.model_name,
                        'type': 'shap'
                    })
            
            # Extract factors from LIME explanation
            if explanation.lime_explanation and 'feature_importance' in explanation.lime_explanation:
                for feature, importance in explanation.lime_explanation['feature_importance']:
                    all_factors.append({
                        'feature': feature,
                        'importance': abs(importance),
                        'value': 0,  # LIME doesn't provide feature values
                        'model': explanation.model_name,
                        'type': 'lime'
                    })
        
        # Sort by importance and categorize
        all_factors.sort(key=lambda x: x['importance'], reverse=True)
        
        # Top contributing factors
        trace.top_contributing_factors = all_factors[:10]
        
        # Categorize as risk or opportunity factors
        for factor in all_factors:
            if factor['importance'] > 0.1:  # Threshold for significant factors
                if factor['value'] < 0:  # Negative impact
                    trace.risk_factors.append(factor)
                else:  # Positive impact
                    trace.opportunity_factors.append(factor)
    
    async def _save_decision_trace(self, trace: TradeDecisionTrace) -> bool:
        """Save decision trace to storage."""
        try:
            timestamp_str = trace.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"decision_trace_{trace.decision_id}_{timestamp_str}.json"
            filepath = self.traces_storage / filename
            
            # Convert to serializable format
            trace_dict = {
                'decision_id': trace.decision_id,
                'timestamp': trace.timestamp.isoformat(),
                'decision_type': trace.decision_type,
                'final_decision': trace.final_decision,
                'confidence': trace.confidence,
                'top_contributing_factors': trace.top_contributing_factors,
                'risk_factors': trace.risk_factors,
                'opportunity_factors': trace.opportunity_factors,
                'market_conditions': trace.market_conditions,
                'metadata': trace.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(trace_dict, f, indent=2)
            
            self.logger.info(f"💾 Decision trace saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save decision trace: {e}")
            return False