#!/usr/bin/env python3
"""
SHAP/LIME Integration for Enhanced Monitoring

Provides detailed model explanations using SHAP and LIME for comprehensive
trade decision analysis and model interpretability.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports for SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None
    lime_tabular = None

from src.core.decorators import handles_errors
from src.utils.logger import system_logger

@dataclass
class SHAPExplanation:
    """SHAP explanation for a model prediction."""
    model_id: str
    prediction: float
    base_value: float
    shap_values: Dict[str, float]  # feature_name -> shap_value
    feature_names: List[str]
    feature_values: List[float]
    explanation_time_ms: float
    timestamp: datetime

@dataclass
class LIMEExplanation:
    """LIME explanation for a model prediction."""
    model_id: str
    prediction: float
    explanation: Dict[str, float]  # feature_name -> importance
    feature_names: List[str]
    feature_values: List[float]
    explanation_time_ms: float
    timestamp: datetime
    local_fidelity: float

@dataclass
class ModelExplanationRequest:
    """Request for model explanation."""
    model_id: str
    model_type: str
    features: np.ndarray
    feature_names: List[str]
    prediction: Optional[float] = None
    model: Optional[Any] = None
    training_data: Optional[pd.DataFrame] = None

class SHAPAnalyzer:
    """SHAP analyzer for model explanations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SHAP analyzer."""
        self.config = config
        self.logger = system_logger.getChild("SHAPAnalyzer")
        
        # Configuration
        self.shap_config = config.get("shap_analysis", {})
        self.enable_shap = self.shap_config.get("enable_shap", True)
        self.max_features = self.shap_config.get("max_features", 50)
        self.explanation_timeout = self.shap_config.get("explanation_timeout", 30)
        
        # SHAP availability
        self.shap_available = False
        self._check_shap_availability()
        
        # Storage
        self.explanations: List[SHAPExplanation] = []
        self.explainer_cache: Dict[str, Any] = {}
        
        self.logger.info(f"SHAP Analyzer initialized (available: {self.shap_available})")
    
    def _check_shap_availability(self):
        """Check if SHAP is available."""
        try:
            import shap
            self.shap_available = True
            self.logger.info("SHAP library is available")
        except ImportError:
            self.shap_available = False
            self.logger.warning("SHAP library not available - explanations will be disabled")
    
    @handles_errors(default_return=None, context="shap_analyzer.explain_prediction")
    async def explain_prediction(
        self, 
        request: ModelExplanationRequest
    ) -> Optional[SHAPExplanation]:
        """Generate SHAP explanation for a model prediction."""
        try:
            if not self.shap_available or not self.enable_shap:
                self.logger.debug("SHAP explanations disabled")
                return None
            
            start_time = time.time()
            
            # Get or create explainer
            explainer = await self._get_or_create_explainer(request)
            if not explainer:
                self.logger.warning(f"Could not create SHAP explainer for {request.model_id}")
                return None
            
            # Generate explanation
            explanation = await self._generate_shap_explanation(request, explainer)
            if not explanation:
                return None
            
            # Store explanation
            self.explanations.append(explanation)
            
            # Maintain memory limit
            if len(self.explanations) > 1000:
                self.explanations = self.explanations[-1000:]
            
            self.logger.debug(f"Generated SHAP explanation for {request.model_id} in {explanation.explanation_time_ms:.2f}ms")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation for {request.model_id}: {e}")
            return None
    
    async def _get_or_create_explainer(self, request: ModelExplanationRequest) -> Optional[Any]:
        """Get or create SHAP explainer for the model."""
        try:
            if request.model_id in self.explainer_cache:
                return self.explainer_cache[request.model_id]
            
            if not self.shap_available:
                return None
            
            
            # Create explainer based on model type
            if request.model_type.lower() in ['tree', 'random_forest', 'xgboost', 'lightgbm']:
                # Tree-based models
                if request.model and hasattr(request.model, 'predict'):
                    explainer = shap.TreeExplainer(request.model)
                else:
                    self.logger.warning(f"Tree model not available for {request.model_id}")
                    return None
            
            elif request.model_type.lower() in ['linear', 'logistic', 'ridge', 'lasso']:
                # Linear models
                if request.model and hasattr(request.model, 'predict'):
                    explainer = shap.LinearExplainer(request.model, request.training_data)
                else:
                    self.logger.warning(f"Linear model not available for {request.model_id}")
                    return None
            
            elif request.model_type.lower() in ['neural_network', 'mlp', 'deep_learning']:
                # Neural network models
                if request.model and hasattr(request.model, 'predict'):
                    explainer = shap.DeepExplainer(request.model, request.training_data)
                else:
                    self.logger.warning(f"Neural network model not available for {request.model_id}")
                    return None
            
            else:
                # Generic model - use KernelExplainer
                if request.model and hasattr(request.model, 'predict') and request.training_data is not None:
                    # Use a subset of training data for efficiency
                    sample_size = min(100, len(request.training_data))
                    sample_data = request.training_data.sample(n=sample_size, random_state=42)
                    explainer = shap.KernelExplainer(request.model.predict, sample_data)
                else:
                    self.logger.warning(f"Generic model not available for {request.model_id}")
                    return None
            
            # Cache explainer
            self.explainer_cache[request.model_id] = explainer
            return explainer
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer for {request.model_id}: {e}")
            return None
    
    async def _generate_shap_explanation(
        self, 
        request: ModelExplanationRequest, 
        explainer: Any
    ) -> Optional[SHAPExplanation]:
        """Generate SHAP explanation using the explainer."""
        try:
            start_time = time.time()
            
            # Prepare features
            if len(request.features.shape) == 1:
                features = request.features.reshape(1, -1)
            else:
                features = request.features
            
            # Generate SHAP values
            shap_values = explainer.shap_values(features)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class output - use first class
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                # Multiple samples - use first sample
                shap_values = shap_values[0]
            
            # Get base value
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0] if len(base_value) > 0 else 0.0
            
            # Get prediction
            if request.prediction is not None:
                prediction = request.prediction
            else:
                prediction = base_value + np.sum(shap_values)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, (feature_name, shap_value) in enumerate(zip(request.feature_names, shap_values)):
                if i < self.max_features:  # Limit number of features
                    feature_importance[feature_name] = float(shap_value)
            
            explanation_time_ms = (time.time() - start_time) * 1000
            
            return SHAPExplanation(
                model_id=request.model_id,
                prediction=float(prediction),
                base_value=float(base_value),
                shap_values=feature_importance,
                feature_names=request.feature_names[:self.max_features],
                feature_values=request.features[:self.max_features].tolist(),
                explanation_time_ms=explanation_time_ms,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            return None

class LIMEAnalyzer:
    """LIME analyzer for model explanations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LIME analyzer."""
        self.config = config
        self.logger = system_logger.getChild("LIMEAnalyzer")
        
        # Configuration
        self.lime_config = config.get("lime_analysis", {})
        self.enable_lime = self.lime_config.get("enable_lime", True)
        self.max_features = self.lime_config.get("max_features", 20)
        self.num_samples = self.lime_config.get("num_samples", 1000)
        self.explanation_timeout = self.lime_config.get("explanation_timeout", 30)
        
        # LIME availability
        self.lime_available = False
        self._check_lime_availability()
        
        # Storage
        self.explanations: List[LIMEExplanation] = []
        self.explainer_cache: Dict[str, Any] = {}
        
        self.logger.info(f"LIME Analyzer initialized (available: {self.lime_available})")
    
    def _check_lime_availability(self):
        """Check if LIME is available."""
        try:
            import lime
            import lime.lime_tabular
            self.lime_available = True
            self.logger.info("LIME library is available")
        except ImportError:
            self.lime_available = False
            self.logger.warning("LIME library not available - explanations will be disabled")
    
    @handles_errors(default_return=None, context="lime_analyzer.explain_prediction")
    async def explain_prediction(
        self, 
        request: ModelExplanationRequest
    ) -> Optional[LIMEExplanation]:
        """Generate LIME explanation for a model prediction."""
        try:
            if not self.lime_available or not self.enable_lime:
                self.logger.debug("LIME explanations disabled")
                return None
            
            start_time = time.time()
            
            # Get or create explainer
            explainer = await self._get_or_create_explainer(request)
            if not explainer:
                self.logger.warning(f"Could not create LIME explainer for {request.model_id}")
                return None
            
            # Generate explanation
            explanation = await self._generate_lime_explanation(request, explainer)
            if not explanation:
                return None
            
            # Store explanation
            self.explanations.append(explanation)
            
            # Maintain memory limit
            if len(self.explanations) > 1000:
                self.explanations = self.explanations[-1000:]
            
            self.logger.debug(f"Generated LIME explanation for {request.model_id} in {explanation.explanation_time_ms:.2f}ms")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation for {request.model_id}: {e}")
            return None
    
    async def _get_or_create_explainer(self, request: ModelExplanationRequest) -> Optional[Any]:
        """Get or create LIME explainer for the model."""
        try:
            if request.model_id in self.explainer_cache:
                return self.explainer_cache[request.model_id]
            
            if not self.lime_available or request.training_data is None:
                return None
            
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                request.training_data.values,
                feature_names=request.feature_names,
                class_names=['prediction'],
                mode='regression' if request.model_type.lower() in ['regression', 'linear'] else 'classification',
                discretize_continuous=True
            )
            
            # Cache explainer
            self.explainer_cache[request.model_id] = explainer
            return explainer
            
        except Exception as e:
            self.logger.error(f"Error creating LIME explainer for {request.model_id}: {e}")
            return None
    
    async def _generate_lime_explanation(
        self, 
        request: ModelExplanationRequest, 
        explainer: Any
    ) -> Optional[LIMEExplanation]:
        """Generate LIME explanation using the explainer."""
        try:
            start_time = time.time()
            
            # Prepare features
            if len(request.features.shape) == 1:
                features = request.features
            else:
                features = request.features[0]
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                features,
                request.model.predict,
                num_features=self.max_features,
                num_samples=self.num_samples
            )
            
            # Extract explanation data
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                feature_importance[feature_idx] = float(importance)
            
            # Get prediction
            if request.prediction is not None:
                prediction = request.prediction
            else:
                prediction = request.model.predict([features])[0]
            
            # Calculate local fidelity (how well the explanation fits locally)
            local_fidelity = getattr(explanation, 'score', 0.0)
            
            explanation_time_ms = (time.time() - start_time) * 1000
            
            return LIMEExplanation(
                model_id=request.model_id,
                prediction=float(prediction),
                explanation=feature_importance,
                feature_names=request.feature_names[:self.max_features],
                feature_values=features[:self.max_features].tolist(),
                explanation_time_ms=explanation_time_ms,
                timestamp=datetime.now(),
                local_fidelity=float(local_fidelity)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation: {e}")
            return None

class ExplainabilityIntegrator:
    """Integrates SHAP and LIME explanations for comprehensive model interpretability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize explainability integrator."""
        self.config = config
        self.logger = system_logger.getChild("ExplainabilityIntegrator")
        
        # Initialize analyzers
        self.shap_analyzer = SHAPAnalyzer(config)
        self.lime_analyzer = LIMEAnalyzer(config)
        
        # Configuration
        self.integration_config = config.get("explainability_integration", {})
        self.enable_combined_explanations = self.integration_config.get("enable_combined_explanations", True)
        self.explanation_storage_path = self.integration_config.get("explanation_storage_path", "explanations")
        
        # Storage
        self.combined_explanations: List[Dict[str, Any]] = []
        
        # Create storage directory
        self.storage_dir = Path(self.explanation_storage_path)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.logger.info("Explainability Integrator initialized")
    
    @handles_errors(default_return=None, context="explainability_integrator.explain_model_prediction")
    async def explain_model_prediction(
        self, 
        request: ModelExplanationRequest
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive model explanation using SHAP and LIME."""
        try:
            explanations = {}
            
            # Generate SHAP explanation
            if self.shap_analyzer.shap_available and self.shap_analyzer.enable_shap:
                shap_explanation = await self.shap_analyzer.explain_prediction(request)
                if shap_explanation:
                    explanations['shap'] = asdict(shap_explanation)
            
            # Generate LIME explanation
            if self.lime_analyzer.lime_available and self.lime_analyzer.enable_lime:
                lime_explanation = await self.lime_analyzer.explain_prediction(request)
                if lime_explanation:
                    explanations['lime'] = asdict(lime_explanation)
            
            # Create combined explanation if both are available
            if self.enable_combined_explanations and len(explanations) > 1:
                combined_explanation = await self._create_combined_explanation(explanations, request)
                if combined_explanation:
                    explanations['combined'] = combined_explanation
            
            # Store explanation
            if explanations:
                await self._store_explanation(request.model_id, explanations)
            
            return explanations if explanations else None
            
        except Exception as e:
            self.logger.error(f"Error explaining model prediction for {request.model_id}: {e}")
            return None
    
    async def _create_combined_explanation(
        self, 
        explanations: Dict[str, Any], 
        request: ModelExplanationRequest
    ) -> Optional[Dict[str, Any]]:
        """Create combined explanation from SHAP and LIME."""
        try:
            combined = {
                'model_id': request.model_id,
                'model_type': request.model_type,
                'timestamp': datetime.now().isoformat(),
                'feature_importance': {},
                'consensus_score': 0.0,
                'explanation_agreement': 0.0
            }
            
            # Extract feature importance from both explanations
            shap_features = {}
            lime_features = {}
            
            if 'shap' in explanations:
                shap_data = explanations['shap']
                for feature_name, importance in shap_data.get('shap_values', {}).items():
                    shap_features[feature_name] = abs(importance)  # Use absolute value for ranking
            
            if 'lime' in explanations:
                lime_data = explanations['lime']
                for feature_name, importance in lime_data.get('explanation', {}).items():
                    lime_features[feature_name] = abs(importance)  # Use absolute value for ranking
            
            # Combine feature importance
            all_features = set(shap_features.keys()) | set(lime_features.keys())
            
            for feature in all_features:
                shap_importance = shap_features.get(feature, 0.0)
                lime_importance = lime_features.get(feature, 0.0)
                
                # Weighted average (can be adjusted based on confidence)
                combined_importance = (shap_importance + lime_importance) / 2.0
                combined['feature_importance'][feature] = combined_importance
            
            # Sort by importance
            combined['feature_importance'] = dict(
                sorted(combined['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            )
            
            # Calculate consensus score
            if shap_features and lime_features:
                # Calculate correlation between SHAP and LIME rankings
                common_features = set(shap_features.keys()) & set(lime_features.keys())
                if len(common_features) > 1:
                    shap_ranks = [shap_features[f] for f in common_features]
                    lime_ranks = [lime_features[f] for f in common_features]
                    
                    # Calculate Spearman correlation
                    correlation = np.corrcoef(shap_ranks, lime_ranks)[0, 1]
                    combined['consensus_score'] = float(correlation) if not np.isnan(correlation) else 0.0
                    combined['explanation_agreement'] = abs(combined['consensus_score'])
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error creating combined explanation: {e}")
            return None
    
    async def _store_explanation(self, model_id: str, explanations: Dict[str, Any]):
        """Store explanation data."""
        try:
            # Create model-specific directory
            model_dir = self.storage_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save explanation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            explanation_file = model_dir / f"explanation_{timestamp}.json"
            
            with open(explanation_file, 'w') as f:
                json.dump(explanations, f, indent=2, default=str)
            
            # Store in memory
            self.combined_explanations.append({
                'model_id': model_id,
                'timestamp': datetime.now(),
                'explanations': explanations
            })
            
            # Maintain memory limit
            if len(self.combined_explanations) > 1000:
                self.combined_explanations = self.combined_explanations[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error storing explanation: {e}")
    
    def get_explanation_stats(self) -> Dict[str, Any]:
        """Get statistics about explanations."""
        return {
            'shap_available': self.shap_analyzer.shap_available,
            'lime_available': self.lime_analyzer.lime_available,
            'shap_explanations_generated': len(self.shap_analyzer.explanations),
            'lime_explanations_generated': len(self.lime_analyzer.explanations),
            'combined_explanations_generated': len(self.combined_explanations),
            'shap_explainer_cache_size': len(self.shap_analyzer.explainer_cache),
            'lime_explainer_cache_size': len(self.lime_analyzer.explainer_cache),
            'storage_directory': str(self.storage_dir)
        }
    
    @handles_errors(default_return=False, context="explainability_integrator.cleanup_old_explanations")
    async def cleanup_old_explanations(self, days_to_keep: int = 30) -> bool:
        """Clean up old explanation files."""
        try:
            cutoff_time = datetime.now().timestamp() - days_to_keep * 24 * 3600
            files_removed = 0
            
            for model_dir in self.storage_dir.iterdir():
                if model_dir.is_dir():
                    for explanation_file in model_dir.glob("explanation_*.json"):
                        if explanation_file.stat().st_mtime < cutoff_time:
                            explanation_file.unlink()
                            files_removed += 1
            
            self.logger.info(f"Cleaned up {files_removed} old explanation files")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old explanations: {e}")
            return False