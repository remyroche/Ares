"""
Interpretability Analysis for ML Indicators

This module provides comprehensive interpretability analysis for ML-based trading
indicators, including feature importance, model explanations, and consensus analysis.

Key Features:
- SHAP-based feature importance analysis
- Model decision explanations
- Consensus feature identification
- Feature interaction analysis
- Model comparison interpretability
- Real-time explanation generation
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# LIME imports
try:
    import lime
    import lime.tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Core imports
from .ml_candle_pattern_indicators import MLIndicatorGenerator, IndicatorType, ModelType
from .model_comparison_pipeline import ModelComparisonPipeline

logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Methods for generating model explanations."""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    INTEGRATED_GRADIENTS = "integrated_gradients"


class InterpretabilityLevel(Enum):
    """Levels of interpretability analysis."""
    GLOBAL = "global"  # Overall model behavior
    LOCAL = "local"    # Individual prediction explanations
    FEATURE = "feature"  # Feature-level analysis
    INTERACTION = "interaction"  # Feature interaction analysis


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability analysis."""
    # Explanation methods
    explanation_methods: List[ExplanationMethod] = None
    enable_global_analysis: bool = True
    enable_local_analysis: bool = True
    enable_feature_analysis: bool = True
    enable_interaction_analysis: bool = True
    
    # SHAP configuration
    shap_sample_size: int = 1000
    shap_background_size: int = 100
    shap_explainer_type: str = "tree"  # tree, linear, kernel, deep - TreeSHAP preferred for performance
    
    # LIME configuration
    lime_sample_size: int = 5000
    lime_num_features: int = 10
    
    # Feature importance
    importance_threshold: float = 0.01
    top_features_count: int = 20
    
    # Consensus analysis
    enable_consensus_analysis: bool = True
    consensus_threshold: float = 0.5
    
    def __post_init__(self):
        if self.explanation_methods is None:
            self.explanation_methods = [ExplanationMethod.SHAP, ExplanationMethod.FEATURE_IMPORTANCE]


class InterpretabilityAnalyzer:
    """
    Comprehensive interpretability analyzer for ML trading indicators.
    
    This analyzer provides multiple methods for understanding model decisions
    and identifying the most important features for trading indicator generation.
    """
    
    def __init__(self, config: Optional[InterpretabilityConfig] = None):
        self.config = config or InterpretabilityConfig()
        self.explanations = {}
        self.feature_importance = {}
        self.consensus_features = {}
        self.interaction_analysis = {}
        
        # Initialize explainers
        self._initialize_explainers()
        
        logger.info("🔍 Interpretability Analyzer initialized")
    
    def _initialize_explainers(self):
        """Initialize explanation methods based on availability."""
        self.explainers = {}
        
        if SHAP_AVAILABLE and ExplanationMethod.SHAP in self.config.explanation_methods:
            self.explainers['shap'] = True
            logger.info("✅ SHAP explainer available")
        else:
            self.explainers['shap'] = False
            logger.warning("⚠️ SHAP not available")
        
        if LIME_AVAILABLE and ExplanationMethod.LIME in self.config.explanation_methods:
            self.explainers['lime'] = True
            logger.info("✅ LIME explainer available")
        else:
            self.explainers['lime'] = False
            logger.warning("⚠️ LIME not available")
    
    def analyze_model_interpretability(self, model: MLIndicatorGenerator, 
                                     data: pd.DataFrame,
                                     model_name: str = "model") -> Dict[str, Any]:
        """
        Analyze interpretability for a single model.
        
        Args:
            model: Trained ML indicator generator
            data: Data used for analysis
            model_name: Name identifier for the model
            
        Returns:
            Interpretability analysis results
        """
        logger.info(f"🔍 Analyzing interpretability for {model_name}")
        
        analysis_results = {
            'model_name': model_name,
            'analysis_timestamp': datetime.now(),
            'global_analysis': {},
            'local_analysis': {},
            'feature_analysis': {},
            'interaction_analysis': {}
        }
        
        # Generate features for analysis
        try:
            pattern_features = model._generate_pattern_features(data)
            context_features = model._generate_market_context_features(data)
            features = model._combine_features(pattern_features, context_features)
            
            # Generate predictions
            predictions = model._generate_feature(data)
            
        except Exception as e:
            logger.error(f"Failed to generate features for {model_name}: {e}")
            return {'error': str(e)}
        
        # Global analysis
        if self.config.enable_global_analysis:
            analysis_results['global_analysis'] = self._perform_global_analysis(
                model, features, predictions, model_name
            )
        
        # Local analysis
        if self.config.enable_local_analysis:
            analysis_results['local_analysis'] = self._perform_local_analysis(
                model, features, predictions, model_name
            )
        
        # Feature analysis
        if self.config.enable_feature_analysis:
            analysis_results['feature_analysis'] = self._perform_feature_analysis(
                model, features, predictions, model_name
            )
        
        # Interaction analysis
        if self.config.enable_interaction_analysis:
            analysis_results['interaction_analysis'] = self._perform_interaction_analysis(
                model, features, predictions, model_name
            )
        
        # Store results
        self.explanations[model_name] = analysis_results
        
        logger.info(f"✅ Interpretability analysis completed for {model_name}")
        return analysis_results
    
    def _perform_global_analysis(self, model: MLIndicatorGenerator, 
                               features: np.ndarray, predictions: pd.Series,
                               model_name: str) -> Dict[str, Any]:
        """Perform global interpretability analysis."""
        global_analysis = {
            'feature_importance': {},
            'model_summary': {},
            'prediction_distribution': {}
        }
        
        try:
            # Feature importance analysis
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                global_analysis['feature_importance'] = feature_importance
                
                # Store for consensus analysis
                self.feature_importance[model_name] = feature_importance
            
            # Model summary statistics
            global_analysis['model_summary'] = {
                'total_features': features.shape[1],
                'prediction_mean': predictions.mean(),
                'prediction_std': predictions.std(),
                'prediction_min': predictions.min(),
                'prediction_max': predictions.max()
            }
            
            # Prediction distribution analysis
            global_analysis['prediction_distribution'] = {
                'positive_predictions': (predictions > 0.5).sum(),
                'negative_predictions': (predictions < 0.5).sum(),
                'neutral_predictions': ((predictions >= 0.4) & (predictions <= 0.6)).sum()
            }
            
        except Exception as e:
            logger.warning(f"Global analysis failed for {model_name}: {e}")
            global_analysis['error'] = str(e)
        
        return global_analysis
    
    def _perform_local_analysis(self, model: MLIndicatorGenerator, 
                              features: np.ndarray, predictions: pd.Series,
                              model_name: str) -> Dict[str, Any]:
        """Perform local interpretability analysis."""
        local_analysis = {
            'shap_explanations': {},
            'lime_explanations': {},
            'sample_explanations': {}
        }
        
        try:
            # SHAP explanations
            if self.explainers['shap'] and SHAP_AVAILABLE:
                shap_explanations = self._generate_shap_explanations(model, features, model_name)
                local_analysis['shap_explanations'] = shap_explanations
            
            # LIME explanations
            if self.explainers['lime'] and LIME_AVAILABLE:
                lime_explanations = self._generate_lime_explanations(model, features, model_name)
                local_analysis['lime_explanations'] = lime_explanations
            
            # Sample explanations for high-confidence predictions
            high_confidence_indices = self._get_high_confidence_indices(predictions)
            if len(high_confidence_indices) > 0:
                sample_explanations = self._generate_sample_explanations(
                    model, features, predictions, high_confidence_indices[:5], model_name
                )
                local_analysis['sample_explanations'] = sample_explanations
            
        except Exception as e:
            logger.warning(f"Local analysis failed for {model_name}: {e}")
            local_analysis['error'] = str(e)
        
        return local_analysis
    
    def _perform_feature_analysis(self, model: MLIndicatorGenerator, 
                                features: np.ndarray, predictions: pd.Series,
                                model_name: str) -> Dict[str, Any]:
        """Perform feature-level analysis."""
        feature_analysis = {
            'top_features': [],
            'feature_correlations': {},
            'feature_contributions': {},
            'feature_stability': {}
        }
        
        try:
            # Get feature importance
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                
                # Identify top features
                if isinstance(feature_importance, dict):
                    for indicator_type, importance_scores in feature_importance.items():
                        if isinstance(importance_scores, np.ndarray) and len(importance_scores) > 0:
                            # Get top features
                            top_indices = np.argsort(importance_scores)[-self.config.top_features_count:][::-1]
                            top_features = {
                                'indicator_type': indicator_type,
                                'top_feature_indices': top_indices.tolist(),
                                'top_feature_scores': importance_scores[top_indices].tolist()
                            }
                            feature_analysis['top_features'].append(top_features)
            
            # Feature correlation with predictions
            feature_correlations = []
            for i in range(features.shape[1]):
                if features.shape[0] > 1:
                    correlation = np.corrcoef(features[:, i], predictions)[0, 1]
                    if not np.isnan(correlation):
                        feature_correlations.append({
                            'feature_index': i,
                            'correlation': correlation
                        })
            
            feature_analysis['feature_correlations'] = sorted(
                feature_correlations, key=lambda x: abs(x['correlation']), reverse=True
            )[:self.config.top_features_count]
            
        except Exception as e:
            logger.warning(f"Feature analysis failed for {model_name}: {e}")
            feature_analysis['error'] = str(e)
        
        return feature_analysis
    
    def _perform_interaction_analysis(self, model: MLIndicatorGenerator, 
                                    features: np.ndarray, predictions: pd.Series,
                                    model_name: str) -> Dict[str, Any]:
        """Perform feature interaction analysis."""
        interaction_analysis = {
            'feature_interactions': {},
            'interaction_strength': {},
            'top_interactions': []
        }
        
        try:
            # Calculate feature interactions using correlation
            feature_interactions = []
            for i in range(features.shape[1]):
                for j in range(i + 1, features.shape[1]):
                    if features.shape[0] > 1:
                        # Calculate interaction as product of features
                        interaction = features[:, i] * features[:, j]
                        interaction_correlation = np.corrcoef(interaction, predictions)[0, 1]
                        
                        if not np.isnan(interaction_correlation):
                            feature_interactions.append({
                                'feature_1': i,
                                'feature_2': j,
                                'interaction_correlation': interaction_correlation,
                                'interaction_strength': abs(interaction_correlation)
                            })
            
            # Sort by interaction strength
            feature_interactions = sorted(
                feature_interactions, key=lambda x: x['interaction_strength'], reverse=True
            )
            
            interaction_analysis['feature_interactions'] = feature_interactions
            interaction_analysis['top_interactions'] = feature_interactions[:10]
            
        except Exception as e:
            logger.warning(f"Interaction analysis failed for {model_name}: {e}")
            interaction_analysis['error'] = str(e)
        
        return interaction_analysis
    
    def _generate_shap_explanations(self, model: MLIndicatorGenerator, 
                                  features: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Generate SHAP explanations for the model."""
        try:
            # Get the underlying model
            if hasattr(model, 'trained_models'):
                # Use the first available model
                for indicator_type, trained_model in model.trained_models.items():
                    if hasattr(trained_model, 'predict'):
                        # Create SHAP explainer
                        if self.config.shap_explainer_type == "tree":
                            explainer = shap.TreeExplainer(trained_model)
                        elif self.config.shap_explainer_type == "linear":
                            explainer = shap.LinearExplainer(trained_model, features)
                        else:
                            explainer = shap.KernelExplainer(trained_model.predict, features[:self.config.shap_background_size])
                        
                        # Generate SHAP values
                        shap_values = explainer.shap_values(features[:self.config.shap_sample_size])
                        
                        return {
                            'explainer_type': self.config.shap_explainer_type,
                            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                            'feature_names': [f"feature_{i}" for i in range(features.shape[1])],
                            'sample_size': self.config.shap_sample_size
                        }
            
            return {'error': 'No suitable model found for SHAP analysis'}
            
        except Exception as e:
            logger.warning(f"SHAP explanation generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_lime_explanations(self, model: MLIndicatorGenerator, 
                                  features: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Generate LIME explanations for the model."""
        try:
            # This would implement LIME explanation generation
            # For now, return a placeholder
            return {
                'explainer_type': 'lime',
                'sample_size': self.config.lime_sample_size,
                'num_features': self.config.lime_num_features,
                'status': 'placeholder'
            }
            
        except Exception as e:
            logger.warning(f"LIME explanation generation failed: {e}")
            return {'error': str(e)}
    
    def _get_high_confidence_indices(self, predictions: pd.Series) -> List[int]:
        """Get indices of high-confidence predictions."""
        # High confidence: predictions far from 0.5
        confidence_scores = np.abs(predictions - 0.5)
        threshold = confidence_scores.quantile(0.8)  # Top 20% most confident
        return np.where(confidence_scores >= threshold)[0].tolist()
    
    def _generate_sample_explanations(self, model: MLIndicatorGenerator, 
                                    features: np.ndarray, predictions: pd.Series,
                                    indices: List[int], model_name: str) -> Dict[str, Any]:
        """Generate explanations for specific sample predictions."""
        sample_explanations = {}
        
        for idx in indices:
            try:
                sample_features = features[idx:idx+1]
                sample_prediction = predictions.iloc[idx]
                
                # Generate explanation for this sample
                explanation = {
                    'index': idx,
                    'prediction': sample_prediction,
                    'features': sample_features[0].tolist(),
                    'explanation': f"Sample {idx} prediction: {sample_prediction:.4f}"
                }
                
                sample_explanations[f"sample_{idx}"] = explanation
                
            except Exception as e:
                logger.warning(f"Sample explanation failed for index {idx}: {e}")
        
        return sample_explanations
    
    def analyze_consensus_interpretability(self, comparison_pipeline: ModelComparisonPipeline) -> Dict[str, Any]:
        """
        Analyze interpretability across multiple models to find consensus features.
        
        Args:
            comparison_pipeline: Model comparison pipeline with trained models
            
        Returns:
            Consensus interpretability analysis
        """
        if not self.config.enable_consensus_analysis:
            return {'error': 'Consensus analysis not enabled'}
        
        logger.info("🤝 Analyzing consensus interpretability across models")
        
        consensus_analysis = {
            'consensus_features': {},
            'feature_agreement': {},
            'model_consensus_scores': {},
            'top_consensus_features': []
        }
        
        try:
            # Collect feature importance from all models
            all_feature_importance = {}
            for model_name, model_result in comparison_pipeline.models.items():
                if model_result['success']:
                    generator = model_result['generator']
                    if hasattr(generator, 'get_feature_importance'):
                        feature_importance = generator.get_feature_importance()
                        all_feature_importance[model_name] = feature_importance
            
            if len(all_feature_importance) < 2:
                return {'error': 'Insufficient models for consensus analysis'}
            
            # Find consensus features
            consensus_features = self._find_consensus_features(all_feature_importance)
            consensus_analysis['consensus_features'] = consensus_features
            
            # Calculate feature agreement scores
            feature_agreement = self._calculate_feature_agreement(all_feature_importance)
            consensus_analysis['feature_agreement'] = feature_agreement
            
            # Calculate model consensus scores
            model_consensus = self._calculate_model_consensus(all_feature_importance)
            consensus_analysis['model_consensus_scores'] = model_consensus
            
            # Identify top consensus features
            top_consensus = self._identify_top_consensus_features(consensus_features, feature_agreement)
            consensus_analysis['top_consensus_features'] = top_consensus
            
            # Store for later use
            self.consensus_features = consensus_features
            
        except Exception as e:
            logger.error(f"Consensus interpretability analysis failed: {e}")
            consensus_analysis['error'] = str(e)
        
        return consensus_analysis
    
    def _find_consensus_features(self, all_feature_importance: Dict[str, Any]) -> Dict[str, Any]:
        """Find features that are important across multiple models."""
        consensus_features = {
            'feature_importance_consensus': {},
            'high_agreement_features': [],
            'consensus_scores': {}
        }
        
        try:
            # Extract feature importance scores
            feature_scores = {}
            for model_name, importance_dict in all_feature_importance.items():
                for indicator_type, scores in importance_dict.items():
                    if isinstance(scores, np.ndarray) and len(scores) > 0:
                        key = f"{model_name}_{indicator_type}"
                        feature_scores[key] = scores
            
            if not feature_scores:
                return consensus_features
            
            # Calculate consensus for each feature
            feature_consensus = {}
            for i in range(len(list(feature_scores.values())[0])):
                feature_scores_across_models = []
                for scores in feature_scores.values():
                    if i < len(scores):
                        feature_scores_across_models.append(scores[i])
                
                if len(feature_scores_across_models) > 1:
                    # Calculate consensus as mean importance
                    consensus_score = np.mean(feature_scores_across_models)
                    consensus_std = np.std(feature_scores_across_models)
                    
                    feature_consensus[i] = {
                        'consensus_score': consensus_score,
                        'consensus_std': consensus_std,
                        'agreement_level': 1.0 - (consensus_std / (consensus_score + 1e-8))
                    }
            
            consensus_features['feature_importance_consensus'] = feature_consensus
            
            # Identify high agreement features
            high_agreement = [
                (idx, data['consensus_score']) 
                for idx, data in feature_consensus.items()
                if data['agreement_level'] >= self.config.consensus_threshold
            ]
            high_agreement.sort(key=lambda x: x[1], reverse=True)
            consensus_features['high_agreement_features'] = high_agreement
            
        except Exception as e:
            logger.warning(f"Consensus feature finding failed: {e}")
            consensus_features['error'] = str(e)
        
        return consensus_features
    
    def _calculate_feature_agreement(self, all_feature_importance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate agreement scores between models for each feature."""
        agreement_scores = {}
        
        try:
            # Convert to arrays for easier processing
            importance_arrays = []
            for model_name, importance_dict in all_feature_importance.items():
                for indicator_type, scores in importance_dict.items():
                    if isinstance(scores, np.ndarray) and len(scores) > 0:
                        importance_arrays.append(scores)
            
            if len(importance_arrays) < 2:
                return agreement_scores
            
            # Calculate pairwise correlations
            for i in range(len(importance_arrays[0])):
                feature_scores = [arr[i] for arr in importance_arrays if i < len(arr)]
                if len(feature_scores) > 1:
                    # Calculate average correlation
                    correlations = []
                    for j in range(len(feature_scores)):
                        for k in range(j + 1, len(feature_scores)):
                            corr = np.corrcoef([feature_scores[j]], [feature_scores[k]])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                    
                    if correlations:
                        agreement_scores[f"feature_{i}"] = np.mean(correlations)
        
        except Exception as e:
            logger.warning(f"Feature agreement calculation failed: {e}")
        
        return agreement_scores
    
    def _calculate_model_consensus(self, all_feature_importance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consensus scores for each model."""
        model_consensus = {}
        
        try:
            model_names = list(all_feature_importance.keys())
            
            for model_name in model_names:
                # Calculate how much this model agrees with others
                model_agreements = []
                
                for other_model in model_names:
                    if other_model != model_name:
                        # Compare feature importance between models
                        agreement = self._calculate_model_agreement(
                            all_feature_importance[model_name],
                            all_feature_importance[other_model]
                        )
                        if agreement is not None:
                            model_agreements.append(agreement)
                
                if model_agreements:
                    model_consensus[model_name] = np.mean(model_agreements)
        
        except Exception as e:
            logger.warning(f"Model consensus calculation failed: {e}")
        
        return model_consensus
    
    def _calculate_model_agreement(self, importance1: Dict[str, Any], 
                                 importance2: Dict[str, Any]) -> Optional[float]:
        """Calculate agreement between two models' feature importance."""
        try:
            # Find common indicator types
            common_types = set(importance1.keys()) & set(importance2.keys())
            if not common_types:
                return None
            
            agreements = []
            for indicator_type in common_types:
                scores1 = importance1[indicator_type]
                scores2 = importance2[indicator_type]
                
                if isinstance(scores1, np.ndarray) and isinstance(scores2, np.ndarray):
                    if len(scores1) == len(scores2) and len(scores1) > 0:
                        corr = np.corrcoef(scores1, scores2)[0, 1]
                        if not np.isnan(corr):
                            agreements.append(corr)
            
            return np.mean(agreements) if agreements else None
            
        except Exception as e:
            logger.warning(f"Model agreement calculation failed: {e}")
            return None
    
    def _identify_top_consensus_features(self, consensus_features: Dict[str, Any], 
                                       feature_agreement: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify the top consensus features."""
        top_features = []
        
        try:
            # Combine consensus scores and agreement scores
            combined_scores = []
            
            for feature_idx, consensus_data in consensus_features.get('feature_importance_consensus', {}).items():
                agreement_key = f"feature_{feature_idx}"
                agreement_score = feature_agreement.get(agreement_key, 0)
                
                combined_score = (
                    consensus_data['consensus_score'] * 0.7 + 
                    agreement_score * 0.3
                )
                
                combined_scores.append({
                    'feature_index': int(feature_idx),
                    'consensus_score': consensus_data['consensus_score'],
                    'agreement_score': agreement_score,
                    'combined_score': combined_score,
                    'agreement_level': consensus_data['agreement_level']
                })
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            top_features = combined_scores[:self.config.top_features_count]
            
        except Exception as e:
            logger.warning(f"Top consensus feature identification failed: {e}")
        
        return top_features
    
    def generate_interpretability_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive interpretability report."""
        report = {
            'report_timestamp': datetime.now(),
            'model_analyses': self.explanations,
            'consensus_features': self.consensus_features,
            'summary_statistics': self._generate_summary_statistics()
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, default=str, indent=2)
            logger.info(f"💾 Interpretability report saved to {save_path}")
        
        return report
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for interpretability analysis."""
        summary = {
            'total_models_analyzed': len(self.explanations),
            'consensus_features_found': len(self.consensus_features.get('high_agreement_features', [])),
            'average_feature_importance': {},
            'top_consensus_features': []
        }
        
        try:
            # Calculate average feature importance across models
            if self.feature_importance:
                all_importance_scores = []
                for model_importance in self.feature_importance.values():
                    for indicator_type, scores in model_importance.items():
                        if isinstance(scores, np.ndarray):
                            all_importance_scores.extend(scores)
                
                if all_importance_scores:
                    summary['average_feature_importance'] = {
                        'mean': np.mean(all_importance_scores),
                        'std': np.std(all_importance_scores),
                        'max': np.max(all_importance_scores),
                        'min': np.min(all_importance_scores)
                    }
            
            # Get top consensus features
            if self.consensus_features.get('high_agreement_features'):
                summary['top_consensus_features'] = self.consensus_features['high_agreement_features'][:10]
        
        except Exception as e:
            logger.warning(f"Summary statistics generation failed: {e}")
            summary['error'] = str(e)
        
        return summary


def create_interpretability_analyzer(config: Optional[InterpretabilityConfig] = None) -> InterpretabilityAnalyzer:
    """Create an interpretability analyzer with specified configuration."""
    return InterpretabilityAnalyzer(config)


def test_interpretability_analyzer():
    """Test function for the interpretability analyzer."""
    print("🧪 Testing Interpretability Analyzer...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create interpretability analyzer
    config = InterpretabilityConfig(
        explanation_methods=[ExplanationMethod.SHAP, ExplanationMethod.FEATURE_IMPORTANCE],
        enable_consensus_analysis=True
    )
    analyzer = create_interpretability_analyzer(config)
    
    # Create and train a model
    from .ml_candle_pattern_indicators import create_ml_indicator_generator, ModelType
    
    generator = create_ml_indicator_generator(ModelType.LIGHTGBM)
    generator.train_models(data)
    
    # Analyze interpretability
    print("🔍 Analyzing model interpretability...")
    analysis_results = analyzer.analyze_model_interpretability(generator, data, "test_model")
    
    # Display results
    print("\n📊 Interpretability Analysis Results:")
    print(f"   Model analyzed: {analysis_results['model_name']}")
    
    if 'global_analysis' in analysis_results:
        global_analysis = analysis_results['global_analysis']
        if 'model_summary' in global_analysis:
            summary = global_analysis['model_summary']
            print(f"   Total features: {summary.get('total_features', 'N/A')}")
            print(f"   Prediction mean: {summary.get('prediction_mean', 'N/A'):.4f}")
            print(f"   Prediction std: {summary.get('prediction_std', 'N/A'):.4f}")
    
    if 'feature_analysis' in analysis_results:
        feature_analysis = analysis_results['feature_analysis']
        if 'top_features' in feature_analysis:
            print(f"   Top features identified: {len(feature_analysis['top_features'])}")
    
    # Generate report
    print("\n📋 Generating interpretability report...")
    report = analyzer.generate_interpretability_report()
    
    print(f"   Models analyzed: {report['summary_statistics']['total_models_analyzed']}")
    print(f"   Consensus features: {report['summary_statistics']['consensus_features_found']}")
    
    print("\n🎉 Interpretability Analyzer test completed successfully!")
    return analyzer, analysis_results


if __name__ == "__main__":
    test_interpretability_analyzer()