"""
Model Complexity Analysis for Overfitting Prevention

This module provides comprehensive model complexity analysis to assess
overfitting risk and provide recommendations for model simplification.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('ModelComplexityAnalysis')

@dataclass
class ModelComplexityAnalysisConfig:
    """Configuration for model complexity analysis."""

    # Complexity thresholds
    max_complexity_score: float = 0.8
    high_complexity_threshold: float = 0.7
    medium_complexity_threshold: float = 0.5

    # Feature complexity
    max_feature_ratio: float = 0.5
    min_samples_per_feature: int = 10
    max_correlation_threshold: float = 0.95

    # Model-specific thresholds
    max_tree_depth: int = 15
    max_neural_layers: int = 5
    max_parameters: int = 1000000

    # Regularization requirements
    min_regularization_strength: float = 1e-6
    required_dropout_rate: float = 0.1

    # Validation requirements
    min_cv_folds: int = 5
    min_validation_samples: int = 100

class ModelComplexityAnalyzer:
    """
    Comprehensive model complexity analysis for overfitting risk assessment.

    This class provides:
    1. Model architecture complexity analysis
    2. Feature space complexity assessment
    3. Data complexity evaluation
    4. Training complexity monitoring
    5. Overfitting risk scoring
    6. Simplification recommendations
    """

    def __init__(self, config: Optional[ModelComplexityAnalysisConfig] = None):
        """Initialize model complexity analyzer."""
        self.config = config or ModelComplexityAnalysisConfig()
        self.logger = logger.getChild('ModelComplexityAnalyzer')

        # Analysis results storage
        self.complexity_analyses = []
        self.risk_assessments = []
        self.simplification_recommendations = []

        self.logger.info("✅ Model Complexity Analyzer initialized")

    def analyze_model_complexity(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        model_name: str = "unknown_model",
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model complexity analysis.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            model_name: Name of the model
            feature_names: Optional feature names

        Returns:
            Dictionary containing complexity analysis results
        """
        self.logger.info(f"🔍 Analyzing model complexity for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'overall_complexity_score': 0.0,
            'overfitting_risk': 'low',
            'complexity_components': {},
            'feature_complexity': {},
            'model_architecture_complexity': {},
            'data_complexity': {},
            'training_complexity': {},
            'risk_factors': [],
            'simplification_recommendations': []
        }

        try:
            # 1. Feature complexity analysis
            feature_complexity = self._analyze_feature_complexity(X_train, y_train, feature_names)
            results['feature_complexity'] = feature_complexity

            # 2. Model architecture complexity analysis
            model_arch_complexity = self._analyze_model_architecture_complexity(model, model_name)
            results['model_architecture_complexity'] = model_arch_complexity

            # 3. Data complexity analysis
            data_complexity = self._analyze_data_complexity(X_train, y_train)
            results['data_complexity'] = data_complexity

            # 4. Training complexity analysis
            training_complexity = self._analyze_training_complexity(model, X_train, y_train, X_val, y_val)
            results['training_complexity'] = training_complexity

            # Calculate overall complexity score
            overall_score = self._calculate_overall_complexity_score(results)
            results['overall_complexity_score'] = overall_score

            # Assess overfitting risk
            risk_level = self._assess_overfitting_risk(overall_score, results)
            results['overfitting_risk'] = risk_level

            # Identify risk factors
            results['risk_factors'] = self._identify_risk_factors(results)

            # Generate simplification recommendations
            results['simplification_recommendations'] = self._generate_simplification_recommendations(results)

            # Store analysis
            self.complexity_analyses.append({
                'model_name': model_name,
                'timestamp': results['timestamp'],
                'results': results
            })

            self.logger.info(f"✅ Model complexity analysis completed for {model_name}")

        except Exception as e:
            error_msg = f"Model complexity analysis failed for {model_name}: {e}"
            results['error'] = error_msg
            results['simplification_recommendations'].append("Review model complexity analysis setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _analyze_feature_complexity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze feature space complexity."""
        complexity = {
            'n_features': 0,
            'n_samples': 0,
            'feature_sample_ratio': 0.0,
            'feature_correlation_issues': 0,
            'high_correlation_pairs': [],
            'low_variance_features': 0,
            'complexity_score': 0.0
        }

        try:
            n_samples, n_features = X.shape
            complexity['n_samples'] = n_samples
            complexity['n_features'] = n_features

            # Feature-to-sample ratio
            if n_samples > 0:
                complexity['feature_sample_ratio'] = n_features / n_samples

                # Check against threshold
                if complexity['feature_sample_ratio'] > self.config.max_feature_ratio:
                    complexity['risk_factor'] = 'high_feature_ratio'

            # Feature correlation analysis
            if n_features > 1 and isinstance(X, pd.DataFrame):
                correlation_matrix = X.corr().abs()

                # Find highly correlated features
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        corr = correlation_matrix.iloc[i, j]

                        if corr > self.config.max_correlation_threshold:
                            high_corr_pairs.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': float(corr)
                            })

                complexity['feature_correlation_issues'] = len(high_corr_pairs)
                complexity['high_correlation_pairs'] = high_corr_pairs[:10]  # Limit to 10

            # Low variance features
            if isinstance(X, pd.DataFrame):
                variances = X.var()
                low_var_features = (variances < variances.quantile(0.1)).sum()
                complexity['low_variance_features'] = int(low_var_features)

            # Calculate feature complexity score
            complexity_score = 0.0

            # Ratio component
            ratio_score = min(1.0, complexity['feature_sample_ratio'] / self.config.max_feature_ratio)
            complexity_score += ratio_score * 0.4

            # Correlation component
            if n_features > 1:
                corr_score = min(1.0, complexity['feature_correlation_issues'] / (n_features * (n_features - 1) / 2))
                complexity_score += corr_score * 0.3

            # Low variance component
            if n_features > 0:
                low_var_score = complexity['low_variance_features'] / n_features
                complexity_score += low_var_score * 0.3

            complexity['complexity_score'] = complexity_score

        except Exception as e:
            self.logger.warning(f"Feature complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_model_architecture_complexity(self, model: Any, model_name: str) -> Dict[str, Any]:
        """Analyze model architecture complexity."""
        complexity = {
            'model_type': 'unknown',
            'architecture_complexity': 0.0,
            'parameter_count': 0,
            'layer_complexity': 0.0,
            'regularization_strength': 0.0,
            'complexity_score': 0.0
        }

        try:
            model_lower = model_name.lower()

            # Determine model type and analyze accordingly
            if 'randomforest' in model_lower:
                complexity.update(self._analyze_tree_model_complexity(model, 'random_forest'))
            elif 'xgboost' in model_lower or 'xgb' in model_lower:
                complexity.update(self._analyze_tree_model_complexity(model, 'xgboost'))
            elif 'lightgbm' in model_lower or 'lgb' in model_lower:
                complexity.update(self._analyze_tree_model_complexity(model, 'lightgbm'))
            elif 'neural' in model_lower or 'nn' in model_lower:
                complexity.update(self._analyze_neural_network_complexity(model))
            elif 'linear' in model_lower or 'ridge' in model_lower or 'lasso' in model_lower:
                complexity.update(self._analyze_linear_model_complexity(model))
            else:
                complexity['model_type'] = 'unknown'
                complexity['complexity_score'] = 0.5  # Default medium complexity

        except Exception as e:
            self.logger.warning(f"Model architecture complexity analysis failed: {e}")
            complexity['error'] = str(e)
            complexity['complexity_score'] = 0.5  # Default

        return complexity

    def _analyze_tree_model_complexity(self, model: Any, model_type: str) -> Dict[str, Any]:
        """Analyze tree-based model complexity."""
        complexity = {
            'model_type': model_type,
            'n_trees': 0,
            'max_depth': 0,
            'avg_leaf_nodes': 0,
            'tree_complexity_score': 0.0
        }

        try:
            # Get model parameters
            params = model.get_params()

            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)

            complexity['n_trees'] = n_estimators

            # Set default max_depth if not specified
            if max_depth is None:
                if model_type == 'xgboost':
                    max_depth = 6  # XGBoost default
                elif model_type == 'lightgbm':
                    max_depth = -1  # LightGBM default (unlimited, but limited by num_leaves)
                else:
                    max_depth = 10  # General default

            complexity['max_depth'] = max_depth if max_depth > 0 else 10

            # Calculate tree complexity
            if model_type == 'lightgbm' and 'num_leaves' in params:
                num_leaves = params.get('num_leaves', 31)
                complexity['tree_complexity_score'] = min(1.0, (n_estimators * num_leaves) / 10000)
            else:
                complexity['tree_complexity_score'] = min(1.0, (n_estimators * complexity['max_depth']) / 1000)

            # Overall architecture complexity
            architecture_score = complexity['tree_complexity_score']

            # Add regularization factor
            reg_alpha = params.get('reg_alpha', 0)
            reg_lambda = params.get('reg_lambda', 1)
            complexity['regularization_strength'] = (reg_alpha + reg_lambda) / 2

            architecture_score = architecture_score * (1 - min(0.5, complexity['regularization_strength']))

            complexity['architecture_complexity'] = architecture_score
            complexity['complexity_score'] = architecture_score

        except Exception as e:
            self.logger.debug(f"Tree model complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_neural_network_complexity(self, model: Any) -> Dict[str, Any]:
        """Analyze neural network complexity."""
        complexity = {
            'model_type': 'neural_network',
            'n_layers': 0,
            'total_parameters': 0,
            'has_dropout': False,
            'dropout_rate': 0.0,
            'architecture_complexity': 0.0
        }

        try:
            # Analyze Keras/TensorFlow models
            if hasattr(model, 'layers'):
                n_layers = len(model.layers)
                total_params = model.count_params()

                complexity['n_layers'] = n_layers
                complexity['total_parameters'] = total_params

                # Architecture complexity
                layer_complexity = min(1.0, n_layers / self.config.max_neural_layers)
                param_complexity = min(1.0, total_params / self.config.max_parameters)

                complexity['architecture_complexity'] = (layer_complexity + param_complexity) / 2

                # Check for dropout
                for layer in model.layers:
                    layer_config = layer.get_config()
                    if 'dropout' in layer_config.get('name', '').lower():
                        complexity['has_dropout'] = True
                        complexity['dropout_rate'] = layer_config.get('rate', 0.0)

            # Analyze PyTorch models
            elif hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                complexity['total_parameters'] = total_params

                # Estimate layers (rough heuristic)
                complexity['n_layers'] = max(1, total_params // 10000)  # Rough estimate
                complexity['architecture_complexity'] = min(1.0, total_params / self.config.max_parameters)

            # Check regularization
            if hasattr(model, 'get_config'):
                config = model.get_config()
                if 'dropout' in str(config).lower():
                    complexity['has_dropout'] = True

            complexity['complexity_score'] = complexity['architecture_complexity']

        except Exception as e:
            self.logger.debug(f"Neural network complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_linear_model_complexity(self, model: Any) -> Dict[str, Any]:
        """Analyze linear model complexity."""
        complexity = {
            'model_type': 'linear',
            'n_features': 0,
            'regularization_strength': 0.0,
            'complexity_score': 0.1  # Linear models are generally low complexity
        }

        try:
            # Get model parameters
            params = model.get_params()

            # Extract regularization
            alpha = params.get('alpha', 1.0)
            complexity['regularization_strength'] = alpha

            # Linear models have low complexity by nature
            complexity['architecture_complexity'] = 0.1

        except Exception as e:
            self.logger.debug(f"Linear model complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_data_complexity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze data complexity."""
        complexity = {
            'n_samples': 0,
            'n_features': 0,
            'class_imbalance_ratio': 1.0,
            'noise_level_estimate': 0.0,
            'complexity_score': 0.0
        }

        try:
            n_samples, n_features = X.shape
            complexity['n_samples'] = n_samples
            complexity['n_features'] = n_features

            # Class imbalance analysis
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if len(unique_classes) > 1:
                sorted_counts = np.sort(class_counts)
                complexity['class_imbalance_ratio'] = sorted_counts[-1] / sorted_counts[0]

            # Estimate noise level (rough heuristic)
            if hasattr(X, 'corr'):  # DataFrame
                correlation_matrix = X.corr()
                # Average absolute correlation with target proxy
                avg_corr = np.mean(np.abs(correlation_matrix.values))
                complexity['noise_level_estimate'] = 1 - avg_corr  # Higher correlation = lower noise

            # Data complexity score
            imbalance_score = min(1.0, (complexity['class_imbalance_ratio'] - 1) / 9)  # 10:1 ratio = max
            noise_score = complexity['noise_level_estimate']

            complexity['complexity_score'] = (imbalance_score + noise_score) / 2

        except Exception as e:
            self.logger.warning(f"Data complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_training_complexity(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Analyze training complexity and overfitting indicators."""
        complexity = {
            'training_stability': 0.0,
            'validation_effectiveness': 0.0,
            'overfitting_indicators': [],
            'complexity_score': 0.0
        }

        try:
            from sklearn.model_selection import cross_val_score

            # Perform quick cross-validation
            try:
                is_regression = len(np.unique(y_train)) > 10

                if is_regression:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=min(self.config.min_cv_folds, 5),
                        scoring='neg_mean_squared_error'
                    )
                else:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=min(self.config.min_cv_folds, 5),
                        scoring='accuracy'
                    )

                # Training stability (lower variance = more stable)
                cv_std = np.std(cv_scores)
                cv_mean = np.mean(cv_scores)

                if cv_mean != 0:
                    complexity['training_stability'] = 1 - min(1.0, cv_std / abs(cv_mean))
                else:
                    complexity['training_stability'] = 0.5

                # Validation effectiveness (if validation data provided)
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)

                    if is_regression:
                        from sklearn.metrics import mean_squared_error
                        val_score = -mean_squared_error(y_val, val_pred)
                    else:
                        from sklearn.metrics import accuracy_score
                        val_score = accuracy_score(y_val, val_pred)

                    # Compare CV performance to validation performance
                    val_effectiveness = min(1.0, abs(val_score - cv_mean) / abs(cv_mean)) if cv_mean != 0 else 0
                    complexity['validation_effectiveness'] = 1 - val_effectiveness

                # Check for overfitting indicators
                if complexity['training_stability'] < 0.5:
                    complexity['overfitting_indicators'].append('unstable_training')

                if X_val is not None and complexity.get('validation_effectiveness', 0) < 0.5:
                    complexity['overfitting_indicators'].append('poor_validation')

            except Exception as e:
                self.logger.warning(f"Training complexity analysis failed: {e}")
                complexity['error'] = str(e)

            # Training complexity score
            stability_score = complexity['training_stability']
            validation_score = complexity.get('validation_effectiveness', 0.5)

            complexity['complexity_score'] = 1 - (stability_score + validation_score) / 2

        except Exception as e:
            self.logger.warning(f"Training complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _calculate_overall_complexity_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall model complexity score."""
        try:
            weights = {
                'feature': 0.25,
                'architecture': 0.35,
                'data': 0.15,
                'training': 0.25
            }

            scores = {}
            for component, weight in weights.items():
                scores[component] = results[f'{component}_complexity'].get('complexity_score', 0.5)

            # Weighted average
            overall_score = sum(scores[comp] * weights[comp] for comp in scores)

            return min(1.0, overall_score)

        except Exception as e:
            self.logger.warning(f"Overall complexity score calculation failed: {e}")
            return 0.5  # Default medium complexity

    def _assess_overfitting_risk(self, complexity_score: float, results: Dict[str, Any]) -> str:
        """Assess overfitting risk based on complexity score and analysis."""
        risk_factors = []

        # Complexity score assessment
        if complexity_score > self.config.max_complexity_score:
            risk_factors.append('very_high_complexity')
        elif complexity_score > self.config.high_complexity_threshold:
            risk_factors.append('high_complexity')
        elif complexity_score > self.config.medium_complexity_threshold:
            risk_factors.append('medium_complexity')

        # Feature complexity assessment
        feature_complexity = results['feature_complexity']
        if feature_complexity.get('feature_sample_ratio', 0) > self.config.max_feature_ratio:
            risk_factors.append('high_feature_ratio')

        if feature_complexity.get('feature_correlation_issues', 0) > 0:
            risk_factors.append('correlated_features')

        # Model architecture assessment
        arch_complexity = results['model_architecture_complexity']
        model_type = arch_complexity.get('model_type', '')

        if 'neural' in model_type and not arch_complexity.get('has_dropout', False):
            risk_factors.append('no_dropout_regularization')

        if arch_complexity.get('max_depth', 0) > self.config.max_tree_depth:
            risk_factors.append('deep_trees')

        # Training complexity assessment
        training_complexity = results['training_complexity']
        if training_complexity.get('training_stability', 0) < 0.5:
            risk_factors.append('unstable_training')

        # Determine overall risk
        if len([f for f in risk_factors if 'very_high' in f or 'deep' in f]) > 0:
            return 'very_high'
        elif len([f for f in risk_factors if 'high' in f]) > 1:
            return 'high'
        elif len(risk_factors) > 2:
            return 'medium'
        elif len(risk_factors) > 0:
            return 'low'
        else:
            return 'very_low'

    def _identify_risk_factors(self, results: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors for overfitting."""
        risk_factors = []

        try:
            # Feature-related risks
            feature_complexity = results['feature_complexity']
            if feature_complexity.get('feature_sample_ratio', 0) > self.config.max_feature_ratio:
                risk_factors.append('High feature-to-sample ratio')

            if feature_complexity.get('feature_correlation_issues', 0) > 0:
                risk_factors.append('Highly correlated features')

            if feature_complexity.get('low_variance_features', 0) > feature_complexity.get('n_features', 1) * 0.1:
                risk_factors.append('Many low-variance features')

            # Model architecture risks
            arch_complexity = results['model_architecture_complexity']
            model_type = arch_complexity.get('model_type', '')

            if 'neural' in model_type:
                if not arch_complexity.get('has_dropout', False):
                    risk_factors.append('Neural network without dropout')
                if arch_complexity.get('total_parameters', 0) > self.config.max_parameters:
                    risk_factors.append('Excessive parameter count')

            if arch_complexity.get('max_depth', 0) > self.config.max_tree_depth:
                risk_factors.append('Excessively deep trees')

            # Data-related risks
            data_complexity = results['data_complexity']
            if data_complexity.get('class_imbalance_ratio', 1.0) > 10:
                risk_factors.append('Severe class imbalance')

            # Training-related risks
            training_complexity = results['training_complexity']
            if training_complexity.get('training_stability', 0) < 0.5:
                risk_factors.append('Unstable training process')

            if training_complexity.get('validation_effectiveness', 0) < 0.5:
                risk_factors.append('Ineffective validation')

        except Exception as e:
            self.logger.warning(f"Risk factor identification failed: {e}")
            risk_factors.append('Analysis error')

        return risk_factors

    def _generate_simplification_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for model simplification."""
        recommendations = []

        try:
            risk_level = results.get('overfitting_risk', 'low')

            if risk_level in ['high', 'very_high']:
                # High-level recommendations
                recommendations.extend([
                    "Consider implementing strong regularization (L1/L2)",
                    "Reduce model complexity (fewer layers/nodes/deeper trees)",
                    "Increase training data size",
                    "Implement early stopping with validation monitoring",
                    "Consider ensemble methods with simpler base models"
                ])

            # Feature-specific recommendations
            feature_complexity = results['feature_complexity']
            if feature_complexity.get('feature_correlation_issues', 0) > 0:
                recommendations.append("Remove or combine highly correlated features")

            if feature_complexity.get('low_variance_features', 0) > 0:
                recommendations.append("Remove low-variance features that don't contribute to prediction")

            if feature_complexity.get('feature_sample_ratio', 0) > self.config.max_feature_ratio:
                recommendations.append("Reduce feature dimensionality through selection or dimensionality reduction")

            # Model-specific recommendations
            arch_complexity = results['model_architecture_complexity']
            model_type = arch_complexity.get('model_type', '')

            if 'neural' in model_type and not arch_complexity.get('has_dropout', False):
                recommendations.append("Add dropout layers to neural network")

            if arch_complexity.get('max_depth', 0) > self.config.max_tree_depth:
                recommendations.append("Reduce tree depth to prevent overfitting")

            # Training-specific recommendations
            training_complexity = results['training_complexity']
            if training_complexity.get('training_stability', 0) < 0.7:
                recommendations.append("Improve training stability with better regularization or learning rate scheduling")

            # Data-specific recommendations
            data_complexity = results['data_complexity']
            if data_complexity.get('class_imbalance_ratio', 1.0) > 5:
                recommendations.append("Address class imbalance through resampling or class weighting")

            if risk_level == 'low':
                recommendations.append("Model complexity appears appropriate - consider increasing capacity if underfitting")

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Review model complexity and consider regularization")

        return recommendations

# Convenience functions
def create_model_complexity_analyzer(config: Optional[ModelComplexityAnalysisConfig] = None) -> ModelComplexityAnalyzer:
    """Create model complexity analyzer instance."""
    return ModelComplexityAnalyzer(config)

def analyze_model_complexity(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "unknown_model",
    feature_names: Optional[List[str]] = None,
    config: Optional[ModelComplexityAnalysisConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze model complexity.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Optional validation features
        y_val: Optional validation targets
        model_name: Name of the model
        feature_names: Optional feature names
        config: Optional configuration

    Returns:
        Dictionary containing complexity analysis results
    """
    analyzer = ModelComplexityAnalyzer(config)
    return analyzer.analyze_model_complexity(
        model, X_train, y_train, X_val, y_val, model_name, feature_names
    )