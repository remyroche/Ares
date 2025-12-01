"""
Model Complexity Analysis for ML Common

Comprehensive model complexity analysis and overfitting risk assessment with
simplification recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ModelComplexityConfig:
    """Configuration for model complexity analysis."""

    # Complexity scoring
    enable_complexity_scoring: bool = True
    complexity_weights: Dict[str, float] = None

    # Feature analysis
    enable_feature_analysis: bool = True
    max_features_to_analyze: int = 50
    feature_importance_threshold: float = 0.01

    # Overfitting risk assessment
    enable_overfitting_risk_assessment: bool = True
    overfitting_risk_weights: Dict[str, float] = None

    # Simplification analysis
    enable_simplification_analysis: bool = True
    simplification_targets: List[str] = None
    max_simplification_iterations: int = 5

    # Performance impact analysis
    enable_performance_impact_analysis: bool = True
    performance_impact_threshold: float = 0.05

    # Reporting
    save_complexity_reports: bool = True
    report_directory: str = "reports/model_complexity"
    enable_detailed_logging: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.complexity_weights is None:
            self.complexity_weights = {
                "n_parameters": 0.3,
                "max_depth": 0.2,
                "n_features": 0.2,
                "model_type": 0.15,
                "regularization": 0.15
            }
        if self.overfitting_risk_weights is None:
            self.overfitting_risk_weights = {
                "complexity_score": 0.4,
                "feature_concentration": 0.3,
                "performance_gap": 0.3
            }
        if self.simplification_targets is None:
            self.simplification_targets = ["feature_selection", "regularization", "architecture"]

@dataclass
class ComplexityAnalysisReport:
    """Comprehensive model complexity analysis report."""

    # Basic information
    model_name: str = "unknown"
    model_type: str = "unknown"
    analysis_timestamp: str = None

    # Complexity metrics
    overall_complexity_score: float = 0.0
    complexity_components: Dict[str, float] = None
    complexity_level: str = "unknown"  # low, medium, high, very_high

    # Feature analysis
    n_features_used: int = 0
    n_features_total: int = 0
    feature_importance_concentration: float = 0.0
    top_features: List[Dict[str, Any]] = None
    feature_redundancy_score: float = 0.0

    # Model structure analysis
    n_parameters: int = 0
    model_depth: int = 0
    architecture_complexity: float = 0.0
    regularization_strength: float = 0.0

    # Overfitting risk assessment
    overfitting_risk_score: float = 0.0
    overfitting_risk_level: str = "low"  # low, medium, high, critical
    overfitting_indicators: List[str] = None

    # Simplification analysis
    simplification_potential: float = 0.0
    recommended_simplifications: List[Dict[str, Any]] = None
    expected_performance_impact: Dict[str, float] = None

    # Performance vs complexity analysis
    performance_complexity_ratio: float = 0.0
    efficiency_score: float = 0.0
    optimal_complexity_range: Tuple[float, float] = (0.0, 1.0)

    # Recommendations
    primary_recommendations: List[str] = None
    detailed_suggestions: List[str] = None
    complexity_warnings: List[str] = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.complexity_components is None:
            self.complexity_components = {}
        if self.top_features is None:
            self.top_features = []
        if self.overfitting_indicators is None:
            self.overfitting_indicators = []
        if self.recommended_simplifications is None:
            self.recommended_simplifications = []
        if self.expected_performance_impact is None:
            self.expected_performance_impact = {}
        if self.primary_recommendations is None:
            self.primary_recommendations = []
        if self.detailed_suggestions is None:
            self.detailed_suggestions = []
        if self.complexity_warnings is None:
            self.complexity_warnings = []
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()

class ModelComplexityAnalyzer:
    """Comprehensive model complexity analysis and overfitting risk assessment."""

    def __init__(self, config: Optional[ModelComplexityConfig] = None):
        """
        Initialize model complexity analyzer.

        Args:
            config: Configuration for complexity analysis
        """
        self.config = config or ModelComplexityConfig()
        self.analysis_history = []

        # Create report directory
        if self.config.save_complexity_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Model Complexity Analyzer initialized")

    def analyze_model_complexity(self,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                model_name: str = "unknown",
                                model_type: str = "unknown",
                                X_test: Optional[np.ndarray] = None,
                                y_test: Optional[np.ndarray] = None,
                                feature_names: Optional[List[str]] = None) -> ComplexityAnalysisReport:
        """
        Perform comprehensive model complexity analysis.

        Args:
            model: Trained model to analyze
            X: Training feature matrix
            y: Training target vector
            model_name: Name of the model
            model_type: Type of the model
            X_test: Optional test feature matrix
            y_test: Optional test target vector
            feature_names: Optional feature names

        Returns:
            ComplexityAnalysisReport with comprehensive analysis
        """
        report = ComplexityAnalysisReport(
            model_name=model_name,
            model_type=model_type
        )

        try:
            # Basic model structure analysis
            if self.config.enable_complexity_scoring:
                structure_analysis = self._analyze_model_structure(model, model_type)
                report.n_parameters = structure_analysis['n_parameters']
                report.model_depth = structure_analysis['model_depth']
                report.architecture_complexity = structure_analysis['architecture_complexity']
                report.regularization_strength = structure_analysis['regularization_strength']

            # Feature analysis
            if self.config.enable_feature_analysis:
                feature_analysis = self._analyze_feature_usage(
                    model, X, feature_names, model_type
                )
                report.n_features_used = feature_analysis['n_features_used']
                report.n_features_total = feature_analysis['n_features_total']
                report.feature_importance_concentration = feature_analysis['concentration']
                report.top_features = feature_analysis['top_features']
                report.feature_redundancy_score = feature_analysis['redundancy_score']

            # Calculate overall complexity score
            report = self._calculate_complexity_score(report, model_type)

            # Overfitting risk assessment
            if self.config.enable_overfitting_risk_assessment:
                risk_analysis = self._assess_overfitting_risk(
                    model, X, y, X_test, y_test, model_type
                )
                report.overfitting_risk_score = risk_analysis['risk_score']
                report.overfitting_risk_level = risk_analysis['risk_level']
                report.overfitting_indicators = risk_analysis['indicators']

            # Simplification analysis
            if self.config.enable_simplification_analysis:
                simplification_analysis = self._analyze_simplification_potential(
                    model, X, y, model_type
                )
                report.simplification_potential = simplification_analysis['potential']
                report.recommended_simplifications = simplification_analysis['recommendations']
                report.expected_performance_impact = simplification_analysis['performance_impact']

            # Performance vs complexity analysis
            if self.config.enable_performance_impact_analysis:
                perf_analysis = self._analyze_performance_complexity_ratio(
                    model, X, y, report.overall_complexity_score
                )
                report.performance_complexity_ratio = perf_analysis['ratio']
                report.efficiency_score = perf_analysis['efficiency']
                report.optimal_complexity_range = perf_analysis['optimal_range']

            # Generate recommendations
            report = self._generate_complexity_recommendations(report)

            # Store report
            self.analysis_history.append(report)

            # Log results
            self._log_complexity_report(report)

            return report

        except Exception as e:
            logger.error(f"Model complexity analysis failed: {e}")
            report.complexity_warnings.append(f"Analysis failed: {str(e)}")
            return report

    def _analyze_model_structure(self, model: Any, model_type: str) -> Dict[str, Any]:
        """Analyze model structure and parameters."""
        try:
            analysis = {
                'n_parameters': 0,
                'model_depth': 0,
                'architecture_complexity': 0.5,
                'regularization_strength': 0.5
            }

            # Model-specific analysis
            if model_type.lower() in ['xgboost', 'lightgbm', 'catboost']:
                analysis['n_parameters'] = self._count_tree_parameters(model)
                analysis['model_depth'] = getattr(model, 'max_depth', 6)
                analysis['architecture_complexity'] = self._calculate_tree_complexity(model)
                analysis['regularization_strength'] = self._assess_regularization(model, model_type)

            elif model_type.lower() in ['random_forest', 'extra_trees']:
                analysis['n_parameters'] = self._count_ensemble_parameters(model)
                analysis['model_depth'] = getattr(model, 'max_depth', 10)
                analysis['architecture_complexity'] = self._calculate_ensemble_complexity(model)
                analysis['regularization_strength'] = 0.3  # Ensemble methods have built-in regularization

            elif model_type.lower() == 'neural_network':
                analysis['n_parameters'] = self._count_neural_network_parameters(model)
                analysis['model_depth'] = self._calculate_network_depth(model)
                analysis['architecture_complexity'] = self._calculate_network_complexity(model)
                analysis['regularization_strength'] = self._assess_network_regularization(model)

            else:
                # Generic analysis
                analysis['n_parameters'] = 1000  # Default estimate
                analysis['model_depth'] = 3
                analysis['architecture_complexity'] = 0.5
                analysis['regularization_strength'] = 0.5

            return analysis

        except Exception as e:
            logger.error(f"Model structure analysis failed: {e}")
            return {
                'n_parameters': 0,
                'model_depth': 0,
                'architecture_complexity': 0.5,
                'regularization_strength': 0.5
            }

    def _count_tree_parameters(self, model: Any) -> int:
        """Count parameters in tree-based models."""
        try:
            n_estimators = getattr(model, 'n_estimators', 100)
            max_depth = getattr(model, 'max_depth', 6)
            # Rough estimate: each tree has ~2^depth nodes, each node has parameters
            nodes_per_tree = sum(2**d for d in range(max_depth + 1))
            params_per_tree = nodes_per_tree * 3  # Rough estimate
            return n_estimators * params_per_tree
        except Exception as e:
            logger.error(f"Tree parameter counting failed: {e}")
            return 1000

    def _calculate_tree_complexity(self, model: Any) -> float:
        """Calculate complexity score for tree models."""
        try:
            n_estimators = getattr(model, 'n_estimators', 100)
            max_depth = getattr(model, 'max_depth', 6)
            complexity = min(1.0, (n_estimators / 200) * (max_depth / 10))
            return complexity
        except Exception as e:
            logger.error(f"Tree complexity calculation failed: {e}")
            return 0.5

    def _count_ensemble_parameters(self, model: Any) -> int:
        """Count parameters in ensemble models."""
        try:
            n_estimators = getattr(model, 'n_estimators', 100)
            max_depth = getattr(model, 'max_depth', 10)
            nodes_per_tree = sum(2**d for d in range(max_depth + 1))
            return n_estimators * nodes_per_tree * 2
        except Exception as e:
            logger.error(f"Ensemble parameter counting failed: {e}")
            return 5000

    def _calculate_ensemble_complexity(self, model: Any) -> float:
        """Calculate complexity score for ensemble models."""
        try:
            n_estimators = getattr(model, 'n_estimators', 100)
            complexity = min(1.0, n_estimators / 300)
            return complexity
        except Exception as e:
            logger.error(f"Ensemble complexity calculation failed: {e}")
            return 0.7

    def _count_neural_network_parameters(self, model: Any) -> int:
        """Count parameters in neural networks."""
        try:
            # This would need to be implemented based on the specific NN framework
            return getattr(model, 'count_params', lambda: 10000)()
        except Exception as e:
            logger.error(f"Neural network parameter counting failed: {e}")
            return 10000

    def _calculate_network_depth(self, model: Any) -> int:
        """Calculate depth of neural network."""
        try:
            # This would need to be implemented based on the specific NN framework
            return getattr(model, 'n_layers', 5)
        except Exception as e:
            logger.error(f"Network depth calculation failed: {e}")
            return 5

    def _calculate_network_complexity(self, model: Any) -> float:
        """Calculate complexity score for neural networks."""
        try:
            n_params = self._count_neural_network_parameters(model)
            complexity = min(1.0, n_params / 100000)
            return complexity
        except Exception as e:
            logger.error(f"Network complexity calculation failed: {e}")
            return 0.8

    def _assess_regularization(self, model: Any, model_type: str) -> float:
        """Assess regularization strength."""
        try:
            if model_type.lower() in ['xgboost', 'lightgbm', 'catboost']:
                reg_alpha = getattr(model, 'reg_alpha', 0)
                reg_lambda = getattr(model, 'reg_lambda', 1)
                regularization = (reg_alpha + reg_lambda) / 2
                return min(1.0, regularization)
            else:
                return 0.5  # Default regularization assessment
        except Exception as e:
            logger.error(f"Regularization assessment failed: {e}")
            return 0.5

    def _assess_network_regularization(self, model: Any) -> float:
        """Assess regularization in neural networks."""
        try:
            # This would need to be implemented based on the specific NN framework
            return 0.5
        except Exception as e:
            logger.error(f"Network regularization assessment failed: {e}")
            return 0.5

    def _analyze_feature_usage(self,
                              model: Any,
                              X: np.ndarray,
                              feature_names: Optional[List[str]],
                              model_type: str) -> Dict[str, Any]:
        """Analyze feature usage and importance."""
        try:
            analysis = {
                'n_features_used': X.shape[1],
                'n_features_total': X.shape[1],
                'concentration': 0.0,
                'top_features': [],
                'redundancy_score': 0.0
            }

            # Get feature importances
            feature_importances = self._get_feature_importances(model, X, model_type)

            if feature_importances:
                # Calculate concentration
                sorted_importances = sorted(feature_importances.values(), reverse=True)
                cumulative_importance = 0
                n_top_features = 0

                for importance in sorted_importances:
                    cumulative_importance += importance
                    n_top_features += 1
                    if cumulative_importance >= 0.8:  # 80% of importance
                        break

                analysis['concentration'] = n_top_features / len(feature_importances)

                # Get top features
                sorted_features = sorted(
                    feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 features

                for feature_name, importance in sorted_features:
                    analysis['top_features'].append({
                        'name': feature_name,
                        'importance': importance,
                        'relative_importance': importance / max(feature_importances.values())
                    })

                # Calculate redundancy score
                analysis['redundancy_score'] = self._calculate_feature_redundancy(
                    feature_importances, X
                )

            return analysis

        except Exception as e:
            logger.error(f"Feature usage analysis failed: {e}")
            return {
                'n_features_used': X.shape[1],
                'n_features_total': X.shape[1],
                'concentration': 0.0,
                'top_features': [],
                'redundancy_score': 0.0
            }

    def _get_feature_importances(self, model: Any, X: np.ndarray, model_type: str) -> Dict[str, float]:
        """Get feature importances from model."""
        try:
            importances = {}

            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                for i in range(len(model.feature_importances_)):
                    feature_name = f"feature_{i}" if X.shape[1] <= 10 else f"feature_{i:03d}"
                    importances[feature_name] = model.feature_importances_[i]

            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class if multi-class

                for i in range(len(coef)):
                    feature_name = f"feature_{i}"
                    importances[feature_name] = abs(coef[i])

            else:
                # Default: assume all features equally important
                for i in range(X.shape[1]):
                    feature_name = f"feature_{i}"
                    importances[feature_name] = 1.0 / X.shape[1]

            return importances

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}

    def _calculate_feature_redundancy(self, feature_importances: Dict[str, float], X: np.ndarray) -> float:
        """Calculate feature redundancy score."""
        try:
            if len(feature_importances) < 2:
                return 0.0

            # Calculate correlation between important features
            important_features = [
                name for name, importance in feature_importances.items()
                if importance > np.mean(list(feature_importances.values()))
            ]

            if len(important_features) < 2:
                return 0.0

            # This is a simplified redundancy calculation
            # In practice, you'd calculate correlations between features
            redundancy = 0.0

            # Simple heuristic: if many features have similar importance
            importances = list(feature_importances.values())
            importance_std = np.std(importances)
            importance_mean = np.mean(importances)

            if importance_mean > 0:
                redundancy = min(1.0, importance_std / importance_mean)

            return redundancy

        except Exception as e:
            logger.error(f"Feature redundancy calculation failed: {e}")
            return 0.0

    def _calculate_complexity_score(self, report: ComplexityAnalysisReport, model_type: str) -> ComplexityAnalysisReport:
        """Calculate overall complexity score."""
        try:
            components = {}

            # Parameter complexity
            if report.n_parameters > 0:
                param_complexity = min(1.0, report.n_parameters / 10000)
                components['n_parameters'] = param_complexity

            # Depth complexity
            if report.model_depth > 0:
                depth_complexity = min(1.0, report.model_depth / 15)
                components['max_depth'] = depth_complexity

            # Feature complexity
            if report.n_features_used > 0:
                feature_complexity = min(1.0, report.n_features_used / 50)
                components['n_features'] = feature_complexity

            # Model type complexity
            type_complexity = self._get_model_type_complexity(model_type)
            components['model_type'] = type_complexity

            # Regularization effect (inverse - higher regularization = lower complexity)
            reg_complexity = 1.0 - report.regularization_strength
            components['regularization'] = reg_complexity

            # Calculate weighted score
            total_weight = sum(self.config.complexity_weights.values())
            complexity_score = 0.0

            for component, weight in self.config.complexity_weights.items():
                if component in components:
                    complexity_score += components[component] * (weight / total_weight)

            report.overall_complexity_score = complexity_score
            report.complexity_components = components

            # Determine complexity level
            if complexity_score < 0.3:
                report.complexity_level = "low"
            elif complexity_score < 0.6:
                report.complexity_level = "medium"
            elif complexity_score < 0.8:
                report.complexity_level = "high"
            else:
                report.complexity_level = "very_high"

            return report

        except Exception as e:
            logger.error(f"Complexity score calculation failed: {e}")
            report.overall_complexity_score = 0.5
            report.complexity_level = "unknown"
            return report

    def _get_model_type_complexity(self, model_type: str) -> float:
        """Get base complexity score for model type."""
        complexity_map = {
            'linear_regression': 0.1,
            'logistic_regression': 0.2,
            'random_forest': 0.4,
            'xgboost': 0.5,
            'lightgbm': 0.5,
            'catboost': 0.5,
            'neural_network': 0.7,
            'deep_learning': 0.9
        }

        return complexity_map.get(model_type.lower(), 0.5)

    def _assess_overfitting_risk(self,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                X_test: Optional[np.ndarray],
                                y_test: Optional[np.ndarray],
                                model_type: str) -> Dict[str, Any]:
        """Assess overfitting risk."""
        try:
            analysis = {
                'risk_score': 0.0,
                'risk_level': 'low',
                'indicators': []
            }

            # Train vs test performance gap
            if X_test is not None and y_test is not None:
                train_score = self._evaluate_model(model, X, y, True)  # Assuming classification
                test_score = self._evaluate_model(model, X_test, y_test, True)

                if train_score > 0:
                    gap = (train_score - test_score) / train_score

                    if gap > 0.2:
                        analysis['risk_score'] += 0.4
                        analysis['indicators'].append(f"Large train-test gap: {gap:.2f}")
                        analysis['risk_level'] = 'high'
                    elif gap > 0.1:
                        analysis['risk_score'] += 0.2
                        analysis['indicators'].append(f"Moderate train-test gap: {gap:.2f}")
                        analysis['risk_level'] = 'medium'

            # Feature concentration risk
            if hasattr(self, '_analyze_feature_usage'):
                # Calculate actual feature concentration based on feature importance
                concentration = self._calculate_feature_concentration(model, X)
                if concentration > 0.8:
                    analysis['risk_score'] += 0.3
                    analysis['indicators'].append(f"High feature concentration: {concentration:.2f}")
                    if analysis['risk_level'] == 'low':
                        analysis['risk_level'] = 'medium'

            # Complexity-based risk
            complexity_risk = min(1.0, analysis['risk_score'] + 0.2)  # Complexity adds risk
            analysis['risk_score'] = complexity_risk

            # Determine final risk level
            if analysis['risk_score'] > 0.7:
                analysis['risk_level'] = 'critical'
            elif analysis['risk_score'] > 0.4:
                analysis['risk_level'] = 'high'
            elif analysis['risk_score'] > 0.2:
                analysis['risk_level'] = 'medium'

            return analysis

        except Exception as e:
            logger.error(f"Overfitting risk assessment failed: {e}")
            return {
                'risk_score': 0.5,
                'risk_level': 'unknown',
                'indicators': [f'Assessment failed: {str(e)}']
            }

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, is_classification: bool) -> float:
        """Evaluate model performance."""
        try:
            # Some callers may accidentally pass a wrapper dict or other objects
            # without a predict() method (e.g., training metadata). In that case,
            # skip evaluation gracefully instead of logging a noisy error.
            if not hasattr(model, "predict"):
                logger.warning(
                    "Model evaluation skipped: provided model has no predict() method "
                    f"(type={type(model).__name__})"
                )
                return 0.0

            if is_classification:
                y_pred = model.predict(X)
                return accuracy_score(y, y_pred)
            else:
                y_pred = model.predict(X)
                return 1 - np.mean((y - y_pred) ** 2) / np.var(y)
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0

    def _analyze_simplification_potential(self,
                                        model: Any,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        model_type: str) -> Dict[str, Any]:
        """Analyze potential for model simplification."""
        try:
            analysis = {
                'potential': 0.0,
                'recommendations': [],
                'performance_impact': {}
            }

            # Feature selection potential
            if self._check_feature_selection_potential(model, X, y):
                analysis['potential'] += 0.3
                analysis['recommendations'].append({
                    'type': 'feature_selection',
                    'description': 'Remove redundant or low-importance features',
                    'expected_impact': 'minimal'
                })
                analysis['performance_impact']['feature_selection'] = -0.05

            # Regularization potential
            if self._check_regularization_potential(model, model_type):
                analysis['potential'] += 0.2
                analysis['recommendations'].append({
                    'type': 'regularization',
                    'description': 'Increase regularization strength',
                    'expected_impact': 'minimal'
                })
                analysis['performance_impact']['regularization'] = -0.02

            # Architecture simplification potential
            if self._check_architecture_simplification_potential(model, model_type):
                analysis['potential'] += 0.4
                analysis['recommendations'].append({
                    'type': 'architecture',
                    'description': 'Reduce model depth or width',
                    'expected_impact': 'moderate'
                })
                analysis['performance_impact']['architecture'] = -0.1

            analysis['potential'] = min(1.0, analysis['potential'])

            return analysis

        except Exception as e:
            logger.error(f"Simplification potential analysis failed: {e}")
            return {
                'potential': 0.0,
                'recommendations': [],
                'performance_impact': {}
            }

    def _check_feature_selection_potential(self, model: Any, X: np.ndarray, y: np.ndarray) -> bool:
        """Check if feature selection would be beneficial."""
        try:
            # Simple heuristic: if many features have low importance
            feature_importances = self._get_feature_importances(model, X, 'generic')
            low_importance_features = [
                name for name, importance in feature_importances.items()
                if importance < 0.01
            ]
            return len(low_importance_features) > len(feature_importances) * 0.3
        except Exception as e:
            logger.error(f"Feature selection potential check failed: {e}")
            return False

    def _check_regularization_potential(self, model: Any, model_type: str) -> bool:
        """Check if increased regularization would be beneficial."""
        try:
            # Check current regularization strength based on model type
            current_reg = self._get_current_regularization_strength(model, model_type)
            return current_reg < 0.7  # Room for more regularization
        except Exception as e:
            logger.error(f"Regularization potential check failed: {e}")
            return False

    def _check_architecture_simplification_potential(self, model: Any, model_type: str) -> bool:
        """Check if architecture can be simplified."""
        try:
            # Check if model is very complex based on actual model structure
            complexity = self._calculate_model_architecture_complexity(model, model_type)
            return complexity > 0.7
        except Exception as e:
            logger.error(f"Architecture simplification potential check failed: {e}")
            return False

    def _analyze_performance_complexity_ratio(self,
                                             model: Any,
                                             X: np.ndarray,
                                             y: np.ndarray,
                                             complexity_score: float) -> Dict[str, Any]:
        """Analyze performance vs complexity ratio."""
        try:
            # Get model performance
            performance = self._evaluate_model(model, X, y, True)

            # Calculate ratio
            if complexity_score > 0:
                ratio = performance / complexity_score
            else:
                ratio = performance

            # Calculate efficiency score
            efficiency = min(1.0, ratio * 2)  # Normalize

            # Determine optimal complexity range
            optimal_min = max(0.1, performance - 0.2)
            optimal_max = min(1.0, performance + 0.2)

            return {
                'ratio': ratio,
                'efficiency': efficiency,
                'optimal_range': (optimal_min, optimal_max)
            }

        except Exception as e:
            logger.error(f"Performance-complexity analysis failed: {e}")
            return {
                'ratio': 1.0,
                'efficiency': 0.5,
                'optimal_range': (0.3, 0.7)
            }

    def _generate_complexity_recommendations(self, report: ComplexityAnalysisReport) -> ComplexityAnalysisReport:
        """Generate complexity-based recommendations."""
        try:
            # Complexity level recommendations
            if report.complexity_level == "very_high":
                report.primary_recommendations.append("Model complexity is very high - consider significant simplification")
                report.complexity_warnings.append("Very high complexity may lead to overfitting")

            elif report.complexity_level == "high":
                report.primary_recommendations.append("High model complexity - monitor for overfitting")

            # Overfitting risk recommendations
            if report.overfitting_risk_level == "critical":
                report.primary_recommendations.append("Critical overfitting risk - immediate action required")
                report.complexity_warnings.append("Critical overfitting risk detected")

            elif report.overfitting_risk_level == "high":
                report.primary_recommendations.append("High overfitting risk - implement stronger regularization")

            # Feature concentration recommendations
            if report.feature_importance_concentration > 0.8:
                report.detailed_suggestions.append("High feature concentration - consider feature selection")

            # Simplification potential recommendations
            if report.simplification_potential > 0.5:
                report.detailed_suggestions.append("High simplification potential - explore model reduction techniques")

            # Performance vs complexity recommendations
            if report.efficiency_score < 0.5:
                report.detailed_suggestions.append("Low efficiency score - model may be overly complex for performance")

            # Optimal complexity recommendations
            if not (report.optimal_complexity_range[0] <= report.overall_complexity_score <= report.optimal_complexity_range[1]):
                report.detailed_suggestions.append(f"Complexity outside optimal range - consider adjusting to {report.optimal_complexity_range}")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")

        return report

    def _log_complexity_report(self, report: ComplexityAnalysisReport):
        """Log complexity analysis results."""
        if not self.config.enable_detailed_logging:
            return

        logger.info(f"Model Complexity Analysis for {report.model_name}:")
        logger.info(f"  Overall Complexity Score: {report.overall_complexity_score:.3f} ({report.complexity_level})")
        logger.info(f"  Overfitting Risk: {report.overfitting_risk_score:.3f} ({report.overfitting_risk_level})")
        logger.info(f"  Features Used: {report.n_features_used}/{report.n_features_total}")
        logger.info(f"  Feature Concentration: {report.feature_importance_concentration:.3f}")
        logger.info(f"  Simplification Potential: {report.simplification_potential:.3f}")

        if report.complexity_warnings:
            for warning in report.complexity_warnings:
                logger.warning(f"  Warning: {warning}")

        if report.primary_recommendations:
            logger.info(f"  Primary Recommendations: {len(report.primary_recommendations)}")
            for rec in report.primary_recommendations[:2]:
                logger.info(f"    - {rec}")

    def save_complexity_report(self, report: ComplexityAnalysisReport, filename: Optional[str] = None):
        """Save complexity analysis report to file."""
        if not self.config.save_complexity_reports:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complexity_analysis_{report.model_name}_{timestamp}.json"

        filepath = Path(self.config.report_directory) / filename

        try:
            report_dict = asdict(report)
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Complexity analysis report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save complexity analysis report: {e}")

    def get_analysis_history(self) -> List[ComplexityAnalysisReport]:
        """Get analysis history."""
        return self.analysis_history.copy()

    def _calculate_feature_concentration(self, model: Any, X: np.ndarray) -> float:
        """Calculate feature concentration based on feature importance."""
        try:
            feature_importances = self._get_feature_importances(model, X, "unknown")
            if not feature_importances:
                return 0.5  # Default if no feature importance available

            # Calculate concentration as the ratio of top features to total features
            total_features = len(feature_importances)
            if total_features == 0:
                return 0.5

            # Sort features by importance and get top 20%
            sorted_importances = sorted(feature_importances.values(), reverse=True)
            top_20_percent = max(1, int(total_features * 0.2))
            top_features_importance = sum(sorted_importances[:top_20_percent])
            total_importance = sum(sorted_importances)

            if total_importance == 0:
                return 0.5

            concentration = top_features_importance / total_importance
            return min(1.0, max(0.0, concentration))
        except Exception as e:
            logger.warning(f"Failed to calculate feature concentration: {e}")
            return 0.5

    def _get_current_regularization_strength(self, model: Any, model_type: str) -> float:
        """Get current regularization strength of the model."""
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()

                # Check for common regularization parameters
                if model_type.lower() in ['random_forest', 'extra_trees']:
                    # For tree-based models, check min_samples_split, min_samples_leaf
                    min_samples_split = params.get('min_samples_split', 2)
                    min_samples_leaf = params.get('min_samples_leaf', 1)
                    # Higher values = more regularization
                    reg_strength = min(1.0, (min_samples_split + min_samples_leaf) / 20.0)
                    return reg_strength

                elif model_type.lower() in ['linear', 'logistic', 'ridge', 'lasso', 'elasticnet']:
                    # For linear models, check alpha/C parameters
                    alpha = params.get('alpha', 1.0)
                    C = params.get('C', 1.0)
                    # Convert to regularization strength (higher alpha/lower C = more regularization)
                    if 'alpha' in params:
                        reg_strength = min(1.0, alpha / 10.0)
                    elif 'C' in params:
                        reg_strength = min(1.0, 1.0 / C)
                    else:
                        reg_strength = 0.1
                    return reg_strength

                elif model_type.lower() in ['neural_network', 'mlp']:
                    # For neural networks, check dropout, weight decay
                    dropout = params.get('dropout_rate', 0.0)
                    weight_decay = params.get('weight_decay', 0.0)
                    reg_strength = min(1.0, (dropout + weight_decay) / 2.0)
                    return reg_strength

            return 0.1  # Default low regularization
        except Exception as e:
            logger.warning(f"Failed to get regularization strength: {e}")
            return 0.1

    def _calculate_model_architecture_complexity(self, model: Any, model_type: str) -> float:
        """Calculate model architecture complexity."""
        try:
            if model_type.lower() in ['random_forest', 'extra_trees']:
                # For tree-based models, complexity based on number of trees and depth
                n_estimators = getattr(model, 'n_estimators', 100)
                max_depth = getattr(model, 'max_depth', None)
                if max_depth is None:
                    max_depth = 10  # Default assumption

                # Normalize complexity (0-1 scale)
                complexity = min(1.0, (n_estimators * max_depth) / 10000.0)
                return complexity

            elif model_type.lower() in ['neural_network', 'mlp']:
                # For neural networks, complexity based on layers and neurons
                if hasattr(model, 'hidden_layer_sizes'):
                    hidden_sizes = model.hidden_layer_sizes
                    if isinstance(hidden_sizes, int):
                        total_neurons = hidden_sizes
                    else:
                        total_neurons = sum(hidden_sizes)
                    complexity = min(1.0, total_neurons / 1000.0)
                    return complexity
                else:
                    return 0.5  # Default medium complexity

            elif model_type.lower() in ['linear', 'logistic', 'ridge', 'lasso', 'elasticnet']:
                # Linear models are generally simple
                return 0.2

            else:
                # For unknown models, try to estimate based on parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    param_count = len(params)
                    complexity = min(1.0, param_count / 50.0)
                    return complexity
                else:
                    return 0.5  # Default medium complexity

        except Exception as e:
            logger.warning(f"Failed to calculate architecture complexity: {e}")
            return 0.5

# Global instance
DEFAULT_COMPLEXITY_ANALYZER = ModelComplexityAnalyzer()

def get_model_complexity_analyzer(config: Optional[ModelComplexityConfig] = None) -> ModelComplexityAnalyzer:
    """Get model complexity analyzer instance."""
    if config is None:
        return DEFAULT_COMPLEXITY_ANALYZER
    return ModelComplexityAnalyzer(config)

def analyze_model_complexity(model: Any,
                            X: np.ndarray,
                            y: np.ndarray,
                            model_name: str = "unknown",
                            model_type: str = "unknown",
                            X_test: Optional[np.ndarray] = None,
                            y_test: Optional[np.ndarray] = None,
                            feature_names: Optional[List[str]] = None) -> ComplexityAnalysisReport:
    """Convenience function to analyze model complexity."""
    analyzer = get_model_complexity_analyzer()
    return analyzer.analyze_model_complexity(
        model, X, y, model_name, model_type, X_test, y_test, feature_names
    )
