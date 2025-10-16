"""
Enhanced Overfitting Detection with Learning Curve Analysis Integration

This module extends the existing overfitting detection system to include:
- Learning curve analysis for overfitting pattern detection
- VC dimension and Rademacher complexity metrics
- Feature interaction overfitting detection
- Comprehensive model complexity analysis
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings

# Import existing overfitting detection
from .enhanced_overfitting_detection import (
    UniversalOverfittingDetector, 
    OverfittingConfig, 
    OverfittingReport
)

# Import learning curve analysis
from ..evaluation.enhanced_learning_curve_analysis import (
    EnhancedLearningCurveAnalyzer,
    LearningCurveAnalysisResult
)

logger = logging.getLogger(__name__)


@dataclass
class ModelComplexityMetrics:
    """Model complexity metrics for overfitting detection."""
    
    # VC Dimension (approximate)
    vc_dimension: float
    vc_bound: float  # VC generalization bound
    
    # Rademacher Complexity
    rademacher_complexity: float
    rademacher_bound: float  # Rademacher generalization bound
    
    # Model-specific complexity
    parameter_count: int
    effective_parameters: float  # For regularized models
    
    # Feature complexity
    feature_interaction_strength: float
    feature_nonlinearity: float
    
    # Model capacity
    model_capacity: float
    capacity_utilization: float  # How much capacity is being used


@dataclass
class FeatureInteractionAnalysis:
    """Feature interaction analysis for overfitting detection."""
    
    # Interaction strength
    pairwise_interactions: Dict[Tuple[str, str], float]
    high_interaction_pairs: List[Tuple[str, str]]
    interaction_matrix: np.ndarray
    
    # Nonlinear interactions
    nonlinear_interactions: Dict[Tuple[str, str], float]
    polynomial_interactions: Dict[Tuple[str, str], float]
    
    # Overfitting indicators
    interaction_overfitting_score: float
    suspicious_interactions: List[Dict[str, Any]]
    
    # Recommendations
    interaction_recommendations: List[str]


class EnhancedOverfittingDetectorWithLearningCurves(UniversalOverfittingDetector):
    """Enhanced overfitting detector with learning curve analysis and complexity metrics."""
    
    def __init__(self, config: Optional[OverfittingConfig] = None):
        """Initialize enhanced overfitting detector."""
        super().__init__(config)
        self.learning_curve_analyzer = EnhancedLearningCurveAnalyzer()
        self.complexity_history = []
        self.interaction_history = []
        
        logger.info("✅ Enhanced Overfitting Detector with Learning Curves initialized")
    
    def detect_overfitting_with_learning_curves(self, 
                                               model: Any,
                                               X_train: np.ndarray,
                                               X_val: np.ndarray,
                                               y_train: np.ndarray,
                                               y_val: np.ndarray,
                                               X_test: Optional[np.ndarray] = None,
                                               y_test: Optional[np.ndarray] = None,
                                               model_name: str = "unknown",
                                               model_type: str = "unknown",
                                               fold_number: Optional[int] = None) -> OverfittingReport:
        """
        Detect overfitting with comprehensive learning curve analysis.
        
        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            model_name: Name of the model
            model_type: Type of model
            fold_number: Fold number for cross-validation
            
        Returns:
            Enhanced OverfittingReport with learning curve analysis
        """
        try:
            # Get basic predictions
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            
            # Get probabilities if available
            train_probabilities = None
            val_probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    train_probabilities = model.predict_proba(X_train)
                    val_probabilities = model.predict_proba(X_val)
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_).flatten()
            
            # Perform basic overfitting detection
            basic_report = self.detect_overfitting(
                train_predictions=train_predictions,
                val_predictions=val_predictions,
                train_labels=y_train,
                val_labels=y_val,
                train_probabilities=train_probabilities,
                val_probabilities=val_probabilities,
                feature_importance=feature_importance,
                model_name=model_name,
                model_type=model_type,
                fold_number=fold_number
            )
            
            # Perform learning curve analysis
            learning_curve_result = self._analyze_learning_curves(
                model, X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Calculate model complexity metrics
            complexity_metrics = self._calculate_model_complexity(
                model, X_train, y_train, model_type
            )
            
            # Analyze feature interactions
            feature_interactions = self._analyze_feature_interactions(
                model, X_train, y_train, X_val, y_val
            )
            
            # Enhance the basic report with additional analysis
            enhanced_report = self._enhance_report_with_analysis(
                basic_report, learning_curve_result, complexity_metrics, feature_interactions
            )
            
            # Track history
            self._track_enhanced_detection(enhanced_report, complexity_metrics, feature_interactions)
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Enhanced overfitting detection failed: {e}")
            return self._create_error_report(str(e), model_name, model_type, fold_number)
    
    def _analyze_learning_curves(self, 
                                model: Any,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                X_test: Optional[np.ndarray] = None,
                                y_test: Optional[np.ndarray] = None) -> LearningCurveAnalysisResult:
        """Analyze learning curves for overfitting patterns."""
        try:
            # Combine train and val for learning curve analysis
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            # Determine scoring metric
            is_classification = len(np.unique(y_combined)) <= 10
            scoring = 'accuracy' if is_classification else 'r2'
            
            # Perform learning curve analysis
            result = self.learning_curve_analyzer.analyze_learning_curve(
                model=model,
                X_train=X_combined,
                y_train=y_combined,
                X_test=X_test if X_test is not None else X_val,
                y_test=y_test if y_test is not None else y_val,
                scoring=scoring
            )
            
            logger.debug(f"✅ Learning curve analysis completed: {result.overfitting_risk}")
            return result
            
        except Exception as e:
            logger.error(f"Learning curve analysis failed: {e}")
            # Return default result
            return LearningCurveAnalysisResult(
                learning_rate="unknown",
                convergence_stability="unknown",
                overfitting_risk="unknown",
                training_efficiency="unknown",
                max_score_gap=0.0,
                final_score_gap=0.0,
                early_learning_slope=0.0,
                convergence_stability_score=0.0,
                train_sizes=[],
                train_scores_mean=[],
                train_scores_std=[],
                val_scores_mean=[],
                val_scores_std=[],
                score_gaps=[]
            )
    
    def _calculate_model_complexity(self, 
                                  model: Any,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  model_type: str) -> ModelComplexityMetrics:
        """Calculate model complexity metrics."""
        try:
            # Calculate VC dimension (approximate)
            vc_dimension = self._estimate_vc_dimension(model, X, y, model_type)
            vc_bound = self._calculate_vc_bound(vc_dimension, len(X), len(y))
            
            # Calculate Rademacher complexity
            rademacher_complexity = self._estimate_rademacher_complexity(model, X, y, model_type)
            rademacher_bound = self._calculate_rademacher_bound(rademacher_complexity, len(X))
            
            # Calculate parameter count
            parameter_count = self._count_parameters(model)
            effective_parameters = self._estimate_effective_parameters(model, X, y)
            
            # Calculate feature complexity
            feature_interaction_strength = self._calculate_feature_interaction_strength(X, y)
            feature_nonlinearity = self._calculate_feature_nonlinearity(X, y)
            
            # Calculate model capacity
            model_capacity = self._estimate_model_capacity(model, model_type)
            capacity_utilization = effective_parameters / parameter_count if parameter_count > 0 else 0
            
            return ModelComplexityMetrics(
                vc_dimension=vc_dimension,
                vc_bound=vc_bound,
                rademacher_complexity=rademacher_complexity,
                rademacher_bound=rademacher_bound,
                parameter_count=parameter_count,
                effective_parameters=effective_parameters,
                feature_interaction_strength=feature_interaction_strength,
                feature_nonlinearity=feature_nonlinearity,
                model_capacity=model_capacity,
                capacity_utilization=capacity_utilization
            )
            
        except Exception as e:
            logger.error(f"Model complexity calculation failed: {e}")
            return ModelComplexityMetrics(
                vc_dimension=0.0, vc_bound=0.0, rademacher_complexity=0.0, rademacher_bound=0.0,
                parameter_count=0, effective_parameters=0.0, feature_interaction_strength=0.0,
                feature_nonlinearity=0.0, model_capacity=0.0, capacity_utilization=0.0
            )
    
    def _analyze_feature_interactions(self, 
                                    model: Any,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray) -> FeatureInteractionAnalysis:
        """Analyze feature interactions for overfitting detection."""
        try:
            # Calculate pairwise interactions
            pairwise_interactions = self._calculate_pairwise_interactions(X_train, y_train)
            high_interaction_pairs = [
                pair for pair, strength in pairwise_interactions.items() 
                if strength > 0.7  # High interaction threshold
            ]
            
            # Calculate interaction matrix
            n_features = X_train.shape[1]
            interaction_matrix = np.zeros((n_features, n_features))
            for (i, j), strength in pairwise_interactions.items():
                interaction_matrix[i, j] = strength
                interaction_matrix[j, i] = strength
            
            # Calculate nonlinear interactions
            nonlinear_interactions = self._calculate_nonlinear_interactions(X_train, y_train)
            polynomial_interactions = self._calculate_polynomial_interactions(X_train, y_train)
            
            # Calculate overfitting score
            interaction_overfitting_score = self._calculate_interaction_overfitting_score(
                pairwise_interactions, nonlinear_interactions, X_train, X_val, y_train, y_val
            )
            
            # Identify suspicious interactions
            suspicious_interactions = self._identify_suspicious_interactions(
                pairwise_interactions, nonlinear_interactions, interaction_overfitting_score
            )
            
            # Generate recommendations
            interaction_recommendations = self._generate_interaction_recommendations(
                high_interaction_pairs, suspicious_interactions, interaction_overfitting_score
            )
            
            return FeatureInteractionAnalysis(
                pairwise_interactions=pairwise_interactions,
                high_interaction_pairs=high_interaction_pairs,
                interaction_matrix=interaction_matrix,
                nonlinear_interactions=nonlinear_interactions,
                polynomial_interactions=polynomial_interactions,
                interaction_overfitting_score=interaction_overfitting_score,
                suspicious_interactions=suspicious_interactions,
                interaction_recommendations=interaction_recommendations
            )
            
        except Exception as e:
            logger.error(f"Feature interaction analysis failed: {e}")
            return FeatureInteractionAnalysis(
                pairwise_interactions={}, high_interaction_pairs=[], interaction_matrix=np.array([]),
                nonlinear_interactions={}, polynomial_interactions={}, interaction_overfitting_score=0.0,
                suspicious_interactions=[], interaction_recommendations=[]
            )
    
    def _estimate_vc_dimension(self, model: Any, X: np.ndarray, y: np.ndarray, model_type: str) -> float:
        """Estimate VC dimension for the model."""
        try:
            n_samples, n_features = X.shape
            
            if model_type.lower() in ['linear', 'logistic']:
                # Linear models: VC dimension ≈ number of parameters
                if hasattr(model, 'coef_'):
                    return float(model.coef_.size)
                else:
                    return float(n_features + 1)  # +1 for bias
            
            elif model_type.lower() in ['tree', 'decision']:
                # Decision trees: VC dimension ≈ number of leaves
                if hasattr(model, 'tree_'):
                    return float(model.tree_.n_leaves)
                else:
                    return float(2 ** min(10, n_features))  # Conservative estimate
            
            elif model_type.lower() in ['forest', 'random']:
                # Random forests: VC dimension ≈ sum of tree VC dimensions
                if hasattr(model, 'estimators_'):
                    total_leaves = sum(
                        estimator.tree_.n_leaves if hasattr(estimator, 'tree_') else 1
                        for estimator in model.estimators_
                    )
                    return float(total_leaves)
                else:
                    return float(100 * 2 ** min(5, n_features))  # Conservative estimate
            
            elif model_type.lower() in ['neural', 'mlp']:
                # Neural networks: VC dimension ≈ number of parameters
                if hasattr(model, 'coefs_'):
                    total_params = sum(coef.size for coef in model.coefs_)
                    return float(total_params)
                else:
                    return float(n_features * 10)  # Conservative estimate
            
            else:
                # Default: conservative estimate based on parameters
                return float(min(n_samples, n_features * 10))
                
        except Exception as e:
            logger.warning(f"VC dimension estimation failed: {e}")
            return float(X.shape[1] * 5)  # Conservative fallback
    
    def _calculate_vc_bound(self, vc_dimension: float, n_samples: int, n_classes: int) -> float:
        """Calculate VC generalization bound."""
        try:
            if vc_dimension <= 0 or n_samples <= 0:
                return 1.0
            
            # VC bound: sqrt((d * log(2n/d) + log(1/delta)) / n)
            # Using delta = 0.05 (95% confidence)
            delta = 0.05
            bound = np.sqrt((vc_dimension * np.log(2 * n_samples / vc_dimension) + np.log(1 / delta)) / n_samples)
            return min(1.0, max(0.0, bound))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"VC bound calculation failed: {e}")
            return 1.0
    
    def _estimate_rademacher_complexity(self, model: Any, X: np.ndarray, y: np.ndarray, model_type: str) -> float:
        """Estimate Rademacher complexity for the model."""
        try:
            n_samples = len(X)
            
            # Generate Rademacher variables
            n_rademacher_samples = min(100, n_samples)
            rademacher_vars = np.random.choice([-1, 1], size=(n_rademacher_samples, n_samples))
            
            # Calculate empirical Rademacher complexity
            rademacher_complexities = []
            
            for rad_vars in rademacher_vars:
                # Calculate model output on Rademacher-weighted data
                try:
                    # This is a simplified estimation - in practice, you'd need model-specific calculations
                    model_output = model.predict(X[:n_rademacher_samples])
                    if len(model_output.shape) > 1:
                        model_output = model_output.flatten()
                    
                    # Calculate Rademacher complexity
                    rad_complexity = np.mean(rad_vars * model_output)
                    rademacher_complexities.append(abs(rad_complexity))
                    
                except Exception as e:
                    logger.warning(f"Rademacher complexity calculation failed for sample: {e}")
                    continue
            
            if rademacher_complexities:
                return float(np.mean(rademacher_complexities))
            else:
                # Fallback: use model complexity heuristic
                return float(1.0 / np.sqrt(n_samples))
                
        except Exception as e:
            logger.warning(f"Rademacher complexity estimation failed: {e}")
            return float(1.0 / np.sqrt(len(X)))
    
    def _calculate_rademacher_bound(self, rademacher_complexity: float, n_samples: int) -> float:
        """Calculate Rademacher generalization bound."""
        try:
            if rademacher_complexity <= 0 or n_samples <= 0:
                return 1.0
            
            # Rademacher bound: 2 * R_n(F) + sqrt(log(1/delta) / (2n))
            # Using delta = 0.05 (95% confidence)
            delta = 0.05
            bound = 2 * rademacher_complexity + np.sqrt(np.log(1 / delta) / (2 * n_samples))
            return min(1.0, max(0.0, bound))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Rademacher bound calculation failed: {e}")
            return 1.0
    
    def _count_parameters(self, model: Any) -> int:
        """Count the number of parameters in the model."""
        try:
            if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                # Linear models
                coef_params = model.coef_.size if hasattr(model.coef_, 'size') else 0
                intercept_params = 1 if model.intercept_ is not None else 0
                return coef_params + intercept_params
            
            elif hasattr(model, 'coefs_'):
                # Neural networks
                total_params = sum(coef.size for coef in model.coefs_)
                if hasattr(model, 'intercepts_'):
                    total_params += sum(intercept.size for intercept in model.intercepts_)
                return total_params
            
            elif hasattr(model, 'estimators_'):
                # Ensemble models
                total_params = 0
                for estimator in model.estimators_:
                    total_params += self._count_parameters(estimator)
                return total_params
            
            elif hasattr(model, 'tree_'):
                # Decision trees
                return model.tree_.n_nodes
            
            else:
                # Fallback: estimate based on model type
                return 100  # Conservative estimate
                
        except Exception as e:
            logger.warning(f"Parameter counting failed: {e}")
            return 0
    
    def _estimate_effective_parameters(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate effective number of parameters (for regularized models)."""
        try:
            # This is a simplified estimation
            # In practice, you'd need model-specific calculations
            
            if hasattr(model, 'alpha') and model.alpha > 0:
                # Regularized models: effective parameters < total parameters
                total_params = self._count_parameters(model)
                regularization_strength = model.alpha
                effective_params = total_params / (1 + regularization_strength)
                return float(effective_params)
            
            else:
                # Non-regularized models: effective parameters ≈ total parameters
                return float(self._count_parameters(model))
                
        except Exception as e:
            logger.warning(f"Effective parameter estimation failed: {e}")
            return float(self._count_parameters(model))
    
    def _calculate_feature_interaction_strength(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate overall feature interaction strength."""
        try:
            n_features = X.shape[1]
            if n_features < 2:
                return 0.0
            
            # Calculate pairwise correlations
            correlations = np.corrcoef(X.T)
            np.fill_diagonal(correlations, 0)  # Remove self-correlations
            
            # Calculate average absolute correlation
            avg_correlation = np.mean(np.abs(correlations))
            
            # Calculate mutual information between features
            if n_features <= 10:  # Only for small feature sets
                try:
                    mi_scores = []
                    for i in range(n_features):
                        for j in range(i + 1, n_features):
                            mi = mutual_info_classif(X[:, [i, j]], y, discrete_features=False)[0]
                            mi_scores.append(mi)
                    
                    avg_mi = np.mean(mi_scores) if mi_scores else 0.0
                except Exception:
                    avg_mi = 0.0
            else:
                avg_mi = 0.0
            
            # Combine correlation and mutual information
            interaction_strength = (avg_correlation + avg_mi) / 2
            return float(interaction_strength)
            
        except Exception as e:
            logger.warning(f"Feature interaction strength calculation failed: {e}")
            return 0.0
    
    def _calculate_feature_nonlinearity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate feature nonlinearity."""
        try:
            n_features = X.shape[1]
            if n_features < 2:
                return 0.0
            
            # Calculate nonlinearity using polynomial features
            from sklearn.preprocessing import PolynomialFeatures
            
            # Create polynomial features (degree 2)
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X)
            
            # Calculate correlation between original and polynomial features
            if X_poly.shape[1] > X.shape[1]:
                # Calculate correlation between original features and their interactions
                original_features = X_poly[:, :n_features]
                interaction_features = X_poly[:, n_features:]
                
                if interaction_features.shape[1] > 0:
                    # Calculate average correlation between original and interaction features
                    correlations = []
                    for i in range(min(5, interaction_features.shape[1])):  # Sample interactions
                        for j in range(n_features):
                            corr = np.corrcoef(original_features[:, j], interaction_features[:, i])[0, 1]
                            correlations.append(abs(corr))
                    
                    nonlinearity = np.mean(correlations) if correlations else 0.0
                else:
                    nonlinearity = 0.0
            else:
                nonlinearity = 0.0
            
            return float(nonlinearity)
            
        except Exception as e:
            logger.warning(f"Feature nonlinearity calculation failed: {e}")
            return 0.0
    
    def _estimate_model_capacity(self, model: Any, model_type: str) -> float:
        """Estimate model capacity."""
        try:
            if model_type.lower() in ['linear', 'logistic']:
                return 1.0  # Low capacity
            elif model_type.lower() in ['tree', 'decision']:
                return 3.0  # Medium capacity
            elif model_type.lower() in ['forest', 'random']:
                return 5.0  # High capacity
            elif model_type.lower() in ['neural', 'mlp']:
                return 8.0  # Very high capacity
            elif model_type.lower() in ['svm']:
                return 4.0  # High capacity
            else:
                return 2.0  # Default medium capacity
                
        except Exception as e:
            logger.warning(f"Model capacity estimation failed: {e}")
            return 2.0
    
    def _calculate_pairwise_interactions(self, X: np.ndarray, y: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise feature interactions."""
        try:
            n_features = X.shape[1]
            interactions = {}
            
            # Limit to reasonable number of features for computational efficiency
            max_features = min(20, n_features)
            
            for i in range(max_features):
                for j in range(i + 1, max_features):
                    # Calculate interaction using mutual information
                    try:
                        # Create interaction feature
                        interaction_feature = X[:, i] * X[:, j]
                        
                        # Calculate mutual information between interaction and target
                        mi = mutual_info_classif(interaction_feature.reshape(-1, 1), y, discrete_features=False)[0]
                        interactions[(i, j)] = float(mi)
                        
                    except Exception as e:
                        logger.warning(f"Pairwise interaction calculation failed for features {i}, {j}: {e}")
                        interactions[(i, j)] = 0.0
            
            return interactions
            
        except Exception as e:
            logger.error(f"Pairwise interactions calculation failed: {e}")
            return {}
    
    def _calculate_nonlinear_interactions(self, X: np.ndarray, y: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Calculate nonlinear feature interactions."""
        try:
            n_features = X.shape[1]
            interactions = {}
            
            # Limit to reasonable number of features
            max_features = min(15, n_features)
            
            for i in range(max_features):
                for j in range(i + 1, max_features):
                    try:
                        # Calculate nonlinear interaction (e.g., X1^2 * X2)
                        nonlinear_interaction = (X[:, i] ** 2) * X[:, j]
                        
                        # Calculate mutual information
                        mi = mutual_info_classif(nonlinear_interaction.reshape(-1, 1), y, discrete_features=False)[0]
                        interactions[(i, j)] = float(mi)
                        
                    except Exception as e:
                        logger.warning(f"Nonlinear interaction calculation failed for features {i}, {j}: {e}")
                        interactions[(i, j)] = 0.0
            
            return interactions
            
        except Exception as e:
            logger.error(f"Nonlinear interactions calculation failed: {e}")
            return {}
    
    def _calculate_polynomial_interactions(self, X: np.ndarray, y: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Calculate polynomial feature interactions."""
        try:
            n_features = X.shape[1]
            interactions = {}
            
            # Limit to reasonable number of features
            max_features = min(10, n_features)
            
            for i in range(max_features):
                for j in range(i + 1, max_features):
                    try:
                        # Calculate polynomial interaction (e.g., X1^2 + X2^2)
                        poly_interaction = (X[:, i] ** 2) + (X[:, j] ** 2)
                        
                        # Calculate mutual information
                        mi = mutual_info_classif(poly_interaction.reshape(-1, 1), y, discrete_features=False)[0]
                        interactions[(i, j)] = float(mi)
                        
                    except Exception as e:
                        logger.warning(f"Polynomial interaction calculation failed for features {i}, {j}: {e}")
                        interactions[(i, j)] = 0.0
            
            return interactions
            
        except Exception as e:
            logger.error(f"Polynomial interactions calculation failed: {e}")
            return {}
    
    def _calculate_interaction_overfitting_score(self, 
                                               pairwise_interactions: Dict[Tuple[int, int], float],
                                               nonlinear_interactions: Dict[Tuple[int, int], float],
                                               X_train: np.ndarray,
                                               X_val: np.ndarray,
                                               y_train: np.ndarray,
                                               y_val: np.ndarray) -> float:
        """Calculate overfitting score based on feature interactions."""
        try:
            # Calculate interaction strength on training vs validation
            train_interaction_strength = np.mean(list(pairwise_interactions.values())) if pairwise_interactions else 0.0
            val_interaction_strength = np.mean(list(nonlinear_interactions.values())) if nonlinear_interactions else 0.0
            
            # Calculate overfitting score
            if train_interaction_strength > 0:
                overfitting_score = abs(train_interaction_strength - val_interaction_strength) / train_interaction_strength
            else:
                overfitting_score = 0.0
            
            return float(min(1.0, max(0.0, overfitting_score)))
            
        except Exception as e:
            logger.warning(f"Interaction overfitting score calculation failed: {e}")
            return 0.0
    
    def _identify_suspicious_interactions(self, 
                                        pairwise_interactions: Dict[Tuple[int, int], float],
                                        nonlinear_interactions: Dict[Tuple[int, int], float],
                                        overfitting_score: float) -> List[Dict[str, Any]]:
        """Identify suspicious feature interactions that may indicate overfitting."""
        suspicious = []
        
        try:
            # High pairwise interactions
            for (i, j), strength in pairwise_interactions.items():
                if strength > 0.8:  # Very high interaction
                    suspicious.append({
                        'type': 'high_pairwise_interaction',
                        'features': (i, j),
                        'strength': strength,
                        'description': f'Very high pairwise interaction between features {i} and {j}',
                        'severity': 'high' if strength > 0.9 else 'medium'
                    })
            
            # High nonlinear interactions
            for (i, j), strength in nonlinear_interactions.items():
                if strength > 0.7:  # High nonlinear interaction
                    suspicious.append({
                        'type': 'high_nonlinear_interaction',
                        'features': (i, j),
                        'strength': strength,
                        'description': f'High nonlinear interaction between features {i} and {j}',
                        'severity': 'high' if strength > 0.8 else 'medium'
                    })
            
            # Overall overfitting score
            if overfitting_score > 0.5:
                suspicious.append({
                    'type': 'high_interaction_overfitting',
                    'strength': overfitting_score,
                    'description': f'High interaction-based overfitting score: {overfitting_score:.3f}',
                    'severity': 'high' if overfitting_score > 0.7 else 'medium'
                })
            
        except Exception as e:
            logger.warning(f"Suspicious interaction identification failed: {e}")
        
        return suspicious
    
    def _generate_interaction_recommendations(self, 
                                            high_interaction_pairs: List[Tuple[int, int]],
                                            suspicious_interactions: List[Dict[str, Any]],
                                            overfitting_score: float) -> List[str]:
        """Generate recommendations based on feature interaction analysis."""
        recommendations = []
        
        try:
            if high_interaction_pairs:
                recommendations.append(f"Consider feature selection to reduce high interactions between {len(high_interaction_pairs)} feature pairs")
            
            if any(interaction['severity'] == 'high' for interaction in suspicious_interactions):
                recommendations.append("High-severity feature interactions detected - consider regularization or feature engineering")
            
            if overfitting_score > 0.5:
                recommendations.append("High interaction-based overfitting detected - consider ensemble methods or cross-validation")
            
            if len(suspicious_interactions) > 5:
                recommendations.append("Many suspicious interactions detected - consider dimensionality reduction")
            
            if not recommendations:
                recommendations.append("Feature interactions appear normal - continue monitoring")
                
        except Exception as e:
            logger.warning(f"Interaction recommendation generation failed: {e}")
            recommendations.append("Unable to generate interaction recommendations")
        
        return recommendations
    
    def _enhance_report_with_analysis(self, 
                                    basic_report: OverfittingReport,
                                    learning_curve_result: LearningCurveAnalysisResult,
                                    complexity_metrics: ModelComplexityMetrics,
                                    feature_interactions: FeatureInteractionAnalysis) -> OverfittingReport:
        """Enhance the basic overfitting report with additional analysis."""
        try:
            # Add learning curve indicators
            if learning_curve_result.overfitting_risk != "unknown":
                if learning_curve_result.overfitting_risk in ["high", "severe"]:
                    basic_report.indicators.append("learning_curve_overfitting")
                    basic_report.warnings.append("Learning curve analysis indicates overfitting risk")
            
            # Add complexity indicators
            if complexity_metrics.vc_bound > 0.5:
                basic_report.indicators.append("high_vc_bound")
                basic_report.warnings.append(f"High VC generalization bound: {complexity_metrics.vc_bound:.3f}")
            
            if complexity_metrics.rademacher_bound > 0.5:
                basic_report.indicators.append("high_rademacher_bound")
                basic_report.warnings.append(f"High Rademacher generalization bound: {complexity_metrics.rademacher_bound:.3f}")
            
            # Add feature interaction indicators
            if feature_interactions.interaction_overfitting_score > 0.5:
                basic_report.indicators.append("feature_interaction_overfitting")
                basic_report.warnings.append(f"High feature interaction overfitting score: {feature_interactions.interaction_overfitting_score:.3f}")
            
            # Add complexity-based recommendations
            if complexity_metrics.capacity_utilization > 0.8:
                basic_report.recommendations.append("High model capacity utilization - consider regularization")
            
            if complexity_metrics.feature_interaction_strength > 0.7:
                basic_report.recommendations.append("High feature interaction strength - consider feature selection")
            
            # Add learning curve recommendations
            if learning_curve_result.convergence_stability == "poor":
                basic_report.recommendations.append("Poor convergence stability - consider learning rate adjustment")
            
            if learning_curve_result.training_efficiency == "low":
                basic_report.recommendations.append("Low training efficiency - consider model simplification")
            
            # Update severity based on additional analysis
            additional_indicators = len(basic_report.indicators) - len([ind for ind in basic_report.indicators if ind in ["accuracy_gap", "f1_gap"]])
            if additional_indicators > 2:
                if basic_report.severity == "moderate":
                    basic_report.severity = "high"
                elif basic_report.severity == "none":
                    basic_report.severity = "moderate"
            
            return basic_report
            
        except Exception as e:
            logger.error(f"Report enhancement failed: {e}")
            return basic_report
    
    def _track_enhanced_detection(self, 
                                report: OverfittingReport,
                                complexity_metrics: ModelComplexityMetrics,
                                feature_interactions: FeatureInteractionAnalysis):
        """Track enhanced detection results."""
        try:
            self.complexity_history.append(complexity_metrics)
            self.interaction_history.append(feature_interactions)
            
            # Log enhanced detection results
            logger.info(f"Enhanced overfitting detection completed for {report.model_name}")
            logger.info(f"  VC Dimension: {complexity_metrics.vc_dimension:.2f}")
            logger.info(f"  Rademacher Complexity: {complexity_metrics.rademacher_complexity:.4f}")
            logger.info(f"  Feature Interaction Score: {feature_interactions.interaction_overfitting_score:.3f}")
            
        except Exception as e:
            logger.warning(f"Enhanced detection tracking failed: {e}")
    
    def get_complexity_summary(self) -> Dict[str, Any]:
        """Get summary of model complexity analysis."""
        if not self.complexity_history:
            return {'message': 'No complexity analysis available'}
        
        try:
            # Calculate summary statistics
            vc_dimensions = [m.vc_dimension for m in self.complexity_history]
            rademacher_complexities = [m.rademacher_complexity for m in self.complexity_history]
            capacity_utilizations = [m.capacity_utilization for m in self.complexity_history]
            
            return {
                'total_analyses': len(self.complexity_history),
                'vc_dimension': {
                    'mean': float(np.mean(vc_dimensions)),
                    'std': float(np.std(vc_dimensions)),
                    'min': float(np.min(vc_dimensions)),
                    'max': float(np.max(vc_dimensions))
                },
                'rademacher_complexity': {
                    'mean': float(np.mean(rademacher_complexities)),
                    'std': float(np.std(rademacher_complexities)),
                    'min': float(np.min(rademacher_complexities)),
                    'max': float(np.max(rademacher_complexities))
                },
                'capacity_utilization': {
                    'mean': float(np.mean(capacity_utilizations)),
                    'std': float(np.std(capacity_utilizations)),
                    'min': float(np.min(capacity_utilizations)),
                    'max': float(np.max(capacity_utilizations))
                }
            }
            
        except Exception as e:
            logger.error(f"Complexity summary generation failed: {e}")
            return {'error': str(e)}
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of feature interaction analysis."""
        if not self.interaction_history:
            return {'message': 'No interaction analysis available'}
        
        try:
            # Calculate summary statistics
            overfitting_scores = [i.interaction_overfitting_score for i in self.interaction_history]
            suspicious_counts = [len(i.suspicious_interactions) for i in self.interaction_history]
            
            return {
                'total_analyses': len(self.interaction_history),
                'interaction_overfitting_score': {
                    'mean': float(np.mean(overfitting_scores)),
                    'std': float(np.std(overfitting_scores)),
                    'min': float(np.min(overfitting_scores)),
                    'max': float(np.max(overfitting_scores))
                },
                'suspicious_interactions': {
                    'mean': float(np.mean(suspicious_counts)),
                    'std': float(np.std(suspicious_counts)),
                    'min': int(np.min(suspicious_counts)),
                    'max': int(np.max(suspicious_counts))
                }
            }
            
        except Exception as e:
            logger.error(f"Interaction summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def detect_overfitting_with_learning_curves(model, 
                                          X_train: np.ndarray, 
                                          X_val: np.ndarray,
                                          y_train: np.ndarray, 
                                          y_val: np.ndarray,
                                          X_test: Optional[np.ndarray] = None,
                                          y_test: Optional[np.ndarray] = None,
                                          model_name: str = "unknown",
                                          model_type: str = "unknown",
                                          fold_number: Optional[int] = None,
                                          config: Optional[OverfittingConfig] = None) -> OverfittingReport:
    """
    Convenience function to detect overfitting with learning curves and complexity analysis.
    """
    detector = EnhancedOverfittingDetectorWithLearningCurves(config)
    return detector.detect_overfitting_with_learning_curves(
        model, X_train, X_val, y_train, y_val, X_test, y_test, model_name, model_type, fold_number
    )