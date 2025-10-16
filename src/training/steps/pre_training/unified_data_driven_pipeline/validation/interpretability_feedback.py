"""
Interpretability Feedback Loop

This module implements SHAP-based interpretability feedback for interaction generation,
providing iterative pruning based on interpretability metrics.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Import SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class InterpretabilityFeedbackConfig:
    """Configuration for interpretability feedback."""
    
    # SHAP parameters
    min_shap_importance: float = 0.01        # Minimum SHAP importance
    max_shap_consistency: float = 0.8        # Maximum SHAP consistency
    min_feature_consistency: float = 0.6     # Minimum feature consistency
    
    # Iterative pruning
    max_iterations: int = 10                  # Maximum pruning iterations
    pruning_threshold: float = 0.1           # Threshold for pruning
    min_features_remaining: int = 5           # Minimum features to keep
    
    # Model parameters
    model_type: str = "tree"                  # "tree", "linear", "neural"
    explainer_type: str = "tree"              # "tree", "linear", "kernel"
    
    # Validation
    enable_cross_validation: bool = True      # Enable CV for SHAP
    cv_folds: int = 3                        # Number of CV folds
    
    # Logging
    verbose: bool = True


@dataclass
class InterpretabilityMetrics:
    """Interpretability metrics for features."""
    
    feature_name: str
    shap_importance: float
    shap_consistency: float
    feature_consistency: float
    interaction_strength: float
    redundancy_score: float
    overall_score: float
    
    # Feedback signals
    should_keep: bool = True
    should_prune: bool = False
    feedback_reason: str = ""


@dataclass
class InterpretabilityFeedbackResult:
    """Result of interpretability feedback analysis."""
    
    # Metrics for all features
    feature_metrics: List[InterpretabilityMetrics] = field(default_factory=list)
    
    # Pruning results
    features_to_keep: List[str] = field(default_factory=list)
    features_to_prune: List[str] = field(default_factory=list)
    
    # Iteration results
    iteration_results: List[Dict[str, Any]] = field(default_factory=list)
    converged: bool = False
    final_iteration: int = 0
    
    # Performance metrics
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class InterpretabilityFeedbackLoop:
    """
    Interpretability feedback loop for interaction generation.
    
    Provides iterative pruning based on:
    1. SHAP importance and consistency
    2. Feature consistency across time
    3. Interaction strength and redundancy
    4. Cross-validation stability
    """
    
    def __init__(self, config: Optional[InterpretabilityFeedbackConfig] = None):
        """Initialize the interpretability feedback loop."""
        self.config = config or InterpretabilityFeedbackConfig()
        self.logger = logging.getLogger(__name__)
        
        if not SHAP_AVAILABLE:
            tprint_warning("⚠️ SHAP not available. Install with: pip install shap")
        
        if self.config.verbose:
            tprint("🔍 Initializing InterpretabilityFeedbackLoop")
    
    def analyze_interpretability(self, 
                               features: pd.DataFrame,
                               targets: pd.Series,
                               model: Any = None) -> InterpretabilityFeedbackResult:
        """
        Analyze interpretability of features using SHAP.
        
        Args:
            features: Input features
            targets: Target labels
            model: Trained model (optional)
            
        Returns:
            InterpretabilityFeedbackResult
        """
        if self.config.verbose:
            tprint("🔍 Analyzing feature interpretability")
        
        if not SHAP_AVAILABLE:
            tprint_error("❌ SHAP not available for interpretability analysis")
            return InterpretabilityFeedbackResult()
        
        result = InterpretabilityFeedbackResult()
        
        # Train model if not provided
        if model is None:
            model = self._train_model(features, targets)
        
        # Calculate SHAP values
        shap_values = self._calculate_shap_values(model, features)
        
        # Analyze each feature
        for feature_name in features.columns:
            metrics = self._analyze_feature_interpretability(
                feature_name, features, targets, shap_values
            )
            result.feature_metrics.append(metrics)
        
        # Determine pruning decisions
        self._determine_pruning_decisions(result)
        
        # Calculate overall scores
        result.initial_score = self._calculate_overall_score(result.feature_metrics)
        
        if self.config.verbose:
            tprint(f"📊 Initial interpretability score: {result.initial_score:.4f}")
            tprint(f"📊 Features to keep: {len(result.features_to_keep)}")
            tprint(f"📊 Features to prune: {len(result.features_to_prune)}")
        
        return result
    
    def iterative_pruning(self, 
                         features: pd.DataFrame,
                         targets: pd.Series,
                         initial_result: InterpretabilityFeedbackResult) -> InterpretabilityFeedbackResult:
        """
        Perform iterative pruning based on interpretability feedback.
        
        Args:
            features: Input features
            targets: Target labels
            initial_result: Initial interpretability analysis
            
        Returns:
            InterpretabilityFeedbackResult after iterative pruning
        """
        if self.config.verbose:
            tprint("🔄 Starting iterative interpretability pruning")
        
        result = initial_result
        current_features = features.copy()
        
        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                tprint(f"🔄 Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Analyze current features
            iteration_result = self.analyze_interpretability(current_features, targets)
            result.iteration_results.append({
                'iteration': iteration + 1,
                'features_count': len(current_features.columns),
                'score': iteration_result.initial_score,
                'features_to_prune': iteration_result.features_to_prune
            })
            
            # Check convergence
            if self._check_convergence(result, iteration):
                result.converged = True
                result.final_iteration = iteration + 1
                break
            
            # Prune features
            features_to_remove = iteration_result.features_to_prune
            if not features_to_remove:
                result.converged = True
                result.final_iteration = iteration + 1
                break
            
            # Remove features
            current_features = current_features.drop(columns=features_to_remove)
            
            # Check minimum features
            if len(current_features.columns) <= self.config.min_features_remaining:
                result.converged = True
                result.final_iteration = iteration + 1
                break
        
        # Final analysis
        final_result = self.analyze_interpretability(current_features, targets)
        result.features_to_keep = list(current_features.columns)
        result.final_score = final_result.initial_score
        result.improvement = result.final_score - result.initial_score
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        if self.config.verbose:
            tprint_success(f"✅ Iterative pruning completed")
            tprint(f"📊 Final score: {result.final_score:.4f}")
            tprint(f"📊 Improvement: {result.improvement:.4f}")
            tprint(f"📊 Converged: {result.converged}")
        
        return result
    
    def _train_model(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Train a model for SHAP analysis."""
        try:
            if self.config.model_type == "tree":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.config.model_type == "linear":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(features, targets)
            return model
        except Exception as e:
            self.logger.warning(f"Model training failed: {e}")
            return None
    
    def _calculate_shap_values(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for features."""
        try:
            if self.config.explainer_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif self.config.explainer_type == "linear":
                explainer = shap.LinearExplainer(model, features)
            else:
                explainer = shap.KernelExplainer(model.predict, features.sample(100))
            
            shap_values = explainer.shap_values(features)
            return shap_values
        except Exception as e:
            self.logger.warning(f"SHAP calculation failed: {e}")
            return np.zeros((len(features), len(features.columns)))
    
    def _analyze_feature_interpretability(self, 
                                        feature_name: str,
                                        features: pd.DataFrame,
                                        targets: pd.Series,
                                        shap_values: np.ndarray) -> InterpretabilityMetrics:
        """Analyze interpretability of a single feature."""
        try:
            feature_idx = features.columns.get_loc(feature_name)
            feature_shap = shap_values[:, feature_idx]
            
            # SHAP importance (mean absolute SHAP value)
            shap_importance = np.mean(np.abs(feature_shap))
            
            # SHAP consistency (stability across samples)
            shap_consistency = 1.0 - np.std(feature_shap) / (np.mean(np.abs(feature_shap)) + 1e-10)
            
            # Feature consistency (stability across time)
            feature_consistency = self._calculate_feature_consistency(
                features[feature_name], targets
            )
            
            # Interaction strength
            interaction_strength = self._calculate_interaction_strength(
                feature_name, features, targets
            )
            
            # Redundancy score
            redundancy_score = self._calculate_redundancy_score(
                feature_name, features
            )
            
            # Overall score
            overall_score = (
                shap_importance * 0.3 +
                shap_consistency * 0.2 +
                feature_consistency * 0.2 +
                interaction_strength * 0.2 +
                (1.0 - redundancy_score) * 0.1
            )
            
            return InterpretabilityMetrics(
                feature_name=feature_name,
                shap_importance=shap_importance,
                shap_consistency=shap_consistency,
                feature_consistency=feature_consistency,
                interaction_strength=interaction_strength,
                redundancy_score=redundancy_score,
                overall_score=overall_score
            )
        except Exception as e:
            self.logger.warning(f"Feature analysis failed for {feature_name}: {e}")
            return InterpretabilityMetrics(
                feature_name=feature_name,
                shap_importance=0.0,
                shap_consistency=0.0,
                feature_consistency=0.0,
                interaction_strength=0.0,
                redundancy_score=1.0,
                overall_score=0.0
            )
    
    def _calculate_feature_consistency(self, feature: pd.Series, targets: pd.Series) -> float:
        """Calculate feature consistency across time."""
        try:
            # Rolling correlation with targets
            window = min(20, len(feature) // 4)
            if window < 5:
                return 0.0
            
            rolling_corr = feature.rolling(window).corr(targets.rolling(window))
            consistency = 1.0 - rolling_corr.std() if not rolling_corr.std() is np.nan else 0.0
            return max(0.0, min(1.0, consistency))
        except:
            return 0.0
    
    def _calculate_interaction_strength(self, 
                                      feature_name: str,
                                      features: pd.DataFrame,
                                      targets: pd.Series) -> float:
        """Calculate interaction strength with other features."""
        try:
            feature = features[feature_name]
            other_features = features.drop(columns=[feature_name])
            
            # Calculate interaction strength as correlation with other features
            interactions = []
            for other_col in other_features.columns:
                # Simple interaction: product of features
                interaction = feature * other_features[other_col]
                corr = interaction.corr(targets)
                if not np.isnan(corr):
                    interactions.append(abs(corr))
            
            return np.mean(interactions) if interactions else 0.0
        except:
            return 0.0
    
    def _calculate_redundancy_score(self, 
                                  feature_name: str,
                                  features: pd.DataFrame) -> float:
        """Calculate redundancy score with other features."""
        try:
            feature = features[feature_name]
            other_features = features.drop(columns=[feature_name])
            
            # Calculate maximum correlation with other features
            max_corr = 0.0
            for other_col in other_features.columns:
                corr = abs(feature.corr(other_features[other_col]))
                if not np.isnan(corr):
                    max_corr = max(max_corr, corr)
            
            return max_corr
        except:
            return 0.0
    
    def _determine_pruning_decisions(self, result: InterpretabilityFeedbackResult) -> None:
        """Determine which features to prune based on interpretability metrics."""
        for metrics in result.feature_metrics:
            # Pruning criteria
            should_prune = (
                metrics.shap_importance < self.config.min_shap_importance or
                metrics.shap_consistency < self.config.max_shap_consistency or
                metrics.feature_consistency < self.config.min_feature_consistency or
                metrics.overall_score < self.config.pruning_threshold
            )
            
            metrics.should_prune = should_prune
            metrics.should_keep = not should_prune
            
            if should_prune:
                result.features_to_prune.append(metrics.feature_name)
                metrics.feedback_reason = "Low interpretability score"
            else:
                result.features_to_keep.append(metrics.feature_name)
                metrics.feedback_reason = "Good interpretability score"
    
    def _check_convergence(self, result: InterpretabilityFeedbackResult, iteration: int) -> bool:
        """Check if iterative pruning has converged."""
        if iteration < 2:
            return False
        
        # Check if no features were pruned in last iteration
        last_iteration = result.iteration_results[-1]
        if not last_iteration['features_to_prune']:
            return True
        
        # Check if improvement is minimal
        if len(result.iteration_results) >= 3:
            recent_scores = [r['score'] for r in result.iteration_results[-3:]]
            if len(set(recent_scores)) == 1:  # No change in last 3 iterations
                return True
        
        return False
    
    def _calculate_overall_score(self, feature_metrics: List[InterpretabilityMetrics]) -> float:
        """Calculate overall interpretability score."""
        if not feature_metrics:
            return 0.0
        
        scores = [m.overall_score for m in feature_metrics]
        return np.mean(scores)
    
    def _generate_recommendations(self, result: InterpretabilityFeedbackResult) -> List[str]:
        """Generate recommendations based on interpretability analysis."""
        recommendations = []
        
        if result.improvement < 0:
            recommendations.append("Consider relaxing pruning criteria")
        
        if not result.converged:
            recommendations.append("Increase maximum iterations for convergence")
        
        if len(result.features_to_keep) < self.config.min_features_remaining:
            recommendations.append("Increase minimum features threshold")
        
        # Feature-specific recommendations
        low_importance_features = [
            m.feature_name for m in result.feature_metrics 
            if m.shap_importance < self.config.min_shap_importance
        ]
        if low_importance_features:
            recommendations.append(f"Consider removing low-importance features: {low_importance_features[:5]}")
        
        return recommendations
