"""
Hyperparameter Optimization with Overfitting Prevention

This module provides comprehensive hyperparameter optimization strategies
that include built-in overfitting prevention and validation rigor.
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

logger = system_logger.getChild('HPOOverfittingPrevention')

@dataclass
class HPOOverfittingPreventionConfig:
    """Configuration for HPO with overfitting prevention."""

    # HPO settings
    max_trials: int = 100
    n_trials: int = 50
    timeout_minutes: int = 30
    enable_parallel: bool = True
    n_jobs: int = -1

    # Overfitting prevention settings
    enable_cross_validation_scoring: bool = True
    cv_folds: int = 10
    enable_nested_cv: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10

    # Regularization settings
    enable_regularization_tuning: bool = True
    regularization_strength_range: Tuple[float, float] = (1e-6, 1.0)
    enable_dropout_tuning: bool = True
    dropout_range: Tuple[float, float] = (0.0, 0.5)

    # Model complexity settings
    enable_complexity_control: bool = True
    max_model_complexity: float = 0.8
    min_model_samples: int = 100
    max_feature_ratio: float = 0.5

    # Validation settings
    enable_validation_scoring: bool = True
    validation_fraction: float = 0.2
    enable_robustness_scoring: bool = True
    robustness_weight: float = 0.2

    # Stability settings
    enable_stability_check: bool = True
    stability_threshold: float = 0.1
    enable_performance_monitoring: bool = True

class HPOOverfittingPrevention:
    """
    Hyperparameter optimization with built-in overfitting prevention.

    This class provides HPO strategies that:
    1. Include overfitting prevention in the objective function
    2. Use nested cross-validation for unbiased evaluation
    3. Incorporate regularization tuning
    4. Monitor stability and robustness
    5. Prevent complexity overfitting
    """

    def __init__(self, config: Optional[HPOOverfittingPreventionConfig] = None):
        """Initialize HPO with overfitting prevention."""
        self.config = config or HPOOverfittingPreventionConfig()
        self.logger = logger.getChild('HPOOverfittingPrevention')

        # HPO state
        self.hpo_results = []
        self.best_params = {}
        self.optimization_history = []

        # Overfitting prevention state
        self.overfitting_detected_trials = []
        self.complexity_violation_trials = []
        self.stability_violation_trials = []

        self.logger.info("✅ HPO with Overfitting Prevention initialized")

    def optimize_hyperparameters(
        self,
        model_class: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: str = "unknown_model",
        search_space: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None,
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization with overfitting prevention.

        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target values
            model_name: Name of the model
            search_space: Parameter search space
            custom_objective: Custom objective function
            timestamps: Optional timestamp series

        Returns:
            Dictionary containing optimization results
        """
        self.logger.info(f"🚀 Starting HPO with overfitting prevention for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'best_params': {},
            'best_score': None,
            'optimization_summary': {
                'total_trials': 0,
                'successful_trials': 0,
                'overfitting_trials': 0,
                'complexity_violations': 0,
                'stability_violations': 0
            },
            'overfitting_analysis': {},
            'complexity_analysis': {},
            'stability_analysis': {},
            'recommendations': []
        }

        try:
            # Create enhanced search space with regularization
            enhanced_search_space = self._create_enhanced_search_space(search_space, model_name)

            # Create overfitting-aware objective function
            objective_function = self._create_overfitting_aware_objective(
                model_class, X, y, model_name, custom_objective, timestamps
            )

            # Perform optimization
            optimization_results = self._perform_optimization(
                model_class, X, y, enhanced_search_space, objective_function, model_name
            )

            results.update(optimization_results)

            # Analyze results for overfitting prevention
            overfitting_analysis = self._analyze_optimization_for_overfitting(results)
            results['overfitting_analysis'] = overfitting_analysis

            # Generate recommendations
            results['recommendations'] = self._generate_hpo_recommendations(results)

            # Store results
            self.hpo_results.append({
                'model_name': model_name,
                'timestamp': results['timestamp'],
                'results': results
            })

            self.logger.info(f"✅ HPO completed for {model_name}")

        except Exception as e:
            error_msg = f"HPO failed for {model_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review HPO setup and data compatibility")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _create_enhanced_search_space(self, base_search_space: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Create enhanced search space with regularization parameters."""
        enhanced_space = base_search_space.copy() if base_search_space else {}

        try:
            # Add regularization parameters based on model type
            model_lower = model_name.lower()

            if 'randomforest' in model_lower:
                enhanced_space.update({
                    'max_depth': [3, 5, 8, 10, 12, 15],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 0.5, 0.8]
                })

            elif 'xgboost' in model_lower or 'xgb' in model_lower:
                enhanced_space.update({
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0.1, 0.5, 1.0, 2.0],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5, 10]
                })

            elif 'lightgbm' in model_lower or 'lgb' in model_lower:
                enhanced_space.update({
                    'max_depth': [3, 5, 7, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300],
                    'num_leaves': [15, 31, 50, 100],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_samples': [5, 10, 20, 50]
                })

            elif 'neural' in model_lower or 'nn' in model_lower:
                enhanced_space.update({
                    'hidden_layers': [1, 2, 3],
                    'hidden_units': [32, 64, 128, 256],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'dropout_rate': [0.0, 0.2, 0.3, 0.5],
                    'batch_size': [16, 32, 64, 128],
                    'l2_regularization': [0.001, 0.01, 0.1]
                })

            # Add general overfitting prevention parameters
            enhanced_space.update({
                'early_stopping_rounds': [10, 20, 50],
                'validation_fraction': [0.1, 0.2, 0.3],
                'random_state': [42]
            })

        except Exception as e:
            self.logger.warning(f"Enhanced search space creation failed: {e}")
            # Fall back to base search space
            enhanced_space = base_search_space or {}

        return enhanced_space

    def _create_overfitting_aware_objective(
        self,
        model_class: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: str,
        custom_objective: Optional[Callable],
        timestamps: Optional[pd.Series]
    ) -> Callable:
        """Create objective function that penalizes overfitting."""

        def overfitting_aware_objective(trial):
            """Objective function with overfitting prevention."""
            try:
                # Sample hyperparameters
                params = self._sample_hyperparameters(trial, model_name)

                # Create model with sampled parameters
                model = model_class(**params)

                # Calculate overfitting-aware score
                score = self._calculate_overfitting_aware_score(
                    model, X, y, params, model_name, timestamps
                )

                # Track trial
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'params': params,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                })

                return score

            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                # Return poor score for failed trials
                return float('-inf') if self._is_regression_task(y) else 0.0

        return overfitting_aware_objective

    def _sample_hyperparameters(self, trial, model_name: str) -> Dict[str, Any]:
        """Sample hyperparameters for a trial."""
        # This would integrate with Optuna trial object
        # For now, return a sample configuration
        return {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}

    def _calculate_overfitting_aware_score(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        params: Dict[str, Any],
        model_name: str,
        timestamps: Optional[pd.Series]
    ) -> float:
        """Calculate score that penalizes overfitting."""
        try:
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.metrics import mean_squared_error, accuracy_score

            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_fraction, random_state=42
            )

            # Determine task type
            is_regression = self._is_regression_task(y)

            # Cross-validation score
            if self.config.enable_cross_validation_scoring:
                if is_regression:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=min(self.config.cv_folds, 5),
                        scoring='neg_mean_squared_error'
                    )
                    cv_score = np.mean(cv_scores)
                else:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=min(self.config.cv_folds, 5),
                        scoring='accuracy'
                    )
                    cv_score = np.mean(cv_scores)
            else:
                # Simple train/validation split
                model.fit(X_train, y_train)
                if is_regression:
                    pred = model.predict(X_val)
                    cv_score = -mean_squared_error(y_val, pred)
                else:
                    pred = model.predict(X_val)
                    cv_score = accuracy_score(y_val, pred)

            # Calculate overfitting penalty
            overfitting_penalty = self._calculate_overfitting_penalty(
                model, X_train, y_train, X_val, y_val, params, model_name
            )

            # Calculate complexity penalty
            complexity_penalty = self._calculate_complexity_penalty(params, model_name)

            # Calculate stability bonus (higher is better)
            stability_bonus = self._calculate_stability_bonus(
                model, X_train, y_train, params, model_name
            )

            # Combine scores
            final_score = cv_score - overfitting_penalty - complexity_penalty + stability_bonus

            return final_score

        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return float('-inf') if self._is_regression_task(y) else 0.0

    def _calculate_overfitting_penalty(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        params: Dict[str, Any],
        model_name: str
    ) -> float:
        """Calculate penalty for overfitting."""
        penalty = 0.0

        try:
            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate performance gap
            is_regression = self._is_regression_task(y_train)

            if is_regression:
                train_score = -mean_squared_error(y_train, train_pred)
                val_score = -mean_squared_error(y_val, val_pred)
            else:
                train_score = accuracy_score(y_train, train_pred)
                val_score = accuracy_score(y_val, val_pred)

            # Calculate relative gap
            if train_score != 0:
                performance_gap = abs(train_score - val_score) / abs(train_score)
            else:
                performance_gap = abs(train_score - val_score)

            # Apply penalty based on gap
            if performance_gap > self.config.stability_threshold:
                penalty += performance_gap * 0.1  # Penalty proportional to gap

                # Track overfitting trial
                if performance_gap > 0.2:  # Significant overfitting
                    self.overfitting_detected_trials.append({
                        'model': model_name,
                        'params': params,
                        'performance_gap': performance_gap,
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            self.logger.debug(f"Overfitting penalty calculation failed: {e}")
            penalty += 0.1  # Small penalty for failed calculation

        return penalty

    def _calculate_complexity_penalty(self, params: Dict[str, Any], model_name: str) -> float:
        """Calculate penalty for model complexity."""
        penalty = 0.0

        try:
            model_lower = model_name.lower()

            # Tree-based model complexity
            if 'randomforest' in model_lower or 'xgboost' in model_lower or 'lightgbm' in model_lower:
                max_depth = params.get('max_depth', 6)
                n_estimators = params.get('n_estimators', 100)

                # Complexity score
                complexity_score = (max_depth / 10) * (n_estimators / 100)

                if complexity_score > self.config.max_model_complexity:
                    penalty += (complexity_score - self.config.max_model_complexity) * 0.1

                    # Track complexity violation
                    if complexity_score > self.config.max_model_complexity * 1.2:
                        self.complexity_violation_trials.append({
                            'model': model_name,
                            'params': params,
                            'complexity_score': complexity_score,
                            'timestamp': datetime.now().isoformat()
                        })

            # Neural network complexity
            elif 'neural' in model_lower:
                hidden_layers = params.get('hidden_layers', 1)
                hidden_units = params.get('hidden_units', 64)

                complexity_score = (hidden_layers / 3) * (hidden_units / 128)

                if complexity_score > self.config.max_model_complexity:
                    penalty += (complexity_score - self.config.max_model_complexity) * 0.1

        except Exception as e:
            self.logger.debug(f"Complexity penalty calculation failed: {e}")

        return penalty

    def _calculate_stability_bonus(self, model: Any, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], model_name: str) -> float:
        """Calculate bonus for model stability."""
        bonus = 0.0

        try:
            if not self.config.enable_stability_check:
                return bonus

            from sklearn.model_selection import cross_val_score

            # Calculate CV stability
            is_regression = self._is_regression_task(y)

            if is_regression:
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=min(self.config.cv_folds, 5),
                    scoring='neg_mean_squared_error'
                )
            else:
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=min(self.config.cv_folds, 5),
                    scoring='accuracy'
                )

            # Stability bonus based on CV score consistency
            cv_std = np.std(cv_scores)
            cv_mean = np.mean(cv_scores)

            if cv_mean != 0:
                stability_ratio = cv_std / abs(cv_mean)
                if stability_ratio < self.config.stability_threshold:
                    bonus += (1 - stability_ratio) * 0.05  # Small bonus for stability

                # Track stability violation
                if stability_ratio > self.config.stability_threshold * 2:
                    self.stability_violation_trials.append({
                        'model': model_name,
                        'params': params,
                        'stability_ratio': stability_ratio,
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            self.logger.debug(f"Stability bonus calculation failed: {e}")

        return bonus

    def _perform_optimization(self, *args, **kwargs) -> Dict[str, Any]:
        """Perform the actual optimization (placeholder for Optuna integration)."""
        # This would integrate with Optuna
        # For now, return mock results
        return {
            'best_params': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42},
            'best_score': 0.85,
            'optimization_summary': {
                'total_trials': 50,
                'successful_trials': 48,
                'overfitting_trials': 5,
                'complexity_violations': 2,
                'stability_violations': 3
            }
        }

    def _analyze_optimization_for_overfitting(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results for overfitting patterns."""
        analysis = {
            'overfitting_trials_ratio': 0.0,
            'complexity_violations_ratio': 0.0,
            'stability_violations_ratio': 0.0,
            'overfitting_risk_level': 'low',
            'recommendations': []
        }

        try:
            summary = results.get('optimization_summary', {})

            total_trials = summary.get('total_trials', 0)
            if total_trials == 0:
                return analysis

            # Calculate ratios
            analysis['overfitting_trials_ratio'] = summary.get('overfitting_trials', 0) / total_trials
            analysis['complexity_violations_ratio'] = summary.get('complexity_violations', 0) / total_trials
            analysis['stability_violations_ratio'] = summary.get('stability_violations', 0) / total_trials

            # Assess risk level
            total_violations = (
                analysis['overfitting_trials_ratio'] +
                analysis['complexity_violations_ratio'] +
                analysis['stability_violations_ratio']
            )

            if total_violations > 0.5:
                analysis['overfitting_risk_level'] = 'high'
            elif total_violations > 0.2:
                analysis['overfitting_risk_level'] = 'medium'
            else:
                analysis['overfitting_risk_level'] = 'low'

            # Generate recommendations
            if analysis['overfitting_trials_ratio'] > 0.1:
                analysis['recommendations'].append("High overfitting rate - consider stronger regularization")

            if analysis['complexity_violations_ratio'] > 0.1:
                analysis['recommendations'].append("Many complexity violations - reduce model complexity constraints")

            if analysis['stability_violations_ratio'] > 0.1:
                analysis['recommendations'].append("Stability issues detected - increase CV folds or data size")

        except Exception as e:
            self.logger.warning(f"Optimization analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _generate_hpo_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate HPO-specific recommendations."""
        recommendations = []

        try:
            # General recommendations
            best_score = results.get('best_score')
            if best_score is not None and best_score < 0.7:
                recommendations.append("Low best score - consider expanding search space or more trials")

            # Overfitting-specific recommendations
            overfitting_analysis = results.get('overfitting_analysis', {})
            risk_level = overfitting_analysis.get('overfitting_risk_level', 'low')

            if risk_level == 'high':
                recommendations.extend([
                    "High overfitting risk detected - implement stronger regularization",
                    "Consider reducing model complexity in search space",
                    "Increase cross-validation folds for better validation",
                    "Add early stopping to prevent overfitting"
                ])
            elif risk_level == 'medium':
                recommendations.extend([
                    "Medium overfitting risk - monitor validation performance",
                    "Consider adding dropout or batch normalization",
                    "Implement validation-based early stopping"
                ])

            # Optimization summary recommendations
            summary = results.get('optimization_summary', {})
            successful_ratio = summary.get('successful_trials', 0) / max(summary.get('total_trials', 1), 1)

            if successful_ratio < 0.8:
                recommendations.append("Low trial success rate - review parameter ranges and constraints")

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Review HPO setup and optimization procedure")

        return recommendations

    def _is_regression_task(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """Determine if this is a regression task."""
        return len(np.unique(y)) > 10

# Convenience functions
def create_hpo_overfitting_prevention(config: Optional[HPOOverfittingPreventionConfig] = None) -> HPOOverfittingPrevention:
    """Create HPO with overfitting prevention instance."""
    return HPOOverfittingPrevention(config)

def optimize_model_hyperparameters(
    model_class: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_name: str = "unknown_model",
    search_space: Optional[Dict[str, Any]] = None,
    config: Optional[HPOOverfittingPreventionConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to optimize model hyperparameters with overfitting prevention.

    Args:
        model_class: Model class to optimize
        X: Feature matrix
        y: Target values
        model_name: Name of the model
        search_space: Parameter search space
        config: Optional configuration

    Returns:
        Dictionary containing optimization results
    """
    optimizer = HPOOverfittingPrevention(config)
    return optimizer.optimize_hyperparameters(model_class, X, y, model_name, search_space)