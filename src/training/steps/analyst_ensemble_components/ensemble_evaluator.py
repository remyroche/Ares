"""Ensemble evaluation component for analyst ensemble creation."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, KFold

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class EnsembleEvaluator:
    """Handles evaluation of ensemble models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ensemble evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("ensemble_evaluation", {})
        self.logger = system_logger.getChild("ensemble_evaluator")
        
        # Evaluation configuration
        self.cv_folds = self.config.get("cv_folds", 5)
        self.metrics = self.config.get(
            "metrics",
            ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
        )
        self.use_cross_validation = self.config.get("use_cross_validation", True)
        self.test_size = self.config.get("test_size", 0.2)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="ensemble evaluation"
    )
    async def evaluate_ensembles(
        self,
        ensembles: Dict[str, Any],
        features: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple ensemble models.
        
        Args:
            ensembles: Dictionary of ensemble models
            features: Feature data for evaluation
            
        Returns:
            Dictionary of evaluation metrics for each ensemble
        """
        self.logger.info(f"Evaluating {len(ensembles)} ensemble models...")
        
        evaluation_results = {}
        
        for ensemble_name, ensemble_model in ensembles.items():
            metrics = await self.evaluate_single_ensemble(
                ensemble_model,
                features
            )
            evaluation_results[ensemble_name] = metrics
        
        # Log summary
        self._log_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="single ensemble evaluation"
    )
    async def evaluate_single_ensemble(
        self,
        ensemble: Any,
        features: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate a single ensemble model.
        
        Args:
            ensemble: Ensemble model to evaluate
            features: Feature data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if features.empty:
            # Return default metrics if no data
            return {metric: 0.0 for metric in self.metrics}
        
        # Generate synthetic labels for demonstration
        # In practice, these would come from the pipeline state
        y = np.random.randint(0, 2, size=len(features))
        
        # Split data
        test_size = int(len(features) * self.test_size)
        X_train = features.iloc[:-test_size]
        y_train = y[:-test_size]
        X_test = features.iloc[-test_size:]
        y_test = y[-test_size:]
        
        metrics = {}
        
        try:
            # Fit the ensemble (if not already fitted)
            if hasattr(ensemble, 'fit'):
                ensemble.fit(X_train, y_train)
            
            # Make predictions
            y_pred = ensemble.predict(X_test)
            
            # Get probability predictions if available
            y_proba = None
            if hasattr(ensemble, 'predict_proba'):
                try:
                    y_proba = ensemble.predict_proba(X_test)[:, 1]
                except:
                    pass
            
            # Calculate metrics
            if "accuracy" in self.metrics:
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
            
            if "f1_score" in self.metrics:
                metrics["f1_score"] = f1_score(y_test, y_pred, average='weighted')
            
            if "precision" in self.metrics:
                metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
            
            if "recall" in self.metrics:
                metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
            
            if "roc_auc" in self.metrics and y_proba is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                except:
                    metrics["roc_auc"] = 0.5
            
            # Cross-validation if requested
            if self.use_cross_validation:
                cv_scores = await self._cross_validate_ensemble(
                    ensemble, features, y
                )
                for metric_name, score in cv_scores.items():
                    metrics[f"{metric_name}_cv"] = score
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate ensemble: {str(e)}")
            # Return default metrics
            metrics = {metric: 0.0 for metric in self.metrics}
        
        return metrics
    
    async def _cross_validate_ensemble(
        self,
        ensemble: Any,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Perform cross-validation on an ensemble.
        
        Args:
            ensemble: Ensemble model
            X: Feature data
            y: Target labels
            
        Returns:
            Dictionary of cross-validation scores
        """
        cv_scores = {}
        
        try:
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Calculate CV scores for accuracy
            scores = cross_val_score(
                ensemble, X, y,
                cv=kfold,
                scoring='accuracy',
                n_jobs=-1
            )
            
            cv_scores["accuracy"] = scores.mean()
            cv_scores["accuracy_std"] = scores.std()
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
        
        return cv_scores
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="ensemble comparison"
    )
    async def compare_ensembles(
        self,
        ensembles: Dict[str, Any],
        features: pd.DataFrame,
        target_metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """Compare multiple ensembles and identify the best performer.
        
        Args:
            ensembles: Dictionary of ensemble models
            features: Feature data
            target_metric: Metric to use for comparison
            
        Returns:
            Comparison results including best ensemble
        """
        self.logger.info(f"Comparing ensembles using {target_metric}...")
        
        # Evaluate all ensembles
        evaluation_results = await self.evaluate_ensembles(ensembles, features)
        
        # Find best performer
        best_ensemble = None
        best_score = -np.inf
        
        for ensemble_name, metrics in evaluation_results.items():
            score = metrics.get(target_metric, 0.0)
            if score > best_score:
                best_score = score
                best_ensemble = ensemble_name
        
        # Calculate improvement statistics
        scores = [m.get(target_metric, 0.0) for m in evaluation_results.values()]
        
        comparison_results = {
            "best_ensemble": best_ensemble,
            "best_score": best_score,
            "average_score": np.mean(scores),
            "score_std": np.std(scores),
            "all_scores": {
                name: metrics.get(target_metric, 0.0)
                for name, metrics in evaluation_results.items()
            }
        }
        
        self.logger.info(
            f"Best ensemble: {best_ensemble} with {target_metric}={best_score:.4f}"
        )
        
        return comparison_results
    
    def _log_evaluation_summary(
        self,
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Log a summary of evaluation results.
        
        Args:
            evaluation_results: Dictionary of evaluation metrics
        """
        if not evaluation_results:
            return
        
        # Calculate average metrics across all ensembles
        all_metrics = {}
        
        for metrics in evaluation_results.values():
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Log averages
        self.logger.info("Ensemble Evaluation Summary:")
        for metric_name, values in all_metrics.items():
            avg_value = np.mean(values)
            std_value = np.std(values)
            self.logger.info(
                f"  {metric_name}: {avg_value:.4f} ± {std_value:.4f}"
            )
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="ensemble diagnostics"
    )
    async def generate_ensemble_diagnostics(
        self,
        ensemble: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Generate detailed diagnostics for an ensemble.
        
        Args:
            ensemble: Ensemble model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of diagnostic information
        """
        diagnostics = {}
        
        try:
            # Get predictions
            y_pred = ensemble.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            diagnostics["confusion_matrix"] = cm.tolist()
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            diagnostics["classification_report"] = report
            
            # Feature importance if available
            if hasattr(ensemble, 'feature_importances_'):
                importances = ensemble.feature_importances_
                feature_names = X_test.columns.tolist()
                diagnostics["feature_importances"] = dict(zip(
                    feature_names,
                    importances.tolist()
                ))
            
            # Prediction confidence distribution
            if hasattr(ensemble, 'predict_proba'):
                probas = ensemble.predict_proba(X_test)
                diagnostics["confidence_stats"] = {
                    "mean": float(probas.max(axis=1).mean()),
                    "std": float(probas.max(axis=1).std()),
                    "min": float(probas.max(axis=1).min()),
                    "max": float(probas.max(axis=1).max())
                }
            
        except Exception as e:
            self.logger.error(f"Failed to generate diagnostics: {str(e)}")
        
        return diagnostics