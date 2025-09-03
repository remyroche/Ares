"""Step 18: Walk Forward Validation - Updated to use BaseStep pattern."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.core.decorators import handles_errors, log_execution_time
from .base_validation_step import BaseValidationStep


class WalkForwardValidationStep(BaseValidationStep):
    """Step 18: Walk Forward Validation for time series models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Walk Forward Validation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "18", "walk_forward_validation")
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Walk forward configuration
        self.wf_config = {
            "n_splits": self.config.get("walk_forward_splits", 5),
            "train_size": self.config.get("walk_forward_train_size", 0.7),
            "test_size": self.config.get("walk_forward_test_size", 0.1),
            "step_size": self.config.get("walk_forward_step_size", 0.05),
            "retrain_frequency": self.config.get("retrain_frequency", 1),
            "expanding_window": self.config.get("expanding_window", False)
        }
        
        # Storage for validation results
        self.fold_results: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, List[Dict[str, float]]] = {}
    
    def _validate_step_specific_inputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> List[str]:
        """Validate step-specific inputs."""
        errors = []
        
        # Check for time series data
        data_sources = ["tactician_labeled_data", "features", "market_data"]
        has_data = any(source in pipeline_state for source in data_sources)
        
        if not has_data:
            errors.append("No time series data found for walk forward validation")
        
        return errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="walk forward validation logic"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the walk forward validation logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with validation results
        """
        self.logger.info("🚀 Starting walk forward validation...")
        
        # Extract data
        data = self._extract_time_series_data(pipeline_state)
        
        if data.empty:
            self.logger.warning("No data available for walk forward validation")
            return pipeline_state
        
        # Get models to validate
        models = self._get_models_for_validation(pipeline_state)
        
        if not models:
            self.logger.warning("No models available for validation")
            return pipeline_state
        
        # Perform walk forward validation
        for fold_idx in range(self.wf_config["n_splits"]):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.wf_config['n_splits']}...")
            
            # Get fold data
            fold_data = await self._get_fold_data(data, fold_idx)
            
            if fold_data is None:
                continue
            
            train_data, val_data, test_data = fold_data
            
            # Validate each model on this fold
            fold_results = await self._validate_fold(
                models, train_data, val_data, test_data, fold_idx
            )
            
            self.fold_results.append(fold_results)
        
        # Aggregate results
        aggregated_results = self._aggregate_fold_results()
        
        # Update pipeline state
        result = pipeline_state.copy()
        result[f"{self.full_step_name}_results"] = {
            "fold_results": self.fold_results,
            "model_performance": self.model_performance,
            "aggregated_results": aggregated_results,
            "configuration": self.wf_config
        }
        
        # Create summary
        result[f"{self.full_step_name}_summary"] = self._create_validation_summary({
            "model_results": aggregated_results,
            "overall_metrics": self._calculate_overall_wf_metrics(aggregated_results)
        })
        
        return result
    
    def _extract_time_series_data(
        self,
        pipeline_state: Dict[str, Any]
    ) -> pd.DataFrame:
        """Extract time series data from pipeline state."""
        # Try different sources
        if "tactician_labeled_data" in pipeline_state:
            return pipeline_state["tactician_labeled_data"]
        
        if "features" in pipeline_state:
            features = pipeline_state["features"]
            if "labels" in pipeline_state:
                # Combine features and labels
                labels = pipeline_state["labels"]
                data = features.copy()
                data["label"] = labels
                return data
            return features
        
        if "market_data" in pipeline_state:
            return pipeline_state["market_data"]
        
        return pd.DataFrame()
    
    async def _get_fold_data(
        self,
        data: pd.DataFrame,
        fold_idx: int
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Get data for a specific fold.
        
        Args:
            data: Full dataset
            fold_idx: Fold index
            
        Returns:
            Tuple of (train, validation, test) data
        """
        n_samples = len(data)
        
        # Calculate sizes
        train_size = int(n_samples * self.wf_config["train_size"])
        test_size = int(n_samples * self.wf_config["test_size"])
        step_size = int(n_samples * self.wf_config["step_size"])
        
        # Calculate fold start position
        fold_start = fold_idx * step_size
        
        if self.wf_config["expanding_window"]:
            # Expanding window: always start from beginning
            train_start = 0
        else:
            # Rolling window: move start position
            train_start = fold_start
        
        # Calculate end positions
        train_end = train_start + train_size
        val_end = train_end + test_size
        test_end = val_end + test_size
        
        # Check if we have enough data
        if test_end > n_samples:
            return None
        
        # Extract fold data
        train_data = data.iloc[train_start:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:test_end]
        
        return train_data, val_data, test_data
    
    async def _validate_fold(
        self,
        models: Dict[str, Any],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold_idx: int
    ) -> Dict[str, Any]:
        """Validate models on a single fold.
        
        Args:
            models: Models to validate
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            fold_idx: Fold index
            
        Returns:
            Fold validation results
        """
        fold_results = {
            "fold_idx": fold_idx,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "model_results": {}
        }
        
        # Extract features and labels
        X_train, y_train = self._split_features_labels(train_data)
        X_val, y_val = self._split_features_labels(val_data)
        X_test, y_test = self._split_features_labels(test_data)
        
        if X_train.empty or len(y_train) == 0:
            return fold_results
        
        # Validate each model
        for model_name, model in models.items():
            try:
                # Retrain if configured
                if fold_idx % self.wf_config["retrain_frequency"] == 0:
                    self.logger.info(f"  Retraining {model_name} on fold {fold_idx}")
                    model.fit(X_train, y_train)
                
                # Validate on test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                fold_results["model_results"][model_name] = metrics
                
                # Store for aggregation
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = []
                self.model_performance[model_name].append(metrics)
                
            except Exception as e:
                self.logger.error(f"Failed to validate {model_name} on fold {fold_idx}: {str(e)}")
                fold_results["model_results"][model_name] = {"error": str(e)}
        
        return fold_results
    
    def _split_features_labels(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and labels."""
        if "label" in data.columns:
            X = data.drop(columns=["label"])
            y = data["label"]
        else:
            # Generate synthetic labels if needed
            X = data
            if "close" in data.columns:
                returns = data["close"].pct_change()
                y = (returns > 0).astype(int)
            else:
                y = pd.Series(np.random.randint(0, 2, size=len(data)), index=data.index)
        
        return X, y
    
    def _aggregate_fold_results(self) -> Dict[str, Dict[str, float]]:
        """Aggregate results across all folds."""
        aggregated = {}
        
        for model_name, fold_metrics in self.model_performance.items():
            # Calculate mean and std for each metric
            model_agg = {}
            
            metric_names = fold_metrics[0].keys() if fold_metrics else []
            
            for metric in metric_names:
                if metric != "error":
                    values = [fm[metric] for fm in fold_metrics if metric in fm]
                    if values:
                        model_agg[f"{metric}_mean"] = np.mean(values)
                        model_agg[f"{metric}_std"] = np.std(values)
                        model_agg[f"{metric}_min"] = np.min(values)
                        model_agg[f"{metric}_max"] = np.max(values)
            
            aggregated[model_name] = model_agg
        
        return aggregated
    
    def _calculate_overall_wf_metrics(
        self,
        aggregated_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate overall walk forward metrics."""
        metrics = {
            "n_models_validated": len(aggregated_results),
            "n_folds": self.wf_config["n_splits"],
            "avg_accuracy": [],
            "avg_f1": [],
            "stability_score": []
        }
        
        for model_results in aggregated_results.values():
            if "accuracy_mean" in model_results:
                metrics["avg_accuracy"].append(model_results["accuracy_mean"])
            if "f1_score_mean" in model_results:
                metrics["avg_f1"].append(model_results["f1_score_mean"])
            
            # Calculate stability (inverse of std)
            if "f1_score_std" in model_results:
                stability = 1.0 / (1.0 + model_results["f1_score_std"])
                metrics["stability_score"].append(stability)
        
        # Calculate averages
        for key in ["avg_accuracy", "avg_f1", "stability_score"]:
            if metrics[key]:
                metrics[key] = np.mean(metrics[key])
            else:
                metrics[key] = 0.0
        
        return metrics
    
    def _validate_step_specific_outputs(
        self,
        pipeline_state: Dict[str, Any]
    ) -> List[str]:
        """Validate step-specific outputs."""
        errors = []
        
        results_key = f"{self.full_step_name}_results"
        if results_key in pipeline_state:
            results = pipeline_state[results_key]
            if "fold_results" not in results or len(results["fold_results"]) == 0:
                errors.append("No fold results found in walk forward validation")
        
        return errors
    
    def _add_step_specific_summary(
        self,
        summary: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> None:
        """Add step-specific items to summary."""
        overall = validation_results.get("overall_metrics", {})
        
        # Key findings
        if overall.get("avg_f1", 0) > 0:
            summary["key_findings"].append(
                f"Average F1 score across folds: {overall['avg_f1']:.3f}"
            )
        
        if overall.get("stability_score", 0) > 0.8:
            summary["key_findings"].append(
                f"High model stability: {overall['stability_score']:.3f}"
            )
        
        # Warnings
        if overall.get("stability_score", 1.0) < 0.7:
            summary["warnings"].append(
                "Low stability score indicates high variance across time periods"
            )
        
        # Recommendations
        if overall.get("n_folds", 0) < 5:
            summary["recommendations"].append(
                "Consider increasing the number of folds for more robust validation"
            )
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return [
            "tactician_specialist_models",
            "market_data",
            "step15_tactician_specialist_training_completed"
        ]
    
    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return [
            f"{self.full_step_name}_results",
            f"{self.full_step_name}_summary"
        ]
    
    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ["step15_tactician_specialist_training"]