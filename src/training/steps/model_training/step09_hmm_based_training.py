"""Step 9: HMM Based Training - Refactored to use BaseStep.

This module implements HMM-based model training with multi-output support
and regime-specific optimization.
"""

from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.model_training.hmm_training_components import (
import asyncio

    HMMModelTrainer, RegimeSpecificTrainer, MultiOutputTrainer, 
    ModelEvaluator, HyperparameterOptimizer
)


class HmmBasedTrainingStep(BaseStep):
    """Step 9: HMM Based Training using standardized base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HMM based training step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "09", "hmm_based_training")
        
        # Step-specific configuration
        self.training_config = config.get("hmm_training_config", {
            "enable_multi_output": True,
            "enable_regime_specific": True,
            "model_types": ["lightgbm", "random_forest"],
            "optimization": {
                "enable": True,
                "n_trials": 50,
                "cv_folds": 5
            },
            "ensemble": {
                "enable": True,
                "method": "voting",
                "weights": "optimized"
            },
            "sr_integration": {
                "enable": True,
                "use_optimized_params": True
            }
        })
        
        # Components
        self.hmm_trainer = None
        self.regime_trainer = None
        self.multi_output_trainer = None
        self.evaluator = None
        self.optimizer = None
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            # Initialize HMM model trainer
            self.hmm_trainer = HMMModelTrainer(
                config=self.training_config
            )
            
            # Initialize regime-specific trainer if enabled
            if self.training_config.get("enable_regime_specific", True):
                self.regime_trainer = RegimeSpecificTrainer(
                    config=self.training_config
                )
            
            # Initialize multi-output trainer if enabled
            if self.training_config.get("enable_multi_output", True):
                self.multi_output_trainer = MultiOutputTrainer(
                    config=self.training_config
                )
            
            # Initialize model evaluator
            self.evaluator = ModelEvaluator()
            
            # Initialize hyperparameter optimizer if enabled
            if self.training_config.get("optimization", {}).get("enable", True):
                self.optimizer = HyperparameterOptimizer(
                    config=self.training_config.get("optimization", {})
                )
            
            self.logger.info("✅ HMM training components initialized")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Some HMM training components not available: {e}")
            # Will use simplified training
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for engineered data with labels
        has_data = False
        if "engineered_data" in pipeline_state:
            has_data = True
        elif all(f"{split}_data" in pipeline_state for split in ["train", "val", "test"]):
            has_data = True
        
        if not has_data:
            errors.append("No engineered data from previous steps")
        
        # Check for labels
        if has_data:
            # Check any data split for labels
            data_sample = None
            if "engineered_data" in pipeline_state:
                data_sample = next(iter(pipeline_state["engineered_data"].values()))
            elif "train_data" in pipeline_state:
                data_sample = pipeline_state["train_data"]
            
            if data_sample is not None and isinstance(data_sample, pd.DataFrame):
                if "label" not in data_sample.columns and "triple_barrier_label" not in data_sample.columns:
                    errors.append("No labels found in data")
        
        # Check for regime information if regime-specific training is enabled
        if self.training_config.get("enable_regime_specific", True):
            if "regime_labels" not in pipeline_state and "regime_characteristics" not in pipeline_state:
                self.logger.warning("Regime information not available, will use standard training")
        
        # Check for selected features
        if "selected_features" not in pipeline_state:
            self.logger.warning("No selected features, will use all features")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="HMM based training execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute HMM based training logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        self.logger.info("🚀 Starting HMM based model training...")
        
        # Get data splits
        data_splits = self._get_data_splits(pipeline_state)
        selected_features = pipeline_state.get("selected_features", [])
        
        # Prepare training data
        self.logger.info("📊 Preparing training data...")
        prepared_data = self._prepare_training_data(
            data_splits, 
            selected_features
        )
        
        # Initialize results storage
        training_results = {
            "models": {},
            "performance": {},
            "feature_importance": {},
            "optimization_results": {}
        }
        
        # Train models based on configuration
        if self.training_config.get("enable_regime_specific", True) and "regime_labels" in pipeline_state:
            self.logger.info("🎯 Training regime-specific models...")
            regime_results = await self._train_regime_specific_models(
                prepared_data,
                pipeline_state
            )
            training_results["regime_models"] = regime_results
        else:
            self.logger.info("🎯 Training standard models...")
            standard_results = await self._train_standard_models(prepared_data)
            training_results["standard_models"] = standard_results
        
        # Train multi-output models if enabled
        if self.training_config.get("enable_multi_output", True):
            self.logger.info("🎯 Training multi-output models...")
            multi_output_results = await self._train_multi_output_models(prepared_data)
            training_results["multi_output_models"] = multi_output_results
        
        # Evaluate all models
        self.logger.info("📊 Evaluating model performance...")
        evaluation_results = await self._evaluate_models(
            training_results,
            prepared_data
        )
        training_results["evaluation"] = evaluation_results
        
        # Select best models
        best_models = self._select_best_models(training_results)
        
        # Generate reports
        reports = self._generate_training_reports(
            training_results,
            evaluation_results
        )
        
        # Update pipeline state
        pipeline_state.update({
            "trained_models": training_results["models"],
            "model_performance": training_results["performance"],
            "feature_importance": training_results["feature_importance"],
            "best_models": best_models,
            "training_reports": reports,
            "training_config": self.training_config
        })
        
        # Save outputs
        await self._save_outputs(training_input, pipeline_state)
        
        return pipeline_state
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if models were trained
        if "trained_models" not in pipeline_state:
            errors.append("No trained models in pipeline state")
            return False, errors
        
        trained_models = pipeline_state["trained_models"]
        if len(trained_models) == 0:
            errors.append("No models were successfully trained")
        
        # Check model performance
        if "model_performance" not in pipeline_state:
            errors.append("No model performance metrics")
        else:
            performance = pipeline_state["model_performance"]
            # Check if any model has reasonable performance
            has_good_model = False
            for model_name, metrics in performance.items():
                if isinstance(metrics, dict):
                    # For classification models
                    if "accuracy" in metrics and metrics["accuracy"] > 0.5:
                        has_good_model = True
                        break
                    # For regression models
                    if "r2_score" in metrics and metrics["r2_score"] > 0:
                        has_good_model = True
                        break
            
            if not has_good_model:
                self.logger.warning("⚠️ No models achieved good performance")
        
        # Check best models selection
        if "best_models" not in pipeline_state:
            errors.append("No best models selected")
        
        return len(errors) == 0, errors
    
    def _get_data_splits(self, pipeline_state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Get data splits from pipeline state.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary of data splits
        """
        data_splits = {}
        
        # Check for engineered data
        if "engineered_data" in pipeline_state:
            return pipeline_state["engineered_data"]
        
        # Otherwise get individual splits
        for split in ["train", "val", "test"]:
            if f"{split}_data" in pipeline_state:
                data_splits[split] = pipeline_state[f"{split}_data"]
        
        return data_splits
    
    def _prepare_training_data(
        self,
        data_splits: Dict[str, pd.DataFrame],
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """Prepare data for training.
        
        Args:
            data_splits: Dictionary of data splits
            selected_features: List of selected features
            
        Returns:
            Prepared training data
        """
        prepared = {}
        
        for split_name, data in data_splits.items():
            # Get feature columns
            if selected_features:
                feature_cols = [col for col in selected_features if col in data.columns]
            else:
                feature_cols = [col for col in data.columns if col.startswith("feature_")]
            
            # Get label column
            label_col = None
            if "label" in data.columns:
                label_col = "label"
            elif "triple_barrier_label" in data.columns:
                label_col = "triple_barrier_label"
            elif "label_binary" in data.columns:
                label_col = "label_binary"
            
            if label_col and feature_cols:
                # Remove rows with missing labels
                valid_mask = data[label_col].notna()
                
                prepared[split_name] = {
                    "features": data.loc[valid_mask, feature_cols],
                    "labels": data.loc[valid_mask, label_col],
                    "feature_names": feature_cols,
                    "label_name": label_col
                }
                
                # Add profit labels if available (for multi-output)
                if "label_exit_return" in data.columns:
                    prepared[split_name]["profit_labels"] = data.loc[valid_mask, "label_exit_return"]
                
                # Add regime labels if available
                if "regime_label" in data.columns:
                    prepared[split_name]["regime_labels"] = data.loc[valid_mask, "regime_label"]
                
                self.logger.info(
                    f"✅ Prepared {split_name} data: "
                    f"{len(prepared[split_name]['features'])} samples, "
                    f"{len(feature_cols)} features"
                )
        
        return prepared
    
    async def _train_regime_specific_models(
        self,
        prepared_data: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train regime-specific models.
        
        Args:
            prepared_data: Prepared training data
            pipeline_state: Pipeline state with regime information
            
        Returns:
            Training results
        """
        if self.regime_trainer:
            return await self.regime_trainer.train_regime_models(
                prepared_data,
                pipeline_state.get("regime_characteristics", {})
            )
        else:
            # Fallback implementation
            return self._simple_regime_training(prepared_data)
    
    async def _train_standard_models(
        self,
        prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train standard models.
        
        Args:
            prepared_data: Prepared training data
            
        Returns:
            Training results
        """
        if self.hmm_trainer:
            return await self.hmm_trainer.train_models(prepared_data)
        else:
            # Fallback implementation
            return self._simple_model_training(prepared_data)
    
    async def _train_multi_output_models(
        self,
        prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train multi-output models.
        
        Args:
            prepared_data: Prepared training data
            
        Returns:
            Training results
        """
        if self.multi_output_trainer:
            return await self.multi_output_trainer.train_multi_output(prepared_data)
        else:
            # Skip if no multi-output trainer
            return {}
    
    async def _evaluate_models(
        self,
        training_results: Dict[str, Any],
        prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate trained models.
        
        Args:
            training_results: Training results with models
            prepared_data: Prepared data for evaluation
            
        Returns:
            Evaluation results
        """
        if self.evaluator:
            return await self.evaluator.evaluate_all_models(
                training_results,
                prepared_data
            )
        else:
            # Simple evaluation
            return self._simple_evaluation(training_results, prepared_data)
    
    def _simple_model_training(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback model training.
        
        Args:
            prepared_data: Prepared training data
            
        Returns:
            Training results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        results = {"models": {}, "performance": {}}
        
        if "train" in prepared_data and "val" in prepared_data:
            train_data = prepared_data["train"]
            val_data = prepared_data["val"]
            
            # Train a simple Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(train_data["features"], train_data["labels"])
            
            # Evaluate
            val_pred = model.predict(val_data["features"])
            
            results["models"]["random_forest"] = model
            results["performance"]["random_forest"] = {
                "accuracy": accuracy_score(val_data["labels"], val_pred),
                "f1_score": f1_score(val_data["labels"], val_pred, average='weighted')
            }
            
            self.logger.info(
                f"✅ Trained Random Forest: "
                f"Accuracy={results['performance']['random_forest']['accuracy']:.4f}"
            )
        
        return results
    
    def _simple_regime_training(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple regime-specific training.
        
        Args:
            prepared_data: Prepared training data
            
        Returns:
            Training results
        """
        results = {"models": {}, "performance": {}}
        
        if "train" in prepared_data and "regime_labels" in prepared_data["train"]:
            train_data = prepared_data["train"]
            
            # Train one model per regime
            unique_regimes = np.unique(train_data["regime_labels"])
            
            for regime in unique_regimes:
                regime_mask = train_data["regime_labels"] == regime
                
                if np.sum(regime_mask) > 50:  # Minimum samples
                    # Use simple model training for this regime
                    regime_data = {
                        "train": {
                            "features": train_data["features"][regime_mask],
                            "labels": train_data["labels"][regime_mask]
                        }
                    }
                    
                    if "val" in prepared_data:
                        val_mask = prepared_data["val"]["regime_labels"] == regime
                        regime_data["val"] = {
                            "features": prepared_data["val"]["features"][val_mask],
                            "labels": prepared_data["val"]["labels"][val_mask]
                        }
                    
                    regime_results = self._simple_model_training(regime_data)
                    
                    for model_name, model in regime_results["models"].items():
                        results["models"][f"{model_name}_regime_{regime}"] = model
                    
                    for metric_name, metrics in regime_results["performance"].items():
                        results["performance"][f"{metric_name}_regime_{regime}"] = metrics
        
        return results
    
    def _simple_evaluation(
        self,
        training_results: Dict[str, Any],
        prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple model evaluation.
        
        Args:
            training_results: Training results
            prepared_data: Prepared data
            
        Returns:
            Evaluation results
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        evaluation = {}
        
        if "test" in prepared_data:
            test_data = prepared_data["test"]
            
            for model_category, category_results in training_results.items():
                if isinstance(category_results, dict) and "models" in category_results:
                    for model_name, model in category_results["models"].items():
                        if hasattr(model, "predict"):
                            try:
                                predictions = model.predict(test_data["features"])
                                
                                evaluation[model_name] = {
                                    "test_accuracy": accuracy_score(test_data["labels"], predictions),
                                    "test_f1": f1_score(test_data["labels"], predictions, average='weighted'),
                                    "test_precision": precision_score(test_data["labels"], predictions, average='weighted'),
                                    "test_recall": recall_score(test_data["labels"], predictions, average='weighted')
                                }
                            except Exception as e:
                                self.logger.warning(f"Failed to evaluate {model_name}: {e}")
        
        return evaluation
    
    def _select_best_models(self, training_results: Dict[str, Any]) -> Dict[str, str]:
        """Select best models based on performance.
        
        Args:
            training_results: Training results
            
        Returns:
            Dictionary of best models by category
        """
        best_models = {}
        
        # Get evaluation results
        evaluation = training_results.get("evaluation", {})
        
        if evaluation:
            # Find best overall model
            best_score = -1
            best_model_name = None
            
            for model_name, metrics in evaluation.items():
                # Use F1 score as primary metric
                score = metrics.get("test_f1", metrics.get("test_accuracy", 0))
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            
            if best_model_name:
                best_models["best_overall"] = best_model_name
                best_models["best_score"] = best_score
                
                self.logger.info(
                    f"🏆 Best model: {best_model_name} "
                    f"(score={best_score:.4f})"
                )
        
        return best_models
    
    def _generate_training_reports(
        self,
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate training reports.
        
        Args:
            training_results: Training results
            evaluation_results: Evaluation results
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        
        # Summary report
        summary_lines = [
            "HMM Based Training Summary",
            "=" * 40,
            "",
            "Models Trained:"
        ]
        
        model_count = 0
        for category, results in training_results.items():
            if isinstance(results, dict) and "models" in results:
                model_count += len(results["models"])
                summary_lines.append(f"  {category}: {len(results['models'])} models")
        
        summary_lines.extend([
            f"\nTotal models: {model_count}",
            "",
            "Performance Summary:"
        ])
        
        # Add top performing models
        if evaluation_results:
            sorted_models = sorted(
                evaluation_results.items(),
                key=lambda x: x[1].get("test_f1", x[1].get("test_accuracy", 0)),
                reverse=True
            )[:5]
            
            for model_name, metrics in sorted_models:
                summary_lines.append(
                    f"  {model_name}: "
                    f"Accuracy={metrics.get('test_accuracy', 0):.4f}, "
                    f"F1={metrics.get('test_f1', 0):.4f}"
                )
        
        reports["summary"] = "\n".join(summary_lines)
        
        # Detailed performance report
        perf_lines = [
            "Detailed Performance Report",
            "=" * 40,
            ""
        ]
        
        for model_name, metrics in evaluation_results.items():
            perf_lines.extend([
                f"{model_name}:",
                f"  Accuracy: {metrics.get('test_accuracy', 0):.4f}",
                f"  F1 Score: {metrics.get('test_f1', 0):.4f}",
                f"  Precision: {metrics.get('test_precision', 0):.4f}",
                f"  Recall: {metrics.get('test_recall', 0):.4f}",
                ""
            ])
        
        reports["performance"] = "\n".join(perf_lines)
        
        return reports
    
    async def _save_outputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get("output_dir", "output")) / "step09_hmm_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        if "trained_models" in pipeline_state:
            models_dir = output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Save model metadata
            model_metadata = {}
            
            for model_name, model in pipeline_state["trained_models"].items():
                try:
                    # Save model using joblib
                    import joblib
                    model_path = models_dir / f"{model_name}.joblib"
                    joblib.dump(model, model_path)
                    
                    model_metadata[model_name] = {
                        "path": str(model_path),
                        "type": type(model).__name__
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save model {model_name}: {e}")
            
            # Save metadata
            metadata_path = models_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(model_metadata)} models")
        
        # Save performance metrics
        if "model_performance" in pipeline_state:
            perf_path = output_dir / "model_performance.json"
            with open(perf_path, 'w') as f:
                json.dump(pipeline_state["model_performance"], f, indent=2)
            self.logger.info(f"💾 Saved performance metrics")
        
        # Save feature importance
        if "feature_importance" in pipeline_state:
            importance_path = output_dir / "feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(pipeline_state["feature_importance"], f, indent=2)
            self.logger.info(f"💾 Saved feature importance")
        
        # Save reports
        if "training_reports" in pipeline_state:
            for report_name, content in pipeline_state["training_reports"].items():
                report_path = output_dir / f"{report_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(content)
                self.logger.info(f"💾 Saved {report_name} report")
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["engineered_data with labels", "selected_features (optional)", "regime_labels (optional)"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return [
            "trained_models", "model_performance", 
            "feature_importance", "best_models", "training_reports"
        ]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ["07_enhanced_matrix_operations", "05_labeling"]