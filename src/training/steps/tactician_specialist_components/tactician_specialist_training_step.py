"""Step 15: Tactician Specialist Training - Migrated to use BaseStep pattern.

This step trains specialist tactician models with regime-specific tactics.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.core.decorators import handles_errors, log_execution_time, validates
from src.training.base_step import BaseStep
from src.utils.logger import system_logger

# Import component modules
from .specialist_trainer import SpecialistTrainer
from .regime_tactics import RegimeTactics
from .model_selector import ModelSelector
from .performance_evaluator import PerformanceEvaluator


class TacticianSpecialistTrainingStep(BaseStep):
    """Step 15: Tactician Specialist Training with regime-specific models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Tactician Specialist Training step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "15", "tactician_specialist_training")
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Initialize components
        self.specialist_trainer = SpecialistTrainer(self.config)
        self.regime_tactics = RegimeTactics(self.config)
        self.model_selector = ModelSelector(self.config)
        self.performance_evaluator = PerformanceEvaluator(self.config)
        
        # Initialize specialist configuration
        self.specialist_config = self._initialize_specialist_config()
        
        # Storage for regime-specific models and results
        self.regime_specialist_models: Dict[str, Dict[str, Any]] = {}
        self.regime_training_results: Dict[str, Dict[str, Any]] = {}
        self.regime_validation_results: Dict[str, Dict[str, Any]] = {}
        
    def _initialize_specialist_config(self) -> Dict[str, Any]:
        """Initialize specialist training configuration."""
        return {
            # Model types to train
            "model_types": ["lightgbm", "xgboost", "neural_network", "random_forest"],
            "enable_sr_integration": True,  # Support/Resistance integration
            
            # Training configuration
            "regime_specific_training": True,
            "min_regime_samples": 500,
            "validation_split": 0.2,
            "test_split": 0.1,
            
            # Specialist tactics
            "tactics": {
                "breakout": {"enabled": True, "min_confidence": 0.8},
                "reversal": {"enabled": True, "min_confidence": 0.85},
                "continuation": {"enabled": True, "min_confidence": 0.75},
                "range_bound": {"enabled": True, "min_confidence": 0.7}
            },
            
            # Performance thresholds
            "min_precision": 0.85,
            "min_recall": 0.1,
            "min_f1_score": 0.2,
            
            # Feature engineering
            "feature_engineering": {
                "use_technical_indicators": True,
                "use_market_microstructure": True,
                "use_regime_features": True,
                "use_sr_features": True
            },
            
            # Model optimization
            "hyperparameter_optimization": True,
            "feature_selection": True,
            "ensemble_specialists": True,
            
            # Multi-timeframe analysis
            "timeframes": ["1m", "5m", "15m"],
            "primary_timeframe": "1m"
        }
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for required tactician labeled data
        if "step14_tactician_labeling_completed" not in pipeline_state:
            errors.append("Step 14 (Tactician Labeling) must be completed before specialist training")
            
        if "tactician_labeled_data" not in pipeline_state:
            errors.append("No tactician labeled data found in pipeline state")
            
        # Check for regime data
        if "regime_data" not in pipeline_state:
            errors.append("No regime data found for regime-specific training")
            
        # Check for features
        if "features" not in pipeline_state:
            errors.append("No feature data found for training")
            
        # Validate configuration
        required_config = ["specialist_training", "model_types", "tactics"]
        for key in required_config:
            if key not in self.config:
                errors.append(f"Missing required configuration: {key}")
                
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="tactician specialist training logic"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the main tactician specialist training logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with trained specialist models
        """
        self.logger.info("🎯 Starting tactician specialist training...")
        
        # Extract required data
        labeled_data = pipeline_state["tactician_labeled_data"]
        regime_data = pipeline_state["regime_data"]
        features = pipeline_state.get("features", pd.DataFrame())
        
        # Prepare training data
        training_data = self._prepare_training_data(labeled_data, features)
        
        # Train specialists for each regime
        all_specialist_models = {}
        all_training_results = {}
        all_validation_results = {}
        
        for regime_id, regime_info in regime_data.items():
            self.logger.info(f"📊 Training specialists for regime {regime_id}...")
            
            # Get regime-specific data
            regime_training_data = self._get_regime_training_data(
                training_data, regime_info
            )
            
            if len(regime_training_data) < self.specialist_config["min_regime_samples"]:
                self.logger.warning(
                    f"Regime {regime_id} has insufficient samples "
                    f"({len(regime_training_data)} < {self.specialist_config['min_regime_samples']})"
                )
                continue
            
            # Determine regime tactics
            regime_tactics_config = await self.regime_tactics.determine_tactics(
                regime_id, regime_info, regime_training_data
            )
            
            # Train specialist models for this regime
            regime_specialists = await self._train_regime_specialists(
                regime_id,
                regime_training_data,
                regime_tactics_config
            )
            
            if regime_specialists:
                all_specialist_models[regime_id] = regime_specialists["models"]
                all_training_results[regime_id] = regime_specialists["training_results"]
                all_validation_results[regime_id] = regime_specialists["validation_results"]
        
        # Create cross-regime specialists if multiple regimes
        if len(all_specialist_models) > 1:
            self.logger.info("🔄 Creating cross-regime specialist models...")
            cross_regime_specialists = await self._create_cross_regime_specialists(
                all_specialist_models, training_data
            )
            
            if cross_regime_specialists:
                all_specialist_models["cross_regime"] = cross_regime_specialists["models"]
                all_training_results["cross_regime"] = cross_regime_specialists["training_results"]
                all_validation_results["cross_regime"] = cross_regime_specialists["validation_results"]
        
        # Evaluate all specialists
        self.logger.info("📈 Evaluating specialist models...")
        evaluation_results = await self.performance_evaluator.evaluate_all_specialists(
            all_specialist_models, training_data
        )
        
        # Update pipeline state
        result = pipeline_state.copy()
        result["tactician_specialist_models"] = all_specialist_models
        result["specialist_training_results"] = all_training_results
        result["specialist_validation_results"] = all_validation_results
        result["specialist_evaluation_results"] = evaluation_results
        
        # Add training summary
        result["specialist_training_summary"] = self._create_training_summary(
            all_specialist_models,
            all_training_results,
            evaluation_results
        )
        
        return result
    
    async def _train_regime_specialists(
        self,
        regime_id: str,
        training_data: pd.DataFrame,
        tactics_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Train specialist models for a specific regime.
        
        Args:
            regime_id: Regime identifier
            training_data: Training data for this regime
            tactics_config: Regime-specific tactics configuration
            
        Returns:
            Dictionary with trained models and results
        """
        try:
            # Split data
            train_data, val_data, test_data = self._split_training_data(training_data)
            
            # Extract features and labels
            X_train, y_train = self._extract_features_labels(train_data)
            X_val, y_val = self._extract_features_labels(val_data)
            X_test, y_test = self._extract_features_labels(test_data)
            
            # Train models for each enabled tactic
            specialist_models = {}
            training_results = {}
            validation_results = {}
            
            for tactic_name, tactic_config in tactics_config.items():
                if not tactic_config.get("enabled", False):
                    continue
                
                self.logger.info(f"  🎯 Training {tactic_name} specialist for regime {regime_id}...")
                
                # Select and train models for this tactic
                tactic_models = await self.specialist_trainer.train_tactic_models(
                    tactic_name,
                    X_train, y_train,
                    X_val, y_val,
                    self.specialist_config["model_types"],
                    regime_id
                )
                
                # Evaluate models
                tactic_validation = await self._evaluate_tactic_models(
                    tactic_models, X_test, y_test, tactic_config
                )
                
                # Select best models based on performance
                best_models = await self.model_selector.select_best_models(
                    tactic_models,
                    tactic_validation,
                    tactic_config
                )
                
                if best_models:
                    specialist_models[tactic_name] = best_models
                    training_results[tactic_name] = {
                        "models_trained": len(tactic_models),
                        "models_selected": len(best_models),
                        "training_samples": len(X_train)
                    }
                    validation_results[tactic_name] = tactic_validation
            
            self.logger.info(
                f"  ✅ Trained {len(specialist_models)} specialist tactics "
                f"for regime {regime_id}"
            )
            
            return {
                "models": specialist_models,
                "training_results": training_results,
                "validation_results": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train specialists for regime {regime_id}: {str(e)}")
            return None
    
    async def _create_cross_regime_specialists(
        self,
        regime_specialists: Dict[str, Dict[str, Any]],
        full_training_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Create specialist models that work across regimes.
        
        Args:
            regime_specialists: Specialist models for each regime
            full_training_data: Complete training data
            
        Returns:
            Dictionary with cross-regime specialists
        """
        try:
            # Collect best models from each regime
            cross_regime_models = {}
            
            # For each tactic, create ensemble from regime specialists
            all_tactics = set()
            for regime_models in regime_specialists.values():
                all_tactics.update(regime_models.keys())
            
            for tactic in all_tactics:
                tactic_models = []
                
                # Collect models for this tactic from all regimes
                for regime_id, regime_models in regime_specialists.items():
                    if tactic in regime_models:
                        tactic_models.extend([
                            (f"{regime_id}_{model_name}", model)
                            for model_name, model in regime_models[tactic].items()
                        ])
                
                if tactic_models:
                    # Create ensemble
                    ensemble = await self.specialist_trainer.create_tactic_ensemble(
                        tactic, tactic_models
                    )
                    cross_regime_models[tactic] = {"ensemble": ensemble}
            
            # Validate cross-regime specialists
            _, val_data, test_data = self._split_training_data(full_training_data)
            X_test, y_test = self._extract_features_labels(test_data)
            
            validation_results = {}
            for tactic, models in cross_regime_models.items():
                validation = await self._evaluate_tactic_models(
                    models, X_test, y_test, {"min_confidence": 0.7}
                )
                validation_results[tactic] = validation
            
            return {
                "models": cross_regime_models,
                "training_results": {
                    "type": "cross_regime_ensemble",
                    "n_regimes": len(regime_specialists),
                    "tactics": list(cross_regime_models.keys())
                },
                "validation_results": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create cross-regime specialists: {str(e)}")
            return None
    
    def _prepare_training_data(
        self,
        labeled_data: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare data for specialist training.
        
        Args:
            labeled_data: Labeled tactician data
            features: Feature data
            
        Returns:
            Prepared training data
        """
        # Merge labeled data with features
        if not features.empty and len(features) == len(labeled_data):
            training_data = pd.concat([features, labeled_data], axis=1)
        else:
            training_data = labeled_data.copy()
        
        # Remove samples with no labels
        if 'label' in training_data.columns:
            training_data = training_data[training_data['label'] != 0]
        
        # Add derived features if configured
        if self.specialist_config["feature_engineering"]["use_technical_indicators"]:
            # Add technical indicators (placeholder)
            pass
        
        if self.specialist_config["feature_engineering"]["use_regime_features"]:
            # Add regime-based features
            if 'regime_id' in training_data.columns:
                # One-hot encode regime
                regime_dummies = pd.get_dummies(
                    training_data['regime_id'], 
                    prefix='regime'
                )
                training_data = pd.concat([training_data, regime_dummies], axis=1)
        
        return training_data
    
    def _get_regime_training_data(
        self,
        training_data: pd.DataFrame,
        regime_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """Extract training data for a specific regime."""
        regime_mask = regime_info.get("mask", [])
        if isinstance(regime_mask, list) and regime_mask:
            regime_mask = np.array(regime_mask)
            # Ensure mask aligns with filtered training data
            if len(regime_mask) == len(training_data):
                return training_data[regime_mask]
        
        # Fallback to regime_id if available
        if 'regime_id' in training_data.columns:
            regime_id = regime_info.get("id", regime_info.get("regime_id"))
            return training_data[training_data['regime_id'] == regime_id]
        
        return training_data
    
    def _split_training_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        n_samples = len(data)
        
        # Calculate split indices
        val_size = int(n_samples * self.specialist_config["validation_split"])
        test_size = int(n_samples * self.specialist_config["test_split"])
        train_size = n_samples - val_size - test_size
        
        # Time-based split (assuming data is time-ordered)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def _extract_features_labels(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and labels from data."""
        # Identify label column
        label_col = 'label'
        
        # Identify feature columns (exclude metadata columns)
        exclude_cols = [
            label_col, 'regime_id', 'timestamp', 'barrier_type',
            'exit_time', 'potential_profit_pct', 'signal_strength'
        ]
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols]
        y = data[label_col] if label_col in data.columns else pd.Series(index=data.index)
        
        return X, y
    
    async def _evaluate_tactic_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tactic_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate models for a specific tactic.
        
        Args:
            models: Dictionary of models
            X_test: Test features
            y_test: Test labels
            tactic_config: Tactic configuration
            
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation_results = {}
        
        for model_name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "n_predictions": len(y_pred),
                    "positive_rate": (y_pred > 0).mean() if hasattr(y_pred, 'mean') else 0
                }
                
                # Check if model meets minimum requirements
                min_precision = tactic_config.get("min_confidence", self.specialist_config["min_precision"])
                metrics["meets_requirements"] = metrics["precision"] >= min_precision
                
                evaluation_results[model_name] = metrics
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate model {model_name}: {str(e)}")
                evaluation_results[model_name] = {
                    "error": str(e),
                    "meets_requirements": False
                }
        
        return evaluation_results
    
    def _create_training_summary(
        self,
        all_models: Dict[str, Dict[str, Any]],
        training_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a summary of the specialist training process."""
        summary = {
            "total_regimes": len(all_models),
            "total_specialists": sum(
                len(tactics) for tactics in all_models.values()
            ),
            "regime_summaries": {},
            "best_performers": {},
            "average_metrics": {
                "precision": [],
                "recall": [],
                "accuracy": []
            }
        }
        
        # Summarize each regime
        for regime_id, regime_models in all_models.items():
            regime_summary = {
                "tactics": list(regime_models.keys()),
                "total_models": sum(
                    len(models) if isinstance(models, dict) else 1
                    for models in regime_models.values()
                ),
                "training_info": training_results.get(regime_id, {})
            }
            summary["regime_summaries"][regime_id] = regime_summary
        
        # Find best performers
        if evaluation_results:
            for regime_id, regime_eval in evaluation_results.items():
                if isinstance(regime_eval, dict):
                    best_tactic = None
                    best_score = -np.inf
                    
                    for tactic, metrics in regime_eval.items():
                        if isinstance(metrics, dict) and "precision" in metrics:
                            if metrics["precision"] > best_score:
                                best_score = metrics["precision"]
                                best_tactic = tactic
                    
                    if best_tactic:
                        summary["best_performers"][regime_id] = {
                            "tactic": best_tactic,
                            "precision": best_score
                        }
        
        return summary
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for required outputs
        required_outputs = [
            "tactician_specialist_models",
            "specialist_training_results",
            "specialist_validation_results",
            "specialist_evaluation_results",
            "specialist_training_summary"
        ]
        
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f"Missing required output: {output}")
        
        # Validate specialist models
        if "tactician_specialist_models" in pipeline_state:
            models = pipeline_state["tactician_specialist_models"]
            if not isinstance(models, dict):
                errors.append("Specialist models must be a dictionary")
            elif len(models) == 0:
                errors.append("No specialist models were trained")
        
        # Validate summary
        if "specialist_training_summary" in pipeline_state:
            summary = pipeline_state["specialist_training_summary"]
            if summary.get("total_specialists", 0) == 0:
                errors.append("No specialists were successfully trained")
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return [
            "tactician_labeled_data",
            "regime_data",
            "features",
            "step14_tactician_labeling_completed"
        ]
    
    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return [
            "tactician_specialist_models",
            "specialist_training_results",
            "specialist_validation_results",
            "specialist_evaluation_results",
            "specialist_training_summary"
        ]
    
    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ["step14_tactician_labeling"]