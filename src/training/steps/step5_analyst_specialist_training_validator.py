"""
Validator for Step 5: Analyst Specialist Training
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step5AnalystSpecialistTrainingValidator(BaseValidator):
    """Validator for Step 5: Analyst Specialist Training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step5_analyst_specialist_training", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the analyst specialist training step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating analyst specialist training step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("analyst_specialist_training", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Analyst specialist training step had errors")
            return False
        
        # 2. Validate model files existence
        model_files_passed = self._validate_model_files_existence(symbol, exchange, data_dir)
        if not model_files_passed:
            self.logger.error("❌ Model files validation failed")
            return False
        
        # 3. Validate model performance
        performance_passed = self._validate_model_performance(symbol, exchange, data_dir)
        if not performance_passed:
            self.logger.error("❌ Model performance validation failed")
            return False
        
        # 4. Validate training metrics
        metrics_passed = self._validate_training_metrics(symbol, exchange, data_dir)
        if not metrics_passed:
            self.logger.error("❌ Training metrics validation failed")
            return False
        
        # 5. Validate model quality
        quality_passed = self._validate_model_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Model quality validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Analyst specialist training outcome is not favorable")
            return False
        
        self.logger.info("✅ Analyst specialist training validation passed")
        return True
    
    def _validate_model_files_existence(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that all expected model files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if all files exist
        """
        try:
            # Expected model file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_analyst_model.pkl",
                f"{data_dir}/{exchange}_{symbol}_analyst_model_metadata.json",
                f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "model_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing model files: {missing_files}")
                return False
            
            self.logger.info("✅ All model files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating model files existence: {e}")
            return False
    
    def _validate_model_performance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate model performance metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if performance is acceptable
        """
        try:
            # Load training history
            history_file = f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
            
            if not os.path.exists(history_file):
                self.logger.warning(f"⚠️ Training history file not found: {history_file}")
                return True  # Not critical for validation
            
            import json
            with open(history_file, "r") as f:
                training_history = json.load(f)
            
            # Extract performance metrics
            if "metrics" in training_history:
                metrics = training_history["metrics"]
                
                # Validate accuracy
                if "accuracy" in metrics:
                    accuracy = metrics["accuracy"]
                    accuracy_passed, accuracy_metrics = self.validate_model_performance(
                        accuracy, 0.0, "analyst_model"
                    )
                    self.validation_results["model_accuracy"] = accuracy_metrics
                    
                    if not accuracy_passed:
                        self.logger.error(f"❌ Model accuracy too low: {accuracy:.3f}")
                        return False
                
                # Validate loss
                if "loss" in metrics:
                    loss = metrics["loss"]
                    loss_passed, loss_metrics = self.validate_model_performance(
                        0.0, loss, "analyst_model"
                    )
                    self.validation_results["model_loss"] = loss_metrics
                    
                    if not loss_passed:
                        self.logger.error(f"❌ Model loss too high: {loss:.3f}")
                        return False
                
                # Validate other metrics
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ["accuracy", "loss"]:
                        if isinstance(metric_value, (int, float)):
                            # Record custom metric
                            metrics.record_validation_result(
                                step_name=self.step_name,
                                validation_type=f"custom_metric_{metric_name}",
                                passed=metric_value > 0,
                                reason=f"{metric_name}: {metric_value:.3f}"
                            )
            
            self.logger.info("✅ Model performance validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during model performance validation: {e}")
            return False
    
    def _validate_training_metrics(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate training metrics and convergence.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if training metrics are acceptable
        """
        try:
            history_file = f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
            
            if not os.path.exists(history_file):
                self.logger.warning(f"⚠️ Training history file not found: {history_file}")
                return True
            
            import json
            with open(history_file, "r") as f:
                training_history = json.load(f)
            
            # Check for training epochs
            if "epochs" in training_history:
                epochs = training_history["epochs"]
                if epochs < 10:
                    self.logger.warning(f"⚠️ Few training epochs: {epochs}")
                elif epochs > 1000:
                    self.logger.warning(f"⚠️ Many training epochs: {epochs}")
            
            # Check for convergence indicators
            if "converged" in training_history:
                converged = training_history["converged"]
                if not converged:
                    self.logger.warning("⚠️ Model did not converge")
            
            # Check for overfitting indicators
            if "train_accuracy" in training_history and "val_accuracy" in training_history:
                train_acc = training_history["train_accuracy"]
                val_acc = training_history["val_accuracy"]
                
                if train_acc - val_acc > 0.1:  # Overfitting if train > val by more than 10%
                    self.logger.warning(f"⚠️ Potential overfitting: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            
            # Check for training time
            if "training_time" in training_history:
                training_time = training_history["training_time"]
                if training_time > 3600:  # More than 1 hour
                    self.logger.warning(f"⚠️ Long training time: {training_time:.1f}s")
                elif training_time < 60:  # Less than 1 minute
                    self.logger.warning(f"⚠️ Short training time: {training_time:.1f}s")
            
            self.logger.info("✅ Training metrics validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during training metrics validation: {e}")
            return False
    
    def _validate_model_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate model quality characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if model quality is acceptable
        """
        try:
            # Load model metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_analyst_model_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check model type
                if "model_type" in metadata:
                    model_type = metadata["model_type"]
                    self.logger.info(f"Model type: {model_type}")
                
                # Check model parameters
                if "parameters" in metadata:
                    params = metadata["parameters"]
                    param_count = len(params)
                    if param_count < 100:
                        self.logger.warning(f"⚠️ Few model parameters: {param_count}")
                    elif param_count > 1000000:
                        self.logger.warning(f"⚠️ Many model parameters: {param_count}")
                
                # Check model size
                if "model_size_mb" in metadata:
                    model_size = metadata["model_size_mb"]
                    if model_size > 100:  # More than 100MB
                        self.logger.warning(f"⚠️ Large model size: {model_size:.1f}MB")
                    elif model_size < 0.1:  # Less than 0.1MB
                        self.logger.warning(f"⚠️ Small model size: {model_size:.1f}MB")
                
                # Check feature importance
                if "feature_importance" in metadata:
                    feature_importance = metadata["feature_importance"]
                    if isinstance(feature_importance, dict):
                        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        self.logger.info(f"Top 5 features: {top_features}")
            
            # Load and validate the actual model
            model_file = f"{data_dir}/{exchange}_{symbol}_analyst_model.pkl"
            
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    
                    # Basic model validation
                    if hasattr(model, 'predict'):
                        self.logger.info("✅ Model has predict method")
                    else:
                        self.logger.error("❌ Model missing predict method")
                        return False
                    
                    if hasattr(model, 'fit'):
                        self.logger.info("✅ Model has fit method")
                    else:
                        self.logger.warning("⚠️ Model missing fit method")
                    
                    # Check model attributes
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if len(importances) > 0:
                            non_zero_features = np.sum(importances > 0)
                            if non_zero_features < 5:
                                self.logger.warning(f"⚠️ Few non-zero feature importances: {non_zero_features}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading model: {e}")
                    return False
            
            self.logger.info("✅ Model quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during model quality validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 5 Analyst Specialist Training validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step5AnalystSpecialistTrainingValidator(CONFIG)
    return await validator.run_validation(training_input, pipeline_state)


if __name__ == "__main__":
    import asyncio
    
    # Example usage
    async def test_validator():
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training"
        }
        
        pipeline_state = {
            "analyst_specialist_training": {
                "status": "SUCCESS",
                "duration": 300.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
