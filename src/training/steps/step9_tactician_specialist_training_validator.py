"""
Validator for Step 9: Tactician Specialist Training
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


class Step9TacticianSpecialistTrainingValidator(BaseValidator):
    """Validator for Step 9: Tactician Specialist Training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step9_tactician_specialist_training", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the tactician specialist training step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating tactician specialist training step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("tactician_specialist_training", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Tactician specialist training step had errors")
            return False
        
        # 2. Validate tactician model files existence
        model_files_passed = self._validate_tactician_model_files(symbol, exchange, data_dir)
        if not model_files_passed:
            self.logger.error("❌ Tactician model files validation failed")
            return False
        
        # 3. Validate tactician model performance
        performance_passed = self._validate_tactician_model_performance(symbol, exchange, data_dir)
        if not performance_passed:
            self.logger.error("❌ Tactician model performance validation failed")
            return False
        
        # 4. Validate training metrics
        metrics_passed = self._validate_tactician_training_metrics(symbol, exchange, data_dir)
        if not metrics_passed:
            self.logger.error("❌ Tactician training metrics validation failed")
            return False
        
        # 5. Validate model quality
        quality_passed = self._validate_tactician_model_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Tactician model quality validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Tactician specialist training outcome is not favorable")
            return False
        
        self.logger.info("✅ Tactician specialist training validation passed")
        return True
    
    def _validate_tactician_model_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that tactician model files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected tactician model file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_tactician_model.pkl",
                f"{data_dir}/{exchange}_{symbol}_tactician_model_metadata.json",
                f"{data_dir}/{exchange}_{symbol}_tactician_training_history.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "tactician_model_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing tactician model files: {missing_files}")
                return False
            
            self.logger.info("✅ All tactician model files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating tactician model files: {e}")
            return False
    
    def _validate_tactician_model_performance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate tactician model performance metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if performance is acceptable
        """
        try:
            # Load tactician training history
            history_file = f"{data_dir}/{exchange}_{symbol}_tactician_training_history.json"
            
            if not os.path.exists(history_file):
                self.logger.warning(f"⚠️ Tactician training history file not found: {history_file}")
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
                        accuracy, 0.0, "tactician_model"
                    )
                    self.validation_results["tactician_accuracy"] = accuracy_metrics
                    
                    if not accuracy_passed:
                        self.logger.error(f"❌ Tactician model accuracy too low: {accuracy:.3f}")
                        return False
                
                # Validate loss
                if "loss" in metrics:
                    loss = metrics["loss"]
                    loss_passed, loss_metrics = self.validate_model_performance(
                        0.0, loss, "tactician_model"
                    )
                    self.validation_results["tactician_loss"] = loss_metrics
                    
                    if not loss_passed:
                        self.logger.error(f"❌ Tactician model loss too high: {loss:.3f}")
                        return False
                
                # Validate signal prediction accuracy
                if "signal_accuracy" in metrics:
                    signal_acc = metrics["signal_accuracy"]
                    if signal_acc < 0.6:
                        self.logger.warning(f"⚠️ Low signal prediction accuracy: {signal_acc:.3f}")
                
                # Validate confidence calibration
                if "confidence_calibration" in metrics:
                    calibration_score = metrics["confidence_calibration"]
                    if calibration_score < 0.7:
                        self.logger.warning(f"⚠️ Poor confidence calibration: {calibration_score:.3f}")
            
            self.logger.info("✅ Tactician model performance validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during tactician model performance validation: {e}")
            return False
    
    def _validate_tactician_training_metrics(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate tactician training metrics and convergence.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if training metrics are acceptable
        """
        try:
            history_file = f"{data_dir}/{exchange}_{symbol}_tactician_training_history.json"
            
            if not os.path.exists(history_file):
                self.logger.warning(f"⚠️ Tactician training history file not found: {history_file}")
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
                    self.logger.warning("⚠️ Tactician model did not converge")
            
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
                    self.logger.warning(f"⚠️ Long tactician training time: {training_time:.1f}s")
                elif training_time < 60:  # Less than 1 minute
                    self.logger.warning(f"⚠️ Short tactician training time: {training_time:.1f}s")
            
            # Check for signal-specific metrics
            if "signal_precision" in training_history:
                signal_precision = training_history["signal_precision"]
                if signal_precision < 0.6:
                    self.logger.warning(f"⚠️ Low signal precision: {signal_precision:.3f}")
            
            if "signal_recall" in training_history:
                signal_recall = training_history["signal_recall"]
                if signal_recall < 0.6:
                    self.logger.warning(f"⚠️ Low signal recall: {signal_recall:.3f}")
            
            self.logger.info("✅ Tactician training metrics validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during tactician training metrics validation: {e}")
            return False
    
    def _validate_tactician_model_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate tactician model quality characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if model quality is acceptable
        """
        try:
            # Load tactician model metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_tactician_model_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check model type
                if "model_type" in metadata:
                    model_type = metadata["model_type"]
                    self.logger.info(f"Tactician model type: {model_type}")
                
                # Check model parameters
                if "parameters" in metadata:
                    params = metadata["parameters"]
                    param_count = len(params)
                    if param_count < 100:
                        self.logger.warning(f"⚠️ Few tactician model parameters: {param_count}")
                    elif param_count > 1000000:
                        self.logger.warning(f"⚠️ Many tactician model parameters: {param_count}")
                
                # Check model size
                if "model_size_mb" in metadata:
                    model_size = metadata["model_size_mb"]
                    if model_size > 100:  # More than 100MB
                        self.logger.warning(f"⚠️ Large tactician model size: {model_size:.1f}MB")
                    elif model_size < 0.1:  # Less than 0.1MB
                        self.logger.warning(f"⚠️ Small tactician model size: {model_size:.1f}MB")
                
                # Check signal prediction capabilities
                if "signal_prediction_accuracy" in metadata:
                    signal_acc = metadata["signal_prediction_accuracy"]
                    if signal_acc < 0.6:
                        self.logger.warning(f"⚠️ Low signal prediction accuracy: {signal_acc:.3f}")
                
                # Check confidence calibration
                if "confidence_calibration_score" in metadata:
                    calibration_score = metadata["confidence_calibration_score"]
                    if calibration_score < 0.7:
                        self.logger.warning(f"⚠️ Poor confidence calibration: {calibration_score:.3f}")
            
            # Load and validate the actual tactician model
            model_file = f"{data_dir}/{exchange}_{symbol}_tactician_model.pkl"
            
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    
                    # Basic model validation
                    if hasattr(model, 'predict'):
                        self.logger.info("✅ Tactician model has predict method")
                    else:
                        self.logger.error("❌ Tactician model missing predict method")
                        return False
                    
                    if hasattr(model, 'fit'):
                        self.logger.info("✅ Tactician model has fit method")
                    else:
                        self.logger.warning("⚠️ Tactician model missing fit method")
                    
                    # Check for signal prediction capabilities
                    if hasattr(model, 'predict_proba'):
                        self.logger.info("✅ Tactician model has probability prediction")
                    else:
                        self.logger.warning("⚠️ Tactician model missing probability prediction")
                    
                    # Check model attributes
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if len(importances) > 0:
                            non_zero_features = np.sum(importances > 0)
                            if non_zero_features < 5:
                                self.logger.warning(f"⚠️ Few non-zero feature importances: {non_zero_features}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading tactician model: {e}")
                    return False
            
            self.logger.info("✅ Tactician model quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during tactician model quality validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the step9_tactician_specialist_training validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step9TacticianSpecialistTrainingValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    
    return {
        "step_name": "step9_tactician_specialist_training",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time()
    }


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
            "tactician_specialist_training": {
                "status": "SUCCESS",
                "duration": 400.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
