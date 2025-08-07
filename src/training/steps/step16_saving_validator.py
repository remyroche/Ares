"""
Validator for Step 16: Saving
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


class Step16SavingValidator(BaseValidator):
    """Validator for Step 16: Saving."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step16_saving", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the saving step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating saving step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("saving", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Saving step had errors")
            return False
        
        # 2. Validate final model files existence
        model_files_passed = self._validate_final_model_files(symbol, exchange, data_dir)
        if not model_files_passed:
            self.logger.error("❌ Final model files validation failed")
            return False
        
        # 3. Validate pipeline completeness
        completeness_passed = self._validate_pipeline_completeness(symbol, exchange, data_dir)
        if not completeness_passed:
            self.logger.error("❌ Pipeline completeness validation failed")
            return False
        
        # 4. Validate file integrity
        integrity_passed = self._validate_file_integrity(symbol, exchange, data_dir)
        if not integrity_passed:
            self.logger.error("❌ File integrity validation failed")
            return False
        
        # 5. Validate final model quality
        quality_passed = self._validate_final_model_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Final model quality validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Saving outcome is not favorable")
            return False
        
        self.logger.info("✅ Saving validation passed")
        return True
    
    def _validate_final_model_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that final model files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected final model file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_final_model.pkl",
                f"{data_dir}/{exchange}_{symbol}_final_model_metadata.json",
                f"{data_dir}/{exchange}_{symbol}_training_summary.json",
                f"{data_dir}/{exchange}_{symbol}_model_config.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "final_model_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing final model files: {missing_files}")
                return False
            
            self.logger.info("✅ All final model files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating final model files: {e}")
            return False
    
    def _validate_pipeline_completeness(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that all pipeline components are complete.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if pipeline is complete
        """
        try:
            # Check for all expected pipeline files
            pipeline_files = [
                f"{data_dir}/{exchange}_{symbol}_historical_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_regime_classification.json",
                f"{data_dir}/{exchange}_{symbol}_train_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_validation_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_test_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_features.pkl",
                f"{data_dir}/{exchange}_{symbol}_labels.pkl",
                f"{data_dir}/{exchange}_{symbol}_analyst_model.pkl",
                f"{data_dir}/{exchange}_{symbol}_analyst_ensemble.pkl",
                f"{data_dir}/{exchange}_{symbol}_tactician_model.pkl",
                f"{data_dir}/{exchange}_{symbol}_tactician_ensemble.pkl",
                f"{data_dir}/{exchange}_{symbol}_calibrated_models.pkl",
                f"{data_dir}/{exchange}_{symbol}_optimized_parameters.json",
                f"{data_dir}/{exchange}_{symbol}_walk_forward_results.json",
                f"{data_dir}/{exchange}_{symbol}_monte_carlo_results.json",
                f"{data_dir}/{exchange}_{symbol}_ab_testing_results.json"
            ]
            
            missing_pipeline_files = []
            for file_path in pipeline_files:
                if not os.path.exists(file_path):
                    missing_pipeline_files.append(file_path)
            
            if missing_pipeline_files:
                self.logger.warning(f"⚠️ Missing pipeline files: {len(missing_pipeline_files)} files missing")
                # Don't fail validation for missing pipeline files, just warn
            
            # Check training summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_training_summary.json"
            if os.path.exists(summary_file):
                import json
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                
                # Check if all steps are marked as completed
                if "completed_steps" in summary:
                    completed_steps = summary["completed_steps"]
                    expected_steps = 16
                    if len(completed_steps) < expected_steps:
                        self.logger.warning(f"⚠️ Incomplete pipeline: {len(completed_steps)}/{expected_steps} steps completed")
                
                # Check overall training status
                if "training_status" in summary:
                    status = summary["training_status"]
                    if status != "COMPLETED":
                        self.logger.warning(f"⚠️ Training status not completed: {status}")
            
            self.logger.info("✅ Pipeline completeness validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during pipeline completeness validation: {e}")
            return False
    
    def _validate_file_integrity(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate file integrity and accessibility.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if file integrity is acceptable
        """
        try:
            # Check final model file integrity
            model_file = f"{data_dir}/{exchange}_{symbol}_final_model.pkl"
            
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    
                    # Check if model has required methods
                    if not hasattr(model, 'predict'):
                        self.logger.error("❌ Final model missing predict method")
                        return False
                    
                    # Check model size
                    file_size = os.path.getsize(model_file)
                    if file_size < 1000:  # Less than 1KB
                        self.logger.warning(f"⚠️ Small final model file: {file_size} bytes")
                    elif file_size > 100000000:  # More than 100MB
                        self.logger.warning(f"⚠️ Large final model file: {file_size} bytes")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading final model: {e}")
                    return False
            
            # Check metadata file integrity
            metadata_file = f"{data_dir}/{exchange}_{symbol}_final_model_metadata.json"
            
            if os.path.exists(metadata_file):
                try:
                    import json
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    
                    # Check required metadata fields
                    required_fields = ["model_type", "training_date", "version"]
                    missing_fields = [field for field in required_fields if field not in metadata]
                    
                    if missing_fields:
                        self.logger.warning(f"⚠️ Missing metadata fields: {missing_fields}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading metadata: {e}")
                    return False
            
            # Check model config file
            config_file = f"{data_dir}/{exchange}_{symbol}_model_config.json"
            
            if os.path.exists(config_file):
                try:
                    import json
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    
                    # Check config completeness
                    if len(config) < 5:
                        self.logger.warning(f"⚠️ Sparse model config: {len(config)} parameters")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading model config: {e}")
                    return False
            
            self.logger.info("✅ File integrity validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during file integrity validation: {e}")
            return False
    
    def _validate_final_model_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate final model quality metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if model quality is acceptable
        """
        try:
            # Load final model metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_final_model_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check model performance metrics
                if "final_accuracy" in metadata:
                    final_acc = metadata["final_accuracy"]
                    acc_passed, acc_metrics = self.validate_model_performance(
                        final_acc, 0.0, "final_model"
                    )
                    self.validation_results["final_model_accuracy"] = acc_metrics
                    
                    if not acc_passed:
                        self.logger.error(f"❌ Final model accuracy too low: {final_acc:.3f}")
                        return False
                
                # Check model complexity
                if "model_complexity" in metadata:
                    complexity = metadata["model_complexity"]
                    if complexity > 1000000:
                        self.logger.warning(f"⚠️ High model complexity: {complexity}")
                    elif complexity < 100:
                        self.logger.warning(f"⚠️ Low model complexity: {complexity}")
                
                # Check model version
                if "version" in metadata:
                    version = metadata["version"]
                    if not version or version == "0.0.0":
                        self.logger.warning("⚠️ Invalid model version")
                
                # Check training date
                if "training_date" in metadata:
                    training_date = metadata["training_date"]
                    # Could add date validation here if needed
                
                # Check model type
                if "model_type" in metadata:
                    model_type = metadata["model_type"]
                    self.logger.info(f"Final model type: {model_type}")
                
                # Check ensemble information
                if "is_ensemble" in metadata:
                    is_ensemble = metadata["is_ensemble"]
                    if is_ensemble:
                        if "ensemble_size" in metadata:
                            ensemble_size = metadata["ensemble_size"]
                            if ensemble_size < 3:
                                self.logger.warning(f"⚠️ Small ensemble size: {ensemble_size}")
                
                # Check calibration information
                if "is_calibrated" in metadata:
                    is_calibrated = metadata["is_calibrated"]
                    if not is_calibrated:
                        self.logger.warning("⚠️ Final model is not calibrated")
                
                # Check validation metrics
                if "validation_metrics" in metadata:
                    val_metrics = metadata["validation_metrics"]
                    
                    if "cross_validation_score" in val_metrics:
                        cv_score = val_metrics["cross_validation_score"]
                        if cv_score < 0.6:
                            self.logger.warning(f"⚠️ Low cross-validation score: {cv_score:.3f}")
                    
                    if "test_accuracy" in val_metrics:
                        test_acc = val_metrics["test_accuracy"]
                        if test_acc < 0.6:
                            self.logger.warning(f"⚠️ Low test accuracy: {test_acc:.3f}")
            
            self.logger.info("✅ Final model quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during final model quality validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the step16_saving validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step16SavingValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    
    return {
        "step_name": "step16_saving",
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
            "saving": {
                "status": "SUCCESS",
                "duration": 120.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
