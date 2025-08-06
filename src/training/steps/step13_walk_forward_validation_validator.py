"""
Validator for Step 13: Walk Forward Validation
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


class Step13WalkForwardValidationValidator(BaseValidator):
    """Validator for Step 13: Walk Forward Validation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step13_walk_forward_validation", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the walk forward validation step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating walk forward validation step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("walk_forward_validation", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Walk forward validation step had errors")
            return False
        
        # 2. Validate walk forward validation files existence
        validation_files_passed = self._validate_walk_forward_files(symbol, exchange, data_dir)
        if not validation_files_passed:
            self.logger.error("❌ Walk forward validation files validation failed")
            return False
        
        # 3. Validate walk forward performance
        performance_passed = self._validate_walk_forward_performance(symbol, exchange, data_dir)
        if not performance_passed:
            self.logger.error("❌ Walk forward performance validation failed")
            return False
        
        # 4. Validate walk forward stability
        stability_passed = self._validate_walk_forward_stability(symbol, exchange, data_dir)
        if not stability_passed:
            self.logger.error("❌ Walk forward stability validation failed")
            return False
        
        # 5. Validate walk forward consistency
        consistency_passed = self._validate_walk_forward_consistency(symbol, exchange, data_dir)
        if not consistency_passed:
            self.logger.error("❌ Walk forward consistency validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Walk forward validation outcome is not favorable")
            return False
        
        self.logger.info("✅ Walk forward validation validation passed")
        return True
    
    def _validate_walk_forward_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that walk forward validation files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected walk forward validation file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_walk_forward_results.json",
                f"{data_dir}/{exchange}_{symbol}_walk_forward_performance.json",
                f"{data_dir}/{exchange}_{symbol}_walk_forward_metadata.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "walk_forward_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing walk forward validation files: {missing_files}")
                return False
            
            self.logger.info("✅ All walk forward validation files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating walk forward validation files: {e}")
            return False
    
    def _validate_walk_forward_performance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate walk forward validation performance metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if performance is acceptable
        """
        try:
            # Load walk forward performance results
            performance_file = f"{data_dir}/{exchange}_{symbol}_walk_forward_performance.json"
            
            if os.path.exists(performance_file):
                import json
                with open(performance_file, "r") as f:
                    performance = json.load(f)
                
                # Check overall performance metrics
                if "overall_accuracy" in performance:
                    overall_acc = performance["overall_accuracy"]
                    acc_passed, acc_metrics = self.validate_model_performance(
                        overall_acc, 0.0, "walk_forward_model"
                    )
                    self.validation_results["walk_forward_accuracy"] = acc_metrics
                    
                    if not acc_passed:
                        self.logger.error(f"❌ Walk forward accuracy too low: {overall_acc:.3f}")
                        return False
                
                # Check performance stability
                if "performance_stability" in performance:
                    stability = performance["performance_stability"]
                    if stability < 0.7:
                        self.logger.warning(f"⚠️ Low walk forward performance stability: {stability:.3f}")
                
                # Check performance trend
                if "performance_trend" in performance:
                    trend = performance["performance_trend"]
                    if trend < -0.05:  # Declining performance
                        self.logger.warning(f"⚠️ Declining walk forward performance trend: {trend:.3f}")
                
                # Check individual fold performance
                if "fold_performance" in performance:
                    fold_perf = performance["fold_performance"]
                    
                    # Check for consistent performance across folds
                    accuracies = [fold.get("accuracy", 0) for fold in fold_perf]
                    if accuracies:
                        acc_std = np.std(accuracies)
                        if acc_std > 0.1:
                            self.logger.warning(f"⚠️ High walk forward performance variance: {acc_std:.3f}")
                        
                        # Check for poor performing folds
                        poor_folds = sum(1 for acc in accuracies if acc < 0.5)
                        if poor_folds > len(accuracies) * 0.3:  # More than 30% poor folds
                            self.logger.warning(f"⚠️ Many poor performing folds: {poor_folds}/{len(accuracies)}")
            
            self.logger.info("✅ Walk forward performance validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during walk forward performance validation: {e}")
            return False
    
    def _validate_walk_forward_stability(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate walk forward validation stability.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if stability is acceptable
        """
        try:
            # Load walk forward metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_walk_forward_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check number of folds
                if "fold_count" in metadata:
                    fold_count = metadata["fold_count"]
                    if fold_count < 3:
                        self.logger.warning(f"⚠️ Few walk forward folds: {fold_count}")
                    elif fold_count > 20:
                        self.logger.warning(f"⚠️ Many walk forward folds: {fold_count}")
                
                # Check fold size
                if "fold_size" in metadata:
                    fold_size = metadata["fold_size"]
                    if fold_size < 100:
                        self.logger.warning(f"⚠️ Small walk forward fold size: {fold_size}")
                    elif fold_size > 10000:
                        self.logger.warning(f"⚠️ Large walk forward fold size: {fold_size}")
                
                # Check overlap ratio
                if "overlap_ratio" in metadata:
                    overlap = metadata["overlap_ratio"]
                    if overlap > 0.8:
                        self.logger.warning(f"⚠️ High walk forward overlap ratio: {overlap:.3f}")
                    elif overlap < 0.1:
                        self.logger.warning(f"⚠️ Low walk forward overlap ratio: {overlap:.3f}")
                
                # Check temporal consistency
                if "temporal_consistency" in metadata:
                    consistency = metadata["temporal_consistency"]
                    if consistency < 0.6:
                        self.logger.warning(f"⚠️ Low walk forward temporal consistency: {consistency:.3f}")
            
            self.logger.info("✅ Walk forward stability validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during walk forward stability validation: {e}")
            return False
    
    def _validate_walk_forward_consistency(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate walk forward validation consistency.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if consistency is acceptable
        """
        try:
            # Load walk forward results
            results_file = f"{data_dir}/{exchange}_{symbol}_walk_forward_results.json"
            
            if os.path.exists(results_file):
                import json
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                # Check for consistent model performance
                if "model_performance" in results:
                    model_perf = results["model_performance"]
                    
                    # Check accuracy consistency
                    if "accuracy_consistency" in model_perf:
                        acc_consistency = model_perf["accuracy_consistency"]
                        if acc_consistency < 0.7:
                            self.logger.warning(f"⚠️ Low accuracy consistency: {acc_consistency:.3f}")
                    
                    # Check loss consistency
                    if "loss_consistency" in model_perf:
                        loss_consistency = model_perf["loss_consistency"]
                        if loss_consistency < 0.7:
                            self.logger.warning(f"⚠️ Low loss consistency: {loss_consistency:.3f}")
                
                # Check parameter consistency
                if "parameter_consistency" in results:
                    param_consistency = results["parameter_consistency"]
                    if param_consistency < 0.6:
                        self.logger.warning(f"⚠️ Low parameter consistency: {param_consistency:.3f}")
                
                # Check prediction consistency
                if "prediction_consistency" in results:
                    pred_consistency = results["prediction_consistency"]
                    if pred_consistency < 0.7:
                        self.logger.warning(f"⚠️ Low prediction consistency: {pred_consistency:.3f}")
            
            self.logger.info("✅ Walk forward consistency validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during walk forward consistency validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 13 Walk Forward Validation validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step13WalkForwardValidationValidator(CONFIG)
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
            "walk_forward_validation": {
                "status": "SUCCESS",
                "duration": 1200.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
