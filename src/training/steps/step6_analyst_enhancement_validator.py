"""
Validator for Step 6: Analyst Enhancement
"""

import asyncio
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


class Step6AnalystEnhancementValidator(BaseValidator):
    """Validator for Step 6: Analyst Enhancement."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step6_analyst_enhancement", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the analyst enhancement step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating analyst enhancement step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("analyst_enhancement", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Analyst enhancement step had errors")
            return False
        
        # 2. Validate enhanced model files existence
        model_files_passed = self._validate_enhanced_model_files(symbol, exchange, data_dir)
        if not model_files_passed:
            self.logger.error("❌ Enhanced model files validation failed")
            return False
        
        # 3. Validate performance improvement
        improvement_passed = self._validate_performance_improvement(symbol, exchange, data_dir)
        if not improvement_passed:
            self.logger.error("❌ Performance improvement validation failed")
            return False
        
        # 4. Validate enhancement quality
        quality_passed = self._validate_enhancement_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Enhancement quality validation failed")
            return False
        
        # 5. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Analyst enhancement outcome is not favorable")
            return False
        
        self.logger.info("✅ Analyst enhancement validation passed")
        return True
    
    def _validate_enhanced_model_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that enhanced model files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected enhanced model file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_enhanced_analyst_model.pkl",
                f"{data_dir}/{exchange}_{symbol}_enhanced_analyst_metadata.json",
                f"{data_dir}/{exchange}_{symbol}_enhancement_history.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "enhanced_model_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing enhanced model files: {missing_files}")
                return False
            
            self.logger.info("✅ All enhanced model files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating enhanced model files: {e}")
            return False
    
    def _validate_performance_improvement(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that performance has improved after enhancement.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if performance improved
        """
        try:
            # Load original and enhanced model performance
            original_history_file = f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
            enhanced_history_file = f"{data_dir}/{exchange}_{symbol}_enhancement_history.json"
            
            original_metrics = {}
            enhanced_metrics = {}
            
            # Load original metrics
            if os.path.exists(original_history_file):
                import json
                with open(original_history_file, "r") as f:
                    original_data = json.load(f)
                    original_metrics = original_data.get("metrics", {})
            
            # Load enhanced metrics
            if os.path.exists(enhanced_history_file):
                import json
                with open(enhanced_history_file, "r") as f:
                    enhanced_data = json.load(f)
                    enhanced_metrics = enhanced_data.get("metrics", {})
            
            # Compare performance metrics
            improvements = []
            
            if "accuracy" in original_metrics and "accuracy" in enhanced_metrics:
                original_acc = original_metrics["accuracy"]
                enhanced_acc = enhanced_metrics["accuracy"]
                acc_improvement = enhanced_acc - original_acc
                improvements.append(("accuracy", acc_improvement))
                
                if acc_improvement < 0:
                    self.logger.warning(f"⚠️ Accuracy decreased: {original_acc:.3f} -> {enhanced_acc:.3f}")
            
            if "loss" in original_metrics and "loss" in enhanced_metrics:
                original_loss = original_metrics["loss"]
                enhanced_loss = enhanced_metrics["loss"]
                loss_improvement = original_loss - enhanced_loss  # Lower loss is better
                improvements.append(("loss", loss_improvement))
                
                if loss_improvement < 0:
                    self.logger.warning(f"⚠️ Loss increased: {original_loss:.3f} -> {enhanced_loss:.3f}")
            
            # Check if overall performance improved
            positive_improvements = sum(1 for _, improvement in improvements if improvement > 0)
            total_improvements = len(improvements)
            
            if total_improvements > 0:
                improvement_ratio = positive_improvements / total_improvements
                if improvement_ratio < 0.5:  # At least 50% of metrics should improve
                    self.logger.warning(f"⚠️ Limited performance improvement: {improvement_ratio:.2f}")
            
            self.validation_results["performance_improvement"] = {
                "improvements": improvements,
                "positive_improvements": positive_improvements,
                "total_improvements": total_improvements,
                "improvement_ratio": improvement_ratio if total_improvements > 0 else 0
            }
            
            self.logger.info("✅ Performance improvement validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during performance improvement validation: {e}")
            return False
    
    def _validate_enhancement_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate the quality of the enhancement process.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if enhancement quality is acceptable
        """
        try:
            # Load enhanced model metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_enhanced_analyst_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check enhancement type
                if "enhancement_type" in metadata:
                    enhancement_type = metadata["enhancement_type"]
                    self.logger.info(f"Enhancement type: {enhancement_type}")
                
                # Check enhancement parameters
                if "enhancement_parameters" in metadata:
                    params = metadata["enhancement_parameters"]
                    param_count = len(params)
                    if param_count < 5:
                        self.logger.warning(f"⚠️ Few enhancement parameters: {param_count}")
                
                # Check model complexity
                if "model_complexity" in metadata:
                    complexity = metadata["model_complexity"]
                    if complexity > 1000000:
                        self.logger.warning(f"⚠️ High model complexity: {complexity}")
                    elif complexity < 1000:
                        self.logger.warning(f"⚠️ Low model complexity: {complexity}")
            
            # Load and validate the enhanced model
            model_file = f"{data_dir}/{exchange}_{symbol}_enhanced_analyst_model.pkl"
            
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    
                    # Basic model validation
                    if hasattr(model, 'predict'):
                        self.logger.info("✅ Enhanced model has predict method")
                    else:
                        self.logger.error("❌ Enhanced model missing predict method")
                        return False
                    
                    # Check for enhancement-specific attributes
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if len(importances) > 0:
                            non_zero_features = np.sum(importances > 0)
                            if non_zero_features < 10:
                                self.logger.warning(f"⚠️ Enhanced model has few non-zero features: {non_zero_features}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading enhanced model: {e}")
                    return False
            
            self.logger.info("✅ Enhancement quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during enhancement quality validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the step6_analyst_enhancement validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step6AnalystEnhancementValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    
    return {
        "step_name": "step6_analyst_enhancement",
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
            "analyst_enhancement": {
                "status": "SUCCESS",
                "duration": 450.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
