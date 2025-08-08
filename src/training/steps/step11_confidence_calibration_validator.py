"""
Validator for Step 11: Confidence Calibration
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


class Step11ConfidenceCalibrationValidator(BaseValidator):
    """Validator for Step 11: Confidence Calibration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step11_confidence_calibration", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the confidence calibration step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating confidence calibration step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("confidence_calibration", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Confidence calibration step had errors")
            return False
        
        # 2. Validate calibration files existence
        calibration_files_passed = self._validate_calibration_files(symbol, exchange, data_dir)
        if not calibration_files_passed:
            self.logger.error("❌ Calibration files validation failed")
            return False
        
        # 3. Validate calibration quality
        quality_passed = self._validate_calibration_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Calibration quality validation failed")
            return False
        
        # 4. Validate calibration metrics
        metrics_passed = self._validate_calibration_metrics(symbol, exchange, data_dir)
        if not metrics_passed:
            self.logger.error("❌ Calibration metrics validation failed")
            return False
        
        # 5. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Confidence calibration outcome is not favorable")
            return False
        
        self.logger.info("✅ Confidence calibration validation passed")
        return True
    
    def _validate_calibration_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that calibration files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected calibration file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_calibrated_models.pkl",
                f"{data_dir}/{exchange}_{symbol}_calibration_metadata.json",
                f"{data_dir}/{exchange}_{symbol}_calibration_results.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "calibration_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing calibration files: {missing_files}")
                return False
            
            self.logger.info("✅ All calibration files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating calibration files: {e}")
            return False
    
    def _validate_calibration_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate calibration quality metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if calibration quality is acceptable
        """
        try:
            # Load calibration metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_calibration_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check calibration quality metrics
                if "calibration_error" in metadata:
                    cal_error = metadata["calibration_error"]
                    if cal_error > 0.1:  # High calibration error
                        self.logger.warning(f"⚠️ High calibration error: {cal_error:.3f}")
                    elif cal_error < 0.01:  # Very low calibration error
                        self.logger.info(f"✅ Excellent calibration error: {cal_error:.3f}")
                
                # Check reliability diagram quality
                if "reliability_score" in metadata:
                    reliability = metadata["reliability_score"]
                    if reliability < 0.8:
                        self.logger.warning(f"⚠️ Low reliability score: {reliability:.3f}")
                
                # Check calibration curve quality
                if "calibration_curve_quality" in metadata:
                    curve_quality = metadata["calibration_curve_quality"]
                    if curve_quality < 0.7:
                        self.logger.warning(f"⚠️ Poor calibration curve quality: {curve_quality:.3f}")
                
                # Check confidence distribution
                if "confidence_distribution" in metadata:
                    conf_dist = metadata["confidence_distribution"]
                    
                    # Check for reasonable confidence distribution
                    if "mean_confidence" in conf_dist:
                        mean_conf = conf_dist["mean_confidence"]
                        if mean_conf < 0.3 or mean_conf > 0.8:
                            self.logger.warning(f"⚠️ Unusual mean confidence: {mean_conf:.3f}")
                    
                    if "confidence_std" in conf_dist:
                        conf_std = conf_dist["confidence_std"]
                        if conf_std < 0.1:
                            self.logger.warning(f"⚠️ Low confidence variance: {conf_std:.3f}")
                        elif conf_std > 0.4:
                            self.logger.warning(f"⚠️ High confidence variance: {conf_std:.3f}")
            
            self.logger.info("✅ Calibration quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during calibration quality validation: {e}")
            return False
    
    def _validate_calibration_metrics(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate calibration performance metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if calibration metrics are acceptable
        """
        try:
            # Load calibration results
            results_file = f"{data_dir}/{exchange}_{symbol}_calibration_results.json"
            
            if os.path.exists(results_file):
                import json
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                # Check ECE (Expected Calibration Error)
                if "ece" in results:
                    ece = results["ece"]
                    if ece > 0.1:
                        self.logger.warning(f"⚠️ High ECE: {ece:.3f}")
                    elif ece < 0.01:
                        self.logger.info(f"✅ Excellent ECE: {ece:.3f}")
                
                # Check MCE (Maximum Calibration Error)
                if "mce" in results:
                    mce = results["mce"]
                    if mce > 0.2:
                        self.logger.warning(f"⚠️ High MCE: {mce:.3f}")
                
                # Check reliability diagram metrics
                if "reliability_metrics" in results:
                    reliability = results["reliability_metrics"]
                    
                    if "brier_score" in reliability:
                        brier = reliability["brier_score"]
                        if brier > 0.3:
                            self.logger.warning(f"⚠️ High Brier score: {brier:.3f}")
                    
                    if "sharpness" in reliability:
                        sharpness = reliability["sharpness"]
                        if sharpness < 0.1:
                            self.logger.warning(f"⚠️ Low sharpness: {sharpness:.3f}")
                
                # Check calibration by confidence level
                if "confidence_level_metrics" in results:
                    conf_metrics = results["confidence_level_metrics"]
                    
                    for level, metrics in conf_metrics.items():
                        if "accuracy" in metrics and "confidence" in metrics:
                            acc = metrics["accuracy"]
                            conf = metrics["confidence"]
                            
                            # Check if accuracy matches confidence
                            diff = abs(acc - conf)
                            if diff > 0.1:
                                self.logger.warning(f"⚠️ Poor calibration at confidence {level}: acc={acc:.3f}, conf={conf:.3f}")
                
                # Check overall calibration score
                if "overall_calibration_score" in results:
                    overall_score = results["overall_calibration_score"]
                    if overall_score < 0.7:
                        self.logger.warning(f"⚠️ Low overall calibration score: {overall_score:.3f}")
                    elif overall_score > 0.95:
                        self.logger.info(f"✅ Excellent overall calibration score: {overall_score:.3f}")
            
            self.logger.info("✅ Calibration metrics validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during calibration metrics validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the step11_confidence_calibration validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step11ConfidenceCalibrationValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    
    return {
        "step_name": "step11_confidence_calibration",
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
            "confidence_calibration": {
                "status": "SUCCESS",
                "duration": 180.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
