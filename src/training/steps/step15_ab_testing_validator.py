"""
Validator for Step 15: A/B Testing
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


class Step15ABTestingValidator(BaseValidator):
    """Validator for Step 15: A/B Testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step15_ab_testing", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the A/B testing step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating A/B testing step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("ab_testing", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ A/B testing step had errors")
            return False
        
        # 2. Validate A/B testing files existence
        testing_files_passed = self._validate_ab_testing_files(symbol, exchange, data_dir)
        if not testing_files_passed:
            self.logger.error("❌ A/B testing files validation failed")
            return False
        
        # 3. Validate A/B testing statistical significance
        significance_passed = self._validate_ab_statistical_significance(symbol, exchange, data_dir)
        if not significance_passed:
            self.logger.error("❌ A/B testing statistical significance validation failed")
            return False
        
        # 4. Validate A/B testing performance comparison
        comparison_passed = self._validate_ab_performance_comparison(symbol, exchange, data_dir)
        if not comparison_passed:
            self.logger.error("❌ A/B testing performance comparison validation failed")
            return False
        
        # 5. Validate A/B testing sample sizes
        sample_sizes_passed = self._validate_ab_sample_sizes(symbol, exchange, data_dir)
        if not sample_sizes_passed:
            self.logger.error("❌ A/B testing sample sizes validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ A/B testing outcome is not favorable")
            return False
        
        self.logger.info("✅ A/B testing validation passed")
        return True
    
    def _validate_ab_testing_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that A/B testing files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected A/B testing file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_ab_testing_results.json",
                f"{data_dir}/{exchange}_{symbol}_ab_testing_performance.json",
                f"{data_dir}/{exchange}_{symbol}_ab_testing_metadata.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "ab_testing_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing A/B testing files: {missing_files}")
                return False
            
            self.logger.info("✅ All A/B testing files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating A/B testing files: {e}")
            return False
    
    def _validate_ab_statistical_significance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate statistical significance of A/B testing results.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if statistical significance is acceptable
        """
        try:
            # Load A/B testing results
            results_file = f"{data_dir}/{exchange}_{symbol}_ab_testing_results.json"
            
            if os.path.exists(results_file):
                import json
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                # Check p-value for statistical significance
                if "p_value" in results:
                    p_value = results["p_value"]
                    if p_value > 0.05:
                        self.logger.warning(f"⚠️ High p-value (not statistically significant): {p_value:.3f}")
                    elif p_value < 0.001:
                        self.logger.info(f"✅ Very low p-value (highly significant): {p_value:.6f}")
                
                # Check confidence intervals
                if "confidence_intervals" in results:
                    ci = results["confidence_intervals"]
                    
                    if "95_percent_ci" in ci:
                        ci_95 = ci["95_percent_ci"]
                        ci_width = ci_95[1] - ci_95[0]
                        if ci_width > 0.2:
                            self.logger.warning(f"⚠️ Wide 95% confidence interval: {ci_width:.3f}")
                    
                    if "99_percent_ci" in ci:
                        ci_99 = ci["99_percent_ci"]
                        ci_width = ci_99[1] - ci_99[0]
                        if ci_width > 0.3:
                            self.logger.warning(f"⚠️ Wide 99% confidence interval: {ci_width:.3f}")
                
                # Check effect size
                if "effect_size" in results:
                    effect_size = results["effect_size"]
                    if abs(effect_size) < 0.1:
                        self.logger.warning(f"⚠️ Small effect size: {effect_size:.3f}")
                    elif abs(effect_size) > 0.8:
                        self.logger.info(f"✅ Large effect size: {effect_size:.3f}")
                
                # Check power analysis
                if "power" in results:
                    power = results["power"]
                    if power < 0.8:
                        self.logger.warning(f"⚠️ Low statistical power: {power:.3f}")
                
                # Check significance level
                if "significance_level" in results:
                    sig_level = results["significance_level"]
                    if sig_level > 0.1:
                        self.logger.warning(f"⚠️ High significance level: {sig_level:.3f}")
            
            self.logger.info("✅ A/B testing statistical significance validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during A/B testing statistical significance validation: {e}")
            return False
    
    def _validate_ab_performance_comparison(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate A/B testing performance comparison.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if performance comparison is acceptable
        """
        try:
            # Load A/B testing performance results
            performance_file = f"{data_dir}/{exchange}_{symbol}_ab_testing_performance.json"
            
            if os.path.exists(performance_file):
                import json
                with open(performance_file, "r") as f:
                    performance = json.load(f)
                
                # Check group A performance
                if "group_a_performance" in performance:
                    group_a = performance["group_a_performance"]
                    
                    if "accuracy" in group_a:
                        acc_a = group_a["accuracy"]
                        if acc_a < 0.5:
                            self.logger.warning(f"⚠️ Low group A accuracy: {acc_a:.3f}")
                    
                    if "sample_size" in group_a:
                        size_a = group_a["sample_size"]
                        if size_a < 100:
                            self.logger.warning(f"⚠️ Small group A sample size: {size_a}")
                
                # Check group B performance
                if "group_b_performance" in performance:
                    group_b = performance["group_b_performance"]
                    
                    if "accuracy" in group_b:
                        acc_b = group_b["accuracy"]
                        if acc_b < 0.5:
                            self.logger.warning(f"⚠️ Low group B accuracy: {acc_b:.3f}")
                    
                    if "sample_size" in group_b:
                        size_b = group_b["sample_size"]
                        if size_b < 100:
                            self.logger.warning(f"⚠️ Small group B sample size: {size_b}")
                
                # Check performance difference
                if "performance_difference" in performance:
                    diff = performance["performance_difference"]
                    
                    if abs(diff) < 0.01:
                        self.logger.warning(f"⚠️ Minimal performance difference: {diff:.3f}")
                    elif abs(diff) > 0.3:
                        self.logger.warning(f"⚠️ Large performance difference: {diff:.3f}")
                
                # Check relative improvement
                if "relative_improvement" in performance:
                    improvement = performance["relative_improvement"]
                    
                    if improvement < 0.05:
                        self.logger.warning(f"⚠️ Small relative improvement: {improvement:.3f}")
                    elif improvement > 0.5:
                        self.logger.info(f"✅ Large relative improvement: {improvement:.3f}")
                
                # Check effect direction
                if "effect_direction" in performance:
                    direction = performance["effect_direction"]
                    if direction == "negative":
                        self.logger.warning("⚠️ Negative effect direction detected")
            
            self.logger.info("✅ A/B testing performance comparison validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during A/B testing performance comparison validation: {e}")
            return False
    
    def _validate_ab_sample_sizes(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate A/B testing sample sizes and balance.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if sample sizes are acceptable
        """
        try:
            # Load A/B testing metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_ab_testing_metadata.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check total sample size
                if "total_sample_size" in metadata:
                    total_size = metadata["total_sample_size"]
                    if total_size < 200:
                        self.logger.warning(f"⚠️ Small total A/B testing sample size: {total_size}")
                    elif total_size > 100000:
                        self.logger.warning(f"⚠️ Large total A/B testing sample size: {total_size}")
                
                # Check group balance
                if "group_balance" in metadata:
                    balance = metadata["group_balance"]
                    if balance < 0.4 or balance > 0.6:
                        self.logger.warning(f"⚠️ Imbalanced A/B testing groups: {balance:.3f}")
                
                # Check minimum detectable effect
                if "minimum_detectable_effect" in metadata:
                    mde = metadata["minimum_detectable_effect"]
                    if mde > 0.2:
                        self.logger.warning(f"⚠️ High minimum detectable effect: {mde:.3f}")
                
                # Check test duration
                if "test_duration_days" in metadata:
                    duration = metadata["test_duration_days"]
                    if duration < 7:
                        self.logger.warning(f"⚠️ Short A/B test duration: {duration} days")
                    elif duration > 90:
                        self.logger.warning(f"⚠️ Long A/B test duration: {duration} days")
                
                # Check randomization quality
                if "randomization_quality" in metadata:
                    rand_quality = metadata["randomization_quality"]
                    if rand_quality < 0.8:
                        self.logger.warning(f"⚠️ Poor randomization quality: {rand_quality:.3f}")
            
            self.logger.info("✅ A/B testing sample sizes validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during A/B testing sample sizes validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 15 A/B Testing validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step15ABTestingValidator(CONFIG)
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
            "ab_testing": {
                "status": "SUCCESS",
                "duration": 800.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
