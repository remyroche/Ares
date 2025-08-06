"""
Validator for Step 2: Market Regime Classification
"""

import os
import sys
import json
import pandas as pd
from typing import Any, Dict, List
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step2MarketRegimeClassificationValidator(BaseValidator):
    """Validator for Step 2: Market Regime Classification."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step2_market_regime_classification", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the market regime classification step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating market regime classification step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("regime_classification", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Market regime classification step had errors")
            return False
        
        # 2. Validate regime classification file existence
        regime_file_path = f"{data_dir}/{exchange}_{symbol}_regime_classification.json"
        file_passed, file_metrics = self.validate_file_exists(regime_file_path, "regime_classification")
        self.validation_results["file_existence"] = file_metrics
        
        if not file_passed:
            self.logger.error(f"❌ Regime classification file not found: {regime_file_path}")
            return False
        
        # 3. Validate regime classification results
        try:
            with open(regime_file_path, "r") as f:
                regime_results = json.load(f)
            
            regime_passed, regime_metrics = self._validate_regime_classification(regime_results)
            self.validation_results["regime_classification"] = regime_metrics
            
            if not regime_passed:
                self.logger.error("❌ Regime classification validation failed")
                return False
            
            # 4. Validate regime distribution
            distribution_passed = self._validate_regime_distribution(regime_results)
            if not distribution_passed:
                self.logger.error("❌ Regime distribution validation failed")
                return False
            
            # 5. Validate regime transitions
            transitions_passed = self._validate_regime_transitions(regime_results)
            if not transitions_passed:
                self.logger.error("❌ Regime transitions validation failed")
                return False
            
            # 6. Validate outcome favorability
            outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
            self.validation_results["outcome_favorability"] = outcome_metrics
            
            if not outcome_passed:
                self.logger.warning("⚠️ Market regime classification outcome is not favorable")
                return False
            
            self.logger.info("✅ Market regime classification validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during regime classification validation: {e}")
            return False
    
    def _validate_regime_classification(self, regime_results: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """
        Validate regime classification results.
        
        Args:
            regime_results: Regime classification results
            
        Returns:
            Tuple of (passed, metrics_dict)
        """
        try:
            # Check if results contain required keys
            required_keys = ["regimes", "probabilities", "metadata"]
            missing_keys = [key for key in required_keys if key not in regime_results]
            
            if missing_keys:
                return False, {"error": f"Missing required keys: {missing_keys}"}
            
            # Validate regimes array
            regimes = regime_results.get("regimes", [])
            if not isinstance(regimes, list) or len(regimes) == 0:
                return False, {"error": "Regimes array is empty or invalid"}
            
            # Validate probabilities
            probabilities = regime_results.get("probabilities", [])
            if not isinstance(probabilities, list) or len(probabilities) == 0:
                return False, {"error": "Probabilities array is empty or invalid"}
            
            # Check if regimes and probabilities have same length
            if len(regimes) != len(probabilities):
                return False, {"error": "Regimes and probabilities arrays have different lengths"}
            
            # Validate probability values (should sum to 1 for each time point)
            valid_probabilities = True
            for i, prob_set in enumerate(probabilities):
                if isinstance(prob_set, list):
                    prob_sum = sum(prob_set)
                    if abs(prob_sum - 1.0) > 0.01:  # Allow small numerical errors
                        valid_probabilities = False
                        break
            
            if not valid_probabilities:
                return False, {"error": "Probability distributions do not sum to 1"}
            
            # Check for reasonable regime values
            valid_regimes = all(isinstance(r, (int, str)) for r in regimes)
            if not valid_regimes:
                return False, {"error": "Invalid regime values found"}
            
            # Calculate metrics
            unique_regimes = set(regimes)
            regime_count = len(unique_regimes)
            
            metrics_dict = {
                "total_regimes": len(regimes),
                "unique_regimes": regime_count,
                "regime_types": list(unique_regimes),
                "valid_probabilities": valid_probabilities,
                "passed": True
            }
            
            return True, metrics_dict
            
        except Exception as e:
            return False, {"error": f"Error validating regime classification: {str(e)}"}
    
    def _validate_regime_distribution(self, regime_results: Dict[str, Any]) -> bool:
        """
        Validate regime distribution characteristics.
        
        Args:
            regime_results: Regime classification results
            
        Returns:
            bool: True if distribution is valid
        """
        try:
            regimes = regime_results.get("regimes", [])
            if not regimes:
                return False
            
            # Count regime frequencies
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total_regimes = len(regimes)
            
            # Check for balanced distribution (no single regime should dominate)
            max_regime_count = max(regime_counts.values())
            max_regime_ratio = max_regime_count / total_regimes
            
            if max_regime_ratio > 0.8:  # No regime should be more than 80%
                self.logger.warning(f"⚠️ Regime distribution is imbalanced: {max_regime_ratio:.3f}")
                return False
            
            # Check for minimum regime diversity
            if len(regime_counts) < 2:
                self.logger.warning("⚠️ Only one regime detected - insufficient diversity")
                return False
            
            # Check for reasonable regime frequencies
            min_regime_ratio = min(regime_counts.values()) / total_regimes
            if min_regime_ratio < 0.05:  # Each regime should be at least 5%
                self.logger.warning(f"⚠️ Some regimes have very low frequency: {min_regime_ratio:.3f}")
            
            self.logger.info(f"✅ Regime distribution validation passed: {len(regime_counts)} regimes")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during regime distribution validation: {e}")
            return False
    
    def _validate_regime_transitions(self, regime_results: Dict[str, Any]) -> bool:
        """
        Validate regime transition characteristics.
        
        Args:
            regime_results: Regime classification results
            
        Returns:
            bool: True if transitions are valid
        """
        try:
            regimes = regime_results.get("regimes", [])
            if len(regimes) < 2:
                return True  # Single regime, no transitions to validate
            
            # Count transitions
            transitions = {}
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                transition_key = f"{current_regime}->{next_regime}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
            
            # Check for reasonable transition patterns
            total_transitions = len(regimes) - 1
            
            # Check for excessive regime switching (more than 50% of time points)
            regime_changes = sum(1 for i in range(len(regimes) - 1) if regimes[i] != regimes[i + 1])
            change_ratio = regime_changes / total_transitions
            
            if change_ratio > 0.5:
                self.logger.warning(f"⚠️ High regime switching frequency: {change_ratio:.3f}")
            
            # Check for stuck regimes (no changes for long periods)
            max_consecutive_same = 1
            current_consecutive = 1
            for i in range(1, len(regimes)):
                if regimes[i] == regimes[i - 1]:
                    current_consecutive += 1
                    max_consecutive_same = max(max_consecutive_same, current_consecutive)
                else:
                    current_consecutive = 1
            
            # Check if any regime is stuck for too long (more than 30% of data)
            max_stuck_ratio = max_consecutive_same / len(regimes)
            if max_stuck_ratio > 0.3:
                self.logger.warning(f"⚠️ Regime stuck for too long: {max_stuck_ratio:.3f}")
            
            self.logger.info(f"✅ Regime transitions validation passed: {len(transitions)} transition types")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during regime transitions validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 2 Market Regime Classification validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step2MarketRegimeClassificationValidator(CONFIG)
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
            "regime_classification": {
                "status": "SUCCESS",
                "duration": 45.2
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
