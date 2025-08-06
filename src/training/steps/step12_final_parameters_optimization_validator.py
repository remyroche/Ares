"""
Validator for Step 12: Final Parameters Optimization
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


class Step12FinalParametersOptimizationValidator(BaseValidator):
    """Validator for Step 12: Final Parameters Optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step12_final_parameters_optimization", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the final parameters optimization step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating final parameters optimization step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("final_parameters_optimization", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Final parameters optimization step had errors")
            return False
        
        # 2. Validate optimization files existence
        optimization_files_passed = self._validate_optimization_files(symbol, exchange, data_dir)
        if not optimization_files_passed:
            self.logger.error("❌ Optimization files validation failed")
            return False
        
        # 3. Validate optimization quality
        quality_passed = self._validate_optimization_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Optimization quality validation failed")
            return False
        
        # 4. Validate optimization convergence
        convergence_passed = self._validate_optimization_convergence(symbol, exchange, data_dir)
        if not convergence_passed:
            self.logger.error("❌ Optimization convergence validation failed")
            return False
        
        # 5. Validate optimized parameters
        parameters_passed = self._validate_optimized_parameters(symbol, exchange, data_dir)
        if not parameters_passed:
            self.logger.error("❌ Optimized parameters validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Final parameters optimization outcome is not favorable")
            return False
        
        self.logger.info("✅ Final parameters optimization validation passed")
        return True
    
    def _validate_optimization_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that optimization files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected optimization file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_optimized_parameters.json",
                f"{data_dir}/{exchange}_{symbol}_optimization_history.json",
                f"{data_dir}/{exchange}_{symbol}_optimization_results.json"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "optimization_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing optimization files: {missing_files}")
                return False
            
            self.logger.info("✅ All optimization files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating optimization files: {e}")
            return False
    
    def _validate_optimization_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate optimization quality metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if optimization quality is acceptable
        """
        try:
            # Load optimization results
            results_file = f"{data_dir}/{exchange}_{symbol}_optimization_results.json"
            
            if os.path.exists(results_file):
                import json
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                # Check optimization objective value
                if "best_objective_value" in results:
                    best_obj = results["best_objective_value"]
                    if best_obj < 0.5:  # Assuming higher is better
                        self.logger.warning(f"⚠️ Low optimization objective value: {best_obj:.3f}")
                
                # Check optimization improvement
                if "improvement" in results:
                    improvement = results["improvement"]
                    if improvement < 0.01:
                        self.logger.warning(f"⚠️ Minimal optimization improvement: {improvement:.3f}")
                
                # Check parameter stability
                if "parameter_stability" in results:
                    stability = results["parameter_stability"]
                    if stability < 0.7:
                        self.logger.warning(f"⚠️ Low parameter stability: {stability:.3f}")
                
                # Check optimization efficiency
                if "optimization_efficiency" in results:
                    efficiency = results["optimization_efficiency"]
                    if efficiency < 0.6:
                        self.logger.warning(f"⚠️ Low optimization efficiency: {efficiency:.3f}")
            
            self.logger.info("✅ Optimization quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during optimization quality validation: {e}")
            return False
    
    def _validate_optimization_convergence(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate optimization convergence.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if optimization converged properly
        """
        try:
            # Load optimization history
            history_file = f"{data_dir}/{exchange}_{symbol}_optimization_history.json"
            
            if os.path.exists(history_file):
                import json
                with open(history_file, "r") as f:
                    history = json.load(f)
                
                # Check number of iterations
                if "iterations" in history:
                    iterations = history["iterations"]
                    if iterations < 10:
                        self.logger.warning(f"⚠️ Few optimization iterations: {iterations}")
                    elif iterations > 1000:
                        self.logger.warning(f"⚠️ Many optimization iterations: {iterations}")
                
                # Check convergence status
                if "converged" in history:
                    converged = history["converged"]
                    if not converged:
                        self.logger.warning("⚠️ Optimization did not converge")
                
                # Check convergence criteria
                if "convergence_criteria" in history:
                    criteria = history["convergence_criteria"]
                    
                    if "objective_tolerance" in criteria:
                        obj_tol = criteria["objective_tolerance"]
                        if obj_tol > 0.1:
                            self.logger.warning(f"⚠️ High objective tolerance: {obj_tol:.3f}")
                    
                    if "parameter_tolerance" in criteria:
                        param_tol = criteria["parameter_tolerance"]
                        if param_tol > 0.1:
                            self.logger.warning(f"⚠️ High parameter tolerance: {param_tol:.3f}")
                
                # Check optimization progress
                if "progress" in history:
                    progress = history["progress"]
                    
                    if "final_improvement" in progress:
                        final_improvement = progress["final_improvement"]
                        if final_improvement < 0.001:
                            self.logger.warning(f"⚠️ Minimal final improvement: {final_improvement:.6f}")
                    
                    if "stagnation_iterations" in progress:
                        stagnation = progress["stagnation_iterations"]
                        if stagnation > 50:
                            self.logger.warning(f"⚠️ Long stagnation period: {stagnation} iterations")
            
            self.logger.info("✅ Optimization convergence validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during optimization convergence validation: {e}")
            return False
    
    def _validate_optimized_parameters(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate optimized parameters quality.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if parameters are valid
        """
        try:
            # Load optimized parameters
            params_file = f"{data_dir}/{exchange}_{symbol}_optimized_parameters.json"
            
            if os.path.exists(params_file):
                import json
                with open(params_file, "r") as f:
                    params = json.load(f)
                
                # Check parameter count
                param_count = len(params)
                if param_count < 5:
                    self.logger.warning(f"⚠️ Few optimized parameters: {param_count}")
                elif param_count > 100:
                    self.logger.warning(f"⚠️ Many optimized parameters: {param_count}")
                
                # Check parameter ranges
                for param_name, param_value in params.items():
                    if isinstance(param_value, (int, float)):
                        # Check for extreme values
                        if abs(param_value) > 1000:
                            self.logger.warning(f"⚠️ Extreme parameter value for {param_name}: {param_value}")
                        
                        # Check for zero values (might indicate issues)
                        if param_value == 0:
                            self.logger.warning(f"⚠️ Zero parameter value for {param_name}")
                        
                        # Check for negative values (if not expected)
                        if param_value < 0 and "threshold" not in param_name.lower():
                            self.logger.warning(f"⚠️ Negative parameter value for {param_name}: {param_value}")
                
                # Check parameter consistency
                if "parameter_consistency_score" in params:
                    consistency = params["parameter_consistency_score"]
                    if consistency < 0.7:
                        self.logger.warning(f"⚠️ Low parameter consistency: {consistency:.3f}")
                
                # Check parameter sensitivity
                if "parameter_sensitivity" in params:
                    sensitivity = params["parameter_sensitivity"]
                    high_sensitivity_params = [p for p, s in sensitivity.items() if s > 0.5]
                    if len(high_sensitivity_params) > 5:
                        self.logger.warning(f"⚠️ Many high sensitivity parameters: {len(high_sensitivity_params)}")
            
            self.logger.info("✅ Optimized parameters validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during optimized parameters validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 12 Final Parameters Optimization validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step12FinalParametersOptimizationValidator(CONFIG)
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
            "final_parameters_optimization": {
                "status": "SUCCESS",
                "duration": 900.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
