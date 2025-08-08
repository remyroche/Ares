"""
Validator for Step 10: Tactician Ensemble Creation
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


class Step10TacticianEnsembleCreationValidator(BaseValidator):
    """Validator for Step 10: Tactician Ensemble Creation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step10_tactician_ensemble_creation", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the tactician ensemble creation step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating tactician ensemble creation step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("tactician_ensemble_creation", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Tactician ensemble creation step had errors")
            return False
        
        # 2. Validate tactician ensemble files existence
        ensemble_files_passed = self._validate_tactician_ensemble_files(symbol, exchange, data_dir)
        if not ensemble_files_passed:
            self.logger.error("❌ Tactician ensemble files validation failed")
            return False
        
        # 3. Validate tactician ensemble diversity
        diversity_passed = self._validate_tactician_ensemble_diversity(symbol, exchange, data_dir)
        if not diversity_passed:
            self.logger.error("❌ Tactician ensemble diversity validation failed")
            return False
        
        # 4. Validate tactician ensemble performance
        performance_passed = self._validate_tactician_ensemble_performance(symbol, exchange, data_dir)
        if not performance_passed:
            self.logger.error("❌ Tactician ensemble performance validation failed")
            return False
        
        # 5. Validate tactician ensemble stability
        stability_passed = self._validate_tactician_ensemble_stability(symbol, exchange, data_dir)
        if not stability_passed:
            self.logger.error("❌ Tactician ensemble stability validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Tactician ensemble creation outcome is not favorable")
            return False
        
        self.logger.info("✅ Tactician ensemble creation validation passed")
        return True
    
    def _validate_tactician_ensemble_files(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that tactician ensemble files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if files exist
        """
        try:
            # Expected tactician ensemble file patterns
            ensemble_dir = f"{data_dir}/tactician_ensembles"
            expected_files = [
                f"{ensemble_dir}/{exchange}_{symbol}_tactician_ensemble.pkl",
                f"{ensemble_dir}/{exchange}_{symbol}_tactician_ensemble_summary.json",
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "tactician_ensemble_files")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing tactician ensemble files: {missing_files}")
                return False
            
            self.logger.info("✅ All tactician ensemble files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating tactician ensemble files: {e}")
            return False
    
    def _validate_tactician_ensemble_diversity(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate tactician ensemble model diversity.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if ensemble is diverse
        """
        try:
            # Load tactician ensemble metadata
            metadata_file = f"{data_dir}/tactician_ensembles/{exchange}_{symbol}_tactician_ensemble_summary.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check number of models in ensemble
                if "model_count" in metadata:
                    model_count = metadata["model_count"]
                    if model_count < 3:
                        self.logger.warning(f"⚠️ Few tactician models in ensemble: {model_count}")
                    elif model_count > 20:
                        self.logger.warning(f"⚠️ Many tactician models in ensemble: {model_count}")
                
                # Check model types diversity
                if "model_types" in metadata:
                    model_types = metadata["model_types"]
                    unique_types = set(model_types)
                    if len(unique_types) < 2:
                        self.logger.warning(f"⚠️ Low tactician model type diversity: {len(unique_types)} types")
                
                # Check ensemble weights
                weights_file = f"{data_dir}/tactician_ensembles/{exchange}_{symbol}_tactician_ensemble_weights.json"
                if os.path.exists(weights_file):
                    with open(weights_file, "r") as f:
                        weights_data = json.load(f)
                    
                    weights = weights_data.get("weights", [])
                    if weights:
                        # Check weight distribution
                        weight_std = np.std(weights)
                        
                        if weight_std < 0.01:  # Very uniform weights
                            self.logger.warning("⚠️ Very uniform tactician ensemble weights")
                        
                        # Check for extreme weights
                        max_weight = max(weights)
                        min_weight = min(weights)
                        
                        if max_weight > 0.8:  # Single model dominates
                            self.logger.warning(f"⚠️ Single tactician model dominates ensemble: {max_weight:.3f}")
                        
                        if min_weight < 0.01:  # Very small weights
                            self.logger.warning(f"⚠️ Very small tactician ensemble weights: {min_weight:.3f}")
            
            self.logger.info("✅ Tactician ensemble diversity validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during tactician ensemble diversity validation: {e}")
            return False
    
    def _validate_tactician_ensemble_performance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate tactician ensemble performance metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if performance is acceptable
        """
        try:
            # Load tactician ensemble metadata for performance metrics
            metadata_file = f"{data_dir}/tactician_ensembles/{exchange}_{symbol}_tactician_ensemble_summary.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check ensemble performance metrics
                if "ensemble_accuracy" in metadata:
                    ensemble_acc = metadata["ensemble_accuracy"]
                    acc_passed, acc_metrics = self.validate_model_performance(
                        ensemble_acc, 0.0, "tactician_ensemble_model"
                    )
                    self.validation_results["tactician_ensemble_accuracy"] = acc_metrics
                    
                    if not acc_passed:
                        self.logger.error(f"❌ Tactician ensemble accuracy too low: {ensemble_acc:.3f}")
                        return False
                
                if "ensemble_loss" in metadata:
                    ensemble_loss = metadata["ensemble_loss"]
                    loss_passed, loss_metrics = self.validate_model_performance(
                        0.0, ensemble_loss, "tactician_ensemble_model"
                    )
                    self.validation_results["tactician_ensemble_loss"] = loss_metrics
                    
                    if not loss_passed:
                        self.logger.error(f"❌ Tactician ensemble loss too high: {ensemble_loss:.3f}")
                        return False
                
                # Check signal prediction performance
                if "signal_prediction_accuracy" in metadata:
                    signal_acc = metadata["signal_prediction_accuracy"]
                    if signal_acc < 0.6:
                        self.logger.warning(f"⚠️ Low tactician ensemble signal prediction accuracy: {signal_acc:.3f}")
                
                # Check individual model performance
                if "individual_model_performance" in metadata:
                    individual_perf = metadata["individual_model_performance"]
                    
                    # Check if all models have reasonable performance
                    poor_models = 0
                    for model_perf in individual_perf:
                        if model_perf.get("accuracy", 0) < 0.5:
                            poor_models += 1
                    
                    if poor_models > len(individual_perf) * 0.5:  # More than 50% poor models
                        self.logger.warning(f"⚠️ Many poor performing tactician models: {poor_models}/{len(individual_perf)}")
                
                # Check ensemble vs individual performance
                if "ensemble_accuracy" in metadata and "individual_model_performance" in metadata:
                    ensemble_acc = metadata["ensemble_accuracy"]
                    individual_accs = [p.get("accuracy", 0) for p in metadata["individual_model_performance"]]
                    
                    if individual_accs:
                        avg_individual_acc = np.mean(individual_accs)
                        ensemble_improvement = ensemble_acc - avg_individual_acc
                        
                        if ensemble_improvement < 0:
                            self.logger.warning(f"⚠️ Tactician ensemble performs worse than average individual model")
                        elif ensemble_improvement < 0.01:
                            self.logger.warning(f"⚠️ Minimal tactician ensemble improvement: {ensemble_improvement:.3f}")
            
            self.logger.info("✅ Tactician ensemble performance validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during tactician ensemble performance validation: {e}")
            return False
    
    def _validate_tactician_ensemble_stability(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate tactician ensemble stability and robustness.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if ensemble is stable
        """
        try:
            # Load tactician ensemble metadata
            metadata_file = f"{data_dir}/tactician_ensembles/{exchange}_{symbol}_tactician_ensemble_summary.json"
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check ensemble stability metrics
                if "stability_score" in metadata:
                    stability_score = metadata["stability_score"]
                    if stability_score < 0.7:
                        self.logger.warning(f"⚠️ Low tactician ensemble stability: {stability_score:.3f}")
                
                # Check prediction variance
                if "prediction_variance" in metadata:
                    pred_variance = metadata["prediction_variance"]
                    if pred_variance > 0.1:
                        self.logger.warning(f"⚠️ High tactician ensemble prediction variance: {pred_variance:.3f}")
                
                # Check model correlation
                if "model_correlation" in metadata:
                    model_corr = metadata["model_correlation"]
                    if model_corr > 0.9:
                        self.logger.warning(f"⚠️ High tactician model correlation: {model_corr:.3f}")
                
                # Check ensemble robustness
                if "robustness_score" in metadata:
                    robustness_score = metadata["robustness_score"]
                    if robustness_score < 0.6:
                        self.logger.warning(f"⚠️ Low tactician ensemble robustness: {robustness_score:.3f}")
            
            # Load and validate the tactician ensemble model
            ensemble_file = f"{data_dir}/tactician_ensembles/{exchange}_{symbol}_tactician_ensemble.pkl"
            
            if os.path.exists(ensemble_file):
                try:
                    with open(ensemble_file, "rb") as f:
                        ensemble = pickle.load(f)
                    
                    # Basic ensemble validation
                    if hasattr(ensemble, 'predict'):
                        self.logger.info("✅ Tactician ensemble has predict method")
                    else:
                        self.logger.error("❌ Tactician ensemble missing predict method")
                        return False
                    
                    # Check for ensemble-specific attributes
                    if hasattr(ensemble, 'estimators_'):
                        estimator_count = len(ensemble.estimators_)
                        if estimator_count < 3:
                            self.logger.warning(f"⚠️ Few tactician estimators in ensemble: {estimator_count}")
                    
                    if hasattr(ensemble, 'weights_'):
                        weight_count = len(ensemble.weights_)
                        if weight_count < 3:
                            self.logger.warning(f"⚠️ Few tactician weights in ensemble: {weight_count}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading tactician ensemble: {e}")
                    return False
            
            self.logger.info("✅ Tactician ensemble stability validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during tactician ensemble stability validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the step10_tactician_ensemble_creation validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step10TacticianEnsembleCreationValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    
    return {
        "step_name": "step10_tactician_ensemble_creation",
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
            "tactician_ensemble_creation": {
                "status": "SUCCESS",
                "duration": 700.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
