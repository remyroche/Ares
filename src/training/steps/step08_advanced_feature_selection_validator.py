# src/training/steps/step08_advanced_feature_selection_validator.py

"""Validator for Step 8: Advanced Feature Selection."""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.base_validator import BaseValidator


class Step08AdvancedFeatureSelectionValidator(BaseValidator):
    """Validator for Step 8: Advanced Feature Selection."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step08_advanced_feature_selection", config)
        
        # Validation thresholds
        self.min_phase1_features = config.get("validation", {}).get("min_phase1_features", 100)
        self.max_phase1_features = config.get("validation", {}).get("max_phase1_features", 200)
        self.min_consensus_ratio = config.get("validation", {}).get("min_consensus_ratio", 0.3)
        self.min_ts_validation_score = config.get("validation", {}).get("min_ts_validation_score", 0.5)
        self.min_boruta_confirmed_ratio = config.get("validation", {}).get("min_boruta_confirmed_ratio", 0.5)

    async def validate(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> bool:
        """Validate the advanced feature selection step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            True if validation passes, False otherwise

        """
        self.logger.info("🔍 Starting Step 8 Advanced Feature Selection validation...")

        # Extract parameters
        symbol = training_input.get("symbol", "UNKNOWN")
        exchange = training_input.get("exchange", "UNKNOWN") 
        timeframe = training_input.get("timeframe", "1m")

        # Check if step 8 was executed
        if "step08_advanced_feature_selection" not in pipeline_state:
            self.logger.error("❌ Step 8 not found in pipeline state")
            self.validation_results["step_executed"] = False
            return False

        step8_state = pipeline_state["step08_advanced_feature_selection"]

        # Check if step completed successfully
        if step8_state.get("status") != "completed":
            self.logger.error(f"❌ Step 8 status is not 'completed': {step8_state.get('status')}")
            self.validation_results["step_completed"] = False
            return False

        self.validation_results["step_completed"] = True

        # Validate input data exists
        input_validation = await self._validate_input_data(symbol, exchange, timeframe)
        if not input_validation["passed"]:
            return False

        # Validate phase 1 results
        phase1_validation = await self._validate_phase1_results(step8_state)
        if not phase1_validation["passed"]:
            return False

        # Validate phase 2 results
        phase2_validation = await self._validate_phase2_results(step8_state)
        if not phase2_validation["passed"]:
            return False

        # Validate output files
        output_validation = await self._validate_output_files(step8_state)
        if not output_validation["passed"]:
            return False

        # Validate feature quality
        quality_validation = await self._validate_feature_quality(step8_state)
        if not quality_validation["passed"]:
            return False

        # All validations passed
        self.logger.info("✅ Step 8 Advanced Feature Selection validation passed")
        return True

    async def _validate_input_data(
        self, symbol: str, exchange: str, timeframe: str
    ) -> dict[str, Any]:
        """Validate that input data exists."""
        validation_result = {
            "passed": True,
            "filtered_features_exist": False,
            "original_features_exist": False,
            "regime_data_exists": False,
        }

        # Check for filtered features from step 7
        filtered_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet"
        filtered_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet"

        if os.path.exists(filtered_train_path) and os.path.exists(filtered_val_path):
            validation_result["filtered_features_exist"] = True
            self.logger.info("✅ Filtered features from Step 7 found")
        else:
            # Check for original features as fallback
            original_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet"
            original_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet"
            
            if os.path.exists(original_train_path) and os.path.exists(original_val_path):
                validation_result["original_features_exist"] = True
                self.logger.warning("⚠️ Using original features (filtered not found)")
            else:
                self.logger.error("❌ No input features found")
                validation_result["passed"] = False

        # Check for regime data (optional)
        hmm_path = f"data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet"
        if os.path.exists(hmm_path):
            validation_result["regime_data_exists"] = True
            self.logger.info("✅ Regime data found")
        else:
            self.logger.warning("⚠️ No regime data found - regime-aware selection will be skipped")

        self.validation_results["input_validation"] = validation_result
        return validation_result

    async def _validate_phase1_results(
        self, step8_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate Phase 1 (mRMR/RF) results."""
        validation_result = {
            "passed": True,
            "phase1_features_count": 0,
            "consensus_ratio": 0,
            "has_metadata": False,
        }

        # Check phase 1 metadata
        phase1_metadata = step8_state.get("phase1_metadata", {})
        if not phase1_metadata:
            self.logger.error("❌ No Phase 1 metadata found")
            validation_result["passed"] = False
            return validation_result

        validation_result["has_metadata"] = True

        # Check feature count
        phase1_features = step8_state.get("phase1_features", 0)
        validation_result["phase1_features_count"] = phase1_features

        if phase1_features < self.min_phase1_features:
            self.logger.error(f"❌ Too few Phase 1 features: {phase1_features} < {self.min_phase1_features}")
            validation_result["passed"] = False
        elif phase1_features > self.max_phase1_features:
            self.logger.error(f"❌ Too many Phase 1 features: {phase1_features} > {self.max_phase1_features}")
            validation_result["passed"] = False
        else:
            self.logger.info(f"✅ Phase 1 features: {phase1_features}")

        # Check consensus ratio
        consensus_ratio = phase1_metadata.get("consensus_ratio", 0)
        validation_result["consensus_ratio"] = consensus_ratio

        if consensus_ratio < self.min_consensus_ratio:
            self.logger.warning(f"⚠️ Low consensus ratio: {consensus_ratio:.2%} < {self.min_consensus_ratio:.2%}")
            # Not a failure, just a warning

        self.validation_results["phase1_validation"] = validation_result
        return validation_result

    async def _validate_phase2_results(
        self, step8_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate Phase 2 (Boruta) results."""
        validation_result = {
            "passed": True,
            "feature_sets_created": [],
            "validation_scores": {},
            "boruta_confirmed_ratios": {},
        }

        # Check phase 2 results
        phase2_results = step8_state.get("phase2_results", {})
        if not phase2_results:
            self.logger.error("❌ No Phase 2 results found")
            validation_result["passed"] = False
            return validation_result

        # Expected feature sets
        expected_sets = {60, 80, 100}
        feature_sets = step8_state.get("phase2_feature_sets", {})

        for expected_size in expected_sets:
            key = f"top_{expected_size}"
            if key in feature_sets:
                actual_size = feature_sets[key]
                validation_result["feature_sets_created"].append(expected_size)
                
                # Check if size matches
                if actual_size != expected_size:
                    self.logger.warning(f"⚠️ Feature set size mismatch: expected {expected_size}, got {actual_size}")

                # Check validation scores
                if expected_size in phase2_results:
                    ts_validation = phase2_results[expected_size].get("ts_validation", {})
                    mean_score = ts_validation.get("mean_score", 0)
                    validation_result["validation_scores"][expected_size] = mean_score

                    if mean_score < self.min_ts_validation_score:
                        self.logger.warning(
                            f"⚠️ Low validation score for {expected_size} features: {mean_score:.4f}"
                        )

                    # Check Boruta confirmed ratio
                    boruta_ratio = phase2_results[expected_size].get("boruta_confirmed_ratio", 0)
                    validation_result["boruta_confirmed_ratios"][expected_size] = boruta_ratio

                    if boruta_ratio < self.min_boruta_confirmed_ratio:
                        self.logger.warning(
                            f"⚠️ Low Boruta confirmed ratio for {expected_size} features: {boruta_ratio:.2%}"
                        )
            else:
                self.logger.error(f"❌ Missing feature set: {expected_size}")
                validation_result["passed"] = False

        self.validation_results["phase2_validation"] = validation_result
        return validation_result

    async def _validate_output_files(
        self, step8_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that all expected output files exist."""
        validation_result = {
            "passed": True,
            "missing_files": [],
            "existing_files": [],
        }

        output_files = step8_state.get("output_files", {})
        
        # Expected output files
        expected_files = [
            "phase1_results",
            "top60_features", "top80_features", "top100_features",
            "top60_train", "top60_val",
            "top80_train", "top80_val", 
            "top100_train", "top100_val",
            "interpretability_report",
            "selection_report"
        ]

        for expected_file in expected_files:
            if expected_file in output_files:
                file_path = output_files[expected_file]
                if os.path.exists(file_path):
                    validation_result["existing_files"].append(expected_file)
                else:
                    self.logger.error(f"❌ Output file not found: {file_path}")
                    validation_result["missing_files"].append(expected_file)
                    validation_result["passed"] = False
            else:
                self.logger.error(f"❌ Expected output not in results: {expected_file}")
                validation_result["missing_files"].append(expected_file)
                validation_result["passed"] = False

        if validation_result["passed"]:
            self.logger.info(f"✅ All {len(expected_files)} output files exist")

        self.validation_results["output_validation"] = validation_result
        return validation_result

    async def _validate_feature_quality(
        self, step8_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate feature quality and diversity."""
        validation_result = {
            "passed": True,
            "feature_reduction": {},
            "interpretability_available": False,
        }

        # Check feature reduction
        original_features = step8_state.get("original_features", 0)
        phase1_features = step8_state.get("phase1_features", 0)
        
        if original_features > 0 and phase1_features > 0:
            reduction_ratio = 1 - (phase1_features / original_features)
            validation_result["feature_reduction"]["phase1"] = reduction_ratio
            
            if reduction_ratio < 0.2:  # Less than 20% reduction
                self.logger.warning(f"⚠️ Low feature reduction in Phase 1: {reduction_ratio:.1%}")

        # Check interpretability results
        interp_results = step8_state.get("interpretability_results", {})
        if interp_results:
            validation_result["interpretability_available"] = True
            self.logger.info("✅ Interpretability analysis available")
            
            # Check for SHAP/LIME results
            for feature_set in interp_results.values():
                if "shap_importance" in feature_set:
                    self.logger.info("   ✓ SHAP analysis completed")
                if "lime_explanations" in feature_set:
                    self.logger.info("   ✓ LIME analysis completed")
                if "model_performance" in feature_set:
                    perf = feature_set["model_performance"]
                    self.logger.info(f"   ✓ Model performance: ROC-AUC = {perf.get('roc_auc', 0):.4f}")
        else:
            self.logger.warning("⚠️ No interpretability results found")

        self.validation_results["quality_validation"] = validation_result
        return validation_result


# Validator execution function
async def run_validator(
    training_input: dict[str, Any], pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """Run the step08_advanced_feature_selection validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Validation results dictionary

    """
    from src.config.constants import CONFIG
    
    validator = Step08AdvancedFeatureSelectionValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step08_advanced_feature_selection",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": datetime.now().isoformat(),
    }