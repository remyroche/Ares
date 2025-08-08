"""
Validator for Step 7: Analyst Ensemble Creation
"""

import os
import sys
import json
import pickle
import asyncio
from typing import Any, Dict, List, Tuple
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step7AnalystEnsembleCreationValidator(BaseValidator):
    """Validator for Step 7: Analyst Ensemble Creation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("step7_analyst_ensemble_creation", config)
        # Soft thresholds to catch obvious issues without overfitting
        self.min_models_per_regime = 2
        self.min_expected_ensembles = {"stacking_cv", "dynamic_weighting", "voting"}
        self.min_cv_mean = 0.50  # minimal acceptable CV mean for stacking
        self.min_validation_accuracy = 0.50  # minimal acceptable validation accuracy where available
        self.weight_tolerance = 0.05  # sum of weights should be ~1 within this tolerance

    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the analyst ensemble creation step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating analyst ensemble creation step...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state (fallback to reasonable keys)
        step_result = (
            pipeline_state.get("analyst_ensemble_creation")
            or pipeline_state.get("analyst_ensembles")
            or pipeline_state.get("ensemble_creation")
            or {}
        )

        # 1) Critical: error absence
        error_passed, error_metrics = self.validate_error_absence(step_result if isinstance(step_result, dict) else {})
        self.validation_results["error_absence"] = error_metrics
        if not error_passed:
            self.logger.error("❌ Analyst ensemble creation step had critical errors - stopping process")
            return False

        # 2) Critical: artifacts exist (summary JSON and ensemble pkl directory)
        files_passed = self._validate_artifacts_exist(symbol, exchange, data_dir)
        if not files_passed:
            self.logger.error("❌ Analyst ensemble artifacts missing - stopping process")
            return False

        # 3) Critical: summary content integrity and basic metrics
        summary_passed = self._validate_summary_content(symbol, exchange, data_dir)
        if not summary_passed:
            self.logger.error("❌ Analyst ensemble summary validation failed - stopping process")
            return False

        # 4) Critical: pickled ensembles integrity and API checks
        pickles_passed = self._validate_pickled_ensembles(symbol, exchange, data_dir)
        if not pickles_passed:
            self.logger.error("❌ Analyst ensemble pickle validation failed - stopping process")
            return False

        # 5) Warning-level: diversity and performance sanity
        diversity_ok = self._validate_diversity_and_weights(symbol, exchange, data_dir)
        if not diversity_ok:
            self.logger.warning("⚠️ Analyst ensemble diversity/weights validation had issues - continuing with caution")

        # 6) Warning-level: outcome favorability from step result
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result if isinstance(step_result, dict) else {})
        self.validation_results["outcome_favorability"] = outcome_metrics
        if not outcome_passed:
            self.logger.warning("⚠️ Analyst ensemble creation outcome not clearly favorable - continuing with caution")

        self.logger.info("✅ Analyst ensemble creation validation passed")
        return True

    def _validate_artifacts_exist(self, symbol: str, exchange: str, data_dir: str) -> bool:
        try:
            summary_file = f"{data_dir}/{exchange}_{symbol}_analyst_ensemble_summary.json"
            summary_exists, summary_metrics = self.validate_file_exists(summary_file, "analyst_ensemble_summary")
            self.validation_results["summary_file"] = summary_metrics
            if not summary_exists:
                return False

            ensemble_models_dir = f"{data_dir}/analyst_ensembles"
            dir_exists = os.path.isdir(ensemble_models_dir)
            dir_metrics = {"dir_path": ensemble_models_dir, "exists": dir_exists}
            self.validation_results["ensemble_models_dir"] = dir_metrics
            if not dir_exists:
                self.logger.error(f"❌ Ensemble models directory not found: {ensemble_models_dir}")
                return False

            # At least one regime ensemble should be present
            pkl_files = [f for f in os.listdir(ensemble_models_dir) if f.endswith(".pkl")]
            if len(pkl_files) == 0:
                self.logger.error("❌ No pickled analyst ensembles found in directory")
                return False

            self.validation_results["ensemble_models_dir"]["pkl_files_count"] = len(pkl_files)
            return True
        except Exception as e:
            self.logger.error(f"❌ Error validating artifact existence: {e}")
            return False

    def _validate_summary_content(self, symbol: str, exchange: str, data_dir: str) -> bool:
        try:
            summary_file = f"{data_dir}/{exchange}_{symbol}_analyst_ensemble_summary.json"
            with open(summary_file, "r") as f:
                summary = json.load(f)

            if not isinstance(summary, dict) or len(summary) == 0:
                self.logger.error("❌ Ensemble summary is empty or invalid format")
                return False

            regimes_checked = 0
            issues: List[str] = []

            for regime_name, ensembles in summary.items():
                regimes_checked += 1
                # Expect a dict of ensemble types per regime
                if not isinstance(ensembles, dict) or len(ensembles) == 0:
                    issues.append(f"Regime {regime_name} has no ensembles in summary")
                    continue

                # Check expected ensemble types
                present_types = set(ensembles.keys())
                missing_types = self.min_expected_ensembles - present_types
                if missing_types:
                    issues.append(f"Regime {regime_name} missing ensembles: {sorted(missing_types)}")

                for ens_type, ens_info in ensembles.items():
                    # base_models present and count >= min
                    base_models = ens_info.get("base_models", [])
                    if not isinstance(base_models, list) or len(base_models) < self.min_models_per_regime:
                        issues.append(f"Regime {regime_name}/{ens_type}: insufficient base models")

                    # validation metrics (if present)
                    val_metrics = ens_info.get("validation_metrics", {}) or {}
                    if val_metrics:
                        acc = val_metrics.get("accuracy")
                        if isinstance(acc, (int, float)) and acc < self.min_validation_accuracy:
                            issues.append(f"Regime {regime_name}/{ens_type}: low validation accuracy {acc:.3f}")

                    # cv_scores for stacking
                    if ens_type == "stacking_cv":
                        cv_scores = (ens_info.get("cv_scores") or {})
                        cv_mean = cv_scores.get("mean")
                        if isinstance(cv_mean, (int, float)) and cv_mean < self.min_cv_mean:
                            issues.append(f"Regime {regime_name}/stacking_cv: low CV mean {cv_mean:.3f}")

                    # weights for dynamic weighting
                    if ens_type == "dynamic_weighting":
                        weights = (ens_info.get("weights") or {})
                        if isinstance(weights, dict) and weights:
                            weight_sum = sum(float(w) for w in weights.values())
                            if abs(weight_sum - 1.0) > self.weight_tolerance:
                                issues.append(
                                    f"Regime {regime_name}/dynamic_weighting: weight sum {weight_sum:.3f} not ~1"
                                )

            self.validation_results["summary_content"] = {
                "regimes_checked": regimes_checked,
                "issues": issues,
                "passed": len(issues) == 0,
            }

            if issues:
                for msg in issues[:5]:
                    self.logger.warning(f"⚠️ {msg}")
            return len(issues) == 0
        except Exception as e:
            self.logger.error(f"❌ Error validating summary content: {e}")
            return False

    def _validate_pickled_ensembles(self, symbol: str, exchange: str, data_dir: str) -> bool:
        try:
            ensemble_models_dir = f"{data_dir}/analyst_ensembles"
            pkl_files = [f for f in os.listdir(ensemble_models_dir) if f.endswith(".pkl")]
            if not pkl_files:
                return False

            checked = 0
            bad_files: List[str] = []

            for fname in pkl_files[:10]:  # inspect up to 10 to limit runtime
                fpath = os.path.join(ensemble_models_dir, fname)
                try:
                    with open(fpath, "rb") as f:
                        data = pickle.load(f)

                    # Expect a dict of ensembles per regime: keys include stacking_cv, dynamic_weighting, voting
                    if not isinstance(data, dict) or not data:
                        bad_files.append(f"{fname}: invalid pickle format")
                        continue

                    # Validate that for each ensemble entry, the 'ensemble' object exposes predict
                    for ens_key, ens_data in list(data.items())[:3]:  # check up to 3 types
                        if isinstance(ens_data, dict):
                            ensemble_obj = ens_data.get("ensemble")
                            if ensemble_obj is None or not hasattr(ensemble_obj, "predict"):
                                bad_files.append(f"{fname}/{ens_key}: missing predict method")
                except Exception as e:
                    bad_files.append(f"{fname}: load error {e}")
                checked += 1

            self.validation_results["pickle_checks"] = {
                "files_checked": checked,
                "issues": bad_files,
                "passed": len(bad_files) == 0,
            }

            if bad_files:
                for msg in bad_files[:5]:
                    self.logger.warning(f"⚠️ {msg}")
            return len(bad_files) == 0
        except Exception as e:
            self.logger.error(f"❌ Error validating pickled ensembles: {e}")
            return False

    def _validate_diversity_and_weights(self, symbol: str, exchange: str, data_dir: str) -> bool:
        try:
            summary_file = f"{data_dir}/{exchange}_{symbol}_analyst_ensemble_summary.json"
            if not os.path.exists(summary_file):
                return True
            with open(summary_file, "r") as f:
                summary = json.load(f)

            diversity_issues: List[str] = []
            for regime_name, ensembles in summary.items():
                # model type diversity via base_models names
                for ens_type, ens_info in ensembles.items():
                    base_models: List[str] = ens_info.get("base_models", [])
                    if len(set(base_models)) < self.min_models_per_regime:
                        diversity_issues.append(f"{regime_name}/{ens_type}: low base model diversity")

                    if ens_type == "dynamic_weighting":
                        weights = (ens_info.get("weights") or {})
                        nonzero = sum(1 for w in weights.values() if float(w) > 0.0)
                        if nonzero == 0:
                            diversity_issues.append(f"{regime_name}/dynamic_weighting: all weights are zero")

            self.validation_results["diversity"] = {
                "issues": diversity_issues,
                "passed": len(diversity_issues) == 0,
            }

            if diversity_issues:
                for msg in diversity_issues[:5]:
                    self.logger.warning(f"⚠️ {msg}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error validating diversity/weights: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 7 Analyst Ensemble Creation validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step7AnalystEnsembleCreationValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step7_analyst_ensemble_creation",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,
        "timestamp": asyncio.get_event_loop().time(),
    }


if __name__ == "__main__":
    async def test_validator():
        training_input = {"symbol": "ETHUSDT", "exchange": "BINANCE", "data_dir": "data/training"}
        pipeline_state = {"analyst_ensemble_creation": {"status": "SUCCESS", "duration": 200.0}}
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
