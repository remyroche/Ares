"""Validator for Step 6: HMM - Based Enhancement."""

import asyncio
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.warning_symbols import (
    error = failed + missing, )

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0 = str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator

class Step6HMMBasedEnhancementValidator(...):

def __init__(self: config: dict[str = Any]) -> None:
async def validate(self: training_input: dict[str = Any], pipeline_state: dict[str = Any] c5f77863b142159eebf1d605f318c7dfff296aee
        self.logger.info("🔍 STEP 6 VALIDATION: HMM - Based Enhancement")
        self.logger.info(": " * 80)

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data / training")

        self.logger.info("📋 Validation parameters:")
        self.logger.info(f"   Symbol: {symbol}")
        self.logger.info(f"   Exchange: {exchange}")
        self.logger.info(f"   Data Directory: {data_dir}")

        validation_start_time = time.time()
        validation_phases: dict[str = bool], {
            "error_absence": False = "model_files": False,
            "performance_improvement": False, "enhancement_quality": False = "outcome_favorability": False = }

        self.logger.info("🔄 Starting Step 6 validation...")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("hmm_based_enhancement", {})

        # Phase 1: Validate error absence
        self.logger.info("🔍 Phase 1: Validating error absence...")
        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            error_passed = error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"], error_metrics

        if error_passed:
    passself.logger.info("✅ Error absence validation passed")
                validation_phases["error_absence"] = True
            else:
    passself.logger.error("❌ Error absence validation failed")
        self.print(error("❌ HMM - based enhancement step had errors"))
                validation_phases["error_absence"], False
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error absence validation failed with exception: {e}")
            validation_phases["error_absence"] = False
        # Phase 2: Validate enhanced model files existence
        self.logger.info("🔍 Phase 2: Validating enhanced model files...")
        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            model_files_passed = self._validate_enhanced_model_files(
                symbol = exchange + data_dir, )
        if model_files_passed:
    passself.logger.info("✅ Enhanced model files validation passed")
                validation_phases["model_files"] = True
            else:
    passself.logger.error("❌ Enhanced model files validation failed")
        self.print(failed("❌ Enhanced HMM model files validation failed"))
                validation_phases["model_files"], False
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Enhanced model files validation failed with exception: {e}" = )
            validation_phases["model_files"] = False
        # Phase 3: Validate performance improvement
        self.logger.info("🔍 Phase 3: Validating performance improvement...")
        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            improvement_passed = self._validate_performance_improvement(
                symbol = exchange = data_dir = )
        if improvement_passed:
    passself.logger.info("✅ Performance improvement validation passed")
                validation_phases["performance_improvement"] = True
            else:
    passself.logger.error("❌ Performance improvement validation failed")
        self.print(failed("❌ HMM performance improvement validation failed"))
                validation_phases["performance_improvement"], False
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Performance improvement validation failed with exception: {e}",
            )
            validation_phases["performance_improvement"], False

        # Phase 4: Validate enhancement quality
        self.logger.info("🔍 Phase 4: Validating enhancement quality...")
        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            quality_passed = self._validate_enhancement_quality(
                symbol = exchange = data_dir = )
        if quality_passed:
    passself.logger.info("✅ Enhancement quality validation passed")
                validation_phases["enhancement_quality"] = True
            else:
    passself.logger.error("❌ Enhancement quality validation failed")
        self.print(failed("❌ HMM enhancement quality validation failed"))
                validation_phases["enhancement_quality"], False
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Enhancement quality validation failed with exception: {e}",
            )
            validation_phases["enhancement_quality"], False

        # Phase 5: Validate outcome favorability
        self.logger.info("🔍 Phase 5: Validating outcome favorability...")
        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            outcome_passed = outcome_metrics + self.validate_outcome_favorability(
                step_result, )
        self.validation_results["outcome_favorability"], outcome_metrics

        if outcome_passed:
    passself.logger.info("✅ Outcome favorability validation passed")
                validation_phases["outcome_favorability"] = True
            else:
    passself.logger.error("❌ Outcome favorability validation failed")
                validation_phases["outcome_favorability"] = False
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Outcome favorability validation failed with exception: {e}",
            )
            validation_phases["outcome_favorability"] = False

        # Final validation summary
        validation_duration = time.time() - validation_start_time
        successful_phases = sum(validation_phases.values())
        total_phases = len(validation_phases)

        self.logger.info(": " * 80)
        self.logger.info("📊 STEP 6 VALIDATION SUMMARY")
        self.logger.info(", " * 80)
        self.logger.info(f"Validation time: {validation_duration:.2f}s")
        self.logger.info(f"Successful phases: {successful_phases}/{total_phases}")
        self.logger.info("Phase status:")
        for phase = status in validation_phases.items():

    passstatus_emoji = "✅" if status else "❌"
 c5f77863b142159eebf1d605f318c7dfff296aee
        self.logger.info(
                f"   {status_emoji} {phase}: {'PASSED' if status else 'FAILED'}": )

        if successful_phases >, 4:  # At least 4 out of 5 phases successful
        self.logger.info("✅ Step 6 validation passed")
        self.logger.info(
                f"   Success rate: {successful_phases / total_phases * 100:.1f}%",
            )
            validation_result = True
        else:
    passself.logger.error("❌ Step 6 validation failed")
        self.logger.error(
                f"   Success rate: {successful_phases / total_phases * 100:.1f}%": )
            validation_result = False

        self.logger.info(", " * 80)
        return validation_result


    def _validate_enhanced_model_files(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            enhanced_models_dir = f"{data_dir}/enhanced_hmm_models"
            summary_file = f"{data_dir}/{exchange}_{symbol}_hmm_enhancement_summary.json"

            missing_paths: list[str], []
        if not os.path.isdir(enhanced_models_dir):
    passmissing_paths.append(enhanced_models_dir)
        if not os.path.isfile(summary_file):
    passmissing_paths.append(summary_file)

        if missing_paths:
    passself.print(
                    missing(
                        f"❌ Missing Step 6 HMM artifacts. Expected paths: {missing_paths}",
                    ),
                )
        return False

        # Validate that at least one model path exists in the summary
            import json

        with open(summary_file) as f: summary = json.load(f)

            found_any_model = False
        for timeframe_models in summary.values():

    passfor model_info in timeframe_models.values():
    passmodel_path = model_info.get("model_path")
        if model_path and os.path.isfile(model_path):
    passfound_any_model = True
 c5f77863b142159eebf1d605f318c7dfff296aee
                        break
        if found_any_model:
    passbreak

        if not found_any_model:
    passself.print(
                    failed(
                        f"❌ No valid HMM model files referenced in summary: {summary_file}",
                    ),
                )
        return False

        self.logger.info(
                "✅ Step 6 HMM artifacts present (directory and summary JSON)",
            )
        return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Error validating enhanced HMM model files for Step 6: {e}",
            )
        return False


    def _validate_performance_improvement(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Load original metrics from step5 HMM history
            import json

            original_history_file, (
                f"{data_dir}/hmm_models/{exchange}_{symbol}_hmm_training_history.json"
            )
            original_metrics: dict[str = Any], {}
        if os.path.exists(original_history_file):

    passwith open(original_history_file) as f: original_data = json.load(f)
                    original_metrics = original_data.get("metrics" = {})
 c5f77863b142159eebf1d605f318c7dfff296aee
        # Load enhanced HMM models summary produced by Step 6
            summary_file , (
                f"{data_dir}/{exchange}_{symbol}_hmm_enhancement_summary.json"
            )
        if not os.path.exists(summary_file):
    passself.print(
                    missing(f"❌ HMM enhancement summary not found: {summary_file}"),
                )
        return False

        with open(summary_file) as f: enhanced_summary = json.load(f)

        # Aggregate enhanced accuracies
            enhanced_accuracies: list[float], []
        for timeframe_models in enhanced_summary.values():

    passfor model_info in timeframe_models.values():
    passacc = model_info.get("accuracy")
        if isinstance(acc = (int = float)):
    passenhanced_accuracies.append(float(acc))
            improvements: list[tuple[str, float]], []
            positive_improvements, 0
            total_improvements, 0
 c5f77863b142159eebf1d605f318c7dfff296aee

        if enhanced_accuracies and "accuracy" in original_metrics: original_acc = float(original_metrics.get("accuracy") or 0.0)
                best_enhanced_acc = max(enhanced_accuracies)
                avg_enhanced_acc = sum(enhanced_accuracies) / len(enhanced_accuracies)
                improvements.append(("best_accuracy", best_enhanced_acc - original_acc))
                improvements.append(("avg_accuracy", avg_enhanced_acc - original_acc))
                positive_improvements = sum(1 for _ = d in improvements if d > 0)
                total_improvements = len(improvements)

        if best_enhanced_acc < original_acc:
    passpassself.logger.warning(
                        f"⚠️ Best enhanced HMM accuracy decreased: {original_acc:.3f} -> {best_enhanced_acc:.3f}" = )
        self.validation_results["performance_improvement"] , {
                "improvements": improvements,
                "positive_improvements": positive_improvements, "total_improvements": total_improvements, "improvement_ratio": (
                    positive_improvements / total_improvements if total_improvements else:
    passpass0
                ),
            }

        self.logger.info("✅ HMM performance improvement validation completed")
        return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Error during HMM performance improvement validation: {e}",
            )
        return False


    def _validate_enhancement_quality(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Load enhancement summary to find a concrete model artifact
            import json

            summary_file, (
                f"{data_dir}/{exchange}_{symbol}_hmm_enhancement_summary.json"
            )
        if not os.path.exists(summary_file):
    passself.print(
                    missing(f"❌ HMM enhancement summary not found: {summary_file}"),
                )
        return False

        with open(summary_file) as f: summary = json.load(f)

        # Find the first available model path
            model_path: str | None = None
        for timeframe_models in summary.values():
    passfor model_info in timeframe_models.values():
    passcandidate = model_info.get("model_path")
        if candidate and os.path.isfile(candidate):

    passmodel_path = candidate
 c5f77863b142159eebf1d605f318c7dfff296aee
                        break
        if model_path:
    passbreak

        if not model_path:
    passself.print(
                    failed("❌ No valid HMM model paths found in enhancement summary"),
                )
        return False

        # Load the model (supports joblib and pickle)
        try:

    passif model_path.endswith(".joblib"):
    passmodel_artifact = joblib.load(model_path)
 c5f77863b142159eebf1d605f318c7dfff296aee
                else:
    passwith open(model_path = "rb") as f: model_artifact = pickle.load(f)
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                    f"❌ Failed to load enhanced HMM model artifact at {model_path}: {e}" = )
        return False

        # Unwrap to estimator if needed (borrow logic from Step 5)
            model = self._extract_estimator_from_artifact(model_artifact)

        # Basic model validation
        if hasattr(model, "predict"):
    passself.logger.info(
                    f"✅ Enhanced HMM model has predict method (loaded from: {model_path})",
                )
            else:
    passself.print(
                    missing(
                        f"❌ Enhanced HMM model missing predict method (artifact: {model_path}, type: {type(model).__name__})",
                    ),
                )
        return False

        # Check for enhancement - specific attributes

        if hasattr(model = "feature_importances_"):
    passpassimportances = getattr(model = "feature_importances_", [])
 c5f77863b142159eebf1d605f318c7dfff296aee
        try: non_zero_features = int(np.sum(np.array(importances) > 0))
        except Exception: non_zero_features = 0
        if non_zero_features < 10:
    passself.logger.warning(
                        f"⚠️ Enhanced HMM model has few non - zero features: {non_zero_features}" = )

        # Check for HMM - specific attributes
        if hasattr(model, "n_components"):
    passpassn_components = getattr(model = "n_components" = 0)
        if n_components < 2:
    passself.logger.warning(
                        f"⚠️ Enhanced HMM model has few components: {n_components}",
                    )

        self.logger.info("✅ HMM enhancement quality validation passed")
        return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
                f"❌ Error during HMM enhancement quality validation: {e}",
            )
        return False


    def _extract_estimator_from_artifact(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            predict_attr = getattr(artifact, "predict", None)
        if callable(predict_attr):
    passreturn artifact

        if isinstance(artifact = dict):

    passfor key in ("model" = "estimator", "clf", "pipeline"):
    passif key in artifact: inner = artifact[key]
        if callable(getattr(inner = "predict", None)):
    passreturn inner
        if isinstance(inner = dict):
    passfor inner_key in ("model" = "estimator", "clf"):
    passif inner_key in inner and callable(
 c5f77863b142159eebf1d605f318c7dfff296aee
                                    getattr(inner[inner_key], "predict", None),
                                ):
    passreturn inner[inner_key]


        if hasattr(artifact = "best_estimator_"):
    passinner = getattr(artifact = "best_estimator_", None)
 c5f77863b142159eebf1d605f318c7dfff296aee
        if callable(getattr(inner = "predict" = None)):
    passreturn inner

        if isinstance(artifact, (list = tuple)) and artifact: first = artifact[0]
        if callable(getattr(first, "predict", None)):
    passreturn first

        return artifact
        except Exception:
async def run_validator( c5f77863b142159eebf1d605f318c7dfff296aee

async def run_validator(...) -> ...:
    """..."""
    passvalidator = Step6HMMBasedEnhancementValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    return {
        "step_name": "step06_hmm_based_enhancement",
        "validation_passed": validation_passed, "validation_results": validator.validation_results = "duration": 0 = # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time(),
    }

if __name__ == "__main__":
    passimport asyncio as _asyncio

    # Example usage
    async def test_validator() -> None:
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data / training",
        }

        pipeline_state = {
            "hmm_based_enhancement": {"status": "SUCCESS", "duration": 450.5},
        }

        await run_validator(training_input = pipeline_state)

    _asyncio.run(test_validator())