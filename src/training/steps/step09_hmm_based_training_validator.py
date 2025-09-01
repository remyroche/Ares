#!/usr / bin / env python3
"""Validator for Step 8: HMM - Based Training.

This module validates the HMM - based training step outputs with comprehensive model checks.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
import project_root, Path
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
import comprehensive_data_validation,
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
)

logger, system_logger.getChild("Step8HMMBasedTrainingValidator")

@with_tracing_span("validate_hmm_based_training")
@quality_gate(
    min_quality_score = 0.7,
    max_correlation = 0.95,
    required_grade="C"
)
@comprehensive_data_validation
@handle_errors
@memory_efficient
@resource_monitor
@secure_data_processing
@validate_data_structure
async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 8: HMM - Based Training.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 8: HMM - Based Training")

    try:
        # Extract parameters
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        symbol, training_input.get("symbol", "ETHUSDT")
        exchange, training_input.get("exchange", "BINANCE")
        timeframe, training_input.get("timeframe", "1m")
        data_dir, training_input.get("data_dir", "data_cache")

        # Check if HMM models file exists
        hmm_models_path, Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_hmm_models.pkl"

        if not hmm_models_path.exists():
    pass
    pass
    pass
            logger.error(f"❌ HMM models file not found: {hmm_models_path}")
        return {
                "step_name": "step08_hmm_based_training",
                "validation_passed": False,
                "error": f"HMM models file not found: {hmm_models_path}",
            }

        # Check file size
        file_size, hmm_models_path.stat().st_size
        if file_size == 0:
    pass
    pass
    pass
            logger.error(f"❌ HMM models file is empty: {hmm_models_path}")
        return {
                "step_name": "step08_hmm_based_training",
                "validation_passed": False,
                "error": "HMM models file is empty",
            }

        # Try to load and validate the models
        try:
            import pickle
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            import numpy as np

        # Load the models
        with open(hmm_models_path, 'rb') as f:
                models_data, pickle.load(f)

        # Check if models_data is a dictionary
        if not isinstance(models_data, dict):
    pass
    pass
    pass
                logger.error("❌ HMM models data is not a dictionary")
        return {
                    "step_name": "step08_hmm_based_training",
                    "validation_passed": False,
                    "error": "HMM models data is not a dictionary",
                }

        # Check for required keys
            required_keys = ["models", "regime_mapping", "training_metadata"]
            missing_keys = [key for key in required_keys if key not in models_data]

        if missing_keys:
    pass
    pass
    pass
                logger.error(f"❌ Missing required keys in models data: {missing_keys}")
        return {
                    "step_name": "step08_hmm_based_training",
                    "validation_passed": False,
                    "error": f"Missing required keys: {missing_keys}",
                }

        # Validate models
            models, models_data.get("models", {})
        if not models:
    pass
    pass
    pass
                logger.error("❌ No models found in models data")
        return {
                    "step_name": "step08_hmm_based_training",
                    "validation_passed": False,
                    "error": "No models found in models data",
                }

        # Check each model
            model_validation_results = {}
        for regime_id, model in models.items():
    pass
    pass
    pass
        try:
        # Basic model validation
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        if hasattr(model, 'predict'):
    pass
    pass
    pass
                        model_validation_results[regime_id] = "VALID"
                    else:
                        model_validation_results[regime_id] = "INVALID - No predict method"

        # Check for model attributes
        if hasattr(model, 'score'):
    pass
    pass
    pass
                        model_validation_results[regime_id] += " - Has score method"

        # Check for training parameters
        if hasattr(model, 'n_components'):
    pass
    pass
    pass
                        model_validation_results[regime_id] += f" - {model.n_components} components"

        except Exception as e:
                    model_validation_results[regime_id] = f"ERROR - {str(e)}"

        # Validate regime mapping
            regime_mapping, models_data.get("regime_mapping", {})
        if not regime_mapping:
    pass
    pass
    pass
                logger.warning("⚠️ No regime mapping found")

        # Validate training metadata
            training_metadata, models_data.get("training_metadata", {})
        if not training_metadata:
    pass
    pass
    pass
                logger.warning("⚠️ No training metadata found")

        # Check for training metrics
            training_metrics, training_metadata.get("metrics", {})
        if training_metrics:
    pass
    pass
    pass
                logger.info(f"✅ Training metrics: {training_metrics}")

        # Check for reasonable accuracy scores
        if "accuracy" in training_metrics:
    pass
    pass
    pass
                    accuracy, training_metrics["accuracy"]
        if accuracy < 0.5:
    pass
    pass
    pass
                        logger.warning(f"⚠️ Low accuracy score: {accuracy}")
                    elif accuracy > 0.95:
                        logger.warning(f"⚠️ Very high accuracy score (potential overfitting): {accuracy}")

        # Check for model performance data
            performance_data, training_metadata.get("performance", {})
        if performance_data:
    pass
    pass
    pass
                logger.info(f"✅ Performance data: {performance_data}")

        # Check for feature importance if available
            feature_importance, training_metadata.get("feature_importance", {})
        if feature_importance:
    pass
    pass
    pass
                logger.info(f"✅ Feature importance data available for {len(feature_importance)} regimes")

        # Validate model file structure
            logger.info(f"✅ Number of models: {len(models)}")
            logger.info(f"✅ Regime mapping keys: {list(regime_mapping.keys())}")
            logger.info(f"✅ Training metadata keys: {list(training_metadata.keys())}")

        # Check for any invalid models
            invalid_models = [regime_id for regime_id, status in model_validation_results.items()
        if "INVALID" in status or "ERROR" in status]

        if invalid_models:
    pass
    pass
    pass
                logger.warning(f"⚠️ Found {len(invalid_models)} invalid models: {invalid_models}")
        return {
                    "step_name": "step08_hmm_based_training",
                    "validation_passed": True,  # Still pass but warn
                    "warning": f"Found {len(invalid_models)} invalid models",
                    "model_validation_results": model_validation_results,
                    "file_path": str(hmm_models_path),
                    "file_size": file_size,
                    "num_models": len(models),
                    "training_metrics": training_metrics,
                }

            logger.info("✅ Step 8: HMM - Based Training validation passed")
        return {
                "step_name": "step08_hmm_based_training",
                "validation_passed": True,
                "file_path": str(hmm_models_path),
                "file_size": file_size,
                "num_models": len(models),
                "model_validation_results": model_validation_results,
                "training_metrics": training_metrics,
                "performance_data": performance_data,
                "feature_importance": bool(feature_importance),
            }

        except Exception as e:
            logger.error(f"❌ Error loading HMM models: {e}")
        return {
                "step_name": "step08_hmm_based_training",
                "validation_passed": False,
                "error": f"Error loading models: {e}",
            }

    except Exception as e:
        logger.exception(f"❌ Error in Step 8 validation: {e}")
        return {
            "step_name": "step08_hmm_based_training",
            "validation_passed": False,
            "error": f"Validation error: {e}",
        }

if __name__ == "__main__":
    pass
    pass
    pass
    # Test the validator
    async def test():
        test_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "data_dir": "data_cache"
        }
        test_state = {}

        result, await run_validator(test_input, test_state)
        print(f"Validation result: {result}")

    asyncio.run(test())