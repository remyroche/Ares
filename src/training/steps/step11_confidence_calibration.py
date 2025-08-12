# src/training/steps/step11_confidence_calibration.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    failed,
)
from src.training.steps.unified_data_loader import get_unified_data_loader

try:
    import joblib  # Optional; used when loading joblib artifacts
except Exception:  # pragma: no cover
    joblib = None


class ConfidenceCalibrationStep:
    """Step 11: Confidence Calibration for individual models and ensembles."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="confidence calibration step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the confidence calibration step."""
        self.logger.info("Initializing Confidence Calibration Step...")
        self.logger.info("Confidence Calibration Step initialized successfully")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="confidence calibration step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute confidence calibration.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing calibration results
        """
        try:
            self.logger.info("🔄 Executing Confidence Calibration...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load analyst models and ensembles
            analyst_models = {}
            tactician_models = {}

            # Load analyst models
            analyst_models_dir = f"{data_dir}/enhanced_analyst_models"
            if os.path.exists(analyst_models_dir):
                for regime_dir in os.listdir(analyst_models_dir):
                    regime_path = os.path.join(analyst_models_dir, regime_dir)
                    if os.path.isdir(regime_path):
                        regime_models = {}
                        for model_file in os.listdir(regime_path):
                            if model_file.endswith((".pkl", ".joblib")):
                                model_name = model_file.replace(".pkl", "").replace(
                                    ".joblib",
                                    "",
                                )
                                model_path = os.path.join(regime_path, model_file)
                                try:
                                    if model_file.endswith(".joblib"):
                                        import joblib

                                        regime_models[model_name] = joblib.load(
                                            model_path,
                                        )
                                    else:
                                        with open(model_path, "rb") as f:
                                            regime_models[model_name] = pickle.load(f)
                                except Exception as e:
                                    self.logger.warning(
                                        f"Failed to load model {model_file}: {e}",
                                    )
                        analyst_models[regime_dir] = regime_models
            try:
                self.logger.info(
                    f"Analyst models loaded: regimes={len(analyst_models)}",
                )
            except Exception:
                pass

            # Load tactician models
            tactician_models_dir = f"{data_dir}/tactician_models"
            if os.path.exists(tactician_models_dir):
                for model_file in os.listdir(tactician_models_dir):
                    if model_file.endswith(".pkl"):
                        model_name = model_file.replace(".pkl", "")
                        model_path = os.path.join(tactician_models_dir, model_file)

                        with open(model_path, "rb") as f:
                            tactician_models[model_name] = pickle.load(f)
            try:
                self.logger.info(
                    f"Tactician models loaded: count={len(tactician_models)}",
                )
            except Exception:
                pass

            # Load ensembles
            analyst_ensembles = {}
            tactician_ensembles = {}

            # Load analyst ensembles
            analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
            if os.path.exists(analyst_ensembles_dir):
                for ensemble_file in os.listdir(analyst_ensembles_dir):
                    if ensemble_file.endswith("_ensemble.pkl"):
                        regime_name = ensemble_file.replace("_ensemble.pkl", "")
                        ensemble_path = os.path.join(
                            analyst_ensembles_dir,
                            ensemble_file,
                        )

                        with open(ensemble_path, "rb") as f:
                            analyst_ensembles[regime_name] = pickle.load(f)

            # Load tactician ensembles
            tactician_ensembles_dir = f"{data_dir}/tactician_ensembles"
            if os.path.exists(tactician_ensembles_dir):
                # New format: single model pickle per symbol/exchange
                model_path = os.path.join(
                    tactician_ensembles_dir,
                    f"{exchange}_{symbol}_tactician_ensemble.pkl",
                )
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        # Store under a default key for downstream usage
                        tactician_ensembles["blended"] = {"ensemble": pickle.load(f)}
                # Also support any additional ensembles present (e.g., experimental)
                for ensemble_file in os.listdir(tactician_ensembles_dir):
                    if (
                        ensemble_file.endswith("_tactician_ensemble.pkl")
                        and os.path.join(tactician_ensembles_dir, ensemble_file)
                        != model_path
                    ):
                        ensemble_path = os.path.join(
                            tactician_ensembles_dir,
                            ensemble_file,
                        )
                        try:
                            with open(ensemble_path, "rb") as f:
                                tactician_ensembles[ensemble_file] = {
                                    "ensemble": pickle.load(f),
                                }
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to load tactician ensemble {ensemble_file}: {e}",
                            )

            # Load a generic validation frame for calibration fallback
            generic_val = self._load_validation_frame(data_dir, exchange, symbol)

            # Perform calibration
            calibration_results = {}

            # 1. Calibrate individual analyst models (including SR regime separately)
            analyst_calibration = await self._calibrate_analyst_models(
                analyst_models,
                analyst_ensembles,
                generic_val,
                data_dir,
                exchange,
                symbol,
            )
            calibration_results["analyst_models"] = analyst_calibration

            # 2. Calibrate individual tactician models
            tactician_calibration = await self._calibrate_tactician_models(
                tactician_models,
                tactician_ensembles,
                generic_val,
            )
            calibration_results["tactician_models"] = tactician_calibration

            # 3. Calibrate analyst ensembles (SR-aware)
            analyst_ensemble_calibration = await self._calibrate_analyst_ensembles(
                analyst_ensembles,
                generic_val,
                data_dir,
                exchange,
                symbol,
            )
            calibration_results["analyst_ensembles"] = analyst_ensemble_calibration

            # 4. Calibrate tactician ensembles
            tactician_ensemble_calibration = await self._calibrate_tactician_ensembles(
                tactician_ensembles,
                generic_val,
            )
            calibration_results["tactician_ensembles"] = tactician_ensemble_calibration

            # Save calibration results
            calibration_dir = f"{data_dir}/calibration_results"
            os.makedirs(calibration_dir, exist_ok=True)

            calibration_file = (
                f"{calibration_dir}/{exchange}_{symbol}_calibration_results.pkl"
            )
            with open(calibration_file, "wb") as f:
                pickle.dump(calibration_results, f)

            # Save calibration summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_calibration_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self._summarize_calibration(calibration_results), f, indent=2)

            self.logger.info(
                f"✅ Confidence calibration completed. Results saved to {calibration_dir}",
            )

            # Update pipeline state
            pipeline_state["calibration_results"] = calibration_results

            return {
                "calibration_results": calibration_results,
                "calibration_file": calibration_file,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.print(error("❌ Error in Confidence Calibration: {e}"))
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    def _load_validation_frame(
        self,
        data_dir: str,
        exchange: str,
        symbol: str,
    ) -> pd.DataFrame | None:
        """Load generic validation features frame saved by step 4."""
        try:
            path = f"{data_dir}/{exchange}_{symbol}_features_validation.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and "label" in df.columns:
                    return df
        except Exception:
            self.logger.warning("Failed to load generic validation frame from step 4")

        # No fallback - step should fail if validation data is missing
        msg = f"Validation frame not found: {path}. Step 11 requires features from Step 4."
        raise FileNotFoundError(msg)

    def _load_regime_validation(
        self,
        data_dir: str,
        exchange: str,
        symbol: str,
        regime_name: str,
    ) -> pd.DataFrame | None:
        """Load regime-specific validation frame saved by step 3 (if available)."""
        try:
            regime_dir = os.path.join(data_dir, "regime_data")
            path = os.path.join(
                regime_dir,
                f"{exchange}_{symbol}_{regime_name}_data.pkl",
            )
            if os.path.exists(path):
                with open(path, "rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and "label" in df.columns:
                    return df
        except Exception as e:
            self.logger.warning(
                f"Failed to load regime validation for {regime_name}: {e}",
            )
        return None

    def _extract_features(
        self,
        df: pd.DataFrame,
        model: Any,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix X and labels y for a given model from a dataframe."""
        y = df["label"].astype(int)
        if hasattr(model, "feature_names_in_"):
            cols = [
                c for c in model.feature_names_in_ if c in df.columns and c != "label"
            ]
            X = df[cols].copy()
        else:
            X = (
                df.select_dtypes(include=[np.number])
                .drop(columns=["label"], errors="ignore")
                .copy()
            )
        X = X.fillna(0)
        return X, y

    async def _calibrate_analyst_models(
        self,
        models: dict[str, dict[str, Any]],
        ensembles: dict[str, Any],
        generic_val: pd.DataFrame | None,
        data_dir: str,
        exchange: str,
        symbol: str,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for regime_name, regime_models in models.items():
            regime_df = (
                self._load_regime_validation(data_dir, exchange, symbol, regime_name)
                or generic_val
            )
            if regime_df is None:
                self.logger.warning(
                    f"No validation data available for regime {regime_name}; skipping calibration",
                )
                continue
            regime_res = {}
            for model_name, model_data in regime_models.items():
                try:
                    base_model = (
                        model_data
                        if hasattr(model_data, "predict_proba")
                        else model_data.get("model", None)
                    )
                    if base_model is None:
                        continue
                    X_val, y_val = self._extract_features(regime_df, base_model)
                    calibrator = CalibratedClassifierCV(
                        base_estimator=base_model,
                        cv="prefit",
                        method="isotonic",
                    )
                    calibrator.fit(X_val, y_val)
                    acc = accuracy_score(y_val, calibrator.predict(X_val))
                    f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                    regime_res[model_name] = {
                        "calibrated_model": calibrator,
                        "metrics": {"accuracy": acc, "f1": f1},
                        "calibration_method": "isotonic_prefit",
                        "regime": regime_name,
                    }
                except Exception as e:
                    self.logger.warning(
                        f"Calibration failed for analyst model {model_name} in {regime_name}: {e}",
                    )
            results[regime_name] = regime_res
        return results

    async def _calibrate_tactician_models(
        self,
        models: dict[str, Any],
        ensembles: dict[str, Any],
        generic_val: pd.DataFrame | None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if generic_val is None:
            return results
        for model_name, model_data in models.items():
            try:
                base_model = (
                    model_data
                    if hasattr(model_data, "predict_proba")
                    else model_data.get("model", None)
                )
                if base_model is None:
                    continue
                X_val, y_val = self._extract_features(generic_val, base_model)
                calibrator = CalibratedClassifierCV(
                    base_estimator=base_model,
                    cv="prefit",
                    method="isotonic",
                )
                calibrator.fit(X_val, y_val)
                acc = accuracy_score(y_val, calibrator.predict(X_val))
                f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                results[model_name] = {
                    "calibrated_model": calibrator,
                    "metrics": {"accuracy": acc, "f1": f1},
                    "calibration_method": "isotonic_prefit",
                }
            except Exception as e:
                self.logger.warning(
                    f"Calibration failed for tactician model {model_name}: {e}",
                )
        return results

    async def _calibrate_analyst_ensembles(
        self,
        ensembles: dict[str, Any],
        generic_val: pd.DataFrame | None,
        data_dir: str,
        exchange: str,
        symbol: str,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for regime_name, regime_ensembles in ensembles.items():
            # Prefer stacking_cv ensemble if present
            ensemble_obj = None
            if isinstance(regime_ensembles, dict):
                for key in ("stacking_cv", "dynamic_weighting", "voting"):
                    if key in regime_ensembles and isinstance(
                        regime_ensembles[key],
                        dict,
                    ):
                        ensemble_obj = regime_ensembles[key].get("ensemble")
                        if ensemble_obj is not None:
                            break
            if ensemble_obj is None:
                continue
            # Validation data
            regime_df = (
                self._load_regime_validation(data_dir, exchange, symbol, regime_name)
                or generic_val
            )
            if regime_df is None:
                continue
            try:
                X_val, y_val = self._extract_features(regime_df, ensemble_obj)
                wrapper = _PrefitWrapper(ensemble_obj)
                calibrator = CalibratedClassifierCV(
                    base_estimator=wrapper,
                    cv="prefit",
                    method="isotonic",
                )
                calibrator.fit(X_val, y_val)
                acc = accuracy_score(y_val, calibrator.predict(X_val))
                f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                results[regime_name] = {
                    "calibrated_ensemble": calibrator,
                    "metrics": {"accuracy": acc, "f1": f1},
                    "calibration_method": "isotonic_prefit",
                }
            except Exception as e:
                self.logger.warning(
                    f"Calibration failed for analyst ensemble in {regime_name}: {e}",
                )
        return results

    async def _calibrate_tactician_ensembles(
        self,
        ensembles: dict[str, Any],
        generic_val: pd.DataFrame | None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if not ensembles or generic_val is None:
            return results
        # ensembles may be a dict of types -> data
        for ensemble_type, ensemble_data in ensembles.items():
            ensemble_obj = (
                ensemble_data.get("ensemble")
                if isinstance(ensemble_data, dict)
                else None
            )
            if ensemble_obj is None:
                continue
            try:
                X_val, y_val = self._extract_features(generic_val, ensemble_obj)
                wrapper = _PrefitWrapper(ensemble_obj)
                calibrator = CalibratedClassifierCV(
                    base_estimator=wrapper,
                    cv="prefit",
                    method="isotonic",
                )
                calibrator.fit(X_val, y_val)
                acc = accuracy_score(y_val, calibrator.predict(X_val))
                f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                results[ensemble_type] = {
                    "calibrated_ensemble": calibrator,
                    "metrics": {"accuracy": acc, "f1": f1},
                    "calibration_method": "isotonic_prefit",
                }
            except Exception as e:
                self.logger.warning(
                    f"Calibration failed for tactician ensemble {ensemble_type}: {e}",
                )
        return results

    def _summarize_calibration(self, results: dict[str, Any]) -> dict[str, Any]:
        summary = {"generated_at": datetime.now().isoformat(), "sections": {}}
        for key, section in results.items():
            summary["sections"][key] = {
                "items": sum(len(v) for v in section.values())
                if isinstance(section, dict)
                else 0,
            }
        return summary


class _PrefitWrapper:
    """Wrapper to adapt prefit estimators/ensembles to sklearn CalibratedClassifierCV with cv='prefit'."""

    def __init__(self, base):
        self.base = base
        # feature_names_in_ passthrough for feature selection
        if hasattr(base, "feature_names_in_"):
            self.feature_names_in_ = base.feature_names_in_

    def fit(self, X, y):  # noqa: D401
        # No-op: base estimator is prefit
        return self

    def predict(self, X):
        if hasattr(self.base, "predict"):
            return self.base.predict(X)
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        if hasattr(self.base, "predict_proba"):
            return self.base.predict_proba(X)
        # Fallback: construct probabilities from class predictions (uniform confidence)
        preds = self.base.predict(X)
        # Assume classes in set {-1, 0, 1}
        n_classes = 3
        proba = np.zeros((len(preds), n_classes))
        # Map labels to indices: -1 -> 0, 0 -> 1, 1 -> 2
        idx = preds + 1
        valid_mask = (idx >= 0) & (idx < n_classes)
        if np.any(valid_mask):
            proba[np.arange(len(preds))[valid_mask], idx[valid_mask]] = 1.0
        if not np.all(valid_mask):  # log once
            system_logger.warning(
                "Predictions outside expected {-1,0,1} encountered in _PrefitWrapper; ignored in probability mapping",
            )
        return proba


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the confidence calibration step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = ConfidenceCalibrationStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:
        print(failed("Confidence calibration failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
