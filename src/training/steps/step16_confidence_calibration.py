# src/training/steps/step16_*.py

import asyncio
import contextlib
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error

try:
    import joblib  # Optional; used when loading joblib artifacts
except Exception:  # pragma: no cover
    joblib = None

class RegimeAwareConfidenceCalibrationStep:
    """Step 16: Regime-Aware Confidence Calibration for individual models and ensembles."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        
        # Initialize regime-specific configuration
        self.regime_config = self._initialize_regime_config()
        
        # Regime-specific state storage
        self.regime_calibration_results: dict[str, dict[str, Any]] = {}
        self.regime_validation_results: dict[str, dict[str, Any]] = {}

    def _initialize_regime_config(self) -> dict[str, Any]:
        """Initialize regime-specific configuration for confidence calibration."""
        return {
            "regime_specific_calibration": True,
            "regime_specific_validation": True,
            "regime_specific_logging": True,
            "min_regime_samples": 200,  # Minimum samples per regime for calibration
            "regime_validation_split": 0.2,  # Validation split per regime
            "regime_calibration_method": "isotonic",  # Calibration method per regime
            "regime_parallel_processing": True,  # Enable parallel regime processing
            "regime_memory_optimization": True,  # Enable memory optimization per regime
        }

    

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status["all_available"]:
            missing_modules = dependency_status["missing_modules"]
            self.logger.warning(f"Missing modules: {missing_modules}")
            # Continue with available modules, using fallbacks where needed

    @handles_errors(fallback=False)
    async def initialize(self) -> None:
        """Initialize the confidence calibration step."""
        self.logger.info("🚀 Initializing Confidence Calibration Step...")
        self.logger.info("✅ Confidence Calibration Step initialized successfully")

    @handles_errors
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="confidence calibration step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute regime-aware confidence calibration."

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing regime-specific calibration results
        """
        try:
            self.logger.info("🔄 Executing Regime-Aware Confidence Calibration...")
            self.logger.info(f"📊 Regime configuration: {self.regime_config}")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load analyst models and ensembles
            analyst_models: dict[str, dict[str, Any]] = {}
            tactician_models: dict[str, Any] = {}

            # Load analyst models
            analyst_models_dir = f"{data_dir}/enhanced_analyst_models"
            if os.path.exists(analyst_models_dir):
                from src.utils.logger import heartbeat

                with heartbeat(
                    self.logger,
                    name="Step11 load_analyst_models",
                    interval_seconds=60.0,
                ):
                    for regime_dir in os.listdir(analyst_models_dir):
                        regime_path = os.path.join(analyst_models_dir, regime_dir)
                        if os.path.isdir(regime_path):
                            regime_models: dict[str, Any] = {}
                            for model_file in os.listdir(regime_path):
                                if model_file.endswith((".pkl", ".joblib")):
                                    model_name = model_file.replace(".pkl", "").replace(
                                        ".joblib",
                                        "",
                                    )
                                    model_path = os.path.join(regime_path, model_file)
                                    try:
                                        if model_file.endswith(".joblib") and joblib is not None:
                                            regime_models[model_name] = joblib.load(
                                                model_path,
                                            )
                                        else:
                                            with open(model_path, "rb") as f:
                                                regime_models[model_name] = pickle.load(
                                                    f,
                                                )
                                    except Exception as e:
                                        self.logger.warning(
                                            f"⚠️ Failed to load model {model_file}: {e}",
                                        )
                            analyst_models[regime_dir] = regime_models
                with contextlib.suppress(Exception):
                    self.logger.info(
                        f"Analyst models loaded: regimes={len(analyst_models)}",
                    )

            # Load tactician models
            tactician_models_dir = f"{data_dir}/tactician_models"
            if os.path.exists(tactician_models_dir):
                from src.utils.logger import heartbeat

                with heartbeat(
                    self.logger,
                    name="Step11 load_tactician_models",
                    interval_seconds=60.0,
                ):
                    for model_file in os.listdir(tactician_models_dir):
                        if model_file.endswith(".pkl"):
                            model_name = model_file.replace(".pkl", "")
                            model_path = os.path.join(tactician_models_dir, model_file)

                            with open(model_path, "rb") as f:
                                tactician_models[model_name] = pickle.load(f)
                with contextlib.suppress(Exception):
                    self.logger.info(
                        f"Tactician models loaded: count={len(tactician_models)}",
                    )

            # Load ensembles
            analyst_ensembles: dict[str, Any] = {}
            tactician_ensembles: dict[str, Any] = {}

            # Load analyst ensembles
            analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
            if os.path.exists(analyst_ensembles_dir):
                from src.utils.logger import heartbeat

                with heartbeat(
                    self.logger,
                    name="Step11 load_analyst_ensembles",
                    interval_seconds=60.0,
                ):
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
                from src.utils.logger import heartbeat

                with heartbeat(
                    self.logger,
                    name="Step11 load_tactician_ensembles",
                    interval_seconds=60.0,
                ):
                    model_path = os.path.join(
                        tactician_ensembles_dir,
                        f"{exchange}_{symbol}_tactician_ensemble.pkl",
                    )
                    if os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            # Store under a default key for downstream usage
                            tactician_ensembles["blended"] = {
                                "ensemble": pickle.load(f),
                            }
                    # Also support any additional ensembles present (e.g., experimental)
                    for ensemble_file in os.listdir(tactician_ensembles_dir):
                        candidate_path = os.path.join(
                            tactician_ensembles_dir,
                            ensemble_file,
                        )
                        if (
                            ensemble_file.endswith("_tactician_ensemble.pkl")
                            and candidate_path != model_path
                        ):
                            try:
                                with open(candidate_path, "rb") as f:
                                    tactician_ensembles[ensemble_file] = {
                                        "ensemble": pickle.load(f),
                                    }
                            except Exception as e:
                                self.logger.warning(
                                    f"⚠️ Failed to load tactician ensemble {ensemble_file}: {e}",
                                )
                with contextlib.suppress(Exception):
                    pass

            # Load a generic validation frame for calibration fallback
            generic_val = self._load_validation_frame(data_dir, exchange, symbol)
            # Try to augment with 1m meta-labels if present
            try:
                step4_train = f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl"
                if os.path.exists(step4_train) and isinstance(
                    generic_val, pd.DataFrame,
                ):
                    with open(step4_train, "rb") as f:
                        step4_df = pickle.load(f)
                    one_m_cols = [
                        c
                        for c in getattr(step4_df, "columns", [])
                        if isinstance(c, str) and c.startswith("1m_")
                    ]
                    if (
                        one_m_cols
                        and "timestamp" in step4_df.columns
                        and "timestamp" in generic_val.columns
                    ):
                        generic_val = generic_val.merge(
                            step4_df[["timestamp", *one_m_cols]],
                            on="timestamp",
                            how="left",
                        )
                        self.logger.info(
                            f"Augmented validation frame with {len(one_m_cols)} 1m meta-label columns",
                        )
            except Exception as _ce:
                self.logger.warning(
                    f"⚠️ Could not augment validation frame with 1m meta-labels: {_ce}",
                )
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"Validation frame loaded: shape={getattr(generic_val, 'shape', None)}",
                )

            # Perform calibration
            calibration_results: dict[str, Any] = {}

            # 1. Calibrate individual analyst models (including SR regime separately)
            self.logger.info("Step11: Calibrating analyst models...")
            analyst_calibration = await self._calibrate_regime_aware_analyst_models(
                analyst_models,
                analyst_ensembles,
                generic_val,
                data_dir,
                exchange,
                symbol,
            )
            calibration_results["analyst_models"] = analyst_calibration
            with contextlib.suppress(Exception):
                pass

            # 2. Calibrate individual tactician models
            self.logger.info("Step11: Calibrating tactician models...")
            tactician_calibration = await self._calibrate_regime_aware_tactician_models(
                tactician_models,
                tactician_ensembles,
                generic_val,
                data_dir,
                exchange,
                symbol,
            )
            calibration_results["tactician_models"] = tactician_calibration
            with contextlib.suppress(Exception):
                pass

            # 3. Calibrate analyst ensembles (SR-aware)
            self.logger.info("Step11: Calibrating analyst ensembles...")
            analyst_ensemble_calibration = await self._calibrate_analyst_ensembles(
                analyst_ensembles,
                generic_val,
                data_dir,
                exchange,
                symbol,
            )
            calibration_results["analyst_ensembles"] = analyst_ensemble_calibration
            with contextlib.suppress(Exception):
                pass

            # 4. Calibrate tactician ensembles
            self.logger.info("Step11: Calibrating tactician ensembles...")
            tactician_ensemble_calibration = await self._calibrate_tactician_ensembles(
                tactician_ensembles,
                generic_val,
            )
            calibration_results["tactician_ensembles"] = tactician_ensemble_calibration
            with contextlib.suppress(Exception):
                pass

            # Save calibration results
            calibration_dir = f"{data_dir}/calibration_results"
            os.makedirs(calibration_dir, exist_ok=True)

            calibration_file = (
                f"{calibration_dir}/{exchange}_{symbol}_calibration_results.pkl"
            )
            with open(calibration_file, "wb") as f:
                pickle.dump(calibration_results, f)
            try:
                self.logger.info(f"Saved calibration results: {calibration_file}")
                # Compact summary of counts for quick troubleshooting
                summary_counts = {
                    "analyst_models": sum(
                        len(v or {})
                        for v in calibration_results.get("analyst_models", {}).values()
                    ),
                    "tactician_models": len(
                        calibration_results.get("tactician_models", {}),
                    ),
                    "analyst_ensembles": len(
                        calibration_results.get("analyst_ensembles", {}),
                    ),
                    "tactician_ensembles": len(
                        calibration_results.get("tactician_ensembles", {}),
                    ),
                }
                self.logger.info(
                    {"msg": "calibration_saved_summary", "counts": summary_counts},
                )
            except Exception:
                pass

            # Save calibration summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_calibration_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self._summarize_calibration(calibration_results), f, indent=2)

            # Meta-labeling system removed - using only HMM market regimes
            try:
                artifacts_dir = self.config.get("meta_labeling", {}).get(
                    "artifacts_dir", "artifacts/meta_labeling",
                )
                os.makedirs(artifacts_dir, exist_ok=True)
                # Persist reliability if available from pipeline_state or calibration
                reliability: dict[str, float] = (
                    pipeline_state.get("label_reliability", {})
                    if isinstance(pipeline_state, dict)
                    else {}
                )
                if not reliability:
                    # fallback: simple per-label accuracy proxy from analyst_models calibration if present
                    acc_map: dict[str, float] = {}
                    try:
                        for models in (analyst_calibration or {}).values():
                            if isinstance(models, dict):
                                for name, res in models.items():
                                    if isinstance(res, dict) and "metrics" in res:
                                        acc_map[name] = float(
                                            res.get("metrics", {}).get("accuracy", 0.0),
                                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Error during reliability fallback calculation: {e}",
                        )
                    reliability = acc_map
                with open(os.path.join(artifacts_dir, "reliability.json"), "w") as f:
                    json.dump(reliability, f, indent=2)
                # Persist thresholds if provided in pipeline_state
                thresholds = (
                    pipeline_state.get("activation_thresholds", {})
                    if isinstance(pipeline_state, dict)
                    else {}
                )
                if thresholds:
                    with open(os.path.join(artifacts_dir, "thresholds.json"), "w") as f:
                        json.dump(thresholds, f, indent=2)
                self.logger.info(f"Persisted meta-label artifacts to {artifacts_dir}")
            except Exception as _pe:
                self.logger.warning(f"Threshold/reliability persistence skipped: {_pe}")

            self.logger.info(
                f"✅ Confidence calibration completed. Results saved to {calibration_dir}",
            )
            with contextlib.suppress(Exception):
                pass

            # Update pipeline state
            pipeline_state["calibration_results"] = calibration_results

            return {
                "calibration_results": calibration_results,
                "calibration_file": calibration_file,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:  # pragma: no cover - defensive
            self.print(error(f"❌ Error in Confidence Calibration: {e}"))
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    def _load_validation_frame(
        self, data_dir: str, exchange: str, symbol: str,
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
        msg = (
            f"Validation frame not found: {path}. Step 11 requires features from Step 4."
        )
        raise FileNotFoundError(msg)

    def _load_regime_validation(
        self, data_dir: str, exchange: str, symbol: str, regime_name: str,
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
        self, df: pd.DataFrame, model: Any,
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
            regime_res: dict[str, Any] = {}
            for model_name, model_data in regime_models.items():
                try:
                    base_model = (
                        model_data
                        if hasattr(model_data, "predict_proba")
                        else (model_data.get("model", None) if isinstance(model_data, dict) else None)
                    )
                    if base_model is None:
                        continue
                    X_val, y_val = self._extract_features(regime_df, base_model)
                    # Baseline metrics before calibration
                    base_metrics = self._calculate_base_metrics(
                        base_model, X_val, y_val,
                    )
                    calibrator = CalibratedClassifierCV(
                        estimator=base_model,
                        cv="prefit",
                        method="isotonic",
                    )
                    calibrator.fit(X_val, y_val)
                    acc = accuracy_score(y_val, calibrator.predict(X_val))
                    f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                    regime_res[model_name] = {
                        "calibrated_model": calibrator,
                        "metrics": {"accuracy": acc, "f1": f1},
                        "base_metrics": base_metrics,
                        "calibration_method": "isotonic_prefit",
                        "regime": regime_name,
                    }
                    # Log comparison
                    with contextlib.suppress(Exception):
                        self.logger.info(
                            {
                                "msg": "calibration_model_metrics",
                                "regime": regime_name,
                                "model": model_name,
                                "base": base_metrics,
                                "calibrated": {"accuracy": float(acc), "f1": float(f1)},
                            },
                        )
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
                    else (model_data.get("model", None) if isinstance(model_data, dict) else None)
                )
                if base_model is None:
                    continue
                X_val, y_val = self._extract_features(generic_val, base_model)
                # Baseline metrics
                base_metrics = self._calculate_base_metrics(base_model, X_val, y_val)
                calibrator = CalibratedClassifierCV(
                    estimator=base_model,
                    cv="prefit",
                    method="isotonic",
                )
                calibrator.fit(X_val, y_val)
                acc = accuracy_score(y_val, calibrator.predict(X_val))
                f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                results[model_name] = {
                    "calibrated_model": calibrator,
                    "metrics": {"accuracy": acc, "f1": f1},
                    "base_metrics": base_metrics,
                    "calibration_method": "isotonic_prefit",
                }
                with contextlib.suppress(Exception):
                    self.logger.info(
                        {
                            "msg": "calibration_tactician_model_metrics",
                            "model": model_name,
                            "base": base_metrics,
                            "calibrated": {"accuracy": float(acc), "f1": float(f1)},
                        },
                    )
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
            ensemble_obj: Any | None = None
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
                # Baseline metrics
                base_metrics = self._calculate_base_metrics(ensemble_obj, X_val, y_val)
                wrapper = _PrefitWrapper(ensemble_obj)
                calibrator = CalibratedClassifierCV(
                    estimator=wrapper,
                    cv="prefit",
                    method="isotonic",
                )
                calibrator.fit(X_val, y_val)
                acc = accuracy_score(y_val, calibrator.predict(X_val))
                f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                results[regime_name] = {
                    "calibrated_ensemble": calibrator,
                    "metrics": {"accuracy": acc, "f1": f1},
                    "base_metrics": base_metrics,
                    "calibration_method": "isotonic_prefit",
                }
                with contextlib.suppress(Exception):
                    self.logger.info(
                        {
                            "msg": "calibration_analyst_ensemble_metrics",
                            "regime": regime_name,
                            "base": base_metrics,
                            "calibrated": {"accuracy": float(acc), "f1": float(f1)},
                        },
                    )
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
                # Baseline metrics
                base_metrics = self._calculate_base_metrics(ensemble_obj, X_val, y_val)
                wrapper = _PrefitWrapper(ensemble_obj)
                calibrator = CalibratedClassifierCV(
                    estimator=wrapper,
                    cv="prefit",
                    method="isotonic",
                )
                calibrator.fit(X_val, y_val)
                acc = accuracy_score(y_val, calibrator.predict(X_val))
                f1 = f1_score(y_val, calibrator.predict(X_val), average="weighted")
                results[ensemble_type] = {
                    "calibrated_ensemble": calibrator,
                    "metrics": {"accuracy": acc, "f1": f1},
                    "base_metrics": base_metrics,
                    "calibration_method": "isotonic_prefit",
                }
                with contextlib.suppress(Exception):
                    self.logger.info(
                        {
                            "msg": "calibration_tactician_ensemble_metrics",
                            "type": ensemble_type,
                            "base": base_metrics,
                            "calibrated": {"accuracy": float(acc), "f1": float(f1)},
                        },
                    )
            except Exception as e:
                self.logger.warning(
                    f"Calibration failed for tactician ensemble {ensemble_type}: {e}",
                )
        return results

    def _summarize_calibration(self, results: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        # Analyst models
        analyst = results.get("analyst_models", {})
        summary["analyst_models"] = {
            regime: {name: data.get("metrics", {}) for name, data in models.items()}
            for regime, models in analyst.items()
        }
        # Tactician models
        tact_models = results.get("tactician_models", {})
        summary["tactician_models"] = {
            name: data.get("metrics", {}) for name, data in tact_models.items()
        }
        # Analyst ensembles
        analyst_ens = results.get("analyst_ensembles", {})
        summary["analyst_ensembles"] = {
            regime: data.get("metrics", {}) for regime, data in analyst_ens.items()
        }
        # Tactician ensembles
        tact_ens = results.get("tactician_ensembles", {})
        summary["tactician_ensembles"] = {
            etype: data.get("metrics", {}) for etype, data in tact_ens.items()
        }
        return summary

    def _calculate_base_metrics(
        self, model: Any, X_val: pd.DataFrame, y_val: pd.Series,
    ) -> dict[str, float]:
        """Helper to calculate baseline accuracy and F1 score for a model/ensemble."
        Returns {} if metrics cannot be computed.
        """
        try:
            if not hasattr(model, "predict"):
                return {}
            base_pred = model.predict(X_val)
            base_acc = accuracy_score(y_val, base_pred)
            base_f1 = f1_score(y_val, base_pred, average="weighted")
            return {"accuracy": float(base_acc), "f1": float(base_f1)}
        except Exception as e:
            with contextlib.suppress(Exception):
                self.logger.warning(
                    f"Could not calculate base metrics for {type(model).__name__}: {e}",
                )
            return {}

class _PrefitWrapper:
    """Wrapper to adapt prefit estimators/ensembles to sklearn CalibratedClassifierCV with cv='prefit'."""

    def __init__(self, base) -> None:
        self.base = base
        # feature_names_in_ passthrough for feature selection
        if hasattr(base, "feature_names_in_"):
            self.feature_names_in_ = base.feature_names_in_  # type: ignore[attr-defined]

    def fit(self, X: pd.DataFrame, y: pd.Series):  # noqa: D401
        # No-op: base estimator is prefit
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.base, "predict"):
            return np.asarray(self.base.predict(X))
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.base, "predict_proba"):
            return np.asarray(self.base.predict_proba(X))
        # Fallback: construct probabilities from class predictions (uniform confidence)
        preds = np.asarray(self.base.predict(X))
        # Assume classes in set {-1, 0, 1}
        n_classes = 3
        proba = np.zeros((len(preds), n_classes), dtype=float)
        # Map labels to indices: -1 -> 0, 0 -> 1, 1 -> 2
        idx = preds.astype(int) + 1
        valid_mask = (idx >= 0) & (idx < n_classes)
        if np.any(valid_mask):
            proba[np.arange(len(preds))[valid_mask], idx[valid_mask]] = 1.0
        if not np.all(valid_mask):  # log once
            system_logger.warning(
                "Predictions outside expected {-1,0,1} encountered in _PrefitWrapper; ignored in probability mapping",
            )
        return proba

from src.utils.enhanced_mlflow_integration import (
    copy,
    create_detailed_step_report,
    import,
    log_step_artifact_with_standardized_name,
    log_step_dataframe_with_standardized_name,
    log_step_metrics,
    log_step_report,
    os,
    os.path,
    with_enhanced_mlflow_logging,
)

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
)


# For backward compatibility with existing step structure
@deterministic_seed(42)
@idempotent_step(step_key="step11_confidence_calibration")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=2400.0)
@validate_step_prerequisites(
    required_directories=["data/training", "models"],
    min_memory_gb=4.0,
    min_disk_gb=3.0,
    required_packages=["pandas", "numpy", "sklearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "features", "targets"],
    },
    context="Confidence Calibration",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=15000, streaming_processing=True, memory_pool=True, cleanup_frequency=35,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["models/{exchange}_{symbol}_calibrated.pkl"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["predictions", "probabilities"],
    },
    performance_thresholds={"calibration_time_minutes": 60.0},
    format_validation=True,
)
@quality_gate(
    model_performance_thresholds={"calibration_accuracy": 0.7},
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"calibration_score": 0.7},
)
async def _calibrate_regime_aware_analyst_models(
    self,
    models: dict[str, dict[str, Any]],
    ensembles: dict[str, Any],
    generic_val: pd.DataFrame | None,
    data_dir: str,
    exchange: str,
    symbol: str,
) -> dict[str, Any]:
    """Calibrate analyst models with regime-specific logic."""
    try:
        self.logger.info("🚀 Starting regime-aware analyst model calibration")
        
        regime_calibration_results = {}
        
        # Check if models have regime-specific structure
        has_regime_specific_structure = any(
            isinstance(regime_models, dict) and any(
                isinstance(model_data, dict) for model_data in regime_models.values()
            )
            for regime_models in models.values()
        )
        
        if has_regime_specific_structure:
            self.logger.info("🔄 Calibrating models with regime-specific structure")
            
            # Calibrate each regime separately
            for regime_name, regime_models in models.items():
                self.logger.info(f"🔧 Calibrating analyst models for regime: {regime_name}")
                
                # Load regime-specific validation data
                regime_val = self._load_regime_validation(data_dir, exchange, symbol, regime_name) or generic_val
                
                if regime_val is not None and len(regime_val) >= self.regime_config["min_regime_samples"]:
                    # Calibrate models for this regime
                    regime_calibrated = await self._calibrate_regime_models(
                        regime_models, regime_name, regime_val
                    )
                    regime_calibration_results[regime_name] = regime_calibrated
                    
                    # Log regime-specific metrics
                    if self.regime_config["regime_specific_logging"]:
                        self._log_regime_specific_metrics(regime_name, {
                            "models_calibrated": len(regime_calibrated),
                            "validation_samples": len(regime_val),
                            "regime": regime_name
                        }, "analyst_calibration")
                else:
                    self.logger.warning(f"⚠️ Insufficient validation data for regime {regime_name}")
        else:
            # Fallback to traditional calibration
            self.logger.info("🔄 Using traditional calibration (no regime structure)")
            regime_calibration_results = await self._calibrate_analyst_models(
                models, ensembles, generic_val, data_dir, exchange, symbol
            )
        
        # Store regime-specific results
        self.regime_calibration_results["analyst_models"] = regime_calibration_results
        
        self.logger.info(f"✅ Completed regime-aware analyst model calibration for {len(regime_calibration_results)} regimes")
        return regime_calibration_results
        
    except Exception as e:
        self.logger.error(f"❌ Error in regime-aware analyst calibration: {e}")
        raise

async def _calibrate_regime_aware_tactician_models(
    self,
    models: dict[str, Any],
    ensembles: dict[str, Any],
    generic_val: pd.DataFrame | None,
    data_dir: str,
    exchange: str,
    symbol: str,
) -> dict[str, Any]:
    """Calibrate tactician models with regime-specific logic."""
    try:
        self.logger.info("🚀 Starting regime-aware tactician model calibration")
        
        regime_calibration_results = {}
        
        # Check if models have regime-specific structure
        has_regime_specific_structure = any(
            isinstance(regime_models, dict) and any(
                isinstance(model_data, dict) for model_data in regime_models.values()
            )
            for regime_models in models.values()
        )
        
        if has_regime_specific_structure:
            self.logger.info("🔄 Calibrating tactician models with regime-specific structure")
            
            # Calibrate each regime separately
            for regime_name, regime_models in models.items():
                self.logger.info(f"🔧 Calibrating tactician models for regime: {regime_name}")
                
                # Load regime-specific validation data
                regime_val = self._load_regime_validation(data_dir, exchange, symbol, regime_name) or generic_val
                
                if regime_val is not None and len(regime_val) >= self.regime_config["min_regime_samples"]:
                    # Calibrate models for this regime
                    regime_calibrated = await self._calibrate_regime_models(
                        regime_models, regime_name, regime_val
                    )
                    regime_calibration_results[regime_name] = regime_calibrated
                    
                    # Log regime-specific metrics
                    if self.regime_config["regime_specific_logging"]:
                        self._log_regime_specific_metrics(regime_name, {
                            "models_calibrated": len(regime_calibrated),
                            "validation_samples": len(regime_val),
                            "regime": regime_name
                        }, "tactician_calibration")
                else:
                    self.logger.warning(f"⚠️ Insufficient validation data for regime {regime_name}")
        else:
            # Fallback to traditional calibration
            self.logger.info("🔄 Using traditional calibration (no regime structure)")
            regime_calibration_results = await self._calibrate_tactician_models(
                models, ensembles, generic_val
            )
        
        # Store regime-specific results
        self.regime_calibration_results["tactician_models"] = regime_calibration_results
        
        self.logger.info(f"✅ Completed regime-aware tactician model calibration for {len(regime_calibration_results)} regimes")
        return regime_calibration_results
        
    except Exception as e:
        self.logger.error(f"❌ Error in regime-aware tactician calibration: {e}")
        raise

async def _calibrate_regime_models(
    self, regime_models: dict[str, Any], regime_name: str, validation_data: pd.DataFrame
) -> dict[str, Any]:
    """Calibrate models for a specific regime."""
    try:
        self.logger.info(f"🔧 Calibrating models for regime: {regime_name}")
        
        calibrated_models = {}
        
        for model_name, model_data in regime_models.items():
            try:
                # Apply regime-specific calibration
                calibrated_model = await self._apply_regime_calibration(
                    model_data, model_name, regime_name, validation_data
                )
                calibrated_models[model_name] = calibrated_model
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to calibrate {model_name} for regime {regime_name}: {e}")
                calibrated_models[model_name] = model_data  # Use uncalibrated model
        
        return calibrated_models
        
    except Exception as e:
        self.logger.error(f"❌ Error calibrating models for regime {regime_name}: {e}")
        raise

async def _apply_regime_calibration(
    self, model_data: dict[str, Any], model_name: str, regime_name: str, validation_data: pd.DataFrame
) -> dict[str, Any]:
    """Apply calibration to a specific model for a specific regime."""
    try:
        # Extract model and prepare validation data
        model = model_data.get("model")
        if model is None:
            return model_data
        
        # Prepare features and labels for calibration
        feature_columns = [col for col in validation_data.columns 
        if col not in ["timestamp", "exchange", "symbol", "timeframe", "composite_cluster_id"]]
        
        X_val = validation_data[feature_columns].fillna(0)
        y_val = validation_data.get("label", validation_data.get("target", pd.Series([0] * len(validation_data))))
        
        # Apply regime-specific calibration method
        calibration_method = self.regime_config["regime_calibration_method"]
        
        if hasattr(model, "predict_proba"):
            # Use CalibratedClassifierCV for probabilistic models
            calibrated_model = CalibratedClassifierCV(
                model, method=calibration_method, cv=3
            )
            calibrated_model.fit(X_val, y_val)
            
            # Create calibrated model package
            calibrated_package = model_data.copy()
            calibrated_package["model"] = calibrated_model
            calibrated_package["calibration_method"] = calibration_method
            calibrated_package["regime"] = regime_name
            calibrated_package["calibration_samples"] = len(validation_data)
            
            return calibrated_package
        else:
            # For non-probabilistic models, return as-is
            self.logger.warning(f"⚠️ Model {model_name} does not support probability calibration")
            return model_data
            
    except Exception as e:
        self.logger.warning(f"⚠️ Error applying calibration to {model_name} for regime {regime_name}: {e}")
        return model_data

def _log_regime_specific_metrics(self, regime: str, metrics: dict[str, Any], step_name: str) -> None:
    """Log regime-specific metrics if enabled."""
    if self.regime_config["regime_specific_logging"]:
        self.logger.info(f"📊 Regime {regime} {step_name} metrics: {metrics}")

async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the confidence calibration step."

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
        step = RegimeAwareConfidenceCalibrationStep(config)
        await step.initialize()

        # Execute step
        training_input: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:
        return False

if __name__ == "__main__":
    # Test the step
    async def await test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())