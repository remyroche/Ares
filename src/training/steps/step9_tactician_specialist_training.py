# src/training/steps/step9_tactician_specialist_training.py

import asyncio
import contextlib
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.centralized_decorators import (
    PerformanceLevel,
    ValidationLevel,
    adaptive_resource_allocation,
    comprehensive_validation,
    guard_dataframe_nulls,
    handle_errors,
    intelligent_caching,
    model_validation,
    # Advanced decorators
    performance_monitor,
    pipeline_checkpoint,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
)


class TacticianSpecialistTrainingStep:
    """Step 9: Tactician Specialist Models Training with S/R Level Integration."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.models: dict[str, Any] = {}

        # Initialize SRBreakoutPredictor for S/R level integration
        self.sr_predictor = SRBreakoutPredictor(config)

        # Initialize enhanced LM optimizer
        self.enhanced_lm_optimizer = None
        try:
            from src.training.enhanced_lm_optimizer import EnhancedLMOptimizer

            self.enhanced_lm_optimizer = EnhancedLMOptimizer(config)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"⚠️ Failed to initialize enhanced LM optimizer: {e}")

        # Initialize optimized feature selection manager (fallback)
        self.optimized_feature_selection = None
        try:
            from src.training.optimized_feature_selection_manager import (
                OptimizedFeatureSelectionManager,
            )

            self.optimized_feature_selection = OptimizedFeatureSelectionManager(config)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"⚠️ Failed to initialize optimized feature selection: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician specialist training step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the tactician specialist training step."""
        self.logger.info("Initializing Tactician Specialist Training Step...")

        # Initialize SRBreakoutPredictor for S/R level integration
        try:
            sr_init_success = await self.sr_predictor.initialize()
            if sr_init_success:
                self.logger.info(
                    "✅ SRBreakoutPredictor initialized for S/R level integration",
                )
            else:
                self.logger.warning(
                    "⚠️ Failed to initialize SRBreakoutPredictor, continuing without S/R analysis",
                )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"⚠️ Error initializing SRBreakoutPredictor: {e}")

        self.logger.info(
            "Tactician Specialist Training Step initialized successfully",
        )

    async def _enhance_training_data_with_sr_context(
        self, labeled_data: pd.DataFrame, symbol: str, timeframe: str,
    ) -> pd.DataFrame:
        """Enhance training data with S/R context and outcomes using HMM-aware analysis."""
        try:
            if labeled_data.empty:
                return labeled_data

            self.logger.info(
                f"🔄 Enhancing training data with HMM-aware S/R context for {timeframe}...",
            )

            enhanced_data = labeled_data.copy()

            # Check if we have OHLCV data for S/R analysis
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in enhanced_data.columns for col in required_cols):
                self.logger.warning(
                    "⚠️ Missing OHLCV columns for S/R analysis, skipping enhancement",
                )
                return enhanced_data

            # Adaptive sampling based on timeframe
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            sample_interval = max(1, len(enhanced_data) // max(1, (1000 // timeframe_minutes)))
            sample_indices = enhanced_data.index[::sample_interval]

            sr_features: dict[str, list[Any]] = {
                "sr_proximity": [],
                "sr_outcome": [],
                "sr_confidence": [],
                "breakout_probability": [],
                "rebounce_probability": [],
                "consolidation_probability": [],
                "hmm_regime_confidence": [],
                "multi_timeframe_sr_score": [],
            }

            for idx in sample_indices:
                try:
                    row = enhanced_data.loc[idx]
                    current_price = float(row["close"])

                    lookback_bars = min(200, max(50, timeframe_minutes * 2))
                    market_slice = enhanced_data.loc[:idx].tail(lookback_bars)

                    if len(market_slice) < 20:
                        # Default values if insufficient data
                        sr_features["sr_proximity"].append(0.0)
                        sr_features["sr_outcome"].append("consolidation")
                        sr_features["sr_confidence"].append(0.5)
                        sr_features["breakout_probability"].append(0.33)
                        sr_features["rebounce_probability"].append(0.33)
                        sr_features["consolidation_probability"].append(0.34)
                        sr_features["hmm_regime_confidence"].append(0.5)
                        sr_features["multi_timeframe_sr_score"].append(0.5)
                        continue

                    # Get HMM-aware S/R context and outcome prediction
                    sr_context = await self.sr_predictor.get_sr_context(
                        market_slice=market_slice,
                        current_price=current_price,
                    )
                    sr_outcome = await self.sr_predictor.predict_sr_outcome(
                        market_slice=market_slice,
                        sr_context=sr_context,
                    )

                    # Extract HMM regime information if available
                    hmm_confidence = 0.5
                    if "composite_cluster_confidence" in row:
                        hmm_confidence = float(row.get("composite_cluster_confidence", 0.5))
                    elif "hmm_cluster_confidence" in row:
                        hmm_confidence = float(row.get("hmm_cluster_confidence", 0.5))

                    # Extract features
                    is_near_sr = bool(sr_outcome.get("is_near_sr_level", False))
                    sr_features["sr_proximity"].append(1.0 if is_near_sr else 0.0)
                    sr_features["sr_outcome"].append(
                        sr_outcome.get("outcome", "consolidation"),
                    )
                    sr_features["sr_confidence"].append(
                        float(sr_outcome.get("confidence", 0.5)),
                    )

                    probabilities = sr_outcome.get("probabilities", {})
                    sr_features["breakout_probability"].append(
                        float(probabilities.get("breakout", 0.33)),
                    )
                    sr_features["rebounce_probability"].append(
                        float(probabilities.get("rebounce", 0.33)),
                    )
                    sr_features["consolidation_probability"].append(
                        float(probabilities.get("consolidation", 0.34)),
                    )
                    sr_features["hmm_regime_confidence"].append(float(hmm_confidence))

                    # Multi-timeframe S/R score
                    sr_conf = float(sr_outcome.get("confidence", 0.5))
                    multi_tf_score = sr_conf * 0.6 + float(hmm_confidence) * 0.4
                    sr_features["multi_timeframe_sr_score"].append(multi_tf_score)

                except Exception as e:  # noqa: BLE001
                    self.logger.debug(
                        f"Error processing S/R features for index {idx}: {e}",
                    )
                    # Default values on error
                    sr_features["sr_proximity"].append(0.0)
                    sr_features["sr_outcome"].append("consolidation")
                    sr_features["sr_confidence"].append(0.5)
                    sr_features["breakout_probability"].append(0.33)
                    sr_features["rebounce_probability"].append(0.33)
                    sr_features["consolidation_probability"].append(0.34)
                    sr_features["hmm_regime_confidence"].append(0.5)
                    sr_features["multi_timeframe_sr_score"].append(0.5)

            # Interpolate S/R features to all data points
            for feature_name, values in sr_features.items():
                if len(values) > 1:
                    feature_series = pd.Series(values, index=sample_indices)

                    full_feature = (
                        feature_series.reindex(enhanced_data.index)
                        .interpolate(method="linear")
                        .fillna(0.5)
                    )
                    enhanced_data[f"sr_{feature_name}"] = full_feature
                else:
                    enhanced_data[f"sr_{feature_name}"] = values[0] if values else 0.5

            # Enhanced sample weights using HMM regime information
            enhanced_data["sr_sample_weight"] = (
                enhanced_data["sr_proximity"] * 0.3
                + enhanced_data["hmm_regime_confidence"] * 0.4
                + 0.3
            )

            self.logger.info(
                f"✅ Enhanced training data with HMM-aware S/R context for {timeframe}: {len(enhanced_data)} samples",
            )
            return enhanced_data

        except Exception as e:  # noqa: BLE001
            self.logger.exception(
                f"❌ Error enhancing training data with HMM-aware S/R context: {e}",
            )
            return labeled_data

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes for adaptive processing.
        Step9 only supports 1m and 5m timeframes.
        """
        tf = timeframe.lower()
        if tf == "1m":
            return 1
        if tf == "5m":
            return 5
        # Default to 1 minute if unsupported timeframe
        self.logger.warning(
            f"Unsupported timeframe '{timeframe}' for Step9, defaulting to 1m",
        )
        return 1

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician specialist training step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tactician specialist models training."""
        try:
            self.logger.info("🔄 Executing Tactician Specialist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load tactician labeled data
            labeled_data_dir = f"{data_dir}/tactician_labeled_data"
            labeled_file_parquet = f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet"
            labeled_file_pickle = f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"

            if os.path.exists(labeled_file_parquet) or os.path.exists(
                labeled_file_pickle,
            ):
                if os.path.exists(labeled_file_parquet):
                    try:
                        labeled_data = pd.read_parquet(labeled_file_parquet)
                    except Exception:
                        with open(labeled_file_pickle, "rb") as f:
                            labeled_data = pickle.load(f)
                else:
                    with open(labeled_file_pickle, "rb") as f:
                        labeled_data = pickle.load(f)
            else:
                msg = (
                    "Tactician labeled data not found: "
                    f"{labeled_file_parquet} or {labeled_file_pickle}. Step 9 requires labeled data from Step 8."
                )
                raise FileNotFoundError(msg)

            # Convert to DataFrame if needed
            if not isinstance(labeled_data, pd.DataFrame):
                labeled_data = pd.DataFrame(labeled_data)

            # Optionally enhance training data with HMM-aware S/R context
            current_timeframe = training_input.get("timeframe", "1m")
            if current_timeframe not in ["1m", "5m"]:
                self.logger.warning(
                    f"Step9 only supports 1m and 5m timeframes, got: {current_timeframe}",
                )
                current_timeframe = "1m"

            try:
                labeled_data = await self._enhance_training_data_with_sr_context(
                    labeled_data,
                    symbol,
                    current_timeframe,
                )
            except Exception as _e:  # noqa: BLE001
                self.logger.warning(
                    f"Failed to enhance training data with HMM-aware S/R context: {_e}",
                )

            # Train tactician specialist models
            training_results = await self._train_tactician_models(
                labeled_data,
                symbol,
                exchange,
            )

            # Save training results
            models_dir = f"{data_dir}/tactician_models"
            os.makedirs(models_dir, exist_ok=True)

            for model_name, model_data in training_results.items():
                model_file = f"{models_dir}/{model_name}.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump(model_data, f)

            # Save training summary
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_tactician_training_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(training_results, f, indent=2)

            self.logger.info(
                f"✅ Tactician specialist training completed. Results saved to {models_dir}",
            )

            # Update pipeline state
            pipeline_state["tactician_models"] = training_results

            return {
                "tactician_models": training_results,
                "models_dir": models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:  # noqa: BLE001
            self.logger.error(
                f"❌ Error in Tactician Specialist Training: {e}",
                exc_info=True,
            )
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _train_tactician_models(
        self, data: pd.DataFrame, symbol: str, exchange: str,
    ) -> dict[str, Any]:
        """Train tactician specialist models."""
        try:
            self.logger.info(
                f"Training tactician specialist models for {symbol} on {exchange}...",
            )

            # Prepare data - handle data types properly
            target_column = (
                "tactician_label" if "tactician_label" in data.columns else "label"
            )
            if target_column not in data.columns:
                raise ValueError("Target column for tactician training not found")

            y = data[target_column].copy()

            # Drop datetime columns and object columns except target
            datetime_columns = data.select_dtypes(
                include=["datetime64[ns]", "datetime64", "datetime"],
            ).columns.tolist()
            if datetime_columns:
                self.logger.info(f"Dropping datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)

            object_columns = data.select_dtypes(include=["object"]).columns.tolist()
            object_columns_to_drop = [
                col for col in object_columns if col != target_column
            ]
            if object_columns_to_drop:
                self.logger.info(f"Dropping object columns: {object_columns_to_drop}")
                data = data.drop(columns=object_columns_to_drop)

            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [
                col for col in numeric_columns if col != target_column
            ]

            if not feature_columns:
                self.logger.warning(
                    "No numeric feature columns found for tactician training",
                )
                data["simple_feature"] = np.random.randn(len(data))
                feature_columns = ["simple_feature"]

            X = data[feature_columns].copy()

            # Ensure numeric
            for col in list(X.columns):
                if not pd.api.types.is_numeric_dtype(X[col]):
                    self.logger.warning(
                        f"Non-numeric column detected and dropped: {col} ({X[col].dtype})",
                    )
                    X = X.drop(columns=[col])
                    feature_columns.remove(col)

            X = X.fillna(0)

            # Chronological split (80/20)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

            # Feature selection and optimization (optional)
            if self.enhanced_lm_optimizer is not None:
                self.logger.info("🚀 Applying enhanced LM optimization for tactician models...")
                model_type = (
                    "classification"
                    if y_train.dtype == "object" or len(pd.unique(y_train)) < 10
                    else "regression"
                )
                try:
                    optimization_results, optimized_features = await self.enhanced_lm_optimizer.optimize_lm_model(
                        step_name="step9",
                        features_df=X_train,
                        target=y_train,
                        model_type=model_type,
                        architecture="LightGBM",
                    )
                    X_train = optimized_features
                    X_test = X_test[X_train.columns]
                    self.logger.info(
                        f"✅ Applied feature selection: {len(X_train.columns)} features selected",
                    )
                    self.enhancement_results = getattr(self, "enhancement_results", {})
                    self.enhancement_results["enhanced_optimization"] = optimization_results
                except Exception as _opt_e:  # noqa: BLE001
                    self.logger.warning(
                        f"Enhanced LM optimizer failed; proceeding without it: {_opt_e}",
                    )

            models: dict[str, Any] = {}

            # 1. LightGBM
            try:
                models["lightgbm"] = await self._train_lightgbm(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    symbol,
                    exchange,
                )
            except Exception as _e:  # noqa: BLE001
                self.logger.warning(f"LightGBM training failed: {_e}")

            # 2. Calibrated Logistic Regression
            try:
                models["calibrated_logistic"] = await self._train_calibrated_logistic(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    symbol,
                    exchange,
                )
            except Exception as _e:  # noqa: BLE001
                self.logger.warning(f"Calibrated Logistic training failed: {_e}")

            # 3. XGBoost
            try:
                models["xgboost"] = await self._train_xgboost(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    symbol,
                    exchange,
                )
            except Exception as _e:  # noqa: BLE001
                self.logger.warning(f"XGBoost training failed: {_e}")

            # 4. Random Forest
            try:
                models["random_forest"] = await self._train_random_forest(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    symbol,
                    exchange,
                )
            except Exception as _e:  # noqa: BLE001
                self.logger.warning(f"Random Forest training failed: {_e}")

            self.logger.info(f"Trained {len(models)} tactician models")

            return models

        except Exception as e:  # noqa: BLE001
            self.logger.exception(error(f"Error training tactician models: {e}"))
            raise

    async def _train_lightgbm(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str,
    ) -> dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score

            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0

            if overfitting_risk > 0.1:  # High overfitting risk
                reg_alpha = 0.1
                reg_lambda = 0.1
                min_child_samples = 50
                subsample = 0.7
            elif overfitting_risk > 0.05:  # Medium overfitting risk
                reg_alpha = 0.05
                reg_lambda = 0.05
                min_child_samples = 30
                subsample = 0.8
            else:  # Low overfitting risk
                reg_alpha = 0.01
                reg_lambda = 0.01
                min_child_samples = 20
                subsample = 0.9

            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )

            eval_set = [(X_test, y_test)]
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric="logloss",
                verbose=False,
            )

            y_pred = model.predict(X_test)
            _ = model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))

            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "LightGBM",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                },
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(error(f"Error training LightGBM: {e}"))
            raise

    async def _train_calibrated_logistic(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str,
    ) -> dict[str, Any]:
        """Train Calibrated Logistic Regression model."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver="liblinear",
            )

            calibrated_model = CalibratedClassifierCV(
                estimator=base_model,
                cv=5,
                method="isotonic",
            )

            calibrated_model.fit(X_train, y_train)

            y_pred = calibrated_model.predict(X_test)
            _ = calibrated_model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))

            return {
                "model": calibrated_model,
                "accuracy": accuracy,
                "feature_importance": {},
                "model_type": "CalibratedLogisticRegression",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "C": 1.0,
                    "max_iter": 1000,
                    "calibration_method": "isotonic",
                    "cv_folds": 5,
                },
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(error(f"Error training Calibrated Logistic Regression: {e}"))
            raise

    async def _train_xgboost(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str,
    ) -> dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score

            best_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01,
            }

            # Adaptive regularization based on data characteristics
            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0

            if overfitting_risk > 0.1:  # High risk
                reg_alpha = max(0.1, best_params.get("reg_alpha", 0.1))
                reg_lambda = max(0.1, best_params.get("reg_lambda", 0.1))
                min_child_weight = 10
                subsample = 0.7
            elif overfitting_risk > 0.05:  # Medium risk
                reg_alpha = max(0.05, best_params.get("reg_alpha", 0.05))
                reg_lambda = max(0.05, best_params.get("reg_lambda", 0.05))
                min_child_weight = 5
                subsample = 0.8
            else:
                reg_alpha = best_params.get("reg_alpha", 0.01)
                reg_lambda = best_params.get("reg_lambda", 0.01)
                min_child_weight = 1
                subsample = 0.9

            model = xgb.XGBClassifier(
                n_estimators=best_params.get("n_estimators", 200),
                max_depth=best_params.get("max_depth", 6),
                learning_rate=best_params.get("learning_rate", 0.05),
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                subsample=best_params.get("subsample", subsample),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )

            eval_set = [(X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set)

            y_pred = model.predict(X_test)
            _ = model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))

            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": best_params,
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(error(f"Error training XGBoost: {e}"))
            raise

    async def _train_random_forest(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, symbol: str, exchange: str,
    ) -> dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            _ = model.predict_proba(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))

            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "RandomForest",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                },
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(error(f"Error training Random Forest: {e}"))
            raise


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
@idempotent_step(step_key="step9_tactician_specialist_training")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=5400.0)
@performance_monitor(
    enable_profiling=True,
    enable_memory_tracking=True,
    enable_cpu_tracking=True,
    save_profile_data=True,
    level=PerformanceLevel.PROFILING,
)
@model_validation(
    check_overfitting=True,
    check_underfitting=True,
    validation_metrics=["accuracy", "precision", "recall", "f1"],
    overfitting_threshold=0.1,
    underfitting_threshold=0.6,
)
@pipeline_checkpoint(
    save_intermediate_results=True,
    checkpoint_frequency=500,
    enable_rollback=True,
)
@intelligent_caching(
    cache_intermediate_results=True,
    cache_validation_data=True,
    cache_model_artifacts=True,
    cache_ttl_hours=24,
)
@adaptive_resource_allocation(
    dynamic_memory_allocation=True,
    adaptive_batch_sizes=True,
    resource_scaling_threshold=0.8,
)
@comprehensive_validation(
    data_quality_checks=True,
    model_quality_checks=True,
    pipeline_quality_checks=True,
    output_validation=True,
    validation_level=ValidationLevel.WARNING,
)
@validate_step_prerequisites(
    required_directories=["data/training", "models"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "sklearn", "lightgbm", "catboost"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "features", "targets"],
    },
    context="Tactician Specialist Training",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    cross_validation_isolation=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    monitor_interval=60.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=300.0,
    expected_exception=Exception,
    monitor_interval=60.0,
)
@validate_step_output(
    required_files=["models/{exchange}_{symbol}_tactician_specialist.pkl"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["predictions", "probabilities"],
    },
    performance_thresholds={"training_time_minutes": 120.0, "memory_usage_gb": 8.0},
    format_validation=True,
)
@quality_gate(
    model_performance_thresholds={"accuracy": 0.6, "f1_score": 0.5},
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    convergence_checks=True,
    overfitting_detection=True,
    validation_score_requirements={"cross_validation_score": 0.6},
)
async def run_step(
    symbol: str, exchange: str = "BINANCE", data_dir: str = "data/training", force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the tactician specialist training step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful = False otherwise

    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = TacticianSpecialistTrainingStep(config)
        await step.initialize()

        # Execute step
        training_input = {
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
    async def test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())